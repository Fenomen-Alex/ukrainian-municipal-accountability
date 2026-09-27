# Reproducibility audit

**Finding:** `--seed` does not make a training run reproducible. Two runs with
identical arguments and identical data see different batches, in different orders.
The seed covers LoRA initialisation and nothing else.

Environment: `mlx_lm` 0.31.3 in `.venv-mlx`, NumPy 2.5.3, Python 3.12.

## 1. Where the seed goes, and where it stops

`ml/tune/run_train.py` parses `--seed` (default 42) into `args.seed`. That value
reaches two places and is consumed by neither for batch order:

| component | file:line | behaviour |
| --- | --- | --- |
| LoRA init | `mlx_lm/lora.py:223` | `mx.random.seed(args.seed)` — **works** |
| reporting callback | `mlx_lm/lora.py:320` | `np.random.seed(args.seed)` in `run()`, a different entry point |
| batch iterator | `mlx_lm/tuner/trainer.py:102` | accepts `seed=None` |
| … its only use | `mlx_lm/tuner/trainer.py:138-141` | `if seed: np.random.seed(seed)` then `np.random.permutation(...)` |
| the caller | `mlx_lm/tuner/trainer.py:275` | `iterate_batches(...)` — **no `seed=` argument** |

`iterate_batches` is bound as a **default argument** at `def` time
(`trainer.py:225`, and again for `loss` at `:183`). Rebinding the module attribute
`trainer.iterate_batches` therefore has no effect on `train`: the default already
holds the original function object. There is no code path by which `args.seed`
reaches the permutation.

The `if seed:` guard is a second, independent defect: `seed=0` is falsy, so
`--seed 0` silently produces an unseeded run.

## 2. Empirical confirmation

`numpy.random.RandomState` is seeded from OS entropy at interpreter start, so an
unseeded draw genuinely differs per process. Reproducing
`iterate_batches`' structure verbatim, 24 examples, `batch_size=2`, same `--seed 42`,
three separate processes:

```
--seed 42  first 6 batches: [(15, 22), (0, 7), (16, 23), (3, 10), (2, 9), (14, 21)]  sha1=5d6e074a0f9e
--seed 42  first 6 batches: [(0, 7), (15, 22), (5, 12), (17, 4), (14, 21), (19, 6)]  sha1=d6b578d0e422
--seed 42  first 6 batches: [(11, 18), (2, 9), (0, 7), (16, 23), (13, 20), (1, 8)]  sha1=dfc4055349f7
```

With the seed actually forwarded:

```
--seed 42  first 6 batches: [(19, 6), (5, 12), (0, 7), (11, 18), (16, 23), (1, 8)]  sha1=2238813086fc
--seed 42  first 6 batches: [(19, 6), (5, 12), (0, 7), (11, 18), (16, 23), (1, 8)]  sha1=2238813086fc
--seed 42  first 6 batches: [(19, 6), (5, 12), (0, 7), (11, 18), (16, 23), (1, 8)]  sha1=2238813086fc
```

And `seed=0`, passed through to the library as-is, is ignored by its `if seed:`
guard — two runs, two different orders:

```
--seed 0   first 6 batches: [(15, 22), (19, 6), (1, 8), (17, 4), (13, 20), (5, 12)]  sha1=84072ac70b10
--seed 0   first 6 batches: [(11, 18), (2, 9), (0, 7), (19, 6), (3, 10), (1, 8)]  sha1=51d46303a6be
```

A second, subtler coupling: because `iterate_batches` draws from the **global** NumPy
RNG, batch order depends on how much of that RNG anything else consumed earlier in
the process. Seeding 42 and then making one unrelated draw changes the order:

```
np.random.seed(42); np.random.permutation(12)   # unrelated draw
np.random.permutation(12)[:6]  ->  [10  9  0  8  6  3]     (not [10 9 0 8 5 2])
```

So batch order is not merely unseeded, it is coupled to unrelated library
behaviour — which is why it can change without any change to this repository.

## 3. What this invalidates

Attempt-to-attempt metric differences in this project are not attributable. Two runs
of the same recipe on the same data, with the same `--seed`, differ in batch
composition, and with `grad_accumulation_steps=2` and `batch_size=2` each optimiser
step sees 4 examples chosen at random. Over 800 iterations that is enough variance to
move domain accuracy by several points.

Concretely: v2 attempt-10 was selected as the best of a sequence of attempts
(the commit message records it as a reproduction of the attempt-7 recipe). Nothing in
that selection was controlled for batch order.

This does **not** mean the v2 model is bad or the reported metrics are wrong. It means
the *differences between attempts* carry an uncontrolled noise term of unknown size,
so "attempt-10 fixed the multi-topic problem" is not a claim the current harness can
support.

## 4. The fix

`ml/tune/run_train.py` now injects a seeded iterator through the only seam that works:

```python
def make_seeded_batch_iterator(base, seed: int):
    def iterate(*args, **kwargs):
        np.random.seed(seed)      # covers seed == 0, which the library's `if seed:` skips
        kwargs["seed"] = seed
        return base(*args, **kwargs)
    return iterate
```

`install_seeded_batch_order(lora, trainer, seed)` wraps `lora.train` — the name
`train_model` actually calls at `lora.py:295` — and injects
`iterate_batches=<seeded>` as a keyword. Three details that matter:

- **Patch `lora.train`, not `trainer.iterate_batches`.** The default argument is
  already bound; rebinding the module attribute would be a silent no-op.
- **Seed NumPy in the wrapper as well as passing `seed` through.** This is what makes
  `seed=0` behave like any other seed instead of silently producing an unseeded run.
- **Idempotent per seed.** Re-installing for the same seed returns the existing
  wrapper, so wrappers cannot nest and re-seed on every call.

`run_config.json` is now written next to the adapter, recording every resolved
argument including the seed, so a run can be reproduced from its own output rather
than from shell history.

## 5. Verification

`ml/tests/test_train_reproducibility.py` — 9 tests, no MLX and no GPU required. The
stand-in reproduces the library's real structure including the `if seed:` guard, and
covers:

| test | what it pins |
| --- | --- |
| `test_same_seed_gives_the_same_batch_order` | the fix works |
| `test_different_seeds_give_different_batch_orders` | the fix is not just constant output |
| `test_order_is_independent_of_numpy_global_state` | the coupling in §2 is broken |
| `test_seed_zero_is_honoured` | the `if seed:` defect is handled |
| `test_batches_cover_every_example_exactly_once_per_epoch` | seeding does not skip or duplicate examples |
| `test_seed_is_passed_through_to_the_library` | `seed` is set explicitly, not just globally |
| `test_explicit_seed_argument_is_overridden_not_silently_kept` | a caller cannot defeat the run-level seed |
| `test_injection_targets_lora_train_not_the_bound_default` | the patch lands on the right name |
| `test_injection_is_idempotent_for_the_same_seed` | wrappers do not nest |

The last one is `importorskip`-guarded and additionally runs under `.venv-mlx` to
assert against the installed `mlx_lm` that `train` and `evaluate` both accept
`iterate_batches`, that `iterate_batches` accepts `seed`, and that
`train_model` still routes through the module-level `train` — so if a future
`mlx_lm` release moves that call, the test fails instead of the fix silently becoming
a no-op.

## 6. Still not reproducible

Fixing batch order makes a run repeatable **on the same hardware and the same
`mlx_lm` version**. Two things remain outside this repository's control and are not
addressed here:

- **Non-deterministic MLX kernels.** GPU reductions on Apple silicon are not
  bit-reproducible across runs even with fixed input order. Bitwise-identical weights
  are therefore not achievable; run-to-run metric variation should be small but
  non-zero.
- **Dependency drift.** `mlx_lm` 0.31.3 and NumPy 2.5.3 are recorded in
  `run_config.json` going forward, but nothing pins them. A version bump can change
  the chat template, the batching, or the LoRA initialisation.

Practical guidance: treat a metric difference of less than ~1 point on the 329-case
frozen suite as within run-to-run noise, and re-measure the control arm before
attributing anything to a data change. The v3 plan makes that an explicit
precondition.
