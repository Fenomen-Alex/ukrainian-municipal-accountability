# V3 Report: the control arm failed its own validation — no publication

Status: **experiment concluded, gates NOT passed, v3 model NOT published.**

Companion document: `v3_experiment_design.md` (plan + §10 as-built, §11 outcome).

## 1. What was built

The v3 treatment corpus, C1–C4, all verified:

- **C1** boiled the consent / phone-&-letter boilerplate out of complaint issues,
  sentence-level: 59.2 % → 7.1 % boilerplate-bearing streams (measured per stream:
  single 7.5 %, terse 0.0 %, multitopic 7.1 %). Two patterns were tried and
  *withdrawn* for false positives; their numbers are not hidden, they are in §10.
  240 records (230 single + 10 multitopic) whose issue vanished entirely were
  dropped and counted in `meta.json`, so the loss is visible.
- **C2** extracts the requested action clause instead of the whole sentence (verb
  preserved, PII redacted) — whole-sentence redundancy 0.0 %; non-empty actions
  48.6 % → 17.8 %. The generic *"вжити заходи"* rote phrase is emptied rather than
  kept as a fake action.
- **C3** 1872 terse rows (234 pool units × 8). The measured pool is 246, not the
  plan's 178/928; the test suite pins the truth to `meta["terse_pool"]["n_units"]`.
- **C4** the 22 hand-annotated multitopic records are reweighted (24×/2×), not
  inflated, giving a 21.8 % real-row share (not the plan's 34.5 %).

Final treatment: **9448 rows** = 5154 single + 1872 terse + 2422 multitopic.
Control is byte-identical to v2's training data (sha256 `a93a6af4…`) with an
identical, recorded recipe.

## 2. What ran

The control arm trained and fused (`qwen3-8b-lora-v3-control-fused`), then scored
all four suites with `--tag` so nothing overwrote a v2 baseline. The driver that
would then train the treatment was stopped **before** the treatment started,
because the control failed the protocol's precondition.

## 3. The control did not validate the harness

The protocol requires the control to land within 1 point of every recorded v2
baseline. It did not, on the primary suite most of all:

| suite | metric | v2 | control | delta |
|---|---|---|---|---|
| eval_v3 (146) | topic_count_accuracy | 0.7808 | 0.6644 | **−0.116 (p=0.0023)** |
| eval_v3 (146) | domain_set_exact | 0.6301 | 0.5479 | −0.082 (p=0.043) |
| eval_v3 (146) | domain_set_recall | 0.7637 | 0.6849 | −0.079 (sig.) |
| eval_v3 (146) | row-schema + json parse | 0.9863–0.9932 | 0.9658 | −0.02 |
| multitopic 85 | multi_topic_recall | 0.8941 | 0.6824 | −0.212 |
| multitopic 85 | topic_count_accuracy | 0.8588 | 0.6824 | −0.176 |
| smoke 20 | two-topic recall | 1/4 | 0/4 | −1 |
| frozen 329 | domain_macro_f1 | 0.7596 | 0.6765 | −0.083 |
| frozen 329 | domain_accuracy | 0.8389 | 0.8511 | +0.012 |

The control is simultaneously better at single-topic domain attribution and much
worse at the multi-topic skill. Fully-correct cases on eval_v3: 28 → 11.

## 4. Root cause (mechanical, demonstrated)

An 800-step run at batch 2 sees a **uniform-random 18.8 % window** of the
8519-example corpus (1600 of 8519 examples; 800 of 4259 batches; the length-sorted
batches are permuted, so the window is a random subset, not the longest ones). The
seed only makes the window *reproducible*; it does not make it representative. v2
drew one window (unseeded RNG), the control another (seed 42); the windows shared
~18 % of the seen set. Both models were graded on ~81 % of complaints they never
trained on, and the gap on those is subset luck, not signal.

## 5. What this proves

`mlx_lm`'s own loop confirms it (`tuner/trainer.py`: `zip(range(1, iters+1),
iterate_batches(loop=True))`), and the paired statistics make it unmissable: the
only difference between the two models is batch-visit order, and it moved scores by
**up to 21 points**. Therefore the n=1-per-arm design cannot separate a C1–C4
treatment effect from a fresh window draw, and v2's own baselines were, in the
same sense, one draw.

## 6. Treatment: not trained, deliberately

Once the precondition failed, a treatment run's score would have been an
undiagnosable mix of effect and lottery. Training it anyway would have produced
numbers — including possibly good-looking gate passes — attributed to nothing. The
protocol says the treatment number means nothing when the control fails; the
report agrees and stops.

## 7. Gate verdict (recorded, for the record)

`gates_v3 --arm v3-control`: schema FAIL, boilerplate FAIL, topic_count FAIL,
two_topic_recall FAIL, smoke_two_topic FAIL (0/4 vs target 4/4). The five advisory
gates pass. Applied to a treatment, the four FAILs would have been measured anyway
at these tolerances *regardless of the treatment's true effect*, because control
and treatment would each drag along their own window draw.

## 8. No publication

No v3 model exists to publish; nothing was uploaded to HF. The fused artifact on
disk is the *control* and is gitignored. The honest headline is that the v3
experiment, as designed, **cannot support a publishable claim at one training run
per arm**, and the gates the plan authorized are not achievable or interpretable at
that budget.

## 9. What is still true

- The C1–C4 corpus build is verified (310 tests: byte-identical control, per-stream
  leakage, per-topic distinctness of the multitopic relabel, drop accounting,
  terse pool truth). It is a ready, cleaner 9448-row training set, usable by any
  redesign.
- The fused-control/provenance machinery, `--tag` scorers, `compare_arms`,
  `gates_v3`, and `training_state` all work and are tested.
- Frozen v2 baseline serves the v1-era weak-label suites (329, 85) unchanged.

## 10. What a sound redo needs (scope, not promised)

1. **Full corpus coverage**: iters ≥ batch count (≈24–30 h per arm at batch 2) or a
   batch size that fits the corpus in 800 steps. Either changes the frozen v2
   recipe, so the gates' v2-relative baselines must be recomputed on the new
   protocol — they cannot be reused.
2. **Multi-seed arms** (≥3 per arm), reporting distributions and an effect size
   with variance rather than one point score pair.
3. Only then is a publication-grade "treatment beats control" claim possible for
   the C1–C4 corpus.

The measured negative is the deliverable of this experiment. The next experiment is
a different budget, not a different model of the same run.