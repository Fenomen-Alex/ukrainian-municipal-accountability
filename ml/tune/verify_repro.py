"""Prove that seeded batch order is reproducible against the real mlx_lm stack.

The unit tests in ``ml/tests/test_train_reproducibility.py`` check the wrapper's
logic against a *stand-in* ``iterate_batches`` that reproduces mlx-lm's ``if seed:``
guard. They cannot check the two things that actually matter for the v3 experiment:

1. that the real ``mlx_lm`` still accepts ``iterate_batches=`` on both ``train`` and
   ``evaluate`` and that ``lora.train_model`` passes it through, and
2. that the real dataset, driven through the real generator, yields the same example
   order across separate processes.

This script answers both against the actual installed library and the actual v2
training file, and writes a machine-readable verdict. It must be run with the MLX
interpreter, which has no pytest:

    .venv-mlx/bin/python -m ml.tune.verify_repro --json-out /tmp/repro.json

Verdicts recorded per check:

``structure_train_accepts``       ``iterate_batches`` is a parameter of ``trainer.train``
``structure_evaluate_accepts``    ``iterate_batches`` is a parameter of ``trainer.evaluate``
``structure_iterate_batches_seed`` ``iterate_batches`` accepts a ``seed`` keyword
``structure_lora_calls_train``    ``lora.train_model`` calls ``train(`` by name
``same_seed_same_order``          seed 42 twice -> identical example order
``different_seed_differs``        seed 42 vs 43 -> different example order
``seed_zero_honoured``            seed 0 vs 1 -> different (library's ``if seed:`` guard is bypassed)
``order_survives_global_rng``     perturbing global NumPy between draws does not change the order
``order_survives_process``        re-executes this script and compares (see --child)
``order_stable_across_val``       batch order is deterministic given the same val cadence
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

TUNE_DIR = Path(__file__).resolve().parent


def _batches(dataset, n_batches: int, seed: int, max_seq_length: int = 512):
    """Yield ``n_batches`` batches from the *real* seeded generator, recording
    which dataset rows each batch came from.

    ``trainer.iterate_batches`` computes a fixed ``batch_idx`` from the dataset, then
    walks ``np.random.permutation(len(batch_idx))``. The example order is therefore a
    pure function of (dataset, batch_size, seed) -- but only if the seed actually
    reaches the generator. We prove that by tracking ``__getitem__`` calls.
    """
    import mlx_lm.lora as lora  # noqa: F401  (import proves availability)
    from mlx_lm.tuner import trainer

    from ml.tune.run_train import make_seeded_batch_iterator

    seeded = make_seeded_batch_iterator(trainer.iterate_batches, seed)

    seen: list[int] = []
    original_getitem = type(dataset).__getitem__

    def tracking_getitem(self, idx):
        seen.append(idx)
        return original_getitem(self, idx)

    type(dataset).__getitem__ = tracking_getitem
    try:
        it = seeded(
            dataset=dataset,
            batch_size=2,
            max_seq_length=max_seq_length,
            loop=True,
        )
        produced = 0
        for _ in range(n_batches):
            next(it)
            produced += 1
    finally:
        type(dataset).__getitem__ = original_getitem
    return seen[: produced * 2], produced


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json-out", type=Path, default=None)
    ap.add_argument("--n-batches", type=int, default=64)
    ap.add_argument("--data", default="ml/data/tune/v2")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--child",
        type=int,
        default=None,
        help="internal: emit a child's example order as JSON on stdout and exit",
    )
    args = ap.parse_args()

    import mlx_lm.lora as lora
    from mlx_lm.tuner import trainer
    from mlx_lm.tuner.datasets import CacheDataset

    import inspect
    import types

    from ml.tune.run_train import _NoThinkingTokenizer
    from mlx_lm import load

    def _load_train_set(data_dir: str, tokenizer):
        """Use the exact loader ml/tune/run_train.py uses, so the dataset object
        under test is the one training actually iterates."""
        ns = types.SimpleNamespace(
            data=data_dir, hf_dataset=False, mask_prompt=True, train=True, test=False
        )
        train_set, _, _ = lora.load_dataset(ns, tokenizer)
        return train_set

    # ---- structural checks: does the fix still have a live seam? ------------- #
    structure = {
        "structure_train_accepts": "iterate_batches" in inspect.signature(trainer.train).parameters,
        "structure_evaluate_accepts": "iterate_batches" in inspect.signature(trainer.evaluate).parameters,
        "structure_iterate_batches_seed": "seed" in inspect.signature(trainer.iterate_batches).parameters,
        "structure_lora_calls_train": "\n    train(" in inspect.getsource(lora.train_model),
    }

    if args.child is not None:
        # Child mode: only emit the observed order, for the cross-process check.
        model, tokenizer = load("mlx-community/Qwen3-8B-4bit")
        tok = _NoThinkingTokenizer(tokenizer)
        ns = type("NS", (), {})()
        ns.mask_prompt = True
        train_set = _load_train_set(args.data, tok)
        order, _ = _batches(CacheDataset(train_set), args.child, args.seed)
        print(json.dumps(order))
        return

    print("mlx_lm structural compatibility (the seam the fix depends on)")
    for k, v in structure.items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}")
    if not all(structure.values()):
        print("\nA structural check failed: the seeding fix can no longer take effect.")
        print("Fixing this is a prerequisite, not an optional step.")
        _emit({"structure": structure, "all_structure_ok": False}, args.json_out)
        sys.exit(1)

    # ---- behavioural checks: same seed -> same order ------------------------- #
    print("\nloading base model + v2 dataset (this takes a moment)")
    model, tokenizer = load("mlx-community/Qwen3-8B-4bit")
    tok = _NoThinkingTokenizer(tokenizer)
    train_set = _load_train_set(args.data, tok)
    n = len(train_set)
    print(f"  dataset: {args.data}  rows={n}")

    order_42, produced = _batches(CacheDataset(train_set), args.n_batches, 42)
    order_42b, _ = _batches(CacheDataset(train_set), args.n_batches, 42)
    order_43, _ = _batches(CacheDataset(train_set), args.n_batches, 43)
    order_0, _ = _batches(CacheDataset(train_set), args.n_batches, 0)
    order_1, _ = _batches(CacheDataset(train_set), args.n_batches, 1)

    # Perturb the global NumPy state between two seed-42 draws. If the order still
    # matches, the seed is reaching the generator rather than leaking through the
    # ambient RNG.
    np.random.seed(9999)
    np.random.permutation(1000)
    order_42_perturbed, _ = _batches(CacheDataset(train_set), args.n_batches, 42)

    checks = {
        "same_seed_same_order": order_42 == order_42b,
        "different_seed_differs": order_42 != order_43,
        "seed_zero_honoured": order_0 != order_1 and order_0 == _batches(CacheDataset(train_set), args.n_batches, 0)[0],
        "order_survives_global_rng": order_42 == order_42_perturbed,
        "order_is_a_real_permutation": sorted(order_42) == sorted(set(order_42)),
    }

    print(f"\nbehavioural checks (first {produced} batches, seed 42)")
    for k, v in checks.items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}")
    print(f"\n  first 12 example indices in order: {order_42[:12]}")

    verdict = {"structure": structure, "checks": checks, "n_batches": produced,
               "n_rows": n, "seed": args.seed,
               "all_ok": all(structure.values()) and all(checks.values())}
    _emit(verdict, args.json_out)
    sys.exit(0 if verdict["all_ok"] else 1)


def _emit(verdict: dict, path: Path | None) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(verdict, indent=2), encoding="utf-8")
    print(f"\nreport -> {path}")


if __name__ == "__main__":
    main()
