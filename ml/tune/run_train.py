"""QLoRA training driver for the municipal-complaint JSON task.

Wraps the mlx-lm LoRA pipeline with two changes:

1. The tokenizer's chat template is forced to ``enable_thinking=False`` so that
   training targets match the inference format produced by
   :mod:`ml.tune.run_eval` (Qwen3's reasoning prefix is never part of the learned
   format).
2. The batch order is seeded. mlx-lm's ``train_model`` already calls
   ``mx.random.seed(args.seed)``, which covers LoRA initialisation, but
   ``tuner.trainer.train`` never forwards a seed to ``iterate_batches``, so the
   NumPy permutation that orders the batches is drawn from an unseeded global RNG.
   Two runs with the same ``--seed`` therefore see different batches. See
   ``ml/reports/reproducibility_audit.md``.

Run from the repo root with the MLX venv::

    .venv-mlx/bin/python -m ml.tune.run_train --iters 400 \
        --adapter-path ml/data/tune/adapters/qwen3-8b-lora
"""

from __future__ import annotations

import argparse
import json
import types
from pathlib import Path

import numpy as np


def make_seeded_batch_iterator(base, seed: int):
    """Wrap an mlx-lm ``iterate_batches`` so batch order is reproducible.

    mlx-lm's ``iterate_batches`` does accept a ``seed``, but guards it with
    ``if seed:`` (so ``seed=0`` is silently ignored) and ``tuner.trainer.train``
    never passes one at all -- it is bound as a default argument at ``def`` time,
    so rebinding the module attribute would have no effect. Injecting a wrapper
    through ``train(iterate_batches=...)`` is the only seam that works.

    Seeding NumPy here as well as passing ``seed`` through is deliberate: it makes
    ``seed=0`` behave like any other seed, which the library's own guard does not.
    """

    def iterate(*args, **kwargs):
        np.random.seed(seed)
        kwargs["seed"] = seed
        return base(*args, **kwargs)

    return iterate


def install_seeded_batch_order(lora_module, trainer_module, seed: int) -> None:
    """Make ``lora.train_model`` batch order deterministic for ``seed``.

    Patches ``lora.train`` -- the name ``train_model`` actually calls -- rather
    than ``trainer.iterate_batches``, whose default argument is already bound.
    """
    if getattr(lora_module, "_ukraine_seeded_batch_order", None) == seed:
        return
    original_train = lora_module.train
    seeded = make_seeded_batch_iterator(trainer_module.iterate_batches, seed)

    def train_with_seeded_batches(*args, **kwargs):
        kwargs.setdefault("iterate_batches", seeded)
        return original_train(*args, **kwargs)

    lora_module.train = train_with_seeded_batches
    lora_module._ukraine_seeded_batch_order = seed
    lora_module._ukraine_original_train = original_train


class _NoThinkingTokenizer:
    """Tokenizer wrapper that strips Qwen3's reasoning mode from the template."""

    def __init__(self, tokenizer):
        self._t = tokenizer

    def __getattr__(self, item):
        return getattr(self._t, item)

    def apply_chat_template(self, *args, **kwargs):
        kwargs.pop("enable_thinking", None)
        kwargs["enable_thinking"] = False
        return self._t.apply_chat_template(*args, **kwargs)

    @property
    def eos_token_id(self):
        return self._t.eos_token_id

    @property
    def eos_token_ids(self):
        return self._t.eos_token_ids


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="mlx-community/Qwen3-8B-4bit")
    ap.add_argument("--data", default="ml/data/tune")
    ap.add_argument("--adapter-path", required=True)
    ap.add_argument("--fine-tune-type", default="lora")
    ap.add_argument("--num-layers", type=int, default=16)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--iters", type=int, default=400)
    ap.add_argument("--val-batches", type=int, default=10)
    ap.add_argument("--learning-rate", type=float, default=1e-4)
    ap.add_argument("--max-seq-length", type=int, default=2048)
    ap.add_argument("--grad-checkpoint", action="store_true")
    ap.add_argument("--grad-accumulation-steps", type=int, default=4)
    ap.add_argument("--steps-per-report", type=int, default=10)
    ap.add_argument("--steps-per-eval", type=int, default=50)
    ap.add_argument("--save-every", type=int, default=100)
    ap.add_argument("--resume-adapter-file", type=Path, default=None,
                    help="resume training from a numbered checkpoint "
                         "(_adapters.safetensors); the seeded batch order makes "
                         "the replayed prefix from step 1 identical, so the tail "
                         "of a deterministic run is safe to continue")
    ap.add_argument("--seed", type=int, default=42)
    return ap


def main() -> None:
    args = build_parser().parse_args()

    import mlx_lm.lora as lora
    from mlx_lm import load
    from mlx_lm.tuner import trainer

    install_seeded_batch_order(lora, trainer, args.seed)

    print("Loading pretrained model")
    model, tokenizer = load(args.model, tokenizer_config={"trust_remote_code": True})
    tokenizer = _NoThinkingTokenizer(tokenizer)

    print("Loading datasets")
    ns = types.SimpleNamespace(**vars(args))
    ns.config = None
    ns.train = True
    ns.test = False
    ns.hf_dataset = False
    ns.mask_prompt = True
    ns.lora_parameters = {"rank": 8, "dropout": 0.0, "scale": 20.0}
    ns.optimizer = "adamw"
    ns.optimizer_config = {
        "adam": {},
        "adamw": {},
        "muon": {},
        "sgd": {},
        "adafactor": {},
    }
    ns.resume_adapter_file = args.resume_adapter_file
    ns.lr_schedule = None
    ns.test_batches = 500
    ns.clear_cache_threshold = 0
    ns.report_to = None
    ns.project_name = None
    adapter_dir = Path(ns.adapter_path)
    adapter_dir.mkdir(parents=True, exist_ok=True)
    ns.adapter_path = str(adapter_dir)

    train_set, valid_set, test_set = lora.load_dataset(ns, tokenizer)

    # Recorded next to the adapter so a run can be reproduced from its own output
    # rather than from shell history.
    (adapter_dir / "run_config.json").write_text(
        json.dumps({k: v for k, v in sorted(vars(ns).items())}, indent=2, default=str),
        encoding="utf-8",
    )

    lora.train_model(ns, model, train_set, valid_set)
    print(f"Saved final weights to {adapter_dir / 'adapters.safetensors'}")


if __name__ == "__main__":
    main()