"""QLoRA training driver for the municipal-complaint JSON task.

Wraps the mlx-lm LoRA pipeline with one consistent change: the tokenizer's
chat template is forced to ``enable_thinking=False`` so that training targets
match the inference format produced by :mod:`ml.tune.run_eval` (Qwen3's
reasoning prefix is never part of the learned format).

Run from the repo root with the MLX venv::

    .venv-mlx/bin/python -m ml.tune.run_train --iters 400 \
        --adapter-path ml/data/tune/adapters/qwen3-8b-lora
"""

from __future__ import annotations

import argparse
import types


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
    ap.add_argument("--seed", type=int, default=42)
    return ap


def main() -> None:
    args = build_parser().parse_args()
    from pathlib import Path

    import mlx_lm.lora as lora
    from mlx_lm import load

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
    ns.resume_adapter_file = None
    ns.lr_schedule = None
    ns.test_batches = 500
    ns.clear_cache_threshold = 0
    ns.report_to = None
    ns.project_name = None
    adapter_dir = Path(ns.adapter_path)
    adapter_dir.mkdir(parents=True, exist_ok=True)
    ns.adapter_path = str(adapter_dir)

    train_set, valid_set, test_set = lora.load_dataset(ns, tokenizer)
    lora.train_model(ns, model, train_set, valid_set)
    print(f"Adapters saved to {adapter_dir / 'adapters.safetensors'}")


if __name__ == "__main__":
    main()