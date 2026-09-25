"""Generate structured JSON from test complaints with a given MLX model/mode.

Modes:
  - deterministic : run the weak-label transform (baseline, no model)
  - base          : 4-bit base model, no adapter
  - lora          : base model + LoRA adapter (fused via --adapter-path)

Writes per-system JSONL predictions + a metrics report side by side.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from ml.tune.build_dataset import (
    DATA_DIR,
    OUT_DIR,
    SYSTEM_PROMPT,
    label_record,
    load_split,
)
from ml.tune.evaluate import Prediction, export_run, load_validator, parse_payload, eval_predictions


def _deterministic_predictions() -> list[Prediction]:
    preds = []
    for i, rec in enumerate(load_split("test")):
        labeled = label_record(rec)
        target = {
            "topics": [
                {
                    "domain": labeled.domain,
                    "issue": labeled.issue,
                    "object": labeled.object,
                    "requested_action": labeled.requested_action,
                    "attributes": labeled.attributes,
                }
            ]
        }
        p = Prediction(idx=i, uid=labeled.uid, raw=json.dumps(target, ensure_ascii=False))
        p.parsed = target
        p.topics = target["topics"]
        preds.append(p)
    return preds


def _model_predictions(model: str, adapter_path: str | None, max_tokens: int,
                       temperature: float) -> list[Prediction]:
    from mlx_lm import generate, load
    from mlx_lm.sample_utils import make_sampler

    sampler = make_sampler(temp=temperature)

    print(f"loading {model} ...", flush=True)
    t0 = time.time()
    model_, tokenizer = load(model, adapter_path=adapter_path)
    print(f"loaded in {time.time()-t0:.1f}s", flush=True)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": ""},
    ]

    preds = []
    for i, rec in enumerate(load_split("test")):
        messages[1]["content"] = (rec.get("content") or "").strip()
        prompt = tokenizer.apply_chat_template(messages, tokenize=False,
                                               add_generation_prompt=True,
                                               enable_thinking=False)
        t0 = time.time()
        out = generate(model_, tokenizer, prompt=prompt, max_tokens=max_tokens,
                       sampler=sampler, verbose=False)
        p = parse_payload(out)
        p.idx = i
        p.uid = rec.get("uid", "")
        p.schema_error = ""
        preds.append(p)
        print(f"[{len(preds)}/{len(load_split('test'))}] {p.uid} {time.time()-t0:.1f}s",
              flush=True)
    return preds


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=["deterministic", "base", "lora"], required=True)
    ap.add_argument("--model", default="mlx-community/Qwen3-8B-4bit")
    ap.add_argument("--adapter-path", default=None)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--temperature", type=float, default=0.0)
    args = ap.parse_args()

    targets = []
    for i, rec in enumerate(load_split("test")):
        labeled = label_record(rec)
        targets.append(
            {
                "idx": i,
                "uid": labeled.uid,
                "kind": labeled.provenance["kind"],
                "topics": [
                    {
                        "domain": labeled.domain,
                        "issue": labeled.issue,
                        "object": labeled.object,
                        "requested_action": labeled.requested_action,
                        "attributes": labeled.attributes,
                        "source_text": (rec.get("content") or "").strip(),
                    }
                ],
            }
        )

    if args.mode == "deterministic":
        preds = _deterministic_predictions()
    elif args.mode == "lora":
        preds = _model_predictions(args.model, args.adapter_path, args.max_tokens,
                                   args.temperature)
    else:
        preds = _model_predictions(args.model, None, args.max_tokens, args.temperature)

    validator = load_validator()
    metrics = eval_predictions(preds, targets, validator)

    out_dir = OUT_DIR / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    export_run(out_dir / f"{args.mode}.json", args.mode, preds, metrics)
    Path(out_dir / f"{args.mode}_targets.jsonl").write_text(
        "\n".join(json.dumps(t, ensure_ascii=False) for t in targets)
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()