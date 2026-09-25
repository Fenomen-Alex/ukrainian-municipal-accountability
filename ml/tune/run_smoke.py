"""Run the 20-case smoke suite through a model and save raw+parsed outputs.

Works for the fused fine-tuned MLX model (a local directory) and the original
base HF/MLX repo (``mlx-community/Qwen3-8B-4bit``). Uses exactly the same
prompt contract as training/: the build_dataset SYSTEM_PROMPT and Qwen3 chat
template with ``enable_thinking=False``.

Usage::

    .venv-mlx/bin/python -m ml.tune.run_smoke \\
        --model ml/data/tune/adapters/qwen3-8b-lora-fused \\
        --tag finetuned
    .venv-mlx/bin/python -m ml.tune.run_smoke \\
        --model mlx-community/Qwen3-8B-4bit \\
        --tag base

Writes ``ml/data/tune/smoke/results/<tag>/smoke_results.jsonl`` with one JSON
object per case: id, category, expected_domains, text, raw, parsed, schema_error.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from ml.tune.build_dataset import SYSTEM_PROMPT
from ml.tune.evaluate import parse_payload, validate_schema, load_validator

SMOKE_ROOT = Path(__file__).resolve().parent.parent / "data" / "tune" / "smoke"
SMOKE_CASES_PATH = SMOKE_ROOT / "smoke_cases.json"
RESULTS_ROOT = SMOKE_ROOT / "results"


def load_smoke_cases(path: Path = SMOKE_CASES_PATH) -> list[dict]:
    return json.loads(path.read_text())


def run_smoke(model: str, tag: str, max_tokens: int = 800,
              temperature: float = 0.0) -> Path:
    from mlx_lm import generate, load
    from mlx_lm.sample_utils import make_sampler

    sampler = make_sampler(temp=temperature)
    print(f"loading {model} ({tag}) ...", flush=True)
    t0 = time.time()
    model_, tokenizer = load(model)
    print(f"loaded in {time.time()-t0:.1f}s", flush=True)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": ""},
    ]
    validator = load_validator()

    out_dir = RESULTS_ROOT / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for case in load_smoke_cases():
        messages[1]["content"] = case["text"].strip()
        prompt = tokenizer.apply_chat_template(messages, tokenize=False,
                                               add_generation_prompt=True,
                                               enable_thinking=False)
        t0 = time.time()
        raw = generate(model_, tokenizer, prompt=prompt, max_tokens=max_tokens,
                       sampler=sampler, verbose=False)
        pred = parse_payload(raw)
        ok, err = validate_schema(pred, validator)
        rows.append({
            "id": case["id"],
            "category": case["category"],
            "expected_domains": case["expected_domains"],
            "text": case["text"],
            "raw": raw,
            "parsed": pred.parsed,           # None if unparseable
            "schema_valid": ok,
            "schema_error": err if not ok else "",
        })
        print(f"[{case['id']}] {time.time()-t0:.1f}s "
              f"schema={'OK' if ok else 'FAIL'}", flush=True)

    out = out_dir / "smoke_results.jsonl"
    out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")
    print(f"saved {len(rows)} results to {out}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--max-tokens", type=int, default=800)
    ap.add_argument("--temperature", type=float, default=0.0)
    args = ap.parse_args()
    run_smoke(args.model, args.tag, args.max_tokens, args.temperature)


if __name__ == "__main__":
    main()