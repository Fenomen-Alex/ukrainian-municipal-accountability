"""Target-token exposure per stream, for the v3.2 report.

Corpus composition is measured in rows, but what the model actually learns from
is *target tokens*: the assistant message, masked prompt excluded. A stream can
be a large share of rows and a small share of gradient, so the report quotes
both. The previous analysis turned on exactly this -- MT held ~41% of target
tokens across v2, previous v3 and corrected v3, which is why "the model saw
less multitopic data" was rejected as the cause.

Needs the MLX environment (tokenizer), so it is kept out of the builder:

    .venv-mlx/bin/python -m ml.tune.exposure_v3_2

Writes ``ml/data/tune/v3/treatment_v3_2/exposure.json``.
"""

from __future__ import annotations

import json
from pathlib import Path

from ml.tune.build_dataset import DATA_DIR

V3 = DATA_DIR / "tune" / "v3"
CORRECTED = V3 / "treatment"
V32 = V3 / "treatment_v3_2"
MODEL = "mlx-community/Qwen3-8B-4bit"

ARMS = (
    ("previous_v3", Path("/tmp/mtinv/v3prev_train.jsonl")),
    ("corrected_v3", CORRECTED / "train.jsonl"),
    ("v3_2", V32 / "train.jsonl"),
)


def stream_of(uid: str) -> str:
    u = uid.split("#")[0]
    if u.startswith("MT-SYN-SO"):
        return "same_object"
    if u.startswith("V3-TERSE"):
        return "terse"
    if u.startswith("MT-REAL"):
        return "mt_real"
    if u.startswith("MT-SYN"):
        return "mt_synth"
    return "single"


def _load(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def measure(tokenizer, rows: list[dict]) -> dict:
    per: dict[str, int] = {}
    for r in rows:
        target = r["messages"][2]["content"]
        per[stream_of(r["uid"])] = per.get(stream_of(r["uid"]), 0) + len(
            tokenizer.encode(target))
    total = sum(per.values())
    shares = {k: round(v / total, 4) for k, v in sorted(per.items())}
    mt = sum(v for k, v in per.items()
             if k in ("mt_real", "mt_synth", "same_object"))
    return {
        "target_tokens": {k: per[k] for k in sorted(per)},
        "target_token_share": shares,
        "total_target_tokens": total,
        "multitopic_target_tokens": mt,
        "multitopic_target_share": round(mt / total, 4),
    }


def build() -> dict:
    from mlx_lm.tokenizer_utils import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL)
    out = {"model": MODEL, "note": (
        "Target tokens are the assistant message only (prompt is masked). "
        "Stream attribution is by uid prefix, matching the file order the "
        "MLX pairer sees.")}
    for name, path in ARMS:
        if not path.exists():
            out[name] = {"skipped": f"{path} not present"}
            continue
        out[name] = measure(tok, _load(path))
        out[name]["train_sha256"] = __import__("hashlib").sha256(
            path.read_bytes()).hexdigest()
        print(f"{name:<13} total {out[name]['total_target_tokens']:>10,}  "
              f"MT {out[name]['multitopic_target_share']*100:5.2f}%")
        for k, v in out[name]["target_token_share"].items():
            print(f"    {k:<12} {v*100:5.2f}%")
    (V32 / "exposure.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


if __name__ == "__main__":
    build()