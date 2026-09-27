"""Assemble the v2 training set from the frozen v1 train set + the multi-topic augmentation.

v2/train.jsonl      = v1 train chat examples (single-topic, unchanged)
                      + multi-topic train examples upsampled ``MT_UPSAMPLE`` times
v2/validation.jsonl = byte-identical copy of the frozen v1 validation set (comparability)
v2/test.jsonl       = byte-identical copy of the frozen v1 test set (comparability)
v2/meta.json        = composition stats + leakage statement
v2/README.md        = usage notes

The frozen v1 validation/test benchmarks are never touched, so v1 and v2 are
directly comparable on exactly the same held-out data. Multi-topic capability is
measured on the separate suite in ml/data/tune/multitopic/eval.jsonl.

**Why upsampling:** with `mlx_lm`'s iteration scheme (batch-size 2, iters 400) a
single pass sees only ~0.14 epochs; a raw 7% multi-topic share left each
multi-topic example seen <1x, so the model never learned to emit 2 topics (see
ml/reports/finetune_v2_report.md). Upsampling to ~23% of the stream still keeps
single-topic as the majority (~77% of examples) while giving every multi-topic
example meaningful gradient exposure.
"""

from __future__ import annotations

import json
from pathlib import Path

from ml.tune.build_dataset import OUT_DIR
from ml.tune.build_multitopic import OUT_DIR as MT_DIR

V2_DIR = OUT_DIR / "v2"
MT_UPSAMPLE = 3
MT_REAL_UPSAMPLE = 12


def build() -> None:
    V2_DIR.mkdir(parents=True, exist_ok=True)

    v1_train = [
        json.loads(l) for l in (OUT_DIR / "train.jsonl").read_text().splitlines()
    ]
    mt_base = [
        json.loads(l) for l in (MT_DIR / "train.jsonl").read_text().splitlines()
    ]
    mt_train = [
        ex
        for ex in mt_base
        for _ in range(MT_REAL_UPSAMPLE if ex["uid"].startswith("MT-REAL") else MT_UPSAMPLE)
    ]
    examples = v1_train + mt_train

    used_mt_uids = {ex["uid"] for ex in mt_base}
    prov = [
        json.loads(l) for l in (MT_DIR / "provenance.jsonl").read_text().splitlines()
    ]
    mt_prov = [p for p in prov if p["uid"] in used_mt_uids]

    (V2_DIR / "train.jsonl").write_text(
        "\n".join(json.dumps(ex, ensure_ascii=False) for ex in examples) + "\n"
    )

    # Frozen benchmarks copied verbatim for isolated, comparable runs.
    for split in ("validation", "test"):
        src = OUT_DIR / f"{split}.jsonl"
        (V2_DIR / f"{split}.jsonl").write_text(src.read_text())
    alias = V2_DIR / "valid.jsonl"
    if not alias.exists():
        alias.symlink_to(V2_DIR / "validation.jsonl")

    n_mt = sum(1 for p in mt_prov if p["n_topics"] >= 2)
    mt_prov_by_uid = {p["uid"]: p for p in mt_prov}
    real_n = sum(1 for p in mt_prov if not p["synthetic"])
    real_copy_n = sum(
        1 for ex in mt_train if ex["uid"].startswith("MT-REAL")
    )
    meta = {
        "built_by": "ml/tune/build_v2.py",
        "composition": {
            "v1_single_topic": len(v1_train),
            "multitopic_augmentation": len(mt_train),
            "multitopic_upsample": MT_UPSAMPLE,
            "multitopic_real_upsample": MT_REAL_UPSAMPLE,
            "real_examples": real_n,
            "real_copies": real_copy_n,
            "total": len(examples),
            "multitopic_share": round((n_mt * MT_UPSAMPLE) / len(examples), 4),
        },
        "validation": "frozen copy of v1 validation.jsonl",
        "test": "frozen copy of v1 test.jsonl",
        "multitopic_eval_suite": str(MT_DIR / "eval.jsonl"),
        "leakage": {
            "statement": (
                "multitopic augmentation built only from train-split content; "
                "no multitopic or v2 train example text appears in the frozen "
                "validation/test benchmark (see multitopic/meta.json)."
            )
        },
        "synthetic_share": round(
            sum(1 for p in mt_prov if p["synthetic"]) / len(mt_base), 4
        ) if mt_base else 0.0,
    }
    (V2_DIR / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    build()