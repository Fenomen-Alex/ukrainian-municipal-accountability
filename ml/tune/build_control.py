"""Build the v3 CONTROL dataset: the v2 data, re-emitted deterministically.

The control arm exists to answer one question: *did v2 fail on these failure modes,
or did it just get a different random stream than we would have drawn today?*

To answer it, the control has to hold the data fixed and move only the seed. So
this module re-runs the v2 assembly recipe verbatim and then **proves** the result
is byte-identical to what v2 trained on:

    ml/tune/build_v2.py assembles v1 train + multi-topic augmentation upsampled
    3x (12x for MT-REAL) and concatenates. There is no RNG anywhere in it, so
    re-running it must reproduce the file exactly. This module asserts that
    rather than assuming it.

The assertion is the point. If it ever fails, the control is no longer a control
and the experiment's comparison collapses, so the build fails loudly instead of
silently training a third configuration.

It applies **none** of the v3 changes C1-C4. That is the treatment's job
(ml/tune/build_v3.py).

    .venv/bin/python -m ml.tune.build_control

Writes ``ml/data/tune/v3/control/{train,validation,test}.jsonl`` + ``meta.json``.
Never writes to ``ml/data/tune/v2/``.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ml.tune.build_dataset import OUT_DIR as V1_DIR
from ml.tune.build_multitopic import OUT_DIR as MT_DIR
from ml.tune.build_v2 import MT_REAL_UPSAMPLE, MT_UPSAMPLE

V2_DIR = V1_DIR / "v2"
CONTROL_DIR = Path(__file__).resolve().parents[2] / "ml/data/tune/v3/control"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _jsonl(rows: list[dict]) -> str:
    return "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n"


def build() -> dict:
    CONTROL_DIR.mkdir(parents=True, exist_ok=True)

    v1_train = [json.loads(l) for l in (V1_DIR / "train.jsonl").read_text().splitlines()]
    mt_base = [json.loads(l) for l in (MT_DIR / "train.jsonl").read_text().splitlines()]

    # Identical to build_v2.build(); kept in sync by test_build_control.py.
    mt_train = [
        ex
        for ex in mt_base
        for _ in range(MT_REAL_UPSAMPLE if ex["uid"].startswith("MT-REAL") else MT_UPSAMPLE)
    ]
    examples = v1_train + mt_train

    train_path = CONTROL_DIR / "train.jsonl"
    train_path.write_text(_jsonl(examples), encoding="utf-8")

    splits = {}
    for split in ("validation", "test"):
        dst = CONTROL_DIR / f"{split}.jsonl"
        dst.write_text((V2_DIR / f"{split}.jsonl").read_text(), encoding="utf-8")
        splits[split] = {"sha256": _sha256(dst), "identical_to_v2":
                         _sha256(dst) == _sha256(V2_DIR / f"{split}.jsonl")}

    got, want = _sha256(train_path), _sha256(V2_DIR / "train.jsonl")
    meta = {
        "built_by": "ml/tune/build_control.py",
        "arm": "control",
        "purpose": (
            "v2 data, deterministic pipeline, no C1-C4. The only difference from v2 "
            "is that batch order is drawn from an explicit seed instead of whatever "
            "state the process happened to be in."
        ),
        "applies_c1_c4": False,
        "train": {
            "n": len(examples),
            "sha256": got,
            "v2_sha256": want,
            "byte_identical_to_v2": got == want,
        },
        "splits": splits,
        "composition": {
            "v1_single_topic": len(v1_train),
            "multitopic_augmentation": len(mt_train),
            "multitopic_upsample": MT_UPSAMPLE,
            "multitopic_real_upsample": MT_REAL_UPSAMPLE,
        },
    }
    (CONTROL_DIR / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    if not meta["train"]["byte_identical_to_v2"]:
        raise SystemExit(
            "control train.jsonl differs from v2. The control arm would no longer be "
            f"a control.\n  control {got}\n  v2      {want}"
        )
    return meta


if __name__ == "__main__":
    m = build()
    t = m["train"]
    print(f"control -> {CONTROL_DIR}")
    print(f"  train n={t['n']}  byte-identical to v2: {t['byte_identical_to_v2']}")
    print(f"  sha256 {t['sha256']}")
    for s, v in m["splits"].items():
        print(f"  {s:11s} identical_to_v2={v['identical_to_v2']}")
