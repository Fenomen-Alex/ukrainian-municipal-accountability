"""Record and verify SHA256 for the model artifacts of each arm.

v3.2 is only a delta on corrected v3 if every other arm is untouched, and the
fused models are ~4.6 GB each, so "did that change?" is not a question to
answer by eye. This writes a baseline manifest and can re-check it later:

    .venv/bin/python -m ml.tune.artifact_sha --record
    .venv/bin/python -m ml.tune.artifact_sha --verify

Symlinked artifact directories are common here (``qwen3-8b-lora-v2-fused`` is a
symlink to ``qwen3-8b-lora-v2-attempt10-fused``), so paths are resolved before
hashing -- otherwise a symlinked arm hashes as empty and looks unchanged no
matter what happens to it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from ml.tune.build_dataset import DATA_DIR

ADAPTERS = DATA_DIR / "tune" / "adapters"
MANIFEST = DATA_DIR / "tune" / "artifact_sha256.json"

#: Arms that must not move. v3.2's own output is recorded but not pinned.
ARMS = (
    "qwen3-8b-lora",
    "qwen3-8b-lora-v2",
    "qwen3-8b-lora-v2-fused",
    "qwen3-8b-lora-v3-control",
    "qwen3-8b-lora-v3-treatment",
    "qwen3-8b-lora-v3-fused",
    "qwen3-8b-lora-v3-corrected",
    "qwen3-8b-lora-v3-corrected-fused",
)

PINNED = frozenset(ARMS)
V32_ARM = "qwen3-8b-lora-v3-2"
#: v3.2's fused model -- the artifact that would actually be served, so it is
#: recorded alongside the adapter. Like the adapter it belongs to this run and
#: is therefore not pinned.
V32_FUSED_ARM = "qwen3-8b-lora-v3-2-fused"
THIS_RUN = frozenset({V32_ARM, V32_FUSED_ARM})

SKIP_SUFFIXES = (".md",)


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def scan(root: Path) -> dict:
    """Per-file SHA256 for one arm, keyed by path relative to ADAPTERS.

    ``root.resolve()`` matters: two arm directories can be the same directory
    behind a symlink, and both must hash the real bytes.
    """
    real = root.resolve()
    if not real.is_dir():
        return {"present": False}
    files = {}
    for p in sorted(real.rglob("*")):
        if not p.is_file() or p.suffix in SKIP_SUFFIXES:
            continue
        files[str(p.relative_to(real))] = {"sha256": _sha(p), "bytes": p.stat().st_size}
    combined = hashlib.sha256(
        "".join(f"{k}:{v['sha256']}" for k, v in sorted(files.items())).encode()
    ).hexdigest()
    return {"present": True, "resolved": str(real), "n_files": len(files),
            "tree_sha256": combined, "files": files}


def build(arms=ARMS) -> dict:
    return {a: scan(ADAPTERS / a) for a in arms}


def record() -> dict:
    data = build(ARMS + tuple(sorted(THIS_RUN)))
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"recorded {len(data)} arms -> {MANIFEST}")
    for a, v in data.items():
        tag = "pinned" if a in PINNED else "this run"
        if v.get("present"):
            print(f"  {a:<34} {v['n_files']:>3} files  tree {v['tree_sha256'][:16]}  ({tag})")
        else:
            print(f"  {a:<34} absent ({tag})")
    return data


def verify(require_all: bool = False) -> int:
    """Re-check every pinned arm against the recorded baseline.

    An arm whose directory is absent is *not verified*, not *unchanged*. The
    adapter directories are gitignored, so a public clone legitimately has
    almost none of them and a vanished directory there says nothing about
    corruption. Those are counted separately and never folded into "unchanged";
    pass ``require_all`` to restore the strict "every pinned arm must be present
    and identical" behaviour.
    """
    if not MANIFEST.exists():
        print("no baseline; run --record first")
        return 1
    base = json.loads(MANIFEST.read_text(encoding="utf-8"))
    now = build(tuple(base))
    bad = 0
    absent = 0
    for arm, was in base.items():
        if arm in THIS_RUN:
            continue  # this run is expected to change
        got = now.get(arm, {})
        if not was.get("present"):
            print(f"  {arm:<34} skipped (was absent)")
            continue
        if not got.get("present"):
            absent += 1
            print(f"  {arm:<34} NOT VERIFIED (absent from this checkout)")
            continue
        if got["tree_sha256"] != was["tree_sha256"]:
            print(f"  {arm:<34} FAIL: tree {was['tree_sha256'][:16]} -> {got['tree_sha256'][:16]}")
            for f, meta in was["files"].items():
                cur = got["files"].get(f)
                if cur is None:
                    print(f"      removed: {f}")
                elif cur["sha256"] != meta["sha256"]:
                    print(f"      changed: {f}")
            bad += 1
        else:
            print(f"  {arm:<34} unchanged ({got['n_files']} files)")
    if bad:
        print(f"{bad} REFERENCE ARM(S) CHANGED")
        return 1
    if absent:
        print(
            f"ALL PRESENT REFERENCE ARMS UNCHANGED "
            f"({absent} NOT PRESENT IN THIS CHECKOUT - NOT VERIFIED)"
        )
        if require_all:
            print("--require-all: treating absent pinned arms as a failure")
            return 1
        return 0
    print("ALL REFERENCE ARMS UNCHANGED")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", action="store_true")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument(
        "--require-all",
        action="store_true",
        help="fail if any pinned arm is absent, not just on a tree mismatch",
    )
    a = ap.parse_args()
    if a.record:
        record()
    elif a.verify:
        raise SystemExit(verify(require_all=a.require_all))
    else:
        ap.print_help()
