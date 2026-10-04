"""Audit what the public repository actually distributes.

Deterministic and offline. Reads only files tracked by git at the current
checkout, so the findings are reproducible from any clone at a known revision.

For every tracked ``.jsonl`` this records:

* its class (source corpus, benchmark case set, generated model output, ...)
* whether it carries the verbatim source schema (raw 20-column rows)
* whether it carries verbatim source free text
* phone-like and email-like hits, using the same heuristics as
  ``build_dataset._redact_pii`` plus an email pattern
* whether tests read it at runtime

The counts are HEURISTIC. They are an exposure indicator, not a privacy
audit, and they support no legal conclusion.

Usage::

    python -m ml.tune.audit_public_data            # table + JSON manifest
    python -m ml.tune.audit_public_data --json     # manifest only
    python -m ml.tune.audit_public_data --check    # verifier: manifest is current
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MANIFEST = ROOT / "ml" / "data" / "tune" / "public_data_manifest.json"

# The raw municipal schema as it arrives from data.kr-rada.gov.ua.
VERBATIM_SOURCE_KEYS = {
    "uid",
    "receivedDateTime",
    "type",
    "kind",
    "content",
    "accrualMethod",
}

EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")

PHONE = re.compile(
    r"(?:\+?3\s?8[0-9\s\-(]{8,14}"
    r"|\b[0-9]{3}[-\s][0-9]{3}[-\s][0-9]{2}[-\s][0-9]{2}\b"
    r"|\b\d{10}\b)"
)

# Directories that hold model-generated text rather than source rows.
GENERATED_PREFIXES = (
    "ml/data/tune/smoke/results/",
    "ml/data/tune/decoding_probe/",
)

BENCHMARK_PREFIXES = (
    "ml/data/tune/eval_v3/",
)

PROVENANCE_META = (
    "provenance.jsonl",
    "annotation_set.jsonl",
)

CHAT_FILES = frozenset({"messages", "text", "raw", "content"})


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def tracked_jsonl() -> list[str]:
    return sorted(_git("ls-files", "*.jsonl").splitlines())


def tracked_symlinks() -> dict[str, str]:
    out = {}
    for line in _git("ls-files", "-s").splitlines():
        mode, _sha, stage, path = line.split(maxsplit=3)
        if mode == "120000":
            out[path] = (ROOT / path).read_text(encoding="utf-8").strip()
    return out


def classify(rel: str) -> str:
    if rel.startswith(GENERATED_PREFIXES):
        return "generated_model_output"
    if rel.startswith(BENCHMARK_PREFIXES):
        return "benchmark_case_set"
    if rel.endswith(PROVENANCE_META):
        return "annotation_or_provenance_metadata"
    return "source_derived_corpus"


def iter_strings(obj):
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from iter_strings(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from iter_strings(v)


def scan(path: pathlib.Path) -> dict:
    raw = path.read_bytes()
    rows = 0
    keys: set[str] = set()
    phone_rows = 0
    email_rows = 0
    phone_hits = 0
    email_hits = 0
    for line in raw.decode("utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows += 1
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(rec, dict):
            keys.update(rec.keys())
        texts = [s for s in iter_strings(rec) if s and s in CHAT_FILES or (s and len(s) > 12)]
        blob = "\n".join(texts)
        ph = PHONE.findall(blob)
        em = EMAIL.findall(blob)
        if ph:
            phone_rows += 1
            phone_hits += len(ph)
        if em:
            email_rows += 1
            email_hits += len(em)
    verbatim_keys = sorted(VERBATIM_SOURCE_KEYS & keys)
    return {
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "rows": rows,
        "keys_sample": sorted(keys)[:8],
        "verbatim_source_schema": len(verbatim_keys) >= 5,
        "verbatim_source_keys": verbatim_keys,
        "carries_free_text": bool({"text", "content", "messages", "raw"} & keys),
        "phone_like_rows": phone_rows,
        "phone_like_hits": phone_hits,
        "email_like_rows": email_rows,
        "email_like_hits": email_hits,
    }


def test_dependencies() -> dict[str, list[str]]:
    """Tests that reference a corpus.

    Tests reach the corpora through module-level constants such as
    ``DATA / "tune" / "multitopic"`` and then ``MT_DIR / "train.jsonl"``, so
    no literal full path appears in the test source. A corpus is therefore
    credited to a test when the test mentions the corpus's basename together
    with a literal from its parent-directory chain.

    This is a static approximation. The authoritative coupling is the
    *measured* result in ``test_coupling`` below, obtained by hiding the
    source-derived corpora and running the suite.
    """
    deps: dict[str, list[str]] = {}
    bodies = {
        py.name: py.read_text(encoding="utf-8") for py in sorted((ROOT / "ml" / "tests").glob("*.py"))
    }
    for rel in tracked_jsonl():
        parts = rel.split("/")
        basename = parts[-1]
        dir_parts = [p for p in parts[1:-1] if p != "data"]
        for name, body in bodies.items():
            tail = "/".join(parts[-2:])
            if tail in body or (basename in body and any(d in body for d in dir_parts)):
                deps.setdefault(rel, []).append(name)
    return {k: sorted(set(v)) for k, v in deps.items()}


# Measured, not assumed. Hiding the 12 source_derived_corpus files at
# revision 15a44be and running `pytest ml/tests -q --continue-on-collection-errors`
# gave "48 failed, 419 passed, 14 skipped, 1 error". Recorded so the manifest
# states the cost of untracking as a measured number rather than a claim.
TEST_COUPLING = {
    "method": "hide source_derived_corpus files, run pytest ml/tests -q --continue-on-collection-errors",
    "measured_at_revision": "15a44beab21a",
    "tests_failing": 48,
    "collection_errors": 1,
    "tests_passing": 419,
    "files_needing_skip_guards": [
        "ml/tests/test_tune_dataset.py",
        "ml/tests/test_v2_augmentation.py",
        "ml/tests/test_v3_changes.py",
    ],
    "note": (
        "The suite cannot run without the tracked corpora. Removing them is "
        "not a silent change: it requires skip guards in three test files."
    ),
}


CLASS_NOTES = {
    "source_derived_corpus": (
        "rows derived from the CC BY municipal source; may retain contact "
        "details because redaction is known incomplete"
    ),
    "benchmark_case_set": (
        "evaluation case definitions needed by the test suite to run offline"
    ),
    "generated_model_output": (
        "model/eval output produced locally; not source rows"
    ),
    "annotation_or_provenance_metadata": (
        "annotation and lineage metadata; may quote source uids or text"
    ),
}


def build() -> dict:
    symlinks = tracked_symlinks()
    deps = test_dependencies()
    files = {}
    for rel in tracked_jsonl():
        path = ROOT / rel
        info = scan(path)
        info["class"] = classify(rel)
        info["symlink_target"] = symlinks.get(rel)
        info["read_by_tests"] = sorted(set(deps.get(rel, [])))
        info["note"] = CLASS_NOTES[info["class"]]
        files[rel] = info

    # Several tracked corpora are byte-identical copies or symlinks of one
    # another. Counting rows per path would double-count the same rows, so
    # also report exposure per DISTINCT content hash.
    by_sha: dict[str, list[str]] = {}
    for rel, v in files.items():
        if v["symlink_target"]:
            continue  # alias, not independent content
        by_sha.setdefault(v["sha256"], []).append(rel)
    distinct = list(by_sha.values())
    total_rows = sum(v["rows"] for v in files.values())
    distinct_rows = sum(files[g[0]]["rows"] for g in distinct)

    totals = {
        "tracked_jsonl": len(files),
        "distinct_files": sum(1 for v in files.values() if not v["symlink_target"]),
        "distinct_content_hashes": len(distinct),
        "bytes": sum(v["bytes"] for v in files.values()),
        "rows": total_rows,
        "distinct_rows": distinct_rows,
        "duplicate_copies": total_rows - distinct_rows,
        "phone_like_rows": sum(v["phone_like_rows"] for v in files.values()),
        "email_like_rows": sum(v["email_like_rows"] for v in files.values()),
        "distinct_phone_like_rows": sum(files[g[0]]["phone_like_rows"] for g in distinct),
        "distinct_email_like_rows": sum(files[g[0]]["email_like_rows"] for g in distinct),
        "verbatim_schema_files": sum(
            1 for v in files.values() if v["verbatim_source_schema"]
        ),
        "symlinks": len(symlinks),
        "read_by_tests": sum(1 for v in files.values() if v["read_by_tests"]),
    }
    by_class: dict[str, dict] = {}
    for info in files.values():
        agg = by_class.setdefault(
            info["class"], {"files": 0, "bytes": 0, "rows": 0, "phone_like_rows": 0, "email_like_rows": 0}
        )
        agg["files"] += 1
        agg["bytes"] += info["bytes"]
        agg["rows"] += info["rows"]
        agg["phone_like_rows"] += info["phone_like_rows"]
        agg["email_like_rows"] += info["email_like_rows"]

    # Public-data boundary (see ml/tune/public_data_policy.py). A tracked path
    # in NOT_PUBLIC is a hard failure: these corpora are reproducible from the
    # official source and must not be distributed.
    from ml.tune import public_data_policy as policy

    forbidden_tracked = sorted(p for p in files if policy.is_forbidden(p))
    reviewed_absent = sorted(
        p for p in policy.REVIEWED_PUBLIC_FRAGMENTS if p not in files and p.endswith((".jsonl",))
    )
    boundary = {
        "must_not_track_count": len(policy.NOT_PUBLIC),
        "forbidden_tracked": forbidden_tracked,
        "reviewed_public_fragments": sorted(policy.REVIEWED_PUBLIC_FRAGMENTS),
        "reviewed_fragments_absent": reviewed_absent,
        "compliant": not forbidden_tracked,
    }

    return {
        "revision": _git("rev-parse", "HEAD"),
        "totals": totals,
        "by_class": by_class,
        "boundary": boundary,
        "duplicate_groups": [
            {"sha256": sha, "paths": sorted(paths)} for sha, paths in sorted(by_sha.items()) if len(paths) > 1
        ],
        "test_coupling": TEST_COUPLING,
        "files": files,
        "caveat": (
            "Heuristic scan. Phone/email patterns are indicative, not a "
            "privacy audit, and support no legal conclusion."
        ),
    }


def render(manifest: dict) -> None:
    t = manifest["totals"]
    print(f"public data audit @ {manifest['revision'][:12]}")
    print(f"  tracked .jsonl          {t['tracked_jsonl']} ({t['distinct_files']} distinct, {t['symlinks']} symlink)")
    print(f"  distinct content        {t['distinct_content_hashes']} hashes ({t['duplicate_copies']} duplicated rows)")
    print(f"  total size              {t['bytes']/1e6:.1f} MB")
    print(f"  rows                    {t['rows']:,} total / {t['distinct_rows']:,} distinct")
    print(f"  verbatim-source schema  {t['verbatim_schema_files']} files")
    print(f"  read by tests           {t['read_by_tests']} files")
    print()
    print(f"  {'class':<34}{'files':>6}{'MB':>8}{'rows':>8}{'phone':>7}{'email':>7}")
    for name, a in sorted(manifest["by_class"].items()):
        print(f"  {name:<34}{a['files']:>6}{a['bytes']/1e6:>8.2f}{a['rows']:>8}{a['phone_like_rows']:>7}{a['email_like_rows']:>7}")
    print()
    if manifest.get("duplicate_groups"):
        print("  byte-identical duplicate groups (rows counted once above):")
        for g in manifest["duplicate_groups"]:
            print(f"    {g['sha256'][:12]}  {'  ==  '.join(g['paths'])}")
        print()
    print(f"  {'file':<50}{'MB':>7}{'rows':>7}{'phone':>7}{'email':>7}{'src?':>6}{'tests':>7}")
    for rel, v in sorted(manifest["files"].items()):
        print(
            f"  {rel:<50}{v['bytes']/1e6:>7.2f}{v['rows']:>7}"
            f"{v['phone_like_rows']:>7}{v['email_like_rows']:>7}"
            f"{'yes' if v['verbatim_source_schema'] else '-':>6}"
            f"{len(v['read_by_tests']) or '-':>7}"
        )
    print()
    print(f"  PII indicator (per path)      phone {t['phone_like_rows']}  email {t['email_like_rows']}")
    print(f"  PII indicator (per distinct)  phone {t['distinct_phone_like_rows']}  email {t['distinct_email_like_rows']}")


def check() -> int:
    current = build()
    if not MANIFEST.exists():
        print("FAIL: ml/data/tune/public_data_manifest.json missing")
        return 1
    stored = json.loads(MANIFEST.read_text(encoding="utf-8"))
    fails = []
    if current["boundary"]["forbidden_tracked"]:
        for rel in current["boundary"]["forbidden_tracked"]:
            fails.append(f"forbidden path tracked in public repo: {rel}")
    # ``revision_head_tree`` is informational: it records the tree the manifest
    # was generated from. It is not compared here because the manifest is
    # committed together with the paths it describes (its own blob changes the
    # tree). File facts and the boundary are what must stay current.
    if sorted(stored.get("files", {})) != sorted(current["files"]):
        fails.append("tracked .jsonl set differs from manifest")
    for rel, v in current["files"].items():
        s = stored.get("files", {}).get(rel)
        if not s:
            continue
        for k in ("sha256", "rows", "phone_like_rows", "email_like_rows"):
            if s.get(k) != v[k]:
                fails.append(f"{rel}: {k} {s.get(k)} != {v[k]}")
    if fails:
        print(f"public data audit: {len(fails)} mismatch(es)")
        for f in fails[:10]:
            print("  FAIL:", f)
        return 1
    t = current["totals"]
    print(
        f"public data audit: OK ({t['tracked_jsonl']} jsonl, "
        f"{t['bytes']/1e6:.1f} MB, {t['rows']} rows, "
        f"{t['phone_like_rows']} phone-like, {t['email_like_rows']} email-like)"
    )
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", action="store_true", help="print manifest JSON")
    ap.add_argument("--check", action="store_true", help="verify manifest is current")
    ap.add_argument("--write", action="store_true", help="write the manifest")
    args = ap.parse_args()
    manifest = build()
    if args.check:
        return check()
    if args.write:
        manifest["revision_head_tree"] = _git("rev-parse", "HEAD^{tree}")
        MANIFEST.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"wrote {MANIFEST.relative_to(ROOT)}")
        return 0
    if args.json:
        print(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True))
    else:
        render(manifest)
    return 0


if __name__ == "__main__":
    sys.exit(main())