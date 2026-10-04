"""Tests for the public-data boundary policy (``ml/tune/public_data_policy.py``).

The policy is the single source of truth for what may be tracked publicly. Two
things are easy to break silently and are guarded here:

1. A tracked file that contains complaint-derived text but is *not* listed in
   ``REVIEWED_PUBLIC_FRAGMENTS`` is an unaudited exception. The policy claims
   every exception is deliberate and justified; an unlisted one falsifies that.
   ``audit_public_data.py`` only inspects tracked ``*.jsonl``, so ``.json``
   artifacts such as ``ml/data/gold/gemini_batches/`` fall outside it entirely.
   These tests close that gap.

2. ``ml/data/gold/gemini_batches/`` is a reviewed exception only because it
   carries **no incremental source-data exposure**: every record's text is
   byte-identical to a record in the reviewed ``annotation_set.jsonl``. If that
   ever stops being true, the justification in the policy becomes false and the
   exception must be revoked rather than silently relied upon.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from ml.tune import public_data_policy as policy

ROOT = Path(__file__).resolve().parents[2]
GOLD = Path("ml/data/gold")
BATCH_DIR = GOLD / "gemini_batches"
ANNOTATION_SET = GOLD / "annotation_set.jsonl"

#: Complaint-bearing file extensions the manifest audit does not scan.
UNAUDITED_SUFFIXES = (".json",)

#: Record fields that hold complaint or generation free-text. Used to decide
#: whether a ``.json`` file is complaint-bearing without guessing from its name.
COMPLAINT_TEXT_FIELDS = {"text", "issue", "content", "raw"}


def _tracked(rel: str) -> bool:
    out = subprocess.run(
        ["git", "ls-files", "--error-unmatch", rel],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    return out.returncode == 0


def _tracked_under(prefix: str) -> list[str]:
    out = subprocess.run(
        ["git", "ls-files", prefix], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout
    return [p for p in out.splitlines() if p]


def _annotation_records() -> list[dict]:
    with ANNOTATION_SET.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _batch_records() -> list[tuple[str, dict]]:
    records: list[tuple[str, dict]] = []
    for path in sorted(BATCH_DIR.glob("batch-*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for rec in payload["records"]:
            records.append((payload["batch_id"], rec))
    return records


# --- the boundary itself -------------------------------------------------


def test_forbidden_corpora_are_not_tracked() -> None:
    """None of the source-derived corpora may be tracked."""
    tracked_forbidden = sorted(p for p in policy.NOT_PUBLIC if _tracked(p))
    assert tracked_forbidden == [], f"forbidden corpora are tracked: {tracked_forbidden}"


def test_no_path_is_both_forbidden_and_reviewed() -> None:
    """The two lists must not overlap; a contradiction hides a real decision."""
    overlap = sorted(policy.NOT_PUBLIC & set(policy.REVIEWED_PUBLIC_FRAGMENTS))
    assert overlap == [], f"paths classified both ways: {overlap}"


def test_every_reviewed_fragment_states_a_justification() -> None:
    for rel, reason in policy.REVIEWED_PUBLIC_FRAGMENTS.items():
        assert reason.strip(), f"{rel} is an exception with no recorded justification"


# --- the .json blind spot ------------------------------------------------


def test_tracked_complaint_bearing_json_is_classified() -> None:
    """Tracked ``.json`` data files carrying complaint text must be classified.

    ``audit_public_data.py`` scans only tracked ``*.jsonl``, so a tracked
    ``.json`` file is otherwise neither measured nor justified. Every such file
    must be either a reviewed exception or explicitly recorded as pending in
    ``PENDING_PUBLIC_CLASSIFICATION`` -- neither state is a silent pass.
    """
    unclassified = []
    for rel in _complaint_bearing_json():
        if policy.reviewed_reason(rel) is None and policy.pending_reason(rel) is None:
            unclassified.append(rel)
    assert unclassified == [], (
        "tracked complaint-bearing .json file(s) with neither a reviewed exception "
        f"nor a recorded pending classification: {unclassified}"
    )


def _complaint_bearing_json() -> list[str]:
    """Tracked ``.json`` files under ``ml/data`` that carry complaint free-text."""
    found = []
    for rel in _tracked_under("ml/data/"):
        if not rel.endswith(UNAUDITED_SUFFIXES):
            continue
        try:
            payload = json.loads((ROOT / rel).read_text(encoding="utf-8"))
        except (ValueError, OSError):
            continue  # not a JSON data file we can interpret
        if _free_text(payload):
            found.append(rel)
    return found


def _free_text(obj, key=None, out=None):
    """Yield long strings held under a complaint-bearing field name."""
    out = [] if out is None else out
    if isinstance(obj, dict):
        for k, v in obj.items():
            _free_text(v, k, out)
    elif isinstance(obj, list):
        for v in obj:
            _free_text(v, key, out)
    elif isinstance(obj, str) and key in COMPLAINT_TEXT_FIELDS and len(obj) > 40:
        out.append(obj)
    return out


def test_pending_classification_is_non_empty_and_justified() -> None:
    """The pending set must stay explicit and explained.

    It is expected to be non-empty while the boundary is unresolved. What is not
    acceptable is an unrecorded entry, or an entry that tracks nothing.
    """
    for rel, reason in policy.PENDING_PUBLIC_CLASSIFICATION.items():
        assert reason.strip(), f"{rel} is pending with no recorded reason"
        assert "UNRESOLVED" in reason, f"{rel} must be marked UNRESOLVED"
        assert _tracked_under(rel), f"pending entry {rel} tracks nothing"


def test_pending_and_reviewed_sets_do_not_overlap() -> None:
    overlap = set(policy.REVIEWED_PUBLIC_FRAGMENTS) & set(policy.PENDING_PUBLIC_CLASSIFICATION)
    assert overlap == set(), f"paths both approved and pending: {overlap}"


def test_gemini_batches_are_tracked_and_classified() -> None:
    tracked = _tracked_under(str(BATCH_DIR))
    assert tracked, f"no tracked files under {BATCH_DIR}"
    for rel in tracked:
        assert policy.reviewed_reason(rel) is not None, f"{rel} is not a reviewed exception"


def test_directory_style_reviewed_prefixes_actually_match() -> None:
    """Regression guard: directory entries must match the paths they cover.

    ``reviewed_reason`` previously required the prefix to end with ``/``, while
    every directory entry in the mapping is written without one. That silently
    disabled the exception for every file beneath it.
    """
    directory_entries = [
        rel for rel in policy.REVIEWED_PUBLIC_FRAGMENTS if rel.endswith("/")
        or (ROOT / rel).is_dir()
    ]
    assert directory_entries, "expected at least one directory-style exception"
    for rel in directory_entries:
        covered = _tracked_under(rel)
        assert covered, f"directory exception {rel} matches no tracked file"
        for child in covered:
            assert policy.reviewed_reason(child) is not None, (
                f"{child} is under the reviewed exception {rel} but reviewed_reason "
                "does not match it"
            )


# --- the substantive justification: zero incremental exposure ------------


def test_gemini_batches_carry_no_text_absent_from_the_annotation_set() -> None:
    """Every batch record's text already ships in the reviewed annotation set.

    This is what makes ``ml/data/gold/gemini_batches`` a safe exception rather
    than a second copy of source-derived data. If it fails, the exception is no
    longer justified and the data must be reclassified, not re-documented.
    """
    gold_text = {rec["text"] for rec in _annotation_records()}
    unknown = [
        (batch, rec["id"])
        for batch, rec in _batch_records()
        if rec["text"] not in gold_text
    ]
    assert unknown == [], (
        f"{len(unknown)} batch record(s) carry text absent from "
        f"{ANNOTATION_SET}, which breaks the reviewed-exception justification: {unknown[:5]}"
    )


def test_gemini_batch_ids_partition_the_annotation_set() -> None:
    """The batches must be a partition of the gold set: no extra, no missing."""
    gold_ids = {rec["id"] for rec in _annotation_records()}
    batch_ids = [rec["id"] for _, rec in _batch_records()]

    assert len(batch_ids) == len(set(batch_ids)), "duplicate record id across batches"
    assert set(batch_ids) <= gold_ids, "batch contains ids absent from the gold set"
    assert set(batch_ids) == gold_ids, (
        "batches do not cover the annotation set exactly "
        f"(missing {len(gold_ids - set(batch_ids))}, extra {len(set(batch_ids) - gold_ids)})"
    )


def test_gemini_batch_records_carry_only_declared_fields() -> None:
    """Records must not smuggle undeclared fields (e.g. raw contact data)."""
    for batch, rec in _batch_records():
        assert set(rec) == {"id", "text", "source_kind"}, (
            f"{batch} record {rec['id']} has unexpected fields {sorted(set(rec) - {'id', 'text', 'source_kind'})}"
        )


@pytest.mark.parametrize("rel", sorted(policy.REVIEWED_PUBLIC_FRAGMENTS))
def test_reviewed_fragment_is_actually_tracked(rel: str) -> None:
    """A listed exception that is not tracked is stale policy."""
    prefix = rel if rel.endswith("/") else rel
    if rel.endswith((".jsonl", ".json")):
        assert _tracked(rel), f"reviewed fragment {rel} is listed but not tracked"
    else:
        assert _tracked_under(prefix), f"reviewed prefix {rel} matches no tracked file"