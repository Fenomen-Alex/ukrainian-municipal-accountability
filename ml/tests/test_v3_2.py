"""Integrity gates for the v3.2 same-object intervention.

v3.2 is a single-change experiment on top of the corrected v3 arm: one
same-object two-topic stream is appended. That is only interpretable if the
change is *provably* confined, so most of these tests are negative -- they
assert that nothing else moved. If any of them fails, the arm is not comparable
to corrected v3 and its scores mean nothing.

    * the first 9,447 rows are byte-identical to corrected v3 (C1-C4 intact)
    * the new stream teaches the eval_v3 category-E shape and nothing else
    * every new row is two topics, one shared object, two different domains
    * sources are train-only and nothing frozen held out leaks in
    * the rebuild is deterministic and lands on a pinned SHA
    * the total is even, so iters == n_batches and no row is dropped or replayed

Run: ``.venv/bin/python -m pytest ml/tests/test_v3_2.py -q``
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from ml.cleaner import normalize_text
from ml.tune.build_dataset import DATA_DIR

ROOT = Path(__file__).resolve().parents[2]
V3 = DATA_DIR / "tune" / "v3"
CORRECTED = V3 / "treatment"
V32 = V3 / "treatment_v3_2"

#: Pinned so a silent change to the sampler cannot masquerade as "the same
#: corpus". Bump only together with an intentional re-run and a note in the
#: report.
TRAIN_SHA = "c304c374af6aa9b9da06a3c47bf47881cd27249ea9b389abd2e1ecdf43ac622e"

#: The reference arm. v3.2 is only a delta if this is untouched.
CORRECTED_SHA = "22e85d80b7a7abc12d0629ffc41fd125a2b718125e01f62fe9cbcd92815ef9f2"

#: Corrected v3 has 9,447 rows, so v3.2 adds SAME_OBJECT_N on top.
N_CORRECTED = 9447
N_SAME_OBJECT = 261
N_TOTAL = N_CORRECTED + N_SAME_OBJECT


def _load(p: Path) -> list[str]:
    return [l for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def _rows(p: Path) -> list[dict]:
    return [json.loads(l) for l in _load(p)]


def _require(path: Path, builder: str = "ml.tune.build_v3_2"):
    if not (path.parent / "meta.json").exists():
        pytest.skip(f"run `{builder}` first: {path.parent} not built")
    return path


def _topics(chat: dict) -> list[dict]:
    return json.loads(chat["messages"][2]["content"])["topics"]


def _user(chat: dict) -> str:
    return chat["messages"][1]["content"]


@pytest.fixture(scope="module")
def v32() -> list[dict]:
    return _rows(_require(V32 / "train.jsonl"))


@pytest.fixture(scope="module")
def so(v32: list[dict]) -> list[dict]:
    return [e for e in v32 if e["uid"].startswith("MT-SYN-SO")]


# --------------------------------------------------------------------------- #
# 1. the change is confined: corrected v3 is untouched
# --------------------------------------------------------------------------- #


def test_corrected_arm_is_byte_identical_to_the_recorded_sha():
    """Corrected v3 is a reference arm; if it moved, v3.2 is not a delta."""
    from ml.tune.build_v3_2 import _sha256

    assert _sha256(CORRECTED / "train.jsonl") == CORRECTED_SHA


def test_v32_extends_corrected_v3_by_exactly_the_new_stream():
    """C1-C4 byte-identical: every corrected row is still there, unchanged,
    in the same order, and the new rows come after."""
    corrected = _load(_require(CORRECTED / "train.jsonl"))
    v32 = _load(V32 / "train.jsonl")
    assert len(corrected) == N_CORRECTED
    assert len(v32) == N_TOTAL
    assert v32[:N_CORRECTED] == corrected, "a C1-C4 row changed in v3.2"


def test_new_rows_are_appended_last_and_do_not_interleave():
    v32 = _load(V32 / "train.jsonl")
    first = next(i for i, l in enumerate(v32) if json.loads(l)["uid"].startswith("MT-SYN-SO"))
    assert first == N_CORRECTED, "the new stream must be contiguous at the end"


def test_v32_adds_no_duplicate_prompts():
    """Each pair is drawn once; a duplicated prompt would over-weight one
    street and inflate the intervention beyond its nominal size."""

    def dupes(lines: list[str]) -> int:
        texts = [normalize_text(json.loads(l)["messages"][1]["content"]) for l in lines]
        return len(texts) - len(set(texts))

    assert dupes(_load(V32 / "train.jsonl")) == dupes(_load(CORRECTED / "train.jsonl"))


def test_validation_and_test_are_frozen_copies_of_v2():
    v2 = DATA_DIR / "tune" / "v2"
    for split in ("validation", "test"):
        assert _load(V32 / f"{split}.jsonl") == _load(v2 / f"{split}.jsonl")


# --------------------------------------------------------------------------- #
# 2. determinism and pinned corpus
# --------------------------------------------------------------------------- #


def test_rebuild_is_deterministic_and_hits_the_pinned_sha():
    """Rebuilding must reproduce the corpus byte-for-byte, or the trained arm
    cannot be reproduced from its recipe."""
    from ml.tune.build_v3_2 import _sha256

    before = (V32 / "train.jsonl").read_bytes()
    subprocess.run([sys.executable, "-m", "ml.tune.build_v3_2"],
                   cwd=ROOT, check=True, capture_output=True)
    assert (V32 / "train.jsonl").read_bytes() == before
    assert _sha256(V32 / "train.jsonl") == TRAIN_SHA


def test_total_is_even_so_iters_equals_batches():
    """The point of the odd stream size: an even total makes
    ``iters == len(batches)`` exact, so training sees every row once with no
    dropped tail and no wrap into a reshuffled second epoch."""
    meta = json.loads((_require(V32 / "meta.json")).read_text(encoding="utf-8"))
    c = meta["composition"]
    assert c["total"] == N_TOTAL
    assert c["total"] % 2 == 0
    assert c["n_batches"] == N_TOTAL // 2
    assert c["iters_ceil"] == c["n_batches"]
    assert c["iters_equals_batches"] is True


def test_meta_counts_are_self_consistent():
    meta = json.loads((_require(V32 / "meta.json")).read_text(encoding="utf-8"))
    c, s = meta["composition"], meta["same_object_stream"]
    assert s["n_rows_after_upsample"] == N_SAME_OBJECT
    assert s["upsample"] == 1
    assert c["same_object_rows"] == N_SAME_OBJECT
    assert (c["single_topic_v1"] + c["terse_augmented"]
            + c["multitopic_c4_base"] + c["same_object_rows"]) == c["total"]
    assert c["multitopic_real_rows"] + c["multitopic_synthetic_rows"] == c["multitopic_c4_base"]
    assert meta["train_sha256"] == TRAIN_SHA


# --------------------------------------------------------------------------- #
# 3. the new stream teaches the category-E shape
# --------------------------------------------------------------------------- #


def test_every_new_row_is_two_topics(so):
    assert so and all(len(_topics(e)) == 2 for e in so)


def test_every_new_row_shares_one_object(so):
    """The failure being fixed is a shared object being merged into one topic,
    so the training signal must guarantee a shared object."""
    for e in so:
        a, b = _topics(e)
        assert a["object"], "topic without an object cannot demonstrate sharing"
        assert a["object"] == b["object"]


def test_every_new_row_has_two_different_domains(so):
    """Rejects 'two topics merely because there are two verbs' and forbids
    synonym pairs dressed up as two problems."""
    for e in so:
        a, b = _topics(e)
        assert a["domain"] != b["domain"]


def test_every_new_row_has_two_distinct_issues(so):
    for e in so:
        a, b = _topics(e)
        assert normalize_text(a["issue"]) != normalize_text(b["issue"])
        assert a["issue"].strip() and b["issue"].strip()


def test_new_rows_use_the_inline_category_e_join(so):
    """One continuous sentence with a shared frame and a comma before the
    connective -- NOT the boundary-marked '<A>. а також <B>.' form that the
    MT85 same-street family already teaches and that the model already splits.
    """
    for e in so:
        text = _user(e)
        assert text.startswith("По "), text[:40]
        assert ", а також " in text
        assert ". а також " not in text, "boundary-marked form does not target the bug"
        # exactly one frame, shared by both clauses
        assert text.count("По ") == 1


def test_no_boundary_marker_inside_the_new_stream(so):
    """Nothing in the new stream may re-teach the SS pattern."""
    for e in so:
        text = _user(e)
        for marker in ("; а також", ".\n", " а також "):
            if marker == " а також ":
                # the ", а також " form already covers this; guard the space-only one
                continue
            assert marker not in text


def test_new_stream_uses_the_same_system_prompt_as_the_corpus(so, v32):
    """A different system prompt would make this a second, unrelated change."""
    prompts = {json.dumps(e["messages"][0]) for e in v32}
    assert len(prompts) == 1


# --------------------------------------------------------------------------- #
# 4. provenance and leakage
# --------------------------------------------------------------------------- #


def test_no_new_row_leaks_into_validation_or_test(v32):
    frozen = set()
    for split in ("validation", "test"):
        for r in _rows(DATA_DIR / "tune" / "v2" / f"{split}.jsonl"):
            frozen.add(normalize_text(r["messages"][1]["content"]))
    train = [normalize_text(_user(e)) for e in v32]
    assert not (set(train) & frozen)


def test_every_new_row_is_composed_from_train_split_records(so):
    """eval_v3 category E is built from *test*-split records, so the new
    stream must be provably train-only or it trains on its own test set."""
    train_ids = {str(r.get("uid")) for r in
                 (json.loads(l) for l in
                  (DATA_DIR / "train.jsonl").read_text(encoding="utf-8").splitlines() if l.strip())}
    meta = json.loads((_require(V32 / "meta.json")).read_text(encoding="utf-8"))
    assert meta["integrity"]["same_object_sources_all_train_split"] is True
    # sources are recorded in the provenance sidecar
    prov = _rows(V32 / "provenance.jsonl")
    for row in prov:
        for uid in row["sources"]:
            assert uid in train_ids, f"{uid} is not a train-split record"


def test_provenance_sidecar_covers_every_new_row(so):
    prov = _rows(_require(V32 / "provenance.jsonl"))
    assert len(prov) == len(so)
    # train uids carry the "#0" replication suffix; provenance is keyed by base uid
    assert {p["uid"] for p in prov} == {e["uid"].split("#")[0] for e in so}
    assert all(p["synthetic"] and p["same_object"] for p in prov)
    assert all(p["n_topics"] == 2 for p in prov)
    assert all(p["splits"] == ["train"] for p in prov)
    assert all(p["domains"][0] != p["domains"][1] for p in prov)


def test_new_streets_are_diverse(so):
    """One street must not dominate the stream."""
    streets = [json.loads(l) for l in _load(V32 / "provenance.jsonl")]
    counts: dict[str, int] = {}
    for p in streets:
        counts[p["street"]] = counts.get(p["street"], 0) + 1
    assert max(counts.values()) <= 3, "MAX_PER_STREET was not honoured"
    assert len(counts) >= 100, "the stream is concentrated in too few streets"


# --------------------------------------------------------------------------- #
# 5. the intervention stays a single change
# --------------------------------------------------------------------------- #


def test_meta_declares_what_was_left_alone():
    meta = json.loads((_require(V32 / "meta.json")).read_text(encoding="utf-8"))
    unchanged = " ".join(meta["unchanged_from_corrected"]).lower()
    for token in ("c1", "r1", "r2", "r3", "c3", "c4", "evaluator", "serving"):
        assert token in unchanged, f"{token} is not declared unchanged"


def test_single_topic_verbatim_rate_is_untouched(v32):
    """Capping the verbatim-copy rate was considered and rejected: it is the
    mechanism, but changing it here would confound the experiment.

    Recomputed from both corpora with the same function rather than read from
    metadata, so this cannot pass just because two files agree on a number.
    """
    from ml.tune.build_v3_2 import _verbatim_rate

    def singles(rows):
        return [e for e in rows if len(_topics(e)) == 1]

    corrected = _rows(_require(CORRECTED / "train.jsonl"))
    assert _verbatim_rate(singles(v32)) == _verbatim_rate(singles(corrected))


def test_training_recipe_constants_are_reused_not_redefined():
    """The recipe must come from the corrected arm, not be restated here."""
    from ml.tune import v3_changes

    assert v3_changes.MT_REAL_UPSAMPLE_V3 == 24
    assert v3_changes.MT_SYNTHETIC_UPSAMPLE_V3 == 2
    assert v3_changes.TERSE_UPSAMPLE == 8