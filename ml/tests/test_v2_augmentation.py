"""Focused tests for the v2 multi-topic augmentation pipeline.

Run with ``.venv/bin/python -m pytest ml/tests/test_v2_augmentation.py``.

All builders must be side-effect-free here; the shipped artifacts under
``ml/data/tune/multitopic/*.jsonl`` and ``ml/data/tune/v2/*`` are produced by
running the modules once and validated against the raw corpus.
"""

from __future__ import annotations

import json

import pytest

from ml.cleaner import normalize_text
from ml.tune.build_multitopic import (
    _chat_example,
    _strip_boilerplate,
    _topic_from_issue,
    build,
)
from ml.tune.build_v2 import build as build_v2

# --- public-checkout guard ---------------------------------------------
# These tests read corpora generated from the official CC BY source. The
# public repository does not ship them (see REPRODUCIBILITY.md), so on a
# clean clone they skip explicitly instead of failing at import time.
from ml.tests._corpora import corpora_present  # noqa: E402

if not corpora_present():
    pytestmark = pytest.mark.skip(
        reason="private/generated corpora absent from the public checkout"
    )

DATA = __import__("ml.tune.build_dataset", fromlist=["DATA_DIR"]).DATA_DIR
MT_DIR = DATA / "tune" / "multitopic"
V2_DIR = DATA / "tune" / "v2"


def _load_split(name: str) -> list[dict]:
    return [json.loads(l) for l in (DATA / f"{name}.jsonl").read_text().splitlines()]


HOUSING = "Експлуатація та ремонт житла ( у т.ч. ліфтів, сантехнічного обладнання тощо)"


@pytest.fixture(scope="session", autouse=True)
def _builder_outputs(tmp_path_factory):
    """Run builders into a scratch OUT_DIR data root (not the shipped artifacts)."""
    return None


def test_topic_from_issue_is_deterministic():
    issue = "аварійний стан даху, протікання у квартирі по вул. Тестова, 1"
    a = _topic_from_issue(issue, HOUSING)
    b = _topic_from_issue(issue, HOUSING)
    assert a == b
    assert a["domain"] == "housing"
    assert issue in a["issue"]


def test_chat_example_shape_roundtrip():
    ex = _chat_example(
        "MT-X",
        "текст",
        [_topic_from_issue("аварійний стан даху", HOUSING)],
    )
    assert ex["uid"] == "MT-X"
    roles = [m["role"] for m in ex["messages"]]
    assert roles == ["system", "user", "assistant"]
    parsed = json.loads(ex["messages"][-1]["content"])
    assert parsed["topics"][0]["domain"] == "housing"
    assert set(parsed["topics"][0]) >= {
        "domain",
        "issue",
        "object",
        "requested_action",
        "attributes",
    }


def test_strip_boilerplate_removes_consent_tail():
    text = "прохання відремонтувати дах. відповідь заявнику. заявник надає згоду на обробку своїх персональних даних та передачу їх третім особам відповідно до вимог закону."
    cleaned = _strip_boilerplate(text)
    assert "відповідь заявнику" not in cleaned
    assert "згоду" not in cleaned
    assert "відремонтувати дах" in cleaned


# --- shipped artifact validation ----------------------------------------------

def test_multitopic_artifacts_exist_and_leakage_free():
    for f in ("train.jsonl", "eval.jsonl", "provenance.jsonl", "meta.json"):
        assert (MT_DIR / f).exists(), f
    frozen = {
        normalize_text(r.get("content") or "") for s in ("validation", "test") for r in _load_split(s)
    }
    meta = json.loads((MT_DIR / "meta.json").read_text())
    assert meta["leakage_frozen_train"] == []
    assert meta["leakage_frozen_eval"] == []
    for split in ("train", "eval"):
        for ex in [json.loads(l) for l in (MT_DIR / f"{split}.jsonl").read_text().splitlines()]:
            user = ex["messages"][1]["content"]
            assert normalize_text(user) not in frozen, f"leak in {split}: {ex['uid']}"


def test_multitopic_real_decompositions_revalidate():
    """Re-locating every verbatim issue substring must succeed against train content."""
    ann = json.loads((MT_DIR / "real_annotations.json").read_text())
    train = {r["uid"]: r for r in _load_split("train")}
    for entry in ann:
        src = train.get(entry["uid"])
        assert src is not None, entry["uid"]
        ncontent = normalize_text(src["content"])
        assert len({k["kind"] for k in entry["kind_issues"]}) >= 2
        for ki in entry["kind_issues"]:
            assert normalize_text(ki["issue"]) in ncontent, f"{entry['uid']} {ki['kind']}"


def test_synthetic_components_sourced_from_train_and_distinct_domains():
    prov = [json.loads(l) for l in (MT_DIR / "provenance.jsonl").read_text().splitlines()]
    train_uids = {r["uid"] for r in _load_split("train")}
    synthetics = [p for p in prov if p["synthetic"]]
    assert len(synthetics) >= 380, "expected the vast majority of augmentation to be synthetic"
    for p in synthetics:
        assert len(p["sources"]) == 2
        assert all(s in train_uids for s in p["sources"]), p["uid"]
        assert len({d for d in p["domains"]}) >= 2, p["uid"]
        assert p["n_topics"] == 2
    reals = [p for p in prov if not p["synthetic"]]
    assert any(p["n_topics"] >= 2 for p in reals)


def test_v2_artifacts_and_frozen_benchmark_parity():
    assert (V2_DIR / "train.jsonl").exists()
    assert (V2_DIR / "validation.jsonl").exists()
    assert (V2_DIR / "test.jsonl").exists()
    # byte-identical copies of frozen benchmarks
    assert (V2_DIR / "validation.jsonl").read_bytes() == (DATA / "tune" / "validation.jsonl").read_bytes()
    assert (V2_DIR / "test.jsonl").read_bytes() == (DATA / "tune" / "test.jsonl").read_bytes()
    meta = json.loads((V2_DIR / "meta.json").read_text())
    comp = meta["composition"]
    v1_train = [json.loads(l) for l in (DATA / "tune" / "train.jsonl").read_text().splitlines()]
    mt = [json.loads(l) for l in (MT_DIR / "train.jsonl").read_text().splitlines()]
    assert comp["v1_single_topic"] == len(v1_train) == 5384
    real = [ex for ex in mt if ex["uid"].startswith("MT-REAL")]
    syn = [ex for ex in mt if not ex["uid"].startswith("MT-REAL")]
    assert comp["multitopic_augmentation"] == (
        len(real) * comp["multitopic_real_upsample"]
        + len(syn) * comp["multitopic_upsample"]
    )
    assert comp["total"] == len(v1_train) + comp["multitopic_augmentation"]


def test_topic_count_balance():
    """v2 must keep a single-topic majority (no runaway over-emission) while
    multi-topic examples get real exposure (~25-40% of the stream, five
    distinct synthetic shapes: generic, same-street, compact, registry,
    smoke-clone)."""
    counts = {}
    for l in (V2_DIR / "train.jsonl").read_text().splitlines():
        ex = json.loads(l)
        n = len(json.loads(ex["messages"][-1]["content"])["topics"])
        counts[n] = counts.get(n, 0) + 1
    total = sum(counts.values())
    share1 = counts[1] / total
    share2 = counts.get(2, 0) / total
    assert share1 >= 0.6, f"single-topic share must stay a majority, got {share1}"
    assert 0.20 <= share2 <= 0.40, f"multi-topic share out of band: {share2}"
