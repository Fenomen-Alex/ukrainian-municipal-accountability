"""Tests for the SPLIT/MERGE/STOP classifier.

The classifier decides what the v3.2 experiment concludes, so it is pinned
against both arms whose behaviour is already established. If a refactor silently
flips a bucket, these fail instead of the report quietly changing its story.

    .venv/bin/python -m pytest ml/tests/test_behavior_v3.py -q
"""

from __future__ import annotations

import json

import pytest

from ml.tune.behavior_v3 import MERGE_SIM, classify, two_topic_cases, cases

#: From ml/tune/V3_MULTITOPIC_ANALYSIS.md. Order and counts must match exactly.
CORRECTED_BUCKETS = {"SPLIT": 23, "MERGE": 17, "STOP": 6, "JSONFAIL": 6}
PREVIOUS_BUCKETS = {"SPLIT": 34, "MERGE": 5, "STOP": 10, "JSONFAIL": 3}

RAW_CORRECTED = "/tmp/mtinv/rawall_corrected.json"
RAW_PREVIOUS = "/tmp/mtinv/raw_prev_all52.json"


def _case(cid="ev3-001", domains=("roads", "sanitation"), issues=(" poth", " сміт")):
    return {
        "id": cid,
        "category": "E",
        "category_name": "x",
        "expected_topic_count": 2,
        "expected_domains": sorted(domains),
        "expected_topics": [
            {"domain": domains[0], "issue": issues[0]},
            {"domain": domains[1], "issue": issues[1]},
        ],
        "text": "По вул. X: a, а також b.",
    }


def _topics(*pairs):
    return [{"domain": d, "issue": i} for d, i in pairs]


# --------------------------------------------------------------------------- #
# classification rules
# --------------------------------------------------------------------------- #


def test_two_topics_is_a_split():
    r = classify(_case(), "{}", None, _topics(("roads", " poth"), ("sanitation", " сміт")))
    assert r["behaviour"] == "SPLIT"


def test_one_topic_covering_both_problems_is_a_merge():
    """The corrected-v3 regression: both problems read, one topic filed."""
    r = classify(_case(), "{}", None, _topics(("roads", " poth і сміт на вулиці")))
    assert r["behaviour"] == "MERGE"
    assert r["predicted_topic_count"] == 1


def test_one_topic_covering_only_the_first_is_a_stop():
    """Different bug: the second problem was never produced at all."""
    r = classify(_case(), "{}", None, _topics(("roads", " poth на вулиці")))
    assert r["behaviour"] == "STOP"


def test_unparseable_is_a_jsonfail_not_a_stop():
    r = classify(_case(), "not json at all", "unparseable", None)
    assert r["behaviour"] == "JSONFAIL"
    assert r["predicted_topic_count"] is None


def test_more_topics_than_expected_is_an_over():
    r = classify(_case(), "{}", None,
                 _topics(("roads", "a"), ("sanitation", "b"), ("water", "c")))
    assert r["behaviour"] == "OVER"


def test_merge_threshold_is_the_documented_one():
    """Guard the constant: re-tuning it per arm would make the buckets
    incomparable between arms."""
    assert MERGE_SIM == 0.30


def test_an_unrelated_topic_is_a_stop():
    """Low similarity to both expected issues must not be read as a merge."""
    r = classify(_case(), "{}", None, _topics(("roads", "купити квитки на базарі")))
    assert r["behaviour"] == "STOP"


# --------------------------------------------------------------------------- #
# the suite itself
# --------------------------------------------------------------------------- #


def test_there_are_52_two_topic_cases():
    assert len(two_topic_cases(cases())) == 52


def test_category_E_has_6_two_topic_cases():
    assert len([c for c in two_topic_cases(cases()) if c["category"] == "E"]) == 6


# --------------------------------------------------------------------------- #
# pinned against both established arms
# --------------------------------------------------------------------------- #


def _buckets_from_raw(path: str) -> dict[str, int]:
    from ml.tune.run_eval_v3 import parse_payload

    saved = json.loads(open(path, encoding="utf-8").read())
    by_id = {c["id"]: c for c in two_topic_cases(cases())}
    out: dict[str, int] = {}
    for cid, case in by_id.items():
        entry = saved[cid]
        raw = entry["raw"] if isinstance(entry, dict) else entry
        parsed = parse_payload(raw)
        topics = getattr(parsed, "parsed", None)
        topics = topics.get("topics") if isinstance(topics, dict) else None
        r = classify(case, raw, None if topics is not None else "unparseable", topics)
        out[r["behaviour"]] = out.get(r["behaviour"], 0) + 1
    return dict(sorted(out.items()))


@pytest.mark.skipif(not __import__("pathlib").Path(RAW_CORRECTED).exists(),
                    reason="raw corrected-arm dump not present")
def test_corrected_arm_buckets_are_reproduced():
    assert _buckets_from_raw(RAW_CORRECTED) == CORRECTED_BUCKETS


@pytest.mark.skipif(not __import__("pathlib").Path(RAW_PREVIOUS).exists(),
                    reason="raw previous-arm dump not present")
def test_previous_arm_buckets_are_reproduced():
    assert _buckets_from_raw(RAW_PREVIOUS) == PREVIOUS_BUCKETS