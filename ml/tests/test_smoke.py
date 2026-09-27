"""Focused tests for the smoke suite loading + evaluation logic.

These never run a model: they exercise the smoke fixture, the raw-result
loaders, the schema validator, and the pure comparison/evaluation helpers in
`ml.tune.eval_smoke` (domain mapping, grounding heuristic, multi-topic bookkeeping).
"""

from __future__ import annotations

import json

import pytest

from ml.tune.eval_smoke import (
    EXPECTED_TO_SCHEMA,
    MULTI_TOPIC_CASES,
    evaluate_all,
    evaluate_case,
    hallucination_flags,
    load_results,
    load_smoke_cases,
)
from ml.tune.evaluate import load_validator

CASES = load_smoke_cases()
BY_ID = {c["id"]: c for c in CASES}


def test_smoke_fixture_has_20_known_ids():
    assert len(CASES) == 20, f"expected 20 cases, got {len(CASES)}"
    assert {c["id"] for c in CASES} == {f"smoke-{i:02d}" for i in range(1, 21)}


def test_every_smoke_case_has_required_fields():
    for c in CASES:
        assert isinstance(c["text"], str) and c["text"].strip()
        assert isinstance(c["expected_domains"], list) and c["expected_domains"]
        assert c["category"]
        assert isinstance(c["notes"], str) and c["notes"].strip()


def test_every_expected_domain_has_mapping():
    domains = {d for c in CASES for d in c["expected_domains"]}
    assert domains - EXPECTED_TO_SCHEMA.keys() == set(), "unexpected domain lacking mapping"


def test_multi_topic_cases_declare_two_domains():
    for cid in MULTI_TOPIC_CASES:
        assert len(BY_ID[cid]["expected_domains"]) == 2
        assert len(BY_ID[cid]["expected_domains"]) == len(set(BY_ID[cid]["expected_domains"]))


def test_mappings_are_schema_domains():
    validator = load_validator()
    enums = set(validator.schema["definitions"]["topic"]["properties"]["domain"]["enum"])
    for acceptable in EXPECTED_TO_SCHEMA.values():
        for d in acceptable:
            assert d in enums, f"mapped domain {d!r} not in the schema enum"


def test_finetuned_base_results_both_shipped():
    """The 20 smoke outputs for both systems exist and parse."""
    for tag in ("finetuned", "base"):
        rows = load_results(tag)
        assert len(rows) == 20
        assert {r["id"] for r in rows} == set(BY_ID)
        for r in rows:
            assert r["parsed"] is not None, f"{r['id']} did not parse"
            assert r["schema_valid"], f"{r['id']} schema-invalid"

            type_ = "application/json"
            _ = type_
            json.dumps(r["parsed"], ensure_ascii=False)


def test_hallucination_flags_matches_report_counts():
    """Sanity guard for the qualitative numbers quoted in the report."""
    ft = {r["id"]: r for r in load_results("finetuned")}
    base = {r["id"]: r for r in load_results("base")}
    assert not any(hallucination_flags(ft[cid]) for cid in BY_ID)
    flagged = [cid for cid in BY_ID if hallucination_flags(base[cid])]
    assert len(flagged) == 10, flagged
    assert "smoke-06" in flagged  # invented "ст. 125" / "2023-04-05"


def test_grounding_accepts_inflection_but_rejects_invention():
    from ml.tune.eval_smoke import _stem_supported, _tokens

    src = "комуналка прибрала сміття біля школи №5 але в підвалі стоїть вода"
    assert _stem_supported("підвалі", _tokens(src))   # inflected form, grounded
    assert _stem_supported("сміттє", _tokens(src))    # derived from сміття
    assert not _stem_supported("2023-04-05", _tokens(src))
    assert not _stem_supported("міська", _tokens(src))


def test_hallucination_empty_only_for_missing_grounding_not_empty_fields():
    row = {"text": "світлофор", "parsed": {"topics": [
        {"domain": "electricity", "issue": "світлофор", "object": "",
         "requested_action": "", "attributes": {}}]}}
    assert hallucination_flags(row) == []
    row["parsed"]["topics"][0]["attributes"] = {"дата": "2023-04-05"}
    assert hallucination_flags(row) != []


def test_all_fields_evaluated_for_each_case():
    ev = evaluate_all()
    for tag in ("finetuned", "base"):
        assert len(ev[tag]) == 20
        for cid, detail in ev[tag].items():
            allowed = {"id", "category", "expected", "pred_domains", "uncovered",
                       "extra", "issue_present", "object_present", "action_present",
                       "has_attributes", "hallucinations", "multi_ok", "schema_ok",
                       "passed", "notes"}
            assert set(detail) == allowed
            assert detail["id"] == cid


def test_multitopic_cases_fail_when_single_topic():
    """Baseline collapse: base model collapses every multi-topic smoke case.

    The v2 fine-tune fixes at least one (smoke-13), so the "both systems
    collapse" invariant only holds for base.
    """
    ev = evaluate_all()
    for cid in MULTI_TOPIC_CASES:
        assert len(ev["base"][cid]["pred_domains"]) == 1
        assert not ev["base"][cid]["multi_ok"], f"base {cid}"
    # v2 must have improved at least one multi case vs base.
    assert any(ev["finetuned"][cid]["multi_ok"] for cid in MULTI_TOPIC_CASES)


def test_summary_counts_match_report():
    from ml.tune.eval_smoke import summarize

    ev = evaluate_all()
    for tag, expected_passed, expected_multi in (
            ("finetuned", 17, 1), ("base", 8, 0)):
        s = summarize(tag, ev[tag])
        assert s["n"] == 20
        assert s["passed"] == expected_passed, (tag, s)
        assert s["schema_valid"] == 20
        assert s["multi_topic_passed"] == expected_multi


def test_evaluate_case_handles_missing_parsed():
    row = {"id": "smoke-01", "category": "Standard", "expected_domains": ["roads"],
           "parsed": None, "schema_valid": False}
    detail = evaluate_case(row)
    assert detail["schema_ok"] is False
    assert detail["passed"] is False
    assert detail["uncovered"] == ["roads"]


@pytest.mark.parametrize("bad_domain", ["nonsense", ""])
def test_predicted_domains_ignores_non_topic(bad_domain):
    from ml.tune.eval_smoke import _predicted_domains

    row = {"parsed": {"topics": [{"domain": bad_domain}]}}
    doms = _predicted_domains(row)
    expected = [bad_domain]
    # function must reflect the topic array honestly (no silent correction)
    assert doms == expected