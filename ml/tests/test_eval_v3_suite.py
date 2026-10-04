"""Structural tests for the v3 adversarial eval suite and the v3 plan.

No model, no GPU, no network. The suite is a scoring target, so what we lock down
here is that the target is well formed, covers what it claims to cover, and does
not smuggle training text into a held-out benchmark.
"""

from __future__ import annotations

import json
import unicodedata
from pathlib import Path

import jsonschema
import pytest

from ml.tune.build_eval_v3 import (
    CATEGORIES,
    CATEGORY_TOPIC_COUNT,
    train_content_keys,
)
from ml.tune.evaluate import load_validator

# --- public-checkout guard ---------------------------------------------
# Some checks need corpora generated from the official CC BY source, which the
# public repository does not ship (see REPRODUCIBILITY.md). Those tests skip
# individually; the benchmark case set itself is tracked and always tested.
from ml.tests._corpora import skip_if_missing  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
EVAL_DIR = ROOT / "ml" / "data" / "tune" / "eval_v3"
V3_DIR = ROOT / "ml" / "data" / "tune" / "v3"
SCHEMA_PATH = ROOT / "ml" / "data" / "gold" / "annotation_schema.json"

TRAIN_STREAMS = (
    ROOT / "ml" / "data" / "train.jsonl",
    ROOT / "ml" / "data" / "tune" / "multitopic" / "train.jsonl",
    ROOT / "ml" / "data" / "tune" / "v2" / "train.jsonl",
)


def _normalise(text: str) -> str:
    """Fold text for leakage comparison.

    Deliberately stricter than the production ``normalize_text``: it also removes
    apostrophe variants and collapses whitespace, so a case that differs from a
    training text only by a curly/straight quote is still caught.
    """
    text = text.replace("’", "'").replace("ʼ", "'").replace("‘", "'")
    text = unicodedata.normalize("NFKC", text)
    return " ".join(text.lower().split())


def _load_cases() -> list[dict]:
    path = EVAL_DIR / "cases.jsonl"
    if not path.exists():
        pytest.skip("eval_v3 suite not built; run `python -m ml.tune.build_eval_v3`")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_suite_size_is_in_target_range():
    cases = _load_cases()
    assert 100 <= len(cases) <= 150, f"suite has {len(cases)} cases, want 100-150"


def test_every_case_matches_case_schema():
    cases = _load_cases()
    schema = json.loads((EVAL_DIR / "schema.json").read_text(encoding="utf-8"))
    validator = jsonschema.Draft7Validator(schema)
    for case in cases:
        errors = sorted(validator.iter_errors(case), key=str)
        assert not errors, f"{case['id']}: {errors[0].message}"


def test_expected_topics_satisfy_production_schema():
    """The references must be legal predictions under the frozen contract.

    This is the check that keeps the suite honest: a reference that the production
    schema would reject is a broken target, not a hard case.
    """
    cases = _load_cases()
    validator = load_validator()
    for case in cases:
        errors = sorted(validator.iter_errors({"topics": case["expected_topics"]}), key=str)
        assert not errors, f"{case['id']}: {errors[0].message}"


def test_case_ids_are_unique_and_ordered():
    cases = _load_cases()
    ids = [case["id"] for case in cases]
    assert len(set(ids)) == len(ids), "duplicate case ids"
    assert ids == sorted(ids), "case ids should be in build order"


def test_case_texts_are_unique():
    cases = _load_cases()
    seen: dict[str, str] = {}
    for case in cases:
        key = _normalise(case["text"])
        assert key not in seen, f"{case['id']} duplicates {seen.get(key)}"
        seen[key] = case["id"]


def test_all_categories_covered():
    cases = _load_cases()
    present = {case["category"] for case in cases}
    assert present == set(CATEGORIES), (
        f"category gap: missing {sorted(set(CATEGORIES) - present)}, "
        f"unexpected {sorted(present - set(CATEGORIES))}"
    )
    for case in cases:
        assert case["category_name"] == CATEGORIES[case["category"]][0]


def test_every_category_has_multiple_cases():
    """One case per category cannot support a rate; require >= 2 so the
    per-category numbers in v3_experiment_design.md are meaningful."""
    cases = _load_cases()
    counts: dict[str, int] = {}
    for case in cases:
        counts[case["category"]] = counts.get(case["category"], 0) + 1
    thin = {code: n for code, n in counts.items() if n < 2}
    assert not thin, f"categories with fewer than 2 cases: {thin}"


def test_provenance_is_closed_and_labelled():
    cases = _load_cases()
    kinds = {"real", "condensed", "composed"}
    for case in cases:
        prov = case["provenance"]
        assert prov["kind"] in kinds
        assert prov["sources"], f"{case['id']} has no provenance sources"
        assert prov["splits"], f"{case['id']} has no split provenance"
        for split in prov["splits"]:
            assert split in {"validation", "test"}


def test_real_cases_are_verbatim_held_out_records():
    skip_if_missing(ROOT / "ml" / "data" / "validation.jsonl", what="held-out corpus")
    skip_if_missing(ROOT / "ml" / "data" / "test.jsonl", what="held-out corpus")
    cases = _load_cases()
    held = {}
    for split in ("validation", "test"):
        for line in (ROOT / "ml" / "data" / f"{split}.jsonl").read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                held[row["uid"]] = row["content"]

    real = [case for case in cases if case["provenance"]["kind"] == "real"]
    assert len(real) >= 30, f"only {len(real)} real cases"
    for case in real:
        assert len(case["provenance"]["sources"]) == 1
        uid = case["provenance"]["sources"][0]
        assert uid in held, f"{case['id']} source {uid} is not held out"
        assert case["text"] == held[uid].strip(), f"{case['id']} is not verbatim"


def test_composed_cases_join_their_sources():
    """Composition must be concatenation of held-out material, not invention."""
    cases = _load_cases()
    composed = [case for case in cases if case["provenance"]["kind"] == "composed"]
    assert composed
    for case in composed:
        sources = case["provenance"]["sources"]
        assert len(sources) == 2, f"{case['id']} should join exactly two records"
        assert len(case["expected_topics"]) == 2


def test_expected_topic_counts_match_declared_categories():
    """Each case carries the count its category promises, and the references
    honour it -- otherwise a category silently stops probing what it claims."""
    cases = _load_cases()
    for case in cases:
        declared = CATEGORY_TOPIC_COUNT[case["category"]]
        assert case["expected_topic_count"] == declared, (
            f"{case['id']} ({case['category']}) declares "
            f"{case['expected_topic_count']}, category promises {declared}"
        )
        assert len(case["expected_topics"]) == declared, (
            f"{case['id']} ({case['category']}) has "
            f"{len(case['expected_topics'])} expected topics, want {declared}"
        )
        assert len(case["expected_domains"]) == declared


def test_no_case_spans_two_splits():
    """Composition pairs records within one split, so provenance is unambiguous
    and `stats.by_split` sums to the case count."""
    cases = _load_cases()
    spanning = [case["id"] for case in cases if len(case["provenance"]["splits"]) != 1]
    assert not spanning, f"cases drawing from more than one split: {spanning}"


def test_composed_cases_use_distinct_source_records():
    """Guards against a component being paired with itself, which would make a
    'two distinct problems' case vacuous."""
    for case in _load_cases():
        sources = case["provenance"]["sources"]
        assert len(set(sources)) == len(sources), f"{case['id']} repeats a source record"


def test_multi_topic_cases_use_distinct_domains():
    for case in _load_cases():
        domains = [topic["domain"] for topic in case["expected_topics"]]
        if len(domains) == 2:
            assert domains[0] != domains[1], f"{case['id']} repeats domain {domains[0]}"


def test_no_request_verb_case_has_empty_requested_action():
    cases = [case for case in _load_cases() if case["category"] == "H"]
    assert len(cases) >= 5
    for case in cases:
        for topic in case["expected_topics"]:
            assert not topic["requested_action"].strip(), f"{case['id']} should have no action"


def test_no_case_text_leaks_into_training():
    """The hard guarantee: a held-out benchmark must contain no training text.

    Composed cases are built by joining two held-out records, so a naive
    containment check is not enough; we compare whole normalised texts against
    every training user message.
    """
    cases = _load_cases()
    skip_if_missing(
        ROOT / "ml" / "data" / "train.jsonl", what="training corpus"
    )
    keys = train_content_keys()
    assert keys, "training key set is empty; leakage check would be vacuous"
    leaks = [case["id"] for case in cases if _normalise(case["text"]) in keys]
    assert not leaks, f"case text found in a training stream: {leaks}"


def test_suite_is_deterministically_reproducible():
    """Re-running the builder must not change a single case."""
    import ml.tune.build_eval_v3 as builder

    skip_if_missing(
        ROOT / "ml" / "data" / "train.jsonl", what="training corpus"
    )
    before = {case["id"]: _normalise(case["text"]) for case in _load_cases()}
    second = builder.build()
    after = {case["id"]: _normalise(case["text"]) for case in second["cases"]}
    assert after == before, "builder is not deterministic"
    assert second["stats"]["n_cases"] == len(before)


def test_stats_json_agrees_with_cases():
    cases = _load_cases()
    stats = json.loads((EVAL_DIR / "stats.json").read_text(encoding="utf-8"))
    assert stats["n_cases"] == len(cases)

    by_category: dict[str, int] = {}
    by_provenance: dict[str, int] = {}
    for case in cases:
        by_category[case["category"]] = by_category.get(case["category"], 0) + 1
        kind = case["provenance"]["kind"]
        by_provenance[kind] = by_provenance.get(kind, 0) + 1
    assert stats["by_category"] == by_category
    assert stats["by_provenance"] == by_provenance
    assert sum(stats["by_split"].values()) == len(cases)


def test_suite_covers_the_terse_register():
    """The smoke failures live in very short inputs; the suite must reach them."""
    cases = _load_cases()
    lengths = [len(case["text"]) for case in cases]
    assert min(lengths) <= 120, f"shortest case is {min(lengths)} chars"
    assert sum(1 for n in lengths if n < 150) >= 10, "too few terse cases"
    assert max(lengths) >= 500, "no long-input cases"


# --------------------------------------------------------------------------- #
# v3 plan
# --------------------------------------------------------------------------- #


def test_v3_plan_is_present_and_machine_readable():
    path = V3_DIR / "plan.json"
    if not path.exists():
        pytest.skip("v3 plan not written")
    plan = json.loads(path.read_text(encoding="utf-8"))
    for key in ("objective", "baseline", "arms", "changes", "benchmarks", "success_gates", "budget"):
        assert key in plan, f"plan.json missing '{key}'"

    arms = plan["arms"]
    for arm in ("control", "treatment"):
        assert arm in arms, f"plan.json missing the '{arm}' arm"
        assert arms[arm]["dataset"], f"the '{arm}' arm names no dataset"
    assert arms["arm_order"] == ["control", "treatment"], (
        "the control arm must run first, or a treatment regression is uninterpretable"
    )

    # Every change must say what it is trying to fix and how it will be measured,
    # so a plan entry cannot degrade into a wish.
    assert plan["changes"], "plan.json declares no changes"
    for change in plan["changes"]:
        for key in ("id", "name", "targets", "action", "measurable_by"):
            assert change.get(key), f"change {change.get('id')!r} missing '{key}'"


def test_v3_success_gates_are_measurable():
    path = V3_DIR / "plan.json"
    if not path.exists():
        pytest.skip("v3 plan not written")
    gates = json.loads(path.read_text(encoding="utf-8"))["success_gates"]
    assert gates, "success_gates must not be empty"
    for name, spec in gates.items():
        assert "metric" in spec and "target" in spec, f"gate '{name}' not measurable"
        assert isinstance(spec["target"], (int, float)), f"gate '{name}' target is not numeric"
        assert "baseline" in spec, f"gate '{name}' missing a baseline to beat"
        assert "blocking" in spec, f"gate '{name}' does not say whether it blocks the run"


def test_v3_plan_baselines_match_the_recorded_v2_metrics():
    """The plan's floors only mean something if they match what v2 actually scored."""
    path = V3_DIR / "plan.json"
    if not path.exists():
        pytest.skip("v3 plan not written")
    plan = json.loads(path.read_text(encoding="utf-8"))
    base = plan["baseline"]
    assert base["frozen_eval"]["n"] == 329
    assert base["frozen_eval"]["domain_accuracy"] == pytest.approx(0.8389, abs=5e-4)
    assert base["frozen_eval"]["schema_valid"] == pytest.approx(0.997, abs=5e-4)
    assert base["multitopic_eval"]["n"] == 85
    assert base["multitopic_eval"]["topic_count_accuracy"] == pytest.approx(0.8588, abs=5e-4)
    assert base["smoke"]["overall_pass"] == 17
    assert base["smoke"]["multi_topic"] == 1
    assert base["smoke"]["multi_topic_expected"] == 4
