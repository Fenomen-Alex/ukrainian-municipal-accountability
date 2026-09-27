"""Focused tests for the multi-topic evaluation harness (no GPU needed).

The deterministic mode is the oracle ceiling: it reproduces the target verbatim,
so every multi-topic metric must be 1.0. That locks in the harness semantics.
"""

from __future__ import annotations

import json

from ml.tune.evaluate import load_validator
from ml.tune.run_eval_multitopic import (
    _load_suite,
    _targets_and_inputs,
    eval_multitopic,
)

import ml.tune.run_eval_multitopic as m


def test_suite_loading_and_targets():
    suite = _load_suite()
    assert len(suite) >= 30
    targets, inputs = _targets_and_inputs(suite)
    assert len(targets) == len(inputs) == len(suite)
    for t, inp in zip(targets, inputs):
        assert t["source_text"] == inp
        assert len(t["topics"]) == 2, f"{t['uid']} should be multi-topic"
        domains = {top["domain"] for top in t["topics"]}
        assert len(domains) >= 2, f"{t['uid']}: {domains}"
        # user text must carry the substance (target not trivially decoupled)
        assert t["topics"][0]["issue"]


def test_deterministic_is_oracle_ceiling():
    suite = _load_suite()
    targets, _ = _targets_and_inputs(suite)
    preds = m._deterministic_predictions(targets)
    validator = load_validator()
    metrics = eval_multitopic(preds, targets, validator)
    for k in ("json_parse_rate", "schema_validity_rate", "topic_count_accuracy",
              "multi_topic_recall", "domain_set_exact", "domain_set_precision",
              "domain_set_recall", "issue_rouge_l_best"):
        assert metrics[k] == 1.0, k
    assert len(metrics["per_example"]) == len(targets)


def test_deterministic_roundtrips_valid_schema():
    suite = _load_suite()
    targets, _ = _targets_and_inputs(suite)
    preds = m._deterministic_predictions(targets)
    validator = load_validator()
    for p in preds:
        assert list(validator.iter_errors(p.parsed)) == []