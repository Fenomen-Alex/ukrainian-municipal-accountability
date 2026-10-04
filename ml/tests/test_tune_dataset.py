"""Tests for the MLX fine-tune dataset build + evaluation pipeline.

These use only the real train/validation splits and the deterministic baseline;
no model inference (no network, no weights) is invoked.
"""

from __future__ import annotations

import json

import jsonschema
import pytest

from ml.tune.build_dataset import (
    KIND_TO_DOMAIN,
    SYSTEM_PROMPT,
    build_chat_example,
    label_record,
    load_split,
    to_json,
)
from ml.tune.evaluate import (

    Prediction,
    eval_predictions,
    first_topic_domain,
    load_validator,
    parse_payload,
)

# --- public-checkout guard ---------------------------------------------
# These tests read corpora generated from the official CC BY source. The
# public repository does not ship them (see REPRODUCIBILITY.md), so on a
# clean clone they skip explicitly instead of failing at import time.
from ml.tests._corpora import corpora_present  # noqa: E402

if not corpora_present():
    pytestmark = pytest.mark.skip(
        reason="private/generated corpora absent from the public checkout"
    )


class _FakeTokenizer:
    """Records kwargs so we can assert thinking is always disabled."""

    def __init__(self):
        self.calls = []

    def apply_chat_template(self, *args, **kwargs):
        self.calls.append(kwargs)
        return "encoded"

    def __getattr__(self, item):
        return None

TEST_ROWS = [] if not corpora_present() else load_split("test")


@pytest.fixture(scope="module")
def validator():
    return load_validator()


def test_kind_to_domain_covers_all_labels(validator):
    kinds = {r.get("kind") for r in TEST_ROWS}
    assert kinds, "test set unexpectedly empty"
    for k in kinds:
        assert k in KIND_TO_DOMAIN, f"kind {k!r} missing from KIND_TO_DOMAIN"


def test_all_test_rows_label_to_schema_valid_json():
    for rec in TEST_ROWS:
        labeled = label_record(rec)
        payload = to_json(labeled)
        errs = list(load_validator().iter_errors(payload))
        assert not errs, f"{rec.get('uid')}: {errs[0].message}"


def test_domain_within_enum(validator):
    enums = set(validator.schema["definitions"]["topic"]["properties"]["domain"]["enum"])
    for rec in TEST_ROWS:
        assert label_record(rec).domain in enums


def test_empty_stays_empty():
    """A record without a request verb must not get an invented action."""
    rec = {"kind": "Теплопостачання",
           "content": "Повідомлення про розповітрення системи теплопостачання.",
           "uid": "T-1"}
    labeled = label_record(rec)
    assert labeled.requested_action == ""


def test_object_from_text_not_metadata():
    """Object must come from the complaint text, never from metadata columns."""
    rec = {"kind": "Теплопостачання",
           "content": "Прошу відремонтувати ліфт по вул. Героїв України, 22, корп. 3.",
           "uid": "T-2",
           "addressThoroughfare": "вул. Зовсім Інша"}
    labeled = label_record(rec)
    assert "Зовсім Інша" not in labeled.object
    assert "Героїв України" in labeled.object


def test_no_pii_in_target():
    """Full names and phone numbers must not leak into the structured target."""
    rec = {"kind": "Санітарний стан, благоустрій населених пунктів, прибудинкових територій",
           "content": "Заявник Іваненко І.І. просить прибрати сміття по вул. Сонячній, буд. 5. "
                      "Телефон 095-123-45-67.",
           "uid": "T-3"}
    labeled = label_record(rec)
    blob = json.dumps(to_json(labeled), ensure_ascii=False)
    assert "Іваненко" not in blob
    assert "095" not in blob


def test_chat_example_layout():
    rec = TEST_ROWS[0]
    ex = build_chat_example(rec)
    assert ex["messages"][0]["role"] == "system"
    assert ex["messages"][1]["role"] == "user"
    assert ex["messages"][2]["role"] == "assistant"
    payload = json.loads(ex["messages"][2]["content"])
    assert "topics" in payload
    assert ex["uid"] == rec.get("uid")


def test_system_prompt_is_ukrainian_and_states_rules():
    assert "JSON" in SYSTEM_PROMPT
    assert "вигадуй" in SYSTEM_PROMPT  # "do not invent"


def test_parse_payload_extracts_json_from_noise():
    p = parse_payload('Інформація\n```json\n{"topics": [{"domain": "roads"}]}\n```')
    assert p.parsed is not None
    assert p.topics[0]["domain"] == "roads"


def test_degenerate_payload_is_unparseable():
    p = parse_payload("Щось зовсім не те")
    assert p.parsed is None


def test_deterministic_baseline_is_upper_bound(validator):
    from ml.tune.run_eval import _deterministic_predictions

    targets = []
    for i, rec in enumerate(TEST_ROWS):
        labeled = label_record(rec)
        targets.append({"idx": i, "uid": labeled.uid,
                        "topics": [{"domain": labeled.domain,
                                    "issue": labeled.issue,
                                    "object": labeled.object,
                                    "requested_action": labeled.requested_action,
                                    "attributes": labeled.attributes,
                                    "source_text": (rec.get("content") or "").strip()}]})
    preds = _deterministic_predictions()
    metrics = eval_predictions(preds, targets, validator)
    assert metrics["json_parse_rate"] == 1.0
    assert metrics["schema_validity_rate"] == 1.0
    assert metrics["domain_accuracy"] == 1.0
    assert metrics["issue_rouge_l"] == 1.0
    assert metrics["object_exact"] == 1.0
    assert metrics["action_presence_match"] == 1.0
    assert metrics["hallucination_rate"] == 0.0


def test_eval_handles_duplicate_uids(validator):
    """Test set contains duplicate uids; eval must be positional, not by uid."""
    from ml.tune.run_eval import _deterministic_predictions

    targets = []
    for i, rec in enumerate(TEST_ROWS):
        labeled = label_record(rec)
        targets.append({"idx": i, "uid": "SAME-UID",
                        "topics": [{"domain": labeled.domain,
                                    "issue": labeled.issue,
                                    "object": labeled.object,
                                    "requested_action": labeled.requested_action,
                                    "attributes": labeled.attributes,
                                    "source_text": (rec.get("content") or "").strip()}]})
    preds = _deterministic_predictions()
    metrics = eval_predictions(preds, targets, validator)
    assert metrics["domain_accuracy"] == 1.0


def test_first_topic_domain_empty_on_no_topics():
    p = Prediction(idx=0, uid="x", raw="")
    assert first_topic_domain(p) == "NO_TOPIC"


def test_schema_validator_rejects_wrong_domain(validator):
    bad = {"topics": [{"domain": "not-a-domain", "issue": "x", "object": "",
                       "requested_action": "", "attributes": {}}]}
    assert any(validator.iter_errors(bad))


def test_no_thinking_tokenizer_always_disables_thinking():
    from ml.tune.run_train import _NoThinkingTokenizer

    fake = _FakeTokenizer()
    wrapped = _NoThinkingTokenizer(fake)
    wrapped.apply_chat_template("x")
    wrapped.apply_chat_template("y", enable_thinking=True)
    assert fake.calls == [{"enable_thinking": False}, {"enable_thinking": False}]


def test_schema_validator_rejects_missing_fields(validator):
    bad = {"topics": [{"domain": "roads", "issue": "x"}]}
    errs = list(validator.iter_errors(bad))
    assert any("requested_action" in str(e.message) for e in errs)
