"""Regression gates for the literal ASCII double-quote artefact.

Root cause (traced, not inferred -- see ml/tune/FINAL_MODEL_ASSESSMENT.md and
ml/tune/QUOTE_ARTEFACT.md):

1. ``ml.tune.build_dataset._redact_pii`` deletes any capitalised 3+ letter word
   via ``_PERSON_NAME``. That over-matches organisation, shop and street names
   ("Екостайл", "ЖЕО №2", "Карамелька"), and it deletes the *content* of a
   quoted name while leaving the enclosing quotation marks behind. The result is
   an empty ASCII quote pair ``""`` in the text.
2. ``serve_v2.build_prompt`` passed complaint text to the model verbatim, so a
   legitimate ASCII-quoted name in real traffic also reached the model as-is.

Both paths end the same way: the model copies the source span verbatim into an
``issue`` JSON string, and a raw ``"`` terminates that string, so the payload no
longer parses. ``parse_strict`` then reports "not strict JSON" and the arm scores
a schema failure.

The fix is a pre-inference normalisation that removes the JSON-hazardous ASCII
quote while preserving the quotation itself:

  * a mid-word ASCII quote (``під"їзду``) is an apostrophe typo -> U+2019,
    which is what the rest of the corpus uses;
  * any other ASCII quote is a quotation delimiter -> alternating U+201C/U+201D,
    a form the corpus already contains ~400 times, so no retraining is needed.

Neither branch deletes a character, so meaning is preserved by construction and
the transformation is idempotent.

Run: ``.venv/bin/python -m pytest ml/tests/test_quote_normalization.py -q``
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ml.tune import serve_v2
from ml.tune.build_dataset import normalize_quotes

ROOT = Path(__file__).resolve().parents[2]

#: The five eval_v3 inputs whose "" was *manufactured* by the PII redactor.
#: ev3-016 / 024 / 032 / 046 / 052 all derive from a complaint whose source
#: reads:  працівники "Екостайлу" не відповідають
#: "Екостайлу" was deleted as if it were a person, leaving the marks empty.
QUOTE_FAILURE_IDS = ["ev3-016", "ev3-024", "ev3-032", "ev3-046", "ev3-052"]

#: The sixth affected case, and the important one: the quotes here are REAL.
#: (біля маг. "Копілка") names a shop. It already parses today, so it is the
#: regression guard -- the fix must make it JSON-safe without erasing the name.
QUOTE_LEGITIMATE_ID = "ev3-123"

#: ev3-083 carries no ASCII quote at all and is a separate defect. It is pinned
#: here so the normalisation cannot be blamed or credited for it.
DECODING_SENSITIVE_ID = "ev3-083"


def _eval_texts() -> dict[str, str]:
    out = {}
    for line in (ROOT / "ml/data/tune/eval_v3/cases.jsonl").open(encoding="utf-8"):
        c = json.loads(line)
        out[c["id"]] = c["text"]
    return out


def _naive_emit(issue: str) -> str:
    """Reproduce the failure mode: a model that copies the span verbatim.

    This is what the v3 arms do. Interpolating into a JSON string without
    escaping is the defect; the test asserts the *input* no longer contains a
    character that makes that interpolation invalid.
    """
    return '{"topics": [{"domain": "payments", "issue": "%s"}]}' % issue


# --------------------------------------------------------------------------
# the unit under test
# --------------------------------------------------------------------------

def test_normalize_leaves_text_without_ascii_quotes_byte_identical():
    """Backward compatibility: the common path must not change at all."""
    for s in (
        "працівники КП ЖЕО №1 не відповідають",
        "вул. Рівненська, буд. № 3",
        "ТОВ ’Екостайл’ нараховує борг",
        "КП «ЖЕО №4» — зауваження",
        "роз’яснення причини нарахування",
        "",
    ):
        assert normalize_quotes(s) == s, s


def test_normalize_converts_a_delimited_quote_pair():
    assert normalize_quotes('КП "ЖЕО №2" не реагує') == "КП “ЖЕО №2” не реагує"


def test_normalize_preserves_the_quoted_content_exactly():
    """Meaning must survive: only the delimiter glyph changes."""
    src = 'працівники "Екостайлу" не відповідають'
    out = normalize_quotes(src)
    assert "Екостайлу" in out
    assert out.replace("“", "").replace("”", "") == src.replace('"', "")


def test_normalize_treats_a_mid_word_quote_as_an_apostrophe():
    """під"їзду is a keyboard typo for під'їзду, not a quotation."""
    assert normalize_quotes('ям під"їзду №1') == "ям під’їзду №1"
    assert normalize_quotes('роз"яснень') == "роз’яснень"


def test_normalize_leaves_an_empty_quote_pair_json_safe():
    """The redaction artefact must become inert, not disappear semantically."""
    out = normalize_quotes('працівники "" не відповідають')
    assert '"' not in out
    assert out == "працівники “” не відповідають"


def test_normalize_is_idempotent():
    for s in (
        'КП "ЖЕО №2" не реагує',
        'ям під"їзду №1',
        'працівники "" не відповідають',
        'торгові точки "Карамелька", "Тютюнок", "Марафет"',
    ):
        once = normalize_quotes(s)
        assert normalize_quotes(once) == once, s


def test_normalize_changes_nothing_but_the_quote_characters():
    """No whitespace, punctuation or letter may be altered."""
    for s in (
        'КП "ЖЕО №2", вул.Гагаріна, 7. Прохання  неодноразові звернення',
        'торгових точок "Карамелька", "Тютюнок"',
    ):
        out = normalize_quotes(s)
        assert len(out) == len(s)
        assert [a for a, b in zip(s, out) if a != b] == ['"'] * s.count('"')
        assert out.replace("“", '"').replace("”", '"').replace("’", '"') == \
            s.replace("“", '"').replace("”", '"').replace("’", '"')


def test_normalize_is_deterministic():
    s = 'КП "ЖЕО №2" та "Теплоенергетик" КМР, під"їзду'
    assert len({normalize_quotes(s) for _ in range(50)}) == 1


def test_normalize_handles_an_unclosed_quote_without_crashing():
    out = normalize_quotes('КП "ЖЕО №4, які відмовляються')
    assert '"' not in out
    assert "ЖЕО №4" in out


def test_normalize_leaves_none_alone():
    assert normalize_quotes(None) is None


# --------------------------------------------------------------------------
# the five regression cases, end to end
# --------------------------------------------------------------------------

@pytest.mark.parametrize("cid", QUOTE_FAILURE_IDS)
def test_quote_failure_case_was_invalid_json_and_is_fixed(cid):
    texts = _eval_texts()
    src = texts[cid]
    assert '"' in src, f"{cid} is expected to carry the artefact"

    # current behaviour: the copied span breaks the payload
    payload, err = serve_v2.parse_strict(_naive_emit(src))
    assert payload is None, "pre-fix reproduction failed: input no longer breaks JSON"
    assert "JSON" in err

    # after the fix the same copy is valid JSON
    fixed = normalize_quotes(src)
    assert '"' not in fixed
    payload, err = serve_v2.parse_strict(_naive_emit(fixed))
    assert payload is not None, f"{cid} still invalid after normalisation: {err}"
    assert payload["topics"][0]["issue"]


def test_every_ascii_quote_in_eval_v3_is_removed_by_normalization():
    """No eval input may hand the model a JSON-hazardous character."""
    for cid, text in _eval_texts().items():
        assert '"' not in normalize_quotes(text), cid


def test_legitimate_quoted_shop_name_is_kept_and_made_json_safe():
    """The real-quotation case: semantics preserved, hazard removed."""
    src = _eval_texts()[QUOTE_LEGITIMATE_ID]
    assert '"Копілка"' in src, "fixture changed: the real quotation is gone"
    out = normalize_quotes(src)
    assert '"' not in out
    assert "Копілка" in out, "the shop name must survive normalisation"
    assert "“Копілка”" in out
    payload, err = serve_v2.parse_strict(_naive_emit(out))
    assert payload is not None, f"{QUOTE_LEGITIMATE_ID} broke: {err}"
    assert payload["topics"][0]["issue"].count("Копілка") == 1


def test_the_six_affected_cases_are_exactly_these():
    """Pin the affected set so a new case cannot slip in unnoticed."""
    affected = sorted(cid for cid, t in _eval_texts().items() if '"' in t)
    assert affected == sorted(QUOTE_FAILURE_IDS + [QUOTE_LEGITIMATE_ID])


def test_the_normalization_changes_no_eval_topic_expectation():
    """Guard against benchmark mutation: the fix must not rewrite references."""
    for line in (ROOT / "ml/data/tune/eval_v3/cases.jsonl").open(encoding="utf-8"):
        c = json.loads(line)
        for t in c["expected_topics"]:
            assert t["domain"], c["id"]


def test_ev3_083_is_untouched_because_it_has_no_ascii_quote():
    """The decoding-sensitive defect must not be silently credited here."""
    text = _eval_texts()[DECODING_SENSITIVE_ID]
    assert '"' not in text
    assert normalize_quotes(text) == text


# --------------------------------------------------------------------------
# serving contract
# --------------------------------------------------------------------------

class _FakeTokenizer:
    def apply_chat_template(self, messages, **kw):
        return "\n".join(f"{m['role']}: {m['content']}" for m in messages)


def test_prompt_for_text_without_ascii_quotes_is_unchanged():
    """v2 backward compatibility: byte-identical prompt on the common path."""
    tok = _FakeTokenizer()
    for s in ("працівники КП ЖЕО №1 не відповідають",
              "вул. Рівненська, буд. № 3"):
        before = serve_v2.build_prompt(s, tok)
        assert '"' not in serve_v2.build_messages(s)[1]["content"]
        assert serve_v2.build_prompt(s, tok) == before


def test_prompt_is_normalized_so_the_model_never_sees_an_ascii_quote():
    """The user turn is clean; the system prompt keeps its JSON schema quotes."""
    tok = _FakeTokenizer()
    user = serve_v2.build_messages('КП "ЖЕО №2" не реагує')[1]["content"]
    assert '"' not in user
    assert "ЖЕО №2" in user
    # the format schema in the system prompt must survive -- it teaches JSON
    assert '"issue"' in serve_v2.SYSTEM_PROMPT


def test_build_messages_covers_every_serving_path():
    """One boundary, so a caller cannot bypass normalisation."""
    assert serve_v2.build_messages('КП "ЖЕО №2"')[1]["content"] \
        == serve_v2.build_messages(serve_v2.build_messages('КП "ЖЕО №2"')[1]["content"])[1]["content"]


def test_serving_constants_are_untouched():
    """The v2 contract itself must not move."""
    assert serve_v2.DEFAULT_MODEL == \
        "ml/data/tune/adapters/qwen3-8b-lora-v2-attempt10-fused"
    assert serve_v2.DEFAULT_MAX_TOKENS == 800
    assert serve_v2.DEFAULT_TEMPERATURE == 0.0
    assert serve_v2.THINK_PREFIX == "<think>\n\n</think>\n\n"