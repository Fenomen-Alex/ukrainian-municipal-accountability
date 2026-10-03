"""Canonical-v2 release smoke suite.

Eight complaint shapes that a fresh integrator is most likely to send first,
checked against the *contract* rather than against model wording. Wording is not
a project contract: these tests never assert what the model should say, only
that the prompt is built correctly, that quote normalization behaves, that a
conforming payload satisfies the schema, and that malformed output is surfaced
instead of repaired.

Two layers:

* model-free (always runs) -- prompt construction, normalization, schema
  constants, strict-parse behaviour, absence of post-processing;
* model-gated (skipped when the fused artifact is absent) -- the real v2 arm on
  all eight shapes, asserting strict-JSON validity and no output mutation.

Runtime for the canonical path is MLX via ``ml.tune.serve_v2``; this suite
intentionally needs no server.

Run: ``.venv/bin/python -m pytest ml/tests/test_release_smoke.py -q``
"""

from __future__ import annotations

import importlib.util
import json
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from ml.tune.build_dataset import SYSTEM_PROMPT, normalize_quotes
from ml.tune.serve_v2 import (
    BASE_MODEL,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL,
    DEFAULT_TEMPERATURE,
    THINK_PREFIX,
    build_messages,
    build_prompt,
    parse_strict,
)
from ml.tune.verify_v2 import SCHEMA_DOMAINS, TOPIC_KEYS, Case as V2Case, check_case

FUSED_DIR = Path(DEFAULT_MODEL)


@dataclass
class SmokeCase:
    id: str
    text: str
    n_topics: int
    #: ASCII quotes in the raw text, by expectation
    ascii_delimited: bool = False   # "name" -> typographic pair
    ascii_apostrophe: bool = False  # mid-word -> U+2019
    #: True when the complaint itself names a remedy, which is what licenses a
    #: non-empty requested_action. The model must not invent one otherwise.
    explicit_request: bool = False
    note: str = ""
    expect_words: list[str] = field(default_factory=list)


SMOKE_CASES: list[SmokeCase] = [
    SmokeCase(
        id="single-topic",
        text="На вулиці Шевченка вже тиждень не горять ліхтарі.",
        n_topics=1,
        note="the simplest single-issue complaint",
        expect_words=["ліхтарі"],
    ),
    SmokeCase(
        id="multitopic",
        text=("На вулиці Шевченка вже тиждень не горять ліхтарі. "
              "Також біля будинку 27 розбите дорожнє покриття, прошу відремонтувати."),
        n_topics=2,
        explicit_request=True,
        note="two distinct issues in one complaint",
        expect_words=["ліхтарі", "покриття"],
    ),
    SmokeCase(
        id="quoted-organisation",
        text='Біля маг. "Копілка" на вул. Героїв Маріуполя три тижні тече каналізація.',
        n_topics=1,
        ascii_delimited=True,
        note="real quoted shop name; the case that motivated the quote fix",
        expect_words=["Копілка", "каналізація"],
    ),
    SmokeCase(
        id="ukrainian-apostrophe",
        text='Прошу відремонтувати яму під"їзду №1, роз"яснення прохання не надано.',
        n_topics=1,
        ascii_apostrophe=True,
        note="mid-word ASCII quotes are apostrophe typos, not quotation marks",
        expect_words=["їзду", "яснення"],
    ),
    SmokeCase(
        id="quote-free",
        text="Прошу відремонтувати дорожнє покриття на вул. Рівненській, буд. № 3.",
        n_topics=1,
        note="the common case: prompt must be byte-identical to the pre-fix path",
        expect_words=["Рівненській"],
    ),
    SmokeCase(
        id="terse",
        text="Не горять ліхтарі.",
        n_topics=1,
        note="minimum viable complaint, no politeness padding",
        expect_words=["ліхтарі"],
    ),
    SmokeCase(
        id="explicit-request",
        text="Прошу негайно відремонтувати асфальтне покриття на вул. Київській, 12.",
        n_topics=1,
        explicit_request=True,
        note="explicit requested action",
        expect_words=["ремонтувати"],
    ),
    SmokeCase(
        id="no-explicit-request",
        text="Біля будинку 27 вже другий місяць не горять ліхтарі.",
        n_topics=1,
        note="states a problem without naming a remedy",
        expect_words=["ліхтарі"],
    ),
]

CASE_IDS = [c.id for c in SMOKE_CASES]


def _case(cid: str) -> SmokeCase:
    return next(c for c in SMOKE_CASES if c.id == cid)


class _FakeTokenizer:
    """Renders the chat template without needing MLX or a tokenizer file."""

    def apply_chat_template(self, messages, **kw):
        return "\n".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>"
                         for m in messages)


# --------------------------------------------------------------------------
# the eight shapes are present and well formed
# --------------------------------------------------------------------------

def test_the_eight_required_shapes_are_covered():
    assert CASE_IDS == [
        "single-topic", "multitopic", "quoted-organisation",
        "ukrainian-apostrophe", "quote-free", "terse",
        "explicit-request", "no-explicit-request",
    ]


@pytest.mark.parametrize("case", SMOKE_CASES, ids=CASE_IDS)
def test_case_text_is_nonempty_and_has_the_expected_quote_shape(case):
    assert case.text.strip()
    n = case.text.count('"')
    if case.ascii_delimited:
        assert n == 2 and case.text.count('""') == 0
    elif case.ascii_apostrophe:
        assert n == 2 and '""' not in case.text
    else:
        assert n == 0
    for word in case.expect_words:
        assert word in case.text, f"{case.id}: fixture lost {word!r}"


# --------------------------------------------------------------------------
# prompt construction
# --------------------------------------------------------------------------

@pytest.mark.parametrize("case", SMOKE_CASES, ids=CASE_IDS)
def test_messages_are_system_then_user(case):
    msgs = build_messages(case.text)
    assert [m["role"] for m in msgs] == ["system", "user"]
    assert msgs[0]["content"] == SYSTEM_PROMPT
    assert msgs[1]["content"] == normalize_quotes(case.text).strip()


@pytest.mark.parametrize("case", SMOKE_CASES, ids=CASE_IDS)
def test_system_prompt_is_never_mutated(case):
    """The system turn must be byte-identical for every complaint shape."""
    baseline = build_messages("")[0]["content"]
    for c in SMOKE_CASES:
        assert build_messages(c.text)[0]["content"] == baseline
    assert baseline == SYSTEM_PROMPT
    # the schema quotes in the system prompt are what teach the output format
    assert '"issue"' in baseline and '"requested_action"' in baseline


@pytest.mark.parametrize("case", SMOKE_CASES, ids=CASE_IDS)
def test_prompt_never_carries_an_ascii_quote_in_the_user_turn(case):
    prompt = build_prompt(case.text, _FakeTokenizer())
    user = build_messages(case.text)[1]["content"]
    assert '"' not in user
    if case.text.count('"'):
        assert '"' in prompt, "system prompt schema quotes must survive"


def test_quote_free_cases_produce_byte_identical_prompts():
    """The compatibility guarantee: no-quote traffic is untouched."""
    tok = _FakeTokenizer()
    for cid in ("quote-free", "terse", "single-topic"):
        text = _case(cid).text
        assert '"' not in text
        assert build_messages(text)[1]["content"] == text.strip()
        assert build_prompt(text, tok) == build_prompt(text, tok)


def test_pinned_generation_settings_are_unchanged():
    assert DEFAULT_MODEL == "ml/data/tune/adapters/qwen3-8b-lora-v2-attempt10-fused"
    assert BASE_MODEL == "mlx-community/Qwen3-8B-4bit"
    assert DEFAULT_MAX_TOKENS == 800
    assert DEFAULT_TEMPERATURE == 0.0
    assert THINK_PREFIX == "<think>\n\n</think>\n\n"


# --------------------------------------------------------------------------
# quote normalization
# --------------------------------------------------------------------------

def test_quoted_organisation_becomes_typographic_but_keeps_the_name():
    out = normalize_quotes(_case("quoted-organisation").text)
    assert "Копілка" in out, "the shop name must survive"
    assert "“Копілка”" in out
    assert '"' not in out


def test_mid_word_quotes_become_apostrophes_not_quotes():
    out = normalize_quotes(_case("ukrainian-apostrophe").text)
    assert "під’їзду" in out and "роз’яснення" in out
    assert "“" not in out and "”" not in out
    assert '"' not in out


def test_normalization_is_idempotent_across_all_shapes():
    for case in SMOKE_CASES:
        once = normalize_quotes(case.text)
        assert normalize_quotes(once) == once, case.id


def test_normalization_preserves_every_word_of_every_shape():
    """Meaning is preserved: only quote glyphs change, never content."""
    for case in SMOKE_CASES:
        stripped = normalize_quotes(case.text)
        assert len(stripped) == len(case.text), case.id
        assert [a for a, b in zip(case.text, stripped) if a != b] == \
            ['"'] * case.text.count('"'), case.id


# --------------------------------------------------------------------------
# schema validity (model-free: the schema itself)
# --------------------------------------------------------------------------

def test_schema_matches_the_production_contract():
    assert TOPIC_KEYS == {"domain", "issue", "object",
                          "requested_action", "attributes"}
    assert "roads" in SCHEMA_DOMAINS and "other" in SCHEMA_DOMAINS
    for d in ("roads", "water", "heating", "payments", "sanitation"):
        assert d in SCHEMA_DOMAINS


def _conforming_payload(domains: list[str], explicit_request: bool = True) -> str:
    """A schema-valid answer that also respects the action regime: a remedy is
    only emitted when the complaint actually asks for one."""
    return json.dumps({"topics": [
        {"domain": d, "issue": f"проблема {i + 1}", "object": "вул. Тестова, 1",
         "requested_action": "відремонтувати" if explicit_request else "",
         "attributes": {}}
        for i, d in enumerate(domains)
    ]}, ensure_ascii=False)


@pytest.mark.parametrize("case", SMOKE_CASES, ids=CASE_IDS)
def test_a_conforming_payload_passes_the_contract_checker(case):
    """A well-formed answer to this shape must be accepted, with no notes."""
    dom = ["roads", "sanitation"][: case.n_topics]
    vc = V2Case(id=case.id, text=case.text, n_topics=case.n_topics, domains=dom,
                has_request_verb=case.explicit_request)
    fatal, notes, _ = check_case(
        vc, _conforming_payload(dom, case.explicit_request), True)
    assert fatal == [], fatal
    assert notes == [], notes


@pytest.mark.parametrize("case", SMOKE_CASES, ids=CASE_IDS)
def test_strict_parse_returns_the_generation_unchanged(case):
    """No post-processing: the parsed payload is exactly what json.loads gives."""
    raw = _conforming_payload(["roads"][: case.n_topics], case.explicit_request)
    payload, err = parse_strict(raw)
    assert err == ""
    assert payload == json.loads(raw)
    assert payload["topics"]


def test_strict_parse_still_surfaces_malformed_output():
    """Nothing is repaired; bad output is reported, not cleaned."""
    for bad in ('{"topics": [{"issue": "a", }]}',
                '```json\n{"topics": []}\n```',
                '! {"topics": []}',
                'Here is the JSON: {"topics": []}',
                'не JSON взагалі',
                ''):
        payload, err = parse_strict(bad)
        assert payload is None, bad
        assert err, bad


# --------------------------------------------------------------------------
# model-gated: the real canonical v2 arm
# --------------------------------------------------------------------------

@pytest.mark.skipif(not FUSED_DIR.exists(),
                    reason=f"fused artifact not present: {FUSED_DIR}")
@pytest.mark.skipif(importlib.util.find_spec("mlx_lm") is None,
                    reason="mlx_lm not installed (canonical runtime is .venv-mlx)")
@pytest.mark.parametrize("case", SMOKE_CASES, ids=CASE_IDS)
def test_canonical_model_returns_strict_json_for_every_shape(case):
    from ml.tune.serve_v2 import generate_text, load_model

    model, tok = load_model()
    raw = generate_text(model, tok, case.text)
    payload, err = parse_strict(raw)
    assert err == "", f"{case.id}: {err}\nraw={raw[:300]}"
    assert isinstance(payload["topics"], list) and payload["topics"]
    for topic in payload["topics"]:
        assert set(topic) == TOPIC_KEYS, f"{case.id}: {sorted(topic)}"
        assert topic["domain"] in SCHEMA_DOMAINS, f"{case.id}: {topic['domain']}"
    # the output must not have been post-processed into shape
    assert payload == json.loads(raw.strip())
    assert "<think>" not in raw