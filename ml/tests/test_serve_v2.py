"""Lock the v2 serving contract.

The v2 fine-tune is prompt-anchored: every one of its 8,519 training examples
carried the exact ``build_dataset.SYSTEM_PROMPT`` as ``messages[0]`` and was
tokenized with Qwen3's chat template at ``enable_thinking=False``. Serving it
with anything else produces no JSON at all.

These tests need no model, no MLX and no server -- they assert the *contract*
that :mod:`ml.tune.serve_v2` encodes, and they fail loudly if someone edits the
system prompt, drops the empty-``<think>`` prefix, or weakens the strict parser
into the permissive regex that originally hid this bug. The behaviour that needs
a GPU is covered by ``ml/tune/verify_v2.py``.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from ml.tune.build_dataset import OUT_DIR, SYSTEM_PROMPT
from ml.tune.serve_v2 import (
    BASE_MODEL,
    DEFAULT_MODEL,
    THINK_PREFIX,
    build_messages,
    build_prompt,
    parse_strict,
)
from ml.tune.verify_v2 import (
    ALL_CASES,
    SCHEMA_DOMAINS,
    TOPIC_KEYS,
    Case,
    check_case,
    has_request_verb,
)

TRAIN_JSONL = OUT_DIR / "v2" / "train.jsonl"
FUSED_DIR = Path("ml/data/tune/adapters/qwen3-8b-lora-v2-attempt10-fused")


# --- the contract itself --------------------------------------------------
def test_every_v2_training_example_carries_the_system_prompt():
    """The premise of the whole serving contract. If this fails, v2 is unservable
    as a plain chat model and the model card's instructions are wrong."""
    if not TRAIN_JSONL.exists():
        pytest.skip(f"{TRAIN_JSONL} not built")
    rows = [json.loads(l) for l in TRAIN_JSONL.read_text().splitlines()]
    assert rows, "v2 train set is empty"
    for i, row in enumerate(rows):
        msgs = row["messages"]
        assert msgs[0]["role"] == "system", f"row {i} does not start with a system message"
        assert msgs[0]["content"] == SYSTEM_PROMPT, f"row {i} has a different system prompt"
        assert [m["role"] for m in msgs] == ["system", "user", "assistant"]


def test_build_messages_puts_the_system_prompt_first():
    msgs = build_messages("  Не горять ліхтарі  ")
    assert msgs[0]["role"] == "system"
    assert msgs[0]["content"] == SYSTEM_PROMPT
    assert msgs[1] == {"role": "user", "content": "Не горять ліхтарі"}, "must be stripped"


def test_system_prompt_declares_exactly_the_schema_domains():
    """If a domain is added to the schema, the system prompt has to teach it,
    otherwise the model cannot emit it. Keep these two in lockstep."""
    declared = set(re.search(r"domain — одна зі значень: ([^.]+)\.", SYSTEM_PROMPT).group(1)
                   .replace(" ", "").split(","))
    assert declared == SCHEMA_DOMAINS, (
        "SYSTEM_PROMPT domain list has drifted from the verification enum; "
        f"prompt-only={sorted(declared - SCHEMA_DOMAINS)} "
        f"enum-only={sorted(SCHEMA_DOMAINS - declared)}"
    )


def test_think_prefix_matches_qwen_template_output():
    """`enable_thinking=False` must yield exactly this, or the prompt differs from
    every training example."""
    assert THINK_PREFIX == "<think>\n\n</think>\n\n"


def test_default_model_and_base_model_are_the_documented_ones():
    assert DEFAULT_MODEL.endswith("qwen3-8b-lora-v2-attempt10-fused")
    assert BASE_MODEL == "mlx-community/Qwen3-8B-4bit"


@pytest.mark.skipif(not FUSED_DIR.exists(), reason="fused artifact not present")
def test_fused_artifact_ships_a_chat_template_that_supports_enable_thinking():
    template = (FUSED_DIR / "chat_template.jinja").read_text()
    assert "enable_thinking" in template, "template cannot disable thinking"
    assert "'<think>\\n\\n</think>\\n\\n'" in template, (
        "template does not inject the empty-<think> block the fine-tune expects"
    )
    assert (FUSED_DIR / "model.safetensors").exists()
    assert (FUSED_DIR / "tokenizer.json").exists()


# --- strict parsing: must NOT tolerate the wrapper text that hid the bug ---
def test_parse_strict_accepts_clean_json():
    payload, err = parse_strict('{"topics": [{"domain": "roads"}]}')
    assert err == ""
    assert payload["topics"][0]["domain"] == "roads"


def test_parse_strict_tolerates_the_learned_surrounding_whitespace():
    payload, err = parse_strict('\n{"topics": []}\n')
    assert err == "", "v2 emits a leading newline; that must not be an error"
    assert payload == {"topics": []}


@pytest.mark.parametrize("bad,reason", [
    ("!\n{\"topics\": []}", "literal '!' wrapper"),
    ("Ось відповідь:\n{\"topics\": []}", "prose before the JSON"),
    ("```json\n{\"topics\": []}\n```", "markdown fence"),
    ("<think>міркуємо</think>{\"topics\": []}", "leaked thinking"),
    ("", "empty output"),
    ("[{\"domain\": \"roads\"}]", "top level is an array, not an object"),
    ("{\"items\": []}", "no 'topics' key"),
    ("не json", "prose only"),
])
def test_parse_strict_rejects_wrapper_and_non_json(bad, reason):
    """Each of these is a real LM Studio failure mode observed in the wild."""
    payload, err = parse_strict(bad)
    assert payload is None, f"{reason} should not parse"
    assert err, f"{reason} should produce an error"


def test_parse_strict_does_not_greedy_extract_from_a_degenerate_generation():
    """The historical scorer used `\\{.*\\}` and would have scored garbage as a
    pass. Guard that regression explicitly."""
    degenerate = '!\nЗвичайно, я бажаю зробити. {"topics": [{"domain": "roads"}]}'
    payload, err = parse_strict(degenerate)
    assert payload is None
    assert "not strict JSON" in err


# --- the contract checks themselves ---------------------------------------
def _ok_payload(**over) -> str:
    topic = {
        "domain": "roads", "issue": "яма на дорозі", "object": "",
        "requested_action": "", "attributes": {},
    }
    topic.update(over.pop("topic", {}))
    return json.dumps({"topics": [topic]}, ensure_ascii=False)


def test_check_case_accepts_a_contract_conforming_output():
    case = Case(id="t", text="яма на дорозі без прохання", n_topics=1,
                domains=["roads"], has_request_verb=False)
    fatal, notes, obs = check_case(case, _ok_payload(), True)
    assert fatal == []
    assert obs["n_topics"] == 1


def test_check_case_flags_non_enum_domain():
    case = Case(id="t", text="яма", n_topics=1, has_request_verb=False)
    raw = _ok_payload(topic={"domain": "вирішення та узбереження будинків"})
    fatal, _, _ = check_case(case, raw, True)
    assert any("not a schema enum value" in f for f in fatal)


def test_check_case_flags_invented_requested_action():
    case = Case(id="t", text="яма на дорозі", n_topics=1, has_request_verb=False)
    raw = _ok_payload(topic={"requested_action": "звернутися до мерії"})
    fatal, _, _ = check_case(case, raw, True)
    assert any("invented requested_action" in f for f in fatal)


def test_check_case_flags_over_emission_on_single_topic_input():
    case = Case(id="t", text="яма на дорозі", n_topics=1, has_request_verb=False)
    raw = json.dumps({"topics": [
        {"domain": "roads", "issue": "а", "object": "", "requested_action": "", "attributes": {}},
        {"domain": "water", "issue": "б", "object": "", "requested_action": "", "attributes": {}},
    ]}, ensure_ascii=False)
    fatal, _, _ = check_case(case, raw, True)
    assert any("single-topic input emitted 2 topics" in f for f in fatal)


def test_check_case_flags_missing_prompt_anchor():
    case = Case(id="t", text="яма", n_topics=1, has_request_verb=False)
    fatal, _, _ = check_case(case, _ok_payload(), False)
    assert any("empty-<think> prefix" in f for f in fatal)


def test_quality_gaps_are_notes_not_failures():
    """Known model limitations must not masquerade as serving failures, or the
    suite cannot be used to decide whether to publish."""
    case = Case(id="t", text="яма і гілки", n_topics=2,
                domains=["roads", "sanitation"], has_request_verb=True)
    merged = json.dumps({"topics": [
        {"domain": "sanitation", "issue": "яма і гілки разом", "object": "",
         "requested_action": "", "attributes": {}},
    ]}, ensure_ascii=False)
    fatal, notes, _ = check_case(case, merged, True)
    assert fatal == [], "under-splitting a multi-topic case is a quality gap, not a format bug"
    assert any("topic count 1" in n for n in notes)


# --- the case corpus ------------------------------------------------------
def test_verification_corpus_is_well_formed():
    assert len(ALL_CASES) == 9
    ids = [c.id for c in ALL_CASES]
    assert len(set(ids)) == len(ids), "duplicate case ids"
    # the brief's three complaints must be present verbatim
    texts = {c.id: c.text for c in ALL_CASES}
    assert texts["A-two-problems"] == (
        "На вулиці Шевченка вже тиждень не горять ліхтарі. "
        "Також біля будинку 27 розбите дорожнє покриття, прошу відремонтувати."
    )
    assert texts["B-two-services"] == "Немає гарячої води і ще не вивезли сміття біля будинку."
    assert texts["C-colloquial-two-topic"].startswith("Добрий день, скільки можна вже з цими проблемами, біля 15 будинку")


def test_corpus_has_the_required_mix():
    singles = [c for c in ALL_CASES if c.n_topics == 1]
    multis = [c for c in ALL_CASES if c.n_topics >= 2]
    assert len(singles) >= 2, "need at least 2 single-topic cases"
    assert len(multis) >= 2, "need at least 2 multi-topic cases"
    assert any(c.id.startswith("smoke-") for c in ALL_CASES), "need repo smoke cases"
    with_request = [c for c in ALL_CASES if c.has_request_verb]
    without = [c for c in ALL_CASES if not c.has_request_verb]
    assert len(with_request) >= 2 and len(without) >= 2, (
        "need both request-bearing and request-free cases to test requested_action"
    )


def test_has_request_verb_agrees_with_the_case_flags():
    for c in ALL_CASES:
        assert c.has_request_verb == has_request_verb(c.text), (
            f"{c.id}: case flag says has_request_verb={c.has_request_verb} but the "
            f"detector says {has_request_verb(c.text)}"
        )


def test_expected_domains_are_schema_enums():
    for c in ALL_CASES:
        for d in c.domains:
            assert d in SCHEMA_DOMAINS, f"{c.id} expects non-enum domain {d!r}"


def test_topic_keys_match_the_production_schema():
    """Verification must grade against the frozen schema, not a copy of it."""
    schema = json.loads(Path("ml/data/gold/annotation_schema.json").read_text())
    topic = schema["definitions"]["topic"]
    assert TOPIC_KEYS == set(topic["properties"]), (
        "verification TOPIC_KEYS drifted from the frozen schema"
    )
    assert set(topic["required"]) == TOPIC_KEYS, "all five topic keys are required"
    assert topic["additionalProperties"] is False
    assert set(topic["properties"]["domain"]["enum"]) == SCHEMA_DOMAINS
