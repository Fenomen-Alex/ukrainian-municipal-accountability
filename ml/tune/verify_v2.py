"""Deterministic end-to-end verification of the v2 artifact.

Phase 5 of the serving investigation. Drives the *canonical* inference contract
(:mod:`ml.tune.serve_v2`) and asserts the full production contract.

The checks are deliberately split into two tiers, because conflating them would
hide the very thing this investigation is about:

**Contract** (fatal). Serving-format conformance. These are the properties that
must hold for the artifact to be published as an API model, and they are
format-sensitive -- a wrong chat template or thinking mode breaks them:

* the rendered prompt ends in the exact ``<think>\\n\\n</think>\\n\\n`` prefix
* strict JSON, with *no* wrapper tolerance (unlike the historical scorer, which
  greps ``\\{.*\\}`` and would happily accept a leading ``!``)
* every topic validates against the frozen production schema
* every ``domain`` is a schema enum member -- a free-text "domain" is a
  hallucination, which is exactly the failure reported from LM Studio
* no thinking text, no markdown fence, no leading ``!``, no prose before the JSON
* no malformed or empty topic objects
* single-topic input yields exactly one topic (no over-emission)
* ``requested_action`` is empty when the text contains no request verb

**Quality** (reported, non-fatal). Model capability, already characterised in
``ml/reports/v2_error_analysis.md`` and ``ml/reports/finetune_v2_report.md``:

* topic count and domain set on multi-topic input (v2 held-out real multi-topic
  recall is 0/2; smoke multi-topic is 1/4)
* ``requested_action`` recall when a request *is* present (the weak labeler puts
  the request sentence inside ``issue`` 98% of the time, so the model tends to
  merge it rather than split it out)

These are reported, never silently repaired.

Usage::

    .venv-mlx/bin/python -m ml.tune.verify_v2 --native
    .venv-mlx/bin/python -m ml.tune.verify_v2 --base-url http://127.0.0.1:1234/v1
"""

from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

from ml.tune.build_dataset import SYSTEM_PROMPT, _derive_requested_action
from ml.tune.serve_v2 import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL,
    DEFAULT_TEMPERATURE,
    THINK_PREFIX,
    build_messages,
    parse_strict,
)

SCHEMA_DOMAINS = {
    "roads", "water", "heating", "housing", "transport", "sanitation",
    "electricity", "construction", "benefits", "government", "commerce",
    "payments", "other",
}
TOPIC_KEYS = {"domain", "issue", "object", "requested_action", "attributes"}


@dataclass
class Case:
    id: str
    text: str
    #: 1 for single-topic, 2 for multi-topic. Contract only checks n==1.
    n_topics: int
    domains: list[str] = field(default_factory=list)
    has_request_verb: bool = False
    note: str = ""


# --- the three complaints from the investigation brief -------------------
CASE_A = Case(
    id="A-two-problems",
    text=("На вулиці Шевченка вже тиждень не горять ліхтарі. "
          "Також біля будинку 27 розбите дорожнє покриття, прошу відремонтувати."),
    n_topics=2, domains=["electricity", "roads"], has_request_verb=True,
    note="street lighting + road surface, one explicit request",
)
CASE_B = Case(
    id="B-two-services",
    text="Немає гарячої води і ще не вивезли сміття біля будинку.",
    n_topics=2, domains=["water", "sanitation"], has_request_verb=False,
    note="hot water + waste removal joined by 'і ще'",
)
CASE_C = Case(
    id="C-colloquial-two-topic",
    text=("Добрий день, скільки можна вже з цими проблемами, біля 15 будинку яма "
          "така що дитину з коляскою нормально не проїхати, і ще гілки дерева "
          "звисають прямо на тротуар, зробіть щось будь ласка."),
    n_topics=2, domains=["roads", "sanitation"], has_request_verb=False,
    note="colloquial pothole + overhanging branch, rambling register. "
         "'зробіть' is a real request but the labeler does not match it, so the "
         "reference target is an empty requested_action",
)

# --- repository smoke cases, verbatim from smoke_cases.json --------------
# expected_domains there is a free-form vocabulary; eval_smoke.EXPECTED_TO_SCHEMA
# maps it to schema enums, and these are the mapped enums.
CASE_SMOKE13 = Case(
    id="smoke-13",
    text=("По вулиці Незалежності від будинку 1 до 15 не працює жоден вуличний "
          "ліхтар, а також на всій цій ділянці глибокі ями на дорозі. Прошу "
          "відновити освітлення та відремонтувати дорогу."),
    n_topics=2, domains=["electricity", "roads"], has_request_verb=True,
    note="repo smoke-13 (Multi-Topic), the one multi case attempt-10 fixed",
)
CASE_SMOKE01 = Case(
    id="smoke-01",
    text=("На вулиці Шевченка, біля будинку №24, утворилася глибока яма на "
          "проїжджій частині. Прошу провести ямковий ремонт асфальтного покриття."),
    n_topics=1, domains=["roads"], has_request_verb=True,
    note="repo smoke-01 (Standard Concrete)",
)

# --- single-topic --------------------------------------------------------
CASE_SINGLE_A = Case(
    id="single-no-request",
    text="Вже другий тиждень у під'їзді будинку 27 не працює ліфт.",
    n_topics=1, domains=["housing"], has_request_verb=False,
    note="single topic, no request verb -> requested_action must be empty",
)
CASE_SINGLE_B = Case(
    id="single-with-request",
    text="Прошу відремонтувати дитячий майданчик у парку на вулиці Шевченка.",
    n_topics=1, domains=["sanitation"], has_request_verb=True,
    note="single topic, explicit request",
)

# --- multi-topic ---------------------------------------------------------
CASE_MULTI_A = Case(
    id="multi-lift-and-leak",
    text=("Надати роз'яснення чому працівники від'єднали кабель від будинку по "
          "вул. Шатила, 3, а також відновити роботу ліфтів у будинку."),
    n_topics=2, domains=["electricity", "housing"], has_request_verb=False,
    note="same shape as repo MT-REAL-Б-292. 'Надати роз'яснення' is a real request "
         "the labeler misses, so the reference target is an empty requested_action",
)
CASE_MULTI_B = Case(
    id="multi-two-sentences",
    text=("Біля будинку 27 на вулиці Шевченка немає гарячої води. "
          "Ще не вивезли сміття біля того ж будинку."),
    n_topics=2, domains=["water", "sanitation"], has_request_verb=False,
    note="two clean sentences, no request verb",
)

ALL_CASES = [
    CASE_A, CASE_B, CASE_C,
    CASE_SMOKE13, CASE_SMOKE01,
    CASE_SINGLE_A, CASE_SINGLE_B,
    CASE_MULTI_A, CASE_MULTI_B,
]


# The anti-hallucination rule ("requested_action empty when the text has no
# request") is checked against the *labeler that produced the training targets*,
# not a hand-kept verb list, so the two cannot drift apart.
def has_request_verb(text: str) -> bool:
    """True when ``build_dataset._derive_requested_action`` yields a request.

    This is deliberately the labeler's own definition rather than an
    independent verb list. The model was trained to reproduce this transform, so
    matching it is the correct bar. Note it is a *weak* oracle: it misses real
    requests such as "Надати роз'яяснення ...", which is itself one of the
    documented labeler defects in ml/reports/contract_audit.md.
    """
    return bool(_derive_requested_action(text))


def check_case(case: Case, raw: str, prompt_tail_ok: bool) -> tuple[list[str], list[str], dict]:
    """Return (contract_failures, quality_notes, observed). Never repairs output."""
    fatal: list[str] = []
    notes: list[str] = []
    obs: dict = {"id": case.id, "raw": raw}

    # --- contract: format sensitivity -----------------------------------
    if not prompt_tail_ok:
        fatal.append("prompt did not end with the empty-<think> prefix")

    stripped = raw.lstrip()
    if "<think>" in raw or "</think>" in raw:
        fatal.append("thinking text present in output")
    if stripped.startswith("```"):
        fatal.append("markdown fence in output")
    if stripped.startswith("!"):
        fatal.append("wrapper '!' before JSON")
    if stripped[:1].isalpha():
        fatal.append("prose before JSON")

    payload, err = parse_strict(raw)
    if err:
        fatal.append(err)
        return fatal, notes, obs
    obs["payload"] = payload

    topics = payload["topics"]
    obs["n_topics"] = len(topics)

    for i, t in enumerate(topics):
        if not isinstance(t, dict):
            fatal.append(f"topic {i} is {type(t).__name__}, expected object")
            continue
        missing = TOPIC_KEYS - set(t)
        extra = set(t) - TOPIC_KEYS
        if missing:
            fatal.append(f"topic {i} missing keys {sorted(missing)}")
        if extra:
            fatal.append(f"topic {i} has unexpected keys {sorted(extra)}")
        dom = t.get("domain")
        if not isinstance(dom, str) or dom not in SCHEMA_DOMAINS:
            fatal.append(f"topic {i} domain {dom!r} is not a schema enum value")
        for f in ("issue", "object", "requested_action", "attributes"):
            v = t.get(f)
            if f == "attributes":
                if v is not None and not isinstance(v, dict):
                    fatal.append(f"topic {i} attributes is {type(v).__name__}, expected object")
            elif v is not None and not isinstance(v, str):
                fatal.append(f"topic {i} {f} is {type(v).__name__}, expected string")
        if not str(t.get("issue", "")).strip():
            fatal.append(f"topic {i} issue is empty (malformed topic)")

    # --- contract: no over-emission, no invented action ------------------
    if case.n_topics == 1 and len(topics) != 1:
        fatal.append(f"single-topic input emitted {len(topics)} topics")

    want_action = has_request_verb(case.text)
    if not want_action:
        for i, t in enumerate(topics):
            if isinstance(t, dict) and str(t.get("requested_action", "")).strip():
                fatal.append(
                    f"topic {i} invented requested_action where the text has no request verb"
                )

    # --- quality: known model limitations, reported not fatal ------------
    if len(topics) != case.n_topics:
        notes.append(f"topic count {len(topics)} (target {case.n_topics})")
    if case.domains:
        got = [t.get("domain") for t in topics if isinstance(t, dict)]
        if got != case.domains:
            notes.append(f"domains {got} (target {case.domains})")
    if case.n_topics >= 2 and len(topics) >= 2:
        issues = [str(t.get("issue", "")) for t in topics if isinstance(t, dict)]
        if len(set(issues)) != len(issues):
            notes.append("two topics carry identical issue text (over-conflation)")
    if case.has_request_verb and topics:
        if not any(str(t.get("requested_action", "")).strip()
                   for t in topics if isinstance(t, dict)):
            notes.append("requested_action left empty although a request verb is present")

    return fatal, notes, obs


# --- transports ----------------------------------------------------------
def run_native(model_path, cases, max_tokens, temperature):
    from ml.tune.serve_v2 import build_prompt, generate_text, load_model

    model, tokenizer = load_model(model_path)
    for case in cases:
        prompt = build_prompt(case.text, tokenizer)
        raw = generate_text(model, tokenizer, case.text,
                            max_tokens=max_tokens, temperature=temperature)
        yield case, raw, prompt.endswith(THINK_PREFIX)


def _post(url: str, body: dict) -> dict:
    req = urllib.request.Request(
        url, data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=900) as resp:
        return json.loads(resp.read().decode("utf-8"))


def run_http(base_url, model_id, cases, max_tokens, temperature):
    """The correct LM Studio call: explicit system message, thinking disabled."""
    for case in cases:
        body = {
            "model": model_id,
            "messages": build_messages(case.text),
            "temperature": temperature,
            "max_tokens": max_tokens,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        data = _post(f"{base_url}/chat/completions", body)
        yield case, data["choices"][0]["message"]["content"], True


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--native", action="store_true",
                    help="run in-process with MLX instead of over HTTP")
    ap.add_argument("--base-url", default="http://127.0.0.1:1234/v1")
    ap.add_argument("--model-id", default=None)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    ap.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    cases = ALL_CASES
    if args.only:
        cases = [c for c in ALL_CASES if c.id in args.only]
        if not cases:
            raise SystemExit(f"no cases matched {args.only}")

    if args.native:
        stream = run_native(args.model, cases, args.max_tokens, args.temperature)
    else:
        stream = run_http(args.base_url, args.model_id or args.model, cases,
                          args.max_tokens, args.temperature)

    rows, fatal_all, notes_all = [], [], []
    for case, raw, tail_ok in stream:
        fatal, notes, obs = check_case(case, raw, tail_ok)
        obs["text"] = case.text
        obs["note"] = case.note
        obs["contract_failures"] = fatal
        obs["quality_notes"] = notes
        rows.append(obs)
        fatal_all.extend(f"{case.id}: {f}" for f in fatal)
        notes_all.extend(f"{case.id}: {n}" for n in notes)
        print(f"[{'PASS' if not fatal else 'FAIL'}] {case.id}  "
              f"({obs.get('n_topics', '?')} topics)")
        for f in fatal:
            print(f"        CONTRACT: {f}")
        for n in notes:
            print(f"        quality:   {n}")
        if fatal:
            print(f"        raw: {raw[:500]}")

    n_pass = sum(1 for r in rows if not r["contract_failures"])
    print()
    print(f"CONTRACT: {n_pass}/{len(cases)} cases pass")
    print(f"QUALITY notes: {len(notes_all)}")
    if fatal_all:
        print("\nCONTRACT FAILURES:")
        for f in fatal_all:
            print(f"  - {f}")

    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\nreport -> {args.json_out}")

    raise SystemExit(1 if fatal_all else 0)


if __name__ == "__main__":
    main()
