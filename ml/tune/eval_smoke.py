"""Targeted smoke evaluation: fine-tuned vs base model on 20 hand-written cases.

This is a QUALITATIVE / TARGETED smoke suite, NOT a statistical benchmark
(N=20, written by hand, no gold labels from the platform). It exists to find
behavioural differences and spot regressions, then to be inspected by a human.

Method
------
Each smoke case declares ``expected_domains`` in a richer, free-form taxonomy
(e.g. ``lighting``, ``waste_management``, ``water_leak``). The model was
trained to emit only the 13 schema domains. So before comparison the expected
domains are mapped onto acceptable *schema* domains via ``EXPECTED_TO_SCHEMA``
(below). The mapping is documented here so it can be audited; where a free-form
category genuinely spans several schema domains we allow all of them.

Mechanical per-case checks (all computed, nothing silently corrected):
  - JSON parse + schema validity
  - expected-domain coverage (each expected domain has >=1 predicted topic
    whose domain is in its acceptable set)
  - extra/unexpected domains (predicted domains not acceptable for any
    expected domain of that case)
  - field presence: non-empty issue / object / requested_action
  - attribute extraction: at least one non-empty attribute value
  - multi-topic recall: for cases with >1 expected domain, model emitted >=2
    distinct topics
  - hallucination flags: non-empty object/attributes values whose token overlap
    with the source text is below a threshold (i.e. not supported by the input)

A case is counted PASS only if: schema_valid AND full expected-domain coverage
AND no hallucination flag AND (for multi-topic cases) >=2 topics.

Usage::

    .venv-mlx/bin/python -m ml.tune.eval_smoke

Writes ``ml/reports/smoke_test_report.md`` and prints a per-case table.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from ml.tune.evaluate import load_validator

SMOKE_ROOT = Path(__file__).resolve().parent.parent / "data" / "tune" / "smoke"
SMOKE_CASES_PATH = SMOKE_ROOT / "smoke_cases.json"
RESULTS_ROOT = SMOKE_ROOT / "results"
REPORT_PATH = Path(__file__).resolve().parent.parent.parent / "ml" / "reports" / "smoke_test_report.md"

# free-form smoke domain -> acceptable schema domain(s)
# Audited against ml/data/gold/annotation_schema.json domain enum and the
# training KIND_TO_DOMAIN mapping. Only schema domains the model can emit.
EXPECTED_TO_SCHEMA: dict[str, set[str]] = {
    "roads": {"roads"},
    "lighting": {"electricity"},
    "water_supply": {"water"},
    "waste_management": {"sanitation"},
    "landscaping": {"sanitation", "construction", "other"},
    "hot_water": {"water", "heating"},
    "road_work_pedestrian": {"roads", "construction", "sanitation"},
    "manholes_infrastructure": {"water", "sanitation", "roads"},
    "water_leak": {"water"},
    "playgrounds": {"construction", "other", "housing", "sanitation"},
    "animals": {"other"},
    "elevators": {"housing", "other"},
    "sewage_basement": {"water", "housing", "sanitation"},
    "traffic_signs": {"roads", "construction"},
    "hot_water_leak": {"water", "heating"},
    "traffic_lights": {"electricity", "transport", "roads"},
    "housing_dispute_or_other": {"housing", "other"},
    "public_transport": {"transport"},
}

# Smoke case ids expected to contain >1 independent problem.
MULTI_TOPIC_CASES = {"smoke-13", "smoke-14", "smoke-15", "smoke-16"}

# categories that need qualitative judgement about "handling"
IMPLICIT_ACTION_CATEGORY = "Edge Case - Implicit Action"
VAGUE_LOCATION_CATEGORY = "Edge Case - Vague Location"
SARCASTIC_CATEGORY = "Messy & Sarcastic"
SURZHYK_CATEGORY = "Messy & Surzhyk"
PRIVATE_CATEGORY = "Edge Case - Non-Municipal / Private"


def load_smoke_cases(path: Path = SMOKE_CASES_PATH) -> list[dict]:
    return json.loads(path.read_text())


def load_results(tag: str) -> list[dict]:
    rows = (RESULTS_ROOT / tag / "smoke_results.jsonl").read_text().splitlines()
    return [json.loads(l) for l in rows]


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"\w+", text.lower()))


def _stem_supported(value_token: str, src_tokens: set[str]) -> bool:
    """Token is grounded if it (or a >=5-char stem) appears in the source."""
    if value_token in src_tokens:
        return True
    for s in src_tokens:
        common = 0
        for a, b in zip(value_token, s):
            if a != b:
                break
            common += 1
        if common >= 5:
            return True
    return False


def _supported(source: str, value: str, threshold: float = 0.6) -> bool:
    """True if the value is plausibly grounded in the source text.

    Rough heuristic: every non-trivial value token must be token-equal to a
    source token or share a >=5-char stem (catches Ukrainian inflection, e.g.
    підвал / підвалі). Genuine fabrications (ст. 125, 2023-04-05, міська рада)
    share no stem and fall through.
    """
    if not value.strip():
        return True
    src_tokens = _tokens(source)
    val_tokens = _tokens(value)
    if not src_tokens or not val_tokens:
        return False
    supported = sum(1 for t in val_tokens if _stem_supported(t, src_tokens))
    return supported / len(val_tokens) >= threshold


def hallucination_flags(row: dict) -> list[str]:
    """Field paths (topic index + key) whose value is not supported by the text."""
    flags: list[str] = []
    parsed = row.get("parsed")
    if not parsed or "topics" not in parsed:
        return flags
    src = row.get("text", "")
    for i, topic in enumerate(parsed["topics"]):
        if not isinstance(topic, dict):
            flags.append(f"topic[{i}]: not-an-object")
            continue
        for key in ("object", "requested_action"):
            val = topic.get(key, "")
            if val and not _supported(src, val):
                flags.append(f"topic[{i}].{key}")
        attrs = topic.get("attributes")
        if isinstance(attrs, dict):
            for v in attrs.values():
                if isinstance(v, str) and v and not _supported(src, v):
                    flags.append(f"topic[{i}].attributes:{v[:30]}")
    return flags


def _schema_ok(row: dict) -> bool:
    return bool(row.get("schema_valid"))


def _predicted_domains(row: dict) -> list[str]:
    parsed = row.get("parsed")
    if not parsed or "topics" not in parsed:
        return []
    return [str(t.get("domain", "")) for t in parsed["topics"] if isinstance(t, dict)]


def _fields(row: dict) -> tuple[bool, bool, bool]:
    parsed = row.get("parsed")
    if not parsed or "topics" not in parsed:
        return False, False, False
    topics = [t for t in parsed["topics"] if isinstance(t, dict)]
    issue = any(str(t.get("issue", "")).strip() for t in topics)
    obj = any(str(t.get("object", "")).strip() for t in topics)
    action = any(str(t.get("requested_action", "")).strip() for t in topics)
    return issue, obj, action


def _has_attributes(row: dict) -> bool:
    parsed = row.get("parsed")
    if not parsed or "topics" not in parsed:
        return False
    for t in parsed["topics"]:
        attrs = t.get("attributes") if isinstance(t, dict) else None
        if isinstance(attrs, dict) and any(str(v).strip() for v in attrs.values()):
            return True
    return False


def evaluate_case(row: dict) -> dict:
    cid = row["id"]
    expected = row["expected_domains"]
    pred_domains = _predicted_domains(row)

    # expected-domain coverage via the documented mapping
    covered = []
    uncovered = []
    for exp in expected:
        acceptable = EXPECTED_TO_SCHEMA.get(exp, {exp})
        if acceptable & set(pred_domains):
            covered.append(exp)
        else:
            uncovered.append(exp)

    accepted_all = {d for e in expected for d in EXPECTED_TO_SCHEMA.get(e, {e})}
    extra = sorted(set(pred_domains) - accepted_all)

    issue, obj, action = _fields(row)
    has_attrs = _has_attributes(row)
    hallu = hallucination_flags(row)

    is_multi = cid in MULTI_TOPIC_CASES
    multi_ok = (not is_multi) or len(set(pred_domains)) >= 2

    schema_ok = _schema_ok(row)
    coverage_ok = len(uncovered) == 0
    passed = bool(schema_ok and coverage_ok and not hallu and multi_ok)

    # qualitative notes for edge categories (informational, not pass/fail)
    notes: list[str] = []
    cat = row["category"]
    if cat == SARCASTIC_CATEGORY:
        notes.append("sarcasm input")
    elif cat == SURZHYK_CATEGORY:
        notes.append("surzhyk input")
    elif cat == IMPLICIT_ACTION_CATEGORY:
        notes.append("implicit-action input")
    elif cat == VAGUE_LOCATION_CATEGORY:
        notes.append("vague-location input")
    elif cat == PRIVATE_CATEGORY:
        notes.append("out-of-scope/private input")

    return {
        "id": cid,
        "category": cat,
        "expected": expected,
        "pred_domains": pred_domains,
        "uncovered": uncovered,
        "extra": extra,
        "issue_present": issue,
        "object_present": obj,
        "action_present": action,
        "has_attributes": has_attrs,
        "hallucinations": hallu,
        "multi_ok": multi_ok,
        "schema_ok": schema_ok,
        "passed": passed,
        "notes": notes,
    }


def evaluate_all() -> dict[str, dict]:
    cases = {c["id"]: c for c in load_smoke_cases()}
    out = {}
    for tag in ("finetuned", "base"):
        rows = load_results(tag)
        keyed = {r["id"]: r for r in rows}
        out[tag] = {cid: evaluate_case(keyed[cid]) for cid in cases}
    return out


def _normalize(raw: str) -> str:
    """Concise normalized form used in the report table."""
    return raw.replace("\n", " ").strip()


def summarize(tag: str, evaluated: dict[str, dict]) -> dict:
    n = len(evaluated)
    passed = sum(1 for e in evaluated.values() if e["passed"])
    schema = sum(1 for e in evaluated.values() if e["schema_ok"])
    coverage = sum(1 for e in evaluated.values() if e["passed"] or not e["uncovered"])
    hallu = sum(1 for e in evaluated.values() if e["hallucinations"])
    multi_expected = sum(1 for e in evaluated.values()
                         if e["id"] in MULTI_TOPIC_CASES)
    multi_passed = sum(1 for e in evaluated.values()
                       if e["id"] in MULTI_TOPIC_CASES and e["multi_ok"])
    return {
        "system": tag,
        "n": n,
        "passed": passed,
        "schema_valid": schema,
        "full_coverage": coverage,
        "cases_with_hallucination": hallu,
        "multi_topic_expected": multi_expected,
        "multi_topic_passed": multi_passed,
    }


def render_report(evaluated: dict[str, dict]) -> str:
    lines: list[str] = []
    lines.append("# Smoke-test report: fine-tuned vs base model (targeted suite)")
    lines.append("")
    lines.append("**Scope:** 20 hand-written Ukrainian municipal complaints, run at "
                 "temperature 0.0, `max_tokens=800`, same system prompt and Qwen3 chat "
                 "template (`enable_thinking=False`) as training.")
    lines.append("")
    lines.append("**Warning:** this is a qualitative / targeted smoke suite (N=20, "
                 "hand-authored, no platform gold labels). It is NOT a statistically "
                 "rigorous benchmark. It is meant to surface behavioural differences "
                 "and regressions for human inspection.")
    lines.append("")
    lines.append("The smoke cases use a *richer* free-form domain vocabulary "
                 "(`lighting`, `water_leak`, ...) than the model's 13 schema domains. "
                 "Comparison maps each expected domain to acceptable schema domains "
                 "via `EXPECTED_TO_SCHEMA` in `ml/tune/eval_smoke.py`. A case PASSes "
                 "if schema-valid AND full expected-domain coverage AND no "
                 "hallucination flags AND (for multi-topic cases) >=2 distinct topics.")
    lines.append("")
    lines.append("## Aggregate")
    lines.append("")
    lines.append("| system | n | passed | schema-valid | full coverage | with hallucination | multi-topic (exp) | multi-topic (ok) |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for tag in ("finetuned", "base"):
        s = summarize(tag, evaluated[tag])
        lines.append(f"| {tag} | {s['n']} | {s['passed']} | {s['schema_valid']} | "
                     f"{s['full_coverage']} | {s['cases_with_hallucination']} | "
                     f"{s['multi_topic_expected']} | {s['multi_topic_passed']} |")
    lines.append("")

    lines.append("## Failure patterns")
    lines.append("")
    for tag in ("finetuned", "base"):
        evs = evaluated[tag]
        lines.append(f"### {tag}")
        multi_fail = [e["id"] for e in evs.values()
                      if e["id"] in MULTI_TOPIC_CASES and not e["multi_ok"]]
        hallu_ids = [e["id"] for e in evs.values() if e["hallucinations"]]
        lines.append(f"- **Multi-topic collapse:** {len(multi_fail)}/{len(MULTI_TOPIC_CASES)} "
                     f"multi-topic cases emitted only one topic "
                     f"({', '.join(sorted(multi_fail)) or 'none'}).")
        lines.append(f"- **Hallucination flags:** {len(hallu_ids)} cases "
                     f"({', '.join(hallu_ids) or 'none'}).")
        for e in evs.values():
            if e["hallucinations"]:
                lines.append(f"  - {e['id']}: {'; '.join(e['hallucinations'])}")
        lines.append("")
    lines.append("")

    lines.append("## Per-case detail (fine-tuned vs base)")
    lines.append("")
    lines.append("| id | category | expected | system | domains | object | action | attributes | flags |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    cases = {c["id"]: c for c in load_smoke_cases()}
    for cid, case in cases.items():
        for tag in ("finetuned", "base"):
            ev = evaluated[tag][cid]
            rows = load_results(tag)
            row = next(r for r in rows if r["id"] == cid)
            parsed = row.get("parsed") or {}
            topics = parsed.get("topics", [])
            t0 = topics[0] if topics and isinstance(topics[0], dict) else {}
            obj = str(t0.get("object", ""))[:45] or "∅"
            act = str(t0.get("requested_action", ""))[:30] or "∅"
            attrs = json.dumps(t0.get("attributes", {}), ensure_ascii=False)[:30] or "∅"
            flags = []
            if not ev["schema_ok"]:
                flags.append("schema")
            if ev["uncovered"]:
                flags.append("uncovered:" + ",".join(ev["uncovered"]))
            if ev["extra"]:
                flags.append("extra:" + ",".join(ev["extra"]))
            if ev["hallucinations"]:
                flags.append("H:" + ";".join(ev["hallucinations"]))
            if cid in MULTI_TOPIC_CASES and not ev["multi_ok"]:
                flags.append("multi-fail")
            flag_str = ", ".join(flags) or "–"
            cat = cid if cid == "smoke-01" else ""
            lines.append(f"| {cid} | {case['category'].split(' - ')[-1]} | "
                         f"{','.join(ev['expected'])} | {tag} | "
                         f"{','.join(ev['pred_domains']) or '∅'} | {obj} | {act} | "
                         f"{attrs} | {flag_str} |")
    lines.append("")

    lines.append("## Qualitative notes")
    lines.append("")
    for tag in ("finetuned", "base"):
        lines.append(f"### {tag}")
        lines.append("")
        rows = load_results(tag)
        for row in rows:
            ev = evaluated[tag][row["id"]]
            raw_norm = _normalize(row.get("raw", ""))
            lines.append(f"**{row['id']}** [{row['category']}] expected="
                         f"{','.join(row['expected_domains'])}")
            lines.append(f"- schema_valid={row.get('schema_valid')} "
                         f"notes={ev['notes'] or '–'}")
            lines.append(f"- raw: `{raw_norm[:400]}`")
            lines.append("")
    lines.append("---")
    lines.append("_Generated by `ml/tune/eval_smoke.py`. Raw outputs live in "
                 "`ml/data/tune/smoke/results/{finetuned,base}/smoke_results.jsonl`._")
    return "\n".join(lines)


def main() -> None:
    evaluated = evaluate_all()
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(render_report(evaluated))
    for tag in ("finetuned", "base"):
        s = summarize(tag, evaluated[tag])
        print(f"{tag}: passed {s['passed']}/{s['n']} | schema {s['schema_valid']} | "
              f"coverage {s['full_coverage']} | hallu {s['cases_with_hallucination']} | "
              f"multi {s['multi_topic_passed']}/{s['multi_topic_expected']}")
    print(f"report -> {REPORT_PATH}")


if __name__ == "__main__":
    main()