"""Shared evaluation metrics for the MLX fine-tune experiment.

Compares three systems on the held-out test split:
  1. Deterministic baseline  (the weak-label transform itself -- an upper bound on
     *this* weak target, honest reference point, not a ceiling on real quality)
  2. Base model              (Qwen3-8B-4bit, no adapter)
  3. Fine-tuned model        (Qwen3-8B-4bit + LoRA adapter, fused)

All inputs in a run are recorded (input content, raw generation, parsed JSON)
so results are fully auditable.

Metrics (test set):
  - JSON parse rate
  - schema-validity rate (jsonschema Draft7 over annotation_schema.json)
  - domain accuracy and macro-F1 vs the weak target
  - issue ROUGE-L overlap vs the cleaned source
  - object exact-match rate and token-overlap vs the derived target
  - requested-action presence match (empty vs non-empty)
  - hallucination rate: non-empty fields the source text does not support
  - multi-topic rate (>=2 topics)
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import jsonschema

from ml.tune.build_dataset import SCHEMA_PATH, label_record


@dataclass
class Prediction:
    idx: int
    uid: str
    raw: str
    parsed: dict | None = None
    schema_error: str = ""
    topics: list[dict] = field(default_factory=list)


def _rouge_l(a: str, b: str) -> float:
    """ROUGE-L F1 over words (quick, dependency-free LCS)."""
    wa, wb = a.split(), b.split()
    if not wa or not wb:
        return 1.0 if wa == wb else 0.0
    n, m = len(wa), len(wb)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if wa[i - 1] == wb[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    lcs = dp[n][m]
    prec = lcs / n
    rec = lcs / m
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def _token_overlap(a: str, b: str) -> float:
    if not a or not b:
        return 1.0 if a == b else 0.0
    sa = set(a.lower().split())
    sb = set(b.lower().split())
    return len(sa & sb) / max(len(sa), 1)


def parse_payload(text: str) -> Prediction:
    """Best-effort extraction of the JSON payload from a generation."""
    pred = Prediction(idx=-1, uid="", raw=text)
    candidate = text.strip()
    m = re.search(r"\{.*\}", candidate, re.DOTALL)
    if m:
        candidate = m.group(0)
    try:
        parsed = json.loads(candidate)
        if isinstance(parsed, dict) and isinstance(parsed.get("topics"), list):
            pred.parsed = parsed
            pred.topics = parsed["topics"]
            return pred
        pred.parsed = parsed if isinstance(parsed, dict) else None
        return pred
    except (json.JSONDecodeError, ValueError):
        return pred


def validate_schema(pred: Prediction, validator) -> tuple[bool, str]:
    if pred.parsed is None:
        return False, "unparseable"
    errs = list(validator.iter_errors(pred.parsed))
    if errs:
        return False, errs[0].message
    return True, ""


def first_topic_domain(pred: Prediction) -> str:
    if not pred.topics:
        return "NO_TOPIC"
    return str(pred.topics[0].get("domain", ""))


def eval_predictions(preds: list[Prediction], targets: list[dict], validator) -> dict:
    """Compute all metrics. ``targets`` are the derived weak-label dicts per idx."""
    by_idx = {t.get("idx", i): t for i, t in enumerate(targets)}
    n = len(preds)

    json_rate = sum(1 for p in preds if p.parsed is not None) / n
    valid = [p for p in preds if not validate_schema(p, validator)[1]]
    schema_rate = len(valid) / n

    domains_true, domains_pred = [], []
    rl_issues, obj_exact, obj_overlap, obj_disc = [], [], [], []
    ra_present_t, ra_present_p, ra_match = [], [], []
    hallu = 0
    multi = 0
    for p in preds:
        target_topics = (by_idx.get(p.idx) or {}).get("topics", [])
        t = target_topics[0] if target_topics else {}
        p_domain = first_topic_domain(p)
        domains_true.append(t.get("domain", "other"))
        domains_pred.append(p_domain)

        t_issue = t.get("issue", "")
        p_issue = p.topics[0].get("issue", "") if p.topics else ""
        rl_issues.append(_rouge_l(t_issue, p_issue))

        t_obj, p_obj = t.get("object", ""), (p.topics[0].get("object", "") if p.topics else "")
        obj_exact.append(1.0 if t_obj.lower() == p_obj.lower() else 0.0)
        obj_overlap.append(_token_overlap(t_obj, p_obj))
        obj_disc.append(1.0 if (t_obj and not p_obj) or (not t_obj and p_obj) else 0.0)

        t_ra, p_ra = t.get("requested_action", ""), (p.topics[0].get("requested_action", "") if p.topics else "")
        ra_present_t.append(1.0 if t_ra else 0.0)
        ra_present_p.append(1.0 if p_ra else 0.0)
        ra_match.append(1.0 if ((t_ra != "") == (p_ra != "")) else 0.0)

        if len(p.topics if p.topics else []) >= 2:
            multi += 1
        if _hallucinated(p, t):
            hallu += 1

    return {
        "n": n,
        "json_parse_rate": round(json_rate, 4),
        "schema_validity_rate": round(schema_rate, 4),
        "domain_accuracy": round(sum(1 for a, b in zip(domains_true, domains_pred) if a == b) / n, 4),
        "domain_macro_f1": round(macro_f1(domains_true, domains_pred), 4),
        "issue_rouge_l": round(mean(rl_issues, 0), 4),
        "object_exact": round(mean(obj_exact, 0), 4),
        "object_token_overlap": round(mean(obj_overlap, 0), 4),
        "object_mismatch_discriminator": round(mean(obj_disc, 0), 4),
        "action_presence_target": round(mean(ra_present_t, 0), 4),
        "action_presence_pred": round(mean(ra_present_p, 0), 4),
        "action_presence_match": round(mean(ra_match, 0), 4),
        "hallucination_rate": round(hallu / n, 4),
        "multi_topic_rate": round(multi / n, 4),
    }


def _hallucinated(p: Prediction, target: dict) -> bool:
    """True if model emitted a non-empty field where the source-derived target is empty,
    or an issue that shares no tokens with the cleaned source."""
    if not p.topics:
        return False
    top = p.topics[0]
    t = target
    for key in ("object", "requested_action"):
        tv, pv = t.get(key, ""), top.get(key, "")
        if tv.strip() == "" and pv.strip() != "":
            return True
    src = t.get("source_text", "")
    p_issue = top.get("issue", "")
    if src and p_issue and _token_overlap(src, p_issue) < 0.01:
        return True
    return False


def mean(vals: list, default: float = 0.0) -> float:
    return sum(vals) / len(vals) if vals else default


def macro_f1(y_true: list[str], y_pred: list[str]) -> float:
    classes = set(y_true)
    f1s = []
    for c in classes:
        tp = sum(1 for a, b in zip(y_true, y_pred) if a == c and b == c)
        fp = sum(1 for a, b in zip(y_true, y_pred) if a != c and b == c)
        fn = sum(1 for a, b in zip(y_true, y_pred) if a == c and b != c)
        denom = 2 * tp + fp + fn
        f1s.append((2 * tp) / denom if denom else 0.0)
    return sum(f1s) / len(f1s) if f1s else math.nan


def load_validator() -> jsonschema.Draft7Validator:
    schema = json.loads(SCHEMA_PATH.read_text())
    return jsonschema.Draft7Validator(schema)


def export_run(path: Path, name: str, preds: list[Prediction], metrics: dict) -> None:
    payload = {
        "system": name,
        "metrics": metrics,
        "predictions": [
            {
                "idx": x.idx,
                "uid": x.uid,
                "raw": x.raw,
                "parsed": x.parsed,
                "schema_error": x.schema_error,
            }
            for x in preds
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))