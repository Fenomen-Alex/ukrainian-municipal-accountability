"""Score a model on the 146-case adversarial suite (``ml/data/tune/eval_v3``).

Why this module exists
----------------------
``ml/tune/build_eval_v3.py`` builds the suite and ``ml/tests/test_eval_v3_suite.py``
tests that it is deterministic and leak-free, but **nothing could score a model on
it**. ``run_eval.py`` (frozen 329) and ``run_eval_multitopic.py`` (85) both read
different corpora. So the suite that the v3 design names as its *primary* benchmark
had never been run against v2, and the baselines quoted in
``ml/data/tune/v3/plan.json`` (``topic_count_accuracy`` 0.824, ``boilerplate_leak``
0.659) are hand-authored planning figures with no measured provenance.

This module measures them. It is the only place eval_v3 numbers come from.

Comparability
-------------
eval_v3 scores against the *cleaned* reference label (``build_eval_v3`` closes the
``надає згоду`` gap that ``_ADMIN_CLAUSE`` misses). The frozen 329 and multitopic 85
score against the *uncleaned* v1 weak label. **The two are not comparable and must
not be pooled** -- a model that correctly strips boilerplate looks worse on the old
suites. See ``ml/reports/v3_experiment_design.md`` section 3.

Usage
-----
    .venv-mlx/bin/python -m ml.tune.run_eval_v3 --model <fused-dir> --tag v2
    .venv-mlx/bin/python -m ml.tune.run_eval_v3 --model <fused-dir> --tag v3-control

Results land in ``ml/data/tune/eval_v3/results/<tag>.json``.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

from ml.cleaner import normalize_text
from ml.tune.build_dataset import SYSTEM_PROMPT
from ml.tune.build_eval_v3 import BOILER_SENT, OUT_DIR as EVAL_V3_DIR
from ml.tune.evaluate import (
    Prediction,
    _rouge_l,
    load_validator,
    mean,
    parse_payload,
)
from ml.tune.serve_v2 import parse_strict

CASES_PATH = EVAL_V3_DIR / "cases.jsonl"
RESULTS_DIR = EVAL_V3_DIR / "results"

_BOILER_RE = re.compile("|".join(BOILER_SENT), re.IGNORECASE)
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")

SCHEMA_DOMAINS = {
    "roads", "water", "heating", "housing", "transport", "sanitation",
    "electricity", "construction", "benefits", "government", "commerce",
    "payments", "other",
}

#: Categories the gates and the error analysis single out.
TERSE_CATEGORIES = {"A", "B", "C"}
MULTI_CATEGORIES = {"B", "C", "D", "E", "F", "G"}
EMPTY_ACTION_CATEGORY = "H"


def _load_cases() -> list[dict]:
    return [json.loads(l) for l in CASES_PATH.read_text(encoding="utf-8").splitlines()]


def _boilerplate_leak(topics: list[dict]) -> bool:
    """True if any predicted ``issue`` carries response-delivery boilerplate."""
    for t in topics:
        if not isinstance(t, dict):
            continue
        for sent in _SENT_SPLIT.split(str(t.get("issue", ""))):
            if sent and _BOILER_RE.search(sent):
                return True
    return False


def _action_redundant(topics: list[dict]) -> bool:
    """True if a predicted ``requested_action`` is contained in its own ``issue``.

    The audit measured this at 98.1% of v2 predictions: the field carried no
    information the model could generalise from.
    """
    for t in topics:
        if not isinstance(t, dict):
            continue
        action = str(t.get("requested_action", "")).strip()
        issue = str(t.get("issue", "")).strip()
        if action and action in issue:
            return True
    return False


def _model_predictions(model_path: str, cases: list[dict], max_tokens: int,
                       temperature: float, batch_report: int = 25) -> list[Prediction]:
    """Run the model with the canonical v2 serving contract.

    Identical to ``ml/tune/V2_SERVING.md``: exact system prompt, thinking off,
    ``strip()`` then a *strict* JSON parse. ``parse_payload`` is recorded alongside
    so we can tell a genuine format failure from a case where the permissive
    historical regex salvaged a wrapper.
    """
    from mlx_lm import generate, load
    from mlx_lm.sample_utils import make_sampler

    from ml.tune.serve_v2 import build_prompt

    model, tokenizer = load(model_path)
    sampler = make_sampler(temp=temperature)

    preds: list[Prediction] = []
    t0 = time.time()
    for i, case in enumerate(cases):
        prompt = build_prompt(case["text"], tokenizer)
        raw = generate(model, tokenizer, prompt=prompt,
                       max_tokens=max_tokens, sampler=sampler, verbose=False)
        permissive = parse_payload(raw)
        payload, strict_err = parse_strict(raw)
        p = Prediction(
            idx=i,
            uid=case["id"],
            raw=raw,
            parsed=payload,
            topics=[t for t in (payload or {}).get("topics", []) if isinstance(t, dict)],
        )
        # Keep the comparison honest: note when the loose parser would have found
        # something the strict one rejected.
        if payload is None:
            p.schema_error = ("permissive_parser_recovered"
                              if permissive.parsed is not None else strict_err)
        preds.append(p)
        if batch_report and (i + 1) % batch_report == 0:
            el = time.time() - t0
            print(f"  {i + 1}/{len(cases)}  {el:.0f}s elapsed", flush=True)
    return preds


def _deterministic_predictions(cases: list[dict]) -> list[Prediction]:
    """The weak labeler applied to the eval_v3 *references*.

    Not a model. It exists so the suite has a floor: it says what a model that had
    learned nothing would score, which makes a model score interpretable.
    """
    preds = []
    for i, case in enumerate(cases):
        topics = [dict(t) for t in case["expected_topics"]]
        preds.append(Prediction(idx=i, uid=case["id"],
                                raw=json.dumps({"topics": topics}, ensure_ascii=False),
                                parsed={"topics": topics}, topics=topics))
    return preds


def score(preds: list[Prediction], cases: list[dict], validator) -> dict:
    """Suite-level, per-category, and subset metrics."""
    assert len(preds) == len(cases)

    json_ok, schema_ok = [], []
    count_ok, dset_exact, dprec, drec, rouge = [], [], [], [], []
    boiler_leak, action_redun = [], []
    out_of_enum, dup_domain = [], []
    n_pred_topics, n_exp_topics = [], []

    per_category: dict[str, list] = defaultdict(list)
    per_case = []

    for pred, case in zip(preds, cases):
        cat = case["category"]
        exp_topics = [t for t in case["expected_topics"] if isinstance(t, dict)]
        exp_domains = [t.get("domain", "") for t in exp_topics]
        topics = [t for t in pred.topics if isinstance(t, dict)]
        got_domains = [t.get("domain", "") for t in topics]

        j = pred.parsed is not None
        s = False
        if j:
            errs = sorted(validator.iter_errors(pred.parsed), key=lambda e: e.path)
            s = not errs

        c_ok = len(topics) == case["expected_topic_count"]
        ex, dp, dr = _domain_set_scores(set(got_domains), set(exp_domains))
        rl = _issue_rouge_best(exp_topics, topics)

        leak = _boilerplate_leak(topics)
        redun = _action_redundant(topics)
        ooe = [d for d in got_domains if d not in SCHEMA_DOMAINS]
        dup = len(got_domains) != len(set(got_domains))

        json_ok.append(j); schema_ok.append(s); count_ok.append(c_ok)
        dset_exact.append(ex); dprec.append(dp); drec.append(dr); rouge.append(rl)
        boiler_leak.append(leak); action_redun.append(redun)
        out_of_enum.append(bool(ooe)); dup_domain.append(dup)
        n_pred_topics.append(len(topics)); n_exp_topics.append(len(exp_topics))

        rec = {
            "id": case["id"], "category": cat, "category_name": case.get("category_name", ""),
            "provenance": case.get("provenance", {}).get("kind", ""),
            "text_len": len(case["text"]),
            "json_ok": j, "schema_ok": s,
            "expected_topic_count": case["expected_topic_count"],
            "predicted_topic_count": len(topics),
            "topic_count_ok": c_ok,
            "expected_domains": exp_domains, "predicted_domains": got_domains,
            "domain_set_exact": ex, "domain_set_precision": dp, "domain_set_recall": dr,
            "issue_rouge_l_best": rl,
            "boilerplate_leak": leak, "action_redundant": redun,
            "out_of_enum_domain": bool(ooe), "out_of_enum_values": ooe,
            "duplicate_domain": dup,
            "predicted_actions": [t.get("requested_action", "") for t in topics],
            "expected_actions": [t.get("requested_action", "") for t in exp_topics],
        }
        per_case.append(rec)
        per_category[cat].append(rec)

    def cat_summary(rows: list[dict]) -> dict:
        return {
            "n": len(rows),
            "schema_valid": round(mean([r["schema_ok"] for r in rows]), 4),
            "topic_count_accuracy": round(mean([r["topic_count_ok"] for r in rows]), 4),
            "domain_set_exact": round(mean([r["domain_set_exact"] for r in rows]), 4),
            "domain_set_precision": round(mean([r["domain_set_precision"] for r in rows]), 4),
            "domain_set_recall": round(mean([r["domain_set_recall"] for r in rows]), 4),
            "issue_rouge_l_best": round(mean([r["issue_rouge_l_best"] for r in rows]), 4),
            "boilerplate_leak_rate": round(mean([r["boilerplate_leak"] for r in rows]), 4),
            "action_redundancy_rate": round(mean([r["action_redundant"] for r in rows]), 4),
            "out_of_enum_rate": round(mean([r["out_of_enum_domain"] for r in rows]), 4),
            "duplicate_domain_rate": round(mean([r["duplicate_domain"] for r in rows]), 4),
        }

    by_category = {c: cat_summary(rows) for c, rows in sorted(per_category.items())}

    # --- subsets the gates name explicitly ---------------------------------- #
    def sub(rows: list[dict]) -> dict:
        return cat_summary(rows) if rows else {"n": 0}

    terse_rows = [r for r in per_case if r["category"] in TERSE_CATEGORIES]
    terse_lt150 = [r for r in per_case if r["text_len"] < 150]
    multi_rows = [r for r in per_case if r["category"] in MULTI_CATEGORIES]
    h_rows = [r for r in per_case if r["category"] == EMPTY_ACTION_CATEGORY]
    h_nonempty = [r for r in h_rows
                  if any(str(a).strip() for a in r["predicted_actions"])]

    metrics = {
        "n": len(per_case),
        "json_parse_rate": round(mean(json_ok), 4),
        "schema_validity_rate": round(mean(schema_ok), 4),
        "topic_count_accuracy": round(mean(count_ok), 4),
        "domain_set_exact": round(mean(dset_exact), 4),
        "domain_set_precision": round(mean(dprec), 4),
        "domain_set_recall": round(mean(drec), 4),
        "issue_rouge_l_best": round(mean(rouge), 4),
        "boilerplate_leak_rate": round(mean(boiler_leak), 4),
        "action_redundancy_rate": round(mean(action_redun), 4),
        "out_of_enum_rate": round(mean(out_of_enum), 4),
        "duplicate_domain_rate": round(mean(dup_domain), 4),
        "mean_predicted_topics": round(mean(n_pred_topics), 4),
        "mean_expected_topics": round(mean(n_exp_topics), 4),
    }
    subsets = {
        "terse_categories_ABC": sub(terse_rows),
        "terse_text_under_150": sub(terse_lt150),
        "multi_topic_categories": sub(multi_rows),
        "empty_action_category_H": {
            **sub(h_rows),
            "cases_with_nonempty_predicted_action": len(h_nonempty),
            "share_nonempty_predicted_action": round(len(h_nonempty) / len(h_rows), 4) if h_rows else None,
        },
    }
    return {"metrics": metrics, "by_category": by_category, "subsets": subsets,
            "per_case": per_case}


def _domain_set_scores(got: set, exp: set) -> tuple[float, float, float]:
    if not exp:
        return (1.0, 1.0, 1.0) if not got else (0.0, 0.0, 0.0)
    inter = len(got & exp)
    exact = 1.0 if got == exp else 0.0
    prec = inter / len(got) if got else 0.0
    rec = inter / len(exp)
    return exact, prec, rec


def _issue_rouge_best(exp_topics: list[dict], got_topics: list[dict]) -> float:
    """Best-match ROUGE-L, so a correct split is not punished for ordering."""
    if not exp_topics:
        return 1.0 if not got_topics else 0.0
    if not got_topics:
        return 0.0
    scores = []
    for et in exp_topics:
        scores.append(max(
            (_rouge_l(str(et.get("issue", "")), str(gt.get("issue", "")))
             for gt in got_topics), default=0.0))
    return sum(scores) / len(scores)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, help="fused model directory")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--max-tokens", type=int, default=800)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--deterministic", action="store_true",
                    help="score the reference labeler instead of a model (floor)")
    ap.add_argument("--only", nargs="*", default=None, help="restrict to case ids")
    args = ap.parse_args()

    cases = _load_cases()
    if args.only:
        cases = [c for c in cases if c["id"] in set(args.only)]
    validator = load_validator()

    if args.deterministic:
        preds = _deterministic_predictions(cases)
    else:
        print(f"scoring {args.model} on {len(cases)} eval_v3 cases (tag={args.tag})")
        t0 = time.time()
        preds = _model_predictions(args.model, cases, args.max_tokens, args.temperature)
        print(f"  inference done in {time.time() - t0:.0f}s")

    result = score(preds, cases, validator)
    result["model"] = args.model
    result["tag"] = args.tag
    result["generation"] = {
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "enable_thinking": False,
        "system_prompt_sha256": "db983edd70b4ba2005cb2e0428f88eec76725e741cdaff7e8d23205ecc2905eb",
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"{args.tag}.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    m = result["metrics"]
    print(f"\n=== eval_v3 [{args.tag}] ===")
    for k in ("n", "json_parse_rate", "schema_validity_rate", "topic_count_accuracy",
              "domain_set_exact", "domain_set_precision", "domain_set_recall",
              "issue_rouge_l_best", "boilerplate_leak_rate", "action_redundancy_rate",
              "out_of_enum_rate", "duplicate_domain_rate"):
        print(f"  {k:26s} {m[k]}")
    h = result["subsets"]["empty_action_category_H"]
    print(f"  H non-empty action          {h['cases_with_nonempty_predicted_action']}/{h['n']}")
    print(f"  terse(<150) topic-count acc {result['subsets']['terse_text_under_150'].get('topic_count_accuracy')}")
    print(f"  terse(<150) domain-set exct {result['subsets']['terse_text_under_150'].get('domain_set_exact')}")
    print(f"\nreport -> {out}")


if __name__ == "__main__":
    main()
