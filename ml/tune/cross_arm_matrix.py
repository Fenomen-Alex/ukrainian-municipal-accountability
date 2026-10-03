"""Cross-arm matrix over the four arms, with the evidence behind every cell.

For each dimension the module reports the value per arm, the documented gate or
pre-registered target where one exists, the change against public v2 and
against corrected-v3, and whether the change is statistically supported wherever
the underlying suite stores per-case results.

Nothing here re-implements an evaluator. ``eval_v3`` and ``multitopic`` already
persist per-case/per-example records and are read directly. The frozen 329 suite
persists only predictions, so its per-case values are obtained by calling the
evaluator's own :func:`ml.tune.evaluate.eval_predictions` on single-element
lists; the per-case decomposition is accepted only if averaging it reproduces
the recorded aggregate exactly, and the assertion is part of the output.

Paired significance uses an exact McNemar test for binary per-case outcomes and a
paired bootstrap over cases for continuous ones. Both arms answer the *same*
items, so a paired test is the correct choice and an unpaired one would
overstate significance. Where a suite offers no per-case records the cell says so
instead of implying support it does not have.

Usage::

    .venv/bin/python -m ml.tune.cross_arm_matrix
    .venv/bin/python -m ml.tune.cross_arm_matrix --json-out FILE
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TUNE = ROOT / "ml/data/tune"

ARMS = ["v2", "v3-treatment", "v3-corrected", "v3-2"]
LABEL = {"v2": "v2 (public)", "v3-treatment": "previous-v3",
         "v3-corrected": "corrected-v3", "v3-2": "v3.2"}
REF = "v2"
CMP = "v3-corrected"

BEHAVIOUR = {"v3-treatment": "v3-prev-recheck.json",
             "v3-corrected": "v3-corrected-recheck.json",
             "v3-2": "v3-2.json"}
MT_FILE = {"v2": "lora.json"}
FROZEN_FILE = {"v2": "lora.json"}
SMOKE_DIR = {"v2": "finetuned-v2"}

#: Anything whose absolute change is at or below this is reported as unchanged
#: rather than as a movement. Rates are rounded to 4 dp by the evaluators, so
#: this is "did not move by even one evaluable case".
TOL_RATE = 0.005
TOL_COUNT = 0.5


def _read(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def _lines(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def ev3(arm: str) -> dict | None:
    p = TUNE / "eval_v3/results" / f"{arm}.json"
    return _read(p) if p.exists() else None


def multitopic(arm: str) -> dict | None:
    p = TUNE / "multitopic/results" / MT_FILE.get(arm, f"{arm}.json")
    return _read(p) if p.exists() else None


def frozen(arm: str) -> dict | None:
    p = TUNE / "eval" / FROZEN_FILE.get(arm, f"{arm}.json")
    return _read(p) if p.exists() else None


def behaviour(arm: str) -> dict | None:
    p = TUNE / "eval_v3/behaviour" / BEHAVIOUR.get(arm, f"{arm}.json")
    return _read(p) if p.exists() else None


def smoke(arm: str) -> list[dict] | None:
    for sub in (arm, SMOKE_DIR.get(arm, "")):
        if not sub:
            continue
        p = TUNE / "smoke/results" / sub / "smoke_results.jsonl"
        if p.exists():
            return _lines(p)
    return None


# ------------------------------------------------------------------ statistics

def mcnemar(a: list[bool], b: list[bool]) -> dict:
    """Exact two-sided McNemar on the discordant pairs."""
    b_only = sum(1 for x, y in zip(a, b) if not x and y)   # A misses, B hits
    a_only = sum(1 for x, y in zip(a, b) if x and not y)
    n = len(a)
    if a_only + b_only == 0:
        p = 1.0
    else:
        k = min(a_only, b_only)
        m = a_only + b_only
        p = min(1.0, 2 * sum(math.comb(m, i) for i in range(k + 1)) / 2 ** m)
    return {"test": "exact McNemar", "n": n,
            "A_worse": a_only, "B_worse": b_only,
            "delta": round((sum(b) - sum(a)) / n, 4) if n else 0.0,
            "p": round(p, 4), "supported": p < 0.05}


def bootstrap(pa: list[float], pb: list[float], iters: int = 4000,
              seed: int = 20260902) -> dict:
    """Paired bootstrap over cases for a continuous per-case score."""
    n = len(pa)
    if not n:
        return {"test": "paired bootstrap", "n": 0, "delta": 0.0,
                "ci95": [0.0, 0.0], "supported": False}
    rng = random.Random(seed)
    d = [y - x for x, y in zip(pa, pb)]
    obs = sum(d) / n
    boots = sorted(sum(d[rng.randrange(n)] for _ in range(n)) / n for _ in range(iters))
    lo, hi = boots[int(0.025 * iters)], boots[min(iters - 1, int(0.975 * iters))]
    return {"test": "paired bootstrap", "n": n, "delta": round(obs, 4),
            "ci95": [round(lo, 4), round(hi, 4)],
            "supported": bool(lo > 0 or hi < 0)}


# --------------------------------------------------- frozen per-case derivation

_FROZEN_KEY = {
    "domain_accuracy": "domain_accuracy",
    "issue_rouge_l": "issue_rouge_l",
    "object_exact": "object_exact",
    "hallucination_rate": "hallucination_rate",
    "multi_topic_rate": "multi_topic_rate",
    "schema_validity_rate": "schema_validity_rate",
    "action_presence_match": "action_presence_match",
}


def frozen_per_case(arm: str) -> dict:
    """Per-case frozen-329 values, plus a faithfulness check on the aggregates.

    Built by calling the evaluator on one prediction at a time so the metric
    definitions are the evaluator's own. ``faithful`` is False unless averaging
    these reproduces every mean-based recorded metric to the recorded rounding.
    """
    from ml.tune.evaluate import Prediction, eval_predictions, load_validator, parse_payload

    run, tgt_p = frozen(arm), TUNE / "eval" / f"{FROZEN_FILE.get(arm, f'{arm}.json')[:-5]}_targets.jsonl"
    if run is None or not tgt_p.exists():
        return {"per_case": {}, "faithful": False, "reason": "missing targets file"}
    targets = {t["idx"]: t for t in _lines(tgt_p)}
    validator = load_validator()
    per: dict[str, float] = {}
    for p in run["predictions"]:
        pr = parse_payload(p["raw"])
        pr.idx = p["idx"]
        pr.uid = p.get("uid", "")
        one = eval_predictions([pr], [targets[p["idx"]]], validator)
        for key in _FROZEN_KEY:
            per.setdefault(key, {})[str(p["idx"])] = one[key]
    faithful, bad = True, []
    for key, sub in per.items():
        want = run["metrics"].get(key)
        got = sum(sub.values()) / len(sub)
        # The evaluator rounds every aggregate to 4 dp, so compare at that
        # rounding. Comparing raw floats would flag a faithful decomposition as
        # broken over differences in the 6th decimal.
        if want is None or round(got, 4) != want:
            faithful = False
            bad.append(f"{key}: recorded {want} vs per-case mean {got:.6f}")
    return {"per_case": per, "faithful": faithful, "mismatch": bad,
            "reason": None if faithful else "; ".join(bad)}


_FROZEN_CACHE: dict[str, dict] = {}


def frozen_pc(arm: str) -> dict:
    if arm not in _FROZEN_CACHE:
        _FROZEN_CACHE[arm] = frozen_per_case(arm)
    return _FROZEN_CACHE[arm]


# ------------------------------------------------------------------- scorers

def _ev3_pc(arm: str, fn) -> dict | None:
    d = ev3(arm)
    return {c["id"]: fn(c) for c in d["per_case"]} if d else None


def _mt_pc(arm: str, fn) -> dict | None:
    d = multitopic(arm)
    return {e["uid"]: fn(e) for e in d["metrics"]["per_example"]} if d else None


def _fr_pc(arm: str, key: str) -> dict | None:
    f = frozen_pc(arm)
    return f["per_case"].get(key) if f["faithful"] else None


# ------------------------------------------------------------------ classify

def classify(delta: float | None, direction: str, tol: float,
             support: dict | None) -> str:
    """improved / regressed / unchanged / unresolved for one cell."""
    if delta is None:
        return "unresolved"
    if abs(delta) <= tol:
        return "unchanged"
    if direction not in ("higher", "lower"):
        return "unresolved"
    good = delta > 0 if direction == "higher" else delta < 0
    return "improved" if good else "regressed"


def _support(a: dict | None, b: dict | None, kind: str) -> dict | None:
    if not a or not b:
        return None
    keys = sorted(set(a) & set(b))
    if not keys:
        return None
    xa = [a[k] for k in keys]
    xb = [b[k] for k in keys]
    if kind == "binary":
        return mcnemar([bool(v) for v in xa], [bool(v) for v in xb])
    return bootstrap([float(v) for v in xa], [float(v) for v in xb])


NO_PERCASE = {"note": "suite stores aggregate only; no per-case records"}

# ------------------------------------------------------------- action_copy

_WORD = re.compile(r"[^\w]+", re.UNICODE)
COPY_MIN_CHARS = 8


def _norm(s: str) -> str:
    return _WORD.sub(" ", (s or "").lower()).strip()


def is_copied(action: str, text: str) -> bool:
    """A requested_action that is a verbatim span of the complaint text.

    Defined here, not by any evaluator: the suites measure redundancy (the same
    action repeated across topics) and presence, but not whether the action is
    lifted from the prompt rather than written. Reported as a diagnostic and
    kept separate from the gate metrics.
    """
    a, t = _norm(action), _norm(text)
    if len(a) < COPY_MIN_CHARS:
        return False
    return a in t


def _case_text() -> dict[str, str]:
    p = TUNE / "eval_v3/cases.jsonl"
    return {json.loads(l)["id"]: json.loads(l)["text"] for l in p.read_text(encoding="utf-8").splitlines() if l.strip()}


def action_copy_ev3(arm: str) -> float | None:
    d = ev3(arm)
    if not d:
        return None
    texts = _case_text()
    hit = sum(1 for c in d["per_case"]
              if any(is_copied(a, texts.get(c["id"], "")) for a in c["predicted_actions"]))
    return round(hit / len(d["per_case"]), 4)


def action_copy_ev3_pc(arm: str) -> dict | None:
    d = ev3(arm)
    if not d:
        return None
    texts = _case_text()
    return {c["id"]: any(is_copied(a, texts.get(c["id"], ""))
                         for a in c["predicted_actions"])
            for c in d["per_case"]}


def action_copy_frozen(arm: str) -> float | None:
    run, tgt = frozen(arm), TUNE / "eval" / f"{FROZEN_FILE.get(arm, f'{arm}.json')[:-5]}_targets.jsonl"
    if not run or not tgt.exists():
        return None
    src = {t["idx"]: t["topics"][0].get("source_text", "") for t in _lines(tgt)}
    hit = 0
    for p in run["predictions"]:
        topics = ((p.get("parsed") or {}).get("topics") or []) if isinstance(p.get("parsed"), dict) else []
        hit += 1 if any(is_copied(t.get("requested_action", ""), src.get(p["idx"], ""))
                        for t in topics) else 0
    return round(hit / len(run["predictions"]), 4)


def empty_action_invention(arm: str) -> float | None:
    d = ev3(arm)
    if not d:
        return None
    h = [c for c in d["per_case"] if c["category"] == "H"]
    return sum(1 for c in h if any((a or "").strip() for a in c["predicted_actions"]))


def smoke_two_topic(arm: str) -> float | None:
    rows = smoke(arm)
    if rows is None:
        return None
    mt = [r for r in rows if r.get("category") == "Multi-Topic"]
    return float(sum(1 for r in mt
                     if len((r.get("parsed") or {}).get("topics") or []) >= 2))


def smoke_two_topic_pc(arm: str) -> dict | None:
    rows = smoke(arm)
    if rows is None:
        return None
    return {str(i): len((r.get("parsed") or {}).get("topics") or []) >= 2
            for i, r in enumerate(rows) if r.get("category") == "Multi-Topic"}


def _beh(arm: str) -> dict | None:
    return behaviour(arm)


def bucket(arm: str, name: str) -> float | None:
    d = behaviour(arm)
    return float(d["summary"]["buckets"].get(name, 0)) if d else None


def bucket_pc(arm: str, name: str) -> dict | None:
    d = behaviour(arm)
    if not d:
        return None
    return {c["id"]: c["behaviour"] == name for c in d["cases"]}


def split_acc(arm: str) -> float | None:
    d = behaviour(arm)
    if not d:
        return None
    return d["summary"]["expected_two_topic_accuracy"]


def split_acc_pc(arm: str) -> dict | None:
    d = behaviour(arm)
    if not d:
        return None
    return {c["id"]: c["behaviour"] == "SPLIT" for c in d["cases"]}


def cat_e_split(arm: str) -> float | None:
    d = behaviour(arm)
    return float(d["summary"]["category_E_splits"]) if d else None


def cat_e_split_pc(arm: str) -> dict | None:
    d = behaviour(arm)
    if not d:
        return None
    return {c["id"]: (c["behaviour"] == "SPLIT" and c["category"] == "E")
            for c in d["cases"] if c["category"] == "E"}




# ---------------------------------------------------------------- dimensions
# direction: "higher" or "lower" is better. gate = documented target from
# gates_v3 (blocking/advisory) or the v3.2 pre-registration.

def _m(arm: str, path: str) -> float | None:
    """Fetch a metric from eval_v3 by dotted path."""
    d = ev3(arm)
    cur = d if d else None
    for part in path.split("."):
        if cur is None:
            return None
        cur = cur.get(part)
    return float(cur) if isinstance(cur, (int, float)) else None


def _mt(arm: str, key: str) -> float | None:
    d = multitopic(arm)
    v = (d or {}).get("metrics", {}).get(key)
    return float(v) if isinstance(v, (int, float)) else None


def _fr(arm: str, key: str) -> float | None:
    d = frozen(arm)
    v = (d or {}).get("metrics", {}).get(key)
    return float(v) if isinstance(v, (int, float)) else None


def _mtopic_count_error(arm: str) -> float | None:
    d = ev3(arm)
    if not d:
        return None
    pc = d["per_case"]
    return sum(abs(c["predicted_topic_count"] - c["expected_topic_count"])
               for c in pc) / len(pc)


def _mt_count(arm: str) -> float | None:
    d = multitopic(arm)
    if not d:
        return None
    pe = d["metrics"]["per_example"]
    return float(sum(1 for e in pe if e["pred_n"] == e["target_n"]))


DIMS: list[dict] = [
    # ---- eval_v3 core, each with a blocking gate
    dict(key="ev3_schema", label="schema validity", suite="eval_v3 (146)",
         gate="= 1.00", blocking=True, direction="higher", tol=TOL_RATE,
         val=lambda a: _m(a, "metrics.schema_validity_rate"),
         pc=lambda a: _ev3_pc(a, lambda c: c["schema_ok"]), kind="binary"),
    dict(key="ev3_boilerplate", label="boilerplate leak", suite="eval_v3 (146)",
         gate="<= 0.02", blocking=True, direction="lower", tol=TOL_RATE,
         val=lambda a: _m(a, "metrics.boilerplate_leak_rate"),
         pc=lambda a: _ev3_pc(a, lambda c: c["boilerplate_leak"]), kind="binary"),
    dict(key="ev3_topic_count", label="topic_count accuracy", suite="eval_v3 (146)",
         gate=">= 0.90", blocking=True, direction="higher", tol=TOL_RATE,
         val=lambda a: _m(a, "metrics.topic_count_accuracy"),
         pc=lambda a: _ev3_pc(a, lambda c: c["topic_count_ok"]), kind="binary"),
    # ---- expected-two behaviour, pre-registered for v3.2
    dict(key="two_topic_acc", label="expected-two-topic accuracy",
         suite="behaviour (52)", gate=">= 0.60 (v3.2 pre-reg)", blocking=False,
         direction="higher", tol=TOL_RATE, val=split_acc, pc=split_acc_pc, kind="binary"),
    dict(key="split", label="SPLIT", suite="behaviour (52)",
         gate="higher is better", blocking=False, direction="higher", tol=TOL_COUNT,
         val=lambda a: bucket(a, "SPLIT"), pc=lambda a: bucket_pc(a, "SPLIT"), kind="binary"),
    dict(key="merge", label="MERGE", suite="behaviour (52)",
         gate="<= 8 (v3.2 pre-reg)", blocking=False, direction="lower", tol=TOL_COUNT,
         val=lambda a: bucket(a, "MERGE"), pc=lambda a: bucket_pc(a, "MERGE"), kind="binary"),
    dict(key="stop", label="STOP", suite="behaviour (52)",
         gate="lower is better", blocking=False, direction="lower", tol=TOL_COUNT,
         val=lambda a: bucket(a, "STOP"), pc=lambda a: bucket_pc(a, "STOP"), kind="binary"),
    dict(key="jsonfail", label="JSONFAIL", suite="behaviour (52)",
         gate="0", blocking=False, direction="lower", tol=TOL_COUNT,
         val=lambda a: bucket(a, "JSONFAIL"), pc=lambda a: bucket_pc(a, "JSONFAIL"),
         kind="binary"),
    dict(key="cat_e", label="category E splits", suite="behaviour cat E (6)",
         gate=">= 5/6 (v3.2 pre-reg)", blocking=False, direction="higher", tol=TOL_COUNT,
         val=cat_e_split, pc=cat_e_split_pc, kind="binary"),
    # ---- multitopic suite
    dict(key="mt_topic_count", label="MT85 topic_count (of 85)", suite="multitopic (85)",
         gate=">= 80/85 (v3.2 pre-reg)", blocking=False, direction="higher", tol=TOL_COUNT,
         # counted from per_example rather than rate*85: the recorded rate is
         # rounded to 4 dp, so multiplying it back gives 72.998, not 73.
         val=lambda a: _mt_count(a),
         pc=lambda a: _mt_pc(a, lambda e: e["pred_n"] == e["target_n"]), kind="binary"),
    dict(key="mt_recall", label="MT85 multi_topic recall", suite="multitopic (85)",
         gate=">= 0.894", blocking=True, direction="higher", tol=TOL_RATE,
         val=lambda a: _mt(a, "multi_topic_recall"),
         pc=lambda a: _mt_pc(a, lambda e: e["pred_n"] >= 2), kind="binary"),
    dict(key="mt_domain_exact", label="MT85 domain-set exact", suite="multitopic (85)",
         gate="none", blocking=False, direction="higher", tol=TOL_RATE,
         val=lambda a: _mt(a, "domain_set_exact"),
         pc=lambda a: _mt_pc(a, lambda e: bool(e["domain_set_exact"])), kind="binary"),
    # ---- frozen 329
    dict(key="fr_multi_topic", label="frozen multi_topic_rate", suite="frozen 329",
         gate="none", blocking=False, direction="higher", tol=TOL_RATE,
         val=lambda a: _fr(a, "multi_topic_rate"),
         pc=lambda a: _fr_pc(a, "multi_topic_rate"), kind="binary"),
    dict(key="fr_domain_acc", label="frozen domain accuracy", suite="frozen 329",
         gate=">= 0.82", blocking=True, direction="higher", tol=TOL_RATE,
         val=lambda a: _fr(a, "domain_accuracy"),
         pc=lambda a: _fr_pc(a, "domain_accuracy"), kind="binary"),
    dict(key="fr_domain_f1", label="frozen domain macro F1", suite="frozen 329",
         gate="none", blocking=False, direction="higher", tol=TOL_RATE,
         val=lambda a: _fr(a, "domain_macro_f1"), kind="aggregate"),
    dict(key="fr_issue_rl", label="frozen issue ROUGE-L", suite="frozen 329",
         gate=">= 0.80", blocking=False, direction="higher", tol=TOL_RATE,
         val=lambda a: _fr(a, "issue_rouge_l"),
         pc=lambda a: _fr_pc(a, "issue_rouge_l"), kind="continuous"),
    dict(key="fr_object_exact", label="frozen object exact", suite="frozen 329",
         gate="none", blocking=False, direction="higher", tol=TOL_RATE,
         val=lambda a: _fr(a, "object_exact"),
         pc=lambda a: _fr_pc(a, "object_exact"), kind="continuous"),
    dict(key="fr_halluc", label="frozen hallucination rate", suite="frozen 329",
         gate="none", blocking=False, direction="lower", tol=TOL_RATE,
         val=lambda a: _fr(a, "hallucination_rate"),
         pc=lambda a: _fr_pc(a, "hallucination_rate"), kind="binary"),
    dict(key="fr_action_match", label="frozen action presence match", suite="frozen 329",
         gate="none", blocking=False, direction="higher", tol=TOL_RATE,
         val=lambda a: _fr(a, "action_presence_match"),
         pc=lambda a: _fr_pc(a, "action_presence_match"), kind="continuous"),
    # ---- action behaviour
    dict(key="ev3_action_redund", label="action redundancy", suite="eval_v3 (146)",
         gate="none", blocking=False, direction="lower", tol=TOL_RATE,
         val=lambda a: _m(a, "metrics.action_redundancy_rate"),
         pc=lambda a: _ev3_pc(a, lambda c: c["action_redundant"]), kind="binary"),
    dict(key="ev3_action_copy", label="action copy (verbatim from prompt)",
         suite="eval_v3 (146)", gate="none", blocking=False, direction="lower",
         tol=TOL_RATE, val=action_copy_ev3, pc=action_copy_ev3_pc, kind="binary",
         note="diagnostic defined in this module; not an evaluator metric"),
    dict(key="fr_action_copy", label="action copy (verbatim from prompt)",
         suite="frozen 329", gate="none", blocking=False, direction="lower",
         tol=TOL_RATE, val=action_copy_frozen, kind="aggregate",
         note="diagnostic defined in this module; not an evaluator metric"),
    dict(key="empty_action", label="empty-action invention (cat H)",
         suite="eval_v3 cat H (12)", gate="0", blocking=False, direction="lower",
         tol=TOL_COUNT, val=empty_action_invention, kind="aggregate"),
    # ---- terse / R1
    dict(key="terse_A", label="terse cat A domain_set_exact", suite="eval_v3 cat A (10)",
         gate=">= 0.70", blocking=False, direction="higher", tol=TOL_RATE,
         val=lambda a: _m(a, "by_category.A.domain_set_exact"),
         pc=lambda a: ({c["id"]: c["domain_set_exact"] == 1.0
                        for c in ev3(a)["per_case"] if c["category"] == "A"}
                       if ev3(a) else None), kind="binary"),
    dict(key="terse_abc", label="terse A+B+C domain_set_exact", suite="eval_v3 (26)",
         gate="none", blocking=False, direction="higher", tol=TOL_RATE,
         val=lambda a: _m(a, "subsets.terse_categories_ABC.domain_set_exact"),
         pc=lambda a: ({c["id"]: c["domain_set_exact"] == 1.0
                        for c in ev3(a)["per_case"] if c["category"] in ("A", "B", "C")}
                       if ev3(a) else None), kind="binary"),
    dict(key="terse_short", label="terse text<150 domain_set_exact", suite="eval_v3 (14)",
         gate="none", blocking=False, direction="higher", tol=TOL_RATE,
         val=lambda a: _m(a, "subsets.terse_text_under_150.domain_set_exact"), kind="aggregate"),
    # ---- smoke
    dict(key="smoke_mt", label="smoke Multi-Topic (of 4)", suite="smoke (20)",
         gate="4 of 4", blocking=True, direction="higher", tol=TOL_COUNT,
         val=smoke_two_topic, pc=smoke_two_topic_pc, kind="binary"),
    # ---- multi-topic slice schema
    dict(key="mt_schema", label="multi-topic slice schema validity",
         suite="eval_v3 multi-topic (44)", gate="none", blocking=False,
         direction="higher", tol=TOL_RATE,
         val=lambda a: _m(a, "subsets.multi_topic_categories.schema_valid"),
         pc=lambda a: ({c["id"]: c["schema_ok"] for c in ev3(a)["per_case"]
                        if c["expected_topic_count"] >= 2} if ev3(a) else None),
         kind="binary"),
    # ---- supporting
    dict(key="ev3_domain_exact", label="eval_v3 domain_set_exact", suite="eval_v3 (146)",
         gate="none", blocking=False, direction="higher", tol=TOL_RATE,
         val=lambda a: _m(a, "metrics.domain_set_exact"),
         pc=lambda a: _ev3_pc(a, lambda c: c["domain_set_exact"] == 1.0), kind="binary"),
    # Topic-count error is used instead of raw mean predicted topics: the mean
    # has no natural direction (high is not better), whereas the signed error
    # against the expected count is a proper lower-is-better score with
    # per-case support.
    dict(key="ev3_topic_count_error", label="eval_v3 topic-count error (abs)",
         suite="eval_v3 (146)", gate="none", blocking=False, direction="lower",
         tol=TOL_RATE, val=lambda a: _mtopic_count_error(a),
         pc=lambda a: _ev3_pc(a, lambda c: abs(c["predicted_topic_count"] - c["expected_topic_count"])),
         kind="continuous"),
    dict(key="ev3_out_of_enum", label="eval_v3 out-of-enum rate", suite="eval_v3 (146)",
         gate="none", blocking=False, direction="lower", tol=TOL_RATE,
         val=lambda a: _m(a, "metrics.out_of_enum_rate"),
         pc=lambda a: _ev3_pc(a, lambda c: c["out_of_enum_domain"]), kind="binary"),
]


def build() -> dict:
    """Assemble the matrix. Every cell is recomputed from committed artefacts."""
    rows = []
    for spec in DIMS:
        vals = {}
        for a in ARMS:
            try:
                v = spec["val"](a)
            except Exception:
                v = None
            vals[a] = round(v, 4) if isinstance(v, float) else v
        pcs = {}
        if spec["kind"] != "aggregate":
            for a in ARMS:
                try:
                    pcs[a] = spec["pc"](a)
                except Exception:
                    pcs[a] = None

        row = {"key": spec["key"], "label": spec["label"], "suite": spec["suite"],
               "gate": spec["gate"], "blocking": spec["blocking"],
               "direction": spec["direction"], "values": vals,
               "kind": spec["kind"], "note": spec.get("note")}

        for label, ref in (("vs_v2", REF), ("vs_corrected", CMP)):
            cells = {}
            for a in ARMS:
                sup = _support(pcs.get(ref), pcs.get(a), spec["kind"]) if pcs.get(ref) and pcs.get(a) else None
                delta = None
                if vals.get(a) is not None and vals.get(ref) is not None:
                    delta = round(vals[a] - vals[ref], 4)
                cells[a] = {
                    "delta": delta,
                    "class": ("baseline" if a == ref else
                              classify(delta, spec["direction"], spec["tol"], sup)),
                    "support": sup,
                }
            row[label] = cells
        rows.append(row)

    fidelity = {a: {"faithful": frozen_pc(a)["faithful"],
                    "mismatch": frozen_pc(a).get("mismatch")}
                for a in ARMS}
    return {"arms": ARMS, "label": LABEL, "ref": REF, "cmp": CMP,
            "rows": rows, "frozen_per_case_fidelity": fidelity}


def _fmt(v, kind: str) -> str:
    if v is None:
        return "n/a"
    if kind in ("binary", "continuous") and 0 <= v <= 1 and v not in (0.0, 1.0):
        return f"{v:.4f}"
    return f"{v:g}"


def to_markdown(mx: dict) -> str:
    L = ["# Cross-arm matrix\n",
         "Values recomputed from committed artefacts. `class` is against the named "
         "reference arm (`vs_v2`, `vs_corrected`); `supp` says whether a paired test "
         "supports the change. Cells without a per-case record say so.\n"]
    for label in ("vs_v2", "vs_corrected"):
        ref = mx["ref"] if label == "vs_v2" else mx["cmp"]
        L.append(f"\n## Change {label.replace('_', ' ')}  (reference: {mx['label'][ref]})\n")
        L.append("| dimension | suite | gate | " + " | ".join(
            mx["label"][a] for a in mx["arms"]) + " |")
        L.append("|---|---|---|" + "---|" * len(mx["arms"]))
        for row in mx["rows"]:
            vals = " | ".join(_fmt(row["values"].get(a), row["kind"]) for a in mx["arms"])
            cells = []
            for a in mx["arms"]:
                c = row[label][a]
                if c["class"] == "baseline":
                    cells.append("—")
                    continue
                s = c.get("support")
                tag = ""
                if s and "p" in s:
                    tag = " ✓" if s["supported"] else " (ns)"
                elif s and "ci95" in s:
                    tag = " ✓" if s["supported"] else " (ns)"
                d = c["delta"]
                cells.append(f"{d:+.4f} {c['class']}{tag}" if d is not None else "n/a")
            gate = row["gate"] + (" *(blocking)*" if row["blocking"] else "")
            L.append(f"| {row['label']} | {row['suite']} | {gate} | {vals} | " + " | ".join(cells) + " |")

    L.append("\n## Per-arm detail\n")
    for a in mx["arms"]:
        L.append(f"\n### {mx['label'][a]}\n")
        L.append("| dimension | value | Δ v2 | Δ corrected-v3 | class vs corrected-v3 | support |")
        L.append("|---|---|---|---|---|---|")
        for row in mx["rows"]:
            c = row["vs_corrected"][a]
            s = c.get("support") or {}
            if "p" in s:
                sup = f"McNemar p={s['p']} ({s['A_worse']}/{s['B_worse']} discordant)"
            elif "ci95" in s:
                sup = f"bootstrap CI {s['ci95'][0]:+.4f}..{s['ci95'][1]:+.4f}"
            else:
                sup = "no per-case records"
            d = c["delta"]
            L.append(f"| {row['label']} | {_fmt(row['values'].get(a), row['kind'])} | "
                     f"{'' if a == mx['ref'] else f'{row['vs_v2'][a]['delta']:+.4f}' if row['vs_v2'][a]['delta'] is not None else 'n/a'} | "
                     f"{'—' if a == mx['cmp'] else f'{d:+.4f}' if d is not None else 'n/a'} | "
                     f"{c['class']} | {sup} |")
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--json-out", type=Path)
    a = ap.parse_args()
    mx = build()
    payload = json.dumps(mx, ensure_ascii=False, indent=2)
    if a.json_out:
        a.json_out.write_text(payload, encoding="utf-8")
        print(f"wrote {a.json_out}")
    print(to_markdown(mx))


if __name__ == "__main__":
    main()
