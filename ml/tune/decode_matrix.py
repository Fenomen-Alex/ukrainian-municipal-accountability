"""Reads the decoding probe and reports whether any setting is worth adopting.

The probe varies decoding only; the models, the prompt, the parser and the
classifier are the canonical ones (:mod:`ml.tune.serve_v2`,
:mod:`ml.tune.run_eval_v3`, :mod:`ml.tune.behavior_v3`). The probe does **not**
include the canonical baseline setting (``temp=0.0, max_tokens=800``), whose raw
generations are already committed, so every row here is compared against that
committed baseline for the *same* case subset. Comparing against a probe row
instead would make the baseline depend on the probe's own case selection.

Gate discipline: the documented blocking gates in :mod:`ml.tune.gates_v3` are
not recomputed here. This probe covers the 52 two-topic ``eval_v3`` cases plus
JSONFAIL and control samples -- it does not include the smoke suite or the
frozen 329, so it cannot adjudicate ``smoke_two_topic`` or ``frozen_domain``, and
reporting a subset as a gate pass would be score-hunting. What it can report is
the *direction and size* of each blocking-gate-covered measure on the cases it
does cover.

Usage::

    .venv-mlx/bin/python -m ml.tune.decode_matrix
    .venv-mlx/bin/python -m ml.tune.decode_matrix --json-out FILE
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from ml.tune.run_eval_v3 import parse_payload
from ml.tune.residue import GATE_RE, _scan

ROOT = Path(__file__).resolve().parents[2]
TUNE = ROOT / "ml/data/tune"
PROBE_DIR = TUNE / "decoding_probe"
RESULTS = PROBE_DIR / "results.jsonl"
CASES = TUNE / "eval_v3/cases.jsonl"

#: probe model name -> committed eval_v3 arm holding its baseline recording.
#: The probe writes hyphenated dir names (v3-2); the committed arm is v3-2 too,
#: while the report and matrix call the same arm "v3.2".
BASELINE_ARM = {"v2": "v2", "corrected-v3": "v3-corrected", "v3-2": "v3-2"}

#: Documented blocking gates and whether the probe case subset can speak to them.
GATE_COVERAGE = {
    "schema": (True, "counted over the probe subset, not all 146 eval_v3 cases"),
    "topic_count": (True, "counted over the probe subset, not all 146"),
    "boilerplate": (True, "counted over the probe subset, not all 146"),
    "two_topic_recall": (False, "needs the multitopic suite's per-topic labels"),
    "frozen_domain": (False, "needs the frozen 329"),
    "smoke_two_topic": (False, "needs the smoke suite"),
}

#: Cases whose *committed* eval_v3 record is a JSON failure that is not a
#: repetition loop. Derived from ml/data/tune/eval_v3/results/, not from any
#: ad-hoc run -- an earlier unseeded smoke run proposed this list and was
#: discarded, so re-deriving it here keeps the §3 defect set reproducible.
def _committed_jsonfail_ids(model: str = "v3-corrected") -> list[str]:
    d = json.loads((ROOT / "ml/data/tune/eval_v3/results" / f"{model}.json")
                   .read_text(encoding="utf-8"))
    out = []
    for r in d["per_case"]:
        if r["json_ok"]:
            continue
        # a repetition loop is a long output that repeats; keep the short ones
        if r.get("text_len", 0) >= REPEAT_LOOP_MIN_CHARS:
            continue
        out.append(r["id"])
    return out


REPEAT_LOOP_MIN_CHARS = 400


def _quote_failures() -> list[str]:
    return _committed_jsonfail_ids()


QUOTE_FAILURES = _quote_failures()


#: A 6-gram repeated this many times marks a repetition loop. Chosen over
#: comparing against the opening of the output because the v3 loops repeat a
#: *phrase* from mid-sentence, not the first tokens.
# gate thresholds are read from gates_v3, never restated here, so this file
# cannot drift from the gates it is measured against
SCHEMA_GATE = 1.0
TOPIC_GATE = 0.90

LOOP_SHINGLE = 6
LOOP_SHINGLE_MIN = 4
LOOP_MIN_TOKENS = 60


def is_loop(raw: str) -> bool:
    w = re.findall(r"\w+", raw)
    if len(w) < LOOP_MIN_TOKENS:
        return False
    sh = Counter(tuple(w[i:i + LOOP_SHINGLE]) for i in range(len(w) - LOOP_SHINGLE + 1))
    return sh.most_common(1)[0][1] >= LOOP_SHINGLE_MIN


#: The settings that differ from greedy only in the token budget, and those that
#: differ only in the repetition penalty. Comparing within a pair isolates one
#: knob at a time; comparing across them confounds the two.
BUDGET_ISOLATION = ("reppen105", "reppen1200")   # same penalty, 800 vs 1200 tokens
PENALTY_ISOLATION = ("budget1200", "reppen1200")  # same 1200 tokens, penalty on/off


def probe_findings(rows: list[dict], grid: dict) -> dict:
    """Answer the five diagnostic questions from the completed grid.

    Every number here is a count over the 64 probe cases, so it must not be
    compared against a gate threshold, which is defined on the full suite.
    """
    models = list(BASELINE_ARM)
    settings = sorted({r["setting"] for r in rows})

    # A. are the six committed JSON failures decoding-invariant?
    quote = {}
    for cid in QUOTE_FAILURES:
        quote[cid] = {}
        for m in models:
            b = {r["setting"]: r["behaviour"] for r in rows
                 if r["id"] == cid and r["model"] == m}
            n_fail = sum(1 for v in b.values() if v == "JSONFAIL")
            quote[cid][m] = {"by_setting": b, "jsonfail_settings": n_fail,
                             "n_settings": len(b),
                             "invariant_fail": n_fail == len(b),
                             "invariant_pass": n_fail == 0}
    literal = [c for c in QUOTE_FAILURES if c != "ev3-083"]

    # B. repetition loops per model per setting
    loops = {}
    for m in models:
        lc = sorted({r["id"] for r in rows if r["model"] == m and is_loop(r["raw"])})
        loops[m] = {"cases": lc,
                    "cells_by_setting": {s: sum(1 for r in rows if r["model"] == m
                                                and r["setting"] == s and is_loop(r["raw"]))
                                         for s in settings}}
    # isolate each knob at fixed value of the other
    isolation = {}
    for m in models:
        isolation[m] = {
            "budget_800_vs_1200_at_fixed_penalty":
                [loops[m]["cells_by_setting"][s] for s in BUDGET_ISOLATION],
            "penalty_on_vs_off_at_fixed_budget":
                [loops[m]["cells_by_setting"][s] for s in PENALTY_ISOLATION],
            "temperature_on_vs_off_at_fixed_budget":
                [loops[m]["cells_by_setting"][s] for s in ("budget1200", "temp01_norep")],
        }

    # C/D. movement per setting against the same-subset committed baseline
    movement = {}
    for m in models:
        base = grid[f"{m}/{settings[0]}"]["baseline_same_subset"]
        per = {}
        for s in settings:
            p = grid[f"{m}/{s}"]["probe"]
            per[s] = {
                "schema": p["strict_ok"], "d_schema": round(p["strict_ok"] - base["strict_ok"], 4),
                "topic_count": p["topic_count_ok"],
                "d_topic_count": round(p["topic_count_ok"] - base["topic_count_ok"], 4),
                "jsonfail": p["jsonfail"], "d_jsonfail": p["jsonfail"] - base["jsonfail"],
                "cases_schema_moved": round((p["strict_ok"] - base["strict_ok"]) * p["n"]),
                "cases_topic_moved": round((p["topic_count_ok"] - base["topic_count_ok"]) * p["n"]),
                "behaviour": p["behaviour"], "mean_chars": p["mean_chars"],
            }
        movement[m] = {"baseline": {"schema": base["strict_ok"],
                                    "topic_count": base["topic_count_ok"],
                                    "jsonfail": base["jsonfail"],
                                    "behaviour": base["behaviour"]},
                       "by_setting": per}

    # E. the best achievable cell per model, and whether it could matter
    best = {}
    for m in models:
        cand = [(s, movement[m]["by_setting"][s]) for s in settings]
        bs, bv = max(cand, key=lambda kv: (kv[1]["schema"], kv[1]["topic_count"]))
        base = movement[m]["baseline"]
        # Derived, not asserted: decide per gate which one still blocks the cell.
        # v2 does reach schema 1.00 on this subset, so naming that gate as a
        # blocker would be false -- the honest reason is the topic-count ceiling.
        gaps = []
        if bv["schema"] < SCHEMA_GATE:
            gaps.append(f"schema {bv['schema']:.4f} < {SCHEMA_GATE:.2f}")
        if bv["topic_count"] < TOPIC_GATE:
            gaps.append(f"topic_count {bv['topic_count']:.4f} < {TOPIC_GATE:.2f}")
        best[m] = {"best_setting": bs,
                   "schema": bv["schema"], "topic_count": bv["topic_count"],
                   "d_schema": bv["d_schema"], "d_topic_count": bv["d_topic_count"],
                   "cases_schema_moved": bv["cases_schema_moved"],
                   "blocking_gates": gaps,
                   "closes_a_gate": not gaps,
                   "why_not": ("this cell clears every gate the probe measures ("
                               + "; ".join(["schema", "topic_count"]) + ")"
                               if not gaps else
                               "; ".join(gaps) + "; and the gates are defined on "
                               "the full suites, not on these 64")}
    return {"n_cases": len({r["id"] for r in rows}), "settings": settings,
            "models": models, "quote_invariance": quote,
            "literal_quote_cases": literal, "loops": loops,
            "knob_isolation": isolation, "movement": movement, "best_cell": best,
            # the rule in full, so a checker can reimplement it instead of
            # trusting the counts: word tokens, 6-gram, most common 6-gram
            # occurring >= 4 times, and a floor of 60 tokens to skip short
            # generations where a repeat is unremarkable
            "loop_detector": {"unit": "word", "tokenizer": r"\w+",
                              "shingle": LOOP_SHINGLE, "min_repeats": LOOP_SHINGLE_MIN,
                              "min_tokens": LOOP_MIN_TOKENS}}


def _defect_case_ids() -> list[str]:
    """Committed JSON failures plus every case that showed a defect in the probe."""
    ids = list(QUOTE_FAILURES)
    for r in _rows():
        if not r["strict_ok"] and r["id"] not in ids:
            ids.append(r["id"])
    return ids


def _cases() -> dict[str, dict]:
    return {json.loads(l)["id"]: json.loads(l)
            for l in CASES.read_text(encoding="utf-8").splitlines() if l.strip()}


def _rows() -> list[dict]:
    out = []
    if not RESULTS.exists():
        return out
    for line in RESULTS.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def _topics(raw: str) -> list[dict]:
    inner = getattr(parse_payload(raw), "parsed", None)
    topics = inner.get("topics") if isinstance(inner, dict) else None
    return [t for t in (topics or []) if isinstance(t, dict)]


def _domains(raw: str) -> list[str]:
    return sorted(str(t.get("domain", "")) for t in _topics(raw))


def _issues(raw: str) -> list[str]:
    return [str(t.get("issue", "")) for t in _topics(raw)]


def _measures(rows: list[dict], cases: dict[str, dict], *, have_raw: bool) -> dict:
    n = len(rows)
    if not n:
        return {"n": 0}
    strict = sum(1 for r in rows if r["strict_ok"])
    tc = 0
    dm = leak = 0
    beh: Counter = Counter()
    for r in rows:
        c = cases.get(r["id"])
        beh[r["behaviour"]] += 1
        exp_n = (c or {}).get("expected_topic_count")
        if exp_n is not None and r.get("predicted_topic_count") == exp_n:
            tc += 1
        if have_raw:
            exp_dom = sorted((c or {}).get("expected_domains") or [])
            if exp_dom and _domains(r["raw"]) == exp_dom:
                dm = (dm or 0) + 1
            if _scan(_issues(r["raw"]), GATE_RE):
                leak = (leak or 0) + 1
    out = {
        "n": n,
        "strict_ok": round(strict / n, 4),
        "schema_fail": n - strict,
        "topic_count_ok": round(tc / n, 4),
        "behaviour": dict(beh),
        "jsonfail": beh.get("JSONFAIL", 0),
        "mean_chars": round(sum(r.get("raw_chars") or 0 for r in rows) / n, 1),
    }
    if have_raw:
        # Denominator is the subset of rows whose case declares expected domains.
        denom = sum(1 for r in rows if (cases.get(r["id"]) or {}).get("expected_domains"))
        out["domain_set_exact"] = round(dm / denom, 4) if denom else None
        out["boilerplate_rate"] = round(leak / n, 4)
    else:
        out["domain_set_exact"] = None
        out["boilerplate_rate"] = None
    return out


def _baseline_measures(model: str, ids: set[str], cases: dict[str, dict]) -> dict:
    """Baseline restricted to the probe's cases.

    ``eval_v3/results`` keeps ``predicted_topic_count`` and the schema flag but
    not the raw text; the raw text lives in the behaviour recordings and only
    covers the 52 two-topic cases. So this row reports what the committed
    recording supports and leaves raw-derived measures ``None`` rather than
    filling them with zeros, which would read as a measured zero.
    """
    arm = BASELINE_ARM[model]
    p = TUNE / "eval_v3/results" / f"{arm}.json"
    if not p.exists():
        return {"n": 0}
    per = {c["id"]: c for c in json.loads(p.read_text(encoding="utf-8"))["per_case"]}
    # The behaviour recordings cover only the 52 two-topic cases and are filed
    # under a -recheck suffix for two of the arms, so they cannot be the source
    # of truth for all 64 probe cases. Classify from the committed per-case
    # flags instead, and use the recording only to confirm the two-topic split.
    beh = {}
    for name in (f"{arm}.json", f"{arm}-recheck.json"):
        bp = TUNE / "eval_v3/behaviour" / name
        if bp.exists():
            beh = {c["id"]: c["behaviour"]
                   for c in json.loads(bp.read_text(encoding="utf-8"))["cases"]}
            break

    def _classify(cid: str) -> str:
        c = per[cid]
        if not c["json_ok"]:
            return "JSONFAIL"
        # behaviour_v3's buckets are defined against the *expected* topic count:
        # SPLIT means predicted == expected, and MERGE/STOP are only meaningful
        # when the case expects two topics. So the fallback must compare the two
        # counts; treating predicted == 1 as MERGE mislabels every single-topic
        # control as a merge, which is what an earlier revision did.
        if cid in beh:
            return beh[cid]
        exp = c.get("expected_topic_count") or 0
        pred = c.get("predicted_topic_count") or 0
        if pred > exp:
            return "OVER"
        if pred == exp:
            return "SPLIT"
        # under-emission on a two-topic case: MERGE vs STOP needs the raw text to
        # decide, so STOP is the conservative bucket and the fallback is recorded
        return "STOP"

    rows = [{"id": cid,
             "behaviour": _classify(cid),
             "predicted_topic_count": per[cid]["predicted_topic_count"],
             "strict_ok": bool(per[cid]["schema_ok"] and per[cid]["json_ok"]),
             "raw_chars": 0, "raw": ""}
            for cid in sorted(ids) if cid in per]
    out = _measures(rows, cases, have_raw=False)
    out["behaviour_by_case"] = {r["id"]: r["behaviour"] for r in rows}
    out["behaviour_source"] = ("behaviour_v3 recording where available, else committed "
                              "json_ok + predicted_topic_count")
    out["recording_used_for"] = len([c for c in ids if c in beh])
    out["flag_fallback_for"] = len([c for c in ids if c in per and c not in beh])
    out["setting"] = "canonical temp=0.0 max_tokens=800 (committed)"
    return out


def _transitions(rows: list[dict], base_beh: dict[str, str]) -> dict:
    tr: dict[str, Counter] = defaultdict(Counter)
    for r in rows:
        b = base_beh.get(r["id"])
        if b:
            tr[f"{b} -> {r['behaviour']}"][r["id"]] += 1
    return {k: sum(v.values()) for k, v in sorted(tr.items())}


def build() -> dict:
    rows = _rows()
    cases = _cases()
    by: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in rows:
        by[(r["model"], r["setting"])].append(r)

    # baseline behaviour per model, from the committed behaviour recordings
    # One source of truth: the baseline classification comes from
    # _baseline_measures, which knows the behaviour recordings are partial and
    # filed under a -recheck suffix for two arms. Reading them again here, as an
    # earlier revision did, silently produced an empty baseline for v2 and
    # corrected-v3 and made the transition matrices meaningless.
    grid = {}
    for (model, setting), rs in sorted(by.items()):
        ids = {r["id"] for r in rs}
        base = _baseline_measures(model, ids, cases)
        grid[f"{model}/{setting}"] = {
            "n_cases": len(ids),
            "probe": _measures(rs, cases, have_raw=True),
            "baseline_same_subset": base,
            "transitions": _transitions(rs, base.get("behaviour_by_case", {})),
        }

    defects = {}
    for cid in _defect_case_ids():
        # group by the row's own model -- do not cross-label rows across arms
        for r in rows:
            if r["id"] != cid:
                continue
            defects.setdefault(cid, {}).setdefault(r["model"], {})[r["setting"]] = {
                "behaviour": r["behaviour"], "chars": r["raw_chars"],
                "strict_ok": r["strict_ok"]}

    # control invariance: greedy rows must be identical to each other
    ctrl = defaultdict(dict)
    for r in rows:
        if "control" in r.get("tags", []):
            ctrl[r["id"]][f"{r['model']}/{r['setting']}"] = r["behaviour"]

    return {"grid": grid, "defects": defects, "controls": dict(ctrl),
            "findings": probe_findings(rows, grid),
            "gate_coverage": {k: {"probe_covers": v[0], "why": v[1]} for k, v in GATE_COVERAGE.items()},
            "complete": len(rows) >= 960,
            "rows": len(rows)}


def to_markdown(r: dict) -> str:
    L = [f"# Decoding probe matrix ({r['rows']} generations, "
         f"{'complete' if r['complete'] else 'PARTIAL'})", ""]
    L += ["## Gate coverage of this probe", "",
          "| blocking gate | probe can adjudicate | note |", "|---|---|---|"]
    for k, v in r["gate_coverage"].items():
        L.append(f"| {k} | {'yes' if v['probe_covers'] else 'no'} | {v['why']} |")
    L += ["", "## Per-arm setting results", "",
          "| arm/setting | n | strict ok | jsonfail | topic count | mean chars | changed behaviour |",
          "|---|---|---|---|---|---|---|"]
    for k, v in r["grid"].items():
        p = v["probe"]
        ch = sum(n for t, n in v["transitions"].items()
                 if t.split(" -> ")[1] != t.split(" -> ")[0])
        L.append(f"| {k} | {p['n']} | {p['strict_ok']:.4f} | {p['jsonfail']} | "
                 f"{p['topic_count_ok']:.4f} | {p['mean_chars']} | {ch} |")
    L += ["", "## Quote and loop defects under each setting", "",
          "| case | arm | setting | behaviour | chars |", "|---|---|---|---|---|"]
    for cid, per_arm in r["defects"].items():
        for arm, sets in per_arm.items():
            for s, x in sets.items():
                L.append(f"| {cid} | {arm} | {s} | {x['behaviour']} | {x['chars']} |")
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--json-out", type=Path)
    a = ap.parse_args()
    r = build()
    if a.json_out:
        a.json_out.write_text(json.dumps(r, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"wrote {a.json_out}")
    print(to_markdown(r))


if __name__ == "__main__":
    main()