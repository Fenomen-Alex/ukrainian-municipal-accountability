"""Multitopic mechanism check on the 52 expected-two-topic cases.

Answers one question: did the failed v3.2 same-object stream teach the model
anything about inferring independent topics from inline structure, or was the
augmentation simply too synthetic and narrow?

Three parts:

1. **Transitions.** SPLIT/MERGE/STOP/JSONFAIL movement for
   previous-v3 -> corrected-v3 and corrected-v3 -> v3.2, over all 52 cases.
2. **Ranking.** The four buckets are ordered by how much of the request survives
   to a consumer: ``JSONFAIL < STOP < MERGE < SPLIT``. JSONFAIL is worst because
   nothing parseable is emitted at all; STOP is better than that because a valid
   single topic is still returned; MERGE is better still because the one topic
   does cover both problems; SPLIT is correct. This ordering is what makes
   "improved" and "regressed" well defined for a single case.
3. **Features.** Per-case structural descriptors, each cross-tabulated against
   the corrected-v3 -> v3.2 movement, so the question "which shapes did v3.2
   get better or worse on" has an answer rather than an impression.

Features are computed from the *input* text and the *expected* topics only.
No model output feeds a feature, so a "feature" cannot be an artefact of how a
particular arm answered.

Usage::

    .venv/bin/python -m ml.tune.multitopic_mechanism
    .venv/bin/python -m ml.tune.multitopic_mechanism --json-out FILE
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from ml.tune.behavior_v3 import CASES, _sim

ROOT = Path(__file__).resolve().parents[2]
TUNE = ROOT / "ml/data/tune"
BEHAVIOUR = TUNE / "eval_v3/behaviour"

ARMS = [("previous-v3", "v3-prev-recheck.json"),
        ("corrected-v3", "v3-corrected-recheck.json"),
        ("v3.2", "v3-2.json")]
BUCKETS = ["SPLIT", "MERGE", "STOP", "JSONFAIL"]

#: survivability ordering; index == rank
RANK = {b: i for i, b in enumerate(reversed(BUCKETS))}  # JSONFAIL 0 .. SPLIT 3

#: Explicit additive markers that join a second problem statement. Deliberately
#: narrow: a bare "і"/"та" appears in almost every sentence of ordinary
#: Ukrainian prose and matched 42 of the 52 cases on its own, so it carries no
#: discriminating information and is reported separately as ``coord_conjunction``.
ADDITIVE = re.compile(r"(?:а також|і також|та також|и також|а ще|і ще|та ще|а також)", re.IGNORECASE)
COORD = re.compile(r"(?:^|[\s,])[іта](?=[\s,])", re.IGNORECASE)
SENT_END = re.compile(r"[.!?;:\n]+")

#: Categories whose two topics are separated by a sentence boundary (D) versus
#: joined inline by a conjunction (B/C). Taken from ``category_name``.
BOUNDARY_CATEGORIES = {"D"}
INLINE_CATEGORIES = {"B", "C"}
WORD = re.compile(r"[^\w]+", re.UNICODE)


def _norm(s: str) -> str:
    return WORD.sub(" ", (s or "").lower()).strip()


def _load(fn: str) -> dict[str, dict]:
    d = json.loads((BEHAVIOUR / fn).read_text(encoding="utf-8"))
    return {c["id"]: c for c in d["cases"]}


def _cases() -> dict[str, dict]:
    return {json.loads(l)["id"]: json.loads(l)
            for l in CASES.read_text(encoding="utf-8").splitlines() if l.strip()}


def features(case: dict) -> dict:
    t = case.get("text", "")
    tops = case.get("expected_topics", [])
    a, b = (tops + [{}, {}])[:2]
    objs = [_norm(tp.get("object", "")) for tp in tops]
    doms = [_norm(tp.get("domain", "")) for tp in tops]
    acts = [(tp.get("requested_action") or "").strip() for tp in tops]
    sents = [s for s in SENT_END.split(t) if s.strip()]
    cat = case.get("category")
    return {
        "category": cat,
        "category_name": case.get("category_name"),
        "same_object": bool(len(objs) == 2 and objs[0] and objs[0] == objs[1]),
        "different_object": bool(len(objs) == 2 and (not objs[0] or not objs[1] or objs[0] != objs[1])),
        "same_domain": bool(len(doms) == 2 and doms[0] and doms[0] == doms[1]),
        "different_domain": bool(len(doms) == 2 and doms[0] != doms[1]),
        "explicit_additive_marker": bool(ADDITIVE.search(t)),
        "coord_conjunction": bool(COORD.search(t)),
        # Boundary vs inline is taken from the suite's own taxonomy, which is
        # authoritative: category D is the two-sentence boundary-marked shape and
        # B/C are the inline conjunction shapes. Counting sentence terminators
        # was tried first and is wrong -- SENT_END includes ':' and ';', so a
        # single sentence carrying a colon was counted as boundary-marked and
        # the feature disagreed with the categories it was meant to summarise
        # (it put 44 cases on the "boundary" side against 8 for category D).
        "boundary_marked": cat in BOUNDARY_CATEGORIES,
        "inline": cat in INLINE_CATEGORIES,
        # Kept only so the disagreement above stays measurable rather than
        # merely asserted; not used for any conclusion.
        "punct_sentence_count": len(sents),
        "punct_boundary": len(sents) >= 2,
        "multi_action": sum(1 for x in acts if x) >= 2,
        "action_present": sum(1 for x in acts if x) >= 1,
        "issue_sim": round(_sim(a.get("issue", ""), b.get("issue", "")), 3),
        "second_similar": _sim(a.get("issue", ""), b.get("issue", "")) >= 0.30,
        "issue_sim_ge_60": _sim(a.get("issue", ""), b.get("issue", "")) >= 0.60,
        "short_text": len(t) < 200,
        "text_len": len(t),
        "sentences": len(sents),
    }


def transitions(a: dict[str, dict], b: dict[str, dict], ids: list[str]) -> Counter:
    return Counter(f"{a[i]['behaviour']} -> {b[i]['behaviour']}" for i in ids)


def movement(a: dict[str, dict], b: dict[str, dict], ids: list[str]) -> dict:
    imp = [i for i in ids if RANK[b[i]["behaviour"]] > RANK[a[i]["behaviour"]]]
    reg = [i for i in ids if RANK[b[i]["behaviour"]] < RANK[a[i]["behaviour"]]]
    same = [i for i in ids if RANK[b[i]["behaviour"]] == RANK[a[i]["behaviour"]]]
    return {"improved": imp, "regressed": reg, "unchanged": same,
            "counts": {"improved": len(imp), "regressed": len(reg),
                       "unchanged": len(same), "n": len(ids)}}


def label_movement(a: dict[str, dict], b: dict[str, dict],
                    ids: list[str]) -> dict[str, str]:
    """Per-case movement label under the survivability ranking."""
    out = {}
    for i in ids:
        ra, rb = RANK[a[i]["behaviour"]], RANK[b[i]["behaviour"]]
        out[i] = "improved" if rb > ra else "regressed" if rb < ra else "unchanged"
    return out


def crosstab(ids: list[str], feats: dict[str, dict], labels: dict[str, str],
             keys: list[str]) -> list[dict]:
    """For each feature value, how the corrected-v3 -> v3.2 movement splits."""
    out = []
    for key in keys:
        groups: dict[object, dict] = defaultdict(
            lambda: {"improved": 0, "regressed": 0, "unchanged": 0, "ids": []})
        for i in ids:
            b = groups[feats[i][key]]
            b["ids"].append(i)
            b[labels[i]] += 1
        rows = [{"value": val, **b} for val, b in
                sorted(groups.items(), key=lambda kv: str(kv[0]))]
        out.append({"feature": key, "groups": rows})
    return out


def build() -> dict:
    cases = _cases()
    beh = {name: _load(fn) for name, fn in ARMS}
    ids = sorted(set.intersection(*(set(b) for b in beh.values())))
    two = [i for i in ids if cases[i].get("expected_topic_count") == 2]
    feats = {i: features(cases[i]) for i in two}

    prev, corr, v32 = (beh[n] for n, _ in ARMS)
    pairs = [("previous-v3", "corrected-v3", prev, corr),
             ("corrected-v3", "v3.2", corr, v32)]

    cat_rows = []
    for cat in sorted({feats[i]["category"] for i in two}):
        sub = [i for i in two if feats[i]["category"] == cat]
        m = movement(corr, v32, sub)
        cat_rows.append({
            "category": cat,
            "category_name": sub and feats[sub[0]]["category_name"],
            "n": len(sub),
            "prev": Counter(prev[i]["behaviour"] for i in sub),
            "corrected": Counter(corr[i]["behaviour"] for i in sub),
            "v3.2": Counter(v32[i]["behaviour"] for i in sub),
            "corrected_to_v32": {"improved": m["counts"]["improved"],
                                 "regressed": m["counts"]["regressed"],
                                 "unchanged": m["counts"]["unchanged"]},
            "improved_ids": m["improved"], "regressed_ids": m["regressed"],
        })

    keys = ["same_object", "different_object", "same_domain", "different_domain",
            "explicit_additive_marker", "coord_conjunction", "boundary_marked",
            "punct_boundary",
            "inline", "multi_action", "action_present", "second_similar",
            "issue_sim_ge_60", "short_text", "category"]
    return {
        "n_two_topic": len(two),
        "ranking": {b: RANK[b] for b in BUCKETS},
        "buckets": {name: Counter(b[i]["behaviour"] for i in two)
                    for name, b in ((n, beh[n]) for n, _ in ARMS)},
        "transitions": {f"{x} -> {y}": dict(transitions(a, b, two))
                        for x, y, a, b in pairs},
        "movement": {f"{x} -> {y}": movement(a, b, two)["counts"]
                     for x, y, a, b in pairs},
        "by_category": cat_rows,
        "feature_crosstab": crosstab(two, feats, label_movement(corr, v32, two), keys),
        "features": feats,
        "jsonfail_ids": {name: [i for i in two if b[i]["behaviour"] == "JSONFAIL"]
                         for name, b in ((n, beh[n]) for n, _ in ARMS)},
        "jsonfail_change": {
            "corrected_only": sorted({i for i in two if corr[i]["behaviour"] == "JSONFAIL"}
                                     - {i for i in two if v32[i]["behaviour"] == "JSONFAIL"}),
            "v32_only": sorted({i for i in two if v32[i]["behaviour"] == "JSONFAIL"}
                               - {i for i in two if corr[i]["behaviour"] == "JSONFAIL"}),
            "both": sorted({i for i in two if v32[i]["behaviour"] == "JSONFAIL"
                            and corr[i]["behaviour"] == "JSONFAIL"}),
        },
    }


def to_markdown(r: dict) -> str:
    L = ["# Multitopic mechanism check\n",
         f"N = {r['n_two_topic']} expected-two-topic cases. Bucket rank "
         f"(higher is better): {r['ranking']}.\n",
         "\n## Buckets per arm\n",
         "| arm | SPLIT | MERGE | STOP | JSONFAIL |", "|---|---|---|---|---|"]
    for name, c in r["buckets"].items():
        L.append(f"| {name} | " + " | ".join(str(c.get(b, 0)) for b in BUCKETS) + " |")

    L.append("\n## Transitions, full source -> destination matrix\n")
    for label, tr in r["transitions"].items():
        L.append(f"\n**{label}**\n")
        L.append("| from \\ to | " + " | ".join(BUCKETS) + " | total |")
        L.append("|---" * (len(BUCKETS) + 2) + "|")
        for src in BUCKETS:
            cells = [str(tr.get(f"{src} -> {dst}", 0)) for dst in BUCKETS]
            row_total = sum(int(x) for x in cells)
            L.append(f"| **{src}** | " + " | ".join(cells) + f" | {row_total} |")
        col = [sum(tr.get(f"{src} -> {dst}", 0) for src in BUCKETS) for dst in BUCKETS]
        L.append("| **total** | " + " | ".join(str(x) for x in col)
                 + f" | {sum(col)} |")

    L.append("\n## Movement\n")
    L.append("| transition | improved | regressed | unchanged |")
    L.append("|---|---|---|---|")
    for label, m in r["movement"].items():
        L.append(f"| {label} | {m['improved']} | {m['regressed']} | {m['unchanged']} |")

    L.append("\n## JSONFAIL membership\n")
    L.append("| arm | JSONFAIL ids |")
    L.append("|---|---|")
    for name, ids in r["jsonfail_ids"].items():
        L.append(f"| {name} | {', '.join(ids)} |")
    ch = r["jsonfail_change"]
    L.append(f"\n- failed only in corrected-v3 (repaired by v3.2): {', '.join(ch['corrected_only']) or 'none'}")
    L.append(f"- failed only in v3.2 (new regressions): {', '.join(ch['v32_only']) or 'none'}")
    L.append(f"- failed in both: {', '.join(ch['both']) or 'none'}")

    L.append("\n## By category (corrected-v3 -> v3.2)\n")
    L.append("| cat | name | n | prev SPLIT | corr SPLIT | v3.2 SPLIT | improved | regressed | unchanged |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for c in r["by_category"]:
        L.append(f"| {c['category']} | {c['category_name']} | {c['n']} | "
                 f"{c['prev'].get('SPLIT', 0)} | {c['corrected'].get('SPLIT', 0)} | "
                 f"{c['v3.2'].get('SPLIT', 0)} | {c['corrected_to_v32']['improved']} | "
                 f"{c['corrected_to_v32']['regressed']} | {c['corrected_to_v32']['unchanged']} |")

    L.append("\n## Feature crosstab (corrected-v3 -> v3.2 movement)\n")
    for ft in r["feature_crosstab"]:
        L.append(f"\n### {ft['feature']}\n")
        L.append("| value | n | improved | regressed | unchanged |")
        L.append("|---|---|---|---|---|")
        for g in ft["groups"]:
            n = len(g["ids"])
            L.append(f"| {g['value']} | {n} | {g['improved']} | {g['regressed']} | {g['unchanged']} |")
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--json-out", type=Path)
    a = ap.parse_args()
    r = build()
    if a.json_out:
        a.json_out.write_text(json.dumps(r, ensure_ascii=False, indent=2, default=dict),
                              encoding="utf-8")
        print(f"wrote {a.json_out}")
    print(to_markdown(r))


if __name__ == "__main__":
    main()