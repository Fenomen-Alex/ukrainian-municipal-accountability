"""Action-behaviour analysis across the four arms.

The action defect was carried into every v3 arm, so the question is where it
came from. This module separates the three candidate explanations that the data
can actually distinguish -- training-distribution, label noise, and model
copying -- instead of assuming one:

* **training-distribution** — compare the action rate of the *corpus targets*
  against the rate the frozen test labels expect. If the training prior does not
  match the label prior, the model is faithfully reproducing the distribution it
  was given and the defect is in the data.
* **label noise** — separate the two directions of disagreement. Omitting an
  action the label wants and inventing one the label does not are different
  failures with different causes.
* **model copying** — fraction of emitted actions that are verbatim spans of the
  complaint text.

It also tests whether decomposing more topics costs action extraction, by
conditioning action emission on the SPLIT/MERGE/STOP bucket.

Usage::

    .venv/bin/python -m ml.tune.action_analysis
    .venv/bin/python -m ml.tune.action_analysis --json-out FILE
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from ml.tune.cross_arm_matrix import (ARMS, COPY_MIN_CHARS, behaviour, ev3, frozen,
                                     is_copied, smoke)
from ml.tune.evaluate import parse_payload

ROOT = Path(__file__).resolve().parents[2]
TUNE = ROOT / "ml/data/tune"
FROZEN_FILE = {"v2": "lora.json"}
BEHAVIOUR = {"v3-treatment": "v3-prev-recheck.json",
             "v3-corrected": "v3-corrected-recheck.json",
             "v3-2": "v3-2.json"}


def corpus_action_rate(path: Path) -> dict:
    """Share of targets that carry a non-empty requested_action.

    Handles both on-disk shapes: training rows carry the target as the last chat
    message, while the frozen label file stores the derived target directly.
    """
    n = t = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if "messages" in rec:
            try:
                payload = json.loads(rec["messages"][-1]["content"])
            except (KeyError, json.JSONDecodeError):
                continue
        elif "topics" in rec:
            payload = rec
        else:
            continue
        n += 1
        if any((x.get("requested_action") or "").strip()
               for x in payload.get("topics", [])):
            t += 1
    return {"n": n, "with_action": t, "rate": round(t / n, 4) if n else None}


def frozen_disagreement(arm: str) -> dict:
    """Split frozen action-presence mismatches into omission vs invention."""
    run = frozen(arm)
    tag = FROZEN_FILE.get(arm, f"{arm}.json")[:-5]
    tgt_p = TUNE / "eval" / f"{tag}_targets.jsonl"
    if run is None or not tgt_p.exists():
        return {}
    tgt = {json.loads(l)["idx"]: json.loads(l)
           for l in tgt_p.read_text(encoding="utf-8").splitlines() if l.strip()}
    omit = invent = agree = 0
    for p in run["predictions"]:
        t = tgt[p["idx"]]["topics"][0]
        topics = parse_payload(p["raw"]).topics or []
        pred = (topics[0].get("requested_action", "") if topics else "")
        has_t = bool((t.get("requested_action") or "").strip())
        has_p = bool(pred.strip())
        if has_t and not has_p:
            omit += 1
        elif has_p and not has_t:
            invent += 1
        else:
            agree += 1
    m = run["metrics"]
    return {"n": len(run["predictions"]), "agree": agree,
            "omitted_action_wanted_by_label": omit,
            "invented_action_not_in_label": invent,
            "label_action_rate": m["action_presence_target"],
            "predicted_action_rate": m["action_presence_pred"],
            "presence_match": m["action_presence_match"]}


def cat_h_invention(arm: str) -> dict:
    d = ev3(arm)
    if not d:
        return {}
    h = [c for c in d["per_case"] if c["category"] == "H"]
    invented = [c["id"] for c in h
                if any((a or "").strip() for a in c["predicted_actions"])]
    return {"n": len(h), "invented": len(invented), "ids": invented}


def redundancy(arm: str) -> dict:
    d = ev3(arm)
    if not d:
        return {}
    m = d["metrics"]
    return {"action_redundancy_rate": m["action_redundancy_rate"]}


def copy_rate_ev3(arm: str) -> dict:
    from ml.tune.cross_arm_matrix import action_copy_ev3, action_copy_ev3_pc
    d = ev3(arm)
    if not d:
        return {}
    pc = action_copy_ev3_pc(arm)
    copied = sorted(i for i, v in (pc or {}).items() if v)
    return {"rate": action_copy_ev3(arm), "n_copied": len(copied),
            "cases": len(d["per_case"]), "min_action_chars": COPY_MIN_CHARS}


def copy_rate_frozen(arm: str) -> dict:
    from ml.tune.cross_arm_matrix import action_copy_frozen
    r = action_copy_frozen(arm)
    return {"rate": r, "n": len(frozen(arm)["predictions"])}


def emission_by_bucket(arm: str) -> dict:
    """Action emission conditioned on the multitopic bucket.

    Answers directly whether emitting more topics costs emitting an action.
    """
    d = behaviour(arm)
    if not d:
        return {}
    by = defaultdict(lambda: {"n": 0, "any_action": 0, "mean_actions": 0.0})
    for c in d["cases"]:
        topics = parse_payload(c.get("raw", "")).topics or []
        acts = [(t.get("requested_action") or "").strip() for t in topics]
        b = by[c["behaviour"]]
        b["n"] += 1
        b["any_action"] += 1 if any(acts) else 0
        b["mean_actions"] += len(acts)
    out = {}
    for k, v in by.items():
        out[k] = {"n": v["n"], "any_action": round(v["any_action"] / v["n"], 4),
                  "mean_actions": round(v["mean_actions"] / v["n"], 4)}
    return out


def generation_action_rate_by_regime() -> dict:
    """Action rate the v3 arms *generate*, split by corpus regime.

    The corpus rate alone only says what the targets contain. This measures the
    model's behaviour on held-out generation for the two regimes the v3 training
    mix distinguishes: single-topic rows and multitopic rows. If the model simply
    reproduces its 21% training prior, both numbers land near it and the action
    loss is a training-distribution effect rather than something specific to
    multi-topic decomposition.
    """
    out = {}
    # eval_v3 holds 94 single-topic and 52 two-topic cases. Measuring "single"
    # as the whole suite silently folds the two-topic cases back in and makes
    # the single-vs-multitopic comparison circular, so the expected topic
    # count is used to split them.
    for label, path in (("eval_v3_single_topic", TUNE / "eval_v3/results"),
                        ("eval_v3_two_topic", TUNE / "eval_v3/results"),
                        ("multitopic", TUNE / "multitopic/results")):
        for arm in ("v2", "v3-treatment", "v3-corrected", "v3-2"):
            p = path / f"{arm}.json"
            if not p.exists():
                continue
            d = json.loads(p.read_text(encoding="utf-8"))
            if label.startswith("eval_v3"):
                want = 1 if label == "eval_v3_single_topic" else 2
                acts_of = lambda c: [(a or "").strip() for a in (c.get("predicted_actions") or [])]
            else:
                acts_of = lambda c: [(t.get("requested_action") or "").strip()
                                     for t in (parse_payload(c.get("raw", "")).topics or [])
                                     if isinstance(t, dict)]
            n = nonempty = 0
            for c in (d.get("per_case") or d.get("predictions") or []):
                if label.startswith("eval_v3") and c.get("expected_topic_count") != want:
                    continue
                acts = acts_of(c)
                if not acts:
                    continue
                n += 1
                nonempty += 1 if any(acts) else 0
            if n:
                out[f"{label}/{arm}"] = {"n": n, "nonempty": nonempty,
                                        "rate": round(nonempty / n, 4)}
    return out


#: Markers of an explicitly requested action in the complaint text. Used only to
#: *audit* category H, whose definition is "no action requested"; a case whose
#: text contains one of these but whose label is empty is a candidate label
#: defect. The marker list is printed with every hit so the calls can be
#: adjudicated by hand rather than trusted blindly.
REQUEST_MARKERS = re.compile(
    r"(прошу|просить|проха|вжити заход|вжити заходів|необхідно вжити|організувати|зобов'язати)",
    re.IGNORECASE)


def cat_h_label_audit() -> dict:
    """Category H cases whose text asks for an action but whose label is empty."""
    p = TUNE / "eval_v3/cases.jsonl"
    rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    h = [r for r in rows if r["category"] == "H"]
    suspects = []
    for r in h:
        m = REQUEST_MARKERS.search(r["text"])
        label_empty = not any((t.get("requested_action") or "").strip()
                              for t in r["expected_topics"])
        if m and label_empty:
            suspects.append({"id": r["id"], "marker": m.group(0),
                             "excerpt": r["text"][max(0, m.start() - 60):m.start() + 120]})
    return {"n": len(h), "suspect_count": len(suspects),
            "suspect_rate": round(len(suspects) / len(h), 4) if h else None,
            "suspects": suspects,
            "markers": sorted({m for r in h for m in REQUEST_MARKERS.findall(r["text"])})}


def build() -> dict:
    corpora = {
        "v2 train": TUNE / "v2/train.jsonl",
        "v3.2 train (== v3 family)": TUNE / "v3/treatment_v3_2/train.jsonl",
        "frozen test (labels)": TUNE / "eval/lora_targets.jsonl",
    }
    return {
        "corpus_action_rate": {k: corpus_action_rate(p) for k, p in corpora.items()},
        "frozen_presence": {a: frozen_disagreement(a) for a in ARMS},
        "empty_action_invention_cat_h": {a: cat_h_invention(a) for a in ARMS},
        "cat_h_label_audit": cat_h_label_audit(),
        "action_redundancy": {a: redundancy(a) for a in ARMS},
        "copy_eval_v3": {a: copy_rate_ev3(a) for a in ARMS},
        "copy_frozen": {a: copy_rate_frozen(a) for a in ARMS},
        "emission_by_bucket": {a: emission_by_bucket(a)
                               for a in ARMS if behaviour(a)},
        "generation_action_rate_by_regime": generation_action_rate_by_regime(),
    }


def to_markdown(r: dict) -> str:
    L = ["# Action-behaviour analysis\n",
         "\n## Training prior vs label prior\n",
         "| corpus | rows | targets with an action | rate |", "|---|---|---|---|"]
    for k, v in r["corpus_action_rate"].items():
        L.append(f"| {k} | {v['n']} | {v['with_action']} | {v['rate']:.4f} |")

    L.append("\n## Frozen 329 action presence, split by failure direction\n")
    L.append("| arm | label wants action | model emits | match | omits wanted | invents unwanted | agrees |")
    L.append("|---|---|---|---|---|---|---|")
    for a, v in r["frozen_presence"].items():
        L.append(f"| {a} | {v['label_action_rate']:.4f} | {v['predicted_action_rate']:.4f} | "
                 f"{v['presence_match']:.4f} | {v['omitted_action_wanted_by_label']} | "
                 f"{v['invented_action_not_in_label']} | {v['agree']} |")

    L.append("\n## Empty-action invention (eval_v3 category H)\n")
    L.append("| arm | n | invented | ids |")
    L.append("|---|---|---|---|")
    for a, v in r["empty_action_invention_cat_h"].items():
        L.append(f"| {a} | {v['n']} | {v['invented']} | {', '.join(v['ids']) or '—'} |")

    au = r["cat_h_label_audit"]
    L.append("\n### Category H label audit\n")
    L.append(f"{au['suspect_count']} of {au['n']} category-H cases contain an explicit "
             "request marker in the text while their expected action is empty.\n")
    L.append("| id | marker | excerpt |")
    L.append("|---|---|---|")
    for x in au["suspects"]:
        L.append(f"| {x['id']} | `{x['marker']}` | {x['excerpt'][:110]} |")

    L.append("\n## Redundancy and copying\n")
    L.append("| arm | action redundancy (eval_v3) | copy rate (eval_v3) | copy rate (frozen) |")
    L.append("|---|---|---|---|")
    for a in ARMS:
        L.append(f"| {a} | {r['action_redundancy'][a]['action_redundancy_rate']:.4f} | "
                 f"{r['copy_eval_v3'][a]['rate']:.4f} | {r['copy_frozen'][a]['rate']:.4f} |")

    L.append("\n## Action emission conditioned on multitopic bucket (52 cases)\n")
    L.append("| arm | bucket | n | any action | mean actions |")
    L.append("|---|---|---|---|---|")
    for a, by in r["emission_by_bucket"].items():
        for k in ("SPLIT", "MERGE", "STOP", "JSONFAIL"):
            if k in by:
                v = by[k]
                L.append(f"| {a} | {k} | {v['n']} | {v['any_action']:.4f} | {v['mean_actions']:.4f} |")
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