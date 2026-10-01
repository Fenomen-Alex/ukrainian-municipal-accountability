"""Per-case behaviour on the eval_v3 two-topic cases: SPLIT / MERGE / STOP.

``run_eval_v3`` reports ``topic_count_accuracy``, which collapses three very
different failures into one number. On a two-topic case, emitting one topic can
mean:

  SPLIT    two topics, both problems accounted for -- correct
  MERGE    one topic carrying *both* problems -- the corrected-v3 regression
  STOP     one topic carrying only the first, second problem not mentioned
  OVER     more than two topics -- over-splitting
  JSONFAIL unparseable, so nothing can be assessed

SPLIT vs MERGE is the distinction the whole v3.2 experiment turns on, and it is
not visible in any aggregate metric. MERGE means the model *did* read both
problems and still filed them as one; STOP means it never produced the second
one at all. Different bugs, different fixes.

MERGE is decided by content, not by topic count alone: the single predicted
topic's issue is compared against both expected issues, and only counts as a
merge if it covers each at or above ``MERGE_SIM``. A one-topic output that
merely restates problem one is a STOP, not a merge.

Generation settings deliberately mirror ``run_eval_v3`` (same prompt builder,
same greedy sampler, same ``--max-tokens``) so behaviour numbers and the scored
metrics describe the same outputs.

    .venv-mlx/bin/python -m ml.tune.behavior_v3 --model <fused> --tag v3-2
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
import time
from pathlib import Path

from ml.tune.build_dataset import DATA_DIR

EVAL_DIR = DATA_DIR / "tune" / "eval_v3"
CASES = EVAL_DIR / "cases.jsonl"
OUT_DIR = EVAL_DIR / "behaviour"

#: Word-level similarity at which a predicted issue counts as "covering" an
#: expected one. 0.30 was the value used to establish the merge/stop split in
#: V3_MULTITOPIC_ANALYSIS.md, so it is reused rather than re-tuned to flatter
#: any arm.
MERGE_SIM = 0.30


def _norm(s: str) -> list[str]:
    return re.sub(r"[^\w\s]", " ", (s or "").lower()
                  .replace("’", "'").replace("‘", "'")).split()


def _sim(a: str, b: str) -> float:
    A, B = _norm(a), _norm(b)
    return difflib.SequenceMatcher(None, A, B).ratio() if A and B else 0.0


def cases() -> list[dict]:
    return [json.loads(l) for l in CASES.read_text(encoding="utf-8").splitlines() if l.strip()]


def two_topic_cases(all_cases: list[dict]) -> list[dict]:
    return [c for c in all_cases
            if c.get("expected_topic_count", len(c.get("expected_topics", []))) == 2]


def classify(case: dict, raw: str, strict_err: str | None,
             parsed_topics: list[dict] | None) -> dict:
    expected = case.get("expected_topics", [])
    rec: dict = {
        "id": case["id"],
        "category": case["category"],
        "category_name": case.get("category_name"),
        "expected_topic_count": len(expected),
        "expected_domains": case.get("expected_domains"),
        "strict_error": strict_err,
        "raw_chars": len(raw or ""),
    }
    if parsed_topics is None:
        rec.update(behaviour="JSONFAIL", predicted_topic_count=None,
                   cover=(None, None), detail=strict_err or "unparseable")
        return rec

    rec["predicted_topic_count"] = len(parsed_topics)
    issues = [t.get("issue") or "" for t in parsed_topics]

    if len(parsed_topics) > len(expected):
        rec.update(behaviour="OVER", predicted_domains=[t.get("domain") for t in parsed_topics],
                   detail=f"{len(parsed_topics)} topics for {len(expected)} expected")
        return rec

    if len(parsed_topics) == len(expected):
        rec.update(behaviour="SPLIT", predicted_domains=[t.get("domain") for t in parsed_topics])
        return rec

    # Under-emission: is the single topic covering both problems, or just one?
    sims = [_sim(issues[0], e.get("issue", "")) for e in expected]
    covers_both = sum(1 for s in sims if s >= MERGE_SIM) >= 2
    rec.update(
        cover=tuple(round(s, 3) for s in sims),
        predicted_domains=[t.get("domain") for t in parsed_topics],
        detail=("single topic covers both expected issues" if covers_both
                else "single topic covers only the first; second problem absent"),
    )
    rec["behaviour"] = "MERGE" if covers_both else "STOP"
    return rec


def generate(model_path: str, only: list[dict], max_tokens: int = 800,
             temperature: float = 0.0) -> dict[str, tuple[str, str | None, list | None]]:
    from mlx_lm import generate as gen, load
    from mlx_lm.sample_utils import make_sampler

    from ml.tune.run_eval_v3 import parse_payload
    from ml.tune.serve_v2 import build_prompt

    t0 = time.time()
    model, tok = load(model_path)
    print(f"loaded {model_path} in {time.time() - t0:.0f}s", flush=True)
    sampler = make_sampler(temp=temperature)
    out = {}
    for i, c in enumerate(only):
        raw = gen(model, tok, prompt=build_prompt(c["text"], tok),
                  max_tokens=max_tokens, sampler=sampler, verbose=False)
        parsed = parse_payload(raw)
        topics = getattr(parsed, "parsed", None)
        topics = topics.get("topics") if isinstance(topics, dict) else None
        err = None if topics is not None else "unparseable"
        out[c["id"]] = (raw, err, topics)
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(only)}  {time.time() - t0:.0f}s", flush=True)
    return out


def run(model_path: str, tag: str, max_tokens: int = 800,
        temperature: float = 0.0, from_raw: Path | None = None) -> dict:
    all_cases = cases()
    by_id = {c["id"]: c for c in all_cases}
    selected = ([by_id[i] for i in only_ids if i in by_id] if only_ids
                else two_topic_cases(all_cases))

    if from_raw is not None:
        # Re-classify previously dumped raw text instead of re-generating. Lets
        # the classifier be checked against an arm whose numbers are already
        # established, without spending GPU on a duplicate generation pass.
        from ml.tune.run_eval_v3 import parse_payload

        saved = json.loads(from_raw.read_text(encoding="utf-8"))
        raws = {}
        for c in selected:
            entry = saved[c["id"]]
            raw = entry["raw"] if isinstance(entry, dict) else entry
            parsed = parse_payload(raw)
            topics = getattr(parsed, "parsed", None)
            topics = topics.get("topics") if isinstance(topics, dict) else None
            raws[c["id"]] = (raw, None if topics is not None else "unparseable", topics)
        print(f"classified from saved raw text: {from_raw}")
    else:
        print(f"classifying {len(selected)} two-topic cases of {len(all_cases)}")
        raws = generate(model_path, selected, max_tokens, temperature)

    rows = []
    for c in selected:
        raw, err, topics = raws[c["id"]]
        rec = classify(c, raw, err, topics)
        rec["raw"] = raw
        rows.append(rec)

    buckets: dict[str, int] = {}
    for r in rows:
        buckets[r["behaviour"]] = buckets.get(r["behaviour"], 0) + 1
    correct = sum(1 for r in rows if r["behaviour"] == "SPLIT")
    summary = {
        "tag": tag,
        "model": model_path,
        "from_raw": str(from_raw) if from_raw else None,
        "n_two_topic_cases": len(rows),
        "n_all_cases": len(all_cases),
        "buckets": dict(sorted(buckets.items())),
        "expected_two_topic_accuracy": round(correct / len(rows), 4) if rows else 0.0,
        "merge_sim_threshold": MERGE_SIM,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "category_E": {
            r["id"]: r["behaviour"] for r in rows if r["category"] == "E"},
        "category_E_splits": sum(1 for r in rows
                                 if r["category"] == "E" and r["behaviour"] == "SPLIT"),
        "category_E_n": sum(1 for r in rows if r["category"] == "E"),
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / f"{tag}.json").write_text(
        json.dumps({"summary": summary, "cases": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8")

    print(f"\n{tag}: {summary['buckets']}")
    print(f"  expected-two-topic accuracy {summary['expected_two_topic_accuracy']*100:.1f}% "
          f"({correct}/{len(rows)})")
    print(f"  category E {summary['category_E_splits']}/{summary['category_E_n']} "
          f"{summary['category_E']}")
    print(f"  -> {OUT_DIR / (tag + '.json')}")
    return summary


only_ids: list[str] = []

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--max-tokens", type=int, default=800)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--from-raw", type=Path, default=None,
                    help="classify saved raw text instead of generating")
    a = ap.parse_args()
    only_ids = a.only or []
    run(a.model, a.tag, a.max_tokens, a.temperature, a.from_raw)