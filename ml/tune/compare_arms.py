"""Compare the v3 arms against v2, with significance, and classify the errors.

Why this is not just a subtraction
----------------------------------
The adversarial suite has 146 cases but they are not independent: 198 source
references resolve to 122 UIDs and 52 cases are composed from 39 of them. A
point difference of a few percent on n=146 is well inside what that overlap can
manufacture, so reporting "v3 is 4 points better" from the raw numbers would be
over-reading. Every comparison here is paired per case and tested:

  * binary per-case metrics  -> McNemar's exact test on the discordant pairs
  * continuous per-case metrics -> paired bootstrap CI over the differences

The paired design also removes case difficulty as a confounder: both arms see
the same 146 complaints, so a case that is hard for one is hard for the other.

Usage::

    .venv/bin/python -m ml.tune.compare_arms --arms v2 v3-control v3-treatment
    .venv/bin/python -m ml.tune.compare_arms --arms v2 v3-treatment --markdown
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from math import comb
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "ml/data/tune/eval_v3/results"

#: Direction of improvement per metric. ``None`` means "lower is not better",
#: which is how a metric that is only informative should be treated.
LOWER_IS_BETTER = {"boilerplate_leak_rate", "action_redundancy_rate",
                   "out_of_enum_rate", "duplicate_domain_rate",
                   "mean_predicted_topics", "mean_expected_topoms"}

BINARY_PER_CASE = {
    "json_ok": "json_parse_rate",
    "schema_ok": "schema_validity_rate",
    "topic_count_ok": "topic_count_accuracy",
    "domain_set_exact": "domain_set_exact",
    "boilerplate_leak": "boilerplate_leak_rate",
    "action_redundant": "action_redundancy_rate",
    "out_of_enum_domain": "out_of_enum_rate",
    "duplicate_domain": "duplicate_domain_rate",
}

CONTINUOUS_PER_CASE = {
    "issue_rouge_l_best": "issue_rouge_l_best",
    "domain_set_precision": "domain_set_precision",
    "domain_set_recall": "domain_set_recall",
}

#: Failure taxonomy. First match wins, so order encodes specificity: a case that
#: failed to parse cannot also be credited with a domain error.
TAXONOMY = [
    ("not_json", lambda c: not c["json_ok"]),
    ("bad_schema", lambda c: c["json_ok"] and not c["schema_ok"]),
    ("out_of_enum", lambda c: c["out_of_enum_domain"]),
    ("duplicate_domain", lambda c: c["duplicate_domain"]),
    ("topic_count", lambda c: c["schema_ok"] and not c["topic_count_ok"]),
    ("domain_missed", lambda c: c["topic_count_ok"] and not c["domain_set_exact"]
     and c["domain_set_recall"] < 1.0),
    ("domain_invented", lambda c: c["topic_count_ok"] and not c["domain_set_exact"]
     and c["domain_set_precision"] < 1.0),
    ("boilerplate_leak", lambda c: c["boilerplate_leak"]),
    ("action_copy", lambda c: c["action_redundant"]),
]


def classify(case: dict) -> str | None:
    """The single failure this case represents, or None if it is clean.

    Deliberately not "every predicate that fires": one case that produced
    unparseable JSON is not additionally evidence of a domain error, a
    boilerplate leak and a bad action. Counting it in four buckets would make
    the taxonomy say more about how the buckets were written than about the
    model.
    """
    return next((name for name, pred in TAXONOMY if pred(case)), None)


def load(tag: str) -> dict:
    p = RESULTS / f"{tag}.json"
    if not p.exists():
        raise SystemExit(f"missing results for {tag!r}: {p}\n"
                         f"run: .venv-mlx/bin/python -m ml.tune.run_eval_v3 "
                         f"--model <fused> --tag {tag}")
    return json.loads(p.read_text(encoding="utf-8"))


def by_id(result: dict) -> dict[str, dict]:
    return {c["id"]: c for c in result["per_case"]}


def mcnemar(a: list[bool], b: list[bool]) -> dict:
    """Exact McNemar on paired binary outcomes: a only, b only, p."""
    n01 = sum(1 for x, y in zip(a, b) if x and not y)   # wins for a
    n10 = sum(1 for x, y in zip(a, b) if y and not x)   # wins for b
    n = n01 + n10
    if n == 0:
        return {"a_only": 0, "b_only": 0, "p": 1.0}
    # Two-sided exact binomial test at p=0.5.
    k = min(n01, n10)
    tail = sum(comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return {"a_only": n01, "b_only": n10, "p": min(1.0, 2 * tail)}


def paired_bootstrap(a: list[float], b: list[float], iters: int = 10000,
                     seed: int = 0) -> dict:
    """CI on the mean paired difference (a - b) by resampling cases."""
    rng = random.Random(seed)
    n = len(a)
    diffs = [x - y for x, y in zip(a, b)]
    means = []
    for _ in range(iters):
        s = 0.0
        for _ in range(n):
            s += diffs[rng.randrange(n)]
        means.append(s / n)
    means.sort()
    return {
        "delta": sum(diffs) / n,
        "ci95": (means[int(0.025 * iters)], means[int(0.975 * iters)]),
    }


def _flag(case: dict, key: str) -> bool | None:
    v = case.get(key)
    if key in ("boilerplate_leak", "action_redundant", "out_of_enum_domain",
               "duplicate_domain"):
        return None if v is None else bool(v)
    return None if v is None else bool(v)


def compare(base_tag: str, arm_tag: str) -> dict:
    """Paired comparison of one arm against the baseline."""
    base, arm = load(base_tag), load(arm_tag)
    b_cases, a_cases = by_id(base), by_id(arm)
    shared = sorted(set(b_cases) & set(a_cases))
    if len(shared) != len(b_cases) or len(shared) != len(a_cases):
        raise SystemExit(
            f"arms do not cover the same cases: {base_tag}={len(b_cases)}, "
            f"{arm_tag}={len(a_cases)}, shared={len(shared)}. Refusing to compare "
            "different suites.")

    out: dict = {"base": base_tag, "arm": arm_tag, "n": len(shared), "binary": {},
                 "continuous": {}}
    for key, metric in BINARY_PER_CASE.items():
        a = [_flag(a_cases[i], key) for i in shared]
        b = [_flag(b_cases[i], key) for i in shared]
        pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
        if not pairs:
            continue
        av = [x for x, _ in pairs]
        bv = [y for _, y in pairs]
        t = mcnemar(av, bv)
        out["binary"][metric] = {
            "base": sum(bv) / len(bv), "arm": sum(av) / len(av), **t,
        }
    for key, metric in CONTINUOUS_PER_CASE.items():
        av = [float(a_cases[i].get(key) or 0.0) for i in shared]
        bv = [float(b_cases[i].get(key) or 0.0) for i in shared]
        out["continuous"][metric] = {
            "base": sum(bv) / len(bv), "arm": sum(av) / len(av),
            **paired_bootstrap(av, bv),
        }
    return out


def taxonomy(tags: list[str]) -> dict:
    """Failure counts per arm, overall and per category."""
    out: dict = {}
    for tag in tags:
        cases = load(tag)["per_case"]
        overall = Counter()
        per_cat: dict[str, Counter] = defaultdict(Counter)
        clean = 0
        for c in cases:
            hit = classify(c)
            if hit is None:
                clean += 1
            else:
                overall[hit] += 1
                per_cat[c["category_name"]][hit] += 1
        out[tag] = {
            "n": len(cases),
            "no_error": clean,
            "counts": dict(overall.most_common()),
            "by_category": {k: dict(v.most_common()) for k, v in sorted(per_cat.items())},
        }
    return out


def to_markdown(comps: list[dict], tax: dict, tags: list[str]) -> str:
    L: list[str] = []
    L.append("## Paired comparison against v2\n")
    L.append(f"n={comps[0]['n']} cases, paired per case. McNemar exact p on binary "
             "metrics; 95% CI from a 10k paired bootstrap on continuous metrics.\n")
    for c in comps:
        L.append(f"### {c['arm']} vs {c['base']}\n")
        L.append("| metric | v2 | arm | delta | test |")
        L.append("|---|---|---|---|---|")
        for m, d in c["binary"].items():
            delta = d["arm"] - d["base"]
            sign = "+" if delta >= 0 else ""
            sig = "significant" if d["p"] < 0.05 else "n.s."
            arrow = " (lower is better)" if m in LOWER_IS_BETTER else ""
            L.append(f"| {m}{arrow} | {d['base']:.4f} | {d['arm']:.4f} | "
                     f"{sign}{delta:.4f} | McNemar p={d['p']:.4f} ({sig}) |")
        for m, d in c["continuous"].items():
            lo, hi = d["ci95"]
            crosses = "n.s." if lo <= 0 <= hi else "significant"
            L.append(f"| {m} | {d['base']:.4f} | {d['arm']:.4f} | "
                     f"{d['delta']:+.4f} | 95% CI [{lo:+.4f}, {hi:+.4f}] ({crosses}) |")
        L.append("")
    L.append("## Error taxonomy\n")
    keys = sorted({k for t in tax.values() for k in t["counts"]})
    L.append("| failure | " + " | ".join(tags) + " |")
    L.append("|---" * (len(tags) + 1) + "|")
    L.append("| **no error** | " + " | ".join(str(tax[t]["no_error"]) for t in tags) + " |")
    for k in keys:
        L.append(f"| {k} | " + " | ".join(str(tax[t]["counts"].get(k, 0)) for t in tags) + " |")
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--arms", nargs="+", default=["v2", "v3-control", "v3-treatment"],
                    help="tags to report; the first is the baseline")
    ap.add_argument("--markdown", action="store_true", help="write report to stdout")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    comps = [compare(a.arms[0], t) for t in a.arms[1:]]
    tax = taxonomy(a.arms)
    if a.markdown:
        md = to_markdown(comps, tax, a.arms)
        if a.out:
            a.out.write_text(md + "\n", encoding="utf-8")
            print(f"wrote {a.out}")
        else:
            print(md)
    else:
        print(json.dumps({"comparisons": comps, "taxonomy": tax},
                         ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
