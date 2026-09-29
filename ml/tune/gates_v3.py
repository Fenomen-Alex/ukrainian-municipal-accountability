"""Evaluate the v3 success gates and emit an auditable verdict.

The gates are specified in ``ml/reports/v3_experiment_design.md`` §5. This script
exists so the verdict is computed from the recorded numbers instead of argued
from them, and so each gate states the suite it was measured on -- the 329 and 85
suites score against v1 weak labels that still contain the boilerplate C1
deletes, so a model implementing C1 correctly *must* look worse on them, and
pooling those numbers with eval_v3 would hide the fix.

Usage::

    .venv/bin/python -m ml.tune.gates_v3 --arm v3-treatment
    .venv/bin/python -m ml.tune.gates_v3 --arm v3-treatment --json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EVAL_V3 = ROOT / "ml/data/tune/eval_v3/results"
EVAL_329 = ROOT / "ml/data/tune/eval"
EVAL_MT = ROOT / "ml/data/tune/multitopic/results"
SMOKE = ROOT / "ml/data/tune/smoke/results"

#: tag -> (dir, filename) for the 329-case suite and the multitopic suite. v2 was
#: scored before those runners grew --tag, so its results carry the default name.
FROZEN_SUITES = {
    "v2": (EVAL_329 / "lora.json", EVAL_MT / "lora.json", "finetuned-v2"),
}


@dataclass
class Gate:
    name: str
    suite: str
    target: str
    baseline: str
    observed: float | None
    blocking: bool
    passed: bool | None = None
    note: str = ""


@dataclass
class Report:
    arm: str
    gates: list[Gate] = field(default_factory=list)

    @property
    def blocking_failed(self) -> list[Gate]:
        return [g for g in self.gates if g.blocking and g.passed is False]

    @property
    def blocking_missing(self) -> list[Gate]:
        return [g for g in self.gates if g.blocking and g.passed is None]

    @property
    def advisory_failed(self) -> list[Gate]:
        return [g for g in self.gates if not g.blocking and g.passed is False]

    def verdict(self) -> str:
        if self.blocking_missing:
            return "INCOMPLETE"
        if self.blocking_failed:
            return "FAIL"
        return "PASS"

    def to_dict(self) -> dict:
        return {
            "arm": self.arm,
            "verdict": self.verdict(),
            "gates": [
                {"name": g.name, "suite": g.suite, "target": g.target,
                 "baseline": g.baseline, "observed": g.observed,
                 "blocking": g.blocking, "passed": g.passed, "note": g.note}
                for g in self.gates
            ],
        }


def _eval_v3(tag: str) -> dict:
    p = EVAL_V3 / f"{tag}.json"
    if not p.exists():
        raise SystemExit(f"missing {p}\nrun: .venv-mlx/bin/python -m ml.tune.run_eval_v3 "
                         f"--model <fused-{tag}> --tag {tag}")
    return json.loads(p.read_text())


def _suite(arm: str, index: int) -> tuple[dict | None, str]:
    """Return (metrics, path) for the 329 suite (0) or multitopic suite (1)."""
    if arm in FROZEN_SUITES:
        p = FROZEN_SUITES[arm][index]
    else:
        p = (EVAL_329 if index == 0 else EVAL_MT) / f"{arm}.json"
    return (json.loads(p.read_text()), str(p)) if p.exists() else (None, str(p))


def _smoke(arm: str) -> tuple[list[dict] | None, str]:
    # run_smoke writes smoke/results/<tag>/; the finetuned- prefix is a legacy
    # name that only the original v2 run used. Try both.
    legacy = f"finetuned-{arm}" if arm not in FROZEN_SUITES else FROZEN_SUITES[arm][2]
    for sub in (arm, legacy):
        p = SMOKE / sub / "smoke_results.jsonl"
        if p.exists():
            return [json.loads(l) for l in p.read_text().splitlines() if l.strip()], str(p)
    return None, str(SMOKE / f"{arm}" / "smoke_results.jsonl")


def _smoke_two_topic(rows: list[dict]) -> tuple[int, int]:
    """(correct, total) among the smoke cases authored as Multi-Topic."""
    mt = [r for r in rows if r.get("category") == "Multi-Topic"]
    ok = sum(1 for r in mt
             if len((r.get("parsed") or {}).get("topics") or []) >= 2)
    return ok, len(mt)


def _empty_action_invention(ev3: dict) -> int:
    """Category H asks for no action; a non-empty action is an invention."""
    h = [c for c in ev3["per_case"] if c["category"] == "H"]
    return sum(1 for c in h if any((a or "").strip() for a in c["predicted_actions"]))


def evaluate(arm: str) -> Report:
    ev3 = _eval_v3(arm)
    m, v3 = ev3["metrics"], ev3
    by_cat = ev3["by_category"]
    frozen329, p329 = _suite(arm, 0)
    mt, pmt = _suite(arm, 1)
    smoke, psmoke = _smoke(arm)

    def g(**kw) -> Gate:
        return Gate(**kw)

    gates = [
        g(name="schema", suite="eval_v3 (146)", target="= 1.00",
          baseline="v2 0.9863", observed=m["schema_validity_rate"], blocking=True,
          passed=m["schema_validity_rate"] >= 1.0),
        g(name="boilerplate", suite="eval_v3 (146)", target="<= 0.02",
          baseline="v2 0.3699", observed=m["boilerplate_leak_rate"], blocking=True,
          passed=m["boilerplate_leak_rate"] <= 0.02,
          note="primary gate for C1; falls from 0.3699 only if the fix generalises"),
        g(name="topic_count", suite="eval_v3 (146)", target=">= 0.90",
          baseline="v2 0.7808", observed=m["topic_count_accuracy"], blocking=True,
          passed=m["topic_count_accuracy"] >= 0.90),
        g(name="two_topic_recall", suite="multitopic (85)", target=">= 0.894",
          baseline="v2 0.8941",
          observed=(mt or {}).get("metrics", {}).get("multi_topic_recall"),
          blocking=True,
          passed=((mt or {}).get("metrics", {}).get("multi_topic_recall") is not None
                  and mt["metrics"]["multi_topic_recall"] >= 0.894),
          note="floor: C3/C4 must not cost the multitopic arm anything"),
        g(name="frozen_domain", suite="frozen 329", target=">= 0.82",
          baseline="v2 0.8389",
          observed=(frozen329 or {}).get("metrics", {}).get("domain_accuracy"),
          blocking=True,
          passed=((frozen329 or {}).get("metrics", {}).get("domain_accuracy") is not None
                  and frozen329["metrics"]["domain_accuracy"] >= 0.82),
          note="v1 weak labels; the floor is 0.82 because the label itself is "
               "wrong on cases the model answers correctly"),
        g(name="smoke_two_topic", suite="smoke (20)", target="4 of 4",
          baseline="v2 1 of 4",
          observed=(_smoke_two_topic(smoke)[0] if smoke else None), blocking=True,
          passed=(_smoke_two_topic(smoke)[0] == 4 if smoke else None)),
        g(name="terse_single", suite="eval_v3 cat A (10)", target=">= 0.70",
          baseline="v2 0.80", observed=by_cat["A"]["domain_set_exact"],
          blocking=False, passed=by_cat["A"]["domain_set_exact"] >= 0.70,
          note="advisory; first v3 run establishes it"),
        g(name="empty_action", suite="eval_v3 cat H (12)", target="0 invented",
          baseline="v2 0 invented", observed=_empty_action_invention(ev3),
          blocking=False, passed=_empty_action_invention(ev3) == 0,
          note="advisory; C2 should make this safer, not worse"),
        g(name="issue_rouge", suite="frozen 329", target=">= 0.80",
          baseline="v2 0.9234",
          observed=(frozen329 or {}).get("metrics", {}).get("issue_rouge_l"),
          blocking=False,
          passed=((frozen329 or {}).get("metrics", {}).get("issue_rouge_l") is not None
                  and frozen329["metrics"]["issue_rouge_l"] >= 0.80),
          note="a drop is the expected consequence of C1 removing boilerplate the "
               "v1 reference still contains; the floor catches an unrelated collapse"),
    ]
    return Report(arm=arm, gates=gates)


def to_markdown(rep: Report) -> str:
    L = [f"## v3 gates — {rep.arm}  ·  **{rep.verdict()}**\n",
         "| gate | suite | target | v2 baseline | observed | blocking | result |",
         "|---|---|---|---|---|---|---|"]
    for g in rep.gates:
        res = "n/a" if g.passed is None else ("pass" if g.passed else "**FAIL**")
        if g.observed is None:
            obs = "missing"
        elif "of" in g.target:      # count-valued gate, not a rate
            obs = f"{int(g.observed)} of {g.target.rsplit('of', 1)[1].strip()}"
        else:
            obs = f"{g.observed:.4f}"
        L.append(f"| {g.name} | {g.suite} | {g.target} | {g.baseline} | {obs} | "
                 f"{'yes' if g.blocking else 'no'} | {res} |")
    notes = [g for g in rep.gates if g.note]
    if notes:
        L.append("")
        for g in notes:
            L.append(f"- **{g.name}** — {g.note}")
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--arm", required=True)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    rep = evaluate(a.arm)
    print(json.dumps(rep.to_dict(), ensure_ascii=False, indent=2) if a.json
          else to_markdown(rep))
    raise SystemExit(0 if rep.verdict() in ("PASS", "INCOMPLETE") else 1)


if __name__ == "__main__":
    main()
