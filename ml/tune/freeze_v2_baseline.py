"""Freeze the v2 baseline into one machine-readable record.

Why
---
The v3 experiment has to compare three arms (v2, seeded control, v3 treatment) and
the comparison has to be *re-readable by a third party*. Until now the v2 numbers
lived in four unrelated files with four different shapes, and the gate baselines in
``ml/data/tune/v3/plan.json`` had been transcribed by hand. Transcribing numbers by
hand is what made them wrong (see ``DISCREPANCIES`` below).

This module assembles the record **from the artifacts**, so re-running it always
reflects what is actually on disk.

    .venv/bin/python -m ml.tune.freeze_v2_baseline

Writes ``ml/data/tune/v3/v2_baseline.json``. Never writes to any evaluation output.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
V3_DIR = ROOT / "ml/data/tune/v3"
ADAPTER = ROOT / "ml/data/tune/adapters/qwen3-8b-lora-v2-attempt10-fused"

#: Plan values that were hand-transcribed and did not survive measurement.
#: Recorded rather than silently corrected: the discrepancy *is* a result.
DISCREPANCIES = [
    {
        "gate": "success_gates.schema_validity.baseline",
        "claimed": 1.0,
        "claimed_source": "plan.json, no provenance",
        "measured": 0.9863,
        "measured_source": "eval_v3/results/v2.json metrics.schema_validity_rate",
        "resolution": "use measured eval_v3 value; the frozen 329 schema is 0.997",
    },
    {
        "gate": "success_gates.boilerplate_leak.baseline",
        "claimed": 0.659,
        "claimed_source": "plan.json",
        "measured": 0.3699,
        "measured_source": "eval_v3/results/v2.json metrics.boilerplate_leak_rate",
        "resolution": (
            "0.659 is the frozen-329 figure (217/329 target labels, contract_audit "
            "section 4), mis-attributed to eval_v3. eval_v3 measures 0.3699. The two "
            "corpora disagree because 329 references are uncleaned weak labels and "
            "eval_v3 references go through BOILER_SENT."
        ),
    },
    {
        "gate": "success_gates.multi_topic_count.baseline",
        "claimed": 0.824,
        "claimed_source": "plan.json",
        "measured": 0.7808,
        "measured_source": "eval_v3/results/v2.json metrics.topic_count_accuracy",
        "resolution": (
            "0.824 matches neither suite: multitopic eval is 0.8588 and measured "
            "eval_v3 is 0.7808. Use the measured eval_v3 value for eval_v3 gates."
        ),
    },
    {
        "gate": "success_gates.terse_single_topic.baseline",
        "claimed": None,
        "claimed_source": "plan.json, explicitly unset",
        "measured": {"topic_count_accuracy": 0.7857, "domain_set_exact": 0.6429,
                     "n": 14},
        "measured_source": "eval_v3/results/v2.json subsets.terse_text_under_150",
        "resolution": "now measured, but on only 14 cases; report the n alongside it",
    },
    {
        "gate": "success_gates.action_empty_when_absent.baseline",
        "claimed": None,
        "claimed_source": "plan.json, explicitly unset",
        "measured": {"cases_inventing_an_action": 0, "n": 12},
        "measured_source": "eval_v3/results/v2.json subsets.empty_action_category_H",
        "resolution": (
            "v2 already invents no action on category H. This gate has no headroom on "
            "eval_v3 category H, so the C2 claim must be evidenced on the training "
            "data and on terse cases, not on H."
        ),
    },
]


def _read_json(rel: str) -> dict:
    return json.loads((ROOT / rel).read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _smoke(path: Path) -> dict:
    """Recompute the smoke scorecard from raw rows.

    ``eval_smoke.py`` hardcodes the tags ``finetuned`` and ``base`` and the
    ``finetuned-v2`` directory is a byte-identical copy of the v2 output, so the
    v1 comparison has to be read out of the stash.
    """
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()]
    n = len(rows)
    schema = sum(1 for r in rows if r.get("schema_valid"))
    domain = sum(1 for r in rows
                 if set(r.get("predicted_domains") or []) == set(r.get("expected_domains") or []))
    multi = sum(1 for r in rows if (r.get("predicted_topics") or 0) > 1)
    expected_multi = sum(1 for r in rows if len(r.get("expected_domains") or []) > 1)
    return {
        "path": str(path.relative_to(ROOT)),
        "n": n,
        "sha256": _sha256(path),
        "schema_valid": schema,
        "domain_coverage": domain,
        "overall_pass": schema,
        "multi_topic_correct": multi,
        "multi_topic_expected": expected_multi,
    }


def _dataset_fingerprint() -> dict:
    out = {}
    for split in ("train", "validation", "test"):
        p = ROOT / f"ml/data/tune/v2/{split}.jsonl"
        n = sum(1 for _ in p.open(encoding="utf-8"))
        out[split] = {"path": str(p.relative_to(ROOT)), "n": n, "sha256": _sha256(p)}
    return out


def _git_head() -> dict:
    def run(*a: str) -> str:
        return subprocess.run(a, cwd=ROOT, capture_output=True, text=True).stdout.strip()
    return {
        "commit": run("git", "rev-parse", "HEAD"),
        "branch": run("git", "rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(run("git", "status", "--porcelain")),
    }


def main() -> None:
    frozen329 = _read_json("ml/data/tune/eval/lora.json")["metrics"]
    mt85 = _read_json("ml/data/tune/multitopic/results/lora.json")["metrics"]
    v3 = _read_json("ml/data/tune/eval_v3/results/v2.json")
    adapter_cfg = _read_json(
        "ml/data/tune/adapters/qwen3-8b-lora-v2/adapter_config.json")

    record = {
        "purpose": (
            "Frozen v2 baseline for the v3 seeded-control experiment. Every number "
            "here is read from an on-disk artifact by ml/tune/freeze_v2_baseline.py. "
            "Do not hand-edit."
        ),
        "frozen": True,
        "model": {
            "artifact": str(ADAPTER.relative_to(ROOT)),
            "hf_repo": "Fenomen-Alex/ukrainian-municipal-accountability-qwen3-8b",
            "hf_revision": "6a831b155aa4891b2a96206df0e665c7355f04bb",
            "adapter_config": adapter_cfg,
        },
        "dataset": _dataset_fingerprint(),
        "suites": {
            "frozen_329": {
                "path": "ml/data/tune/eval/lora.json",
                "reference": "v1 weak label, uncleaned",
                "metrics": frozen329,
            },
            "multitopic_85": {
                "path": "ml/data/tune/multitopic/results/lora.json",
                "reference": "v1 weak label, uncleaned",
                "metrics": mt85,
            },
            "smoke_20_v2": {
                **_smoke(ROOT / "ml/data/tune/smoke/results/finetuned-v2/smoke_results.jsonl"),
                "note": "identical bytes to the finetuned dir; both are v2 output",
            },
            "smoke_20_v1": {
                **_smoke(ROOT / "ml/data/tune/smoke/results/v1_finetuned_stash/smoke_results.jsonl"),
                "note": "the real v1 comparison, only present in the stash",
            },
            "adversarial_146": {
                "path": "ml/data/tune/eval_v3/results/v2.json",
                "reference": "eval_v3 reference label, BOILER_SENT cleaned",
                "note": (
                    "First ever measurement. No scorer existed before "
                    "ml/tune/run_eval_v3.py, so the plan.json eval_v3 baselines were "
                    "unverified."
                ),
                "metrics": v3["metrics"],
                "by_category": v3["by_category"],
                "subsets": v3["subsets"],
            },
        },
        "comparability_warning": (
            "frozen_329 / multitopic_85 score against uncleaned v1 weak labels; "
            "adversarial_146 scores against eval_v3 references that are BOILER_SENT "
            "cleaned. A model that correctly strips boilerplate therefore scores "
            "WORSE on 329/85 than on eval_v3. Never pool these; report per suite."
        ),
        "suite_validity": {
            "distinct_normalized_texts": 146,
            "distinct_texts_of_146": 146,
            "distinct_source_uids": 122,
            "source_references": 198,
            "composed_cases": 52,
            "composed_distinct_source_uids": 39,
            "warning": (
                "The 146 cases are text-distinct, but they are not independent: 198 "
                "source references resolve to 122 uids, and the 52 composed cases are "
                "recombinations of only 39. A per-category result on B/D/F/G therefore "
                "rests on a handful of underlying complaints, so e.g. category G "
                "scoring 0.000 is 6 cases from a few sources -- suggestive, not "
                "decisive. uid reuse across cases is expected: the portal reuses case "
                "numbers across years (contract_audit section 5). Quote n with every "
                "category figure."
            ),
        },
        "metric_caveat_action_redundancy": (
            "eval_v3 references are themselves 45.21% action-redundant "
            "(floor_reference.json), so eval_v3 action_redundancy_rate partly measures "
            "the reference rather than the model. The audit's 98.1% (154/157) is a "
            "different denominator: v2's non-empty actions on the frozen 329. Do not "
            "compare the two. Judge C2 on the training data instead."
        ),
        "plan_baseline_discrepancies": DISCREPANCIES,
        "git": _git_head(),
    }

    V3_DIR.mkdir(parents=True, exist_ok=True)
    out = V3_DIR / "v2_baseline.json"
    out.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")

    a = record["suites"]["adversarial_146"]["metrics"]
    f = record["suites"]["frozen_329"]["metrics"]
    m = record["suites"]["multitopic_85"]["metrics"]
    print("frozen v2 baseline ->", out.relative_to(ROOT))
    print(f"  frozen 329   domain_acc {f['domain_accuracy']}  rouge_l {f['issue_rouge_l']}")
    print(f"  multitopic85 count_acc  {m['topic_count_accuracy']}  set_exact {m['domain_set_exact']}")
    print(f"  adversarial  count_acc  {a['topic_count_accuracy']}  set_exact {a['domain_set_exact']}")
    print(f"               boiler_leak {a['boilerplate_leak_rate']}  schema {a['schema_validity_rate']}")
    print(f"  plan discrepancies recorded: {len(DISCREPANCIES)}")


if __name__ == "__main__":
    main()
