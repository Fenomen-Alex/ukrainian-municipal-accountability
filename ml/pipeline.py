"""Dataset preparation pipeline for the Kropyvnytskyi appeals benchmark.

Reads the raw CSV using only the Python standard library, applies the
established filters (administrative-noise, broad kinds, min-examples, exact
deduplication), performs a deterministic chronological split, and writes:

* ``train.jsonl`` / ``validation.jsonl`` / ``test.jsonl``
* ``labels.json``
* ``statistics.json`` / ``statistics.md``

Run:  python3 -m ml.pipeline --input /tmp/kropyvnytskyi-appeals.csv --out dir
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Any

from ml.cleaner import is_administrative_noise, is_broad_kind
from ml.dedup import deduplicate_pairs
from ml.labels import build_label_map, filter_min_examples, label_distribution
from ml.split import (
    chronological_split,
    find_cross_split_text_leakage,
    parse_datetime,
)

OUTPUT_COLUMNS = ["uid", "receivedDateTime", "kind", "content"]


def read_csv(input_path: str) -> list[dict[str, Any]]:
    with open(input_path, newline="", encoding="utf-8-sig") as f:
        return [dict(row) for row in csv.DictReader(f)]


def write_jsonl(records: list[dict[str, Any]], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_json(obj: Any, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, sort_keys=True)


def run_pipeline(
    input_csv: str,
    out_dir: str,
    train_start: str = "2026-01-01",
    val_start: str = "2026-07-01",
    min_examples: int = 30,
) -> dict[str, Any]:
    train_start_dt = parse_datetime(train_start + "T00:00")
    val_start_dt = parse_datetime(val_start + "T00:00")
    assert train_start_dt is not None and val_start_dt is not None

    os.makedirs(out_dir, exist_ok=True)

    raw = read_csv(input_csv)
    n_raw = len(raw)

    # 1. remove administrative/referral boilerplate
    after_noise = [r for r in raw if not is_administrative_noise(r.get("content"))]
    n_after_noise = len(after_noise)

    # 2. exclude broad kinds (Інше / Інші питання / null)
    after_kinds = [r for r in after_noise if not is_broad_kind(r.get("kind"))]
    n_after_kinds = len(after_kinds)

    # 3. exact (content, kind) deduplication
    after_dedup = deduplicate_pairs(after_kinds)
    n_after_dedup = len(after_dedup)

    # 4. keep only labels with >= min_examples examples
    after_min = filter_min_examples(after_dedup, min_examples=min_examples)
    n_after_min = len(after_min)

    # 5. chronological split
    train, val, test = chronological_split(
        after_min, train_start_dt, val_start_dt
    )

    # 6. leakage guard (must be zero)
    leakage = find_cross_split_text_leakage(train, val, test)

    # outputs
    for name, records in (("train", train), ("validation", val), ("test", test)):
        write_jsonl(records, os.path.join(out_dir, f"{name}.jsonl"))

    label_map = build_label_map(after_min)
    write_json(label_map, os.path.join(out_dir, "labels.json"))

    splits = {"train": train, "validation": val, "test": test}
    split_counts = {name: len(records) for name, records in splits.items()}
    split_distributions = {
        name: label_distribution(records) for name, records in splits.items()
    }

    report_counts = {
        "raw": n_raw,
        "after_noise_filter": n_after_noise,
        "after_excluding_broad_kinds": n_after_kinds,
        "after_deduplication": n_after_dedup,
        "after_min_examples": n_after_min,
        "final_splits": split_counts,
    }

    report = {
        "input": input_csv,
        "parameters": {
            "train_start": train_start,
            "val_start": val_start,
            "min_examples": min_examples,
        },
        "counts": report_counts,
        "labels": {
            "label_count": len(label_map),
            "label_map": label_map,
        },
        "class_distribution": {
            name: {kind: count for kind, count in dist}
            for name, dist in split_distributions.items()
        },
        "leakage": {
            "total": len(leakage),
        },
        "outputs": {
            name: os.path.join(out_dir, f"{name}.jsonl") for name in splits
        },
    }
    write_json(report, os.path.join(out_dir, "statistics.json"))

    _write_markdown(report, os.path.join(out_dir, "statistics.md"))
    return report


def _write_markdown(report: dict[str, Any], path: str) -> None:
    lines = [
        "# Dataset statistics — Ukrainian Municipal Accountability (Kropyvnytskyi)",
        "",
        f"- Input: `{report['input']}`",
        f"- Raw records: {report['counts']['raw']}",
        f"- After noise filter: {report['counts']['after_noise_filter']}",
        f"- After excluding broad kinds (Інше/Інші питання/null): "
        f"{report['counts']['after_excluding_broad_kinds']}",
        f"- After exact (content, kind) deduplication: "
        f"{report['counts']['after_deduplication']}",
        f"- After min-examples filter (>= {report['parameters']['min_examples']}): "
        f"{report['counts']['after_min_examples']}",
        "",
        "## Final splits",
        "",
        "| Split | Count |",
        "| --- | --- |",
    ]
    for name, count in report["counts"]["final_splits"].items():
        lines.append(f"| {name} | {count} |")

    lines += [
        "",
        "## Split date ranges",
        "",
        f"- train: < {report['parameters']['train_start']}",
        f"- validation: [{report['parameters']['train_start']}, "
        f"{report['parameters']['val_start']})",
        f"- test: >= {report['parameters']['val_start']}",
        "",
        "## Labels",
        "",
        f"Total labels: {report['labels']['label_count']}",
        "",
        "## Class distribution",
        "",
    ]
    dist = report["class_distribution"]
    for split_name in ("train", "validation", "test"):
        lines.append(f"### {split_name}")
        lines.append("")
        lines.append("| kind | count |")
        lines.append("| --- | --- |")
        for kind, count in dist.get(split_name, {}).items():
            lines.append(f"| {kind} | {count} |")
        lines.append("")

    lines += [
        "## Cross-split text leakage",
        "",
        f"Total leaked normalized texts: {report['leakage']['total']}",
        "",
    ]

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Path to raw CSV")
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument("--train-start", default="2026-01-01")
    parser.add_argument("--val-start", default="2026-07-01")
    parser.add_argument("--min-examples", type=int, default=30)
    args = parser.parse_args()

    report = run_pipeline(
        input_csv=args.input,
        out_dir=args.out,
        train_start=args.train_start,
        val_start=args.val_start,
        min_examples=args.min_examples,
    )
    print(json.dumps(report["counts"], ensure_ascii=False, indent=2, sort_keys=True))
    print("Full report:", os.path.join(args.out, "statistics.json"))


if __name__ == "__main__":
    main()