"""Review-friendly report over merged Gemini proposals.

Reads the merged unreviewed proposals file (default
``ml/data/gold/gemini_proposals.jsonl``) and produces a markdown report
(``ml/data/gold/gemini_review_report.md``) with aggregate statistics and
cross-tabulations.

The report describes the proposals; it never declares any proposal correct or
incorrect -- determination is left to a human reviewer.

Run::

    python3 -m ml.gold.review_report --data-dir ml/data
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, OrderedDict
from pathlib import Path

from ml.gold.select import KIND_ORDER

REPORT_VERSION = "1.0.0"
LARGE_ATTRIBUTES_THRESHOLD = 6

PII_LIKE_RE = re.compile(r"(\+?\d[\d\s\-()]{6,}\d|[\w.+-]+@[\w-]+\.[\w.]+)")

# domain values that appear in a topic's attributes values should not be
# flagged as PII; keep the checker fields scoped below.
TOPIC_FIELDS = ("domain", "issue", "object", "requested_action", "attributes")


def load_proposals(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"proposals file not found: {path}")
    return [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]


def _is_empty(value) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, (dict, list)):
        return len(value) == 0
    return False


def _topic_field_text(value) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        parts = []
        for key, val in value.items():
            if isinstance(val, str):
                parts.append(val)
        return " ".join(parts)
    return ""


def analyze(proposals: list[dict], source_kinds: list[str]) -> OrderedDict:
    total = len(proposals)
    with_topics = 0
    zero_topics = 0
    multi_topic_count = 0
    domain_freq: Counter = Counter()
    empty_field_freq: Counter = Counter()
    large_attributes: list[str] = []
    pii_like_records: list[dict] = []
    topic_count_sum = 0
    cross_tab: Counter = Counter()

    for rec in proposals:
        annotation = rec.get("annotation") or {}
        topics = annotation.get("topics") or []
        n = len(topics)
        topic_count_sum += n
        if n > 0:
            with_topics += 1
        if n == 0:
            zero_topics += 1
        if n > 1:
            multi_topic_count += 1
        for topic in topics:
            if not isinstance(topic, dict):
                continue
            domain = topic.get("domain")
            domain_freq[domain] += 1
            for field in ("issue", "object", "requested_action"):
                if _is_empty(topic.get(field)):
                    empty_field_freq[field] += 1
            attrs = topic.get("attributes")
            if _is_empty(attrs):
                empty_field_freq["attributes"] += 1
            if isinstance(attrs, dict) and len(attrs) > LARGE_ATTRIBUTES_THRESHOLD:
                large_attributes.append(rec["id"])
            # PII-like scan across every textual topic field + attribute value.
            texts = [
                _topic_field_text(topic.get(f)) for f in TOPIC_FIELDS
            ]
            if isinstance(attrs, dict):
                texts.extend(str(v) for v in attrs.values())
            joined = " ".join(texts)
            if PII_LIKE_RE.search(joined):
                pii_like_records.append({"id": rec["id"], "domain": domain})
            cross_tab[(rec.get("source_kind"), domain)] += 1

    domains = sorted({d for (_, d) in cross_tab})
    matrix = [
        {"kind": kind, "counts": {d: cross_tab[(kind, d)] for d in domains}}
        for kind in source_kinds
    ]

    stats = OrderedDict()
    stats["total_records"] = total
    stats["records_with_at_least_one_topic"] = with_topics
    stats["records_with_zero_topics"] = zero_topics
    stats["multi_topic_annotations"] = multi_topic_count
    stats["topics_total"] = topic_count_sum
    stats["avg_topics_per_record"] = (
        round(topic_count_sum / total, 3) if total else 0.0
    )
    stats["topic_domain_frequency"] = OrderedDict(sorted(
        domain_freq.items(), key=lambda kv: (-kv[1], str(kv[0]))
    ))
    stats["empty_field_frequency"] = OrderedDict(sorted(empty_field_freq.items()))
    stats["records_with_large_attribute_sets"] = large_attributes
    stats["records_with_pii_like_strings"] = pii_like_records
    stats["cross_tab_domains"] = domains
    stats["cross_tab_matrix"] = matrix
    return stats


def render_markdown(stats: OrderedDict) -> str:
    lines = []
    lines.append("# Gemini Proposal Review Report")
    lines.append("")
    lines.append(
        "*Status: unreviewed proposals. Nothing in this report should be read "
        "as a correctness declaration for any proposal.*"
    )
    lines.append("")
    lines.append(f"- Total records: **{stats['total_records']}**")
    lines.append(f"- Records with ≥1 topic: {stats['records_with_at_least_one_topic']}")
    lines.append(f"- Records with 0 topics: {stats['records_with_zero_topics']}")
    lines.append(f"- Multi-topic annotations: {stats['multi_topic_annotations']}")
    lines.append(f"- Total topics: {stats['topics_total']} "
                 f"(avg {stats['avg_topics_per_record']}/record)")
    lines.append("")

    lines.append("## Topic/domain frequency")
    lines.append("")
    lines.append("| domain | count |")
    lines.append("| --- | ---: |")
    for domain, count in stats["topic_domain_frequency"].items():
        lines.append(f"| {domain or '(none)'} | {count} |")
    lines.append("")

    lines.append("## Empty-field frequency")
    lines.append("")
    lines.append("| field | empty count |")
    lines.append("| --- | ---: |")
    for field, count in stats["empty_field_frequency"].items():
        lines.append(f"| {field} | {count} |")
    lines.append("")

    large = stats["records_with_large_attribute_sets"]
    lines.append(f"## Records with suspiciously large attribute sets "
                 f"(> {LARGE_ATTRIBUTES_THRESHOLD} attributes)  ({len(large)})")
    lines.append("")
    if large:
        for rid in large:
            lines.append(f"- `{rid}`")
    else:
        lines.append("_none_")
    lines.append("")

    pii = stats["records_with_pii_like_strings"]
    lines.append(f"## Records with potentially unnecessary PII-like strings ({len(pii)})")
    lines.append("")
    if pii:
        for rec in pii:
            lines.append(f"- `{rec['id']}` (domain: {rec['domain'] or '(none)'})")
    else:
        lines.append("_none_")
    lines.append("")

    domains = stats["cross_tab_domains"]
    lines.append("## Source-kind vs proposed-domain cross-tabulation")
    lines.append("")
    lines.append("| source_kind | " + " | ".join(domains or ["(none)"]) + " | total |")
    lines.append("| --- | " + " --- |" * (len(domains) + 1))
    for row in stats["cross_tab_matrix"]:
        counts = [row["counts"].get(d, 0) for d in domains]
        total_row = sum(counts)
        cells = " | ".join(str(c) for c in counts)
        lines.append(f"| {row['kind']} | {cells} | {total_row} |")
    lines.append("")
    return "\n".join(lines)


def build_report(data_dir: Path, proposals: list[dict] | None = None,
                 output: Path | None = None) -> str:
    if proposals is None:
        proposals = load_proposals(data_dir / "gold" / "gemini_proposals.jsonl")
    if output is None:
        output = data_dir / "gold" / "gemini_review_report.md"
    stats = analyze(proposals, KIND_ORDER)
    md = render_markdown(stats)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(md, encoding="utf-8")
    return md


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python3 -m ml.gold.review_report",
        description="Build a review report over merged Gemini proposals.",
    )
    parser.add_argument("--data-dir", default="ml/data", type=Path)
    parser.add_argument("--proposals", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    data_dir = args.data_dir
    output = args.output or (data_dir / "gold" / "gemini_review_report.md")
    proposals_path = args.proposals or (data_dir / "gold" / "gemini_proposals.jsonl")
    if not proposals_path.exists():
        print(f"error: no proposals file at {proposals_path}", file=sys.stderr)
        raise SystemExit(1)
    build_report(data_dir, load_proposals(proposals_path), output)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()