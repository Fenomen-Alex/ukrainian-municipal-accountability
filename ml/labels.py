"""Label-map building and minimum-support filtering."""

from __future__ import annotations

from typing import Any


def build_label_map(records: list[dict[str, Any]]) -> dict[str, int]:
    """Return a stable ``{kind: index}`` mapping, sorted by kind name."""
    kinds = sorted({r["kind"] for r in records})
    return {kind: idx for idx, kind in enumerate(kinds)}


def label_distribution(records: list[dict[str, Any]]) -> list[tuple[str, int]]:
    """Return ``[(kind, count), ...]`` sorted by count descending."""
    counts: dict[str, int] = {}
    for record in records:
        kind = record["kind"]
        counts[kind] = counts.get(kind, 0) + 1
    return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))


def filter_min_examples(
    records: list[dict[str, Any]], min_examples: int = 30
) -> list[dict[str, Any]]:
    """Keep only records whose ``kind`` has at least ``min_examples`` examples."""
    counts: dict[str, int] = {}
    for record in records:
        kind = record["kind"]
        counts[kind] = counts.get(kind, 0) + 1
    keep = {kind for kind, count in counts.items() if count >= min_examples}
    return [r for r in records if r["kind"] in keep]