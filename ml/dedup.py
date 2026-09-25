"""Deterministic exact-duplicate removal."""

from __future__ import annotations

from typing import Any


def deduplicate_pairs(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Remove records that duplicate an already-seen exact ``(content, kind)`` pair.

    The first occurrence by input order is kept; identical content under a
    different ``kind`` is considered a *different* example and is kept.
    Whitespace variants are intentionally NOT merged here — leakage prevention
    across splits is handled separately via normalized text.
    """
    seen: set[tuple[str, str]] = set()
    result: list[dict[str, Any]] = []
    for record in records:
        key = (record["content"], record["kind"])
        if key in seen:
            continue
        seen.add(key)
        result.append(record)
    return result