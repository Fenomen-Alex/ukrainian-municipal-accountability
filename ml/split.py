"""Chronological splitting and cross-split text-leakage detection."""

from __future__ import annotations

import re
from typing import Any

from ml.cleaner import normalize_text

_DATE_RE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2})")


def parse_datetime(value: str | None) -> tuple[int, int, int, int, int] | None:
    """Parse ``%Y-%m-%dT%H:%M`` into a comparable tuple; None when malformed."""
    if value is None:
        return None
    m = _DATE_RE.match(value.strip())
    if not m:
        return None
    try:
        parts = tuple(int(g) for g in m.groups())
    except ValueError:
        return None
    return parts  # type: ignore[return-value]


def chronological_split(
    records: list[dict[str, Any]],
    train_start: tuple[int, int, int, int, int],
    val_start: tuple[int, int, int, int, int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Split records chronologically.

    * train:      records strictly before ``train_start``
    * validation: ``train_start`` <= date < ``val_start``
    * test:       date >= ``val_start``
    """
    train: list[dict[str, Any]] = []
    val: list[dict[str, Any]] = []
    test: list[dict[str, Any]] = []
    for record in records:
        dt = parse_datetime(record.get("receivedDateTime"))
        if dt is None:
            raise ValueError(
                f"Unparseable receivedDateTime: {record.get('receivedDateTime')!r}"
            )
        if dt < train_start:
            train.append(record)
        elif dt < val_start:
            val.append(record)
        else:
            test.append(record)
    return train, val, test


def compute_normalized_text(content: str | None) -> str:
    """Canonical form used to detect identical text across splits."""
    return normalize_text(content)


def find_cross_split_text_leakage(
    train: list[dict[str, Any]],
    val: list[dict[str, Any]],
    test: list[dict[str, Any]],
) -> dict[str, set[str]]:
    """Return ``{normalized_text: {split_names}}`` for texts present in >1 split.

    Leakage is judged on normalized (whitespace-collapsed, lowercased) content,
    so a typo or spacing difference does not hide a shared sentence.
    """
    buckets: dict[str, set[str]] = {}
    for split_name, records in (
        ("train", train),
        ("val", val),
        ("test", test),
    ):
        for record in records:
            text = compute_normalized_text(record.get("content"))
            buckets.setdefault(text, set()).add(split_name)
    return {text: splits for text, splits in buckets.items() if len(splits) > 1}


def ensure_no_text_leakage(
    train: list[dict[str, Any]],
    val: list[dict[str, Any]],
    test: list[dict[str, Any]],
) -> None:
    """Raise ``ValueError`` if any normalized text occurs in more than one split."""
    leakage = find_cross_split_text_leakage(train, val, test)
    if leakage:
        sample = next(iter(leakage.items()))
        raise ValueError(
            f"Cross-split text leakage detected ({len(leakage)} texts), "
            f"e.g. {sample[0]!r} in {sorted(sample[1])}"
        )