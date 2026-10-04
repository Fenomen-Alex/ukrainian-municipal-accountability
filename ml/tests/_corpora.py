"""Shared helpers for tests that need locally generated (private) corpora.

The public repository does not ship the source-derived corpora: they are
generated from the official CC BY source by the deterministic pipeline and
kept local. See ``REPRODUCIBILITY.md``.

A fresh clone therefore has no ``ml/data/train.jsonl`` and friends. Tests that
genuinely need those rows must skip with an explicit reason instead of failing
at collection time, and must not silently weaken their assertions.

The corpora list lives in ``ml/tune/public_data_policy.py`` so the policy and
the tests cannot drift apart.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ml.tune.public_data_policy import GENERATED_CORPORA

ROOT = Path(__file__).resolve().parents[2]

SKIP_REASON = (
    "private/generated corpora not present: this test needs corpora built from "
    "the official CC BY source, which the public repository does not ship. "
    "Rebuild them locally (see REPRODUCIBILITY.md) to run this test."
)


def corpus_path(directory: str, split: str) -> Path:
    return ROOT / directory / f"{split}.jsonl"


def missing_corpora() -> list[str]:
    """Return the generated corpora that are absent from this checkout."""
    missing = []
    for directory, split in GENERATED_CORPORA:
        path = corpus_path(directory, split)
        if not path.exists():
            missing.append(f"{directory}/{split}.jsonl")
    return missing


def corpora_present() -> bool:
    return not missing_corpora()


def skip_if_no_corpora(*, reason: str = SKIP_REASON) -> None:
    """Skip the calling test when generated corpora are absent."""
    if not corpora_present():
        pytest.skip(reason)


def skip_if_missing(path: Path | str, *, what: str = "corpus") -> None:
    """Skip the calling test when a specific required file is absent."""
    target = Path(path)
    if not target.is_absolute():
        target = ROOT / target
    if not target.exists():
        pytest.skip(f"{what} not present: {target.name} (see REPRODUCIBILITY.md)")