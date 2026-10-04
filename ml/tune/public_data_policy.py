"""Public-data policy: what must not be tracked, and what stays public.

Single source of truth for the public-data boundary, consumed by
``ml/tune/audit_public_data.py`` and by the test guards in
``ml/tests/_corpora.py``.

Policy
------
Source-derived and generated corpora are **not** publicly distributed. They are
reproducible from the official CC BY source via the deterministic pipeline, so
shipping the rows adds privacy exposure without adding reproducibility.

Tracked instead: build code, schema, manifests, hashes/counts, documentation,
and the small reviewed artifacts listed in ``REVIEWED_PUBLIC_FRAGMENTS``.

See ``REPRODUCIBILITY.md``, ``PUBLIC_DATA_HISTORY.md`` and
``HISTORY_REWRITE_PLAN.md``.
"""

from __future__ import annotations

import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]

# (directory, split) pairs the pipeline materialises from the official source.
GENERATED_CORPORA: tuple[tuple[str, str], ...] = (
    ("ml/data", "train"),
    ("ml/data", "validation"),
    ("ml/data", "test"),
    ("ml/data/tune/v2", "train"),
    ("ml/data/tune/v2", "validation"),
    ("ml/data/tune/v2", "valid"),
    ("ml/data/tune/v2", "test"),
    ("ml/data/tune/multitopic", "train"),
    ("ml/data/tune/multitopic", "eval"),
    ("ml/data/tune/v3/treatment", "train"),
    ("ml/data/tune/v3/treatment", "validation"),
    ("ml/data/tune/v3/treatment", "test"),
)

#: Paths that must never be tracked in the public repository.
NOT_PUBLIC: frozenset[str] = frozenset(
    f"{directory}/{split}.jsonl" for directory, split in GENERATED_CORPORA
)

#: Tracked artifacts that do contain complaint-derived text, kept deliberately.
#: Each entry states the justification so the decision is auditable.
REVIEWED_PUBLIC_FRAGMENTS: dict[str, str] = {
    "ml/data/gold/annotation_set.jsonl": (
        "400 human-annotated gold records used by the frozen evaluation "
        "harness. Annotation labels only, no verbatim contact details."
    ),
    "ml/data/tune/eval_v3/cases.jsonl": (
        "146 held-out evaluation cases forming the frozen benchmark. Verbatim "
        "complaint text, but contact details are redacted. Required to "
        "reproduce published evaluation numbers."
    ),
    "ml/data/tune/smoke/results": (
        "20-case smoke-suite model outputs. Generated text, not source rows."
    ),
    "ml/data/tune/decoding_probe/results.jsonl": (
        "Decoding probe metadata. Generated, no personal data."
    ),
    "ml/data/tune/multitopic/provenance.jsonl": (
        "Augmentation lineage (uid and topic composition). Metadata only."
    ),
    "ml/data/tune/v3/treatment_v3_2/provenance.jsonl": (
        "v3.2 lineage metadata. No complaint text."
    ),
}


def is_forbidden(rel: str) -> bool:
    """True if ``rel`` must not appear in the public repository."""
    return rel in NOT_PUBLIC


def reviewed_reason(rel: str) -> str | None:
    """Justification for tracking ``rel``, if it is a reviewed exception."""
    if rel in REVIEWED_PUBLIC_FRAGMENTS:
        return REVIEWED_PUBLIC_FRAGMENTS[rel]
    for prefix, reason in REVIEWED_PUBLIC_FRAGMENTS.items():
        if prefix.endswith("/") and rel.startswith(prefix):
            return reason
    return None