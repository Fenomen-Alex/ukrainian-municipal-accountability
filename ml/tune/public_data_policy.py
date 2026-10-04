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
#:
#: These justifications record what the file *actually* contains, including
#: measured PII indicators. An exception is only auditable if its stated reason
#: is true; an earlier version of the ``annotation_set.jsonl`` entry claimed
#: "no verbatim contact details", which was wrong (see its entry below).
REVIEWED_PUBLIC_FRAGMENTS: dict[str, str] = {
    "ml/data/gold/annotation_set.jsonl": (
        "400 human-annotated gold records used by the frozen evaluation "
        "harness. Carries the verbatim complaint text of each record, so it is "
        "review-gated rather than synthetic, and it is NOT PII-free: the "
        "heuristic PII indicators measured on this file are 10 phone-like and 3 "
        "email-like rows, because the source-side redaction is incomplete. "
        "Retained because the frozen evaluation harness cannot be reproduced "
        "or re-scored without it."
    ),
    "ml/data/gold/gemini_batches": (
        "The 10 annotation batches (plus manifest.json) that ml/gold/"
        "prepare_gemini.py emits and ml/gold/validate_gemini.py verifies for "
        "the gold-annotation procedure. Each record is {id, text, "
        "source_kind} and its text is byte-identical to the corresponding "
        "record's text in annotation_set.jsonl (verified for all 400), so it "
        "adds ZERO incremental source-data exposure beyond that reviewed "
        "exception -- it only records how the 400 records were partitioned "
        "into batches. Its PII indicators (10 phone-like, 3 email-like rows) "
        "are the same rows already counted in annotation_set.jsonl, not "
        "additional ones. Retained because the annotation pipeline and its "
        "tests (ml/tests/test_gemini_pipeline.py) read these files and pass on "
        "a clean checkout."
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


#: Tracked complaint-bearing artifacts whose public classification is **not yet
#: decided**. These are recorded rather than silently tolerated.
#:
#: ``audit_public_data.py`` only scans tracked ``*.jsonl``, so tracked ``*.json``
#: files fall outside enforcement entirely. An inventory of the tracked ``.json``
#: files that carry complaint free-text found the entries below; several of them
#: contain phone-like/email-like indicators, and
#: ``ml/data/tune/multitopic/results/*.json`` is classified ``source_derived_corpus``
#: (i.e. forbidden) by :func:`audit_public_data.classify` if it were a ``.jsonl``.
#:
#: These are NOT approved for publication. They are the unresolved set that must
#: be classified before the public-data boundary can be called clean, and each
#: entry states the measured indicators so the decision is auditable.
PENDING_PUBLIC_CLASSIFICATION: dict[str, str] = {
        "ml/data/tune/multitopic/results": (
        "Model generations for the multitopic suite (predictions[*].raw) plus "
        "their reference issue labels. Previously contained 4 phone-like hits "
        "per arm copied by the model from input; those numbers were redacted in "
        "b6b1ab8 to placeholders, removing incremental exposure (UNRESOLVED: "
        "whether this prefix should be reclassified as generated model output, or "
        "purged from history, given its dependence on published multitopic metrics). "
        "audit_public_data.classify() returns 'source_derived_corpus' for this "
        "prefix; retained because published multitopic metrics depend on it."
    ),
    "ml/data/error_analysis.json": (
        "Error-taxonomy sample: 45 complaint texts, 0 phone-like / 0 email-like "
        "indicators, 37 of which appear in no reviewed pool. UNRESOLVED: "
        "source-derived complaint input under a class that audit_public_data "
        "would call source_derived_corpus."
    ),
    "ml/data/tune/smoke/smoke_cases.json": (
        "18 complaint texts, 0 phone-like / 0 email-like indicators, none "
        "present in any reviewed pool. UNRESOLVED: provenance (synthetic vs "
        "source-derived) is not recorded anywhere in the repository."
    ),
    "ml/data/tune/multitopic/real_annotations.json": (
        "46 human reference labels including street-level addresses "
        "(e.g. 'вул.Попова, 18, корп.4, кв.43'). 0 phone-like / 0 email-like "
        "indicators. UNRESOLVED: reference labels are source-derived, and this "
        "path is likewise source_derived_corpus under classify()."
    ),
    "ml/data/tune/eval_v3/behaviour": (
        "SPLIT/MERGE/STOP behaviour probes: predictions[*].raw only, 0 "
        "phone-like / 0 email-like indicators. Belongs with generated model "
        "output, but is not listed under GENERATED_PREFIXES and is unlisted "
        "here as an exception. UNRESOLVED by omission."
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
        # Directory entries are written without a trailing slash
        # ("ml/data/tune/smoke/results"), so normalise before testing. Doing it
        # the other way round -- requiring ``prefix.endswith("/")`` -- silently
        # disabled every directory-style exception in this mapping.
        if rel.startswith(prefix.rstrip("/") + "/"):
            return reason
    return None


def pending_reason(rel: str) -> str | None:
    """Recorded-but-undecided classification for ``rel``, if any."""
    for prefix, reason in PENDING_PUBLIC_CLASSIFICATION.items():
        if rel == prefix or rel.startswith(prefix.rstrip("/") + "/"):
            return reason
    return None