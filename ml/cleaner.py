"""Administrative-noise filtering for municipal appeal records.

Data-prep observation (from prior exploration of the dataset): a large share
of records are not citizen complaints at all but administrative docket
entries that merely give the caller a reference phone number ("надано номер
телефону ..."). Those carry no learnable signal, so they are removed before
any benchmark is created.

Filtering rule (explicit and stable):

1. ``normalize_text`` lowercases and collapses all whitespace.
2. ``is_administrative_noise`` returns True when the normalized content
   *begins with* a variant of the referral boilerplate: a declined past-tense
   form of "надати" followed by "номер"/"номери" (with common OCR/typing
   variations such as "надно", "надоно", "надані", "надан"). The match is
   anchored to the start of the string only, so a complaint that merely
   *mentions* a forwarded number later in the text is kept.
"""

from __future__ import annotations

import re

BROAD_KINDS = {"Інше", "Інші питання", "null", None}

# Matches normalized content that begins with a declined form of
# "надати номер(и)" where the first word starts with "над" and the second
# token starts with "номер".
_NUMBER_REFERRAL_RE = re.compile(r"^над\w*\s+номер")


def normalize_text(content: str | None) -> str:
    """Lowercase and collapse all whitespace for stable matching."""
    if content is None:
        return ""
    return " ".join(content.lower().split())


def is_administrative_noise(content: str | None) -> bool:
    """True if the content begins with referral boilerplate text."""
    text = normalize_text(content)
    if not text:
        return False
    return bool(_NUMBER_REFERRAL_RE.match(text))


def is_broad_kind(kind: str | None) -> bool:
    """True when the kind label carries no specific topic signal."""
    return kind in BROAD_KINDS