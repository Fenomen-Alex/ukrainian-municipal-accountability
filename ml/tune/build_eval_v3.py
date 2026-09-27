"""Build the v3 adversarial evaluation suite (``ml/data/tune/eval_v3/``).

Why this suite exists
---------------------
Two audits of the shipped v2 model (attempt-10) found that the existing
benchmarks structurally *cannot* see the failure mode we care about most:

1. ``ml/data/tune/eval`` (frozen 329) and ``ml/data/tune/multitopic/eval.jsonl``
   (85) are both dominated by long complaints (median content length 391 and 377
   chars). Every terse multi-topic failure in the 20-case smoke suite is a short
   note. The frozen metrics were blind to it.
2. The corpus splits by *date* (train < 2026-01-01, validation 2026-01..07,
   test >= 2026-07) and terse complaints are almost exclusively 2023 records,
   which all landed in train. There is therefore **no terse complaint at all**
   in validation or test, so no terse case can be borrowed from the held-out
   pool without leaking training text.

This module therefore builds a suite of 100-150 cases that covers the structural
patterns the v2 model gets wrong, using three provenance classes, all recorded
per case:

``real``
    A held-out (validation/test) complaint used verbatim. Expected output is the
    reference label of that record. Zero leakage risk.
``condensed``
    A held-out complaint reduced to a terse note by :func:`condense` -- a
    deterministic, fact-preserving reduction (boilerplate sentences dropped,
    wrapper words stripped, no new words introduced). Needed because the
    held-out pool contains no terse records at all.
``composed``
    Two held-out complaints joined by one structural template (the conjunctions
    and clause shapes we need to stress). The components are real held-out
    records; only the *combination* is synthetic. Expected output is the union
    of the two reference labels.

Reference labeler
-----------------
``_ADMIN_CLAUSE`` in :mod:`ml.tune.build_dataset` only strips the ``"згоду
надає"`` word order, while the corpus overwhelmingly writes ``"надає згоду"``.
As a result 53-64% of v1/v2 target ``issue`` fields still contain the
consent / response-delivery boilerplate. This module uses :func:`ref_topic`,
which is the contract labeler *with that gap closed* (``BOILER_SENT``), so the
expected outputs in this suite are clean. Scores on this suite are therefore
NOT directly comparable with the frozen-329 numbers and are reported separately
-- that difference is itself a measurement of the label-noise problem.

This module never reads or writes model artefacts and never runs a model.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from ml.cleaner import normalize_text
from ml.tune.build_dataset import (
    DATA_DIR,
    KIND_TO_DOMAIN,
    SYSTEM_PROMPT,
    _BUILDING_NO,
    _clean_text,
    _derive_attributes,
    _derive_object,
    _derive_requested_action,
    _redact_pii,
    _STREET_PHRASE,
)

OUT_DIR = DATA_DIR / "tune" / "eval_v3"
CASES_PATH = OUT_DIR / "cases.jsonl"
SCHEMA_PATH = OUT_DIR / "schema.json"
STATS_PATH = OUT_DIR / "stats.json"

# --------------------------------------------------------------------------- #
# Reference labeler (contract labeler + the boilerplate gap closed)
# --------------------------------------------------------------------------- #

#: Docket / consent / response-delivery boilerplate. Kept as sentence-level
#: patterns so the *whole* sentence is dropped rather than a fragment of it.
BOILER_SENT: tuple[str, ...] = (
    r"нада[єи]\s+згоду",
    r"згоду\s+нада[єи]",
    r"згоду\s+на\s+оброб",
    r"надати\s+телефоном",
    r"надати\s+поштою",
    r"надати\s+письмово",
    r"надати\s+відповідь",
    r"відповідь\s+нада[єи]",
    r"відповідь\s+надати",
    r"бажа[єи]\s+отримати",
    r"просить\s+надати",
    r"просимо\s+надати",
    r"повідомити\s+засобами",
    r"за(хист|щит)\s+персональних\s+даних",
    r"персональних\s+даних",
    r"відповідь\s+заявник",
    r"звернення\s+опрацювати",
    r"відповідь\s+заявник\w*",
    r"в\s+телефонному\s+режимі",
    r"в\s+письмовому\s+режимі",
    r"відповідь\s+нада[єи]ться",
    r"відповідь\s+буде\s+надана",
    r"заявник\w*\s+нада[єи]",
    r"заявниц\w*\s+нада[єи]",
)

_BOILER_RE = re.compile("|".join(BOILER_SENT), re.IGNORECASE)
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")

#: Wrapper phrases that carry no topical content.
_WRAPPERS = ("щодо ", "скарга на ", "скаржиться на ", "звертається з проханням ")


def strip_boilerplate(text: str) -> str:
    """Drop consent / response-delivery sentences entirely."""
    out = [s for s in _SENT_SPLIT.split((text or "").strip()) if s and not _BOILER_RE.search(s)]
    return _clean_text(" ".join(out)).strip(" .,;:-«»")


#: Dangling identity fragments left behind when the identity clause is only
#: partially matched (e.g. "... по вул. Державності. заявниці").
_DANGLING = re.compile(
    r"(?:^|[\s.;,])за(?:явник|явниц|явники|явницям|явкою|я)\w*\s*$",
    re.IGNORECASE,
)


def _clean_clause(text: str) -> str:
    """Cleaner used for the reference ``issue``: contract rules + boilerplate gap."""
    cleaned = strip_boilerplate(text)
    cleaned = re.sub(r"^[Сс]карга\s+на\s+", "", cleaned)
    cleaned = re.sub(r"^Щодо\s+", "", cleaned)
    cleaned = _redact_pii(_clean_text(cleaned)).strip(" .,;:-«»")
    previous = None
    while previous != cleaned:
        previous = cleaned
        cleaned = _DANGLING.sub("", cleaned).strip(" .,;:-«»")
    return cleaned


def ref_topic(record: dict) -> dict:
    """One reference topic for a single-topic record (contract labeler, cleaned)."""
    content = record.get("content") or ""
    return {
        "domain": KIND_TO_DOMAIN.get(record.get("kind") or "", "other"),
        "issue": _clean_clause(content)[:400],
        "object": _derive_object(content, record),
        "requested_action": _action_clause(record),
        "attributes": _derive_attributes(content, record),
    }


def _action_clause(record: dict) -> str:
    """Requested action, with boilerplate stripped (contract rule + gap closed)."""
    raw = _derive_requested_action(record.get("content") or "")
    raw = strip_boilerplate(raw)
    return _redact_pii(_clean_text(raw)).strip(" .,;:-«»")[:200]


# --------------------------------------------------------------------------- #
# Terse condensation (needed: the held-out pool has zero terse records)
# --------------------------------------------------------------------------- #

_ACTION_HEAD = re.compile(
    r"^(прошу|прохання|необхідно|треба|зверта[ює]|просимо|надай|виконай)\b[,:]?\s*",
    re.IGNORECASE,
)
_TERSE_MAX = 170


def condense(record: dict, max_len: int = _TERSE_MAX) -> str:
    """Deterministic, fact-preserving reduction of a complaint to a terse note.

    Only deletion happens: boilerplate sentences, wrapper words and a leading
    request-verb frame are removed, and the text is cut at a sentence or
    clause boundary. No word is ever added or reworded, so every fact in the
    condensed note is a fact of the source record.
    """
    text = _clean_clause(record.get("content") or "")
    text = _ACTION_HEAD.sub("", text)
    text = _clean_text(text)
    if len(text) <= max_len:
        return text.strip(" .,;:-«»")
    # Prefer cutting at a clause boundary so the remaining text stays a sentence.
    for sep in ("; ", ", а також ", ", а ", ". "):
        cut = text.rfind(sep, 0, max_len)
        if cut > max_len // 3:
            return text[:cut].strip(" .,;:-«»")
    return text[:max_len].rsplit(" ", 1)[0].strip(" .,;:-«»")


def _street(record: dict) -> str:
    m = _STREET_PHRASE.search(record.get("content") or "")
    return f"{m.group('stem')} {m.group('name')}".strip() if m else ""


# --------------------------------------------------------------------------- #
# Corpus access
# --------------------------------------------------------------------------- #


def load_split(name: str) -> list[dict]:
    return [json.loads(l) for l in (DATA_DIR / f"{name}.jsonl").read_text().splitlines()]


def held_out() -> list[dict]:
    """Validation + test records, deduplicated and sorted for deterministic selection.

    Deduplicated by normalised content, not by ``uid``: the portal reuses case
    numbers across years (``Б-67`` is both a 2023 lift complaint and a 2026
    waste-invoice complaint), and 130 uids appear in more than one split with
    unrelated text. 58 uids are duplicated inside the held-out pool. ``uid`` is
    therefore traceability metadata, not identity.
    """
    rows = load_split("validation") + load_split("test")
    seen: set[str] = set()
    unique = []
    for r in rows:
        key = normalize_text(r.get("content") or "")
        if key in seen:
            continue
        seen.add(key)
        unique.append(r)
    return sorted(unique, key=lambda r: (str(r.get("uid")), str(r.get("content"))))


def held_out_splits() -> dict[str, str]:
    """Normalised text -> the split it first appears in, for held-out records.

    Keyed by text rather than ``uid`` for the same reason as :func:`held_out`.
    """
    out: dict[str, str] = {}
    for split in ("validation", "test"):
        for r in load_split(split):
            out.setdefault(normalize_text(r.get("content") or ""), split)
    return out


def train_content_keys() -> set[str]:
    """Normalised text of every training example the model has (or could) see."""
    keys: set[str] = set()
    for split in ("train",):
        keys |= {normalize_text(r.get("content") or "") for r in load_split(split)}
    mt = DATA_DIR / "tune" / "multitopic" / "train.jsonl"
    if mt.exists():
        for line in mt.read_text().splitlines():
            keys.add(normalize_text(json.loads(line)["messages"][1]["content"]))
    v2 = DATA_DIR / "tune" / "v2" / "train.jsonl"
    if v2.exists():
        for line in v2.read_text().splitlines():
            keys.add(normalize_text(json.loads(line)["messages"][1]["content"]))
    return keys


# --------------------------------------------------------------------------- #
# Category definitions
# --------------------------------------------------------------------------- #

#: code -> (name, prose description, keyword filter or None)
CATEGORIES: dict[str, tuple[str, str, str | None]] = {
    "A": ("terse_single", "Short single-problem note (condensed real complaint)", None),
    "B": ("terse_two_topic_i", "Two terse problems joined by 'і'", None),
    "C": ("terse_two_topic_also", "Two terse problems joined by 'а також'", None),
    "D": ("two_sentences", "Two problems in two separate sentences", None),
    "E": ("same_object_two_problems", "One object/location, two different problems", None),
    "F": ("different_objects_two_problems", "Two different objects, two problems", None),
    "G": ("explicit_and_implicit_action", "One explicit request + one implicit problem", None),
    "H": ("no_explicit_action", "Problem statement with no explicit request", None),
    "I": ("long_rambling", "Long complaint with back-story and repetition", None),
    "J": ("irrelevant_prose", "Heavy administrative/legal boilerplate around the problem", None),
    "K": ("ontology_ambiguity", "Domain genuinely confusable between coarse buckets", None),
    "L": ("lighting", "Street / outdoor lighting", r"освітлен|ліхтар"),
    "M": ("traffic_lights", "Traffic signals", r"світлофор"),
    "N": ("playgrounds", "Playgrounds and play equipment", r"майданчик|дитяч\w* майдан|качелі|горка|песочниц"),
    "O": ("trees", "Trees, branches, bushes, mowing, weeds", r"дерев|кущ|гілк|амброз|бур.?ян|покос"),
    "P": ("stray_animals", "Stray / nuisance animals", r"безпритульн|тварин"),
    "Q": ("waste", "Waste, dumpsters, illegal dumping", r"смітт|сміття|відход|контейнер"),
    "R": ("water_sewerage", "Water supply and sewerage", r"водопостач|каналізац|порив|протік"),
    "S": ("roads_sidewalks", "Roads, asphalt, potholes, sidewalks, manholes", r"дорог|асфальт|ямк|тротуар|пішохід|люк"),
    "T": ("public_transport", "Public transport and drivers", r"маршрутк|автобус|тролейбус|трамва|зупинк"),
    "U": ("housing_heating", "Housing maintenance and heating", r"тепл|опален|дах|стел|стіна|будинок|квартир"),
    "V": ("elevators", "Elevators", r"ліфт"),
}

#: How many topics each category is expected to contain. Declared here so the
#: case carries its own contract and the test suite does not have to duplicate it.
CATEGORY_TOPIC_COUNT: dict[str, int] = {
    "A": 1, "B": 2, "C": 2, "D": 2, "E": 2, "F": 2, "G": 2,
    "H": 1, "I": 1, "J": 1, "K": 2,
    "L": 1, "M": 1, "N": 1, "O": 1, "P": 1, "Q": 1,
    "R": 1, "S": 1, "T": 1, "U": 1, "V": 1,
}

#: Domain pairs that the coarse 13-value ontology genuinely cannot separate well.
#: Used to pick components for category K.
AMBIGUOUS_PAIRS: tuple[tuple[str, str], ...] = (
    ("roads", "sanitation"),
    ("water", "housing"),
    ("electricity", "sanitation"),
    ("water", "sanitation"),
    ("housing", "payments"),
    ("heating", "payments"),
    ("roads", "construction"),
    ("transport", "roads"),
)


def _area_hits(text: str, pattern: str) -> bool:
    return re.search(pattern, text or "", re.IGNORECASE) is not None


# --------------------------------------------------------------------------- #
# Case assembly
# --------------------------------------------------------------------------- #


class _Pool:
    """Deterministic, de-duplicated pool of held-out records per domain."""

    def __init__(self, records: list[dict]) -> None:
        self.records = records
        self._by_domain: dict[str, list[dict]] = defaultdict(list)
        for r in records:
            if not (r.get("content") or "").strip():
                continue
            self._by_domain[KIND_TO_DOMAIN.get(r.get("kind") or "", "other")].append(r)

    def domain(self, domain: str) -> list[dict]:
        return self._by_domain.get(domain, [])

    def area(self, pattern: str) -> list[dict]:
        return [r for r in self.records if _area_hits(r.get("content") or "", pattern)]

    def all(self) -> list[dict]:
        return list(self.records)


def _slice(items: list[dict], n: int, stride: int = 7, skip: set[str] | None = None) -> list[dict]:
    """Deterministic spread-out sample (no RNG, stable across runs).

    ``skip`` holds uids already consumed by an earlier category so a real
    held-out record is used at most once in the whole suite.
    """
    if n <= 0 or not items:
        return []
    pool = [r for r in items if skip is None or str(r.get("uid")) not in skip]
    if len(pool) <= n:
        return list(pool)
    out, i = [], 0
    while len(out) < n and i < len(pool) * 3:
        out.append(pool[i % len(pool)])
        i += stride
    return out[:n]


def _case(
    cid: str,
    category: str,
    text: str,
    topics: list[dict],
    *,
    kind: str,
    sources: list[str],
    splits: list[str],
    note: str,
) -> dict:
    return {
        "id": cid,
        "category": category,
        "category_name": CATEGORIES[category][0],
        "expected_topic_count": CATEGORY_TOPIC_COUNT[category],
        "provenance": {
            "kind": kind,
            "sources": sources,
            "splits": splits,
            "note": note,
        },
        "text": text,
        "expected_topics": topics,
        "expected_topic_count": len(topics),
        "expected_domains": sorted({t["domain"] for t in topics}),
    }


def _split_of(records: list[dict], split: str) -> list[str]:
    return [split] * len(records)


def build() -> dict:
    held = held_out()
    pool = _Pool(held)
    split_of = held_out_splits()

    def split_key(record: dict) -> str:
        return normalize_text(record.get("content") or "")

    def sp(records: list[dict]) -> list[str]:
        return sorted({split_of.get(split_key(r), "?") for r in records})

    cases: list[dict] = []
    used_real: set[str] = set()
    n = 0

    def nid() -> str:
        nonlocal n
        n += 1
        return f"ev3-{n:03d}"

    def usable(r: dict) -> bool:
        return bool(ref_topic(r)["issue"].strip())

    # -- A: terse single (condensed) -------------------------------------- #
    terse_src = [r for r in pool.all() if usable(r) and len(condense(r)) < 150]
    for r in _slice(terse_src, 6, stride=11, skip=used_real):
        used_real.add(str(r.get("uid")))
        cases.append(_case(
            nid(), "A", condense(r), [ref_topic(r)],
            kind="condensed", sources=[str(r.get("uid"))], splits=sp([r]),
            note="held-out complaint reduced to a terse note by deleting boilerplate "
                 "sentences and wrapper words only",
        ))

    # A few cases pushed to a genuinely minimal note, so the suite's length
    # distribution covers the register the smoke failures live in rather than
    # only the mid-length bulk of the held-out pool.
    for r in _slice(terse_src, 4, stride=29, skip=used_real):
        used_real.add(str(r.get("uid")))
        cases.append(_case(
            nid(), "A", condense(r, max_len=110), [ref_topic(r)],
            kind="condensed", sources=[str(r.get("uid"))], splits=sp([r]),
            note="held-out complaint reduced to a minimal note (max 110 chars) by "
                 "deleting boilerplate sentences and wrapper words only",
        ))

    # -- B/C/D: two terse problems joined by a structural template -------- #
    # Components are real held-out complaints from *different* domains so the
    # expected topic count is unambiguously 2. Both components come from the
    # same split, so every case has exactly one provenance split.
    def two_terse_pairs(n_pairs: int) -> list[tuple[dict, dict]]:
        # Group by *domain*, not by source kind: two different kinds can map to
        # the same domain, which would make a "two problems" case collapse to one.
        per_split: dict[str, list[tuple[str, list[dict]]]] = {}
        for split in ("validation", "test"):
            by_dom: dict[str, list[dict]] = {}
            for r in held:
                if split_of.get(split_key(r)) != split:
                    continue
                dom = KIND_TO_DOMAIN.get(r.get("kind") or "")
                if dom:
                    by_dom.setdefault(dom, []).append(r)
            per_split[split] = sorted(by_dom.items())

        pairs: list[tuple[dict, dict]] = []
        used: set[str] = set()
        for split in ("validation", "test"):
            doms = [(d, rs) for d, rs in per_split[split] if rs]
            if len(doms) < 2:
                continue
            i = 0
            while len(pairs) < n_pairs and i < 4000:
                da, la = doms[i % len(doms)]
                db, lb = doms[(i + 1 + (i // len(doms)) // 2) % len(doms)]
                i += 1
                if da == db:
                    continue
                ra, rb = la[(i // 3) % len(la)], lb[(i // 5) % len(lb)]
                ua, ub = str(ra.get("uid")), str(rb.get("uid"))
                if ua == ub or ua in used or ub in used:
                    continue
                if not usable(ra) or not usable(rb):
                    continue
                used.update((ua, ub))
                pairs.append((ra, rb))
                if len(pairs) >= n_pairs:
                    break
        return pairs

    for ra, rb in two_terse_pairs(8):
        a, b = condense(ra), condense(rb)
        cases.append(_case(
            nid(), "B", f"{a}, і {b[0].lower() + b[1:]}.",
            [ref_topic(ra), ref_topic(rb)],
            kind="composed", sources=[str(ra.get("uid")), str(rb.get("uid"))], splits=sp([ra, rb]),
            note="two real held-out complaints condensed and joined with 'і'",
        ))
    for ra, rb in two_terse_pairs(8):
        a, b = condense(ra), condense(rb)
        cases.append(_case(
            nid(), "C", f"{a}, а також {b[0].lower() + b[1:]}.",
            [ref_topic(ra), ref_topic(rb)],
            kind="composed", sources=[str(ra.get("uid")), str(rb.get("uid"))], splits=sp([ra, rb]),
            note="two real held-out complaints condensed and joined with 'а також'",
        ))
    for ra, rb in two_terse_pairs(8):
        # D: full (un-condensed) issues in two sentences -- deliberately different
        # surface form from F, which uses condensed clauses.
        a, b = ref_topic(ra)["issue"], ref_topic(rb)["issue"]
        cases.append(_case(
            nid(), "D", f"{a} {b[0].lower() + b[1:]}.",
            [ref_topic(ra), ref_topic(rb)],
            kind="composed", sources=[str(ra.get("uid")), str(rb.get("uid"))], splits=sp([ra, rb]),
            note="two real held-out complaints stated as two full sentences (first "
                 "sentence boundary kept)",
        ))

    # -- E: one object, two problems -------------------------------------- #
    # Keyed by (split, street) so both complaints about a street come from the
    # same split, keeping one provenance split per case.
    by_street: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in pool.all():
        s = _street(r)
        if s:
            by_street[(split_of.get(split_key(r), "?"), s)].append(r)
    same_obj = 0
    for split_s, s in sorted(by_street):
        recs = [r for r in by_street[(split_s, s)] if usable(r)]
        pair = None
        for i, ra in enumerate(recs):
            for rb in recs[i + 1:]:
                if KIND_TO_DOMAIN.get(ra.get("kind") or "") != KIND_TO_DOMAIN.get(rb.get("kind") or ""):
                    pair = (ra, rb)
                    break
            if pair:
                break
        if not pair:
            continue
        ra, rb = pair
        cases.append(_case(
            nid(), "E", f"По {s}: {condense(ra)}, а також {condense(rb)[0].lower() + condense(rb)[1:]}.",
            [ref_topic(ra), ref_topic(rb)],
            kind="composed", sources=[str(ra.get("uid")), str(rb.get("uid"))], splits=sp([ra, rb]),
            note=f"two held-out complaints about the same street ({s})",
        ))
        same_obj += 1
        if same_obj >= 6:
            break

    # -- F: different objects, two problems ------------------------------- #
    for ra, rb in two_terse_pairs(8):
        a, b = condense(ra), condense(rb)
        cases.append(_case(
            nid(), "F", f"{a}. {b[0].upper() + b[1:]}.",
            [ref_topic(ra), ref_topic(rb)],
            kind="composed", sources=[str(ra.get("uid")), str(rb.get("uid"))], splits=sp([ra, rb]),
            note="two held-out complaints about different locations, stated in sequence",
        ))

    # -- G: one explicit request + one implicit problem ------------------- #
    g_done = 0
    for ra, rb in two_terse_pairs(24):
        act = _action_clause(rb)
        if len(act) < 8 or KIND_TO_DOMAIN.get(ra.get("kind") or "") == KIND_TO_DOMAIN.get(rb.get("kind") or ""):
            continue
        a, b = condense(ra), condense(rb)
        ta, tb = ref_topic(ra), ref_topic(rb)
        tb["requested_action"] = _clean_text(act)
        tb["issue"] = b[:400]
        cases.append(_case(
            nid(), "G", f"{a}. Прошу {act[0].lower() + act[1:]}.",
            [ta, tb],
            kind="composed", sources=[str(ra.get("uid")), str(rb.get("uid"))], splits=sp([ra, rb]),
            note="problem A stated without a request, problem B with an explicit 'Прошу ...'",
        ))
        g_done += 1
        if g_done >= 6:
            break

    # -- H: no explicit action -------------------------------------------- #
    h_src = [r for r in pool.all() if usable(r) and not ref_topic(r)["requested_action"].strip()]
    for r in _slice(h_src, 12, stride=13, skip=used_real):
        used_real.add(str(r.get("uid")))
        cases.append(_case(
            nid(), "H", (r.get("content") or "").strip(), [ref_topic(r)],
            kind="real", sources=[str(r.get("uid"))], splits=sp([r]),
            note="held-out complaint with no request verb: requested_action must be empty",
        ))

    # -- I: long rambling -------------------------------------------------- #
    long_src = [r for r in pool.all() if usable(r) and len((r.get("content") or "").strip()) > 600]
    for r in _slice(long_src, 8, stride=5, skip=used_real):
        used_real.add(str(r.get("uid")))
        cases.append(_case(
            nid(), "I", (r.get("content") or "").strip(), [ref_topic(r)],
            kind="real", sources=[str(r.get("uid"))], splits=sp([r]),
            note="long complaint with back-story, dates and repetition",
        ))

    # -- J: irrelevant prose ----------------------------------------------- #
    j_src = [r for r in pool.all() if usable(r) and _BOILER_RE.search(r.get("content") or "")]
    for r in _slice(j_src, 6, stride=17, skip=used_real):
        used_real.add(str(r.get("uid")))
        cases.append(_case(
            nid(), "J", (r.get("content") or "").strip(), [ref_topic(r)],
            kind="real", sources=[str(r.get("uid"))], splits=sp([r]),
            note="heavy consent / response-delivery boilerplate that must not reach the output",
        ))

    # -- K: ontology ambiguity --------------------------------------------- #
    # Pick a *distinct*, not-yet-used record per domain, and keep both components
    # in the same split, so the ambiguity is genuine rather than an artefact of
    # reusing one record across pairs.
    k_used: set[str] = set()

    def _pick_domain_uid(records: list[dict], exclude: set[str]) -> dict | None:
        for r in records:
            uid = str(r.get("uid"))
            if uid in exclude or uid in k_used or not usable(r):
                continue
            return r
        return None

    for da, db in AMBIGUOUS_PAIRS:
        la, lb = pool.domain(da), pool.domain(db)
        if not la or not lb:
            continue
        # keep both components in one split
        for split in ("validation", "test"):
            la_s = [r for r in la if split_of.get(split_key(r)) == split]
            lb_s = [r for r in lb if split_of.get(split_key(r)) == split]
            ra = _pick_domain_uid(la_s, set())
            rb = _pick_domain_uid(lb_s, {str(ra.get("uid"))} if ra else set())
            if ra and rb:
                break
        else:
            continue
        if not ra or not rb or str(ra.get("uid")) == str(rb.get("uid")):
            continue
        k_used.update((str(ra.get("uid")), str(rb.get("uid"))))
        a, b = condense(ra), condense(rb)
        cases.append(_case(
            nid(), "K", f"{a}, а також {b[0].lower() + b[1:]}.",
            [ref_topic(ra), ref_topic(rb)],
            kind="composed", sources=[str(ra.get("uid")), str(rb.get("uid"))], splits=sp([ra, rb]),
            note=f"conflatable coarse domains ({da} vs {db}): the pair a human reviewer "
                 f"would also struggle to separate",
        ))

    # -- L..V: topical coverage ------------------------------------------- #
    for code in ("L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V"):
        _, _, pattern = CATEGORIES[code]
        quota = {"M": 4, "P": 3, "V": 4, "N": 5, "L": 6, "O": 6, "Q": 6, "R": 6, "S": 7, "T": 5, "U": 6}[code]
        src = [r for r in pool.area(pattern) if usable(r)]
        for r in _slice(src, quota, stride=3, skip=used_real):
            used_real.add(str(r.get("uid")))
            cases.append(_case(
                nid(), code, (r.get("content") or "").strip(), [ref_topic(r)],
                kind="real", sources=[str(r.get("uid"))], splits=sp([r]),
                note=f"held-out complaint matching the {CATEGORIES[code][0]} area",
            ))

    return _finalise(cases)


def _finalise(cases: list[dict]) -> dict:
    """Deduplicate, sort, and compute the suite's own statistics."""
    seen_text: dict[str, str] = {}
    seen_ids: set[str] = set()
    kept: list[dict] = []
    for c in cases:
        key = normalize_text(c["text"])
        if not key or key in seen_text or c["id"] in seen_ids:
            continue
        if not c["text"].strip() or c["expected_topic_count"] < 1:
            continue
        for t in c["expected_topics"]:
            if not t["issue"].strip():
                raise ValueError(f"{c['id']}: empty expected issue")
        seen_text[key] = c["id"]
        seen_ids.add(c["id"])
        kept.append(c)
    kept.sort(key=lambda c: c["id"])

    by_cat = Counter(c["category"] for c in kept)
    by_prov = Counter(c["provenance"]["kind"] for c in kept)
    # Counted per case, not per source: composition pairs records from the same
    # split, so every case has exactly one provenance split and the parts sum to
    # the case count.
    by_split = Counter(c["provenance"]["splits"][0] for c in kept)
    spanning = [c["id"] for c in kept if len(c["provenance"]["splits"]) != 1]
    n_topics = Counter(c["expected_topic_count"] for c in kept)
    doms = Counter(d for c in kept for d in c["expected_domains"])
    lens = sorted(len(c["text"]) for c in kept)

    stats = {
        "built_by": "ml/tune/build_eval_v3.py",
        "reference_labeler": "contract labeler (build_dataset) with _ADMIN_CLAUSE "
                             "gap closed by BOILER_SENT; see module docstring",
        "n_cases": len(kept),
        "by_category": dict(sorted(by_cat.items())),
        "by_provenance": dict(sorted(by_prov.items())),
        "by_split": dict(sorted(by_split.items())),
        "cases_spanning_splits": spanning,
        "expected_topic_count_dist": dict(sorted(n_topics.items())),
        "expected_domain_counts": dict(doms.most_common()),
        "text_length": {
            "min": lens[0], "median": lens[len(lens) // 2],
            "max": lens[-1],
            "p25": lens[len(lens) // 4], "p75": lens[3 * len(lens) // 4],
        },
        "comparability": (
            "NOT directly comparable with ml/data/tune/eval (frozen 329) or "
            "multitopic/eval (85): those score against the v1 weak label, which "
            "retains administrative boilerplate in 53-64% of issue fields, while "
            "this suite scores against the cleaned reference label."
        ),
        "training_exclusion": (
            "No case text is drawn from the train split. 'real' cases are verbatim "
            "validation/test records. 'condensed' and 'composed' cases are derived "
            "from those same held-out records. All texts are checked against every "
            "training text (train split, multitopic/train.jsonl, v2/train.jsonl) by "
            "ml/tests/test_eval_v3_suite.py::test_no_case_text_leaks_into_training."
        ),
    }
    return {"cases": kept, "stats": stats}


# --------------------------------------------------------------------------- #
# Case-record JSON Schema
# --------------------------------------------------------------------------- #

CASE_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "$id": "ukrainian-municipal-accountability/eval-v3-case",
    "title": "One case of the v3 adversarial evaluation suite",
    "type": "object",
    "required": [
        "id", "category", "category_name", "provenance", "text",
        "expected_topics", "expected_topic_count", "expected_domains",
    ],
    "additionalProperties": False,
    "properties": {
        "id": {"type": "string", "pattern": r"^ev3-\d{3}$"},
        "category": {"type": "string", "pattern": "^[A-V]$"},
        "category_name": {"type": "string", "minLength": 1},
        "expected_topic_count": {"type": "integer", "enum": [1, 2]},
        "provenance": {
            "type": "object",
            "required": ["kind", "sources", "splits", "note"],
            "additionalProperties": False,
            "properties": {
                "kind": {"enum": ["real", "condensed", "composed"]},
                "sources": {
                    "type": "array", "minItems": 1,
                    "items": {"type": "string", "minLength": 1},
                },
                "splits": {
                    "type": "array", "minItems": 1,
                    "items": {"enum": ["validation", "test"]},
                },
                "note": {"type": "string", "minLength": 1},
            },
        },
        "text": {"type": "string", "minLength": 5},
        "expected_topics": {
            "type": "array", "minItems": 1,
            "items": {
                "type": "object",
                "required": ["domain", "issue", "object", "requested_action", "attributes"],
                "additionalProperties": False,
                "properties": {
                    "domain": {"type": "string", "minLength": 1},
                    "issue": {"type": "string", "minLength": 1},
                    "object": {"type": "string"},
                    "requested_action": {"type": "string"},
                    "attributes": {"type": "object"},
                },
            },
        },
        "expected_topic_count": {"type": "integer", "minimum": 1, "maximum": 2},
        "expected_domains": {
            "type": "array", "minItems": 1,
            "items": {"type": "string", "minLength": 1},
        },
    },
}


def write(cases: list[dict], stats: dict) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CASES_PATH.write_text(
        "\n".join(json.dumps(c, ensure_ascii=False) for c in cases) + "\n"
    )
    SCHEMA_PATH.write_text(json.dumps(CASE_SCHEMA, ensure_ascii=False, indent=2) + "\n")
    STATS_PATH.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n")


def main() -> None:
    result = build()
    write(result["cases"], result["stats"])
    print(f"wrote {len(result['cases'])} cases -> {CASES_PATH}")
    print(json.dumps(result["stats"]["by_category"], ensure_ascii=False))
    print(json.dumps(result["stats"]["by_provenance"], ensure_ascii=False))
    print(json.dumps(result["stats"]["expected_topic_count_dist"], ensure_ascii=False))
    _ = SYSTEM_PROMPT  # prompt contract is shared with training; kept for reference


if __name__ == "__main__":
    main()
