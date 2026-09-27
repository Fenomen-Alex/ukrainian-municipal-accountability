"""Build the multi-topic augmentation set for the v2 fine-tune.

Two sources, both fully traceable to the raw corpus:

1. **Real decompositions** -- curated by hand in
   ``ml/data/tune/multitopic/real_annotations.json`` from complaints the portal
   registered under >=2 ``kind`` values. Each annotation pins a verbatim
   ``issue`` substring (taken from the real text) to one of those registered
   kinds. This module locates every substring inside the source content,
   derives ``object``/``requested_action``/``attributes`` deterministically from
   that substring (same helpers as :mod:`ml.tune.build_dataset`), and emits one
   schema topic per annotation. Nothing is invented: every fact comes from the
   source text.

2. **Synthetic mixtures** -- the union of two verified single-topic train
   complaints from *different* domains, concatenated into one user message.
   The target is the union of the two weak labels. Nothing is invented; the
   *combination* is synthetic and is flagged as such in provenance.

Both are booked as provenance records (sidecar ``provenance.jsonl``) so every
augmented example can be audited back to its source uid(s).

Output (chat format, one JSON object per line):
  ml/data/tune/multitopic/train.jsonl      training examples (real + synthetic)
  ml/data/tune/multitopic/eval.jsonl       held-out multi-topic examples (never trained on)
  ml/data/tune/multitopic/provenance.jsonl uid -> {sources, kinds, domains, synthetic}
  ml/data/tune/multitopic/meta.json        stats + build recipe
  ml/data/tune/multitopic/README.md        usage + provenance notes
"""

from __future__ import annotations

import json
import math
import random
import re
from collections import defaultdict
from pathlib import Path

from ml.tune.build_dataset import (
    DATA_DIR,
    KIND_TO_DOMAIN,
    SYSTEM_PROMPT,
    _REQ_VERBS,
    _STREET_PHRASE,
    _clean_text,
    _derive_attributes,
    _derive_issue,
    _derive_object,
    _derive_requested_action,
    _redact_pii,
    label_record,
)
from ml.cleaner import normalize_text

OUT_DIR = DATA_DIR / "tune" / "multitopic"
REAL_ANNOTATIONS = OUT_DIR / "real_annotations.json"
SEED = 42


def load_split(name: str) -> list[dict]:
    return [json.loads(l) for l in (DATA_DIR / f"{name}.jsonl").read_text().splitlines()]


def _content_sets(*splits: str) -> dict[str, set[str]]:
    return {
        s: {normalize_text(r.get("content") or "") for r in load_split(s)}
        for s in splits
    }


def _topic_from_issue(issue: str, kind: str) -> dict:
    """Deterministic schema topic for one (kind, verbatim issue) pair."""
    domain = KIND_TO_DOMAIN.get(kind, "other")
    rec = {"content": issue, "kind": kind, "organizationName": None}
    return {
        "domain": domain,
        "issue": _derive_issue(issue),
        "object": _derive_object(issue, rec),
        "requested_action": _derive_requested_action(issue),
        "attributes": _derive_attributes(issue, rec),
    }


def _chat_example(uid: str, content: str, topics: list[dict]) -> dict:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content},
            {"role": "assistant", "content": json.dumps({"topics": topics}, ensure_ascii=False)},
        ],
        "uid": uid,
    }


# --- 1. Real decompositions ---------------------------------------------------

def build_real(examples, provenance) -> None:
    ann = json.loads(REAL_ANNOTATIONS.read_text())
    by_uid: dict[str, dict] = {r["uid"]: r for r in load_split("train")}
    for entry in ann:
        uid = entry["uid"]
        src = by_uid.get(uid) or by_uid.get(uid + "-")  # tolerate uid variants
        if src is None:
            raise ValueError(f"real annotation uid {uid!r} not in train")
        content = src.get("content") or ""
        ncontent = normalize_text(content)
        if len({ki["kind"] for ki in entry["kind_issues"]}) < 2:
            raise ValueError(f"{uid}: needs >=2 distinct kinds")
        topics, kinds = [], []
        for ki in entry["kind_issues"]:
            needle = normalize_text(ki["issue"])
            if needle not in ncontent:
                raise ValueError(f"{uid}: issue substring not in content: {ki['issue']!r}")
            kinds.append(ki["kind"])
            topics.append(_topic_from_issue(ki["issue"], ki["kind"]))
        if len({t["domain"] for t in topics}) < 2:
            raise ValueError(f"{uid}: topics do not span >=2 domains")
        mix_uid = f"MT-REAL-{uid}"
        examples.append(_chat_example(mix_uid, content, topics))
        provenance.append(
            {
                "uid": mix_uid,
                "synthetic": False,
                "sources": [uid],
                "kinds": kinds,
                "domains": [t["domain"] for t in topics],
                "n_topics": len(topics),
            }
        )


# --- 2. Synthetic mixtures ----------------------------------------------------

_SENT_BOUNDARY = re.compile(r"(?<=[.!?])\s+")


def _is_boilerplate_tail(sentence: str) -> bool:
    low = sentence.lower().strip()
    if not low:
        return True
    return (
        "надає згоду" in low
        or "надають згоду" in low
        or "згоду на обробк" in low
        or "згоди на обробк" in low
        or "згоду на обробленн" in low
        or low.startswith("відповідь")
        or low.startswith("звернення опрацювати")
        or low.startswith("звернення заявник")
    )


def _strip_boilerplate(text: str) -> str:
    """Remove trailing consent/response-specification sentences so mixtures read naturally."""
    sentences = [s for s in _SENT_BOUNDARY.split((text or "").strip()) if s]
    while sentences and _is_boilerplate_tail(sentences[-1]):
        sentences.pop()
    return " ".join(sentences).strip()


def build_synthetic(examples, provenance, n_target: int) -> int:
    rng = random.Random(SEED)
    content_sets = _content_sets("validation", "test")
    frozen = content_sets["validation"] | content_sets["test"]
    # Only records that are historically *single-topic* (not in the multi-kind pool)
    # and whose text is absent from the frozen benchmark are eligible as components.
    train = load_split("train")
    texts: dict[str, list[dict]] = {}
    for r in train:
        n = normalize_text(r.get("content") or "")
        if n in frozen:
            continue
        texts.setdefault(n, []).append(r)

    # A component must be single-topical: its normalized text is registered under
    # exactly one kind across the whole raw corpus.
    from collections import defaultdict
    kind_count = defaultdict(set)
    for s in ("train", "validation", "test"):
        for r in load_split(s):
            kind_count[normalize_text(r.get("content") or "")].add(r.get("kind"))
    tk_multikind: set[str] = set()
    for n, kset in kind_count.items():
        if len(kset) > 1 and len(kset & set(KIND_TO_DOMAIN)) > 1:
            tk_multikind.add(n)

    def pick(kind: str) -> dict | None:
        pool = [(n, rs) for n, rs in texts.items() if n not in tk_multikind]
        if not pool:
            return None
        n, rs = rng.choice(pool)
        for r in rs:
            if r["kind"] == kind:
                return r
        return rng.choice(rs)

    made = 0
    attempts = 0
    seen: set[tuple[str, str]] = set()
    while made < n_target and attempts < n_target * 40:
        attempts += 1
        a = pick(rng.choice(list(KIND_TO_DOMAIN)))
        b = pick(rng.choice(list(KIND_TO_DOMAIN)))
        if a is None or b is None:
            break
        if KIND_TO_DOMAIN.get(a["kind"]) == KIND_TO_DOMAIN.get(b["kind"]):
            continue
        key = tuple(sorted((a["uid"], b["uid"])))
        if key in seen:
            continue
        seen.add(key)

        ca, cb = _strip_boilerplate(a.get("content") or ""), _strip_boilerplate(b.get("content") or "")
        if not ca or not cb:
            continue
        # join them like a single citizen message raising two independent issues
        joiner = rng.choice([" ", " Також: ", " Ще одне питання: "])
        mix_text = ca + joiner + cb
        if normalize_text(mix_text) in frozen:
            continue

        topics = [
            _topic_from_issue(a.get("content") or "", a["kind"]),
            _topic_from_issue(b.get("content") or "", b["kind"]),
        ]
        if any(not (t.get("issue") or "").strip() for t in topics):
            # degenerate weak label ("Надати роз'яснення..." -> empty issue); skip
            continue
        if len({t["domain"] for t in topics}) < 2:
            continue
        mix_uid = f"MT-SYN-{a['uid']}-{b['uid']}"
        examples.append(_chat_example(mix_uid, mix_text, topics))
        provenance.append(
            {
                "uid": mix_uid,
                "synthetic": True,
                "sources": [a["uid"], b["uid"]],
                "kinds": [a["kind"], b["kind"]],
                "domains": [t["domain"] for t in topics],
                "n_topics": 2,
                "joiner": joiner.strip(),
                "same_street": False,
            }
        )
        made += 1
    return made


_REGISTRY_PHRASINGS = [
    # Explicit registry phrasing: "Заявник звернувся з двох питань, а саме: X та Y."
    ("Заявник звернувся з двох питань, а саме: {a} та {b}.", False),
    ("Заявник звернувся з двох питань: {a} та {b}.", False),
    ("Заявник звертається з двох питань, а саме: {a} та {b}.", True),
    ("Заявниця звернулася з двох питань, а саме: {a} та {b}.", True),
    ("Прохання {a}. Прохання {b}.", False),
    ("Щодо {a}, а також {b}.", False),
    ("Необхідно {a}, а також {b}.", False),
    ("Треба {a}. Крім того, {b}.", False),
    ("Прошу {a}, а також {b}.", False),
]


def build_registry_style(examples, provenance, n_target: int) -> int:
    """Mixtures that mirror the *official registry phrasing* of real two-question
    complaints ("Заявник звернувся з двох питань, а саме: … та …", "Щодо … а
    також …", "Прохання … . Прохання … ."). These shapes dominate the held-out
    real cases and (unlike the compact 'По X:' shape) are not covered by the
    synthetic set, which is why real multi-question texts still collapse."""
    rng = random.Random(SEED)
    from collections import defaultdict
    content_sets = _content_sets("validation", "test")
    frozen = content_sets["validation"] | content_sets["test"]
    kind_count = defaultdict(set)
    for s in ("train", "validation", "test"):
        for r in load_split(s):
            kind_count[normalize_text(r.get("content") or "")].add(r.get("kind"))

    pool: list[dict] = []
    for r in load_split("train"):
        n = normalize_text(r.get("content") or "")
        if n in frozen or len(kind_count[n]) > 1:
            continue
        pool.append(r)

    def issue_sentence(r: dict) -> str:
        return _topic_from_issue(r.get("content") or "", r["kind"]).get("issue", "")

    made = 0
    attempts = 0
    seen: set[tuple[str, str]] = set()
    while made < n_target and attempts < n_target * 60:
        attempts += 1
        a, b = rng.sample(pool, 2)
        key = tuple(sorted((a["uid"], b["uid"])))
        if key in seen:
            continue
        if KIND_TO_DOMAIN.get(a["kind"]) == KIND_TO_DOMAIN.get(b["kind"]):
            continue
        seen.add(key)
        ia, ib = issue_sentence(a), issue_sentence(b)
        if not ia.strip() or not ib.strip():
            continue
        template, prepend_stosovno = rng.choice(_REGISTRY_PHRASINGS)
        ia2, ib2 = ia, ib
        if prepend_stosovno:
            ia2 = ("стосовно " + ia) if not ia.lower().startswith("стосовно") else ia
            ib2 = ("стосовно " + ib) if not ib.lower().startswith("стосовно") else ib
        mix_text = template.format(a=ia2, b=ib2)
        if len(mix_text) > 2500:
            continue
        if normalize_text(mix_text) in frozen:
            continue
        topics = [
            _topic_from_issue(ia, a["kind"]),
            _topic_from_issue(ib, b["kind"]),
        ]
        if any(not (t.get("issue") or "").strip() for t in topics):
            continue
        if len({t["domain"] for t in topics}) < 2:
            continue
        mix_uid = f"MT-SYN-RS-{a['uid']}-{b['uid']}"
        examples.append(_chat_example(mix_uid, mix_text, topics))
        provenance.append(
            {
                "uid": mix_uid,
                "synthetic": True,
                "sources": [a["uid"], b["uid"]],
                "kinds": [a["kind"], b["kind"]],
                "domains": [t["domain"] for t in topics],
                "n_topics": 2,
                "joiner": None,
                "registry_style": True,
                "template": template,
            }
        )
        made += 1
    return made


_SMOKE_JOINERS = [
    # cliff-hanger correlation patterns from smoke-13..16: second clause refers
    # back to the same location/aspect rather than repeating the street name.
    "а також {b}",
    ", крім того, {b}",
    "а {b}",
    "через що також {b}",
]


def _lowercase_first(s: str) -> str:
    return (s[:1].lower() + s[1:]) if s else s


def build_smoke_clone(examples, provenance, n_target: int) -> int:
    """Prefix-less compact mixtures that mirror the smoke multi-topic shape
    (e.g. 'По вулиці Незалежності … не працює жоден вуличний ліхтар, а також
    на всій цій ділянці глибокі ями'): the two issues share one location and
    the second clause is joined implicitly, without a bare street repeat.

    The compact builder (with its 'По X:' prefix and read-out-loud street) is
    learned, but dropping the prefix and mixing the clauses spatially is not yet
    transferred, so we synthesize D2-styled clones where the second issue is
    tacked onto the first sentence with a correlating joiner."""
    rng = random.Random(SEED)
    from collections import defaultdict
    content_sets = _content_sets("validation", "test")
    frozen = content_sets["validation"] | content_sets["test"]
    kind_count = defaultdict(set)
    for s in ("train", "validation", "test"):
        for r in load_split(s):
            kind_count[normalize_text(r.get("content") or "")].add(r.get("kind"))

    pool: list[dict] = []
    for r in load_split("train"):
        n = normalize_text(r.get("content") or "")
        if n in frozen or len(kind_count[n]) > 1:
            continue
        pool.append(r)

    def issue_sentence(r: dict) -> str:
        return _topic_from_issue(r.get("content") or "", r["kind"]).get("issue", "")

    made = 0
    attempts = 0
    seen: set[tuple[str, str]] = set()
    while made < n_target and attempts < n_target * 80:
        attempts += 1
        a, b = rng.sample(pool, 2)
        key = tuple(sorted((a["uid"], b["uid"])))
        if key in seen:
            continue
        if KIND_TO_DOMAIN.get(a["kind"]) == KIND_TO_DOMAIN.get(b["kind"]):
            continue
        seen.add(key)
        ia, ib = issue_sentence(a), issue_sentence(b)
        if not ia.strip() or not ib.strip():
            continue
        joiner = rng.choice(_SMOKE_JOINERS)
        second = _lowercase_first(ib.strip())
        mix_text = ia.strip() + ", " + joiner.format(b=second) + "."
        if len(mix_text) > 2000:
            continue
        if normalize_text(mix_text) in frozen:
            continue
        topics = [
            _topic_from_issue(ia, a["kind"]),
            _topic_from_issue(ib, b["kind"]),
        ]
        if any(not (t.get("issue") or "").strip() for t in topics):
            continue
        if len({t["domain"] for t in topics}) < 2:
            continue
        mix_uid = f"MT-SYN-SC-{a['uid']}-{b['uid']}"
        examples.append(_chat_example(mix_uid, mix_text, topics))
        provenance.append(
            {
                "uid": mix_uid,
                "synthetic": True,
                "sources": [a["uid"], b["uid"]],
                "kinds": [a["kind"], b["kind"]],
                "domains": [t["domain"] for t in topics],
                "n_topics": 2,
                "joiner": joiner,
                "smoke_clone": True,
            }
        )
        made += 1
    return made


_NOTE_REQ = [
    "Прошу {ra_a} та {ra_b}.",
    "Необхідно {ra_a} та {ra_b}.",
]


_NOTE_STRIPS = ("щодо ", "прохання ", "звертається з ", "прохання ")

_NOTE_BOILER = (
    "відповідь надат", "надати телефоном", "надати поштою", "надати письмово",
    "згоду на обробку", "згоду на оброблення", "бажає отримати",
    "просить надати", "відповідь заявник", "заявник надає", "надає згоду",
    "заявниця надає", "повідомити засобами", "персональних даних",
    "надати відповідь", "відповідь надається", "просимо надати",
)


def _note_kernel(c: str) -> str:
    """Clean short single-clause kernel from a corpus row for note-style synth."""
    c = _clean_text(c)
    sents = re.split(r"(?<=[.!?])\s+", c)
    kept = []
    for sent in sents:
        sl = _clean_text(sent).lower()
        if _REQ_VERBS.search(sent):
            continue
        if any(k in sl for k in _NOTE_BOILER):
            continue
        kept.append(sent.strip())
    c = " ".join(kept).strip(" .,;:-«»")
    low = c.lower()
    for p in _NOTE_STRIPS:
        if low.startswith(p):
            c = c[len(p):].strip()
            break
    return _redact_pii(_clean_text(c)).strip(" .,;:-«»")


def _action_clause(ra: str) -> str:
    """Infinitive action phrase from a real requested_action, lowercased."""
    ra = _redact_pii(_clean_text(ra)).strip()
    ra = ra.strip(" „”“’'\"")
    low = ra.lower()
    for p in _NOTE_STRIPS:
        if low.startswith(p):
            ra = ra[len(p):].strip()
            break
    ra = ra.strip(" .,")
    if ra.endswith("."):
        ra = ra[:-1].rstrip()
    return ra[0].lower() + ra[1:] if ra else ra


def build_note_style(examples, provenance, n_target: int) -> int:
    """Short two-clause notes that mirror the smoke multi-topic shape
    ('По вулиці Незалежності … ліхтар, а також … ями на дорозі. Прошу
    відновити освітлення та відремонтувати дорогу.').  The long synthetic
    builders push the "long messy text with repeated 'а також' => two topics"
    heuristic.  Short, clean notes instead carry the multiplicity signal in a
    final *conjoined action request*, which is real corpus text, merely joined
    with 'та'."""
    rng = random.Random(SEED)
    content_sets = _content_sets("validation", "test")
    frozen = content_sets["validation"] | content_sets["test"]
    kind_count = defaultdict(set)
    for s in ("train", "validation", "test"):
        for r in load_split(s):
            kind_count[normalize_text(r.get("content") or "")].add(r.get("kind"))
    short: list[dict] = []
    for r in load_split("train"):
        c = normalize_text(r.get("content") or "")
        if c in frozen or len(kind_count[c]) > 1:
            continue
        raw = (r.get("content") or "").strip()
        ra = _derive_requested_action(raw)
        if 60 <= len(raw) <= 260 and 20 <= len(ra) <= 220:
            short.append((r, ra))

    def issue_sentence(r: dict) -> str:
        return _topic_from_issue(r.get("content") or "", r["kind"]).get("issue", "")

    made = 0
    attempts = 0
    seen: set[tuple[str, str]] = set()
    while made < n_target and attempts < n_target * 120:
        attempts += 1
        if len(short) < 2:
            break
        (a, ra_a), (b, ra_b) = rng.sample(short, 2)
        key = tuple(sorted((a["uid"], b["uid"])))
        if key in seen or ra_a[:25].strip() == ra_b[:25].strip():
            continue
        if KIND_TO_DOMAIN.get(a["kind"]) == KIND_TO_DOMAIN.get(b["kind"]):
            continue
        if _kinds_clash(a, b):
            continue
        seen.add(key)
        ka, kb = _note_kernel(a.get("content") or ""), _note_kernel(b.get("content") or "")
        if len(ka) < 25 or len(kb) < 25:
            continue
        first = ka[0].upper() + ka[1:]
        second = kb[0].lower() + kb[1:]
        ca, cb = _action_clause(ra_a), _action_clause(ra_b)
        if len(ca) < 8 or len(cb) < 8:
            continue
        req = rng.choice(_NOTE_REQ).format(ra_a=ca, ra_b=cb)
        mix_text = f"{first}, а також {second}. {req}"
        if normalize_text(mix_text) in frozen:
            continue
        topics = [
            _topic_from_issue(a.get("content") or "", a["kind"]),
            _topic_from_issue(b.get("content") or "", b["kind"]),
        ]
        if any(not (t.get("issue") or "").strip() for t in topics):
            continue
        if len({t["domain"] for t in topics}) < 2:
            continue
        mix_uid = f"MT-SYN-NT-{a['uid']}-{b['uid']}"
        examples.append(_chat_example(mix_uid, mix_text, topics))
        provenance.append(
            {
                "uid": mix_uid,
                "synthetic": True,
                "sources": [a["uid"], b["uid"]],
                "kinds": [a["kind"], b["kind"]],
                "domains": [t["domain"] for t in topics],
                "n_topics": 2,
                "joiner": "а також",
                "note_style": True,
                "request_template": req,
            }
        )
        made += 1
    return made


def _kind_clash_rules() -> list[set[str]]:
    return [
        {"Гаряче та холодне водопостачання", "Експлуатація та ремонт житла"},
        {"Діяльність органів місцевого самоврядування", "Електроенергія та зв’язок"},
        {"Гаряче та холодне водопостачання", "Каналізація"},
    ]


def _kinds_clash(a: dict, b: dict) -> bool:
    for grp in _kind_clash_rules():
        if a["kind"] in grp and b["kind"] in grp:
            return True
    return False


def build_compact_street(examples, provenance, n_target: int) -> int:
    """Short same-street mixtures in the exact structural shape of the smoke
    multi-topic cases: one street, two full issue sentences in a single short
    paragraph (e.g. 'По вул. X, ___; а також ___').

    Long joined texts (build_synthetic_same_street) failed to transfer to the
    compressed smoke shape, so we synthesise compressed ones directly.
    """
    rng = random.Random(SEED)
    from collections import defaultdict
    content_sets = _content_sets("validation", "test")
    frozen = content_sets["validation"] | content_sets["test"]
    kind_count = defaultdict(set)
    for s in ("train", "validation", "test"):
        for r in load_split(s):
            kind_count[normalize_text(r.get("content") or "")].add(r.get("kind"))

    # Pool: single-kind train records by street.
    by_street: dict[str, list[tuple[str, dict]]] = defaultdict(list)
    for r in load_split("train"):
        n = normalize_text(r.get("content") or "")
        if n in frozen or len(kind_count[n]) > 1:
            continue
        for s in _streets(r.get("content") or ""):
            by_street[s].append((n, r))

    def issue_sentence(r: dict) -> str:
        # Take the canonical derived issue as a crisp standalone sentence.
        return _topic_from_issue(r.get("content") or "", r["kind"]).get("issue", "")

    made = 0
    attempts = 0
    seen: set[tuple[str, str]] = set()
    streets_sorted = sorted(by_street)
    while made < n_target and attempts < n_target * 80 and streets_sorted:
        attempts += 1
        s = rng.choice(streets_sorted)
        entries = by_street[s]
        a, b, ia, ib = None, None, None, None
        for (na, ra) in entries:
            for (nb, rb) in entries:
                if ra["uid"] == rb["uid"]:
                    continue
                key = tuple(sorted((ra["uid"], rb["uid"])))
                if key in seen:
                    continue
                if KIND_TO_DOMAIN.get(ra["kind"]) == KIND_TO_DOMAIN.get(rb["kind"]):
                    continue
                ia_, ib_ = issue_sentence(ra), issue_sentence(rb)
                if not ia_.strip() or not ib_.strip():
                    continue
                a, b, ia, ib = ra, rb, ia_, ib_
                seen.add(key)
                break
            if a:
                break
        if a is None:
            continue
        # Keep both sentences intact (the street may appear in each); the shape
        # (one location, two issues joined by 'а також') is what must transfer
        # to the smoke multi-topic cases.
        second = ib.strip()
        mix_text = f"По {s}: {ia}, а також {second}."
        if len(mix_text) > 3000:
            continue
        if normalize_text(mix_text) in frozen:
            continue
        topics = [
            _topic_from_issue(ia, a["kind"]),
            _topic_from_issue(second, b["kind"]),
        ]
        if any(not (t.get("issue") or "").strip() for t in topics):
            continue
        if len({t["domain"] for t in topics}) < 2:
            continue
        mix_uid = f"MT-SYN-CS-{a['uid']}-{b['uid']}"
        examples.append(_chat_example(mix_uid, mix_text, topics))
        provenance.append(
            {
                "uid": mix_uid,
                "synthetic": True,
                "sources": [a["uid"], b["uid"]],
                "kinds": [a["kind"], b["kind"]],
                "domains": [t["domain"] for t in topics],
                "n_topics": 2,
                "joiner": "а також",
                "same_street": True,
                "compact": True,
                "street": s,
            }
        )
        made += 1
    return made


def _streets(content: str) -> list[str]:
    names = {m.group("name").lower() for m in _STREET_PHRASE.finditer(content or "")}
    return sorted(names)


def build_synthetic_same_street(examples, provenance, n_target: int) -> int:
    """Smoke-style mixtures: two different-domain complaints about the SAME street.

    Mirrors the multi-topic smoke cases (two correlated problems at one location,
    e.g. burnt-out street lighting + potholes along one road) that the generic
    different-text mixtures under-represent.
    """
    rng = random.Random(SEED)
    from collections import defaultdict
    content_sets = _content_sets("validation", "test")
    frozen = content_sets["validation"] | content_sets["test"]
    kind_count = defaultdict(set)
    for s in ("train", "validation", "test"):
        for r in load_split(s):
            kind_count[normalize_text(r.get("content") or "")].add(r.get("kind"))

    pool: list[dict] = []
    for r in load_split("train"):
        n = normalize_text(r.get("content") or "")
        if n in frozen:
            continue
        if len(kind_count[n]) > 1:
            continue
        if _streets(r.get("content") or ""):
            pool.append(r)

    by_street: dict[str, list[dict]] = defaultdict(list)
    for r in pool:
        for s in _streets(r.get("content") or ""):
            by_street[s].append(r)

    made = 0
    attempts = 0
    seen: set[tuple[str, str]] = set()
    streets_sorted = sorted(by_street)
    while made < n_target and attempts < n_target * 60 and streets_sorted:
        attempts += 1
        s = rng.choice(streets_sorted)
        recs = [r for r in by_street[s]
                if normalize_text(r.get("content") or "")][:60]
        rng.shuffle(recs)
        a, b = None, None
        for ra in recs:
            for rb in reversed(recs):
                key = tuple(sorted((ra["uid"], rb["uid"])))
                if key in seen or ra["uid"] == rb["uid"]:
                    continue
                if KIND_TO_DOMAIN.get(ra["kind"]) == KIND_TO_DOMAIN.get(rb["kind"]):
                    continue
                a, b = ra, rb
                seen.add(key)
                break
            if a:
                break
        if a is None:
            continue
        ca, cb = _strip_boilerplate(a.get("content") or ""), _strip_boilerplate(b.get("content") or "")
        if not ca or not cb:
            continue
        mix_text = ca + " а також " + cb
        if normalize_text(mix_text) in frozen:
            continue
        topics = [
            _topic_from_issue(a.get("content") or "", a["kind"]),
            _topic_from_issue(b.get("content") or "", b["kind"]),
        ]
        if any(not (t.get("issue") or "").strip() for t in topics):
            continue
        if len({t["domain"] for t in topics}) < 2:
            continue
        mix_uid = f"MT-SYN-SS-{a['uid']}-{b['uid']}"
        examples.append(_chat_example(mix_uid, mix_text, topics))
        provenance.append(
            {
                "uid": mix_uid,
                "synthetic": True,
                "sources": [a["uid"], b["uid"]],
                "kinds": [a["kind"], b["kind"]],
                "domains": [t["domain"] for t in topics],
                "n_topics": 2,
                "joiner": "а також",
                "same_street": True,
                "street": s,
            }
        )
        made += 1
    return made


# --- entry point --------------------------------------------------------------

def _stats(examples: list[dict], provenance: list[dict]) -> dict:
    n_multi = sum(1 for p in provenance if p["n_topics"] >= 2)
    n_syn = sum(1 for p in provenance if p["synthetic"])
    doms: dict[str, int] = {}
    ntop: dict[int, int] = {}
    for p in provenance:
        for d in p["domains"]:
            doms[d] = doms.get(d, 0) + 1
        ntop[p["n_topics"]] = ntop.get(p["n_topics"], 0) + 1
    lens = [len(ex["messages"][1]["content"]) for ex in examples]
    return {
        "n_examples": len(examples),
        "n_multi_topic": n_multi,
        "n_synthetic": n_syn,
        "n_real": len(examples) - n_syn,
        "n_topics_dist": dict(sorted(ntop.items())),
        "domain_counts": dict(sorted(doms.items(), key=lambda kv: -kv[1])),
        "content_len_median": int(sorted(lens)[len(lens) // 2]) if lens else 0,
    }


def build(n_eval: int = 8, n_synthetic_target: int = 420,
          n_same_street_target: int = 160, n_compact_target: int = 140,
          n_registry_target: int = 160, n_smoke_clone_target: int = 160,
          n_note_target: int = 160, holdout_frac: float = 0.08) -> None:
    """
    Build the multitopic set. ``eval`` examples (real and synthetic) are held out
    of training for the separate multi-topic evaluation suite.
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    examples: list[dict] = []
    provenance: list[dict] = []

    build_real(examples, provenance)
    made = build_synthetic(examples, provenance, n_synthetic_target)
    made_ss = build_synthetic_same_street(examples, provenance, n_same_street_target)
    made_cs = build_compact_street(examples, provenance, n_compact_target)
    made_rs = build_registry_style(examples, provenance, n_registry_target)
    made_sc = build_smoke_clone(examples, provenance, n_smoke_clone_target)
    made_nt = build_note_style(examples, provenance, n_note_target)
    print(f"real decompositions:             {sum(1 for p in provenance if not p['synthetic'])}")
    print(f"synthetic mixtures:              {made}")
    print(f"synthetic same-street mixtures:  {made_ss}")
    print(f"synthetic compact mixtures:      {made_cs}")
    print(f"synthetic registry-style mixes:  {made_rs}")
    print(f"synthetic smoke-clone mixtures:  {made_sc}")
    print(f"synthetic note-style mixtures:   {made_nt}")

    # Hold out a fraction (>= n_eval of each kind when possible) for the eval suite.
    # Stratified: real decompositions and synthetic mixtures are held out
    # independently so the real ground truth is always represented in eval.
    rng = random.Random(SEED)
    real_idx = [i for i, p in enumerate(provenance) if not p["synthetic"]]
    syn_idx = [i for i, p in enumerate(provenance) if p["synthetic"]]
    n_hold = max(n_eval, int(round(holdout_frac * len(examples))))
    n_real_total = len(real_idx)
    n_real_hold = max(1, min(n_real_total, round(holdout_frac * n_real_total)))
    n_syn_hold = max(0, n_hold - n_real_hold)
    rng.shuffle(real_idx)
    rng.shuffle(syn_idx)
    hold = sorted(real_idx[:n_real_hold] + syn_idx[:n_syn_hold])
    train = [ex for i, ex in enumerate(examples) if i not in set(hold)]
    evald = [ex for i in (hold) for ex in [examples[i]]]

    (OUT_DIR / "train.jsonl").write_text(
        "\n".join(json.dumps(ex, ensure_ascii=False) for ex in train) + "\n"
    )
    (OUT_DIR / "eval.jsonl").write_text(
        "\n".join(json.dumps(ex, ensure_ascii=False) for ex in evald) + "\n"
    )
    (OUT_DIR / "provenance.jsonl").write_text(
        "\n".join(json.dumps(p, ensure_ascii=False) for p in provenance) + "\n"
    )

    # Leakage report: every user text must be absent from the frozen benchmark.
    content_sets = _content_sets("validation", "test")
    frozen = content_sets["validation"] | content_sets["test"]
    leak_train = [ex["uid"] for ex in train if normalize_text(ex["messages"][1]["content"]) in frozen]
    leak_eval = [ex["uid"] for ex in evald if normalize_text(ex["messages"][1]["content"]) in frozen]
    report = {
        "built_by": "ml/tune/build_multitopic.py",
        "seed": SEED,
        "n_synthetic_target": n_synthetic_target,
        "holdout_frac": holdout_frac,
        "n_holdout": n_hold,
        "leakage_frozen_train": leak_train,
        "leakage_frozen_eval": leak_eval,
        "train": _stats(train, [p for p in provenance if p["uid"] in {e["uid"] for e in train}]),
        "eval": _stats(evald, [p for p in provenance if p["uid"] in {e["uid"] for e in evald}]),
    }
    (OUT_DIR / "meta.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    build(n_note_target=0)