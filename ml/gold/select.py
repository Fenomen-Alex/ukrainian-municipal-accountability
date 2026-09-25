"""Deterministic selection of a ~400-record gold annotation set.

Reads the frozen train/validation/test JSONL splits plus the cached Qwen3
embeddings and writes ``ml/data/gold/*`` outputs. No LLM, no API calls, no
downloads; annotations are left empty for a human/LLM annotator to fill.

Run::

    python3 -m ml.gold.select --data-dir ml/data

Everything is deterministic for a fixed seed and fixed input files.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, OrderedDict
from pathlib import Path

import numpy as np

from ml.cleaner import normalize_text

SEED = 20240917
CLASS_BUDGET = 360
MIN_PER_KIND = 12
MAX_KMEANS_ITER = 50
OUTLIER_FRACTION = 0.08
MULTITOPIC_MAX = 40
FINAL_TARGET = 400
SPLIT_ORDER = ("train", "validation", "test")
METHOD_ORDER = ("kmeans_centroid", "kmeans_outlier", "multitopic_upsample")

# Canonical kind enumeration (labels.json order).
KIND_ORDER = [
    "Будівництво та ремонт доріг, вулиць",
    "Будівництво, містобудування, архітектура",
    "Гаряче та холодне водопостачання",
    "Діяльність органів місцевого самоврядування",
    "Експлуатація та ремонт житла ( у т.ч. ліфтів, сантехнічного обладнання тощо)",
    "Електропостачання населених пунктів, будинків",
    "Питання пов’язані з торгівлею (у тому числі стихійна торгівля)",
    "Плата за житло та комунальні послуги ( у т.ч. підвищення тарифів)",
    "Пільгове перевезення пасажирів",
    "Робота пасажирського транспорту (у т. ч. електричного транспорту)",
    "Санітарний стан, благоустрій населених пунктів, прибудинкових територій",
    "Теплопостачання",
]

_CLIP_RE = re.compile(r"\bкрім того\b|\bа також\b")


def boilerplate_score(text: str) -> float:
    """Higher means more administrative boilerplate / less signal.

    Scores are additive weights of common operator/docket phrasing plus a
    weak penalty for very short text. Only de-prioritizes; never excludes.
    """
    low = text.lower()
    score = 0.0
    if "відповідь заявнику" in low:
        score += 3.0
    if "надає згоду на обробку персональних даних" in low:
        score += 3.0
    if "персональних даних" in low:
        score += 1.0
    if "надано номер телефону" in low or "надано номер контакт" in low:
        score += 2.0
    if len(text) < 40:
        score += 3.0
    elif len(text) < 80:
        score += 1.5
    elif len(text) < 160:
        score += 0.5
    return score


def multi_topic_score(text: str) -> int:
    """Deterministic heuristic score of likely multi-topic complaints.

    Only a selection signal -- NOT a gold label. Higher score suggests
    several independent complaint clauses. Signals:
      * repeated ``щодо`` / ``про`` topic markers
      * ``а також`` / ``крім того`` connectors
      * repeated ``прошу`` / ``вимагаю`` request verbs
      * clause-initial request verbs across sentence boundaries
    """
    low = text.lower()
    score = 0
    score += max(0, low.count("щодо"))
    if "а також" in low:
        score += 1
    if "крім того" in low:
        score += 1
    proshu = low.count("прошу")
    vymahaiu = low.count("вимагаю")
    if proshu >= 2:
        score += 1
    if vymahaiu >= 2:
        score += 1
    # Separable clauses: sentence/period/clause boundaries beginning with a
    # request or topic verb.
    clauses = re.split(r"[.;\n]", low)
    starters = 0
    for cl in clauses:
        cl = cl.strip()
        if cl.startswith("щодо") or cl.startswith("прошу") or cl.startswith(
            "вимагаю"
        ):
            starters += 1
    if starters >= 2:
        score += 1
    if len(_CLIP_RE.findall(low)) >= 2:
        score += 1
    return score


def is_multitopic_candidate(text: str) -> bool:
    return multi_topic_score(text) >= 2


def load_record_index(data_dir: Path) -> dict[str, list]:
    """Return ``{split: [original raw records]}`` in fixed file order."""
    records: dict[str, list] = OrderedDict((s, []) for s in SPLIT_ORDER)
    for split in SPLIT_ORDER:
        path = data_dir / f"{split}.jsonl"
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                records[split].append(rec)
    return records


def load_embeddings(data_dir: Path) -> dict[str, np.ndarray]:
    """Load cached embeddings per split, aligned to the JSONL file order."""
    out: dict[str, np.ndarray] = {}
    for split in SPLIT_ORDER:
        arr = np.load(data_dir / "embeddings" / f"{split}.npy")
        with open(data_dir / "embeddings" / f"{split}.meta.json", encoding="utf-8") as fh:
            meta = json.load(fh)
        if arr.shape[0] != meta["n"]:
            raise ValueError(
                f"{split}: {arr.shape[0]} embedding rows but meta n={meta['n']}"
            )
        if arr.shape[1] != meta["dim"]:
            raise ValueError(
                f"{split}: {arr.shape[1]} embedding columns but meta dim={meta['dim']}"
            )
        out[split] = arr
    return out


def build_resources(data_dir: Path) -> tuple[list[dict], dict[str, list]]:
    """Flatten records + embeddings into one per-split bundle order.

    Returns ``(records, split_records)``:
      ``records``      -- list of dicts, each with ``split/uid/kind/text/
                          index/emb/split_index`` keys, ordered by
                          ``(split order, file order)``.
      ``split_records``-- ``{split: [original raw records]}``.
    """
    split_records = load_record_index(data_dir)
    embeddings = load_embeddings(data_dir)
    records: list[dict] = []
    for split in SPLIT_ORDER:
        rows = split_records[split]
        arr = embeddings[split]
        if len(rows) != arr.shape[0]:
            raise ValueError(
                f"{split}: {len(rows)} records but {arr.shape[0]} embeddings"
            )
        for i, raw in enumerate(rows):
            records.append(
                {
                    "split": split,
                    "split_index": SPLIT_ORDER.index(split),
                    "uid": raw["uid"],
                    "kind": raw["kind"],
                    "text": raw["content"],
                    "index": i,
                    "emb": arr[i],
                }
            )
    return records, split_records


def compute_quotas(
    supports: dict[str, int],
    budget: int = CLASS_BUDGET,
    min_per_kind: int = MIN_PER_KIND,
    seed: int = SEED,
) -> dict[str, int]:
    """Deterministic per-kind quota *summing exactly* to ``budget``.

    ``quota ∝ support**0.6`` scaled to ``budget``, floored at
    ``min_per_kind``, capped at the kind's support. Because the floor can
    push the total above ``budget``, seats are then removed from the kinds
    that (a) are above their raw proportional share and (b) have room to
    spare above the floor, ties broken by canonical kind order. The result
    sums exactly to ``budget`` whenever ``budget >= 12 * len(kinds)`` and no
    kind is smaller than ``min_per_kind``.
    """
    kinds = [k for k in KIND_ORDER if supports.get(k)]
    if not kinds:
        return {}

    weights = {k: max(supports[k], 0) ** 0.6 for k in kinds}
    total_w = sum(weights.values())
    raw = {k: weights[k] / total_w * budget for k in kinds}

    quota: dict[str, int] = {
        k: int(max(min_per_kind, raw[k])) for k in kinds
    }
    for k in kinds:
        quota[k] = min(quota[k], supports[k])

    total = sum(quota.values())

    def excess_key(k: str):
        # Remove the seats that exceed their raw share most; order among
        # equals is deterministic by canonical KIND_ORDER.
        return (quota[k] - raw[k], KIND_ORDER.index(k))

    while total != budget:
        if total > budget:
            removable = [k for k in kinds if quota[k] > min_per_kind]
            if not removable:
                raise ValueError(
                    "cannot shrink quotas to budget without going below "
                    f"min_per_kind={min_per_kind}: {quota}"
                )
            k = max(removable, key=excess_key)
            quota[k] -= 1
            total -= 1
        else:
            addable = [k for k in kinds if quota[k] < supports[k]]
            if not addable:
                raise ValueError(
                    "cannot grow quotas to budget without exceeding supports"
                )
            k = min(addable, key=lambda k: (quota[k] - raw[k], KIND_ORDER.index(k)))
            quota[k] += 1
            total += 1

    return quota


def deterministic_kmeans(
    X: np.ndarray, k: int, seed: int, max_iter: int = MAX_KMEANS_ITER
) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic k-means with seeded k-means++-style initialization.

    ``labels[i]`` is cluster id, row order preserved; ``centroids`` has one
    row per cluster. Euclidean distance is used (equivalent to cosine
    distance for L2-normalized rows).
    """
    n = X.shape[0]
    if k <= 0:
        raise ValueError("k must be > 0")
    if k > n:
        raise ValueError(f"k={k} exceeds row count {n}")

    rng = np.random.RandomState(seed)
    centers = np.empty((k, X.shape[1]), dtype=np.float64)
    indices = np.empty(k, dtype=np.intp)

    first = int(rng.randint(0, n))
    centers[0] = X[first].astype(np.float64)
    indices[0] = first
    d2 = np.sum((X - centers[0]) ** 2, axis=1)

    for c in range(1, k):
        probs = d2 / d2.sum()
        pick = int(rng.choice(n, size=1, p=probs)[0])
        centers[c] = X[pick].astype(np.float64)
        indices[c] = pick
        d2 = np.minimum(d2, np.sum((X - centers[c]) ** 2, axis=1))

    labels = np.zeros(n, dtype=np.intp)
    for _ in range(max_iter):
        dist = np.sum(
            (X[:, None, :].astype(np.float64) - centers[None, :, :]) ** 2, axis=2
        )
        new_labels = np.argmin(dist, axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for c in range(k):
            mask = labels == c
            if np.any(mask):
                centers[c] = X[mask].mean(axis=0)
    return labels, centers


def build_id_table(records: list[dict]) -> None:
    """Attach ``_uid_repeat_count`` / ``_uid_occurrence_order`` in place.

    ``{split}:{uid}`` that appear only once keep the plain compound id;
    repeats get a deterministic ``#{ordinal}`` suffix (ordinal 0, 1, ... in
    fixed order). ``uid`` alone is never used (duplicated across splits).
    """
    counter: Counter = Counter()
    for r in records:
        counter[f"{r['split']}:{r['uid']}"] += 1
    seen: Counter = Counter()
    for r in records:
        base = f"{r['split']}:{r['uid']}"
        r["_uid_repeat_count"] = counter[base]
        r["_uid_occurrence_order"] = seen[base]
        seen[base] += 1


def id_for_record(rec: dict) -> str:
    base = f"{rec['split']}:{rec['uid']}"
    if rec["_uid_repeat_count"] == 1:
        return base
    return f"{base}#{rec['_uid_occurrence_order']}"


def select_records(records: list[dict]) -> tuple[list[dict], int]:
    """Run the full deterministic selection.

    Returns ``(selected, conflict_count)`` where conflict_count counts
    candidates skipped because their normalized text was already taken.
    """
    supports = Counter(r["kind"] for r in records)
    quotas = compute_quotas(supports)

    for rec in records:
        rec["is_multitopic_candidate"] = is_multitopic_candidate(rec["text"])

    taken: set[str] = set()
    conflicts = 0
    selected: list[dict] = []

    def claim(rec: dict, method: str) -> bool:
        nonlocal conflicts
        key = normalize_text(rec["text"])
        if key in taken:
            conflicts += 1
            return False
        taken.add(key)
        rec["selection_method"] = method
        rec["is_multitopic_candidate"] = is_multitopic_candidate(rec["text"])
        selected.append(rec)
        return True

    for kind in KIND_ORDER:
        q = quotas.get(kind, 0)
        if not q:
            continue
        kind_records = [r for r in records if r["kind"] == kind]
        if not kind_records:
            continue
        idx_of: dict[int, dict] = {id(r): r for r in kind_records}
        X = np.stack([np.asarray(r["emb"], dtype=np.float64) for r in kind_records])
        labels, centers = deterministic_kmeans(
            X, k=q, seed=_kind_seed(kind, SEED)
        )
        dist_to_center = np.sum((X - centers[labels]) ** 2, axis=1)

        # 1) One centroid representative per cluster.
        def rank_key(i: int):
            rec = kind_records[i]
            return (
                dist_to_center[i],
                boilerplate_score(rec["text"]),
                rec["split_index"],
                rec["uid"],
            )

        picked: set[int] = set()
        for c in range(q):
            members = [
                i for i in np.where(labels == c)[0] if i not in picked
            ]
            members.sort(key=rank_key)
            for i in members:
                if claim(kind_records[i], "kmeans_centroid"):
                    picked.add(i)
                    break

        # 2) Farthest-from-centroid outliers for boundary diversity.
        outlier_budget = max(0, min(int(q * OUTLIER_FRACTION), q - MIN_PER_KIND))
        if outlier_budget:
            remaining = [i for i in range(len(kind_records)) if i not in picked]
            remaining.sort(
                key=lambda i: (
                    -dist_to_center[i],
                    boilerplate_score(kind_records[i]["text"]),
                    kind_records[i]["split_index"],
                    kind_records[i]["uid"],
                )
            )
            for i in remaining[:outlier_budget]:
                if claim(kind_records[i], "kmeans_outlier"):
                    pass

    # 3) Multi-topic upsampling up to MULTITOPIC_MAX slots. The budget is
    #    derived so ordinary (quota+outlier) + multitopic == FINAL_TARGET.
    multitopic_budget = min(MULTITOPIC_MAX, FINAL_TARGET - len(selected))
    taken_ids = {id_for_record(r) for r in selected}
    remaining_pool = [
        r for r in records if id_for_record(r) not in taken_ids
    ]
    remaining_pool.sort(
        key=lambda r: (
            not r["is_multitopic_candidate"],
            -multi_topic_score(r["text"]),
            boilerplate_score(r["text"]),
            r["split_index"],
            r["uid"],
        )
    )
    multi_share = 0
    for r in remaining_pool:
        if multi_share >= multitopic_budget:
            break
        if not r["is_multitopic_candidate"]:
            continue
        if claim(r, "multitopic_upsample"):
            multi_share += 1

    # 4) Deterministic top-up to reach exactly FINAL_TARGET, filling from
    #    still-unselected multitopic candidates first, then any remaining
    #    candidates, all in a fixed deterministic order.
    if len(selected) < FINAL_TARGET:
        taken_ids = {id_for_record(r) for r in selected}
        fill_pool = [
            r for r in remaining_pool if id_for_record(r) not in taken_ids
        ]
        fill_pool.sort(
            key=lambda r: (
                -is_multitopic_candidate(r["text"]),
                -multi_topic_score(r["text"]),
                boilerplate_score(r["text"]),
                r["split_index"],
                r["uid"],
            )
        )
        for r in fill_pool:
            if len(selected) >= FINAL_TARGET:
                break
            method = "multitopic_upsample" if r["is_multitopic_candidate"] else "kmeans_centroid"
            claim(r, method)

    if len(selected) != FINAL_TARGET:
        raise RuntimeError(
            f"selection ended with {len(selected)} records, expected {FINAL_TARGET}"
        )

    # 5) Deterministic final ordering.
    selected.sort(
        key=lambda r: (
            KIND_ORDER.index(r["kind"]),
            METHOD_ORDER.index(r["selection_method"]),
            id_for_record(r),
        )
    )
    return selected, conflicts


def _kind_seed(kind: str, seed: int) -> int:
    return seed + KIND_ORDER.index(kind)


def gather_statistics(
    records: list[dict],
    selected: list[dict],
    split_records: dict[str, list],
    uid_unique: tuple[int, int, bool],
    conflicts: int,
) -> OrderedDict:
    stats = OrderedDict()
    stats["total_selected"] = len(selected)
    stats["source_total"] = len(records)
    stats["selection_parameters"] = {
        "seed": SEED,
        "class_budget": CLASS_BUDGET,
        "min_per_kind": MIN_PER_KIND,
        "outlier_fraction": OUTLIER_FRACTION,
        "multitopic_max": MULTITOPIC_MAX,
        "final_target": FINAL_TARGET,
        "max_kmeans_iter": MAX_KMEANS_ITER,
        "quota_exponent": 0.6,
        "embedding_model": "G37A/Qwen3-Embedding-0.6B-80k-squad",
    }
    stats["per_kind_counts"] = OrderedDict(
        (k, sum(1 for r in selected if r["kind"] == k)) for k in KIND_ORDER
    )
    stats["per_kind_support"] = OrderedDict(
        (k, sum(1 for r in records if r["kind"] == k)) for k in KIND_ORDER
    )
    stats["per_split_counts"] = OrderedDict(
        (s, sum(1 for r in selected if r["split"] == s)) for s in SPLIT_ORDER
    )
    stats["multitopic_candidate_total"] = sum(
        1 for r in records if is_multitopic_candidate(r["text"])
    )
    stats["multitopic_candidate_selected"] = sum(
        1
        for r in selected
        if r["is_multitopic_candidate"] and r["selection_method"] == "multitopic_upsample"
    )
    stats["selection_method_counts"] = OrderedDict(
        (m, sum(1 for r in selected if r["selection_method"] == m))
        for m in METHOD_ORDER
    )
    stats["near_normalized_duplicate_conflicts"] = conflicts
    stats["final_normalized_duplicate_count"] = len(selected) - len(
        {normalize_text(r["text"]) for r in selected}
    )
    stats["uid_uniqueness"] = {
        "total": uid_unique[0],
        "unique": uid_unique[1],
        "globally_unique": uid_unique[2],
        "id_scheme": "{split}:{uid} (+#{ordinal} on repeat)",
    }
    stats["per_source_file"] = OrderedDict(
        (s, {"records": len(split_records[s]),
             "selected": sum(1 for r in selected if r["split"] == s)})
        for s in SPLIT_ORDER
    )
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python3 -m ml.gold.select",
        description="Deterministic gold annotation set selection.",
    )
    parser.add_argument("--data-dir", default="ml/data", type=Path)
    args = parser.parse_args()

    data_dir = args.data_dir
    out_dir = data_dir / "gold"
    out_dir.mkdir(parents=True, exist_ok=True)

    records, split_records = build_resources(data_dir)
    build_id_table(records)
    selected, conflicts = select_records(records)

    total = len(records)
    unique = len({f"{r['split']}:{r['uid']}" for r in records})
    uid_tuple = (total, unique, unique == total)

    compile_schema(out_dir)
    compile_guide(out_dir)
    stats = gather_statistics(
        records, selected, split_records, uid_tuple, conflicts
    )

    with open(out_dir / "annotation_set.jsonl", "w", encoding="utf-8") as fh:
        for rec in selected:
            fh.write(
                json.dumps(
                    {
                        "id": id_for_record(rec),
                        "text": rec["text"],
                        "source_kind": rec["kind"],
                        "source_split": rec["split"],
                        "selection_method": rec["selection_method"],
                        "is_multitopic_candidate": rec["is_multitopic_candidate"],
                        "annotation": {"topics": []},
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    with open(out_dir / "selection_statistics.json", "w", encoding="utf-8") as fh:
        json.dump(stats, fh, ensure_ascii=False, indent=2)
        fh.write("\n")

    print(f"Wrote {len(selected)} records -> {out_dir / 'annotation_set.jsonl'}")
    print("Per-kind:", dict(stats["per_kind_counts"]))
    print("Methods:", dict(stats["selection_method_counts"]))
    print(f"Conflicts (normalized dups skipped): {conflicts}")


def compile_schema(out_dir: Path) -> None:
    schema = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "$id": "ukrainian-municipal-accountability/gold-annotation",
        "title": "Gold annotation for Ukrainian municipal complaints",
        "type": "object",
        "required": ["topics"],
        "additionalProperties": False,
        "properties": {
            "topics": {
                "type": "array",
                "description": (
                    "One or more independent complaints/requests in the text. "
                    "Empty when the text carries no actionable complaint."
                ),
                "items": {"$ref": "#/definitions/topic"},
            }
        },
        "definitions": {
            "topic": {
                "type": "object",
                "required": ["domain", "issue", "object", "requested_action", "attributes"],
                "additionalProperties": False,
                "properties": {
                    "domain": {
                        "type": "string",
                        "description": "Broad municipal domain of the topic.",
                        "enum": [
                            "roads",
                            "water",
                            "heating",
                            "housing",
                            "transport",
                            "sanitation",
                            "electricity",
                            "construction",
                            "benefits",
                            "government",
                            "commerce",
                            "payments",
                            "other",
                        ],
                    },
                    "issue": {
                        "type": "string",
                        "description": "Concrete problem, stated neutrally.",
                    },
                    "object": {
                        "oneOf": [
                            {
                                "type": "object",
                            },
                            {
                                "type": "string",
                                "description": "Legacy free-text object until structured refactor.",
                                "deprecated": True,
                            },
                        ],
                        "description": (
                            "Structured target of the complaint (address, "
                            "facility, service, person/body) or free text."
                        ),
                    },
                    "requested_action": {
                        "type": "string",
                        "description": "What the complainant asks the municipality to do.",
                    },
                    "attributes": {
                        "type": "object",
                        "description": "Additional structured facts present in the text.",
                        "additionalProperties": True,
                    },
                },
            }
        },
    }
    with open(out_dir / "annotation_schema.json", "w", encoding="utf-8") as fh:
        json.dump(schema, fh, ensure_ascii=False, indent=2)
        fh.write("\n")


def compile_guide(out_dir: Path) -> None:
    guide = """\
# Gold Annotation Guide — Ukrainian Municipal Complaints

This guide defines how to annotate the 400 records in
`annotation_set.jsonl`. The corpus holds citizen appeals to a Ukrainian
municipality; each record is one appeal. Annotation captures *what is
actually written*, nothing more.

This dataset is intended for evaluation and future supervision. Do not
train models with these records while annotating them.

## Purpose

The gold set measures how well automatic systems (TF-IDF baseline, embedding
classifier, future production models) recover the real topics of municipal
complaints. High-quality, consistent annotation therefore matters more than
quantity.

## Record format

Each JSONL line is:

```json
{
  "id": "train:Ш-2",
  "text": "original complaint text, unchanged",
  "source_kind": "label from the source dataset",
  "source_split": "train|validation|test",
  "selection_method": "kmeans_centroid|kmeans_outlier|multitopic_upsample",
  "is_multitopic_candidate": false,
  "annotation": {"topics": []}
}
```

Only `annotation.topics` is filled by you. Never edit `text`, `id`,
`source_kind`, `source_split`, `selection_method`, or
`is_multitopic_candidate`.

## WARNING about `source_kind`

`source_kind` is **weak supervision** from the source dataset. It may be
wrong, coarse, or assigned automatically. Treat it as a hint at most.

**Never treat `source_kind` as the gold annotation.** The gold answer is the
`topics` you write. If `source_kind` disagrees with the text, the text wins.

## Topics array

`topics` is an array, not a single object, because one complaint frequently
contains several independent requests (one address needs garbage pickup *and*
a pothole fix; a pensioner asks about *heating* and about *a benefit*).

Each element of `topics` describes ONE distinct complaint/request:

| Field | Meaning |
| --- | --- |
| `domain` | Broad municipal domain (controlled vocabulary, see schema). |
| `issue` | The concrete problem, stated neutrally. |
| `object` | The target of the complaint: address, facility, service, body. |
| `requested_action` | What the complainant asks the municipality to do. |
| `attributes` | Extra structured facts present in the text (dates, numbers, repeated incidents). |

## Domain vocabulary

Use one of the `enum` values in `annotation_schema.json`:

- `roads` — repair, potholes, street works, sidewalks
- `water` — hot/cold water supply, leaks, outages, quality
- `heating` — district heating, radiator temperature, boiler schedules
- `housing` — building maintenance, elevators, plumbing in flats, yards
- `transport` — municipal buses/trolleys, stops, schedules, fares
- `sanitation` — waste collection, stray animals, street cleaning, greenery
- `electricity` — power outages, street lights, wiring
- `construction` — building works, permits, illegal constructions
- `benefits` — subsidies, allowances, social payments
- `government` — municipal services, officials, document requests
- `commerce` — trade, markets, shop complaints
- `payments` — tariffs, utility bills, billing disputes
- `other` — anything not covered above

## `issue`

Write the concrete problem in 2–10 words, neutral, no judgement:

| Weak | Strong |
| --- | --- |
| "bad road" | "potholes on Dniprovska street near house 3" |
| "no water again" | "intermittent cold-water supply since 12 March" |

## `object`

The thing the complaint targets. Prefer a structured object:

```json
{"address": "вул. Центральна, 12", "kind": "residential_building"}
```

When no structure fits, a short free string is acceptable (see schema). If
the object is unclear or absent, use `null`.

## `requested_action`

The ask, impersonally stated:

- "repair the roadway"
- "recalculate the heating bill"
- "arrange waste collection on the scheduled day"
- "investigate and respond to the complainant"

If there is no explicit ask, describe the implied one. If there truly is no
action, leave the field empty.

## `attributes`

Structured facts *present in the text only*: addresses, dates, building
numbers, vehicle fleets, repeated consultations, quantities. Do not add
context from outside the record.

## Multi-topic handling

Split the text into distinct complaints BEFORE writing topics. A single
`topics` element must be internally coherent: one problem, one object, one
domain. Two different problems (even on the same street) → two elements.

`is_multitopic_candidate` is a heuristic hint; verify by reading, do not
trust it.

## Ambiguous complaints

- If a topic could fit two domains, choose the one the author most directly
  complains about; put the secondary aspect in `attributes` if stated.
- If the text is genuinely too vague to annotate, write `{"topics": []}`
  and add an `attributes` note `{"annotation_note": "ambiguous"}`.

## When to leave fields empty

- No actionable request → empty `requested_action`.
- Object not identifiable → `null`/empty `object`.
- No extra facts → empty `attributes`.
- No complaint at all (pure docket text) → `{"topics": []}`.

## Facts and PII

- Never infer facts that are not in the text. `source_kind`, dates, prior
  complaints, and repair history must be evidenced by the text.
- Never record PII gratuitously. Names and phone numbers belong in
  `attributes` only when directly relevant (e.g., the caller's reference
  number); otherwise omit them.

## What counts as a distinct topic

A distinct topic is a complaint that could be acted on independently by the
municipality. If acting on A does not resolve B, they are separate topics —
regardless of whether they appear in one paragraph.

## Examples

### Single topic

```json
{
  "id": "train:В-101",
  "text": "Прошу відремонтувати дорогу на вулиці Шевченка біля будинку 5.",
  "annotation": {
    "topics": [
      {
        "domain": "roads",
        "issue": "damaged road surface on Shevchenka street",
        "object": {"address": "вул. Шевченка, 5", "kind": "road"},
        "requested_action": "repair the roadway",
        "attributes": {}
      }
    ]
  }
}
```

### Multi-topic

```json
{
  "id": "train:K-42",
  "text": "Щодо будинку 12 по вулиці Лесі Українки: прошу вивезти сміття, а також полагодити під'їзд.",
  "annotation": {
    "topics": [
      {
        "domain": "sanitation",
        "issue": "uncollected waste",
        "object": {"address": "вул. Лесі Українки, 12", "kind": "residential_building"},
        "requested_action": "arrange waste removal",
        "attributes": {}
      },
      {
        "domain": "housing",
        "issue": "broken entrance door",
        "object": {"address": "вул. Лесі Українки, 12", "kind": "entrance"},
        "requested_action": "repair the entrance",
        "attributes": {}
      }
    ]
  }
}
```

### No complaint (docket noise)

```json
{
  "id": "validation:А-7",
  "text": "Надано номер телефону ГЛ Пенсійного фонду (0800503753).",
  "annotation": {"topics": []}
}
```
"""
    with open(out_dir / "annotation_guide.md", "w", encoding="utf-8") as fh:
        fh.write(guide)


if __name__ == "__main__":
    main()