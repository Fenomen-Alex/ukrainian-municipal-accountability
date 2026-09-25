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
