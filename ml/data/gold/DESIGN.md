# Gold Annotation Selection — Design

Goal: deterministically select exactly 400 records from the 6,837-record
corpus (train/validation/test) to form a human/LLM gold annotation set. The
selection uses the cached Qwen3 1024-dim L2-normalized embeddings only. No
LLM, no API, no downloads, no training. Annotation starts empty; humans/LLMs
label `annotation.topics` later.

Source files (`ml/data/{train,validation,test}.jsonl`, `ml/data/embeddings/*`)
are never modified. Labels: `ml/data/labels.json` defines the canonical
12-kind order used everywhere for deterministic iteration.

## ID scheme

`uid` is NOT globally unique across splits (6,301 of 6,837) and is not even
unique within a split (up to 3 records share a `{split}:{uid}` pair). So each
record's `id` is:

- if the `{split}:{uid}` pair is unique in the whole corpus: `{split}:{uid}`
- otherwise: `{split}:{uid}#{occurrence}` where `occurrence` is a
  deterministic 1-based index of the record among sharing records, in fixed
  file order.

Guaranteed unique. Recorded in statistics as `uid_strategy`.

## Kinds / iteration order

Kinds iterate in `labels.json` order (values 0..11). Splits iterate in fixed
order train, validation, test. All loops use explicit indices / sorted keys;
no reliance on set/dict iteration order anywhere.

## 1. Quota (sum exactly 360)

For each kind, support = number of records of that kind across all three
splits. Compute raw quota proportional to `support^0.6`, scale so the sum of
raw quotas is ~360, floor each at 12, cap each at its kind's support, then
deterministically nudge (round residual via largest-remainder with seed-based
stable tie-breaks) so the quotas sum to exactly **360**.

- `min_per_kind = 12`
- `class_budget = 360`
- never exceed a kind's support.

## 2. Per-kind clustering

For each kind with quota `k`: run deterministic NumPy k-means on that kind's
stacked normalized embeddings, `k = quota` clusters, one representative per
cluster.

- seeded `RandomState(seed + kind_index)` for reproducibility
- k-means++-style seeded centroid initialization
- fixed iteration cap 50, early stop on stable labels
- ties broken by index (argmin), which is deterministic given fixed input
  order.

## 3. Representative selection

For each cluster: rank members by

1. distance to that cluster's centroid (ascending)
2. boilerplate score (ascending)
3. `source_split` order (train < validation < test)
4. `uid` (ascending)

Take the nearest valid candidate. `boilerplate_score` is a small integer or
float; higher = more boilerplate. Signals (never exclusive, only
de-prioritize):

- `Відповідь заявнику`
- `надає згоду на обробку персональних даних`
- leading `Щодо` operator phrasing
- very short / signal-poor text.

## 4. Outliers

`outlier_budget = min(class_budget, total_outlier_quota)` where
`total_outlier_quota = 40` per class budgets divided across kinds
proportionally to quota, deterministically; capped so outliers never displace
the 12-minimum (outliers only come from kinds with quota above the minimum).

Each kind additionally selects its outlier share as farthest-from-centroid
records within that kind, ranked by distance descending then boilerplate
ascending then uid. Outlier picks do not count against the `360` quota; they
are additional.

## 5. Deduplication

Maintain a global set of normalized texts (`ml.cleaner.normalize_text`). If a
candidate's normalized text is already taken:

- skip it, take the next-ranked candidate for that slot (same cluster for
  centroid; next-farthest for outliers)
- record the conflict count.

If a kind cannot fill its slot from the cluster, fall through to the next
best still-unselected record of that kind. The quota is never silently
discarded. Any remaining unfilled slots are `recounted`; final dataset still
exactly 400.

Guarantee: the 400 selected records have pairwise distinct
`normalize_text(text)` strings.

## 6. Multi-topic upsampling

Deterministic heuristic flags records likely to contain several independent
complaints. Signals (counted per record, combined into a deterministic
boolean `is_multitopic_candidate`):

- 2+ separable occurrences of `щодо` / `про` clauses
- presence of `а також`
- presence of `крім того`
- 2+ occurrences of `прошу`
- 2+ occurrences of `вимагаю`
- other clearly separable complaint clauses (e.g. 2+ enumerated requests)

This is a selection signal only — NOT a gold label.

After ordinary (quota + outlier) selection, records flagged as multi-topic
candidates that were NOT yet selected are ranked deterministically
(multi-topic signal strength desc, then boilerplate asc, then uid) and the
top ones fill the remaining slots up to **40**. The last slots (to reach
exactly 400) are filled from the remaining unselected records ranked by
boilerplate asc, then uid. Multi-topic upsample selections keep their
original `kind` and source split; their `selection_method` is
`multitopic_upsample` (or `kmeans_*` if later re-used); the JSONL only
records the method that actually placed the record.

## 7. Final deterministic ordering

Records are emitted sorted by `source_kind` (labels.json order), then
`selection_method` (fixed enum order), then `id`. Every sort uses explicit
keys; no set ordering.

## Outputs (all written under `ml/data/gold/`)

| File | Contents |
| --- | --- |
| `annotation_set.jsonl` | 400 records, one JSON object per line |
| `annotation_schema.json` | JSON Schema (draft-07) for the annotation |
| `annotation_guide.md` | human/LLM annotation instructions |
| `selection_statistics.json` | all counters + parameters |
| `DESIGN.md` | this document |

Each `annotation_set.jsonl` record:

```json
{
  "id": "...",
  "text": "<original content byte-for-byte>",
  "source_kind": "<label>",
  "source_split": "train|validation|test",
  "selection_method": "kmeans_centroid|kmeans_outlier|multitopic_upsample",
  "is_multitopic_candidate": false,
  "annotation": { "topics": [] }
}
```

`id` is either `{split}:{uid}` or the disambiguated `{split}:{uid}#{n}`.
`text` equals the original `content` exactly. `annotation.topics` is always
`[]` at this stage — no automatic annotation.

## Determinism

A single fixed seed (default `20240917`) drives k-means init and any
random-ish nudge decisions. All downstream choices (ranking, dedup, ordering)
are pure functions of the fixed input order. Re-running the CLI must produce
byte-identical outputs.