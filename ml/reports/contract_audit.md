# Contract audit — v2 / attempt-10

What the deployed model actually promises, what it is actually trained on, and
where the two disagree. No schema change, no retraining; this is a description of
the contract as it stands.

## 1. Input

A record from `ml/data/{train,validation,test}.jsonl` is a single complaint:

| field | role |
| --- | --- |
| `uid` | case number, **not a stable identity** (see §5) |
| `receivedDateTime` | receipt timestamp; drives the date split |
| `type` | channel |
| `kind` | portal category; the **only** source of the training `domain` |
| `content` | the complaint text |
| `address`, `organization`, `result` | routing metadata, not used for labelling |

`ml/tune/build_dataset.py` turns each into a chat example: one system message
(`SYSTEM_PROMPT`), the raw `content` as the user message, and the weak label as the
assistant message. The v2 multitopic stream is built by `ml/tune/build_multitopic.py`
and v2 composition by `ml/tune/build_v2.py`.

`ml/tune/run_train.py::_NoThinkingTokenizer` forces `enable_thinking=False` on the
Qwen3 chat template, so the learned format matches what `ml/tune/run_eval.py` asks
for at inference. This is the one deliberate divergence from stock `mlx-lm`.

## 2. Output

Strictly `{"topics": [...]}`. Each topic in
`ml/data/gold/annotation_schema.json` (`definitions/topic`) requires exactly:

`domain`, `issue`, `object`, `requested_action`, `attributes`

- `domain`: 13-value enum — `roads, water, heating, housing, transport, sanitation,
  electricity, construction, benefits, government, commerce, payments, other`
- `issue`: free text, the problem statement
- `object`: `oneOf [object, string]`, the string branch marked `deprecated: true`
- `requested_action`: free text
- `attributes`: optional structured fields

`ml/tune/evaluate.py` scores JSON parse rate, Draft-07 schema validity, domain
accuracy and macro-F1, issue ROUGE-L, object exact/token overlap, requested-action
presence match, hallucination rate, and multi-topic rate.
`ml/tune/run_eval_multitopic.py` adds topic-count accuracy, multi-topic recall, and
domain-set precision/recall/exact.

## 3. The label is a projection of `kind`, not a reading of the text

`label_record` maps `kind` → `domain` through `KIND_TO_DOMAIN` (12 entries) and then
derives `issue`, `object`, `requested_action` and `attributes` by string surgery on
`content`. Two consequences dominate every metric in this repository:

1. **The model is never shown a domain that the portal `kind` does not license.**
   It learns the portal's taxonomy, including the portal's mistakes. See
   `ontology_audit.md`.
2. **The `issue` target is whatever survives the cleaner**, and the cleaner leaks
   administrative boilerplate. See §4 and `v2_error_analysis.md`.

## 4. Boilerplate leak in the label

`ml/tune/build_dataset.py::_ADMIN_CLAUSE` is the sentence filter meant to drop
consent and response-delivery boilerplate. It misses the most common word order in
the corpus. It expects `згоду надає` / `заявник надає`; the corpus overwhelmingly says
**`надає згоду`**, which no pattern matches. A second miss: `відповідь\s+заявник`
does not match the declined form `заявниці`.

Measured with a broad substring detector (15 response-delivery phrases):

| corpus | flagged |
| --- | --- |
| v1 train targets | 3035 / 5384 (56.4 %) |
| v2 train examples | 5270 / 8519 (61.9 %) |
| frozen test targets | 211 / 329 (64.1 %) |
| multitopic train examples | 745 / 979 (76.1 %) |
| multitopic train topics | 1061 / 1201 (88.3 %) |
| multitopic eval examples | 69 / 85 (81.2 %) |

This is a *detector* count, not a hand-verified one; the true figure is lower. But
the direction is not in doubt, because the model reproduces it: 212 / 314 (67.5 %)
of frozen predictions carry boilerplate in `issue`, and 217 / 329 (66.0 %) of the
targets do too. The model is faithfully learning the labeler.

`ml/tune/build_eval_v3.py::BOILER_SENT` closes the gap for the eval-v3 reference
label only. It is deliberately **not** applied to the training data here: doing so
would invalidate every recorded v2 number, so it is proposed as v3 change C1 instead.

## 5. `uid` is not an identity

The portal reuses case numbers across years. `Б-67` is a 2023 lift complaint *and* a
2026 waste-invoice complaint; `К-64` is a 2023 housing complaint *and* a 2026
sanitation one.

- 130 uids appear in more than one split, always with unrelated text. None are
  near-duplicates: 0 of 130 share more than 50 % of their 40-character shingles, and
  0 are substrings of each other.
- 58 uids are duplicated **inside** the held-out pool (`validation` + `test`).
- Exact duplicate texts **within** a single split: train 220, validation 46, test 15.
  So the effective size of the 329-row frozen test set is about 314.
- Exact text duplicates **across** splits: 0. The date-based split is text-clean,
  which is the one reassuring result here.

Consequence: anything that dedups or checks leakage by `uid` silently passes.
`ml/tune/build_eval_v3.py` and its test key on normalised text for this reason.

## 6. Requested-action redundancy

In v1 train, 2616 / 5384 (48.6 %) of examples have a non-empty `requested_action`,
and in **every** one of those the action already appears inside the `issue`: 333
exactly, 2124 as a substring. 93.9 % of action-bearing examples are therefore a
copy of part of their own `issue`, and the model reproduces that — 154 / 157 (98.1 %)
of its non-empty predicted actions are substrings of its predicted `issue`.

The field is close to unlearnable as a distinct skill. v3 change C2 makes it empty
unless an explicit request verb is present.

## 7. Split and leakage

`ml/data/statistics.md`: train `<2026-01-01`, validation `[2026-01-01, 2026-07-01)`,
test `>=2026-07-01`. Because the split is by date and the corpus is dominated by 2023
material, the splits are **not exchangeable** — see `short_note_analysis.md`.

v2 `meta.json` reports `multitopic_share: 0.3448` while
`ml/data/tune/v2/README.md` states ≈0.368. Recomputing from the emitted file gives
3135 / 8519 = 0.368. The declared metadata disagrees with the data it describes;
this is left as a finding rather than silently corrected.

`multitopic/meta.json` records empty `leakage_frozen_train` and
`leakage_frozen_eval` lists, consistent with the zero cross-split text overlap above.

## 8. Unresolved inconsistencies

| # | issue | status |
| --- | --- | --- |
| 1 | 12 `kind` labels → 13 schema domains; `benefits` and `other` have zero support | documented, not fixed |
| 2 | two `kind`s (`Пільгове перевезення пасажирів`, `Робота пасажирського транспорту`) both map to `transport` | documented |
| 3 | `_ADMIN_CLAUSE` misses `надає згоду` | v3 change C1 |
| 4 | `object` schema still allows the deprecated string branch; the labeler only ever emits strings | documented |
| 5 | `v2/meta.json` multitopic_share 0.3448 vs 0.368 actual | v3 change C4 |
| 6 | `eval_smoke.py` hardcodes the result tag `finetuned`; the stale `finetuned-v2` artifact produced a wrong 0/4 reading | documented; `smoke_cases.json` tags were corrected |
| 7 | `uid` reused across years | documented; nothing keys on it |

Items 1, 2, 4 and 7 are properties of the frozen schema and corpus, not bugs to be
patched here. Fixing any of them would invalidate the recorded baselines, so they are
carried into v3 as constraints rather than changes.
