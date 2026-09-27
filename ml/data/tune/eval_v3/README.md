# Eval suite v3 (adversarial / diagnostic)

Built by `ml/tune/build_eval_v3.py`. **146 cases**, generated deterministically
(no RNG, no sampling seed) so the suite is byte-reproducible.

```bash
.venv/bin/python -m ml.tune.build_eval_v3
```

| file | contents |
| --- | --- |
| `cases.jsonl` | one case per line: `id`, `text`, `category`, `category_name`, `expected_topics`, `provenance`, `note` |
| `schema.json` | JSON Schema for a case object |
| `stats.json` | composition, length distribution, domain counts, comparability note |

## What this suite is for

It is a **diagnostic**, not a headline benchmark. Every `expected_topics` value is
produced by the contract labeler (`ml.tune/build_dataset.py::label_record`) with one
specific gap closed — `_ADMIN_CLAUSE` does not match the very common word order
`надає згоду` (it expects `згоду надає`), so response-delivery boilerplate survives
into the `issue` field for a large share of training targets. `build_eval_v3.py`
adds `BOILER_SENT` / `_DANGLING` to close that gap.

Consequence: **scores here are not comparable** with

- `ml/data/tune/eval/` (frozen 329 cases) or
- `ml/data/tune/multitopic/eval.jsonl` (85 cases),

because those score against the *uncleaned* v1 weak label, which carries
administrative boilerplate in 53–64 % of `issue` fields. A model that correctly
learned to strip boilerplate looks *worse* on the old suites. Use those suites for
regression continuity, and this suite for measuring the target behaviour.

## Categories (A–V)

| id | name | probes |
| --- | --- | --- |
| A | `terse_single` | one complaint, minimal or mid-length note (16 cases < 150 chars, shortest 65) |
| B | `terse_two_topic_i` | two terse complaints concatenated, connective only |
| C | `terse_two_topic_also` | two terse complaints joined by "а також" |
| D | `two_sentences` | two complaints as two **full** sentences |
| E | `same_object_two_problems` | one location, two different problems |
| F | `different_objects_two_problems` | two different objects, two problems |
| G | `explicit_and_implicit_action` | one explicit request plus one implicit problem |
| H | `no_explicit_action` | `requested_action` must be empty |
| I | `long_rambling` | back-story, dates, repetition |
| J | `irrelevant_prose` | heavy consent / response-delivery boilerplate |
| K | `ontology_ambiguity` | domain pair a human reviewer would also struggle to split |
| L | `lighting` | street / outdoor lighting |
| M | `traffic_lights` | traffic signals |
| N | `playgrounds` | playgrounds and play equipment |
| O | `trees` | trees, branches, bushes, mowing, weeds |
| P | `stray_animals` | stray / nuisance animals |
| Q | `waste` | waste, dumpsters, illegal dumping |
| R | `water_sewerage` | water supply and sewerage |
| S | `roads_sidewalks` | roads, asphalt, potholes, sidewalks, manholes |
| T | `public_transport` | public transport and drivers |
| U | `housing_heating` | housing maintenance and heating |
| V | `elevators` | elevators |

Per-category counts are 4–12; `stats.json` has the exact figures.

## Provenance

| kind | n | meaning |
| --- | --- | --- |
| `real` | 84 | verbatim held-out record |
| `condensed` | 10 | held-out record with boilerplate sentences and wrapper words deleted |
| `composed` | 52 | two held-out records joined, joined by a connective / sentence boundary / "а також" |

Composition always pairs two records from the **same split**, so every case has
exactly one provenance split (`stats.cases_spanning_splits` is empty): 116
validation, 30 test. 94 cases expect one topic, 52 expect two.

Every `provenance.sources` uid comes from `ml/data/validation.jsonl` or
`ml/data/test.jsonl`. **No case text is drawn from `ml/data/train.jsonl`.** This is
enforced by
`ml/tests/test_eval_v3_suite.py::test_no_case_text_leaks_into_training`, which checks
each case text against `train.jsonl`, `multitopic/train.jsonl` and `v2/train.jsonl`
after `normalize_text`.

No case is synthesised from scratch. Composition only concatenates or deletes
material that is present in a held-out record; nothing is invented.

### `uid` is not an identity

The builder deduplicates held-out records by **normalised content**, never by `uid`.
The portal reuses case numbers across years: `Б-67` is a 2023 lift complaint *and* a
2026 waste-invoice complaint, and `К-64` is a 2023 housing complaint *and* a 2026
sanitation one. Across the frozen splits 130 uids appear in more than one split with
unrelated text (none are near-duplicates: 0 share more than 50 % of their 40-char
shingles), and 58 uids are duplicated inside the held-out pool alone. Treat `uid` as
traceability metadata only. See `ml/reports/contract_audit.md`.

## Known reference-labeler limitations

These are properties of the existing contract, inherited by the references here.
They are reported in `ml/reports/ontology_audit.md` and
`ml/reports/contract_audit.md` rather than hidden here.

1. **Domain is inherited from the portal `kind` field**, not derived from the text.
   Broken traffic lights are registered under `світлофор` and therefore labelled
   `sanitation`; playgrounds, animals, trees and waste are also largely `sanitation`.
   Semantic edge cases (category B) built on those kinds carry the portal label, not
   a human one.
2. **The PII redactor is aggressive.** `КП ’Міськсвітло’ КМР` becomes `КП ’Р`, and a
   long single-letter street name can survive only as `вул. Є`. Present in expected
   `object` values for a handful of cases.
3. **Boilerplate detection is sentence-level.** A response-delivery clause merged
   into the same sentence as the real issue survives; residual fragments like
   `можливість отримати 21.01.2026 відповідь на питання` remain in a small number of
   references.
4. **`requested_action` is largely redundant with `issue`** (93.9 % of v1 training
   examples contain the action inside the issue), so it is a weak training signal
   rather than an independent target.

## Usage

The suite is a scoring target, not an executable test — no model is run against it
in CI. `ml/tests/test_eval_v3_suite.py` validates schema conformance, production
schema conformance of every `expected_topics`, category coverage, provenance
integrity and leakage. Score it after a training run with a v3 trainer
(`ml/tune/run_train.py`) and compare against the gates in
`ml/reports/v3_experiment_design.md`.
