# v2 error analysis

Ten failure modes, each with a count and a cause. Counts are from the recorded v2
attempt-10 artifacts:

- frozen single-topic: `ml/data/tune/eval/lora.json` (n=329, 314 parsed to ≥1 topic)
- multi-topic: `ml/data/tune/multitopic/results/lora.json` (n=85)
- smoke: `ml/data/tune/smoke/results/finetuned/smoke_results.jsonl` (n=20)

Headline metrics for reference:

| | frozen 329 | multitopic 85 | smoke 20 |
| --- | --- | --- | --- |
| JSON parse | 1.000 | 0.988 | — |
| schema valid | 0.997 | 0.988 | 1.00 |
| domain accuracy | 0.839 | 0.682 (set exact) | 0.90 (coverage) |
| issue ROUGE-L | 0.923 | 0.799 | — |
| topic-count acc | — | 0.859 | — |
| multi-topic rate | 0.037 | 0.894 (recall) | 1/4 |
| hallucination | 0.015 | — | 0 |

---

## 1. Administrative boilerplate copied into `issue` — 212/314 (67.5 %)

**Cause.** The training label leaks it. `_ADMIN_CLAUSE` in
`ml/tune/build_dataset.py` expects `згоду надає` / `заявник надає`; the corpus says
`надає згоду`, which no pattern matches. Targets carry boilerplate in 211/329 frozen
cases, 5270/8519 v2 training examples, and 88.3 % of multitopic training topics.
Predictions carry it in 212/314 (67.5 %) — a rate indistinguishable from the label's.

**Not a model failure.** The model is doing exactly what it was taught. This is the
single largest source of apparent error in the whole system, and it is a labeler bug.

**Fix.** v3 change C1. Note the consequence: `issue_rouge_l` against the frozen v1
label *will* fall, because that label contains the boilerplate being removed. The v3
plan records an expected drop as a non-blocking gate with a 0.80 floor so it is not
later mistaken for a regression.

## 2. `requested_action` is a copy of `issue` — 154/157 (98.1 %)

**Cause.** Redundancy in the label. 2616/5384 v1 training examples have a non-empty
action, and in all of them the action already appears in the `issue` (333 exact, 2124
substring). 93.9 % of action-bearing examples are self-referential. The model learned
the shortcut, as it should.

**Effect.** The field carries no independent signal. `requested_action_presence_match`
is 0.9909, which looks excellent and means almost nothing.

**Fix.** v3 change C2 — emit an empty action unless an explicit request verb is
present. The eval-v3 category H (12 cases) asserts this.

## 3. Topic-count errors — 12/85 (14.1 %) on multi-topic input

Breakdown, all against a target of 2:

| predicted | count | reading |
| --- | --- | --- |
| 1 | 8 | under-extraction: second problem missed |
| 0 | 1 | nothing extracted at all |
| 3 | 2 | over-extraction |
| 4 | 1 | severe over-extraction |

8 of 12 are **under**-extractions, and the model picks the wrong one of the two
domains to keep rather than emitting a spurious one. Examples:
`MT-SYN-Ш-2860-Ю-1915` → predicted `transport` only, target `construction`+`transport`;
`MT-SYN-Р-2204-С-2902` → predicted `water` only, target `heating`+`housing`.
Only one case (`MT-SYN-К-434-Г-1532`) extracts nothing at all.

## 4. Domain substitution at the correct topic count — 16/85 (18.8 %)

**Cause.** Ontology, not extraction. The model splits the complaint correctly and
then names one bucket wrongly. The substitution pairs are the semantically adjacent
ones from `ontology_audit.md`: `sanitation`↔`water` (4), `construction`↔`sanitation`
(2), `roads`↔`water` (2).

**This is the most fixable of the multi-topic failures without a schema change**,
because the topic boundary was right — only the vocabulary was wrong.

## 5. Duplicate-domain topics — 3/85

`MT-SYN-SS-Д-2579-І-442` predicts 4 topics that collapse to one domain (`sanitation`);
`MT-SYN-SS-М-3355-Л-2372` and `MT-SYN-SC-М-709-Р-897` predict 2 topics sharing a
domain. The model split a single problem into several and then could not name them
differently. `domain_set_exact` treats these as failures, correctly.

## 6. The out-of-enum domain — 1/329

`Г-2332` is predicted `address`, which is not one of the 13 schema values. This is the
sole cause of `schema_valid` 0.997. Interesting rather than alarming: the model
invented a plausible-sounding domain name instead of copying the enum. One case in
329, and it is a vocabulary failure, not a format failure.

## 7. Object field — strong, with a redactor artefact

`object_exact` 0.9422, `object_token_overlap` 0.9574. The weakest cases are PII
redaction artefacts rather than extraction errors: the labeler's `_redact_pii` turns
`КП 'Міськсвітло' КМР` into `КП 'Р`, and a long single-letter street name survives
only as `вул. Є`. The model is being asked to reproduce mangled strings.

## 8. Multi-topic rate on single-topic input — 12/329 (3.7 %)

The frozen benchmark contains single-topic complaints; the model spuriously emitted
multiple topics in 3.7 % of them. Well below the 18.8 % domain-substitution rate, so
topic-count calibration is not the problem — naming is.

## 9. The terse register is not measured at all

Not a model failure — a **benchmark** failure, and the most consequential gap in the
evaluation. Terse complaints (<150 chars) are 57.6 % of the 2023 slice and **0 of
1124 validation and 0 of 329 test records** are under 150 chars, because the split is
by date and terse notes are a 2023 phenomenon. Every frozen and multitopic number
above is measured on mid-to-long inputs.

The smoke suite is the only thing touching this register, and it is where the
multi-topic failures show: 1/4. See `short_note_analysis.md`.

## 10. Smoke: the failures are all terse two-topic cases

`smoke-14`, `smoke-15`, `smoke-16` fail; `smoke-13` now passes. All three failures are
two-topic inputs where only one problem is extracted, and all three are short. The 17
smoke cases that pass are dominated by single-topic inputs.

Reading: 1/4 multi-topic on terse input versus 89.4 % recall on the multitopic suite
is a single, consistent story — **the model handles multi-topic extraction on
well-formed mid-length text and fails on terse notes.** The eval-v3 suite is built to
measure exactly this, with 52 two-topic cases and 16 cases under 150 characters.

---

## Attribution summary

| # | failure | whose fault | fixable without a schema change |
| --- | --- | --- | --- |
| 1 | boilerplate in `issue` | labeler | yes — C1 |
| 2 | action redundancy | labeler | yes — C2 |
| 3 | under-extraction | model / data | yes — C3 |
| 4 | domain substitution | ontology | partly — C4 |
| 5 | duplicate-domain topics | model | partly |
| 6 | out-of-enum domain | model | yes — enum is in the prompt |
| 7 | object redactor artefact | labeler | yes — narrow the redactor |
| 8 | spurious multi-topic | model | yes |
| 9 | terse register unmeasured | **evaluation** | yes — eval-v3 |
| 10 | terse multi-topic failure | model / data | yes — C3 |

Six of ten are labeler or evaluation defects, not model defects. That is the single
most useful conclusion of this audit: the v2 numbers understate the model on the
things it was asked to do, and overstate it on the things it was taught wrongly.
