# v3 experiment design

Machine-readable form: `ml/data/tune/v3/plan.json`. Rationale: `ml/data/tune/v3/README.md`.

**Status: the plan below is the original design, kept as written. §10 records where
implementation departed from it.** Where the two disagree, §10 and the measured
record in `ml/data/tune/v3/v2_baseline.json` are authoritative — `plan.json` carried
several baselines that were mis-sourced, two of which were blocking gates.

## 1. What v3 is trying to fix

The audit found four failures with independent causes, and — more usefully — that
**six of the ten error modes are labeler or evaluation defects, not model defects**
(`v2_error_analysis.md` §attribution). So v3 is not "a bigger model run". It is:

| change | failure it targets | whose defect |
| --- | --- | --- |
| C1 boilerplate cleaning | 67.5 % of predictions carry admin boilerplate in `issue` | labeler |
| C2 decoupled `requested_action` | 98.1 % of predicted actions are substrings of their own `issue` | labeler |
| C3 terse-note upsampling | terse multi-topic recall 1/4 vs 89.4 % on mid-length | data |
| C4 multitopic share + real fraction | 8.4 % real, and `meta.json` disagrees with the data | data |

None of the four requires a schema change, which matters: the schema is frozen and
changing it would invalidate every recorded number.

## 2. Arms

| arm | data | n | purpose |
| --- | --- | --- | --- |
| **control** | `ml/data/tune/v2/train.jsonl`, unchanged | 8519 | reproduce the baseline under the seeded pipeline |
| **treatment** | `ml/data/tune/v3/treatment/train.jsonl` | **9468** (plan said ~10500) | C1–C4 |

**The control arm is not optional.** Each change trades against a number v2 is
currently good at:

- C1 removes boilerplate from the target, so `issue_rouge_l` against the frozen v1
  label **must** fall. The drop is the fix working.
- C2 empties `requested_action` where no request verb exists, so
  `requested_action_presence_match` **must** fall. Same reasoning.

With one arm, a metric that moves is unattributable. With both, the control arm
isolates harness variance from the data change. Order is fixed: control first.

## 3. Benchmarks

| suite | n | reference | role |
| --- | --- | --- | --- |
| `ml/data/tune/eval` | 329 | v1 weak label, boilerplate intact | regression |
| `ml/data/tune/multitopic/eval.jsonl` | 85 | v1 weak label, boilerplate intact | regression |
| `ml/data/tune/eval_v3` | 146 | cleaned reference | **primary** |
| `ml/data/tune/smoke` | 20 | human-curated | guardrail |

The first two are **not comparable to eval_v3** and must not be pooled with it. They
score against a label that contains the boilerplate C1 deletes, so a model that
implements C1 correctly looks worse on them. Every gate in §5 says which suite it is
measured on for exactly this reason.

## 4. The adversarial suite

146 cases, 22 categories A–V, built deterministically by
`ml/tune/build_eval_v3.py` (no RNG, byte-reproducible):

| axis | coverage |
| --- | --- |
| provenance | 84 `real`, 10 `condensed`, 52 `composed` |
| split | 116 validation, 30 test; 0 cases span both |
| topic count | 94 single, 52 two-topic |
| length | min 65, median 369, max 779; 16 cases under 150 characters |
| leakage | 0 case texts in `train.jsonl`, `multitopic/train.jsonl` or `v2/train.jsonl` |

Design choices worth stating:

- **No case is invented.** `real` is verbatim; `condensed` deletes boilerplate
  sentences and wrapper words only; `composed` concatenates two held-out records.
- **Composition pairs records from the same split**, so provenance is unambiguous.
- **Real cases are globally disjoint** — each held-out record is consumed at most once
  across the whole suite, so quotas are met by breadth rather than repetition.
- **References are cleaned**, which is what makes eval_v3 the only suite that can
  reward C1. 0/194 reference topics carry boilerplate, against 53–64 % in v1.
- Every reference validates against the **production** schema, so a case is never
  scored against a target the model could not legally emit.

## 5. Success gates

Nine gates, each naming the suite it is measured on.

| gate | metric | target | baseline | blocking |
| --- | --- | --- | --- | --- |
| schema | eval_v3 `schema_valid` | 1.00 | 1.00 | yes |
| boilerplate | eval_v3 boilerplate leak rate | ≤ 0.02 | **0.3699** (plan said 0.675) | yes |
| topic count | eval_v3 `topic_count_accuracy` | ≥ 0.90 | **0.7808** (plan said 0.824) | yes |
| two-topic recall | multitopic `multi_topic_recall` | ≥ 0.894 | 0.8941 | yes (floor) |
| frozen domain | frozen `domain_accuracy` | ≥ 0.82 | 0.8389 | yes (floor) |
| smoke two-topic | smoke `multi_topic` | 4 | 1 | yes |
| terse single | eval_v3 category A domain accuracy | ≥ 0.70 | **null** | no |
| empty action | eval_v3 category H, non-empty predicted action | 0 | **null** | no |
| issue ROUGE | frozen `issue_rouge_l` | ≥ 0.80 | 0.9234 | no (expected to drop) |

The two eval_v3 baselines marked above were measured by scoring the frozen v2
artifact with `ml/tune/run_eval_v3.py`; they are in
`ml/data/tune/eval_v3/results/v2.json` and hashed in `ml/data/tune/v3/v2_baseline.json`.
`plan.json` carried 0.675 and 0.824 instead, which came from neither the frozen
benchmark nor eval_v3. Since both gates are blocking, the wrong baseline would have
made them either unreachable or vacuous.

Three of these need explanation:

- **`frozen domain_accuracy` floors at 0.82, not 1.0.** `ontology_audit.md` shows the
  label itself is wrong on cases the model answers correctly — the one traffic-light
  case in the test set is predicted `electricity` and is right. Chasing 1.0 means
  chasing the portal's mistakes. v1 scored 0.845, so 0.82 leaves headroom for ontology
  noise while still catching a real regression.
- **`issue_rouge_l` is a non-blocking floor at 0.80.** A drop is the *expected*
  consequence of C1. The floor exists to catch an unrelated collapse, and the gate is
  recorded so the drop is not later read as a regression.
- **Two gates have a null baseline.** Terse single-topic and empty-action behaviour
  have never been measured, because no held-out record is under 197 characters. The
  first treatment run *establishes* these numbers rather than being judged on them.

## 6. Leakage and identity policy

- Augmentation derives only from train-split content. Held-out records are used only
  to build eval_v3.
- Enforced by `ml/tests/test_eval_v3_suite.py::test_no_case_text_leaks_into_training`,
  comparing every case text against all three training streams after normalisation.
- **Identity is normalised text, never `uid`.** The portal reuses case numbers across
  years (`Б-67` is both a 2023 lift complaint and a 2026 waste-invoice complaint);
  130 uids span splits with unrelated text and 58 are duplicated inside the held-out
  pool. A `uid`-keyed check would pass while testing nothing.

## 7. Preconditions

1. **Run the control arm first** and require it to land within 1 point of every
   recorded baseline. If it does not, the harness is untrustworthy and the treatment
   number means nothing.
2. **The batch-order defect must be fixed** — it is, in `ml/tune/run_train.py`, with
   tests in `ml/tests/test_train_reproducibility.py`. Without it the two arms differ by
   batch order as well as by dataset. See `reproducibility_audit.md`.
3. ~~**Resolve the `multitopic_share` discrepancy** (0.3448 declared vs 0.368
   actual) in the rebuilt meta, not carried forward.~~ Done: the v3 meta reports
   measured values and never recomputes a share from declared factors.
4. **Treat sub-1-point differences as noise.** Batch order is now fixed, but MLX
   reductions on Apple silicon are not bit-reproducible, so small variance remains.

## 8. Budget

LoRA rank 8, alpha 20, dropout 0, 16 target modules, AdamW 1e-4, batch 2, grad accum 2,
800 iterations, seed 42 — matching v2 attempt-10 so the arms differ only in data. Target
~10500 training examples (range 9000–12500). Roughly 2× the v2 attempt-10 wall time
across two arms. No run was performed to produce these figures.

## 9. Known limits of this design

- **Eval-v3 references are transformed, not hand-annotated.** `condensed` and
  `composed` cases derive their expected topics from the same deterministic labeler
  that produced the training targets, with the boilerplate gap closed. They are
  internally consistent and schema-valid, but they are not independent human labels.
  Treat absolute scores as indicative and deltas as the signal.
- **Category K is confusable by construction** — 8 cases pair domains the ontology
  cannot separate. A low score there is expected and should not be read as regression.
- **The suite is 146 cases.** Per-category rates rest on 3–12 cases each, which is too
  few for stable per-category thresholds. The gates use suite-level metrics for that
  reason.
- **Significance testing is paired, not across seeds.** `ml/tune/compare_arms.py`
  gives a McNemar exact p-value and paired bootstrap CI per case, which controls for
  case difficulty and for the suite's internal overlap. It does **not** control for
  seed: each arm is one training run, so a delta smaller than seed-to-seed variance
  is still not attributable to the data change. Multi-seed replication is the honest
  fix and is out of budget here. Treat the paired p-values as necessary, not
  sufficient.

## 10. As built — where implementation departed from the plan

Everything below is measured on the built dataset, not estimated. The builders are
`ml/tune/build_control.py`, `ml/tune/build_v3.py`, `ml/tune/v3_changes.py`; the
composition and hashes are in `ml/data/tune/v3/*/meta.json`; the gates are the 28
tests in `ml/tests/test_v3_changes.py`.

### C1 — boilerplate: 59.2 % → 7.1 %

v2's `_ADMIN_CLAUSE` only matched a `Заявник повідомляє` template, so the plan's
premise held: 59.2 % of v1 issues still carried consent or reply boilerplate.

The plan did not specify a granularity, and **clause-level deletion was tried first
and was the wrong choice.** It leaves wreckage — `Прохання … та надати згоду` with the
tail cut is a target no citizen wrote — and a boilerplate tail stripped from every
example collapses the distinct-issue count. Sentence-level deletion is correct here
because a consent or reply-delivery sentence is a whole non-issue and can be removed
outright while the issue sentence survives.

Two patterns had to be withdrawn after reading real rows:

- `в телефонному режимі` is boilerplate in `Відповідь в телефонному режимі` but **is
  the substance** of a complaint about how staff spoke to the citizen. It is now
  removed only inside the reply-delivery phrase.
- `Відповідь заявниці.` occurs 1828 times and the `BOILER_SENT` probe misses it
  entirely: Ukrainian turns заявник's `к` into `ц`, so `заявник\w*` cannot match
  `заявниці`. A leak rate computed from that probe alone is an undercount.

**230 records were dropped** because they were *only* boilerplate — a consent block
with no complaint. Keeping them would train the model to emit an empty `issue`. This
is a deliberate data loss on integrity grounds, not a way of hitting a target share,
and the count is in `meta.json` so the loss is visible rather than silent.

### C2 — action: whole-sentence redundancy → 0.0, non-empty 48.6 % → 17.8 %

The action is the request clause, not a whole sentence. `_redact_pii` treats any
capitalised word as a person name, which ate sentence-initial imperatives
(`Усунути порив` → `""`), so the verb is matched against the raw text first.

`requested_action` being a substring of `issue` **stays high by construction** — a
specific request is genuinely part of the issue. The 98.1 % figure in §1 is a labeler
defect only in the *whole-sentence* form. Both rates are therefore reported, and the
frozen 329 `action_presence_match` gate (0.9909) is expected to fall for the same
reason `issue_rouge_l` does.

### C3 — terse pool is 246, not 178 or 928

`plan.json` claims 178; `short_note_analysis.md` claims 928. Measured against the
current `_note_kernel`: 841 records under 150 characters, of which 246 yield a usable
kernel ≥25 characters, 393 non-empty and 448 empty. At 8× that is 1872 rows; 12 pool
records whose kernel C1 empties are skipped, so 234 units × 8. A test pins the
measured count, so a later change to `_note_kernel` cannot silently resize the
augmentation.

### C4 — real rows reach 21.6 %, not 34.5 %

Only 22 of the 979 multitopic records are hand-annotated. v2 upsampled 12×/3×
(8.4 % real rows); v3 uses 24×/2× → 528 real of 2442 = **21.6 %**. The plan's 34.48 %
came from declaring real weights without upsampling; the 36.80 % that contradicted it
was itself a recomputation of a declared figure. The v3 meta reports only measured
values.

**This reweights 22 real records. It does not add real data**, and the report must not
describe C4 as though it did.

### Composition and other deviations

- Treatment is **9468** rows: 5154 single + 1872 terse + 2442 multitopic (terse
  19.8 %, multitopic 25.8 %), against the plan's ~10500.
- **Multitopic rows are reweighted but not passed through C1/C2.** C1 and C2 are
  applied to the single-topic and terse streams only. If the intent was C1–C4 on all
  training topics, this is a gap, and it is the one place where the built dataset does
  not match the plan's title.
- The control was launched with `--steps-per-report 25` against v2's recorded 10.
  This changes logging cadence, not the trained weights, but it is a visible
  difference from the frozen v2 recipe and is recorded in the control's
  `run_config.json`.
- `ml/tune/run_eval.py` and `run_eval_multitopic.py` gained `--tag`. Without it the
  two arms would have written to the same `{mode}.json` and silently overwritten the
  v2 baseline.
- `ml/tune/fuse_v3.py` inherits the tokenizer and chat template from the verified v2
  artifact rather than from the base repo, because `mlx_lm.fuse` copies the template
  from the base repo — which is how v2's fused directory ended up needing a
  hand-patched template that honours `enable_thinking`.
