# v3 experiment design

Machine-readable form: `ml/data/tune/v3/plan.json`. Rationale: `ml/data/tune/v3/README.md`.
**Nothing here has been trained.** No adapter, fused model, or training run was produced.

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
| **treatment** | `ml/data/tune/v3/train.jsonl` | ~10500 | C1–C4 |

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
| boilerplate | eval_v3 boilerplate leak rate | ≤ 0.02 | 0.675 | yes |
| topic count | eval_v3 `topic_count_accuracy` | ≥ 0.90 | 0.824 | yes |
| two-topic recall | multitopic `multi_topic_recall` | ≥ 0.894 | 0.894 | yes (floor) |
| frozen domain | frozen `domain_accuracy` | ≥ 0.82 | 0.839 | yes (floor) |
| smoke two-topic | smoke `multi_topic` | 4 | 1 | yes |
| terse single | eval_v3 category A domain accuracy | ≥ 0.70 | **null** | no |
| empty action | eval_v3 category H, non-empty predicted action | 0 | **null** | no |
| issue ROUGE | frozen `issue_rouge_l` | ≥ 0.80 | 0.923 | no (expected to drop) |

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
3. **Resolve the `multitopic_share` discrepancy** (0.3448 declared vs 0.368 actual) in
   the rebuilt meta, not carried forward.
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
- **No significance testing.** With one control and one treatment arm, a difference
  cannot be given a p-value. Multi-seed replication is the honest way to tighten this
  and is out of budget here.
