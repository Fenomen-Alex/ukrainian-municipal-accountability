# Model experiments — v1, v2 and the v3 lineage

A single-page history of every training arm in this repository, why **v2 is the
canonical model**, and why nothing in the v3 lineage was released.

This is a *history* document. For the current status see
[FINAL_STATUS.md](FINAL_STATUS.md); for the full evidence and reasoning see
[FINAL_MODEL_ASSESSMENT.md](FINAL_MODEL_ASSESSMENT.md), which is authoritative
whenever the two disagree.

**Model development is frozen.** The decision and its prerequisites are in
`FINAL_MODEL_ASSESSMENT.md` §8 (Option A). Nothing below is a plan.

## The arms

| Arm | Adapter directory | Outcome |
|---|---|---|
| v1 | `qwen3-8b-lora/` | Superseded by v2. Report: `ml/reports/finetune_report.md`. |
| **v2 (attempt10, fused)** | `qwen3-8b-lora-v2-attempt10-fused/` | **Canonical public model.** |
| v3 control | `qwen3-8b-lora-v3-control/` | Baseline for the v3 comparison; no independent claim. |
| previous-v3 | `qwen3-8b-lora-v3-treatment/` | Rejected: lost output discipline. |
| corrected-v3 | `qwen3-8b-lora-v3-corrected/` | Rejected: best content arm, action prior collapsed. |
| v3.2 | `qwen3-8b-lora-v3-2/` | Rejected: solved boilerplate, regressed elsewhere. |

(`v2` also has unfused `qwen3-8b-lora-v2/` and `qwen3-8b-lora-v2-fused/`
intermediates. Use the `-attempt10-fused` directory; it is the published one.)

Label mapping used by `ml/data/tune/cross_arm_matrix.json`:
`v3-treatment` → previous-v3, `v3-corrected` → corrected-v3, `v3-2` → v3.2.

## What each generation tried

**v1 → v2.** v2 was the first arm that met the output contract under greedy
decoding, and it remains the only arm simultaneously strong on action presence
(0.9909), hallucination (0.0152), issue ROUGE-L (0.9234) and domain macro-F1
(0.7596). Its two measurable weaknesses are boilerplate (0.3699) and topic
count (0.7808) — precisely what the v3 lineage was built to fix.

**The v3 lineage** added two interventions:

* **C1 — output discipline.** Target the boilerplate form and the topic-count
  gate. It worked: boilerplate fell 0.3699 → 0.2260 → 0.0068 → **0.0000** across
  the lineage.
* **C2 — action prior.** Flatten the model's tendency to copy a requested
  action. It backfired: frozen action presence match collapsed from v2's
  **0.9909** to 0.5471 / 0.5775 / 0.5653, i.e. the v3 arms *lost* the ability
  to tell "a remedy was requested" from "no remedy was requested".

## Why none of the v3 arms shipped

Blocking gates, from `FINAL_MODEL_ASSESSMENT.md` §2:

| gate | requirement | v2 | previous-v3 | corrected-v3 | v3.2 |
|---|---|---|---|---|---|
| schema validity (146) | 1.00 | 0.9863 | 0.9795 | 0.9589 | 0.9110 |
| boilerplate (146) | ≤ 0.02 | 0.3699 | 0.2260 | 0.0068 | **0.0000** |
| topic count (146) | ≥ 0.90 | 0.7808 | 0.8699 | 0.8014 | 0.7740 |
| MT85 recall | ≥ 0.894 | 0.8941 | 0.9412 | 0.9176 | 0.9059 |
| smoke two-topic (of 4) | 4 | 1 | 1 | 0 | 2 |

**No arm passes topic count or the two-topic smoke gate**, and the two v3 arms
that fixed boilerplate lost schema validity. corrected-v3 was the best *content*
extractor of the four (terse 1.00 / 0.8571, object exact 0.9848, best MT
domain-set exact) but paid for it with the collapsed action prior and the worst
macro-F1 (0.6775). v3.2 won boilerplate outright and still finished worst on
schema (0.9110), worst on expected-two (0.3846), and below corrected-v3 on its
own target category.

The arms trade off against each other rather than ranking; that trade-off is the
finding. Read the full table in `FINAL_MODEL_ASSESSMENT.md` §2 — the rows above
are the blocking subset, not the evidence.

### Three confounders that inflate the apparent v3 regression

Documented in `FINAL_MODEL_ASSESSMENT.md` §4–§6 and worth knowing before
re-reading the table:

1. **The `""` inputs.** Five eval cases contain redaction-created empty quotes.
   They cost both v3 arms 5 schema failures on an input the model never sees at
   inference. Fixed at the prompt boundary rather than by training — see
   [QUOTE_ARTEFACT.md](QUOTE_ARTEFACT.md).
2. **The action prior (C2).** Most of the frozen action-regression is C2's
   doing, not a content regression.
3. **Reference style.** v2's ROUGE-L and macro-F1 advantages are partly
   reference-style bias: v2 was trained to match the references.

## The known `ev3-083` defect

`ev3-083` is a decoding defect carried into every arm including v2. A 50-prompt
probe at char 125 reproduced it in 50/50 previous-v3 generations and 0/50 for
corrected-v3 and v3.2 — v2 is unaffected (25/25 valid). **v2 therefore still
carries `ev3-083`; it is not repaired in the canonical model.** The completed
decoding probe found no configuration that fixes it without changing topic
behaviour (`FINAL_MODEL_ASSESSMENT.md` §7.5), so it remains a documented
limitation rather than a known workaround.

## Reproducing these numbers

```bash
.venv/bin/python -m ml.tune.verify_final_assessment   # 380 claim-level checks
.venv/bin/python -m ml.tune.verify_v3_2_report         # 197 checks, historical
.venv-mlx/bin/python -m ml.tune.verify_repro           # reproducibility
```

These need the gitignored adapters and generated eval corpora. A fresh clone
cannot run them unaided; see the provenance section of
`FINAL_RELEASE_READINESS.md`.