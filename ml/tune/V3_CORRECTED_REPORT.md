# v3 CORRECTED — full-corpus treatment run and evaluation report

**Arm:** `v3-corrected` (corrected v3 treatment)
**Status:** trained, fused, evaluated on all four suites. **Gates: FAIL** (3 blocking + 2 advisory fail, 3 blocking pass).
**Headline:** both prior hypotheses are confirmed — the R1 regression is gone and boilerplate is essentially eliminated — but the corrections made the model markedly *more single-topic*, which costs multitopic and topic-count accuracy and introduces a new unescaped-quote failure mode.

Everything below is a faithful record of one run on one machine. No dataset change, no C3′, no recipe change, no evaluator change, no control retrain, no publication.

---

## 1. What this run was

The previous v3 arm (`v3-treatment`) was trained on a corpus built with three data bugs
(R1, R2, R3 — see `ml/tune/V3_1_ANALYSIS.md`). The corrections were committed in `e8f336c`
and analysed in `e9094a6`. This run retrains on the **corrected** corpus using the
**identical** recipe, so the only variable is the training data.

## 2. Provenance

| item | value |
|---|---|
| training start commit | `70cb1e8d1dda279ff99cbc44e46c6242ffbfd7d7` (`run_v3_corrected.sh`) |
| tree at start / at fusion | clean / `git_dirty: false` |
| corpus | `ml/data/tune/v3/treatment/train.jsonl` |
| corpus sha256 | `22e85d80b7a7abc12d0629ffc41fd125a2b718125e01f62fe9cbcd92815ef9f2` |
| meta sha256 | `5db70ce863a966ac6151e75b106874b5438e9fdac7734ef81d8d081f0fc026f7` |
| rows | 9,447 = 5,157 single / 1,872 terse / 2,418 multitopic |
| final adapter sha256 | `11582e036e5f4de4b910ec387fa8aa8fa24d6d705935a31af34e61676661fe7b` |
| driver | `ml/tune/run_v3_corrected.sh` |
| eval driver | `ml/tune/run_v3_corrected_eval.sh` |
| log | `ml/data/tune/v3/logs/corrected_train.log` |

## 3. Exact configuration

Unchanged from the previous v3 run and from the v3.1 analysis recipe.

```
base            mlx-community/Qwen3-8B-4bit
LoRA            rank 8, scale 20.0, dropout 0.0, target 16 layers
trainable       0.118 % (9.699 M / 8190.735 M)
batch size      2        grad accumulation 2 (effective 4)
iterations      4724     (= ceil(9447 / 2), one epoch, 100 % coverage)
optimizer       AdamW    lr 1e-4    lr_schedule none    seed 42
max seq         2048     mask_prompt true    grad checkpoint on
save_every      500      report 25   eval cadence 50   val_batches 10
serving         inherited from the v2 canonical artifact, thinking off
```

## 4. Runtime, loss, checkpoints

| | |
|---|---|
| start | 2026-09-30 00:26:49 |
| complete | 2026-09-30 22:07:53 |
| **wall clock** | **21 h 41 m** (≈16.5 s/iter, ~0.060 it/s) |
| trainable peak memory | 10.33 GB |
| train loss | 0.340 @ iter 25 → 0.010 @ iter 4724 |
| checkpoints | `0000500 … 0004500`, 9 of 9 expected, **none missing** |
| final weights | `adapters.safetensors` present |
| log final marker | present |
| `training_state` | `complete: true`, `missing: []` |

Verified by `ml.tune.training_state --iters 4724 --save-every 500 --describe`.
The validation-set warning ("Validation set not found or empty") is identical to the
previous v3 run and to the recipe; it is a known, accepted property of this setup, not a
run anomaly.

## 5. Fusion

`.venv-mlx/bin/python -m ml.tune.fuse_v3 --adapter ml/data/tune/adapters/qwen3-8b-lora-v3-corrected --out ml/data/tune/adapters/qwen3-8b-lora-v3-corrected-fused`

| | |
|---|---|
| fused modules | 112 |
| dequantize | `false` |
| `config_matches_v2` | **true** |
| inherited from | `ml/data/tune/adapters/qwen3-8b-lora-v2-attempt10-fused` |
| inherited `chat_template.jinja` | `87a2728cb8dc9fe424d624542f6060ec05a1d285ebbec578bb078900e33396b5` |
| inherited `tokenizer.json` | `be75606093db2094d7cd20f3c2f385c212750648bd6ea4fb2bf507a6a4c55506` |
| inherited `tokenizer_config.json` | `d93ac1a2c7adb9ad022f354d822f1f0f57e32e62ef2989e902782d6ab896acfe` |

All ten per-checkpoint adapter sha256 values are recorded in `fuse_manifest.json`.
Independent re-verification: `config.json`, `chat_template.jinja`, `tokenizer.json` and
`tokenizer_config.json` in the new fused directory are **byte-identical** to the v2
canonical artifact. The serving contract is therefore unchanged, as required.

The v2 canonical artifact and the previous v3 artifacts were not modified.

## 6. Results

`v2` = the shipped v2 model. `v3-treatment` = previous v3. `v3-corrected` = this run.

### 6.1 eval_v3, 146 cases (cleaned labels)

| metric | v2 | v3-treatment | v3-corrected |
|---|---|---|---|
| json_parse_rate | 0.9932 | 0.9795 | 0.9589 |
| schema_validity_rate | 0.9863 | 0.9795 | 0.9589 |
| topic_count_accuracy | 0.7808 | 0.8699 | **0.8014** |
| domain_set_exact | 0.6301 | 0.6712 | **0.7055** |
| domain_set_precision | 0.8356 | 0.7808 | **0.8836** |
| domain_set_recall | 0.7637 | 0.7397 | **0.8048** |
| duplicate_domain_rate ↓ | 0.0137 | 0.0068 | 0.0068 |
| out_of_enum_rate ↓ | 0.0068 | 0.0000 | 0.0000 |
| boilerplate_leak_rate ↓ | 0.3699 | 0.2260 | **0.0068** |
| action_redundancy_rate ↓ | 0.2055 | 0.0890 | 0.1575 |
| issue_rouge_l_best | 0.6719 | 0.7135 | **0.7545** |
| mean predicted topics | 1.2123 | 1.2192 | 1.1164 |
| mean expected topics | 1.3562 | 1.3562 | 1.3562 |

Subsets:

| subset | n | metric | v2 | v3-treatment | v3-corrected |
|---|---|---|---|---|---|
| terse_categories_ABC | 26 | topic_count | 0.7692 | 0.7692 | **0.8462** |
| | | domain_set_exact | 0.5769 | 0.2308 | **0.6923** |
| | | domain_set_recall | 0.7500 | 0.3654 | **0.8077** |
| | | issue_rouge | 0.6840 | 0.5874 | 0.6568 |
| terse_text_under_150 | 14 | topic_count | 0.7857 | 0.9286 | 0.8571 |
| | | domain_set_exact | 0.6429 | 0.1429 | **0.8571** |
| | | domain_set_recall | 0.7500 | 0.1786 | **0.9286** |
| | | issue_rouge | 0.7133 | 0.7157 | 0.7385 |
| multi_topic_categories | 44 | topic_count | 0.4318 | 0.6136 | **0.4091** |
| | | domain_set_exact | 0.3409 | 0.3864 | 0.2727 |
| | | domain_set_recall | 0.6477 | 0.5568 | 0.5682 |
| | | boilerplate_leak ↓ | 0.0455 | 0.0227 | **0.0000** |
| empty_action_category_H | 12 | topic_count | 0.9167 | 1.0000 | 1.0000 |
| | | domain_set_exact | 0.7500 | 0.9167 | **1.0000** |
| | | domain_set_recall | 0.8333 | 0.9167 | **1.0000** |
| | | boilerplate_leak ↓ | 0.9167 | 0.5000 | **0.0000** |
| | | issue_rouge | 0.6614 | 0.8900 | **0.9331** |

### 6.2 Multitopic, 85 cases

| metric | v2 | v3-treatment | v3-corrected |
|---|---|---|---|
| json_parse_rate | 0.9882 | 1.0000 | 1.0000 |
| schema_validity_rate | 0.9882 | 1.0000 | 1.0000 |
| multi_topic_recall | 0.8941 | **0.9412** | 0.9176 |
| topic_count_accuracy | 0.8588 | 0.9412 | 0.9176 |
| domain_set_exact | 0.6824 | 0.7765 | 0.7529 |
| domain_set_precision | 0.8961 | 0.9176 | 0.9176 |
| domain_set_recall | 0.8353 | 0.8824 | 0.8765 |
| issue_rouge_l_best | 0.7988 | 0.7365 | 0.7258 |

Per-case topic-count behaviour (all 85 cases have `target_n = 2`):
v3-treatment under-emits on 5, over-emits on 0, exact on 80.
v3-corrected under-emits on **7**, over-emits on 0, exact on **78**.
There is no over-emission in either arm; the failure mode is one-sided.

### 6.3 Frozen 329 (v1 weak labels)

> `ml/data/tune/eval/` is git-ignored and no frozen result is tracked in this repository
> (true for v2 and previous v3 as well), so these numbers live in this report only.

| metric | v2 | v3-treatment | v3-corrected |
|---|---|---|---|
| json_parse_rate | 1.0000 | 1.0000 | 1.0000 |
| schema_validity_rate | 0.9970 | 0.9970 | **1.0000** |
| domain_accuracy | 0.8389 | 0.8602 | **0.8663** |
| domain_macro_f1 | 0.7596 | 0.7556 | **0.6775** |
| issue_rouge_l | 0.9234 | 0.6611 | 0.6610 |
| object_exact | 0.9422 | 0.9696 | **0.9848** |
| object_token_overlap | 0.9574 | 0.9773 | **0.9848** |
| object_mismatch_discriminator ↓ | 0.0334 | 0.0182 | **0.0091** |
| action_presence_target | 0.5015 | 0.5015 | 0.5015 |
| action_presence_pred | 0.4985 | 0.1337 | 0.1702 |
| action_presence_match | 0.9909 | 0.5471 | 0.5775 |
| hallucination_rate ↓ | 0.0152 | 0.0456 | 0.0578 |
| multi_topic_rate | 0.0365 | 0.0122 | **0.0000** |

`multi_topic_rate = 0.0000` is the single most diagnostic number in this report: on the
frozen set the corrected model **never** emits two topics.

### 6.4 Smoke, 20 cases

| | v3-treatment | v3-corrected |
|---|---|---|
| two-topic correct | 0 of 4 | 0 of 4 |

| case | expected | v3-treatment | v3-corrected |
|---|---|---|---|
| smoke-13 | lighting, roads | electricity, roads | electricity |
| smoke-14 | waste_management, animals | sanitation | sanitation |
| smoke-15 | elevators, sewage_basement | housing | housing |
| smoke-16 | landscaping, traffic_signs | sanitation | sanitation |

## 7. Gates

`.venv-mlx/bin/python -m ml.tune.gates_v3 --arm v3-corrected` → **FAIL**

| gate | suite | target | v2 | v3-treatment | v3-corrected | blocking | result |
|---|---|---|---|---|---|---|---|
| boilerplate | eval_v3 | ≤ 0.02 | 0.3699 | 0.2260 | **0.0068** | yes | **pass** |
| two_topic_recall | multitopic | ≥ 0.894 | 0.8941 | 0.9412 | **0.9176** | yes | **pass** |
| frozen_domain | frozen 329 | ≥ 0.82 | 0.8389 | 0.8602 | **0.8663** | yes | **pass** |
| terse_single | eval_v3 cat A | ≥ 0.70 | 0.80 | 0.0 | **1.0000** | no | pass |
| schema | eval_v3 | = 1.00 | 0.9863 | 0.9795 | 0.9589 | yes | **FAIL** |
| topic_count | eval_v3 | ≥ 0.90 | 0.7808 | 0.8699 | 0.8014 | yes | **FAIL** |
| smoke_two_topic | smoke | 4 of 4 | 1 of 4 | 0 of 4 | 0 of 4 | yes | **FAIL** |
| empty_action | eval_v3 cat H | 0 invented | 0 | 1 | 1 | no | **FAIL** |
| issue_rouge | frozen 329 | ≥ 0.80 | 0.9234 | 0.6611 | 0.6610 | no | **FAIL** |

Three blocking gates pass. All three blocking failures are inherited from the previous v3
run or worsened by it: `smoke_two_topic` was already 0 of 4, `topic_count` was already
below 0.90, and `schema` was already below 1.00. No blocking gate regressed from
*passing* to *failing*.

## 8. Paired comparison

`ml.tune.compare_arms`, n = 146, McNemar exact on binary metrics, 10k paired bootstrap CI
on continuous metrics.

### 8.1 v3-corrected vs v3-treatment (the R1/R2/R3 effect)

| metric | v3-treatment | v3-corrected | delta | test |
|---|---|---|---|---|
| boilerplate_leak_rate ↓ | 0.2260 | 0.0068 | −0.2192 | **p < 0.0001 (significant)** |
| domain_set_precision | 0.7808 | 0.8836 | +0.1027 | **+0.0377 … +0.1678 (significant)** |
| issue_rouge_l_best | 0.7135 | 0.7545 | +0.0410 | **+0.0113 … +0.0718 (significant)** |
| domain_set_recall | 0.7397 | 0.8048 | +0.0651 | +0.0000 … +0.1301 (borderline) |
| domain_set_exact | 0.6712 | 0.7055 | +0.0342 | p = 0.4869 (n.s.) |
| duplicate_domain_rate ↓ | 0.0068 | 0.0068 | 0.0000 | p = 1.0000 (n.s.) |
| out_of_enum_rate ↓ | 0.0000 | 0.0000 | 0.0000 | p = 1.0000 (n.s.) |
| topic_count_accuracy | 0.8699 | 0.8014 | −0.0685 | p = 0.0639 (n.s., trending down) |
| action_redundancy_rate ↓ | 0.0890 | 0.1575 | **+0.0685** | **p = 0.0213 (significant regression)** |
| json_parse_rate | 0.9795 | 0.9589 | −0.0205 | p = 0.3750 (n.s.) |
| schema_validity_rate | 0.9795 | 0.9589 | −0.0205 | p = 0.3750 (n.s.) |

### 8.2 v3-corrected vs v2

| metric | v2 | v3-corrected | delta | test |
|---|---|---|---|---|
| boilerplate_leak_rate ↓ | 0.3699 | 0.0068 | −0.3630 | **p < 0.0001 (significant)** |
| issue_rouge_l_best | 0.6719 | 0.7545 | +0.0826 | **+0.0392 … +0.1251 (significant)** |
| domain_set_exact | 0.6301 | 0.7055 | +0.0753 | **p = 0.0433 (significant)** |
| domain_set_precision | 0.8356 | 0.8836 | +0.0479 | −0.0137 … +0.1096 (n.s.) |
| domain_set_recall | 0.7637 | 0.8048 | +0.0411 | −0.0137 … +0.0959 (n.s.) |
| topic_count_accuracy | 0.7808 | 0.8014 | +0.0205 | p = 0.6476 (n.s.) |
| action_redundancy_rate ↓ | 0.2055 | 0.1575 | −0.0479 | p = 0.2962 (n.s.) |
| json_parse_rate | 0.9932 | 0.9589 | −0.0342 | p = 0.1250 (n.s.) |
| schema_validity_rate | 0.9863 | 0.9589 | −0.0274 | p = 0.2891 (n.s.) |
| duplicate_domain_rate ↓ | 0.0137 | 0.0068 | −0.0068 | p = 1.0000 (n.s.) |
| out_of_enum_rate ↓ | 0.0068 | 0.0000 | −0.0068 | p = 1.0000 (n.s.) |

## 9. Error taxonomy

| failure | v2 | v3-treatment | v3-corrected |
|---|---|---|---|
| **no error** | 28 | 64 | **86** |
| action_copy | 18 | 6 | 16 |
| bad_schema | 1 | 0 | 0 |
| boilerplate_leak | 46 | 28 | **1** |
| domain_missed | 20 | 28 | **13** |
| duplicate_domain | 2 | 1 | 1 |
| not_json | 1 | 3 | 6 |
| topic_count | 30 | 16 | 23 |

The corrected arm produces the **fewest defective cases of the three** (86/146 = 58.9 %
fully clean, versus 64 and 28), and the largest share of the remaining defects are
`action_copy` and `topic_count` — both consistent with a model that has become terser and
more single-topic, not with a model that has become sloppier.

## 10. Terse per-case (category A, n = 10)

| id | expected | v3-treatment | v3-corrected | v2 |
|---|---|---|---|---|
| ev3-001 | sanitation | other | **sanitation** | roads |
| ev3-002 | sanitation | other | **sanitation** | sanitation |
| ev3-003 | sanitation | other | **sanitation** | sanitation |
| ev3-004 | sanitation | other | **sanitation** | sanitation |
| ev3-005 | sanitation | other | **sanitation** | sanitation |
| ev3-006 | construction | other | **construction** | sanitation |
| ev3-007 | housing | other | **housing** | housing |
| ev3-008 | roads | other | **roads** | roads |
| ev3-009 | roads | other | **roads** | roads |
| ev3-010 | sanitation | other | **sanitation** | sanitation |
| **exact** | | **0/10** | **10/10** | 8/10 |

The previous arm returned the literal domain `other` on all ten terse single-topic cases.
That was the R1 signature. It is completely gone, and the corrected arm beats v2 on this
subset (10/10 vs 8/10).

## 11. Boilerplate

One boilerplate leak remains in 146 cases: `ev3-117` (category Q), predicted `sanitation`.
In category H the leak rate is 0.0000 (was 0.5000 for previous v3, 0.9167 for v2), and in
the multi-topic subset it is 0.0000 (was 0.0227). The eval_v3 leak rate of 0.0068 is
1/146, comfortably inside the ≤ 0.02 gate.

## 12. Schema / parse failures

6 of 146 cases are unparseable (previous v3: 3). No case is schema-invalid
(`bad_schema` = 0 for both v3 arms), so this is a JSON-emission failure, not an
out-of-enum failure. Diagnosis from the raw generations (deterministic, temperature 0):

| case | category | cause |
|---|---|---|
| ev3-016 | B | unescaped `"` inside the `issue` string, break at char 125 |
| ev3-024 | C | same, char 125 |
| ev3-032 | D | same, char 125 |
| ev3-046 | F | same, char 125 |
| ev3-052 | G | same, char 125 |
| ev3-083 | K | repetition degeneration (`по вул. по вул. …`) then truncation at max_tokens 800 |

All five quote failures emit the same fragment, `На дзвінки працівники "" не відповідають`.
**This is not inherited from the training data:** the corrected corpus contains 0 rows
with that fragment and 0 target `issue`/`object`/`requested_action` values containing a
bare double quote. The model manufactures the empty quoted span itself. The cause is
therefore unresolved and is a genuine new finding, not a data artefact.

## 13. Why topic count fell

Predicted topic-count distribution on eval_v3 (expected: 94 single, 52 two):

| arm | 0 topics | 1 topic | 2 topics | 3 topics |
|---|---|---|---|---|
| v2 | 1 | 114 | 30 | 1 |
| v3-treatment | 3 | 108 | 35 | 0 |
| v3-corrected | **6** | **117** | **23** | 0 |

MT85 under-emission rose 5 → 7, and frozen `multi_topic_rate` fell to 0.0000. The
corrected model is systematically more reluctant to emit a second topic. Since the gates
were already failing on `topic_count` and `smoke_two_topic` before this run, this is an
aggravation of a pre-existing weakness rather than a new one — but it is the reason the
blocking `topic_count` gate got worse (0.8699 → 0.8014).

## 14. Hypotheses

| # | hypothesis | verdict | evidence |
|---|---|---|---|
| H1 | R1 (terse labels forced to `other`) caused the terse collapse | **CONFIRMED** | cat A exact 0/10 → 10/10; under-150 domain exact 0.143 → 0.857; every `other` output gone |
| H2 | R2/R3 drove the boilerplate leak | **CONFIRMED** | 0.2260 → 0.0068, p < 0.0001; taxonomy 28 → 1; boilerplate gate now passes |
| H3 | Removing boilerplate costs the multitopic arm nothing | **PARTIALLY REFUTED** | MT recall 0.9412 → 0.9176, still above the 0.894 floor so the gate holds, but eval_v3 multi-topic topic_count 0.6136 → 0.4091 and under-emission rose |
| H4 | The category-H invented action was caused by R2 | **REFUTED** | still exactly 1 invented action (`ev3-062`), unchanged from previous v3 |
| H5 | The schema/parse failures relate to R1/R2/R3 | **NOT SUPPORTED** | 3 → 6 is a *new* failure mode (unescaped quote ×5, truncation ×1) with no matching artefact in the corpus |
| H6 | The frozen-329 rouge drop is boilerplate removal | **REFUTED** | 0.6611 → 0.6610, i.e. unchanged by the corrections; the drop was already present in previous v3 |
| H7 | Correcting R1 would fix smoke two-topic | **REFUTED** | still 0 of 4; on smoke-13 the second topic is now dropped where previous v3 kept it |

## 15. Secondary regressions to carry forward

These are new relative to previous v3 and are not explained by the above:

- `action_redundancy_rate` 0.0890 → 0.1575, **p = 0.0213**; `action_copy` 6 → 16.
- `domain_macro_f1` on frozen 329: 0.7556 → 0.6775.
- `hallucination_rate` on frozen 329: 0.0456 → 0.0578.
- `not_json` 3 → 6 (section 12).

`action_presence_match` on frozen 329 (0.5775) remains far below v2 (0.9909). That is a
v3-inherited weakness, not new here.

## 16. Overall assessment

The corrections did what they were supposed to do, and they did it decisively. The two
bugs they targeted are fixed: terse single-topic behaviour went from 0/10 to 10/10, and
boilerplate leakage went from 0.2260 to 0.0068 with p < 0.0001. The corrected arm is the
best of the three on domain-set precision, domain-set recall, domain-set exact, issue
ROUGE on eval_v3, category-A terse, category-H, frozen domain accuracy, frozen schema
validity, frozen object exact, and the number of fully clean cases.

It also has a real, new cost: it is markedly more single-topic than either predecessor,
which is what keeps the `topic_count` and `smoke_two_topic` blocking gates failing and
what produces the new unescaped-quote failures. The previous arm's terse regression was
the more severe defect and it is now gone; the price is a sharper version of a
pre-existing multitopic weakness.

**This arm is not shippable as-is** — three blocking gates fail — but it is strictly
better than the previous v3 treatment on the primary hypothesis and on the primary
boilerplate gate, and it is now the best starting point for the next iteration.

## 17. Suggested next steps

1. Attack the under-emission directly, since it is now the dominant failure: inspect the
   1,872-row terse and 2,418-row multitopic slices of the corrected corpus for a
   post-R2/R3 shortening that made multi-topic targets degenerate.
2. Find the source of the `""` fragment in generation; it is absent from the corpus, so it
   is a model-side decoding artefact worth isolating on a single case.
3. Re-examine `ev3-062` (the surviving category-H invented action) directly — it survived
   R2 and R3 untouched.
4. Only after (1) is settled, revisit `topic_count` and `smoke_two_topic`.

## 18. Reproduction

```bash
# train (21 h 41 m on an M-series Mac)
bash ml/tune/run_v3_corrected.sh

# fuse
.venv-mlx/bin/python -m ml.tune.fuse_v3 \
  --adapter ml/data/tune/adapters/qwen3-8b-lora-v3-corrected \
  --out    ml/data/tune/adapters/qwen3-8b-lora-v3-corrected-fused

# evaluate (1 h 23 m)
bash ml/tune/run_v3_corrected_eval.sh

# gates and paired comparison
.venv-mlx/bin/python -m ml.tune.gates_v3 --arm v3-corrected
.venv-mlx/bin/python -m ml.tune.compare_arms --arms v2 v3-treatment v3-corrected --markdown
```

New tracked result files: `ml/data/tune/eval_v3/results/v3-corrected.json`,
`ml/data/tune/multitopic/results/v3-corrected.json`,
`ml/data/tune/smoke/results/v3-corrected/smoke_results.jsonl`.
Frozen-329 output (`ml/data/tune/eval/v3-corrected.json`) is git-ignored by repository
policy, as are the v2 and previous-v3 frozen outputs.
