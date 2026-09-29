# V3 report: full-corpus treatment vs v2 — NOT PUBLISHED

**Date:** 2026-09-29 · **Arm:** `qwen3-8b-lora-v3-treatment` (C1–C4, full exposure)
**Verdict summary:** the treatment is a large, statistically significant
improvement over the published v2 on the primary adversarial suite, but it
still misses four of six *blocking* publish gates. Following the policy in
`ml/tune/gates_v3.py` and `ml/reports/v3_experiment_design.md`, **v3 is not
published and the published v2 artifact is unchanged.**

This report supersedes the negative result of the 800-step control run. The
previous run's failure was traced to *partial corpus exposure* (18.8 % of
training examples); this run re-trains with every example shown exactly once,
which confirms the diagnosis: the treatment now beats v2 — decisively on the
primary suite — while still not meeting the publish bar.

---

## 1. The context this run replaces

The short v3 run (`800` iterations, batch 2) drew its batch order from a random
permutation and showed the model only `1600/8519 ≈ 19 %` of the corpus. Its
control scored far below v2 on the byte-identical suite (eval_v3
`topic_count_accuracy` 0.7808 → ~0.66, no-error cases 28 → 11), i.e. the model
memorised a random slice and regressed elsewhere. See `ml/reports/V3_REPORT.md`
and `v3_experiment_design.md` §11.

## 2. Dataset

Built by `ml/tune/build_v3.py` with C1–C4 applied (see `v3_experiment_design.md`
§5–8):

- **train** `ml/data/tune/v3/treatment/train.jsonl` — **9 448 rows** =
  5 154 (v2 corpus, C1+C2 cleaned) + 1 872 (terse, C3: 246 units × 8) +
  2 422 (multitopic, C4: 1 075 real × 2 + 272 synthetic × N). Terse share
  0.1981, multitopic share 0.2564, real multitopic 21.8 %.
- **validation** 1 124 and **test** 329 rows — byte-identical to the v2 splits
  (sha256 `c7cea07ec00f60e8…` / `49a7c35185586849…`), so evaluations stay
  directly comparable to the published baselines.
- train sha256 `ce56c9588fa4c8dd…`; official leakage/schema/determinism tests
  pass (311 passed, 3 skipped, 4 subtests).
- No `valid.jsonl` live-validation set — deliberate parity with how v2 and the
  control were trained (mlx-lm warns and continues).

## 3. Training configuration (full-corpus run)

| knob | value |
|---|---|
| base model | `mlx-community/Qwen3-8B-4bit` |
| LoRA | rank 8, scale 20.0, dropout 0, layers 16 (~9.7 M trainable, 0.118 %) |
| batch / grad-accum | 2 / 2 (effective 4) |
| **iterations** | **4 724** = `ceil(9448/2)` → exactly one epoch |
| **corpus coverage** | **100 %** — every train row emitted once (4 724 unique length-sorted batches) |
| seed | 42 (seeded batch order, `install_seeded_batch_order`) |
| learning rate | 1e-4 adamw, no schedule |
| max-seq / mask-prompt / grad-checkpoint | 2048 / True / True |
| save-every | 500 (9 numbered checkpoints, one final) |
| wall time | 21.1 h (2026-09-28 22:21:39 → 2026-09-29 19:27:27), 2 214 iterations saved per checkpoint; train loss 0.251 (iter 25) → 0.015 (4724) |
| final adapter | `ml/data/tune/adapters/qwen3-8b-lora-v3-treatment/adapters.safetensors`, sha256 `e80b16da0fe048db600625014d5f9ce1aa97d80ce29c813e561615bd55085a60` |
| reliability | numbered adapter checkpoints (~38 MB each) every 500 iters + `training_state` completion check (continuous checkpoints *and* end-of-run log marker); the driver auto-resumes from the highest checkpoint after an infrastructure crash, and the seeded batch order replays the prefix identically |

## 4. Fusion

`ml/tune/fuse_v3.py` fused the final adapter into
**`ml/data/tune/adapters/qwen3-8b-lora-v3-fused`** (4.3 GB). 112 modules fused
`dequantize=False`; the tokenizer files and chat template are inherited from the
verified v2 artifact (sha256 `87a2728c…`, `be75606093…`, `d93ac1a2c…` inherited
unchanged); `config.json` is byte-identical to v2 (`config_matches_v2: true`) so
the arms differ in weights only. Manifest: `fuse_manifest.json` in the fused dir.

## 5. Evaluation results

All four suites scored against the fused v3 model, `--tag v3-treatment`, exact
same prompt/schema contract as every other arm, temperature 0.

### 5.1 Primary suite — eval_v3 (146 adversarial cases, cleaned labels)

| metric | v2 | v3-treatment | delta | test |
|---|---|---|---|---|
| json_parse_rate | 0.9932 | 0.9795 | −0.0137 | n.s. |
| schema_validity_rate | 0.9863 | 0.9795 | −0.0068 | n.s. |
| **topic_count_accuracy** | 0.7808 | **0.8699** | **+0.0890** | **McNemar p=0.0241 sig.** |
| domain_set_exact | 0.6301 | 0.6712 | +0.0411 | n.s. |
| **boilerplate_leak_rate** ↓ | 0.3699 | **0.2260** | **−0.1438** | **McNemar p=0.0001 sig.** |
| **action_redundancy_rate** ↓ | 0.2055 | **0.0890** | **−0.1164** | **McNemar p=0.0060 sig.** |
| out_of_enum_rate ↓ | 0.0068 | 0.0000 | −0.0068 | n.s. |
| duplicate_domain_rate ↓ | 0.0137 | 0.0068 | −0.0068 | n.s. |
| **issue_rouge_l_best** | 0.6719 | **0.7135** | **+0.0416** | **bootstrap CI [0.002, 0.080] sig.** |
| domain_set_precision | 0.8356 | 0.7808 | −0.0548 | n.s. |
| domain_set_recall | 0.7637 | 0.7397 | −0.0240 | n.s. |

Error taxonomy (eval_v3): **no-error 28 → 64**, boilerplate_leak 46 → 28,
action_copy 18 → 6, topic_count 30 → 16, domain_missed 20 → 28,
not_json 1 → 3. C1 (boilerplate removal) and C2 (action targeting) generalise:
20 previously-leaking cases cleaned for 4 new leaks.

### 5.2 Multitopic (85) and frozen (329) — v1 weak labels, not pooled with 5.1

Multitopic 85: `multi_topic_recall` **0.8941 → 0.9412** (beats the 0.894 floor),
`topic_count_accuracy` 0.8588 → 0.9412, `domain_set_exact` 0.6824 → 0.7765,
`schema_validity_rate` 0.9882 → **1.0**.

Frozen 329: `domain_accuracy` 0.8389 → **0.8602** (floor 0.82 met), `object_exact`
0.9422 → 0.9696, `multi_topic_rate` 0.0365 → 0.0122 (less over-emission).
`issue_rouge_l` 0.9234 → 0.6611 — the *expected* C1 consequence on the frozen
suite, whose v1 reference still contains the boilerplate the model now strips;
this is why 5.1 and the frozen suite must not be pooled.

### 5.3 Smoke (20)

`two_topic` recall 1 of 4 — unchanged from v2 (0 of 4 under the failed control).
The trained model can emit two topics on the MT suite (5.2) but still does not
on the four multi-topic smoke prompts.

### 5.4 Gates (`ml/tune/gates_v3.py --arm v3-treatment`) — **FAIL**

| gate | suite | target | v2 baseline | observed | blocking | result |
|---|---|---|---|---|---|---|
| schema | eval_v3 | = 1.00 | 0.9863 | 0.9795 | yes | **FAIL** |
| boilerplate | eval_v3 | <= 0.02 | 0.3699 | 0.2260 | yes | **FAIL** |
| topic_count | eval_v3 | >= 0.90 | 0.7808 | 0.8699 | yes | **FAIL** |
| two_topic_recall | multitopic (85) | >= 0.894 | 0.8941 | 0.9412 | yes | pass |
| frozen_domain | frozen 329 | >= 0.82 | 0.8389 | 0.8602 | yes | pass |
| smoke_two_topic | smoke (20) | 4 of 4 | 1 of 4 | 1 of 4 | yes | **FAIL** |
| terse_single | eval_v3 cat A (10) | >= 0.70 | 0.80 | 0.00 | no | **FAIL** |
| empty_action | eval_v3 cat H (12) | 0 invented | 0 | 1 | no | **FAIL** |
| issue_rouge | frozen 329 | >= 0.80 | 0.9234 | 0.6611 | no | **FAIL** |

## 6. Error analysis

- **Terse-domain collapse (biggest regression).** On all 10 category-A (≤150-char)
  single-topic cases the model now emits domain `other` even though the content
  is plainly `sanitation`/`roads`/`housing`/`construction` (v2 got 8/10). It
  correctly emits 1 topic (count accuracy 1.0) but abandons the domain
  vocabulary on very short inputs — 8 of the 16 new domain misses come from
  these cases. This alone fails the advisory `terse_single` gate at 0.00.
- **3 hard JSON failures** (ev3-013, ev3-032, ev3-046; temp 0, deterministic):
  the model emits an unparseable empty array instead of a topic list. All three
  reference a `payments`+`roads` or `government`+`heating` pair. These three
  drive schema to 0.9795 (v2 had 1 failure). v2 handled all three.
- **Boilerplate geography shifts the frozen ROUGE.** 20 cases fixed, 4 new leaks
  (`ev3-030/070/101/128`). The frozen-suite ROUGE drop is the C1 trade-off, not
  a collapse (no-error count actually doubles on the *cleaned* primary suite).
- **Domain substitutions in multitopic.** The 19 remaining MT misses are mostly
  one-topic substitutions (`heating→housing`, `water→roads`, `construction→
  roads`, drop `government` for a `water`/`sanitation` tail) — recall is high
  (0.94) but exact-set remains 0.78.

## 7. Publication decision

**No publication.** Blocking gates are not met: schema (0.9795 ≠ 1.00,
3 unparseable outputs), boilerplate (0.226 > 0.02 — improved 4× but not gone),
topic_count (0.8699 < 0.90), smoke two-topic (1/4). The published `v2` artifact
(`qwen3-8b-lora-v2-attempt10-fused`) is untouched. No HF upload, no GGUF
conversion, no fresh-download check was performed (publishing gate not reached).

What the run did establish, useful for a v4:

- The full-corpus redesign hypothesis is **confirmed**: with 100 % exposure the
  model beats v2 significantly on the primary suite; the 800-step control's
  regression was an exposure artefact, not an inherent failure of C1–C4.
- C1 and C2 generalise (boilerplate −0.144 p=0.0001; action redundancy −0.116
  p=0.006; no-error 28 → 64).
- The remaining gaps are targeted, not distributional: (a) terse-domain collapse
  to `other` (10/10 cat A — a domain-vocabulary loss on short inputs),
  (b) three unparseable outputs, (c) smoke two-topic. A v4 candidate would
  address (a) by resampling short training rows, (b) by checking the schema
  gate at 1.00, and (c) per the smoke gate.

## 8. Artifacts and disk

- Trained adapter (final): `ml/data/tune/adapters/qwen3-8b-lora-v3-treatment/`
  (final `adapters.safetensors` + `run_config.json`; the nine numbered
  checkpoints were removed after fusion — shas recorded in `fuse_manifest.json`).
- Fused model: `ml/data/tune/adapters/qwen3-8b-lora-v3-fused/` (4.3 GB) with
  `fuse_manifest.json`.
- Control adapter `qwen3-8b-lora-v3-control` and its recorded (negative) results
  are kept as the executable record of the failed experiment.
- Published v2 canonical dir `qwen3-8b-lora-v2-attempt10-fused` and its symlink
  alias `qwen3-8b-lora-v2-fused` unchanged; `~30 GB` of obsolete fused copies
  were removed during this phase (inventory: `ml/reports/artifact_inventory.md`).

## 9. Limitations

- eval_v3 is the only suite whose labels were cleaned of the C1 boilerplate;
  the frozen 329 / multitopic 85 numbers are v1 weak labels and must not be
  pooled with eval_v3 (the model is penalised on them for implementing C1).
- Generation at temperature 0 is deterministic but the 3 hard JSON failures
  show output-format robustness is not yet guaranteed on adversarial inputs.
- The terse category is only 10 cases; the `other`-collapse reading is
  consistent (10/10) but the sample is small.
- One epoch at lr 1e-4 with no schedule: no claim of optimality, only of
  full-corpus exposure parity with the v2 recipe.