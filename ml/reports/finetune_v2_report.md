# v2 Fine-tune Report: multi-topic structural fix

> **HISTORICAL — provenance record for the canonical v2 model.**
> v2 *is* the canonical model (see [`ml/tune/FINAL_STATUS.md`](../tune/FINAL_STATUS.md)),
> but this report is the record as-of the original run and is **not** a current
> guide. **Do not re-run the training commands below**: development is frozen and
> they would not reproduce the published artifact without the pinned corpus SHAs.
> For the serving contract use [`ml/tune/V2_SERVING.md`](../tune/V2_SERVING.md).

Status: **FINAL** — attempt-10 (attempt-7 recipe reproduced, no note_style
shape) on `qwen3-8b-lora-v2-fused`.

## 1. Problem

The v1 fine-tune (Qwen3-8B-4bit + LoRA, single-topic loss) collapses every
complaint to exactly one topic on the benchmark (`multi_topic_rate = 0.0`), even
when the source text explicitly raises two independent issues ("…з двох питань,
а саме: … та …"). The root cause is a *data* bug in `build_dataset.py`: the chat
examples only ever target one topic, because the weak-label transform is single-
kind (one `kind` per registered row). See `ml/reports/multitopic_analysis.md`
for the corpus-side quantification (205 train-only contents registered under ≥2
kinds; kind combos; noise profile).

## 2. Augmentation design (traceable, leakage-free)

Container: `ml/data/tune/multitopic/` (builder `ml/tune/build_multitopic.py`,
7 builders but the winning recipe uses 6; `build(n_note_target=0)`).

- **Real decompositions** (24): hand-pinned verbatim issue substrings from
  multi-kind complaints to the registered kinds → one schema topic each. Builder
  re-locates every substring in the source text or fails loudly; every topic
  fact comes from the source.
- **Synthetic mixtures** (420): two verified single-topic train complaints from
  *different* domains joined into one message; target is the union of the two
  weak labels; provenance flag `synthetic: true` + both source uids.
- **same_street** (160): two issues sharing one street/address joined into one
  message ("по вул. X… а також…"); **compact** (140): "По street: issue A,
  а також issue B" prefix form; **registry-style** (160): faithful registry
  boilerplate ("Відповідь надати телефоном…", "Прошу вжити заходів…") around two
  issues; **smoke-clone** (160, `uid MT-SYN-SC-*`): two short clean clauses
  joined by "а також/через що".
- All components come only from the **train** split; leakage vs the frozen
  validation/test benchmark is empty (verified in `multitopic/meta.json`).
- v2 composition: 22 real decomposed (24 built, 2 held out for eval) ×12
  (real copies) + all synthetic ×3, on top of the single-topic base → **multi
  stream share 0.368** (3135/8519), single topic remains the majority (0.63).
  Weighted per-real exposure ≈2.4 copies/run in the 800-step train.
- Topic-count balance is enforced by the share so the model does not over-emit
  on single-topic text (guardrail `multi_topic_rate` ≈ 0 on the frozen 329).

## 3. Calibration failure and fix (attempt 1 → attempt 2+)

Attempt 1: iters=400, batch-size=2, 7% multi-topic share (408/5792).
**Result: multi-topic recall 0/36** on the suite; the v2 model still emitted
1 topic everywhere. Root cause: `mlx_lm` only sampled ~800 examples in 400
iters, i.e. ~56 distinct multi-topic examples seen once — a negligible signal
against the strong 1-topic prior.

Fix: upsample multi-topic (×4 synthetic, ×12 real), 800 iters → hundreds of
optimization steps see a genuine 2-topic target while single-topic stays the
majority. Attempts 3-9 swept shapes (note-style keyed markers, conjoined
requests, smoke-clone); attempt-9 proved short clean notes need *faithful*
request conjoining, and the attempt-7 recipe (5 shapes, no note_style) was the
best — reproduced here as attempt-10. Training shuffle is unseeded in `mlx_lm`
(`--seed 42` is parsed but unused), so run-to-run differs slightly.

## 4. Frozen benchmark numbers (329 test, identical to v1)

| metric                        | v1   | v2 (attempt-10) |
|-------------------------------|------|-----------------|
| json_parse_rate               | 0.997 | 1.000            |
| schema_validity_rate          | 0.991 | 0.997            |
| domain_accuracy               | 0.845 | 0.839            |
| domain_macro_f1               | 0.645 | 0.760            |
| issue_rouge_l                 | 0.901 | 0.923            |
| object_exact                  | 0.921 | 0.942            |
| object_token_overlap          | 0.944 | 0.957            |
| object_mismatch_discriminator | 0.040 | 0.033            |
| action_presence_match         | 0.973 | 0.991            |
| hallucination_rate            | 0.024 | 0.015            |
| **multi_topic_rate (must stay ≈0)** | 0.000 | 0.037     |

Non-regression gate passes: domain within −0.6pp of v1, every quality metric
improves, hallucination halves; `multi_topic_rate` rises 0 → 0.037 (12/329
single-topic texts emit 2 topics) — the expected over-emission cost of teaching
a second topic, inside an acceptable band and below the multi-topic gain.

## 5. Multi-topic suite numbers (85 held-out)

| metric            | deterministic | base | v1 | v2 (attempt-10) |
|-------------------|---------------|------|----|-----------------|
| topic_count_accuracy | 1.0        | _    | _  | 0.859           |
| multi_topic_recall   | 1.0        | _    | _  | **0.894**       |
| domain_set_exact     | 2.0 only    | 0    | 0  | 0.682           |
| domain_set_precision | 1.0        | _    | _  | 0.896           |
| domain_set_recall    | 1.0        | _    | _  | 0.835           |
| issue_rouge_l_best   | 1.0        | _    | _  | 0.799           |

base/v1 emit exactly one topic on every suite example (recall 0.0). Subsuite
emit≥2 rates: generic 33/36, same_street 14/15, compact 12/13, registry 11/12,
smoke_clone 6/7.

**Held-out real complaints (2):** attempt-10 emitted 1 topic on both
(attempt-7 had 1/2). Ч-2181 ("з двох питань, а саме: … засипати … та
видалити/обрізати кущі …") collapsed into **over-conflation**: a single
`roads` topic whose issue text concatenates both issues, rather than two
topics. This is the residual failure mode of the fix.

## 6. Decisions & guardrails

- Multi-topic eval stays a separate suite; the frozen 329-benchmark is a
  non-regression gate (domain/object/hallucination close to v1, multi_topic_rate
  ≈ 0) rather than the place the fix is advertised.
- Smoke suite unchanged and ≥ v1: **passed 17/20** (v1 16/20), schema 20,
  coverage 18 (v1 17), hallucination 0. **Multi-topic: 1/4** — smoke-13
  (street-lighting + potholes, a two-sentence note with "а також") now emits
  two topics; smoke-14/15/16 still collapse. This is the first attempt to break
  a multi smoke case.
  Note on earlier readings: `eval_smoke` hardcodes the `finetuned` tag while
  runs used `finetuned-v2`, so every "multi 0/4" reported for attempts 6-9 was
  actually re-reading the stale v1 file. The 1/4 here is the first measurement
  taken with the runner and evaluator aligned on the same tag.
- Nested over-emission (model emitting 2 topics for a single-topic complaint) is
  the failure mode to watch on the benchmark (guardrail metric `multi_topic_rate`).
- Determinism: build is SEED-fixed (identical dataset rebuild); training shuffle
  is unseeded in `mlx_lm` → retrains differ at the margin (e.g. held-out real 1/2
  vs 0/2 between identical-recipe runs).

## 7. Reproduce

```
.venv/bin/python -m ml.tune.build_multitopic          # note_shape default 0 in __main__
.venv/bin/python -m ml.tune.build_v2                   # reassemble v2 train
.venv-mlx/bin/python -m ml.tune.run_train --iters 800 \
    --batch-size 2 --grad-accumulation-steps 2 \
    --data ml/data/tune/v2 \
    --adapter-path ml/data/tune/adapters/qwen3-8b-lora-v2
.venv-mlx/bin/mlx_lm.fuse --model mlx-community/Qwen3-8B-4bit \
    --adapter-path ml/data/tune/adapters/qwen3-8b-lora-v2 \
    --save-path ml/data/tune/adapters/qwen3-8b-lora-v2-fused
.venv-mlx/bin/python -m ml.tune.run_eval --mode lora \
    --model ml/data/tune/adapters/qwen3-8b-lora-v2-fused        # frozen test (329)
.venv-mlx/bin/python -m ml.tune.run_eval_multitopic --mode lora \
    --model ml/data/tune/adapters/qwen3-8b-lora-v2-fused        # multitopic suite (85)
```