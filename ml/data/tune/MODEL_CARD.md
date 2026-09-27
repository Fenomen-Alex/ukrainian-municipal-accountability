# Model card: Qwen3-8B LoRA fine-tune for Ukrainian municipal complaint extraction

**Status:** research prototype (experimental, hand-verified on a small targeted suite)

This is a hybrid model card: the train/loss numbers below come from the *held-out
eval split* (see `ml/reports/finetune_report.md`), the smoke numbers from a small
hand-written targeted suite (see `ml/reports/smoke_test_report.md`). The two are
**not the same measurement** and must not be conflated. Smoke-suite numbers are
NOT a statistically rigorous benchmark.

## Model

- **Base model:** `mlx-community/Qwen3-8B-4bit` (Qwen3-8B, 4-bit MLX
  group-quantized, bits=4, group_size=64)
- **Fine-tune technique:** QLoRA (rank 8, alpha/scale 20, dropout 0), 16 target
  modules, AdamW lr 1e-4, single epoch pass of 400 steps, batch 2 with grad
  accumulation 2 (per-step batch 4), mask_prompt training (`ChatDataset`)
- **Adapter (v1, historical):** `ml/data/tune/adapters/qwen3-8b-lora/adapters.safetensors`
  (~38.8 MB), plus decaying checkpoints 0000100–0000400
- **Fused (deployed) artifact:** `ml/data/tune/adapters/qwen3-8b-lora-v2-fused/`
  (~4.3 GB MLX weights; 8,190,735,360 params). **Not a GGUF file** — it is a
  MLX/HuggingFace-style directory (config.json + safetensors + tokenizer).
- **No thinking block:** generation uses Qwen3 chat template with
  `enable_thinking=False`; makes JSON output reliable and ~3x faster.

## Training data

- 6,837 deduplicated Ukrainian municipal complaints (portal texts), split
  train / valid / test = 5,384 / 1,124 / 329 (14 duplicate uids → positional eval)
- Input: complaint text (≤456 tokens). Output: JSON with `topics[]`; each topic:
  `domain` (13-value enum), `issue`, `object` (street/building, prefer standard
  forms), `requested_action` (only for explicit requests, else empty),
  `attributes` (street/building, only when present). Complaint may mention
  multiple topics.
- 13 domains: roads, water, heating, housing, transport, sanitation,
  electricity, construction, benefits, government, commerce, payments, other

## Fine-tuned eval (held-out split, n=329)

From `ml/reports/finetune_report.md`, deterministic single run (temperature 0):

| metric | base | fine-tuned |
|---|---|---|
| JSON parse | 0.988 | 0.997 |
| schema validity | 0.973 | 0.991 |
| domain accuracy | 0.778 | 0.845 |
| domain macro-F1 | 0.495 | 0.645 |
| issue ROUGE-L | 0.124 | 0.901 |
| object exact | 0.067 | 0.921 |
| object token overlap | 0.281 | 0.944 |
| hallucination rate | 0.739 | 0.024 |
| action-presence match | 0.587 | 0.973 |

Primary improvement: drastic reduction of hallucination (invented objects/
actions/attributes) and enablement of clean structured extraction.

## Targeted smoke suite (20 hand-written cases)

From `ml/reports/smoke_test_report.md`, same settings (temperature 0). The smoke
suite uses a *richer free-form* domain vocabulary (`lighting`, `water_leak`,
...) than the model's 13 schema domains; comparison maps both to the schema via
an explicit documented mapping.

| system | passed/20 | schema-valid | full coverage | hallucination flags | multi-topic ok |
|---|---|---|---|---|---|
| base | 8 | 20 | 17 | 10 | 0/4 |
| fine-tuned (v1) | 16 | 20 | 17 | 0 | 0/4 |
| fine-tuned v2 | 17 | 20 | 18 | 0 | 1/4 |

Known behaviours (from the smoke suite):
- **Excess vocabulary:** `issue` repeats input and may include noise (e.g.
  `траншею ... і кинули` glued to object). `requested_action` often empty even
  when an explicit request exists (pulls the training prior, where ~50% of
  records had no action). Not harmful but leaks minor noise into `object`.
- **Multi-topic regression:** base and v1 emit a single topic for all four
  multi-topic smoke cases. The v2 fine-tune (below) fixes three of the failure
  modes on the multitopic suite (recall 0.89) but only one of the four smoke
  multi cases splits (smoke-13); smoke-14/15/16 still collapse to one correct
  domain. To extract every topic, teach input/output `topics[]` structures more
  strongly or post-process by splitting.
- **Domain mapping:** fine-tuned maps to a reasonable schema domain in most
  cases, but everything "adjacent to green space / playground / manhole" tends
  to go `sanitation`; the schema has no finer graining. Acceptable for the
  current 13-domain design.

## v2 fine-tune (multi-topic structural fix) — current deployed model

`ml/data/tune/adapters/qwen3-8b-lora-v2-fused`: same base, same QLoRA config
(rank 8/scale 20, 16 modules, lr 1e-4, batch 2 + grad-accum 2, 800 iters,
temperature 0), trained on the single-topic base plus a multi-topic
augmentation set (see `ml/reports/finetune_v2_report.md`). This replaces v1
`qwen3-8b-lora-fused` as the artifact served as `qwen3-8b-municipal-finetune`.

Multi-topic suite (85 held-out — v1/base emit 1 topic on all): recall 0.894,
topic-count accuracy 0.859. Frozen-329 stays within the v1 non-regression gate
(domain 0.839 vs 0.845, hallucination 0.015 vs 0.024; guardrail
`multi_topic_rate` 0.037). Smoke 17/20 with smoke-13 multi fixed (see target
suite table below).

## Context lengths / limits

- `ml/data/tune/meta.json` — dataset build metadata (undup, token counts)
- Generation caps: `max_tokens=800`, temperature 0.0 in smoke and eval harnesses
- No evaluation on extremely long inputs; complain limit tested to 456 input
  tokens. Longer complaints are truncated at tokenizer max (512) — see
  `build_dataset.py`. For production, raise `max_seq_len` and retrain/retest.