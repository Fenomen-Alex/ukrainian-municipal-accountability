# MLX fine-tuning experiment: Ukrainian municipal complaints -> structured JSON

Status: EXECUTED - see ml/reports/finetune_report.md for results (QLoRA 400 iters, lora vs base eval, fused artifact)
Date: 2026-09-24

## Goal

Produce a small, demonstrable, reproducible experiment that turns free-form
Ukrainian municipal citizen complaints into the machine-consumable structured
JSON format already frozen in `ml/data/gold/annotation_schema.json`:

```json
{"topics": [{"domain": "...", "issue": "...", "object": "...", "requested_action": "...", "attributes": {...}}]}
```

This is **not** a chatbot, and **not** a 47-class classifier. It must be a
task-specific model fine-tuned on Apple Silicon, reproducible at any point.

## Hardware (verified)

- MacBook Pro M1 Pro, 32 GB unified memory, no discrete GPU (`hw.memsize` = 32 GB)
- Disk: ~102 Gi free (less than the earlier estimate of 100 GB)
- Toolchain: MLX-LM 0.31.3 + mlx 0.32.2 + mlx-metal in a **separate venv**
  `.venv-mlx` (Python 3.12.14). The repo `.venv` is Python 3.14 and is **not**
  compatible with current mlx wheels, so all training/inference work uses
  `.venv-mlx/bin/python` and `.venv-mlx/bin/mlx_lm.*`.

## Candidate base models (researched 2026-09-24)

| Model | License | Ukrainian | JSON discipline | MLX-QLoRA path | Silicon fit |
|---|---|---|---|---|---|
| `INSAIT-Institute/MamayLM-Gemma-3-12B-IT-v2.0` | Gemma TOU | Best available (INSAIT, native) | Good (v2.0 improved instruction following) | No prefab 12B MLX quant; 27B fp16/6bit only on `mlx-community`; convert or run fp16 (heavy) | Feasible but slow; Gemma-3 multimodal, vision tower must be dropped for text-only LoRA |
| `lapa-llm` family (Lapa is model+dataset project) | CC-BY-* datasets | Good | Unknown | Datasets reusable; models at `lapa-llm` | Dataset source, not base-model choice |
| `Qwen/Qwen3-8B` (base) + `mlx-community/Qwen3-8B-4bit` | Apache-2.0 | Good (official 119 langs, WMT usable uk) | Excellent (Qwen3 is the extract-the-JSON model; hard-coded system-follow) | First-class: 4-bit prefab MLX exists (133k downloads), QLoRA auto-enabled on quantized weights | Sweet spot: ~5 GB download, sub-hour training per run |

Also considered: `Qwen3-4B-2507` (faster, weaker), `MamayLM` 27B (too big for
32 GB comfort), Russian-retail trails (Vtmpas/Mazepa/Zhilyuk) — rejected, wrong
language, dead repos.

### Decision

**Base: `mlx-community/Qwen3-8B-4bit` (Apache-2.0, Qwen3-8B).**

Rationale:
1. **Apache-2.0** — cleanest licensing for a redistributable task-specific model
   (Gemma TOU adds redistribution conditions).
2. **JSON/instruction discipline** — the entire task is structured output; Qwen3
   is the strongest open option for obeying a schema in its output.
3. **Turnkey MLX** — a maintained, pre-quantized 4-bit MLX build exists; QLoRA
   + LoRA come auto-configured on quantized weights; no conversion step, no
   vision-tower surgery.
4. **Good Ukrainian** — official 119-language support, WMT-usable English↔Ukrainian;
   not native-grade like MamayLM, but our downstream task is *structured
   extraction* whose labels come from the data, not from the base model's
   grammar, so weaker Ukrainian is tolerable for the first credible experiment.
5. **Reasonable fit on M1 Pro 32 GB** — 4-bit 8B family: within a couple GB of
   unified memory for QLoRA, sub-hour for a 300-500 step run. MamayLM-12B is the
   documented *next* quality-priority candidate if this pipeline proves out.

MamayLM stays the strongest Ukrainian-quality candidate; it is explicitly
deferred (no prefab 12B MLX quant + Gemma-3 modal/folding overhead + Gemma TOU),
not dismissed.

## Data

Existing corpus (already deduplicated, split, and leakage-checked — reuse as-is,
do not re-derive):

- `ml/data/train.jsonl` — 5384
- `ml/data/validation.jsonl` — 1124
- `ml/data/test.jsonl` — 329
- Fields per row: `content` (free text), `kind` (weak source-class label),
  `result`, `organizationName`, `organizationId`, `receivedDateTime`, address
  fields, `status`, `type`, `accrualMethod`.

`kind` is the previous pipeline's weak label; **we do not blindly copy it** — it
feeds a deterministic `kind -> domain-enum` map and is honestly reported as a
weak source with a loose 1:1 fallback to `other` when the map has no entry.

Training corpus = existing 6837 unique rows, honored splits
(train/valid/test); no test leakage, nothing re-split.

## Output schema (target)

Reuse `ml/data/gold/annotation_schema.json` verbatim (topics + 5 fields, draft-07).

## Labeling strategy (defensible, deterministic)

For the first credible experiment we build **weak structured targets** purely by
deterministic transforms of existing fields:

1. `domain` — from `kind` via a fixed `labels.json` -> schema enum map
   (documented in code). Unknown kinds fall back to `other`.
2. `issue` — a **neutral restatement**: we take the complaint text, strip
   leading greeting/salutation boilerplate, and emit the cleaned sentence as the
   issue. This is *not* a paraphrase (no synthesis budget); it is a defensible,
   auditable deterministic transform. Weakness disclosed in eval.
3. `object` — the address-derived target: prefer the fullest address present
   (`addressThoroughfare addressLocatorDesignator addressPostName …`), else
   `organizationName`. Determined by field presence, never synthesized.
4. `requested_action` — derived from the sentence containing imperative/request
   verbs (`прошу`, `просимо`, `вимагаю», …) that appears in `content`, else
   empty string. Empty stays empty — **no hallucinated actions**.
5. `attributes` — a small fixed bag derived from fields that genuinely coexist
   with the answer in the source: `organization`, `street`, `building`,
   `status`, `receivedDateTime`. Only present keys are emitted; `additionalProperties: true`
   allows exactly what we emit.

All transforms are pure functions over the record dict, unit-tested, and the
provenance (`derived_from`, `weak=1`) is documented per field in code, not
embedded in the JSON (schema has `additionalProperties: false`, so provenance
lives only in the build log / report).

This is **weak supervision**, honestly labelled as such. We do not claim these
are gold labels; eval measures how far the fine-tune gets against this weak
target AND against a deterministic baseline.

## Training

- Tool: `mlx_lm.lora` (MLX-LM 0.31.3) — QLoRA on the 4-bit base (auto).
- Data format: `{"messages": [...]}` chat JSONL (ChatDataset).
- System prompt embeds the schema + extraction instructions + honesty rules
  (empty instead of invented, no PII).
- Config: LoRA layers 16, batch 1-2, grad-checkpoint, `--mask-prompt`,
  `--iters ~300-500`, `--learning-rate 1e-4 (1e-5 fallback)`, `--val-batches -1`.
- Records training wall-clock and config to `git`-friendly YAML/JSON.

## Evaluation (base vs fine-tuned vs deterministic baseline)

1. Deterministic baseline = exactly the labeling transform above applied to
   `test.jsonl` (upper bound on *this* weak target, honest reference point).
2. Base model (`mlx-community/Qwen3-8B-4bit`, no adapter): generate JSON for each
   test complaint.
3. Fine-tuned: same, with adapter fused.

Metrics (test set, N=329):
- JSON parse-rate (valid output)
- Schema-validity rate (jsonschema Draft7 with our schema)
- Domain accuracy + macro-F1 vs weak target
- Issue overlap (ROUGE-L or token overlap) with cleaned source
- Object exact/partial match vs derived object
- Requested-action presence-match (empty vs non-empty correctly predicted)
- Hallucination rate: fields set when source is empty / tokens absent from source
- Multi-topic rate (`len(topics) >= 2`)

Report: one markdown with all numbers, config of record, artifact paths, exact
inference command, limitations, next experiment.

## Artifacts

- `ml/tune/` — dataset build, train, eval scripts
- `ml/data/tune/` — generated chat JSONL (train/valid/test), eval output, report
- Adapter: `ml/data/tune/adapters/…`; fused model for inference, if disk allows.