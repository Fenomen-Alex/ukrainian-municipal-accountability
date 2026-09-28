---
license: apache-2.0
library_name: mlx
base_model: mlx-community/Qwen3-8B-4bit
tags:
  - ukrainian
  - municipal
  - complaint
  - topic-segmentation
  - multi-topic
  - qwen3
  - lora
  - mlx
  - weak-supervision
  - 4bit
language:
  - uk
pipeline_tag: text-generation
---

# Ukrainian municipal accountability — Qwen3-8B v2

Fine-tuned **Qwen3-8B (4-bit MLX)** that converts a free-form Ukrainian
municipal complaint into a structured JSON payload describing its topic(s).

This is the **v2 (attempt-10)** checkpoint. It is a LoRA fine-tune that has been
**merged into the base weights**, so this repository contains a self-contained
`safetensors` file and does not require a PEFT adapter at load time.

## What it does

Input is a raw citizen complaint, often messy, rambling, and multi-issue. Output
is a single JSON object with a `topics` array:

```json
{
  "topics": [
    {
      "domain": "roads",
      "issue": "біля будинку 27 розбите дорожнє покриття",
      "object": "",
      "requested_action": "",
      "attributes": {}
    }
  ]
}
```

Each topic has exactly five keys — `domain`, `issue`, `object`,
`requested_action`, `attributes`. `domain` is one of 13 values:

`benefits`, `commerce`, `construction`, `electricity`, `government`, `heating`,
`housing`, `other`, `payments`, `roads`, `sanitation`, `transport`, `water`

The payload is a **weak-label approximation**, not an authoritative
classification. See [Limitations](#limitations) before relying on it.

## Serving contract — read this before deploying

The model is highly sensitive to the **exact** prompt format. Serving it with a
generic system prompt, a paraphrase, or with thinking enabled produces
degenerate output (literal `!`, repeated prose, unrelated markdown). This is the
single most common failure mode and it looks like a broken model when it is
actually a prompt-format mismatch.

**Requirements**

1. System message must be the **byte-exact** system prompt from the project's
   [`build_dataset.py`](https://github.com/Fenomen-Alex/ukrainian-municipal-accountability/blob/master/ml/tune/build_dataset.py).
2. `enable_thinking=false` — the training target begins with an empty
   `<think>\n\n</think>` block. Do not let a chat template inject reasoning.
3. `temperature=0.0`, `max_tokens=800`.
4. Use the bundled `chat_template.jinja`; do not install a custom template.
5. `strip()` the completion before parsing, then parse with a real JSON parser.

The rendered prefix the model was trained on:

```text
<|im_start|>system
{SYSTEM_PROMPT}<|im_end|>
<|im_start|>user
{complaint}<|im_end|>
<|im_start|>assistant
<think>

</think>

```

**LM Studio:** load this model, paste the exact system prompt into the system
field, switch thinking **off**, set temperature `0` and a 800-token limit.

**OpenAI-compatible server** (LM Studio, mlx-lm, vLLM, TGI):

```bash
curl http://127.0.0.1:1234/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "<model-id>",
    "temperature": 0.0,
    "max_tokens": 800,
    "chat_template_kwargs": {"enable_thinking": false},
    "messages": [
      {"role": "system", "content": "<SYSTEM_PROMPT, byte-exact>"},
      {"role": "user",   "content": "<complaint>"}
    ]
  }'
```

A full copy of the system prompt, ready to paste, lives in
[`ml/tune/V2_SERVING.md`](https://github.com/Fenomen-Alex/ukrainian-municipal-accountability/blob/master/ml/tune/V2_SERVING.md).
A reference loader with the exact prompt is
[`ml/tune/serve_v2.py`](https://github.com/Fenomen-Alex/ukrainian-municipal-accountability/blob/master/ml/tune/serve_v2.py).

**Never** repair the output with a regex that hunts for `{...}`. That hides
wrappers and thinking blocks instead of fixing the cause.

## Verification

`ml/tune/verify_v2.py` checks two things separately, because conflating them
hides real defects:

- **serving contract** (must pass) — parseable JSON, valid schema, in-vocabulary
  domains, all five keys present, non-empty `issue`, no fabricated
  `requested_action`, no extra topics on single-topic input.
- **model quality** (advisory) — topic count, domain set, and request-field
  recall.

Contract: **9/9 pass** on both native MLX and an LM Studio HTTP server, over the
three reported complaints, two verbatim repository smoke cases, and four
constructed single/multi-topic cases. Full suite: **251 passed, 1 skipped**.

Native and server runtimes are each self-consistent at `temperature=0`, but
produced identical text on only 5/9 cases; the differences were confined to
marginal fields (`object` extraction, and one word inside an `issue`). The
serving contract holds on both paths, so this is numerical variation between
runtimes, not instability.

## Training

| | |
|---|---|
| Base | `mlx-community/Qwen3-8B-4bit` |
| Method | LoRA, merged into base weights |
| LoRA rank / alpha / dropout | 8 / 20 / 0.0 |
| Adapted modules | 16 (decoder layers 20–35), 224 tensors |
| Iterations | 800 |
| Batch / grad accumulation | 2 / 2 (effective 4) |
| Optimizer / LR | AdamW / 1e-4 |
| Max sequence length | 2048 |
| Gradient checkpointing | off |
| Seed | 42 |
| Loss masking | prompt tokens masked |
| Weights on disk | 4,607,731,712 bytes across 907 tensors |
| Quantization | 4-bit, group size 64 |

**Dataset** — 8,519 training examples:

| Component | Count |
|---|---|
| v1 single-topic | 5,384 |
| Multi-topic augmentation | 3,135 |
| — real decompositions (×12) | 264 |
| — synthetic (×3) | 2,871 |
| **Total** | **8,519** |

Validation 1,124 and test 329 are frozen across all iterations and were
verified free of overlap with both the frozen training set and the multi-topic
set.

**97.75% of the multi-topic augmentation is synthetic.** Domain coverage in
that set is skewed (`sanitation` 567, `roads` 347, `water` 293) because the
synthetic generator is not domain-balanced.

## Results

### Frozen test set (n=329)

| Metric | Value |
|---|---|
| JSON parse rate | 1.000 |
| Schema validity | 0.997 |
| Domain accuracy | 0.839 |
| Domain macro F1 | 0.760 |
| `issue` ROUGE-L | 0.923 |
| `object` exact match | 0.942 |
| `object` token overlap | 0.957 |
| Hallucination rate | 0.015 |
| `requested_action` presence match | 0.991 |

### Multi-topic suite (n=85, all two-topic)

| Metric | Value |
|---|---|
| JSON parse / schema validity | 0.988 |
| Topic-count accuracy | 0.859 |
| Multi-topic recall | 0.894 |
| Domain-set exact | 0.682 |
| Domain-set precision | 0.896 |
| Domain-set recall | 0.835 |
| `issue` ROUGE-L (best-match) | 0.799 |

### Smoke suite (n=20)

| | v1 | v2 (this model) |
|---|---|---|
| Passed | 16/20 | **17/20** |
| Schema valid | 20/20 | 20/20 |
| Hallucination | 0 | **0** |
| Multi-topic | 0/4 | 1/4 |

v2 fixes the multi-topic failure on `smoke-13`; `smoke-14`, `smoke-15`, and
`smoke-16` still fail. Multi-topic detection is the model's weakest capability.

## Limitations

Read these before using the output for triage, routing, or reporting.

- **Labels are weak.** Annotations were produced by a heuristic labeler, not by
  humans. The model reproduces the labeler, including its blind spots — most
  notably the `requested_action` field, which is empty whenever the labeler's
  verb list fails to match. Real requests such as "Надати роз'яснення…" or
  "зробіть…" are therefore often left empty, or folded into `issue`. Treat
  `requested_action` as unreliable and validate downstream.
- **`sanitation` is over-broad.** The labeler routes recreation, greenery,
  construction, and illegal-advertising complaints into `sanitation`, which
  inflates that class and distorts the domain distribution. For a real
  deployment this needs a human-in-the-loop pass.
- **Multi-topic under-splitting is the main failure mode.** Short or rambling
  complaints with two problems are often returned as a single topic. On the 85-case
  multi-topic suite, domain-set exact match is 0.682.
- **The dataset is 97.75% synthetic for multi-topic.** Real multi-topic
  performance rests on only 22 real decompositions (×12) plus 2 held-out real cases.
- **Not a legal or safety tool.** Do not use these predictions to make
  determinations about individual residents, to allocate public funds
  automatically, or as the sole basis for any enforcement action.
- **Apple Silicon only.** These are 4-bit MLX weights. They will not load in a
  CUDA-only runtime. Convert to GGUF for llama.cpp if you need broader hardware
  support — do not expect bit-identical results after conversion.
- **Ukrainian only.** No other language was represented in training.

## Intended use

Triage and structuring of Ukrainian municipal complaints: splitting a free-form
message into topics, tagging each with a municipal domain, and extracting the
object of the complaint. Suitable as a drafting aid or a queue-routing
pre-filter where a human reviews the result.

**Out of scope:** autonomous decision-making, enforcement, eligibility
determination, or any use where an incorrect `requested_action` could cause
harm to a resident.

## License and data provenance

The **model weights are Apache-2.0**, matching the base model
[`Qwen/Qwen3-8B`](https://huggingface.co/Qwen/Qwen3-8B), whose full license text
was verified before publication. The MLX 4-bit conversion of the base declares
the same license.

The **training data is not redistributed here and its license is not
documented** in the source project. The municipal complaint corpus has no
recorded source URL, license, or redistribution terms. Only the derived weights
are published. Anyone redistributing or building on this checkpoint should
confirm the upstream data terms independently.

## Citation and provenance

Training data, evaluation harness, verification suite, and the full
serving contract live in the source repository:

<https://github.com/Fenomen-Alex/ukrainian-municipal-accountability>

Cite that repository. This checkpoint is `v2 (attempt-10)` of its fine-tuning
track.
