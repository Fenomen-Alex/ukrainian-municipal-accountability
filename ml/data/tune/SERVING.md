# Serving / packaging the fine-tuned model

Native format: **MLX** (not GGUF). A fused HuggingFace-style directory:
`ml/data/tune/adapters/qwen3-8b-lora-v2-fused/` with `config.json`,
`model.safetensors` + `model.safetensors.index.json`, tokenizer files and
`chat_template.jinja` (library_name `mlx`, apache-2.0 license).

## Local inference (MLX) — no GPU, no server

```bash
.venv-mlx/bin/python -m ml.tune.run_smoke \
  --model ml/data/tune/adapters/qwen3-8b-lora-v2-fused --tag finetuned
```

`run_smoke.py` calls `mlx_lm.load(model, adapter_path=None)` (thin on `generate
`), passes the system prompt, sets `enable_thinking=False` in the Qwen3 chat
template, temperature 0.0. Also verified in `ml/tune/run_eval.py` (vanilla and
LoRA modes).

## LM Studio (local OpenAI-compatible server) — tested, works

LM Studio understands the MLX format natively. The fused directory was installed
by symlinking it under the models folder and restarting the server; the model
then indexes and loads as `qwen3-8b-municipal-finetune`.

> **The system prompt is mandatory.** v2 is prompt-anchored: every one of its
> 8,519 training examples carried the exact `SYSTEM_PROMPT` from
> `ml/tune/build_dataset.py`. If you omit it (which the LM Studio chat GUI does
> by default) the model produces no JSON at all — it emits `!`, Ukrainian prose
> or repetition loops. This is the single most common way to make v2 look
> broken. Read **`ml/tune/V2_SERVING.md`** for the canonical contract before
> serving.

```bash
ln -s "$PWD/ml/data/tune/adapters/qwen3-8b-lora-v2-fused" \
  "$HOME/.lmstudio/models/mlx-community/qwen3-8b-municipal-finetune"
lms server stop && lms server start   # re-scan (lms import only takes single files)
lms load qwen3-8b-municipal-finetune
```

Ready-to-run request (the system message is the real one, not a placeholder —
`V2_SERVING.md` has the copy-pasteable version):

```bash
curl -s http://localhost:1234/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d "$(.venv-mlx/bin/python -m ml.tune.serve_v2 --request-only \
        --text 'На вулиці Шевченка, біля будинку №24, яма.')"
```

Verified on this machine with the v2 fused model: with the system message
present the endpoint returns valid JSON (two topics for the two-issue smoke
case) at temperature 0, matching native MLX inference. Without it, the same
request returns non-JSON garbage.

Note: this repo serves a **directory of per-file symlinks** into the fused model
under the LM Studio models folder; replace the target dir to point the same
model name at a new fused release.

## Ollama

Ollama **requires GGUF**; it cannot load the MLX directory. MLX can export GGUF
(`mlx_lm fuse --export-gguf`), but that is a **lossy conversion** from 4-bit MLX
and for this release we deliberately did **not** convert. If Ollama execution is
required, the supported pipeline is to export GGUF from the *fused MLX* weights
or to start from a GGUF base; document the conversion in the release notes and
re-run the smoke suite afterwards to confirm parity.

## What is not supported / not attempted

- No proprietary license hosts, no mcp server, no container app image.
- The 4.3 GB fused weight set is a development artifact; it is excluded from the
  git repo and treated as a binary release asset.