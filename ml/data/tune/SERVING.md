# Serving / packaging the fine-tuned model

Native format: **MLX** (not GGUF). A fused HuggingFace-style directory:
`ml/data/tune/adapters/qwen3-8b-lora-fused/` with `config.json`,
`model.safetensors` + `model.safetensors.index.json`, tokenizer files and
`chat_template.jinja` (library_name `mlx`, apache-2.0 license).

## Local inference (MLX) — no GPU, no server

```bash
.venv-mlx/bin/python -m ml.tune.run_smoke \
  --model ml/data/tune/adapters/qwen3-8b-lora-fused --tag finetuned
```

`run_smoke.py` calls `mlx_lm.load(model, adapter_path=None)` (thin on `generate
`), passes the system prompt, sets `enable_thinking=False` in the Qwen3 chat
template, temperature 0.0. Also verified in `ml/tune/run_eval.py` (vanilla and
LoRA modes).

## LM Studio (local OpenAI-compatible server) — tested, works

LM Studio understands the MLX format natively. The fused directory was installed
by symlinking it under the models folder and restarting the server; the model
then indexes and loads as `qwen3-8b-municipal-finetune`:

```bash
ln -s "$PWD/ml/data/tune/adapters/qwen3-8b-lora-fused" \
  "$HOME/.lmstudio/models/mlx-community/qwen3-8b-municipal-finetune"
lms server stop && lms server start   # re-scan (lms import only takes single files)
lms load qwen3-8b-municipal-finetune
curl -s http://localhost:1234/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3-8b-municipal-finetune",
       "messages":[{"role":"system","content":"...system prompt..."},
                   {"role":"user","content":"На вулиці Шевченка, біля будинку №24, яма."}],
       "temperature":0}'
```

Verified on this machine: the model returned valid JSON for a smoke case through
the `/v1/chat/completions` endpoint.

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