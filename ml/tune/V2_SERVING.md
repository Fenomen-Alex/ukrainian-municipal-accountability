# v2 inference contract (canonical)

The v2 fine-tune is a **prompt-anchored** model. It was trained on 8,519 chat
examples that *all* carried one specific system message, tokenized with Qwen3's
chat template at `enable_thinking=False`. If you do not reproduce that prompt,
the adapter has nothing to key on and the model reverts to raw base-Qwen3 chat
behaviour (prose, repetition, `!` prefixes, no JSON at all). This is not a
quality problem and it is not fixable by sampling settings.

This file is the single canonical description of how to run v2. The Python
implementation is `ml/tune/serve_v2.py`; the verification suite is
`ml/tune/verify_v2.py`.

## The one rule

> Always send the **exact** system message below as `messages[0]`, then the
> citizen complaint as the user message. Do not paraphrase, translate, shorten or
> omit it.

## Model

| | |
|---|---|
| **Published** | <https://huggingface.co/Fenomen-Alex/ukrainian-municipal-accountability-qwen3-8b> |
| Model card | [`MODEL_CARD_V2.md`](MODEL_CARD_V2.md) |
| Artifact | `ml/data/tune/adapters/qwen3-8b-lora-v2-attempt10-fused/` |
| Byte-identical alias | `ml/data/tune/adapters/qwen3-8b-lora-v2-fused/` (same sha256) |
| Base | `mlx-community/Qwen3-8B-4bit` (Qwen3-8B, 4-bit MLX quantization) |
| Format | MLX / safetensors, `Qwen3ForCausalLM`. **Not** GGUF. |
| Files | `model.safetensors`, `model.safetensors.index.json`, `config.json`, `tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja`, `README.md` |
| EOS | `<|im_end|>` (id 151645) |
| License | Apache-2.0 (matches base). Training data is **not** redistributed — see the model card. |

Prefer the Hub copy. The published repo is byte-identical to the local artifact
(all 7 files match on sha256), and a fresh anonymous download reproduced all 9
verification cases byte-for-byte. To fetch it:

```bash
hf download Fenomen-Alex/ukrainian-municipal-accountability-qwen3-8b \
  --local-dir ./qwen3-v2
```

## The exact prompt

Immediately before generation the model expects this text, and only this:

```
<|im_start|>system
Ти — система, що перетворює звернення громадян до органів місцевого самоврядування у структурований JSON-формат.

Виведи виключно валідний JSON без пояснень, без markdown-рамок, без коментарів. Схема:
{"topics": [{"domain": str, "issue": str, "object": str, "requested_action": str, "attributes": {}}]}

Правила:
- domain — одна зі значень: roads, water, heating, housing, transport, sanitation, electricity, construction, benefits, government, commerce, payments, other.
- issue — конкретна проблема, сформульована нейтрально, без скарг на конкретних осіб.
- object — адреса або установа, якої стосується звернення. Якщо немає — порожній рядок.
- requested_action — що заявник просить зробити. Якщо прохання немає — порожній рядок.
- attributes — додаткові факти (організація, вулиця, будинок, статус, дата). Лише те, що реально є у тексті.
- topics може містити декілька незалежних проблем. Якщо звернення не містить дієвої проблеми — topics порожній масив.
- НЕ вигадуй факти. Якщо чогось немає у тексті — не додавай. Не включай персональних даних (ПІБ, номери телефонів).<|im_end|>
<|im_start|>user
На вулиці Шевченка, біля будинку №24, утворилася глибока яма на проїжджій частині. Прошу провести ямковий ремонт асфальтного покриття.<|im_end|>
<|im_start|>assistant
<think>

</think>


```

i.e. structurally:

```text
<|im_start|>system
{SYSTEM_PROMPT}<|im_end|>
<|im_start|>user
{complaint}<|im_end|>
<|im_start|>assistant
<think>

</think>

```

The literal trailing `<think>\n\n</think>\n\n` is what
`enable_thinking=False` produces in Qwen3's `chat_template.jinja`. It is part of
every training target's prefix, so do not strip it.

### The system message, byte-exact

```text
Ти — система, що перетворює звернення громадян до органів місцевого самоврядування у структурований JSON-формат.

Виведи виключно валідний JSON без пояснень, без markdown-рамок, без коментарів. Схема:
{"topics": [{"domain": str, "issue": str, "object": str, "requested_action": str, "attributes": {}}]}

Правила:
- domain — одна зі значень: roads, water, heating, housing, transport, sanitation, electricity, construction, benefits, government, commerce, payments, other.
- issue — конкретна проблема, сформульована нейтрально, без скарг на конкретних осіб.
- object — адреса або установа, якої стосується звернення. Якщо немає — порожній рядок.
- requested_action — що заявник просить зробити. Якщо прохання немає — порожній рядок.
- attributes — додаткові факти (організація, вулиця, будинок, статус, дата). Лише те, що реально є у тексті.
- topics може містити декілька незалежних проблем. Якщо звернення не містить дієвої проблеми — topics порожній масив.
- НЕ вигадуй факти. Якщо чогось немає у тексті — не додавай. Не включай персональних даних (ПІБ, номери телефонів).
```

Source of truth: `SYSTEM_PROMPT` in `ml/tune/build_dataset.py`. Do not retype
it; import it.

```python
from ml.tune.build_dataset import SYSTEM_PROMPT
```

## Quote normalization (user turn only)

The user complaint is normalized before templating. This is a root-cause fix,
not cosmetic cleanup — see `ml/tune/QUOTE_ARTEFACT.md` for the full account.

`normalize_quotes()` in `ml/tune/build_dataset.py`:

* **Delimited** ASCII quotes become a typographic pair, alternating `“` / `”`:
  `Біля маг. "Копілка"` → `Біля маг. “Копілка”`.
* **Mid-word** ASCII quotes become U+2019: `під"їзду` → `під’їзду`,
  `роз"яснення` → `роз’яснення`.
* Idempotent; never deletes text; only quote *glyphs* change, so word content
  and length are preserved.

```python
from ml.tune.serve_v2 import build_messages
build_messages('Біля маг. "Копілка" три тижні тече каналізація.')
```

Two constraints make this safe to add:

* **User turn only.** The system prompt's own schema quotes (`"issue"`,
  `"requested_action"`, …) are what teach the output format and are never
  touched. `SYSTEM_PROMPT` is byte-identical on every request.
* **Complaints with no ASCII quotes are byte-identical** to the pre-fix path,
  so existing traffic is unaffected. `test_release_smoke.py` asserts both.

If you are sending requests over HTTP rather than calling `build_messages`,
apply the same normalization to the user content yourself — the server will not
do it for you. `--request-only` prints the already-normalized messages, so the
easiest integration is to take its output verbatim.

## Why this is load-bearing (measured, not assumed)

All rows are the same model on the same complaint via the LM Studio HTTP server
on `127.0.0.1:1234`, at `temperature=0`.

| # | Request | Result |
|---|---|---|
| 1 | `/v1/chat/completions`, exact system message, `enable_thinking=false` | **valid JSON, 2 topics** |
| 2 | `/v1/chat/completions`, exact system message, thinking **on** | **valid JSON, 2 topics** |
| 3 | `/v1/chat/completions`, **no** system message (what the GUI does) | `!` + Ukrainian prose, loops until `finish_reason=length` |
| 4 | `/v1/chat/completions`, **empty** system message | identical failure to #3 |
| 5 | `/v1/chat/completions`, *different* system message | degenerate (`![](https://i.imgur.com/...)` spam) |
| 6 | `/v1/completions`, prompt hand-rendered as above | **valid JSON, 2 topics** |
| 7 | `/v1/completions`, bare complaint text, no chat markers | echoes the input in a loop |

Conclusions:

* The **system message is the determining factor.** Thinking mode is *not*:
  rows 1 and 2 both work. `enable_thinking=false` is still the contract because
  it is the trained condition, but omitting the system prompt is what breaks it.
* It must be **this** system message. A generic or empty system message fails
  just as hard as none (rows 4, 5).
* `/v1/completions` works **only** with the fully rendered prompt (row 6 vs 7),
  because the model never saw bare text.
* The historical scorer would never have caught this: `parse_payload` in
  `ml/tune/evaluate.py` greps `\{.*\}` and silently discards surrounding text.
  Use `ml.tune.serve_v2.parse_strict` when you are checking a serving contract.

## Canonical: native MLX

Pinned serving runtime: `ml/requirements-serve.txt` (Apple Silicon).

```bash
.venv-mlx/bin/python -m ml.tune.serve_v2 \
  --model ml/data/tune/adapters/qwen3-8b-lora-v2-attempt10-fused \
  --text "На вулиці Шевченка, біля будинку №24, утворилася глибока яма на проїжджій частині. Прошу провести ямковий ремонт асфальтного покриття."
```

```python
from ml.tune.serve_v2 import load_model, generate_text, parse_strict

model, tok = load_model("ml/data/tune/adapters/qwen3-8b-lora-v2-attempt10-fused")
raw = generate_text(model, tok, complaint)        # temp 0.0, max_tokens 800
payload, error = parse_strict(raw)                # no wrapper tolerance
```

## Canonical: HTTP

Both endpoints work, provided the prompt is built as above.

`POST /v1/chat/completions` — the recommended path; let the server's chat
template do the rendering.

```bash
curl -s http://127.0.0.1:1234/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3-8b-municipal-finetune",
    "messages": [
      {"role": "system", "content": "Ти — система, що перетворює звернення громадян до органів місцевого самоврядування у структурований JSON-формат.\n\nВиведи виключно валідний JSON без пояснень, без markdown-рамок, без коментарів. Схема:\n{\"topics\": [{\"domain\": str, \"issue\": str, \"object\": str, \"requested_action\": str, \"attributes\": {}}]}\n\nПравила:\n- domain — одна зі значень: roads, water, heating, housing, transport, sanitation, electricity, construction, benefits, government, commerce, payments, other.\n- issue — конкретна проблема, сформульована нейтрально, без скарг на конкретних осіб.\n- object — адреса або установа, якої стосується звернення. Якщо немає — порожній рядок.\n- requested_action — що заявник просить зробити. Якщо прохання немає — порожній рядок.\n- attributes — додаткові факти (організація, вулиця, будинок, статус, дата). Лише те, що реально є у тексті.\n- topics може містити декілька незалежних проблем. Якщо звернення не містить дієвої проблеми — topics порожній масив.\n- НЕ вигадуй факти. Якщо чогось немає у тексті — не додавай. Не включай персональних даних (ПІБ, номери телефонів)."},
      {"role": "user", "content": "На вулиці Шевченка, біля будинку №24, утворилася глибока яма на проїжджій частині. Прошу провести ямковий ремонт асфальтного покриття."}
    ],
    "temperature": 0.0,
    "max_tokens": 800,
    "chat_template_kwargs": {"enable_thinking": false}
  }'
```

`chat_template_kwargs` is an `mlx_lm.server` extension. If your server ignores
it, thinking mode still works (row 2) — but keep it if you can.

`POST /v1/completions` — completion semantics. You must render the prompt
yourself, `add_generation_prompt=True` and `enable_thinking=False` included.

```python
prompt = tokenizer.apply_chat_template(
    [{"role": "system", "content": SYSTEM_PROMPT},
     {"role": "user", "content": complaint}],
    tokenize=False, add_generation_prompt=True, enable_thinking=False,
)
```

## LM Studio

### Which name to use

Three different names refer to this model. They are not interchangeable.

| Name | Where it belongs |
|---|---|
| `Fenomen-Alex/ukrainian-municipal-accountability-qwen3-8b` | The **Hugging Face repo id** — use this to download from HF. |
| `qwen3-8b-lora-v2-attempt10-fused` | The **fused artifact directory** on disk (gitignored), the default for `--model`. |
| `qwen3-8b-municipal-finetune` | A **local LM Studio folder label**, not a model path and not an HF id. It is only the `model` field your LM Studio server echoes back. |

So `qwen3-8b-municipal-finetune` is a naming convention for the folder you load
into LM Studio; renaming that folder locally is fine, and renaming it in the
documentation is not, because LM Studio matches the request's `model` field
against whatever it has loaded. Keep the label consistent between what LM Studio
has loaded and what you send. `--request-only` defaults to this label; override
it with `--model-id` to match your own setup.

1. Model Developer / load the fused directory (symlink works, see
   `ml/data/tune/SERVING.md`).
2. **Do not use a custom chat template.** The shipped `chat_template.jinja` is
   already correct, and a custom template is the usual way people accidentally
   drop the system message.
3. Paste the system prompt into the **System** box of the chat, or send
   `messages` via the API. The GUI has no separate "system" slot per se — use the
   system-prompt field, or call the endpoint.
4. Turn **thinking / reasoning off** if the UI offers it.
5. Temperature `0.0`, max tokens `800`.

The symptom of getting this wrong is unmistakable: output that starts with `!`,
repeats itself, is not JSON, or leaks a Ukrainian phrase into `domain`.

## Generation settings

| Setting | Value | Why |
|---|---|---|
| `temperature` | `0.0` | every recorded v2 metric was produced greedily |
| `max_tokens` | `800` | matches `run_smoke.py`; ample for multi-topic |
| `top_p` / penalties | leave default | untested; do not invent them |
| seed | not needed at `temperature=0.0` | measured: 3 repeats, 1 unique output per case |

The model is **self-consistent** within a runtime at `temperature=0.0`, but
**not bit-identical across runtimes**: 5 of 9 verification cases match native
MLX byte-for-byte, and the other 4 differ only in marginal fields (`object`
most often, plus one word in an `issue`). Different MLX kernels and batch shapes
move a borderline greedy decision. Treat `domain` and topic count as the stable
signals; do not assert on `object` when comparing runtimes.

## Output contract

```json
{
  "topics": [
    {
      "domain": "...",
      "issue": "...",
      "object": "...",
      "requested_action": "...",
      "attributes": {}
    }
  ]
}
```

`domain` must be one of the 13 schema enums and nothing else:

```text
roads, water, heating, housing, transport, sanitation,
electricity, construction, benefits, government, commerce,
payments, other
```

A Ukrainian phrase in `domain` means the prompt was wrong, not that the model
invented a category.

### Consumer requirements

* **Strip leading/trailing whitespace before parsing.** v2 commonly emits a bare
  newline before `{` (e.g. `"\n{\"topics\": ...}"`). That is the trained
  behaviour, not corruption. `parse_strict()` and `JSON.parse(x.trim())` both
  handle it.
* Do **not** "repair" output with a `\{.*\}` regex. That is what the historical
  scorer did, and it hides prompt failures instead of surfacing them.
* One topic object per problem, always with all five keys. `object`,
  `requested_action` and `attributes` are legitimately empty; `issue` and
  `domain` are not.

## Worked example

Input:

```text
На вулиці Шевченка, біля будинку №24, утворилася глибока яма на проїжджій частині. Прошу провести ямковий ремонт асфальтного покриття.
```

Actual v2 output at `temperature=0.0` (native MLX):

```json
{"topics": [{"domain": "roads", "issue": "вулиці Шевченка, біля будинку №24, утворилася глибока яма на проїжджій частині. провести ямковий ремонт асфальтного покриття.", "object": "вулиці Шевченка", "requested_action": "", "attributes": {"street": "вулиці Шевченка"}}]}
```

## Verification

```bash
# native
.venv-mlx/bin/python -m ml.tune.verify_v2 --native

# over HTTP against a running server
.venv-mlx/bin/python -m ml.tune.verify_v2 \
  --base-url http://127.0.0.1:1234/v1 --model-id qwen3-8b-municipal-finetune
```

The suite grades the serving **contract** (strict JSON, schema, enum domains, no
wrapper text, no thinking text, no invented `requested_action`, no over-emission
on single-topic input) as fatal, and reports **quality** notes (multi-topic topic
count, domain set, `requested_action` recall) without failing. That split is
deliberate: the contract is format-sensitive, the quality gaps are known model
limitations documented in `ml/reports/v2_error_analysis.md`.

## Known repository defect: the smoke v1/v2 comparison

`ml/data/tune/smoke/results/finetuned/` and `.../finetuned-v2/` are
**byte-identical** (same md5), and both hold **v2** output. The real v1 output
survives only in `.../v1_finetuned_stash/`.

Because `ml/tune/eval_smoke.py:evaluate_all()` only scores the tags `finetuned`
and `base`, running it reports the **v2** numbers in the v1 slot. To reproduce
the v1 baseline you must score the stash explicitly:

```python
import ml.tune.eval_smoke as E
rows = E.load_results("v1_finetuned_stash")
ev = {r["id"]: E.evaluate_case(r) for r in rows}
print(E.summarize("v1", ev))
```

True scores, both scored with the same evaluator:

| | v1 (stash) | v2 (this model) |
|---|---|---|
| Passed | 16/20 | 17/20 |
| Failed cases | `smoke-13`, `14`, `15`, `16` | `smoke-14`, `15`, `16` |
| Multi-topic | 0/4 | 1/4 |

v2 does improve on v1 — the multi-topic `smoke-13` case flips to pass — but the
comparison is only meaningful via the stash. Do not cite `eval_smoke.py`'s
default output as a v1 number.
