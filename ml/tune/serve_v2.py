"""Canonical inference contract for the v2 Qwen3-8B municipal extractor.

The v2 fine-tune was trained on chat examples that *always* carried the
``build_dataset.SYSTEM_PROMPT`` system message and were tokenized with Qwen3's
chat template at ``enable_thinking=False``. Both facts are load-bearing, and
this module is the single place that encodes them.

Exact text the fine-tune expects immediately before generation::

    <|im_start|>system
    {SYSTEM_PROMPT}<|im_end|>
    <|im_start|>user
    {complaint}<|im_end|>
    <|im_start|>assistant
    <think>

    </think>

The model then emits the JSON object followed by ``<|im_end|>``.

Do not hand-edit the prompt. Build it with :func:`build_prompt` so the system
message and the empty-``<think>`` block cannot be dropped by accident. See
``ml/tune/V2_SERVING.md`` for the equivalent LM Studio / HTTP settings.

Usage::

    .venv-mlx/bin/python -m ml.tune.serve_v2 --text "Не горять ліхтарі"
    echo "Не горять ліхтарі" | .venv-mlx/bin/python -m ml.tune.serve_v2 --stdin
"""

from __future__ import annotations

import argparse
import json
import sys

from ml.tune.build_dataset import SYSTEM_PROMPT, normalize_quotes

#: The exact v2 artifact. Byte-identical to ``qwen3-8b-lora-v2-fused``.
DEFAULT_MODEL = "ml/data/tune/adapters/qwen3-8b-lora-v2-attempt10-fused"

#: Base model the LoRA was trained against (4-bit MLX quantization).
BASE_MODEL = "mlx-community/Qwen3-8B-4bit"

#: Every v2 training target was a single compact JSON object; 800 is the same
#: budget ``run_smoke`` uses and leaves ample room for multi-topic output.
DEFAULT_MAX_TOKENS = 800

#: Greedy decoding. The recorded v2 metrics were all produced at temp=0.0.
DEFAULT_TEMPERATURE = 0.0

#: Empty thinking block that ``enable_thinking=False`` injects. The fine-tune was
#: trained with this present and it must not be stripped.
THINK_PREFIX = "<think>\n\n</think>\n\n"


def build_messages(complaint: str) -> list[dict]:
    """The v2 message list: the fixed system prompt plus the citizen text.

    ``normalize_quotes`` is applied here, at the single boundary every serving
    path goes through, so a raw ASCII double quote in live traffic can never be
    copied verbatim into a JSON ``issue`` string.  Text without ASCII quotes is
    returned byte-identical, so the v2 contract is unchanged for it.
    """
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": (normalize_quotes(complaint) or "").strip()},
    ]


def build_prompt(complaint: str, tokenizer) -> str:
    """Render the exact prompt the v2 fine-tune was trained on.

    ``enable_thinking=False`` is mandatory: it is what produces the
    ``<think>\\n\\n</think>\\n\\n`` generation prefix.
    """
    return tokenizer.apply_chat_template(
        build_messages(complaint),
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def load_model(model_path: str = DEFAULT_MODEL):
    """Load the MLX model + tokenizer."""
    from mlx_lm import load

    return load(model_path)


def generate_text(
    model,
    tokenizer,
    complaint: str,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
) -> str:
    """Greedy-decode one complaint. Returns the raw continuation."""
    from mlx_lm import generate
    from mlx_lm.sample_utils import make_sampler

    prompt = build_prompt(complaint, tokenizer)
    sampler = make_sampler(temp=temperature)
    return generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=max_tokens,
        sampler=sampler,
        verbose=False,
    )


def parse_strict(text: str) -> tuple[dict | None, str]:
    """Parse the generation as JSON with *no* wrapper tolerance.

    ``ml.tune.evaluate.parse_payload`` searches for ``\\{.*\\}`` and silently
    discards surrounding text. That is right for scoring historical runs but
    wrong for verifying a serving contract, because a leading ``!``, a stray
    sentence, or a leaked chat turn are exactly the defects being checked for.
    Returns ``(payload, error)``.
    """
    raw = (text or "").strip()
    if not raw:
        return None, "empty output"
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        return None, f"not strict JSON: {exc}"
    if not isinstance(payload, dict):
        return None, f"top level is {type(payload).__name__}, expected object"
    if not isinstance(payload.get("topics"), list):
        return None, "missing 'topics' array"
    return payload, ""


def extract(complaint: str, model_path: str = DEFAULT_MODEL, **kw) -> dict:
    """One-shot convenience: load, generate, strictly parse."""
    model, tokenizer = load_model(model_path)
    prompt = build_prompt(complaint, tokenizer)
    raw = generate_text(model, tokenizer, complaint, **kw)
    payload, error = parse_strict(raw)
    return {"prompt": prompt, "raw": raw, "payload": payload, "error": error}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--text", default=None, help="complaint text")
    ap.add_argument("--stdin", action="store_true", help="read complaint from stdin")
    ap.add_argument("--show-prompt", action="store_true", help="print the rendered prompt")
    ap.add_argument("--request-only", action="store_true",
                    help="print a ready-to-paste /v1/chat/completions request body "
                         "and exit without loading the model")
    ap.add_argument("--model-id", default="qwen3-8b-municipal-finetune",
                    help="model id to put in --request-only output")
    ap.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    ap.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    args = ap.parse_args()

    if args.stdin:
        complaint = sys.stdin.read()
    elif args.text is not None:
        complaint = args.text
    else:
        complaint = sys.stdin.read()

    complaint = complaint.strip()
    if not complaint:
        ap.error("no complaint text given (use --text or --stdin)")

    if args.request_only:
        print(json.dumps({
            "model": args.model_id,
            "messages": build_messages(complaint),
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "chat_template_kwargs": {"enable_thinking": False},
        }, ensure_ascii=False))
        return

    model, tokenizer = load_model(args.model)
    if args.show_prompt:
        print(build_prompt(complaint, tokenizer))
        print("---", file=sys.stderr)

    raw = generate_text(
        model,
        tokenizer,
        complaint,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )
    print(raw)
    payload, error = parse_strict(raw)
    if error:
        print(f"[strict parse failed: {error}]", file=sys.stderr)
        raise SystemExit(1)
    print("[strict parse ok]", file=sys.stderr)


if __name__ == "__main__":
    main()
