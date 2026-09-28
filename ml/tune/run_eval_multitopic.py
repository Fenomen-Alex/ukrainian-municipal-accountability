"""Multi-topic evaluation driver.

Runs a model/mode over the held-out multi-topic suite
(``ml/data/tune/multitopic/eval.jsonl``) and reports the metrics that matter for
the v2 goal: can the model emit more than one topic when the input is genuinely
multi-topic, without that spilling over into over-emission (measured separately
on the single-topic frozen test via ``run_eval.py`` ``multi_topic_rate``).

Modes match ``run_eval.py``: deterministic | base | lora.

Metrics:
  - json_parse_rate / schema_validity_rate        (reuse shared validator)
  - topic_count_accuracy   pred topic count == target topic count
  - multi_topic_recall     target multi-topic AND pred emitted >=2 topics
  - domain_set_exact       predicted domain set == target domain set
  - domain_set_precision/recall (predictive duplicates collapse to a set)
  - issue_rouge_l_best     per target topic, best ROUGE-L across predicted topics
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from ml.tune.build_multitopic import OUT_DIR as MT_DIR
from ml.tune.build_dataset import SYSTEM_PROMPT
from ml.tune.evaluate import (
    Prediction,
    _rouge_l,
    export_run,
    load_validator,
    parse_payload,
)


def _load_suite() -> list[dict]:
    return [
        json.loads(l) for l in (MT_DIR / "eval.jsonl").read_text().splitlines()
    ]


def _targets_and_inputs(suite: list[dict]) -> tuple[list[dict], list[str]]:
    targets, inputs = [], []
    for i, ex in enumerate(suite):
        target = json.loads(ex["messages"][-1]["content"])
        targets.append({"idx": i, "uid": ex["uid"], "topics": target["topics"],
                        "source_text": ex["messages"][1]["content"].strip()})
        inputs.append(ex["messages"][1]["content"].strip())
    return targets, inputs


def _deterministic_predictions(targets: list[dict]) -> list[Prediction]:
    """Baseline: reproduce the (oracle) target verbatim; measures ceiling of the
    metric harness itself, not a real system."""
    preds = []
    for t in targets:
        payload = {"topics": t["topics"]}
        p = Prediction(idx=t["idx"], uid=t["uid"],
                       raw=json.dumps(payload, ensure_ascii=False))
        p.parsed = payload
        p.topics = payload["topics"]
        preds.append(p)
    return preds


def _model_predictions(model: str, adapter_path: str | None, inputs: list[str],
                       uids: list[str], max_tokens: int, temperature: float) -> list[Prediction]:
    from mlx_lm import generate, load
    from mlx_lm.sample_utils import make_sampler

    sampler = make_sampler(temp=temperature)
    print(f"loading {model} ...", flush=True)
    t0 = time.time()
    model_, tokenizer = load(model, adapter_path=adapter_path)
    print(f"loaded in {time.time()-t0:.1f}s", flush=True)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": ""},
    ]

    preds = []
    for i, user_text in enumerate(inputs):
        messages[1]["content"] = user_text
        prompt = tokenizer.apply_chat_template(messages, tokenize=False,
                                               add_generation_prompt=True,
                                               enable_thinking=False)
        t0 = time.time()
        out = generate(model_, tokenizer, prompt=prompt, max_tokens=max_tokens,
                       sampler=sampler, verbose=False)
        p = parse_payload(out)
        p.idx = i
        p.uid = uids[i]
        preds.append(p)
        n = len(inputs)
        print(f"[{len(preds)}/{n}] {p.uid} {time.time()-t0:.1f}s", flush=True)
    return preds


def eval_multitopic(preds: list[Prediction], targets: list[dict], validator) -> dict:
    by_idx = {t["idx"]: t for t in targets}
    n = len(preds)
    if n == 0:
        return {}

    json_parse = sum(1 for p in preds if p.parsed is not None)
    schema_ok = 0
    topic_count_match = 0
    multi_recall_true, multi_recall_hit = 0, 0
    set_exact, set_prec, set_rec = [], [], []
    rl_best = []
    row = []
    for p in preds:
        errors = None if p.parsed is None else list(validator.iter_errors(p.parsed))
        if p.parsed is not None and not errors:
            schema_ok += 1

        t = by_idx[p.idx]
        t_domains = {t.get("domain") for t in t["topics"]}
        p_topics = p.topics if p.topics else []
        p_domains = {t.get("domain") for t in p_topics}

        if len(p_topics) == len(t["topics"]):
            topic_count_match += 1
        if set(t_domains) == set(p_domains):
            set_exact.append(1.0)
        inter = len(t_domains & p_domains)
        set_prec.append(inter / len(p_domains) if p_domains else 0.0)
        set_rec.append(inter / len(t_domains) if t_domains else 0.0)

        if len(t["topics"]) >= 2:
            multi_recall_true += 1
            if len(p_topics) >= 2:
                multi_recall_hit += 1

        for target_top in t["topics"]:
            best = max((_rouge_l(target_top.get("issue", ""), pc.get("issue", ""))
                        for pc in p_topics), default=0.0)
            rl_best.append(best)

        row.append({
            "uid": p.uid,
            "target_n": len(t["topics"]),
            "target_domains": sorted(t_domains),
            "pred_n": len(p_topics),
            "pred_domains": sorted(p_domains),
            "domain_set_exact": sorted(t_domains) == sorted(p_domains),
        })

    return {
        "n": n,
        "json_parse_rate": round(json_parse / n, 4),
        "schema_validity_rate": round(schema_ok / n, 4),
        "topic_count_accuracy": round(topic_count_match / n, 4),
        "multi_topic_recall": round(multi_recall_hit / multi_recall_true, 4) if multi_recall_true else float("nan"),
        "domain_set_exact": round(sum(set_exact) / n, 4),
        "domain_set_precision": round(mean(set_prec), 4),
        "domain_set_recall": round(mean(set_rec), 4),
        "issue_rouge_l_best": round(mean(rl_best), 4),
        "per_example": row,
    }


def mean(vals: list[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=["deterministic", "base", "lora"], required=True)
    ap.add_argument("--model", default="mlx-community/Qwen3-8B-4bit")
    ap.add_argument("--adapter-path", default=None)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument(
        "--tag", default=None,
        help=(
            "name for the result file; defaults to --mode. See the note in "
            "ml/tune/run_eval.py -- the multitopic suite is shared by every arm, "
            "so overwriting it destroys the comparison"
        ),
    )
    args = ap.parse_args()

    suite = _load_suite()
    targets, inputs = _targets_and_inputs(suite)
    uids = [t["uid"] for t in targets]

    if args.mode == "deterministic":
        preds = _deterministic_predictions(targets)
    elif args.mode == "lora":
        preds = _model_predictions(args.model, args.adapter_path, inputs, uids,
                                   args.max_tokens, args.temperature)
    else:
        preds = _model_predictions(args.model, None, inputs, uids,
                                   args.max_tokens, args.temperature)

    validator = load_validator()
    metrics = eval_multitopic(preds, targets, validator)

    tag = args.tag or args.mode
    out_dir = MT_DIR / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    export_run(out_dir / f"{tag}.json", tag, preds, metrics)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()