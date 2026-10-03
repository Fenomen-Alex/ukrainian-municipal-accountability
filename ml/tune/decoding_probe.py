"""Inference-only decoding probe: does generation settings, not training, cause the
remaining schema failures?

This is a **diagnostic**, not a scoring run. It changes nothing about how the
model was trained and nothing about how it is judged:

* the prompt comes from :func:`ml.tune.serve_v2.build_prompt`, the canonical
  serving contract, imported rather than reimplemented so it cannot drift;
* behaviour comes from :func:`ml.tune.behavior_v3.classify`, the same classifier
  that produced the recorded SPLIT/MERGE/STOP/JSONFAIL buckets;
* tolerant parsing comes from :func:`ml.tune.run_eval_v3.parse_payload`, exactly
  as ``behavior_v3`` uses it, so buckets stay comparable with the committed run;
* strict parsing additionally comes from :func:`ml.tune.serve_v2.parse_strict`
  so the serving contract's own JSON requirement is measured separately.

The only thing this module varies is the sampler and the token budget. There is
no prompt edit, no post-processing, no JSON repair and no evaluator change.

Case selection is derived from committed artefacts so it is reproducible:

``jsonfail``   every case that failed strict JSON under *any* of the three arms
               in ``eval_v3/behaviour/*.json`` (14 cases).
``twotopic``   the 52 expected-two-topic cases.
``control``    single-topic cases that were schema- *and* json-valid under all
               four arms, i.e. cases where the model already succeeds, so any
               change under a different setting is a regression introduced by
               the setting rather than noise.

Usage::

    .venv-mlx/bin/python -m ml.tune.decoding_probe            # full grid
    .venv-mlx/bin/python -m ml.tune.decoding_probe --resume   # continue
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from ml.tune.behavior_v3 import CASES, classify
from ml.tune.run_eval_v3 import parse_payload
from ml.tune.serve_v2 import build_prompt, parse_strict

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "tune"
OUT_DIR = DATA_DIR / "decoding_probe"
OUT = OUT_DIR / "results.jsonl"

#: ``temp=0.0 / max_tokens=800`` is the canonical serving contract
#: (``serve_v2.DEFAULT_TEMPERATURE`` / ``DEFAULT_MAX_TOKENS``) and therefore the
#: *baseline*. Its raw generations are already committed, so it is not
#: regenerated here; the other rows are one-factor-at-a-time departures from it.
SETTINGS: dict[str, dict] = {
    "budget1200": {"temp": 0.0, "top_p": 0.0, "max_tokens": 1200, "rep": None},
    "reppen105": {"temp": 0.0, "top_p": 0.0, "max_tokens": 800, "rep": (1.05, 20)},
    "reppen1200": {"temp": 0.0, "top_p": 0.0, "max_tokens": 1200, "rep": (1.05, 20)},
    # Temperature isolated from the penalty: identical to ``temp01`` but with no
    # repetition processor, so the two rows separate "sampling helps" from
    # "penalising repetition helps". Without this pair a ``temp01`` repair could
    # not be attributed to either factor.
    "temp01_norep": {"temp": 0.1, "top_p": 0.9, "max_tokens": 1200, "rep": None},
    "temp01": {"temp": 0.1, "top_p": 0.9, "max_tokens": 1200, "rep": (1.05, 20)},
}

#: Seed for every sampled row. Greedy rows (temp 0.0) are deterministic in
#: mlx_lm, but the sampled rows are not, so without a seed the committed
#: artefact could not be reproduced. Seeded per generation, not once per run, so
#: a row's output depends only on (model, setting, case) and not on how many
#: generations preceded it -- otherwise ``--resume`` and a single-shot run would
#: disagree.
SEED = 20260902

MODELS = {
    "v2": "ml/data/tune/adapters/qwen3-8b-lora-v2-attempt10-fused",
    "corrected-v3": "ml/data/tune/adapters/qwen3-8b-lora-v3-corrected-fused",
    "v3-2": "ml/data/tune/adapters/qwen3-8b-lora-v3-2-fused",
}

#: The control sample spreads single-topic controls over categories. A category
#: may repeat, so selection is by occurrence index into that category's pool
#: rather than always the first id: taking ``sorted(pool)[0]`` for every slot
#: collapses repeated categories onto one case and silently yields 6 controls
#: where 12 were asked for.
CONTROL_CATEGORIES = ["A", "A", "A", "H", "H", "H", "I", "I", "S", "S", "M", "V"]


def _cases() -> list[dict]:
    return [json.loads(l) for l in CASES.read_text(encoding="utf-8").splitlines() if l.strip()]


def _arm_result(name: str) -> dict[str, dict]:
    p = DATA_DIR / "eval_v3" / "results" / f"{name}.json"
    return {c["id"]: c for c in json.loads(p.read_text(encoding="utf-8"))["per_case"]}


def build_caselist() -> list[dict]:
    """Reproducible case selection: 52 two-topic + 14 jsonfail + 12 control."""
    cases = {c["id"]: c for c in _cases()}
    jsonfail: set[str] = set()
    for p in sorted((DATA_DIR / "eval_v3" / "behaviour").glob("*.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        jsonfail |= {c["id"] for c in d["cases"] if c["behaviour"] == "JSONFAIL"}

    two = [c for c in cases.values()
           if c.get("expected_topic_count", len(c.get("expected_topics", []))) == 2]

    per_case = {a: _arm_result(a) for a in ("v2", "v3-treatment", "v3-corrected", "v3-2")}
    ctrl_pool: dict[str, list[str]] = {}
    for cid, c in per_case["v2"].items():
        if c["expected_topic_count"] != 1:
            continue
        if all(per_case[a][cid]["schema_ok"] and per_case[a][cid]["json_ok"] for a in per_case):
            ctrl_pool.setdefault(c["category"], []).append(cid)
    control: list[str] = []
    seen_control: set[str] = set()
    for k, cat in enumerate(CONTROL_CATEGORIES):
        pool = sorted(ctrl_pool.get(cat, []))
        if not pool:
            raise SystemExit(f"no valid control case in category {cat}")
        # kth *distinct* member of this category's pool; distinct pools are
        # keyed by category so ids cannot collide across categories.
        offset = sum(1 for c in CONTROL_CATEGORIES[:k] if c == cat)
        pick = pool[offset % len(pool)]
        if pick in seen_control:
            raise SystemExit(f"control slot {k} in category {cat} reused case {pick}")
        seen_control.add(pick)
        control.append(pick)

    out: dict[str, dict] = {}
    for cid in jsonfail:
        out[cid] = {"tags": ["jsonfail"]}
    for c in two:
        tags = out.setdefault(c["id"], {"tags": []})["tags"]
        tags.append("twotopic")
    for cid in control:
        tags = out.setdefault(cid, {"tags": []})["tags"]
        tags.append("control")
    return [{"id": cid, "tags": sorted(v["tags"]), "category": cases[cid]["category"]}
            for cid, v in sorted(out.items())]


def _gen_kwargs(cfg: dict) -> dict:
    """Build ``generate()`` kwargs for one probe setting.

    ``mlx_lm`` 0.31.x takes a *sampler* ``(logprobs) -> token`` and, separately,
    ``logits_processors`` ``(tokens, logits) -> logits``. Repetition penalty is a
    logit processor, not a sampler, so it is wired through the native
    ``logits_processors`` argument rather than being composed onto the sampler
    by hand. Nothing else about decoding is touched.
    """
    from mlx_lm.sample_utils import make_repetition_penalty, make_sampler

    kw = {"sampler": make_sampler(temp=cfg["temp"], top_p=cfg["top_p"])}
    if cfg["rep"] is not None:
        penalty, ctx = cfg["rep"]
        kw["logits_processors"] = [make_repetition_penalty(penalty, ctx)]
    return kw


def run(resume: bool = True) -> None:
    from mlx_lm import generate, load

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "cases.json").write_text(
        json.dumps(build_caselist(), indent=2), encoding="utf-8")

    done: set[tuple] = set()
    if resume and OUT.exists():
        for line in OUT.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                done.add((r["model"], r["setting"], r["id"]))

    todo = [(m, s, c) for m in MODELS for s in SETTINGS for c in build_caselist()]
    todo = [t for t in todo if (t[0], t[1], t[2]["id"]) not in done]
    print(f"{len(todo)} generations to run ({len(done)} already present)", flush=True)
    if not todo:
        return

    fh = OUT.open("a", encoding="utf-8")
    for mi, (mname, mpath) in enumerate(MODELS.items(), 1):
        pending = [t for t in todo if t[0] == mname]
        if not pending:
            continue
        t0 = time.time()
        model, tok = load(mpath)
        print(f"[{mi}/{len(MODELS)}] loaded {mname} in {time.time() - t0:.0f}s "
              f"({len(pending)} gens)", flush=True)
        for n, (_, sname, case) in enumerate(pending, 1):
            cfg = SETTINGS[sname]
            full = next(c for c in _cases() if c["id"] == case["id"])
            # Per-generation seed, derived from the coordinates, so the row is
            # reproducible from its own identity and independent of run order.
            seed = SEED + (list(MODELS).index(mname) * 1000003
                           + sorted(SETTINGS).index(sname) * 10007
                           + int(case["id"].split("-")[1]))
            import mlx.core as mx
            mx.random.seed(seed)
            raw = generate(model, tok, prompt=build_prompt(full["text"], tok),
                           max_tokens=cfg["max_tokens"], **_gen_kwargs(cfg),
                           verbose=False)
            parsed = parse_payload(raw)
            # Mirror ml.tune.behavior_v3.generate exactly: the bucket is decided
            # by parse_payload, not by parse_strict, so recorded buckets stay
            # comparable with the committed run.
            inner = getattr(parsed, "parsed", None)
            topics = inner.get("topics") if isinstance(inner, dict) else None
            payload_err = None if topics is not None else "unparseable"
            # parse_strict is measured separately: it is the serving contract's
            # own requirement and catches output parse_payload's {.*} search
            # would have silently salvaged.
            strict_payload, strict_err = parse_strict(raw)
            beh = classify(full, raw, payload_err, topics)
            fh.write(json.dumps({
                "model": mname, "setting": sname, "id": case["id"],
                "tags": case["tags"], "category": case["category"],
                "max_tokens": cfg["max_tokens"], "temp": cfg["temp"],
                "top_p": cfg["top_p"], "rep": cfg["rep"], "seed": seed,
                "raw": raw, "raw_chars": len(raw),
                "strict_ok": strict_err == "" and strict_payload is not None,
                "strict_error": strict_err,
                "behaviour": beh["behaviour"],
                "predicted_topic_count": beh.get("predicted_topic_count"),
            }, ensure_ascii=False) + "\n")
            fh.flush()
            if n % 10 == 0 or n == len(pending):
                print(f"  {mname}/{sname} {n}/{len(pending)}", flush=True)
        del model
    fh.close()
    print("done", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--resume", action="store_true",
                    help="skip (model,setting,id) already in results.jsonl")
    args = ap.parse_args()
    run(resume=args.resume)
