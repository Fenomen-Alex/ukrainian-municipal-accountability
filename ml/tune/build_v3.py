"""Assemble the v3 TREATMENT training set: v1 data + the C1-C4 changes.

    ml/data/tune/v3/treatment/train.jsonl
        = v1 train, relabelled with C1 (issue) and C2 (requested_action)
        + C3: terse note kernels upsampled TERSE_UPSAMPLE times
        + C4: multi-topic augmentation reweighted toward the real examples

    validation.jsonl / test.jsonl
        = byte-identical copies of the frozen v2 splits, so all three arms are
          compared on exactly the same held-out data.

    meta.json
        = composition, before/after counts for every change, and the leakage
          statement.

The control arm (``ml.tune.build_control``) is byte-identical to v2, so the only
difference between control and treatment is C1-C4. That is what makes the
comparison causal rather than merely suggestive.

    .venv/bin/python -m ml.tune.build_v3
"""

from __future__ import annotations

import json
from pathlib import Path

from ml.tune.build_dataset import (
    DATA_DIR,
    SYSTEM_PROMPT,
    _clean_text,
    _derive_attributes,
    _derive_object,
)
from ml.tune.build_multitopic import _note_kernel
from ml.tune.build_v2 import MT_REAL_UPSAMPLE, MT_UPSAMPLE
from ml.tune.v3_changes import (
    MT_REAL_UPSAMPLE_V3,
    MT_SYNTHETIC_UPSAMPLE_V3,
    TERSE_UPSAMPLE,
    clean_issue_c1,
    extract_action_c2,
    terse_pool,
)

V1_DIR = DATA_DIR / "tune"
MT_DIR = V1_DIR / "multitopic"
V2_DIR = V1_DIR / "v2"
OUT = V1_DIR / "v3" / "treatment"

#: Assistant turn, i.e. the training target.
_ASSISTANT = 2


def _load(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines()]


def _user_text(chat: dict) -> str:
    for m in chat["messages"]:
        if m["role"] == "user":
            return m["content"]
    return ""


def _topics_of(chat: dict) -> list[dict]:
    return json.loads(_assistant_text(chat))["topics"]


def _assistant_text(chat: dict) -> str:
    for m in chat["messages"]:
        if m["role"] == "assistant":
            return m["content"]
    raise ValueError(f"no assistant turn in {chat.get('uid')!r}")


def _relabel(chat: dict) -> tuple[dict, dict]:
    """Apply C1 and C2 to every topic of a chat example.

    C1 rewrites ``issue`` from the citizen's own text. C2 re-derives
    ``requested_action`` as a request *clause*. ``object`` and ``attributes`` are
    left alone: they were never defective, and re-deriving them would change
    fields the experiment is not about.

    Returns (new chat, before/after counters).
    """
    text = _user_text(chat)
    topics = _topics_of(chat)
    before = {
        "boilerplate_issue": 0,
        "action_whole_sentence": 0,
        "action_empty": 0,
        "action_nonempty": 0,
    }
    out_topics = []
    for t in topics:
        nt = dict(t)
        new_issue = clean_issue_c1(text)
        if new_issue != t.get("issue", ""):
            before["boilerplate_issue"] += 1
        nt["issue"] = new_issue
        new_action = extract_action_c2(text)
        if not new_action:
            before["action_empty"] += 1
        else:
            before["action_nonempty"] += 1
        nt["requested_action"] = new_action
        out_topics.append(nt)
    new = dict(chat)
    new["messages"] = list(chat["messages"])
    new["messages"][_ASSISTANT] = {
        "role": "assistant",
        "content": json.dumps({"topics": out_topics}, ensure_ascii=False),
    }
    return new, before


def _terse_examples(records: list[dict]) -> list[dict]:
    """C3: one chat example per terse record, kernel as the visible text.

    The kernel is the complaint with boilerplate and wrapper words deleted, so
    the target stays faithful to what the model is shown -- augmentation by
    condensing, never by inventing a second problem.
    """
    out = []
    for i, r in enumerate(records):
        kernel = _clean_text(_note_kernel(r.get("content") or ""))
        if len(kernel) < 25:
            continue
        if not clean_issue_c1(kernel).strip():
            continue
        domain = "other"
        topics = [{
            "domain": domain,
            "issue": clean_issue_c1(kernel),
            "object": _derive_object(kernel, r),
            "requested_action": extract_action_c2(kernel),
            "attributes": _derive_attributes(kernel, r),
        }]
        out.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": kernel},
                {"role": "assistant",
                 "content": json.dumps({"topics": topics}, ensure_ascii=False)},
            ],
            "uid": f"V3-TERSE-{i:05d}",
        })
    return out


def _upsample(chat: dict) -> list[dict]:
    """C4: weight the 22 real multi-topic texts above the templated ones."""
    real = chat["uid"].startswith("MT-REAL")
    n = MT_REAL_UPSAMPLE_V3 if real else MT_SYNTHETIC_UPSAMPLE_V3
    out = []
    for c in range(n):
        cp = dict(chat)
        cp["uid"] = f"{chat['uid']}#{c}"
        out.append(cp)
    return out


def _measure(examples: list[dict], source: str) -> dict:
    """Composition of a slice, so the report can show before/after per change."""
    topics = action_n = action_whole = 0
    n_multi = 0
    real_rows = 0
    for ex in examples:
        ts = _topics_of(ex)
        topics += len(ts)
        if len(ts) >= 2:
            n_multi += 1
        if ex["uid"].split("#")[0].startswith("MT-REAL"):
            real_rows += 1
        text = _user_text(ex)
        for _ in ts:
            a = json.loads(_assistant_text(ex))["topics"]
            break
        for t in a:
            act = t.get("requested_action", "")
            if act:
                action_n += 1
            if act and any(act == s.strip()
                            for s in __import__("re").split(r"(?<=[.!?])\s+", _clean_text(text))):
                action_whole += 1
    return {
        "source": source,
        "n_examples": len(examples),
        "n_topics": topics,
        "n_multi_topic_examples": n_multi,
        "n_real_multitopic_rows": real_rows,
        "real_row_share": round(real_rows / len(examples), 4) if examples else 0.0,
        "n_nonempty_action": action_n,
        "nonempty_action_share": round(action_n / topics, 4) if topics else 0.0,
        "action_is_whole_sentence": action_whole,
        "action_whole_sentence_share": round(action_whole / action_n, 4) if action_n else 0.0,
    }


def build() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)

    v1_train = _load(V1_DIR / "train.jsonl")
    raw_train = _load(DATA_DIR / "train.jsonl")

    # --- C1 + C2 on the single-topic stream ------------------------------- #
    single, dropped_empty, c12 = [], 0, {
        "boilerplate_issue": 0, "action_whole_sentence": 0,
        "action_empty": 0, "action_nonempty": 0}
    for ex in v1_train:
        new, before = _relabel(ex)
        # C1 can empty an ``issue`` when the record was *only* boilerplate (a
        # consent block with no complaint). Keeping it would train the model to
        # emit an empty issue, so these are dropped -- on integrity grounds, not
        # to hit a target share. The count is reported so the loss is visible.
        if not any(t["issue"].strip() for t in _topics_of(new)):
            dropped_empty += 1
            continue
        single.append(new)
        for k in c12:
            c12[k] += before[k]

    # --- C3 terse upsampling ---------------------------------------------- #
    pool = terse_pool(raw_train, _note_kernel)
    terse_units = _terse_examples(pool)
    terse = [dict(e) for e in terse_units for _ in range(TERSE_UPSAMPLE)]

    # --- C4 multitopic reweighting ---------------------------------------- #
    mt_base = _load(MT_DIR / "train.jsonl")
    mt = [c for ex in mt_base for c in _upsample(ex)]

    examples = single + terse + mt
    (OUT / "train.jsonl").write_text(
        "\n".join(json.dumps(e, ensure_ascii=False) for e in examples) + "\n",
        encoding="utf-8")

    for split in ("validation", "test"):
        (OUT / f"{split}.jsonl").write_text(
            (V2_DIR / f"{split}.jsonl").read_text(encoding="utf-8"), encoding="utf-8")

    meta = {
        "built_by": "ml/tune/build_v3.py",
        "arm": "treatment",
        "changes_applied": ["C1 boilerplate clean", "C2 requested_action",
                            "C3 terse upsampling", "C4 multitopic real-share"],
        "upsample_factors": {
            "terse": TERSE_UPSAMPLE,
            "mt_real": f"{MT_UPSAMPLE}/{MT_REAL_UPSAMPLE} -> "
                       f"{MT_SYNTHETIC_UPSAMPLE_V3}/{MT_REAL_UPSAMPLE_V3}",
        },
        "terse_pool": {
            "n_records": len(pool),
            "n_units": len(terse_units),
            "n_rows_after_upsample": len(terse),
            "criterion": "train record < 150 chars and note kernel >= 25 chars",
        },
        "c1_c2_counters": c12,
        "dropped_empty_issue": {
            "n": dropped_empty,
            "reason": (
                "record was only boilerplate (consent/response template, no "
                "complaint), so C1 leaves no issue to learn from; kept would teach "
                "the model to emit an empty issue"
            ),
        },
        "composition": {
            "single_topic_v1": len(single),
            "terse_augmented": len(terse),
            "multitopic_augmented": len(mt),
            "total": len(examples),
            "terse_share": round(len(terse) / len(examples), 4),
            "multitopic_share": round(len(mt) / len(examples), 4),
        },
        "slices": {
            "single": _measure(single, "v1 train, C1+C2 relabelled"),
            "terse": _measure(terse, "C3 terse augmentation"),
            "multitopic": _measure(mt, "C4 reweighted multitopic"),
        },
        "validation": "frozen copy of v2 validation.jsonl",
        "test": "frozen copy of v2 test.jsonl",
        "leakage": (
            "every augmentation source is a train-split record; the terse pool and "
            "the multi-topic set are disjoint by construction and neither appears in "
            "the frozen validation/test copies, which are byte-identical to v2."
        ),
    }
    (OUT / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


if __name__ == "__main__":
    m = build()
    print(f"treatment -> {OUT}")
    c = m["composition"]
    print(f"  total {c['total']}  single {c['single_topic_v1']}  "
          f"terse {c['terse_augmented']}  multitopic {c['multitopic_augmented']}")
    print(f"  terse share {c['terse_share']}  multitopic share {c['multitopic_share']}")
    for name, s in m["slices"].items():
        print(f"  [{name}] real_share={s['real_row_share']} "
              f"action_nonempty={s['nonempty_action_share']} "
              f"action_whole_sent={s['action_whole_sentence_share']}")
