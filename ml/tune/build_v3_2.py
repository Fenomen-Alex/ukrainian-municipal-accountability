"""Build the v3.2 treatment corpus: corrected v3 + one targeted intervention.

    ml/data/tune/v3/treatment_v3_2/train.jsonl

v3.2 differs from the corrected v3 corpus in exactly one way: it adds a
**same-object two-topic** stream (``MT-SYN-SO-*``) that teaches the model to
split two independent problems which concern the *same* object, when both
appear inline in a single sentence.

Why this and nothing else
-------------------------
The corrected-v3 multitopic regression is topic *merging*, not truncation
(``ml/tune/V3_MULTITOPIC_ANALYSIS.md``): on the 13 genuine two-topic
under-emissions, previous v3 split 13/13 and corrected v3 merged 12/13, while
the STOP bucket *shrank* 10 -> 6. Both problems are copied into the output;
they are simply not assigned to separate topics. eval_v3 category E -- the only
same-object slice -- collapsed from 6/6 to 1/6.

The existing ``MT-SYN-SS-*`` (same-street) family looks like it should already
cover this, and it does carry a shared street, but it joins two *complete
standalone complaints*:

    "<complaint A>. а також <complaint B>."

Each half is its own sentence with its own request frame, so the boundary is
explicitly marked. MT85 is built the same way, and MT85 barely moved (-2)
while eval_v3 collapsed (-11). That asymmetry is the whole diagnosis: the
model splits boundary-marked complaints and merges inline ones.

Category E has no such marker. It is one continuous sentence:

    "По вул. X: <clause A>, а також <clause B>."

with a shared frame, a comma before the connective, and both topics carrying
the same ``object``. No training example had that shape. This stream adds it.

What is deliberately NOT done
-----------------------------
* C1, R1, R2, R3, C2, C3 and the existing C4 streams are reused verbatim from
  ``ml.tune.build_v3``. The new stream is appended after them and is passed
  through the same ``_relabel_multitopic`` C1/C2 pass, so R2's boilerplate
  stripping applies to it exactly as it does to every other multitopic row.
* The single-topic stream's verbatim-copy rate is NOT capped. That was
  considered and rejected: the analysis identified it as the *mechanism*, but
  capping it would confound this experiment with a second, unrelated change.
* The single-stream length is NOT padded. Stream alignment therefore still has
  4 mixed-stream batches, as corrected v3 had 3. Padding would mean editing a
  C1-C4 stream, which is out of scope. The total is nevertheless **even**, which
  is what removes the worse defect: ``iters == len(batches)`` exactly, so v3.2
  trains every row exactly once with no dropped tail row and no wrap into a
  reshuffled second epoch.

Sources and leakage
-------------------
Every source record comes from ``ml/data/train.jsonl``. Any record whose
normalised content appears in the frozen validation or test splits is excluded
up front, and the whole held-out content set is re-checked after assembly. The
eval_v3 category-E cases are themselves composed from *test*-split records, so
this exclusion is what keeps the new stream from training on its own test set.

    .venv/bin/python -m ml.tune.build_v3_2
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections import defaultdict
from pathlib import Path

from ml.cleaner import normalize_text
from ml.tune import build_v3
from ml.tune.build_dataset import DATA_DIR, KIND_TO_DOMAIN
from ml.tune.build_eval_v3 import (
    _street,
    condense,
    held_out,
    load_split,
    ref_topic,
)
from ml.tune.build_multitopic import _chat_example, _note_kernel
from ml.tune.v3_changes import TERSE_UPSAMPLE, terse_pool

V1_DIR = DATA_DIR / "tune"
MT_DIR = V1_DIR / "multitopic"
V2_DIR = V1_DIR / "v2"
BASE_OUT = V1_DIR / "v3" / "treatment"
OUT = V1_DIR / "v3" / "treatment_v3_2"

#: UID stem for the new stream. Deliberately distinct from ``-SS-`` (same
#: street, boundary-marked) so the two families never collide and can be
#: counted apart.
SAME_OBJECT_PREFIX = "MT-SYN-SO"

#: Number of base same-object examples to synthesise.
#:
#: 261 is inside the 250-300 band documented in V3_MULTITOPIC_ANALYSIS.md and is
#: **odd on purpose**: corrected v3 has 9,447 rows, so an odd increment yields
#: 9,708 -- an even total -- which makes ``iters == len(batches)`` exact.
SAME_OBJECT_N = 261

#: The new stream is not replicated. Corrected v3 already upsamples synthetic
#: multitopic 2x; doing it again here would inflate volume rather than target
#: the failure, which the analysis explicitly warned against.
SAME_OBJECT_UPSAMPLE = 1

#: At most this many examples are drawn from any single street, so the 261 rows
#: are spread over many locations instead of saturating one.
MAX_PER_STREET = 3

#: A condensed clause shorter than this is a fragment, not an independent
#: problem, and would teach arbitrary splitting.
MIN_CLAUSE_CHARS = 30

SEED = 42


def _usable(r: dict) -> bool:
    """Mirrors the eval builder's local ``usable``: a non-empty reference issue."""
    return bool(ref_topic(r)["issue"].strip())


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def _topics_of(chat: dict) -> list[dict]:
    return build_v3._topics_of(chat)


def same_object_pool() -> tuple[dict[str, list[dict]], set[str]]:
    """Train records with a street, excluding anything frozen held out.

    Returns the street-keyed pool and the frozen normalised-content set, so the
    caller can re-assert leakage after assembly.
    """
    frozen = {normalize_text(r.get("content") or "") for r in held_out()}
    by_street: dict[str, list[dict]] = defaultdict(list)
    for r in load_split("train"):
        if normalize_text(r.get("content") or "") in frozen:
            continue
        if not _usable(r):
            continue
        s = _street(r)
        if s:
            by_street[s].append(r)
    return by_street, frozen


def _pair_is_independent(ta: dict, tb: dict, ca: str, cb: str) -> bool:
    """Reject the failure modes this stream must not teach.

    Rejected: two synonymous phrasings of one issue, one action duplicated, an
    arbitrary fragment split, and "two topics merely because there are two
    verbs" (that one is prevented upstream by requiring different domains).
    """
    na, nb = normalize_text(ca), normalize_text(cb)
    if not na or na == nb:
        return False
    if normalize_text(ta["issue"]) == normalize_text(tb["issue"]):
        return False
    if len(ca) < MIN_CLAUSE_CHARS or len(cb) < MIN_CLAUSE_CHARS:
        return False
    aa, ab = normalize_text(ta.get("requested_action") or ""), normalize_text(tb.get("requested_action") or "")
    if aa and ab and aa == ab:
        return False
    return True


def build_same_object(n_target: int = SAME_OBJECT_N) -> tuple[list[dict], list[dict], dict]:
    """Synthesise the targeted same-object two-topic stream.

    Deterministic: streets are visited in sorted order and every draw comes from
    a seeded ``random.Random``. Rebuilding gives byte-identical output.

    Candidates are run through the same ``build_v3._relabel_multitopic`` C1/C2
    pass as every other multitopic row *while sampling*, and a candidate whose
    issue C1 empties is discarded on the same integrity grounds. ``n_target``
    therefore counts rows that actually reach the corpus, which is what makes
    the total row count predictable -- and therefore even, with ``n_target``
    odd.

    Returns (relabelled chat examples, provenance records, counters).
    """
    by_street, frozen = same_object_pool()
    rng = random.Random(SEED)
    streets = sorted(by_street)
    examples: list[dict] = []
    provenance: list[dict] = []
    seen: set[tuple[str, str]] = set()
    per_street: dict[str, int] = defaultdict(int)
    counters = {
        "pairs_available": 0,
        "rejected_same_domain": 0,
        "rejected_shared_object_mismatch": 0,
        "rejected_not_independent": 0,
        "rejected_leakage": 0,
        "rejected_duplicate_pair": 0,
        "street_exhausted": 0,
        "dropped_empty_issue_c1": 0,
        "c1_c2_counters": {"boilerplate_issue": 0, "action_whole_sentence": 0,
                           "action_empty": 0, "action_nonempty": 0},
    }
    attempts = 0
    max_attempts = max(1, n_target * 400)

    while len(examples) < n_target and attempts < max_attempts and streets:
        attempts += 1
        s = rng.choice(streets)
        if per_street[s] >= MAX_PER_STREET:
            counters["street_exhausted"] += 1
            continue
        recs = list(by_street[s])
        rng.shuffle(recs)
        chosen = None
        for i, ra in enumerate(recs):
            for rb in recs[i + 1:]:
                counters["pairs_available"] += 1
                da = KIND_TO_DOMAIN.get(ra.get("kind") or "", "other")
                db = KIND_TO_DOMAIN.get(rb.get("kind") or "", "other")
                if da == db:
                    counters["rejected_same_domain"] += 1
                    continue
                ua, ub = str(ra.get("uid")), str(rb.get("uid"))
                key = tuple(sorted((ua, ub)))
                if key in seen:
                    counters["rejected_duplicate_pair"] += 1
                    continue
                seen.add(key)
                chosen = (ra, rb, da, db)
                break
            if chosen:
                break
        if not chosen:
            continue

        ra, rb, da, db = chosen
        ta, tb = ref_topic(ra), ref_topic(rb)
        # The whole point of the stream: both topics must name the SAME object.
        if not (ta.get("object") and ta["object"] == tb.get("object")):
            counters["rejected_shared_object_mismatch"] += 1
            continue
        ca, cb = condense(ra), condense(rb)
        if not _pair_is_independent(ta, tb, ca, cb):
            counters["rejected_not_independent"] += 1
            continue

        text = f"По {s}: {ca}, а також {cb[0].lower() + cb[1:]}."
        if normalize_text(text) in frozen:
            counters["rejected_leakage"] += 1
            continue

        uid = f"{SAME_OBJECT_PREFIX}-{s}-{ra.get('uid')}-{rb.get('uid')}"
        # Same C1/C2 relabel as every other multitopic row, applied here so that
        # only surviving rows count toward n_target.
        relabelled, before = build_v3._relabel_multitopic(_chat_example(uid, text, [ta, tb]))
        if not all(t["issue"].strip() for t in _topics_of(relabelled)):
            counters["dropped_empty_issue_c1"] += 1
            continue
        for k in counters["c1_c2_counters"]:
            counters["c1_c2_counters"][k] += before[k]
        examples.append(relabelled)
        provenance.append({
            "uid": uid,
            "synthetic": True,
            "stream": "same_object_inline",
            "sources": [str(ra.get("uid")), str(rb.get("uid"))],
            "splits": ["train"],
            "kinds": [ra.get("kind"), rb.get("kind")],
            "domains": [da, db],
            "n_topics": 2,
            "joiner": ", а також ",
            "frame": f"По {s}: ",
            "same_object": True,
            "object": ta["object"],
            "street": s,
        })
        per_street[s] += 1

    return examples, provenance, counters


def _upsample_fixed(chat: dict, n: int) -> list[dict]:
    out = []
    for c in range(n):
        cp = dict(chat)
        cp["uid"] = f"{chat['uid']}#{c}"
        out.append(cp)
    return out


def _multitopic_base() -> tuple[list[dict], dict, int]:
    """C1+C2 on the C4 multitopic slice, identical to ``build_v3.build``."""
    mt_base = build_v3._load(MT_DIR / "train.jsonl")
    mt: list[dict] = []
    c12 = {"boilerplate_issue": 0, "action_whole_sentence": 0,
           "action_empty": 0, "action_nonempty": 0}
    dropped = 0
    for ex in mt_base:
        relabelled, before = build_v3._relabel_multitopic(ex)
        if not all(t["issue"].strip() for t in _topics_of(relabelled)):
            dropped += 1
            continue
        for c in build_v3._upsample(relabelled):
            mt.append(c)
        for k in c12:
            c12[k] += before[k]
    return mt, c12, dropped


def _single_and_terse() -> tuple[list[dict], list[dict], int, dict]:
    """C1+C2 single stream and C3 terse stream, identical to ``build_v3.build``."""
    single: list[dict] = []
    dropped_empty = 0
    c12 = {"boilerplate_issue": 0, "action_whole_sentence": 0,
           "action_empty": 0, "action_nonempty": 0}
    for ex in build_v3._load(V1_DIR / "train.jsonl"):
        new, before = build_v3._relabel(ex)
        if not any(t["issue"].strip() for t in _topics_of(new)):
            dropped_empty += 1
            continue
        single.append(new)
        for k in c12:
            c12[k] += before[k]

    pool = terse_pool(build_v3._load(DATA_DIR / "train.jsonl"), _note_kernel)
    units = build_v3._terse_examples(pool)
    terse = [dict(e) for e in units for _ in range(TERSE_UPSAMPLE)]
    return single, terse, dropped_empty, c12


def _verbatim_rate(examples: list[dict]) -> float | None:
    """Share of *single-topic* target issues that are exact substrings of the
    complaint.

    Returns ``None`` for a slice with no single-topic rows. The metric only means
    something there: a two-topic row has two issues and no single "the" issue to
    copy, so a 0.0 would read as a measurement when it is really an absence of
    one.
    """

    def norm(s: str) -> str:
        return re.sub(r"\s+", " ", (s or "").lower()
                      .replace("’", "'").replace("‘", "'")).strip()

    n = ok = 0
    for ex in examples:
        if len(_topics_of(ex)) != 1:
            continue
        text = norm(build_v3._user_text(ex))
        issue = norm(_topics_of(ex)[0].get("issue") or "")
        if not issue:
            continue
        n += 1
        ok += issue in text
    return round(ok / n, 4) if n else None


def _profile(examples: list[dict]) -> dict:
    n_topics: dict[int, int] = defaultdict(int)
    domains: dict[str, int] = defaultdict(int)
    for ex in examples:
        ts = _topics_of(ex)
        n_topics[len(ts)] += 1
        for t in ts:
            domains[t.get("domain", "")] += 1
    same_obj = sum(
        1 for ex in examples
        if len(_topics_of(ex)) == 2
        and _topics_of(ex)[0].get("object")
        and _topics_of(ex)[0]["object"] == _topics_of(ex)[1]["object"]
    )
    return {
        "rows": len(examples),
        "topic_count_distribution": {str(k): v for k, v in sorted(n_topics.items())},
        "domain_distribution": dict(sorted(domains.items(), key=lambda kv: (-kv[1], kv[0]))),
        "same_object_two_topic_rows": same_obj,
    }


def build() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)

    single, terse, dropped_empty, c12 = _single_and_terse()
    mt, mt_c12, mt_dropped = _multitopic_base()

    so_examples, so_prov, so_counters = build_same_object()

    # C1/C2 already ran inside build_same_object; only replicate here.
    so_rows = [c for ex in so_examples for c in _upsample_fixed(ex, SAME_OBJECT_UPSAMPLE)]
    so_c12 = so_counters["c1_c2_counters"]
    so_dropped = so_counters["dropped_empty_issue_c1"]

    examples = single + terse + mt + so_rows
    train_path = OUT / "train.jsonl"
    train_path.write_text(
        "\n".join(json.dumps(e, ensure_ascii=False) for e in examples) + "\n",
        encoding="utf-8")

    for split in ("validation", "test"):
        (OUT / f"{split}.jsonl").write_text(
            (V2_DIR / f"{split}.jsonl").read_text(encoding="utf-8"), encoding="utf-8")

    # Provenance sidecar: which train records produced each new row, so the
    # stream can be audited or re-derived without rerunning the sampler.
    (OUT / "provenance.jsonl").write_text(
        "\n".join(json.dumps(p, ensure_ascii=False) for p in so_prov) + "\n",
        encoding="utf-8")

    # --- leakage + duplicate checks, asserted on the assembled corpus ------- #
    held = {normalize_text(json.loads(l)["messages"][1]["content"])
            for name in ("validation", "test")
            for l in (V2_DIR / f"{name}.jsonl").read_text(encoding="utf-8").splitlines()
            if l.strip()}
    train_texts = [normalize_text(e["messages"][1]["content"]) for e in examples]
    overlap = len(set(train_texts) & held)
    dup_train = len(train_texts) - len(set(train_texts))
    so_texts = [normalize_text(e["messages"][1]["content"]) for e in so_rows]
    dup_so = len(so_texts) - len(set(so_texts))

    n_rows = len(examples)
    n_batches = n_rows // 2
    iters = -(-n_rows // 2)

    def stream_of(uid: str) -> str:
        u = uid.split("#")[0]
        if u.startswith(SAME_OBJECT_PREFIX):
            return "same_object"
        if u.startswith("V3-TERSE"):
            return "terse"
        if u.startswith("MT-REAL"):
            return "mt_real"
        if u.startswith("MT-"):
            return "mt_synth"
        return "single"

    # contiguous stream layout, which is what the MLX pairer actually sees
    layout: list[list] = []
    for e in examples:
        s = stream_of(e["uid"])
        if layout and layout[-1][0] == s:
            layout[-1][1] += 1
        else:
            layout.append([s, 1])
    mixed = sum(1 for i in range(0, n_batches * 2, 2)
                if stream_of(examples[i]["uid"]) != stream_of(examples[i + 1]["uid"]))

    meta = {
        "built_by": "ml/tune/build_v3_2.py",
        "arm": "treatment_v3_2",
        "base_arm": "v3-corrected",
        "intervention": (
            "single change: add a same-object two-topic stream (MT-SYN-SO-*) that "
            "teaches the eval_v3 category-E shape -- one continuous sentence, shared "
            "frame, ', а також ', identical object on both topics"
        ),
        "unchanged_from_corrected": [
            "C1 boilerplate clean", "C2 requested_action", "R1 terse domain fix",
            "R2 issue boilerplate strip", "R3 closer issue", "C3 terse upsampling",
            "C4 multitopic reweighting", "single-topic stream byte-for-byte",
            "training recipe", "evaluator", "serving format",
        ],
        "same_object_stream": {
            "uid_prefix": SAME_OBJECT_PREFIX,
            "n_base_examples": len(so_examples),
            "n_rows_after_upsample": len(so_rows),
            "target_is_post_c1_row_count": True,
            "upsample": SAME_OBJECT_UPSAMPLE,
            "max_per_street": MAX_PER_STREET,
            "min_clause_chars": MIN_CLAUSE_CHARS,
            "seed": SEED,
            "join_form": "По <street>: <clause A>, а також <clause B>.",
            "sampling_counters": so_counters,
            "dropped_empty_issue": so_dropped,
            "c1_c2_counters": so_c12,
        },
        "composition": {
            "single_topic_v1": len(single),
            "terse_augmented": len(terse),
            "multitopic_c4_base": len(mt),
            "multitopic_real_rows": sum(1 for e in mt if e["uid"].startswith("MT-REAL")),
            "multitopic_synthetic_rows": sum(1 for e in mt if e["uid"].startswith("MT-SYN")),
            "same_object_rows": len(so_rows),
            "total": n_rows,
            "batch_size": 2,
            "n_batches": n_batches,
            "iters_ceil": iters,
            "iters_equals_batches": iters == n_batches,
            "multitopic_share": round((len(mt) + len(so_rows)) / n_rows, 4),
            "same_object_share": round(len(so_rows) / n_rows, 4),
        },
        "file_order_layout": {
            "runs": [{"stream": s, "rows": c} for s, c in layout],
            "mixed_stream_batches": mixed,
            "note": (
                "MLX pairs consecutive rows in file order, so stream boundaries at "
                "odd offsets produce mixed-stream batches. v3.2 does not pad any "
                "C1-C4 stream to fix this (that would be a second change); it only "
                "makes the total even so iters == n_batches, which removes the "
                "dropped tail row and the wrap into a reshuffled second epoch."
            ),
        },
        "profiles": {
            "single": _profile(single),
            "terse": _profile(terse),
            "multitopic_c4": _profile(mt),
            "same_object": _profile(so_rows),
        },
        "verbatim_copy_rate": {
            "single": _verbatim_rate(single),
            "terse": _verbatim_rate(terse),
            "multitopic_c4": _verbatim_rate(mt),
            "same_object": _verbatim_rate(so_rows),
        },
        "integrity": {
            "train_text_overlap_with_validation_or_test": overlap,
            "duplicate_prompts_within_train": dup_train,
            "duplicate_prompts_within_same_object_stream": dup_so,
            "same_object_streams_all_two_topics": all(
                len(_topics_of(e)) == 2 for e in so_rows),
            "same_object_streams_all_shared_object": all(
                _topics_of(e)[0].get("object")
                and _topics_of(e)[0]["object"] == _topics_of(e)[1]["object"]
                for e in so_rows),
            "same_object_streams_all_distinct_domain": all(
                _topics_of(e)[0]["domain"] != _topics_of(e)[1]["domain"] for e in so_rows),
            "same_object_streams_all_use_inline_join": all(
                ", а також " in e["messages"][1]["content"] for e in so_rows),
            "same_object_sources_all_train_split": True,
        },
        "c1_c2_counters": {"single": c12, "multitopic": mt_c12},
        "dropped_empty_issue": {"n_single": dropped_empty, "n_multitopic": mt_dropped,
                                "n_same_object": so_dropped},
        "validation": "frozen copy of v2 validation.jsonl",
        "test": "frozen copy of v2 test.jsonl",
        "leakage": (
            "every augmentation source is a train-split record; any train record "
            "whose normalised content appears in the frozen validation or test "
            "splits is excluded before sampling, and the assembled corpus is "
            "re-checked against both splits. eval_v3 category E is composed from "
            "test-split records, so those are excluded by the same rule."
        ),
        "train_sha256": _sha256(train_path),
        "validation_sha256": _sha256(OUT / "validation.jsonl"),
        "test_sha256": _sha256(OUT / "test.jsonl"),
    }
    (OUT / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


if __name__ == "__main__":
    m = build()
    c = m["composition"]
    print(f"v3.2 treatment -> {OUT}")
    print(f"  total {c['total']}  single {c['single_topic_v1']}  terse {c['terse_augmented']}")
    print(f"  multitopic C4 {c['multitopic_c4_base']} (real {c['multitopic_real_rows']} / "
          f"synthetic {c['multitopic_synthetic_rows']})")
    print(f"  same-object rows {c['same_object_rows']}")
    print(f"  batches {c['n_batches']}  iters {c['iters_ceil']}  "
          f"iters==batches: {c['iters_equals_batches']}")
    print(f"  mixed-stream batches {m['file_order_layout']['mixed_stream_batches']}")
    print(f"  integrity {m['integrity']}")
    print(f"  train sha256 {m['train_sha256']}")