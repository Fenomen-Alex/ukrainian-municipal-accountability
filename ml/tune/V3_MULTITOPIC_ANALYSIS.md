# v3 corrected — multitopic regression root-cause analysis

**Arms compared:** `v3-treatment` (previous v3) vs `v3-corrected`
**Status:** root cause identified. **Corrected v3 is a failed release candidate.**
**Headline:** the regression is **topic merging**, not early stopping. On the 13 genuine
two-topic under-emissions, previous v3 splits **13/13**; corrected v3 splits **1/13** and
merges the two complaints into a single topic in **12/13**. The mechanism is traceable to the
R2 correction: it raised verbatim-copy supervision in the single-topic stream from **83.0% to
91.5%**, which is the same behaviour that makes the model absorb both problems into one topic.

Everything below is a read-only analysis of existing artefacts plus regenerated inference on
the existing adapters. **No retraining, no corpus edit, no evaluator change, no publication.**

---

## 1. Bottom line

Previous v3 emitted two topics where two complaints were present. Corrected v3 still *sees*
both complaints — it copies both of them into the output — but assigns them to **one** topic
instead of two.

`ev3-035` (category E, expected `roads` + `sanitation`), corrected v3 verbatim:

```json
{"topics": [{"domain": "sanitation", "issue": "прибрати суху деревину, яке звалило після
 буревію, біля корп. 1, по вул. Є, а також проведення ремонтних робіт дорожнього покриття
 по вул. Є", ...}]}
```

Both problems are present. The connective `а також` is preserved verbatim. They are simply
**not split**. This is the single most important finding in the report: nothing was dropped
from the model's perception, so this is not an under-reading failure and cannot be fixed by
supplying more multitopic data volume alone.

## 2. Measured impact

### 2.1 eval_v3, 52 expected-two-topic cases (regenerated raw, both arms)

| behaviour | v3-prev | v3-corrected |
|---|---|---|
| SPLIT (2 topics, correct count) | **34** | **23** |
| MERGE (1 topic containing both problems) | 5 | **17** |
| STOP (1 topic, second problem omitted) | 10 | 6 |
| invalid JSON | 3 | 6 |
| topic-count accuracy | 65.4% | **44.2%** |

Split rate fell **65% → 44%**. Merge count more than tripled, **5 → 17**. Stop count *fell*
(10 → 6), which is the direct evidence that this is a merge regression rather than a
truncation regression: if the model were losing track of the complaint or stopping early, the
STOP bucket would grow. It shrank. The content moved from the dropped bucket into the MERGE
bucket.

Per category:

| cat | n | v3-prev | v3-corrected |
|---|---|---|---|
| B | 8 | JSONFAIL 1, MERGE 1, SPLIT 4, STOP 2 | JSONFAIL 1, SPLIT 5, STOP 2 |
| C | 8 | SPLIT 6, STOP 2 | JSONFAIL 1, SPLIT 7 |
| D | 8 | JSONFAIL 1, MERGE 2, SPLIT 5 | JSONFAIL 1, MERGE 5, SPLIT 2 |
| E | 6 | **SPLIT 6** | **MERGE 5, SPLIT 1** |
| F | 8 | JSONFAIL 1, SPLIT 4, STOP 3 | JSONFAIL 1, MERGE 3, SPLIT 2, STOP 2 |
| G | 6 | MERGE 1, SPLIT 2, STOP 3 | JSONFAIL 1, MERGE 2, SPLIT 1, STOP 2 |
| K | 8 | MERGE 1, SPLIT 7 | JSONFAIL 1, MERGE 2, SPLIT 5 |

Category E is the cleanest signal in the whole analysis: previous v3 was **perfect (6/6)**,
corrected v3 collapses to **1/6**. A metric going 100% → 17% on a fixed 6-case slice is not
noise.

### 2.2 MT85 (85 real multitopic complaints)

| metric | v2 | v3-prev | v3-corrected |
|---|---|---|---|
| topic-count correct | 73/85 | **80/85** | 78/85 |
| domain_set_exact | 58/85 | **66/85** | 64/85 |
| json parse rate | 0.99 | 1.00 | 1.00 |

MT85 moves by only **−2** on both metrics, versus **−11** on eval_v3 expected-two-topic.
Per-case classification of MT85: **0 fixed**, **2 regressed**
(`MT-SYN-SS-Г-2973-Д-448`, `MT-SYN-Д-1680-Г-3180`), 5 wrong in both v3 arms, 71 correct in
all three arms.

**This asymmetry is diagnostic.** MT85 prompts are two *separate* complaints concatenated with
clear boundaries, which is exactly how MT training data is built. eval_v3 categories D/E/F/G
embed two problems inside *one* continuous sentence. The model handles the boundary-marked
shape and fails the inline shape. A pure multitopic-volume shortage would hurt both roughly
equally; it does not.

## 3. Root cause

### 3.1 The R2 correction is the mechanism

R2 stripped response-delivery boilerplate from target `issue` fields and moved the request
phrase into the issue, making targets **more verbatim copies of the complaint**. Measured
over unique UIDs, the fraction of target issues that are exact substrings of the complaint:

| stream | v3-prev | v3-corrected |
|---|---|---|
| single | 83.0% | **91.5%** |
| terse | 94.4% | 94.4% |
| mt_real | 79.5% | 79.5% |
| mt_synth | 50.1% | 53.2% |

The single-topic stream — **5,157 rows, the majority of the corpus** — moved +8.5 points
toward "copy the complaint span into `issue`". That is a strong, direct prior for
*one topic per complaint span*. Applied to an inline two-problem sentence, the same prior
produces exactly one topic whose `issue` covers both problems.

R2 was correct on its own terms. Boilerplate in `issue` was a real defect and removing it
improved single-topic quality (B rose 0.500 → 0.625, C rose 0.750 → 0.875). The defect is
that it was applied corpus-wide without adding a counterweight for the inline-multitopic case.

### 3.2 The counterweight was already weak, and R1 removed part of it

The single stream already encoded a **merge prior** before any correction: of 163 single-stream
rows whose prompt contains `а також`, **123 (75.5%)** have a single topic whose `issue` spans
**both** sides of the connective. Corrected has **125 (76.7%)** — essentially unchanged.

So R2 did not invent merging. It **amplified a pre-existing bias in the largest stream** past
the point where it overrides the MT stream's split demonstration. The MT stream does
demonstrate splitting across the same connective (1,022 split vs 120 merged examples), but MT
is only **25.6%** of rows and **41.1%** of target tokens — outvoted by a 5,157-row
single-topic stream that was just taught to copy harder.

### 3.3 Why category E specifically

Categories are cleanly separated by construction. E is the only category where the two problems
share the **same object**:

| cat | same object | connective |
|---|---|---|
| B, D, F, G | no | no |
| C, K | no | yes |
| **E** | **yes** | **yes** |

Conditioning on the expected output, same-object cases score **17%** versus **48%** for
different-object cases. When both problems share an object, "copy the span mentioning that
object" and "emit one topic" become the same action, and the verbatim-copy prior wins.

The MT stream supplies only weak same-object supervision to counteract this. Pooled over
MT-REAL and MT-SYN, same-object two-topic pairs are **386 of 2,418 MT rows (16.0%)**, heavily
concentrated in synthetic data (**19.2%** MT-SYN) and rare in real complaints (**4.5%**
MT-REAL). Same-domain two-topic pairs are **0.0%** in both. So the inline same-object shape
that dominates category E is barely represented in real supervision, and never represented
with a repeated domain.

### 3.4 Competing explanations, and why they fail

| hypothesis | verdict | evidence |
|---|---|---|
| Multitopic dilution (C3) | **rejected as the cause** | MT target-token share is 41.37% (prev) vs 41.10% (corrected) — a 0.27-point change cannot produce a 21-point split-rate drop. MT85 falls only 2. |
| Groundedness / hallucinated targets | **rejected** | MT topic-1 coverage 0.998 real / 0.980 synth; topic-2 0.993 / 0.981; topic order correct 95.5% real / 86.2% synth. Targets are well grounded in both arms. |
| Train/eval leakage or overlap | **rejected** | Zero MT85 base-UID and zero complaint-text overlap with training. |
| Sequence truncation | **rejected** | No sequence exceeds `max_seq_length=2048`. |
| Optimization/order variance | **contributing, not sufficient** | See §4. Order differs, but variance cannot produce a *directional* merge shift on the same-object slice. |
| **R2 verbatim-copy amplification** | **supported** | 83.0% → 91.5% in the dominant stream; merge 5 → 17; same-object cases 17% vs 48%; STOP bucket shrank while MERGE grew. |

## 4. Secondary finding: the previous "batch order" concern is real but secondary

An earlier pass concluded that length sorting was a no-op and that corrected introduced
stream-mixing batches. Both are correct and worth recording, but neither explains the merge
shift.

`CacheDataset.itemlen()` returns `len(self._data[idx])`, and `_data[idx]` is a raw
`{messages, uid}` dict, so **every** row has length 2. Length sorting is therefore a stable
no-op; batches are contiguous pairs in file order, permuted by seeded shuffle.

| corpus | rows | batches | mixed-stream batches | tail rows dropped |
|---|---|---|---|---|
| v2 | 8,519 | 4,259 | 0 | 0 |
| v3-prev | 9,448 | 4,724 | 0 | 0 |
| v3-corrected | 9,447 | 4,723 | **3** | **1** |

Corrected has an **odd** row count, which shifts stream starts to odd indices and produces
exactly three mixed batches at the single→terse, terse→mt_real and mt_real→mt_synth boundaries.
The final mt_synth row is never trained, and the 4,724th iteration wraps into a reshuffled
second epoch. Batch *sequence* overlap between the two arms is only **12.3%**, and **0%** of
MT batch pairs are shared.

This is a genuine reproducibility defect and should be fixed in v3.2. It is **not** the
multitopic root cause: it perturbs MT examples almost identically in both arms, and a
random perturbation would not concentrate failures on same-object cases.

## 5. C4 audit (data quality, both arms)

| property | v3-prev | v3-corrected |
|---|---|---|
| target issue verbatim substring of complaint (single) | 83.0% | 91.5% |
| merge prior in single stream (`а також` rows) | 75.5% | 76.7% |
| MT split vs merged across connective | 1022 / 120 | 1022 / 120 |
| same-object two-topic pairs (MT-REAL / MT-SYN) | 4.5% / 19.2% | identical |
| same-domain two-topic pairs | 0.0% | 0.0% |
| MT issue coverage topic1 / topic2 (real) | 0.998 / 0.993 | identical |
| empty `requested_action` (MT-REAL / MT-SYN) | 68.2% / 80.8% | identical |
| sequences over 2048 tokens | 0 | 0 |

## 6. v3.2 — one minimal controlled change

**Change:** add same-object two-topic multitopic supervision. Keep R1, R2, R3 as they are.
Everything else in the recipe is unchanged.

Rationale: the merge prior lives in the single stream and the inline two-topic shape is
under-represented (4.5% same-object in MT-REAL, 19.2% in MT-SYN, 0% same-domain). The model
has never been shown, in volume, that two problems sharing one object must yield two topics.
This targets the exact failure rather than adding generic multitopic volume, which the MT85
evidence shows is not the binding constraint.

**Concretely:**

- Generate MT examples where both problems share the same street/object, differing domain
  (`roads` + `sanitation`, `heating` + `housing`, `water` + `sanitation`, `commerce` +
  `construction`).
- Prompt form mirrors category E: single continuous sentence, `а також` connective, one shared
  object.
- Target keeps **two** topics with the shared object in `object` and distinct domains — the
  exact opposite of what the single stream teaches.
- Size to roughly double the same-object share of MT-REAL from **4.5% to ~10%**, which at the
  current 528 real rows means adding **~28** real same-object rows. Because single real rows
  are already replicated 24x, pair that with a synthetic same-object block of **~250–300**
  rows so the shape is not swamped by one source.
- Upsample by repetition to restore the previous MT target-token share of **41.4%** so the
  change is not confounded by a second dilution shift.

**Recipe, unchanged otherwise:** batch 2, grad accumulation 2, lr 1e-4, no schedule, seed 42,
`max_seq_length` 2048, `mask_prompt` true, checkpointing on, one epoch.

**Runtime estimate:** +250–400 rows is ~3–4% more data, so one epoch moves from 4,723 to
roughly **4,850–4,925** optimizer iterations. Previous full run was well under a day on this
machine; expect the same order of magnitude, plus roughly 15–20 minutes for fusion and about
40 minutes for the eval sweep across all four suites.

**Stop / success criteria — decide before running:**

- **Success:** eval_v3 expected-two-topic topic-count accuracy **≥ 60%** (vs 44.2% corrected,
  65.4% previous); category E split rate **≥ 5/6**; MT85 topic-count **≥ 80/85** (no
  regression against previous v3); merge count on eval_v3 **≤ 8** (vs 17).
- **Stop and revert:** MT85 topic-count below **78/85**, or any category below previous v3
  minus 10 points, or eval_v3 merge count above **15**.

**Confounds to record, not to fix:** keep the row count even so stream starts stay
pair-aligned, and log the realised batch sequence hash so order sensitivity is measurable next
time.

## 7. Reproduction

Analysis scripts, all read-only:

- `/tmp/mtinv/copyfaith.py` — verbatim-copy rates per stream per corpus
- `/tmp/mtinv/c4merge.py` — merge prior and MT split/merge counts
- `/tmp/mtinv/objshare.py` — same-object / same-domain pair rates
- `/tmp/mtinv/order2.py` — batching reconstruction
- `/tmp/mtinv/merge3.py`, `/tmp/mtinv/final_cmp.py` — merge vs stop classification
- `/tmp/mtinv/rawall_corrected.json`, `/tmp/mtinv/raw_prev_all52.json` — regenerated raw
  generations

Note: run inference with `.venv-mlx/bin/python`; `.venv` lacks `mlx_lm`. Avoid naming scratch
scripts `copy.py` — it shadows the stdlib `copy` module and breaks `transformers` imports.

## 8. Scope

Read-only analysis plus inference on existing adapters. No retraining, no corpus or evaluator
modification, no change to v2 or previous-v3 artefacts, no publication.