# v3.2: the same-object intervention

**Status: training in progress.** The corpus, its gates and its provenance are
complete and committed. Model results are filled in below once the run finishes;
everything after this marker is written from measurements, not expectations.

Companion documents:

* `ml/tune/V3_MULTITOPIC_ANALYSIS.md` — the diagnosis this experiment acts on.
* `ml/tune/build_v3_2.py` — the builder. `ml/tests/test_v3_2.py` — 22 gates.
* `ml/tune/behavior_v3.py` — the SPLIT/MERGE/STOP classifier.
* `ml/tune/artifact_sha.py` — SHA256 pinning of every other arm.

---

## 1. What changed, in one line

One stream was appended to the corrected v3 corpus: 261 two-topic rows in which
**both problems concern the same object and appear inline in one sentence.**
Nothing else was touched.

```
ml/data/tune/v3/treatment_v3_2/train.jsonl
sha256  c304c374af6aa9b9da06a3c47bf47881cd27249ea9b389abd2e1ecdf43ac622e
9708 rows = 5157 single + 1872 terse + 2418 multitopic (C4) + 261 same-object
```

The first 9,447 rows are **byte-identical to corrected v3, in the same order**
(`test_v32_extends_corrected_v3_by_exactly_the_new_stream`). That is the
property that makes this a delta rather than a new corpus.

## 2. Why this stream and not another

The corrected-v3 multitopic regression is **merging, not truncation**. On the 13
genuine two-topic under-emissions, previous v3 split 13/13 while corrected v3
merged 12/13, and the STOP bucket *shrank* 10 → 6. Both problems are copied into
the output; they are simply filed as one topic. eval_v3 category E — the only
same-object slice — collapsed 6/6 → 1/6, and the recheck in
`ml/data/tune/eval_v3/behaviour/v3-corrected-recheck.json` shows exactly how: of
those 6, **five are MERGE and one is SPLIT**. The model is reading both problems
and then not separating them.

The corpus already had a same-object family, `MT-SYN-SS-*` (same street), 145
rows of which 103 share an object. It does not fix this, and the reason is
specific:

```
"<complete complaint A>. а також <complete complaint B>."
```

Each half is its own sentence with its own request frame, so the boundary is
**explicitly marked**. MT85 is built the same way, and MT85 barely moved (80 → 78)
while eval_v3 collapsed (44 → 23). That asymmetry is the whole diagnosis: the
model splits boundary-marked complaints and merges inline ones.

Category E has no marker. It is one continuous sentence with a shared frame:

```
По вул. X: <clause A>, а також <clause B>.
```

No training example had that shape. That is the gap this stream fills.

## 3. Corpus

### 3.1 The new stream

| | |
|---|---|
| uid prefix | `MT-SYN-SO-` (disjoint from `MT-SYN-SS-`) |
| rows | 261, upsample 1 |
| shape | `По <street>: <clause A>, а також <clause B>.` |
| topics | exactly 2, identical `object`, **different** `domain` |
| up to | 3 rows per street, across 153 streets |
| sources | `ml/data/train.jsonl` only |

Example:

> По вул. Олефіренка: з'ясувати та усунути причину відсутності теплопостачання
> в квартирі №4 за адресою вул.Олефіренка, 5/9, а також заборонити незаконне
> будівництво та знищення дерев у дворі житлових будинків по вул.Олефіренка, 18/21.

*(topics: heating / construction, both `object = "вул. Олефіренка"`)*

### 3.2 Rejected candidates, and why

Sampler counters over the 730 pairs it examined:

| rejected | n | reason |
|---|---|---|
| same domain | 303 | two topics merely because there are two verbs |
| duplicate pair | 127 | would over-weight one street |
| not independent | 38 | synonymous issues, duplicated action, or a clause under 30 chars |
| street already at cap | 16 | diversity |
| issue emptied by C1 | 1 | nothing left to learn from; dropped as elsewhere |
| leakage | 0 | — |

Domain pairs span 32 distinct combinations; the largest are roads+sanitation
(54), sanitation+water (44), housing+sanitation (24). This tracks the eval-E
distribution, which is dominated by sanitation+roads.

### 3.3 Target-token exposure

`ml/tune/exposure_v3_2.py`, target = assistant message only (prompt is masked).

| stream | previous v3 | corrected v3 | **v3.2** |
|---|---|---|---|
| single | 47.66% | 47.90% | 45.54% |
| terse | 10.75% | 10.78% | 10.25% |
| mt_real | 8.24% | 8.23% | 7.82% |
| mt_synth | 33.35% | 33.09% | 31.46% |
| **same_object** | — | — | **4.93%** |
| **multitopic total** | 41.59% | 41.32% | **44.21%** |
| total target tokens | 1,359,057 | 1,361,123 | 1,431,735 |

Multitopic share rises to 44.21%. Single-topic share dilutes 47.90% → 45.54%,
a −2.36pp shift that is comparable in magnitude to corrected v3's own dilution
(previous → corrected moved single by the same order), which the analysis showed
is **not sufficient on its own** to cause merging.

### 3.4 Batch parity

Corrected v3 had 9,447 rows — odd. `iters = ceil(9447/2) = 4724` but
`batches = floor(9447/2) = 4723`, so the last row was dropped and the wrap
replayed a reshuffled second epoch. **v3.2's total is even**, by sizing the new
stream at an odd 261:

```
n_batches = 4854     iters = ceil(9708/2) = 4854     iters == batches  ✓
```

Every row is seen exactly once, with no dropped tail and no second epoch. Stream
alignment is *not* fixed — 4 mixed-stream batches remain, because the single
stream's 5,157 rows start the terse stream at an odd offset. Padding it would
mean editing a C1–C4 stream, which is out of scope for a single-change
experiment. This is reported, not hidden.

## 4. What was deliberately not changed

C1, R1, R2, R3, C3, the existing C4 streams, the single-topic stream, the
single-topic verbatim-copy rate, the training recipe, the evaluator, and serving.
`ml/tune/v3_changes.py` is imported, never restated, so the recipe cannot drift
with the corpus.

Capping the verbatim-copy rate deserves a note, because it is the *mechanism*
behind the single-topic behaviour (83.0% → 91.5%) and it would be the obvious
thing to "fix". It was rejected: doing both at once makes a null result
uninterpretable, because you cannot tell which change did nothing.

`test_single_topic_verbatim_rate_is_untouched` recomputes the rate from both
corpora rather than trusting two metadata files to agree.

## 5. Leakage

Every source is a train-split record. Any train record whose normalised content
appears in the frozen validation or test splits is excluded **before** sampling,
because eval_v3 category E is itself composed from *test*-split records — the
stream would otherwise be training on its own test set.

Post-assembly: `train_text_overlap_with_validation_or_test = 0`,
`duplicate_prompts_within_same_object_stream = 0`, and the duplicate-prompt count
of the whole corpus is **identical to corrected v3's** (3,322), so the new stream
adds no duplicated gradient.

## 6. Known limitations

Stated up front, because they bound what the result can mean.

1. **`requested_action` coverage is low.** Only 34.9% of new rows carry any
   action and 2.7% carry one on both topics, against 100% / 16.7% for the eval-E
   reference. This is inherited C2 behaviour, not new — the existing multitopic
   rows behave the same way — but it means the new stream teaches the *split*,
   not the *action fields*, for same-object pairs. The distinct-action gate in
   `_pair_is_independent` is consequently near-inert on this corpus.
2. **Dominant pattern is both-actions-empty** (65%), which is arguably
   desirable: it teaches the model not to invent actions. But it is a weaker
   signal than the eval set expects.
3. **261 rows over 153 streets** is a small sample relative to the corpus; the
   intervention is a nudge, not a re-weighting.
4. **Street-extraction noise** (a truncated `вул` stem) appears in 2.3% of new
   rows, 0% in eval E. It comes from the eval builder's own `_street` regex, so
   it is faithful to how the eval set was built.

<!-- RESULTS -->

## Appendix: reproduction

```bash
.venv/bin/python -m ml.tune.build_v3_2          # corpus + meta + provenance
.venv-mlx/bin/python -m ml.tune.exposure_v3_2   # target-token shares
.venv/bin/python -m pytest ml/tests -q           # 352 passed
.venv/bin/python -m ml.tune.artifact_sha --record
bash ml/tune/run_v3_2.sh                         # training, ~22 h
bash ml/tune/run_v3_2_eval.sh                    # fuse + all suites + verify
.venv/bin/python -m ml.tune.artifact_sha --verify
```