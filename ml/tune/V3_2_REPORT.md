# v3.2: the same-object intervention

**Status: FAILED EXPERIMENT — do not ship.** One epoch completed cleanly
(27.55 h, final loss 0.008, 4,854 iterations) and every suite was scored. The
model met 0 of 4 pre-registered success criteria and tripped 2 of 3 revert
triggers. Everything after `<!-- RESULTS -->` is written from measurements, not
expectations.

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

## 7. Result: FAILED EXPERIMENT — do not ship

v3.2 is **not** a release candidate. It met **0 of 4** pre-registered success
criteria and tripped **2 of 3** stop/revert triggers. It is strictly worse than
corrected-v3 on 8 of 12 eval_v3 metrics and on 7 categories' topic-count accuracy.

The intervention did move the thing it was designed to move — MERGE fell
17 → 12 and category E recovered 1/6 → 3/6 — but **only one third of that merge
reduction turned into a real split**. The rest decayed into STOP and, worse,
into a **new degenerate repetition-loop failure mode** that destroyed seven
cases every earlier arm had answered correctly.

### 7.1 The pre-registered rule, applied mechanically

From `V3_MULTITOPIC_ANALYSIS.md` §6, fixed before the run:

| rule | criterion | target | v3.2 observed | result |
|---|---|---|---|---|
| success | expected-two-topic topic-count | ≥ 60% | **38.5%** (20/52) | **FAIL** |
| success | category E split rate | ≥ 5/6 | **3/6** | **FAIL** |
| success | MT85 topic-count | ≥ 80/85 | **77/85** | **FAIL** |
| success | eval_v3 merge count | ≤ 8 | **12** | **FAIL** |
| stop → revert | MT85 topic-count | < 78/85 triggers revert | **77/85** | **TRIGGERED** |
| stop → revert | any category vs previous v3 | −10 pts triggers revert | **7 categories** (B C D E F G K) | **TRIGGERED** |
| stop → revert | eval_v3 merge count | > 15 triggers revert | 12 | not triggered |

Verdict: **0/4 success criteria, 2/3 revert triggers → FAILED EXPERIMENT.**

### 7.2 Behaviour on the 52 expected-two-topic cases

| arm | SPLIT | MERGE | STOP | JSONFAIL | expected-two accuracy | category E |
|---|---|---|---|---|---|---|
| prev-v3 | 34 | 5 | 10 | 3 | **65.4%** (34/52) | 6/6 |
| corrected-v3 | 23 | 17 | 6 | 6 | **44.2%** (23/52) | 1/6 |
| v3.2 | 20 | 12 | 8 | 12 | **38.5%** (20/52) | 3/6 |

v3.2 has the **lowest expected-two accuracy of the three v3 arms** and the
**highest JSON-failure count of any arm evaluated**. Accuracy fell while the
MERGE count fell too — that combination is the whole story of this experiment.

### 7.3 Where the merge reduction actually went

Of corrected-v3's 17 MERGEs, 9 left the bucket in v3.2:

| corrected-v3 MERGE → v3.2 | n | real progress? |
|---|---|---|
| → SPLIT (`ev3-034`, `ev3-035`, `ev3-039`) | 3 | **yes** — the intended effect |
| → STOP (`ev3-027`, `ev3-041`, `ev3-050`) | 3 | no — regression to under-emission |
| → JSONFAIL (`ev3-036`, `ev3-037`, `ev3-049`) | 3 | no — **new failure mode** |
| → still MERGE | 8 | no |

So **1 in 3** merge corrections produced a split. Meanwhile 4 cases corrected-v3
split correctly regressed *into* MERGE (`ev3-011`, `ev3-014`, `ev3-022`,
`ev3-082`).

Net SPLIT movement, corrected-v3 → v3.2: **16 kept, 4 gained, 7 lost → 20**.

Across all 52 cases, **19 changed: 4 better, 15 worse.**

### 7.4 eval_v3 (146 cases)

| metric | v2 | prev-v3 | corrected-v3 | v3.2 | Δ vs corrected |
|---|---|---|---|---|---|
| `json_parse_rate` | 0.9932 | 0.9795 | 0.9589 | **0.9178** | -0.0411 **regression** |
| `schema_validity_rate` | 0.9863 | 0.9795 | 0.9589 | **0.9110** | -0.0479 **regression** |
| `topic_count_accuracy` | 0.7808 | 0.8699 | 0.8014 | **0.7740** | -0.0274 **regression** |
| `domain_set_exact` | 0.6301 | 0.6712 | 0.7055 | **0.6644** | -0.0411 **regression** |
| `domain_set_precision` | 0.8356 | 0.7808 | 0.8836 | **0.8082** | -0.0754 **regression** |
| `domain_set_recall` | 0.7637 | 0.7397 | 0.8048 | **0.7500** | -0.0548 **regression** |
| `issue_rouge_l_best` | 0.6719 | 0.7135 | 0.7545 | **0.7265** | -0.0280 **regression** |
| `boilerplate_leak_rate` | 0.3699 | 0.2260 | 0.0068 | **0.0000** | -0.0068 ✓ |
| `action_redundancy_rate` | 0.2055 | 0.0890 | 0.1575 | **0.0753** | -0.0822 ✓ |
| `duplicate_domain_rate` | 0.0137 | 0.0068 | 0.0068 | **0.0000** | -0.0068 ✓ |
| `out_of_enum_rate` | 0.0068 | 0.0000 | 0.0000 | **0.0068** | +0.0068 **regression** |
| `mean_predicted_topics` | 1.2123 | 1.2192 | 1.1164 | **1.0616** | -0.0548 |

`mean_predicted_topics` is a diagnostic rather than a score, but it is the
clearest single indicator of the failure: the model's mean commitment to topics
fell **1.1164 → 1.0616**, i.e. it became *more* conservative exactly where the
intervention was supposed to make it split more.

### 7.5 Multi-topic subset of eval_v3 (n=44) and MT85 (85 cases)

| metric | v2 | prev-v3 | corrected-v3 | v3.2 | Δ vs corrected |
|---|---|---|---|---|---|
| mt44 `schema_valid` | 1.0000 | 0.9318 | 0.8864 | **0.7273** | -0.1591 **regression** |
| mt44 `topic_count_accuracy` | 0.4318 | 0.6136 | 0.4091 | **0.3409** | -0.0682 **regression** |
| mt44 `domain_set_exact` | 0.3409 | 0.3864 | 0.2727 | **0.2273** | -0.0454 **regression** |
| mt85 `topic_count_accuracy` | 0.8588 | 0.9412 | 0.9176 | **0.9059** | -0.0117 **regression** |
| mt85 `multi_topic_recall` | 0.8941 | 0.9412 | 0.9176 | **0.9059** | -0.0117 **regression** |
| mt85 `domain_set_exact` | 0.6824 | 0.7765 | 0.7529 | **0.7765** | +0.0236 ✓ |
| mt85 `issue_rouge_l_best` | 0.7988 | 0.7365 | 0.7258 | **0.7280** | +0.0022 ✓ |

The multi-topic **schema** collapse on eval_v3's own multi-topic slice,
0.8864 → **0.7273**, is the 7-case repetition-loop defect showing up as an
aggregate. MT85, a different suite, holds schema at 0.9882 and only loses one
topic-count case — which is why the damage is invisible unless both are read.

### 7.6 Frozen 329-case suite

| metric | v2 | prev-v3 | corrected-v3 | v3.2 | Δ vs corrected |
|---|---|---|---|---|---|
| `json_parse_rate` | 1.0000 | 1.0000 | 1.0000 | **1.0000** | +0.0000 |
| `schema_validity_rate` | 0.9970 | 0.9970 | 1.0000 | **1.0000** | +0.0000 |
| `domain_accuracy` | 0.8389 | 0.8602 | 0.8663 | **0.8693** | +0.0030 ✓ |
| `domain_macro_f1` | 0.7596 | 0.7556 | 0.6775 | **0.7209** | +0.0434 ✓ |
| `issue_rouge_l` | 0.9234 | 0.6611 | 0.6610 | **0.6412** | -0.0198 **regression** |
| `object_exact` | 0.9422 | 0.9696 | 0.9848 | **0.9483** | -0.0365 **regression** |
| `object_token_overlap` | 0.9574 | 0.9773 | 0.9848 | **0.9605** | -0.0243 **regression** |
| `hallucination_rate` | 0.0152 | 0.0456 | 0.0578 | **0.0213** | -0.0365 ✓ |
| `multi_topic_rate` | 0.0365 | 0.0122 | 0.0000 | **0.0030** | +0.0030 **regression** |
| `action_presence_pred` | 0.4985 | 0.1337 | 0.1702 | **0.1033** | -0.0669 **regression** |
| `action_presence_match` | 0.9909 | 0.5471 | 0.5775 | **0.5653** | -0.0122 **regression** |

This is the only suite where v3.2 is broadly competitive: schema stays perfect,
domain macro-F1 and hallucination both improve. But `object_exact` falls back
to roughly v2 levels and **action emission keeps shrinking** — 0.4985 → 0.1033
against a 0.5015 target. The model is not inventing actions (that half of the
goal holds) but it is also barely emitting any.

### 7.7 Categories breaching the −10-point stop trigger

| category | prev-v3 | corrected-v3 | v3.2 | v3.2 − prev-v3 |
|---|---|---|---|---|
| B | 0.5000 | 0.6250 | 0.2500 | -0.2500 |
| C | 0.7500 | 0.8750 | 0.6250 | -0.1250 |
| D | 0.6250 | 0.2500 | 0.2500 | -0.3750 |
| E | 1.0000 | 0.1667 | 0.5000 | -0.5000 |
| F | 0.5000 | 0.2500 | 0.2500 | -0.2500 |
| G | 0.3333 | 0.1667 | 0.1667 | -0.1666 |
| K | 0.8750 | 0.6250 | 0.6250 | -0.2500 |

All 15 single-topic categories (A, H–V) are unchanged at 1.0000, which confirms
the damage is confined to the multi-topic generation path and is not general
decay.

### 7.8 The 12 JSON failures, classified

The 12 `JSONFAIL` cases split cleanly into exactly the two mechanisms asked
about, and the split is diagnostic rather than cosmetic — brace balance
separates them:

**Five quote-related failures — pre-existing, not introduced here.**
`ev3-016`, `ev3-024`, `ev3-032`, `ev3-046`, `ev3-052` all emit the same defect:
an unescaped double-quote pair inside a JSON string value
(`На дзвінки працівники "" не відповідають`), which terminates the string early
and corrupts the document. All five are brace-balanced — the output is
well-formed except for the quoting — and **all five were already JSONFAIL in
corrected-v3**. v3.2 neither fixed nor introduced them. Notably `ev3-016`,
`ev3-024` and `ev3-052` were correct SPLITs in previous-v3, so the corrected-v3
corpus rewrite is what introduced this defect; it remains unaddressed.

**Seven repetition-loop failures — new regressions introduced by v3.2.**

| case | prev-v3 | corrected-v3 | v3.2 | chars |
|---|---|---|---|---|
| `ev3-017` | SPLIT | STOP | JSONFAIL | 1645 |
| `ev3-025` | SPLIT | **SPLIT** | JSONFAIL | 1645 |
| `ev3-028` | SPLIT | **SPLIT** | JSONFAIL | 1221 |
| `ev3-036` | SPLIT | MERGE | JSONFAIL | 1566 |
| `ev3-037` | SPLIT | MERGE | JSONFAIL | 1637 |
| `ev3-047` | STOP | STOP | JSONFAIL | 1482 |
| `ev3-049` | STOP | MERGE | JSONFAIL | 1824 |

Every one of these is **brace-unbalanced** (`{` = 2 or 4 against `}` = 0 or 2),
i.e. generation ran to the 800-token budget mid-string and never closed the
object. The repeated unit is a clause copied from the input, e.g.
`…Коваленка, 15 в’їзд до будинку зі сторони вул. Коваленка, 15 в’їзд до
будинку зі сторони…` (`ev3-017`) and `…їх не відповідальна, але відповідальна за
дорогу, їх не…` (`ev3-036`).

Two of them — `ev3-025` and `ev3-028` — were correct SPLITs in corrected-v3, so
this is a genuine new capability loss, not a reshuffle.

### 7.9 Category E, case by case

| case | prev-v3 | corrected-v3 | v3.2 | reading |
|---|---|---|---|---|
| `ev3-035` | SPLIT | MERGE | **SPLIT** | recovered, correct domains |
| `ev3-036` | SPLIT | MERGE | JSONFAIL | recovery destroyed by repetition loop |
| `ev3-037` | SPLIT | MERGE | JSONFAIL | recovery destroyed by repetition loop |
| `ev3-038` | SPLIT | MERGE | MERGE | genuine remaining merge |
| `ev3-039` | SPLIT | MERGE | **SPLIT** | recovered, correct domains |
| `ev3-040` | SPLIT | SPLIT | **SPLIT** | preserved |
| | 6/6 | 1/6 | **3/6** | target ≥ 5/6 → FAIL |

The stream taught the split for two of the six. Had the repetition-loop defect
not eaten the other two, this would have been 5/6 and would have passed. The
category-E failure is a **partial success ruined by a bug elsewhere**, which is a
different diagnosis from "the intervention does not work".

### 7.10 Terse / boilerplate / schema regression guard

| stream | corrected-v3 | v3.2 | Δ | verdict |
|---|---|---|---|---|
| boilerplate leak (eval_v3) | 0.0068 | 0.0000 | -0.0068 | ✓ improved |
| schema validity (eval_v3) | 0.9589 | 0.9110 | -0.0479 | **regression** |
| json parse (eval_v3) | 0.9589 | 0.9178 | -0.0411 | **regression** |
| cat A terse_single topic_count | 1.0000 | 1.0000 | +0.0000 | ✓ held |
| cat A terse_single domain_exact | 1.0000 | 0.8000 | -0.2000 | **regression** |
| cat B terse_two_topic topic_count | 0.6250 | 0.2500 | -0.3750 | **regression** |
| smoke Multi-Topic | 0/4 | 2/4 | +2 | ✓ improved |

The "no material terse, boilerplate or schema regression" condition is
**violated**: schema, json parse, terse single-domain exactness and terse
two-topic topic-count all regressed. Boilerplate improved further and the smoke
multi-topic count doubled, but neither offsets a 15.9-point schema collapse on
the multi-topic slice.

### 7.11 Why it failed

The diagnosis in `V3_MULTITOPIC_ANALYSIS.md` was right that corrected-v3 was
**merging**, and the stream did attack it: MERGE 17 → 12, category E 1/6 → 3/6,
smoke multi-topic 0/4 → 2/4. The hypothesis was that same-object pairs were
under-represented. That part is supported.

The failure is in **how** the split was taught. The 261 new rows are rigidly
templated — `По X: A, а також B.` — and the existing streams already teach that
the `issue` field is a near-verbatim copy of the input. 261 rows of a fixed
two-topic frame over 153 streets is a strong, narrow template signal, and
`mean_predicted_topics` collapsing to 1.0616 plus 7 new verbatim-copy loops is
the signature of a model that has learned *the surface form of a two-topic
answer* rather than *the decision to emit two topics*. Where the template does
not cleanly apply, the copy behaviour has no stopping criterion and runs to the
token budget.

This is a **behavioural** failure, not a capability one: the frozen 329 suite,
which contains almost no same-object pairs, is the best v3.2 performs anywhere
(schema 1.0, hallucination down to 0.0213). The damage is entirely concentrated
in the multi-topic path the stream targeted — which is why 261 rows moved the
target behaviour and simultaneously broke the adjacent behaviour.

### 7.12 Release decision

**FAILED EXPERIMENT. v3.2 must not be shipped, and no serving or release
material should be prepared from it.**

Corrected-v3 remains the best v3 arm. Previous-v3 remains the best v3 arm on
expected-two-topic accuracy (65.4%) and category E (6/6) and is the only one
that has never introduced a new failure mode.

Retained from this experiment, all of it independently of the model:

* the finding that same-object pair density is the actionable lever (§7.11);
* proof that the 5 quote-related JSON failures are a **corrected-v3 defect**
  that must be fixed in the corpus, not in post-processing;
* the identification of repetition-loop truncation as a new failure mode worth
  a dedicated detector;
* the corpus, its 22 integrity gates, provenance and exposure tooling, which
  are model-independent and reusable.

Not retried here, and deliberately so: the recipe was fixed in advance and the
stop/revert rule fired twice. Re-running with a different stream size or a
different ratio would be a new experiment and needs its own pre-registered
criteria.

### 7.13 Artefact integrity

| artefact | sha256 |
|---|---|
| v3.2 corpus | `c304c374af6aa9b9da06a3c47bf47881cd27249ea9b389abd2e1ecdf43ac622e` |
| final adapter | `fbc8dbc089cc4d0ae91dbb93b912220e5b6bb1f25412fa4681ef31f4d155c2d5` |
| iteration-4500 checkpoint | `17f15245d719d37ab8dedf600c5c6276e0a5e47bcbb39769c46bc02f404eae63` |
| fused `model.safetensors` | `c6745eed54b242515d961574bb16892b3f5842940fc46995c48e4e3078fc795b` |

Fusion: 112 modules, `config_matches_v2 = true`; `config.json`,
`tokenizer_config.json`, `tokenizer.json` and `chat_template.jinja` are all
**byte-identical** to the canonical v2 fused arm, and all four fused arms hold
112 modules. All four `model.safetensors` hashes are distinct, confirming four
genuinely different models. Every pre-existing reference arm verified unchanged
after the full sweep.

## Appendix: reproduction

```bash
.venv/bin/python -m ml.tune.build_v3_2          # corpus + meta + provenance
.venv-mlx/bin/python -m ml.tune.exposure_v3_2   # target-token shares
.venv/bin/python -m pytest ml/tests -q           # 352 passed
.venv/bin/python -m ml.tune.artifact_sha --record
bash ml/tune/run_v3_2.sh                         # training, 27.55 h actual
bash ml/tune/run_v3_2_eval.sh                    # fuse + all suites + verify
.venv/bin/python -m ml.tune.gates_v3 --arm v3-2  # legacy v3 gate table
.venv/bin/python -m ml.tune.artifact_sha --verify
```

Wall-clock for the run this report describes: training 2026-10-01 10:05:41 →
2026-10-02 13:38:41; the fusion-and-evaluate sweep 13:49:48 → 15:22:44.

Every number in §7 is regenerated from the result files by
`.venv/bin/python -m ml.tune.verify_v3_2_report`, which re-reads this
document and compares each figure against the artefact it claims to summarise —
**197 claims**, all passing. It parses the report rather than holding its own
copy of the expected values, because a second list of numbers would only be a
second thing to get wrong. `ml/tests/test_verify_v3_2_report.py` corrupts a
claim seven different ways and requires each to fail, so the verifier cannot
quietly stop checking.

The `prev-v3` and `corrected-v3` behaviour columns come from the `-recheck`
files, which reuse the raw generations captured during the diagnosis and re-run
the identical classifier that `behavior_v3.py` validated against both arms.

**Known limitation of the verification.** `ml/data/tune/eval/` is gitignored
(repo policy predating this experiment), so the 55 frozen-329 claims in §7.6
cannot be checked from a fresh clone — 142 of 197 are fully covered by tracked
artifacts, and the rest require `bash ml/tune/run_v3_2_eval.sh` to have been run
locally. The verifier fails with an explicit "missing artefact" message rather
than silently skipping.