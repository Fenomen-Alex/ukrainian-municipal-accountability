# V3.1 analysis: what to fix before the next full training run

**Scope.** This is the diagnosis behind the v3 → v3.1 data change, written after
the R1/R2/R3 correction (commit `e8f336c`). It answers one question per section
(A–H) with measured evidence, then states one exact recipe.

**Two corpora, do not confuse them.**

| name | file | rows | status |
| --- | --- | --- | --- |
| **v3** (what was trained) | `ml/data/tune/v3/treatment/train.jsonl` @ `fbfb281` | 9 448 | the corpus behind every v3 number below |
| **v3.1** (this document) | `ml/data/tune/v3/treatment/train.jsonl` @ `e8f336c` | 9 447 | corrected R1/R2/R3 |

The model scores quoted here (`v2`, `v3-treatment`, `v3-control`) were measured on
the **v3** corpus; they are the diagnostic baseline the correction is aimed at.
No model has been trained on the v3.1 corpus, so v3.1 has **no scores** — only
data measurements. Where a number is inferred rather than measured it is marked
*expected*.

**v3.1 corpus identity** (`sha256`): `train.jsonl`
`22e85d80b7a7abc12d0629ffc41fd125a2b718125e01f62fe9cbcd92815ef9f2`, `meta.json`
`5db70ce863a966ac6151e75b106874b5438e9fdac7734ef81d8d081f0fc026f7`.

**Sources.** `eval_v3/results/{v2,v3-treatment,v3-control}.json` (metrics,
by_category, subsets, per_case); `ml/tune/V3_REPORT.md` §5–6 (v3 run);
`ml/reports/v3_experiment_design.md` §5 (gates) and §10 (as-built);
`eval_v3/{stats.json,README.md}` (suite); `v3/treatment/meta.json` (composition);
`build_v3.py`, `v3_changes.py`, `build_eval_v3.py` (code).

---

## A. Terse regression — a data bug, now fixed

**Observation (measured).** On eval_v3 category A (10 short single-topic cases),
`domain_set_exact` for v3-treatment is **0.00 (0/10)** against v2's **0.80
(8/10)**. Every one of the ten predicted `other` while getting the topic count
right (`topic_count_accuracy` = 1.0):

| case | expected | v3 | v2 |
| --- | --- | --- | --- |
| ev3-001…005, 010 | sanitation | **other** | sanitation (001: roads) |
| ev3-006 | construction | **other** | sanitation |
| ev3-007 | housing | **other** | housing |
| ev3-008/009 | roads | **other** | roads |

The report calls this "the biggest regression" and attributes 8 of the 16 new
domain misses to it (`V3_REPORT.md` §6). It is the sole reason the advisory
`terse_single` gate (≥ 0.70) reads 0.00.

**Root cause (exact).** `build_v3.py::_terse_examples` hard-coded the domain:

```python
domain = "other"          # pre-fix, line 174
```

The terse slice is C3's augmentation — **1 872 rows = 234 units × 8**, i.e.
**19.8 % of the whole corpus** (`meta.json`). Every one of those rows carried
`"domain": "other"` in its target. The model did not lose the domain vocabulary
by accident: one fifth of its training taught it to emit `other` for exactly the
input shape category A probes.

**Fix (R1, applied).**

```python
domain = KIND_TO_DOMAIN.get(r.get("kind") or "", "other")
```

The corrected terse slice carries **10 real domains, zero `other`** (row counts,
each ×8):

| domain | rows | | domain | rows |
| --- | --- | --- | --- | --- |
| sanitation | 560 | | heating | 56 |
| housing | 384 | | construction | 24 |
| water | 320 | | payments | 16 |
| roads | 312 | | government | 8 |
| transport | 112 | | `other` | **0** |
| electricity | 80 | | | |

No other property of C3 changed (pool 246 records → 234 units → 1 872 rows). This
is the one blocking-adjacent regression with a single-line, testable cause.

---

## B. C3 quality — repaired by R1, but the ×8 multiplier still amplifies junk

**What C3 is.** The terse slice is one chat per short record, the visible text
being the record's *kernel* (complaint with boilerplate/wrapper deleted), domain
now from R1. 234 units, ×8 = 1 872 rows, 19.8 % of the corpus.

**Measured kernel properties (v3.1).**

- length: mean **65.1**, median **68**, min **1**, max **135** chars;
- action: **32 / 1 872** rows (1.7 %) have a non-empty `requested_action`, and
  **all 32 are whole-sentence** (`meta.json` `slices.terse`). The terse regime
  therefore teaches an empty action almost always — which is the right prior for
  category H and for short notes that state a problem but not a request.

**Defect B1 — the ≥25-char filter is applied pre-C1, then C1 strips it further
(measured).** `meta.json` records the pool criterion as *"train record < 150 chars
and note kernel ≥ 25 chars"*. But the built `issue` is `clean_issue_c1(kernel)`,
and C1 can shorten a 25-char kernel to near nothing. Result: **144 terse rows
(18 units × 8) have a post-C1 kernel under 25 characters**, and **4 units are
under 10**. Examples:

| kernel | source note |
| --- | --- |
| `О` (1 char) | `О. роз’яснення щодо пільг у проїзді у громадському транспорті ВПО` |
| `забере особисто` | `роз’яснення чи законно буде проводитись підсипання щебнем …` |
| `03.2023 року` | `роз’яснення чому в ДНЗ №19 відсутнє опалення з 27.03.2023 року` |
| `Ушакова` | `та заасфальтувати яму біля магазину ’ по вул.Ушакова` |

These are degenerate targets — an address fragment or a closing phrase — repeated
8×. They are not the cause of the category-A collapse (R1 is), but they are the
same class of contamination and are cheap to remove (see §H, variant C3′).

**Defect B2 — the terse pool is a fixed 234 units.** The pool is small relative
to the 8× weight it carries, so the same 234 kernels account for a fifth of the
corpus. The model memorises them (train loss falls to 0.015, `V3_REPORT.md` §3),
and it still generalised on multitopic, so this is a risk note, not a proven
failure. If a re-run keeps ×8 it should state that C3 is a *format/domain prior*,
not new data.

**Verdict for §H:** C3 is retained. Before R1 it was actively harmful; after R1
it is the only mechanism that trains the short-input regime, which category A
measures and which no other slice covers.

---

## C. C1 / boilerplate — phone-mode strand was the real residue

**Measured on the corpora** (topic-level, pattern match; rows: v3 9 447 / 11 865
topics, v2 8 519 / 11 654 topics):

| residue pattern | v2 | v3.1 |
| --- | --- | --- |
| consent (`згод… на оброб…`) | **3 991 (34.25 %)** | **0** |
| phone-mode (`[ву] … телефонному/письмовому режимі`) | **1 306 (11.21 %)** | **6 (0.05 %)** |
| generic closer (`вжити заход…`) | 783 (6.72 %) | 325 (2.74 %) |
| docket/`№ … від` (admin) | 54 (0.46 %) | 42 (0.35 %) |
| `просить надати` | 84 (0.72 %) | 55 (0.46 %) |
| response stub (`відповідь заяв…/надати`) | 13 (0.11 %) | **0** |

**Two C1 defects were left by the v3 corpus, both now fixed:**

1. **Euphony.** The drop rule was `в`-only, but the corpus writes `у телефонному
   режимі` **127** times against `в телефонному режимі` **584** (and 3 vs 6 for
   `письмов-`). A `в`-only rule silently left every `у` record behind. All R2
   patterns are now `[ву]\s+у?\s*` (shared `_R2_MODE`).
2. **Intervening subject.** The dominant surviving shape was `Відповідь заявниця
   бажає отримати у телефонному режимі` (**111** raw records): the subject sits
   between `Відповідь` and the verb, so the response-delivery rule never fired
   and clause surgery stranded a bare `у телефонному режимі`. R2 now matches by
   sentence *shape* (anchored `^відповідь … _R2_MODE`, subject-first
   `заявниця бажає отримати …`), not by adjacency.

**The 6 surviving phone-mode topics are the meaningful instrumentals** (e.g.
`некоректно спілкувалася з заявником, в телефонному режимі`). They use the
instrumental (`з заявником` / `із заявницею`) and have no delivery verb, so the
rules leave them; a test pins this. That the residual is exactly the substance
documents in `v3_experiment_design.md` §C1 ("`в телефонному режимі` is boilerplate
inside the delivery phrase but is the substance of a complaint about how staff
spoke to the citizen") is the intended behaviour.

**The 325 `вжити заход…` hits are R3 rescues, not residue.** R3 end-anchors the
generic closer so `Прохання вжити заходів та усунути причину витоку води` is
kept whole; the hits are concrete-object closers the previous rule deleted.

**The 42 admin / 55 `просить надати` hits are mostly real content.** Inspection
shows citizens citing their own prior docket (`№ 6180 від 17.08.2022`) or an
embedded request inside a longer issue. They are not removable boilerplate; the
pattern flags real text. Treat 0.35–0.46 % as the floor of this probe, not a
target to drive to zero.

**Eval alignment (measured).** The eval_v3 reference labeler does **not** share
the euphony bug: **0 of 198** reference topics contain a phone-mode phrase,
because `BOILER_SENT` also has the sentence-level `відповідь заявник\w*`, which
catches the delivery sentence regardless of `в`/`у`. Training and reference now
agree, so **no eval change is needed** for R2.

**Caveat on the gate.** The v3 model's `boilerplate_leak_rate` was **0.226**
against the ≤ 0.02 gate. Corpus residue was 11.2 % (now 0.05 %), so C1's share of
that leak is now small; the rest is the model *over-generating* boilerplate on
inputs it did not clean internally. R2 will help, but C1 alone does not
guarantee the gate — see §H.

---

## D. Multitopic — clean per-topic, four integrity drops

**Slice (measured).** 2 418 rows, 4 836 topics; **real_row_share 0.2184** (528
real of 2 418). C1 + C2 run **per topic** via `_relabel_multitopic`, cleaning each
topic from its own `issue` clause (not from the merged user text, which would
collapse every topic to the same string). `meta.json`
`multitopic_c1_c2_counters.boilerplate_issue` = 1 861 of 4 836 topics touched.

**Corrected behaviour.** The same C1 level applies to the multitopic slice, so
its residual consent/phone-mode count is now **0 / 0** (one scan over the whole
corpus finds only the 6 meaningful instrumentals, all in the single slice).

**Four chats dropped (v3 → v3.1), on integrity grounds.** `build_v3.py:279-283`
drops a whole multitopic chat when any topic empties, because dropping one topic
"would silently turn a two-topic example into a one-topic one and corrupt the
multitopic suite". R2 made one topic delivery-only in four chats
(`MT-SYN-SC-К-2090-М-699#0/#1`, `MT-SYN-М-1848-С-2879#0/#1`), each showing a
pure-delivery topic (`в телефонному режимі` / `надати заявниці в телефонному
режимі`) alongside real content. `dropped_empty_issue.n_multitopic` is **12**
(v3.1) vs 10 (v3).

**What the v3 run measured** (v1 weak labels, the multitopic 85 suite —
`V3_REPORT.md` §5.2): `multi_topic_recall` 0.8941 → **0.9412**,
`topic_count_accuracy` 0.8588 → **0.9412**, `domain_set_exact` 0.6824 →
**0.7765**, `schema_validity_rate` 0.9882 → **1.0**. Smoke two-topic stayed
**1 / 4**.

**Remaining multitopic error mode (§6):** single-topic domain substitutions
(`heating→housing`, `water→roads`, `construction→roads`, a dropped `government`
tail) — recall is high, exact-set is not. Neither R2 nor R3 addresses this;
it is an ontology boundary issue, not boilerplate.

---

## E. Unparseable outputs — three hard JSON failures, not addressed by R1/R2/R3

**Observation (measured).** v3-treatment `json_parse_rate` **0.9795**,
`schema_validity_rate` **0.9795** (v2: 0.9932 / 0.9863). Three cases fail,
deterministic at temperature 0:

| case | category | expected topics | predicted | text head |
| --- | --- | --- | --- | --- |
| ev3-013 | B `terse_two_topic_i` | 2 | **0** | `збільшення кількості посадкових місць … і направити спеціалістів …` |
| ev3-032 | D `two_sentences` | 2 | **0** | `надання роз’яснення причини … боргу за вивіз сміття. … вирівняти грейдером …` |
| ev3-046 | F `different_objects_two_problems` | 2 | **0** | same payments+roads pair as ev3-032 |

All three are `composed` two-topic cases and the model emits an unparseable empty
array instead of a topic list (`V3_REPORT.md` §6). v2 handled all three.

**Root cause (inferred, from the inputs).** The three texts are merged
complaints with no sentence-separating device the model recognises as a topic
boundary, and **all three open lower-case with a nominalised verb** (`збільшення
…`, `надання …`) rather than a subject or imperative — unlike every training row,
which starts with a `Заявник`/`Щодо`/imperative frame. The model appears to fall
into a degenerate continuation rather than emitting JSON. This is a
**format/robustness** failure on adversarial two-topic inputs, not a
boilerplate or domain failure.

**v3.1 impact: none.** R1/R2/R3 change content, not the output contract or the
input distribution of these three cases. The correction will not fix them and
must not be credited with fixing them. The schema gate stays open.

---

## F. Empty-action failure — one reference imprecision, not an invention

**Observation (measured).** v3-treatment category H: **1 of 12** cases predicted a
non-empty action (`share_nonempty_predicted_action` 0.0833); the advisory
`empty_action` gate wants 0. The case is **ev3-062** (provenance `real`):

- reference action: `''`
- predicted: `вжити заходів для перенесення туалету в інше місце (через сильний
  сморід стояти неможливо) та організувати місце для скидання сміття за межами
  кладовища`

**Analysis.** The reference `''` is a labeler artifact, not a model invention.
The case text genuinely contains a request — `Просить вжити заходів для
перенесення туалету …` — and category H is defined as *"no explicit action"*. The
reference labeler did not extract it, but the corrected R3 clause rule (which
fires on clause-final `вжити заход…` and rescues concrete-object closers) would
extract exactly that shape. So the model produced the *better* label and was
penalised for it. This is the same class of artifact as the frozen-329 ROUGE
drop: the reference is stale relative to the intended contract.

**v3.1 impact.** R3 raises `nonempty_action` on request-bearing rows (12 rescued
rows carry a concrete-object closer), which trains the model to extract such
requests — i.e. it makes ev3-062-like outputs *more* likely, not less. That is
correct contract behaviour. The reference is frozen, the case is kept, and
**v3.1 does not attempt to suppress it**. If the gate must read 0, that is an
eval-label decision (out of scope here), not a data change.

**Secondary (measured).** Terse rows contribute **0** non-empty actions except
the 32 whole-sentence ones (1.7 %), all of which are the kernel itself stated as
a request — negligible and not an H risk.

---

## G. Overall distribution shift — shorter, cleaner, and rare-domain-poor

**Corpus-level (measured).**

| | v2 | v3.1 | change |
| --- | --- | --- | --- |
| rows | 8 519 | **9 447** | +10.9 % |
| topics | 11 654 | 11 865 | +1.8 % |
| issue length mean | 190.2 | **127.6** | **−32.9 %** |
| issue length median | 185 | **105** | −43 % |
| p90 / max | 328 / 400 | 239 / 400 | — |
| topic-count 1 / 2 | 5 384 / 3 135 | 7 029 / 2 418 | multitopic share 36.8 % → 25.6 % |

The length collapse is the C1 signature: consent (34 %) and phone-mode (11 %)
text removed. The model now trains on the *complaint*, not the complaint plus
its administrative tail.

**Composition (v3.1):** single 5 157 (54.6 %), terse 1 872 (19.8 %), multitopic
2 418 (25.6 %); action non-empty by slice: single 19.3 %, terse 1.7 %, multitopic
22.0 %.

**Domain balance vs the eval suite (measured).** Training topic shares against
eval_v3's 198 expected topics:

| domain | train share | eval share | eval/train |
| --- | --- | --- | --- |
| sanitation | 30.0 % | 32.3 % | 1.08 |
| water | 15.0 % | 8.1 % | 0.54 |
| roads | 16.8 % | 12.1 % | 0.72 |
| housing | 12.1 % | 11.1 % | 0.92 |
| electricity | 6.4 % | 7.1 % | 1.11 |
| transport | 5.1 % | 7.1 % | 1.38 |
| heating | 9.2 % | 5.1 % | 0.55 |
| payments | 2.6 % | 5.6 % | 2.11 |
| construction | 1.7 % | 4.0 % | 2.32 |
| **government** | **0.6 %** | **5.1 %** | **8.18** |
| commerce | 0.4 % | 2.5 % | 6.68 |

The eval suite deliberately over-samples rare domains (`government`, `commerce`,
`payments`, `construction`) by 2–8×. This is a **training-data gap**, not an eval
flaw: `government` is 73 topics in the whole corpus. It plausibly contributes to
the v3 `domain_missed` count going 20 → 28 (§6). v3.1 does **not** add rare-domain
rows; a future run should consider whether the eval is weighted by frequency or
purpose (it is diagnostic, so the over-sampling is defensible — but then domain
misses on rare classes must not be read as regressions).

**Cross-suite consequence (unchanged).** The frozen 329 `issue_rouge_l` falls
0.9234 → 0.6611 because its v1 reference still contains the boilerplate the model
now strips. v3.1 pushes in the same direction; the frozen suite must not be
pooled with eval_v3.

---

## H. V3.1 proposal

### H.1 Recommendation

**Run one more full-corpus training pass, same budget, on the corrected corpus
(commit `e8f336c`).** Justification: R1 is the direct cause of the category-A
collapse and its fix is a one-line domain assignment; R2 removes the 11.2 %
phone-mode residue that C1 left in the target (now 0.05 %); R3 rescues 12
rows of real content the previous rule deleted. All three change exactly the
quantities three gates measure, add no data, and carry no new risk. The previous
run already established the treatment *beats* v2 significantly on the primary
suite (`boilerplate −0.144 p=0.0001`, `action_redundancy −0.116 p=0.006`,
`topic_count +0.089 p=0.024`, `issue_rouge +0.042`), so the marginal value of
correcting its known data bugs is high.

### H.2 Exact recipe

| knob | value |
| --- | --- |
| **dataset** | `ml/data/tune/v3/treatment/train.jsonl` @ `e8f336c`, sha256 `22e85d80…` |
| **rows** | **9 447** = single **5 157** + terse **1 872** (234 × 8) + multitopic **2 418** (real share 0.2184) |
| **iters** | **4 724** = `ceil(9447 / 2)` → exactly one epoch, 100 % coverage |
| **changing** | batch 2 / grad-accum 2 (effective 4); seed 42 (seeded batch order) |
| base / LoRA | `mlx-community/Qwen3-8B-4bit`; rank 8, scale 20.0, dropout 0, layers 16 |
| lr | 1e-4 adamw, no schedule; max-seq 2048, mask-prompt True, grad-checkpoint True |
| save-every | 500 |
| validation / test | byte-identical to v2 (tokenizer + chat template inherited from the v2 artifact) |
| expected wall | ~21 h |
| **changes vs the v3 run** | R1 domain from `KIND_TO_DOMAIN`; R2 euphony (`[ву]`) + sentence-shape + clause ordering; R3 end-anchored sentence + clause-final guarded closer |

The run scripts already compute iters from the row count
(`run_v3_treatment.sh`), so no iteration constant needs editing.

**Optional variant C3′ (recommended).** Apply the terse 25-char threshold to the
**post-C1** kernel, dropping the 18 degenerate units (see §B):

| | as-is | C3′ |
| --- | --- | --- |
| terse rows | 1 872 | 1 728 |
| total rows | 9 447 | 9 303 |
| iters | 4 724 | 4 652 |

C3′ is a data-integrity improvement (removes 144 rows whose target is an address
fragment or closing phrase), not a fix for a measured gate failure. It should be
adopted only with a matching test pin; if the goal is to isolate the R1/R2/R3
effect, run the as-is corpus first.

### H.3 C3 decision

**Retain, repaired.** Do not remove: it is the only augmentation covering the
short-input regime, and the category-A collapse was a *label* bug (R1), not a
slice-design bug — removing C3 would discard the mechanism and leave the terse
gate unmeasured. Do not redesign the multiplier yet: ×8 is defensible while the
model still generalised on multitopic, but §B lists the degenerate-kernel removal
(C3′) as the one redesign worth doing.

### H.4 What a v3.1 run will and will not fix

| gate (target, blocking) | v3 | v3.1 expectation | basis |
| --- | --- | --- | --- |
| schema (= 1.00, **yes**) | 0.9795 | **unchanged · open** | §E; R1/R2/R3 do not touch format |
| boilerplate (≤ 0.02, **yes**) | 0.2260 | improves, may still miss | §C: residue 11.2 % → 0.05 %; model-side leak remains |
| topic_count (≥ 0.90, **yes**) | 0.8699 | modest improvement | cleaner targets; not a targeted fix |
| two_topic_recall (≥ 0.894, yes) | 0.9412 | holds | passed |
| frozen_domain (≥ 0.82, yes) | 0.8602 | holds | passed |
| smoke_two_topic (4/4, **yes**) | 1/4 | **unchanged · open** | not addressed by any R-change |
| terse_single (≥ 0.70, no) | **0.00** | **≥ 0.70 expected** | §A: R1 is the direct cause |
| empty_action (0 invented, no) | 1 | likely 1 | §F: reference artifact |
| issue_rouge (≥ 0.80, no) | 0.6611 | ~unchanged / lower | expected C1 trade-off on v1 labels |

**Is another full training run justified? Yes — but it will not, by itself,
produce a publishable model.** The data correction closes the terse regression
and most of the boilerplate gap; the schema gate (3 unparseable two-topic
outputs) and the smoke two-topic gate are **format/robustness** problems that no
amount of C1/R1/R2/R3 fixes. A publish-bound run must either also add a
strict-JSON/formatting mitigation (and re-measure schema) or accept that the
re-run is a *measurement* of the corrected data, not a publication attempt.

### H.5 Known residual issues carried forward

- `build_v3.py __main__` raises `KeyError` in its summary print after writing
  the dataset and `meta.json` (it iterates non-measure keys of `slices`). Files
  are correct; the CLI exit code is 1. Fix before automating a re-run.
- The 3 unparseable outputs (§E) and smoke two-topic (§D) remain open.
- Rare-domain under-representation (§G) is unfixed; `government` at 0.6 % of
  training against 5.1 % of eval is the worst case.
