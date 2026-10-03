# Final model assessment

**Date:** 2026-10-02
**Question:** after v2, previous-v3, corrected-v3 and v3.2, is there a releasable
arm, and is another training run justified?
**Answer:** no arm is releasable, **including v2**, which fails four blocking
gates and stays canonical only as the deployed contract. The decoding probe is
complete and yields **no** usable serving change. Another training run is **not**
justified yet: the two gates that block every arm have un-isolated causes, and
the cheapest real win — stripping the `""` source-text artefact — needs no
training. See §7 and §8.

Scope constraint for this document: inference-only analysis and reuse of
already-committed measurements. No weights were trained, no benchmark case or
evaluator definition was changed, and no score-hunting decoding configuration is
proposed.

> **Update (2026-10-03).** The `""` source-text artefact analysed in §5 has since
> been fixed at the prompt boundary rather than by retraining, as §8 anticipated
> would be cheapest. The fix, its scope and its real-generation evidence are in
> **[QUOTE_ARTEFACT.md](QUOTE_ARTEFACT.md)**. The numbers in this document are
> unchanged and remain historical; where the two overlap, the quotation artefact
> is now fixed and only v2 remains canonical.

## How to reproduce every number here

| claim source | command |
|---|---|
| cross-arm matrix, gates | `.venv/bin/python -m ml.tune.cross_arm_matrix`, `.venv/bin/python -m ml.tune.gates_v3 --arm ARM` |
| multitopic mechanism | `.venv/bin/python -m ml.tune.multitopic_mechanism` |
| action behaviour | `.venv/bin/python -m ml.tune.action_analysis` |
| R1/R2/R3 residue | `.venv/bin/python -m ml.tune.residue` |
| decoding probe (960 generations, complete) | `.venv-mlx/bin/python -m ml.tune.decoding_probe` then `.venv/bin/python -m ml.tune.decode_matrix` |
| `""` source-text artefact | `.venv/bin/python -m ml.tune.residue` (the `quote_artefact` block) |
| this document's claims | `.venv/bin/python -m ml.tune.verify_final_assessment` |

`ml/data/tune/eval/` is gitignored, so claims that read the frozen 329 need that
directory present. `verify_final_assessment` cannot run without it: the reference
style is computed live before the first check.

The decoding probe is a completed, seeded 960-generation grid and is treated here
as authoritative for §3 and for P11. An earlier **unseeded** smoke run was
discarded and is cited nowhere in this document; every decoding number below comes
from `ml/data/tune/decode_matrix.json`.

---

## 1. Cross-arm matrix

Full matrix with per-dimension classification and paired statistics:
`ml/data/tune/cross_arm_matrix.json`. Blocks: **v2**, **previous-v3**,
**corrected-v3**, **v3.2**.

### eval_v3 (146 cases)

| dimension | v2 | previous-v3 | corrected-v3 | v3.2 | blocking? |
|---|---|---|---|---|---|
| schema / JSON validity | 0.9863 | 0.9795 | 0.9589 | 0.9110 | yes |
| boilerplate leak | 0.3699 | 0.2260 | 0.0068 | **0.0000** | yes |
| topic count exact | 0.7808 | 0.8699 | 0.8014 | 0.7740 | yes |
| domain-set exact (cat A terse) | 0.80 | **0.00** | **1.00** | 0.80 | no |

### multitopic (85 two-topic cases)

| dimension | v2 | previous-v3 | corrected-v3 | v3.2 | blocking? |
|---|---|---|---|---|---|
| topic-count hits / 85 | 73 | 80 | 78 | 77 | — |
| MT85 recall | 0.8941 | 0.9412 | 0.9176 | 0.9059 | yes |
| frozen multi-topic exact | 0.0365 | 0.0122 | **0.0000** | 0.0030 | no |

### frozen 329

| dimension | v2 | previous-v3 | corrected-v3 | v3.2 | blocking? |
|---|---|---|---|---|---|
| domain accuracy | 0.8389 | 0.8602 | 0.8663 | 0.8693 | yes |
| macro-F1 | **0.7596** | 0.7556 | 0.6775 | 0.7209 | no |
| issue ROUGE-L | **0.9234** | 0.6611 | 0.6610 | 0.6412 | no |
| action presence match | **0.9909** | 0.5471 | 0.5775 | 0.5653 | no |
| smoke two-topic (of 4) | 1 | 1 | 0 | 2 | yes |

Paired tests, from `cross_arm_matrix.json` (`vs_v2` and `vs_corrected`):

| comparison | delta | p | supported |
|---|---|---|---|
| v3.2 schema vs v2 | −0.0753 | 0.0034 | yes (exact McNemar) |
| v3.2 schema vs corrected-v3 | −0.0479 | 0.0391 | yes |
| v3.2 frozen multi-topic exact vs v2 | −0.0334 | 0.0034 | yes |
| v3.2 topic count vs v2 | −0.0068 | 1.0000 | no |
| v3.2 topic count vs corrected-v3 | −0.0274 | 0.3877 | no |
| MT85 recall, any v3 arm vs v2 | +0.0118 … +0.0471 | 0.125 … 1.000 | no |
| frozen domain accuracy, any v3 arm vs v2 | +0.0213 … +0.0304 | 0.053 … 0.144 | no |

Only three of those are statistically separable from noise. The v3.2 schema
regression is real and significant. Its topic-count movement is **not** — the
0.7740 figure is not distinguishable from v2's 0.7808 or from corrected-v3's
0.8014, so the topic-count gate fails everywhere but no arm is *significantly*
better than another. The MT85 recall and frozen domain-accuracy improvements are
directionally consistent across all three v3 arms but individually
insignificant at n = 85 and n = 329.

Three results are decided by the evidence:

- **Boilerplate is solved.** 0.3699 → 0.0000, on the suite built to detect it.
- **Topic count and smoke two-topic are not solved by anything**, including v2 —
  and the differences between arms on topic count are not statistically real.
- **The frozen-329 macro-F1 and issue ROUGE-L regressions are largely a
  reference-style artefact** — see §4. They are not evidence of worse content
  extraction, and the style-independent domain accuracy gate moved the *other*
  way for every v3 arm, though not significantly.

## 2. Problem classification

Each item is classified by evidence, not by plausibility.

| # | problem | classification | evidence |
|---|---|---|---|
| P1 | boilerplate leak | **training-distribution** (fixed) | corpus boilerplate 0.6447 → 0.0085; generation 0.6657 → 0.0091 |
| P2 | R2 euphony / intervening-subject residue | **training-distribution** (fixed in generation) | generation R2 0.2736 → 0.0000; but the gate cannot see the `у` form — §5 |
| P3 | R3 generic closer | **model-generation** (not fixed) | corpus 0.0000 yet generation 0.0182–0.0304; v2 emits 0.0000 |
| P4 | terse domain collapse | **training-distribution** (fixed by corrected-v3, partly lost by v3.2) | cat A domain exact 0.80 → 0.00 (previous-v3) → 1.00 (corrected-v3) → 0.80 (v3.2) |
| P5 | topic count < 0.90 | **unresolved** | best is 0.8699; nothing reaches the gate |
| P6 | strict schema repetition loops | **model-generation at greedy** | greedy loop cells: v2 0, corrected-v3 1 (`ev3-083`), v3.2 7; §3 tests whether any decoding setting removes them |
| P7 | frozen issue ROUGE-L 0.9234 → 0.6412 | **evaluator limitation** | references are 0.6474 boilerplate / 293.9 chars; v3 emits 203.7–208.6 |
| P8 | frozen macro-F1 regression | **evaluator limitation** + real action-prior regression | domain accuracy rose; the F1 loss tracks action presence |
| P9 | action presence match 0.9909 → 0.5653 | **training-distribution**, in two opposite regimes | v3 arms fail on single-topic (0.064–0.181 vs v2 0.312), v2 fails on two-topic (0.019 vs 0.122–0.130) — §6 |
| P10 | "empty action invented" on cat H | **label problem** | 4 of 12 cat-H cases request an action in text and are labelled empty; ev3-062's label is wrong — §6 |
| P11 | repetition loops | **diagnosed by §3, not fixed** | no setting removes the 5 v3.2 loops that persist everywhere; an earlier unseeded smoke run hinted at a sampling repair, was discarded and is cited nowhere |
| P12 | quote JSON failures (5 cases) | **source-text + copying**, not model error | all 5 eval_v3 inputs contain a literal `""`; v2 parses all 5, both v3 arms parse 0 — §5 |
| P12b | `ev3-083` JSON failure | **unresolved** | no quote characters in the input; all arms emit 230 chars; only corrected-v3 fails |
| P13 | smoke two-topic never reaches 4/4 | **unresolved** | 1 / 1 / 0 / 2 |
| P14 | frozen multi-topic exact 0.0365 → 0.0030 | **unresolved** | all below 4% |

### Defect classification

Every defect named in this report, classified by what the evidence isolates:

| defect | classification | what isolates it |
|---|---|---|
| terse collapse (cat A 0.80 → 0.00) | **training-distribution** | corrected-v3 restores 1.00 with no other change; v3.2 gives it back |
| boilerplate leak (0.3699 → 0.0000) | **training-distribution** (fixed) | corpus boilerplate 0.6447 → 0.0085 tracks generation 0.6657 → 0.0091 |
| topic merge (2-topic SPLIT 26 → 20) | **unresolved** | no arm reaches the 0.90 topic-count gate; v2 fails it with zero malformed output, so it is decomposition, not JSON |
| JSON quote failures (5 cases) | **data + learned copying** | all 5 inputs contain the artefact; v2 parses 5/5, both v3 arms 0/5; present in both training sets (12,528 / 15,274) |
| repetition loops | **model behaviour** (v3.2) | 5 of 8 v3.2 loop cases persist under all five settings; budget changes nothing; the penalty reproducibly cures 2 and temperature creates 1 |
| R3 generic closer | **model behaviour** (not fixed) | corpus 0.0000, generation 0.0182–0.0304; v2 emits 0.0000 |
| R2 euphony | **training-distribution** (fixed) + **evaluator limitation** | generation 0.2736 → 0.0000, but `BOILER_SENT` matches only the `в` form |
| action behaviour | **training-distribution**, regime-specific | v3 corpus 0.2119 vs frozen labels 0.5015; v2 collapses on two-topic, v3 on single-topic |
| macro-F1 regression | **evaluator limitation** | frozen references are 64.74% boilerplate / 293.9 chars; v3 emits 203.7–208.6 |
| issue ROUGE-L regression | **evaluator limitation** | same reference-style bias; domain accuracy (style-independent) rose |
| cat-H empty-action labels | **data/label** | 4 of 12 request an action in text and are labelled empty |
| smoke two-topic 0–2 of 4 | **unresolved** | no arm reaches 4/4; not explained by JSON or action rate |
| serving/decoding | **not a defect** | probe: budget is inert, penalty reproducibly cures 2 of 8 v3.2 loops, temperature trades one loop for another, no gate closable |

Three of these are *not* the model being bad at extraction. P7, P8 and P10 are
measurement or label defects that make the v3 arms look worse than they are, and
P1/P2/P4 are real wins.

### Final cross-arm table

All blocking metrics plus the load-bearing diagnostics. `BLOCK` marks a documented
blocking gate. No global ranking — the arms trade off against each other and the
trade-off is the finding.

| metric | v2 | previous-v3 | corrected-v3 | v3.2 | gate | interpretation |
|---|---|---|---|---|---|---|
| schema (146) | 0.9863 | 0.9795 | 0.9589 | 0.9110 | BLOCK = 1.00 | only v2 is close; the `""` cases cost both v3 arms 5 |
| boilerplate (146) | 0.3699 | 0.2260 | 0.0068 | **0.0000** | BLOCK ≤ 0.02 | C1 solved it; the gate is blind to the `у` form but generation is genuinely clean |
| topic count (146) | 0.7808 | 0.8699 | 0.8014 | 0.7740 | BLOCK ≥ 0.90 | **no arm passes**; differences between arms are not significant (p ≥ 0.39) |
| domain-set exact (146) | 0.6301 | 0.6712 | 0.7055 | 0.6644 | — | corrected-v3 best; all within noise |
| out-of-enum (146) | 0.0068 | 0.0000 | 0.0000 | 0.0068 | — | all clean |
| expected-two exact (52) | — | 0.6538 | 0.4423 | 0.3846 | — | previous-v3 best; v3.2 worst |
| category E (of 8) | — | 6 | 1 | 3 | — | v3.2's own target, still below previous-v3 |
| MT85 topic count | 73 | 80 | 78 | 77 | — | best 80/85, gate-adjacent |
| MT85 recall | 0.8941 | 0.9412 | 0.9176 | 0.9059 | BLOCK ≥ 0.894 | all pass; improvements vs v2 insignificant (p ≥ 0.13) |
| MT domain-set exact | 0.6824 | 0.7765 | 0.7529 | 0.7765 | — | v3 arms better, insignificant |
| frozen multi-topic rate | 0.0365 | 0.0122 | 0.0000 | 0.0030 | — | all far below usable |
| frozen domain accuracy | 0.8389 | 0.8602 | 0.8663 | 0.8693 | BLOCK ≥ 0.82 | all pass; v3 arms highest (insignificant) |
| frozen domain macro-F1 | **0.7596** | 0.7556 | 0.6775 | 0.7209 | — | v2 best; the gap is reference-style biased (§4) |
| frozen issue ROUGE-L | **0.9234** | 0.6611 | 0.6610 | 0.6412 | — | v2 best; measures v2's style, not content |
| frozen object exact | 0.9422 | 0.9696 | 0.9848 | 0.9483 | — | corrected-v3 best |
| frozen hallucination | **0.0152** | 0.0456 | 0.0578 | 0.0213 | — | v2 cleanest; v3.2 nearly as good |
| frozen action presence match | **0.9909** | 0.5471 | 0.5775 | 0.5653 | — | v2 dominant; the 0.21 training prior |
| frozen action copy | 0.4407 | 0.0912 | 0.1246 | 0.0669 | — | v2 copies 4× more |
| action redundancy (146) | 0.2055 | 0.0890 | 0.1575 | 0.0753 | — | v3.2 cleanest |
| terse cat A | 0.80 | 0.00 | **1.00** | 0.80 | — | the one clean v3 win; v3.2 partly gave it back |
| terse A/B/C | 0.5769 | 0.2308 | 0.6923 | 0.4615 | — | same pattern |
| terse short | 0.6429 | 0.1429 | 0.8571 | 0.7143 | — | same pattern |
| smoke two-topic (of 4) | 1 | 1 | 0 | 2 | BLOCK = 4 | **no arm passes**; unresolved |

Read as trade-offs, not a ranking:

- **v2 is the only arm that is simultaneously strong on action presence
  (0.9909), hallucination (0.0152), issue ROUGE-L (0.9234) and macro-F1
  (0.7596).** Its two weaknesses — 0.3699 boilerplate and 0.7808 topic count —
  are the two that C1 later fixed in the v3 lineage.
- **corrected-v3 is the best content arm**: highest terse scores (1.00 / 0.8571),
  highest object exact (0.9848), highest MT domain-set exact. Its cost is the
  action prior (0.5775) and the worst macro-F1 (0.6775).
- **v3.2 trades breadth for its own target and loses**: it wins boilerplate
  outright and beats v2 on frozen domain accuracy and action redundancy, but it
  is worst on schema among the v3 arms' peers (0.9110), worst on expected-two
  (0.3846), below corrected-v3 on category E, and has the highest single-action
  hallucination gap in the v3 family. Its 12 JSON failures on 52 two-topic
  cases are the cost.
- **The 5 `""` inputs, the action prior, and the reference style account for most
  of the apparent v3 regression.** Correcting for them, the v3 lineage is
  genuinely better at content extraction than v2 and worse at output discipline.

## 3. Decoding probe

`ml/tune/decoding_probe.py` → `ml/data/tune/decode_matrix.py` →
`ml/data/tune/decode_matrix.json`. The grid is complete and was checked before
any interpretation: **960 of 960 cells** (3 models × 5 settings × 64 cases), zero
duplicates, zero missing, zero empty generations, and 960 distinct seeds. 64
cases = 52 expected-two-topic + 6 committed JSON failures + 12 unique controls
(6 overlap the JSONFAIL set).

Settings, chosen so that each knob can be isolated at a fixed value of the
others:

| setting | temperature | repetition penalty | max tokens |
|---|---|---|---|
| `budget1200` | 0.0 | — | 1200 |
| `reppen105` | 0.0 | 1.05 / 20 | 800 |
| `reppen1200` | 0.0 | 1.05 / 20 | 1200 |
| `temp01_norep` | 0.1 | — | 1200 |
| `temp01` | 0.1 | 1.05 / 20 | 1200 |

`reppen105` vs `reppen1200` isolates the **budget**; `budget1200` vs `reppen1200`
isolates the **penalty**; `budget1200` vs `temp01_norep` isolates the
**temperature**.

### Baseline and movement

The baseline is the committed greedy recording (`temp=0.0, max_tokens=800`)
restricted to the same 64 cases. `eval_v3/behaviour/` files cover only the 52
two-topic cases and two of the three arms are filed under a `-recheck` suffix,
so the baseline uses the semantic `SPLIT/MERGE/STOP` recording where it exists and
falls back to `json_ok` + topic count elsewhere. v2 has no recording at all, so
its baseline behaviour column is flag-derived and its STOP count is not
comparable to the v3 arms'.

| model | metric | baseline | best setting | Δ | cases moved |
|---|---|---|---|---|---|
| v2 | schema | 1.000 | 1.000 | +0.000 | 0 |
| v2 | topic count | 0.594 | 0.609 | +0.016 | 1 |
| corrected-v3 | schema | 0.906 | 0.922 | +0.016 | 1 |
| corrected-v3 | topic count | 0.547 | 0.609 | +0.062 | 4 |
| v3.2 | schema | 0.812 | 0.844 | +0.031 | 2 |
| v3.2 | topic count | 0.500 | 0.578 | +0.078 | 5 |

Full per-setting grid:

| setting | v2 schema / topic | corrected-v3 schema / topic | v3.2 schema / topic |
|---|---|---|---|
| `budget1200` | 1.000 / 0.594 | 0.906 / 0.547 | 0.812 / 0.500 |
| `reppen105` | 1.000 / 0.578 | 0.922 / 0.578 | 0.844 / 0.547 |
| `reppen1200` | 1.000 / 0.578 | 0.922 / 0.578 | 0.844 / 0.547 |
| `temp01_norep` | 1.000 / 0.609 | 0.906 / 0.547 | 0.844 / 0.547 |
| `temp01` | 1.000 / 0.594 | 0.922 / 0.609 | 0.844 / 0.578 |

### A. Are the quote failures invariant across decoding settings?

The five literal-quote cases fail in **5 of 5 settings on both v3 arms and 0 of 5
on v2** — 30 failing cells out of the 30 v3 cells that could have succeeded.
They are completely decoding-invariant, so no serving change touches them, which
is consistent with §5: the model is copying an artefact that is present in the
input.

`ev3-083` is the single exception and it is genuinely decoding-sensitive:

| case | v2 | corrected-v3 | v3.2 |
|---|---|---|---|
| `ev3-016` | 0 / 5 fail | **5 / 5 fail** | **5 / 5 fail** |
| `ev3-024` | 0 / 5 fail | **5 / 5 fail** | **5 / 5 fail** |
| `ev3-032` | 0 / 5 fail | **5 / 5 fail** | **5 / 5 fail** |
| `ev3-046` | 0 / 5 fail | **5 / 5 fail** | **5 / 5 fail** |
| `ev3-052` | 0 / 5 fail | **5 / 5 fail** | **5 / 5 fail** |
| **all 5 literal-quote cases** | **0 / 5 fail** | **5 / 5 fail** | **5 / 5 fail** |
| `ev3-083` (separate defect) | 0 / 5 fail | **2 / 5 fail** | 0 / 5 fail |

corrected-v3 fails `ev3-083` only under `budget1200` and `temp01_norep` — the
two settings **without** a repetition penalty — and succeeds under all three
settings that have one.

### B. Are repetition loops sensitive to the budget or the repetition penalty?

A loop is a **word-level** 6-gram occurring at least four times, in a generation
of at least 60 word tokens. (Tokens, not characters: character shingles flag
ordinary Ukrainian prose as repetition.) The v3 loops repeat a phrase from
mid-sentence, so comparing against the opening of the output detects none of
them; that is why the earlier "all decoding-invariant" reading was wrong. The
rule is recorded in `decode_matrix.json` so it can be reimplemented and checked
rather than trusted.

| model | cases that ever loop | greedy-1200 | penalty-800 | penalty-1200 | temp-0.1 | temp+penalty |
|---|---|---|---|---|---|---|
| v2 | 0 | 0 | 0 | 0 | 0 | 0 |
| corrected-v3 | 1 | 1 | 0 | 0 | 1 | 0 |
| v3.2 | 8 | 7 | 5 | 5 | 7 | 5 |

Isolating one knob at a time:

| comparison (loop cells) | v2 | corrected-v3 | v3.2 |
|---|---|---|---|
| budget 800 → 1200, penalty fixed | 0 → 0 | 0 → 0 | **5 → 5** |
| penalty off → on, budget fixed at 1200 | 0 → 0 | **1 → 0** | **7 → 5** |
| temperature off → on, no penalty | 0 → 0 | 1 → 1 | 7 → 7 |

Three conclusions, and they are the useful ones:

- **The token budget does nothing.** 800 and 1200 tokens give identical loop
  counts in every arm. Raising the budget cannot fix this, and the earlier
  committed failure to notice that cost real generations.
- **The repetition penalty is the only knob with a reproducible effect**, and it
  is small and arm-dependent: 0 loop cells removed for v2, 1 for corrected-v3,
  2 for v3.2.
- **Temperature is not inert — it trades one loop for another** (1 → 1 and
  7 → 7 in aggregate). Against the same no-penalty baseline, `temp-0.1` clears
  `ev3-028` and creates `ev3-081`. Its apparent help in the `temp01` column comes
  entirely from the penalty that column also carries.

Per case, so the aggregate counts above can be audited:

| case | greedy-1200 | penalty-800 | penalty-1200 | temp-0.1 | temp+penalty |
|---|---|---|---|---|---|
| ev3-017 | loop | loop | loop | loop | loop |
| ev3-025 | loop | loop | loop | loop | loop |
| ev3-037 | loop | loop | loop | loop | loop |
| ev3-047 | loop | loop | loop | loop | loop |
| ev3-049 | loop | loop | loop | loop | loop |
| ev3-028 | loop | clean | clean | clean | clean |
| ev3-036 | loop | clean | clean | loop | clean |
| ev3-081 | clean | clean | clean | **loop** | clean |

Four things follow that the totals hide:

- **5 of v3.2's 8 loop cases** (`ev3-017`, `ev3-025`, `ev3-037`, `ev3-047`,
  `ev3-049`) loop under **every one of the five settings**, including at 800
  tokens with the penalty on. Those loops are intrinsic to the arm, not a
  serving artifact. That is the number that matters, and no decoding setting
  touches it.
- **The penalty is reproducible and monotone.** `ev3-028` and `ev3-036` both
  loop at greedy and both are cleared by all three penalty settings, at both
  1.05 and 1.20 and at 800 and 1200 tokens. So "removes 2 of 8" is a real
  effect, not sampling noise — it is simply far too small to matter, because 5
  cases are untouchable.
- **Temperature is not inert; it trades.** The `temperature 1 → 1` in the
  isolation table above is a coincidence of counting, not a null result.
  Comparing the two no-penalty settings case by case, temperature at 0.1
  **clears `ev3-028` and creates `ev3-081`** — a one-for-one swap that leaves
  the total at 7. Anyone reading only the aggregate would have called temperature
  inert and been wrong about which cases it touches.
- **`ev3-081` needs the absence of a penalty to appear.** It is clean at greedy
  and clean under all three penalty settings. So the two useful knobs pull in
  opposite directions on different cases, and neither is a fix.

### C. Can schema validity improve without materially changing topic behaviour?

Yes, but by one or two cases, and only for arms that are already failing.
corrected-v3's entire schema gain is `ev3-083` (+1 case, +0.016). v3.2's is two
cases (+0.031). v2 has nothing to gain — it is already at 1.000 and stays there
in all five settings.

The topic-count deltas are larger in relative terms (+4 and +5 cases) but they
are the *same* cases moving into parseable output, not new decomposition
ability: v3.2's `SPLIT` count rises 32 → 35 → 37 as JSONFAIL cases collapse into
parseable single-topic output, while `MERGE` falls 12 → 9 and `STOP` holds at 8.

### D. Do decoding settings alter SPLIT / MERGE / STOP?

On the **52 expected-two-topic cases** only, greedy (`budget1200`, which
reproduces the committed baseline exactly on these cases) → best setting:

| model | SPLIT | MERGE | STOP | JSONFAIL |
|---|---|---|---|---|
| v2 → `temp01_norep` | 26 → 27 | 20 → 19 | 6 → 6 | 0 → 0 |
| corrected-v3 → `temp01` | 23 → 27 | 17 → 14 | 6 → 6 | 6 → 5 |
| v3.2 → `temp01` | 20 → 25 | 12 → 9 | 8 → 8 | 12 → 10 |

The movement is **out of JSONFAIL**, not between SPLIT and MERGE: corrected-v3
gains 2 parseable cases (JSONFAIL 6 → 5) plus 2 SPLIT↔MERGE reclassifications;
v3.2 gains 3 parseable cases (12 → 10) plus 2 reclassifications. STOP never
moves for either arm. Genuine SPLIT↔MERGE reclassification is 2–3 cases per
arm out of 52.

### E. Is there a decoding configuration that gives a meaningful general
robustness improvement?

**No.** The best cell for each arm is `temp01` (or `temp01_norep` for v2), and:

| model | best setting | schema | topic count | cases moved | closes a gate? |
|---|---|---|---|---|---|
| v2 | `temp01_norep` | 1.0000 (+0.0000) | 0.6094 (+0.0156) | 0 / 1 | no — topic count |
| corrected-v3 | `temp01` | 0.9219 (+0.0156) | 0.6094 (+0.0625) | 1 / 4 | no — schema, topic count |
| v3.2 | `temp01` | 0.8438 (+0.0313) | 0.5781 (+0.0781) | 2 / 5 | no — schema, topic count |

Note v2 is the one arm that **reaches** schema 1.000 on this subset, so its only
remaining blocker is topic count; naming schema as a v2 blocker would be false.
`closes a gate?` is derived from the blockers listed, not asserted.

Three reasons this is not a serving recommendation:

1. **No cell reaches a gate.** Schema would have to be 1.000 and topic count
   0.90. The best v3 cell is 0.922 and 0.609. The gates are defined on the full
   146-case and 85-case suites, so no number in this table is comparable to a
   threshold in the first place.
2. **The only release-ready model is v2, and it does not need it.** v2 is at
   1.000 schema with zero JSON failures and zero loops in all 320 of its cells.
   Its best setting moves one case of topic count (+0.016), which is noise.
   Changing the canonical serving config would perturb a frozen, SHA-verified
   public baseline for no measured benefit.
3. **The gains are probe-subset sized.** +1 and +2 cases out of 64. Choosing the
   best of five settings on the same 64 cases that selected it is selection on
   the evaluation data, not a general improvement.

The defensible reading: the repetition penalty removes a small, reproducible
number of loops from the *broken* arms and nothing at all from the healthy one.
It is a mitigation for arms that are not being released, applied to a defect
that persists in 5 of 8 v3.2 cases regardless.

### Multitopic transition matrices (committed greedy, 52 cases)

These come from `ml/data/tune/multitopic_mechanism.py` on the committed
recordings, not from the probe, and are the historical comparison the probe
does not replace. Rows are the source bucket, columns the destination.

**previous-v3 → corrected-v3**

| from \ to | SPLIT | MERGE | STOP | JSONFAIL | total |
|---|---|---|---|---|---|
| **SPLIT** | 17 | 12 | 1 | 4 | 34 |
| **MERGE** | 1 | 4 | 0 | 0 | 5 |
| **STOP** | 4 | 1 | 5 | 0 | 10 |
| **JSONFAIL** | 1 | 0 | 0 | 2 | 3 |
| **total** | 23 | 17 | 6 | 6 | 52 |

**corrected-v3 → v3.2**

| from \ to | SPLIT | MERGE | STOP | JSONFAIL | total |
|---|---|---|---|---|---|
| **SPLIT** | 16 | 4 | 1 | 2 | 23 |
| **MERGE** | 3 | 8 | 3 | 3 | 17 |
| **STOP** | 0 | 0 | 4 | 2 | 6 |
| **JSONFAIL** | 1 | 0 | 0 | 5 | 6 |
| **total** | 20 | 12 | 8 | 12 | 52 |

Movement: previous-v3 → corrected-v3 = 7 improved / 17 regressed / 28 unchanged.
corrected-v3 → v3.2 = **4 improved / 15 regressed / 33 unchanged**.

What v3.2 actually did, cell by cell:

- **It did not improve decomposition.** SPLIT as a destination *fell* 23 → 20.
  Three of its four improvements are `MERGE → SPLIT`, and the fourth is
  `JSONFAIL → SPLIT` — so every gain is a case that used to parse or merge now
  splitting, and nothing else moved up. It bought 4 by trading 15.
- **It mostly converted parseable output into failure.** SPLIT → MERGE is 4 and
  MERGE → STOP is 3, but SPLIT → JSONFAIL is 2, STOP → JSONFAIL is 2, MERGE →
  JSONFAIL is 3, and `JSONFAIL → JSONFAIL` is 5. JSONFAIL as a destination rose
  6 → 12, **doubling**. Five of v3.2's 12 JSON failures were already failures
  under corrected-v3.
- **It introduced new malformed output.** 5 of the 6 JSONFAIL cases under v3.2
  were fine under corrected-v3. Those are the repetition loops, and the probe
  shows 5 of v3.2's 8 loop cases persist under every decoding setting.
- **Category E, the intervention's actual target, is the only category that
  moved in its favour**: 2 improved / 2 regressed / 2 unchanged. Every other
  category regressed or held.

One measurement caveat that limits all of the above: the mechanism taxonomy
proposes 15 candidate features, and **4 of them have no variance at all** across
the 85 multitopic cases — `same_domain`, `different_domain`,
`lexically similar second issue` (`second_similar`) and `issue_sim_ge_60`. A
feature with no variance across every case carries no information and can explain
nothing, so the "which-candidate-wins" reading can only ever rest on the 11 that
do vary. Those four are not quoted as evidence anywhere in this document.

So the honest verdict on v3.2 is that it **failed to improve the underlying
decision boundary** and degraded an otherwise working arm: it halved the parseable
two-topic output while buying a token-net-positive result on the one shape it was
trained on, at the cost of regressions in all seven categories and a doubled
JSON-failure rate.

## 4. What the frozen ROUGE-L and macro-F1 regressions actually are

The frozen 329 references were produced by the **v2 contract labeler**, which
does not strip boilerplate. Measured directly:

| | boilerplate rate | mean `issue` chars |
|---|---|---|
| frozen 329 references | 0.6474 | 293.9 |
| v2 generations | 0.6657 | 297.2 |
| previous-v3 generations | 0.2888 | 208.6 |
| corrected-v3 generations | 0.0152 | 206.5 |
| v3.2 generations | 0.0091 | 203.7 |

ROUGE-L rewards reproducing reference text. v2 reproduces it almost exactly and
scores 0.9234. Every v3 arm was trained by C1 to *delete* the text the reference
still contains, so it scores 0.64 while emitting an issue 90 characters shorter.
That is the metric measuring style.

The corroboration is that the style-independent measure moved the other way:
frozen **domain accuracy rose** for all three v3 arms (0.8389 → 0.8602 / 0.8663 /
0.8693). `build_eval_v3` already documents that its own reference labeler closes
the boilerplate gap ("the contract labeler *with that gap closed*"); the frozen
329 were never re-derived that way.

This is a reason to **re-derive the frozen references under the C1 labeler before
using `issue_rouge_l` or macro-F1 as release evidence**, not a reason to trust
v3 on content. It is not a license to re-score the existing arms and declare a
winner.

## 5. R2 is invisible to its own gate

`build_eval_v3.BOILER_SENT` contains `в\s+телефонному\s+режимі` and **not**
`у\s+телефонному\s+режимі`. `v3_changes` documents this exact euphony blind spot
and fixes it in the corpus rules ("584 `в` against 127 `у`"). The gate therefore
cannot see the form C1 was specifically written to remove, so
`boilerplate_leak_rate = 0` is not by itself proof that R2 is gone.

Measuring both forms from the raw generations:

| arm | frozen 329 | 52 two-topic | corpus |
|---|---|---|---|
| v2 | 0.2736 | — | 0.1477 |
| previous-v3 | 0.2736 | 0.0192 | — |
| corrected-v3 | 0.0061 | 0.0000 | — |
| v3.2 | 0.0000 | 0.0000 | 0.0006 |

R2 is genuinely fixed in generation by corrected-v3 and v3.2 — the conclusion
holds, but it required a detector the suite does not ship.

### The five quote JSON failures are a source-text defect

Five of the 146 `eval_v3` inputs contain a literal `""` — two double quotes with
nothing between them. It is not Ukrainian quoting; it is an artefact of whatever
normalised the source text. It is also **not JSON-safe**: a model that copies the
span into a string value terminates that value early, and the output will not
parse.

Those five cases are the entire quote-failure story:

| arm | parses the 5 `""` inputs |
|---|---|
| v2 | **5 / 5** |
| corrected-v3 | 0 / 5 |
| v3.2 | 0 / 5 |

The obvious reading — "the v3 arms hallucinate broken quotes" — is wrong, because
the artefact is in the **inputs**. Nor is it a case of v2 never seeing it: both
training sets contain it in abundance (v2 `12,528` occurrences, v3.2 `15,274`).
The split is explained by the copying behaviour measured in §6: the v3 arms
reproduce input spans verbatim, so they inherit the artefact; v2 normalises the
span away. This is the same mechanism as the action-prior regression, and it is
why the fix belongs in the data, not the model.

`ev3-083` is a separate and unexplained failure: its input contains no quote
characters at all, all three arms emit 230 characters, and only corrected-v3
fails to parse.

## 6. Action behaviour

Source: `ml/data/tune/action_analysis.json`.

### C2 flattened the action prior, and generation falls below even that

**Training targets**

| population | action-bearing rate |
|---|---|
| v2 training targets | 0.4571 |
| v3 training targets | 0.2119 |
| v3 same-object stream specifically | 0.1877 (98 of 522) |
| frozen 329 **labels** | 0.5015 |

**What the models actually emit on held-out cases**

`eval_v3` holds 94 single-topic and 52 two-topic cases, so they are split by
expected topic count rather than pooled — pooling them makes the
single-vs-multitopic comparison circular.

| arm | frozen 329 | eval_v3 single-topic | eval_v3 two-topic | multitopic |
|---|---|---|---|---|
| v2 | **0.4985** | **0.3118** | 0.0192 | — |
| previous-v3 | 0.1337 | 0.0745 | **0.1224** | 0.2235 |
| corrected-v3 | 0.1702 | 0.1809 | **0.1304** | 0.2471 |
| v3.2 | 0.1033 | 0.0638 | **0.1250** | 0.2381 |

v2 reproduces the frozen label rate almost exactly (0.4985 against 0.5015). Every
v3 arm lands far below its own training rate on the frozen set — 0.1033 for v3.2
against a 0.2119 target — so the shortfall is **worse than prior reproduction**,
not merely a faithful copy of the corpus.

The per-regime split shows the action loss is **not** a multitopic-decomposition
artefact, and it is not uniform across the two model families. The two families
fail in **opposite regimes**:

- **v2 collapses on two-topic cases.** 0.3118 → 0.0192, a 16× drop, while
  beating every v3 arm on the same cases (0.0192 against 0.1224–0.1304).
- **The v3 arms collapse on single-topic cases.** 0.0638–0.1809 against v2's
  0.3118, and every one of them is worse than v2 there.

So "the model omits actions" is two different defects that happened to point the
same way on the frozen set. Both are **training-distribution** effects — v2's
own multitopic prior suppressed two-topic actions — and neither is fixed by
emitting more topics. The two-topic `n` is 40–49 rather than 52 because cases
where the model emits no topics at all contribute no action slot.

### The frozen mismatch is omission, not invention

| arm | omits | invents | agrees | match |
|---|---|---|---|---|
| v2 | 2 | 1 | 326 | 0.9909 |
| previous-v3 | 135 | 14 | 180 | 0.5471 |
| corrected-v3 | 124 | 15 | 190 | 0.5775 |
| v3.2 | 137 | 6 | 186 | 0.5653 |

The models fail by **dropping** the requested action, which is what a 21%
training prior predicts. Invention stays low, and v3.2 is the best v3 arm at it.

### C2 did not cause an action/copy trade-off

| | eval_v3 redundant | eval_v3 verbatim copy | frozen verbatim copy |
|---|---|---|---|
| v2 | 0.2055 | 0.1986 | 0.4407 |
| previous-v3 | 0.0890 | 0.0753 | 0.0912 |
| corrected-v3 | 0.1575 | 0.1507 | 0.1246 |
| v3.2 | 0.0753 | 0.0753 | 0.0669 |

v3 arms copy far *less* than v2 and are less redundant. There is no
copy-versus-extraction trade-off to recover.

### The cat-H "invention" is a label defect

Every v3 arm "invents" exactly one cat-H action, `ev3-062`, and v2 invents none.
`ev3-062`'s text reads *"Просить вжити заходів для перенесення туалету в інше
місце ... та організувати місце для скидання сміття"* — an explicit request — while
its expected `requested_action` is empty. The v3 arms extracted the action
correctly.

A full audit finds **4 of the 12** category-H cases have an explicit request
marker in the text with an empty label:

| case | marker | excerpt |
|---|---|---|
| ev3-056 | `просить` | Заявник просить видалити високу траву … |
| ev3-062 | `Просить` | Просить вжити заходів для перенесення туалету … |
| ev3-065 | `Просить` | Просить допомоги у вирішенні даної проблеми … |
| ev3-066 | `просить` | Заявниця просить видалити аварійне сухе дерево … |

So the `empty_action` advisory gate penalises a model for being *correct* on
roughly a third of its cases, and the 21% training prior suppressed the action in
three of the four. Per the no-benchmark-changes constraint these cases are
**reported, not fixed**; fixing them is a prerequisite for any future action work.

## 7. Release decision

### 1. What is the canonical public model?

`ml/data/tune/adapters/qwen3-8b-lora-v2-attempt10-fused`, exactly what
`serve_v2.DEFAULT_MODEL` loads. Unchanged, still published, still SHA-verified.
Nothing in this report alters it.

### 2. Is any v3 arm release-ready?

**No.** previous-v3, corrected-v3 and v3.2 each fail at least three blocking
gates:

| arm | failed blocking gates |
|---|---|
| previous-v3 | schema, boilerplate, topic count, smoke |
| corrected-v3 | schema, topic count, smoke |
| v3.2 | schema, topic count, smoke |

### 3. Does any existing model satisfy all blocking gates?

**No — not even v2.** v2 fails schema (0.9863), boilerplate (0.3699), topic count
(0.7808) and smoke (1/4). v2 remains the canonical public model because it is
the deployed contract and the arm the gates were baselined against, **not**
because it passes the v3 gates. There is no release candidate for this project
right now, on any arm.

### 4. What exactly blocks release?

Two gates are unresolved in **every** arm, including v2:

- **topic count ≥ 0.90.** Best is 0.8699. For v2 this is pure decomposition
  error, not malformed output — v2 emits zero JSON failures, so it is merging or
  stopping where it should split. For the v3 arms the `""` artefact costs 5 cases
  outright and repetition loops cost more. Neither cause is fixed.
- **smoke two-topic = 4/4.** Best is 2/4. The mechanism analysis does not
  explain it.

A fourth blocker is the **action prior**, which is understood but not fixed. C2
dropped the training action rate from v2's **0.4571** to **0.2119** against
frozen labels at **0.5015**, and that single number explains the frozen action
presence collapsing from v2's **0.9909** to v3.2's **0.5653**. corrected-v3 → v3.2
also regressed **15** of the 52 two-topic cases.

Three further blockers are pipeline-owned rather than model-owned:

- **The frozen-329 references are style-biased** (64.74% boilerplate, 293.9 mean
  issue chars) so `issue_rouge_l` and macro-F1 cannot judge a candidate.
- **4 of 12 cat-H action labels are wrong**, so action presence is currently
  scored against partly incorrect labels.
- **`build_eval_v3.BOILER_SENT` cannot see the `у` form of R2**, so the boilerplate
  gate is a floor, not a complete test.

### 5. Does the completed decoding probe provide a usable serving improvement?

**No.** The probe's own conclusion:

- The token budget is **inert** — 800 and 1200 tokens give identical loop counts
  in every arm.
- The repetition penalty reproducibly removes **2 of 8** v3.2 loop cases
  (`ev3-028`, `ev3-036`, both cleared at 1.05 and 1.20 and at 800 and 1200
  tokens) and **0** for v2. Temperature at 0.1 with no penalty *creates* one
  (`ev3-081`). The remaining 5 v3.2 cases loop under every setting, so the
  penalty is a mitigation,
  not a fix.
- The best cell for any arm is `temp01`, worth +1 case of schema and +4 of topic
  count for corrected-v3, +2/+5 for v3.2 — and **zero** for v2, the only arm that
  could be shipped.
- No probe cell reaches any gate, and the gates are defined on the full suites, so
  no number in §3 is comparable to a threshold at all.

Adopting any of these settings would change a frozen, SHA-verified public
baseline for no measured benefit, and would mean selecting the best of five
settings on the same 64 cases that chose it.

### 6. Is a new training experiment justified?

**Not justified yet** — see §8 for the decision and the exact candidate that
becomes available once the measurement is repaired.

## 8. Decision on further training

### Option A — STOP MODEL DEVELOPMENT FOR NOW. **This is the decision.**

Further training is **not justified yet**.

This is **not** "we do not understand the model". §2–§6 now have root causes for
most defects. It is "we cannot yet score a new arm, and the two gates that block
every arm have un-isolated causes":

1. **Topic count ≥ 0.90 fails in every arm and nobody knows why.** For v2 it is
   not a JSON problem at all — v2 emits no malformed output and still merges or
   stops where it should split. This is the single largest un-isolated defect in
   the project and it is not a data-rate problem.
2. **Smoke 4/4 fails in every arm**, best 2/4, unexplained.
3. **Two of the metrics that would rank a new arm are known to be biased** —
   the style-loaded frozen references and the 4 wrong cat-H labels.
4. **The cheapest real win does not need training at all.** Stripping the `""`
   artefact from the corpora and eval inputs removes 5 of the 6 non-loop JSON
   failures. That is a data fix worth more than any candidate training run, and it
   should be done first and measured before spending 20h on the rest.

Running another 20h experiment now would produce an arm scored against known-biased
metrics, aimed at an un-isolated decomposition failure, while a free data fix sits
untested. That is not a good trade.

### Rejected: Option B (one specific next training experiment now)

The action prior **is** a clean single-variable hypothesis — C2 dropped the
training action rate from 0.4571 to 0.2119 against frozen labels at 0.5015, and
that one number explains the 0.9909 → 0.5653 collapse. But it cannot be run yet:

- The `""` fix is free and changes the JSON-failure denominator, so re-running
  first would confound the two.
- The topic-count gate is unresolved and independent, so an action-rate-only run
  could not pass release gates even if it worked perfectly.
- Judging it would use the biased frozen metrics and the wrong cat-H labels.

**When the prerequisites below are met, this is the next controlled experiment**,
and it is worth stating now so the next pass does not have to re-derive it:

- **Hypothesis**: restoring the action-bearing target rate toward v2's 0.4571
  (from 0.2119), with no change to decomposition, recovers frozen action presence
  from ~0.57 toward 0.99 without costing schema or topic count.
- **Corpus change**: rebalance the C1/C2 sampler so the released corpus reaches
  ≈ 0.45 action-bearing targets. Add no new decomposition data, no category-E
  synthesis, and no prompt change — the v3.2 over-correction is the thing to
  avoid repeating.
- **Success**: frozen action presence ≥ 0.90, schema ≥ 0.97, topic count ≥ 0.85,
  and no category regressing by more than 2 cases on the 52.
- **Revert**: any of schema < 0.95, topic count < 0.77, or a repetition-loop case
  count above corrected-v3's.

### Rejected: Option C (freeze v3 and ship corrected-v3 or v3.2)

Both fail three blocking gates. corrected-v3's content quality is the best in the
project, but it regresses action presence to 0.5775 and doubles JSON failures, and
shipping either would ship against the biased metrics above.

### Ordered prerequisites before any run

1. Strip the literal `""` from corpora and eval inputs; re-measure §5. (Free,
   highest value.)
2. Re-derive the frozen-329 references under the C1 labeler, or retire
   `issue_rouge_l` and macro-F1 from release evidence.
3. Repair the 4 mislabeled cat-H cases and re-audit `requested_action`.
4. Isolate the topic-count ceiling — v2's decomposition failure on cases where it
   emits valid JSON is the actual research question.
5. Only then run the action-rate experiment above.

### Next actions

1. Strip the `""` artefact from both corpora and the eval inputs.
2. Rebuild the frozen references under the C1 labeler.
3. Repair the four cat-H labels and re-audit the action labels.
4. Investigate the topic-count ceiling on v2.
5. **Leave v2 serving unchanged.** No prompt, template or decoding change is
   recommended.

## 9. Limitations

- The decoding probe covers the 52 two-topic `eval_v3` cases plus JSONFAIL and
  control samples. It cannot adjudicate `smoke_two_topic`, `frozen_domain` or
  `two_topic_recall`; it reports direction and size on the cases it covers.
- `ml/data/tune/eval/` is gitignored, and `verify_final_assessment` cannot run
  without it: the frozen-329 reference style is computed live from
  `lora_targets.jsonl` and each frozen generation file, both loaded before the
  first check. The per-arm frozen-329 *values* quoted above are independently
  traceable to the tracked `cross_arm_matrix.json`; the style measurement is not,
  and would have to be recomputed on a machine that has the directory.
- The R3 detector is a documented regex over sentence-final generic closers. The
  six hit sentences are quoted in `residue.json` so the calls can be adjudicated
  by hand.
- The cat-H audit is the same: a marker regex plus the quoted excerpt for every
  hit.
- The `""` finding in §5 establishes *where* the five quote failures come from
  and *which arm* inherits them. It does not establish that stripping the
  artefact fixes them: that would need a new arm, which this report does not
  recommend. The claim is that the mechanism is identified and cheap to remove,
  not that removal is proven.
- No causal claim is made from a single arm pair. §2 attributes a cause only where
  a measurement isolates it.