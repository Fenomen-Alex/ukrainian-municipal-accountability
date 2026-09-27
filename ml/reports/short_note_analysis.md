# Short-note analysis

Why the terse register — the one where the deployed model actually fails — is
invisible to every recorded benchmark, and what there is to train on.

## 1. The finding

The frozen split is by **date**, and terse complaints are a **2023 phenomenon**.

| year | rows | median length | < 150 chars | < 200 chars | split |
| --- | --- | --- | --- | --- | --- |
| 2023 | 2099 | 173 | **839** | 1210 | train |
| 2024 | 1691 | 333 | 1 | 6 | train |
| 2025 | 1594 | 370 | 1 | 1 | train |
| 2026 | 1453 | 391 | 0 | 1 | validation, test |

Per split:

| split | n | min length | median | < 150 | < 200 |
| --- | --- | --- | --- | --- | --- |
| train | 5384 | 41 | 296 | 841 | 1217 |
| validation | 1124 | **197** | 388 | **0** | 1 |
| test | 329 | **218** | 407 | **0** | **0** |

**The held-out sets contain no terse records at all.** The shortest validation
complaint is 197 characters and the shortest test complaint is 218. Not one
benchmark number recorded in this repository — frozen 329, multitopic 85, v1, the
deterministic oracle — was measured on an input shorter than 197 characters.

Meanwhile 2023 accounts for 839 of the 841 terse training records. Short notes are
essentially a training-only phenomenon, and the date split puts every one of them on
the train side.

## 2. Why this matters more than the numbers suggest

The smoke suite is the only artefact that touches the terse register, because it was
hand-written rather than sampled from the corpus. It reports **1/4** on two-topic
cases. The multitopic suite reports **0.894** multi-topic recall.

Both are true. They measure different populations:

- multitopic 85: synthetic compositions of mid-length complaints
- smoke two-topic: short, hand-written notes

The 89.4 % vs 1/4 gap is not noise. It is the difference between a 300-character
input that states both problems and a 60-character note that mentions two in passing.
The model has never been evaluated on the second kind, and — per §1 — could not have
been, because the corpus contains no such held-out record to sample.

Note also that all three smoke failures (`smoke-14`, `smoke-15`, `smoke-16`) are
terse two-topic cases, while `smoke-13` — the same structural pattern, longer input —
passes. The discriminator is length, not construction.

## 3. What is available to train on

Terse pool, train split, raw content 60–260 characters:

| measure | n |
| --- | --- |
| candidate records | 2010 |
| non-empty note kernel after boilerplate removal | **1209** |
| empty kernel (every sentence filtered as boilerplate) | 801 |
| kernel ≥ 25 characters | **928** |
| kernel < 25 characters | 281 |
| kernels carrying an explicit two-topic connective | 35 |
| distinct portal kinds represented | 12 of 12 |

Kernel length min/median/max: 1 / 69 / 227.

`_note_kernel` is `ml/tune/build_multitopic.py::_note_kernel`: it splits on
sentences, drops any sentence matching the boilerplate list, then strips request
wrappers. It is the same transformation the eval-v3 condensed cases use, so training
and evaluation share a definition of "terse note".

Two observations:

- **928 usable kernels** is a workable upsampling pool. At 8–12× that yields
  7400–11000 terse examples, more than enough to change the register without
  dominating an ~10k training set. Proposed as v3 change C3.
- **The 801 empty kernels are the boilerplate bug again.** 40 % of the terse pool is
  a record whose every sentence is administrative filler. That is not a coincidence:
  short records in this corpus are disproportionately boilerplate, which is exactly
  why `_ADMIN_CLAUSE`'s `надає згоду` gap hurts this register most.

Only 35 kernels contain an explicit two-topic connective. Synthetic note-style
composition is therefore the only practical way to produce terse **multi-topic**
training data — the corpus will not supply it. The v2 stream already does this
(`build_note_style`, `n_note_target`, currently 0 in the v2 build), and C3 turns it
back on with a stated target rather than leaving it at a default.

## 4. What this rules out

- **Re-splitting to get terse held-out records.** Any split that puts 2023 material
  on both sides leaks the date confound in a different direction, and the current
  split is text-clean (0 exact cross-split duplicates). Not worth it for 1453
  validation rows and 841 terse training records.
- **Claiming a terse metric from the frozen suite.** Any terse number reported
  against `ml/data/tune/eval` would be a category-A-only figure over cases the
  builder composed from held-out records, and must be labelled as such.
- **Treating `issue_rouge_l` 0.9234 as evidence the model handles short inputs.** It
  is measured on a population with a 218-character floor.

## 5. How it is measured now

`ml/data/tune/eval_v3` closes the gap without touching the split:

| | n | note |
| --- | --- | --- |
| cases under 150 characters | 16 | shortest 65 |
| category A (terse single) | 10 | 6 mid-length, 4 forced to ≤ 110 chars |
| category B/C/D/E/F/G/K (two-topic) | 52 | the 1/4 failure mode |
| category J (boilerplate around the problem) | 6 | |

The condensed cases are built by deleting boilerplate sentences and wrapper words
from a **held-out** record — deletion only, nothing invented — so they carry the terse
register while remaining traceable to real text. Provenance is recorded per case as
`real` / `condensed` / `composed`, and 0 of 146 case texts appear in any training
stream.

Scores on eval_v3 category A are the first terse measurement this project will have.
The v3 plan records the gate as **non-blocking with a null baseline** for exactly
this reason: there is no prior number to beat, so the first run establishes one rather
than being judged against a guess.

## 6. Caveat on the A-category references

A condensed case's `expected_topics` inherits its domain from the portal `kind`, and
`_note_kernel` can drop the sentence that carried the substance — in which case the
reference is weaker than the text. The builder skips records whose reference `issue`
is empty after cleaning (`usable()`), and the test suite asserts every reference still
validates against the production schema, but a condensed reference is a
transformation of a label, not a hand annotation. Treat category A as directional
until a human labels ~30 of them.
