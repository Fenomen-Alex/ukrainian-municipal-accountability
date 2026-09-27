# v3 plan

`plan.json` is the machine-readable plan; this file is the reasoning behind it.
**Nothing here has been trained.** No adapter, fused model, or training run was
produced by this work.

## The one-paragraph version

The v2 audit isolated four failures that have independent causes and independent
fixes: administrative boilerplate is copied verbatim into `issue`; `requested_action`
is 94 % redundant with `issue` and so is not learnable; terse complaints are 58 % of
the 2023 slice and 0 % of the held-out sets, so nothing in the current benchmarks
measures the register where multi-topic extraction actually fails; and the 13-value
ontology collapses distinguishable problems (traffic lights and playgrounds are both
`sanitation`). v3 trains one treatment arm against the unchanged v2 data as a
control, so any regression is attributable rather than guessed at.

## Why a control arm at all

Each of C1–C4 trades against a number that v2 is currently good at. Stripping
boilerplate will drop `issue_rouge_l` against the frozen v1 label, because that label
contains the boilerplate — the drop is the point, not a regression. Emitting an empty
`requested_action` when the text has no request verb will drop
`requested_action_presence_match`, because the frozen target has actions the text
never asked for.

A single treatment run therefore cannot tell you whether a metric moved because the
fix worked or because something else broke. The control arm exists to be run first
and land within one point of every recorded baseline. If it does not, the harness is
suspect and the treatment number is uninterpretable — so this is a hard precondition,
not a formality.

## Benchmarks are not interchangeable

| suite | n | what it scores against | role |
| --- | --- | --- | --- |
| `ml/data/tune/eval` | 329 | v1 weak label, boilerplate intact | regression only |
| `ml/data/tune/multitopic/eval.jsonl` | 85 | v1 weak label, boilerplate intact | regression only |
| `ml/data/tune/eval_v3` | 146 | cleaned reference label | **primary** |
| `ml/data/tune/smoke` | 20 | human-curated | guardrail |

A model that correctly learns to strip boilerplate scores *worse* on the first two.
That is why the plan states an expected `issue_rouge_l` drop as a non-blocking gate
with a floor of 0.80: recorded so the drop is not later mistaken for a regression.

## Leakage and identity

Augmentation may only be derived from train-split content. Held-out records are used
only to build `eval_v3`. The check is
`ml/tests/test_eval_v3_suite.py::test_no_case_text_leaks_into_training`.

Both the builder and that test key on **normalised text, never `uid`**. The portal
reuses case numbers across years — `Б-67` is both a 2023 lift complaint and a 2026
waste-invoice complaint — so 130 uids appear in more than one split with unrelated
text, and 58 are duplicated inside the held-out pool alone. Any dedup or leakage
check keyed on `uid` would silently pass.

## Out of scope, deliberately

- **The schema is not changed.** The 12-label / 13-enum mismatch is documented in
  `ml/reports/ontology_audit.md`, not fixed. Changing it would invalidate every
  recorded metric in this repository and every gate in this plan.
- `benefits` and `other` are left empty; they have zero training support and no
  defensible weak-label rule.
- No re-splitting. The date-based boundary is what makes the terse-note finding
  visible in the first place; re-splitting would hide it.
- No model artifacts.

## Files

- `plan.json` — objective, arms, changes C1–C4, benchmarks, nine success gates, budget, preconditions, leakage policy, out-of-scope list.
- `../eval_v3/` — the primary benchmark, 146 cases, categories A–V.
- `../../reports/v3_experiment_design.md` — the same design in prose, with the audit evidence behind each change.
- `../../reports/ontology_audit.md` — the 12/13 mismatch and the domain confusions behind C4.
- `../../reports/reproducibility_audit.md` — the batch-shuffle defect that must be fixed before the control arm is trusted.
