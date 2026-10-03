# Final release readiness — canonical v2

**Date:** 2026-10-03
**Subject:** the canonical v2 model (`Fenomen-Alex/ukrainian-municipal-accountability-qwen3-8b`)
**Verdict:** **READY FOR CANONICAL USE as a documented contract — NOT a released model.**

Those two halves are not a hedge; they are the precise finding:

* **Ready** — the serving path is pinned, reproducible, documented, and tested
  end-to-end against the real model on Apple Silicon. A fresh integrator can
  reproduce it from this repository.
* **Not a released model** — v2 **fails four blocking gates** of the project's
  own release criteria, and `ml/tune/FINAL_MODEL_ASSESSMENT.md` §7 concludes that
  *no arm is releasable, including v2*. v2 is canonical as the **deployed
  contract**, not as a model that clears the bar. Nothing here should be read as
  clearing that bar, and the v3 lineage is not a near-miss.

---

## 1. Verification matrix — results from this run

Every command below was executed against the working tree in this release-hardening
pass.

| # | Check | Command | Result |
|---|---|---|---|
| 1 | Full test suite | `.venv/bin/python -m pytest ml/tests -q` | **486 passed, 11 skipped, 4 subtests** |
| 2 | Release smoke suite, model-free | `.venv/bin/python -m pytest ml/tests/test_release_smoke.py -q` | **57 passed, 8 skipped** |
| 3 | Release smoke suite, real model (MLX) | `.venv-mlx/bin/python -m pytest ml/tests/test_release_smoke.py -q` | **65 passed** (47.96 s) |
| 4 | Assessment claim verifier | `.venv/bin/python -m ml.tune.verify_final_assessment` | **OK (380/380)** |
| 5 | v3.2 historical report | `.venv/bin/python -m ml.tune.verify_v3_2_report` | **PASS (197/197)** |
| 6 | Reproducibility (MLX) | `.venv-mlx/bin/python -m ml.tune.verify_repro` | **exit 0**, 9 checks PASS, 0 FAIL |
| 7 | Reference-arm SHA manifest | `.venv/bin/python -m ml.tune.artifact_sha --verify` | **ALL REFERENCE ARMS UNCHANGED** |
| 8 | Prompt-only round trip | `.venv/bin/python -m ml.tune.serve_v2 --request-only --text …` | valid request body, exit 0 |
| 9 | Whitespace / conflict markers | `git diff --check` | **clean** |

Skipped tests are the model-gated smoke cases in the non-MLX venv, where
`mlx_lm` is deliberately absent. They are not silently passing: check #3 runs
them for real.

### The release smoke suite

`ml/tests/test_release_smoke.py` covers eight complaint shapes a new integrator
is most likely to send first, and asserts the **contract**, never model wording:

ordinary single-topic · multitopic · quoted organisation · Ukrainian
mid-word apostrophe · quote-free · terse · explicit requested action · no
explicit requested action.

It asserts: messages are system-then-user with a byte-exact system prompt; the
system prompt is never mutated; the user turn carries no ASCII quote; quote-free
complaints are byte-identical to the pre-fix path; normalization is idempotent
and changes glyphs only, never content; pinned generation settings are unchanged;
a conforming payload satisfies `check_case`; and `parse_strict` **surfaces**
malformed output rather than repairing it.

It deliberately asserts **no** exact wording, because wording is not a project
contract. Check #3 confirms the real model returns strict, schema-valid JSON for
all eight shapes.

One fixture defect was caught by the project's own `check_case` during
authoring — two topics sharing identical `issue` text is over-conflation — and
the fixture was corrected rather than the checker relaxed.

## 2. What changed in this pass

Documentation and tests only. **No weights, benchmark cases, evaluator
definitions, thresholds, or the system prompt were modified.**

| Change | Purpose |
|---|---|
| `README.md` (new) | Canonical entry point: quickstart, contract summary, limitations, data/privacy notice |
| `ml/requirements-serve.txt` (new) | Pinned serving runtime; the repo previously had **no** dependency manifest |
| `ml/tune/FINAL_STATUS.md` (new) | Frozen / canonical / no-further-candidates status with required content |
| `ml/tune/MODEL_EXPERIMENTS.md` (new) | v1 → v2 → v3 lineage, why v2 is canonical, why v3 was rejected |
| `ml/tests/test_release_smoke.py` (new) | The eight-shape release smoke suite |
| `ml/tune/V2_SERVING.md` | Added quote normalization (user turn only) and the three-name model-identity table |
| `ml/tune/MODEL_CARD_V2.md` | Scoped the data-redistribution claim across HF vs GitHub |
| `ml/tune/FINAL_MODEL_ASSESSMENT.md` | Pointer to the quote artifact; measurements unchanged |
| 5 stale docs | Superseded / historical banners (§4) |

The quote fix itself is already released in `e39bed4` and is documented in
`ml/tune/QUOTE_ARTEFACT.md`.

## 3. Data, privacy and provenance

**No credentials, keys, `.env` files, tokens or model weights are tracked.**
Verified by pattern scan over all tracked files and by explicit path checks.

Tracked corpora: **25 `.jsonl` files, 78.2 MB**, largest being
`ml/data/tune/v2/train.jsonl` (26.4 MB).

Findings:

* **The tracked corpora are not PII-free.** Raw municipal complaint text contains
  real-world phone numbers and email-like strings (roughly 53 train rows with
  phones, including a third-party number, and about 22 with email-like strings).
  A naive regex scan matches every raw row through the substring `CATUTTC` — a
  false positive, which is itself worth knowing before anyone scans this data.
* **Adapters and generated eval outputs are gitignored**, as are the corpora the
  claim verifiers consume. No weights are in Git.
* **Over-broad `_PERSON_NAME` redaction** is a training-data defect inherited by
  every arm: the heuristic strips legitimate names. Documented, not fixed —
  fixing it means rebuilding corpora and retraining, which the freeze forbids.

### Provenance — an open question, not a cleared item

**The project records no source URL, licence, or redistribution terms for the
municipal complaint corpus anywhere in the repository.** The model weights are
Apache-2.0; that grant covers the weights and **does not extend to the corpus**.

Consequence, stated plainly: this repository cannot be redistributed on an
informed basis today. If your use involves redistributing the repo or its data,
that is blocked pending provenance, independently of the model's quality. The HF
artifact is unaffected — it ships weights only.

## 4. Stale references resolved

Each now carries a banner at the top of the file; none was deleted, since they
are the evidence behind the release decision.

| File | Marked as |
|---|---|
| `ml/data/tune/SERVING.md` | Superseded → `V2_SERVING.md`; retains the best LM Studio loading notes |
| `ml/data/tune/MODEL_CARD.md` | Superseded — this is the v1 card |
| `ml/reports/finetune_report.md` | Superseded v1 report; **training commands marked do-not-run** |
| `ml/reports/finetune_v2_report.md` | Historical provenance for the canonical model; commands marked do-not-run |
| `ml/reports/artifact_inventory.md` | Historical inventory, explicitly **not** a deletion list |

The vestigial Node scaffold (`package.json`, `src/index.ts`, `tests/index.test.ts`,
all empty) is documented in `README.md` as **not a runtime**. It was left in
place rather than deleted: removing it is a separate call, not a release-blocking
one.

## 5. Environmental limitations — read before trusting §1

**Fresh HF download verification is blocked in this environment.** `hf download`
of the public artifact fails with `urllib.error.HTTPError: HTTP Error 400: Bad
Request`, so `ml/tune/verify_v2.py` could not be re-run here. **No workaround was
attempted and none is documented as working.** Consequently:

* A **historical** anonymous download and 9-case byte-comparison are recorded in
  `ml/tune/V2_SERVING.md`; that evidence stands as historical, not as a
  verification performed in this pass.
* The public artifact's current reachability is **unverified here**. Confirm it
  before relying on the HF route in the quickstart.
* All §1 results except checks 1, 2, 4, 5, 7 and 9 ran against the **local**
  fused artifact.

**A fresh clone cannot fully verify itself.** Checks 4–7 require the gitignored
adapters and generated eval corpora. A fresh clone reports *uncovered claims*,
never a false pass — by design. Only the model-free layers of the smoke suite and
the core suite run unaided.

**Apple Silicon only.** `mlx` does not build elsewhere. On other runtimes, expect
the documented cross-runtime variance: output is self-consistent within a runtime
at `temperature=0.0` but not bit-identical across runtimes, and `object` fields
vary most. Do not assert on them.

## 6. The four gates v2 does not clear

From `ml/tune/FINAL_MODEL_ASSESSMENT.md` §2, restated because it is the reason
this is "not a released model":

| Gate | Requirement | v2 |
|---|---|---|
| schema validity (146 cases) | 1.00 | 0.9863 |
| boilerplate (146 cases) | ≤ 0.02 | 0.3699 |
| topic count (146 cases) | ≥ 0.90 | 0.7808 |
| smoke two-topic | 4 of 4 | 1 of 4 |

Two of these are now understood and partly addressed: the `""` quote artefact is
fixed at the prompt boundary (`QUOTE_ARTEFACT.md`), and boilerplate is what the v3
lineage solved — at the cost of schema validity and action detection. That
trade-off is documented rather than resolved, which is why the freeze stands
instead of another run.

## 7. Final position

**Ship the contract, not the claim.** v2 is documented, pinned, reproducible and
tested; use it with the limitations in §6 and in `ml/tune/FINAL_STATUS.md` in
view. Do not represent it as having passed a release review — it did not.

**Two items block any broader release**, and neither is a modelling problem:

1. **Corpus provenance** (§3) — undocumented source licence.
2. **v2's four failing gates** (§6) — accepted historically, not cleared.

Neither is fixed here, and neither should be quietly dropped from a release
checklist. Model development remains **frozen**; the cheapest remaining win — the
quote artefact — is already taken.