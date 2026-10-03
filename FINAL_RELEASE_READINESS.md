# Final release readiness — canonical v2

**Date:** 2026-10-03
**Subject:** the canonical v2 model (`Fenomen-Alex/ukrainian-municipal-accountability-qwen3-8b`)
**Verdict:** **READY FOR CANONICAL USE as a documented contract — NOT a released model.**
**Distribution readiness:** **`BLOCKED`** — see §3a. One documentation defect on
the public model card, not a model problem.

Those halves are not a hedge; they are the precise finding:

* **Ready** — the serving path is pinned, reproducible, documented, and tested
  end-to-end against the real model on Apple Silicon. A fresh integrator can
  reproduce it from this repository.
* **Not a released model** — v2 **fails four blocking gates** of the project's
  own release criteria, and `ml/tune/FINAL_MODEL_ASSESSMENT.md` §7 concludes that
  *no arm is releasable, including v2*. v2 is canonical as the **deployed
  contract**, not as a model that clears the bar. Nothing here should be read as
  clearing that bar, and the v3 lineage is not a near-miss.

**Two questions are now settled by primary-source evidence**, superseding the
earlier "unknown/pending" wording in this document:

* **Source-data licensing: CC BY 4.0 International**, publisher Executive
  Committee of the Kropyvnytskyi City Council. Redistribution is explicitly
  permitted **with attribution**. Record:
  [DATA_PROVENANCE.md](DATA_PROVENANCE.md).
* **Public HF artifact: reachable, and fully verified.** All six contract files
  downloaded fresh and matched the recorded SHA-256 manifest byte-for-byte,
  including the 4.6 GB `model.safetensors`. Details in §5.

| Canonical identifiers | |
|---|---|
| Public artifact | `Fenomen-Alex/ukrainian-municipal-accountability-qwen3-8b` |
| Local artifact | `ml/data/tune/adapters/qwen3-8b-lora-v2-attempt10-fused` (gitignored) |
| SHA manifest key | `qwen3-8b-lora-v2-fused` in `ml/data/tune/artifact_sha256.json` |
| HF revision verified | `6a831b155aa4891b2a96206df0e665c7355f04bb` |

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
| 10 | Canonical contract, native MLX | `.venv-mlx/bin/python -m ml.tune.verify_v2 --native` | **9/9 contract cases pass**, exit 0 |
| 11 | Public HF artifact, fresh download | `curl -L …/resolve/main/<file>` ×6 | **all 6 files 200** |
| 12 | HF SHA-256 vs recorded manifest | sha256 of downloaded files | **6/6 MATCH byte-for-byte** |
| 13 | Source-data provenance | `CKAN package_show` on publisher portal | **CC BY 4.0 confirmed** |
| 14 | Provenance claims consistency | `.venv/bin/python -m ml.tune.verify_release_provenance` | see §1a |

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

### Provenance — established, with a live attribution obligation

**RESOLVED, and it resolved in the project's favour.** The source corpus is
**not** unlicensed. It was verified directly against the publisher's own CKAN
metadata:

* **Dataset:** «Дані про надходження звернень на телефонні "гарячі лінії", в
  аварійно-диспетчерські служби, телефонні центри тощо», published by the
  **Executive Committee of the Kropyvnytskyi City Council** at
  `https://data.kr-rada.gov.ua`, dataset id
  `9109c3a3-980f-45d0-a83e-b60f87d583d4`.
* **Licence:** `license_id: cc-by`, *Creative Commons Attribution*; the dataset
  page declares Open Definition conformance; the portal-wide default is
  **Creative Commons Attribution 4.0 International**.
* **Terms:** CC BY "allows re-distribution and re-use of a licensed work on the
  condition that the creator is appropriately credited."

Redistribution is therefore **explicitly permitted**, conditional on attribution.
Full evidence, exact wording, attribution owed, and the two corrections to the
project's earlier source reference (a one-character dataset-id typo, and a
filename that no longer exists) are in
[DATA_PROVENANCE.md](DATA_PROVENANCE.md).

**Two consequences:**

1. **The attribution condition is now satisfied** by that document. It was not
   satisfied before, and the earlier claim in this file — that no licence was
   documented — was wrong.
2. **PII remains in the distributed corpora.** The heuristic redaction is
   incomplete: **187** phone-like and **114** email-like rows survive across the
   25 tracked `.jsonl` files (27,235 rows scanned). CC BY permits this; it is
   nonetheless a live distribution of personal data.

## 3a. Distribution readiness

**Status: `BLOCKED`**

This is a **distribution/readiness** verdict, not a model-quality verdict. It is
blocked on a single, specific, fixable item — and it is *not* the model.

**Model-quality statements above are unchanged and remain in force:**

* **v2 is canonical and deployed.** It is the published artifact and the served
  contract.
* **No arm clears all internal model-quality gates**, including v2, which fails
  four blocking gates (§6).
* **Model development is frozen.**

### Why distribution is BLOCKED

Attribution is a *condition* of the only licence that permits redistribution, and
this project's own **public model card currently contradicts it**. The published
Hugging Face card states:

> "The **training data is not redistributed here and its license is not
> documented** in the source project. The municipal complaint corpus has no
> recorded source URL, license, or redistribution terms."

That is **false** — a source URL is recorded and the licence is CC BY 4.0. So as
things stand:

* the repository ships derived CC BY material **and** satisfies attribution
  locally, but
* the **public artifact** misstates the data terms, understates the grant, and
  omits the attribution condition to anyone who reads only the model card.

Redistributing under CC BY while the project's own published card tells users the
data has no licence is the blocker. It is a documentation defect, not a legal one,
and it is **fixable by editing the public model card** — which was deliberately
**not** done in this task, because the public artifact must not be modified
without the owner's decision.

### What would change the status

* **To `READY FOR PUBLIC DISTRIBUTION`:** update the public model card's
  "License and data provenance" section to state CC BY 4.0 with the attribution
  notice and link `DATA_PROVENANCE.md`. Nothing else is outstanding on the
  distribution side — the corpus terms are settled and the attribution is already
  written.
* **To `READY EXCEPT FOR EXTERNAL VERIFICATION`:** not applicable. External
  verification did not fail (§5); it **succeeded**.

### Not blockers, recorded so they are not mistaken for blockers

* **Residual PII in tracked corpora** — permitted by CC BY, but a legitimate
  remediation target. Removing it would require rebuilding corpora and
  retraining, which the freeze forbids; a corpus-only redaction without
  retraining is possible but would change SHA-pinned training data.
* **The unversioned source** (`appeals.csv`, no checksum) — means the corpora
  cannot be re-derived later. It does not block distribution of what exists.
* **The erroneous dataset id and dead filename** in earlier documentation —
  corrected in `DATA_PROVENANCE.md`.

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

## 5. External verification — corrected, and now PASSING

**The previously recorded blocker was a misdiagnosis, and it is resolved.**

Earlier revisions of this document stated that "fresh HF download verification is
blocked by `HTTP Error 400: Bad Request`". That was wrong about the cause. The
400 never involved Hugging Face. Traced to source:

```
File "ml/tune/verify_v2.py", line 291, in run_http
    data = _post(f"{base_url}/chat/completions", body)
```

`verify_v2` defaults to `--base-url http://127.0.0.1:1234/v1`, i.e. a **local
LM Studio server**, not HF. Reproducing the request directly returns:

```json
{"error":{"message":"No models loaded. Please load a model in the developer page
 or use the 'lms load' command.","type":"invalid_request_error","param":"model"}}
```

So the failure class is **endpoint-level on a local server with no model
loaded** — not repository-level, not file-level, not authentication-related, and
not sandbox-specific. Hugging Face was never contacted by that code path.

### Public HF artifact — independently verified

| Operation | Command | Result |
|---|---|---|
| Repository existence | `GET /api/models/Fenomen-Alex/…-qwen3-8b` | **200** |
| Public visibility | same | `private:false`, `gated:false`, `disabled:false` |
| File listing | same | **8 files**, `sha=6a831b15…`, `lastModified 2026-09-28` |
| Model card | `GET /resolve/main/README.md` | **200**, 9,998 bytes |
| Single-file download | `GET /resolve/main/config.json` | **200**, 939 bytes |
| Library access | `HfApi().model_info` / `hf_hub_download` | **both OK** |

**Full fresh artifact verification succeeded.** All six contract files were
downloaded to a temporary directory and hashed:

| File | bytes | SHA-256 vs recorded manifest |
|---|---:|---|
| `chat_template.jinja` | 4,116 | **MATCH** |
| `config.json` | 939 | **MATCH** |
| `model.safetensors` | 4,607,835,164 | **MATCH** |
| `model.safetensors.index.json` | 64,105 | **MATCH** |
| `tokenizer.json` | 11,422,650 | **MATCH** |
| `tokenizer_config.json` | 413 | **MATCH** |

**All 6 contract files verified byte-for-byte**, no unexpected files present, and
no symlinks — the weights file hashed as a real 4.6 GB file. The manifest entry
used is `qwen3-8b-lora-v2-fused` in `ml/data/tune/artifact_sha256.json`, which
resolves to the canonical `…-v2-attempt10-fused` directory.

Additionally, `verify_v2 --native` now runs the canonical contract cases in
process under MLX, with no LM Studio dependency:

```
$ .venv-mlx/bin/python -m ml.tune.verify_v2 --native
[9 cases] CONTRACT: 9/9 cases pass        QUALITY notes: 8        exit=0
```

The 8 quality notes are the documented v2 weakness profile (topic-count
under-splitting, `requested_action` left empty), not contract failures.

### Remaining environmental limitations

* **`verify_v2` in its default HTTP mode still cannot run here** — it requires a
  model to be loaded in LM Studio's developer page or via `lms load`. That is a
  local environment state, not a repository defect, and no state was changed to
  force it. Use `--native` for an equivalent check. The HTTP path itself is
  documented as working in `V2_SERVING.md` when a model is loaded.
* **A fresh clone still cannot fully verify itself.** Checks 4–7 require the
  gitignored adapters and generated eval corpora. A fresh clone reports
  *uncovered claims*, never a false pass — by design.
* **Apple Silicon only.** `mlx` does not build elsewhere. On other runtimes,
  expect the documented cross-runtime variance: output is self-consistent within
  a runtime at `temperature=0.0` but not bit-identical across runtimes, and
  `object` fields vary most. Do not assert on them.

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