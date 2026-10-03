# Release-hardening audit inventory

Audit of the repository at `e39bed4` (post-quote-fix, model development frozen).
Scope: stale references, canonical-contract agreement, serving path, data/privacy,
fresh-clone usability. Nothing was deleted; historical reports are preserved.

## A. Canonical documents that do not exist yet

| # | Missing | Consequence |
|---|---|---|
| A1 | root `README.md` | a fresh clone has no entry point; purpose, canonical model and quickstart must be rediscovered |
| A2 | `ml/tune/FINAL_STATUS.md` | no single statement of current canonical state |
| A3 | `ml/tune/MODEL_EXPERIMENTS.md` | the v1 → v2 → v3 progression exists only as scattered reports |

## B. Stale or competing docs that could send a fresh user to the wrong artifact

| # | File | Problem | Action |
|---|---|---|---|
| B1 | `ml/data/tune/SERVING.md` | reads as *the* serving doc; names `qwen3-8b-lora-v2-fused` and never mentions `serve_v2.py`, the `enable_thinking` contract or quote normalization | superseded banner → `ml/tune/V2_SERVING.md`; keep its unique LM Studio and Ollama/GGUF content |
| B2 | `ml/data/tune/MODEL_CARD.md` | v1 card, "research prototype"; points at the v1 adapter and an old name `qwen3-8b-municipal-finetune` | superseded banner → `ml/tune/MODEL_CARD_V2.md` |
| B3 | `ml/reports/finetune_report.md` | v1 report containing **runnable** `run_train --iters 400` / `--save-path …-v2-fused` commands | historical banner, explicitly not to be re-run |
| B4 | `ml/reports/finetune_v2_report.md` | v2 report with `run_train --iters 800` commands | historical banner |
| B5 | `ml/reports/artifact_inventory.md` | v1/v2 artifact decisions incl. `DELETE` rows | historical banner; it is an audit record, not a to-do list |

## C. Canonical contract docs that are correct but incomplete

| # | File | Gap | Action |
|---|---|---|---|
| C1 | `ml/tune/V2_SERVING.md` | does not document the prompt-boundary quote normalization added in `e39bed4` | add the normalization to the contract |
| C2 | `ml/tune/FINAL_MODEL_ASSESSMENT.md` | consistent with the fix (5 *failing* cases) but does not point at the root-cause note | add pointer to `ml/tune/QUOTE_ARTEFACT.md` |

Consistency check: the assessment says "five cases", `ml/tune/residue` reports
five because its detector is `""`-only, and `QUOTE_ARTEFACT.md` reports six
eval cases carrying an ASCII quote. These agree: five fail, and the sixth
(`ev3-123`, a legitimate `"Копілка"`) parses today and is the regression guard.

## D. Machine-specific absolute paths in tracked metadata

| # | File | Detail |
|---|---|---|
| D1 | `ml/data/tune/artifact_sha256.json` | every `resolved` field is an absolute `/Users/alex/...` path |
| D2 | `ml/data/tune/v2/meta.json:15` | `multitopic_eval_suite` is an absolute path |

These are provenance records, not secrets, and their sha256 values are load-bearing
for `verify_v3_2_report` check 7.13. The values are **not** rewritten; instead the
generator is made repo-relative for future runs and the existing absolute paths are
documented as provenance.

## E. Tracked data and privacy

78 MB of `.jsonl` is tracked. It is not all equivalent, and one widely-repeated
metric was misleading on first pass:

| File | Rows | Size | Content |
|---|---|---|---|
| `ml/data/train.jsonl` | 5,384 | 7.9M | raw municipal complaints |
| `ml/data/validation.jsonl` | 1,124 | 1.8M | raw municipal complaints |
| `ml/data/test.jsonl` | 329 | 568K | raw municipal complaints |
| `ml/data/tune/v2/train.jsonl` | 8,519 | 26M | derived v2 training records |
| `ml/data/gold/annotation_set.jsonl` | 400 | 332K | gold annotations |

Personal-data measurement (counts only; no values reproduced):

* A naive phone regex matches **every** raw row, but that is the `CATUTTC`
  territorial-unit code, **not** a phone number. CONFIRMED against the upstream
  source: `CATUTTC` is a genuine column of `appeals.csv` (one of 20), holding the
  territorial-unit code.
* Real phone numbers inside complaint text occur in ~53 `train` rows, and the
  contexts include a third party's number (`номер чоловіка заявниці`), not only
  the applicant's own. ~22 rows contain email-like strings.
* Free text also carries addresses, timestamps, organization names and citizen
  references. Redaction exists in `_redact_pii` precisely because such spans exist.

**E1 — documentation contradiction. RESOLVED.** `ml/tune/MODEL_CARD_V2.md` stated
"The training data is not redistributed here". That is accurate for the
**Hugging Face model repo** and inaccurate for the **GitHub repo**, which tracks
the raw and derived corpora above. The claim has been scoped.

**The premise of this finding was itself wrong, and has been corrected.** This
audit previously recorded that "the repository records no source licence". Primary
source metadata contradicts that: the corpus derives from a Kropyvnytskyi City
Council open-data dataset published under **Creative Commons Attribution 4.0
International**, which permits redistribution with attribution. Two further
errors surfaced while establishing it: the recorded dataset id was wrong by one
character (it 404s), and the recorded filename `appeals_2026-08-01.csv` no longer
exists — the portal serves a single mutable `appeals.csv` with no checksum.

See `DATA_PROVENANCE.md` for the full evidence, the attribution owed, the
measured residual-PII figures, and `ml/tune/verify_release_provenance.py` for the
machine check that keeps these claims consistent.

Clean results: no credentials, `.env`, keys or tokens are tracked (all "token"
matches are NLP metrics such as `object_token_overlap`). Adapters (4.3 GB),
`ml/data/tune/eval/`, embeddings and v3.2 generated corpora are correctly ignored.

## F. What a fresh clone can actually run

| Check | Fresh clone | Why |
|---|---|---|
| `pytest ml/tests -q` | runs, degrades to skips | tests use `skipif` for absent artifacts |
| `ml.tune.artifact_sha --verify` | cannot fully verify | adapters gitignored |
| `ml.tune.verify_v3_2_report` | check 7.13 fails | needs adapters + v3.2 corpus |
| `ml.tune.verify_final_assessment` | reports uncovered claims | needs `ml/data/tune/eval/` (gitignored, produced by `run_eval.py`) |
| `ml.tune.verify_v2` | network-dependent | HF download |

`verify_final_assessment` is deliberately honest here: a missing `ml/data/tune/eval/`
is reported as uncovered claims, never as a pass. This must be documented, not
worked around, and the verifier must not be weakened.

## G. Vestigial scaffold

`src/index.ts` and `tests/index.test.ts` are both **0 bytes**, and `package.json`
declares Fastify/zod/vitest dependencies plus a `test` script that exits 1. This
implies a Node serving path that does not exist; the canonical runtime is Python +
MLX. Recorded as vestigial rather than removed, and no second inference framework
is introduced.

## H. Historical evidence retained (not defects)

`ml/data/tune/smoke/results/v1_finetuned_stash/` and the `previous-v3` arm rows
are historical experiment evidence and are kept. `ml/data/tune/decoding_probe/probe.log`
is a 2.7 KB progress log containing no complaint text.