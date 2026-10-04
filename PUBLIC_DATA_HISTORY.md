# Public Data History

The repository previously tracked source-derived complaint corpora in Git. As of the remediation at commit `a6862f6` (and consolidated in subsequent current-HEAD cleanup), those files have been removed from the **tracked set** using `git rm`. However:

- **Historical exposure remains.** The public Git history (including commits reachable from `origin/master`, notably `e217197` and its ancestors) still contains copies of the corpus files (`ml/data/*.jsonl`, `ml/data/tune/*/*.jsonl`, etc.). `git rm` does **not** remove those historical blobs.
- **Exact reconstruction is not currently possible.** The original historical snapshot `appeals_2026-08-01.csv` referenced in earlier provenance notes is unavailable, and no authoritative source checksum for that exact snapshot exists. This means the exact byte-for-byte corpus content from historical commits cannot be independently verified from a single canonical snapshot today.
- **Reproducibility is via the deterministic pipeline.** Authorised users can rebuild equivalent corpora from the official CC BY source (`data.kr-rada.gov.ua` dataset `9109c3a3-980f-45d0-a83e-b60f87d583d4`, resource `ddf7ffe9-d8e0-4646-93b2-b10d0b4bfe63`). The rebuilt corpora will be byte-deterministic given the same source snapshot and the pipeline in this repository (see `REPRODUCIBILITY.md`).
- **Model artifacts are unchanged.** The released model files, SHA-256 manifest (`ml/data/tune/artifact_sha256.json`), and the frozen evaluation harness remain independent of the corpus tracking status. HF provenance corrections were published separately; this repository remediation affects only what is tracked in Git going forward.
- **Historical PII mitigation is a separate action.** Complete removal of historical PII exposure requires a history rewrite (e.g. `git filter-repo`), which is **not performed** in this task. See `HISTORY_REWRITE_PLAN.md` for details and prerequisites (including explicit owner approval and coordination to update forks, clones, mirrors, and CI).

## Classification

- Current HEAD: source-derived/generated corpora are **not tracked**. 
- Public history: contains **source-derived/generated corpora** (historical). This cannot be claimed as "eradicated" without rewriting history. The repository's public posture after this remediation is: corpora are not distributed going forward; historical distribution exists and requires separate remediation if desired.
## Audit of the `*.json` blind spot (`da1c759`)

`ml/tune/audit_public_data.py` measures tracked `*.jsonl` files only. Tracked
`*.json` data files were therefore neither measured nor justified. A sweep of
every tracked `ml/data/**/*.json` file for complaint free-text found **21 files**
carrying it. Two conclusions:

**`ml/data/gold/gemini_batches/` — resolved, safe.** All 400 record texts are
byte-identical to records in `gold/annotation_set.jsonl`, which is already a
reviewed public fragment. It therefore adds **zero incremental source-data
exposure** and only records how those records were partitioned into batches. Its
measured indicators — 10 phone-like and 3 email-like rows — are the same rows
already counted in the annotation set, not additional ones. It is now listed in
`REVIEWED_PUBLIC_FRAGMENTS`, and `ml/tests/test_public_data_policy.py` fails if
any batch text ever appears that is absent from the annotation set.

**Everything else — unresolved, and publication-blocking.** The remaining
complaint-bearing `.json` files have no reviewed exception. The significant ones:

| Path | Finding |
|---|---|
| `ml/data/tune/multitopic/results/*.json` | 4 phone-like hits per arm (e.g. `099 110 34 11`, `099 203 73 07`) in `predictions[*].raw`, copied by the model from its input. `audit_public_data.classify()` returns `source_derived_corpus` — the **forbidden** class — for this prefix. |
| `ml/data/error_analysis.json` | 45 complaint texts, 37 of them present in no reviewed pool. 0 phone-like / 0 email-like indicators. |
| `ml/data/tune/smoke/smoke_cases.json` | 18 complaint texts, none in any reviewed pool; whether they are synthetic or source-derived is not recorded anywhere. |
| `ml/data/tune/multitopic/real_annotations.json` | 46 human reference labels containing street-level addresses (e.g. `вул.Попова, 18, корп.4, кв.43`). |
| `ml/data/tune/eval_v3/behaviour/*.json` | `predictions[*].raw` only, no indicators — belongs with generated model output but was unlisted. |

These are recorded in `PENDING_PUBLIC_CLASSIFICATION` in
`ml/tune/public_data_policy.py` so the open set is explicit, is measurable, and
cannot grow silently: a test fails if a new tracked complaint-bearing `.json`
appears with neither a reviewed exception nor a recorded pending entry. **They are
not approved for publication.** Each needs an explicit decision — redact,
reclassify as generated model output, or purge from history — and purging would
break the published multitopic metrics that are computed from them.

The history rewrite (see `HISTORY_REWRITE_VALIDATION.md`) was therefore **not
published**: the remote still points at the pre-rewrite tip, and these files need
resolving before a clean public boundary can be claimed.
