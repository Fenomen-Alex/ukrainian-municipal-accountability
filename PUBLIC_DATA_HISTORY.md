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