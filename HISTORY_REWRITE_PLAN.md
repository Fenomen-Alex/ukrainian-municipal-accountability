# History Rewrite Plan (EXECUTED — rewrite published to remote)

**Status (current):** The rewrite described below was executed and validated, and the rewritten history has been published to `origin/master`. All 12 paths are absent from every branch/tag reachable from origin; only local safety tags (`pre-rewrite-e393337`, `pre-rewrite-redacted-e393337`) and transient GitHub orphaned-object retention remain. No further execution is required.

Original plan (retained for audit):

## Affected paths (historical)

The following files have historically contained source-derived or generated complaint corpora and must be purged from history:

- `ml/data/test.jsonl`
- `ml/data/train.jsonl`
- `ml/data/validation.jsonl`
- `ml/data/tune/multitopic/eval.jsonl`
- `ml/data/tune/multitopic/train.jsonl`
- `ml/data/tune/v2/test.jsonl`
- `ml/data/tune/v2/train.jsonl`
- `ml/data/tune/v2/valid.jsonl`
- `ml/data/tune/v2/validation.jsonl`
- `ml/data/tune/v3/treatment/test.jsonl`
- `ml/data/tune/v3/treatment/train.jsonl`
- `ml/data/tune/v3/treatment/validation.jsonl`

Note: `ml/data/tune/v2/valid.jsonl` is a symlink in the working tree but is tracked as a blob in history; it must be removed.

## Protected artifacts (must remain intact in rewritten history)

The following must not be modified or corrupted by history rewrite:
- Deterministic build code: `ml/tune/build_*.py`, `ml/cleaner.py`, schema under `ml/schema/`
- Manifests/hashes: `ml/data/tune/artifact_sha256.json`, `ml/data/tune/public_data_manifest.json`, `ml/data/tune/MULTITOPIC_STATS.json`
- Model artifacts (if any large model files are tracked in history): verify with `git rev-list --objects --all | grep -E '\.(bin|safetensors|gguf|onnx)$'` before rewrite; current HEAD has no model weight files tracked (weights live in release artifacts/MLX). Confirm against `ml/tune/artifact_sha256.json`.
- Evaluation fixtures explicitly kept: `ml/data/gold/annotation_set.jsonl`, `ml/data/tune/eval_v3/cases.jsonl` (reviewed public fragments) — confirm their historical content is acceptable to keep; if they contain sensitive data, include them in purge.
- Documentation, provenance, release docs.

## Recommended tool

Use [`git filter-repo`](https://github.com/newren/git-filter-repo) (not `git filter-branch`, not BFG alone without verification). BFG can also work but requires care with paths and symlinks.

## High-level procedure

1. **Pre-requisites & approval**
   - Obtain explicit owner approval to rewrite history.
   - Notify downstream consumers (forks, clones, CI, release mirrors) in advance.
   - Create a full backup of the repository (including all refs/tags).
   - Confirm no uncommitted work; ensure current HEAD is clean as desired.

2. **Inventory & dry-run**
   - `git rev-list --count --all` to count commits.
   - `git rev-list --objects --all | grep -E 'ml/data/(test|train|validation)\.jsonl|ml/data/tune/.*\.jsonl' | head -20` to see which blobs/commits affected.
   - Dry-run with `git filter-repo --path ml/data/test.jsonl --path ml/data/train.jsonl --path ml/data/validation.jsonl --path ml/data/tune --invert-paths`? Or explicitly remove the listed paths. Alternatively `--path` with specific files and `--invert-paths` is not correct for removal; to remove, specify paths to drop: use `--path ml/data/test.jsonl --path ml/data/train.jsonl ... --force` to remove those paths entirely from history.
   - Verify protected paths remain: ensure no accidental removal of reviewed fixtures unless intended.

3. **Rewrite**
   - `git filter-repo --path ml/data/test.jsonl --path ml/data/train.jsonl --path ml/data/validation.jsonl --path ml/data/tune/multitopic/eval.jsonl --path ml/data/tune/multitopic/train.jsonl --path ml/data/tune/v2/test.jsonl --path ml/data/tune/v2/train.jsonl --path ml/data/tune/v2/valid.jsonl --path ml/data/tune/v2/validation.jsonl --path ml/data/tune/v3/treatment/test.jsonl --path ml/data/tune/v3/treatment/train.jsonl --path ml/data/tune/v3/treatment/validation.jsonl --invert-paths --force` (if using `--invert-paths` with `--path` to keep only those? Or easier: remove specific paths by excluding them - standard: `git filter-repo --path <files-to-remove> --invert-paths` keeps everything else; no, `--path` selects paths to process; to remove them completely, use `--path` and also `--replace-refs delete-no-add`? Or simply `git filter-repo --path ml/data/test.jsonl --force` will keep only that path? No. Correct form: to **remove** specific paths from all history, use `git filter-repo --path-regex '(ml/data/(test|train|validation)\.jsonl|ml/data/tune/(multitopic|v2|v3/treatment)/(eval|train|test|valid|validation)\.jsonl)$' --invert-paths`? Wait: `--invert-paths` means process paths that do **not** match; combined with `--path-regex` selecting those to remove means we drop them. Yes.
   - Example: `git filter-repo --path-regex '^(ml/data/(test|train|validation)\.jsonl|ml/data/tune/multitopic/(eval|train)\.jsonl|ml/data/tune/v2/(test|train|valid|validation)\.jsonl|ml/data/tune/v3/treatment/(test|train|validation)\.jsonl)$' --invert-paths --force`

4. **Post-rewrite validation**
   - Check no removed paths remain: `git log --all --name-only --oneline --max-count=1 -- ml/data/train.jsonl` should return nothing.
   - Verify protected paths still exist in history where expected: spot-check `git log --oneline -1 -- ml/data/gold/annotation_set.jsonl`.
   - Run `git fsck --full --no-reflogs` to verify integrity.
   - Recompute size: `git count-objects -vH`.
   - Regenerate audit/manifest if needed (they reflect current state; manifest is advisory).
   - Run full test suite on rewritten history tip: `pytest ml/tests` (corpus-dependent tests skip; non-dependent tests must pass).
   - Verify model artifact hashes still match: `ml/tune/verify_release_provenance.py` and `ml/tune/artifact_sha256.json`.

5. **Force-push (requires coordination)**
   - Force-push all refs: `git push origin --force --all` and `git push origin --force --tags`. 
   - All consumers must re-clone or run `git fetch --all --prune && git reset --hard origin/master` (data loss risk if local branches contain work based on old history).

## Safety notes

- Never rewrite history without backups.
- Coordinate with HF, mirrors, CI caches, release downloads.
- This is irreversible in the public sense once force-pushed; ensure all stakeholders agree.
- The current task **does not** perform these steps. This document exists solely to record the full remediation path and prerequisites.