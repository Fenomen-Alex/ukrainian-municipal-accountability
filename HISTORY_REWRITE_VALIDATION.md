# History rewrite validation

Record of the local execution and validation of the history rewrite planned in
[HISTORY_REWRITE_PLAN.md](HISTORY_REWRITE_PLAN.md). The plan was deferred
pending owner approval; approval to execute the rewrite **locally** was given,
with an explicit hard stop before any remote modification.

> ## LOCAL HISTORY REWRITE VALIDATED — REMOTE NOT YET UPDATED

| | |
|---|---|
| Pre-rewrite public tip | `e3933372e173c36b5fde26349b5bece8f910c665` |
| Rewritten local tip | `682b7c318028b0e8c121a7cfddddff92929faec3` |
| Remediation commit on top | `da1c759ccf8fc4de9a5895003ae9bc8de2b01dba` |
| `origin/master` | `e393337` — **unchanged** |
| Safety reference | local tag `pre-rewrite-e393337` → `e393337` (retained, not pushed) |
| Tool | `git filter-repo` a40bce548d2c |
| Method | exact 12 paths, `--invert-paths`, no regex, no wildcards |

## What was removed

Exactly the 12 paths documented in the plan, and nothing else:

1. `ml/data/test.jsonl`
2. `ml/data/train.jsonl`
3. `ml/data/validation.jsonl`
4. `ml/data/tune/multitopic/eval.jsonl`
5. `ml/data/tune/multitopic/train.jsonl`
6. `ml/data/tune/v2/test.jsonl`
7. `ml/data/tune/v2/train.jsonl`
8. `ml/data/tune/v2/valid.jsonl`  *(tracked as a blob, symlink in the worktree)*
9. `ml/data/tune/v2/validation.jsonl`
10. `ml/data/tune/v3/treatment/test.jsonl`
11. `ml/data/tune/v3/treatment/train.jsonl`
12. `ml/data/tune/v3/treatment/validation.jsonl`

## Method

The rewrite was performed in an isolated `--mirror` clone under `/tmp` and
imported back only after passing every check below. Nothing was downloaded, no
model was copied or fused, `.lmstudio` and HF caches were not duplicated, and the
temporary clone and mirror were deleted afterwards.

```bash
git clone --mirror --no-local <repo> /tmp/hrw.git
git -C /tmp/hrw.git filter-repo --force \
  --path ml/data/test.jsonl --path ml/data/train.jsonl \
  --path ml/data/validation.jsonl \
  --path ml/data/tune/multitopic/eval.jsonl \
  --path ml/data/tune/multitopic/train.jsonl \
  --path ml/data/tune/v2/test.jsonl --path ml/data/tune/v2/train.jsonl \
  --path ml/data/tune/v2/valid.jsonl --path ml/data/tune/v2/validation.jsonl \
  --path ml/data/tune/v3/treatment/test.jsonl \
  --path ml/data/tune/v3/treatment/train.jsonl \
  --path ml/data/tune/v3/treatment/validation.jsonl \
  --invert-paths --replace-refs delete-no-add
```

## Verification performed

### History integrity

| Check | Result |
|---|---|
| All 12 forbidden paths reachable from `master` | **absent** |
| Forbidden paths reachable from `origin/master` (pre-rewrite) | 12 — expected, remote not yet updated |
| Forbidden paths reachable from `--all` | 12 — held only by the local safety tag `pre-rewrite-e393337` |
| Commit count before / after | 38 / 38 — no commits dropped |
| Author, committer, both dates, subject of all 38 commits | **byte-identical** to pre-rewrite |
| `git fsck --full --no-reflogs` | **clean**, 0 errors, 0 dangling |
| `refs/replace/*` | none |
| Current HEAD tree | `4710b43ced40e19e1da38b4c06fe81c40528353c` — **identical** pre- and post-rewrite |
| Working tree after import | clean |

### No collateral drift

The decisive check. For every commit in both histories, every `(path, blob)`
pair was enumerated and compared as sets:

| Check | Result |
|---|---|
| Paths present before but missing after | **exactly the 12** forbidden paths — no unexpected loss |
| Paths present after but not before | **0** |
| Paths whose blob-SHA set changed | **0** |
| Build code, `ml/cleaner.py`, `ml/schema/`, manifests, `artifact_sha256.json`, `public_data_manifest.json`, `MULTITOPIC_STATS.json` | unchanged |
| Benchmarks (`eval_v3/cases.jsonl`, `gold/annotation_set.jsonl`), evaluators, verifiers | unchanged |
| Serving code (`serve_v2.py`), model metadata, model card, all documentation | unchanged |
| Evaluators (`cleaner.py`, `schema/`, `eval_v3`, scoring) | unchanged — no evaluator semantics were modified |
| Tests | **modified in `da1c759`, see 'Test changes' below** — guards added, no assertion weakened |

Because commit and tree SHAs necessarily change when paths are removed, content
integrity was proven at blob level rather than by SHA comparison of the trees.

### Source-derived material no longer reachable

Scan of every reachable `.jsonl`/`.json` blob using the project's own
`audit_public_data` heuristics (same `PHONE`/`EMAIL` patterns):

| | phone-like | email-like |
|---|---:|---:|
| Before (`e393337`) | 621 | 286 |
| After (`682b7c3`) | 188 | 12 |

All residual hits are in deliberately tracked artifacts and are pattern
false positives or known reviewed exceptions — `artifact_sha256.json` (hex
substring matches), `public_data_manifest.json` (records the historical
indicators as data), `eval_v3/results/*.json` (metric values), and
`gold/annotation_set.jsonl` (reviewed public fragment). No source-derived
complaint corpus is reachable from `master`.

### Model and reference SHA

The canonical v2 fused model was verified directly against the recorded manifest
(`qwen3-8b-lora-v2-fused`, which resolves to `qwen3-8b-lora-v2-attempt10-fused`):

```
MATCH  chat_template.jinja            4,116 bytes
MATCH  config.json                      939 bytes
MATCH  model.safetensors   4,607,835,164 bytes
MATCH  model.safetensors.index.json  64,105 bytes
MATCH  tokenizer.json              11,422,650 bytes
MATCH  tokenizer_config.json            413 bytes

tree_sha256 recorded   : 5352b260df5925a83a9532aa9cdcaa0b0c5a97ff7028780bc4f3b1008b6dc4f5
tree_sha256 recomputed : 5352b260df5925a83a9532aa9cdcaa0b0c5a97ff7028780bc4f3b1008b6dc4f5
```

**Canonical v2 matches the manifest — weights unchanged.** No weight was
modified, re-fused, downloaded, or published. Model development remains frozen.

### Test and verifier results

Run in the working repository at the rewritten tip:

| Command | Result |
|---|---|
| `pytest ml/tests/test_quote_normalization.py test_serve_v2.py test_release_smoke.py -q` (targeted) | **108 passed, 9 skipped** |
| `pytest ml/tests -q` | **362 passed, 153 skipped, 4 subtests, 0 failed** (at `da1c759`) |
| `ml.tune.audit_public_data --check` | **OK** (13 jsonl, 2.4 MB, 2991 rows, 11 phone-like, 3 email-like) — matches `DATA_PROVENANCE.md` exactly |
| `ml.tune.verify_release_provenance` | **OK (19/19)** |
| `ml.tune.verify_final_assessment` | **OK (380/380)** |
| `ml.tune.verify_repro` (MLX) | **skipped as designed** — corpora not distributed (path B); explicit guidance, not a false pass |
| `ml.tune.verify_v3_2_report` | **exit 1, cannot run** — the multi-GB fused v3.2 adapter is absent from this checkout; it declines to report a pass rather than faking one |
| `ml.tune.artifact_sha --verify` | **canonical model `qwen3-8b-lora-v2-fused` unchanged (6 files)**; 6 further arms **NOT VERIFIED** (gitignored, absent). `--require-all` still fails on absence |
| `pytest ml/tests/test_release_smoke.py -q` (real model, MLX) | **65 passed (48.4 s)** — matches the documented baseline of 65 |
| `pytest ml/tests -q` in a detached worktree (public shape) | **349 passed, 166 skipped, 0 failed** |

#### Test changes made in `da1c759`, and what they do not cover

Every one of the previous failures was caused by an artefact that is gitignored
and therefore absent from a public checkout — never by the rewrite. That is now
stated as an explicit skip rather than an error:

| Cause | Handling |
|---|---|
| `ml/data/tune/adapters/qwen3-8b-lora-v3-2-fused/` absent (multi-GB private build) | `test_verify_v3_2_report.py` skips all 10 tests with a stated reason |
| `ml/data/tune/eval/` absent (gitignored frozen per-case evaluator output) | 5 functions in `test_final_assessment.py` skip with a stated reason; the other 27 still run |

No assertion was weakened, removed, or made conditional on the outcome. Where an
artefact is present the tests run in full: the 12 final-assessment tests execute in
the working tree (`39 passed, 4 skipped`) and skip in a clean checkout
(`27 passed, 16 skipped`).

**Not covered, stated plainly:** the v3.2 fused adapter is absent from *both* the
working tree and a clean checkout, so those 10 v3.2 verifier tests have not been
observed to pass in this session. They skip everywhere here and will run only where
the adapter is rebuilt. The v3.2 report claim is therefore **unverified** in this
record, and is backed only by the tracked `artifact_sha256.json` and
`FINAL_MODEL_ASSESSMENT.md` (380/380).

### Fresh-clone validation

A `--no-local` clone of the rewritten repository (10 MB, zero garbage; a plain
local clone was rejected because it copied a 2.5 GiB orphaned git temp object
from the source objects directory):

| Check | Result |
|---|---|
| Forbidden paths reachable from clone `master` / `origin/master` | **absent** |
| `git fsck --full --no-reflogs` | clean |
| HEAD tree | `4710b43c…` — identical to pre-rewrite |
| `audit_public_data --check` | **OK** (13 jsonl, 2.4 MB, 11 phone-like, 3 email-like) |
| `verify_release_provenance` | **OK (19/19)** |
| `pytest ml/tests -q` (pre-`da1c759`) | 22 failed, 331 passed, 144 skipped |
| `pytest ml/tests -q` (at `da1c759`) | **0 failed, 349 passed, 166 skipped** |

All 22 fresh-clone failures trace to a single missing **gitignored** artefact,
`ml/data/tune/eval/lora.json` (`.gitignore:9`). It is untracked and **never
committed in any history**, pre- or post-rewrite, so no rewrite can affect it.
This is the documented behaviour in `FINAL_RELEASE_READINESS.md` §5: a fresh
clone reports *uncovered claims*, never a false pass.

**Proof:** the identical 22 failures occur at `e393337` and at `682b7c3` in the
same clone (`22 failed, 331 passed, 144 skipped` both times). The failure count is
independent of which commit is checked out.

The temporary clone was deleted immediately after validation.

## LM Studio link repair

The LM Studio folder `~/.lmstudio/models/mlx-community/qwen3-8b-municipal-finetune`
held seven per-file symlinks into the deleted obsolete directory
`ml/data/tune/adapters/qwen3-8b-lora-v2-fused/`. All seven were broken. They were
repointed to the canonical
`ml/data/tune/adapters/qwen3-8b-lora-v2-attempt10-fused/` with `ln -sfn`.

* No model file was copied; the directory still occupies **0 bytes**.
* No second model directory was created.
* All seven links resolve, including the 4.3 GB `model.safetensors`.
* The `qwen3-8b-municipal-finetune` label is intentional and documented in
  `V2_SERVING.md` — it is an LM Studio folder label, not a model path, and was
  preserved.

## Remaining risks

1. **The remote is unchanged.** `origin/master` is still `e393337` and the
   corpora remain publicly downloadable at that ref. This validation is only
   meaningful once the rewritten history is published. **Nothing has been pushed.**
2. **The local safety tag retains the old history.** `pre-rewrite-e393337` keeps
   all 12 paths reachable via `--all`. It must **not** be pushed, and it is the
   only local reason the pre-rewrite corpora are still on this machine. Delete it
   only after the remote rewrite is confirmed published.
3. **The canonical model is now verified by the release tool itself.** The
   `adapters/qwen3-8b-lora-v2-fused` symlink had been lost with the earlier local
   cleanup, so `artifact_sha --verify` never actually hashed the shipped model.
   Restoring that symlink (a symlink, not a copy) re-enables it:
   `qwen3-8b-lora-v2-fused unchanged (6 files)`. The remaining 6 arms are absent
   and are now reported as **NOT VERIFIED** rather than as corruption; use
   `--require-all` for the strict behaviour.
4. **`verify_v3_2_report` still cannot run** without the gitignored v3.2 fused
   adapter, and its 10 tests now skip with that reason. This is unchanged in
   substance — only the reporting improved. See risk 9.
5. **`ml/data/gold/gemini_batches/` is resolved.** It is now a reviewed exception
   in `REVIEWED_PUBLIC_FRAGMENTS`: all 400 texts are byte-identical to
   `gold/annotation_set.jsonl`, so it adds zero incremental source-data exposure,
   and its 10 phone-like / 3 email-like rows are exactly the rows already counted
   there. A test now fails if that ever stops being true.
6. **The HF model card still carries one stale sentence** claiming the source
   project tracks source-derived corpora. Unchanged here; still queued for the
   next authorised HF metadata edit. No HF repository was modified.
7. **A 2.5 GiB orphaned git temp object** remains at
   `.git/objects/c3/tmp_obj_5lm9yF` (dated 4 Oct, predating this work). Git
   flags it as garbage and it is referenced by nothing. It was **not** deleted —
   it is not this task's artefact and there was no disk pressure (171 GiB free).
   It is also why a plain `git clone` of this repository produces a 2.7 GB clone;
   use `--no-local`.
8. **Verification claims that need gitignored artefacts remain uncovered** on any
   clone. This is by design under path B, and is recorded rather than papered
   over.
9. **PUBLICATION IS BLOCKED — unclassified complaint-bearing data in tracked
   `*.json` files.** Auditing the `audit_public_data.py` `*.jsonl`-only blind spot
   found 21 tracked `.json` files carrying complaint free-text that no reviewed
   exception covers. The most serious is `ml/data/tune/multitopic/results/`: it
   holds **4 phone-like hits per arm** (e.g. `099 110 34 11`, `099 203 73 07`) in
   `predictions[*].raw`, where the model copied a real contact number from its
   input. `audit_public_data.classify()` returns **`source_derived_corpus`** —
   the *forbidden* class — for this prefix, so as a `.jsonl` it would already be
   rejected by the audit; it escapes only because it is `.json`.
   `ml/data/error_analysis.json` (37 of 45 texts in no reviewed pool),
   `ml/data/tune/smoke/smoke_cases.json` (18 texts, provenance unrecorded) and
   `ml/data/tune/multitopic/real_annotations.json` (46 reference labels including
   street-level addresses) are in the same unresolved state.
   All of these are now recorded in `PENDING_PUBLIC_CLASSIFICATION` so the set is
   explicit and cannot grow silently, and a test fails if a new unclassified
   complaint-bearing `.json` appears. **They are not approved for publication.**
   Each needs an owner decision — redact, reclassify as generated model output, or
   purge from history — and purging would break the published multitopic metrics
   that depend on them. The rewritten history has deliberately **not** been
   pushed.

## Reproducing this validation

```bash
# the 12 forbidden paths, from the plan
ml/data/test.jsonl ml/data/train.jsonl ml/data/validation.jsonl
ml/data/tune/multitopic/{eval,train}.jsonl
ml/data/tune/v2/{test,train,valid,validation}.jsonl
ml/data/tune/v3/treatment/{test,train,validation}.jsonl

# absent from the rewritten public history
git rev-list --objects master | grep -E ' ml/data/(test|train|validation)\.jsonl$'

# current HEAD intact
git rev-parse HEAD^{tree}   # 4710b43ced40e19e1da38b4c06fe81c40528353c

# boundary policy
python -m ml.tune.audit_public_data --check
```

## Boundary

Not done, and deliberately so: no force-push, no change to `origin/master`, no
Hugging Face modification, no tag or GitHub release created, no model weight
touched, no training or evaluation job run, and the pre-rewrite safety reference
left in place.