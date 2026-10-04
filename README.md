# Ukrainian municipal accountability — structured extraction

Turns a Ukrainian citizen's municipal complaint into a strict JSON structure:
the problems described, the municipal domain each belongs to, the object it
concerns, and the remedy requested.

**Status: model development frozen.** One canonical model is designated for use:
**v2**, published as `Fenomen-Alex/ukrainian-municipal-accountability-qwen3-8b`
(MLX 4-bit, based on `mlx-community/Qwen3-8B-4bit`). Later training arms exist
in this repository as *historical experiments only* — see
[ml/tune/FINAL_STATUS.md](ml/tune/FINAL_STATUS.md). There is no in-progress
release candidate.

| | |
|---|---|
| Canonical model | v2 (attempt10, fused) |
| Runtime | MLX via `mlx_lm` on Apple Silicon (canonical), LM Studio / OpenAI-compatible HTTP (compatible) |
| Decoding | `temperature=0.0`, `max_tokens=800`, thinking disabled |
| Output | JSON only, no prose, no markdown fences |
| Serving contract | [ml/tune/V2_SERVING.md](ml/tune/V2_SERVING.md) |

## Quickstart

**1. Environment.** Python 3.12, Apple Silicon:

```bash
python3.12 -m venv .venv-mlx
.venv-mlx/bin/pip install -r ml/requirements-serve.txt
```

**2. Get the model.** Either the public artifact or a local fused build.

```bash
# public artifact
.venv-mlx/bin/hf download Fenomen-Alex/ukrainian-municipal-accountability-qwen3-8b \
  --local-dir ml/data/tune/adapters/qwen3-8b-lora-v2-attempt10-fused
```

or, if you built it locally, point `--model` at the existing fused directory
`ml/data/tune/adapters/qwen3-8b-lora-v2-attempt10-fused/` (gitignored; adapters
are not distributed with this repository).

**3. Run it.**

```bash
# one complaint, end to end
.venv-mlx/bin/python -m ml.tune.serve_v2 \
  --model ml/data/tune/adapters/qwen3-8b-lora-v2-attempt10-fused \
  --text "На вулиці Шевченка вже тиждень не горять ліхтарі."

# print the exact request body without loading the model (no download needed)
.venv/bin/python -m ml.tune.serve_v2 --request-only \
  --text 'Біля маг. "Копілка" три тижні тече каналізація.'
```

Step 3's `--request-only` form is the fastest way to see the prompt contract:
it emits the exact `messages` array and generation settings to send to any
OpenAI-compatible endpoint, and needs no weights at all.

## The contract in one rule

Build the prompt exactly, generate greedily, and **surface** malformed output
rather than repairing it. Specifics and the reasoning behind each of them are in
[ml/tune/V2_SERVING.md](ml/tune/V2_SERVING.md). The short version:

* The system message is byte-exact. Do not retype it, reformat it, or let a
  client substitute its own.
* Quote normalization runs **on the user turn only**. `"Копілка"` becomes
  `“Копілка”`, and a mid-word `під"їзду` becomes `під’їзду`. This is a
  root-cause fix for redaction-created empty quotes that made the model emit
  broken JSON; see [ml/tune/QUOTE_ARTEFACT.md](ml/tune/QUOTE_ARTEFACT.md).
* `temperature=0.0`, `max_tokens=800`, thinking off.
* Parse strictly. Leading/trailing whitespace is tolerated; nothing else is.
* Do not regex-repair output. A `\{.*\}` fallback hides exactly the prompt
  failures you need to see.

## Verification

```bash
# full suite
.venv/bin/python -m pytest ml/tests -q

# release smoke suite: eight complaint shapes against the contract,
# then against the real model when MLX and weights are present
.venv/bin/python -m pytest ml/tests/test_release_smoke.py -q
.venv-mlx/bin/python -m pytest ml/tests/test_release_smoke.py -q

# claim-level verifiers
.venv/bin/python -m ml.tune.verify_final_assessment
.venv-mlx/bin/python -m ml.tune.verify_repro
```

The model-free layer of the smoke suite needs no weights and no MLX; the
model-gated layer runs only where both are available. Evidence, environment
limitations and per-command results are recorded in
[FINAL_RELEASE_READINESS.md](FINAL_RELEASE_READINESS.md).

## Repository layout

| Path | What it is |
|---|---|
| `ml/tune/` | Training, serving and verification code; the canonical contract |
| `ml/tests/` | Test suite, including the release smoke suite |
| `ml/data/tune/` | Corpora, adapters (gitignored), and metadata |
| `ml/reports/` | Generated evaluation reports |
| `src/`, `tests/` | **Vestigial Node scaffold.** Not a runtime, not used by serving. |

Adapters, generated eval outputs and model weights are gitignored and are not
part of this repository; `ml/data/tune/artifact_sha256.json` records their
expected hashes where applicable.

## Known limitations

These are properties of the canonical v2 model and are not fixed. They are
listed so that integrators can decide whether v2 is fit for their use.

* **Multitopic is inline.** Multiple problems in one complaint are returned in
  one response; there is no per-topic split or routing stage.
* **Action regime.** When a complaint names no remedy, `requested_action` is
  empty; when it does, the model may still under- or over-reach, so treat the
  field as a suggestion rather than a commitment.
* **`ev3-083` is a known decoding defect** carried into every later arm and not
  repaired.
* **Taxonomy is coarse.** Thirteen fixed domains; anything outside them
  collapses to `other`, and sub-domain distinctions (e.g. which specific
  benefit, or which municipal body) are lost.
* **Training-label quality.** The weak-label redaction heuristic over-matches
  person names, so names are sometimes stripped where they were legitimate
  identifiers. This is a data artefact, documented rather than hidden.

## Data and privacy

The source-derived and generated training corpora are **not tracked in this
GitHub repository**. They are reproducible from the official open-data source via
the deterministic pipeline (see [REPRODUCIBILITY.md](REPRODUCIBILITY.md)). What
is tracked is build code, schema, manifests/hashes, documentation, and a small
reviewed set of benchmark/provenance artifacts (~2.4 MB of `.jsonl`).

**This is a current-HEAD policy, not a history purge.** The former corpora are
still present in public Git history (`e217197`); removing them from history
requires a separately-approved rewrite. See
[PUBLIC_DATA_HISTORY.md](PUBLIC_DATA_HISTORY.md) and
[HISTORY_REWRITE_PLAN.md](HISTORY_REWRITE_PLAN.md).

**The source is openly licensed.** The corpora derive from a published
Kropyvnytskyi City Council open-data dataset under **Creative Commons
Attribution 4.0 International**, which permits redistribution and re-use with
attribution. That attribution is recorded in
[DATA_PROVENANCE.md](DATA_PROVENANCE.md), which is the authoritative provenance
record. **The Apache-2.0 licence on the model weights does not cover the data.**

Two facts worth knowing before you rebuild or use the corpora:

* **The rebuilt corpora are not PII-free.** The heuristic redaction is
  pattern-based and incomplete. The former tracked source corpora carried 210
  phone-like and 118 email-like rows; those files are no longer distributed. Treat
  any locally rebuilt corpus as sensitive rather than as reference data.
* **The source is not versioned.** It is a single mutable `appeals.csv` with no
  published checksum, so the historical corpora cannot be re-derived byte-for-byte
  or proven against the source later.

No credentials, keys or `.env` files are tracked, and no model weights are in
Git.