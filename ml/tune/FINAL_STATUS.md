# Final status

**MODEL DEVELOPMENT FROZEN.** No further training is planned, authorised or in
progress. See `ml/tune/FINAL_MODEL_ASSESSMENT.md` §8 — Option A is the decision.

**CANONICAL MODEL: v2** (attempt10, fused).

| | |
|---|---|
| Public artifact | `Fenomen-Alex/ukrainian-municipal-accountability-qwen3-8b` |
| Local artifact | `ml/data/tune/adapters/qwen3-8b-lora-v2-attempt10-fused/` (gitignored) |
| Base | `mlx-community/Qwen3-8B-4bit` |
| Architecture | `Qwen3ForCausalLM` |
| Format | MLX / safetensors — **not** GGUF |
| Canonical runtime | MLX via `mlx_lm` (Apple Silicon) |
| Compatible runtimes | LM Studio, any OpenAI-compatible endpoint |
| Serving contract | [V2_SERVING.md](V2_SERVING.md) |

## Release candidates

**There are no release candidates beyond canonical v2.** Every other arm in this
repository — v1, v3 control, previous-v3, corrected-v3, v3.2 — is a completed
historical experiment retained for the record. None is production-ready, and the
v3 lineage in particular is **not** "nearly ready": it fixes boilerplate but
fails the topic-count and two-topic gates outright and regresses schema validity
and action detection relative to v2. See
[MODEL_EXPERIMENTS.md](MODEL_EXPERIMENTS.md).

## The quote fix (shipped)

A root-cause fix is in the canonical serving path. Redaction-created empty
quotes (`""`) and mid-word ASCII quotes made the model emit broken JSON.

* Mid-word quotes → apostrophe: `під"їзду` → `під’їзду`.
* Delimited quotes → typographic pair: `"Копілка"` → `“Копілка”`.
* Applied to the **user turn only**; the system prompt is byte-exact.
* Complaints containing no ASCII quotes are byte-identical to the pre-fix path.

Repaired 5/5 affected cases on corrected-v3 and v3.2 in real generation; v2 was
never affected and is unchanged by it. Full account, evidence and the honest
nuance about how the defect reached eval:
[QUOTE_ARTEFACT.md](QUOTE_ARTEFACT.md).

## Known limitations of canonical v2

These are **not fixed** and are not planned to be. Any integrator should
evaluate them against their use case.

* **Multitopic output is inline.** Several problems in one complaint come back
  as several entries in a single `topics` array. There is no per-topic split,
  routing or partial-output path.
* **Action regime is imperfect.** When a complaint names no remedy,
  `requested_action` should be empty; when it does, the model may
  under- or over-reach. v2's frozen action presence match (0.9909) and action
  copy (0.4407) are strong but not perfect — treat `requested_action` as a
  suggestion, not a commitment.
* **`ev3-083` is a known decoding defect.** It is present in v2 and is not
  repaired. A completed decoding probe found no configuration that fixes it
  without altering topic behaviour, so there is no sanctioned workaround.
* **Coarse taxonomy.** Thirteen fixed `domain` values; anything outside them
  collapses to `other`, and finer distinctions are lost.
* **Training-label quality — `_PERSON_NAME` over-matching.** The weak-label
  redaction heuristic used to build the corpora matches person names too
  broadly, so legitimate names are sometimes stripped from training text. This
  is a defect in the *data*, inherited by every arm including v2. It is
  documented rather than hidden; fixing it would require rebuilding the corpora
  and retraining, which is out of scope under the freeze.
* **Cross-runtime variance.** Output is self-consistent within a runtime at
  `temperature=0.0` but **not** bit-identical across runtimes; `object` fields
  in particular vary. Do not assert on them.

## Data and privacy

The source-derived and generated training corpora are **not tracked** in this
GitHub repository; they are reproducible from the official source via the
deterministic pipeline ([REPRODUCIBILITY.md](../../REPRODUCIBILITY.md)). The
former tracked source corpora were **not** PII-free: raw municipal complaint text
contains real-world phone numbers and email-like strings, and the heuristic
redaction is incomplete (210 phone-like and 114 email-like rows in the last
tracked audit). Those files are no longer distributed, but the blobs remain in
public Git history; see
[PUBLIC_DATA_HISTORY.md](../../PUBLIC_DATA_HISTORY.md) and
[HISTORY_REWRITE_PLAN.md](../../HISTORY_REWRITE_PLAN.md).

The source is openly licensed: a Kropyvnytskyi City Council open-data dataset
under **Creative Commons Attribution 4.0 International**, which permits
redistribution and re-use **on condition the creator is credited**. That
attribution, the exact publisher wording and the measured figures are in
[DATA_PROVENANCE.md](../../DATA_PROVENANCE.md). The Apache-2.0 licence on the
model weights does **not** cover the corpus and does not discharge that
attribution condition. The Hugging Face artifact ships weights only.

## Where to look next

| Question | Document |
|---|---|
| How do I run it? | [../README.md](../../README.md) |
| Exact prompt, decoding, output contract | [V2_SERVING.md](V2_SERVING.md) |
| Why v2, and what was tried | [MODEL_EXPERIMENTS.md](MODEL_EXPERIMENTS.md) |
| Full evidence and reasoning | [FINAL_MODEL_ASSESSMENT.md](FINAL_MODEL_ASSESSMENT.md) |
| What was fixed, and what wasn't | [QUOTE_ARTEFACT.md](QUOTE_ARTEFACT.md) |
| Is this ready to ship? | [../FINAL_RELEASE_READINESS.md](../../FINAL_RELEASE_READINESS.md) |