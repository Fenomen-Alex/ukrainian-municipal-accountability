# Manual Gemini annotation step

The 400-record gold set is prepared as 10 deterministic batches in
`ml/data/gold/gemini_batches/`. Each `batch-XXX.json` already contains:

- `batch_id` — must be echoed back unchanged (Format A) or matched by the
  output filename (Format B).
- `instructions` — the full annotation prompt to send to Gemini.
- `records` — `[{id, text, source_kind}]`. Do NOT copy `source_kind` into the
  annotation; classify from the text only.
- `schema_version` — must match `ml/data/gold/gemini_batches/manifest.json`.

Also reference `ml/data/gold/annotation_guide.md` for labelling rules and
`ml/data/gold/annotation_schema.json` for the exact JSON Shape.

## Per-batch procedure (manual, repeated for batch-001..batch-010)

1. Open `ml/data/gold/gemini_batches/batch-XXX.json` and send its entire
   content (or just the `records` array plus the `instructions` field) to
   Gemini.
2. Ask Gemini to return its annotations as **raw JSON only** (no markdown
   fences, no commentary), in either:
   - Format A: `{"batch_id": "...", "annotations": [{"id": ..., "annotation": {...}}, ...]}`
   - Format B: a plain JSON array `[{"id": ..., "annotation": {...}}, ...]`
3. Save the response verbatim to
   `ml/data/gold/gemini_outputs/batch-XXX.json`.

## Validate a single batch before merging

    python3 -m ml.gold.validate_gemini --input ml/data/gold/gemini_outputs/batch-XXX.json --batch batch-XXX --data-dir ml/data

It exits 0 on success and 1 on any error (missing/unknown/duplicate ids,
schema violations, changed text, wrong batch), printing
`category / record_id / path / message` for each problem. Fix and re-save the
file until this says `Validation OK`.

## Validate all 10 batches and merge into proposals

    python3 -m ml.gold.validate_gemini --input-dir ml/data/gold/gemini_outputs --merge-output ml/data/gold/gemini_proposals.jsonl --data-dir ml/data

This requires all 400 ids, no cross-batch duplicates, and produces
`ml/data/gold/gemini_proposals.jsonl` with status
`gemini_proposed_unreviewed`. It never modifies the gold set.

## Build the review report

    python3 -m ml.gold.review_report --data-dir ml/data

Writes `ml/data/gold/gemini_review_report.md` (totals, domain frequency,
empty-field frequency, large attribute sets, PII-like strings, and the
`source_kind` x proposed-domain cross-tabulation). Nothing in this report
judges correctness; a human reviewer must do that.