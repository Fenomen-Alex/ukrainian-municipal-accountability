# MVP Documentation

## Purpose
The Ukrainian municipal accountability MVP provides a minimal vertical slice to help citizens of Kropyvnytskyi prepare official appeals for municipal problems. It uses the frozen canonical v2 model to assistively structure complaints; the user always reviews and edits before submission. No automatic submission occurs.

## User flow
1. **Landing (/)** — Explains purpose and primary CTA "Описати проблему".
2. **Complaint input** — Free-form complaint text (required), location (optional), contact (optional). Validates non-empty text and length.
3. **AI structuring** — Calls v2 inference to extract structured topics per schema; treats results as suggestions and allows editing/removal/addition.
4. **Review** — User verifies all fields before generating final appeal.
5. **Final appeal** — Generates human-readable Ukrainian appeal; editable; copy to clipboard; open official channel. Saves local draft with status.
6. **Draft/receipt** — Stored in browser localStorage (ready → submitted_to_user after opening official channel).

## Architecture
- Browser UI (static HTML/JS in public/)
- Fastify API (src/index.ts, src/routes/api.ts)
- Inference adapter (src/services/inference.ts) calling ml.tune.serve_v2
- Domain logic: schema (Zod), normalization (quote handling), appeal generation
- Persistence: localStorage for drafts (easy to replace)

## Local startup
```bash
npm install
npx tsx src/index.ts
```
Open http://localhost:3000/

## Environment variables
See `.env.example`. Defaults provided; `V2_MODEL_PATH`, `PYTHON_PATH`, `HOST`, `PORT`, `OFFICIAL_CHANNEL_URL`.

## Model service requirement
Uses canonical v2 model via `ml.tune.serve_v2` (MLX on Apple Silicon). Canonical HF: `Fenomen-Alex/ukrainian-municipal-accountability-qwen3-8b`.

## Official submission flow
Configurable via `OFFICIAL_CHANNEL_URL`. App copies appeal text and opens official channel; marks draft as "submitted_to_user" (not delivered).

## Current limitations
- Local-only drafts (browser localStorage)
- Attachments UI not implemented
- No auth/persistence server-side
- Inference requires MLX environment

## Not implemented
Auth, accounts, municipal API, automatic email, feed, admin, analytics, payments, maps, mobile, ML retraining.
