# Deployment (Cloud Run)

## Target
- Project: ua-complaint-form-assistant
- Service: uma-app
- Region: us-central1
- Image: us-central1-docker.pkg.dev/ua-complaint-form-assistant/uma-app/app:latest
- URL: https://uma-app-686770229755.us-central1.run.app

## Build
```bash
gcloud builds submit --tag us-central1-docker.pkg.dev/ua-complaint-form-assistant/uma-app/app:latest
```

Local builds must target the Cloud Run platform explicitly (a plain
`docker build` on Apple Silicon produces an arm64 OCI index that Cloud Run
rejects with "must support amd64/linux"):
```bash
docker build --provenance=false --platform linux/amd64 \
  -t us-central1-docker.pkg.dev/ua-complaint-form-assistant/uma-app/app:latest .
docker push us-central1-docker.pkg.dev/ua-complaint-form-assistant/uma-app/app:latest
```

## Deploy
```bash
gcloud run deploy uma-app --image us-central1-docker.pkg.dev/ua-complaint-form-assistant/uma-app/app:latest --region us-central1 --platform managed --allow-unauthenticated --port 8080 --memory 2Gi --cpu 2 --timeout 600 --max-instances 5
```
Deploying by tag reuses existing service config (env vars are preserved).

## Endpoints
- `/` Landing
- `/app` App (form flow)
- `/app.js`, `/styles.css` Static assets (served by @fastify/static)
- `/health` OK
- POST `/api/analyze` Structured complaint analysis via OpenAI-compatible
  `/v1/chat/completions` on the inference service (`INFERENCE_SERVICE_URL`).
  Returns `{structured}` only; rate-limited per IP; input capped at 64 KB.

## Env
- PORT=8080, HOST=0.0.0.0
- INFERENCE_SERVICE_URL=https://uma-inference-...-a.run.app (required in prod;
  when unset, dev falls back to local inference)
- INFERENCE_AUTH=metadata (required in prod; sends a Google identity token as
  `Authorization: Bearer` to the inference service via the Cloud Run metadata
  server. Unset in dev = plain request.)

## Inference service
- Service: uma-inference, region europe-west4 (GPU), URL
  https://uma-inference-j67732jniq-ez.a.run.app
- Auth: NOT public. `allUsers` invoker removed; only
  `686770229755-compute@developer.gserviceaccount.com` (uma-app's runtime
  service account) holds `roles/run.invoker`. Anonymous calls get 403 at the
  Cloud Run edge.
  ```bash
  gcloud run services remove-iam-policy-binding uma-inference \
    --region europe-west4 --member=allUsers --role=roles/run.invoker
  gcloud run services add-iam-policy-binding uma-inference \
    --region europe-west4 \
    --member=serviceAccount:686770229755-compute@developer.gserviceaccount.com \
    --role=roles/run.invoker
  ```
- Scaling: min-instances 0 (scale-to-zero), max-instances 1, 1 GPU, CPU only
  during requests. Cold start = container start + model load (observed ~1-3
  min); the web fetch timeout is 120 s, so a cold inference can surface as a
  generic 502 — retry succeeds once warm.

## Notes
- Web container is node:20-slim + prod node_modules + src + public (~90 MB).
  No Python, no MLX, no `ml/` directory in the image; the local-MLX fallback
  cannot run there (fails closed if `INFERENCE_SERVICE_URL` is ever unset).
- Inference runs on a separate Cloud Run service (llama-server, GGUF model)
  and is treated as untrusted upstream: response is validated with zod,
  errors are sanitized, and upstream failure surfaces as 502 with a generic
  Ukrainian message.
- Rate limiting defaults to 10 requests/min per IP and is per-instance
  in-memory (multi-instance Cloud Run means effective limits are per
  instance, not global).
