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

## Notes
- Web container is Node + tsx; no Python/MLX in the request path.
- Inference runs on a separate Cloud Run service (llama-server, GGUF model)
  and is treated as untrusted upstream: response is validated with zod,
  errors are sanitized, and upstream failure surfaces as 502 with a generic
  Ukrainian message.
- Rate limiting defaults to 10 requests/min per IP and is per-instance
  in-memory (multi-instance Cloud Run means effective limits are per
  instance, not global).
