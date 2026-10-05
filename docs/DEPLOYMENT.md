# Deployment (Cloud Run)

## Target
- Project: ua-complaint-form-assistant
- Service: uma-app
- Region: us-central1
- Image: us-central1-docker.pkg.dev/ua-complaint-form-assistant/uma-app/app:latest

## Build
```bash
gcloud builds submit --tag us-central1-docker.pkg.dev/ua-complaint-form-assistant/uma-app/app:latest
```

## Deploy
```bash
gcloud run deploy uma-app --image us-central1-docker.pkg.dev/ua-complaint-form-assistant/uma-app/app:latest --region us-central1 --platform managed --allow-unauthenticated --port 8080 --memory 2Gi --cpu 2 --timeout 600 --max-instances 5
```

## Endpoints
- `/` Landing
- `/app` App
- `/health` OK
- POST `/api/analyze` Calls v2 inference via ml.tune.serve_v2

## Env
- PYTHON_PATH=python
- PORT=8080, HOST=0.0.0.0
- V2_MODEL_PATH configurable (canonical v2)

## Notes
Inference requires Python deps from ml/requirements-serve.txt. MLX only runs on Apple Silicon; on x86_64 Cloud Run this v2 inference path will fail without a compatible runtime (as observed). The web UI and API structure are production-ready; model inference is the deployment-specific dependency.
