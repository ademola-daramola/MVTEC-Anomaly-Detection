# Anomaly Detection Service

NestJS API -> Redis (BullMQ) -> FastAPI/PyTorch worker. Frozen ResNet18 patch features + per-patch Mahalanobis scoring, trained on MVTec AD "good" images only.

## 1. Train the profiles (once, on your machine)

    pip install -r requirements.txt        # torch, torchvision, numpy, scipy, scikit-learn, opencv, pillow
    set MVTEC_ROOT=C:\path\to\MVTEC        # PowerShell: $env:MVTEC_ROOT="..."
    set MODELS_DIR=models
    python train.py                        # writes models/<category>.npz and results/auroc.json

The `models/` folder is mounted read-only into the worker container. The `.npz` profiles are large (roughly 100+ MB each) and are git-ignored, so every clone must run `train.py` first. MVTec AD must be downloaded separately (non-commercial licence).

**Results (MVTec AD, image-level AUROC):** mean 0.918 over 15 categories; best leather 1.000, bottle 0.998; weakest screw 0.773. Details in `docs/architecture.md`.

## 2. Run the stack

    docker compose up --build

Only the API is published, on host port 3001. The worker's /health and /metrics stay inside the Compose network.

## 3. Try it

    curl -F image=@bottle_defect.png -F category=bottle http://localhost:3001/inspections
    # -> {"job_id":"<uuid>","status":"queued"}

    curl http://localhost:3001/inspections/<uuid>
    # -> {"status":"done","score":...,"threshold":...,"is_anomalous":...,"heatmap_url":"/inspections/<uuid>/heatmap"}

    curl -o heatmap.png http://localhost:3001/inspections/<uuid>/heatmap

Statuses: queued -> processing -> done | failed. Failed jobs retry 3 times with exponential backoff.

## Layout

    common.py, train.py, scorer.py   model code (shared by training and serving)
    worker/                          FastAPI app + BullMQ consumer + Dockerfile
    api/                             minimal NestJS app: POST /inspections, GET /inspections/:id, GET /:id/heatmap
    docker-compose.yml               redis, api, worker, shared storage volume

## Not done yet

- PostgreSQL / TypeORM persistence: job state currently lives in Redis only (kept 24 h).
- First job after an idle period can wait ~16 s before pickup (cause not confirmed).
- Authentication on the API.
- Drift logging and queue-depth alerts beyond the worker's /metrics.
- Contrastive fine-tuning of the backbone.
- Thresholds are tuned on the test set, so treat them as provisional.
