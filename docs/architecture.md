# System Design: Visual Inspection & Anomaly Detection Service

**Author:** Alistair | **Date:** Oct 23, 2026 | **Status:** v2, prototype implemented and tested end to end in Docker Compose

## 1. Purpose

Accept industrial product images, score them for anomalies using a model trained only on defect-free ("good") samples, and return an anomaly score, a pass/fail decision, and a heatmap showing where the deviation is. Uploads never block the web layer, so inference runs asynchronously.

## 2. Components

| Component | Tech | Status | Responsibility |
| --- | --- | --- | --- |
| API layer | NestJS (TypeScript) | Implemented | Upload validation, job creation, status and heatmap endpoints |
| Queue | Redis + BullMQ | Implemented | Durable job queue between the API and the worker |
| Inference worker | Python, FastAPI, PyTorch | Implemented | Loads per-category profiles, extracts patch features, scores, writes heatmaps |
| Shared storage | Docker volume | Implemented | Uploaded images and heatmap overlays |
| Orchestration | Docker Compose | Implemented | redis, api, worker on one network; only the API publishes a port |
| Database | PostgreSQL + TypeORM | Planned | Durable job records, scores, drift statistics (job state currently lives in Redis for 24 h) |
| Auth | NestJS guard | Planned | API authentication |

## 3. Request Flow

1. Client calls `POST /inspections` (multipart: `image`, `category`).
2. The API validates category, file type (PNG/JPEG) and size (10 MB), saves the image to shared storage, enqueues `{image_path, category}` on the `inspections` queue (3 attempts, exponential backoff), and returns `202 {job_id, status: "queued"}`.
3. The worker picks up the job, loads the cached profile for that category, extracts patch features, computes per-patch Mahalanobis distance, smooths the score map, and takes the maximum as the image score.
4. The worker writes the heatmap overlay to shared storage and returns the result as the BullMQ job's return value. There is no callback to the API.
5. The client polls `GET /inspections/:id`. The API reads the job state and result from the queue and maps it to `queued`, `processing`, `done` or `failed`.
6. The client downloads the overlay with `GET /inspections/:id/heatmap`.

## 4. API Contracts

**API (public, host port 3001 -> container port 3000)**

- `POST /inspections` returns `202 {job_id, status}`
- `GET /inspections/:id` returns `{status, score, threshold, is_anomalous, model_version, latency_ms, heatmap_url, created_at, finished_at}` or `{status: "failed", error}`
- `GET /inspections/:id/heatmap` returns the PNG overlay
- `GET /health`

**Worker (internal, not published)**

- `GET /health` lists the categories with loaded profiles
- `GET /metrics` returns queue depth, processed and failed counts, and average latency
- `POST /score` is a synchronous endpoint for debugging

Job IDs are UUIDs, validated before any file lookup. The worker only reads images located under the shared storage directory.

## 5. Job States

`queued` -> `processing` -> `done` | `failed`

Failed jobs retry 3 times with exponential backoff before staying `failed`. Completed and failed jobs are kept for 24 hours.

## 6. Model

- **Backbone:** frozen ImageNet ResNet18; features from layers 1, 2 and 3 are upsampled to 56x56 and concatenated (448 channels), from which 100 are selected with a fixed seed.
- **Normal-state profile:** per-category, per-patch mean and inverse covariance, computed from `train/good` images only.
- **Scoring:** Mahalanobis distance per patch, Gaussian smoothing (sigma 2), image score = maximum of the smoothed map.
- **Threshold:** per category, chosen at the F1-optimal point. It is currently tuned on the test set, so it is provisional.
- **Artifacts:** `models/<category>.npz` with a version tag (`padim-r18-l123-v2`). Each file is roughly 100+ MB because of the per-patch covariance matrices, so profiles are not stored in Git.
- **Upgrade path:** replace the frozen backbone with a contrastively fine-tuned one (SimCLR-style). The serving interface does not change.

## 7. Results (MVTec AD, image-level AUROC)

Mean over 15 categories: **0.918** (first iteration: 0.879). Strongest: leather 1.000, bottle 0.998, wood 0.990, tile 0.987. Weakest: screw 0.773, pill 0.849, toothbrush 0.864.

End-to-end check in Docker: a defective bottle scored 19.6 against a threshold of 10.4 (flagged); a good bottle scored 6.7 (passed). Processing took 0.1 to 2 s once the worker was warm.

## 8. Monitoring and Drift

Implemented: per-job latency in results and worker `/metrics`.
Planned: log every score per category, compare rolling mean and variance with the training baseline to detect input drift, alert on queue depth and failure rate.

## 9. Deployment

`docker compose up --build` starts `redis`, `api` and `worker` with a shared `storage` volume. The `models/` folder is mounted read-only into the worker. The worker image uses CPU PyTorch and bakes the ImageNet weights into the image.

## 10. Risks and Open Questions

- **Pick-up delay after idle:** the first jobs after a quiet period waited about 16 s before processing started, while a job submitted right after activity finished in 0.12 s. Cause not yet confirmed; likely idle polling in the Python queue client.
- **Weak categories:** screw, pill and toothbrush. Smoothing and feature choices are being tuned per category.
- **Thresholds:** tuned on the test set; a held-out validation split is needed.
- **Persistence:** Redis-only state means results expire after 24 hours.
- **Profile size:** about 100+ MB per category; consider fewer channels, float16, or a lower-dimensional projection.
- **Dataset licence:** MVTec AD is for non-commercial use, so this is a research demo.

## 11. Next Steps

1. Tune per-category smoothing and features; recheck the four categories that regressed.
2. Add PostgreSQL persistence with TypeORM and API authentication.
3. Investigate the idle pick-up delay.
4. Add drift logging.
5. Contrastive fine-tuning of the backbone.
