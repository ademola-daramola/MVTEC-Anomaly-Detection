"""FastAPI inference worker.

Consumes jobs from the BullMQ queue in Redis (produced by the NestJS API), scores the image
with the saved per-category profile, writes a heatmap, and returns the result to BullMQ.
HTTP endpoints are for health checks, metrics and debugging only.
"""
import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import redis.asyncio as aioredis
from bullmq import Worker
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from scorer import Scorer

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("worker")

QUEUE_NAME = os.environ.get("QUEUE_NAME", "inspections")
REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379")
STORAGE_DIR = Path(os.environ.get("STORAGE_DIR", "./storage")).resolve()

state = {"scorer": None, "worker": None, "redis": None}
stats = {"processed": 0, "failed": 0, "latency_total_ms": 0.0}


def _safe_image_path(raw: str) -> Path:
    """Only read images that live under the shared storage directory."""
    path = Path(raw).resolve()
    if not path.is_relative_to(STORAGE_DIR):
        raise ValueError(f"image_path must be inside {STORAGE_DIR}")
    if not path.is_file():
        raise FileNotFoundError(f"image not found: {path}")
    return path


def _score(image_path: Path, category: str, heatmap_path: Path) -> dict:
    return state["scorer"].score_image(str(image_path), category, str(heatmap_path))


async def process(job, token):
    """BullMQ processor. The returned dict is stored as the job's return value."""
    started = time.perf_counter()
    try:
        image_path = _safe_image_path(job.data["image_path"])
        category = job.data["category"]
        heatmap_path = STORAGE_DIR / "heatmaps" / f"{job.id}.png"
        result = await asyncio.to_thread(_score, image_path, category, heatmap_path)
    except Exception:
        stats["failed"] += 1
        log.exception("job %s failed", job.id)
        raise  # BullMQ applies the retry/backoff policy set by the producer

    latency_ms = round((time.perf_counter() - started) * 1000, 1)
    stats["processed"] += 1
    stats["latency_total_ms"] += latency_ms
    log.info("job %s done: category=%s score=%.3f (%.0f ms)", job.id, category, result["score"], latency_ms)
    return {
        "score": result["score"],
        "threshold": result["threshold"],
        "is_anomalous": result["is_anomalous"],
        "model_version": result["model_version"],
        "category": category,
        "heatmap": f"heatmaps/{job.id}.png",
        "latency_ms": latency_ms,
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    (STORAGE_DIR / "heatmaps").mkdir(parents=True, exist_ok=True)
    state["scorer"] = await asyncio.to_thread(Scorer)  # loads the backbone once
    state["redis"] = aioredis.from_url(REDIS_URL)
    state["worker"] = Worker(QUEUE_NAME, process, {"connection": REDIS_URL})
    log.info("worker listening on queue '%s' | categories: %s", QUEUE_NAME, state["scorer"].available_categories())
    yield
    await state["worker"].close()
    await state["redis"].aclose()


app = FastAPI(title="Anomaly inference worker", lifespan=lifespan)


class ScoreRequest(BaseModel):
    image_path: str
    category: str


@app.get("/health")
async def health():
    categories = state["scorer"].available_categories() if state["scorer"] else []
    return {"status": "ok" if categories else "no_models", "categories": categories}


@app.get("/metrics")
async def metrics():
    depth = await state["redis"].llen(f"bull:{QUEUE_NAME}:wait")
    done = stats["processed"]
    return {
        "queue_depth": depth,
        "processed": done,
        "failed": stats["failed"],
        "avg_latency_ms": round(stats["latency_total_ms"] / done, 1) if done else None,
    }


@app.post("/score")
async def score(req: ScoreRequest):
    """Synchronous scoring, for testing without the queue."""
    try:
        image_path = _safe_image_path(req.image_path)
        heatmap_path = STORAGE_DIR / "heatmaps" / f"debug_{image_path.stem}.png"
        return await asyncio.to_thread(_score, image_path, req.category, heatmap_path)
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))
