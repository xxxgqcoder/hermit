"""Hermit FastAPI application — server entry point.

Run via: uvicorn hermit.app:app --host 0.0.0.0 --port 8000
"""

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI

from hermit.api.routes import router
from hermit.ingestion.task_queue import start_task_worker
from hermit.retrieval import embedder, reranker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ── Server state ────────────────────────────────────────────────
_server_start_time: float | None = None
_server_ready: bool = False


def get_server_state() -> dict:
    return {
        "start_time": _server_start_time,
        "ready": _server_ready,
        "uptime": time.time() - _server_start_time if _server_start_time else 0,
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _server_start_time, _server_ready
    _server_start_time = time.time()

    # Auto-download missing models before loading them
    from hermit.models import ensure_models
    ensure_models()

    logger.info("Starting Hermit — loading models...")
    embedder.warmup()
    reranker.warmup()
    start_task_worker()

    # Check if embedding models changed since last run
    from hermit.storage.model_signature import check_model_changed, save_signature
    model_changed, old_sig, new_sig = check_model_changed()
    if model_changed:
        logger.warning(
            "Embedding model change detected! Old: %s, New: %s. "
            "All collections will be re-indexed in background.",
            old_sig, new_sig,
        )

    # Reload persisted collections and run startup scan
    from hermit.storage.registry import get_all
    from hermit.ingestion.scanner import scan_folder, rebuild_collection
    from hermit.ingestion.watcher import start_watching
    from hermit.api.routes import _collections

    for name, cfg in get_all().items():
        logger.info("Restoring collection '%s' from %s", name, cfg["folder_path"])
        try:
            if model_changed:
                logger.warning(
                    "Queuing full re-index for collection '%s' due to model change.", name
                )
                rebuild_collection(
                    name,
                    cfg["folder_path"],
                )
            else:
                stats = scan_folder(
                    name,
                    cfg["folder_path"],
                    defer_indexing=True,
                )
                logger.info("Startup scan for '%s': %s", name, stats)

            start_watching(name, cfg["folder_path"])
            _collections[name] = cfg
        except Exception:
            logger.exception("Failed to restore collection '%s'", name)

    # Save current model signature after successful startup
    if model_changed:
        save_signature()
        logger.info("Model signature updated.")

    _server_ready = True
    logger.info("Hermit ready.")
    yield
    logger.info("Shutting down Hermit.")


app = FastAPI(title="Hermit", version="0.1.0", lifespan=lifespan)
app.include_router(router)
