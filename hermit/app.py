"""Hermit FastAPI application — server entry point.

Run via: uvicorn hermit.app:app --host 0.0.0.0 --port 8000
"""

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

from fastapi import FastAPI

from hermit.api.routes import router
from hermit.config import SEARCH_THREADS
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
_search_executor: ThreadPoolExecutor | None = None


def get_search_executor() -> ThreadPoolExecutor:
    assert _search_executor is not None, "search executor not initialised"
    return _search_executor


def get_server_state() -> dict:
    return {
        "start_time": _server_start_time,
        "ready": _server_ready,
        "uptime": time.time() - _server_start_time if _server_start_time else 0,
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _server_start_time, _server_ready, _search_executor
    _server_start_time = time.time()

    _search_executor = ThreadPoolExecutor(
        max_workers=SEARCH_THREADS,
        thread_name_prefix="search",
    )
    logger.info("Search thread pool: %d threads", SEARCH_THREADS)

    # Auto-download missing models before loading them
    from hermit.models import ensure_models, ensure_quantized_models
    ensure_models()
    ensure_quantized_models()

    logger.info("Starting Hermit — loading models...")
    embedder.warmup()
    reranker.warmup()
    start_task_worker()

    # In standalone mode, initialise the Qdrant connection eagerly so that
    # any Docker image pull or container startup happens here — with clear
    # log output — rather than silently inside the first collection scan.
    from hermit.config import QDRANT_HOST
    if QDRANT_HOST:
        logger.info("Standalone 模式：提前初始化 Qdrant 连接 (%s)...", QDRANT_HOST)
        from hermit.storage.qdrant import client as _qdrant_client
        _qdrant_client()  # triggers ensure_qdrant_running + image pull if needed
        logger.info("Qdrant 连接已就绪。")

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
        ig_pat = cfg.get("ignore_patterns", [])
        ig_ext = cfg.get("ignore_extensions", [])
        try:
            if model_changed:
                logger.warning(
                    "Queuing full re-index for collection '%s' due to model change.", name
                )
                rebuild_collection(
                    name,
                    cfg["folder_path"],
                    ignore_patterns=ig_pat,
                    ignore_extensions=ig_ext,
                )
            else:
                stats = scan_folder(
                    name,
                    cfg["folder_path"],
                    defer_indexing=True,
                    ignore_patterns=ig_pat,
                    ignore_extensions=ig_ext,
                )
                logger.info("Startup scan for '%s': %s", name, stats)

            start_watching(name, cfg["folder_path"],
                           ignore_patterns=ig_pat, ignore_extensions=ig_ext)
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
    if _search_executor:
        _search_executor.shutdown(wait=False)
    # Explicitly stop the managed Qdrant container during graceful shutdown.
    # This is more reliable than atexit (which is skipped on SIGKILL) and
    # fires deterministically as part of the ASGI lifespan shutdown event.
    from hermit.config import QDRANT_HOST, QDRANT_MANAGED
    if QDRANT_HOST and QDRANT_MANAGED:
        from hermit.config import QDRANT_CONTAINER_NAME
        from hermit.storage.qdrant_docker import stop_qdrant_container
        stop_qdrant_container(QDRANT_CONTAINER_NAME)


app = FastAPI(title="Hermit", version="0.1.0", lifespan=lifespan)
app.include_router(router)
