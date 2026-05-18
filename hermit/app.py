"""Hermit FastAPI application — server entry point.

Run via: uvicorn hermit.app:app --host 0.0.0.0 --port 8000
"""

import logging
import time
from concurrent.futures import ThreadPoolExecutor
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

# Search-request executor pool size. Default 1 keeps the historical serialized
# behavior (one reranker / one ONNX session = small steady-state arena). Agent-
# style deep-search workloads can fan out — bump via HERMIT_SEARCH_WORKERS=N
# at your own memory-cost risk: ONNX sessions are thread-safe and release the
# GIL during inference, but each concurrent run grows the arena high-water-mark.
# Pair with reranker idle-unload (HERMIT_RERANKER_IDLE_TIMEOUT) to reclaim
# the peak after bursts. See problems/concurrent-search-rss-blowup.md for
# measured costs at 2/4 workers on this codebase.
import os as _os
_SEARCH_WORKERS = int(_os.environ.get("HERMIT_SEARCH_WORKERS", 1))

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
        max_workers=_SEARCH_WORKERS,
        thread_name_prefix="search",
    )
    logger.info(
        "Search executor: %s (%d worker%s)",
        "serialized" if _SEARCH_WORKERS == 1 else "parallel",
        _SEARCH_WORKERS,
        "" if _SEARCH_WORKERS == 1 else "s",
    )

    # Auto-download missing models before loading them
    from hermit.models import ensure_models, ensure_quantized_models
    ensure_models()
    ensure_quantized_models()

    logger.info("Starting Hermit — loading models...")
    embedder.warmup()
    embedder.start_idle_unloader()
    reranker.warmup()
    reranker.start_idle_unloader()
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
        ig_pat = cfg.get("ignore_patterns", [])
        ig_ext = cfg.get("ignore_extensions", [])
        try:
            if model_changed:
                logger.warning(
                    "Queuing full re-index for collection '%s' due to model change.",
                    name,
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

    if model_changed:
        save_signature()
        logger.info("Model signature updated.")

    _server_ready = True
    logger.info("Hermit ready.")
    yield
    logger.info("Shutting down Hermit.")
    if _search_executor:
        _search_executor.shutdown(wait=False)


app = FastAPI(title="Hermit", version="0.1.0", lifespan=lifespan)
app.include_router(router)
