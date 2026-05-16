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

# Search requests are intentionally serialized.  The ONNX models are large
# shared sessions with retained native buffers; concurrent request execution
# raises memory pressure without enough benefit for Hermit's local use case.
_SEARCH_WORKERS = 1

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
    logger.info("Search executor: serialized (%d worker)", _SEARCH_WORKERS)

    # Auto-download missing models before loading them
    from hermit.models import ensure_models, ensure_quantized_models
    ensure_models()
    ensure_quantized_models()

    logger.info("Starting Hermit — loading models...")
    embedder.warmup()
    reranker.warmup()
    reranker.start_idle_unloader()
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

    # Check if the Qdrant deployment mode changed since last run.
    # The embedded qdrant-client (local) and the Rust qdrant-server
    # (standalone) use incompatible on-disk layouts, so a switch in
    # either direction must trigger a full re-index — otherwise the
    # new engine sees an empty store while Hermit's metadata still
    # claims N indexed files.
    from hermit.storage.qdrant_mode_signature import (
        check_mode_changed, save_mode,
    )
    mode_changed, old_mode, new_mode = check_mode_changed(QDRANT_HOST)
    if mode_changed:
        logger.warning(
            "Qdrant deployment mode change detected! Old: %s, New: %s. "
            "All collections will be re-indexed in background "
            "(local and standalone engines use incompatible on-disk formats).",
            old_mode, new_mode,
        )

    needs_rebuild = model_changed or mode_changed

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
            if needs_rebuild:
                reason = "model change" if model_changed else f"mode change ({old_mode} → {new_mode})"
                logger.warning(
                    "Queuing full re-index for collection '%s' due to %s.",
                    name, reason,
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

    # Save current signatures after successful startup
    if model_changed:
        save_signature()
        logger.info("Model signature updated.")
    if mode_changed:
        save_mode(new_mode)
        logger.info("Qdrant mode signature updated to '%s'.", new_mode)

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
