import gc
import logging
import threading
import time

# Patch fastembed before importing TextCrossEncoder so the constructor can
# accept enable_mem_pattern alongside enable_cpu_mem_arena.
from hermit.retrieval import fastembed_patch  # noqa: F401  (import-for-side-effect)
from fastembed.rerank.cross_encoder import TextCrossEncoder

from hermit.config import (
    MODEL_ROOT,
    ONNX_ARENA,
    ONNX_THREADS,
    RERANKER_IDLE_CHECK_INTERVAL,
    RERANKER_IDLE_TIMEOUT,
    RERANKER_MODEL,
)

logger = logging.getLogger(__name__)

_reranker: TextCrossEncoder | None = None
_reranker_lock = threading.Lock()
_last_use: float = 0.0
_unloader_started = False
_unloader_thread: threading.Thread | None = None


_ARENA_OPTS = {
    "enable_cpu_mem_arena": ONNX_ARENA,
    "enable_mem_pattern": ONNX_ARENA,
}


def _build_reranker() -> TextCrossEncoder:
    from hermit.storage.quantizer import get_quantized_dir, is_quantized
    if is_quantized(RERANKER_MODEL):
        q_dir = get_quantized_dir(RERANKER_MODEL)
        logger.info(
            "Loading quantized reranker model from %s (threads=%d, arena=%s)",
            q_dir, ONNX_THREADS, ONNX_ARENA,
        )
        return TextCrossEncoder(
            model_name=RERANKER_MODEL,
            cache_dir=str(MODEL_ROOT),
            threads=ONNX_THREADS,
            specific_model_path=str(q_dir),
            **_ARENA_OPTS,
        )
    logger.info(
        "Loading reranker model: %s (threads=%d, arena=%s)",
        RERANKER_MODEL, ONNX_THREADS, ONNX_ARENA,
    )
    return TextCrossEncoder(
        model_name=RERANKER_MODEL,
        cache_dir=str(MODEL_ROOT),
        threads=ONNX_THREADS,
        **_ARENA_OPTS,
    )


def _get_reranker() -> TextCrossEncoder:
    global _reranker, _last_use
    with _reranker_lock:
        if _reranker is None:
            t0 = time.monotonic()
            _reranker = _build_reranker()
            logger.info(
                "Reranker model loaded (cold start %.2fs).",
                time.monotonic() - t0,
            )
        _last_use = time.monotonic()
        return _reranker


def rerank(query: str, passages: list[str], top_k: int) -> list[int]:
    """Rerank passages and return indices of top_k most relevant (descending score)."""
    if not passages:
        return []
    model = _get_reranker()
    scores = list(model.rerank(query, passages))
    # Touch _last_use again after inference so a long rerank doesn't expire
    # mid-flight from the idle check.
    global _last_use
    _last_use = time.monotonic()
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    return ranked[:top_k]


def _unload_locked() -> bool:
    """Drop the reranker reference and trigger GC. Caller must hold lock."""
    global _reranker
    if _reranker is None:
        return False
    _reranker = None
    gc.collect()
    return True


def _idle_unloader_loop():
    while True:
        time.sleep(RERANKER_IDLE_CHECK_INTERVAL)
        if RERANKER_IDLE_TIMEOUT <= 0:
            continue
        with _reranker_lock:
            if _reranker is None:
                continue
            idle = time.monotonic() - _last_use
            if idle <= RERANKER_IDLE_TIMEOUT:
                continue
            logger.info(
                "Reranker idle %.0fs (> %ds), unloading.",
                idle, RERANKER_IDLE_TIMEOUT,
            )
            _unload_locked()


def start_idle_unloader():
    """Spawn the background idle-unload thread once per process."""
    global _unloader_started, _unloader_thread
    if _unloader_started:
        return
    if RERANKER_IDLE_TIMEOUT <= 0:
        logger.info("Reranker idle unload disabled (timeout <= 0).")
        _unloader_started = True
        return
    _unloader_thread = threading.Thread(
        target=_idle_unloader_loop,
        name="reranker-idle-unloader",
        daemon=True,
    )
    _unloader_thread.start()
    _unloader_started = True
    logger.info(
        "Reranker idle unloader started (timeout=%ds, check=%ds).",
        RERANKER_IDLE_TIMEOUT, RERANKER_IDLE_CHECK_INTERVAL,
    )


def is_loaded() -> bool:
    """Return whether the reranker model is currently resident."""
    with _reranker_lock:
        return _reranker is not None


def unload_now() -> bool:
    """Force-unload the reranker immediately. Returns True if a model was dropped."""
    with _reranker_lock:
        return _unload_locked()


def warmup():
    """Load reranker model and run a dummy inference to compile ONNX graphs."""
    logger.info("Warming up reranker model...")
    _get_reranker()
    # Trigger ONNX JIT compilation before server reports ready.
    list(_get_reranker().rerank("warmup", ["warmup passage"]))
    logger.info("Reranker model ready.")
