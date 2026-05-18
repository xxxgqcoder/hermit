import gc
import logging
import threading
import time
from concurrent.futures import Future
from dataclasses import dataclass, field
from queue import Queue, Empty

# Patch fastembed before importing TextEmbedding so the model constructor can
# accept enable_mem_pattern alongside enable_cpu_mem_arena.
from hermit.retrieval import fastembed_patch  # noqa: F401  (import-for-side-effect)
from fastembed import TextEmbedding

from hermit.config import (
    DENSE_IDLE_CHECK_INTERVAL,
    DENSE_IDLE_TIMEOUT,
    DENSE_MODEL,
    MODEL_ROOT,
    ONNX_ARENA,
    ONNX_THREADS,
)
from hermit.retrieval import embed_cache

logger = logging.getLogger(__name__)

# ── Batch scheduler settings ───────────────────────────────────
_BATCH_SIZE = 64       # max texts to accumulate before flushing
_BATCH_TIMEOUT = 0.05  # seconds to wait for more texts before flushing

_dense_model: TextEmbedding | None = None
_model_lock = threading.Lock()  # protects model load / unload
_last_use: float = 0.0
_unloader_started = False
_unloader_thread: threading.Thread | None = None


_ARENA_OPTS = {
    "enable_cpu_mem_arena": ONNX_ARENA,
    "enable_mem_pattern": ONNX_ARENA,
}


def _build_dense_model() -> TextEmbedding:
    from hermit.storage.quantizer import get_quantized_dir, is_quantized
    if is_quantized(DENSE_MODEL):
        q_dir = get_quantized_dir(DENSE_MODEL)
        logger.info(
            "Loading quantized dense model from %s (threads=%d, arena=%s)",
            q_dir, ONNX_THREADS, ONNX_ARENA,
        )
        return TextEmbedding(
            model_name=DENSE_MODEL,
            cache_dir=str(MODEL_ROOT),
            threads=ONNX_THREADS,
            specific_model_path=str(q_dir),
            **_ARENA_OPTS,
        )
    logger.info(
        "Loading dense embedding model: %s (threads=%d, arena=%s)",
        DENSE_MODEL, ONNX_THREADS, ONNX_ARENA,
    )
    return TextEmbedding(
        model_name=DENSE_MODEL,
        cache_dir=str(MODEL_ROOT),
        threads=ONNX_THREADS,
        **_ARENA_OPTS,
    )


def _get_dense_model() -> TextEmbedding:
    global _dense_model, _last_use
    with _model_lock:
        if _dense_model is None:
            t0 = time.monotonic()
            _dense_model = _build_dense_model()
            logger.info(
                "Dense embedding model loaded (cold start %.2fs).",
                time.monotonic() - t0,
            )
        _last_use = time.monotonic()
        return _dense_model


# ── Batch embedding request ────────────────────────────────────

@dataclass
class _EmbedRequest:
    texts: list[str]
    future: Future = field(default_factory=Future)
    count: int = 0

    def __post_init__(self):
        self.count = len(self.texts)


class _EmbedScheduler:
    """Dedicated thread that batches embedding requests from multiple workers.

    Workers submit texts and block on the returned Future.  The scheduler
    accumulates texts until BATCH_SIZE is reached or BATCH_TIMEOUT expires,
    then runs inference once and distributes results back via Futures.
    """

    def __init__(self, name: str, embed_fn):
        self._name = name
        self._embed_fn = embed_fn
        self._queue: Queue[_EmbedRequest] = Queue()
        self._thread: threading.Thread | None = None
        self._started = False

    def start(self):
        if self._started:
            return
        self._started = True
        self._thread = threading.Thread(
            target=self._run, name=f"embed-{self._name}", daemon=True
        )
        self._thread.start()

    def submit(self, texts: list[str]) -> Future:
        req = _EmbedRequest(texts=texts)
        self._queue.put(req)
        return req.future

    def _run(self):
        while True:
            # Block until at least one request arrives
            try:
                first = self._queue.get(timeout=1.0)
            except Empty:
                continue

            batch_requests: list[_EmbedRequest] = [first]
            total = first.count

            # Collect more requests until batch is full or timeout
            deadline = time.monotonic() + _BATCH_TIMEOUT
            while total < _BATCH_SIZE:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    req = self._queue.get(timeout=remaining)
                    batch_requests.append(req)
                    total += req.count
                except Empty:
                    break

            # Merge all texts into one batch
            all_texts: list[str] = []
            for req in batch_requests:
                all_texts.extend(req.texts)

            try:
                all_results = self._embed_fn(all_texts)
                # Distribute results back to each request's Future
                offset = 0
                for req in batch_requests:
                    req.future.set_result(all_results[offset:offset + req.count])
                    offset += req.count
            except Exception as exc:
                for req in batch_requests:
                    if not req.future.done():
                        req.future.set_exception(exc)


def _dense_embed_fn(texts: list[str]) -> list[list[float]]:
    model = _get_dense_model()
    embeddings = list(model.embed(texts, batch_size=_BATCH_SIZE))
    return [e.tolist() for e in embeddings]


_dense_scheduler = _EmbedScheduler("dense", _dense_embed_fn)


# ── Public API (index path — batched) ──────────────────────────

def embed_dense(texts: list[str]) -> list[list[float]]:
    """Submit texts for dense embedding. Blocks until the batch is processed.

    Pure cache hits skip the scheduler queue and ONNX path entirely; misses
    still flow through the scheduler so batching across workers is preserved.
    """
    cached, miss_idx = embed_cache.lookup_dense(texts)
    if not miss_idx:
        return cached  # type: ignore[return-value]
    miss_texts = [texts[i] for i in miss_idx]
    _dense_scheduler.start()
    fresh = _dense_scheduler.submit(miss_texts).result()
    for slot, vec in zip(miss_idx, fresh):
        cached[slot] = vec
        embed_cache.store_dense(texts[slot], vec)
    # Touch _last_use again after the batch — long indexing batches can run
    # past the idle threshold; bumping here keeps the unloader from killing
    # the session mid-flight.
    global _last_use
    _last_use = time.monotonic()
    return cached  # type: ignore[return-value]


# ── Public API (query path — immediate, no batching) ───────────

def embed_query_dense(query: str) -> list[float]:
    model = _get_dense_model()
    result = list(model.query_embed(query))[0].tolist()
    global _last_use
    _last_use = time.monotonic()
    return result


# ── Idle unloader ──────────────────────────────────────────────


def _unload_locked() -> bool:
    """Drop the dense model reference and trigger GC. Caller must hold lock."""
    global _dense_model
    if _dense_model is None:
        return False
    _dense_model = None
    gc.collect()
    return True


def _idle_unloader_loop():
    while True:
        time.sleep(DENSE_IDLE_CHECK_INTERVAL)
        if DENSE_IDLE_TIMEOUT <= 0:
            continue
        with _model_lock:
            if _dense_model is None:
                continue
            idle = time.monotonic() - _last_use
            if idle <= DENSE_IDLE_TIMEOUT:
                continue
            logger.info(
                "Dense model idle %.0fs (> %ds), unloading.",
                idle, DENSE_IDLE_TIMEOUT,
            )
            _unload_locked()


def start_idle_unloader():
    """Spawn the background idle-unload thread once per process."""
    global _unloader_started, _unloader_thread
    if _unloader_started:
        return
    if DENSE_IDLE_TIMEOUT <= 0:
        logger.info("Dense model idle unload disabled (timeout <= 0).")
        _unloader_started = True
        return
    _unloader_thread = threading.Thread(
        target=_idle_unloader_loop,
        name="dense-idle-unloader",
        daemon=True,
    )
    _unloader_thread.start()
    _unloader_started = True
    logger.info(
        "Dense model idle unloader started (timeout=%ds, check=%ds).",
        DENSE_IDLE_TIMEOUT, DENSE_IDLE_CHECK_INTERVAL,
    )


def is_loaded() -> bool:
    """Return whether the dense model is currently resident."""
    with _model_lock:
        return _dense_model is not None


def unload_now() -> bool:
    """Force-unload the dense model immediately. Returns True if dropped."""
    with _model_lock:
        return _unload_locked()


def warmup():
    """Pre-load model, start scheduler thread, and run a dummy inference.

    Running a dummy embed call here triggers ONNX Runtime JIT graph compilation
    so that the first real indexing/search request is served at full speed.
    Without this, ONNX compilation happens on the first production call and can
    stall the worker for 30–90 s, causing indexing timeouts.
    """
    logger.info("Warming up embedding model...")
    _get_dense_model()
    _dense_scheduler.start()
    logger.info("Running dummy inference to compile ONNX graphs...")
    embed_dense(["warmup"])
    logger.info("Embedding model ready.")
