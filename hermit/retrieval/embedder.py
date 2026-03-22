import logging
import threading
import time
from concurrent.futures import Future
from dataclasses import dataclass, field
from queue import Queue, Empty
from fastembed import TextEmbedding, SparseTextEmbedding

from hermit.config import MODEL_ROOT, DENSE_MODEL, SPARSE_MODEL, ONNX_THREADS

logger = logging.getLogger(__name__)

# ── Batch scheduler settings ───────────────────────────────────
_BATCH_SIZE = 64       # max texts to accumulate before flushing
_BATCH_TIMEOUT = 0.05  # seconds to wait for more texts before flushing

_dense_model: TextEmbedding | None = None
_sparse_model: SparseTextEmbedding | None = None
_model_lock = threading.Lock()  # protects lazy model init only


def _get_dense_model() -> TextEmbedding:
    global _dense_model
    if _dense_model is None:
        with _model_lock:
            if _dense_model is None:
                logger.info("Loading dense embedding model: %s (threads=%d)", DENSE_MODEL, ONNX_THREADS)
                _dense_model = TextEmbedding(
                    model_name=DENSE_MODEL,
                    cache_dir=str(MODEL_ROOT),
                    threads=ONNX_THREADS,
                )
                logger.info("Dense embedding model loaded.")
    return _dense_model


def _get_sparse_model() -> SparseTextEmbedding:
    global _sparse_model
    if _sparse_model is None:
        with _model_lock:
            if _sparse_model is None:
                logger.info("Loading sparse embedding model: %s", SPARSE_MODEL)
                _sparse_model = SparseTextEmbedding(
                    model_name=SPARSE_MODEL,
                    cache_dir=str(MODEL_ROOT),
                )
                logger.info("Sparse embedding model loaded.")
    return _sparse_model


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


def _sparse_embed_fn(texts: list[str]) -> list:
    model = _get_sparse_model()
    return list(model.embed(texts, batch_size=_BATCH_SIZE))


_dense_scheduler = _EmbedScheduler("dense", _dense_embed_fn)
_sparse_scheduler = _EmbedScheduler("sparse", _sparse_embed_fn)


# ── Public API (index path — batched) ──────────────────────────

def embed_dense(texts: list[str]) -> list[list[float]]:
    """Submit texts for dense embedding. Blocks until the batch is processed."""
    _dense_scheduler.start()
    return _dense_scheduler.submit(texts).result()


def embed_sparse(texts: list[str]) -> list:
    """Submit texts for sparse embedding. Blocks until the batch is processed."""
    _sparse_scheduler.start()
    return _sparse_scheduler.submit(texts).result()


# ── Public API (query path — immediate, no batching) ───────────

def embed_query_dense(query: str) -> list[float]:
    model = _get_dense_model()
    return list(model.query_embed(query))[0].tolist()


def embed_query_sparse(query: str):
    model = _get_sparse_model()
    return list(model.query_embed(query))[0]


def warmup():
    """Pre-load models and start scheduler threads."""
    logger.info("Warming up embedding models...")
    _get_dense_model()
    _get_sparse_model()
    _dense_scheduler.start()
    _sparse_scheduler.start()
    logger.info("Embedding models ready.")
