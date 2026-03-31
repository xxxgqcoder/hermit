"""Model download, verification, and self-check utilities."""

import contextlib
import logging
import threading
import time
from pathlib import Path

from huggingface_hub import snapshot_download

from hermit.config import (
    DENSE_DIM,
    DENSE_MODEL,
    MODEL_ROOT,
    RERANKER_MODEL,
    SPARSE_MODEL,
)

logger = logging.getLogger(__name__)

# ── Model registry ──────────────────────────────────────────────
MODELS = [
    {
        "repo_id": DENSE_MODEL,
        "description": f"Dense embedding ({DENSE_DIM}-dim, Chinese-English)",
        "allow_patterns": [
            "onnx/model.onnx",
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
        ],
    },
    {
        "repo_id": SPARSE_MODEL,
        "description": "Sparse embedding (BM25)",
        "allow_patterns": None,  # small model, download everything
    },
    {
        "repo_id": RERANKER_MODEL,
        "description": "Reranker (multilingual cross-encoder)",
        "allow_patterns": [
            "onnx/model.onnx",
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
        ],
    },
]

MAX_RETRIES = 5
RETRY_DELAY = 3  # seconds


def _model_cache_dir(repo_id: str) -> Path:
    """Return the expected HuggingFace cache directory for a model."""
    # HF cache layout: models--{org}--{name}/
    return MODEL_ROOT / f"models--{repo_id.replace('/', '--')}"


@contextlib.contextmanager
def _log_heartbeat(message: str, interval: float = 30.0):
    """Log *message* every *interval* seconds in a background thread.

    Use as a context manager around long synchronous calls (downloads,
    quantization) so the server log never goes silent for extended periods.
    """
    stop = threading.Event()

    def _worker():
        elapsed = interval
        while not stop.wait(interval):
            logger.info("%s (%.0fs elapsed)", message, elapsed)
            elapsed += interval

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    try:
        yield
    finally:
        stop.set()
        t.join(timeout=1)


def check_models_exist() -> dict[str, bool]:
    """Check which models are present in the local cache."""
    result = {}
    for m in MODELS:
        cache_dir = _model_cache_dir(m["repo_id"])
        # A model is considered present if its cache dir exists and has snapshots
        snapshots = cache_dir / "snapshots"
        result[m["repo_id"]] = snapshots.is_dir() and any(snapshots.iterdir())
    return result


def download_model(repo_id: str, allow_patterns: list[str] | None, force: bool) -> str:
    """Download a single model with retry logic. Returns snapshot path."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with _log_heartbeat(f"Downloading {repo_id}..."):
                path = snapshot_download(
                    repo_id=repo_id,
                    cache_dir=str(MODEL_ROOT),
                    allow_patterns=allow_patterns,
                    force_download=force,
                )
            return path
        except Exception as e:
            if attempt == MAX_RETRIES:
                raise
            logger.warning(
                "Attempt %d/%d for %s failed: %s — retrying in %ds",
                attempt, MAX_RETRIES, repo_id, e, RETRY_DELAY,
            )
            time.sleep(RETRY_DELAY)
    raise RuntimeError("unreachable")


def download_all(force: bool = False):
    """Download all models."""
    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    logger.info("Model cache: %s", MODEL_ROOT)

    for m in MODELS:
        logger.info("Downloading %s — %s", m["repo_id"], m["description"])
        path = download_model(m["repo_id"], m["allow_patterns"], force)
        logger.info("  -> %s", path)

    logger.info("All downloads complete.")


def ensure_models():
    """Check for missing models and download them automatically."""
    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    status = check_models_exist()
    missing = [repo_id for repo_id, exists in status.items() if not exists]

    if not missing:
        logger.info("All models present in %s", MODEL_ROOT)
        return

    logger.info("Missing models detected: %s — downloading...", missing)
    models_by_id = {m["repo_id"]: m for m in MODELS}
    for repo_id in missing:
        m = models_by_id[repo_id]
        logger.info("Downloading %s — %s", m["repo_id"], m["description"])
        path = download_model(m["repo_id"], m["allow_patterns"], force=False)
        logger.info("  -> %s", path)

    logger.info("All missing models downloaded.")


def ensure_quantized_models():
    """Quantize ONNX models to INT8 once; subsequent calls are no-ops."""
    from hermit.storage.quantizer import is_quantized, quantize

    for repo_id in [DENSE_MODEL, RERANKER_MODEL]:
        if is_quantized(repo_id):
            logger.info("INT8 quantized model already present for %s", repo_id)
            continue
        logger.info(
            "Running one-time INT8 quantization for %s — this may take a few minutes...",
            repo_id,
        )
        success = quantize(repo_id, model_file="onnx/model.onnx")
        if not success:
            logger.warning("Quantization failed for %s — fp32 will be used instead", repo_id)


def verify_models():
    """Quick smoke test: load each model and run a dummy inference."""
    from fastembed import SparseTextEmbedding, TextEmbedding
    from fastembed.rerank.cross_encoder import TextCrossEncoder

    logger.info("Verifying models...")
    cache = str(MODEL_ROOT)

    dense = TextEmbedding(model_name=DENSE_MODEL, cache_dir=cache)
    vec = list(dense.embed(["test"]))[0]
    assert len(vec) == DENSE_DIM, f"Expected dim {DENSE_DIM}, got {len(vec)}"
    logger.info("  Dense embedding OK (dim=%d)", len(vec))

    sparse = SparseTextEmbedding(model_name=SPARSE_MODEL, cache_dir=cache)
    svec = list(sparse.embed(["test"]))[0]
    logger.info("  Sparse embedding OK (indices=%d)", len(svec.indices))

    reranker = TextCrossEncoder(model_name=RERANKER_MODEL, cache_dir=cache)
    scores = list(reranker.rerank("query", ["relevant doc", "irrelevant"]))
    assert scores[0] > scores[1], f"Reranker scoring seems wrong: {scores}"
    logger.info("  Reranker OK (scores=%s)", scores)

    logger.info("All models verified.")
