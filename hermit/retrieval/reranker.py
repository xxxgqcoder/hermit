import logging
from fastembed.rerank.cross_encoder import TextCrossEncoder

from hermit.config import MODEL_ROOT, ONNX_THREADS, RERANKER_MODEL

logger = logging.getLogger(__name__)

_reranker: TextCrossEncoder | None = None


def _get_reranker() -> TextCrossEncoder:
    global _reranker
    if _reranker is None:
        from hermit.storage.quantizer import get_quantized_dir, is_quantized
        if is_quantized(RERANKER_MODEL):
            q_dir = get_quantized_dir(RERANKER_MODEL)
            logger.info(
                "Loading quantized reranker model from %s (threads=%d)",
                q_dir, ONNX_THREADS,
            )
            _reranker = TextCrossEncoder(
                model_name=RERANKER_MODEL,
                cache_dir=str(MODEL_ROOT),
                threads=ONNX_THREADS,
                specific_model_path=str(q_dir),
            )
        else:
            logger.info(
                "Loading reranker model: %s (threads=%d)",
                RERANKER_MODEL, ONNX_THREADS,
            )
            _reranker = TextCrossEncoder(
                model_name=RERANKER_MODEL,
                cache_dir=str(MODEL_ROOT),
                threads=ONNX_THREADS,
            )
        logger.info("Reranker model loaded.")
    return _reranker


def rerank(query: str, passages: list[str], top_k: int) -> list[int]:
    """Rerank passages and return indices of top_k most relevant (descending score)."""
    if not passages:
        return []
    model = _get_reranker()
    scores = list(model.rerank(query, passages))
    # scores is list of float, same order as passages
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    return ranked[:top_k]


def warmup():
    logger.info("Warming up reranker model...")
    _get_reranker()
    logger.info("Reranker model ready.")
