import logging
from fastembed import TextEmbedding, SparseTextEmbedding

from hermit.config import MODEL_ROOT, DENSE_MODEL, SPARSE_MODEL

logger = logging.getLogger(__name__)

_dense_model: TextEmbedding | None = None
_sparse_model: SparseTextEmbedding | None = None


def _get_dense_model() -> TextEmbedding:
    global _dense_model
    if _dense_model is None:
        logger.info("Loading dense embedding model: %s", DENSE_MODEL)
        _dense_model = TextEmbedding(
            model_name=DENSE_MODEL,
            cache_dir=str(MODEL_ROOT),
        )
        logger.info("Dense embedding model loaded.")
    return _dense_model


def _get_sparse_model() -> SparseTextEmbedding:
    global _sparse_model
    if _sparse_model is None:
        logger.info("Loading sparse embedding model: %s", SPARSE_MODEL)
        _sparse_model = SparseTextEmbedding(
            model_name=SPARSE_MODEL,
            cache_dir=str(MODEL_ROOT),
        )
        logger.info("Sparse embedding model loaded.")
    return _sparse_model


def embed_dense(texts: list[str]) -> list[list[float]]:
    model = _get_dense_model()
    embeddings = list(model.embed(texts, batch_size=32))
    return [e.tolist() for e in embeddings]


def embed_sparse(texts: list[str]) -> list:
    model = _get_sparse_model()
    return list(model.embed(texts, batch_size=32))


def embed_query_dense(query: str) -> list[float]:
    model = _get_dense_model()
    return list(model.query_embed(query))[0].tolist()


def embed_query_sparse(query: str):
    model = _get_sparse_model()
    return list(model.query_embed(query))[0]


def warmup():
    """Pre-load models by running a dummy embed."""
    logger.info("Warming up embedding models...")
    _get_dense_model()
    _get_sparse_model()
    logger.info("Embedding models ready.")
