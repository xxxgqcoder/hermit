import logging
from qdrant_client import models

from hermit.config import DEFAULT_RERANK_CANDIDATES, DEFAULT_W_DENSE, DEFAULT_W_SPARSE
from hermit.retrieval import embedder, reranker
from hermit.storage import qdrant

logger = logging.getLogger(__name__)


def search(
    collection_name: str,
    query: str,
    top_k: int = 5,
    w_dense: float = DEFAULT_W_DENSE,
    w_sparse: float = DEFAULT_W_SPARSE,
    rerank_candidates: int = DEFAULT_RERANK_CANDIDATES,
) -> list[dict]:
    """Hybrid search with reranking. Returns list of result dicts."""
    dense_vec = embedder.embed_query_dense(query)
    sparse_vec = embedder.embed_query_sparse(query)
    candidate_limit = max(top_k, rerank_candidates)

    # Thread-safe query via qdrant module lock
    results = qdrant.query_points(
        collection_name=collection_name,
        prefetch=[
            models.Prefetch(
                query=dense_vec,
                using="dense",
                limit=candidate_limit,
            ),
            models.Prefetch(
                query=models.SparseVector(
                    indices=sparse_vec.indices.tolist(),
                    values=sparse_vec.values.tolist(),
                ),
                using="sparse",
                limit=candidate_limit,
            ),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=candidate_limit,
        with_payload=True,
    ).points

    if not results:
        return []

    # Extract texts for reranking
    passages = [r.payload["text"] for r in results]

    # Rerank
    top_indices = reranker.rerank(query, passages, top_k=top_k)

    output = []
    for idx in top_indices:
        output.append(_result_to_dict(results[idx]))
    return output


def _result_to_dict(result) -> dict:
    return {
        "text": result.payload["text"],
        "source_file": result.payload["source_file"],
        "chunk_index": result.payload["chunk_index"],
        "total_chunks": result.payload["total_chunks"],
        "score": result.score,
    }
