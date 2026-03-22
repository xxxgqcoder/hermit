import logging
from qdrant_client import models

from hermit.config import DEFAULT_RERANK_CANDIDATES, DEFAULT_W_DENSE, DEFAULT_W_SPARSE
from hermit.retrieval import embedder, reranker
from hermit.storage.qdrant import client

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

    c = client()

    # Qdrant prefetch-based hybrid search
    results = c.query_points(
        collection_name=collection_name,
        prefetch=[
            models.Prefetch(
                query=dense_vec,
                using="dense",
                limit=rerank_candidates,
            ),
            models.Prefetch(
                query=models.SparseVector(
                    indices=sparse_vec.indices.tolist(),
                    values=sparse_vec.values.tolist(),
                ),
                using="sparse",
                limit=rerank_candidates,
            ),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=rerank_candidates,
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
        r = results[idx]
        output.append({
            "text": r.payload["text"],
            "source_file": r.payload["source_file"],
            "chunk_index": r.payload["chunk_index"],
            "total_chunks": r.payload["total_chunks"],
            "score": r.score,
        })
    return output
