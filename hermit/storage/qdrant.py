import logging
from qdrant_client import QdrantClient, models

from hermit.config import DATA_ROOT

logger = logging.getLogger(__name__)


def get_client() -> QdrantClient:
    qdrant_path = DATA_ROOT / "qdrant"
    qdrant_path.mkdir(parents=True, exist_ok=True)
    return QdrantClient(path=str(qdrant_path))


_client: QdrantClient | None = None


def client() -> QdrantClient:
    global _client
    if _client is None:
        _client = get_client()
    return _client


def ensure_collection(name: str):
    """Create collection with named dense + sparse vectors if it doesn't exist."""
    c = client()
    if c.collection_exists(name):
        return
    from hermit.config import DENSE_DIM
    c.create_collection(
        collection_name=name,
        vectors_config={
            "dense": models.VectorParams(size=DENSE_DIM, distance=models.Distance.COSINE),
        },
        sparse_vectors_config={
            "sparse": models.SparseVectorParams(),
        },
    )
    # Payload index on source_file for efficient filtering/deletion
    c.create_payload_index(
        collection_name=name,
        field_name="source_file",
        field_schema=models.PayloadSchemaType.KEYWORD,
    )
    logger.info("Created collection '%s'", name)


def delete_collection(name: str):
    c = client()
    if c.collection_exists(name):
        c.delete_collection(name)
        logger.info("Deleted collection '%s'", name)


def delete_by_source_file(collection_name: str, source_file: str):
    """Delete all points whose source_file matches."""
    c = client()
    c.delete(
        collection_name=collection_name,
        points_selector=models.FilterSelector(
            filter=models.Filter(
                must=[models.FieldCondition(
                    key="source_file",
                    match=models.MatchValue(value=source_file),
                )]
            )
        ),
    )


def upsert_chunks(
    collection_name: str,
    ids: list[str],
    dense_vectors: list[list[float]],
    sparse_vectors: list,
    payloads: list[dict],
):
    """Upsert chunk points with named dense + sparse vectors."""
    c = client()
    points = []
    for i, point_id in enumerate(ids):
        sv = sparse_vectors[i]
        points.append(models.PointStruct(
            id=point_id,
            vector={
                "dense": dense_vectors[i],
                "sparse": models.SparseVector(
                    indices=sv.indices.tolist(),
                    values=sv.values.tolist(),
                ),
            },
            payload=payloads[i],
        ))
    c.upsert(collection_name=collection_name, points=points)
