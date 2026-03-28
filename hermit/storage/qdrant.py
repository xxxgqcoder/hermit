import logging
import threading
from qdrant_client import QdrantClient, models

from hermit.config import DATA_ROOT

logger = logging.getLogger(__name__)

# Global lock protecting all local Qdrant client operations.
# qdrant_client's local mode uses numpy arrays internally which are
# NOT thread-safe; concurrent upsert/delete can corrupt the index.
_lock = threading.Lock()


class CollectionCorruptedError(Exception):
    """Raised when local Qdrant data is corrupted and the collection was recreated."""


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


def _create_collection_unlocked(c: QdrantClient, name: str):
    """Internal helper to create a collection. Caller must already hold _lock."""
    from hermit.config import DENSE_DIM
    c.create_collection(
        collection_name=name,
        hnsw_config=models.HnswConfigDiff(on_disk=True),
        optimizers_config=models.OptimizersConfigDiff(default_segment_number=2),
        vectors_config={
            "dense": models.VectorParams(
                size=DENSE_DIM, 
                distance=models.Distance.COSINE,
                on_disk=True
            ),
        },
        sparse_vectors_config={
            "sparse": models.SparseVectorParams(),
        },
    )
    c.create_payload_index(
        collection_name=name,
        field_name="source_file",
        field_schema=models.PayloadSchemaType.KEYWORD,
    )


def ensure_collection(name: str):
    """Create collection with named dense + sparse vectors if it doesn't exist."""
    with _lock:
        c = client()
        if c.collection_exists(name):
            return
        _create_collection_unlocked(c, name)
        logger.info("Created collection '%s'", name)


def delete_collection(name: str):
    with _lock:
        c = client()
        if c.collection_exists(name):
            c.delete_collection(name)
            logger.info("Deleted collection '%s'", name)


def delete_by_source_file(collection_name: str, source_file: str):
    """Delete all points whose source_file matches.

    Raises CollectionCorruptedError if the local Qdrant data is corrupted;
    the collection is automatically recreated in that case.
    """
    with _lock:
        c = client()
        try:
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
        except IndexError:
            logger.warning(
                "Local Qdrant data corrupted for collection '%s', recreating",
                collection_name,
            )
            c.delete_collection(collection_name)
            _create_collection_unlocked(c, collection_name)
            logger.info("Recreated collection '%s'", collection_name)
            raise CollectionCorruptedError(collection_name)


def upsert_chunks(
    collection_name: str,
    ids: list[str],
    dense_vectors: list[list[float]],
    sparse_vectors: list,
    payloads: list[dict],
):
    """Upsert chunk points with named dense + sparse vectors."""
    points = _build_points(ids, dense_vectors, sparse_vectors, payloads)
    with _lock:
        client().upsert(collection_name=collection_name, points=points)


def replace_file_chunks(
    collection_name: str,
    source_file: str,
    ids: list[str],
    dense_vectors: list[list[float]],
    sparse_vectors: list,
    payloads: list[dict],
):
    """Delete old points for source_file and upsert new ones in a single lock.

    Raises CollectionCorruptedError if the local Qdrant data is corrupted.
    """
    points = _build_points(ids, dense_vectors, sparse_vectors, payloads)
    with _lock:
        c = client()
        try:
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
        except IndexError:
            logger.warning(
                "Local Qdrant data corrupted for collection '%s', recreating",
                collection_name,
            )
            c.delete_collection(collection_name)
            _create_collection_unlocked(c, collection_name)
            logger.info("Recreated collection '%s'", collection_name)
            raise CollectionCorruptedError(collection_name)
        c.upsert(collection_name=collection_name, points=points)


def _build_points(ids, dense_vectors, sparse_vectors, payloads):
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
    return points


def query_points(collection_name: str, **kwargs):
    """Thread-safe wrapper around client().query_points()."""
    with _lock:
        return client().query_points(collection_name=collection_name, **kwargs)
