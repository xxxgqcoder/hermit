import atexit
import contextlib
import fcntl
import logging
import os
import socket
import threading
from pathlib import Path

from qdrant_client import QdrantClient, models

from hermit.config import DATA_ROOT

logger = logging.getLogger(__name__)

# Global lock protecting all local Qdrant client operations.
# qdrant_client's local mode uses numpy arrays internally which are
# NOT thread-safe; concurrent upsert/delete can corrupt the index.
# In Stand-alone mode this lock is bypassed via _get_lock().
_lock = threading.Lock()

# True when QDRANT_HOST is set and we are connected to an external Qdrant server.
_standalone_mode: bool = False

# File descriptor for the app-level exclusive lock on the local data directory.
# None when not held (standalone mode or not yet acquired).
_app_lock_fd: int | None = None

# Guard to register the Docker atexit handler only once per process.
_docker_atexit_registered: bool = False


class CollectionCorruptedError(Exception):
    """Raised when local Qdrant data is corrupted and the collection was recreated."""


# ── Safeguard helpers ────────────────────────────────────────────

def _get_lock():
    """Return real threading lock in local mode; no-op context in standalone mode."""
    return contextlib.nullcontext() if _standalone_mode else _lock


def _check_no_qdrant_service() -> None:
    """Safeguard #1 — port probe.

    If the Qdrant HTTP or gRPC port is already listening, a stand-alone Qdrant
    process (e.g. Docker) is likely running.  Sharing the directory with the
    local-mode Python client would cause data corruption / deadlocks.
    """
    from hermit.config import QDRANT_PORT, QDRANT_GRPC_PORT
    for port in (QDRANT_PORT, QDRANT_GRPC_PORT):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.5)
            if s.connect_ex(("127.0.0.1", port)) == 0:
                raise RuntimeError(
                    f"端口 {port} 已被占用，检测到可能存在运行中的 Qdrant 服务。"
                    "请设置 QDRANT_HOST 环境变量以使用 Stand-alone 模式，或停止该服务后重试。"
                )


def _acquire_app_lock(qdrant_path: Path) -> None:
    """Safeguard #2 — application-level exclusive file lock.

    Prevents two Hermit processes from opening the same local Qdrant directory
    simultaneously.
    """
    global _app_lock_fd
    if _app_lock_fd is not None:
        return  # already locked by this process
    lock_file = qdrant_path / ".hermit.lock"
    fd = os.open(str(lock_file), os.O_CREAT | os.O_RDWR)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        os.close(fd)
        raise RuntimeError(
            "Qdrant 数据目录已被另一个 Hermit 进程占用，"
            "请检查是否有其他 Hermit 实例正在运行。"
        )
    _app_lock_fd = fd
    atexit.register(_release_app_lock)


def _release_app_lock() -> None:
    """Release the application-level file lock if held."""
    global _app_lock_fd
    if _app_lock_fd is not None:
        try:
            fcntl.flock(_app_lock_fd, fcntl.LOCK_UN)
            os.close(_app_lock_fd)
        except OSError:
            pass
        _app_lock_fd = None


# ── Client initialization ────────────────────────────────────────

def get_client() -> QdrantClient:
    """Return a QdrantClient for either Stand-alone or Local mode.

    Stand-alone mode (QDRANT_HOST set):
      - If QDRANT_MANAGED=true (default for localhost), Hermit starts the
        Qdrant Docker container automatically and registers cleanup on exit.
      - Connects via HTTP/gRPC; global lock is bypassed for concurrent writes.

    Local mode (default, QDRANT_HOST unset):
      - Path-based embedded client with three safeguards.
    """
    global _standalone_mode, _docker_atexit_registered
    from hermit.config import QDRANT_HOST, QDRANT_PORT, QDRANT_GRPC_PORT, QDRANT_MANAGED

    if QDRANT_HOST:
        _standalone_mode = True

        if QDRANT_MANAGED:
            from hermit.config import QDRANT_CONTAINER_NAME, QDRANT_IMAGE
            from hermit.storage.qdrant_docker import ensure_qdrant_running, stop_qdrant_container
            ensure_qdrant_running(
                host=QDRANT_HOST,
                port=QDRANT_PORT,
                grpc_port=QDRANT_GRPC_PORT,
                qdrant_data_path=DATA_ROOT / "qdrant",
                container_name=QDRANT_CONTAINER_NAME,
                image=QDRANT_IMAGE,
            )
            if not _docker_atexit_registered:
                atexit.register(stop_qdrant_container, QDRANT_CONTAINER_NAME)
                _docker_atexit_registered = True

        logger.info(
            "Stand-alone mode: connecting to Qdrant at %s:%d (gRPC:%d)",
            QDRANT_HOST, QDRANT_PORT, QDRANT_GRPC_PORT,
        )
        return QdrantClient(
            host=QDRANT_HOST,
            port=QDRANT_PORT,
            grpc_port=QDRANT_GRPC_PORT,
        )

    # ── Local mode ───────────────────────────────────────────────
    _standalone_mode = False
    qdrant_path = DATA_ROOT / "qdrant"
    qdrant_path.mkdir(parents=True, exist_ok=True)

    # Safeguard #1: abort if a Qdrant service is already listening on the port
    _check_no_qdrant_service()

    # Safeguard #2: exclusive file lock — prevent two Hermit processes
    _acquire_app_lock(qdrant_path)

    try:
        return QdrantClient(path=str(qdrant_path))
    except Exception as e:
        _release_app_lock()
        # Safeguard #3: catch engine-level file lock errors and surface a friendly message
        err_lower = str(e).lower()
        if any(kw in err_lower for kw in ("lock", "temporarily unavailable", "being used", "already opened")):
            raise RuntimeError(
                "Qdrant 数据目录已被占用，请检查是否已启动了 Stand-alone Docker 容器。"
                f"\n原始错误: {e}"
            ) from e
        raise


_client: QdrantClient | None = None


def client() -> QdrantClient:
    global _client
    if _client is None:
        _client = get_client()
    return _client


def _create_collection_unlocked(c: QdrantClient, name: str):
    """Internal helper to create a collection. Caller must already hold the appropriate lock."""
    from hermit.config import DENSE_DIM
    c.create_collection(
        collection_name=name,
        hnsw_config=models.HnswConfigDiff(on_disk=True),
        optimizers_config=models.OptimizersConfigDiff(default_segment_number=2),
        vectors_config={
            "dense": models.VectorParams(
                size=DENSE_DIM,
                distance=models.Distance.COSINE,
                on_disk=True,
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
    with _get_lock():
        c = client()
        if c.collection_exists(name):
            return
        _create_collection_unlocked(c, name)
        logger.info("Created collection '%s'", name)


def delete_collection(name: str):
    with _get_lock():
        c = client()
        if c.collection_exists(name):
            c.delete_collection(name)
            logger.info("Deleted collection '%s'", name)


def delete_by_source_file(collection_name: str, source_file: str):
    """Delete all points whose source_file matches.

    Raises CollectionCorruptedError if the local Qdrant data is corrupted;
    the collection is automatically recreated in that case.
    """
    with _get_lock():
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
    with _get_lock():
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
    with _get_lock():
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
    with _get_lock():
        return client().query_points(collection_name=collection_name, **kwargs)
