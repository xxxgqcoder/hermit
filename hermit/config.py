import json
import os
import re
from pathlib import Path

# ── HERMIT_HOME ─────────────────────────────────────────────────
# Default: ~/.hermit/  — override with HERMIT_HOME env var
HERMIT_HOME = Path(os.environ.get("HERMIT_HOME", Path.home() / ".hermit"))

# Model storage
MODEL_ROOT = HERMIT_HOME / "models"

# Data storage (Qdrant + SQLite)
DATA_ROOT = HERMIT_HOME / "data"

# Logs
LOG_DIR = HERMIT_HOME / "logs"

# PID file for daemon management
PID_FILE = HERMIT_HOME / "hermit.pid"

# Default chunking parameters (token-based, using embedding model tokenizer)
DEFAULT_CHUNK_TOKENS = 256
DEFAULT_CHUNK_OVERLAP_TOKENS = 32

# Default search parameters
DEFAULT_TOP_K = 5
DEFAULT_W_DENSE = 0.7
DEFAULT_W_SPARSE = 0.3
DEFAULT_RERANK_CANDIDATES = 50

# Embedding models (fastembed-supported)
DENSE_MODEL = "jinaai/jina-embeddings-v2-base-zh"
DENSE_DIM = 768
SPARSE_MODEL = "Qdrant/bm25"

# Reranker model
RERANKER_MODEL = "jinaai/jina-reranker-v2-base-multilingual"

# Maximum number of knowledge base collections
MAX_COLLECTIONS = 4
MAX_COLLECTION_NAME_LENGTH = 64
COLLECTION_NAME_PATTERN = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]*$")

# Qdrant connection — Local mode (default) vs Stand-alone mode
# Set QDRANT_HOST to connect to an external Qdrant server (e.g. Docker).
# When unset/empty, Hermit uses the embedded local mode (path-based client).
QDRANT_HOST: str | None = os.environ.get("QDRANT_HOST") or None
QDRANT_PORT: int = int(os.environ.get("QDRANT_PORT", 6333))
QDRANT_GRPC_PORT: int = int(os.environ.get("QDRANT_GRPC_PORT", 6334))

# Hermit-managed Qdrant Docker container (Stand-alone mode only).
# Set QDRANT_MANAGED=true to let Hermit start/stop the container automatically.
# Defaults to true when QDRANT_HOST is localhost/127.0.0.1; false for remote hosts.
QDRANT_CONTAINER_NAME: str = os.environ.get("QDRANT_CONTAINER_NAME", "hermit_qdrant")
QDRANT_IMAGE: str = os.environ.get("QDRANT_IMAGE", "qdrant/qdrant:v1.17.0")
_local_hosts = {"localhost", "127.0.0.1"}
QDRANT_MANAGED: bool = os.environ.get(
    "QDRANT_MANAGED",
    "true" if (os.environ.get("QDRANT_HOST") or "").lower() in _local_hosts else "false",
).lower() == "true"
del _local_hosts

# Indexing concurrency
INDEX_WORKERS = int(os.environ.get("HERMIT_INDEX_WORKERS", 2))

# ONNX Runtime thread control — prevents oversubscription on Linux.
# Default: half of available CPU cores (min 2).
ONNX_THREADS = int(os.environ.get("HERMIT_ONNX_THREADS", max(2, os.cpu_count() // 2)))

# Polling interval for knowledge base file change detection (seconds)
# Default: 900s (15 minutes). Override with HERMIT_POLL_INTERVAL env var.
POLL_INTERVAL_SECONDS = int(os.environ.get("HERMIT_POLL_INTERVAL", 900))

# FastAPI
HOST = "0.0.0.0"
DEFAULT_PORT = 8000
PORT_FILE = HERMIT_HOME / "port.json"


def load_port() -> int:
    """Read persisted port from PORT_FILE; fall back to DEFAULT_PORT."""
    if PORT_FILE.exists():
        try:
            data = json.loads(PORT_FILE.read_text())
            return int(data["port"])
        except (json.JSONDecodeError, KeyError, ValueError, TypeError):
            pass
    return DEFAULT_PORT


def save_port(port: int) -> None:
    """Persist *port* to PORT_FILE."""
    PORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    PORT_FILE.write_text(json.dumps({"port": port}))


def resolve_port() -> int:
    """Return a usable port: try the persisted port, then DEFAULT_PORT, then ask the OS."""
    import socket as _sock

    def _available(p: int) -> bool:
        with _sock.socket(_sock.AF_INET, _sock.SOCK_STREAM) as s:
            return s.connect_ex(("127.0.0.1", p)) != 0

    candidate = load_port()
    if _available(candidate):
        return candidate

    if candidate != DEFAULT_PORT and _available(DEFAULT_PORT):
        return DEFAULT_PORT

    # Let the OS pick a free port
    with _sock.socket(_sock.AF_INET, _sock.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]
