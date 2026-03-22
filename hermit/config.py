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
DEFAULT_RERANK_CANDIDATES = 30

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
