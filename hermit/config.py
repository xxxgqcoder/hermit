import json
import os
import re
from pathlib import Path

# ── HERMIT_HOME ─────────────────────────────────────────────────
# Default: ~/.hermit/  — override with HERMIT_HOME env var
HERMIT_HOME = Path(os.environ.get("HERMIT_HOME", Path.home() / ".hermit"))

# Model storage
MODEL_ROOT = HERMIT_HOME / "models"

# Data storage (LanceDB + SQLite)
DATA_ROOT = HERMIT_HOME / "data"

# Logs
LOG_DIR = HERMIT_HOME / "logs"
LOG_MAX_BYTES: int = 256 * 1024 * 1024
LOG_BACKUP_COUNT: int = 1

# Embedding cache (chunk text → vector). Keyed by sha256 of the exact text
# passed to the model (title-prefixed chunk), namespaced by the model name so
# model swaps don't poison the cache. Always on — there is no env kill-switch
# by design; the cache is self-healing (dim/length validation on hit) and the
# 30-day TTL ensures any pathological state is reaped automatically.
CACHE_ROOT = HERMIT_HOME / "cache"
EMBED_CACHE_TTL_SECONDS: int = 30 * 86400

# PID file for daemon management
PID_FILE = HERMIT_HOME / "hermit.pid"

# Default chunking parameters (token-based, using embedding model tokenizer)
DEFAULT_CHUNK_TOKENS = 256
DEFAULT_CHUNK_OVERLAP_TOKENS = 32

# Default search parameters
DEFAULT_TOP_K = 5
DEFAULT_RERANK_CANDIDATES = 20
DEFAULT_SEARCH_MODE = "hybrid"
SEARCH_MODES = ("hybrid", "semantic", "keyword", "fuzzy")

# ONNX Runtime thread control — intra/inter-op threads per ONNX session.
# Default to 2: ONNX Runtime retains per-thread arenas, so each extra thread
# inflates resident memory by tens of MB without proportional latency benefit
# on the single-worker search executor. Raise via HERMIT_ONNX_THREADS only when
# you have measured single-request latency and accept the memory cost.
ONNX_THREADS: int = int(os.environ.get("HERMIT_ONNX_THREADS", 2))

# ONNX Runtime arena allocator control. Default OFF — without this, the dense
# embedder and reranker both ratchet their MALLOC_LARGE high-water-mark up over
# time and never give it back to the OS (only a full session destruction does,
# which is what the reranker idle-unload exploits). Disabling the arena makes
# each Run() allocate fresh through plain malloc — a few % slower per call,
# but RSS stays flat across bursts and accumulating indexing runs.
# See problems/concurrent-search-rss-blowup.md and
# problems/dense-embedder-arena-creep.md for the motivating measurements.
# Set HERMIT_ONNX_ARENA=true to re-enable the arena (old fastembed default).
ONNX_ARENA: bool = os.environ.get("HERMIT_ONNX_ARENA", "false").lower() in {"1", "true", "yes"}

# Embedding model (fastembed-supported). Keyword recall is handled by LanceDB's
# tantivy FTS index, so no sparse model is needed.
DENSE_MODEL = "jinaai/jina-embeddings-v2-base-zh"
DENSE_DIM = 768

# Reranker model
RERANKER_MODEL = "jinaai/jina-reranker-v2-base-multilingual"

# Reranker idle unload — drop the in-memory TextCrossEncoder after this many
# seconds without a request to release the ONNX Runtime arena (~2GB resident).
# Next request lazily reloads at the cost of a one-time ~1-3s cold start.
# Set to 0 (or negative) to disable.
RERANKER_IDLE_TIMEOUT: int = int(os.environ.get("HERMIT_RERANKER_IDLE_TIMEOUT", 300))
# How often the background thread wakes to check idleness. Lower bound on
# unload latency past the timeout.
RERANKER_IDLE_CHECK_INTERVAL: int = int(
    os.environ.get("HERMIT_RERANKER_IDLE_CHECK_INTERVAL", 60),
)

# Dense embedder idle unload — same mechanism as reranker but with a longer
# default threshold. Dense is touched by every search (query embed) and every
# indexing batch, so a working session typically keeps it warm; the unload
# fires across genuinely quiet stretches (overnight, between bursts of edits)
# to reclaim the ~1-2GB activation pool that indexing accumulates. Cold start
# is ~0.2-0.5s. Set to 0 (or negative) to disable.
DENSE_IDLE_TIMEOUT: int = int(os.environ.get("HERMIT_DENSE_IDLE_TIMEOUT", 1800))
DENSE_IDLE_CHECK_INTERVAL: int = int(
    os.environ.get("HERMIT_DENSE_IDLE_CHECK_INTERVAL", 120),
)

# Maximum number of knowledge base collections
MAX_COLLECTIONS = 4
MAX_COLLECTION_NAME_LENGTH = 64
COLLECTION_NAME_PATTERN = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]*$")

# Indexing concurrency
# Default 1: personal-use scenario where incremental updates are infrequent.
# A single worker avoids contention with search requests on the shared ONNX session.
# Override with HERMIT_INDEX_WORKERS=2 (or more) for initial bulk indexing.
INDEX_WORKERS = int(os.environ.get("HERMIT_INDEX_WORKERS", 1))

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
