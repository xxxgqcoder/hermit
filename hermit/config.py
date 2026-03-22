import re
from pathlib import Path

# Project root: directory containing main.py
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Model storage
MODEL_ROOT = PROJECT_ROOT / "models"

# Data storage (Qdrant + SQLite)
DATA_ROOT = PROJECT_ROOT / "data"

# Default chunking parameters
DEFAULT_CHUNK_SIZE = 512
DEFAULT_CHUNK_OVERLAP = 64

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
PORT = 8000
