from pydantic import BaseModel, Field

from hermit.config import (
    DEFAULT_RERANK_CANDIDATES,
    DEFAULT_TOP_K,
    DEFAULT_W_DENSE,
    DEFAULT_W_SPARSE,
)


class SearchRequest(BaseModel):
    query: str
    collection: str
    top_k: int = DEFAULT_TOP_K
    w_dense: float = Field(DEFAULT_W_DENSE, ge=0, le=1)
    w_sparse: float = Field(DEFAULT_W_SPARSE, ge=0, le=1)
    rerank_candidates: int = DEFAULT_RERANK_CANDIDATES


class SearchResult(BaseModel):
    text: str
    source_file: str
    chunk_index: int
    total_chunks: int
    score: float | None = None


class SearchResponse(BaseModel):
    results: list[SearchResult]


class CollectionStatus(BaseModel):
    name: str
    folder_path: str
    indexed_files: int
    total_chunks: int
    watching: bool


class CollectionTaskStatus(BaseModel):
    collection: str
    pending_tasks: int
    queued_tasks: int
    in_progress_tasks: int
    worker_alive: bool


class SyncResponse(BaseModel):
    added: int
    updated: int
    deleted: int


class HealthCollectionInfo(BaseModel):
    name: str
    indexed_files: int
    total_chunks: int


class HealthResponse(BaseModel):
    status: str  # "ready", "starting"
    uptime: float
    models_loaded: bool
    collections: list[HealthCollectionInfo]
    pending_index_tasks: int
