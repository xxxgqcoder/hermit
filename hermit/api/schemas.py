from typing import Literal

from pydantic import BaseModel, model_validator

from hermit.config import (
    DEFAULT_RERANK_CANDIDATES,
    DEFAULT_SEARCH_MODE,
    DEFAULT_TOP_K,
)


SearchMode = Literal["hybrid", "semantic", "keyword", "fuzzy"]


class SearchRequest(BaseModel):
    query: str = ""
    collection: str
    top_k: int = DEFAULT_TOP_K
    rerank_candidates: int = DEFAULT_RERANK_CANDIDATES
    mode: SearchMode = DEFAULT_SEARCH_MODE
    # Substring or glob (`*`/`?`) match against the filename stem (basename
    # without extension).  Orthogonal to `mode`.
    filename: str | None = None
    # Override the per-mode default for cross-encoder reranking. None = use the
    # mode's default (hybrid/semantic on, keyword/fuzzy off).
    rerank: bool | None = None

    @model_validator(mode="after")
    def _check_query_or_filename(self):
        if self.mode == "fuzzy":
            # fuzzy supports query-only, filename-only, or both
            if not self.query and not self.filename:
                raise ValueError(
                    "fuzzy mode requires either `query` or `filename`"
                )
        else:
            if not self.query:
                raise ValueError(f"`query` is required for mode={self.mode!r}")
        return self


class SearchResult(BaseModel):
    text: str
    source_file: str
    chunk_index: int
    total_chunks: int
    score: float | None = None


class SearchResponse(BaseModel):
    results: list[SearchResult]


class CollectionCreateRequest(BaseModel):
    name: str
    folder_path: str
    ignore_patterns: list[str] = []
    ignore_extensions: list[str] = []


class CollectionCreateResponse(BaseModel):
    status: str
    name: str
    folder_path: str
    ignore_patterns: list[str] = []
    ignore_extensions: list[str] = []


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


class CollectionRemoveResponse(BaseModel):
    status: str
    name: str


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
    storage: str  # always "lance"
