import logging
from fastapi import APIRouter, HTTPException

from hermit.api.schemas import (
    CollectionStatus,
    CollectionTaskStatus,
    HealthCollectionInfo,
    HealthResponse,
    SearchRequest,
    SearchResponse,
    SearchResult,
    SyncResponse,
)
from hermit.ingestion.scanner import scan_folder
from hermit.ingestion.task_queue import get_collection_task_status
from hermit.retrieval.searcher import search
from hermit.storage.metadata import MetadataStore

logger = logging.getLogger(__name__)

router = APIRouter()

# In-memory registry of collections: name -> config
_collections: dict[str, dict] = {}


@router.post("/search", response_model=SearchResponse)
def do_search(req: SearchRequest):
    if req.collection not in _collections:
        raise HTTPException(status_code=404, detail=f"Collection '{req.collection}' not found")
    results = search(
        collection_name=req.collection,
        query=req.query,
        top_k=req.top_k,
        w_dense=req.w_dense,
        w_sparse=req.w_sparse,
        rerank_candidates=req.rerank_candidates,
    )
    return SearchResponse(results=[SearchResult(**r) for r in results])


@router.post("/collections/{name}/sync", response_model=SyncResponse)
def sync_collection(name: str):
    if name not in _collections:
        raise HTTPException(status_code=404, detail=f"Collection '{name}' not found")
    cfg = _collections[name]
    stats = scan_folder(
        name,
        cfg["folder_path"],
        defer_indexing=True,
        ignore_patterns=cfg.get("ignore_patterns", []),
        ignore_extensions=cfg.get("ignore_extensions", []),
    )
    return SyncResponse(**stats)


@router.get("/collections/{name}/status", response_model=CollectionStatus)
def collection_status(name: str):
    if name not in _collections:
        raise HTTPException(status_code=404, detail=f"Collection '{name}' not found")
    cfg = _collections[name]
    meta = MetadataStore(name)
    status = meta.get_status()
    return CollectionStatus(
        name=name,
        folder_path=cfg["folder_path"],
        watching=True,
        **status,
    )


@router.get("/health", response_model=HealthResponse)
def health():
    from hermit.app import get_server_state
    state = get_server_state()

    collections_info = []
    total_pending = 0
    for name in _collections:
        meta = MetadataStore(name)
        status = meta.get_status()
        collections_info.append(HealthCollectionInfo(
            name=name,
            indexed_files=status["indexed_files"],
            total_chunks=status["total_chunks"],
        ))
        task_status = get_collection_task_status(name)
        total_pending += task_status["pending_tasks"]

    return HealthResponse(
        status="ready" if state["ready"] else "starting",
        uptime=state["uptime"],
        models_loaded=state["ready"],
        collections=collections_info,
        pending_index_tasks=total_pending,
    )


@router.get("/collections/{name}/tasks", response_model=CollectionTaskStatus)
def collection_tasks_status(name: str):
    if name not in _collections:
        raise HTTPException(status_code=404, detail=f"Collection '{name}' not found")
    return CollectionTaskStatus(**get_collection_task_status(name))
