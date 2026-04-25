import asyncio
import logging
from fastapi import APIRouter, HTTPException

from hermit.api.schemas import (
    CollectionCreateRequest,
    CollectionCreateResponse,
    CollectionStatus,
    CollectionRemoveResponse,
    CollectionTaskStatus,
    HealthCollectionInfo,
    HealthResponse,
    SearchRequest,
    SearchResponse,
    SearchResult,
    SyncResponse,
)
from hermit.ingestion.scanner import scan_folder
from hermit.ingestion.task_queue import (
    cancel_collection_tasks,
    get_collection_task_status,
    wait_for_collection_tasks_idle,
)
from hermit.ingestion.watcher import start_watching, stop_watching
from hermit.retrieval.searcher import search
from hermit.storage import qdrant
from hermit.storage.metadata import MetadataStore
from hermit.storage.registry import register, unregister

logger = logging.getLogger(__name__)

router = APIRouter()

# In-memory registry of collections: name -> config
_collections: dict[str, dict] = {}


@router.post("/search", response_model=SearchResponse)
async def do_search(req: SearchRequest):
    if req.collection not in _collections:
        raise HTTPException(status_code=404, detail=f"Collection '{req.collection}' not found")
    from hermit.app import get_search_executor
    loop = asyncio.get_running_loop()
    results = await loop.run_in_executor(
        get_search_executor(),
        lambda: search(
            collection_name=req.collection,
            query=req.query,
            top_k=req.top_k,
            w_dense=req.w_dense,
            w_sparse=req.w_sparse,
            rerank_candidates=req.rerank_candidates,
        ),
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


@router.post("/collections", response_model=CollectionCreateResponse)
def add_collection(req: CollectionCreateRequest):
    if req.name in _collections:
        raise HTTPException(status_code=409, detail=f"Collection '{req.name}' already exists")

    try:
        register(
            req.name,
            req.folder_path,
            ignore_patterns=req.ignore_patterns,
            ignore_extensions=req.ignore_extensions,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    cfg = {
        "folder_path": req.folder_path,
        "ignore_patterns": list(req.ignore_patterns),
        "ignore_extensions": [e.lower() for e in req.ignore_extensions],
    }

    try:
        _collections[req.name] = cfg
        stats = scan_folder(
            req.name,
            req.folder_path,
            defer_indexing=True,
            ignore_patterns=cfg["ignore_patterns"],
            ignore_extensions=cfg["ignore_extensions"],
        )
        start_watching(
            req.name,
            req.folder_path,
            ignore_patterns=cfg["ignore_patterns"],
            ignore_extensions=cfg["ignore_extensions"],
        )
        logger.info("Initial scan for '%s': %s", req.name, stats)
    except Exception as e:
        _collections.pop(req.name, None)
        stop_watching(req.name)
        MetadataStore(req.name).destroy()
        qdrant.delete_collection(req.name)
        unregister(req.name)
        raise HTTPException(status_code=500, detail=str(e)) from e

    return CollectionCreateResponse(
        status="added",
        name=req.name,
        folder_path=req.folder_path,
        ignore_patterns=cfg["ignore_patterns"],
        ignore_extensions=cfg["ignore_extensions"],
    )


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

    from hermit.config import QDRANT_HOST
    return HealthResponse(
        status="ready" if state["ready"] else "starting",
        uptime=state["uptime"],
        models_loaded=state["ready"],
        collections=collections_info,
        pending_index_tasks=total_pending,
        qdrant_mode="standalone" if QDRANT_HOST else "local",
        qdrant_host=QDRANT_HOST,
    )


@router.get("/collections/{name}/tasks", response_model=CollectionTaskStatus)
def collection_tasks_status(name: str):
    if name not in _collections:
        raise HTTPException(status_code=404, detail=f"Collection '{name}' not found")
    return CollectionTaskStatus(**get_collection_task_status(name))


@router.delete("/collections/{name}", response_model=CollectionRemoveResponse)
def remove_collection(name: str):
    cfg = _collections.get(name)
    if cfg is None:
        raise HTTPException(status_code=404, detail=f"Collection '{name}' not found")

    stop_watching(name)
    _collections.pop(name, None)

    try:
        cancel_collection_tasks(name)
        if not wait_for_collection_tasks_idle(name, timeout=30.0):
            raise HTTPException(
                status_code=409,
                detail=(
                    f"Collection '{name}' still has indexing tasks in progress; "
                    "please retry shortly"
                ),
            )

        qdrant.delete_collection(name)
        MetadataStore(name).destroy()
        unregister(name)
    except HTTPException:
        _collections[name] = cfg
        start_watching(
            name,
            cfg["folder_path"],
            ignore_patterns=cfg.get("ignore_patterns", []),
            ignore_extensions=cfg.get("ignore_extensions", []),
        )
        raise
    except Exception as e:
        _collections[name] = cfg
        start_watching(
            name,
            cfg["folder_path"],
            ignore_patterns=cfg.get("ignore_patterns", []),
            ignore_extensions=cfg.get("ignore_extensions", []),
        )
        raise HTTPException(status_code=500, detail=str(e)) from e

    return CollectionRemoveResponse(status="removed", name=name)
