import logging
from fastapi import APIRouter, HTTPException

from hermit.api.schemas import (
    CollectionStatus,
    CreateCollectionRequest,
    SearchRequest,
    SearchResponse,
    SearchResult,
    SyncResponse,
)
from hermit.ingestion.scanner import scan_folder
from hermit.ingestion.watcher import start_watching, stop_watching
from hermit.retrieval.searcher import search
from hermit.storage.metadata import MetadataStore
from hermit.storage.qdrant import delete_collection, ensure_collection

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


@router.post("/collections", response_model=SyncResponse)
def create_collection(req: CreateCollectionRequest):
    if req.name in _collections:
        raise HTTPException(status_code=409, detail=f"Collection '{req.name}' already exists")

    ensure_collection(req.name)
    stats = scan_folder(req.name, req.folder_path, req.chunk_size, req.chunk_overlap)
    start_watching(req.name, req.folder_path, req.chunk_size, req.chunk_overlap)

    _collections[req.name] = {
        "folder_path": req.folder_path,
        "chunk_size": req.chunk_size,
        "chunk_overlap": req.chunk_overlap,
    }
    return SyncResponse(**stats)


@router.delete("/collections/{name}")
def remove_collection(name: str):
    if name not in _collections:
        raise HTTPException(status_code=404, detail=f"Collection '{name}' not found")
    stop_watching(name)
    delete_collection(name)
    MetadataStore(name).destroy()
    _collections.pop(name, None)
    return {"detail": f"Collection '{name}' deleted"}


@router.post("/collections/{name}/sync", response_model=SyncResponse)
def sync_collection(name: str):
    if name not in _collections:
        raise HTTPException(status_code=404, detail=f"Collection '{name}' not found")
    cfg = _collections[name]
    stats = scan_folder(name, cfg["folder_path"], cfg["chunk_size"], cfg["chunk_overlap"])
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
