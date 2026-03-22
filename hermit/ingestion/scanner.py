import hashlib
import logging
import uuid
from pathlib import Path

from hermit.ingestion.chunker import chunk_text
from hermit.ingestion.task_queue import enqueue_index_task
from hermit.retrieval import embedder
from hermit.storage.metadata import MetadataStore
from hermit.storage import qdrant

logger = logging.getLogger(__name__)


def _file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8192), b""):
            h.update(block)
    return h.hexdigest()


def _collect_files(folder: Path) -> set[str]:
    """Collect all non-hidden files under folder."""
    files: set[str] = set()
    for file_path in folder.rglob("*"):
        if not file_path.is_file():
            continue
        if any(part.startswith(".") for part in file_path.parts):
            continue
        files.add(str(file_path))
    return files


def _index_file(
    collection_name: str,
    file_path: Path,
    meta: MetadataStore,
    chunk_size: int,
    chunk_overlap: int,
) -> bool:
    """Read, chunk, embed, and upsert a single file. Returns True on success."""
    fpath_str = str(file_path)

    try:
        text = file_path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        logger.warning("Failed to read %s: %s", fpath_str, e)
        return False

    chunks = chunk_text(text, chunk_size=chunk_size, overlap=chunk_overlap)
    if not chunks:
        return False

    # Delete old entries (safe even if none exist)
    qdrant.delete_by_source_file(collection_name, fpath_str)

    # Prepend document title to each chunk for embedding
    title = file_path.stem
    embed_inputs = [f"[{title}]\n{chunk}" for chunk in chunks]

    dense_vectors = embedder.embed_dense(embed_inputs)
    sparse_vectors = embedder.embed_sparse(embed_inputs)

    ids = [str(uuid.uuid4()) for _ in chunks]
    payloads = [
        {
            "text": chunk,
            "title": title,
            "source_file": fpath_str,
            "chunk_index": i,
            "total_chunks": len(chunks),
        }
        for i, chunk in enumerate(chunks)
    ]

    qdrant.upsert_chunks(collection_name, ids, dense_vectors, sparse_vectors, payloads)

    fhash = _file_hash(file_path)
    fmtime = file_path.stat().st_mtime
    meta.upsert(fpath_str, fhash, fmtime, len(chunks))

    logger.info("Indexed %s (%d chunks)", fpath_str, len(chunks))
    return True


def scan_folder(
    collection_name: str,
    folder_path: str,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
    defer_indexing: bool = True,
) -> dict:
    """Scan folder and sync index with three-way diff. Returns stats.

    If defer_indexing=True, added/updated files are queued for background indexing.
    """
    folder = Path(folder_path).resolve()
    if not folder.is_dir():
        raise ValueError(f"Not a directory: {folder}")

    qdrant.ensure_collection(collection_name)
    meta = MetadataStore(collection_name)

    # Step 1: Collect current disk files and indexed records
    disk_files = _collect_files(folder)
    indexed = meta.get_all_records()  # {path: (hash, mtime)}
    indexed_set = set(indexed.keys())

    # Step 2: Three-way diff
    to_delete = indexed_set - disk_files     # in SQLite but not on disk
    to_add = disk_files - indexed_set        # on disk but not in SQLite
    to_check = disk_files & indexed_set      # in both — need hash comparison

    added = 0
    updated = 0
    deleted = 0

    # --- Handle deletions: indexed but file gone ---
    for fpath_str in to_delete:
        qdrant.delete_by_source_file(collection_name, fpath_str)
        meta.delete(fpath_str)
        deleted += 1
        logger.info("Removed deleted file from index: %s", fpath_str)

    # --- Handle new files: on disk but not indexed ---
    for fpath_str in sorted(to_add):
        if defer_indexing:
            if enqueue_index_task(collection_name, fpath_str, chunk_size, chunk_overlap):
                added += 1
        else:
            if _index_file(collection_name, Path(fpath_str), meta, chunk_size, chunk_overlap):
                added += 1

    # --- Handle existing: check hash for changes ---
    for fpath_str in sorted(to_check):
        old_hash, _ = indexed[fpath_str]
        current_hash = _file_hash(Path(fpath_str))
        if current_hash == old_hash:
            continue
        if defer_indexing:
            if enqueue_index_task(collection_name, fpath_str, chunk_size, chunk_overlap):
                updated += 1
        else:
            if _index_file(collection_name, Path(fpath_str), meta, chunk_size, chunk_overlap):
                updated += 1

    return {"added": added, "updated": updated, "deleted": deleted}
