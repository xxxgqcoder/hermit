import hashlib
import logging
import uuid
from pathlib import Path

from hermit.ingestion.chunker import chunk_text
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


def scan_folder(
    collection_name: str,
    folder_path: str,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> dict:
    """Scan folder and sync index. Returns stats."""
    folder = Path(folder_path).resolve()
    if not folder.is_dir():
        raise ValueError(f"Not a directory: {folder}")

    qdrant.ensure_collection(collection_name)
    meta = MetadataStore(collection_name)

    existing = meta.get_all_records()  # {path: (hash, mtime)}
    current_files: set[str] = set()

    added = 0
    updated = 0
    deleted = 0

    for file_path in sorted(folder.rglob("*")):
        if not file_path.is_file():
            continue
        # Skip hidden files
        if any(part.startswith(".") for part in file_path.parts):
            continue

        fpath_str = str(file_path)
        current_files.add(fpath_str)
        fhash = _file_hash(file_path)
        fmtime = file_path.stat().st_mtime

        # Check if unchanged
        if fpath_str in existing:
            old_hash, old_mtime = existing[fpath_str]
            if old_hash == fhash:
                continue

        # Read and chunk
        try:
            text = file_path.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            logger.warning("Failed to read %s: %s", fpath_str, e)
            continue

        chunks = chunk_text(text, chunk_size=chunk_size, overlap=chunk_overlap)
        if not chunks:
            continue

        # Delete old entries
        qdrant.delete_by_source_file(collection_name, fpath_str)

        # Prepend document title to each chunk for embedding
        title = file_path.stem
        embed_inputs = [f"[{title}]\n{chunk}" for chunk in chunks]

        # Embed (title-augmented text for better retrieval)
        dense_vectors = embedder.embed_dense(embed_inputs)
        sparse_vectors = embedder.embed_sparse(embed_inputs)

        # Build payloads and ids
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

        # Upsert
        qdrant.upsert_chunks(collection_name, ids, dense_vectors, sparse_vectors, payloads)
        meta.upsert(fpath_str, fhash, fmtime, len(chunks))

        if fpath_str in existing:
            updated += 1
        else:
            added += 1

        logger.info("Indexed %s (%d chunks)", fpath_str, len(chunks))

    # Handle deletions
    for old_path in existing:
        if old_path not in current_files:
            qdrant.delete_by_source_file(collection_name, old_path)
            meta.delete(old_path)
            deleted += 1
            logger.info("Removed %s from index", old_path)

    return {"added": added, "updated": updated, "deleted": deleted}
