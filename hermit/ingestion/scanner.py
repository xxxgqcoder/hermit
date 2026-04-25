import fnmatch
import hashlib
import logging
import uuid
from pathlib import Path

from hermit.ingestion.chunker import chunk_text, chunk_markdown
from hermit.ingestion.task_queue import enqueue_index_task
from hermit.retrieval import embedder
from hermit.storage.metadata import MetadataStore
from hermit.storage import qdrant
from hermit.storage.qdrant import CollectionCorruptedError

logger = logging.getLogger(__name__)

# Whitelist of extensions that contain indexable plain text.
# Files with any other extension (or no extension) are skipped with a warning.
_TEXT_EXTENSIONS = frozenset({
    # Notes / documentation
    '.md', '.markdown', '.txt', '.rst', '.org',
    # Structured text
    '.json', '.jsonl', '.yaml', '.yml', '.toml', '.csv', '.tsv', '.xml',
    # Web
    '.html', '.htm',
    # Source code
    '.py', '.js', '.ts', '.jsx', '.tsx', '.go', '.rs', '.java',
    '.c', '.cpp', '.h', '.hpp', '.cs', '.rb', '.php', '.swift', '.kt',
    '.sh', '.zsh', '.bash', '.fish',
    # Config / logs
    '.ini', '.cfg', '.conf', '.env', '.log',
})


def _file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8192), b""):
            h.update(block)
    return h.hexdigest()


def _collect_files(
    folder: Path,
    ignore_patterns: list[str] | None = None,
    ignore_extensions: list[str] | None = None,
) -> set[str]:
    """Collect all non-hidden files under folder, applying ignore rules."""
    _patterns = ignore_patterns or []
    _extensions = {e.lower() for e in (ignore_extensions or [])}
    files: set[str] = set()
    for file_path in folder.rglob("*"):
        if file_path.is_symlink():
            logger.info("Ignoring symlink: %s", file_path)
            continue
        if not file_path.is_file():
            continue
        rel_parts = file_path.relative_to(folder).parts
        if any(part.startswith(".") for part in rel_parts):
            continue
        if file_path.suffix.lower() not in _TEXT_EXTENSIONS:
            logger.warning("Skipping non-text file: %s", file_path)
            continue
        if _extensions and file_path.suffix.lower() in _extensions:
            continue
        if _patterns:
            rel = str(file_path.relative_to(folder))
            if any(fnmatch.fnmatch(rel, pat) for pat in _patterns):
                continue
        files.add(str(file_path))
    return files


def _index_file(
    collection_name: str,
    file_path: Path,
    meta: MetadataStore,
    file_hash: str | None = None,
) -> bool:
    """Read, chunk, embed, and upsert a single file. Returns True on success."""
    fpath_str = str(file_path)

    try:
        text = file_path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        logger.warning("Failed to read %s: %s", fpath_str, e)
        return False

    chunks = chunk_markdown(text) if file_path.suffix.lower() == '.md' else chunk_text(text)
    if not chunks:
        return False

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

    # Delete old + upsert new in a single lock acquisition
    try:
        qdrant.replace_file_chunks(
            collection_name, fpath_str, ids, dense_vectors, sparse_vectors, payloads
        )
    except CollectionCorruptedError:
        logger.warning(
            "Collection '%s' was recreated due to data corruption; "
            "clearing metadata so all files are re-indexed on next scan",
            collection_name,
        )
        meta.destroy()
        return False

    fhash = file_hash or _file_hash(file_path)
    fmtime = file_path.stat().st_mtime
    meta.upsert(fpath_str, fhash, fmtime, len(chunks))

    logger.info("Indexed %s (%d chunks)", fpath_str, len(chunks))
    return True


def scan_folder(
    collection_name: str,
    folder_path: str,
    defer_indexing: bool = True,
    ignore_patterns: list[str] | None = None,
    ignore_extensions: list[str] | None = None,
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
    disk_files = _collect_files(folder, ignore_patterns, ignore_extensions)
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
        try:
            qdrant.delete_by_source_file(collection_name, fpath_str)
        except CollectionCorruptedError:
            logger.warning(
                "Collection '%s' was recreated due to data corruption; "
                "clearing metadata and aborting scan — re-scan to re-index all files",
                collection_name,
            )
            meta.destroy()
            return {"added": 0, "updated": 0, "deleted": 0, "corrupted": True}
        meta.delete(fpath_str)
        deleted += 1
        logger.info("Removed deleted file from index: %s", fpath_str)

    # --- Handle new files: on disk but not indexed ---
    for fpath_str in sorted(to_add):
        if defer_indexing:
            if enqueue_index_task(collection_name, fpath_str):
                added += 1
        else:
            if _index_file(collection_name, Path(fpath_str), meta):
                added += 1

    # --- Handle existing: check mtime first, then hash for changes ---
    for fpath_str in sorted(to_check):
        old_hash, old_mtime = indexed[fpath_str]
        try:
            current_mtime = Path(fpath_str).stat().st_mtime
        except OSError:
            continue
        if current_mtime == old_mtime:
            continue  # mtime unchanged — skip expensive hash
        current_hash = _file_hash(Path(fpath_str))
        if current_hash == old_hash:
            # Content identical despite mtime change — update mtime in metadata
            meta.upsert(fpath_str, old_hash, current_mtime,
                        meta.get_chunk_count(fpath_str))
            continue
        if defer_indexing:
            if enqueue_index_task(collection_name, fpath_str, file_hash=current_hash):
                updated += 1
        else:
            if _index_file(collection_name, Path(fpath_str), meta, file_hash=current_hash):
                updated += 1

    return {"added": added, "updated": updated, "deleted": deleted}


def rebuild_collection(
    collection_name: str,
    folder_path: str,
    ignore_patterns: list[str] | None = None,
    ignore_extensions: list[str] | None = None,
):
    """Drop and recreate a collection, then re-index all files via task queue."""
    logger.info("Rebuilding index for collection '%s'...", collection_name)
    qdrant.delete_collection(collection_name)
    MetadataStore(collection_name).destroy()
    return scan_folder(
        collection_name,
        folder_path,
        defer_indexing=True,
        ignore_patterns=ignore_patterns,
        ignore_extensions=ignore_extensions,
    )
