"""LanceDB-backed storage for Hermit collections.

Single-mode embedded vector store. Each Hermit collection maps to a LanceDB
table under ``DATA_ROOT/lance/``. Tables carry dense vectors plus a tantivy
FTS index on ``text`` (for hybrid/keyword recall) and a separate FTS index on
``filename`` (for substring filename filters). The ``source_file`` column has
a BTREE scalar index since every ``delete_by_source_file`` hits it.

Vector indexes (IVF_PQ / HNSW) are built lazily once a table grows past
``VECTOR_INDEX_THRESHOLD`` rows — under that, brute-force scan beats both
build cost and index maintenance churn.
"""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING

import lancedb
import pyarrow as pa

from hermit.config import DATA_ROOT, DENSE_DIM

if TYPE_CHECKING:
    from lancedb.db import DBConnection
    from lancedb.table import Table

logger = logging.getLogger(__name__)

# Build a vector index once the table grows past this row count. LanceDB
# recommends ~50k as the break-even point for IVF/HNSW vs brute-force scan.
VECTOR_INDEX_THRESHOLD = 50_000

_db: "DBConnection | None" = None
_db_lock = threading.Lock()


def _schema() -> pa.Schema:
    """PyArrow schema for a Hermit chunk row."""
    return pa.schema([
        pa.field("id", pa.string()),
        pa.field("text", pa.string()),
        pa.field("title", pa.string()),
        pa.field("filename", pa.string()),
        pa.field("source_file", pa.string()),
        pa.field("chunk_index", pa.int32()),
        pa.field("total_chunks", pa.int32()),
        pa.field("vector", pa.list_(pa.float32(), DENSE_DIM)),
    ])


def db() -> "DBConnection":
    """Lazy singleton LanceDB connection rooted at ``DATA_ROOT/lance``."""
    global _db
    if _db is None:
        with _db_lock:
            if _db is None:
                lance_dir = DATA_ROOT / "lance"
                lance_dir.mkdir(parents=True, exist_ok=True)
                _db = lancedb.connect(str(lance_dir))
                logger.info("Connected to LanceDB at %s", lance_dir)
    return _db


def _ensure_indexes(tbl: "Table") -> None:
    """Create FTS + scalar indexes on a table. Idempotent — skips existing."""
    existing = {idx.name for idx in tbl.list_indices()}
    if "text_idx" not in existing:
        tbl.create_fts_index(
            "text",
            use_tantivy=False,
            with_position=False,
            replace=True,
        )
    if "filename_idx" not in existing:
        tbl.create_fts_index(
            "filename",
            use_tantivy=False,
            with_position=False,
            replace=True,
        )
    if "source_file_idx" not in existing:
        tbl.create_scalar_index("source_file", index_type="BTREE")


def ensure_collection(name: str) -> "Table":
    """Open a collection table, creating + indexing if it doesn't exist."""
    connection = db()
    if name in connection.list_tables().tables:
        return connection.open_table(name)
    logger.info("Creating LanceDB table '%s'", name)
    tbl = connection.create_table(name, schema=_schema(), mode="create")
    _ensure_indexes(tbl)
    return tbl


def open_table(name: str) -> "Table":
    """Open an existing table. Raises if the collection wasn't created."""
    return db().open_table(name)


def delete_collection(name: str) -> None:
    connection = db()
    if name in connection.list_tables().tables:
        connection.drop_table(name)
        logger.info("Dropped LanceDB table '%s'", name)


def _escape_sql(value: str) -> str:
    """Escape single quotes for SQL string literals."""
    return value.replace("'", "''")


def delete_by_source_file(collection_name: str, source_file: str) -> None:
    tbl = open_table(collection_name)
    tbl.delete(f"source_file = '{_escape_sql(source_file)}'")


def _maybe_build_vector_index(tbl: "Table") -> None:
    """Build IVF/HNSW vector index opportunistically once the table is big enough."""
    existing = {idx.name for idx in tbl.list_indices()}
    if "vector_idx" in existing:
        return
    if tbl.count_rows() < VECTOR_INDEX_THRESHOLD:
        return
    logger.info(
        "Building vector index on '%s' (%d rows >= %d)",
        tbl.name, tbl.count_rows(), VECTOR_INDEX_THRESHOLD,
    )
    tbl.create_index(metric="cosine", vector_column_name="vector")


def replace_file_chunks(
    collection_name: str,
    source_file: str,
    ids: list[str],
    vectors: list[list[float]],
    payloads: list[dict],
) -> None:
    """Atomically drop existing rows for ``source_file`` and insert new ones.

    ``payloads`` carries the per-chunk metadata (text/title/filename/
    chunk_index/total_chunks). ``vectors`` is the dense embedding list,
    aligned with ``ids``.
    """
    tbl = ensure_collection(collection_name)
    tbl.delete(f"source_file = '{_escape_sql(source_file)}'")
    if not ids:
        return
    rows = [
        {
            "id": ids[i],
            "vector": vectors[i],
            **payloads[i],
        }
        for i in range(len(ids))
    ]
    tbl.add(rows)
    # LanceDB's native FTS index needs an explicit optimize for newly-appended
    # rows to become searchable across reader connections — without it,
    # subsequent ``search(query_type="fts")`` calls miss those rows.
    tbl.optimize()
    _maybe_build_vector_index(tbl)
