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
import os
import shutil
import threading
from collections import defaultdict
from datetime import timedelta
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

# ── Optimize cadence ──────────────────────────────────────────
# ``Table.optimize()`` makes newly-appended rows visible to the native FTS
# index and runs compact_files + cleanup_old_versions internally. Each call
# *rebuilds* the FTS index under a fresh ``_indices/<uuid>`` directory and
# orphans the previous one — and LanceDB's cleanup_old_versions reaps stale
# data versions but NOT those orphaned index directories (verified on
# lancedb 0.30.2: 40 delete+add+optimize cycles with cleanup_older_than=0
# still leave ~79 orphan index dirs on disk). So two things matter:
#
#   1. Don't optimize per file. ``replace_file_chunks`` only writes rows;
#      the task queue flushes one ``optimize_collection`` per indexing burst
#      (see ingestion/task_queue.py). This cut what was ~600k optimizes on a
#      busy collection down to one-per-burst.
#   2. Periodically ``vacuum_collection`` — the only reliable way to reclaim
#      orphaned index dirs is to rebuild the table from its live rows. It is
#      gated behind ``maybe_vacuum`` so it only fires once orphans pile up.
#
# The cleanup window stays short so transient versions from a single burst
# get reaped promptly. The latest version is never removed regardless.
_OPTIMIZE_CLEANUP_OLDER_THAN = timedelta(minutes=1)

# Rebuild a table once its on-disk index directory count exceeds this. A
# healthy table has one dir per live index (~3: text/filename/source_file,
# plus vector once built). The threshold leaves generous headroom over that
# baseline so vacuum only fires on genuine orphan accumulation, not normal churn.
VACUUM_INDEX_DIR_THRESHOLD = 32

# Suffix for the scratch table vacuum builds before swapping it into place.
_VACUUM_TMP_SUFFIX = "__vacuum"

_db: "DBConnection | None" = None
_db_lock = threading.Lock()

# Per-collection lock serializing writers (replace/optimize) against the
# destructive drop+recreate in ``vacuum_collection`` — they share the single
# ``db()`` connection, so a vacuum must not interleave with an in-flight add.
_collection_locks: "defaultdict[str, threading.Lock]" = defaultdict(threading.Lock)


def _collection_lock(name: str) -> threading.Lock:
    with _db_lock:
        return _collection_locks[name]


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


def _table_dir(name: str) -> "os.PathLike[str]":
    """On-disk directory backing a LanceDB table."""
    return DATA_ROOT / "lance" / f"{name}.lance"


def _index_dir_count(name: str) -> int:
    """Number of ``_indices/<uuid>`` directories on disk for a collection.

    A healthy table has roughly one per live index; a much larger count is
    orphaned FTS-index churn that ``optimize`` cannot reclaim.
    """
    idx_dir = DATA_ROOT / "lance" / f"{name}.lance" / "_indices"
    if not idx_dir.is_dir():
        return 0
    return sum(1 for p in idx_dir.iterdir() if p.is_dir())


def recover_vacuum_temp(name: str) -> None:
    """Reconcile a leftover ``<name>__vacuum.lance`` from a crashed vacuum.

    ``vacuum_collection`` builds the replacement table in a temp directory and
    only then drops the original and renames the temp into place, so a crash
    leaves one of two recoverable states:

    - **final present + temp present** → the crash happened while the temp was
      still being built; the original is intact, so discard the temp.
    - **final missing + temp present** → the crash landed in the tiny window
      between dropping the original and the rename; the temp is the fully-built,
      row-count-verified replacement, so promote it.

    Idempotent and cheap (a couple of ``exists`` checks). Call at startup
    before opening the collection. The original data is never at risk: every
    expensive, panic-prone step runs against the temp while the original
    stands untouched.
    """
    final_dir = _table_dir(name)
    tmp_dir = _table_dir(f"{name}{_VACUUM_TMP_SUFFIX}")
    if not os.path.exists(tmp_dir):
        return
    with _collection_lock(name):
        if os.path.exists(final_dir):
            logger.warning(
                "Discarding aborted vacuum temp for '%s' (original intact)", name
            )
            shutil.rmtree(tmp_dir, ignore_errors=True)
        else:
            logger.warning(
                "Promoting vacuum temp for '%s' (crash between drop and rename)",
                name,
            )
            os.rename(tmp_dir, final_dir)


def vacuum_collection(name: str) -> bool:
    """Rebuild a table from its live rows to reclaim orphaned index dirs.

    LanceDB's ``optimize``/``cleanup_old_versions`` reaps stale data versions
    but leaves orphaned ``_indices/<uuid>`` directories behind, so a table
    that has seen heavy delete+add+optimize churn keeps growing without bound.
    The only reliable reclaim is to recreate the table from its live rows so
    it carries only its live indices, collapsing on-disk size to the logical
    data size.

    Crash-safe by construction: the replacement is built in a temp table
    (``<name>__vacuum``) — where all the expensive, panic-prone work (FTS
    index build) happens — and only after its row count is verified do we drop
    the original and ``os.rename`` the temp into place. The original is dropped
    only in a tiny, compute-free window; any failure before that leaves the
    original fully intact, and ``recover_vacuum_temp`` reconciles a crash
    inside the window on next startup.

    Returns True if a rebuild happened. Holds the per-collection lock so it
    never interleaves with a concurrent ``replace_file_chunks`` add.

    NOTE: ``to_arrow()`` materializes every row (incl. dense vectors) in
    memory. Fine at the current scale (tens of thousands of rows ≈ hundreds
    of MB); revisit with batched streaming if collections grow toward the
    1M-chunk deep-search target.
    """
    connection = db()
    if name not in connection.list_tables().tables:
        return False
    tmp_name = f"{name}{_VACUUM_TMP_SUFFIX}"
    tmp_dir = _table_dir(tmp_name)
    final_dir = _table_dir(name)
    with _collection_lock(name):
        tbl = connection.open_table(name)
        expected = tbl.count_rows()
        data = tbl.to_arrow()
        logger.info(
            "Vacuuming collection '%s': %d rows, %d index dirs on disk",
            name, expected, _index_dir_count(name),
        )

        # Clear any leftover temp from a previously aborted run.
        if tmp_name in connection.list_tables().tables:
            connection.drop_table(tmp_name)
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir, ignore_errors=True)

        # 1) Build the replacement in a temp table. Every expensive,
        #    panic-prone step (FTS index build) happens here while the
        #    original table stays fully intact on disk.
        if data.num_rows == 0:
            new_tbl = connection.create_table(tmp_name, schema=_schema(), mode="create")
        else:
            new_tbl = connection.create_table(tmp_name, data=data, mode="create")
        _ensure_indexes(new_tbl)
        _maybe_build_vector_index(new_tbl)

        rebuilt = new_tbl.count_rows()
        if rebuilt != expected:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            raise RuntimeError(
                f"vacuum '{name}': rebuilt row count {rebuilt} != {expected}; "
                "aborted with original intact"
            )

        # 2) Swap. Tiny, compute-free window: drop the original, then a
        #    same-filesystem atomic rename moves the temp into place.
        connection.drop_table(name)
        os.rename(tmp_dir, final_dir)
    logger.info(
        "Vacuumed '%s': now %d index dirs on disk",
        name, _index_dir_count(name),
    )
    return True


def maybe_vacuum(name: str) -> bool:
    """Vacuum a collection only if orphaned index dirs have piled up.

    Cheap to call repeatedly — a single ``listdir`` when there's nothing to
    do. This replaces the old startup ``compact_collection``, whose
    ``optimize(cleanup_older_than=0)`` never actually reclaimed the orphaned
    index directories it was meant to.
    """
    if _index_dir_count(name) <= VACUUM_INDEX_DIR_THRESHOLD:
        return False
    return vacuum_collection(name)


def optimize_collection(name: str) -> None:
    """Make newly-appended rows searchable via the native FTS index.

    Called once per indexing burst (on task-queue drain) instead of per file.
    Runs under the per-collection lock so it doesn't race a vacuum.
    """
    connection = db()
    if name not in connection.list_tables().tables:
        return
    with _collection_lock(name):
        tbl = connection.open_table(name)
        tbl.optimize(cleanup_older_than=_OPTIMIZE_CLEANUP_OLDER_THAN)


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
    optimize: bool = True,
) -> None:
    """Atomically drop existing rows for ``source_file`` and insert new ones.

    ``payloads`` carries the per-chunk metadata (text/title/filename/
    chunk_index/total_chunks). ``vectors`` is the dense embedding list,
    aligned with ``ids``.

    With ``optimize=True`` (default) the FTS index is refreshed immediately so
    the new rows are searchable — convenient for synchronous/one-off writes.
    The background indexing queue passes ``optimize=False`` and instead calls
    ``optimize_collection`` once per burst, because optimizing per file both
    burns CPU and orphans an FTS-index directory each call (see the module
    docstring on ``_OPTIMIZE_CLEANUP_OLDER_THAN``).
    """
    with _collection_lock(collection_name):
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
        if optimize:
            # Native FTS needs an explicit optimize for appended rows to
            # become searchable across reader connections. Short cleanup
            # window so the per-cycle fragments get reaped promptly.
            tbl.optimize(cleanup_older_than=_OPTIMIZE_CLEANUP_OLDER_THAN)
        _maybe_build_vector_index(tbl)
