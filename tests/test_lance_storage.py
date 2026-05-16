"""Unit tests for the LanceDB-backed storage module.

Fully offline. Each test gets a fresh ``DATA_ROOT`` under ``tmp_path`` so
collections don't bleed between tests.
"""

from __future__ import annotations

import uuid

import pytest

import hermit.config as cfg
import hermit.storage.lance as lance_mod


@pytest.fixture()
def lance_env(tmp_path, monkeypatch):
    """Isolate the LanceDB connection in a per-test tmp dir."""
    monkeypatch.setattr(cfg, "DATA_ROOT", tmp_path / "data")
    monkeypatch.setattr(lance_mod, "DATA_ROOT", tmp_path / "data")
    monkeypatch.setattr(lance_mod, "_db", None)
    yield lance_mod
    monkeypatch.setattr(lance_mod, "_db", None)


def _vec(seed: int) -> list[float]:
    """Cheap deterministic dense vector of the right shape."""
    return [float((seed + i) % 17) / 17.0 for i in range(cfg.DENSE_DIM)]


def _payload(source_file: str, i: int, text: str = "hello world", filename: str = "doc") -> dict:
    return {
        "text": text,
        "title": "Doc",
        "filename": filename,
        "source_file": source_file,
        "chunk_index": i,
        "total_chunks": 3,
    }


def _add_chunks(lance, name, source_file, count, seed_base=0):
    ids = [str(uuid.uuid4()) for _ in range(count)]
    vectors = [_vec(seed_base + i) for i in range(count)]
    payloads = [_payload(source_file, i) for i in range(count)]
    lance.replace_file_chunks(name, source_file, ids, vectors, payloads)


# ── Collection lifecycle ─────────────────────────────────────────


def test_ensure_collection_creates_table_with_schema_and_indexes(lance_env):
    tbl = lance_env.ensure_collection("docs")
    assert tbl.count_rows() == 0
    columns = {f.name for f in tbl.schema}
    assert {"id", "text", "title", "filename", "source_file",
            "chunk_index", "total_chunks", "vector"} == columns

    index_names = {idx.name for idx in tbl.list_indices()}
    # FTS on text + filename, scalar on source_file
    assert "text_idx" in index_names
    assert "filename_idx" in index_names
    assert "source_file_idx" in index_names


def test_ensure_collection_is_idempotent(lance_env):
    a = lance_env.ensure_collection("docs")
    b = lance_env.ensure_collection("docs")
    assert a.name == b.name
    # Second call must not duplicate indices.
    assert len([i for i in b.list_indices() if i.name == "text_idx"]) == 1


def test_delete_collection_drops_table(lance_env):
    lance_env.ensure_collection("docs")
    assert "docs" in lance_env.db().list_tables().tables
    lance_env.delete_collection("docs")
    assert "docs" not in lance_env.db().list_tables().tables


def test_delete_collection_missing_is_noop(lance_env):
    # Should not raise when the table was never created.
    lance_env.delete_collection("never")


# ── replace_file_chunks ──────────────────────────────────────────


def test_replace_file_chunks_inserts_rows(lance_env):
    _add_chunks(lance_env, "docs", "/a.md", count=3)
    tbl = lance_env.open_table("docs")
    assert tbl.count_rows() == 3
    rows = tbl.search().to_list()
    assert {r["source_file"] for r in rows} == {"/a.md"}


def test_replace_file_chunks_deletes_old_then_inserts_new(lance_env):
    _add_chunks(lance_env, "docs", "/a.md", count=5, seed_base=0)
    _add_chunks(lance_env, "docs", "/b.md", count=2, seed_base=100)
    assert lance_env.open_table("docs").count_rows() == 7

    # Re-insert /a.md with a smaller chunk list — should drop the old 5 and add 2.
    _add_chunks(lance_env, "docs", "/a.md", count=2, seed_base=200)
    tbl = lance_env.open_table("docs")
    assert tbl.count_rows() == 4
    rows = tbl.search().where("source_file = '/a.md'").to_list()
    assert len(rows) == 2


def test_replace_file_chunks_with_empty_ids_only_deletes(lance_env):
    """Regression: shrinking to zero chunks must still wipe the prior rows."""
    _add_chunks(lance_env, "docs", "/a.md", count=3)
    assert lance_env.open_table("docs").count_rows() == 3

    lance_env.replace_file_chunks("docs", "/a.md", [], [], [])
    assert lance_env.open_table("docs").count_rows() == 0


def test_replace_file_chunks_escapes_single_quotes(lance_env):
    """File paths with single quotes must not break the SQL DELETE."""
    tricky = "/notes/it's-a-file.md"
    ids = [str(uuid.uuid4())]
    vectors = [_vec(0)]
    payloads = [_payload(tricky, 0)]
    lance_env.replace_file_chunks("docs", tricky, ids, vectors, payloads)

    # Calling again with the same source_file must drop the prior row without
    # raising a SQL parse error.
    lance_env.replace_file_chunks("docs", tricky, ids, vectors, payloads)
    assert lance_env.open_table("docs").count_rows() == 1


# ── delete_by_source_file ────────────────────────────────────────


def test_delete_by_source_file_removes_only_matching(lance_env):
    _add_chunks(lance_env, "docs", "/a.md", count=2)
    _add_chunks(lance_env, "docs", "/b.md", count=3)
    lance_env.delete_by_source_file("docs", "/a.md")

    tbl = lance_env.open_table("docs")
    remaining = tbl.search().to_list()
    assert {r["source_file"] for r in remaining} == {"/b.md"}
    assert len(remaining) == 3


# ── Lazy vector index ────────────────────────────────────────────


def test_vector_index_skipped_below_threshold(lance_env, monkeypatch):
    monkeypatch.setattr(lance_mod, "VECTOR_INDEX_THRESHOLD", 100)
    _add_chunks(lance_env, "docs", "/a.md", count=10)
    tbl = lance_env.open_table("docs")
    assert all(idx.name != "vector_idx" for idx in tbl.list_indices())


def test_vector_index_built_once_threshold_crossed(lance_env, monkeypatch):
    # LanceDB's IVF_PQ trainer needs >= 256 rows to fit centroids. Set a
    # threshold just under that and feed enough rows to satisfy both.
    monkeypatch.setattr(lance_mod, "VECTOR_INDEX_THRESHOLD", 256)
    _add_chunks(lance_env, "docs", "/a.md", count=260)
    tbl = lance_env.open_table("docs")
    vector_indices = [
        idx for idx in tbl.list_indices()
        if "vector" in (idx.columns or [])
    ]
    assert vector_indices, "expected a vector index after crossing the threshold"
