"""Unit tests for the LanceDB-backed storage module.

Fully offline. Each test gets a fresh ``DATA_ROOT`` under ``tmp_path`` so
collections don't bleed between tests.
"""

from __future__ import annotations

import os
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


# ── vacuum / orphan-index reclaim ────────────────────────────────


def test_vacuum_reclaims_orphan_index_dirs(lance_env):
    """Repeated optimize churn leaks ``_indices`` dirs; vacuum reclaims them.

    LanceDB's optimize/cleanup never reaps orphaned index directories, so a
    table that has seen many delete+add+optimize cycles accumulates them. A
    vacuum rebuild must collapse the count back to the live-index set while
    preserving rows and search.
    """
    name = "docs"
    for i in range(12):
        _add_chunks(lance_env, name, f"/f{i % 3}.md", count=4, seed_base=i * 10)

    before = lance_env._index_dir_count(name)
    rows_before = lance_env.open_table(name).count_rows()
    assert before > 3, f"expected orphan churn, got only {before} index dirs"

    assert lance_env.vacuum_collection(name) is True

    after = lance_env._index_dir_count(name)
    assert after < before
    tbl = lance_env.open_table(name)
    assert tbl.count_rows() == rows_before
    assert {"text_idx", "filename_idx", "source_file_idx"} <= {
        idx.name for idx in tbl.list_indices()
    }
    hits = tbl.search("hello", query_type="fts", fts_columns="text").limit(5).to_list()
    assert hits


def test_maybe_vacuum_noop_below_threshold(lance_env, monkeypatch):
    monkeypatch.setattr(lance_mod, "VACUUM_INDEX_DIR_THRESHOLD", 10_000)
    _add_chunks(lance_env, "docs", "/a.md", count=3)
    assert lance_env.maybe_vacuum("docs") is False


def test_maybe_vacuum_fires_above_threshold(lance_env, monkeypatch):
    monkeypatch.setattr(lance_mod, "VACUUM_INDEX_DIR_THRESHOLD", 3)
    for i in range(8):
        _add_chunks(lance_env, "docs", f"/f{i % 2}.md", count=2, seed_base=i)
    assert lance_env._index_dir_count("docs") > 3
    assert lance_env.maybe_vacuum("docs") is True
    assert lance_env.open_table("docs").count_rows() > 0


def test_optimize_false_skips_optimize_but_rows_present(lance_env):
    """Background path writes with optimize=False; rows land, FTS lags until flush."""
    ids = [str(uuid.uuid4()) for _ in range(3)]
    vectors = [_vec(i) for i in range(3)]
    payloads = [_payload("/a.md", i) for i in range(3)]
    lance_env.replace_file_chunks("docs", "/a.md", ids, vectors, payloads, optimize=False)
    assert lance_env.open_table("docs").count_rows() == 3
    lance_env.optimize_collection("docs")
    hits = (
        lance_env.open_table("docs")
        .search("hello", query_type="fts", fts_columns="text")
        .limit(5)
        .to_list()
    )
    assert hits


# ── vacuum crash-safety (temp-build + atomic rename) ──────────────


def test_vacuum_leaves_no_temp_dir(lance_env):
    """A clean vacuum swaps the temp into place and leaves nothing behind."""
    _add_chunks(lance_env, "docs", "/a.md", count=3)
    assert lance_env.vacuum_collection("docs") is True
    assert not os.path.exists(lance_env._table_dir("docs__vacuum"))
    assert lance_env.open_table("docs").count_rows() == 3


def test_recover_discards_temp_when_final_present(lance_env):
    """Crash while building the temp: original intact → temp discarded."""
    _add_chunks(lance_env, "docs", "/a.md", count=3)
    # Fabricate a leftover temp from an aborted build.
    _add_chunks(lance_env, "docs__vacuum", "/x.md", count=1)
    assert os.path.exists(lance_env._table_dir("docs__vacuum"))

    lance_env.recover_vacuum_temp("docs")

    assert not os.path.exists(lance_env._table_dir("docs__vacuum"))
    # Original data untouched.
    assert lance_env.open_table("docs").count_rows() == 3


def test_recover_promotes_temp_when_final_missing(lance_env):
    """Crash between drop and rename: final gone, fully-built temp → promoted."""
    # The temp is the verified replacement; the original was already dropped.
    _add_chunks(lance_env, "docs__vacuum", "/a.md", count=4)
    final = lance_env._table_dir("docs")
    assert not os.path.exists(final)
    assert os.path.exists(lance_env._table_dir("docs__vacuum"))

    lance_env.recover_vacuum_temp("docs")

    assert os.path.exists(final)
    assert not os.path.exists(lance_env._table_dir("docs__vacuum"))
    t = lance_env.open_table("docs")
    assert t.count_rows() == 4


def test_recover_noop_without_temp(lance_env):
    """No leftover temp → recovery is a cheap no-op that leaves data alone."""
    _add_chunks(lance_env, "docs", "/a.md", count=2)
    lance_env.recover_vacuum_temp("docs")  # must not raise
    assert lance_env.open_table("docs").count_rows() == 2


def test_vacuum_clears_stale_temp_before_rebuild(lance_env):
    """A stale temp from a prior aborted run must not break a fresh vacuum."""
    _add_chunks(lance_env, "docs", "/a.md", count=3)
    # Leftover temp on disk from an earlier crash.
    _add_chunks(lance_env, "docs__vacuum", "/stale.md", count=9)

    assert lance_env.vacuum_collection("docs") is True
    assert not os.path.exists(lance_env._table_dir("docs__vacuum"))
    # The vacuum rebuilt from the *real* table, not the stale temp.
    assert lance_env.open_table("docs").count_rows() == 3
