"""Tests for scanner three-way diff, task queue, registry, and task status API."""

import json
import os
import shutil
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ── Registry tests ──────────────────────────────────────────────


@pytest.fixture
def tmp_data_root(tmp_path, monkeypatch):
    monkeypatch.setattr("hermit.storage.registry.DATA_ROOT", tmp_path)
    monkeypatch.setattr("hermit.storage.registry._REGISTRY_PATH", tmp_path / "collections.json")
    return tmp_path


def test_registry_round_trip(tmp_data_root):
    from hermit.storage.registry import register, unregister, get_all

    assert get_all() == {}

    register("col1", "/tmp/docs")
    all_ = get_all()
    assert "col1" in all_
    assert all_["col1"]["folder_path"] == "/tmp/docs"

    register("col2", "/tmp/other")
    assert len(get_all()) == 2

    unregister("col1")
    all_ = get_all()
    assert "col1" not in all_
    assert "col2" in all_


def test_registry_unregister_missing(tmp_data_root):
    from hermit.storage.registry import unregister, get_all

    unregister("nonexistent")  # should not raise
    assert get_all() == {}


# ── Scanner three-way diff tests ────────────────────────────────


@pytest.fixture
def scan_env(tmp_path):
    """Set up a temp folder with files, mock qdrant/embedder/metadata."""
    folder = tmp_path / "docs"
    folder.mkdir()
    (folder / "a.md").write_text("hello world")
    (folder / "b.md").write_text("goodbye world")
    return folder


@patch("hermit.ingestion.scanner.qdrant")
@patch("hermit.ingestion.scanner.enqueue_index_task")
def test_scan_folder_deferred_new_files(mock_enqueue, mock_qdrant, scan_env, tmp_path, monkeypatch):
    """New files on disk should be enqueued when defer_indexing=True."""
    mock_enqueue.return_value = True

    # Patch MetadataStore to return empty (no previously indexed files)
    mock_meta = MagicMock()
    mock_meta.get_all_records.return_value = {}
    monkeypatch.setattr(
        "hermit.ingestion.scanner.MetadataStore",
        lambda name: mock_meta,
    )

    from hermit.ingestion.scanner import scan_folder

    stats = scan_folder("test", str(scan_env), defer_indexing=True)

    assert stats["added"] == 2
    assert stats["deleted"] == 0
    assert stats["updated"] == 0
    assert mock_enqueue.call_count == 2
    # qdrant.delete_by_source_file should NOT be called for new files
    mock_qdrant.delete_by_source_file.assert_not_called()


@patch("hermit.ingestion.scanner.qdrant")
@patch("hermit.ingestion.scanner.enqueue_index_task")
def test_scan_folder_deletion(mock_enqueue, mock_qdrant, scan_env, tmp_path, monkeypatch):
    """Files in SQLite but missing from disk should be deleted immediately."""
    gone_path = str(scan_env / "gone.md")  # does not exist on disk

    mock_meta = MagicMock()
    mock_meta.get_all_records.return_value = {
        gone_path: ("oldhash", 1000.0),
    }
    monkeypatch.setattr(
        "hermit.ingestion.scanner.MetadataStore",
        lambda name: mock_meta,
    )

    from hermit.ingestion.scanner import scan_folder

    stats = scan_folder("test", str(scan_env), defer_indexing=True)

    assert stats["deleted"] == 1
    mock_qdrant.delete_by_source_file.assert_any_call("test", gone_path)
    mock_meta.delete.assert_any_call(gone_path)


@patch("hermit.ingestion.scanner.qdrant")
@patch("hermit.ingestion.scanner.enqueue_index_task")
def test_scan_folder_hash_unchanged_skips(mock_enqueue, mock_qdrant, scan_env, tmp_path, monkeypatch):
    """Files with unchanged hash should not be enqueued or re-indexed."""
    from hermit.ingestion.scanner import _file_hash

    a_path = str(scan_env / "a.md")
    b_path = str(scan_env / "b.md")
    a_hash = _file_hash(Path(a_path))
    b_hash = _file_hash(Path(b_path))

    mock_meta = MagicMock()
    mock_meta.get_all_records.return_value = {
        a_path: (a_hash, 1000.0),
        b_path: (b_hash, 1000.0),
    }
    monkeypatch.setattr(
        "hermit.ingestion.scanner.MetadataStore",
        lambda name: mock_meta,
    )

    from hermit.ingestion.scanner import scan_folder

    stats = scan_folder("test", str(scan_env), defer_indexing=True)

    assert stats["added"] == 0
    assert stats["updated"] == 0
    assert stats["deleted"] == 0
    mock_enqueue.assert_not_called()


@patch("hermit.ingestion.scanner.qdrant")
@patch("hermit.ingestion.scanner.enqueue_index_task")
def test_scan_folder_hash_changed_enqueues(mock_enqueue, mock_qdrant, scan_env, tmp_path, monkeypatch):
    """Files with changed hash should be enqueued for re-indexing."""
    mock_enqueue.return_value = True
    a_path = str(scan_env / "a.md")
    b_path = str(scan_env / "b.md")

    mock_meta = MagicMock()
    mock_meta.get_all_records.return_value = {
        a_path: ("stale_hash", 1000.0),
        b_path: ("stale_hash", 1000.0),
    }
    monkeypatch.setattr(
        "hermit.ingestion.scanner.MetadataStore",
        lambda name: mock_meta,
    )

    from hermit.ingestion.scanner import scan_folder

    stats = scan_folder("test", str(scan_env), defer_indexing=True)

    assert stats["updated"] == 2
    assert mock_enqueue.call_count == 2


@patch("hermit.ingestion.scanner.qdrant")
@patch("hermit.ingestion.scanner._index_file")
def test_scan_folder_sync_mode(mock_index, mock_qdrant, scan_env, tmp_path, monkeypatch):
    """When defer_indexing=False, _index_file should be called directly."""
    mock_index.return_value = True

    mock_meta = MagicMock()
    mock_meta.get_all_records.return_value = {}
    monkeypatch.setattr(
        "hermit.ingestion.scanner.MetadataStore",
        lambda name: mock_meta,
    )

    from hermit.ingestion.scanner import scan_folder

    stats = scan_folder("test", str(scan_env), defer_indexing=False)

    assert stats["added"] == 2
    assert mock_index.call_count == 2


# ── Task queue tests ────────────────────────────────────────────


def test_task_queue_dedup():
    """Duplicate tasks for same (collection, file) should be rejected."""
    from hermit.ingestion.task_queue import IndexTask, _IndexTaskQueue

    q = _IndexTaskQueue()
    # Don't start worker — just test enqueue dedup logic
    q._worker = threading.Thread()  # fake alive check
    q._worker.start = lambda: None

    task = IndexTask("col", "/tmp/a.md")
    # Manually add to pending to simulate enqueue without worker
    q._pending.add((task.collection_name, task.file_path))
    assert q.enqueue(task) is False  # should reject duplicate


def test_task_queue_status():
    """get_status should report correct counts."""
    from hermit.ingestion.task_queue import _IndexTaskQueue

    q = _IndexTaskQueue()
    q._pending = {("col1", "/a"), ("col1", "/b"), ("col2", "/c")}
    q._in_progress = {("col1", "/a")}

    status = q.get_status("col1")
    assert status["pending_tasks"] == 2
    assert status["in_progress_tasks"] == 1
    assert status["queued_tasks"] == 1

    status2 = q.get_status("col2")
    assert status2["pending_tasks"] == 1
    assert status2["in_progress_tasks"] == 0
    assert status2["queued_tasks"] == 1


# ── API route tests ─────────────────────────────────────────────


@pytest.fixture
def client():
    """Create a TestClient with mocked heavy dependencies."""
    with patch("hermit.retrieval.embedder.warmup"), \
         patch("hermit.retrieval.reranker.warmup"), \
         patch("hermit.ingestion.task_queue.start_task_worker"):
        from hermit.app import app
        from fastapi.testclient import TestClient
        return TestClient(app)


def test_tasks_endpoint_404(client):
    resp = client.get("/collections/nonexistent/tasks")
    assert resp.status_code == 404


@patch("hermit.api.routes.get_collection_task_status")
def test_tasks_endpoint_ok(mock_status, client):
    from hermit.api.routes import _collections

    _collections["demo"] = {"folder_path": "/tmp"}
    mock_status.return_value = {
        "collection": "demo",
        "pending_tasks": 3,
        "queued_tasks": 2,
        "in_progress_tasks": 1,
        "worker_alive": True,
    }

    try:
        resp = client.get("/collections/demo/tasks")
        assert resp.status_code == 200
        data = resp.json()
        assert data["collection"] == "demo"
        assert data["pending_tasks"] == 3
        assert data["queued_tasks"] == 2
        assert data["in_progress_tasks"] == 1
        assert data["worker_alive"] is True
    finally:
        _collections.pop("demo", None)
