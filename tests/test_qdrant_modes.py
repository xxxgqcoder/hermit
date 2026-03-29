"""Integration tests for Qdrant dual-mode (Local Mode & Stand-alone Mode).

Test strategy
-------------
* Mock embeddings (random numpy vectors) — no model loading required, fast CI.
* Local mode tests: fresh tmp qdrant dir, no Docker needed.
* Stand-alone mode tests: auto-launch qdrant/qdrant:v1.13 Docker on port 16333
  (non-conflicting with production 6333); skip gracefully if Docker unavailable.

Run:
    pytest tests/test_qdrant_modes.py -v
"""

import contextlib
import os
import subprocess
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
from qdrant_client import models as qmodels

# ── Constants ────────────────────────────────────────────────────

DENSE_DIM = 768
COLLECTION = "test_dual_mode"
STANDALONE_HTTP_PORT = 16333
STANDALONE_GRPC_PORT = 16334
DOCKER_CONTAINER = "hermit_test_qdrant"
# Matches qdrant-client 1.17.x in uv.lock
DOCKER_IMAGE = "qdrant/qdrant:v1.17.0"


# ── Mock sparse vector ────────────────────────────────────────────

@dataclass
class FakeSparseVec:
    """Minimal sparse vector compatible with _build_points()."""
    indices: np.ndarray
    values: np.ndarray


def _make_sparse() -> FakeSparseVec:
    idx = np.array([1, 42, 100], dtype=np.int32)
    vals = np.array([0.3, 0.7, 0.5], dtype=np.float32)
    return FakeSparseVec(indices=idx, values=vals)


def _make_dense() -> list[float]:
    rng = np.random.default_rng(seed=42)
    vec = rng.random(DENSE_DIM, dtype=np.float32)
    vec = vec / np.linalg.norm(vec)
    return vec.tolist()


def _build_test_points(n: int = 5, source_file: str = "test_doc.md", seed_offset: int = 0):
    rng = np.random.default_rng(seed=7 + seed_offset)
    ids, dense_vecs, sparse_vecs, payloads = [], [], [], []
    for i in range(n):
        # Qdrant local mode requires valid UUID strings for string point IDs
        ids.append(str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{source_file}-chunk-{i}")))
        v = rng.random(DENSE_DIM, dtype=np.float32)
        v = v / np.linalg.norm(v)
        dense_vecs.append(v.tolist())
        sparse_vecs.append(_make_sparse())
        payloads.append({
            "text": f"这是第 {i + 1} 个测试文本块",
            "source_file": source_file,
            "chunk_index": i,
            "total_chunks": n,
        })
    return ids, dense_vecs, sparse_vecs, payloads


# ── Module reset fixture ─────────────────────────────────────────

@pytest.fixture(autouse=True)
def reset_qdrant_module(tmp_path, monkeypatch):
    """Reset all global singletons in the qdrant storage module between tests."""
    import hermit.storage.qdrant as qmod
    import hermit.config as cfg

    # Point DATA_ROOT at isolated tmp dir so tests never touch ~/.hermit/data
    monkeypatch.setattr(cfg, "DATA_ROOT", tmp_path / "data")
    monkeypatch.setattr(qmod, "DATA_ROOT", tmp_path / "data")

    # Reset module-level singletons
    monkeypatch.setattr(qmod, "_client", None)
    monkeypatch.setattr(qmod, "_standalone_mode", False)
    monkeypatch.setattr(qmod, "_app_lock_fd", None)
    monkeypatch.setattr(qmod, "_docker_atexit_registered", False)

    yield

    # Teardown: close client and release lock
    if qmod._client is not None:
        with contextlib.suppress(Exception):
            qmod._client.close()
        qmod._client = None
    qmod._release_app_lock()


# ── Docker fixture ───────────────────────────────────────────────

def _wait_for_port(host: str, port: int, timeout: float = 30.0) -> bool:
    import socket
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(1.0)
            if s.connect_ex((host, port)) == 0:
                return True
        time.sleep(0.5)
    return False


@pytest.fixture(scope="session")
def qdrant_docker(tmp_path_factory):
    """Start a Qdrant Docker container via Hermit's managed lifecycle.

    Yields (host, http_port, grpc_port) if Docker is available and the
    container starts successfully; yields None otherwise (tests will skip).
    """
    import hermit.storage.qdrant_docker as docker_mod

    if not docker_mod._is_docker_available():
        yield None
        return

    # Clean up any leftover container from a previous run
    subprocess.run(["docker", "rm", "-f", DOCKER_CONTAINER], capture_output=True)

    data_path = tmp_path_factory.mktemp("qdrant_standalone_data")
    try:
        docker_mod.ensure_qdrant_running(
            host="localhost",
            port=STANDALONE_HTTP_PORT,
            grpc_port=STANDALONE_GRPC_PORT,
            qdrant_data_path=data_path,
            container_name=DOCKER_CONTAINER,
            image=DOCKER_IMAGE,
        )
    except RuntimeError as exc:
        print(f"\nWarning: Could not start Qdrant container: {exc}")
        yield None
        return

    yield ("localhost", STANDALONE_HTTP_PORT, STANDALONE_GRPC_PORT)

    docker_mod.stop_qdrant_container(DOCKER_CONTAINER)
    docker_mod._container_created = False


# ── Local mode tests ─────────────────────────────────────────────

class TestLocalMode:

    def test_collection_create_and_query(self, tmp_path, monkeypatch):
        """Local mode: create collection, upsert points, query returns results."""
        import hermit.storage.qdrant as qmod
        import hermit.config as cfg

        # Ensure QDRANT_HOST is unset (local mode)
        monkeypatch.setattr(cfg, "QDRANT_HOST", None)

        qmod.ensure_collection(COLLECTION)
        assert qmod._standalone_mode is False

        ids, dense_vecs, sparse_vecs, payloads = _build_test_points(5)
        qmod.upsert_chunks(COLLECTION, ids, dense_vecs, sparse_vecs, payloads)

        # Query with a random dense vector
        query_vec = _make_dense()
        results = qmod.query_points(
            COLLECTION,
            query=query_vec,
            using="dense",
            limit=3,
            with_payload=True,
        ).points

        assert len(results) > 0
        assert results[0].payload["source_file"] == "test_doc.md"

    def test_collection_delete_by_source_file(self, monkeypatch):
        """Local mode: delete_by_source_file removes only matching points."""
        import hermit.storage.qdrant as qmod
        import hermit.config as cfg

        monkeypatch.setattr(cfg, "QDRANT_HOST", None)

        qmod.ensure_collection(COLLECTION)

        # Insert 3 chunks from file_a and 2 from file_b
        ids_a, dv_a, sv_a, pay_a = _build_test_points(3, "file_a.md")
        ids_b, dv_b, sv_b, pay_b = _build_test_points(2, "file_b.md")
        # IDs are UUID5 keyed by filename+index, so they won't collide

        qmod.upsert_chunks(COLLECTION, ids_a, dv_a, sv_a, pay_a)
        qmod.upsert_chunks(COLLECTION, ids_b, dv_b, sv_b, pay_b)

        qmod.delete_by_source_file(COLLECTION, "file_a.md")

        c = qmod.client()
        remaining = c.scroll(
            collection_name=COLLECTION,
            scroll_filter=qmodels.Filter(
                must=[qmodels.FieldCondition(
                    key="source_file",
                    match=qmodels.MatchValue(value="file_a.md"),
                )]
            ),
            with_payload=True,
            limit=100,
        )[0]
        assert len(remaining) == 0

        remaining_b = c.scroll(
            collection_name=COLLECTION,
            scroll_filter=qmodels.Filter(
                must=[qmodels.FieldCondition(
                    key="source_file",
                    match=qmodels.MatchValue(value="file_b.md"),
                )]
            ),
            with_payload=True,
            limit=100,
        )[0]
        assert len(remaining_b) == 2

    def test_replace_file_chunks(self, monkeypatch):
        """Local mode: replace_file_chunks atomically replaces points for a file."""
        import hermit.storage.qdrant as qmod
        import hermit.config as cfg

        monkeypatch.setattr(cfg, "QDRANT_HOST", None)
        qmod.ensure_collection(COLLECTION)

        # Initial insert
        ids, dv, sv, pay = _build_test_points(3, "replaceable.md")
        qmod.upsert_chunks(COLLECTION, ids, dv, sv, pay)

        # Now replace with 2 new chunks
        rng = np.random.default_rng(seed=99)
        new_ids = [
            str(uuid.uuid5(uuid.NAMESPACE_DNS, f"replaceable.md-new-{i}"))
            for i in range(2)
        ]
        new_dv = []
        for _ in new_ids:
            v = rng.random(DENSE_DIM, dtype=np.float32)
            v = v / np.linalg.norm(v)
            new_dv.append(v.tolist())
        new_sv = [_make_sparse() for _ in new_ids]
        new_pay = [
            {"text": f"替换后文本 {i}", "source_file": "replaceable.md",
             "chunk_index": i, "total_chunks": 2}
            for i in range(2)
        ]
        qmod.replace_file_chunks(COLLECTION, "replaceable.md", new_ids, new_dv, new_sv, new_pay)

        c = qmod.client()
        pts = c.scroll(
            collection_name=COLLECTION,
            with_payload=True,
            limit=100,
        )[0]
        assert len(pts) == 2
        assert all(p.payload["source_file"] == "replaceable.md" for p in pts)

    def test_port_conflict_raises(self, tmp_path, monkeypatch):
        """Local mode: RuntimeError raised when Qdrant ports are already in use."""
        import hermit.storage.qdrant as qmod
        import hermit.config as cfg

        monkeypatch.setattr(cfg, "QDRANT_HOST", None)

        # Simulate a port being occupied
        import socket as _socket
        original_connect_ex = _socket.socket.connect_ex

        def _fake_connect_ex(self, addr):
            if addr[1] in (cfg.QDRANT_PORT, cfg.QDRANT_GRPC_PORT):
                return 0  # pretend the port is open
            return original_connect_ex(self, addr)

        monkeypatch.setattr(_socket.socket, "connect_ex", _fake_connect_ex)

        with pytest.raises(RuntimeError, match="端口.*已被占用"):
            qmod.get_client()

    def test_app_lock_acquired_in_local_mode(self, monkeypatch):
        """Local mode: app-level file lock is acquired after client initialisation."""
        import hermit.storage.qdrant as qmod
        import hermit.config as cfg

        monkeypatch.setattr(cfg, "QDRANT_HOST", None)
        qmod.client()  # triggers get_client()

        assert qmod._app_lock_fd is not None
        assert qmod._standalone_mode is False

    def test_delete_collection(self, monkeypatch):
        """Local mode: delete_collection removes the collection from Qdrant."""
        import hermit.storage.qdrant as qmod
        import hermit.config as cfg

        monkeypatch.setattr(cfg, "QDRANT_HOST", None)

        qmod.ensure_collection(COLLECTION)
        assert qmod.client().collection_exists(COLLECTION)

        qmod.delete_collection(COLLECTION)
        assert not qmod.client().collection_exists(COLLECTION)


# ── Lock-bypass logic tests ───────────────────────────────────────

class TestLockBypassLogic:

    def test_get_lock_local_mode_returns_real_lock(self, monkeypatch):
        """_get_lock() should return the real threading.Lock in local mode."""
        import threading
        import hermit.storage.qdrant as qmod

        monkeypatch.setattr(qmod, "_standalone_mode", False)
        lock = qmod._get_lock()
        assert isinstance(lock, type(threading.Lock()))

    def test_get_lock_standalone_mode_returns_nullcontext(self, monkeypatch):
        """_get_lock() should return nullcontext in standalone mode (no lock overhead)."""
        import hermit.storage.qdrant as qmod

        monkeypatch.setattr(qmod, "_standalone_mode", True)
        lock = qmod._get_lock()
        # nullcontext is a context manager but NOT a threading.Lock
        import threading
        assert not isinstance(lock, type(threading.Lock()))
        # Verify it's a no-op context manager
        with lock:
            pass  # should not raise


# ── Stand-alone mode tests ───────────────────────────────────────

class TestStandaloneMode:

    @pytest.fixture(autouse=True)
    def _patch_standalone(self, monkeypatch, qdrant_docker):
        """Configure QDRANT_HOST/PORT env vars to point at Docker container."""
        if qdrant_docker is None:
            pytest.skip("Docker unavailable or Qdrant container failed to start")

        host, http_port, grpc_port = qdrant_docker
        import hermit.config as cfg

        monkeypatch.setattr(cfg, "QDRANT_HOST", host)
        monkeypatch.setattr(cfg, "QDRANT_PORT", http_port)
        monkeypatch.setattr(cfg, "QDRANT_GRPC_PORT", grpc_port)
        monkeypatch.setattr(cfg, "QDRANT_CONTAINER_NAME", DOCKER_CONTAINER)
        # Container is already running via session fixture; disable managed auto-start
        # so individual tests don't re-trigger Docker logic.
        monkeypatch.setattr(cfg, "QDRANT_MANAGED", False)

    def test_standalone_mode_flag_set(self, monkeypatch, qdrant_docker):
        """Stand-alone mode: _standalone_mode is True after get_client()."""
        import hermit.storage.qdrant as qmod

        qmod.client()
        assert qmod._standalone_mode is True

    def test_no_app_lock_in_standalone_mode(self, monkeypatch, qdrant_docker):
        """Stand-alone mode: app-level file lock is NOT acquired."""
        import hermit.storage.qdrant as qmod

        qmod.client()
        assert qmod._app_lock_fd is None

    def test_standalone_collection_create_and_query(self, monkeypatch, qdrant_docker):
        """Stand-alone mode: create collection, upsert, query returns results."""
        import hermit.storage.qdrant as qmod

        # Use unique name to avoid collision with parallel runs
        col = f"{COLLECTION}_standalone"

        # Clean up any previous run
        if qmod.client().collection_exists(col):
            qmod.delete_collection(col)

        qmod.ensure_collection(col)

        ids, dense_vecs, sparse_vecs, payloads = _build_test_points(5, "standalone_doc.md")
        qmod.upsert_chunks(col, ids, dense_vecs, sparse_vecs, payloads)

        query_vec = _make_dense()
        results = qmod.query_points(
            col,
            query=query_vec,
            using="dense",
            limit=3,
            with_payload=True,
        ).points

        assert len(results) > 0
        assert results[0].payload["source_file"] == "standalone_doc.md"
        assert qmod._standalone_mode is True

        # Cleanup
        qmod.delete_collection(col)

    def test_standalone_delete_by_source_file(self, monkeypatch, qdrant_docker):
        """Stand-alone mode: delete_by_source_file works correctly."""
        import hermit.storage.qdrant as qmod

        col = f"{COLLECTION}_standalone_del"
        if qmod.client().collection_exists(col):
            qmod.delete_collection(col)
        qmod.ensure_collection(col)

        ids_a, dv_a, sv_a, pay_a = _build_test_points(3, "sa_file_a.md")
        ids_b, dv_b, sv_b, pay_b = _build_test_points(2, "sa_file_b.md")
        # IDs are UUID5 keyed by filename+index, so they won't collide

        qmod.upsert_chunks(col, ids_a, dv_a, sv_a, pay_a)
        qmod.upsert_chunks(col, ids_b, dv_b, sv_b, pay_b)
        qmod.delete_by_source_file(col, "sa_file_a.md")

        c = qmod.client()
        remaining_a = c.scroll(
            collection_name=col,
            scroll_filter=qmodels.Filter(
                must=[qmodels.FieldCondition(
                    key="source_file",
                    match=qmodels.MatchValue(value="sa_file_a.md"),
                )]
            ),
            limit=100,
        )[0]
        assert len(remaining_a) == 0

        remaining_b = c.scroll(
            collection_name=col,
            scroll_filter=qmodels.Filter(
                must=[qmodels.FieldCondition(
                    key="source_file",
                    match=qmodels.MatchValue(value="sa_file_b.md"),
                )]
            ),
            limit=100,
        )[0]
        assert len(remaining_b) == 2

        qmod.delete_collection(col)


# ── Managed container lifecycle tests ───────────────────────────────────────

class TestManagedContainerLifecycle:
    """Tests for Hermit-managed Docker container startup/shutdown."""

    def test_ensure_running_removes_and_recreates(self, qdrant_docker, tmp_path):
        """ensure_qdrant_running() removes any existing container and creates a fresh one."""
        if qdrant_docker is None:
            pytest.skip("Docker unavailable")

        import hermit.storage.qdrant_docker as docker_mod

        # Call a second time — should rm -f the existing container and run a new one
        docker_mod.ensure_qdrant_running(
            host="localhost",
            port=STANDALONE_HTTP_PORT,
            grpc_port=STANDALONE_GRPC_PORT,
            qdrant_data_path=tmp_path / "qdrant",
            container_name=DOCKER_CONTAINER,
            image=DOCKER_IMAGE,
        )
        # Container was re-created successfully
        assert docker_mod._container_created is True
        assert _wait_for_port("localhost", STANDALONE_HTTP_PORT, timeout=5.0)

    def test_managed_mode_get_client_registers_atexit(self, monkeypatch, qdrant_docker):
        """get_client() in managed standalone mode sets _docker_atexit_registered."""
        if qdrant_docker is None:
            pytest.skip("Docker unavailable")

        import hermit.storage.qdrant as qmod
        import hermit.config as cfg

        monkeypatch.setattr(cfg, "QDRANT_HOST", "localhost")
        monkeypatch.setattr(cfg, "QDRANT_PORT", STANDALONE_HTTP_PORT)
        monkeypatch.setattr(cfg, "QDRANT_GRPC_PORT", STANDALONE_GRPC_PORT)
        monkeypatch.setattr(cfg, "QDRANT_CONTAINER_NAME", DOCKER_CONTAINER)
        monkeypatch.setattr(cfg, "QDRANT_IMAGE", DOCKER_IMAGE)
        monkeypatch.setattr(cfg, "QDRANT_MANAGED", True)

        c = qmod.client()  # triggers ensure_qdrant_running (idempotent — port already open)
        assert qmod._standalone_mode is True
        assert qmod._docker_atexit_registered is True
        assert c is not None

    def test_unmanaged_mode_skips_docker(self, monkeypatch, qdrant_docker):
        """get_client() with QDRANT_MANAGED=False skips Docker management entirely."""
        if qdrant_docker is None:
            pytest.skip("Docker unavailable")

        import hermit.storage.qdrant as qmod
        import hermit.config as cfg

        monkeypatch.setattr(cfg, "QDRANT_HOST", "localhost")
        monkeypatch.setattr(cfg, "QDRANT_PORT", STANDALONE_HTTP_PORT)
        monkeypatch.setattr(cfg, "QDRANT_GRPC_PORT", STANDALONE_GRPC_PORT)
        monkeypatch.setattr(cfg, "QDRANT_MANAGED", False)

        qmod.client()
        assert qmod._docker_atexit_registered is False

    def test_no_docker_raises_friendly_error(self, monkeypatch, tmp_path):
        """ensure_qdrant_running raises RuntimeError with helpful message when docker absent."""
        import hermit.storage.qdrant_docker as docker_mod

        monkeypatch.setattr(docker_mod, "_is_docker_available", lambda: False)

        with pytest.raises(RuntimeError, match="Docker"):
            docker_mod.ensure_qdrant_running(
                host="localhost",
                port=19999,
                grpc_port=20000,
                qdrant_data_path=tmp_path / "qdrant",
                container_name="no_container",
                image=DOCKER_IMAGE,
            )

    def test_stop_is_noop_if_not_created(self):
        """stop_qdrant_container() is a no-op when _container_created is False."""
        import hermit.storage.qdrant_docker as docker_mod

        original = docker_mod._container_created
        try:
            docker_mod._container_created = False
            # Should not invoke docker; no assertion needed beyond no exception
            docker_mod.stop_qdrant_container("ghost_container")
        finally:
            docker_mod._container_created = original

