"""Unified end-to-end deployment tests for Hermit.

Covers both deployment modes — Local (embedded Qdrant) and Stand-alone
(Hermit-managed Docker container) — in a single suite so that whatever
holds for one path is also exercised on the other. Mode-switching
scenarios (Local → Stand-alone with the same data directory) live here
too, since they are the whole point of having a unified test file.

Run with:
    pytest tests/test_e2e_deployment.py -v
    pytest tests/test_e2e_deployment.py -v -k local        # only local mode
    pytest tests/test_e2e_deployment.py -v -k standalone   # only standalone

A working Docker daemon is REQUIRED — the deployment suite refuses to
run without it (``pytest.exit`` at module load). The pinned
``TEST_QDRANT_IMAGE`` is auto-pulled once if not yet cached.

Test groups
───────────
1.  Smoke per mode             — clean boot, /health, isolation
2.  KB lifecycle per mode      — empty boot → add → index → search
3.  Persistence per mode       — restart preserves data
4.  Recovery per mode          — SIGKILL during indexing, KB folder
                                  emptied before restart
5.  Mode-switching             — Local → Stand-alone reuses the same
                                  ``~/.hermit/data/qdrant`` directory
6.  Stand-alone-only           — IPv6 startup, container persistence
                                  (stopped not removed), orphan adoption,
                                  atexit container stop, friendly errors
                                  when Docker / image unavailable
7.  Image-pin invariants       — no Docker required; assert tag is fixed
                                  and matches ``hermit.config.QDRANT_IMAGE``
8.  Embedding cache (#27)      — second add of the same KB is dramatically
                                  faster and writes no new cache entries
9.  KB lifecycle edge cases    — remove during indexing, incremental file
                                  modification, multi-collection isolation,
                                  search quality regression
10. Graceful shutdown / concurrency — SIGTERM mid-indexing, search under
                                       indexing load
11. Reverse mode-switching     — Stand-alone → Local symmetry
12. Stand-alone resilience     — `docker restart` of the Qdrant container
                                  while Hermit is up
13. Server lifecycle           — port-conflict fallback when port.json
                                  points at an occupied port
"""

import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
import uuid
import urllib.request
from pathlib import Path

import pytest

from tests.conftest import (
    _link_models,
    _poll_health,
    _poll_health_while_process_alive,
    _read_port_file,
    _terminate_hermit_server,
)


# ── Constants ────────────────────────────────────────────────────

# Fixed Qdrant image. Must use an explicit ``vX.Y.Z`` tag (no ``latest``)
# to keep the on-disk format stable per design-doc §3.3.
# Aligned with the qdrant-client major version pinned in uv.lock (1.17.x).
TEST_QDRANT_IMAGE = "qdrant/qdrant:v1.17.0"


# ── Docker requirement ───────────────────────────────────────────
#
# Deployment tests REQUIRE a working Docker daemon. The standalone
# deployment path is not optional coverage — it is a first-class mode
# of Hermit, so we refuse to skip it when the host is missing Docker.
# A missing daemon is a setup error, not a CI signal to ignore.
#
# The pinned image is auto-pulled once at session start if not cached.


def _ensure_docker_daemon() -> None:
    """Abort the test session if Docker is not usable."""
    if not shutil.which("docker"):
        pytest.exit(
            "Docker CLI not found on PATH. Deployment tests require a "
            "working Docker daemon — install Docker Desktop (or the "
            "engine) and retry.",
            returncode=2,
        )
    try:
        subprocess.run(
            ["docker", "info"], check=True, capture_output=True, timeout=10,
        )
    except (subprocess.SubprocessError, FileNotFoundError) as exc:
        pytest.exit(
            f"Docker daemon is not responding ({exc}). Deployment tests "
            "require a running Docker daemon — start Docker Desktop "
            "(or `systemctl start docker`) and retry.",
            returncode=2,
        )


def _ensure_pinned_image() -> None:
    """Make sure ``TEST_QDRANT_IMAGE`` is cached locally; pull it if not."""
    inspect = subprocess.run(
        ["docker", "image", "inspect", TEST_QDRANT_IMAGE],
        capture_output=True,
    )
    if inspect.returncode == 0:
        return
    print(
        f"\n[deployment-tests] Image '{TEST_QDRANT_IMAGE}' not cached — "
        "pulling once for the session...",
        flush=True,
    )
    pull = subprocess.run(["docker", "pull", TEST_QDRANT_IMAGE])
    if pull.returncode != 0:
        pytest.exit(
            f"Failed to pull '{TEST_QDRANT_IMAGE}' (exit {pull.returncode}). "
            "Deployment tests require this image — check network access "
            "or override QDRANT_IMAGE locally and update TEST_QDRANT_IMAGE.",
            returncode=2,
        )


_ensure_docker_daemon()
_ensure_pinned_image()


# ── HTTP / process helpers ───────────────────────────────────────


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _get(port: int, path: str) -> dict:
    url = f"http://127.0.0.1:{port}{path}"
    with urllib.request.urlopen(url, timeout=10) as resp:
        return json.loads(resp.read())


def _post(port: int, path: str, body: dict) -> dict:
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


def _run_hermit(args: list[str], env: dict) -> tuple[int, dict]:
    """Run ``hermit <args>`` in a subprocess and parse the JSON stdout."""
    result = subprocess.run(
        [sys.executable, "-m", "hermit.cli"] + args,
        env=env, capture_output=True, text=True,
    )
    try:
        output = json.loads(result.stdout.strip())
    except json.JSONDecodeError:
        output = {"raw": result.stdout.strip(), "stderr": result.stderr.strip()}
    return result.returncode, output


def _poll_indexing_done(port: int, collection: str, timeout: int = 60) -> bool:
    """Poll until the indexing queue for *collection* is drained."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            data = _get(port, f"/collections/{collection}/tasks")
            if data.get("pending_tasks", 1) == 0:
                return True
        except Exception:
            pass
        time.sleep(1)
    return False


def _start_server(env: dict, hermit_home: Path) -> tuple[subprocess.Popen, int]:
    """Launch ``hermit start`` and wait for ``/health`` ``ready``."""
    proc = subprocess.Popen(
        [sys.executable, "-m", "hermit.cli", "start"],
        env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    try:
        port = _read_port_file(hermit_home, timeout=10)
    except RuntimeError:
        proc.kill(); proc.wait()
        _terminate_hermit_server(hermit_home)
        raise

    timeout = int(env.get("HERMIT_START_TIMEOUT", 120))
    if not _poll_health_while_process_alive(port, proc, timeout=timeout):
        proc.kill(); proc.wait()
        _terminate_hermit_server(hermit_home)
        log_file = hermit_home / "logs" / "hermit.log"
        if log_file.exists():
            print("\n=== hermit.log (tail) ===\n", log_file.read_text()[-4000:])
        pytest.fail(f"Server did not become ready within {timeout}s")
    return proc, port


def _stop_server(env: dict, hermit_home: Path, proc: subprocess.Popen) -> None:
    """Best-effort graceful shutdown: ``hermit stop`` + reap PID."""
    subprocess.run(
        [sys.executable, "-m", "hermit.cli", "stop"],
        env=env, check=False, capture_output=True,
    )
    _terminate_hermit_server(hermit_home)
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
    (hermit_home / "hermit.pid").unlink(missing_ok=True)


# ── Docker helpers ───────────────────────────────────────────────


def _container_running(container_name: str) -> bool:
    r = subprocess.run(
        ["docker", "ps", "--filter", f"name={container_name}",
         "--format", "{{.Names}}"],
        capture_output=True, text=True,
    )
    return container_name in r.stdout


def _container_exists(container_name: str) -> bool:
    r = subprocess.run(
        ["docker", "ps", "-a", "--filter", f"name={container_name}",
         "--format", "{{.Names}}"],
        capture_output=True, text=True,
    )
    return container_name in r.stdout


def _container_image(container_name: str) -> str | None:
    r = subprocess.run(
        ["docker", "inspect", "--format", "{{.Config.Image}}", container_name],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        return None
    return r.stdout.strip()


def _wait_until(predicate, timeout: float = 15.0, step: float = 0.5) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(step)
    return False


def _remove_container(container_name: str) -> None:
    subprocess.run(
        ["docker", "rm", "-f", container_name],
        check=False, capture_output=True,
    )


# ── Env builders (one per deployment mode) ───────────────────────


def _make_local_env(hermit_home: Path) -> tuple[dict, str | None]:
    """Build env for local (embedded) Qdrant mode.

    Returns ``(env, container_name=None)`` so the caller can use the same
    cleanup path as the standalone version.
    """
    env = os.environ.copy()
    env.update({
        "HERMIT_HOME": str(hermit_home),
        "HERMIT_START_TIMEOUT": "120",
        # Set Qdrant ports to free ports so the local-mode port-probe
        # safeguard doesn't clash with a real Qdrant on 6333/6334.
        "QDRANT_PORT": str(_free_port()),
        "QDRANT_GRPC_PORT": str(_free_port()),
    })
    # Strip any inherited standalone settings to be sure we are in local mode.
    for k in ("QDRANT_HOST", "QDRANT_MANAGED", "QDRANT_CONTAINER_NAME"):
        env.pop(k, None)
    return env, None


def _make_standalone_env(
    hermit_home: Path,
    *,
    qdrant_host: str = "127.0.0.1",
    managed: bool = True,
    image: str = TEST_QDRANT_IMAGE,
    container_name: str | None = None,
    qdrant_port: int | None = None,
    qdrant_grpc_port: int | None = None,
) -> tuple[dict, str]:
    """Build env for standalone (Docker-managed Qdrant) mode.

    Always pins ``QDRANT_IMAGE`` to *image* so tests are reproducible
    regardless of any ``QDRANT_IMAGE`` exported in the developer's shell.
    """
    if container_name is None:
        container_name = f"hermit_qdrant_test_{uuid.uuid4().hex[:8]}"
    if qdrant_port is None:
        qdrant_port = _free_port()
    if qdrant_grpc_port is None:
        qdrant_grpc_port = _free_port()

    env = os.environ.copy()
    env.update({
        "HERMIT_HOME": str(hermit_home),
        "HERMIT_START_TIMEOUT": "300",  # docker pull can take >2 min on CI
        "QDRANT_HOST": qdrant_host,
        "QDRANT_MANAGED": "true" if managed else "false",
        "QDRANT_PORT": str(qdrant_port),
        "QDRANT_GRPC_PORT": str(qdrant_grpc_port),
        "QDRANT_CONTAINER_NAME": container_name,
        "QDRANT_IMAGE": image,
    })
    return env, container_name


# ── Mode-parametrized fixtures ───────────────────────────────────

# Both deployment modes are exercised by every parametrized test.
# Docker availability has already been enforced at module load — neither
# mode is conditionally skipped.
_MODE_PARAMS = [
    pytest.param("local", id="local"),
    pytest.param("standalone", id="standalone"),
]


@pytest.fixture(params=_MODE_PARAMS)
def deployment(request, tmp_path):
    """Yield a deployment context for one of the supported modes.

    Yields a dict with keys::

        {
            "mode": "local" | "standalone",
            "env": dict,                    # subprocess env
            "hermit_home": Path,
            "container_name": str | None,   # None in local mode
        }

    Tear-down: stops the server, force-removes the Docker container in
    standalone mode, and cleans the PID file. Always runs even on test
    failure.
    """
    mode = request.param
    hermit_home = tmp_path / f"hermit_home_{mode}"
    hermit_home.mkdir()
    _link_models(hermit_home)

    if mode == "local":
        env, container_name = _make_local_env(hermit_home)
    else:
        env, container_name = _make_standalone_env(hermit_home)

    yield {
        "mode": mode,
        "env": env,
        "hermit_home": hermit_home,
        "container_name": container_name,
    }

    # Always attempt graceful stop, even if no proc was tracked here —
    # tests usually start their own subprocesses and stop them explicitly,
    # but this catches anything left running.
    subprocess.run(
        [sys.executable, "-m", "hermit.cli", "stop"],
        env=env, check=False, capture_output=True,
    )
    _terminate_hermit_server(hermit_home)
    if container_name is not None:
        _remove_container(container_name)


@pytest.fixture()
def deployment_server(deployment):
    """Same as ``deployment``, but with a server already running.

    Yields ``(port, deployment)`` and stops the server in tear-down.
    """
    proc, port = _start_server(deployment["env"], deployment["hermit_home"])
    try:
        yield port, deployment
    finally:
        _stop_server(deployment["env"], deployment["hermit_home"], proc)


@pytest.fixture()
def standalone_only_server(tmp_path):
    """Standalone-only running server.

    Use this in tests that exercise behaviour unique to the Docker-managed
    path (image inspect, container lifecycle, ...). Yields the same shape
    as ``deployment_server`` so tests can read it the same way.
    """
    hermit_home = tmp_path / "hermit_home_standalone"
    hermit_home.mkdir()
    _link_models(hermit_home)
    env, container_name = _make_standalone_env(hermit_home)

    proc, port = _start_server(env, hermit_home)
    deployment = {
        "mode": "standalone",
        "env": env,
        "hermit_home": hermit_home,
        "container_name": container_name,
    }
    try:
        yield port, deployment
    finally:
        _stop_server(env, hermit_home, proc)
        _remove_container(container_name)


@pytest.fixture()
def large_docs_dir(tmp_path):
    """Many small Markdown files — enough to leave indexing in flight after boot.

    Used by the SIGKILL-during-indexing recovery scenario where we need
    pending tasks to still exist when we kill the server.
    """
    docs = tmp_path / "large_docs"
    docs.mkdir()
    for i in range(40):
        body = (
            f"# Document {i}\n\n"
            "Hermit indexes Markdown documents semantically.\n"
            f"This is body paragraph A for document number {i}.\n\n"
            "## Section 2\n\n"
            f"Paragraph B mentions hybrid retrieval and reranking — doc {i}.\n"
            f"Some more filler text repeated across files: knowledge base, embedding, search. {i}\n"
        )
        (docs / f"doc_{i:03d}.md").write_text(body)
    return docs


# ────────────────────────────────────────────────────────────────
#                      1. Smoke per mode
# ────────────────────────────────────────────────────────────────


def test_server_starts_clean(deployment_server):
    """Server boots cleanly from an empty HERMIT_HOME and reports ready."""
    port, deployment = deployment_server
    data = _get(port, "/health")
    assert data["status"] == "ready"
    assert data["models_loaded"] is True
    assert data["qdrant_mode"] == (
        "standalone" if deployment["mode"] == "standalone" else "local"
    )


def test_local_isolation_from_default_home(tmp_path):
    """Local mode never touches ``~/.hermit/`` when HERMIT_HOME is set."""
    hermit_home = tmp_path / "hermit_iso"
    hermit_home.mkdir()
    _link_models(hermit_home)
    env, _ = _make_local_env(hermit_home)

    default_pid = Path.home() / ".hermit" / "hermit.pid"
    before_mtime = default_pid.stat().st_mtime if default_pid.exists() else None

    proc, _port = _start_server(env, hermit_home)
    try:
        assert (hermit_home / "hermit.pid").exists()
        if before_mtime is not None:
            assert default_pid.stat().st_mtime == before_mtime, (
                "Default ~/.hermit/hermit.pid was modified during isolated test"
            )
    finally:
        _stop_server(env, hermit_home, proc)


# ────────────────────────────────────────────────────────────────
#                  2. KB lifecycle per mode
# ────────────────────────────────────────────────────────────────


def test_empty_kb_then_add_then_search(deployment, test_docs_dir):
    """Empty deploy → ``kb add`` → indexing → search returns content.

    Boots the server with no collections, registers one via the CLI,
    waits for indexing, then verifies search hits and ``/collections/<>``
    bookkeeping.
    """
    env = deployment["env"]
    hermit_home = deployment["hermit_home"]

    proc, port = _start_server(env, hermit_home)
    try:
        # Empty boot — no collections registered yet
        health = _get(port, "/health")
        assert health["collections"] == [] or all(
            c.get("total_chunks", 0) == 0 for c in health["collections"]
        )

        rc, output = _run_hermit(["kb", "add", "lifecycle_col", str(test_docs_dir)], env=env)
        assert rc == 0, f"kb add failed: {output}"
        assert output.get("status") == "added"

        # Trigger a manual sync so we don't wait on the 15-min poll loop
        sync = _post(port, "/collections/lifecycle_col/sync", {})
        assert "added" in sync

        assert _poll_indexing_done(port, "lifecycle_col", timeout=60), (
            "Indexing did not complete within 60s"
        )

        status = _get(port, "/collections/lifecycle_col/status")
        assert status["indexed_files"] >= 1
        assert status["total_chunks"] > 0

        results = _post(port, "/search", {
            "collection": "lifecycle_col",
            "query": "hermit knowledge base",
            "top_k": 3,
        })["results"]
        assert len(results) > 0, "Expected at least one search hit"
        for r in results:
            assert r.get("text"), "Search result missing text field"
    finally:
        _stop_server(env, hermit_home, proc)


# ────────────────────────────────────────────────────────────────
#                  3. Persistence per mode
# ────────────────────────────────────────────────────────────────


def test_data_persists_across_restart(deployment, test_docs_dir):
    """Stop and re-start the server: vectors and metadata must survive.

    For Local mode this exercises the embedded SQLite + on-disk Qdrant
    files. For Stand-alone mode it exercises the host-mounted volume +
    persistent-container design (§2 of the standalone design doc).
    """
    env = deployment["env"]
    hermit_home = deployment["hermit_home"]

    rc, output = _run_hermit(["kb", "add", "persist_col", str(test_docs_dir)], env=env)
    assert rc == 0, output

    # ── First boot: index ────────────────────────────────────
    proc, port = _start_server(env, hermit_home)
    first_chunks: int | None = None
    first_files: int | None = None
    try:
        assert _poll_indexing_done(port, "persist_col", timeout=60)
        s = _get(port, "/collections/persist_col/status")
        first_chunks = s["total_chunks"]
        first_files = s["indexed_files"]
        assert first_chunks > 0, s
        first_results = _post(port, "/search", {
            "collection": "persist_col",
            "query": "hermit knowledge base",
            "top_k": 3,
        })["results"]
        assert len(first_results) > 0
    finally:
        _stop_server(env, hermit_home, proc)

    # ── Second boot: data should still be there ─────────────
    proc, port = _start_server(env, hermit_home)
    try:
        assert _poll_indexing_done(port, "persist_col", timeout=60)
        s = _get(port, "/collections/persist_col/status")
        assert s["total_chunks"] == first_chunks, (
            f"Chunk count drift across restart: {first_chunks} → {s['total_chunks']}"
        )
        assert s["indexed_files"] == first_files
        results = _post(port, "/search", {
            "collection": "persist_col",
            "query": "hermit knowledge base",
            "top_k": 3,
        })["results"]
        assert len(results) > 0, "Search returned nothing — Qdrant data was lost"
    finally:
        _stop_server(env, hermit_home, proc)


# ────────────────────────────────────────────────────────────────
#                  4. Recovery per mode
# ────────────────────────────────────────────────────────────────


def test_indexing_interrupted_then_recovered(deployment, large_docs_dir):
    """SIGKILL the server while indexing is still in flight, then restart.

    Exercises the design invariant that ``metadata.upsert`` happens
    *after* a successful Qdrant write per file, so a mid-batch crash
    leaves orphan-free state: the next boot's startup scan re-detects
    the un-indexed files and re-queues them. Final state must be a
    fully indexed collection with no errors in the log.
    """
    env = deployment["env"]
    hermit_home = deployment["hermit_home"]

    rc, output = _run_hermit(
        ["kb", "add", "recover_col", str(large_docs_dir)], env=env,
    )
    assert rc == 0, output

    # ── First boot: kill mid-indexing ───────────────────────
    proc, port = _start_server(env, hermit_home)
    interrupted = False
    try:
        # Wait briefly for the worker to actually start chewing through tasks
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            try:
                tasks = _get(port, "/collections/recover_col/tasks")
                if tasks.get("pending_tasks", 0) > 0:
                    interrupted = True
                    break
            except Exception:
                pass
            time.sleep(0.1)

        # SIGKILL the server (no chance to checkpoint) by reading the PID file
        pid_file = hermit_home / "hermit.pid"
        server_pid = int(pid_file.read_text().strip())
        try:
            os.kill(server_pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        # Wait for the process to actually die
        _wait_until(
            lambda: not _proc_alive(server_pid),
            timeout=5.0, step=0.1,
        )
    finally:
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        # Stale pid file blocks the next start; remove it.
        (hermit_home / "hermit.pid").unlink(missing_ok=True)

    # We don't strictly require *catching* the in-flight window — fast
    # machines can finish indexing before the kill — but log it so a
    # green run on a slow box still tells us what we exercised.
    if not interrupted:
        print("NOTE: indexing finished before kill window; testing clean reboot only")

    # ── Second boot: must come up cleanly and finish indexing ──
    proc, port = _start_server(env, hermit_home)
    try:
        assert _poll_indexing_done(port, "recover_col", timeout=120), (
            "Post-kill indexing failed to drain"
        )
        status = _get(port, "/collections/recover_col/status")
        # All large_docs_dir files should ultimately be indexed
        on_disk = sum(1 for _ in large_docs_dir.glob("*.md"))
        assert status["indexed_files"] == on_disk, (
            f"Recovered indexed_files={status['indexed_files']} ≠ "
            f"on-disk files={on_disk}"
        )
        assert status["total_chunks"] > 0

        # Search still works
        results = _post(port, "/search", {
            "collection": "recover_col",
            "query": "hybrid retrieval reranking",
            "top_k": 3,
        })["results"]
        assert len(results) > 0
    finally:
        _stop_server(env, hermit_home, proc)


def test_kb_files_all_deleted_then_restart(deployment, test_docs_dir):
    """All KB files removed off-disk between two boots → graceful cleanup.

    The collection registration stays (it lives in ``collections.json``),
    but the next startup scan reports ``deleted=N`` and prunes both
    Qdrant points and SQLite metadata. Search on the now-empty collection
    returns no results without erroring.
    """
    env = deployment["env"]
    hermit_home = deployment["hermit_home"]

    rc, output = _run_hermit(["kb", "add", "ghost_col", str(test_docs_dir)], env=env)
    assert rc == 0, output

    # ── First boot: index normally ──────────────────────────
    proc, port = _start_server(env, hermit_home)
    initial_chunks: int | None = None
    try:
        assert _poll_indexing_done(port, "ghost_col", timeout=60)
        status = _get(port, "/collections/ghost_col/status")
        initial_chunks = status["total_chunks"]
        assert initial_chunks > 0
    finally:
        _stop_server(env, hermit_home, proc)

    # ── Wipe ALL files (folder remains, simulating user deletion) ──
    for child in test_docs_dir.iterdir():
        if child.is_file():
            child.unlink()
        else:
            shutil.rmtree(child)
    assert not any(test_docs_dir.iterdir()), "Pre-restart wipe incomplete"

    # ── Second boot: scan should observe the deletions ──────
    proc, port = _start_server(env, hermit_home)
    try:
        # Wait for any post-scan tasks to drain (deletes are inline,
        # so this is mostly a safety net)
        assert _poll_indexing_done(port, "ghost_col", timeout=30)

        status = _get(port, "/collections/ghost_col/status")
        assert status["indexed_files"] == 0, (
            f"Expected 0 indexed_files after wipe, got {status}"
        )
        assert status["total_chunks"] == 0, (
            f"Expected 0 chunks after wipe, got {status}"
        )

        # Collection still exists and is searchable — just empty
        response = _post(port, "/search", {
            "collection": "ghost_col",
            "query": "hermit knowledge base",
            "top_k": 3,
        })
        assert response.get("results") == [], (
            f"Empty collection returned non-empty results: {response}"
        )
    finally:
        _stop_server(env, hermit_home, proc)


# ────────────────────────────────────────────────────────────────
#                  5. Mode switching (local ↔ standalone)
# ────────────────────────────────────────────────────────────────


def test_switch_local_to_standalone_preserves_data(tmp_path, test_docs_dir):
    """Index in Local mode, stop, switch the same HERMIT_HOME to Stand-alone.

    The embedded ``qdrant-client`` (local) and the Rust ``qdrant-server``
    (standalone) write incompatible on-disk layouts. Hermit detects the
    mode change at startup via ``qdrant_mode_signature`` and triggers
    ``rebuild_collection`` for every registered collection — same path
    as embedding-model changes. Files on disk are unchanged, so the
    rebuild is fully automatic from the user's point of view: by the
    time ``/health`` reports ready and the indexing queue drains,
    search works in the new mode.
    """
    hermit_home = tmp_path / "hermit_switch"
    hermit_home.mkdir()
    _link_models(hermit_home)

    # ── Phase 1: Local mode — index test_docs_dir ───────────
    local_env, _ = _make_local_env(hermit_home)
    rc, _ = _run_hermit(["kb", "add", "switch_col", str(test_docs_dir)], env=local_env)
    assert rc == 0

    proc, port = _start_server(local_env, hermit_home)
    try:
        assert _poll_indexing_done(port, "switch_col", timeout=60)
        local_status = _get(port, "/collections/switch_col/status")
        assert local_status["total_chunks"] > 0
        local_chunks = local_status["total_chunks"]
        local_files = local_status["indexed_files"]

        local_results = _post(port, "/search", {
            "collection": "switch_col",
            "query": "hermit knowledge base",
            "top_k": 3,
        })["results"]
        assert len(local_results) > 0
    finally:
        _stop_server(local_env, hermit_home, proc)

    # ── Phase 2: Stand-alone mode — same data dir, Docker-mounted ──
    sa_env, container_name = _make_standalone_env(hermit_home)
    proc, port = _start_server(sa_env, hermit_home)
    try:
        health = _get(port, "/health")
        assert health["qdrant_mode"] == "standalone"

        assert _poll_indexing_done(port, "switch_col", timeout=60), (
            "Standalone-mode startup did not drain (unexpected re-index)"
        )

        sa_status = _get(port, "/collections/switch_col/status")
        assert sa_status["total_chunks"] == local_chunks, (
            f"Cross-mode chunk drift: local={local_chunks} → "
            f"standalone={sa_status['total_chunks']}"
        )
        assert sa_status["indexed_files"] == local_files

        sa_results = _post(port, "/search", {
            "collection": "switch_col",
            "query": "hermit knowledge base",
            "top_k": 3,
        })["results"]
        assert len(sa_results) > 0, (
            "Search returned nothing after switching to standalone — "
            "data dir was not adopted"
        )
    finally:
        _stop_server(sa_env, hermit_home, proc)
        _remove_container(container_name)


# ────────────────────────────────────────────────────────────────
#                  6. Stand-alone-only behavior
# ────────────────────────────────────────────────────────────────


def test_standalone_starts_with_localhost(tmp_path):
    """``QDRANT_HOST=localhost`` works (IPv6 resolution regression — Issue #1)."""
    hermit_home = tmp_path / "hermit_localhost"
    hermit_home.mkdir()
    _link_models(hermit_home)
    env, container_name = _make_standalone_env(hermit_home, qdrant_host="localhost")

    proc, port = _start_server(env, hermit_home)
    try:
        assert _get(port, "/health")["status"] == "ready"
    finally:
        _stop_server(env, hermit_home, proc)
        _remove_container(container_name)


def test_standalone_kb_add_succeeds(standalone_only_server, test_docs_dir):
    """``hermit kb add`` succeeds when the running server is in standalone mode (Issue #2)."""
    _port, deployment = standalone_only_server
    rc, output = _run_hermit(
        ["kb", "add", "sa_kbadd", str(test_docs_dir)],
        env=deployment["env"],
    )
    assert rc == 0, f"kb add failed: {output}"
    assert output.get("status") == "added"


def test_standalone_container_is_running_and_pinned(standalone_only_server):
    """Container is up after start and was launched from ``TEST_QDRANT_IMAGE``."""
    _port, deployment = standalone_only_server
    name = deployment["container_name"]
    assert _container_running(name), f"Container '{name}' not running"
    image = _container_image(name)
    assert image == TEST_QDRANT_IMAGE, (
        f"Container '{name}' was launched from '{image}', "
        f"expected '{TEST_QDRANT_IMAGE}'"
    )


def test_standalone_atexit_stops_container(tmp_path):
    """``hermit stop`` triggers ``docker stop`` on the managed container.

    Persistent-container design: container should be stopped (not removed)
    so that the next start can resume via ``docker start``.
    """
    hermit_home = tmp_path / "hermit_atexit"
    hermit_home.mkdir()
    _link_models(hermit_home)
    env, container_name = _make_standalone_env(hermit_home)

    proc, _port = _start_server(env, hermit_home)
    try:
        subprocess.run(
            [sys.executable, "-m", "hermit.cli", "stop"],
            env=env, check=False, capture_output=True,
        )
        assert _wait_until(
            lambda: not _container_running(container_name),
            timeout=15.0,
        ), f"Container '{container_name}' still running after hermit stop"
        assert _container_exists(container_name), (
            "Container was removed instead of just stopped — "
            "violates persistent-container invariant"
        )
    finally:
        _terminate_hermit_server(hermit_home)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        _remove_container(container_name)


def test_standalone_orphan_container_adopted(tmp_path):
    """SIGKILL leaves the container running; the next start adopts it.

    Mirrors the crash-and-restart path in ``ensure_qdrant_running``: the
    fast-path healthy check sees Qdrant on the expected port and skips
    the create/restart branches entirely.
    """
    hermit_home = tmp_path / "hermit_orphan"
    hermit_home.mkdir()
    _link_models(hermit_home)
    env, container_name = _make_standalone_env(hermit_home)

    proc1, port1 = _start_server(env, hermit_home)
    try:
        pid_file = hermit_home / "hermit.pid"
        server_pid = int(pid_file.read_text().strip())
        os.kill(server_pid, signal.SIGKILL)
        _wait_until(lambda: not _proc_alive(server_pid), timeout=5.0, step=0.1)
    finally:
        try:
            proc1.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc1.kill()
            proc1.wait()
        (hermit_home / "hermit.pid").unlink(missing_ok=True)

    assert _container_running(container_name), (
        "Container died with hermit — orphan adoption test cannot proceed"
    )

    proc2, port2 = _start_server(env, hermit_home)
    try:
        h = _get(port2, "/health")
        assert h["status"] == "ready"
        assert h["qdrant_mode"] == "standalone"
    finally:
        _stop_server(env, hermit_home, proc2)
        _remove_container(container_name)


def test_standalone_bad_image_fails_fast(tmp_path):
    """A non-existent Qdrant image yields a friendly error and a non-ready server.

    Uses an obviously bogus tag so ``docker pull`` fails immediately on
    the first authoritative response (no successful manifest to retry).
    """
    hermit_home = tmp_path / "hermit_badimage"
    hermit_home.mkdir()
    _link_models(hermit_home)
    bogus_image = "qdrant/qdrant:no-such-tag-xyz-99999"
    env, container_name = _make_standalone_env(hermit_home, image=bogus_image)
    env["HERMIT_START_TIMEOUT"] = "120"  # we want a shorter ceiling here

    proc = subprocess.Popen(
        [sys.executable, "-m", "hermit.cli", "start"],
        env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    try:
        # Either port.json never appears, or health never goes ready —
        # in both cases the wrapper subprocess should exit non-zero.
        try:
            port = _read_port_file(hermit_home, timeout=10)
            became_ready = _poll_health_while_process_alive(port, proc, timeout=120)
        except RuntimeError:
            became_ready = False
        assert not became_ready, (
            "Server unexpectedly became ready with a bogus Qdrant image"
        )

        # Wrapper must exit non-zero when startup fails
        rc = proc.wait(timeout=15)
        assert rc != 0, f"hermit start exited 0 despite docker failure (rc={rc})"

        # Log should mention the image / pull failure so users can debug
        log_file = hermit_home / "logs" / "hermit.log"
        if log_file.exists():
            log_text = log_file.read_text(errors="replace")
            assert (
                bogus_image in log_text
                or "镜像" in log_text
                or "image" in log_text.lower()
                or "pull" in log_text.lower()
            ), "Log lacks any image/pull diagnostic for the failure"
    finally:
        _terminate_hermit_server(hermit_home)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        # Defensively clean up: a partially-launched container would be unusual
        # for this failure mode but we don't want to leave orphans on retry.
        _remove_container(container_name)


def test_standalone_no_docker_cli_fails_fast(tmp_path):
    """If the ``docker`` CLI is missing from PATH, hermit aborts standalone start.

    Uses ``PATH=/nonexistent`` for the subprocess so ``shutil.which("docker")``
    returns ``None`` and ``ensure_qdrant_running`` raises a friendly error.
    """
    hermit_home = tmp_path / "hermit_nodocker"
    hermit_home.mkdir()
    _link_models(hermit_home)
    env, container_name = _make_standalone_env(hermit_home)
    env["PATH"] = "/nonexistent"  # hide docker from the child
    env["HERMIT_START_TIMEOUT"] = "30"

    proc = subprocess.Popen(
        [sys.executable, "-m", "hermit.cli", "start"],
        env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    try:
        try:
            port = _read_port_file(hermit_home, timeout=10)
            became_ready = _poll_health_while_process_alive(port, proc, timeout=30)
        except RuntimeError:
            became_ready = False
        assert not became_ready, "Server became ready without a docker CLI"

        rc = proc.wait(timeout=15)
        assert rc != 0, "hermit start exited 0 despite missing docker CLI"

        log_file = hermit_home / "logs" / "hermit.log"
        if log_file.exists():
            log_text = log_file.read_text(errors="replace")
            assert (
                "Docker" in log_text or "docker" in log_text
            ), "Log should explain the missing-Docker failure"
    finally:
        _terminate_hermit_server(hermit_home)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        _remove_container(container_name)


# ────────────────────────────────────────────────────────────────
#                  7. Image-pin invariants (no Docker)
# ────────────────────────────────────────────────────────────────


def test_pinned_image_tag_is_not_floating():
    """``TEST_QDRANT_IMAGE`` must use an explicit ``vX.Y.Z`` tag.

    Floating tags (``latest``, ``stable``, ``vX.Y``) silently shift the
    on-disk format between machines — design-doc §3.3 forbids them.
    """
    assert ":" in TEST_QDRANT_IMAGE
    tag = TEST_QDRANT_IMAGE.split(":", 1)[1]
    assert tag not in {"latest", "stable", "main"}, (
        f"Floating tag '{tag}' is forbidden — pin to vX.Y.Z"
    )
    assert re.fullmatch(r"v\d+\.\d+\.\d+", tag), (
        f"Tag '{tag}' must match 'vX.Y.Z'"
    )


def test_pinned_image_matches_config_default():
    """Drift guard: keep ``TEST_QDRANT_IMAGE`` == ``hermit.config.QDRANT_IMAGE``.

    A bumped config without a bumped test pin would silently leave tests
    covering an old version while users get a new one. Bump together.
    """
    from hermit.config import QDRANT_IMAGE as CONFIG_QDRANT_IMAGE
    assert TEST_QDRANT_IMAGE == CONFIG_QDRANT_IMAGE, (
        f"Test pin '{TEST_QDRANT_IMAGE}' diverged from "
        f"hermit.config.QDRANT_IMAGE='{CONFIG_QDRANT_IMAGE}'"
    )


# ── Misc tiny helpers ────────────────────────────────────────────


def _proc_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _du_bytes(path: Path) -> int:
    """Recursively sum sizes of all files under *path*. Missing dir → 0."""
    if not path.exists():
        return 0
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            try:
                total += p.stat().st_size
            except OSError:
                pass
    return total


# ────────────────────────────────────────────────────────────────
#                  8. Embedding cache (post-#27)
# ────────────────────────────────────────────────────────────────


def test_embedding_cache_speeds_up_reindex(tmp_path, large_docs_dir):
    """Re-adding the same KB with a warm cache must skip ONNX inference.

    Indexes ``large_docs_dir`` once cold (writes vectors into
    ``HERMIT_HOME/cache/{dense,sparse}``), removes the collection, then
    indexes the *same* directory again and asserts:
      * second-round wall time is at most 1/3 of the first round
      * cache directory size after round 2 is identical to round 1
        (proving every chunk was a hit and nothing was re-written)

    Local mode only — the cache lives under ``HERMIT_HOME`` and is
    deployment-mode-independent, so running both modes would just
    duplicate the same assertion.
    """
    hermit_home = tmp_path / "hermit_home_cache"
    hermit_home.mkdir()
    _link_models(hermit_home)
    env, _ = _make_local_env(hermit_home)

    proc, port = _start_server(env, hermit_home)
    try:
        # ── Round 1: cold cache ─────────────────────────────
        rc, _ = _run_hermit(["kb", "add", "cache_col", str(large_docs_dir)], env=env)
        assert rc == 0
        _post(port, "/collections/cache_col/sync", {})
        cold_start = time.monotonic()
        assert _poll_indexing_done(port, "cache_col", timeout=120)
        cold_elapsed = time.monotonic() - cold_start
        cold_status = _get(port, "/collections/cache_col/status")
        assert cold_status["total_chunks"] > 0

        cache_root = hermit_home / "cache"
        cold_cache_size = _du_bytes(cache_root)
        assert cold_cache_size > 0, "Cache should have entries after cold round"

        rc, _ = _run_hermit(["kb", "remove", "cache_col"], env=env)
        assert rc == 0

        # ── Round 2: warm cache ─────────────────────────────
        rc, _ = _run_hermit(["kb", "add", "cache_col", str(large_docs_dir)], env=env)
        assert rc == 0
        _post(port, "/collections/cache_col/sync", {})
        warm_start = time.monotonic()
        assert _poll_indexing_done(port, "cache_col", timeout=120)
        warm_elapsed = time.monotonic() - warm_start
        warm_status = _get(port, "/collections/cache_col/status")

        warm_cache_size = _du_bytes(cache_root)

        assert warm_status["total_chunks"] == cold_status["total_chunks"]
        assert warm_cache_size == cold_cache_size, (
            f"Cache grew between rounds ({cold_cache_size} → {warm_cache_size}); "
            "warm round wrote new entries instead of hitting cache"
        )
        # Warm should be at least 3× faster. In practice it's >10× — keep
        # the assertion loose so CPU-starved CI doesn't false-flag.
        assert warm_elapsed * 3 <= cold_elapsed, (
            f"Warm-cache reindex was not significantly faster: "
            f"cold={cold_elapsed:.2f}s warm={warm_elapsed:.2f}s"
        )
    finally:
        _stop_server(env, hermit_home, proc)


# ────────────────────────────────────────────────────────────────
#                  9. KB lifecycle edge cases
# ────────────────────────────────────────────────────────────────


def test_kb_remove_during_indexing(deployment, large_docs_dir):
    """``kb remove`` mid-indexing cancels the worker and cleans up cleanly.

    The DELETE /collections/<name> handler calls cancel_collection_tasks
    and then waits up to 30s for the worker to drain. Verifies:
      * remove returns success without 409 (drained in time)
      * collection no longer present in /health
      * the same name can be re-added immediately afterwards
    """
    env = deployment["env"]
    hermit_home = deployment["hermit_home"]

    rc, _ = _run_hermit(["kb", "add", "remove_mid", str(large_docs_dir)], env=env)
    assert rc == 0

    proc, port = _start_server(env, hermit_home)
    try:
        # Wait until at least one task is in flight so we're really
        # exercising the cancel path, not just a pre-empty queue.
        deadline = time.monotonic() + 10
        saw_pending = False
        while time.monotonic() < deadline:
            try:
                tasks = _get(port, f"/collections/remove_mid/tasks")
                if tasks.get("pending_tasks", 0) > 0:
                    saw_pending = True
                    break
            except Exception:
                pass
            time.sleep(0.05)
        if not saw_pending:
            print("NOTE: indexing finished before remove window")

        rc, output = _run_hermit(["kb", "remove", "remove_mid"], env=env)
        assert rc == 0, f"kb remove failed: {output}"
        assert output.get("status") == "removed"

        health = _get(port, "/health")
        names = {c["name"] for c in health.get("collections", [])}
        assert "remove_mid" not in names

        # /collections/<>/tasks must 404 once the collection is gone
        tasks_url = f"http://127.0.0.1:{port}/collections/remove_mid/tasks"
        try:
            urllib.request.urlopen(tasks_url, timeout=5)
            pytest.fail("tasks endpoint should 404 after remove")
        except urllib.error.HTTPError as e:
            assert e.code == 404

        # Re-adding the same name must succeed (registry / qdrant slot freed)
        rc, _ = _run_hermit(["kb", "add", "remove_mid", str(large_docs_dir)], env=env)
        assert rc == 0
    finally:
        _stop_server(env, hermit_home, proc)


def test_kb_incremental_file_modification(deployment, tmp_path):
    """Editing a file → /sync re-embeds only that file; peers untouched.

    Uses three docs with distinctive made-up tokens so search ordering
    is deterministic. Verifies the modified doc's chunks were replaced
    (old token no longer searchable, new token returned) while peers
    keep returning their original tokens.
    """
    env = deployment["env"]
    hermit_home = deployment["hermit_home"]

    docs = tmp_path / f"incdocs_{deployment['mode']}"
    docs.mkdir()
    (docs / "doc_a.md").write_text(
        "# Doc A\n\nThe rare token zorblax appears in document A only.\n"
    )
    (docs / "doc_b.md").write_text(
        "# Doc B\n\nThe rare token kwirvex appears in document B only.\n"
    )
    (docs / "doc_c.md").write_text(
        "# Doc C\n\nThe rare token plumbat appears in document C only.\n"
    )

    rc, _ = _run_hermit(["kb", "add", "inc_col", str(docs)], env=env)
    assert rc == 0

    proc, port = _start_server(env, hermit_home)
    try:
        _post(port, "/collections/inc_col/sync", {})
        assert _poll_indexing_done(port, "inc_col", timeout=60)
        before = _get(port, "/collections/inc_col/status")
        assert before["indexed_files"] == 3

        def _top_source(query: str) -> str | None:
            r = _post(port, "/search", {
                "collection": "inc_col", "query": query, "top_k": 3,
            })["results"]
            return Path(r[0]["source_file"]).name if r else None

        assert _top_source("zorblax") == "doc_a.md"
        assert _top_source("kwirvex") == "doc_b.md"
        assert _top_source("plumbat") == "doc_c.md"

        # Modify only doc_a: replace zorblax → snurfle
        (docs / "doc_a.md").write_text(
            "# Doc A\n\nThe rare token snurfle now appears in document A.\n"
        )

        _post(port, "/collections/inc_col/sync", {})
        assert _poll_indexing_done(port, "inc_col", timeout=60)
        after = _get(port, "/collections/inc_col/status")
        assert after["indexed_files"] == 3
        # Chunk count should be unchanged because we replaced one chunk
        # with another of similar size — this is a soft check, allow ±2
        assert abs(after["total_chunks"] - before["total_chunks"]) <= 2

        # New token finds doc_a
        assert _top_source("snurfle") == "doc_a.md"
        # Untouched peers still found by their tokens
        assert _top_source("kwirvex") == "doc_b.md"
        assert _top_source("plumbat") == "doc_c.md"
        # Old token no longer points at doc_a (chunk was replaced)
        # We don't assert "no results" because BM25 may still match the
        # query string lexically against unrelated chunks; we assert
        # only that doc_a is no longer the top result.
        assert _top_source("zorblax") != "doc_a.md"
    finally:
        _stop_server(env, hermit_home, proc)


def test_multiple_collections_isolated(deployment, tmp_path):
    """Two collections on the same server: searches don't bleed across them."""
    env = deployment["env"]
    hermit_home = deployment["hermit_home"]

    apples_dir = tmp_path / f"apples_{deployment['mode']}"
    apples_dir.mkdir()
    (apples_dir / "fruit.md").write_text(
        "# Apples\n\nThe orchard grows crisp red apples and golden delicious "
        "varieties. Apple cider is pressed from McIntosh apples in autumn.\n"
    )

    cars_dir = tmp_path / f"cars_{deployment['mode']}"
    cars_dir.mkdir()
    (cars_dir / "engine.md").write_text(
        "# Cars\n\nThe internal combustion engine powers most automobiles. "
        "Pistons, cylinders, and crankshafts convert fuel to mechanical motion.\n"
    )

    rc, _ = _run_hermit(["kb", "add", "kb_apples", str(apples_dir)], env=env)
    assert rc == 0
    rc, _ = _run_hermit(["kb", "add", "kb_cars", str(cars_dir)], env=env)
    assert rc == 0

    proc, port = _start_server(env, hermit_home)
    try:
        _post(port, "/collections/kb_apples/sync", {})
        _post(port, "/collections/kb_cars/sync", {})
        assert _poll_indexing_done(port, "kb_apples", timeout=60)
        assert _poll_indexing_done(port, "kb_cars", timeout=60)

        # Apple query in kb_apples returns apple content
        a_results = _post(port, "/search", {
            "collection": "kb_apples", "query": "apple cider orchard", "top_k": 3,
        })["results"]
        assert len(a_results) > 0
        for r in a_results:
            assert "apple" in r["text"].lower(), (
                f"kb_apples returned non-apple content: {r['text'][:80]}"
            )

        # Same query in kb_cars: source must be from cars_dir, never apples_dir
        c_results = _post(port, "/search", {
            "collection": "kb_cars", "query": "apple cider orchard", "top_k": 3,
        })["results"]
        for r in c_results:
            src = r["source_file"]
            assert "apples_" not in src and "fruit.md" not in src, (
                f"kb_cars leaked apples content: {src}"
            )

        # Engine query in kb_cars returns car content; in kb_apples it must
        # not return car-source files.
        e_apples = _post(port, "/search", {
            "collection": "kb_apples", "query": "internal combustion engine", "top_k": 3,
        })["results"]
        for r in e_apples:
            src = r["source_file"]
            assert "cars_" not in src and "engine.md" not in src, (
                f"kb_apples leaked cars content: {src}"
            )
    finally:
        _stop_server(env, hermit_home, proc)


# ────────────────────────────────────────────────────────────────
#                  10. Graceful shutdown / concurrency
# ────────────────────────────────────────────────────────────────


def test_sigterm_during_indexing_drains_cleanly(deployment, large_docs_dir):
    """``hermit stop`` (SIGTERM) mid-indexing exits cleanly; restart finishes.

    Validates that interrupting the worker via the user-facing graceful
    path leaves a consistent state — restart must complete indexing
    without losing or duplicating files (final indexed_files matches
    on-disk file count).
    """
    env = deployment["env"]
    hermit_home = deployment["hermit_home"]

    rc, _ = _run_hermit(["kb", "add", "term_col", str(large_docs_dir)], env=env)
    assert rc == 0

    proc, port = _start_server(env, hermit_home)
    saw_pending = False
    try:
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            try:
                tasks = _get(port, "/collections/term_col/tasks")
                if tasks.get("pending_tasks", 0) > 0:
                    saw_pending = True
                    break
            except Exception:
                pass
            time.sleep(0.05)

        # Send SIGTERM via the user-facing path (hermit stop)
        stop_start = time.monotonic()
        result = subprocess.run(
            [sys.executable, "-m", "hermit.cli", "stop"],
            env=env, capture_output=True, text=True, timeout=30,
        )
        stop_elapsed = time.monotonic() - stop_start

        assert result.returncode == 0, (
            f"hermit stop exited {result.returncode}: {result.stderr}"
        )
        # The CLI's stop wrapper bounds itself to ~10s before SIGKILL
        assert stop_elapsed < 30, f"hermit stop took {stop_elapsed:.1f}s"
        # Process should be gone
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            pytest.fail("Hermit server still running after hermit stop")
    finally:
        _terminate_hermit_server(hermit_home)
        (hermit_home / "hermit.pid").unlink(missing_ok=True)

    if not saw_pending:
        print("NOTE: indexing finished before SIGTERM window; "
              "testing clean shutdown only")

    # Restart and verify the collection finishes indexing
    proc, port = _start_server(env, hermit_home)
    try:
        assert _poll_indexing_done(port, "term_col", timeout=120)
        status = _get(port, "/collections/term_col/status")
        on_disk = sum(1 for _ in large_docs_dir.glob("*.md"))
        assert status["indexed_files"] == on_disk, (
            f"After SIGTERM+restart: indexed_files={status['indexed_files']} "
            f"≠ on-disk={on_disk}"
        )
        results = _post(port, "/search", {
            "collection": "term_col",
            "query": "hybrid retrieval reranking",
            "top_k": 3,
        })["results"]
        assert len(results) > 0
    finally:
        _stop_server(env, hermit_home, proc)


def test_search_concurrent_with_indexing(deployment, large_docs_dir):
    """Search remains responsive while a large indexing job is in flight.

    Validates the single-INDEX_WORKERS / shared-ONNX-session design: a
    background indexing job must not block search calls. Hammers /search
    every 100ms while indexing is in progress and asserts:
      * zero non-2xx responses
      * the indexing job still completes
    """
    import threading

    env = deployment["env"]
    hermit_home = deployment["hermit_home"]

    rc, _ = _run_hermit(["kb", "add", "concur_col", str(large_docs_dir)], env=env)
    assert rc == 0

    proc, port = _start_server(env, hermit_home)
    stop_evt = threading.Event()
    errors: list[str] = []
    success_count = 0
    success_lock = threading.Lock()

    def _hammer():
        nonlocal success_count
        while not stop_evt.is_set():
            try:
                _post(port, "/search", {
                    "collection": "concur_col",
                    "query": "hybrid retrieval reranking",
                    "top_k": 3,
                })
                with success_lock:
                    success_count += 1
            except Exception as exc:
                errors.append(str(exc))
            time.sleep(0.1)

    t = threading.Thread(target=_hammer, daemon=True)
    try:
        t.start()
        assert _poll_indexing_done(port, "concur_col", timeout=120)
        # Drain a few more search hits after indexing finishes so the
        # success_count reflects steady-state behaviour too.
        time.sleep(0.5)
    finally:
        stop_evt.set()
        t.join(timeout=5)
        _stop_server(env, hermit_home, proc)

    assert errors == [], (
        f"Search calls failed during indexing ({len(errors)} errors): "
        f"first={errors[0] if errors else None}"
    )
    assert success_count >= 5, (
        f"Only {success_count} successful searches — fixture too short?"
    )


# ────────────────────────────────────────────────────────────────
#                  11. Reverse mode-switching
# ────────────────────────────────────────────────────────────────


def test_switch_standalone_to_local_preserves_data(tmp_path, test_docs_dir):
    """Mirror of test_switch_local_to_standalone, in the opposite direction.

    Index in Stand-alone mode (Docker-managed), stop, switch the same
    HERMIT_HOME to Local mode. The signature mismatch must trigger a
    transparent rebuild and search must continue to work.
    """
    hermit_home = tmp_path / "hermit_revswitch"
    hermit_home.mkdir()
    _link_models(hermit_home)

    sa_env, container_name = _make_standalone_env(hermit_home)
    rc, _ = _run_hermit(["kb", "add", "rev_col", str(test_docs_dir)], env=sa_env)
    assert rc == 0

    proc, port = _start_server(sa_env, hermit_home)
    sa_chunks: int | None = None
    sa_files: int | None = None
    try:
        assert _poll_indexing_done(port, "rev_col", timeout=60)
        s = _get(port, "/collections/rev_col/status")
        sa_chunks, sa_files = s["total_chunks"], s["indexed_files"]
        assert sa_chunks > 0
    finally:
        _stop_server(sa_env, hermit_home, proc)
        # Standalone container must be removed before switching modes —
        # otherwise port 6333 from this run could collide with the local
        # mode's safeguard probe.
        _remove_container(container_name)

    local_env, _ = _make_local_env(hermit_home)
    proc, port = _start_server(local_env, hermit_home)
    try:
        health = _get(port, "/health")
        assert health["qdrant_mode"] == "local"
        assert _poll_indexing_done(port, "rev_col", timeout=60)
        s = _get(port, "/collections/rev_col/status")
        assert s["total_chunks"] == sa_chunks, (
            f"Reverse-switch chunk drift: standalone={sa_chunks} → "
            f"local={s['total_chunks']}"
        )
        assert s["indexed_files"] == sa_files
        results = _post(port, "/search", {
            "collection": "rev_col",
            "query": "hermit knowledge base",
            "top_k": 3,
        })["results"]
        assert len(results) > 0
    finally:
        _stop_server(local_env, hermit_home, proc)


# ────────────────────────────────────────────────────────────────
#                  12. Stand-alone container resilience
# ────────────────────────────────────────────────────────────────


def test_standalone_container_restart_is_transparent(
    standalone_only_server, test_docs_dir,
):
    """Manually ``docker restart`` the Qdrant container; Hermit recovers.

    Simulates the realistic case where a user restarts Docker Desktop or
    the container itself. The qdrant-client transparently reconnects, so
    a search after the restart must succeed without restarting Hermit.
    """
    port, deployment = standalone_only_server
    env = deployment["env"]
    container_name = deployment["container_name"]

    rc, _ = _run_hermit(["kb", "add", "restart_col", str(test_docs_dir)], env=env)
    assert rc == 0
    _post(port, "/collections/restart_col/sync", {})
    assert _poll_indexing_done(port, "restart_col", timeout=60)
    pre = _get(port, "/collections/restart_col/status")
    assert pre["total_chunks"] > 0

    # Restart the Qdrant container (NOT hermit)
    r = subprocess.run(
        ["docker", "restart", container_name],
        capture_output=True, text=True, timeout=30,
    )
    assert r.returncode == 0, f"docker restart failed: {r.stderr}"

    # Wait for Qdrant to come back up — the client retries internally,
    # so we just give it some headroom before issuing a search.
    assert _wait_until(
        lambda: _container_running(container_name), timeout=15.0,
    )
    time.sleep(2.0)  # let Qdrant actually start serving

    deadline = time.monotonic() + 30
    last_exc: Exception | None = None
    while time.monotonic() < deadline:
        try:
            results = _post(port, "/search", {
                "collection": "restart_col",
                "query": "hermit knowledge base",
                "top_k": 3,
            })["results"]
            assert len(results) > 0, "Search returned 0 results after restart"
            break
        except Exception as exc:
            last_exc = exc
            time.sleep(1.0)
    else:
        pytest.fail(f"Search never recovered after docker restart: {last_exc}")

    post = _get(port, "/collections/restart_col/status")
    assert post["total_chunks"] == pre["total_chunks"], (
        "Container restart caused chunk count drift"
    )


# ────────────────────────────────────────────────────────────────
#                  13. Server lifecycle
# ────────────────────────────────────────────────────────────────


def test_port_conflict_falls_back_to_alternate_port(tmp_path):
    """If port.json points at an occupied port, ``resolve_port`` falls back.

    Pre-writes port.json with a port we hold open via a bound socket;
    expects hermit to detect the conflict and pick a different port,
    then to write the new port back so subsequent CLI calls find it.
    """
    hermit_home = tmp_path / "hermit_portconflict"
    hermit_home.mkdir()
    _link_models(hermit_home)
    env, _ = _make_local_env(hermit_home)

    # Bind a socket and hold it for the duration of the test
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 0)
    sock.bind(("127.0.0.1", 0))
    sock.listen(1)
    occupied = sock.getsockname()[1]

    port_file = hermit_home / "port.json"
    port_file.write_text(json.dumps({"port": occupied}))
    pre_mtime = port_file.stat().st_mtime

    try:
        # Launch hermit start manually — _start_server's _read_port_file
        # would race against hermit overwriting the pre-seeded port.json.
        proc = subprocess.Popen(
            [sys.executable, "-m", "hermit.cli", "start"],
            env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        try:
            # Wait for hermit to overwrite port.json with its chosen port
            deadline = time.monotonic() + 30
            actual_port: int | None = None
            while time.monotonic() < deadline:
                if port_file.exists() and port_file.stat().st_mtime > pre_mtime:
                    try:
                        actual_port = int(json.loads(port_file.read_text())["port"])
                        break
                    except (json.JSONDecodeError, KeyError, ValueError):
                        pass
                time.sleep(0.1)
            assert actual_port is not None, "port.json was never overwritten"
            assert actual_port != occupied, (
                f"Hermit started on occupied port {occupied} — "
                "port-conflict fallback failed"
            )
            assert _poll_health(actual_port, timeout=120)
            assert _get(actual_port, "/health")["status"] == "ready"
        finally:
            _stop_server(env, hermit_home, proc)
    finally:
        sock.close()
