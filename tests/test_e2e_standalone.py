"""E2E regression tests — standalone (Docker) mode.

All tests are skipped automatically when Docker is not available.

Run with:
    pytest tests/test_e2e_standalone.py -v

Issue regressions covered:
  - Issue #1: QDRANT_HOST=localhost fails due to IPv6 resolution on macOS
  - Issue #2: kb add fails in standalone mode
"""

import json
import os
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

from tests.conftest import _poll_health, _read_port_file, PROJECT_ROOT

# Skip entire module if Docker is not available
pytestmark = pytest.mark.skipif(
    not shutil.which("docker"),
    reason="Docker CLI not found — standalone tests require Docker",
)


# ── helpers ─────────────────────────────────────────────────────

def _free_port() -> int:
    """Ask the OS for a free TCP port."""
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
    result = subprocess.run(
        [sys.executable, "-m", "hermit.cli"] + args,
        env=env,
        capture_output=True,
        text=True,
    )
    try:
        output = json.loads(result.stdout.strip())
    except json.JSONDecodeError:
        output = {"raw": result.stdout.strip(), "stderr": result.stderr.strip()}
    return result.returncode, output


def _poll_indexing_done(port: int, collection: str, timeout: int = 60) -> bool:
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


def _remove_container(container_name: str) -> None:
    subprocess.run(
        ["docker", "rm", "-f", container_name],
        check=False, capture_output=True,
    )


# ── fixtures ─────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def standalone_env(tmp_path_factory):
    """Module-scoped isolated HERMIT_HOME for standalone (Docker) mode.

    Allocates unique ports and container name to avoid collision with the
    user's production Qdrant container.
    """
    hermit_home = tmp_path_factory.mktemp("hermit_standalone")

    # Symlink models
    models_src = PROJECT_ROOT / "models"
    if models_src.exists():
        (hermit_home / "models").symlink_to(models_src)

    qdrant_port = _free_port()
    qdrant_grpc_port = _free_port()
    container_name = f"hermit_qdrant_test_{uuid.uuid4().hex[:8]}"

    env = os.environ.copy()
    env.update({
        "HERMIT_HOME": str(hermit_home),
        "HERMIT_START_TIMEOUT": "300",  # Docker image pull can take >2 min on CI
        "QDRANT_HOST": "127.0.0.1",
        "QDRANT_MANAGED": "true",
        "QDRANT_PORT": str(qdrant_port),
        "QDRANT_GRPC_PORT": str(qdrant_grpc_port),
        "QDRANT_CONTAINER_NAME": container_name,
    })

    yield {
        "env": env,
        "hermit_home": hermit_home,
        "qdrant_port": qdrant_port,
        "container_name": container_name,
    }

    # Safety-net cleanup in case tests left the container running
    _remove_container(container_name)


@pytest.fixture()
def standalone_server(standalone_env):
    """Function-scoped running Hermit server in standalone (Docker) mode.

    Yields (port: int, standalone_env: dict).
    """
    env = standalone_env["env"]
    hermit_home = standalone_env["hermit_home"]

    proc = subprocess.Popen(
        [sys.executable, "-m", "hermit.cli", "start"],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    try:
        port = _read_port_file(hermit_home, timeout=10)
    except RuntimeError:
        proc.kill()
        proc.wait()
        pytest.fail("Timed out waiting for port.json")

    start_timeout = int(env.get("HERMIT_START_TIMEOUT", 300))
    if not _poll_health(port, timeout=start_timeout):
        proc.kill()
        proc.wait()
        log_file = hermit_home / "logs" / "hermit.log"
        if log_file.exists():
            print("\n=== hermit.log ===\n", log_file.read_text()[-4000:])
        pytest.fail(f"Standalone server did not become ready within {start_timeout}s")

    yield port, standalone_env

    # ── Teardown ────────────────────────────────────────────────
    subprocess.run(
        [sys.executable, "-m", "hermit.cli", "stop"],
        env=env, check=False, capture_output=True,
    )

    # Wait for atexit to docker rm the container (up to 5s)
    for _ in range(10):
        time.sleep(0.5)
        result = subprocess.run(
            ["docker", "ps", "--filter", f"name={standalone_env['container_name']}",
             "--format", "{{.Names}}"],
            capture_output=True, text=True,
        )
        if standalone_env["container_name"] not in result.stdout:
            break

    # Fallback: force-remove the container
    _remove_container(standalone_env["container_name"])

    # Ensure process is reaped
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()

    # Cleanup pid file so next test starts clean
    pid_file = hermit_home / "hermit.pid"
    pid_file.unlink(missing_ok=True)


# ── Tests ────────────────────────────────────────────────────────


def test_standalone_server_starts_ipv4(standalone_server):
    """Standalone server starts with QDRANT_HOST=127.0.0.1 — baseline IPv4 path."""
    port, env_info = standalone_server
    data = _get(port, "/health")
    assert data["status"] == "ready"
    assert data["models_loaded"] is True
    assert data["qdrant_mode"] == "standalone"


def test_standalone_server_starts_localhost(tmp_path_factory, standalone_env):
    """Standalone server starts with QDRANT_HOST=localhost — IPv6 regression (Issue #1).

    Uses its own isolated HERMIT_HOME to avoid port.json / hermit.pid conflicts
    with the standalone_env fixture.
    """
    hermit_home = tmp_path_factory.mktemp("hermit_localhost")

    models_src = PROJECT_ROOT / "models"
    if models_src.exists():
        (hermit_home / "models").symlink_to(models_src)

    qdrant_port = _free_port()
    qdrant_grpc_port = _free_port()
    container_name = f"hermit_qdrant_test_{uuid.uuid4().hex[:8]}"

    env = os.environ.copy()
    env.update({
        "HERMIT_HOME": str(hermit_home),
        "HERMIT_START_TIMEOUT": "300",
        "QDRANT_HOST": "localhost",
        "QDRANT_MANAGED": "true",
        "QDRANT_PORT": str(qdrant_port),
        "QDRANT_GRPC_PORT": str(qdrant_grpc_port),
        "QDRANT_CONTAINER_NAME": container_name,
    })

    proc = subprocess.Popen(
        [sys.executable, "-m", "hermit.cli", "start"],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    try:
        port = _read_port_file(hermit_home, timeout=10)
        assert _poll_health(port, timeout=300), (
            "Server with QDRANT_HOST=localhost did not become ready — "
            "possible IPv6 regression (Issue #1)"
        )
        data = _get(port, "/health")
        assert data["status"] == "ready"
    finally:
        subprocess.run(
            [sys.executable, "-m", "hermit.cli", "stop"],
            env=env, check=False, capture_output=True,
        )
        _remove_container(container_name)
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


def test_standalone_docker_container_running(standalone_server, standalone_env):
    """Qdrant Docker container is actually running after server start."""
    container_name = standalone_env["container_name"]
    result = subprocess.run(
        ["docker", "ps", "--filter", f"name={container_name}", "--format", "{{.Names}}"],
        capture_output=True, text=True,
    )
    assert container_name in result.stdout, (
        f"Expected Docker container '{container_name}' to be running"
    )


def test_standalone_kb_add_succeeds(standalone_server, test_docs_dir, standalone_env):
    """hermit kb add succeeds in standalone mode (Issue #2 regression)."""
    port, _ = standalone_server
    env = standalone_env["env"]

    rc, output = _run_hermit(["kb", "add", "test_col", str(test_docs_dir)], env=env)
    assert rc == 0, f"kb add failed: {output}"
    assert output.get("status") == "added"
    assert output.get("name") == "test_col"


def test_standalone_indexing_and_search(standalone_env, test_docs_dir):
    """Documents indexed and searchable in standalone mode.

    Registers the collection BEFORE starting the server so the server loads
    it at startup and immediately queues indexing tasks.
    """
    env = standalone_env["env"]
    hermit_home = standalone_env["hermit_home"]

    # Register collection before server starts so startup scan queues indexing
    rc, output = _run_hermit(["kb", "add", "test_col_idx", str(test_docs_dir)], env=env)
    assert rc == 0, f"kb add failed: {output}"

    # Start server — will restore test_col_idx at startup
    proc = subprocess.Popen(
        [sys.executable, "-m", "hermit.cli", "start"],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        port = _read_port_file(hermit_home, timeout=10)
        start_timeout = int(env.get("HERMIT_START_TIMEOUT", 300))
        assert _poll_health(port, timeout=start_timeout), "Standalone server did not become ready"

        assert _poll_indexing_done(port, "test_col_idx", timeout=60), (
            "Indexing did not complete within 60s"
        )

        status = _get(port, "/collections/test_col_idx/status")
        assert status["total_chunks"] > 0, f"Expected chunks > 0: {status}"

        response = _post(port, "/search", {
            "collection": "test_col_idx",
            "query": "hermit knowledge base",
            "top_k": 3,
        })
        results = response.get("results", [])
        assert len(results) > 0, "Expected at least one search result"
        for r in results:
            assert r.get("text"), "Each result must have a non-empty text field"
    finally:
        subprocess.run(
            [sys.executable, "-m", "hermit.cli", "stop"],
            env=env, check=False, capture_output=True,
        )
        # Wait for atexit/lifespan to clean up Docker container (up to 10s)
        for _ in range(20):
            time.sleep(0.5)
            result = subprocess.run(
                ["docker", "ps", "--filter",
                 f"name={standalone_env['container_name']}", "--format", "{{.Names}}"],
                capture_output=True, text=True,
            )
            if standalone_env["container_name"] not in result.stdout:
                break
        _remove_container(standalone_env["container_name"])
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        # Clean pid file so next test can start fresh
        (hermit_home / "hermit.pid").unlink(missing_ok=True)


def test_standalone_container_stopped_after_stop(standalone_env, tmp_path):
    """Docker container is stopped (not removed) after `hermit stop`.

    Persistent-container design: atexit calls `docker stop`, preserving
    the container so subsequent `hermit start` can resume with `docker start`.
    """
    hermit_home = tmp_path / "hermit_stop_test"
    hermit_home.mkdir()

    models_src = PROJECT_ROOT / "models"
    if models_src.exists():
        (hermit_home / "models").symlink_to(models_src)

    qdrant_port = _free_port()
    qdrant_grpc_port = _free_port()
    container_name = f"hermit_qdrant_test_{uuid.uuid4().hex[:8]}"

    env = os.environ.copy()
    env.update({
        "HERMIT_HOME": str(hermit_home),
        "HERMIT_START_TIMEOUT": "300",
        "QDRANT_HOST": "127.0.0.1",
        "QDRANT_MANAGED": "true",
        "QDRANT_PORT": str(qdrant_port),
        "QDRANT_GRPC_PORT": str(qdrant_grpc_port),
        "QDRANT_CONTAINER_NAME": container_name,
    })

    proc = subprocess.Popen(
        [sys.executable, "-m", "hermit.cli", "start"],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    try:
        port = _read_port_file(hermit_home, timeout=10)
        assert _poll_health(port, timeout=300), "Server did not become ready"

        # Stop server — atexit should docker stop (not rm) the container
        subprocess.run(
            [sys.executable, "-m", "hermit.cli", "stop"],
            env=env, check=False, capture_output=True,
        )

        # Poll for container to stop running (up to 15s for atexit handler)
        # docker ps (no -a) only shows RUNNING containers
        deadline = time.monotonic() + 15
        container_stopped = False
        while time.monotonic() < deadline:
            running = subprocess.run(
                ["docker", "ps", "--filter", f"name={container_name}", "--format", "{{.Names}}"],
                capture_output=True, text=True,
            )
            if container_name not in running.stdout:
                container_stopped = True
                break
            time.sleep(1)

        assert container_stopped, (
            f"Docker container '{container_name}' is still running after hermit stop"
        )

        # Container should still EXIST (just stopped) — persistent design
        all_containers = subprocess.run(
            ["docker", "ps", "-a", "--filter", f"name={container_name}", "--format", "{{.Names}}"],
            capture_output=True, text=True,
        )
        assert container_name in all_containers.stdout, (
            f"Container '{container_name}' should still exist (stopped) after hermit stop, "
            "but was removed — persistent-container invariant violated"
        )
    finally:
        _remove_container(container_name)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


def test_standalone_orphan_container_adopted(standalone_env, tmp_path):
    """Orphaned container (from crash) is adopted on next hermit start.

    Simulates: hermit start → SIGKILL (container left running) →
               hermit start → container adopted via fast-path healthy check →
               hermit stop → container stopped (not removed).
    """
    hermit_home = tmp_path / "hermit_orphan_test"
    hermit_home.mkdir()

    models_src = PROJECT_ROOT / "models"
    if models_src.exists():
        (hermit_home / "models").symlink_to(models_src)

    qdrant_port = _free_port()
    qdrant_grpc_port = _free_port()
    container_name = f"hermit_qdrant_test_{uuid.uuid4().hex[:8]}"

    env = os.environ.copy()
    env.update({
        "HERMIT_HOME": str(hermit_home),
        "HERMIT_START_TIMEOUT": "300",
        "QDRANT_HOST": "127.0.0.1",
        "QDRANT_MANAGED": "true",
        "QDRANT_PORT": str(qdrant_port),
        "QDRANT_GRPC_PORT": str(qdrant_grpc_port),
        "QDRANT_CONTAINER_NAME": container_name,
    })

    # ── First start: create container ───────────────────────────
    proc1 = subprocess.Popen(
        [sys.executable, "-m", "hermit.cli", "start"],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        port1 = _read_port_file(hermit_home, timeout=10)
        assert _poll_health(port1, timeout=300), "First server did not become ready"

        # Simulate crash: SIGKILL the uvicorn server process (not hermit CLI)
        # The CLI already exited; we kill the server process via hermit stop --signal SIGKILL
        # Simpler: just SIGKILL proc1 if it's the server (but proc1 is the 'hermit start' CLI
        # which already exited). We need to kill the actual server by PID file.
        pid_file = hermit_home / "hermit.pid"
        server_pid = int(pid_file.read_text().strip()) if pid_file.exists() else None
        if server_pid:
            import signal as _signal
            try:
                os.kill(server_pid, _signal.SIGKILL)
            except ProcessLookupError:
                pass
            # Wait briefly for process to die
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                try:
                    os.kill(server_pid, 0)
                    time.sleep(0.2)
                except ProcessLookupError:
                    break
    finally:
        try:
            proc1.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc1.kill()
            proc1.wait()

    # Container should still be RUNNING after crash (orphaned)
    running = subprocess.run(
        ["docker", "ps", "--filter", f"name={container_name}", "--format", "{{.Names}}"],
        capture_output=True, text=True,
    )
    assert container_name in running.stdout, (
        f"Container '{container_name}' should still be running after hermit crash"
    )

    # ── Second start: adopt orphaned container ───────────────────
    # Remove stale pid file so hermit start doesn't refuse
    (hermit_home / "hermit.pid").unlink(missing_ok=True)

    proc2 = subprocess.Popen(
        [sys.executable, "-m", "hermit.cli", "start"],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        port2 = _read_port_file(hermit_home, timeout=10)
        assert _poll_health(port2, timeout=300), "Second server (post-crash) did not become ready"

        health = _get(port2, "/health")
        assert health["status"] == "ready"
        assert health["qdrant_mode"] == "standalone"

        # Stop cleanly — container should be stopped, not removed
        subprocess.run(
            [sys.executable, "-m", "hermit.cli", "stop"],
            env=env, check=False, capture_output=True,
        )

        # Wait for container to stop
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            r = subprocess.run(
                ["docker", "ps", "--filter", f"name={container_name}", "--format", "{{.Names}}"],
                capture_output=True, text=True,
            )
            if container_name not in r.stdout:
                break
            time.sleep(1)

        # Container exists (stopped) but is not running
        all_containers = subprocess.run(
            ["docker", "ps", "-a", "--filter", f"name={container_name}", "--format", "{{.Names}}"],
            capture_output=True, text=True,
        )
        assert container_name in all_containers.stdout, (
            "Container should still exist (stopped) after clean hermit stop"
        )
    finally:
        _remove_container(container_name)
        try:
            proc2.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc2.kill()
            proc2.wait()
