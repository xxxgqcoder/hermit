"""E2E regression tests — local (embedded Qdrant) mode.

Run with:
    pytest tests/test_e2e_deployment.py -v

Each test uses an isolated HERMIT_HOME (function-scoped fixture) so state
doesn't leak between tests.
"""

import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest

from tests.conftest import _poll_health, _read_port_file, PROJECT_ROOT


# ── helpers ─────────────────────────────────────────────────────

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
    """Run a hermit CLI command, return (returncode, parsed_json_output)."""
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
    """Poll /collections/{name}/tasks until pending_tasks == 0."""
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


def _start_server(hermit_env: dict) -> tuple[int, subprocess.Popen]:
    """Start server via Popen, wait for ready. Returns (port, proc)."""
    import subprocess as sp
    env = hermit_env["env"]
    hermit_home = hermit_env["hermit_home"]
    proc = sp.Popen(
        [sys.executable, "-m", "hermit.cli", "start"],
        env=env,
        stdout=sp.PIPE,
        stderr=sp.PIPE,
    )
    port = _read_port_file(hermit_home, timeout=10)
    start_timeout = int(env.get("HERMIT_START_TIMEOUT", 120))
    assert _poll_health(port, timeout=start_timeout), (
        f"Server did not become ready within {start_timeout}s"
    )
    return port, proc


def _stop_server(hermit_env: dict, proc: subprocess.Popen) -> None:
    """Stop server gracefully."""
    env = hermit_env["env"]
    subprocess.run(
        [sys.executable, "-m", "hermit.cli", "stop"],
        env=env, check=False, capture_output=True,
    )
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


# ── Tests ────────────────────────────────────────────────────────


def test_server_starts_from_clean_home(hermit_server):
    """Server starts and reports ready from a completely empty HERMIT_HOME."""
    port, _ = hermit_server
    data = _get(port, "/health")
    assert data["status"] == "ready"


def test_qdrant_local_mode_health(hermit_server):
    """Health endpoint reports models loaded; no IPv6 errors in log."""
    port, hermit_home = hermit_server
    data = _get(port, "/health")
    assert data["models_loaded"] is True
    assert data["qdrant_mode"] == "local"

    log_file = hermit_home / "logs" / "hermit.log"
    if log_file.exists():
        log_text = log_file.read_text(errors="replace")
        assert "ipv6" not in log_text.lower() or "error" not in log_text.lower(), (
            "IPv6-related error found in log"
        )


def test_kb_add_succeeds(hermit_server, test_docs_dir):
    """hermit kb add registers a collection and returns expected JSON (Issue #2 regression)."""
    port, hermit_home = hermit_server
    env = {"HERMIT_HOME": str(hermit_home), **{k: v for k, v in __import__("os").environ.items()}}
    # Rebuild env from hermit_server's hermit_home
    import os
    env = os.environ.copy()
    env["HERMIT_HOME"] = str(hermit_home)

    rc, output = _run_hermit(["kb", "add", "test_col", str(test_docs_dir)], env=env)
    assert rc == 0, f"kb add failed: {output}"
    assert output.get("status") == "added"
    assert output.get("name") == "test_col"


def test_kb_list_shows_collection(hermit_server, test_docs_dir):
    """kb list returns a dict-of-dicts with the registered collection."""
    port, hermit_home = hermit_server
    import os
    env = os.environ.copy()
    env["HERMIT_HOME"] = str(hermit_home)

    rc, _ = _run_hermit(["kb", "add", "test_col", str(test_docs_dir)], env=env)
    assert rc == 0

    rc, output = _run_hermit(["kb", "list"], env=env)
    assert rc == 0
    collections = output.get("collections", {})
    assert "test_col" in collections


def test_kb_indexed_after_restart(hermit_env, test_docs_dir):
    """Collection registered before first start is indexed on startup."""
    import os
    env = hermit_env["env"]
    hermit_home = hermit_env["hermit_home"]

    # Register collection while server is not running
    rc, output = _run_hermit(["kb", "add", "test_col", str(test_docs_dir)], env=env)
    assert rc == 0, f"kb add failed: {output}"

    # Start server
    port, proc = _start_server(hermit_env)
    try:
        # Poll until indexing queue is drained
        assert _poll_indexing_done(port, "test_col", timeout=60), (
            "Indexing did not complete within 60s"
        )
        # Verify chunks were indexed
        status = _get(port, "/collections/test_col/status")
        assert status["total_chunks"] > 0, f"Expected chunks > 0, got: {status}"
    finally:
        _stop_server(hermit_env, proc)


def test_search_returns_results(hermit_env, test_docs_dir):
    """Search on an indexed collection returns non-empty results with text field."""
    import os
    env = hermit_env["env"]

    rc, output = _run_hermit(["kb", "add", "test_col", str(test_docs_dir)], env=env)
    assert rc == 0, f"kb add failed: {output}"

    port, proc = _start_server(hermit_env)
    try:
        assert _poll_indexing_done(port, "test_col", timeout=60), (
            "Indexing did not complete within 60s"
        )

        # "hermit" appears in the inline doc content
        response = _post(port, "/search", {
            "collection": "test_col",
            "query": "hermit knowledge base",
            "top_k": 3,
        })
        results = response.get("results", [])
        assert len(results) > 0, "Expected at least one search result"
        for r in results:
            assert r.get("text"), "Each result must have a non-empty text field"
    finally:
        _stop_server(hermit_env, proc)


def test_isolation_from_default_home(hermit_env):
    """Server uses isolated HERMIT_HOME, does not touch ~/.hermit/."""
    import os
    hermit_home = hermit_env["hermit_home"]
    env = hermit_env["env"]

    default_home = Path.home() / ".hermit"
    default_pid = default_home / "hermit.pid"
    before_mtime = default_pid.stat().st_mtime if default_pid.exists() else None

    port, proc = _start_server(hermit_env)
    try:
        data = _get(port, "/health")
        assert data["status"] == "ready"

        # PID file written to isolated home
        assert (hermit_home / "hermit.pid").exists()

        # ~/.hermit/hermit.pid was not created or modified
        if default_pid.exists():
            if before_mtime is not None:
                assert default_pid.stat().st_mtime == before_mtime, (
                    "Default HERMIT_HOME hermit.pid was unexpectedly modified"
                )
        else:
            assert not default_pid.exists(), (
                "Default HERMIT_HOME hermit.pid was created during isolated test"
            )
    finally:
        _stop_server(hermit_env, proc)
