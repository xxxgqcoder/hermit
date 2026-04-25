"""Shared fixtures for Hermit E2E tests.

Provides:
  - hermit_env  — function-scoped isolated HERMIT_HOME with models symlink
  - hermit_server — function-scoped running server (local mode)
  - test_docs_dir — function-scoped temp dir with 3 small .md files
"""

import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest

# Project root: two levels up from tests/
PROJECT_ROOT = Path(__file__).parent.parent


def _free_port() -> int:
    """Ask the OS to allocate a free TCP port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


# ── helpers ─────────────────────────────────────────────────────

def _poll_health(port: int, timeout: int = 60) -> bool:
    """Poll GET /health until status == 'ready' or timeout."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            url = f"http://127.0.0.1:{port}/health"
            with urllib.request.urlopen(url, timeout=3) as resp:
                data = json.loads(resp.read())
                if data.get("status") == "ready":
                    return True
        except Exception:
            pass
        time.sleep(1)
    return False


def _read_port_file(hermit_home: Path, timeout: int = 10) -> int:
    """Poll for port.json to appear and return the port."""
    port_file = hermit_home / "port.json"
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if port_file.exists():
            try:
                data = json.loads(port_file.read_text())
                return int(data["port"])
            except Exception:
                pass
        time.sleep(0.1)
    raise RuntimeError(f"port.json not found in {hermit_home} after {timeout}s")


# ── fixtures ─────────────────────────────────────────────────────


@pytest.fixture()
def hermit_env(tmp_path):
    """Function-scoped isolated HERMIT_HOME.

    Creates a fresh HERMIT_HOME in tmp_path and symlinks the project
    models directory so models don't need to be re-downloaded.

    Yields dict with keys:
      "env"        — os.environ copy with HERMIT_HOME and HERMIT_START_TIMEOUT overridden
      "hermit_home" — Path to the isolated home directory
    """
    hermit_home = tmp_path / "hermit_home"
    hermit_home.mkdir()

    # Symlink models so startup is fast
    models_src = PROJECT_ROOT / "models"
    if models_src.exists():
        (hermit_home / "models").symlink_to(models_src)

    env = os.environ.copy()
    env["HERMIT_HOME"] = str(hermit_home)
    env["HERMIT_START_TIMEOUT"] = "120"
    # Set QDRANT_PORT/GRPC_PORT to free ports so _check_no_qdrant_service()
    # doesn't clash with any real Qdrant service running on the host (e.g. 6333).
    # In local (embedded) mode these ports are never actually used; they only
    # appear in the safeguard check.
    env["QDRANT_PORT"] = str(_free_port())
    env["QDRANT_GRPC_PORT"] = str(_free_port())

    yield {"env": env, "hermit_home": hermit_home}


@pytest.fixture()
def hermit_server(hermit_env):
    """Function-scoped running Hermit server (local mode).

    Launches the server with subprocess.Popen (cmd_start blocks internally).
    Reads port from port.json (written synchronously before uvicorn spawns).
    Polls /health until ready.

    Yields (port: int, hermit_home: Path).

    Teardown: hermit stop (check=False) + SIGKILL if PID file lingers.
    """
    env = hermit_env["env"]
    hermit_home = hermit_env["hermit_home"]

    proc = subprocess.Popen(
        [sys.executable, "-m", "hermit.cli", "start"],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # port.json is written synchronously before uvicorn spawns
    try:
        port = _read_port_file(hermit_home, timeout=10)
    except RuntimeError:
        proc.kill()
        proc.wait()
        pytest.fail("Timed out waiting for port.json to appear")

    start_timeout = int(env.get("HERMIT_START_TIMEOUT", 120))
    if not _poll_health(port, timeout=start_timeout):
        proc.kill()
        proc.wait()
        # Dump logs for debugging
        log_file = hermit_home / "logs" / "hermit.log"
        if log_file.exists():
            print("\n=== hermit.log ===\n", log_file.read_text()[-4000:])
        pytest.fail(f"Server did not become ready within {start_timeout}s")

    yield port, hermit_home

    # ── Teardown ────────────────────────────────────────────────
    subprocess.run(
        [sys.executable, "-m", "hermit.cli", "stop"],
        env=env,
        check=False,
        capture_output=True,
    )

    # Wait for PID file to disappear (up to 5s), then force-kill
    pid_file = hermit_home / "hermit.pid"
    for _ in range(10):
        if not pid_file.exists():
            break
        time.sleep(0.5)
    else:
        if pid_file.exists():
            try:
                pid = int(pid_file.read_text().strip())
                os.kill(pid, signal.SIGKILL)
            except (ValueError, ProcessLookupError, OSError):
                pass
            pid_file.unlink(missing_ok=True)

    # Ensure the Popen process is reaped
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


@pytest.fixture()
def test_docs_dir(tmp_path):
    """Function-scoped temp dir with 3 small Markdown files."""
    docs = tmp_path / "docs"
    docs.mkdir()

    (docs / "intro.md").write_text(
        "# Introduction\n\nThis document introduces the hermit knowledge base system.\n"
        "Hermit uses hybrid dense and sparse search for retrieval.\n"
    )
    (docs / "usage.md").write_text(
        "# Usage Guide\n\nUse the hermit CLI to add knowledge bases.\n"
        "Run `hermit kb add mydb /path/to/docs` to register a collection.\n"
    )
    (docs / "faq.md").write_text(
        "# FAQ\n\nQ: How does hermit perform semantic search?\n"
        "A: Hermit embeds documents with jina embeddings and stores them in Qdrant.\n"
    )

    return docs
