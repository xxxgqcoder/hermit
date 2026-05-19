"""Shared fixtures for Hermit E2E tests.

Provides:
  - hermit_env  — function-scoped isolated HERMIT_HOME with models symlink
  - hermit_server — function-scoped running server (local mode)
  - test_docs_dir — function-scoped temp dir with 3 small .md files
"""

import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest

# Project root: two levels up from tests/
PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_MODEL_ROOT = Path.home() / ".hermit" / "models"



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


def _poll_health_while_process_alive(
    port: int,
    proc: subprocess.Popen,
    timeout: int = 60,
) -> bool:
    """Poll health, but stop early if the startup wrapper exits unsuccessfully."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if _poll_health(port, timeout=1):
            return True
        returncode = proc.poll()
        if returncode is not None and returncode != 0:
            return False
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


def _link_models(hermit_home: Path) -> None:
    """Symlink an existing model cache into an isolated HERMIT_HOME."""
    for models_src in (PROJECT_ROOT / "models", DEFAULT_MODEL_ROOT):
        if models_src.exists():
            (hermit_home / "models").symlink_to(models_src)
            return


def _terminate_hermit_server(hermit_home: Path, timeout: float = 5.0) -> None:
    """Best-effort cleanup for daemonized uvicorn started by hermit cli."""
    pid_file = hermit_home / "hermit.pid"
    if not pid_file.exists():
        return

    try:
        pid = int(pid_file.read_text().strip())
    except ValueError:
        pid_file.unlink(missing_ok=True)
        return

    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        pid_file.unlink(missing_ok=True)
        return
    except OSError:
        return

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            pid_file.unlink(missing_ok=True)
            return
        time.sleep(0.1)

    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    pid_file.unlink(missing_ok=True)


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

    _link_models(hermit_home)

    env = os.environ.copy()
    env["HERMIT_HOME"] = str(hermit_home)
    env["HERMIT_START_TIMEOUT"] = "120"

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
        _terminate_hermit_server(hermit_home)
        pytest.fail("Timed out waiting for port.json to appear")

    start_timeout = int(env.get("HERMIT_START_TIMEOUT", 120))
    if not _poll_health_while_process_alive(port, proc, timeout=start_timeout):
        proc.kill()
        proc.wait()
        _terminate_hermit_server(hermit_home)
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

    _terminate_hermit_server(hermit_home)

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
        "A: Hermit embeds documents with jina embeddings and stores them in LanceDB.\n"
    )

    return docs
