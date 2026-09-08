"""Tests for bounded server logging."""

import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermit import cli
from hermit.cli import _tail_log
from hermit.config import LOG_BACKUP_COUNT, LOG_MAX_BYTES
from hermit.server import build_log_config, trim_log_file


def test_log_config_uses_256_mib_rotating_handler(tmp_path: Path):
    log_file = tmp_path / "hermit.log"

    config = build_log_config(log_file)
    handler = config["handlers"]["rotating_file"]

    assert handler["class"] == "logging.handlers.RotatingFileHandler"
    assert handler["filename"] == str(log_file)
    assert handler["maxBytes"] == 256 * 1024 * 1024 == LOG_MAX_BYTES
    assert handler["backupCount"] == 1 == LOG_BACKUP_COUNT


def test_trim_log_file_keeps_newest_complete_lines(tmp_path: Path):
    log_file = tmp_path / "hermit.log"
    log_file.write_bytes(b"old-line\npartial-boundary\nnew-line-1\nnew-line-2\n")

    trim_log_file(log_file, max_bytes=27)

    assert log_file.read_bytes() == b"new-line-1\nnew-line-2\n"
    assert log_file.stat().st_size <= 27


def test_tail_log_restarts_from_zero_after_rotation(tmp_path: Path):
    log_file = tmp_path / "hermit.log"
    log_file.write_text("old content that was rotated\n")
    old_position = log_file.stat().st_size
    log_file.write_text("new startup line\n")

    new_position, lines = _tail_log(log_file, old_position)

    assert new_position == log_file.stat().st_size
    assert lines == ["new startup line"]


def test_start_trims_log_before_spawning_server(tmp_path: Path, monkeypatch):
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    log_file = log_dir / "hermit.log"
    log_file.write_text("legacy log\n")
    events = []

    monkeypatch.setattr(cli, "HERMIT_HOME", tmp_path)
    monkeypatch.setattr(cli, "LOG_DIR", log_dir)
    monkeypatch.setattr(cli, "PID_FILE", tmp_path / "hermit.pid")
    monkeypatch.setattr(cli, "_read_pid", lambda: None)
    monkeypatch.setattr(cli, "resolve_port", lambda: 8000)
    monkeypatch.setattr(cli, "save_port", lambda _port: None)
    monkeypatch.setattr(cli.time, "sleep", lambda _seconds: None)

    def fake_trim(path):
        assert path == log_file
        events.append("trim")

    monkeypatch.setattr("hermit.server.trim_log_file", fake_trim)

    class FakeProcess:
        pid = 12345

        @staticmethod
        def poll():
            return None

    def fake_popen(command, **kwargs):
        events.append("spawn")
        assert events == ["trim", "spawn"]
        assert command[1:3] == ["-m", "hermit.server"]
        assert kwargs["stdout"] == subprocess.DEVNULL
        assert kwargs["stderr"] == subprocess.DEVNULL
        return FakeProcess()

    monkeypatch.setattr(cli.subprocess, "Popen", fake_popen)

    class ReadyResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        @staticmethod
        def read():
            return json.dumps({"status": "ready"}).encode()

    monkeypatch.setattr(cli.urllib.request, "urlopen", lambda *_args, **_kwargs: ReadyResponse())

    with pytest.raises(SystemExit) as exc_info:
        cli.cmd_start(SimpleNamespace(verbose=False))

    assert exc_info.value.code == 0


@pytest.mark.parametrize("fail_startup", [False, True])
def test_server_runner_records_logs_and_rotates_without_console_noise(tmp_path, fail_startup):
    """Exercise real Uvicorn and file handlers with a model-free ASGI app."""
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    script = '''
import logging
import sys
import types
from contextlib import asynccontextmanager
from fastapi import FastAPI
from hermit import server

fail_startup = sys.argv.pop() == "fail"

@asynccontextmanager
async def lifespan(app):
    if fail_startup:
        raise RuntimeError("startup failure sentinel")
    for i in range(100):
        logging.getLogger("hermit.probe").info("rotation sentinel %s", i)
    yield

module = types.ModuleType("hermit.app")
module.app = FastAPI(lifespan=lifespan)
@module.app.get("/health")
def health():
    logging.getLogger("hermit.probe").info("health sentinel")
    return {"status": "ready"}
sys.modules["hermit.app"] = module
server.LOG_MAX_BYTES = 1024
server.main()
'''
    env = {**os.environ, "HERMIT_HOME": str(tmp_path)}
    proc = subprocess.Popen(
        [sys.executable, "-c", script, "--host", "127.0.0.1", "--port", str(port),
         "fail" if fail_startup else "ready"],
        env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    try:
        if fail_startup:
            out, err = proc.communicate(timeout=15)
            assert proc.returncode != 0
        else:
            deadline = time.monotonic() + 15
            while time.monotonic() < deadline:
                assert proc.poll() is None, "server exited before ready"
                try:
                    with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1) as resp:
                        assert json.loads(resp.read())["status"] == "ready"
                    break
                except (urllib.error.URLError, TimeoutError):
                    time.sleep(0.1)
            else:
                pytest.fail("probe server did not become ready")
            proc.terminate()
            out, err = proc.communicate(timeout=10)
            assert proc.returncode in (0, -15)
        assert out == b""
        assert err == b""
        logs = list((tmp_path / "logs").glob("hermit.log*"))
        content = "\n".join(path.read_text() for path in logs)
        if fail_startup:
            assert "startup failure sentinel" in content
        else:
            assert {path.name for path in logs} == {"hermit.log", "hermit.log.1"}
            assert all(path.stat().st_size <= 1024 for path in logs)
            assert "health sentinel" in content
            assert 'GET /health HTTP/1.1' in content
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.communicate(timeout=10)
