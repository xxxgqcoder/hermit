"""Tests for bounded server logging."""

import json
import subprocess
from pathlib import Path

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
        cli.cmd_start(None)

    assert exc_info.value.code == 0
