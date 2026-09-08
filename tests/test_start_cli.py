"""Startup output contract without starting models or touching the user's service."""

import io
import json
import subprocess
from types import SimpleNamespace

import pytest

import hermit.cli as cli


@pytest.fixture
def startup(monkeypatch, tmp_path):
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    log_file = log_dir / "hermit.log"
    log_file.write_text("old startup 日志\n" * 1000)
    proc = SimpleNamespace(pid=12345, poll=lambda: None)
    state = SimpleNamespace(log_file=log_file, proc=proc, health="ready", seconds=0)

    monkeypatch.setattr(cli, "HERMIT_HOME", tmp_path)
    monkeypatch.setattr(cli, "LOG_DIR", log_dir)
    monkeypatch.setattr(cli, "PID_FILE", tmp_path / "hermit.pid")
    monkeypatch.setattr(cli, "_read_pid", lambda: None)
    monkeypatch.setattr(cli, "resolve_port", lambda: 8000)
    monkeypatch.setattr(cli, "save_port", lambda port: None)
    monkeypatch.setattr(cli, "_pretty", False)
    monkeypatch.setenv("HERMIT_START_TIMEOUT", "12")

    def spawn(*args, **kwargs):
        assert kwargs["stderr"] == subprocess.DEVNULL
        assert kwargs["stdout"] == subprocess.DEVNULL
        assert kwargs["start_new_session"] is True
        with log_file.open("a") as log:
            log.write("new startup 日志\n" * 1000)
            log.write('127.0.0.1 - "GET /health HTTP/1.1" 200 OK\n')
        return proc

    def sleep(seconds):
        state.seconds += seconds

    monkeypatch.setattr(cli.subprocess, "Popen", spawn)
    monkeypatch.setattr(cli.time, "sleep", sleep)
    monkeypatch.setattr(cli.time, "monotonic", lambda: state.seconds)
    monkeypatch.setattr(
        cli.urllib.request, "urlopen",
        lambda *a, **kw: io.BytesIO(json.dumps({"status": state.health}).encode()),
    )
    return state


@pytest.mark.parametrize("tty", [False, True])
def test_default_start_suppresses_logs(startup, monkeypatch, capsys, tty):
    monkeypatch.setattr(cli.sys.stderr, "isatty", lambda: tty)
    with pytest.raises(SystemExit) as exc:
        cli.cmd_start(SimpleNamespace(verbose=False))
    assert exc.value.code == 0
    out, err = capsys.readouterr()
    assert json.loads(out) == {"status": "started", "pid": 12345, "port": 8000}
    assert "日志" not in err
    if tty:
        assert "starting..." in err
        assert "server ready" in err
        assert len(err.splitlines()) == 2
    else:
        assert err == ""
    assert "new startup 日志" in startup.log_file.read_text()


def test_verbose_only_streams_current_startup(startup, capsys):
    with pytest.raises(SystemExit):
        cli.cmd_start(SimpleNamespace(verbose=True))
    out, err = capsys.readouterr()
    assert json.loads(out)["status"] == "started"
    assert "old startup" not in err
    assert err.count("new startup 日志") == 1000
    assert "GET /health" not in err


def test_verbose_does_not_replay_trimmed_legacy_log(startup, monkeypatch, capsys):
    import hermit.server as server

    original_trim = server.trim_log_file
    monkeypatch.setattr(server, "trim_log_file", lambda path: original_trim(path, 128))
    with pytest.raises(SystemExit):
        cli.cmd_start(SimpleNamespace(verbose=True))
    out, err = capsys.readouterr()
    assert json.loads(out)["status"] == "started"
    assert "old startup" not in err
    assert err.count("new startup 日志") == 1000


@pytest.mark.parametrize("tty", [False, True])
def test_start_timeout_keeps_warning_and_bounded_progress(startup, monkeypatch, capsys, tty):
    monkeypatch.setattr(cli.sys.stderr, "isatty", lambda: tty)
    startup.health = "starting"
    with pytest.raises(SystemExit) as exc:
        cli.cmd_start(SimpleNamespace(verbose=False))
    assert exc.value.code == 0
    out, err = capsys.readouterr()
    result = json.loads(out)
    assert result["status"] == "starting"
    assert str(startup.log_file) in result["warning"]
    assert "12s" in result["warning"]
    assert "server ready" not in err
    assert "日志" not in err
    assert ("still starting..." in err) is tty
    assert len(err.splitlines()) == (2 if tty else 0)


def test_start_failure_preserves_error_and_cleans_pid(startup, capsys):
    startup.proc.poll = lambda: 1
    with pytest.raises(SystemExit) as exc:
        cli.cmd_start(SimpleNamespace(verbose=False))
    assert exc.value.code == 1
    out, err = capsys.readouterr()
    assert str(startup.log_file) in json.loads(out)["error"]
    assert not cli.PID_FILE.exists()
    assert "old startup" not in err


@pytest.mark.parametrize("flags,verbose", [([], False), (["--verbose"], True), (["-v"], True)])
def test_start_parser(monkeypatch, flags, verbose):
    captured = {}
    monkeypatch.setattr(cli, "cmd_start", lambda args: captured.update(verbose=args.verbose))
    monkeypatch.setattr(cli.sys, "argv", ["hermit", "start", *flags])
    cli.main()
    assert captured == {"verbose": verbose}
