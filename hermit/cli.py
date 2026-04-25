"""Hermit CLI — all commands output JSON by default (agent-first design).

Usage:
    hermit start                          # start server in background
    hermit stop                           # stop the running server
    hermit status                         # show server health (JSON)
    hermit logs                           # tail server logs (streaming)
    hermit download [--force]             # download models
    hermit search <collection> <query>    # semantic search
    hermit kb add <name> <dir>            # add a knowledge base
    hermit kb remove <name>               # remove a knowledge base
    hermit kb list                        # list knowledge bases
    hermit collection status <name>       # collection indexing status
    hermit collection sync <name>         # trigger sync
    hermit collection tasks <name>        # indexing task status
    hermit install-skills                 # install skills to ~/.agents/skills/
    hermit install-skills --uninstall     # remove installed skills

All commands output JSON to stdout. Use --pretty for indented output.
Errors: {"error": "message"} with non-zero exit code.
"""

import argparse
import json
import logging
import os
import re
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request

from hermit.config import (
    DEFAULT_RERANK_CANDIDATES,
    DEFAULT_TOP_K,
    HERMIT_HOME,
    HOST,
    LOG_DIR,
    PID_FILE,
    load_port,
    resolve_port,
    save_port,
)

logger = logging.getLogger(__name__)

# ── Global output state (set by --pretty flag) ──────────────────
_pretty = False

# Uvicorn HTTP access log lines (e.g. '127.0.0.1 - "GET /health ..." 200 OK')
# are filtered out during startup tailing so users only see hermit app logs.
_ACCESS_LOG_RE = re.compile(r'\d+\.\d+\.\d+\.\d+.*HTTP')


def _tail_log(log_path, last_pos: int) -> tuple[int, list[str]]:
    """Return new lines written to log_path since last_pos (byte offset).

    Filters out uvicorn HTTP access log lines to reduce noise.
    Returns (new_byte_pos, list_of_new_lines).
    """
    try:
        with open(log_path, "rb") as f:
            f.seek(last_pos)
            data = f.read()
        new_pos = last_pos + len(data)
        lines = [
            line for line in data.decode(errors="replace").splitlines()
            if line.strip() and not _ACCESS_LOG_RE.search(line)
        ]
        return new_pos, lines
    except OSError:
        return last_pos, []


# ── JSON output helpers ─────────────────────────────────────────


def _output(data: dict) -> None:
    """Print JSON to stdout and exit 0."""
    indent = 2 if _pretty else None
    print(json.dumps(data, indent=indent, ensure_ascii=False))
    raise SystemExit(0)


def _error(msg: str, code: int = 1) -> None:
    """Print JSON error to stdout and exit with code."""
    indent = 2 if _pretty else None
    print(json.dumps({"error": msg}, indent=indent, ensure_ascii=False))
    raise SystemExit(code)


# ── HTTP client helper ──────────────────────────────────────────


def _api_request(method: str, path: str, body: dict | None = None) -> dict:
    """Send HTTP request to the running server. Returns parsed JSON.

    Raises SystemExit with JSON error on failure.
    """
    url = f"http://127.0.0.1:{load_port()}{path}"
    data = json.dumps(body).encode() if body else None
    headers = {"Content-Type": "application/json"} if body else {}
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        try:
            detail = json.loads(e.read()).get("detail", str(e))
        except Exception:
            detail = str(e)
        _error(detail, code=1)
    except urllib.error.URLError:
        _error("server is not running (connection refused)")
    except TimeoutError:
        _error("server request timed out")
    return {}  # unreachable


def _api_request_noexit(
    method: str,
    path: str,
    body: dict | None = None,
) -> tuple[dict | None, str | None, int | None]:
    """Send HTTP request and return (result, error_message, status_code)."""
    url = f"http://127.0.0.1:{load_port()}{path}"
    data = json.dumps(body).encode() if body else None
    headers = {"Content-Type": "application/json"} if body else {}
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read()), None, resp.status
    except urllib.error.HTTPError as e:
        try:
            detail = json.loads(e.read()).get("detail", str(e))
        except Exception:
            detail = str(e)
        return None, detail, e.code
    except urllib.error.URLError:
        return None, "server is not running (connection refused)", None
    except TimeoutError:
        return None, "server request timed out", None


# ── Internal helpers ────────────────────────────────────────────


def _read_pid() -> int | None:
    """Read PID from file, return None if missing or stale."""
    if not PID_FILE.exists():
        return None
    try:
        pid = int(PID_FILE.read_text().strip())
        os.kill(pid, 0)  # check if process is alive
        return pid
    except (ValueError, ProcessLookupError, PermissionError):
        PID_FILE.unlink(missing_ok=True)
        return None


# ── Server commands ─────────────────────────────────────────────


def cmd_start(_args):
    pid = _read_pid()
    if pid is not None:
        _output({"status": "already_running", "pid": pid, "port": load_port()})

    port = resolve_port()
    save_port(port)

    # Ensure directories exist
    HERMIT_HOME.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    log_file = LOG_DIR / "hermit.log"

    # Spawn uvicorn as a single-process daemon.
    # Concurrency is handled by an internal ThreadPoolExecutor (search) and
    # daemon threads (indexing/watching), avoiding multi-process duplication.
    with open(log_file, "a") as lf:
        proc = subprocess.Popen(
            [
                sys.executable, "-m", "uvicorn",
                "hermit.app:app",
                "--host", HOST,
                "--port", str(port),
            ],
            stdout=lf,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    PID_FILE.parent.mkdir(parents=True, exist_ok=True)
    PID_FILE.write_text(str(proc.pid))

    # Wait for server to become ready.
    # Default 300s: standalone mode may need to pull the Qdrant Docker image
    # on first run, which can take several minutes on slow networks.
    # Override with HERMIT_START_TIMEOUT env var (seconds).
    start_timeout = int(os.environ.get("HERMIT_START_TIMEOUT", 300))

    log_pos: int = 0
    last_output_t = time.monotonic()  # tracks when we last printed to stderr

    for elapsed in range(start_timeout):
        time.sleep(1)
        try:
            os.kill(proc.pid, 0)
        except ProcessLookupError:
            PID_FILE.unlink(missing_ok=True)
            _error(f"server process exited unexpectedly, check {log_file}")

        # ── Stream new server log lines to stderr ───────────────
        log_pos, new_lines = _tail_log(log_file, log_pos)
        for line in new_lines:
            print(line, file=sys.stderr, flush=True)
            last_output_t = time.monotonic()

        # Heartbeat: if the log has been silent for >10s, reassure the user.
        if time.monotonic() - last_output_t > 10:
            print(
                f"[hermit] still starting... ({elapsed + 1}s elapsed)",
                file=sys.stderr, flush=True,
            )
            last_output_t = time.monotonic()

        # ── Poll /health ────────────────────────────────────────
        try:
            url = f"http://127.0.0.1:{port}/health"
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=3) as resp:
                health = json.loads(resp.read())
        except Exception:
            continue

        if health.get("status") == "ready":
            print(
                f"[hermit] server ready ({elapsed + 1}s elapsed)",
                file=sys.stderr, flush=True,
            )
            _output({"status": "started", "pid": proc.pid, "port": port})

    _output({"status": "starting", "pid": proc.pid, "port": port,
             "warning": (
                 f"health check timed out after {start_timeout}s, server may still be loading. "
                 f"Check logs: {log_file}. "
                 "Increase timeout with HERMIT_START_TIMEOUT env var."
             )})


def cmd_stop(_args):
    pid = _read_pid()
    if pid is None:
        _error("server is not running")

    os.kill(pid, signal.SIGTERM)

    # Wait for graceful shutdown (up to 10s)
    for _ in range(10):
        time.sleep(1)
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            PID_FILE.unlink(missing_ok=True)
            _output({"status": "stopped", "pid": pid})

    # Force kill
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    PID_FILE.unlink(missing_ok=True)
    _output({"status": "killed", "pid": pid})


def cmd_status(_args):
    pid = _read_pid()
    if pid is None:
        _output({"status": "stopped"})

    port = load_port()
    try:
        url = f"http://127.0.0.1:{port}/health"
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=3) as resp:
            health = json.loads(resp.read())
    except Exception:
        _output({"status": "starting", "pid": pid})

    health["pid"] = pid
    _output(health)


def cmd_logs(_args):
    """Tail server logs — streaming output, not JSON."""
    log_file = LOG_DIR / "hermit.log"
    if not log_file.exists():
        _error("no log file found")

    try:
        subprocess.run(["tail", "-f", str(log_file)])
    except KeyboardInterrupt:
        pass


# ── Download command ────────────────────────────────────────────


def cmd_download(args):
    from hermit.models import MODELS, download_all, verify_models

    download_all(force=args.force)
    if not args.skip_verify:
        verify_models()
    _output({"status": "complete", "models": [m["repo_id"] for m in MODELS]})


# ── Search command ──────────────────────────────────────────────


def cmd_search(args):
    body = {
        "query": args.query,
        "collection": args.collection,
        "top_k": args.top_k,
        "rerank_candidates": args.rerank_candidates,
    }
    result = _api_request("POST", "/search", body)
    _output(result)


# ── Knowledge base management ───────────────────────────────────


def cmd_kb_add(args):
    from pathlib import Path
    from hermit.storage.registry import register

    folder = Path(args.dir).resolve()
    if not folder.is_dir():
        _error(f"'{folder}' is not a directory")

    result = {
        "status": "added",
        "name": args.name,
        "folder_path": str(folder),
    }
    if args.ignore:
        result["ignore_patterns"] = args.ignore
    if args.ignore_ext:
        result["ignore_extensions"] = [e.lower() for e in args.ignore_ext]

    pid = _read_pid()
    if pid is not None:
        api_result, error, _status = _api_request_noexit(
            "POST",
            "/collections",
            {
                "name": args.name,
                "folder_path": str(folder),
                "ignore_patterns": args.ignore or [],
                "ignore_extensions": args.ignore_ext or [],
            },
        )
        if api_result is None:
            _error(error or "failed to add collection")
        _output(api_result)

    try:
        register(
            args.name,
            str(folder),
            ignore_patterns=args.ignore or None,
            ignore_extensions=args.ignore_ext or None,
        )
    except ValueError as e:
        _error(str(e))
    _output(result)


def _remove_collection_local(name: str) -> dict:
    from hermit.storage import qdrant
    from hermit.storage.metadata import MetadataStore
    from hermit.storage.registry import get_all, unregister

    existing = get_all()
    if name not in existing:
        _error(f"collection '{name}' not found")

    qdrant.delete_collection(name)
    MetadataStore(name).destroy()
    unregister(name)
    return {"status": "removed", "name": name}


def cmd_kb_remove(args):
    pid = _read_pid()
    if pid is not None:
        result, error, status = _api_request_noexit("DELETE", f"/collections/{args.name}")
        if result is None:
            from hermit.storage.registry import get_all

            existing = get_all()
            if status == 404 and args.name in existing:
                result = _remove_collection_local(args.name)
            else:
                _error(error or "failed to remove collection")
    else:
        result = _remove_collection_local(args.name)
    _output(result)


def cmd_kb_update(args):
    from hermit.storage.registry import get_all, update

    existing = get_all()
    if args.name not in existing:
        _error(f"collection '{args.name}' not found")

    try:
        update(
            args.name,
            ignore_patterns=args.ignore or None,
            ignore_extensions=args.ignore_ext or None,
            clear_ignore=args.clear_ignore,
            clear_ignore_ext=args.clear_ignore_ext,
        )
    except ValueError as e:
        _error(str(e))

    updated = get_all()[args.name]
    _output({"status": "updated", "name": args.name, **updated})


def cmd_kb_list(_args):
    from hermit.storage.registry import get_all

    _output({"collections": get_all()})


# ── Collection commands (HTTP forwarding) ───────────────────────


def cmd_collection_status(args):
    result = _api_request("GET", f"/collections/{args.name}/status")
    _output(result)


def cmd_collection_sync(args):
    result = _api_request("POST", f"/collections/{args.name}/sync")
    _output(result)


def cmd_collection_tasks(args):
    result = _api_request("GET", f"/collections/{args.name}/tasks")
    _output(result)


# ── Skill distribution ──────────────────────────────────────────


def _find_skills_dir() -> "Path | None":
    """Locate the skills source directory.

    Priority:
    1. Package-internal: hermit/_skills/ (installed mode)
    2. Repository: .agents/skills/ relative to project root (dev mode)
    """
    from pathlib import Path

    # 1. Package-internal (_skills/ injected by hatchling force-include)
    pkg_skills = Path(__file__).parent / "_skills"
    if pkg_skills.is_dir():
        return pkg_skills

    # 2. Dev mode: walk up from hermit/ to repo root, look for .agents/skills/
    repo_root = Path(__file__).parent.parent
    dev_skills = repo_root / ".agents" / "skills"
    if dev_skills.is_dir():
        return dev_skills

    return None


# Project name used as origin marker for installed skills
_PROJECT_NAME = "hermit"


def cmd_install_skills(args):
    import shutil
    from pathlib import Path

    skills_src = _find_skills_dir()
    if skills_src is None:
        _error("no skills directory found")

    # Install directly under ~/.agents/skills/{skill_name}/
    global_skills_dir = Path.home() / ".agents" / "skills"
    # Collect skill names from source
    skill_names = [d.name for d in skills_src.iterdir() if d.is_dir() and (d / "SKILL.md").exists()]

    if not skill_names:
        _error("no skills found in source directory")

    if args.uninstall:
        removed = []
        for name in skill_names:
            target = global_skills_dir / name
            if target.exists():
                shutil.rmtree(target)
                removed.append(name)
        _output({"status": "uninstalled", "skills": removed})

    # Install
    global_skills_dir.mkdir(parents=True, exist_ok=True)
    installed = []
    for name in skill_names:
        src = skills_src / name
        dst = global_skills_dir / name
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        # Write origin marker for version tracking
        origin = dst / ".origin"
        origin.write_text(json.dumps({"package": _PROJECT_NAME, "version": "0.1.0"}))
        installed.append(name)

    _output({"status": "installed", "skills": installed, "target": str(global_skills_dir)})


# ── Argument parser ─────────────────────────────────────────────


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    parser = argparse.ArgumentParser(prog="hermit", description="Hermit CLI (JSON output)")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output")
    sub = parser.add_subparsers(dest="command")

    # Server lifecycle
    sub.add_parser("start", help="Start the server in background")
    sub.add_parser("stop", help="Stop the running server")
    sub.add_parser("status", help="Show server status")
    sub.add_parser("logs", help="Tail server logs (streaming, not JSON)")

    # Model download
    dl = sub.add_parser("download", help="Download all required models")
    dl.add_argument("--force", action="store_true", help="Force re-download")
    dl.add_argument("--skip-verify", action="store_true", help="Skip model verification")

    # Search
    sr = sub.add_parser("search", help="Semantic search")
    sr.add_argument("collection", help="Collection name")
    sr.add_argument("query", help="Search query")
    sr.add_argument("--top-k", type=int, default=DEFAULT_TOP_K,
                    help=f"Number of results (default: {DEFAULT_TOP_K})")
    sr.add_argument("--rerank-candidates", type=int, default=DEFAULT_RERANK_CANDIDATES,
                    help=f"Rerank candidate pool size (default: {DEFAULT_RERANK_CANDIDATES})")

    # kb subcommand
    kb = sub.add_parser("kb", help="Manage knowledge base collections")
    kb_sub = kb.add_subparsers(dest="kb_command")

    kb_add = kb_sub.add_parser("add", help="Add a knowledge base directory")
    kb_add.add_argument("name", help="Collection alias")
    kb_add.add_argument("dir", help="Path to the directory")
    kb_add.add_argument("--ignore", action="append", default=[],
                        help="Glob pattern for paths to ignore (repeatable)")
    kb_add.add_argument("--ignore-ext", action="append", default=[],
                        help="File extension to ignore, e.g. .pdf (repeatable)")

    kb_rm = kb_sub.add_parser("remove", help="Remove a knowledge base")
    kb_rm.add_argument("name", help="Collection name")

    kb_update = kb_sub.add_parser("update", help="Update ignore rules for a knowledge base")
    kb_update.add_argument("name", help="Collection name")
    kb_update.add_argument("--ignore", action="append", default=[],
                           help="Glob pattern for paths to ignore (replaces existing, repeatable)")
    kb_update.add_argument("--ignore-ext", action="append", default=[],
                           help="File extension to ignore, e.g. .pdf (replaces existing, repeatable)")
    kb_update.add_argument("--clear-ignore", action="store_true",
                           help="Clear all path ignore patterns")
    kb_update.add_argument("--clear-ignore-ext", action="store_true",
                           help="Clear all extension ignore patterns")

    kb_sub.add_parser("list", help="List all knowledge bases")

    # collection subcommand (HTTP forwarding)
    col = sub.add_parser("collection", help="Query collection state (requires running server)")
    col_sub = col.add_subparsers(dest="col_command")

    col_status = col_sub.add_parser("status", help="Collection indexing status")
    col_status.add_argument("name", help="Collection name")

    col_sync = col_sub.add_parser("sync", help="Trigger collection sync")
    col_sync.add_argument("name", help="Collection name")

    col_tasks = col_sub.add_parser("tasks", help="Indexing task status")
    col_tasks.add_argument("name", help="Collection name")

    # Skill distribution
    sk = sub.add_parser("install-skills", help="Install agent skills to ~/.agents/skills/")
    sk.add_argument("--uninstall", action="store_true", help="Remove installed skills")

    args = parser.parse_args()

    global _pretty
    _pretty = args.pretty

    if args.command == "start":
        cmd_start(args)
    elif args.command == "stop":
        cmd_stop(args)
    elif args.command == "status":
        cmd_status(args)
    elif args.command == "logs":
        cmd_logs(args)
    elif args.command == "download":
        cmd_download(args)
    elif args.command == "search":
        cmd_search(args)
    elif args.command == "kb":
        if args.kb_command == "add":
            cmd_kb_add(args)
        elif args.kb_command == "remove":
            cmd_kb_remove(args)
        elif args.kb_command == "update":
            cmd_kb_update(args)
        elif args.kb_command == "list":
            cmd_kb_list(args)
        else:
            kb.print_help()
    elif args.command == "collection":
        if args.col_command == "status":
            cmd_collection_status(args)
        elif args.col_command == "sync":
            cmd_collection_sync(args)
        elif args.col_command == "tasks":
            cmd_collection_tasks(args)
        else:
            col.print_help()
    elif args.command == "install-skills":
        cmd_install_skills(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
