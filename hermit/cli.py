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

All commands output JSON to stdout. Use --pretty for indented output.
Errors: {"error": "message"} with non-zero exit code.
"""

import argparse
import json
import logging
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

from hermit.config import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_RERANK_CANDIDATES,
    DEFAULT_TOP_K,
    HERMIT_HOME,
    HOST,
    LOG_DIR,
    PID_FILE,
    PORT,
)

logger = logging.getLogger(__name__)

# ── Global output state (set by --pretty flag) ──────────────────
_pretty = False


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
    url = f"http://127.0.0.1:{PORT}{path}"
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


def _is_port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", port)) == 0


# ── Server commands ─────────────────────────────────────────────


def cmd_start(_args):
    pid = _read_pid()
    if pid is not None:
        _output({"status": "already_running", "pid": pid, "port": PORT})

    if _is_port_in_use(PORT):
        _error(f"port {PORT} is already in use")

    # Ensure directories exist
    HERMIT_HOME.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    log_file = LOG_DIR / "hermit.log"

    # Spawn uvicorn as a detached process
    with open(log_file, "a") as lf:
        proc = subprocess.Popen(
            [
                sys.executable, "-m", "uvicorn",
                "hermit.app:app",
                "--host", HOST,
                "--port", str(PORT),
            ],
            stdout=lf,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    PID_FILE.parent.mkdir(parents=True, exist_ok=True)
    PID_FILE.write_text(str(proc.pid))

    # Wait for server to become ready (up to 120s for model downloads)
    for _ in range(120):
        time.sleep(1)
        try:
            os.kill(proc.pid, 0)
        except ProcessLookupError:
            PID_FILE.unlink(missing_ok=True)
            _error(f"server process exited unexpectedly, check {log_file}")

        try:
            url = f"http://127.0.0.1:{PORT}/health"
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=3) as resp:
                health = json.loads(resp.read())
        except Exception:
            continue

        if health.get("status") == "ready":
            _output({"status": "started", "pid": proc.pid, "port": PORT})

    _output({"status": "starting", "pid": proc.pid, "port": PORT,
             "warning": "health check timed out, server may still be loading models"})


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

    try:
        url = f"http://127.0.0.1:{PORT}/health"
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

    try:
        register(args.name, str(folder), args.chunk_size, args.chunk_overlap)
    except ValueError as e:
        _error(str(e))

    _output({"status": "added", "name": args.name, "folder_path": str(folder)})


def cmd_kb_remove(args):
    from hermit.storage.registry import get_all, unregister
    from hermit.storage.metadata import MetadataStore

    existing = get_all()
    if args.name not in existing:
        _error(f"collection '{args.name}' not found")

    unregister(args.name)
    MetadataStore(args.name).destroy()
    _output({"status": "removed", "name": args.name})


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
    kb_add.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE,
                        help=f"Chunk size in characters (default: {DEFAULT_CHUNK_SIZE})")
    kb_add.add_argument("--chunk-overlap", type=int, default=DEFAULT_CHUNK_OVERLAP,
                        help=f"Chunk overlap in characters (default: {DEFAULT_CHUNK_OVERLAP})")

    kb_rm = kb_sub.add_parser("remove", help="Remove a knowledge base")
    kb_rm.add_argument("name", help="Collection name")

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
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
