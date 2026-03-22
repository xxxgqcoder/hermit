"""Hermit CLI — deployment & management commands.

Usage:
    hermit start                 # start the server in background
    hermit stop                  # stop the running server
    hermit status                # show server status
    hermit logs                  # tail server logs
    hermit download              # download all models (resumes interrupted)
    hermit download --force      # force re-download
    hermit download --skip-verify
    hermit kb add <name> <dir>   # add a knowledge base directory
    hermit kb remove <name>      # remove a knowledge base
    hermit kb list               # list all knowledge bases
"""

import argparse
import logging
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.request
import urllib.error
import json

from hermit.config import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    HERMIT_HOME,
    HOST,
    LOG_DIR,
    PID_FILE,
    PORT,
)

logger = logging.getLogger(__name__)


# ── Helper utilities ────────────────────────────────────────────


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


def _health_check() -> dict | None:
    """Query the /health endpoint, return parsed JSON or None."""
    try:
        url = f"http://127.0.0.1:{PORT}/health"
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=3) as resp:
            return json.loads(resp.read())
    except Exception:
        return None


# ── Server commands ─────────────────────────────────────────────


def cmd_start(_args):
    pid = _read_pid()
    if pid is not None:
        print(f"Hermit is already running (PID {pid}).")
        raise SystemExit(0)

    if _is_port_in_use(PORT):
        print(f"Error: port {PORT} is already in use.")
        raise SystemExit(1)

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

    print(f"Hermit starting (PID {proc.pid})...")
    print(f"  Log: {log_file}")
    print(f"  Home: {HERMIT_HOME}")

    # Wait for server to become ready (up to 120s for model downloads)
    for i in range(120):
        time.sleep(1)
        # Check process still alive
        try:
            os.kill(proc.pid, 0)
        except ProcessLookupError:
            print("Error: server process exited unexpectedly. Check logs:")
            print(f"  {log_file}")
            PID_FILE.unlink(missing_ok=True)
            raise SystemExit(1)

        health = _health_check()
        if health and health.get("status") == "ready":
            print(f"Hermit is ready at http://127.0.0.1:{PORT}")
            return

    print("Warning: server started but health check timed out.")
    print("It may still be downloading models. Check status with: hermit status")


def cmd_stop(_args):
    pid = _read_pid()
    if pid is None:
        print("Hermit is not running.")
        return

    print(f"Stopping Hermit (PID {pid})...")
    os.kill(pid, signal.SIGTERM)

    # Wait for graceful shutdown (up to 10s)
    for _ in range(10):
        time.sleep(1)
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            PID_FILE.unlink(missing_ok=True)
            print("Hermit stopped.")
            return

    # Force kill
    print("Graceful shutdown timed out, sending SIGKILL...")
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    PID_FILE.unlink(missing_ok=True)
    print("Hermit killed.")


def cmd_status(_args):
    pid = _read_pid()
    if pid is None:
        print("Hermit is not running.")
        return

    print(f"Hermit is running (PID {pid})")

    health = _health_check()
    if health is None:
        print("  Status: starting (not yet responding to health checks)")
        return

    status = health.get("status", "unknown")
    uptime = health.get("uptime", 0)
    mins, secs = divmod(int(uptime), 60)
    hours, mins = divmod(mins, 60)
    print(f"  Status: {status}")
    print(f"  Uptime: {hours}h {mins}m {secs}s")
    print(f"  Models loaded: {health.get('models_loaded', 'unknown')}")

    collections = health.get("collections", [])
    if collections:
        print(f"  Collections ({len(collections)}):")
        for c in collections:
            print(f"    {c['name']}: {c['indexed_files']} files, {c['total_chunks']} chunks")
    else:
        print("  Collections: none")

    pending = health.get("pending_index_tasks", 0)
    if pending > 0:
        print(f"  Indexing: {pending} tasks pending")


def cmd_logs(_args):
    log_file = LOG_DIR / "hermit.log"
    if not log_file.exists():
        print("No log file found.")
        return

    # Tail the log file using subprocess for cross-platform support
    try:
        subprocess.run(["tail", "-f", str(log_file)])
    except KeyboardInterrupt:
        pass


# ── Download command ────────────────────────────────────────────


def cmd_download(args):
    from hermit.models import download_all, verify_models

    download_all(force=args.force)
    if not args.skip_verify:
        verify_models()


# ── Knowledge base management ───────────────────────────────────


def cmd_kb_add(args):
    from pathlib import Path
    from hermit.storage.registry import register

    folder = Path(args.dir).resolve()
    if not folder.is_dir():
        print(f"Error: '{folder}' is not a directory")
        raise SystemExit(1)

    try:
        register(args.name, str(folder), args.chunk_size, args.chunk_overlap)
    except ValueError as e:
        print(f"Error: {e}")
        raise SystemExit(1)

    print(f"Added collection '{args.name}' → {folder}")
    print("Restart the service to apply changes.")


def cmd_kb_remove(args):
    from hermit.storage.registry import get_all, unregister
    from hermit.storage.metadata import MetadataStore

    existing = get_all()
    if args.name not in existing:
        print(f"Error: collection '{args.name}' not found")
        raise SystemExit(1)

    unregister(args.name)
    MetadataStore(args.name).destroy()
    print(f"Removed collection '{args.name}'")
    print("Restart the service to apply changes.")


def cmd_kb_list(_args):
    from hermit.storage.registry import get_all

    collections = get_all()
    if not collections:
        print("No collections registered.")
        return

    for name, cfg in collections.items():
        print(f"  {name}: {cfg['folder_path']}")


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    parser = argparse.ArgumentParser(prog="hermit", description="Hermit management CLI")
    sub = parser.add_subparsers(dest="command")

    # Server lifecycle commands
    sub.add_parser("start", help="Start the server in background")
    sub.add_parser("stop", help="Stop the running server")
    sub.add_parser("status", help="Show server status")
    sub.add_parser("logs", help="Tail server logs")

    # Model download
    dl = sub.add_parser("download", help="Download all required models")
    dl.add_argument("--force", action="store_true", help="Force re-download")
    dl.add_argument("--skip-verify", action="store_true", help="Skip model verification")

    # kb subcommand
    kb = sub.add_parser("kb", help="Manage knowledge base collections")
    kb_sub = kb.add_subparsers(dest="kb_command")

    kb_add = kb_sub.add_parser("add", help="Add a knowledge base directory")
    kb_add.add_argument("name", help="Collection alias (max 64 chars, must be unique)")
    kb_add.add_argument("dir", help="Path to the directory")
    kb_add.add_argument(
        "--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE,
        help=f"Chunk size in characters (default: {DEFAULT_CHUNK_SIZE})",
    )
    kb_add.add_argument(
        "--chunk-overlap", type=int, default=DEFAULT_CHUNK_OVERLAP,
        help=f"Chunk overlap in characters (default: {DEFAULT_CHUNK_OVERLAP})",
    )

    kb_rm = kb_sub.add_parser("remove", help="Remove a knowledge base")
    kb_rm.add_argument("name", help="Collection name")

    kb_sub.add_parser("list", help="List all knowledge bases")

    args = parser.parse_args()
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
    elif args.command == "kb":
        if args.kb_command == "add":
            cmd_kb_add(args)
        elif args.kb_command == "remove":
            cmd_kb_remove(args)
        elif args.kb_command == "list":
            cmd_kb_list(args)
        else:
            kb.print_help()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
