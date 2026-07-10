"""Uvicorn runner with bounded on-disk logging."""

import argparse
import shutil
from pathlib import Path

import uvicorn

from hermit.config import HOST, LOG_BACKUP_COUNT, LOG_DIR, LOG_MAX_BYTES


def trim_log_file(log_file: Path, max_bytes: int = LOG_MAX_BYTES) -> None:
    """Keep only the newest complete lines when an existing log is oversized."""
    if max_bytes <= 0 or not log_file.exists() or log_file.stat().st_size <= max_bytes:
        return

    temp_file = log_file.with_suffix(log_file.suffix + ".trim")
    try:
        with log_file.open("rb") as source, temp_file.open("wb") as target:
            source.seek(-max_bytes, 2)
            source.readline()  # Drop the partial line at the retention boundary.
            shutil.copyfileobj(source, target, length=1024 * 1024)
        temp_file.replace(log_file)
    finally:
        temp_file.unlink(missing_ok=True)


def build_log_config(log_file: Path) -> dict:
    """Return a Uvicorn logging config backed by one rotating file handler."""
    return {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "format": "%(asctime)s %(levelname)s %(name)s: %(message)s",
            },
        },
        "handlers": {
            "rotating_file": {
                "class": "logging.handlers.RotatingFileHandler",
                "filename": str(log_file),
                "maxBytes": LOG_MAX_BYTES,
                "backupCount": LOG_BACKUP_COUNT,
                "encoding": "utf-8",
                "formatter": "default",
            },
        },
        "loggers": {
            "uvicorn": {
                "handlers": ["rotating_file"],
                "level": "INFO",
                "propagate": False,
            },
            "uvicorn.error": {
                "level": "INFO",
            },
            "uvicorn.access": {
                "handlers": ["rotating_file"],
                "level": "INFO",
                "propagate": False,
            },
        },
        "root": {
            "handlers": ["rotating_file"],
            "level": "INFO",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Hermit API server")
    parser.add_argument("--host", default=HOST)
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOG_DIR / "hermit.log"
    trim_log_file(log_file)

    uvicorn.run(
        "hermit.app:app",
        host=args.host,
        port=args.port,
        log_config=build_log_config(log_file),
    )


if __name__ == "__main__":
    main()
