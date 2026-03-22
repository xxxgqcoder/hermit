import logging
import threading

from hermit.config import POLL_INTERVAL_SECONDS
from hermit.ingestion.scanner import scan_folder

logger = logging.getLogger(__name__)


class _PollingWatcher:
    """Periodically scans a folder for changes instead of using OS file events.

    Runs scan_folder every POLL_INTERVAL_SECONDS in a daemon thread.
    The first scan is skipped because app.py already runs one at startup.
    """

    def __init__(self, collection_name: str, folder_path: str,
                 ignore_patterns: list[str] | None = None,
                 ignore_extensions: list[str] | None = None):
        self.collection_name = collection_name
        self.folder_path = folder_path
        self.ignore_patterns = ignore_patterns
        self.ignore_extensions = ignore_extensions
        self._stop_event = threading.Event()

    def start(self):
        t = threading.Thread(target=self._run, daemon=True,
                             name=f"poll-{self.collection_name}")
        t.start()

    def stop(self):
        self._stop_event.set()

    def _run(self):
        logger.info("Polling watcher started for '%s' every %ds",
                     self.collection_name, POLL_INTERVAL_SECONDS)
        while not self._stop_event.wait(POLL_INTERVAL_SECONDS):
            try:
                stats = scan_folder(
                    self.collection_name,
                    self.folder_path,
                    defer_indexing=True,
                    ignore_patterns=self.ignore_patterns,
                    ignore_extensions=self.ignore_extensions,
                )
                if stats["added"] or stats["updated"] or stats["deleted"]:
                    logger.info("Poll scan for '%s': %s",
                                self.collection_name, stats)
            except Exception:
                logger.exception("Poll scan failed for '%s'",
                                 self.collection_name)


_watchers: dict[str, _PollingWatcher] = {}


def start_watching(collection_name: str, folder_path: str,
                   ignore_patterns: list[str] | None = None,
                   ignore_extensions: list[str] | None = None):
    """Start periodic polling for a folder."""
    if collection_name in _watchers:
        return  # Already watching

    watcher = _PollingWatcher(collection_name, folder_path,
                              ignore_patterns, ignore_extensions)
    watcher.start()
    _watchers[collection_name] = watcher
    logger.info("Watching '%s' at %s (poll every %ds)",
                collection_name, folder_path, POLL_INTERVAL_SECONDS)


def stop_watching(collection_name: str):
    """Stop polling a folder."""
    watcher = _watchers.pop(collection_name, None)
    if watcher:
        watcher.stop()
        logger.info("Stopped watching '%s'", collection_name)
