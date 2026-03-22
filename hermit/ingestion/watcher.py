import logging
import threading
from pathlib import Path

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileSystemEvent

from hermit.ingestion.scanner import scan_folder

logger = logging.getLogger(__name__)


class _Handler(FileSystemEventHandler):
    """Triggers a full rescan on any file change.

    Debouncing is kept simple: a timer resets on each event, and only fires
    the scan after 2 seconds of quiet.
    """

    def __init__(self, collection_name: str, folder_path: str, chunk_size: int, chunk_overlap: int):
        self.collection_name = collection_name
        self.folder_path = folder_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._timer: threading.Timer | None = None
        self._lock = threading.Lock()

    def _schedule_scan(self):
        with self._lock:
            if self._timer is not None:
                self._timer.cancel()
            self._timer = threading.Timer(2.0, self._do_scan)
            self._timer.daemon = True
            self._timer.start()

    def _do_scan(self):
        logger.info("Watcher triggered rescan for '%s'", self.collection_name)
        try:
            stats = scan_folder(
                self.collection_name,
                self.folder_path,
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                defer_indexing=True,
            )
            logger.info("Watcher rescan done: %s", stats)
        except Exception:
            logger.exception("Watcher rescan failed for '%s'", self.collection_name)

    def on_any_event(self, event: FileSystemEvent):
        if event.is_directory:
            return
        self._schedule_scan()


_observers: dict[str, Observer] = {}


def start_watching(collection_name: str, folder_path: str, chunk_size: int = 512, chunk_overlap: int = 64):
    """Start watching a folder for changes."""
    if collection_name in _observers:
        return  # Already watching

    handler = _Handler(collection_name, folder_path, chunk_size, chunk_overlap)
    observer = Observer()
    observer.schedule(handler, folder_path, recursive=True)
    observer.daemon = True
    observer.start()
    _observers[collection_name] = observer
    logger.info("Watching '%s' at %s", collection_name, folder_path)


def stop_watching(collection_name: str):
    """Stop watching a folder."""
    obs = _observers.pop(collection_name, None)
    if obs:
        obs.stop()
        logger.info("Stopped watching '%s'", collection_name)
