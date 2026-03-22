import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from queue import Queue

from hermit.storage import qdrant
from hermit.storage.metadata import MetadataStore

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IndexTask:
    collection_name: str
    file_path: str
    chunk_size: int
    chunk_overlap: int


class _IndexTaskQueue:
    def __init__(self):
        self._queue: Queue[IndexTask] = Queue()
        self._pending: set[tuple[str, str]] = set()
        self._in_progress: set[tuple[str, str]] = set()
        self._lock = threading.Lock()
        self._worker: threading.Thread | None = None

    def start(self):
        with self._lock:
            if self._worker is not None and self._worker.is_alive():
                return
            self._worker = threading.Thread(target=self._run, name="index-task-worker", daemon=True)
            self._worker.start()
            logger.info("Index task worker started")

    def enqueue(self, task: IndexTask) -> bool:
        self.start()
        key = (task.collection_name, task.file_path)
        with self._lock:
            if key in self._pending:
                return False
            self._pending.add(key)
        self._queue.put(task)
        return True

    def _run(self):
        while True:
            task = self._queue.get()
            key = (task.collection_name, task.file_path)
            with self._lock:
                self._in_progress.add(key)
            try:
                self._handle_task(task)
            except Exception:
                logger.exception("Index task failed for %s", task.file_path)
            finally:
                with self._lock:
                    self._in_progress.discard(key)
                    self._pending.discard(key)
                self._queue.task_done()

    def get_status(self, collection_name: str) -> dict:
        with self._lock:
            pending = [k for k in self._pending if k[0] == collection_name]
            in_progress = [k for k in self._in_progress if k[0] == collection_name]
            in_progress_set = set(in_progress)
            queued = [k for k in pending if k not in in_progress_set]
            worker_alive = self._worker is not None and self._worker.is_alive()

        return {
            "collection": collection_name,
            "pending_tasks": len(pending),
            "queued_tasks": len(queued),
            "in_progress_tasks": len(in_progress),
            "worker_alive": worker_alive,
        }

    def _handle_task(self, task: IndexTask):
        from hermit.ingestion.scanner import _index_file

        file_path = Path(task.file_path)
        meta = MetadataStore(task.collection_name)

        if not file_path.exists() or not file_path.is_file():
            qdrant.delete_by_source_file(task.collection_name, task.file_path)
            meta.delete(task.file_path)
            logger.info("Skipped stale task and cleaned missing file: %s", task.file_path)
            return

        ok = _index_file(
            task.collection_name,
            file_path,
            meta,
            chunk_size=task.chunk_size,
            chunk_overlap=task.chunk_overlap,
        )
        if ok:
            logger.info("Indexed by background task: %s", task.file_path)


_QUEUE = _IndexTaskQueue()


def start_task_worker():
    _QUEUE.start()


def enqueue_index_task(
    collection_name: str,
    file_path: str,
    chunk_size: int,
    chunk_overlap: int,
) -> bool:
    return _QUEUE.enqueue(
        IndexTask(
            collection_name=collection_name,
            file_path=file_path,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    )


def get_collection_task_status(collection_name: str) -> dict:
    return _QUEUE.get_status(collection_name)
