import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from queue import Queue

from hermit.config import INDEX_WORKERS
from hermit.storage import qdrant
from hermit.storage.qdrant import CollectionCorruptedError
from hermit.storage.metadata import MetadataStore

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IndexTask:
    collection_name: str
    file_path: str


class _IndexTaskQueue:
    def __init__(self, num_workers: int = INDEX_WORKERS):
        self._queue: Queue[IndexTask] = Queue()
        self._pending: set[tuple[str, str]] = set()
        self._in_progress: set[tuple[str, str]] = set()
        self._lock = threading.Lock()
        self._workers: list[threading.Thread] = []
        self._num_workers = max(1, num_workers)

    def start(self):
        with self._lock:
            alive = [w for w in self._workers if w.is_alive()]
            if len(alive) >= self._num_workers:
                return
            # Launch missing workers
            for i in range(self._num_workers - len(alive)):
                idx = len(alive) + i
                w = threading.Thread(
                    target=self._run,
                    name=f"index-worker-{idx}",
                    daemon=True,
                )
                w.start()
                alive.append(w)
            self._workers = alive
            logger.info(
                "Index worker pool: %d workers running", len(self._workers)
            )

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
            alive = [w for w in self._workers if w.is_alive()]

        return {
            "collection": collection_name,
            "pending_tasks": len(pending),
            "queued_tasks": len(queued),
            "in_progress_tasks": len(in_progress),
            "worker_alive": len(alive) > 0,
        }

    def _handle_task(self, task: IndexTask):
        from hermit.ingestion.scanner import _index_file

        file_path = Path(task.file_path)
        meta = MetadataStore(task.collection_name)

        if not file_path.exists() or not file_path.is_file():
            try:
                qdrant.delete_by_source_file(task.collection_name, task.file_path)
            except CollectionCorruptedError:
                logger.warning(
                    "Collection '%s' was recreated due to data corruption; "
                    "clearing metadata — re-scan to re-index all files",
                    task.collection_name,
                )
                meta.destroy()
                return
            meta.delete(task.file_path)
            logger.info("Skipped stale task and cleaned missing file: %s", task.file_path)
            return

        ok = _index_file(
            task.collection_name,
            file_path,
            meta,
        )
        if ok:
            logger.info("Indexed by background task: %s", task.file_path)


_QUEUE = _IndexTaskQueue()


def start_task_worker():
    _QUEUE.start()


def enqueue_index_task(
    collection_name: str,
    file_path: str,
) -> bool:
    return _QUEUE.enqueue(
        IndexTask(
            collection_name=collection_name,
            file_path=file_path,
        )
    )


def get_collection_task_status(collection_name: str) -> dict:
    return _QUEUE.get_status(collection_name)
