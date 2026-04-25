import logging
import threading
import time
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
    file_hash: str | None = None


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

    def cancel_collection(self, collection_name: str) -> dict:
        """Remove queued tasks for a collection and report remaining in-progress work."""
        with self._lock:
            queued_removed = 0
            with self._queue.mutex:
                kept = []
                for task in list(self._queue.queue):
                    if task.collection_name == collection_name:
                        queued_removed += 1
                    else:
                        kept.append(task)
                self._queue.queue.clear()
                self._queue.queue.extend(kept)
                if queued_removed:
                    self._queue.unfinished_tasks = max(
                        0,
                        self._queue.unfinished_tasks - queued_removed,
                    )
                    self._queue.not_full.notify_all()

            self._pending = {
                key for key in self._pending if key[0] != collection_name
            }
            in_progress_count = sum(
                1 for key in self._in_progress if key[0] == collection_name
            )

        return {
            "collection": collection_name,
            "queued_removed": queued_removed,
            "in_progress_tasks": in_progress_count,
        }

    def wait_until_collection_idle(
        self,
        collection_name: str,
        timeout: float = 30.0,
        poll_interval: float = 0.05,
    ) -> bool:
        """Wait until a collection has no in-progress index tasks."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._lock:
                busy = any(key[0] == collection_name for key in self._in_progress)
            if not busy:
                return True
            time.sleep(poll_interval)
        return False

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
            file_hash=task.file_hash,
        )
        if ok:
            logger.info("Indexed by background task: %s", task.file_path)


_QUEUE = _IndexTaskQueue()


def start_task_worker():
    _QUEUE.start()


def enqueue_index_task(
    collection_name: str,
    file_path: str,
    file_hash: str | None = None,
) -> bool:
    return _QUEUE.enqueue(
        IndexTask(
            collection_name=collection_name,
            file_path=file_path,
            file_hash=file_hash,
        )
    )


def get_collection_task_status(collection_name: str) -> dict:
    return _QUEUE.get_status(collection_name)


def cancel_collection_tasks(collection_name: str) -> dict:
    return _QUEUE.cancel_collection(collection_name)


def wait_for_collection_tasks_idle(
    collection_name: str,
    timeout: float = 30.0,
    poll_interval: float = 0.05,
) -> bool:
    return _QUEUE.wait_until_collection_idle(
        collection_name,
        timeout=timeout,
        poll_interval=poll_interval,
    )
