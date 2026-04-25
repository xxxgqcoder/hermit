import sqlite3
import threading
import time
from pathlib import Path

from hermit.config import DATA_ROOT


class MetadataStore:
    """SQLite-based metadata store for tracking indexed files."""

    _instances: dict[str, "MetadataStore"] = {}
    _cls_lock = threading.Lock()

    def __new__(cls, collection_name: str):
        with cls._cls_lock:
            if collection_name in cls._instances:
                return cls._instances[collection_name]
            instance = super().__new__(cls)
            cls._instances[collection_name] = instance
            return instance

    def __init__(self, collection_name: str):
        if hasattr(self, "_initialized"):
            return
        db_dir = DATA_ROOT / "metadata"
        db_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = db_dir / f"{collection_name}.db"
        self._collection_name = collection_name
        self._local = threading.local()
        self._initialized = True
        self._init_db()

    def _init_db(self):
        with self._conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS files (
                    file_path TEXT PRIMARY KEY,
                    file_hash TEXT NOT NULL,
                    file_mtime REAL NOT NULL,
                    chunk_count INTEGER NOT NULL,
                    last_indexed_at REAL NOT NULL
                )
            """)

    def _conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(str(self.db_path))
            self._local.conn = conn
        return conn

    def _reset_conn(self) -> None:
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            self._local.conn = None

    def _recover_missing_table(self) -> None:
        self._reset_conn()
        self._init_db()

    def get_all_records(self) -> dict[str, tuple[str, float]]:
        """Return {file_path: (file_hash, file_mtime)} for all indexed files."""
        try:
            with self._conn() as conn:
                rows = conn.execute("SELECT file_path, file_hash, file_mtime FROM files").fetchall()
        except sqlite3.OperationalError as e:
            if "no such table: files" not in str(e):
                raise
            self._recover_missing_table()
            with self._conn() as conn:
                rows = conn.execute("SELECT file_path, file_hash, file_mtime FROM files").fetchall()
        return {row[0]: (row[1], row[2]) for row in rows}

    def upsert(self, file_path: str, file_hash: str, file_mtime: float, chunk_count: int):
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO files (file_path, file_hash, file_mtime, chunk_count, last_indexed_at)
                   VALUES (?, ?, ?, ?, ?)
                   ON CONFLICT(file_path) DO UPDATE SET
                       file_hash=excluded.file_hash,
                       file_mtime=excluded.file_mtime,
                       chunk_count=excluded.chunk_count,
                       last_indexed_at=excluded.last_indexed_at""",
                (file_path, file_hash, file_mtime, chunk_count, time.time()),
            )

    def delete(self, file_path: str):
        with self._conn() as conn:
            conn.execute("DELETE FROM files WHERE file_path = ?", (file_path,))

    def get_status(self) -> dict:
        try:
            with self._conn() as conn:
                row = conn.execute(
                    "SELECT COUNT(*), SUM(chunk_count) FROM files"
                ).fetchone()
        except sqlite3.OperationalError as e:
            if "no such table: files" not in str(e):
                raise
            self._recover_missing_table()
            with self._conn() as conn:
                row = conn.execute(
                    "SELECT COUNT(*), SUM(chunk_count) FROM files"
                ).fetchone()
        return {
            "indexed_files": row[0] or 0,
            "total_chunks": row[1] or 0,
        }

    def get_chunk_count(self, file_path: str) -> int:
        """Return chunk_count for a file, or 0 if not found."""
        try:
            with self._conn() as conn:
                row = conn.execute(
                    "SELECT chunk_count FROM files WHERE file_path = ?", (file_path,)
                ).fetchone()
        except sqlite3.OperationalError as e:
            if "no such table: files" not in str(e):
                raise
            self._recover_missing_table()
            with self._conn() as conn:
                row = conn.execute(
                    "SELECT chunk_count FROM files WHERE file_path = ?", (file_path,)
                ).fetchone()
        return row[0] if row else 0

    def destroy(self):
        # Close thread-local connections
        self._reset_conn()
        if self.db_path.exists():
            self.db_path.unlink()
        with self._cls_lock:
            self._instances.pop(self._collection_name, None)
            # Allow re-initialization
        if hasattr(self, "_initialized"):
            del self._initialized
