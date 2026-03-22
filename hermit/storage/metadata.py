import sqlite3
import time
from pathlib import Path

from hermit.config import DATA_ROOT


class MetadataStore:
    """SQLite-based metadata store for tracking indexed files."""

    def __init__(self, collection_name: str):
        db_dir = DATA_ROOT / "metadata"
        db_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = db_dir / f"{collection_name}.db"
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
        return sqlite3.connect(str(self.db_path))

    def get_all_records(self) -> dict[str, tuple[str, float]]:
        """Return {file_path: (file_hash, file_mtime)} for all indexed files."""
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
        with self._conn() as conn:
            row = conn.execute(
                "SELECT COUNT(*), SUM(chunk_count) FROM files"
            ).fetchone()
        return {
            "indexed_files": row[0] or 0,
            "total_chunks": row[1] or 0,
        }

    def destroy(self):
        if self.db_path.exists():
            self.db_path.unlink()
