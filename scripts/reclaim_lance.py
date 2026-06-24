"""One-shot reclaim of orphaned LanceDB index directories.

LanceDB's ``optimize``/``cleanup_old_versions`` reaps stale data versions but
never removes orphaned ``_indices/<uuid>`` directories, so a Hermit data dir
that optimized the FTS index on every file write balloons to hundreds of GB
against a few hundred MB of logical data. This rebuilds each collection from
its live rows, collapsing on-disk size back to the logical data size.

Run with the Hermit server STOPPED (the rebuild drops + recreates each table
and must not race the server's indexing writes):

    pkill -f 'uvicorn hermit.app'      # or stop however you started it
    uv run python scripts/reclaim_lance.py

Honors HERMIT_HOME (defaults to ~/.hermit), so it operates on the real data
dir, not a test fixture.
"""

from __future__ import annotations

from pathlib import Path

from hermit.config import DATA_ROOT
from hermit.storage import lance


def _dir_size(p: Path) -> int:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def main() -> None:
    lance_dir = DATA_ROOT / "lance"
    if not lance_dir.is_dir():
        print(f"No lance dir at {lance_dir}; nothing to do.")
        return

    conn = lance.db()
    tables = list(conn.list_tables().tables)
    print(f"Data dir: {DATA_ROOT}")
    print(f"Collections: {tables}\n")

    total_before = _dir_size(lance_dir)
    for name in tables:
        tdir = lance_dir / f"{name}.lance"
        before = _dir_size(tdir)
        idx_before = lance._index_dir_count(name)
        print(f"[{name}] before: {before / 1e9:.2f} GB, {idx_before} index dirs")
        lance.vacuum_collection(name)
        after = _dir_size(tdir)
        idx_after = lance._index_dir_count(name)
        print(
            f"[{name}] after:  {after / 1e9:.2f} GB, {idx_after} index dirs "
            f"(reclaimed {(before - after) / 1e9:.2f} GB)\n"
        )

    total_after = _dir_size(lance_dir)
    print(
        f"TOTAL: {total_before / 1e9:.2f} GB -> {total_after / 1e9:.2f} GB "
        f"(reclaimed {(total_before - total_after) / 1e9:.2f} GB)"
    )


if __name__ == "__main__":
    main()
