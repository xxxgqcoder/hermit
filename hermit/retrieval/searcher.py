import fnmatch
import logging
from pathlib import PurePath
from typing import Literal

from lancedb.rerankers import RRFReranker

from hermit.config import DEFAULT_RERANK_CANDIDATES, DEFAULT_SEARCH_MODE
from hermit.retrieval import embedder, reranker
from hermit.storage import lance

logger = logging.getLogger(__name__)


SearchMode = Literal["hybrid", "semantic", "keyword", "fuzzy"]

# Modes that run cross-encoder rerank by default.
_RERANK_DEFAULT = {
    "hybrid": True,
    "semantic": True,
    "keyword": False,
    "fuzzy": False,
}

_GLOB_CHARS = set("*?[")


def search(
    collection_name: str,
    query: str,
    top_k: int = 5,
    rerank_candidates: int = DEFAULT_RERANK_CANDIDATES,
    mode: SearchMode = DEFAULT_SEARCH_MODE,
    filename: str | None = None,
    rerank: bool | None = None,
) -> list[dict]:
    """Search a collection. Returns list of result dicts.

    Modes:
      - hybrid:   dense + FTS (BM25) with RRF fusion, then cross-encoder rerank
      - semantic: dense only, then cross-encoder rerank
      - keyword:  FTS only (LanceDB tantivy), rerank off by default
      - fuzzy:    substring scan over text/filename, rerank off by default
    """
    where_clause, glob_pattern = _build_filename_filter(filename)

    if mode == "fuzzy":
        return _fuzzy_search(
            collection_name,
            query=query,
            base_where=where_clause,
            glob_pattern=glob_pattern,
            top_k=top_k,
        )

    candidates = _vector_recall(
        collection_name,
        query=query,
        mode=mode,
        rerank_candidates=rerank_candidates,
        top_k=top_k,
        where_clause=where_clause,
    )

    if glob_pattern is not None:
        candidates = [
            r for r in candidates
            if _glob_matches(r.get("source_file", ""), glob_pattern)
        ]

    if not candidates:
        return []

    do_rerank = _RERANK_DEFAULT[mode] if rerank is None else rerank
    if do_rerank:
        passages = [r["text"] for r in candidates]
        top_indices = reranker.rerank(query, passages, top_k=top_k)
        return [_result_to_dict(candidates[i]) for i in top_indices]

    return [_result_to_dict(r) for r in candidates[:top_k]]


# ── Mode implementations ──────────────────────────────────────────


def _vector_recall(
    collection_name: str,
    query: str,
    mode: SearchMode,
    rerank_candidates: int,
    top_k: int,
    where_clause: str | None,
) -> list[dict]:
    """Run vector / FTS recall for hybrid / semantic / keyword."""
    tbl = lance.open_table(collection_name)
    candidate_limit = max(top_k, rerank_candidates)

    if mode == "hybrid":
        dense_vec = embedder.embed_query_dense(query)
        builder = (
            tbl.search(query_type="hybrid", fts_columns="text")
            .vector(dense_vec)
            .text(query)
            .rerank(RRFReranker())
        )
    elif mode == "semantic":
        dense_vec = embedder.embed_query_dense(query)
        builder = tbl.search(dense_vec, query_type="vector")
    elif mode == "keyword":
        builder = tbl.search(query, query_type="fts", fts_columns="text")
    else:
        raise ValueError(f"unsupported vector recall mode: {mode!r}")

    if where_clause:
        builder = builder.where(where_clause, prefilter=True)
    return builder.limit(candidate_limit).to_list()


def _fuzzy_search(
    collection_name: str,
    query: str,
    base_where: str | None,
    glob_pattern: str | None,
    top_k: int,
) -> list[dict]:
    """LIKE-style substring match on chunk text and/or filename.

    Strategy:
      1. Compose a SQL ``WHERE`` predicate combining the filename filter
         (if any) and a ``contains(lower(text), ...)`` substring match.
      2. Scan the table once via ``table.search().where(...)``. LanceDB
         pushes the predicate down — there is no scroll cursor to manage.
      3. Glob filename patterns can't go into SQL, so post-filter in
         Python after the scan.
      4. Sort by (source_file, chunk_index), trim to top_k.
    """
    needle = query.lower() if query else None
    clauses: list[str] = []
    if base_where:
        clauses.append(base_where)
    if needle:
        clauses.append(f"contains(lower(text), '{_escape_sql(needle)}')")
    where = " AND ".join(clauses) if clauses else None

    tbl = lance.open_table(collection_name)
    builder = tbl.search()
    if where:
        builder = builder.where(where)
    rows = builder.select(
        ["text", "title", "filename", "source_file", "chunk_index", "total_chunks"]
    ).to_list()

    if glob_pattern is not None:
        rows = [
            r for r in rows
            if _glob_matches(r.get("source_file", ""), glob_pattern)
        ]

    rows.sort(key=lambda r: (
        r.get("source_file", ""),
        r.get("chunk_index", 0),
    ))
    return [_result_to_dict(r, score=None) for r in rows[:top_k]]


# ── Filter helpers ────────────────────────────────────────────────


def _build_filename_filter(filename: str | None) -> tuple[str | None, str | None]:
    """Return (sql_where_clause, glob_pattern).

    - Plain substring (no glob chars): ``contains(lower(filename), '...')``.
    - Glob pattern (contains *?[): no SQL prefilter, callers must apply
      ``glob_pattern`` as a Python post-filter against ``source_file``.
    """
    if not filename:
        return None, None
    if any(c in filename for c in _GLOB_CHARS):
        return None, filename.lower()
    needle = filename.lower()
    return f"contains(lower(filename), '{_escape_sql(needle)}')", None


def _escape_sql(value: str) -> str:
    return value.replace("'", "''")


def _glob_matches(source_file: str, pattern: str) -> bool:
    """Match pattern against either the full path or just the basename, lowercased."""
    sf = source_file.lower()
    base = PurePath(sf).name
    return fnmatch.fnmatch(sf, pattern) or fnmatch.fnmatch(base, pattern)


# ── Result formatting ─────────────────────────────────────────────


_SCORE_KEYS = ("_relevance_score", "_distance", "_score")


def _result_to_dict(row: dict, score=...) -> dict:
    if score is ...:
        s = next((row[k] for k in _SCORE_KEYS if k in row), None)
    else:
        s = score
    return {
        "text": row["text"],
        "source_file": row["source_file"],
        "chunk_index": row["chunk_index"],
        "total_chunks": row["total_chunks"],
        "score": s,
    }
