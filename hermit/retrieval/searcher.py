import fnmatch
import logging
from typing import Literal

from qdrant_client import models

from hermit.config import (
    DEFAULT_RERANK_CANDIDATES,
    DEFAULT_SEARCH_MODE,
    DEFAULT_W_DENSE,
    DEFAULT_W_SPARSE,
)
from hermit.retrieval import embedder, reranker
from hermit.storage import qdrant

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

# Fuzzy mode pagination knobs.
#
# Why we paginate: scroll() returns at most N points per call, so fetching only
# the first page silently truncates results past the page boundary — for a
# 30k-chunk collection a fuzzy hit that happens to live at chunk 5000 would
# never be returned. We page through scroll until either (a) the cursor is
# exhausted, or (b) we have scanned _FUZZY_MAX_SCAN points (safety cap to
# prevent a query from walking an unbounded collection).
#
# NOTE on local mode: in embedded Qdrant, payload indexes are no-ops, so even
# with a Qdrant-side TEXT filter the engine still does a full Python-level
# linear scan internally. That is fine at personal scale but not at hundreds
# of thousands of chunks. A real fix needs either a separate server (Qdrant
# stand-alone, where TEXT indexes work and we already branch on this) or a
# locally-built inverted index. Tracked as future work.
_FUZZY_PAGE_SIZE = 1024
_FUZZY_MAX_SCAN = 100_000


def search(
    collection_name: str,
    query: str,
    top_k: int = 5,
    w_dense: float = DEFAULT_W_DENSE,
    w_sparse: float = DEFAULT_W_SPARSE,
    rerank_candidates: int = DEFAULT_RERANK_CANDIDATES,
    mode: SearchMode = DEFAULT_SEARCH_MODE,
    filename: str | None = None,
    rerank: bool | None = None,
) -> list[dict]:
    """Search a collection. Returns list of result dicts.

    Modes:
      - hybrid:   dense + sparse RRF + rerank (default)
      - semantic: dense only + rerank
      - keyword:  sparse (BM25) only, rerank off by default
      - fuzzy:    no vectors; Qdrant scroll with MatchText on text/filename,
                  rerank off by default
    """
    qdrant_filter, glob_pattern = _build_filter(filename)

    if mode == "fuzzy":
        return _fuzzy_search(
            collection_name,
            query=query,
            base_filter=qdrant_filter,
            glob_pattern=glob_pattern,
            top_k=top_k,
        )

    candidates = _vector_recall(
        collection_name,
        query=query,
        mode=mode,
        rerank_candidates=rerank_candidates,
        top_k=top_k,
        qdrant_filter=qdrant_filter,
    )

    if glob_pattern is not None:
        candidates = [
            r for r in candidates
            if _glob_matches(r.payload.get("source_file", ""), glob_pattern)
        ]

    if not candidates:
        return []

    do_rerank = _RERANK_DEFAULT[mode] if rerank is None else rerank
    if do_rerank:
        passages = [r.payload["text"] for r in candidates]
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
    qdrant_filter: models.Filter | None,
) -> list:
    """Run vector recall for hybrid/semantic/keyword. Returns list of points."""
    candidate_limit = max(top_k, rerank_candidates)

    if mode == "hybrid":
        dense_vec = embedder.embed_query_dense(query)
        sparse_vec = embedder.embed_query_sparse(query)
        results = qdrant.query_points(
            collection_name=collection_name,
            prefetch=[
                models.Prefetch(
                    query=dense_vec,
                    using="dense",
                    limit=candidate_limit,
                    filter=qdrant_filter,
                ),
                models.Prefetch(
                    query=models.SparseVector(
                        indices=sparse_vec.indices.tolist(),
                        values=sparse_vec.values.tolist(),
                    ),
                    using="sparse",
                    limit=candidate_limit,
                    filter=qdrant_filter,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=candidate_limit,
            with_payload=True,
        ).points
    elif mode == "semantic":
        dense_vec = embedder.embed_query_dense(query)
        results = qdrant.query_points(
            collection_name=collection_name,
            query=dense_vec,
            using="dense",
            query_filter=qdrant_filter,
            limit=candidate_limit,
            with_payload=True,
        ).points
    elif mode == "keyword":
        sparse_vec = embedder.embed_query_sparse(query)
        results = qdrant.query_points(
            collection_name=collection_name,
            query=models.SparseVector(
                indices=sparse_vec.indices.tolist(),
                values=sparse_vec.values.tolist(),
            ),
            using="sparse",
            query_filter=qdrant_filter,
            limit=candidate_limit,
            with_payload=True,
        ).points
    else:
        raise ValueError(f"unsupported vector recall mode: {mode!r}")

    return list(results)


def _fuzzy_search(
    collection_name: str,
    query: str,
    base_filter: models.Filter | None,
    glob_pattern: str | None,
    top_k: int,
) -> list[dict]:
    """LIKE-style substring match on chunk text and/or filename.

    Strategy:
      1. Build a server-side filter:
         - filename substring → MatchText on `filename` payload (always)
         - text query → MatchText on `text` payload (stand-alone mode only;
           local mode payload indexes are no-ops and case-sensitive, so we
           keep the text match in Python there)
      2. Page through scroll until exhausted or _FUZZY_MAX_SCAN reached.
      3. Apply Python post-filters: glob (always) + text substring (always —
         even in stand-alone mode, MatchText is token-level so we still
         verify true substring semantics here).
      4. Sort by (source_file, chunk_index), trim to top_k.
    """
    scroll_filter = _fuzzy_scroll_filter(query, base_filter)
    needle = query.lower() if query else None

    matches: list = []
    offset = None
    scanned = 0
    while True:
        points, next_offset = qdrant.scroll_points(
            collection_name=collection_name,
            scroll_filter=scroll_filter,
            limit=_FUZZY_PAGE_SIZE,
            with_payload=True,
            offset=offset,
        )
        scanned += len(points)

        if glob_pattern is not None:
            points = [
                p for p in points
                if _glob_matches(p.payload.get("source_file", ""), glob_pattern)
            ]
        if needle is not None:
            points = [
                p for p in points
                if needle in p.payload.get("text", "").lower()
            ]
        matches.extend(points)

        if next_offset is None or scanned >= _FUZZY_MAX_SCAN:
            break
        offset = next_offset

    if scanned >= _FUZZY_MAX_SCAN:
        logger.warning(
            "fuzzy search hit scan cap (%d) on collection '%s'; results may be incomplete",
            _FUZZY_MAX_SCAN, collection_name,
        )

    matches.sort(key=lambda p: (
        p.payload.get("source_file", ""),
        p.payload.get("chunk_index", 0),
    ))
    return [_result_to_dict(p, score=None) for p in matches[:top_k]]


def _fuzzy_scroll_filter(
    query: str,
    base_filter: models.Filter | None,
) -> models.Filter | None:
    """Compose the server-side filter for fuzzy mode.

    In stand-alone Qdrant we add `MatchText(text=query)` to leverage the TEXT
    payload index for cheap pre-filtering. In local (embedded) mode we skip
    it: payload indexes are no-ops there and MatchText is case-sensitive, so
    adding it would only mask matches without speeding anything up.
    """
    must = list(base_filter.must) if base_filter else []
    if query and qdrant.is_standalone_mode():
        must.append(models.FieldCondition(
            key="text",
            match=models.MatchText(text=query),
        ))
    return models.Filter(must=must) if must else None


# ── Filter helpers ────────────────────────────────────────────────


def _build_filter(filename: str | None) -> tuple[models.Filter | None, str | None]:
    """Return (qdrant_filter, glob_pattern).

    - Plain substring (no glob chars): MatchText on `filename` payload.
    - Glob pattern (contains *?[): no Qdrant pre-filter, callers must apply
      `glob_pattern` as a Python post-filter against `source_file`.
    """
    if not filename:
        return None, None
    if any(c in filename for c in _GLOB_CHARS):
        return None, filename.lower()
    return models.Filter(must=[
        models.FieldCondition(
            key="filename",
            match=models.MatchText(text=filename.lower()),
        )
    ]), None


def _glob_matches(source_file: str, pattern: str) -> bool:
    """Match pattern against either the full path or just the basename, lowercased."""
    sf = source_file.lower()
    from pathlib import PurePath
    base = PurePath(sf).name
    return fnmatch.fnmatch(sf, pattern) or fnmatch.fnmatch(base, pattern)


# ── Result formatting ─────────────────────────────────────────────


def _result_to_dict(result, score=...) -> dict:
    s = result.score if score is ... else score
    return {
        "text": result.payload["text"],
        "source_file": result.payload["source_file"],
        "chunk_index": result.payload["chunk_index"],
        "total_chunks": result.payload["total_chunks"],
        "score": s,
    }
