"""Verify the search candidate-limit invariant survives the LanceDB swap.

The behaviour we lock down: even with ``rerank_candidates`` set lower than
``top_k``, the underlying recall must still pull ``max(top_k, rerank_candidates)``
rows so the reranker has enough material to fill the user-requested ``top_k``.
"""

import hermit.config as config
from hermit.config import DEFAULT_RERANK_CANDIDATES, ONNX_THREADS
from hermit.retrieval import searcher


def _row(i: int) -> dict:
    return {
        "text": f"text {i}",
        "source_file": f"/tmp/{i}.txt",
        "chunk_index": i,
        "total_chunks": 3,
        "_relevance_score": 1.0 / (i + 1),
    }


class _ChainStub:
    """Mimic LanceDB's chained query-builder API."""

    def __init__(self, captured: dict, rows: list[dict]):
        self._captured = captured
        self._rows = rows

    def vector(self, vec):
        self._captured["vector"] = vec
        return self

    def text(self, query):
        self._captured["text"] = query
        return self

    def rerank(self, _reranker):
        self._captured["reranked"] = True
        return self

    def where(self, clause, prefilter=False):
        self._captured["where"] = clause
        self._captured["prefilter"] = prefilter
        return self

    def limit(self, n):
        self._captured["limit"] = n
        return self

    def to_list(self):
        return list(self._rows[: self._captured["limit"]])


class _TableStub:
    def __init__(self, captured: dict, rows: list[dict]):
        self._captured = captured
        self._rows = rows

    def search(self, query=None, query_type=None, fts_columns=None):
        self._captured["query"] = query
        self._captured["query_type"] = query_type
        self._captured["fts_columns"] = fts_columns
        return _ChainStub(self._captured, self._rows)


def test_default_rerank_candidates_reduced_for_memory():
    assert DEFAULT_RERANK_CANDIDATES == 20


def test_search_request_concurrency_is_not_configurable():
    assert not hasattr(config, "SEARCH_THREADS")
    assert ONNX_THREADS == 2


def test_search_keeps_reranker_and_expands_candidate_limit_to_top_k(monkeypatch):
    monkeypatch.setattr(searcher.embedder, "embed_query_dense", lambda query: [0.1, 0.2])
    monkeypatch.setattr(searcher.reranker, "rerank", lambda query, passages, top_k: [1, 0][:top_k])

    captured: dict = {}
    rows = [_row(0), _row(1), _row(2)]
    monkeypatch.setattr(searcher.lance, "open_table", lambda name: _TableStub(captured, rows))

    results = searcher.search(
        collection_name="docs",
        query="memory",
        top_k=2,
        rerank_candidates=1,
    )

    assert [r["text"] for r in results] == ["text 1", "text 0"]
    assert captured["limit"] == 2, "candidate_limit should be max(top_k=2, rerank_candidates=1)"
    assert captured["query_type"] == "hybrid"
    assert captured["reranked"] is True
