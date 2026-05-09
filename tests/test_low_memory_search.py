from types import SimpleNamespace

import hermit.config as config
from hermit.config import DEFAULT_RERANK_CANDIDATES, ONNX_THREADS
from hermit.retrieval import searcher


class _ArrayLike:
    def __init__(self, values):
        self._values = values

    def tolist(self):
        return self._values


class _SparseVector:
    indices = _ArrayLike([1, 2])
    values = _ArrayLike([0.8, 0.2])


def _point(i: int):
    return SimpleNamespace(
        payload={
            "text": f"text {i}",
            "source_file": f"/tmp/{i}.txt",
            "chunk_index": i,
            "total_chunks": 3,
        },
        score=1.0 / (i + 1),
    )


def test_default_rerank_candidates_reduced_for_memory():
    assert DEFAULT_RERANK_CANDIDATES == 20


def test_search_request_concurrency_is_not_configurable():
    assert not hasattr(config, "SEARCH_THREADS")
    assert ONNX_THREADS == 2


def test_search_keeps_reranker_and_expands_candidate_limit_to_top_k(monkeypatch):
    monkeypatch.setattr(searcher.embedder, "embed_query_dense", lambda query: [0.1, 0.2])
    monkeypatch.setattr(searcher.embedder, "embed_query_sparse", lambda query: _SparseVector())
    monkeypatch.setattr(searcher.reranker, "rerank", lambda query, passages, top_k: [1, 0][:top_k])

    captured = {}

    def fake_query_points(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(points=[_point(0), _point(1), _point(2)])

    monkeypatch.setattr(searcher.qdrant, "query_points", fake_query_points)

    results = searcher.search(
        collection_name="docs",
        query="memory",
        top_k=2,
        rerank_candidates=1,
    )

    assert [r["text"] for r in results] == ["text 1", "text 0"]
    assert captured["limit"] == 2
    assert [prefetch.limit for prefetch in captured["prefetch"]] == [2, 2]
