"""Tests for the four search modes and the orthogonal `filename` filter.

Strategy
--------
Insert real Qdrant points with deterministic text/filename payload, but mock
the embedder + reranker so no model loading is required. Each test runs
against a fresh local-mode collection in an isolated tmp dir.
"""

import contextlib
import uuid
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest


DENSE_DIM = 768
COLLECTION = "test_search_modes"


@dataclass
class FakeSparseVec:
    indices: np.ndarray
    values: np.ndarray


def _rand_dense(seed: int = 0) -> list[float]:
    rng = np.random.default_rng(seed=seed)
    v = rng.random(DENSE_DIM, dtype=np.float32)
    v = v / np.linalg.norm(v)
    return v.tolist()


def _fake_sparse(tokens: list[int]) -> FakeSparseVec:
    return FakeSparseVec(
        indices=np.array(tokens or [1], dtype=np.int32),
        values=np.array([0.5] * (len(tokens) or 1), dtype=np.float32),
    )


# ── Fixture: isolated qdrant + sample data + mocked embedder/reranker ─────


@pytest.fixture()
def search_env(tmp_path, monkeypatch):
    import hermit.config as cfg
    import hermit.storage.qdrant as qmod
    import hermit.retrieval.embedder as emb
    import hermit.retrieval.reranker as rer

    monkeypatch.setattr(cfg, "DATA_ROOT", tmp_path / "data")
    monkeypatch.setattr(qmod, "DATA_ROOT", tmp_path / "data")
    monkeypatch.setattr(cfg, "QDRANT_HOST", None)
    monkeypatch.setattr(qmod, "_client", None)
    monkeypatch.setattr(qmod, "_standalone_mode", False)
    monkeypatch.setattr(qmod, "_app_lock_fd", None)

    # Stable token sets so query/text alignment is predictable for sparse recall.
    def _embed_query_dense(query: str) -> list[float]:
        return _rand_dense(seed=hash(query) & 0xFFFF)

    def _embed_query_sparse(query: str):
        # Map characters to small integer token ids.
        tokens = sorted({ord(c) % 1000 for c in query.lower() if c.isalnum()})
        return _fake_sparse(tokens)

    monkeypatch.setattr(emb, "embed_query_dense", _embed_query_dense)
    monkeypatch.setattr(emb, "embed_query_sparse", _embed_query_sparse)

    # Reranker is identity — preserves recall order, returns top_k indices.
    def _identity_rerank(query: str, passages: list[str], top_k: int) -> list[int]:
        return list(range(min(top_k, len(passages))))

    monkeypatch.setattr(rer, "rerank", _identity_rerank)

    qmod.ensure_collection(COLLECTION)

    # Insert 3 files × N chunks with predictable text and filenames.
    docs = [
        ("/notes/design.md", "design", [
            "Hermit uses hybrid dense and sparse search for retrieval.",
            "The design document describes Qdrant integration.",
        ]),
        ("/notes/usage.md", "usage", [
            "Run hermit kb add to register a knowledge base.",
            "Use hermit search to query the index.",
        ]),
        ("/notes/faq.md", "faq", [
            "How does hermit perform semantic search?",
            "Hermit embeds documents with jina embeddings.",
        ]),
    ]

    ids, dvecs, svecs, payloads = [], [], [], []
    for path, stem, chunks in docs:
        for i, chunk in enumerate(chunks):
            ids.append(str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{path}-{i}")))
            dvecs.append(_rand_dense(seed=hash((path, i)) & 0xFFFF))
            tokens = sorted({ord(c) % 1000 for c in chunk.lower() if c.isalnum()})
            svecs.append(_fake_sparse(tokens))
            payloads.append({
                "text": chunk,
                "title": stem,
                "filename": stem.lower(),
                "source_file": path,
                "chunk_index": i,
                "total_chunks": len(chunks),
            })

    qmod.upsert_chunks(COLLECTION, ids, dvecs, svecs, payloads)

    yield qmod, COLLECTION

    if qmod._client is not None:
        with contextlib.suppress(Exception):
            qmod._client.close()
        qmod._client = None
    qmod._release_app_lock()


# ── Mode behavior ────────────────────────────────────────────────
#
# Note: Qdrant local (embedded) mode does not actually create payload indexes —
# `create_payload_index` is a no-op there. Filters still work, but via a
# Python-side linear scan rather than an index. The TEXT index assertions only
# hold under stand-alone (Docker) mode; see test_qdrant_modes.py for that.


class TestSearchModes:

    def test_hybrid_default_returns_results(self, search_env):
        from hermit.retrieval.searcher import search
        results = search(COLLECTION, "hermit search", top_k=3)
        assert results
        assert len(results) <= 3
        assert all("source_file" in r for r in results)

    def test_semantic_mode_returns_results(self, search_env):
        from hermit.retrieval.searcher import search
        results = search(COLLECTION, "hermit search", top_k=3, mode="semantic")
        assert results

    def test_keyword_mode_skips_rerank_by_default(self, search_env, monkeypatch):
        import hermit.retrieval.reranker as rer
        from hermit.retrieval.searcher import search

        called = {"count": 0}

        def _spy_rerank(*args, **kwargs):
            called["count"] += 1
            return list(range(min(kwargs.get("top_k", 0) or args[2], 5)))

        monkeypatch.setattr(rer, "rerank", _spy_rerank)
        results = search(COLLECTION, "hermit", top_k=3, mode="keyword")
        assert results
        assert called["count"] == 0  # rerank skipped

    def test_keyword_mode_with_explicit_rerank_runs_rerank(self, search_env, monkeypatch):
        import hermit.retrieval.reranker as rer
        from hermit.retrieval.searcher import search

        called = {"count": 0}

        def _spy_rerank(query, passages, top_k):
            called["count"] += 1
            return list(range(min(top_k, len(passages))))

        monkeypatch.setattr(rer, "rerank", _spy_rerank)
        results = search(COLLECTION, "hermit", top_k=3, mode="keyword", rerank=True)
        assert results
        assert called["count"] == 1

    def test_fuzzy_text_substring(self, search_env):
        from hermit.retrieval.searcher import search
        # "qdrant" appears only in design.md chunk index 1
        results = search(COLLECTION, "qdrant", top_k=10, mode="fuzzy")
        assert results
        assert all("qdrant" in r["text"].lower() for r in results)
        assert all(r["source_file"] == "/notes/design.md" for r in results)

    def test_fuzzy_filename_only_lists_matching_files(self, search_env):
        from hermit.retrieval.searcher import search
        # No query, fuzzy by filename only → list all chunks of usage.md
        results = search(COLLECTION, "", top_k=10, mode="fuzzy", filename="usage")
        assert results
        assert all(r["source_file"] == "/notes/usage.md" for r in results)
        # Both usage.md chunks come back, in chunk_index order
        assert [r["chunk_index"] for r in results] == [0, 1]

    def test_fuzzy_requires_query_or_filename(self, search_env):
        from hermit.api.schemas import SearchRequest
        with pytest.raises(ValueError):
            SearchRequest(collection=COLLECTION, mode="fuzzy")


# ── Filename filter (orthogonal) ─────────────────────────────────


class TestFilenameFilter:

    def test_hybrid_with_filename_substring(self, search_env):
        from hermit.retrieval.searcher import search
        results = search(COLLECTION, "hermit search", top_k=10, filename="design")
        assert results
        assert all(r["source_file"] == "/notes/design.md" for r in results)

    def test_hybrid_with_glob_filename(self, search_env):
        from hermit.retrieval.searcher import search
        results = search(COLLECTION, "hermit", top_k=10, filename="*usage*")
        assert results
        assert all(r["source_file"] == "/notes/usage.md" for r in results)

    def test_filename_no_match_returns_empty(self, search_env):
        from hermit.retrieval.searcher import search
        results = search(COLLECTION, "hermit search", top_k=10, filename="nonexistent")
        assert results == []

    def test_filename_glob_dotmd(self, search_env):
        from hermit.retrieval.searcher import search
        # Glob *.md → all three files are eligible (post-filter on source_file)
        results = search(COLLECTION, "hermit", top_k=10, filename="*.md")
        assert results
        assert all(r["source_file"].endswith(".md") for r in results)

    def test_fuzzy_query_plus_filename_filter(self, search_env):
        from hermit.retrieval.searcher import search
        # "hermit" appears in many chunks; constrain to faq.md
        results = search(
            COLLECTION, "hermit", top_k=10, mode="fuzzy", filename="faq",
        )
        assert results
        assert all(r["source_file"] == "/notes/faq.md" for r in results)
        assert all("hermit" in r["text"].lower() for r in results)
