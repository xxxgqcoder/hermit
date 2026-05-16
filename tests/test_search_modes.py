"""Tests for the four search modes and the orthogonal ``filename`` filter.

Each test runs against a fresh LanceDB collection in an isolated tmp dir.
Embeddings + reranker are mocked so no models load — keyword recall is
exercised against LanceDB's real tantivy-based FTS index.
"""

import contextlib
import uuid
from pathlib import Path

import numpy as np
import pytest


DENSE_DIM = 768
COLLECTION = "test_search_modes"


def _rand_dense(seed: int = 0) -> list[float]:
    rng = np.random.default_rng(seed=seed)
    v = rng.random(DENSE_DIM, dtype=np.float32)
    v = v / np.linalg.norm(v)
    return v.tolist()


# ── Fixture: isolated LanceDB + sample data + mocked embedder/reranker ─────


@pytest.fixture()
def search_env(tmp_path, monkeypatch):
    import hermit.config as cfg
    import hermit.storage.lance as lmod
    import hermit.retrieval.embedder as emb
    import hermit.retrieval.reranker as rer

    monkeypatch.setattr(cfg, "DATA_ROOT", tmp_path / "data")
    monkeypatch.setattr(lmod, "DATA_ROOT", tmp_path / "data")
    monkeypatch.setattr(lmod, "_db", None)

    def _embed_query_dense(query: str) -> list[float]:
        return _rand_dense(seed=hash(query) & 0xFFFF)

    monkeypatch.setattr(emb, "embed_query_dense", _embed_query_dense)

    # Reranker is identity — preserves recall order, returns top_k indices.
    def _identity_rerank(query: str, passages: list[str], top_k: int) -> list[int]:
        return list(range(min(top_k, len(passages))))

    monkeypatch.setattr(rer, "rerank", _identity_rerank)

    lmod.ensure_collection(COLLECTION)

    docs = [
        ("/notes/design.md", "design", [
            "Hermit uses hybrid dense and fts search for retrieval.",
            "The design document describes lancedb integration.",
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

    for path, stem, chunks in docs:
        ids, vectors, payloads = [], [], []
        for i, chunk in enumerate(chunks):
            ids.append(str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{path}-{i}")))
            vectors.append(_rand_dense(seed=hash((path, i)) & 0xFFFF))
            payloads.append({
                "text": chunk,
                "title": stem,
                "filename": stem.lower(),
                "source_file": path,
                "chunk_index": i,
                "total_chunks": len(chunks),
            })
        lmod.replace_file_chunks(COLLECTION, path, ids, vectors, payloads)

    yield lmod, COLLECTION

    with contextlib.suppress(Exception):
        lmod.delete_collection(COLLECTION)
    lmod._db = None


# ── Mode behavior ────────────────────────────────────────────────


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
        # "lancedb" appears only in design.md chunk index 1
        results = search(COLLECTION, "lancedb", top_k=10, mode="fuzzy")
        assert results
        assert all("lancedb" in r["text"].lower() for r in results)
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


# ── Filter-builder unit (no fixture needed) ───────────────────────


class TestFilterBuilder:

    def test_substring_builds_contains_clause(self):
        from hermit.retrieval.searcher import _build_filename_filter
        where, glob = _build_filename_filter("design")
        assert glob is None
        assert where is not None
        assert "contains(lower(filename), 'design')" in where

    def test_glob_returns_no_sql_filter(self):
        from hermit.retrieval.searcher import _build_filename_filter
        where, glob = _build_filename_filter("*.md")
        assert where is None
        assert glob == "*.md"

    def test_sql_escaping_for_single_quote(self):
        from hermit.retrieval.searcher import _build_filename_filter
        where, _ = _build_filename_filter("don't")
        # Single quote in the value must be doubled to remain valid SQL.
        assert where is not None and "don''t" in where

    def test_empty_filename_returns_none(self):
        from hermit.retrieval.searcher import _build_filename_filter
        assert _build_filename_filter(None) == (None, None)
        assert _build_filename_filter("") == (None, None)
