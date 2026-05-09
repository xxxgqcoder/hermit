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


# ── Pagination correctness (regression for 1024-truncation bug) ───


class TestFuzzyPagination:

    def test_fuzzy_finds_match_past_first_page(self, search_env, monkeypatch):
        """Insert > _FUZZY_PAGE_SIZE points; needle lives in the last one.

        Without pagination, scroll returns only the first page and the match
        is silently dropped. With pagination, we walk the cursor until the
        match is found.
        """
        import hermit.retrieval.searcher as sch
        from hermit.retrieval.searcher import search

        # Shrink the page size to keep the test fast. The pagination loop
        # itself is what we exercise; absolute counts don't matter.
        monkeypatch.setattr(sch, "_FUZZY_PAGE_SIZE", 50)

        qmod, col = search_env

        # Add 120 filler points (no needle) then 1 needle point at the end.
        ids, dvecs, svecs, payloads = [], [], [], []
        for i in range(120):
            ids.append(str(uuid.uuid5(uuid.NAMESPACE_DNS, f"filler-{i}")))
            dvecs.append(_rand_dense(seed=10000 + i))
            svecs.append(_fake_sparse([i % 100]))
            payloads.append({
                "text": f"unrelated content block {i}",
                "title": "filler",
                "filename": "filler",
                "source_file": f"/filler/file_{i:04d}.md",
                "chunk_index": 0,
                "total_chunks": 1,
            })
        # Final needle chunk
        ids.append(str(uuid.uuid5(uuid.NAMESPACE_DNS, "needle")))
        dvecs.append(_rand_dense(seed=99999))
        svecs.append(_fake_sparse([42]))
        payloads.append({
            "text": "this chunk contains the rare-needle-token UNICORN",
            "title": "needle",
            "filename": "needle",
            "source_file": "/notes/last.md",
            "chunk_index": 0,
            "total_chunks": 1,
        })
        qmod.upsert_chunks(col, ids, dvecs, svecs, payloads)

        results = search(col, "UNICORN", top_k=5, mode="fuzzy")
        assert len(results) == 1
        assert results[0]["source_file"] == "/notes/last.md"

    def test_fuzzy_respects_max_scan_cap(self, search_env, monkeypatch, caplog):
        """When _FUZZY_MAX_SCAN is hit, we stop and emit a warning."""
        import logging
        import hermit.retrieval.searcher as sch
        from hermit.retrieval.searcher import search

        monkeypatch.setattr(sch, "_FUZZY_PAGE_SIZE", 10)
        monkeypatch.setattr(sch, "_FUZZY_MAX_SCAN", 20)

        qmod, col = search_env
        # Add 30 filler points so the scroll has enough to hit the cap.
        ids, dvecs, svecs, payloads = [], [], [], []
        for i in range(30):
            ids.append(str(uuid.uuid5(uuid.NAMESPACE_DNS, f"capfill-{i}")))
            dvecs.append(_rand_dense(seed=70000 + i))
            svecs.append(_fake_sparse([i]))
            payloads.append({
                "text": f"filler {i}",
                "title": "filler",
                "filename": "filler",
                "source_file": f"/cap/file_{i}.md",
                "chunk_index": 0,
                "total_chunks": 1,
            })
        qmod.upsert_chunks(col, ids, dvecs, svecs, payloads)

        with caplog.at_level(logging.WARNING, logger="hermit.retrieval.searcher"):
            search(col, "no-such-needle-token", top_k=5, mode="fuzzy")

        assert any("scan cap" in rec.message for rec in caplog.records)


# ── Standalone-mode TEXT pre-filter ───────────────────────────────


class TestFuzzyStandaloneFilter:

    def test_local_mode_no_text_match_in_filter(self, search_env):
        """Local mode: filter should NOT include MatchText on text (would mask matches)."""
        from hermit.retrieval.searcher import _fuzzy_scroll_filter
        f = _fuzzy_scroll_filter("anything", base_filter=None)
        assert f is None  # no filename filter, no standalone → no filter

    def test_standalone_mode_adds_text_match(self, search_env, monkeypatch):
        """Standalone mode: MatchText(text=query) is added to the filter."""
        from qdrant_client import models as qmodels
        import hermit.storage.qdrant as qmod
        from hermit.retrieval.searcher import _fuzzy_scroll_filter

        monkeypatch.setattr(qmod, "is_standalone_mode", lambda: True)

        f = _fuzzy_scroll_filter("Qdrant", base_filter=None)
        assert f is not None
        assert len(f.must) == 1
        cond = f.must[0]
        assert isinstance(cond, qmodels.FieldCondition)
        assert cond.key == "text"
        assert cond.match.text == "Qdrant"

    def test_standalone_combines_filename_and_text_filters(self, search_env, monkeypatch):
        """Standalone mode: filename + text filters compose under `must`."""
        from qdrant_client import models as qmodels
        import hermit.storage.qdrant as qmod
        from hermit.retrieval.searcher import _build_filter, _fuzzy_scroll_filter

        monkeypatch.setattr(qmod, "is_standalone_mode", lambda: True)

        base, _ = _build_filter("design")
        f = _fuzzy_scroll_filter("Qdrant", base_filter=base)
        assert f is not None
        keys = sorted(c.key for c in f.must)
        assert keys == ["filename", "text"]
