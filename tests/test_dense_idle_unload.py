"""Tests for dense embedder idle-unload behaviour.

Mirrors ``test_reranker_idle_unload.py`` shape — fast unit tests that drive
the state machine via a fake ``TextEmbedding`` stand-in, plus a slow
integration test exercising the real model when its weights are cached.
"""

from __future__ import annotations

import importlib
import logging
import time
from pathlib import Path

import pytest

import hermit.config as config
import hermit.retrieval.embedder as embedder_mod


_MODEL_CACHE_DIR = (
    Path.home() / ".hermit" / "models"
    / "models--jinaai--jina-embeddings-v2-base-zh"
)


@pytest.fixture(autouse=True)
def _reset_embedder_state():
    """Each test starts with a clean module-level state."""
    embedder_mod.unload_now()
    embedder_mod._unloader_started = False
    embedder_mod._unloader_thread = None
    yield
    embedder_mod.unload_now()
    embedder_mod._unloader_started = False
    embedder_mod._unloader_thread = None


class _FakeEmbedding:
    """Stand-in for fastembed.TextEmbedding."""

    instances_alive = 0

    def __init__(self):
        type(self).instances_alive += 1

    def __del__(self):
        type(self).instances_alive -= 1

    def embed(self, texts, batch_size=None):
        import numpy as np
        # Deterministic non-zero vectors so downstream lookups have something
        # to validate. Dimension doesn't matter for the unload tests — they
        # never touch the cache validator.
        return iter([np.array([float(len(t)), 0.0, 1.0]) for t in texts])

    def query_embed(self, query):
        import numpy as np
        return iter([np.array([float(len(query)), 0.0, 1.0])])


def test_idle_unloader_unloads_after_timeout(monkeypatch, caplog):
    _FakeEmbedding.instances_alive = 0
    monkeypatch.setattr(embedder_mod, "_build_dense_model", lambda: _FakeEmbedding())
    monkeypatch.setattr(embedder_mod, "DENSE_IDLE_TIMEOUT", 0.1)
    monkeypatch.setattr(embedder_mod, "DENSE_IDLE_CHECK_INTERVAL", 0.05)

    # Touch dense via query path → model loads.
    embedder_mod.embed_query_dense("hello")
    assert embedder_mod.is_loaded()
    assert _FakeEmbedding.instances_alive == 1

    with caplog.at_level(logging.INFO, logger=embedder_mod.logger.name):
        embedder_mod.start_idle_unloader()
        # Wait up to 2s for unload to fire.
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline and embedder_mod.is_loaded():
            time.sleep(0.02)

    assert not embedder_mod.is_loaded()
    assert _FakeEmbedding.instances_alive == 0
    assert any("unloading" in rec.message for rec in caplog.records)

    # Lazy reload on next call.
    embedder_mod.embed_query_dense("world")
    assert embedder_mod.is_loaded()


def test_idle_unloader_skips_when_recently_used(monkeypatch):
    monkeypatch.setattr(embedder_mod, "_build_dense_model", lambda: _FakeEmbedding())
    monkeypatch.setattr(embedder_mod, "DENSE_IDLE_TIMEOUT", 0.5)
    monkeypatch.setattr(embedder_mod, "DENSE_IDLE_CHECK_INTERVAL", 0.05)

    embedder_mod.embed_query_dense("kickoff")
    embedder_mod.start_idle_unloader()

    # Hammer query embed for 0.6s — longer than the timeout, but each call
    # bumps _last_use so the unloader should never fire.
    end = time.monotonic() + 0.6
    while time.monotonic() < end:
        embedder_mod.embed_query_dense("ping")
        time.sleep(0.05)

    assert embedder_mod.is_loaded()


def test_idle_timeout_zero_disables_unloader(monkeypatch, caplog):
    monkeypatch.setattr(embedder_mod, "_build_dense_model", lambda: _FakeEmbedding())
    monkeypatch.setattr(embedder_mod, "DENSE_IDLE_TIMEOUT", 0)

    with caplog.at_level(logging.INFO, logger=embedder_mod.logger.name):
        embedder_mod.start_idle_unloader()

    assert embedder_mod._unloader_thread is None
    assert any("disabled" in rec.message for rec in caplog.records)


def test_config_env_var_override(monkeypatch):
    monkeypatch.setenv("HERMIT_DENSE_IDLE_TIMEOUT", "1234")
    importlib.reload(config)
    try:
        assert config.DENSE_IDLE_TIMEOUT == 1234
    finally:
        monkeypatch.delenv("HERMIT_DENSE_IDLE_TIMEOUT", raising=False)
        importlib.reload(config)
        importlib.reload(embedder_mod)


# ── Integration test: real model cold reload ────────────────────


@pytest.mark.skipif(
    not _MODEL_CACHE_DIR.exists(),
    reason=f"dense model not cached at {_MODEL_CACHE_DIR}",
)
def test_real_dense_cold_reload_latency():
    """End-to-end: load real model, unload, measure lazy reload latency."""
    t0 = time.monotonic()
    embedder_mod.embed_query_dense("initial warmup query")
    cold_s = time.monotonic() - t0
    assert embedder_mod.is_loaded()
    print(f"\n[dense-idle-unload] initial cold load + query: {cold_s:.2f}s")

    t0 = time.monotonic()
    embedder_mod.embed_query_dense("warm query")
    warm_s = time.monotonic() - t0
    print(f"[dense-idle-unload] warm query:                  {warm_s:.3f}s")

    assert embedder_mod.unload_now()
    assert not embedder_mod.is_loaded()

    t0 = time.monotonic()
    embedder_mod.embed_query_dense("after unload")
    reload_s = time.monotonic() - t0
    assert embedder_mod.is_loaded()
    print(f"[dense-idle-unload] post-unload cold reload:     {reload_s:.2f}s")

    assert reload_s < 10.0, f"cold reload took {reload_s:.1f}s, expected <10s"
