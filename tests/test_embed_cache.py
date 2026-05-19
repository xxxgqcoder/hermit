"""Unit tests for the on-disk dense embedding cache.

Fully offline — no model loads, no storage. Each test gets its own
``tmp_path`` cache root via monkeypatch so they don't share state.
"""

from __future__ import annotations

import time

import numpy as np
import pytest


# ── Fixtures ─────────────────────────────────────────────────────


@pytest.fixture()
def cache_module(tmp_path, monkeypatch):
    """Yield ``hermit.retrieval.embed_cache`` with state isolated to *tmp_path*.

    Resets the module-level Cache singletons before AND after the test,
    and points ``CACHE_ROOT`` at *tmp_path* so files don't escape.
    """
    import hermit.config as cfg
    import hermit.retrieval.embed_cache as ec

    # Close any pre-existing handles from a prior test
    ec.close()

    monkeypatch.setattr(cfg, "CACHE_ROOT", tmp_path / "cache")
    monkeypatch.setattr(ec, "CACHE_ROOT", tmp_path / "cache")

    # Use a realistic model name so the model-namespace path is exercised
    monkeypatch.setattr(ec, "_dense_model_name", "test-dense-model")

    yield ec

    ec.close()


def _vec(seed: int, dim: int) -> list[float]:
    rng = np.random.default_rng(seed=seed)
    return rng.random(dim, dtype=np.float32).astype(float).tolist()


# ── Dense ────────────────────────────────────────────────────────


def test_dense_hit_returns_cached(cache_module):
    ec = cache_module
    from hermit.config import DENSE_DIM

    text = "hello world"
    v = _vec(1, DENSE_DIM)
    ec.store_dense(text, v)

    results, miss = ec.lookup_dense([text])
    assert miss == []
    assert results[0] == v


def test_dense_miss_returns_none_and_indices(cache_module):
    ec = cache_module
    results, miss = ec.lookup_dense(["a", "b", "c"])
    assert results == [None, None, None]
    assert miss == [0, 1, 2]


def test_dense_partial_hit(cache_module):
    ec = cache_module
    from hermit.config import DENSE_DIM

    v = _vec(2, DENSE_DIM)
    ec.store_dense("middle", v)

    results, miss = ec.lookup_dense(["alpha", "middle", "omega"])
    assert results[0] is None
    assert results[1] == v
    assert results[2] is None
    assert miss == [0, 2]


def test_dense_dimension_validation_treats_wrong_dim_as_miss(cache_module):
    """A cached vector with the wrong dim (e.g. after model swap) must miss."""
    ec = cache_module
    from hermit.config import DENSE_DIM

    # Bypass store_dense's own validator — we want a *poisoned* entry on disk.
    cache = ec._dense_cache_ref()
    bad = [0.5] * (DENSE_DIM // 2)  # half the expected dim
    cache.set(ec._key(ec._dense_model_name, "poison"), bad)

    results, miss = ec.lookup_dense(["poison"])
    assert results == [None]
    assert miss == [0]


def test_dense_store_rejects_invalid_input(cache_module):
    """store_dense() drops invalid vectors silently rather than poisoning the cache."""
    ec = cache_module
    from hermit.config import DENSE_DIM

    ec.store_dense("short", [0.1, 0.2])  # wrong dim — should be a no-op
    results, _ = ec.lookup_dense(["short"])
    assert results == [None]

    # Sanity: storing a correct vector still works
    v = _vec(3, DENSE_DIM)
    ec.store_dense("good", v)
    results, _ = ec.lookup_dense(["good"])
    assert results == [v]


# ── Behaviour switches ───────────────────────────────────────────


def test_ttl_expires(cache_module, monkeypatch):
    """Override the TTL to ~10 ms and verify diskcache actually drops the entry."""
    ec = cache_module
    from hermit.config import DENSE_DIM

    monkeypatch.setattr(ec, "EMBED_CACHE_TTL_SECONDS", 0.05)

    v = _vec(6, DENSE_DIM)
    ec.store_dense("expiring", v)

    # Immediately: hit
    results, _ = ec.lookup_dense(["expiring"])
    assert results == [v]

    # Wait for expiry, then expect miss
    time.sleep(0.2)
    results, miss = ec.lookup_dense(["expiring"])
    assert results == [None]
    assert miss == [0]


def test_model_namespace_isolates_entries(cache_module, monkeypatch):
    """Same text under a different model name must not collide."""
    ec = cache_module
    from hermit.config import DENSE_DIM

    v_a = _vec(7, DENSE_DIM)
    ec.store_dense("shared-text", v_a)

    monkeypatch.setattr(ec, "_dense_model_name", "different-model")
    results, miss = ec.lookup_dense(["shared-text"])
    assert results == [None]
    assert miss == [0]

    # Storing under the new model leaves the old entry intact
    v_b = _vec(8, DENSE_DIM)
    ec.store_dense("shared-text", v_b)

    monkeypatch.setattr(ec, "_dense_model_name", "test-dense-model")
    results, _ = ec.lookup_dense(["shared-text"])
    assert results == [v_a], "Original-model entry must still be intact"


def test_default_ttl_is_thirty_days():
    """Sanity: the hard-coded default is what the design says (30 days)."""
    from hermit.config import EMBED_CACHE_TTL_SECONDS

    assert EMBED_CACHE_TTL_SECONDS == 30 * 86400
