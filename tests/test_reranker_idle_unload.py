"""Tests for reranker idle-unload behaviour.

Two layers:

* ``test_idle_unloader_unloads_after_timeout`` — fast, fully mocked: drives
  the loop directly to prove the lock/state machine works without spinning up
  the real ONNX session.
* ``test_real_reranker_cold_reload_latency`` — slow integration test that
  loads the real ``jinaai/jina-reranker-v2-base-multilingual`` model, forces
  an unload, then measures lazy reload + rerank latency. Skipped automatically
  when the model is not cached locally.
"""

from __future__ import annotations

import importlib
import logging
import time
from pathlib import Path

import pytest

import hermit.config as config
import hermit.retrieval.reranker as reranker_mod


# Path to the cached HF snapshot for the real model. Used to skip the slow
# integration test on machines without the weights downloaded.
_MODEL_CACHE_DIR = (
    Path.home() / ".hermit" / "models"
    / "models--jinaai--jina-reranker-v2-base-multilingual"
)


@pytest.fixture(autouse=True)
def _reset_reranker_state():
    """Each test starts with a clean module-level state."""
    reranker_mod.unload_now()
    reranker_mod._unloader_started = False
    reranker_mod._unloader_thread = None
    yield
    reranker_mod.unload_now()
    reranker_mod._unloader_started = False
    reranker_mod._unloader_thread = None


class _FakeEncoder:
    """Stand-in for TextCrossEncoder. Records destruction for the test."""

    instances_alive = 0

    def __init__(self):
        type(self).instances_alive += 1

    def __del__(self):
        type(self).instances_alive -= 1

    def rerank(self, query, passages):
        return [float(len(p)) for p in passages]


def test_idle_unloader_unloads_after_timeout(monkeypatch, caplog):
    """Drive the unloader loop body directly; verify state transitions."""
    _FakeEncoder.instances_alive = 0

    monkeypatch.setattr(reranker_mod, "_build_reranker", lambda: _FakeEncoder())
    # Tight timing: 0.1s timeout, 0.05s check interval.
    monkeypatch.setattr(reranker_mod, "RERANKER_IDLE_TIMEOUT", 0.1)
    monkeypatch.setattr(reranker_mod, "RERANKER_IDLE_CHECK_INTERVAL", 0.05)

    # Touch the reranker → it becomes resident.
    out = reranker_mod.rerank("q", ["abc", "abcd"], top_k=2)
    assert out == [1, 0]
    assert reranker_mod.is_loaded()
    assert _FakeEncoder.instances_alive == 1

    with caplog.at_level(logging.INFO, logger=reranker_mod.logger.name):
        reranker_mod.start_idle_unloader()

        # Wait up to 2s for idle unload to fire. Should fire within ~0.15s.
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline and reranker_mod.is_loaded():
            time.sleep(0.02)

    assert not reranker_mod.is_loaded(), "reranker should have been unloaded"
    assert _FakeEncoder.instances_alive == 0, "old encoder should be GC'd"
    assert any("unloading" in rec.message for rec in caplog.records)

    # Lazy reload on next call.
    reranker_mod.rerank("q", ["x"], top_k=1)
    assert reranker_mod.is_loaded()


def test_idle_unloader_skips_when_recently_used(monkeypatch):
    """If activity keeps touching the model, it stays resident."""
    monkeypatch.setattr(reranker_mod, "_build_reranker", lambda: _FakeEncoder())
    monkeypatch.setattr(reranker_mod, "RERANKER_IDLE_TIMEOUT", 0.5)
    monkeypatch.setattr(reranker_mod, "RERANKER_IDLE_CHECK_INTERVAL", 0.05)

    reranker_mod.rerank("q", ["a"], top_k=1)
    reranker_mod.start_idle_unloader()

    # Hammer rerank for 0.6s — longer than the timeout, but each call resets
    # _last_use, so the unloader should never fire.
    end = time.monotonic() + 0.6
    while time.monotonic() < end:
        reranker_mod.rerank("q", ["a"], top_k=1)
        time.sleep(0.05)

    assert reranker_mod.is_loaded(), "active reranker should not be unloaded"


def test_idle_timeout_zero_disables_unloader(monkeypatch, caplog):
    monkeypatch.setattr(reranker_mod, "_build_reranker", lambda: _FakeEncoder())
    monkeypatch.setattr(reranker_mod, "RERANKER_IDLE_TIMEOUT", 0)

    with caplog.at_level(logging.INFO, logger=reranker_mod.logger.name):
        reranker_mod.start_idle_unloader()

    assert reranker_mod._unloader_thread is None
    assert any("disabled" in rec.message for rec in caplog.records)


def test_config_env_var_override(monkeypatch):
    """HERMIT_RERANKER_IDLE_TIMEOUT must reach the module after reimport."""
    monkeypatch.setenv("HERMIT_RERANKER_IDLE_TIMEOUT", "42")
    importlib.reload(config)
    try:
        assert config.RERANKER_IDLE_TIMEOUT == 42
    finally:
        monkeypatch.delenv("HERMIT_RERANKER_IDLE_TIMEOUT", raising=False)
        importlib.reload(config)
        importlib.reload(reranker_mod)


# ── Integration test: real model cold reload ────────────────────


@pytest.mark.skipif(
    not _MODEL_CACHE_DIR.exists(),
    reason=f"reranker model not cached at {_MODEL_CACHE_DIR}",
)
def test_real_reranker_cold_reload_latency(caplog):
    """End-to-end: load real reranker, unload, measure lazy reload latency.

    Prints two timings:
      * initial cold load + first rerank
      * post-unload cold reload + first rerank

    Design doc target: reload within 1-3s on a warm OS file cache.
    """
    query = "what is hermit"
    passages = [
        "Hermit is a local semantic search service.",
        "Bananas are yellow.",
        "Use the hermit CLI to add knowledge bases.",
    ]

    # Initial cold load.
    t0 = time.monotonic()
    reranker_mod.rerank(query, passages, top_k=2)
    cold_load_s = time.monotonic() - t0
    assert reranker_mod.is_loaded()
    print(f"\n[idle-unload] initial cold load + rerank: {cold_load_s:.2f}s")

    # Warm rerank for reference.
    t0 = time.monotonic()
    reranker_mod.rerank(query, passages, top_k=2)
    warm_s = time.monotonic() - t0
    print(f"[idle-unload] warm rerank:                  {warm_s:.3f}s")

    # Force unload — the path the idle-unloader would take.
    assert reranker_mod.unload_now()
    assert not reranker_mod.is_loaded()

    # Cold reload + rerank.
    t0 = time.monotonic()
    reranker_mod.rerank(query, passages, top_k=2)
    reload_s = time.monotonic() - t0
    assert reranker_mod.is_loaded()
    print(f"[idle-unload] post-unload cold reload:      {reload_s:.2f}s")

    # Loose sanity bound — designed to catch regressions, not pin timings.
    assert reload_s < 30.0, f"cold reload took {reload_s:.1f}s, expected < 30s"
