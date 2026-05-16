"""On-disk dense embedding cache keyed by sha256 of the model input.

Indexing is dominated by ONNX inference. Many real-world re-index events
(file renamed, metadata-only edit on a peer file, second collection over
the same docs) re-embed chunks whose *exact* model input is unchanged.
This module sits *above* the ``_EmbedScheduler`` in ``embedder.py`` so
that pure cache hits skip the scheduler queue and ONNX path entirely;
misses still flow through the scheduler and benefit from its batching.

Cache key
─────────
``sha256(f"{MODEL_NAME}::{text}".encode("utf-8")).hexdigest()`` where
``text`` is exactly the string passed to the embedding model — i.e. the
title-prefixed chunk used in ``scanner._index_file`` (``[{title}]\\n{chunk}``).
The model name is part of the key so swapping models naturally orphans
old entries; the 7-day TTL then reaps them.

Storage
───────
A single ``diskcache.Cache`` instance under ``HERMIT_HOME/cache/dense``.
diskcache is SQLite-backed, supports per-item ``expire``, and is safe for
concurrent access from the multiple indexing workers.

Validation on hit
─────────────────
Cache hits are sanity-checked before being returned (dim == ``DENSE_DIM``).
Anything that fails validation is treated as a miss so the caller falls
back to the real model — defensive against schema drift and partially-
corrupted entries from earlier crashes.
"""

from __future__ import annotations

import hashlib
import logging
import threading
from typing import Optional

from hermit.config import (
    CACHE_ROOT,
    DENSE_DIM,
    DENSE_MODEL,
    EMBED_CACHE_TTL_SECONDS,
)

logger = logging.getLogger(__name__)


# ── Module state ────────────────────────────────────────────────

_lock = threading.Lock()
_dense_cache = None  # type: ignore[assignment]


def _open_cache(subdir: str):
    """Lazily open a diskcache.Cache under CACHE_ROOT/<subdir>.

    Lazy because importing this module shouldn't create directories on
    disk (matters for tests and for the disabled-cache code path).
    """
    import diskcache  # local import: keeps the dep optional at import time

    path = CACHE_ROOT / subdir
    path.mkdir(parents=True, exist_ok=True)
    return diskcache.Cache(str(path))


def _dense_cache_ref():
    global _dense_cache
    if _dense_cache is None:
        with _lock:
            if _dense_cache is None:
                _dense_cache = _open_cache("dense")
    return _dense_cache


# Model name lives in a module-level global so tests can monkeypatch it
# to verify the model-namespace invariant without rebuilding the entire
# config module.
_dense_model_name: str = DENSE_MODEL


def _key(model: str, text: str) -> str:
    return hashlib.sha256(f"{model}::{text}".encode("utf-8")).hexdigest()


# ── Dense ────────────────────────────────────────────────────────


def _valid_dense(v: object) -> bool:
    if not isinstance(v, list) or len(v) != DENSE_DIM:
        return False
    # Spot-check a single element rather than scanning all 768 floats.
    return isinstance(v[0], float)


def lookup_dense(texts: list[str]) -> tuple[list[Optional[list[float]]], list[int]]:
    """Look up dense vectors for *texts*.

    Returns ``(results, miss_indices)`` where ``results[i]`` is the cached
    vector or ``None`` for a miss / failed validation, and ``miss_indices``
    lists positions the caller still has to compute.
    """
    results: list[Optional[list[float]]] = [None] * len(texts)
    cache = _dense_cache_ref()
    miss: list[int] = []
    for i, text in enumerate(texts):
        v = cache.get(_key(_dense_model_name, text))
        if _valid_dense(v):
            results[i] = v  # type: ignore[assignment]
        else:
            miss.append(i)
    return results, miss


def store_dense(text: str, vector: list[float]) -> None:
    if not _valid_dense(vector):
        return
    cache = _dense_cache_ref()
    cache.set(_key(_dense_model_name, text), vector, expire=EMBED_CACHE_TTL_SECONDS)


# ── Lifecycle ────────────────────────────────────────────────────


def close() -> None:
    """Close cache handles. Intended for tests; production processes can rely on GC."""
    global _dense_cache
    with _lock:
        if _dense_cache is not None:
            try:
                _dense_cache.close()
            except Exception:  # pragma: no cover — best-effort
                pass
        _dense_cache = None
