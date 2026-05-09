"""On-disk embedding cache keyed by sha256 of the model input.

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
Two ``diskcache.Cache`` instances under ``HERMIT_HOME/cache/{dense,sparse}``.
diskcache is SQLite-backed, supports per-item ``expire``, and is safe for
concurrent access from the multiple indexing workers.

Validation on hit
─────────────────
Cache hits are sanity-checked before being returned (dense: dim ==
``DENSE_DIM``; sparse: ``len(indices) == len(values) > 0``). Anything that
fails validation is treated as a miss so the caller falls back to the
real model — defensive against schema drift and partially-corrupted
entries from earlier crashes.
"""

from __future__ import annotations

import hashlib
import logging
import threading
from typing import Optional

import numpy as np

from hermit.config import (
    CACHE_ROOT,
    DENSE_DIM,
    DENSE_MODEL,
    EMBED_CACHE_TTL_SECONDS,
    SPARSE_MODEL,
)

logger = logging.getLogger(__name__)


# ── Module state ────────────────────────────────────────────────

_lock = threading.Lock()
_dense_cache = None  # type: ignore[assignment]
_sparse_cache = None  # type: ignore[assignment]


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


def _sparse_cache_ref():
    global _sparse_cache
    if _sparse_cache is None:
        with _lock:
            if _sparse_cache is None:
                _sparse_cache = _open_cache("sparse")
    return _sparse_cache


# Model names live in module-level globals so tests can monkeypatch them
# to verify the model-namespace invariant without rebuilding the entire
# config module.
_dense_model_name: str = DENSE_MODEL
_sparse_model_name: str = SPARSE_MODEL


def _key(model: str, text: str) -> str:
    return hashlib.sha256(f"{model}::{text}".encode("utf-8")).hexdigest()


# ── Sparse roundtrip type ───────────────────────────────────────


class _CachedSparseVec:
    """Mimics fastembed's sparse output shape (.indices / .values ndarrays).

    qdrant.py's ``_build_points`` only calls ``.indices.tolist()`` and
    ``.values.tolist()`` on these objects, so duck-typed numpy arrays
    are sufficient — see hermit/storage/qdrant.py:_build_points.
    """

    __slots__ = ("indices", "values")

    def __init__(self, indices: list[int], values: list[float]):
        self.indices = np.asarray(indices, dtype=np.int32)
        self.values = np.asarray(values, dtype=np.float32)


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


# ── Sparse ───────────────────────────────────────────────────────


def _valid_sparse(payload: object) -> bool:
    if not isinstance(payload, tuple) or len(payload) != 2:
        return False
    indices, values = payload
    return (
        isinstance(indices, list)
        and isinstance(values, list)
        and len(indices) == len(values)
        and len(indices) > 0
    )


def lookup_sparse(
    texts: list[str],
) -> tuple[list[Optional[_CachedSparseVec]], list[int]]:
    results: list[Optional[_CachedSparseVec]] = [None] * len(texts)
    cache = _sparse_cache_ref()
    miss: list[int] = []
    for i, text in enumerate(texts):
        payload = cache.get(_key(_sparse_model_name, text))
        if _valid_sparse(payload):
            indices, values = payload  # type: ignore[misc]
            results[i] = _CachedSparseVec(indices, values)
        else:
            miss.append(i)
    return results, miss


def store_sparse(text: str, indices: list[int], values: list[float]) -> None:
    if not (isinstance(indices, list) and isinstance(values, list)):
        return
    if len(indices) != len(values) or len(indices) == 0:
        return
    cache = _sparse_cache_ref()
    cache.set(
        _key(_sparse_model_name, text),
        (indices, values),
        expire=EMBED_CACHE_TTL_SECONDS,
    )


# ── Lifecycle ────────────────────────────────────────────────────


def close() -> None:
    """Close cache handles. Intended for tests; production processes can rely on GC."""
    global _dense_cache, _sparse_cache
    with _lock:
        for c in (_dense_cache, _sparse_cache):
            if c is not None:
                try:
                    c.close()
                except Exception:  # pragma: no cover — best-effort
                    pass
        _dense_cache = None
        _sparse_cache = None
