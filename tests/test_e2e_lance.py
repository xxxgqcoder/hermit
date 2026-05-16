"""End-to-end smoke test against the LanceDB stack.

Spins up a real Hermit server with an isolated ``HERMIT_HOME``, registers a
small markdown collection, polls for indexing to finish, then runs each of
the four search modes and verifies the lifecycle (add → search → rm).

Skipped automatically when the embedding / reranker models are not cached
locally — these tests require the production model weights and would
otherwise spend several minutes downloading.
"""

from __future__ import annotations

import json
import time
import urllib.request
from pathlib import Path

import pytest


_MODELS_CACHE = Path.home() / ".hermit" / "models"
_REQUIRED_MODELS = [
    "models--jinaai--jina-embeddings-v2-base-zh",
    "models--jinaai--jina-reranker-v2-base-multilingual",
]

if not all((_MODELS_CACHE / m).exists() for m in _REQUIRED_MODELS):
    pytest.skip(
        f"required models not cached under {_MODELS_CACHE}",
        allow_module_level=True,
    )


def _http_get(port: int, path: str) -> dict:
    with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=10) as r:
        return json.loads(r.read())


def _http_post(port: int, path: str, body: dict) -> dict:
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=data,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


def _http_delete(port: int, path: str) -> dict:
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}", method="DELETE",
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


def _wait_for_indexing(port: int, name: str, timeout: float = 60.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status = _http_get(port, f"/collections/{name}/tasks")
        if status["pending_tasks"] == 0:
            return
        time.sleep(0.5)
    raise AssertionError(f"Indexing for '{name}' didn't drain within {timeout}s")


def test_lance_kb_lifecycle(hermit_server, test_docs_dir):
    port, _ = hermit_server

    health = _http_get(port, "/health")
    assert health["status"] == "ready"
    assert health["storage"] == "lance"

    name = "lance_e2e"
    _http_post(port, "/collections", {
        "name": name,
        "folder_path": str(test_docs_dir),
    })

    _wait_for_indexing(port, name)

    status = _http_get(port, f"/collections/{name}/status")
    assert status["indexed_files"] == 3, status
    assert status["total_chunks"] >= 3

    # Each mode returns at least one result for a keyword present in the corpus.
    for mode in ("hybrid", "semantic", "keyword", "fuzzy"):
        resp = _http_post(port, "/search", {
            "collection": name,
            "query": "hermit",
            "mode": mode,
            "top_k": 3,
        })
        assert resp["results"], f"mode={mode} returned empty"

    # Filename filter
    resp = _http_post(port, "/search", {
        "collection": name,
        "query": "hermit",
        "filename": "intro",
    })
    assert resp["results"]
    assert all(r["source_file"].endswith("intro.md") for r in resp["results"])

    # Glob filename filter
    resp = _http_post(port, "/search", {
        "collection": name,
        "query": "hermit",
        "filename": "*.md",
    })
    assert resp["results"]

    # Clean removal
    _http_delete(port, f"/collections/{name}")
    health = _http_get(port, "/health")
    assert all(c["name"] != name for c in health["collections"])
