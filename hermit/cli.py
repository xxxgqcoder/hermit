"""Hermit CLI — deployment & management commands.

Usage:
    hermit download              # download all models (resumes interrupted)
    hermit download --force      # force re-download
    hermit download --skip-verify
"""

import argparse
import logging
import time

from huggingface_hub import snapshot_download

from hermit.config import (
    DENSE_DIM,
    DENSE_MODEL,
    MODEL_ROOT,
    RERANKER_MODEL,
    SPARSE_MODEL,
)

logger = logging.getLogger(__name__)

# ── Model registry ──────────────────────────────────────────────
MODELS = [
    {
        "repo_id": DENSE_MODEL,
        "description": f"Dense embedding ({DENSE_DIM}-dim, Chinese-English)",
        "allow_patterns": [
            "onnx/model.onnx",
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
        ],
    },
    {
        "repo_id": SPARSE_MODEL,
        "description": "Sparse embedding (BM25)",
        "allow_patterns": None,  # small model, download everything
    },
    {
        "repo_id": RERANKER_MODEL,
        "description": "Reranker (multilingual cross-encoder)",
        "allow_patterns": [
            "onnx/model.onnx",
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
        ],
    },
]

MAX_RETRIES = 5
RETRY_DELAY = 3  # seconds


def download_model(repo_id: str, allow_patterns: list[str] | None, force: bool) -> str:
    """Download a single model with retry logic. Returns snapshot path."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            path = snapshot_download(
                repo_id=repo_id,
                cache_dir=str(MODEL_ROOT),
                allow_patterns=allow_patterns,
                force_download=force,
            )
            return path
        except Exception as e:
            if attempt == MAX_RETRIES:
                raise
            logger.warning(
                "Attempt %d/%d for %s failed: %s — retrying in %ds",
                attempt, MAX_RETRIES, repo_id, e, RETRY_DELAY,
            )
            time.sleep(RETRY_DELAY)
    raise RuntimeError("unreachable")


def verify_models():
    """Quick smoke test: load each model and run a dummy inference."""
    from fastembed import SparseTextEmbedding, TextEmbedding
    from fastembed.rerank.cross_encoder import TextCrossEncoder

    logger.info("Verifying models...")
    cache = str(MODEL_ROOT)

    dense = TextEmbedding(model_name=DENSE_MODEL, cache_dir=cache)
    vec = list(dense.embed(["test"]))[0]
    assert len(vec) == DENSE_DIM, f"Expected dim {DENSE_DIM}, got {len(vec)}"
    logger.info("  Dense embedding OK (dim=%d)", len(vec))

    sparse = SparseTextEmbedding(model_name=SPARSE_MODEL, cache_dir=cache)
    svec = list(sparse.embed(["test"]))[0]
    logger.info("  Sparse embedding OK (indices=%d)", len(svec.indices))

    reranker = TextCrossEncoder(model_name=RERANKER_MODEL, cache_dir=cache)
    scores = list(reranker.rerank("query", ["relevant doc", "irrelevant"]))
    assert scores[0] > scores[1], f"Reranker scoring seems wrong: {scores}"
    logger.info("  Reranker OK (scores=%s)", scores)

    logger.info("All models verified.")


def cmd_download(args):
    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    logger.info("Model cache: %s", MODEL_ROOT)

    for m in MODELS:
        logger.info("Downloading %s — %s", m["repo_id"], m["description"])
        path = download_model(m["repo_id"], m["allow_patterns"], args.force)
        logger.info("  -> %s", path)

    logger.info("All downloads complete.")

    if not args.skip_verify:
        verify_models()


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    parser = argparse.ArgumentParser(prog="hermit", description="Hermit management CLI")
    sub = parser.add_subparsers(dest="command")

    dl = sub.add_parser("download", help="Download all required models")
    dl.add_argument("--force", action="store_true", help="Force re-download")
    dl.add_argument("--skip-verify", action="store_true", help="Skip model verification")

    args = parser.parse_args()
    if args.command == "download":
        cmd_download(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
