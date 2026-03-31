"""ONNX INT8 dynamic quantization utilities.

Quantizes fp32 ONNX model weights to INT8 once, saves the result under
MODEL_ROOT/quantized/.  Subsequent startups load the quantized file directly
via fastembed's `specific_model_path` kwarg, bypassing any download.

Directory layout after quantization:
    {MODEL_ROOT}/quantized/{repo_id_slug}/
        onnx/
            model.onnx          ← INT8 quantized weights
        tokenizer.json          ← copied from original snapshot
        config.json             ← copied from original snapshot
        tokenizer_config.json   ← copied from original snapshot (if present)
        special_tokens_map.json ← copied from original snapshot (if present)

The `model_file` field in fastembed's registry is "onnx/model.onnx", so
passing `specific_model_path=str(quantized_dir)` makes fastembed load
`quantized_dir/onnx/model.onnx` without any code changes to the registry.
"""

import logging
import shutil
from pathlib import Path

from hermit.config import MODEL_ROOT

logger = logging.getLogger(__name__)

QUANTIZED_DIR = MODEL_ROOT / "quantized"

# Tokenizer/config files to copy alongside the quantized ONNX
_SIDECAR_FILES = [
    "tokenizer.json",
    "config.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "sentencepiece.bpe.model",
]


def _slug(repo_id: str) -> str:
    return repo_id.replace("/", "--")


def get_quantized_dir(repo_id: str) -> Path:
    """Return the directory that will hold the quantized model."""
    return QUANTIZED_DIR / _slug(repo_id)


def is_quantized(repo_id: str) -> bool:
    """Return True if an INT8-quantized ONNX already exists for this model."""
    return (get_quantized_dir(repo_id) / "onnx" / "model.onnx").exists()


def _find_snapshot_dir(repo_id: str) -> Path | None:
    """Resolve the active snapshot directory for a cached HuggingFace model."""
    cache_dir = MODEL_ROOT / f"models--{_slug(repo_id)}"
    refs_main = cache_dir / "refs" / "main"
    if not refs_main.exists():
        logger.debug("No refs/main for %s, trying snapshots directory", repo_id)
        snapshots = cache_dir / "snapshots"
        if not snapshots.is_dir():
            return None
        # Fall back to the first available snapshot
        revisions = [p for p in snapshots.iterdir() if p.is_dir()]
        return revisions[0] if revisions else None

    revision = refs_main.read_text().strip()
    snapshot = cache_dir / "snapshots" / revision
    return snapshot if snapshot.is_dir() else None


def quantize(repo_id: str, model_file: str = "onnx/model.onnx") -> bool:
    """Quantize a single model's ONNX weights from fp32 to INT8.

    Args:
        repo_id:    HuggingFace repo id, e.g. "jinaai/jina-embeddings-v2-base-zh"
        model_file: Relative path within the snapshot dir, e.g. "onnx/model.onnx"

    Returns:
        True if quantization succeeded (or was already done), False on failure.
        On failure the caller should fall back to the original fp32 model.
    """
    q_dir = get_quantized_dir(repo_id)
    q_onnx = q_dir / "onnx" / "model.onnx"

    if q_onnx.exists():
        logger.info("Quantized model already present: %s", q_onnx)
        return True

    snapshot = _find_snapshot_dir(repo_id)
    if snapshot is None:
        logger.error("Cannot quantize %s: no local snapshot found in %s", repo_id, MODEL_ROOT)
        return False

    src_onnx = snapshot / model_file
    # resolve() follows symlinks (HF cache stores blobs as symlinks)
    src_resolved = src_onnx.resolve()
    if not src_resolved.exists():
        logger.error("Source ONNX not found: %s", src_onnx)
        return False

    logger.info(
        "Quantizing %s to INT8 (source: %.0fMB) — runs once, may take a few minutes...",
        repo_id,
        src_resolved.stat().st_size / 1024 / 1024,
    )

    q_onnx.parent.mkdir(parents=True, exist_ok=True)
    tmp_onnx = q_onnx.with_suffix(".onnx.tmp")

    try:
        from onnxruntime.quantization import QuantType, quantize_dynamic
        from hermit.models import _log_heartbeat

        with _log_heartbeat(f"Quantizing {repo_id} to INT8..."):
            quantize_dynamic(
                str(src_resolved),
                str(tmp_onnx),
                weight_type=QuantType.QInt8,
            )
        tmp_onnx.rename(q_onnx)
    except Exception as exc:
        logger.warning("Quantization failed for %s: %s — will use fp32 model", repo_id, exc)
        if tmp_onnx.exists():
            tmp_onnx.unlink()
        return False

    # Copy sidecar files (tokenizer, config) from snapshot to quantized dir
    for fname in _SIDECAR_FILES:
        src = snapshot / fname
        if src.exists():
            dst = q_dir / fname
            if not dst.exists():
                try:
                    shutil.copy2(str(src.resolve()), str(dst))
                except Exception as exc:
                    logger.warning("Failed to copy %s: %s", fname, exc)

    size_mb = q_onnx.stat().st_size / 1024 / 1024
    logger.info("Quantization complete: %s (%.0fMB)", q_onnx, size_mb)
    return True
