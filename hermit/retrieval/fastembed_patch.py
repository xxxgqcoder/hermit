"""Monkey-patch fastembed to expose ``enable_mem_pattern`` as a session option.

Background
----------
fastembed's ``OnnxModel`` (the base class under both ``TextEmbedding`` and
``TextCrossEncoder``) builds the ONNX Runtime ``SessionOptions`` internally and
only forwards ``enable_cpu_mem_arena`` from user kwargs via the
``EXPOSED_SESSION_OPTIONS`` allow-list. ``enable_mem_pattern`` — equally
important for keeping ONNX from pinning a per-shape memory layout — is not
exposed.

We need both off to stop the ONNX arena from ratcheting up:

  * ``enable_cpu_mem_arena=False``: per-Run allocations go through plain
    malloc instead of the pooled arena, so they're returned to the OS when
    free()'d. Without this the arena high-water-mark is sticky for the
    session's whole lifetime.
  * ``enable_mem_pattern=False``: disables predictive memory-pattern
    pre-allocation. With dynamic shapes (varying batch / token-length) the
    pattern cache adds memory without helping latency much, and on macOS it
    plays poorly with the arena disabled.

The patch is a tiny extension to the existing exposed-options machinery — it
appends ``enable_mem_pattern`` to ``EXPOSED_SESSION_OPTIONS`` and teaches
``add_extra_session_options`` how to apply it. No fork required.

See ``problems/concurrent-search-rss-blowup.md`` and
``problems/dense-embedder-arena-creep.md`` for the motivating measurements.
"""

from __future__ import annotations

import logging

import onnxruntime as ort
from fastembed.common.onnx_model import OnnxModel

logger = logging.getLogger(__name__)

_PATCHED_ATTR = "_hermit_arena_patch_applied"


def apply() -> None:
    """Extend fastembed's OnnxModel to also accept ``enable_mem_pattern``.

    Idempotent — guarded by a flag on the class so repeated imports are no-ops.
    """
    if getattr(OnnxModel, _PATCHED_ATTR, False):
        return

    OnnxModel.EXPOSED_SESSION_OPTIONS = tuple(
        list(OnnxModel.EXPOSED_SESSION_OPTIONS) + ["enable_mem_pattern"]
    )

    _original_add = OnnxModel.add_extra_session_options

    @classmethod
    def _patched_add(
        cls,
        session_options: ort.SessionOptions,
        extra_options: dict,
    ) -> None:
        _original_add(session_options, extra_options)
        if "enable_mem_pattern" in extra_options:
            session_options.enable_mem_pattern = extra_options["enable_mem_pattern"]

    OnnxModel.add_extra_session_options = _patched_add
    setattr(OnnxModel, _PATCHED_ATTR, True)
    logger.debug(
        "fastembed OnnxModel patched: enable_mem_pattern is now an exposed session option",
    )


# Apply on import — callers just `import hermit.retrieval.fastembed_patch`
# (or import this module before constructing any fastembed model).
apply()
