"""Track which Qdrant deployment mode last wrote to ``DATA_ROOT/qdrant``.

Why this exists
───────────────
The embedded ``qdrant-client`` (Local mode) and the Rust ``qdrant`` server
(Stand-alone mode) write incompatible on-disk layouts:

  * Local mode    → ``DATA_ROOT/qdrant/collection/<name>/storage.sqlite``
  * Stand-alone   → ``DATA_ROOT/qdrant/collections/<name>/segments/...``

A blind switch between modes leaves the new engine looking at a directory
it does not understand: it silently creates an empty collection, while
Hermit's own SQLite metadata still claims N indexed files. Search returns
nothing. There is no in-place migration available between these layouts
(see design-doc §2.1 — the stated "100% compatibility" turned out to be
aspirational).

What this module does
─────────────────────
Persist the mode under which Hermit last successfully booted, and
report mismatches at startup so ``app.lifespan`` can trigger a full
``rebuild_collection`` for every registered collection (same path used
when the embedding model changes).
"""

import json
import logging
from pathlib import Path

from hermit.config import DATA_ROOT

logger = logging.getLogger(__name__)

_SIGNATURE_PATH = DATA_ROOT / "qdrant_mode.json"


def _current_mode(qdrant_host: str | None) -> str:
    """Resolve the mode string from runtime config."""
    return "standalone" if qdrant_host else "local"


def load_saved_mode() -> str | None:
    if _SIGNATURE_PATH.exists():
        try:
            return json.loads(_SIGNATURE_PATH.read_text()).get("mode")
        except (json.JSONDecodeError, OSError):
            return None
    return None


def save_mode(mode: str) -> None:
    _SIGNATURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    _SIGNATURE_PATH.write_text(json.dumps({"mode": mode}, indent=2))


def check_mode_changed(
    qdrant_host: str | None,
) -> tuple[bool, str | None, str]:
    """Return ``(changed, old_mode, new_mode)``.

    First-ever boot (no saved file) is treated as *not changed*: there
    is nothing for the old engine to have written that the new engine
    would need to flush. The current mode is persisted on the spot so
    the next boot can detect a real switch.
    """
    saved = load_saved_mode()
    current = _current_mode(qdrant_host)
    if saved is None:
        save_mode(current)
        return False, None, current
    return saved != current, saved, current
