import json
import logging
from pathlib import Path

from hermit.config import DATA_ROOT, DENSE_MODEL, SPARSE_MODEL

logger = logging.getLogger(__name__)

_SIGNATURE_PATH = DATA_ROOT / "model_signature.json"


def _current_signature() -> dict[str, str]:
    return {
        "dense_model": DENSE_MODEL,
        "sparse_model": SPARSE_MODEL,
    }


def load_saved_signature() -> dict[str, str] | None:
    if _SIGNATURE_PATH.exists():
        return json.loads(_SIGNATURE_PATH.read_text())
    return None


def save_signature():
    _SIGNATURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    _SIGNATURE_PATH.write_text(json.dumps(_current_signature(), indent=2))


def check_model_changed() -> tuple[bool, dict[str, str] | None, dict[str, str]]:
    """Return (changed, old_signature, new_signature)."""
    saved = load_saved_signature()
    current = _current_signature()
    if saved is None:
        save_signature()
        return False, None, current
    changed = saved != current
    return changed, saved, current
