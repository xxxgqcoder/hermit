import json
import logging
from pathlib import Path

from hermit.config import DATA_ROOT, MAX_COLLECTIONS, MAX_COLLECTION_NAME_LENGTH, COLLECTION_NAME_PATTERN

logger = logging.getLogger(__name__)

_REGISTRY_PATH = DATA_ROOT / "collections.json"


def _load() -> dict[str, dict]:
    if _REGISTRY_PATH.exists():
        return json.loads(_REGISTRY_PATH.read_text())
    return {}


def _save(data: dict[str, dict]):
    _REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    _REGISTRY_PATH.write_text(json.dumps(data, indent=2))


def register(name: str, folder_path: str, chunk_size: int, chunk_overlap: int):
    if not name:
        raise ValueError("Collection name must not be empty")
    if len(name) > MAX_COLLECTION_NAME_LENGTH:
        raise ValueError(
            f"Collection name must not exceed {MAX_COLLECTION_NAME_LENGTH} characters"
        )
    if not COLLECTION_NAME_PATTERN.match(name):
        raise ValueError(
            f"Invalid collection name '{name}'. "
            "Must start with a letter or digit, and contain only letters, digits, underscores, or hyphens."
        )
    data = _load()
    if name not in data and len(data) >= MAX_COLLECTIONS:
        raise ValueError(f"Maximum {MAX_COLLECTIONS} collections allowed")
    if name in data:
        raise ValueError(f"Collection '{name}' already exists")
    for existing_name, cfg in data.items():
        if cfg["folder_path"] == folder_path and existing_name != name:
            raise ValueError(
                f"Directory '{folder_path}' is already registered as collection '{existing_name}'"
            )
    data[name] = {
        "folder_path": folder_path,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
    }
    _save(data)


def unregister(name: str):
    data = _load()
    data.pop(name, None)
    _save(data)


def get_all() -> dict[str, dict]:
    return _load()
