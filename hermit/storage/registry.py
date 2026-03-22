import json
import logging
from pathlib import Path

from hermit.config import DATA_ROOT

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
    data = _load()
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
