"""Tests for knowledge base CLI commands and registry validation."""

import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest


# ── Registry validation tests ───────────────────────────────────


@pytest.fixture
def tmp_registry(tmp_path, monkeypatch):
    monkeypatch.setattr("hermit.storage.registry.DATA_ROOT", tmp_path)
    monkeypatch.setattr("hermit.storage.registry._REGISTRY_PATH", tmp_path / "collections.json")
    return tmp_path


def test_register_max_collections(tmp_registry):
    """Should reject registration when MAX_COLLECTIONS is reached."""
    from hermit.storage.registry import register, get_all
    from hermit.config import MAX_COLLECTIONS

    for i in range(MAX_COLLECTIONS):
        register(f"col{i}", f"/tmp/dir{i}")

    assert len(get_all()) == MAX_COLLECTIONS

    with pytest.raises(ValueError, match="Maximum"):
        register("one_more", "/tmp/extra")


def test_register_duplicate_folder(tmp_registry):
    """Should reject registering the same folder under a different name."""
    from hermit.storage.registry import register

    register("col1", "/tmp/docs")

    with pytest.raises(ValueError, match="already registered"):
        register("col2", "/tmp/docs")


def test_register_invalid_name_special_chars(tmp_registry):
    """Should reject names with special characters."""
    from hermit.storage.registry import register

    for bad_name in ["my docs", "my/docs", "my.docs", "@alias"]:
        with pytest.raises(ValueError, match="Invalid collection name"):
            register(bad_name, f"/tmp/{id(bad_name)}")


def test_register_invalid_name_start_char(tmp_registry):
    """Should reject names starting with hyphen or underscore."""
    from hermit.storage.registry import register

    for bad_name in ["-start", "_start"]:
        with pytest.raises(ValueError, match="Invalid collection name"):
            register(bad_name, f"/tmp/{bad_name}")


def test_register_valid_name_formats(tmp_registry):
    """Should accept well-formed names."""
    from hermit.storage.registry import register, get_all

    for i, name in enumerate(["docs", "my-docs", "my_docs", "Doc123"]):
        register(name, f"/tmp/dir{i}")

    assert len(get_all()) == 4


def test_register_duplicate_name(tmp_registry):
    """Should reject registering a duplicate collection name."""
    from hermit.storage.registry import register

    register("col1", "/tmp/docs")

    with pytest.raises(ValueError, match="already exists"):
        register("col1", "/tmp/docs_v2")


def test_register_name_too_long(tmp_registry):
    """Should reject name exceeding MAX_COLLECTION_NAME_LENGTH."""
    from hermit.storage.registry import register
    from hermit.config import MAX_COLLECTION_NAME_LENGTH

    long_name = "a" * (MAX_COLLECTION_NAME_LENGTH + 1)
    with pytest.raises(ValueError, match="must not exceed"):
        register(long_name, "/tmp/docs")


def test_register_name_at_max_length(tmp_registry):
    """Name exactly at max length should be accepted."""
    from hermit.storage.registry import register, get_all
    from hermit.config import MAX_COLLECTION_NAME_LENGTH

    name = "a" * MAX_COLLECTION_NAME_LENGTH
    register(name, "/tmp/docs")
    assert name in get_all()


def test_register_empty_name(tmp_registry):
    """Should reject empty name."""
    from hermit.storage.registry import register

    with pytest.raises(ValueError, match="must not be empty"):
        register("", "/tmp/docs")


def test_register_reuse_folder_after_unregister(tmp_registry):
    """After unregistering, the same folder should be available again."""
    from hermit.storage.registry import register, unregister

    register("col1", "/tmp/docs")
    unregister("col1")
    register("col2", "/tmp/docs")  # should succeed


# ── Ignore patterns in registry ─────────────────────────────────


def test_register_with_ignore_patterns(tmp_registry):
    """register() should persist ignore_patterns and ignore_extensions."""
    from hermit.storage.registry import register, get_all

    register("col1", "/tmp/docs", ignore_patterns=["build/**", "*.log"],
             ignore_extensions=[".pdf", ".BIN"])
    cfg = get_all()["col1"]
    assert cfg["ignore_patterns"] == ["build/**", "*.log"]
    assert cfg["ignore_extensions"] == [".pdf", ".bin"]  # lowercased


def test_register_without_ignore_defaults_empty(tmp_registry):
    """register() without ignore args should store empty lists."""
    from hermit.storage.registry import register, get_all

    register("col1", "/tmp/docs")
    cfg = get_all()["col1"]
    assert cfg["ignore_patterns"] == []
    assert cfg["ignore_extensions"] == []


def test_update_ignore_patterns(tmp_registry):
    """update() should replace ignore_patterns for an existing collection."""
    from hermit.storage.registry import register, update, get_all

    register("col1", "/tmp/docs")
    update("col1", ignore_patterns=["*.tmp", "cache/**"])
    cfg = get_all()["col1"]
    assert cfg["ignore_patterns"] == ["*.tmp", "cache/**"]
    assert cfg["ignore_extensions"] == []  # unchanged


def test_update_ignore_extensions(tmp_registry):
    """update() should replace ignore_extensions for an existing collection."""
    from hermit.storage.registry import register, update, get_all

    register("col1", "/tmp/docs", ignore_extensions=[".pdf"])
    update("col1", ignore_extensions=[".bin", ".EXE"])
    cfg = get_all()["col1"]
    assert cfg["ignore_extensions"] == [".bin", ".exe"]  # lowercased


def test_update_clear_ignore(tmp_registry):
    """update() with clear_ignore should empty ignore_patterns."""
    from hermit.storage.registry import register, update, get_all

    register("col1", "/tmp/docs", ignore_patterns=["*.log"])
    update("col1", clear_ignore=True)
    assert get_all()["col1"]["ignore_patterns"] == []


def test_update_clear_ignore_ext(tmp_registry):
    """update() with clear_ignore_ext should empty ignore_extensions."""
    from hermit.storage.registry import register, update, get_all

    register("col1", "/tmp/docs", ignore_extensions=[".pdf"])
    update("col1", clear_ignore_ext=True)
    assert get_all()["col1"]["ignore_extensions"] == []


def test_update_nonexistent_collection(tmp_registry):
    """update() for a missing collection should raise ValueError."""
    from hermit.storage.registry import update

    with pytest.raises(ValueError, match="not found"):
        update("ghost", ignore_patterns=["*.log"])


# ── CLI integration tests (subprocess) ──────────────────────────


@pytest.fixture
def cli_env(tmp_path, monkeypatch):
    """Patch DATA_ROOT so CLI commands use a temp directory."""
    monkeypatch.setattr("hermit.storage.registry.DATA_ROOT", tmp_path)
    monkeypatch.setattr("hermit.storage.registry._REGISTRY_PATH", tmp_path / "collections.json")
    return tmp_path


def test_kb_list_empty(cli_env, capsys):
    """kb list with no collections returns empty JSON."""
    from hermit.cli import main

    with pytest.raises(SystemExit) as exc_info:
        with patch("sys.argv", ["hermit", "kb", "list"]):
            main()
    assert exc_info.value.code == 0

    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert data["collections"] == {}


def test_kb_add_and_list(cli_env, tmp_path, capsys):
    """kb add should register a directory, kb list should show it."""
    from hermit.cli import main

    test_dir = tmp_path / "my_docs"
    test_dir.mkdir()

    with pytest.raises(SystemExit) as exc_info:
        with patch("sys.argv", ["hermit", "kb", "add", "my_docs", str(test_dir)]):
            main()
    assert exc_info.value.code == 0

    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert data["status"] == "added"
    assert data["name"] == "my_docs"

    with pytest.raises(SystemExit) as exc_info:
        with patch("sys.argv", ["hermit", "kb", "list"]):
            main()
    assert exc_info.value.code == 0

    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert "my_docs" in data["collections"]
    assert data["collections"]["my_docs"]["folder_path"] == str(test_dir)


def test_kb_add_invalid_name(cli_env, tmp_path, capsys):
    """kb add with invalid alias should fail."""
    from hermit.cli import main

    test_dir = tmp_path / "docs"
    test_dir.mkdir()

    for bad_name in ["my docs", "my/docs", "@alias"]:
        with pytest.raises(SystemExit, match="1"):
            with patch("sys.argv", ["hermit", "kb", "add", bad_name, str(test_dir)]):
                main()


def test_kb_add_nonexistent_dir(cli_env, tmp_path, capsys):
    """kb add with a non-existent directory should fail."""
    from hermit.cli import main

    with pytest.raises(SystemExit, match="1"):
        with patch("sys.argv", ["hermit", "kb", "add", "nope", str(tmp_path / "nope")]):
            main()


def test_kb_add_duplicate_name(cli_env, tmp_path, capsys):
    """kb add with a name that already exists should fail."""
    from hermit.cli import main

    d1 = tmp_path / "docs1"
    d1.mkdir()
    d2 = tmp_path / "docs2"
    d2.mkdir()

    with pytest.raises(SystemExit) as exc_info:
        with patch("sys.argv", ["hermit", "kb", "add", "myalias", str(d1)]):
            main()
    assert exc_info.value.code == 0

    with pytest.raises(SystemExit) as exc_info:
        with patch("sys.argv", ["hermit", "kb", "add", "myalias", str(d2)]):
            main()
    assert exc_info.value.code == 1


def test_kb_add_duplicate_folder(cli_env, tmp_path, capsys):
    """kb add same folder with different name should fail."""
    from hermit.cli import main

    test_dir = tmp_path / "docs"
    test_dir.mkdir()

    with pytest.raises(SystemExit) as exc_info:
        with patch("sys.argv", ["hermit", "kb", "add", "alias1", str(test_dir)]):
            main()
    assert exc_info.value.code == 0

    with pytest.raises(SystemExit) as exc_info:
        with patch("sys.argv", ["hermit", "kb", "add", "alias2", str(test_dir)]):
            main()
    assert exc_info.value.code == 1


def test_kb_add_name_too_long(cli_env, tmp_path, capsys):
    """kb add with a name exceeding 64 chars should fail."""
    from hermit.cli import main
    from hermit.config import MAX_COLLECTION_NAME_LENGTH

    test_dir = tmp_path / "docs"
    test_dir.mkdir()
    long_name = "x" * (MAX_COLLECTION_NAME_LENGTH + 1)

    with pytest.raises(SystemExit, match="1"):
        with patch("sys.argv", ["hermit", "kb", "add", long_name, str(test_dir)]):
            main()


def test_kb_add_exceeds_max(cli_env, tmp_path, capsys):
    """kb add beyond MAX_COLLECTIONS should fail."""
    from hermit.cli import main
    from hermit.config import MAX_COLLECTIONS

    for i in range(MAX_COLLECTIONS):
        d = tmp_path / f"dir{i}"
        d.mkdir()
        with pytest.raises(SystemExit) as exc_info:
            with patch("sys.argv", ["hermit", "kb", "add", f"col{i}", str(d)]):
                main()
        assert exc_info.value.code == 0

    extra = tmp_path / "extra"
    extra.mkdir()
    with pytest.raises(SystemExit) as exc_info:
        with patch("sys.argv", ["hermit", "kb", "add", "extra", str(extra)]):
            main()
    assert exc_info.value.code == 1


def test_kb_remove(cli_env, tmp_path, capsys, monkeypatch):
    """kb remove should unregister and destroy metadata."""
    from hermit.cli import main
    from hermit.storage.registry import get_all

    test_dir = tmp_path / "docs"
    test_dir.mkdir()

    # Patch MetadataStore.destroy so it doesn't need a real SQLite DB
    mock_destroyed = []

    class FakeMetadataStore:
        def __init__(self, name):
            self.name = name

        def destroy(self):
            mock_destroyed.append(self.name)

    monkeypatch.setattr("hermit.storage.metadata.MetadataStore", FakeMetadataStore)

    with pytest.raises(SystemExit) as exc_info:
        with patch("sys.argv", ["hermit", "kb", "add", "docs", str(test_dir)]):
            main()
    assert exc_info.value.code == 0
    assert "docs" in get_all()

    with pytest.raises(SystemExit) as exc_info:
        with patch("sys.argv", ["hermit", "kb", "remove", "docs"]):
            main()
    assert exc_info.value.code == 0

    captured = capsys.readouterr()
    data = json.loads(captured.out.strip().split("\n")[-1])
    assert data["status"] == "removed"
    assert data["name"] == "docs"
    assert get_all() == {}
    assert "docs" in mock_destroyed


def test_kb_remove_nonexistent(cli_env, capsys):
    """kb remove of a non-existent collection should fail."""
    from hermit.cli import main

    with pytest.raises(SystemExit, match="1"):
        with patch("sys.argv", ["hermit", "kb", "remove", "ghost"]):
            main()


def test_kb_add_registers_collection(cli_env, tmp_path, capsys):
    """kb add should register the collection without chunk params."""
    from hermit.cli import main
    from hermit.storage.registry import get_all

    test_dir = tmp_path / "docs"
    test_dir.mkdir()

    with pytest.raises(SystemExit) as exc_info:
        with patch("sys.argv", [
            "hermit", "kb", "add", "docs", str(test_dir),
        ]):
            main()
    assert exc_info.value.code == 0

    cfg = get_all()["docs"]
    assert cfg["folder_path"] == str(test_dir)
    assert "chunk_size" not in cfg
    assert "chunk_overlap" not in cfg
