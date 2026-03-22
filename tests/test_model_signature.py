"""Tests for model signature change detection and rebuild logic."""

import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest


@pytest.fixture
def tmp_data_root(tmp_path, monkeypatch):
    monkeypatch.setattr("hermit.storage.model_signature.DATA_ROOT", tmp_path)
    monkeypatch.setattr("hermit.storage.model_signature._SIGNATURE_PATH", tmp_path / "model_signature.json")
    return tmp_path


# ── model_signature tests ───────────────────────────────────────


class TestCheckModelChanged:
    """Tests for check_model_changed() logic."""

    def test_first_run_no_signature_file(self, tmp_data_root):
        """First run: no signature file exists, should save and report no change."""
        from hermit.storage.model_signature import check_model_changed

        changed, old_sig, new_sig = check_model_changed()

        assert changed is False
        assert old_sig is None
        assert new_sig["dense_model"] != ""
        assert new_sig["sparse_model"] != ""
        # Signature file should have been created
        assert (tmp_data_root / "model_signature.json").exists()

    def test_no_change_on_second_run(self, tmp_data_root):
        """Second run with same models: should report no change."""
        from hermit.storage.model_signature import check_model_changed

        # First run creates the file
        check_model_changed()
        # Second run should find no change
        changed, old_sig, new_sig = check_model_changed()

        assert changed is False
        assert old_sig == new_sig

    def test_detects_dense_model_change(self, tmp_data_root):
        """Should detect when dense model changes."""
        from hermit.storage.model_signature import check_model_changed, _SIGNATURE_PATH

        sig_path = tmp_data_root / "model_signature.json"
        sig_path.write_text(json.dumps({
            "dense_model": "old-model/dense-v1",
            "sparse_model": "Qdrant/bm25",
        }, indent=2))

        changed, old_sig, new_sig = check_model_changed()

        assert changed is True
        assert old_sig["dense_model"] == "old-model/dense-v1"
        assert new_sig["dense_model"] != "old-model/dense-v1"

    def test_detects_sparse_model_change(self, tmp_data_root):
        """Should detect when sparse model changes."""
        from hermit.storage.model_signature import check_model_changed

        sig_path = tmp_data_root / "model_signature.json"
        sig_path.write_text(json.dumps({
            "dense_model": "jinaai/jina-embeddings-v2-base-zh",
            "sparse_model": "old-model/sparse-v1",
        }, indent=2))

        changed, old_sig, new_sig = check_model_changed()

        assert changed is True
        assert old_sig["sparse_model"] == "old-model/sparse-v1"
        assert new_sig["sparse_model"] != "old-model/sparse-v1"

    def test_detects_both_models_changed(self, tmp_data_root):
        """Should detect when both models change."""
        from hermit.storage.model_signature import check_model_changed

        sig_path = tmp_data_root / "model_signature.json"
        sig_path.write_text(json.dumps({
            "dense_model": "old/dense",
            "sparse_model": "old/sparse",
        }, indent=2))

        changed, old_sig, new_sig = check_model_changed()

        assert changed is True
        assert old_sig["dense_model"] == "old/dense"
        assert old_sig["sparse_model"] == "old/sparse"


class TestSaveAndLoadSignature:
    """Tests for save_signature() and load_saved_signature()."""

    def test_save_and_load_round_trip(self, tmp_data_root):
        from hermit.storage.model_signature import save_signature, load_saved_signature

        assert load_saved_signature() is None
        save_signature()
        sig = load_saved_signature()
        assert sig is not None
        assert "dense_model" in sig
        assert "sparse_model" in sig

    def test_save_overwrites_old(self, tmp_data_root):
        from hermit.storage.model_signature import save_signature, load_saved_signature

        sig_path = tmp_data_root / "model_signature.json"
        sig_path.write_text(json.dumps({"dense_model": "old", "sparse_model": "old"}))

        save_signature()
        sig = load_saved_signature()
        assert sig["dense_model"] != "old"


# ── rebuild_collection tests ────────────────────────────────────


class TestRebuildCollection:
    """Tests for rebuild_collection() in scanner module."""

    @patch("hermit.ingestion.scanner.qdrant")
    @patch("hermit.ingestion.scanner.MetadataStore")
    @patch("hermit.ingestion.scanner.scan_folder")
    def test_rebuild_deletes_and_rescans(self, mock_scan, mock_meta_cls, mock_qdrant):
        from hermit.ingestion.scanner import rebuild_collection

        mock_meta_instance = MagicMock()
        mock_meta_cls.return_value = mock_meta_instance
        mock_scan.return_value = {"added": 3, "updated": 0, "deleted": 0}

        result = rebuild_collection("test_col", "/tmp/docs", chunk_size=256, chunk_overlap=32)

        mock_qdrant.delete_collection.assert_called_once_with("test_col")
        mock_meta_instance.destroy.assert_called_once()
        mock_scan.assert_called_once_with(
            "test_col", "/tmp/docs",
            chunk_size=256, chunk_overlap=32,
            defer_indexing=True,
        )
        assert result == {"added": 3, "updated": 0, "deleted": 0}
