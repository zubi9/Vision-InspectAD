from pathlib import Path

from scripts.download_models import HF_TO_LOCAL, load_manifest, save_manifest, should_download
from src.common import paths


def test_hf_to_local_covers_all_19_models():
    assert len(HF_TO_LOCAL) == 19


def test_hf_to_local_anomalib_names_match_paths():
    for category in paths.MVTEC_CATEGORIES:
        filename = f"patchcore_{category}.onnx"
        assert HF_TO_LOCAL[filename] == paths.ANOMALIB_ONNX_DIR / filename


def test_hf_to_local_specialist_names_match_paths():
    assert HF_TO_LOCAL["dagm-best.onnx"] == paths.DAGM_ONNX_PATH
    assert HF_TO_LOCAL["kolekor-best.onnx"] == paths.KOLEKTOR_ONNX_PATH
    assert HF_TO_LOCAL["magnetic_tile-best.onnx"] == paths.MAGNETIC_TILE_ONNX_PATH
    assert HF_TO_LOCAL["router-best.onnx"] == paths.ROUTER_ONNX_PATH


def test_should_download_missing_local_file(tmp_path):
    assert should_download("x.onnx", tmp_path / "nope.onnx", "etag123", {}, auto_yes=False) is True


def test_should_download_unknown_remote_tag_does_not_overwrite(tmp_path):
    existing = tmp_path / "exists.onnx"
    existing.touch()
    assert should_download("x.onnx", existing, None, {}, auto_yes=False) is False


def test_should_download_up_to_date_skips(tmp_path):
    existing = tmp_path / "exists.onnx"
    existing.touch()
    assert should_download("x.onnx", existing, "abc", {"x.onnx": "abc"}, auto_yes=False) is False


def test_should_download_stale_with_auto_yes(tmp_path):
    existing = tmp_path / "exists.onnx"
    existing.touch()
    assert should_download("x.onnx", existing, "abc", {"x.onnx": "old"}, auto_yes=True) is True


def test_manifest_round_trip(tmp_path, monkeypatch):
    import scripts.download_models as dm
    monkeypatch.setattr(dm, "MANIFEST_PATH", tmp_path / "models" / ".hf_manifest.json")

    save_manifest({"a.onnx": "tag1", "b.onnx": "tag2"})
    loaded = load_manifest()

    assert loaded == {"a.onnx": "tag1", "b.onnx": "tag2"}


def test_manifest_missing_file_returns_empty_dict(tmp_path, monkeypatch):
    import scripts.download_models as dm
    monkeypatch.setattr(dm, "MANIFEST_PATH", tmp_path / "models" / "does_not_exist.json")

    assert load_manifest() == {}
