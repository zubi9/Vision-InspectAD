import pytest

from src.common import model_registry, paths


def test_build_registry_covers_all_18_router_classes():
    registry = model_registry.build_registry()
    assert set(registry.keys()) == set(paths.ROUTER_CLASSES)


def test_mvtec_categories_map_to_anomalib_backend():
    registry = model_registry.build_registry(anomalib_models_dir=paths.ANOMALIB_MODELS_DIR)
    for category in paths.MVTEC_CATEGORIES:
        entry = registry[f"MVTec_{category}"]
        assert entry.backend == "anomalib"
        assert entry.checkpoint.name == f"patchcore_{category}.ckpt"


def test_supervised_classes_map_to_separate_specialist_checkpoints():
    registry = model_registry.build_registry(
        dagm_checkpoint=paths.PROJECT_ROOT / "models" / "dagm" / "weights" / "best.pt",
        kolektor_checkpoint=paths.PROJECT_ROOT / "models" / "kolektor" / "weights" / "best.pt",
        magnetic_tile_checkpoint=paths.PROJECT_ROOT / "models" / "magnetic_tile" / "weights" / "best.pt",
    )
    assert registry["DAGM"].checkpoint != registry["KolektorSDD2"].checkpoint
    assert registry["KolektorSDD2"].checkpoint != registry["Magnetic_Tile"].checkpoint
    assert all(registry[c].backend == "yolo26-seg" for c in paths.SUPERVISED_ROUTER_CLASSES)


def test_resolve_returns_entry_when_checkpoint_exists(tmp_path):
    checkpoint = tmp_path / "patchcore_bottle.ckpt"
    checkpoint.touch()
    registry = {"MVTec_bottle": model_registry.ModelEntry(backend="anomalib", checkpoint=checkpoint)}

    entry = model_registry.resolve(registry, "MVTec_bottle")

    assert entry.checkpoint == checkpoint


def test_resolve_raises_file_not_found_when_checkpoint_missing(tmp_path):
    registry = {
        "MVTec_bottle": model_registry.ModelEntry(backend="anomalib", checkpoint=tmp_path / "missing.ckpt")
    }
    with pytest.raises(FileNotFoundError):
        model_registry.resolve(registry, "MVTec_bottle")


def test_resolve_raises_key_error_for_unknown_class():
    with pytest.raises(KeyError):
        model_registry.resolve({}, "totally_made_up_class")


def test_resolve_raises_value_error_for_unknown_router_output():
    """UNKNOWN is what the router itself returns when unconfident -- resolve()
    should refuse it explicitly rather than doing a registry lookup that would
    raise a less informative KeyError."""
    with pytest.raises(ValueError):
        model_registry.resolve({}, "UNKNOWN")


def test_build_onnx_registry_uses_rglob_discovery(tmp_path, monkeypatch):
    """Anomalib's exporter has changed its output nesting across versions already
    in this project -- build_onnx_registry must find the file by name via rglob,
    not assume a fixed nested path."""
    onnx_dir = tmp_path / "onnx_root"
    nested = onnx_dir / "weights" / "onnx"
    nested.mkdir(parents=True)
    (nested / "patchcore_bottle.onnx").touch()

    monkeypatch.setattr(paths, "ANOMALIB_ONNX_DIR", onnx_dir)

    registry = model_registry.build_onnx_registry()

    assert registry["MVTec_bottle"].checkpoint == nested / "patchcore_bottle.onnx"
