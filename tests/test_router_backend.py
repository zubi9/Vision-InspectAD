from types import SimpleNamespace

import pytest
from PIL import Image

from api.backends import router as router_module


def _fake_yolo_cls(top1_idx, top1conf, names):
    result = SimpleNamespace(
        probs=SimpleNamespace(top1=top1_idx, top1conf=top1conf),
        names=names,
    )

    class FakeYOLO:
        def __init__(self, path):
            self.path = path

        def predict(self, source, verbose=False):
            return [result]

    return FakeYOLO


def test_route_above_threshold_returns_class(tmp_path, monkeypatch):
    model_path = tmp_path / "router.onnx"
    model_path.touch()
    monkeypatch.setattr(
        router_module, "YOLO",
        _fake_yolo_cls(top1_idx=2, top1conf=0.92, names={0: "DAGM", 1: "KolektorSDD2", 2: "MVTec_bottle"}),
    )

    r = router_module.RouterModel(model_path, confidence_threshold=0.6)
    decision = r.route(Image.new("RGB", (224, 224)))

    assert decision.is_confident is True
    assert decision.router_class == "MVTec_bottle"
    assert decision.raw_class == "MVTec_bottle"
    assert decision.confidence == pytest.approx(0.92)


def test_route_below_threshold_returns_none_class(tmp_path, monkeypatch):
    model_path = tmp_path / "router.onnx"
    model_path.touch()
    monkeypatch.setattr(
        router_module, "YOLO",
        _fake_yolo_cls(top1_idx=0, top1conf=0.3, names={0: "DAGM"}),
    )

    r = router_module.RouterModel(model_path, confidence_threshold=0.6)
    decision = r.route(Image.new("RGB", (224, 224)))

    assert decision.is_confident is False
    assert decision.router_class is None
    assert decision.raw_class == "DAGM"  # still reported, just not trusted
    assert decision.confidence == pytest.approx(0.3)


def test_route_exactly_at_threshold_is_confident(tmp_path, monkeypatch):
    model_path = tmp_path / "router.onnx"
    model_path.touch()
    monkeypatch.setattr(
        router_module, "YOLO",
        _fake_yolo_cls(top1_idx=0, top1conf=0.6, names={0: "Magnetic_Tile"}),
    )

    r = router_module.RouterModel(model_path, confidence_threshold=0.6)
    decision = r.route(Image.new("RGB", (224, 224)))

    assert decision.is_confident is True
    assert decision.router_class == "Magnetic_Tile"


def test_init_raises_if_onnx_missing(tmp_path):
    missing_path = tmp_path / "nope.onnx"
    with pytest.raises(FileNotFoundError):
        router_module.RouterModel(missing_path, confidence_threshold=0.6)
