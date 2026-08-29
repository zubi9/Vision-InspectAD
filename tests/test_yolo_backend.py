from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from api.backends import yolo_backend as backend


class FakeTensor:
    """Mimics the .cpu().numpy() chain used on ultralytics' torch tensor results."""

    def __init__(self, array):
        self._array = np.asarray(array)

    def cpu(self):
        return self

    def numpy(self):
        return self._array


def _fake_result(boxes_xyxy, scores, class_ids, names, masks_xy=None):
    boxes = SimpleNamespace(
        xyxy=FakeTensor(boxes_xyxy),
        conf=FakeTensor(scores),
        cls=FakeTensor(class_ids),
    )
    masks = SimpleNamespace(xy=masks_xy) if masks_xy is not None else None
    return SimpleNamespace(boxes=boxes, masks=masks, names=names)


def test_predict_extracts_regions_with_labels(tmp_path, monkeypatch):
    model_path = tmp_path / "model.onnx"
    model_path.touch()

    fake_result = _fake_result(
        boxes_xyxy=[[10, 10, 50, 50]],
        scores=[0.83],
        class_ids=[1],
        names={0: "blowhole", 1: "scratch"},
    )

    class FakeYOLO:
        def __init__(self, path, **kwargs):
            self.path = path

        def predict(self, source, verbose=False):
            return [fake_result]

    monkeypatch.setattr(backend, "YOLO", FakeYOLO)

    result = backend.Yolo26SegBackend().predict(Image.new("RGB", (100, 100)), model_path)

    assert result["defect_detected"] is True
    assert result["image_score"] == pytest.approx(0.83)
    assert len(result["regions"]) == 1
    region = result["regions"][0]
    assert region.label == "scratch"
    assert region.bbox == (10.0, 10.0, 50.0, 50.0)


def test_predict_no_detections_returns_empty(tmp_path, monkeypatch):
    model_path = tmp_path / "model.onnx"
    model_path.touch()

    fake_result = _fake_result(
        boxes_xyxy=np.empty((0, 4)),
        scores=np.empty((0,)),
        class_ids=np.empty((0,)),
        names={},
    )

    class FakeYOLO:
        def __init__(self, path, **kwargs):
            pass

        def predict(self, source, verbose=False):
            return [fake_result]

    monkeypatch.setattr(backend, "YOLO", FakeYOLO)

    result = backend.Yolo26SegBackend().predict(Image.new("RGB", (100, 100)), model_path)

    assert result["defect_detected"] is False
    assert result["image_score"] == 0.0
    assert result["regions"] == []


def test_predict_includes_mask_polygon_when_present(tmp_path, monkeypatch):
    model_path = tmp_path / "model.onnx"
    model_path.touch()

    polygon = np.array([[1.0, 1.0], [2.0, 1.0], [2.0, 2.0]])
    fake_result = _fake_result(
        boxes_xyxy=[[0, 0, 10, 10]],
        scores=[0.5],
        class_ids=[0],
        names={0: "crack"},
        masks_xy=[polygon],
    )

    class FakeYOLO:
        def __init__(self, path, **kwargs):
            pass

        def predict(self, source, verbose=False):
            return [fake_result]

    monkeypatch.setattr(backend, "YOLO", FakeYOLO)

    result = backend.Yolo26SegBackend().predict(Image.new("RGB", (10, 10)), model_path)

    assert result["regions"][0].mask == polygon.tolist()


def test_get_model_raises_if_checkpoint_missing(tmp_path):
    missing_path = tmp_path / "does_not_exist.onnx"
    with pytest.raises(FileNotFoundError):
        backend.Yolo26SegBackend()._get_model(missing_path)


def test_get_model_reuses_cached_instance(tmp_path, monkeypatch):
    model_path = tmp_path / "model.onnx"
    model_path.touch()

    init_calls = []

    class FakeYOLO:
        def __init__(self, path, **kwargs):
            init_calls.append(path)

    monkeypatch.setattr(backend, "YOLO", FakeYOLO)

    be = backend.Yolo26SegBackend()
    be._get_model(model_path)
    be._get_model(model_path)

    assert len(init_calls) == 1
