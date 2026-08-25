from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from api.backends import anomalib_backend as backend


def test_missing_model_raises(tmp_path):
    model = backend.AnomalibBackend()

    with pytest.raises(FileNotFoundError, match="Anomalib ONNX model not found"):
        model._get_inferencer(tmp_path / "missing.onnx")


def test_inferencer_is_cached(tmp_path, monkeypatch):
    model_path = tmp_path / "model.onnx"
    model_path.touch()

    created = []

    class FakeInferencer:
        def __init__(self, **kwargs):
            created.append(kwargs)

    monkeypatch.setattr(backend, "OpenVINOInferencer", FakeInferencer)

    model = backend.AnomalibBackend(cache_size=1)
    first = model._get_inferencer(model_path)
    second = model._get_inferencer(model_path)

    assert first is second
    assert len(created) == 1


def test_cache_evicts_least_recently_used(tmp_path, monkeypatch):
    paths = [tmp_path / f"model{i}.onnx" for i in range(2)]
    for path in paths:
        path.touch()

    monkeypatch.setattr(
        backend,
        "OpenVINOInferencer",
        lambda **kwargs: object(),
    )

    model = backend.AnomalibBackend(cache_size=1)
    model._get_inferencer(paths[0])
    model._get_inferencer(paths[1])

    assert paths[0] not in model._cache
    assert paths[1] in model._cache


def test_heatmap_to_regions():
    model = backend.AnomalibBackend(region_threshold=0.5)

    heatmap = np.array(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.9, 0.9, 0.0],
            [0.0, 0.9, 0.9, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )

    regions = model._heatmap_to_regions(heatmap)

    assert len(regions) == 1
    assert regions[0].bbox == (1.0, 1.0, 3.0, 3.0)
    assert regions[0].score == pytest.approx(0.9)
    assert regions[0].label is None


class FakeDim:
    def __init__(self, length: int):
        self._length = length
        self.is_static = True

    def get_length(self) -> int:
        return self._length


class FakeInputBlob:
    def __init__(self, height: int, width: int, channels: int = 3):
        self.partial_shape = [FakeDim(1), FakeDim(channels), FakeDim(height), FakeDim(width)]


def test_predict_returns_expected_schema(tmp_path, monkeypatch):
    model_path = tmp_path / "model.onnx"
    model_path.touch()

    class FakeInferencer:
        def __init__(self, **kwargs):
            self.input_blob = FakeInputBlob(height=4, width=4)

        def predict(self, image):
            return SimpleNamespace(
                pred_score=0.91,
                pred_label=True,
                anomaly_map=np.array(
                    [
                        [0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.9, 0.9, 0.0],
                        [0.0, 0.9, 0.9, 0.0],
                        [0.0, 0.0, 0.0, 0.0],
                    ],
                    dtype=np.float32,
                ),
            )

    monkeypatch.setattr(backend, "OpenVINOInferencer", FakeInferencer)

    image = Image.new("RGB", (4, 4))
    result = backend.AnomalibBackend().predict(image, model_path)

    assert result["defect_detected"] is True
    assert result["image_score"] == pytest.approx(0.91)
    assert len(result["regions"]) == 1


def test_predict_falls_back_to_score_threshold(tmp_path, monkeypatch):
    model_path = tmp_path / "model.onnx"
    model_path.touch()

    class FakeInferencer:
        def __init__(self, **kwargs):
            self.input_blob = FakeInputBlob(height=4, width=4)

        def predict(self, image):
            return SimpleNamespace(
                pred_score=0.8,
                anomaly_map=np.zeros((4, 4), dtype=np.float32),
            )

    monkeypatch.setattr(backend, "OpenVINOInferencer", FakeInferencer)

    result = backend.AnomalibBackend(region_threshold=0.5).predict(
        Image.new("RGB", (4, 4)),
        model_path,
    )

    assert result["defect_detected"] is True
    assert result["regions"] == []