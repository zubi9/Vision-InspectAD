import io
from dataclasses import dataclass

from fastapi.testclient import TestClient
from PIL import Image

from api.main import app, app_state
from src.common import model_registry
from src.common.schemas import Region

# TestClient(app) without a `with` block does not run the lifespan handler, so
# no real ONNX models get loaded -- app_state is populated per-test with fakes.
client = TestClient(app)


@dataclass
class FakeRoutingDecision:
    router_class: str | None
    raw_class: str
    confidence: float
    is_confident: bool


class FakeRouter:
    def __init__(self, decision: FakeRoutingDecision):
        self._decision = decision

    def route(self, image):
        return self._decision


class FakeAnomalibBackend:
    def predict(self, image, checkpoint):
        return {
            "defect_detected": True,
            "image_score": 0.88,
            "regions": [Region(bbox=(1, 1, 5, 5), score=0.88, label=None)],
            "heatmap_overlay_base64": "ZmFrZQ==",
        }


class FakeYoloBackend:
    def predict(self, image, checkpoint):
        return {
            "defect_detected": True,
            "image_score": 0.7,
            "regions": [Region(bbox=(2, 2, 8, 8), score=0.7, label="crack")],
        }


def _fake_image_bytes():
    buf = io.BytesIO()
    Image.new("RGB", (16, 16)).save(buf, format="PNG")
    return buf.getvalue()


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_dispatches_to_anomalib(tmp_path):
    checkpoint = tmp_path / "patchcore_bottle.ckpt"
    checkpoint.touch()

    app_state.router_model = FakeRouter(
        FakeRoutingDecision(router_class="MVTec_bottle", raw_class="MVTec_bottle", confidence=0.95, is_confident=True)
    )
    app_state.registry = {
        "MVTec_bottle": model_registry.ModelEntry(backend="anomalib", checkpoint=checkpoint)
    }
    app_state.anomalib_backend = FakeAnomalibBackend()
    app_state.yolo_backend = FakeYoloBackend()

    response = client.post("/predict", files={"file": ("test.png", _fake_image_bytes(), "image/png")})

    assert response.status_code == 200
    body = response.json()
    assert body["source_model"] == "anomalib"
    assert body["router_class"] == "MVTec_bottle"
    assert body["defect_detected"] is True
    assert body["heatmap_overlay_base64"] == "ZmFrZQ=="


def test_predict_dispatches_to_yolo(tmp_path):
    checkpoint = tmp_path / "best.pt"
    checkpoint.touch()

    app_state.router_model = FakeRouter(
        FakeRoutingDecision(router_class="DAGM", raw_class="DAGM", confidence=0.81, is_confident=True)
    )
    app_state.registry = {
        "DAGM": model_registry.ModelEntry(backend="yolo26-seg", checkpoint=checkpoint)
    }
    app_state.anomalib_backend = FakeAnomalibBackend()
    app_state.yolo_backend = FakeYoloBackend()

    response = client.post("/predict", files={"file": ("test.png", _fake_image_bytes(), "image/png")})

    assert response.status_code == 200
    body = response.json()
    assert body["source_model"] == "yolo26-seg"
    assert body["regions"][0]["label"] == "crack"


def test_predict_returns_unrecognized_below_threshold():
    app_state.router_model = FakeRouter(
        FakeRoutingDecision(router_class=None, raw_class="MVTec_screw", confidence=0.2, is_confident=False)
    )
    app_state.registry = {}
    app_state.anomalib_backend = FakeAnomalibBackend()
    app_state.yolo_backend = FakeYoloBackend()

    response = client.post("/predict", files={"file": ("test.png", _fake_image_bytes(), "image/png")})

    assert response.status_code == 200
    body = response.json()
    assert body["recognized"] is False
    assert body["router_raw_class"] == "MVTec_screw"
    assert body["router_confidence"] == 0.2


def test_predict_rejects_non_image_content_type():
    response = client.post("/predict", files={"file": ("test.txt", b"not an image", "text/plain")})
    assert response.status_code == 400


def test_predict_rejects_corrupt_image_bytes():
    app_state.router_model = FakeRouter(
        FakeRoutingDecision(router_class="MVTec_bottle", raw_class="MVTec_bottle", confidence=0.9, is_confident=True)
    )
    response = client.post("/predict", files={"file": ("test.png", b"not actually png data", "image/png")})
    assert response.status_code == 400


def test_predict_returns_500_when_checkpoint_missing(tmp_path):
    app_state.router_model = FakeRouter(
        FakeRoutingDecision(router_class="MVTec_bottle", raw_class="MVTec_bottle", confidence=0.9, is_confident=True)
    )
    app_state.registry = {
        "MVTec_bottle": model_registry.ModelEntry(backend="anomalib", checkpoint=tmp_path / "missing.ckpt")
    }
    app_state.anomalib_backend = FakeAnomalibBackend()
    app_state.yolo_backend = FakeYoloBackend()

    response = client.post("/predict", files={"file": ("test.png", _fake_image_bytes(), "image/png")})

    assert response.status_code == 500
