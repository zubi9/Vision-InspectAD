"""
Anomalib inference backend for the API.

Uses anomalib.deploy.OpenVINOInferencer, which -- despite the name -- accepts raw
.onnx files directly (not just OpenVINO IR .xml/.bin), and handles preprocessing
(resize/normalize) and postprocessing consistently with how the model was trained.
This is deliberately NOT a hand-rolled onnxruntime.InferenceSession: Anomalib owns
its own pre/post-processing pipeline, and reimplementing it here risks silently
mismatching training-time normalization -- exactly the kind of version-specific
mismatch this project has already hit more than once with the Anomalib API.

Requires `openvino` installed (pip install openvino), separate from onnxruntime
(which the YOLO backend uses instead, via Ultralytics).
"""

import base64
from collections import OrderedDict
from pathlib import Path
import io

import cv2
import numpy as np
from anomalib.deploy import OpenVINOInferencer
from PIL import Image

from src.common.schemas import Region


def _get_field(pred, *candidate_names, required=True):
    """Anomalib's prediction result attribute names have shifted across versions
    (see this project's earlier image_size / checkpoint-loading API changes) --
    try each candidate name and fail loudly with a clear message if none exist,
    rather than silently returning the wrong thing."""
    for name in candidate_names:
        if hasattr(pred, name):
            return getattr(pred, name)
    if required:
        raise AttributeError(
            f"None of {candidate_names} found on prediction result "
            f"(available: {[a for a in dir(pred) if not a.startswith('_')]}). "
            f"Anomalib's result object shape may differ from what this backend expects -- "
            f"check the installed anomalib version's OpenVINOInferencer.predict() return type."
        )
    return None


def _scalar(value, name: str) -> float:
    """Convert a scalar-like NumPy/PyTorch value to a Python float."""
    array = np.asarray(value).squeeze()
    if array.size != 1:
        raise ValueError(f"{name} must contain exactly one value, got shape {array.shape}")
    return float(array.item())


class AnomalibBackend:
    """LRU-caches inferencers across categories -- holding all 15 in memory at once
    is wasteful, but reloading from disk on every single request is slow. Whichever
    category hasn't been used most recently gets evicted when the cache is full."""

    def __init__(self, device: str = "AUTO", cache_size: int = 3, region_threshold: float = 0.5):
        self.device = device
        self.cache_size = cache_size
        self.region_threshold = region_threshold
        self._cache: OrderedDict[Path, OpenVINOInferencer] = OrderedDict()

    def _get_inferencer(self, onnx_path: Path) -> OpenVINOInferencer:
        if onnx_path in self._cache:
            self._cache.move_to_end(onnx_path)
            return self._cache[onnx_path]

        if not onnx_path.exists():
            raise FileNotFoundError(f"Anomalib ONNX model not found at {onnx_path}")

        inferencer = OpenVINOInferencer(path=onnx_path, device=self.device)
        self._cache[onnx_path] = inferencer
        if len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)  # evict least-recently-used
        return inferencer

    def _prepare_image(
        self,
        image: Image.Image,
        inferencer: OpenVINOInferencer,
    ) -> np.ndarray:
        """Resize the image using static dimensions from the model input shape."""
        partial_shape = inferencer.input_blob.partial_shape

        if len(partial_shape) != 4:
            raise RuntimeError(
                f"Expected a 4D NCHW model input, got shape {partial_shape}"
            )

        _, channels_dim, height_dim, width_dim = partial_shape

        channels = (
            channels_dim.get_length()
            if channels_dim.is_static
            else 3
        )
        height = (
            height_dim.get_length()
            if height_dim.is_static
            else image.height
        )
        width = (
            width_dim.get_length()
            if width_dim.is_static
            else image.width
        )

        if channels != 3:
            raise RuntimeError(
                f"Expected a 3-channel model input, got shape {partial_shape}"
            )

        resized = image.convert("RGB").resize(
            (width, height),
            Image.Resampling.BILINEAR,
        )
        return np.asarray(resized)

    def _heatmap_to_regions(self, anomaly_map: np.ndarray) -> list[Region]:
        """Threshold the anomaly heatmap and extract bounding boxes via connected
        components -- same underlying technique as converting MVTec's ground-truth
        masks into bounding boxes earlier in this project, applied here to a
        predicted heatmap instead of a label."""
        if anomaly_map.ndim > 2:
            anomaly_map = anomaly_map.squeeze()

        normalized = anomaly_map
        if normalized.max() > 1.0 or normalized.min() < 0.0:
            span = normalized.max() - normalized.min()
            normalized = (normalized - normalized.min()) / span if span > 0 else normalized

        mask = (normalized >= self.region_threshold).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        regions = []
        for contour in contours:
            if cv2.contourArea(contour) < 4:  # drop noise-sized specks
                continue
            x, y, w, h = cv2.boundingRect(contour)
            region_score = float(normalized[y:y + h, x:x + w].max())
            regions.append(Region(
                bbox=(float(x), float(y), float(x + w), float(y + h)),
                score=region_score,
                label=None,  # Anomalib has no class label -- it's one-class per category
            ))
        return regions


    def _overlay_base64(self, image: Image.Image, anomaly_map: np.ndarray) -> str:
        heatmap = anomaly_map.squeeze()
        span = heatmap.max() - heatmap.min()
        heatmap = (heatmap - heatmap.min()) / span if span > 0 else heatmap
        heatmap_u8 = (heatmap * 255).astype(np.uint8)
        heatmap_color = cv2.applyColorMap(heatmap_u8, cv2.COLORMAP_JET)
        heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

        base = np.array(image.resize((heatmap.shape[1], heatmap.shape[0])))
        overlay = cv2.addWeighted(base, 0.6, heatmap_color, 0.4, 0)

        buf = io.BytesIO()
        Image.fromarray(overlay).save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")
    

    def predict(self, image: Image.Image, onnx_path: Path) -> dict:
        inferencer = self._get_inferencer(onnx_path)
        input_image = self._prepare_image(image, inferencer)
        pred = inferencer.predict(image=input_image)

        pred_score = _scalar(_get_field(pred, "pred_score"), "pred_score")

        anomaly_map = np.asarray(
            _get_field(pred, "anomaly_map")
        )
        anomaly_map = np.squeeze(anomaly_map)

        pred_label = _get_field(pred, "pred_label", required=False)
        if pred_label is not None:
            pred_label = bool(np.asarray(pred_label).squeeze().item())

        defect_detected = (
            pred_label
            if pred_label is not None
            else pred_score >= self.region_threshold
        )

        regions = self._heatmap_to_regions(anomaly_map)
        overlay_b64 = self._overlay_base64(image, anomaly_map)

        return {
            "defect_detected": defect_detected,
            "image_score": pred_score,
            "regions": regions,
            "heatmap_overlay_base64": overlay_b64,
        }