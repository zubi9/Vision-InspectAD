"""
API-side router wrapper. Loads the ONNX-exported YOLO26n-cls router once at
startup and reuses it across requests.

Same confidence-gating philosophy as src/router_pipeline/inference.py: below
threshold, return None rather than a forced guess, since a misroute sends the
image to a completely unrelated specialist model.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from ultralytics import YOLO


@dataclass
class RoutingDecision:
    router_class: str | None  # None if below confidence threshold ("unrecognized")
    raw_class: str  # what the router said, even if below threshold
    confidence: float
    is_confident: bool


class RouterModel:
    def __init__(self, onnx_path: Path, confidence_threshold: float):
        if not onnx_path.exists():
            raise FileNotFoundError(
                f"Router ONNX model not found at {onnx_path}. "
                f"Check ROUTER_ONNX_PATH in api/config.py or the env var of the same name."
            )
        # Ultralytics' YOLO() loads .onnx transparently, running the same
        # pre/post-processing pipeline as it would for a .pt checkpoint.
        self.model = YOLO(str(onnx_path))
        self.confidence_threshold = confidence_threshold

    def route(self, image: Image.Image) -> RoutingDecision:
        result = self.model.predict(source=np.array(image), verbose=False)[0]
        top1_idx = int(result.probs.top1)
        confidence = float(result.probs.top1conf)
        raw_class = result.names[top1_idx]

        is_confident = confidence >= self.confidence_threshold
        return RoutingDecision(
            router_class=raw_class if is_confident else None,
            raw_class=raw_class,
            confidence=confidence,
            is_confident=is_confident,
        )
