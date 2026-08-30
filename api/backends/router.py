"""
API-side router wrapper. Loads the router once at startup and reuses it across
requests -- either a local ONNX file, or a remote Triton model via Ultralytics'
native Triton support (YOLO("http://host:port/model_name", task=...)), which
still runs all of YOLO's own pre/post-processing locally and only dispatches the
tensor compute to Triton, so no preprocessing is being hand-replicated here.

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
    def __init__(self, model_ref: Path | str | None, confidence_threshold: float):
        if model_ref is None or (isinstance(model_ref, str) and not model_ref.strip()):
            raise ValueError(
                "router model ref is empty or missing. Set VI_ROUTER_ONNX_PATH or enable VI_USE_TRITON=true."
            )

        is_triton = isinstance(model_ref, str) and model_ref.startswith(("http://", "https://", "triton://"))
        if not is_triton:
            model_path = Path(model_ref)
            if not model_path.exists():
                raise FileNotFoundError(
                    f"Router ONNX model not found at {model_ref}. "
                    f"Check VI_ROUTER_ONNX_PATH and ensure the export exists before starting the API."
                )

        # task="classify" is required for Triton remote models (can't be inferred
        # from a URL the way it is from a local .onnx/.pt's embedded metadata) --
        # harmless to always pass it for local loading too.
        self.model = YOLO(str(model_ref), task="classify")
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
