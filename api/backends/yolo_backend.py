"""
YOLO26-seg inference backend for the API.

Ultralytics' YOLO() class loads .onnx transparently and runs the same
pre-processing (letterbox resize) and post-processing (NMS, mask decoding) it would
for a .pt checkpoint -- so this is a thin wrapper, not a reimplementation. It also
natively supports a Triton Inference Server URL (YOLO("http://host:port/model",
task=...)) via the exact same code path, still handling pre/post-processing
locally and only dispatching the tensor compute to Triton -- so switching between
local ONNX and Triton is just a different string/Path passed to predict(), nothing
in this class needs to know which one it's talking to.

Each of the three supervised datasets (DAGM, Magnetic Tiles, Kolektor SDD2) is a
separate specialist model (not one combined model -- see PROVENANCE.md).
"""

from pathlib import Path

import numpy as np
from PIL import Image
from ultralytics import YOLO

from src.common.schemas import Region


def _is_triton_ref(model_ref: Path | str) -> bool:
    return isinstance(model_ref, str) and model_ref.startswith(("http://", "https://", "triton://"))


class Yolo26SegBackend:
    def __init__(self):
        self._model: YOLO | None = None
        self._loaded_ref: Path | str | None = None

    def _get_model(self, model_ref: Path | str) -> YOLO:
        if self._model is not None and self._loaded_ref == model_ref:
            return self._model
        if not _is_triton_ref(model_ref) and not Path(model_ref).exists():
            raise FileNotFoundError(f"YOLO26-seg model not found at {model_ref}")
        # task="segment" required for Triton remote models; harmless for local loads.
        self._model = YOLO(str(model_ref), task="segment")
        self._loaded_ref = model_ref
        return self._model

    def predict(self, image: Image.Image, model_ref: Path | str) -> dict:
        model = self._get_model(model_ref)
        result = model.predict(source=np.array(image), verbose=False)[0]

        regions: list[Region] = []
        if result.boxes is not None:
            boxes_xyxy = result.boxes.xyxy.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy().astype(int)

            masks_xy = None
            if result.masks is not None:
                masks_xy = result.masks.xy  # list of polygon point arrays, one per instance

            for i in range(len(boxes_xyxy)):
                x1, y1, x2, y2 = boxes_xyxy[i].tolist()
                label = result.names[class_ids[i]]
                polygon = None
                if masks_xy is not None and i < len(masks_xy):
                    polygon = masks_xy[i].tolist()  # [[x, y], [x, y], ...]

                regions.append(Region(
                    bbox=(x1, y1, x2, y2),
                    mask=polygon,
                    score=float(scores[i]),
                    label=label,
                ))

        image_score = float(max((r.score for r in regions), default=0.0))
        return {
            "defect_detected": len(regions) > 0,
            "image_score": image_score,
            "regions": regions,
        }
