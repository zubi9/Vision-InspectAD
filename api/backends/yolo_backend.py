"""
YOLO26-seg inference backend for the API.

Ultralytics' YOLO() class loads .onnx transparently and runs the same
pre-processing (letterbox resize) and post-processing (NMS, mask decoding) it would
for a .pt checkpoint -- so this is a thin wrapper, not a reimplementation. Handles
all three supervised datasets (DAGM, Magnetic Tiles, Kolektor SDD2) since they're
trained as one combined model with one shared class head.
"""

from pathlib import Path

import numpy as np
from PIL import Image
from ultralytics import YOLO

from src.common.schemas import Region


class Yolo26SegBackend:
    def __init__(self):
        self._model: YOLO | None = None
        self._loaded_path: Path | None = None

    def _get_model(self, onnx_path: Path) -> YOLO:
        if self._model is not None and self._loaded_path == onnx_path:
            return self._model
        if not onnx_path.exists():
            raise FileNotFoundError(f"YOLO26-seg ONNX model not found at {onnx_path}")
        self._model = YOLO(str(onnx_path))
        self._loaded_path = onnx_path
        return self._model

    def predict(self, image: Image.Image, onnx_path: Path) -> dict:
        model = self._get_model(onnx_path)
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
