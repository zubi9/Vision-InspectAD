"""
Shared prediction schema across the Anomalib and YOLO26-seg backends.

Defined in Phase 1 so both pipelines could be written against a stable contract;
wired into the FastAPI layer here in Phase 3.
"""

from typing import Literal, Optional

from pydantic import BaseModel


class Region(BaseModel):
    """A single flagged region, normalized across backends.

    - Anomalib: derived from thresholding the anomaly heatmap into
      connected components (no native class label -> class is None).
    - YOLO26-seg: one entry per detected instance, with its class label.
    """
    bbox: tuple[float, float, float, float]  # x_min, y_min, x_max, y_max
    mask: Optional[list[list[float]]] = None  # polygon points, if available
    score: float
    label: Optional[str] = None


class PredictionResult(BaseModel):
    defect_detected: bool
    image_score: float  # overall anomaly/confidence score for the image
    regions: list[Region]
    source_model: Literal["anomalib", "yolo26-seg"]
    router_class: str  # e.g. "MVTec_bottle", "DAGM", "KolektorSDD2", "Magnetic_Tile"
    router_confidence: float
    model_version: Optional[str] = None
    heatmap_overlay_base64: Optional[str] = None  # base64-encoded PNG of the heatmap overlay


class UnrecognizedResult(BaseModel):
    """Returned instead of PredictionResult when the router isn't confident enough
    to dispatch. An honest "can't tell" rather than a forced, likely-wrong guess."""
    recognized: Literal[False] = False
    router_raw_class: str  # what the router said, even though confidence was too low
    router_confidence: float
    message: str = "Router confidence below threshold -- image not routed to any model."

