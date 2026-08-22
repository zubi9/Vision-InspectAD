"""
API routes.

/predict flow:
    1. Router decides which of 18 classes the image belongs to.
    2. Below confidence threshold -> UnrecognizedResult (honest failure, not a guess).
    3. Otherwise, model_registry.resolve() finds the right backend + checkpoint.
    4. That backend runs inference and the result is normalized into PredictionResult.
"""

import io

from fastapi import APIRouter, File, HTTPException, UploadFile
from PIL import Image

from src.common import model_registry
from src.common.schemas import PredictionResult, UnrecognizedResult

router = APIRouter()


@router.get("/health")
def health():
    return {"status": "ok"}


@router.post("/predict", response_model=None)
async def predict(file: UploadFile = File(...)):
    from api.main import app_state  # deferred import avoids a circular import with main.py

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail=f"Expected an image, got content-type={file.content_type}")

    raw_bytes = await file.read()
    try:
        image = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not decode image: {e}")

    decision = app_state.router_model.route(image)

    if not decision.is_confident:
        return UnrecognizedResult(
            router_raw_class=decision.raw_class,
            router_confidence=decision.confidence,
        )

    try:
        entry = model_registry.resolve(app_state.registry, decision.router_class)
    except (KeyError, FileNotFoundError) as e:
        raise HTTPException(status_code=500, detail=str(e))

    if entry.backend == "anomalib":
        raw = app_state.anomalib_backend.predict(image, entry.checkpoint)
        source_model = "anomalib"
    else:
        raw = app_state.yolo_backend.predict(image, entry.checkpoint)
        source_model = "yolo26-seg"

    return PredictionResult(
        defect_detected=raw["defect_detected"],
        image_score=raw["image_score"],
        regions=raw["regions"],
        source_model=source_model,
        router_class=decision.router_class,
        router_confidence=decision.confidence,
        heatmap_overlay_base64=raw.get("heatmap_overlay_base64")
    )
