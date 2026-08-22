"""
FastAPI app entrypoint.

Run with:
    uvicorn api.main:app --reload

All models (router + Anomalib backend + YOLO26-seg backend) load once at startup
via the lifespan handler below, not per-request -- per-request loading would make
every single prediction pay full model-load latency.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from api import config
from api.backends.anomalib_backend import AnomalibBackend
from api.backends.router import RouterModel
from api.backends.yolo_backend import Yolo26SegBackend
from api.routes import router as api_router
from src.common import model_registry
from pathlib import Path


class AppState:
    """Simple namespace holding everything loaded at startup. Route handlers reach
    into this directly (see api/routes.py's deferred import of app_state) -- typing
    it explicitly here makes what's available at request time easy to see at a glance."""
    router_model: RouterModel
    registry: dict
    anomalib_backend: AnomalibBackend
    yolo_backend: Yolo26SegBackend


app_state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[startup] loading router model...")
    app_state.router_model = RouterModel(
        onnx_path=config.ROUTER_ONNX_PATH,
        confidence_threshold=config.ROUTER_CONFIDENCE_THRESHOLD,
    )

    print("[startup] building model registry...")
    app_state.registry = model_registry.build_onnx_registry(
        anomalib_models_dir=config.ANOMALIB_MODELS_DIR,
        anomalib_model_name=config.ANOMALIB_MODEL_NAME,
        combined_yolo_checkpoint=Path(
            "/home/zubair/AI Development Directory/visioninspect_v112/models/yolo26seg/weights/best.onnx"
        ),
        dagm_checkpoint=Path(
            "/home/zubair/AI Development Directory/visioninspect_v112/models/dagm/weights/best.onnx"
        ),
        kolektor_checkpoint=Path(
            "/home/zubair/AI Development Directory/visioninspect_v112/models/kolektor/weights/best.onnx"
        ),
        magnetic_tile_checkpoint=Path(
            "/home/zubair/AI Development Directory/visioninspect_v112/models/magnetic_tile/weights/best.onnx"
        ),
    )
    missing = [name for name, entry in app_state.registry.items() if not entry.checkpoint.exists()]
    if missing:
        print(f"[startup] WARNING: {len(missing)}/{len(app_state.registry)} registry entries "
              f"point at missing checkpoints (will 500 if routed to): {missing}")

    app_state.anomalib_backend = AnomalibBackend(
        device=config.ANOMALIB_DEVICE,
        cache_size=config.ANOMALIB_INFERENCER_CACHE_SIZE,
        region_threshold=config.ANOMALY_REGION_THRESHOLD,
    )
    app_state.yolo_backend = Yolo26SegBackend()

    print("[startup] ready.")
    yield
    print("[shutdown] done.")


app = FastAPI(title="VisionInspect API", lifespan=lifespan)
app.include_router(api_router)
