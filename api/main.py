from contextlib import asynccontextmanager

from fastapi import FastAPI

from api import config
from api.backends.anomalib_backend import AnomalibBackend
from api.backends.router import RouterModel
from api.backends.yolo_backend import Yolo26SegBackend
from api.routes import router as api_router
from src.common import model_registry


class AppState:
    router_model: RouterModel
    registry: dict
    anomalib_backend: AnomalibBackend
    yolo_backend: Yolo26SegBackend


app_state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    router_ref = f"{config.TRITON_URL}/router" if config.USE_TRITON else config.ROUTER_ONNX_PATH
    print(f"[startup] loading router model ({'Triton' if config.USE_TRITON else 'local ONNX'})...")
    app_state.router_model = RouterModel(
        model_ref=router_ref,
        confidence_threshold=config.ROUTER_CONFIDENCE_THRESHOLD,
    )

    print(f"[startup] building model registry ({'Triton for YOLO models' if config.USE_TRITON else 'local ONNX'})...")
    if config.USE_TRITON:
        app_state.registry = model_registry.build_triton_registry(
            triton_url=config.TRITON_URL,
            anomalib_model_name=config.ANOMALIB_MODEL_NAME,
        )
    else:
        app_state.registry = model_registry.build_onnx_registry(
            anomalib_model_name=config.ANOMALIB_MODEL_NAME,
            dagm_checkpoint=config.DAGM_ONNX_PATH,
            kolektor_checkpoint=config.KOLEKTOR_ONNX_PATH,
            magnetic_tile_checkpoint=config.MAGNETIC_TILE_ONNX_PATH,
        )

    missing = []
    for name, entry in app_state.registry.items():
        is_triton = isinstance(entry.checkpoint, str) and entry.checkpoint.startswith("http")
        if not is_triton and not entry.checkpoint.exists():
            missing.append(name)
    if missing:
        print(f"[startup] WARNING: {len(missing)}/{len(app_state.registry)} registry entries "
              f"point at missing local checkpoints (will 500 if routed to): {missing}")

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
