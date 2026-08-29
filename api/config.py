import os

from src.common.paths import (  # noqa: F401
    PROJECT_ROOT,
    ANOMALIB_MODELS_DIR,
    ANOMALIB_ONNX_DIR,
    ANOMALIB_MODEL_NAME,
    ROUTER_ONNX_PATH,
    DAGM_ONNX_PATH,
    KOLEKTOR_ONNX_PATH,
    MAGNETIC_TILE_ONNX_PATH,
    MVTEC_CATEGORIES,
    ROUTER_CLASSES,
    SUPERVISED_ROUTER_CLASSES,
)

ROUTER_CONFIDENCE_THRESHOLD = float(os.environ.get("VI_ROUTER_CONFIDENCE_THRESHOLD", "0.6"))
ANOMALY_REGION_THRESHOLD = float(os.environ.get("VI_ANOMALY_REGION_THRESHOLD", "0.5"))
ANOMALIB_DEVICE = os.environ.get("VI_ANOMALIB_DEVICE", "AUTO")
ANOMALIB_INFERENCER_CACHE_SIZE = int(os.environ.get("VI_ANOMALIB_INFERENCER_CACHE_SIZE", "3"))

# Phase 6: YOLO models (router + 3 specialists) can be served via Triton instead of
# local ONNX files. Anomalib stays local regardless -- see model_registry.py's
# build_triton_registry() docstring for why. Default TRITON_URL assumes the
# docker-compose service name "triton" on its internal port 8000 (not the host-
# remapped port -- container-to-container traffic uses the internal port).
USE_TRITON = os.environ.get("VI_USE_TRITON", "false").lower() in ("1", "true", "yes")
TRITON_URL = os.environ.get("VI_TRITON_URL", "http://triton:8000")
