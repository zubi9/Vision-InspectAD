"""
API configuration. Model paths default to where each training script's own
output convention places its ONNX export -- adjust via environment variables if
your actual export locations differ.
"""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Router: yolo_pipeline/train.py's default project="runs/classify", name="specialist_router"
# (see src/router_pipeline/README.md) -> weights land at runs/classify/specialist_router/weights/
ROUTER_ONNX_PATH = Path(
    os.environ.get(
        "ROUTER_ONNX_PATH",
        str(PROJECT_ROOT / "models" / "runs" / "classify" / "specialist_router" / "weights" / "best.onnx"),
    )
)

# Below this confidence, the router returns "unrecognized" rather than a forced guess.
# Matches the 0.6 threshold mentioned when the router was trained (100% accuracy at
# this threshold) -- override via env var if you want to tune it later.
ROUTER_CONFIDENCE_THRESHOLD = float(os.environ.get("ROUTER_CONFIDENCE_THRESHOLD", "0.6"))

ANOMALIB_MODELS_DIR = Path(
    os.environ.get("ANOMALIB_MODELS_DIR", str(PROJECT_ROOT / "models/onnx/weights"))
)
ANOMALIB_MODEL_NAME = os.environ.get("ANOMALIB_MODEL_NAME", "patchcore")

# yolo_pipeline/train.py's default output_dir="./models/yolo26seg" -> weights land at
# models/yolo26seg/weights/
COMBINED_YOLO_ONNX_PATH = Path(
    os.environ.get(
        "COMBINED_YOLO_ONNX_PATH",
        str(PROJECT_ROOT / "models" / "yolo26seg" / "weights" / "best.onnx"),
    )
)

# Anomaly score threshold (0-1) above which a pixel is considered part of a flagged
# region when converting Anomalib's heatmap into bounding boxes for the shared schema.
ANOMALY_REGION_THRESHOLD = float(os.environ.get("ANOMALY_REGION_THRESHOLD", "0.5"))

# Anomalib inference device -- OpenVINOInferencer accepts AUTO/CPU/GPU/NPU.
ANOMALIB_DEVICE = os.environ.get("ANOMALIB_DEVICE", "AUTO")

# Number of Anomalib inferencers to keep warm in memory at once (LRU). Each one is a
# full model, so this bounds memory use when many categories might get hit in a
# session; least-recently-used category gets evicted and reloaded on next request.
ANOMALIB_INFERENCER_CACHE_SIZE = int(os.environ.get("ANOMALIB_INFERENCER_CACHE_SIZE", "3"))

# declare the paths for piplines in the src/router_pipeline/prepare_data.py and src/router_pipeline/link_datasets.py
RAW_DATA_ROOT = Path(os.environ.get("DATA_ROOT", str(PROJECT_ROOT / "data/raw")))
CLS_ROUTER_DATA_ROOT = Path(os.environ.get("OUTPUT_ROOT", str(PROJECT_ROOT / "data/domain_router_ds")))