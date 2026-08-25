import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _p(var: str, default_relative: str) -> Path:
    return Path(os.environ.get(var, str(PROJECT_ROOT / default_relative)))


# --- Data: raw inputs ---
MVTEC_RAW_ROOT = _p("VI_MVTEC_RAW_ROOT", "data/raw")
SUPERVISED_RAW_ROOT = _p("VI_SUPERVISED_RAW_ROOT", "data/raw_supervised")
DAGM_RAW_ROOT = SUPERVISED_RAW_ROOT / "dagm"
MAGNETIC_TILES_RAW_ROOT = SUPERVISED_RAW_ROOT / "magnetic_tiles"
KOLEKTOR_SDD2_RAW_ROOT = SUPERVISED_RAW_ROOT / "kolektor_sdd2"

# --- Data: router pipeline ---
ROUTER_SOURCE_ROOT = _p("VI_ROUTER_SOURCE_ROOT", "data/router_source")
DOMAIN_ROUTER_DATASET_ROOT = _p("VI_DOMAIN_ROUTER_DS_ROOT", "data/domain_router_ds")

# --- Data: YOLO segmentation datasets, one merged dataset per specialist ---
YOLO_SEG_DATASET_ROOT = _p("VI_YOLO_SEG_DS_ROOT", "data/yolo-seg-ds")
DAGM_YOLO_DATA_YAML = YOLO_SEG_DATASET_ROOT / "dagm" / "data.yaml"
KOLEKTOR_YOLO_DATA_YAML = YOLO_SEG_DATASET_ROOT / "kolektor" / "data.yaml"
MAGNETIC_TILE_YOLO_DATA_YAML = YOLO_SEG_DATASET_ROOT / "magnetic_tile" / "data.yaml"

# --- Models: Anomalib (flat layout -- models/patchcore_<category>.ckpt) ---
ANOMALIB_MODELS_DIR = _p("VI_ANOMALIB_MODELS_DIR", "models")
ANOMALIB_ONNX_DIR = _p("VI_ANOMALIB_ONNX_DIR", "models/onnx")
ANOMALIB_MODEL_NAME = os.environ.get("VI_ANOMALIB_MODEL_NAME", "patchcore")

# --- Models: router ---
ROUTER_RUN_DIR = _p("VI_ROUTER_RUN_DIR", "models/runs/classify")
ROUTER_RUN_NAME = os.environ.get("VI_ROUTER_RUN_NAME", "specialist_router")
ROUTER_CHECKPOINT_PT = ROUTER_RUN_DIR / ROUTER_RUN_NAME / "weights" / "best.pt"
ROUTER_ONNX_PATH = _p("VI_ROUTER_ONNX_PATH", "models/runs/classify/specialist_router/weights/best.onnx")

# --- Models: 3 YOLO26-seg specialists, one dir each ---
DAGM_MODEL_DIR = _p("VI_DAGM_MODEL_DIR", "models/dagm")
KOLEKTOR_MODEL_DIR = _p("VI_KOLEKTOR_MODEL_DIR", "models/kolektor")
MAGNETIC_TILE_MODEL_DIR = _p("VI_MAGNETIC_TILE_MODEL_DIR", "models/magnetic_tile")

DAGM_CHECKPOINT_PT = DAGM_MODEL_DIR / "weights" / "best.pt"
KOLEKTOR_CHECKPOINT_PT = KOLEKTOR_MODEL_DIR / "weights" / "best.pt"
MAGNETIC_TILE_CHECKPOINT_PT = MAGNETIC_TILE_MODEL_DIR / "weights" / "best.pt"

DAGM_ONNX_PATH = _p("VI_DAGM_ONNX_PATH", "models/dagm/weights/best.onnx")
KOLEKTOR_ONNX_PATH = _p("VI_KOLEKTOR_ONNX_PATH", "models/kolektor/weights/best.onnx")
MAGNETIC_TILE_ONNX_PATH = _p("VI_MAGNETIC_TILE_ONNX_PATH", "models/magnetic_tile/weights/best.onnx")

# --- Experiment tracking ---
MLFLOW_TRACKING_URI = os.environ.get(
    "VI_MLFLOW_TRACKING_URI", f"sqlite:///{PROJECT_ROOT / 'experiments' / 'mlflow.db'}"
)

# --- Class taxonomy: single source of truth, imported everywhere else ---
MVTEC_CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid", "hazelnut",
    "leather", "metal_nut", "pill", "screw", "tile", "toothbrush",
    "transistor", "wood", "zipper",
]
SUPERVISED_ROUTER_CLASSES = ["DAGM", "KolektorSDD2", "Magnetic_Tile"]
ROUTER_CLASSES = SUPERVISED_ROUTER_CLASSES + [f"MVTec_{c}" for c in MVTEC_CATEGORIES]


def find_anomalib_onnx(category: str, model_name: str = ANOMALIB_MODEL_NAME) -> Path | None:
    """Anomalib's exporter nests ONNX output in a way that's shifted across versions
    already in this project (see model_registry.py history) -- search for the file
    by name instead of assuming an exact nested path, so version differences in the
    exporter's folder structure can't silently break serving."""
    matches = list(ANOMALIB_ONNX_DIR.rglob(f"{model_name}_{category}.onnx"))
    return matches[0] if matches else None
