"""
Central dispatch registry: maps a router class name to which backend handles it
and where that backend's checkpoint lives.

This is the single source of truth for "given a router decision, what do I actually
call" -- used by the inference API (Phase 3) and by any script that needs to resolve
a router class to a runnable model. Keeping this in one place means adding a new
MVTec category, splitting the supervised datasets into separate models, or swapping
a checkpoint is a one-line change here, not a hunt through the API layer.

18 router classes total (matches src/router_pipeline/prepare_data.py's ROUTER_CLASSES):
    - MVTec_bottle, MVTec_cable, ... (15 MVTec AD categories, each its own Anomalib checkpoint)
    - DAGM, KolektorSDD2, Magnetic_Tile (3 supervised-dataset classes)

The three supervised classes are separate router outputs (for diagnostics/future
flexibility -- see src/router_pipeline/README.md's "Architecture boundary" note) but
currently all resolve to the SAME combined YOLO26-seg checkpoint, since DAGM +
Magnetic Tiles + Kolektor SDD2 are trained together as one model. If they're ever
split into separate specialist models, only the three yolo_checkpoint arguments
below need to change.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

MVTEC_CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid", "hazelnut",
    "leather", "metal_nut", "pill", "screw", "tile", "toothbrush",
    "transistor", "wood", "zipper",
]

SUPERVISED_ROUTER_CLASSES = ["DAGM", "KolektorSDD2", "Magnetic_Tile"]

ROUTER_CLASSES = SUPERVISED_ROUTER_CLASSES + [f"MVTec_{c}" for c in MVTEC_CATEGORIES]

Backend = Literal["anomalib", "yolo26-seg"]


@dataclass
class ModelEntry:
    backend: Backend
    checkpoint: Path
    model_class: str | None = None  # e.g. "patchcore" -- only meaningful for anomalib


def build_registry(
    anomalib_models_dir: Path = Path("./models"),
    anomalib_model_name: str = "patchcore",
    dagm_checkpoint: Path | None = None,
    kolektor_checkpoint: Path | None = None,
    magnetic_tile_checkpoint: Path | None = None,
    combined_yolo_checkpoint: Path = Path("./models/yolo26seg/weights/best.pt"),
) -> dict[str, ModelEntry]:
    """Build the router-class -> ModelEntry mapping.

    dagm_checkpoint / kolektor_checkpoint / magnetic_tile_checkpoint default to
    combined_yolo_checkpoint (the current one-model-for-all-three setup). Pass any
    of them explicitly to point that class at a separate specialist model instead,
    without touching the other two.
    """
    registry: dict[str, ModelEntry] = {}

    for category in MVTEC_CATEGORIES:
        checkpoint = anomalib_models_dir / f"{anomalib_model_name}_{category}.ckpt"
        registry[f"MVTec_{category}"] = ModelEntry(
            backend="anomalib",
            checkpoint=checkpoint,
            model_class=anomalib_model_name,
        )

    registry["DAGM"] = ModelEntry(
        backend="yolo26-seg",
        checkpoint=dagm_checkpoint or combined_yolo_checkpoint,
    )
    registry["KolektorSDD2"] = ModelEntry(
        backend="yolo26-seg",
        checkpoint=kolektor_checkpoint or combined_yolo_checkpoint,
    )
    registry["Magnetic_Tile"] = ModelEntry(
        backend="yolo26-seg",
        checkpoint=magnetic_tile_checkpoint or combined_yolo_checkpoint,
    )

    return registry


def build_onnx_registry(
    anomalib_models_dir: Path = Path("./models/onnx/weights"),
    anomalib_model_name: str = "patchcore",
    dagm_checkpoint: Path | None = None,
    kolektor_checkpoint: Path | None = None,
    magnetic_tile_checkpoint: Path | None = None,
    combined_yolo_checkpoint: Path = Path("./models/yolo26seg/weights/best.onnx"),
) -> dict[str, ModelEntry]:
    """Same mapping as build_registry(), but pointing at .onnx exports instead of
    the .ckpt/.pt training-time checkpoints -- this is what the inference API
    (Phase 3) actually loads, since serving is ONNX Runtime-based.

    Assumes each Anomalib category's ONNX export sits alongside its .ckpt with the
    extension swapped (models/patchcore_<category>.onnx) and the combined YOLO
    model's ONNX export sits next to its best.pt (models/yolo26seg/weights/best.onnx,
    matching yolo_pipeline/train.py's output layout). Adjust the arguments here if
    your actual export locations differ.
    """
    registry: dict[str, ModelEntry] = {}

    for category in MVTEC_CATEGORIES:
        checkpoint = anomalib_models_dir / f"{anomalib_model_name}_{category}.onnx"
        registry[f"MVTec_{category}"] = ModelEntry(
            backend="anomalib",
            checkpoint=checkpoint,
            model_class=anomalib_model_name,
        )

    registry["DAGM"] = ModelEntry(
        backend="yolo26-seg",
        checkpoint=dagm_checkpoint or combined_yolo_checkpoint,
    )
    registry["KolektorSDD2"] = ModelEntry(
        backend="yolo26-seg",
        checkpoint=kolektor_checkpoint or combined_yolo_checkpoint,
    )
    registry["Magnetic_Tile"] = ModelEntry(
        backend="yolo26-seg",
        checkpoint=magnetic_tile_checkpoint or combined_yolo_checkpoint,
    )

    return registry


def resolve(registry: dict[str, ModelEntry], router_class: str) -> ModelEntry:
    if router_class == "UNKNOWN":
        raise ValueError(
            "Router returned UNKNOWN (below its confidence threshold) -- there is no "
            "model to dispatch to. Handle this case before calling resolve()."
        )
    if router_class not in registry:
        raise KeyError(
            f"Router predicted class '{router_class}' with no matching registry entry. "
            f"Known classes: {sorted(registry)}"
        )
    entry = registry[router_class]
    if not entry.checkpoint.exists():
        raise FileNotFoundError(
            f"Registry points '{router_class}' -> {entry.checkpoint}, but that checkpoint "
            f"doesn't exist. Has this model been trained yet?"
        )
    return entry


if __name__ == "__main__":
    import sys

    use_onnx = "--onnx" in sys.argv
    registry = build_onnx_registry() if use_onnx else build_registry()
    label = "ONNX (Phase 3 serving)" if use_onnx else "checkpoint (training-time)"
    print(f"Registry view: {label}\n")
    for name in ROUTER_CLASSES:
        entry = registry[name]
        exists = "OK" if entry.checkpoint.exists() else "MISSING"
        print(f"[{exists}] {name:20s} -> {entry.backend:12s} {entry.checkpoint}")
