"""
Maps a router class to which backend handles it and where its checkpoint lives.
Path defaults come from src/common/paths.py -- change values there, not here.

18 router classes: MVTec_<category> x 15, plus DAGM / KolektorSDD2 / Magnetic_Tile,
each of the latter three now its own specialist YOLO26-seg model (not one combined
model -- that was an earlier design, superseded).
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from src.common import paths

Backend = Literal["anomalib", "yolo26-seg"]


@dataclass
class ModelEntry:
    backend: Backend
    checkpoint: Path
    model_class: str | None = None


def build_registry(
    anomalib_models_dir: Path = paths.ANOMALIB_MODELS_DIR,
    anomalib_model_name: str = paths.ANOMALIB_MODEL_NAME,
    dagm_checkpoint: Path = paths.DAGM_CHECKPOINT_PT,
    kolektor_checkpoint: Path = paths.KOLEKTOR_CHECKPOINT_PT,
    magnetic_tile_checkpoint: Path = paths.MAGNETIC_TILE_CHECKPOINT_PT,
) -> dict[str, ModelEntry]:
    registry: dict[str, ModelEntry] = {}

    for category in paths.MVTEC_CATEGORIES:
        checkpoint = anomalib_models_dir / f"{anomalib_model_name}_{category}.ckpt"
        registry[f"MVTec_{category}"] = ModelEntry(
            backend="anomalib", checkpoint=checkpoint, model_class=anomalib_model_name
        )

    registry["DAGM"] = ModelEntry(backend="yolo26-seg", checkpoint=dagm_checkpoint)
    registry["KolektorSDD2"] = ModelEntry(backend="yolo26-seg", checkpoint=kolektor_checkpoint)
    registry["Magnetic_Tile"] = ModelEntry(backend="yolo26-seg", checkpoint=magnetic_tile_checkpoint)

    return registry


def build_onnx_registry(
    anomalib_model_name: str = paths.ANOMALIB_MODEL_NAME,
    dagm_checkpoint: Path = paths.DAGM_ONNX_PATH,
    kolektor_checkpoint: Path = paths.KOLEKTOR_ONNX_PATH,
    magnetic_tile_checkpoint: Path = paths.MAGNETIC_TILE_ONNX_PATH,
) -> dict[str, ModelEntry]:
    """Same mapping, pointing at ONNX exports for serving. Anomalib entries are
    resolved via paths.find_anomalib_onnx() (rglob-based) rather than a fixed
    nested path, since Anomalib's exporter folder structure has already changed
    across versions during this project."""
    registry: dict[str, ModelEntry] = {}

    for category in paths.MVTEC_CATEGORIES:
        found = paths.find_anomalib_onnx(category, anomalib_model_name)
        checkpoint = found or (paths.ANOMALIB_ONNX_DIR / f"{anomalib_model_name}_{category}.onnx")
        registry[f"MVTec_{category}"] = ModelEntry(
            backend="anomalib", checkpoint=checkpoint, model_class=anomalib_model_name
        )

    registry["DAGM"] = ModelEntry(backend="yolo26-seg", checkpoint=dagm_checkpoint)
    registry["KolektorSDD2"] = ModelEntry(backend="yolo26-seg", checkpoint=kolektor_checkpoint)
    registry["Magnetic_Tile"] = ModelEntry(backend="yolo26-seg", checkpoint=magnetic_tile_checkpoint)

    return registry


def resolve(registry: dict[str, ModelEntry], router_class: str) -> ModelEntry:
    if router_class == "UNKNOWN":
        raise ValueError("Router returned UNKNOWN -- no model to dispatch to.")
    if router_class not in registry:
        raise KeyError(f"Unknown router class '{router_class}'. Known: {sorted(registry)}")
    entry = registry[router_class]
    if not entry.checkpoint.exists():
        raise FileNotFoundError(f"'{router_class}' -> {entry.checkpoint} does not exist.")
    return entry


if __name__ == "__main__":
    import sys

    use_onnx = "--onnx" in sys.argv
    registry = build_onnx_registry() if use_onnx else build_registry()
    label = "ONNX (serving)" if use_onnx else "checkpoint (training-time)"
    print(f"Registry view: {label}\n")
    for name in paths.ROUTER_CLASSES:
        entry = registry[name]
        exists = "OK" if entry.checkpoint.exists() else "MISSING"
        print(f"[{exists}] {name:20s} -> {entry.backend:12s} {entry.checkpoint}")
