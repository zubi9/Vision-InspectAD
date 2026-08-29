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
    checkpoint: Path | str  # Path for local files, str (http://...) for a Triton model URL
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


def build_triton_registry(
    triton_url: str = "http://triton:8000",
    anomalib_model_name: str = paths.ANOMALIB_MODEL_NAME,
) -> dict[str, ModelEntry]:
    """YOLO/router entries point at Triton model URLs; Anomalib entries are
    UNCHANGED (still local ONNX via OpenVINOInferencer) -- Anomalib isn't migrated
    to Triton this iteration, since there's no equivalent to Ultralytics' native
    Triton support that would keep its pre/post-processing correct automatically,
    and hand-replicating Anomalib's exact normalization to talk to raw Triton
    tensors is a real correctness risk, not just extra work. See README for the
    full reasoning. Triton model names match router class names 1:1 by design
    (see serving/triton/generate_model_repository.py), so no separate mapping
    table is needed here.
    """
    registry = build_onnx_registry(anomalib_model_name=anomalib_model_name)
    for router_class in paths.SUPERVISED_ROUTER_CLASSES:
        registry[router_class] = ModelEntry(backend="yolo26-seg", checkpoint=f"{triton_url}/{router_class}")
    return registry


def resolve(registry: dict[str, ModelEntry], router_class: str) -> ModelEntry:
    if router_class == "UNKNOWN":
        raise ValueError("Router returned UNKNOWN -- no model to dispatch to.")
    if router_class not in registry:
        raise KeyError(f"Unknown router class '{router_class}'. Known: {sorted(registry)}")
    entry = registry[router_class]
    is_triton = isinstance(entry.checkpoint, str) and entry.checkpoint.startswith(("http://", "https://"))
    if not is_triton and not Path(entry.checkpoint).exists():
        raise FileNotFoundError(f"'{router_class}' -> {entry.checkpoint} does not exist.")
    return entry


if __name__ == "__main__":
    import sys

    mode = "checkpoint (training-time)"
    if "--onnx" in sys.argv:
        registry = build_onnx_registry()
        mode = "ONNX (serving, local)"
    elif "--triton" in sys.argv:
        registry = build_triton_registry()
        mode = "Triton (serving, YOLO models remote)"
    else:
        registry = build_registry()

    print(f"Registry view: {mode}\n")
    for name in paths.ROUTER_CLASSES:
        entry = registry[name]
        is_triton = isinstance(entry.checkpoint, str) and entry.checkpoint.startswith("http")
        exists = "TRITON-URL" if is_triton else ("OK" if Path(entry.checkpoint).exists() else "MISSING")
        print(f"[{exists}] {name:20s} -> {entry.backend:12s} {entry.checkpoint}")
