"""
Builds serving/triton/model_repository/ from the 19 ONNX exports this project
already produces (15 Anomalib categories + 3 YOLO26-seg specialists + router).

Deliberately uses Triton's auto-complete config feature rather than hand-specifying
input/output tensor names and shapes: this project has already hit real API/format
mismatches more than once when guessing at exact tensor conventions (Anomalib's
image_size handling, checkpoint loading, etc.), and Triton can read the actual
input/output signature straight from each ONNX graph. Each config.pbtxt here only
pins what Triton *can't* infer on its own: platform, batching, and CPU/GPU
placement.

GPU placement: since VRAM isn't the constraint here (confirmed for this deployment
target), every model defaults to KIND_GPU for lowest latency -- including the
Anomalib models, since PatchCore's nearest-neighbor search against its memory bank
is matrix-heavy and generally benefits from GPU too, not just the YOLO models.
Override per-model via --cpu-models if you want a specific model on CPU instead.

Usage:
    python generate_model_repository.py
    python generate_model_repository.py --cpu-models MVTec_screw MVTec_toothbrush
    python generate_model_repository.py --dynamic-batch-yolo   # only if your YOLO
                                                                # ONNX exports use dynamic=True
"""

import argparse
import shutil
from pathlib import Path

from src.common import paths

REPO_ROOT = Path(__file__).resolve().parent / "model_repository"

ANOMALIB_CONFIG_TEMPLATE = """\
platform: "onnxruntime_onnx"
max_batch_size: 0
instance_group [
  {{
    kind: {kind}
  }}
]
"""

# max_batch_size: 0 by default -- Ultralytics only exports ONNX with a dynamic batch
# axis if dynamic=True was passed at export time, which this script has no way to
# confirm. Guessing max_batch_size > 0 against a fixed-batch export would make
# Triton fail to load the model outright. If your exports do support dynamic batch,
# switch to YOLO_DYNAMIC_BATCH_CONFIG_TEMPLATE below for real throughput gains.
YOLO_CONFIG_TEMPLATE = """\
platform: "onnxruntime_onnx"
max_batch_size: 0
instance_group [
  {{
    kind: {kind}
  }}
]
"""

YOLO_DYNAMIC_BATCH_CONFIG_TEMPLATE = """\
platform: "onnxruntime_onnx"
max_batch_size: 8
instance_group [
  {{
    kind: {kind}
  }}
]
dynamic_batching {{
  max_queue_delay_microseconds: 5000
}}
"""


def _link_model(source: Path, dest_dir: Path) -> bool:
    if not source.exists():
        print(f"[SKIP] {source} not found -- has this model been exported to ONNX yet?")
        return False
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / "model.onnx"
    if dest.exists() or dest.is_symlink():
        dest.unlink()
    dest.symlink_to(source.resolve())
    return True


def build_repository(cpu_models: set[str], dynamic_batch_yolo: bool = False) -> dict:
    if REPO_ROOT.exists():
        shutil.rmtree(REPO_ROOT)
    REPO_ROOT.mkdir(parents=True)

    summary = {}

    for category in paths.MVTEC_CATEGORIES:
        model_name = f"MVTec_{category}"
        onnx_path = paths.find_anomalib_onnx(category) or (
            paths.ANOMALIB_ONNX_DIR / f"{paths.ANOMALIB_MODEL_NAME}_{category}.onnx"
        )
        model_dir = REPO_ROOT / model_name
        linked = _link_model(onnx_path, model_dir / "1")
        if linked:
            kind = "KIND_CPU" if model_name in cpu_models else "KIND_GPU"
            (model_dir / "config.pbtxt").write_text(ANOMALIB_CONFIG_TEMPLATE.format(kind=kind))
        summary[model_name] = linked

    specialists = {
        "DAGM": paths.DAGM_ONNX_PATH,
        "KolektorSDD2": paths.KOLEKTOR_ONNX_PATH,
        "Magnetic_Tile": paths.MAGNETIC_TILE_ONNX_PATH,
    }
    yolo_template = YOLO_DYNAMIC_BATCH_CONFIG_TEMPLATE if dynamic_batch_yolo else YOLO_CONFIG_TEMPLATE
    for model_name, onnx_path in specialists.items():
        model_dir = REPO_ROOT / model_name
        linked = _link_model(onnx_path, model_dir / "1")
        if linked:
            kind = "KIND_CPU" if model_name in cpu_models else "KIND_GPU"
            (model_dir / "config.pbtxt").write_text(yolo_template.format(kind=kind))
        summary[model_name] = linked

    model_dir = REPO_ROOT / "router"
    linked = _link_model(paths.ROUTER_ONNX_PATH, model_dir / "1")
    if linked:
        kind = "KIND_CPU" if "router" in cpu_models else "KIND_GPU"
        (model_dir / "config.pbtxt").write_text(yolo_template.format(kind=kind))
    summary["router"] = linked

    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cpu-models", nargs="*", default=[],
                         help="Model names (e.g. MVTec_screw, router) to pin to CPU instead of GPU")
    parser.add_argument("--dynamic-batch-yolo", action="store_true",
                         help="Enable dynamic batching for YOLO models -- only use if their ONNX "
                              "exports were done with dynamic=True, otherwise Triton will fail to load them")
    args = parser.parse_args()

    summary = build_repository(set(args.cpu_models), dynamic_batch_yolo=args.dynamic_batch_yolo)

    n_ok = sum(summary.values())
    print(f"\n{n_ok}/{len(summary)} models linked into {REPO_ROOT}")
    for name, ok in summary.items():
        print(f"  [{'OK' if ok else 'MISSING'}] {name}")
    if n_ok < len(summary):
        print("\nMissing models won't be served -- export them to ONNX first, then re-run this script.")


if __name__ == "__main__":
    main()
