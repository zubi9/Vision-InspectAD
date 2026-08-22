#!/usr/bin/env python3

"""
Export VisionInspect PatchCore models trained with Anomalib to ONNX.

Expected structure:

models/
├── bottle/
│   └── ...
├── cable/
│   └── ...
├── capsule/
│   └── ...
└── ...

or checkpoints directly under models/.

The exporter automatically discovers .ckpt files.

Training configuration from train_all_categories.py:
    image_size = 256
    model = patchcore
    15 MVTec categories
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from anomalib.engine import Engine
from anomalib.deploy import ExportType
from anomalib.models import Patchcore


MVTEC_CATEGORIES = [
    "bottle",
    "cable",
    "capsule",
    "carpet",
    "grid",
    "hazelnut",
    "leather",
    "metal_nut",
    "pill",
    "screw",
    "tile",
    "toothbrush",
    "transistor",
    "wood",
    "zipper",
]


def parse_args():

    parser = argparse.ArgumentParser(
        description="Export VisionInspect PatchCore checkpoints to ONNX."
    )

    parser.add_argument(
        "--models-dir",
        type=Path,
        default=Path("./models"),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./models/onnx"),
    )

    parser.add_argument(
        "--image-size",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--categories",
        nargs="+",
        default=None,
        help="Specific MVTec categories to export.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    return parser.parse_args()


def find_checkpoints(models_dir: Path):

    checkpoints = sorted(
        models_dir.rglob("*.ckpt")
    )

    # Don't accidentally export previously generated artifacts.
    checkpoints = [
        p for p in checkpoints
        if "onnx" not in p.parts
    ]

    return checkpoints


def identify_category(checkpoint: Path):

    path_string = str(checkpoint).lower()

    matches = [
        category
        for category in MVTEC_CATEGORIES
        if category.lower() in path_string
    ]

    if len(matches) == 1:
        return matches[0]

    # If checkpoints are stored like:
    #
    # models/bottle/model.ckpt
    #
    # infer from parent directory.
    for parent in checkpoint.parents:

        if parent.name in MVTEC_CATEGORIES:
            return parent.name

    return None


def export_patchcore(
    checkpoint: Path,
    category: str,
    output_dir: Path,
    image_size: int,
    overwrite: bool,
):

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_file = (
        output_dir /
        f"patchcore_{category}.onnx"
    )

    if output_file.exists() and not overwrite:

        print(
            f"[SKIP] {category}: "
            f"{output_file} already exists"
        )

        return {
            "category": category,
            "checkpoint": str(checkpoint),
            "onnx": str(output_file),
            "status": "skipped",
        }

    print()
    print("=" * 70)
    print(f"Category   : {category}")
    print(f"Checkpoint : {checkpoint}")
    print(f"Input size : {image_size} x {image_size}")
    print(f"Output     : {output_file}")
    print("=" * 70)

    # ---------------------------------------------------------
    # IMPORTANT
    #
    # These parameters MUST match train.py.
    #
    # The training script supplied only exposes:
    #
    #     image_size=256
    #
    # It delegates PatchCore construction to train.py.
    #
    # Therefore don't silently invent backbone/layer settings here.
    # ---------------------------------------------------------

    model = Patchcore()

    engine = Engine()

    exported = engine.export(
        model=model,
        export_type=ExportType.ONNX,
        ckpt_path=str(checkpoint),
        export_root=str(output_dir),
        model_file_name=f"patchcore_{category}",
        input_size=(image_size, image_size),
    )

    print(
        f"[OK] {category} -> {exported}"
    )

    return {
        "category": category,
        "checkpoint": str(checkpoint),
        "onnx": str(exported),
        "status": "success",
    }


def check_dependencies():
    try:
        import onnx
        print(f"[OK] ONNX {onnx.__version__}")
    except ImportError:
        raise RuntimeError(
            "ONNX is not installed.\n"
            "Install with:\n"
            "    python -m pip install onnx"
        )

    try:
        import onnxruntime
        print(
            f"[OK] ONNX Runtime "
            f"{onnxruntime.__version__}"
        )
    except ImportError:
        raise RuntimeError(
            "ONNX Runtime is not installed.\n"
            "Install with:\n"
            "    python -m pip install onnxruntime"
        )

def main():

    args = parse_args()

    models_dir = args.models_dir.resolve()
    output_dir = args.output_dir.resolve()

    if not models_dir.exists():

        raise FileNotFoundError(
            f"Models directory not found: {models_dir}"
        )

    checkpoints = find_checkpoints(
        models_dir
    )

    if not checkpoints:

        raise RuntimeError(
            f"No .ckpt files found under {models_dir}"
        )

    print()
    print("VisionInspect PatchCore ONNX Export")
    print("=" * 70)
    print(f"Models : {models_dir}")
    print(f"Output : {output_dir}")
    print(f"Found  : {len(checkpoints)} checkpoints")

    selected = []

    for checkpoint in checkpoints:

        category = identify_category(
            checkpoint
        )

        if category is None:

            print(
                f"[WARN] Cannot identify MVTec category: "
                f"{checkpoint}"
            )

            continue

        if (
            args.categories is not None
            and category not in args.categories
        ):
            continue

        selected.append(
            (checkpoint, category)
        )

    if not selected:

        raise RuntimeError(
            "No matching PatchCore checkpoints found."
        )

    results = []

    for checkpoint, category in selected:

        try:

            result = export_patchcore(
                checkpoint=checkpoint,
                category=category,
                output_dir=output_dir,
                image_size=args.image_size,
                overwrite=args.overwrite,
            )

            results.append(result)

        except Exception as exc:

            print(
                f"[ERROR] {category}: {exc}"
            )

            results.append(
                {
                    "category": category,
                    "checkpoint": str(checkpoint),
                    "status": "failed",
                    "error": str(exc),
                }
            )

    summary = {
        "model": "PatchCore",
        "input_size": [
            args.image_size,
            args.image_size,
        ],
        "results": results,
    }

    summary_file = (
        output_dir /
        "patchcore_onnx_export_summary.json"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_file.write_text(
        json.dumps(
            summary,
            indent=2,
        )
    )

    successful = sum(
        x["status"] == "success"
        for x in results
    )

    failed = sum(
        x["status"] == "failed"
        for x in results
    )

    skipped = sum(
        x["status"] == "skipped"
        for x in results
    )

    print()
    print("=" * 70)
    print("EXPORT SUMMARY")
    print("=" * 70)
    print(f"Successful : {successful}")
    print(f"Skipped    : {skipped}")
    print(f"Failed     : {failed}")
    print(f"Summary    : {summary_file}")


if __name__ == "__main__":
    main()