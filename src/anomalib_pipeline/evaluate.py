"""
Evaluate a trained Anomalib checkpoint and produce a metrics report.

Anomaly-detection-appropriate metrics (NOT precision/recall/mAP, which
assume a bounding-box detection task):

  - Image-level AUROC : how well the model separates normal vs anomalous
                         *images* by their overall anomaly score.
  - Pixel-level AUROC : how well the predicted anomaly heatmap separates
                         normal vs anomalous *pixels* against the ground
                         truth masks.
  - PRO score          : Per-Region Overlap -- rewards detecting every
                         defect region rather than just the largest one;
                         standard in the MVTec AD literature since AUROC
                         alone can look good while missing small defects.

Usage:
    python evaluate.py --checkpoint ./models/patchcore_bottle.ckpt \
                        --data-root ./data/raw --category bottle
"""

import argparse
import json
from pathlib import Path

from anomalib.data import MVTecAD
from anomalib.engine import Engine
from anomalib.models import EfficientAd, Padim, Patchcore
from anomalib.pre_processing import PreProcessor
from torch.serialization import safe_globals
from torchvision.transforms.v2 import Resize

MODEL_REGISTRY = {
    "patchcore": Patchcore,
    "padim": Padim,
    "efficientad": EfficientAd,
}


def evaluate(
    checkpoint: Path,
    data_root: Path,
    category: str,
    model_name: str = "patchcore",
    image_size: int = 256,
    enable_progress_bar: bool = False,
) -> dict:
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{model_name}'. Choose from {list(MODEL_REGISTRY)}")

    eval_batch_size = 1 if model_name == "efficientad" else 32
    # Note: MVTecAD takes `augmentations` (a torchvision v2 transform), not image_size
    # directly -- see train.py's build_datamodule for the same fix and rationale.
    datamodule = MVTecAD(
        root=str(data_root),
        category=category,
        eval_batch_size=eval_batch_size,
        augmentations=Resize((image_size, image_size)),
    )

    model_cls = MODEL_REGISTRY[model_name]
    # PyTorch 2.6+ defaults torch.load's weights_only to True, which blocks unpickling
    # Anomalib's PreProcessor class embedded in the checkpoint. weights_only=False trusts
    # the checkpoint contents -- fine here since it's one you trained yourself locally;
    # safe_globals([PreProcessor]) additionally allowlists that specific class explicitly.
    with safe_globals([PreProcessor]):
        model = model_cls.load_from_checkpoint(str(checkpoint), weights_only=False)
    engine = Engine(enable_progress_bar=enable_progress_bar)

    results = engine.test(model=model, datamodule=datamodule, ckpt_path=str(checkpoint))
    report = results[0] if results else {}

    print("\n=== Evaluation Report ===")
    for key, value in report.items():
        print(f"  {key}: {value}")

    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("./data/raw"))
    parser.add_argument("--category", type=str, required=True)
    parser.add_argument("--model", type=str, default="patchcore", choices=list(MODEL_REGISTRY),
                         help="Must match the model class the checkpoint was trained with")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--output-json", type=Path, default=None,
                         help="Optional path to write the report as JSON")
    args = parser.parse_args()

    report = evaluate(args.checkpoint, args.data_root, args.category, args.model, args.image_size)

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\n[done] report written to {args.output_json}")


if __name__ == "__main__":
    main()
