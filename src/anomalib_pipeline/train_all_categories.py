"""
Train a single Anomalib model across all MVTec AD categories, sequentially.

Important: this does NOT train one unified multi-class model. Anomalib's models
(PatchCore, PaDiM, EfficientAD) are one-class -- each category gets its own
memory bank / distribution / network, fit only on that category's normal images.
This script loops the same training call once per category, matching how the
MVTec AD benchmark is normally reported (a per-category results table).

VRAM notes (relevant for constrained GPUs, e.g. 8GB):
  - Runs are strictly sequential -- only one category's model is ever in VRAM
    at a time, so total VRAM need is the same as a single-category run.
  - torch.cuda.empty_cache() is called between categories to release cached
    allocator memory before the next run starts, since Lightning/Anomalib
    don't always tear down cleanly between successive Engine() instances in
    the same process.
  - If a category still OOMs (most likely with EfficientAD at larger image
    sizes), reduce --image-size or, for PatchCore/PaDiM, --batch-size before
    retrying just that category.
  - One category failing (OOM, corrupt data, etc.) does not stop the rest --
    failures are collected and reported at the end so you can retry only
    what's needed.

Usage:
    python train_all_categories.py --model patchcore --max-epochs 1
    python train_all_categories.py --model efficientad --max-epochs 20 --categories bottle cable
"""

import argparse
import gc
import json
import time
from pathlib import Path

from prepare_data import MVTEC_CATEGORIES
from train import train


def train_all(
    data_root: Path,
    model_name: str,
    image_size: int = 256,
    batch_size: int = 32,
    max_epochs: int = 1,
    output_dir: Path = Path("./models"),
    categories: list[str] | None = None,
    mlflow_experiment: str = "visioninspect-anomalib",
    mlflow_tracking_uri: str = "sqlite:///./experiments/mlflow.db",
) -> dict:
    categories = categories or MVTEC_CATEGORIES
    if model_name == "efficientad" and batch_size != 1:
        print("[info] forcing batch_size=1 for EfficientAd (required by the model)")
        batch_size = 1

    results = {}
    failures = {}
    start = time.time()

    for i, category in enumerate(categories, 1):
        print(f"\n{'=' * 60}")
        print(f"[{i}/{len(categories)}] Training {model_name} on category: {category}")
        print(f"{'=' * 60}")

        try:
            result = train(
                data_root=data_root,
                category=category,
                model_name=model_name,
                image_size=image_size,
                batch_size=batch_size,
                max_epochs=max_epochs,
                output_dir=output_dir,
                mlflow_experiment=mlflow_experiment,
                mlflow_tracking_uri=mlflow_tracking_uri,
                enable_progress_bar=False,
            )
            results[category] = result
        except Exception as e:  # noqa: BLE001 -- intentionally broad: isolate per-category failures
            print(f"[FAILED] {category}: {e}")
            failures[category] = str(e)
        finally:
            # Release VRAM before the next category starts. Import torch lazily
            # so this script doesn't hard-fail on machines without CUDA either.
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
            except ImportError:
                pass
            gc.collect()

    elapsed = time.time() - start
    print(f"\n{'=' * 60}")
    print(f"Done: {len(results)} succeeded, {len(failures)} failed, {elapsed / 60:.1f} min total")
    if failures:
        print("Failed categories (retry individually with train.py):")
        for cat, err in failures.items():
            print(f"  - {cat}: {err}")

    summary = {
        "model": model_name,
        "results": {
            cat: {
                "run_id": r["run_id"],
                "checkpoint": r["checkpoint"],
                "test_results": r["test_results"],
            }
            for cat, r in results.items()
        },
        "failures": failures,
        "elapsed_minutes": elapsed / 60,
    }
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", type=Path, default=Path("./data/raw"))
    parser.add_argument("--model", type=str, default="patchcore", choices=["patchcore", "padim", "efficientad"])
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-epochs", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=Path("./models"))
    parser.add_argument("--categories", type=str, nargs="+", default=None,
                         help=f"Subset of categories to train (default: all 15). Choices: {MVTEC_CATEGORIES}")
    parser.add_argument("--summary-json", type=Path, default=Path("./models/all_categories_summary.json"))
    args = parser.parse_args()

    summary = train_all(
        data_root=args.data_root,
        model_name=args.model,
        image_size=args.image_size,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        output_dir=args.output_dir,
        categories=args.categories,
    )

    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.summary_json, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n[done] summary written to {args.summary_json}")


if __name__ == "__main__":
    main()
