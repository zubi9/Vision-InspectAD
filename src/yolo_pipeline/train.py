"""
Train a YOLO26-seg specialist on one supervised defect dataset (DAGM, Magnetic
Tiles, or Kolektor SDD2 -- each trained separately, not combined).

Expects a per-dataset Ultralytics-format dataset with one data.yaml:

    data/yolo-seg-ds/<dagm|kolektor|magnetic_tile>/
        images/train/*.jpg
        images/val/*.jpg
        labels/train/*.txt      # YOLO-seg polygon format
        labels/val/*.txt
        data.yaml

Converting each dataset's native label format into YOLO-seg polygons is its own
data-engineering task -- this script trains against the converted result.

Called 3x, once per specialist -- see train_all_specialists.py for a wrapper that
runs all three with the right default paths per specialist, rather than three
manual invocations each needing the correct --data/--output-dir typed by hand.

Usage:
    python train.py --data ./data/yolo-seg-ds/dagm/data.yaml --output-dir ./models/dagm --epochs 100

    python train.py --data ./data/supervised_combined/data.yaml --epochs 100
"""

import argparse
from pathlib import Path

import mlflow
from ultralytics import YOLO

from src.common import paths


def train(
    data_yaml: Path,
    epochs: int = 100,
    image_size: int = 640,
    base_weights: str = "yolo26s-seg.pt",
    output_dir: Path = Path("./models/specialist"),
    run_name: str = "yolo26seg-specialist",
    mlflow_experiment: str = "visioninspect-yolo26seg",
    mlflow_tracking_uri: str = paths.MLFLOW_TRACKING_URI,
) -> dict:
    if mlflow_tracking_uri.startswith("sqlite:///"):
        db_path = Path(mlflow_tracking_uri.replace("sqlite:///", "", 1))
        db_path.parent.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(mlflow_experiment)

    output_dir.mkdir(parents=True, exist_ok=True)

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params({
            "base_weights": base_weights,
            "epochs": epochs,
            "image_size": image_size,
            "data_yaml": str(data_yaml),
            "output_dir": str(output_dir),
        })

        model = YOLO(base_weights)
        train_results = model.train(
            data=str(data_yaml),
            epochs=epochs,
            imgsz=image_size,
            project=str(output_dir.parent),
            name=output_dir.name,
            exist_ok=True,
        )

        metrics = getattr(train_results, "results_dict", {}) or {}
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                mlflow.log_metric(key.replace("/", "_").replace("(", "").replace(")", ""), value)

        best_weights = output_dir / "weights" / "best.pt"
        if best_weights.exists():
            mlflow.log_artifact(str(best_weights))

        run_id = run.info.run_id
        print(f"\n[done] run_id={run_id}")
        print(f"[done] best weights: {best_weights}")
        print(f"[done] class names: {model.names}")

    return {"run_id": run_id, "checkpoint": str(best_weights), "class_names": model.names, "metrics": metrics}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, required=True, help="Path to this specialist's data.yaml")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--base-weights", type=str, default="yolo26s-seg.pt")
    parser.add_argument("--output-dir", type=Path, required=True,
                         help="e.g. ./models/dagm, ./models/kolektor, ./models/magnetic_tile")
    parser.add_argument("--run-name", type=str, default="yolo26seg-specialist")
    args = parser.parse_args()

    train(
        data_yaml=args.data,
        epochs=args.epochs,
        image_size=args.image_size,
        base_weights=args.base_weights,
        output_dir=args.output_dir,
        run_name=args.run_name,
    )


if __name__ == "__main__":
    main()
