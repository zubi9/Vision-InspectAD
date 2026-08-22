"""
Train YOLO26-seg on the combined supervised defect dataset (DAGM + Magnetic Tiles +
Kolektor SDD2, merged into one Ultralytics-format dataset).

This assumes the three datasets have already been merged into a single YOLO
segmentation dataset with one data.yaml (standard Ultralytics multi-class format):

    data/supervised_combined/
        images/train/*.jpg
        images/val/*.jpg
        labels/train/*.txt      # YOLO-seg polygon format
        labels/val/*.txt
        data.yaml                # names: [dagm classes..., magnetic tile classes..., kolektor classes...]

Merging DAGM (weak texture-defect labels), Magnetic Tiles (per-defect-type folders),
and Kolektor SDD2 (pixel masks) into one consistent YOLO-seg label format is its own
data-engineering task -- this script trains against the merged result, it doesn't do
the merging itself.

One combined model (not three separate ones) was chosen deliberately: it lets the
router treat "supervised defect" as a single class (see src/router_pipeline/), and
YOLO natively supports multi-class detection in one model, unlike Anomalib's
one-class-per-category constraint.

Usage:
    python train.py --data ./data/supervised_combined/data.yaml --epochs 100
"""

import argparse
from pathlib import Path

import mlflow
from ultralytics import YOLO


def train(
    data_yaml: Path,
    epochs: int = 100,
    image_size: int = 640,
    base_weights: str = "yolo26s-seg.pt",
    output_dir: Path = Path("./models/yolo26seg"),
    mlflow_experiment: str = "visioninspect-yolo26seg",
    mlflow_tracking_uri: str = "sqlite:///./experiments/mlflow.db",
) -> dict:
    if mlflow_tracking_uri.startswith("sqlite:///"):
        db_path = Path(mlflow_tracking_uri.replace("sqlite:///", "", 1))
        db_path.parent.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(mlflow_experiment)

    output_dir.mkdir(parents=True, exist_ok=True)

    with mlflow.start_run(run_name="yolo26seg-combined") as run:
        mlflow.log_params({
            "base_weights": base_weights,
            "epochs": epochs,
            "image_size": image_size,
            "data_yaml": str(data_yaml),
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
        print(f"[done] IMPORTANT: copy or symlink {best_weights} to "
              f"src/yolo_pipeline/weights/best.pt for the API/registry to find it, "
              f"and fill in PROVENANCE.md with this run_id.")

    return {"run_id": run_id, "checkpoint": str(best_weights), "class_names": model.names, "metrics": metrics}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, required=True, help="Path to the merged dataset's data.yaml")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--base-weights", type=str, default="yolo26s-seg.pt")
    parser.add_argument("--output-dir", type=Path, default=Path("./models/yolo26seg"))
    args = parser.parse_args()

    train(
        data_yaml=args.data,
        epochs=args.epochs,
        image_size=args.image_size,
        base_weights=args.base_weights,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
