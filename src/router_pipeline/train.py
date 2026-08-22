#!/usr/bin/env python3
import argparse
import json
import time
from pathlib import Path
form visioninspect_v112.api import config

import mlflow
from ultralytics import YOLO


def args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--model", default="yolo26n-cls.pt")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--imgsz", type=int, default=224)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--experiment", default="VisionInspect-Router")
    p.add_argument("--run-name", default=None)
    p.add_argument("--tracking-uri", default="sqlite:///./experiments/mlflow.db")
    p.add_argument("--project", type=Path, default=Path(config.PROJECT_ROOT) / "models" / "runs" / "classify")
    p.add_argument("--name", default="specialist_router")
    p.add_argument("--export-onnx", action="store_true")
    return p.parse_args()


def main():
    a = args()
    mlflow.set_tracking_uri(a.tracking_uri)
    mlflow.set_experiment(a.experiment)

    with mlflow.start_run(run_name=a.run_name or a.name) as run:
        mlflow.log_params(vars(a) | {"data": str(a.data)})

        meta = a.data / "metadata.json"
        man = a.data / "manifest.csv"

        if meta.exists():
            mlflow.log_artifact(str(meta), "dataset")
            mlflow.log_param(
                "num_router_classes", json.loads(meta.read_text())["num_classes"]
            )

        if man.exists():
            mlflow.log_artifact(str(man), "dataset")

        model = YOLO(a.model)
        t = time.perf_counter()

        model.train(
            data=str(a.data),
            epochs=a.epochs,
            imgsz=a.imgsz,
            batch=a.batch,
            workers=a.workers,
            device=a.device,
            seed=a.seed,
            patience=a.patience,
            pretrained=True,
            project=str(a.project),
            name=a.name,
            exist_ok=True,
            plots=True,
        )

        mlflow.log_metric("training_minutes", (time.perf_counter() - t) / 60)

        best = Path(model.trainer.best)

        val = model.val(data=str(a.data), split="val", imgsz=a.imgsz, batch=a.batch, device=a.device)
        for k, v in getattr(val, "results_dict", {}).items():
            try:
                mlflow.log_metric("val_" + k.replace("/", "_"), float(v))
            except Exception:
                pass

        mlflow.log_artifacts(str(best.parent), "ultralytics_run")
        mlflow.log_artifact(str(best), "model")

        if a.export_onnx:
            out = model.export(format="onnx", imgsz=a.imgsz, simplify=True)
            mlflow.log_artifact(str(Path(out)), "model")
            print("ONNX:", out)

        print("Run:", run.info.run_id)
        print("Best:", best)


if __name__ == "__main__":
    main()
