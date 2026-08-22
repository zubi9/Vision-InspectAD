"""
Train an Anomalib model on a single MVTec AD category.

Designed to be called either:
  - from the CLI:            python train.py --category bottle --model patchcore
  - from the training notebook, parameterized via papermill

Logs to MLflow so every checkpoint under models/ is traceable back to a run.

Model choice (Phase 1 default: PatchCore):
  - PatchCore: strong accuracy, memory-bank based, no training of a neural net
    (fits nucleus of features from training set) -> fast to "train", slower
    at inference due to nearest-neighbor search over the memory bank.
  - PaDiM: similar family, lighter memory footprint, slightly lower accuracy.
  - EfficientAD: trained network, fastest inference, best fit if latency
    matters more than squeezing out the last AUROC points.
  Phase 2 will benchmark these against each other; PatchCore is the Phase 1
  baseline because it has the strongest published results on MVTec AD with
  minimal hyperparameter tuning.
"""

import argparse
from pathlib import Path

import mlflow
from anomalib.data import MVTecAD
from anomalib.engine import Engine
from anomalib.models import EfficientAd, Padim, Patchcore
from torchvision.transforms.v2 import Resize

MODEL_REGISTRY = {
    "patchcore": Patchcore,
    "padim": Padim,
    "efficientad": EfficientAd,
}


def build_datamodule(data_root: Path, category: str, batch_size: int, image_size: int | None = None) -> MVTecAD:
    """
    Note: current Anomalib's MVTecAD does not accept an `image_size` kwarg directly --
    it accepts `augmentations` (a torchvision.transforms.v2 transform) instead. Passing
    image_size=... straight into MVTecAD raises TypeError against current versions.

    This resize is in addition to the model's own pre-processor (see build_model), which
    also resizes + normalizes as part of the forward pass. Having it in both places is
    harmless (resizing to the same target size twice is idempotent) and keeps the
    datamodule's batches consistently sized before they ever reach the model.
    """
    augmentations = Resize((image_size, image_size)) if image_size is not None else None
    return MVTecAD(
        root=str(data_root),
        category=category,
        train_batch_size=batch_size,
        eval_batch_size=batch_size,
        augmentations=augmentations,
    )


def build_model(model_name: str, image_size: int):
    """Build the model with its pre-processor configured for the target image size.

    Image resizing is owned by the model in current Anomalib, not the datamodule --
    each model's configure_pre_processor() returns a PreProcessor that resizes
    (and, for some models, center-crops) inputs before they hit the network.
    """
    model_cls = MODEL_REGISTRY[model_name]
    pre_processor = model_cls.configure_pre_processor(image_size=(image_size, image_size))
    return model_cls(pre_processor=pre_processor)


def train(
    data_root: Path,
    category: str,
    model_name: str,
    image_size: int = 256,
    batch_size: int = 32,
    max_epochs: int = 1,
    output_dir: Path = Path("./models"),
    mlflow_experiment: str = "visioninspect-anomalib",
    mlflow_tracking_uri: str = "sqlite:///./experiments/mlflow.db",
    enable_progress_bar: bool = False,
) -> dict:
    """
    Note on mlflow_tracking_uri: uses a SQLite-backed URI (sqlite:///path/to/file.db)
    rather than a bare directory path. Older MLflow versions accepted a plain folder
    as a file-store tracking URI; current versions expect an explicit backend scheme.
    A plain directory path will silently misbehave or raise deprecation errors depending
    on the installed MLflow version -- sqlite:/// is the reliable default across versions.

    Note on enable_progress_bar: defaults to False. Lightning/Anomalib's rich-based
    progress bar can trigger a RecursionError in some notebook environments (rich's
    Console wrapping sys.stdout gets double-wrapped by the notebook's own output
    capture, e.g. in VS Code notebooks). This is a display bug, not a training bug --
    disabling the fancy progress bar sidesteps it entirely. Set True if your
    environment doesn't hit this (plain terminal, classic Jupyter, etc.).
    """
    if model_name == "efficientad" and batch_size != 1:
        raise ValueError("EfficientAd requires batch_size=1 for both training and evaluation.")

    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{model_name}'. Choose from {list(MODEL_REGISTRY)}")

    if mlflow_tracking_uri.startswith("sqlite:///"):
        db_path = Path(mlflow_tracking_uri.replace("sqlite:///", "", 1))
        db_path.parent.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(mlflow_experiment)

    datamodule = build_datamodule(data_root, category, batch_size, image_size)
    model = build_model(model_name, image_size)
    engine = Engine(max_epochs=max_epochs, enable_progress_bar=enable_progress_bar)

    with mlflow.start_run(run_name=f"{model_name}-{category}") as run:
        mlflow.log_params({
            "model": model_name,
            "category": category,
            "image_size": image_size,
            "batch_size": batch_size,
            "max_epochs": max_epochs,
        })

        engine.fit(model=model, datamodule=datamodule)
        test_results = engine.test(model=model, datamodule=datamodule)

        # Anomalib's Engine.test returns a list of dicts (one per dataloader);
        # log every scalar metric it reports so nothing is silently dropped.
        if test_results:
            for key, value in test_results[0].items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(key, value)

        output_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = output_dir / f"{model_name}_{category}.ckpt"
        engine.trainer.save_checkpoint(str(ckpt_path))
        mlflow.log_artifact(str(ckpt_path))

        run_id = run.info.run_id
        print(f"\n[done] run_id={run_id}")
        print(f"[done] checkpoint saved to {ckpt_path}")
        print(f"[done] IMPORTANT: record this run_id next to the checkpoint -- "
              f"it is how the shipped model is traced back to its training run.")

    return {"run_id": run_id, "checkpoint": str(ckpt_path), "test_results": test_results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("./data/raw"))
    parser.add_argument("--category", type=str, required=True, help="MVTec AD category, e.g. 'bottle'")
    parser.add_argument("--model", type=str, default="patchcore", choices=list(MODEL_REGISTRY))
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-epochs", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=Path("./models"))
    parser.add_argument("--enable-progress-bar", action="store_true",
                         help="Enable rich progress bar (may cause RecursionError in some notebook environments)")
    args = parser.parse_args()

    train(
        data_root=args.data_root,
        category=args.category,
        model_name=args.model,
        image_size=args.image_size,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        output_dir=args.output_dir,
        enable_progress_bar=args.enable_progress_bar,
    )


if __name__ == "__main__":
    main()
