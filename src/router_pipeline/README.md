# VisionInspect Router Pipeline

Three-stage specialist-routing pipeline:

```text
Image -> YOLO26n classifier -> specialist ID -> specialist model
```

Router classes:
- DAGM
- KolektorSDD2
- Magnetic_Tile
- 15 MVTec AD categories, each corresponding to one PatchCore specialist.

## 1. Prepare

Expected relative layout:

```text
datasets/
├── yolo_segmentation/
│   ├── DAGM/
│   ├── KolektorSDD2/
│   └── Magnetic_Tile/
└── MVTec_AD/
    ├── bottle/ ... zipper/
```

Run:

```bash
python prepare_data.py \
  --data-root datasets \
  --output-root datasets/domain_router \
  --val-ratio 0.15 --test-ratio 0.15 \
  --seed 42 --overwrite
```

The discovery logic excludes mask files (`*_mask`, `*_gt`, `*_label`) and common mask/label directories.

## 2. Train + MLflow

```bash
pip install ultralytics mlflow scikit-learn pandas
python train.py \
  --data datasets/domain_router \
  --model yolo26n-cls.pt \
  --epochs 75 --imgsz 224 --batch 32 \
  --experiment VisionInspect-Router \
  --run-name router-yolo26n-v1 \
  --export-onnx
```

MLflow defaults to `./mlruns`.

```bash
mlflow ui --backend-store-uri ./mlruns
```

Tracked artifacts include parameters, dataset metadata/manifest, validation metrics, training outputs, best weights, and optional ONNX export.

## 3. Inference

```bash
python inference.py \
  --model runs/classify/specialist_router/weights/best.pt \
  --source path/to/image.png \
  --threshold 0.80
```

Batch:

```bash
python inference.py \
  --model runs/classify/specialist_router/weights/best.pt \
  --source path/to/images \
  --recursive --threshold 0.80 \
  --output predictions.jsonl
```

A low-confidence prediction becomes `UNKNOWN` instead of forcing an incorrect specialist.

## Architecture boundary

`inference.py` only performs routing. It does not execute PatchCore or YOLO segmentation. A later serving layer can map:

```text
MVTec_bottle -> PatchCore(bottle)
DAGM        -> DAGM YOLO specialist
...
```

This separation is deliberate and makes later FastAPI/Triton integration straightforward.
