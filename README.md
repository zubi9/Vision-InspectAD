# VisionInspect

Industrial defect inspection platform — Anomalib (unsupervised AD) as the primary model track,
YOLO26-seg (supervised, inference-only this iteration) as the secondary track.

See `VisionInspect_Project_Plan.md` for the full phased roadmap and design rationale.

## Phase 1 — Setup

### 1. Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Dataset

MVTec AD requires manual download (registration required):
https://www.mvtec.com/company/research/datasets/mvtec-ad

Extract it under `data/raw/` so the structure looks like:

```
data/raw/
  bottle/
    train/good/*.png
    test/good/*.png
    test/<defect_type>/*.png
    ground_truth/<defect_type>/*.png
  cable/
    ...
```

Verify the download:

```bash
python src/anomalib_pipeline/prepare_data.py --data-root ./data/raw --category bottle
# or check every category at once:
python src/anomalib_pipeline/prepare_data.py --data-root ./data/raw --category all
```

### 3. Train the Phase 1 baseline

Either directly:

```bash
python src/anomalib_pipeline/train.py --category bottle --model patchcore
```

Or reproducibly via the notebook + papermill (recommended — this is the "official" run):

```bash
papermill notebooks/train_anomalib.ipynb notebooks/executed/train_anomalib_bottle.ipynb \
    -p category bottle -p model_name patchcore
```

This logs params/metrics/checkpoint to MLflow under `experiments/mlflow/`. View the dashboard with:

```bash
mlflow ui --backend-store-uri sqlite:///./experiments/mlflow.db
```

**Record the printed `run_id`** — it's how the checkpoint saved under `models/` is traced back to
the run that produced it.

### 4. Evaluate

```bash
python src/anomalib_pipeline/evaluate.py \
    --checkpoint ./models/patchcore_bottle.ckpt \
    --data-root ./data/raw --category bottle \
    --output-json ./models/patchcore_bottle_eval.json
```

Reports image-level AUROC, pixel-level AUROC, and PRO score — not precision/recall/mAP, since this
is an anomaly detection task, not bounding-box object detection.

### 5. Train across all categories (optional)

Anomalib's models are one-class — there's no single multi-class model, so "all categories" means
looping the same training call 15 times, once per category (matching how MVTec AD results are
normally reported, as a per-category table). This runs strictly sequentially, so VRAM needs are
the same as a single-category run regardless of GPU size:

```bash
python src/anomalib_pipeline/train_all_categories.py --model patchcore --max-epochs 1
```

A failure in one category (OOM, corrupt image, etc.) doesn't stop the rest — failures are
collected and reported at the end so you can retry just those categories with `train.py` directly.
A summary JSON (all run IDs, checkpoints, and test metrics) is written to
`models/all_categories_summary.json`.

**Rough time expectations** (varies a lot by GPU): PatchCore/PaDiM are largely inference-speed
bound (no real backpropagation), so 15 categories typically finish in well under an hour on a
mid-range GPU. EfficientAD does real training and is forced to `batch_size=1`, so budget
meaningfully longer per category if you go that route — try one category first to gauge per-run
time before kicking off all 15.

## YOLO26-seg weights

Trained via `src/yolo_pipeline/train.py` against a merged `data.yaml` combining DAGM + Magnetic
Tiles + Kolektor SDD2 (one model, not three — see the project plan's Model Strategy section for
why). Merging the three datasets' different label formats into one Ultralytics-format dataset is a
separate data-engineering step this script assumes is already done. After training, copy/symlink
the resulting `best.pt` to `src/yolo_pipeline/weights/best.pt` and fill in
`src/yolo_pipeline/PROVENANCE.md` (source datasets, run_id, training date).

## Router (YOLO26n-cls)

Decides which of 18 downstream targets — the 15 Anomalib categories, or one of the three
supervised-dataset classes (`DAGM`, `KolektorSDD2`, `Magnetic_Tile`) — should handle an incoming
image. Necessary because Anomalib's models are one-class: a `bottle` checkpoint has no way to
recognize "this input isn't for me," so routing has to happen *before* inference, not inside any
one model. Full details in `src/router_pipeline/README.md`; short version:

```bash
pip install ultralytics mlflow scikit-learn pandas

# 1. Link existing data into the unified layout prepare_data.py expects (symlinks, no duplication)
python src/router_pipeline/link_datasets.py

# 2. Build the router's classification dataset (stratified train/val/test split)
python src/router_pipeline/prepare_data.py \
    --data-root ./data/router_source --output-root ./data/domain_router \
    --val-ratio 0.15 --test-ratio 0.15 --overwrite

# 3. Train the router
python src/router_pipeline/train.py \
    --data ./data/domain_router --model yolo26n-cls.pt \
    --epochs 75 --imgsz 224 --batch 32 \
    --experiment VisionInspect-Router --run-name router-yolo26n-v1 --export-onnx

# 4. Try routing a single image
python src/router_pipeline/inference.py \
    --model runs/classify/specialist_router/weights/best.pt \
    --source path/to/image.png --threshold 0.80
```

Note the `--threshold` default in `inference.py` is `0.0` (accepts everything) — pass it
explicitly, as in the example above, to get the "low confidence → `UNKNOWN`" fallback behavior
rather than a forced guess. `DAGM`, `KolektorSDD2`, and `Magnetic_Tile` are three separate router
classes (not one combined bucket) even though they currently all resolve to the same combined
YOLO26-seg checkpoint — see `src/common/model_registry.py`'s docstring for why. That registry is
the single source of truth mapping a router class to its backend and checkpoint path; run it
directly (`python src/common/model_registry.py`) to sanity-check which checkpoints it can
currently find.

## Inference API (Phase 3)

Router-then-dispatch, served via FastAPI. All models load once at startup, not per-request.

```bash
pip install onnx onnxruntime openvino opencv-python fastapi "uvicorn[standard]" python-multipart

uvicorn api.main:app --reload
```

Then:

```bash
curl -X POST -F "file=@some_image.png" http://localhost:8000/predict
```

**Expects ONNX exports already in place** at the paths `api/config.py` defaults to (matching each
training script's own output layout — router at
`runs/classify/specialist_router/weights/best.onnx`, Anomalib at
`models/patchcore_<category>.onnx`, combined YOLO26-seg at
`models/yolo26seg/weights/best.onnx`). Override any of these via environment variables of the same
name (e.g. `ROUTER_ONNX_PATH=/other/path.onnx uvicorn api.main:app`) if your exports land elsewhere.
At startup the API prints a warning listing any registry entries pointing at checkpoints that
don't exist yet — check that output before assuming everything's wired up.

**Response shapes:**
- Confident routing → `PredictionResult` (`defect_detected`, `image_score`, `regions`,
  `source_model`, `router_class`, `router_confidence`).
- Router below its confidence threshold (default 0.6, matches training) → `UnrecognizedResult`,
  not a forced guess.

**Design notes worth knowing:**
- Anomalib inference goes through `anomalib.deploy.OpenVINOInferencer`, which accepts raw `.onnx`
  directly (not just OpenVINO IR) and owns its own pre/post-processing — deliberately not a
  hand-rolled `onnxruntime.InferenceSession`, to avoid silently mismatching training-time
  normalization (this project has already hit a few Anomalib API version mismatches; this
  sidesteps another category of them).
- YOLO inference (router + combined seg model) goes through Ultralytics' own `YOLO(onnx_path)`,
  which also handles pre/post-processing transparently regardless of `.pt` vs `.onnx`.
- Anomalib inferencers are LRU-cached (`ANOMALIB_INFERENCER_CACHE_SIZE`, default 3) rather than
  holding all 15 categories in memory at once or reloading from disk every request.

## Demo & Deployment (Phase 4)

```bash
docker compose up --build
```

- API: `http://localhost:8000` (docs at `/docs`)
- Streamlit: `http://localhost:8501`

Runs both containers, API healthchecked before Streamlit starts. `models/` and `runs/` are
volume-mounted read-write into the API container so new exports don't require a rebuild.

Without Docker:

```bash
uvicorn api.main:app --reload &
streamlit run app/streamlit_app.py
```

`PredictionResult` now carries `heatmap_overlay_base64` (Anomalib only) alongside `regions`, so
Streamlit shows a heatmap overlay for anomaly results and drawn boxes/masks for YOLO results.

## Testing & CI (Phase 5)

```bash
pip install -r requirements-test.txt   # lightweight -- no torch/anomalib/ultralytics needed
pytest -v
or Try.
PYENV_VERSION=<your-testing-env> python -m pytest -v
```

`tests/conftest.py` stubs `anomalib`, `ultralytics`, and `mlflow` when they're not installed, since
every test mocks their public interfaces (`OpenVINOInferencer`, `YOLO`) directly rather than
needing the real heavy frameworks. This keeps the test job fast; the full dependency graph is
still exercised for real by the Docker build job.

Coverage: dataset structure verification (`test_prepare_data.py`), both inference backends
(`test_anomalib_backend.py`, `test_yolo_backend.py`), the router's confidence gating
(`test_router_backend.py`), the dispatch registry (`test_model_registry.py`), shared schema
normalization (`test_schemas.py`), and API endpoints end-to-end (`test_api.py`, via
`TestClient` — deliberately *not* using the `with` context manager, so the real model-loading
lifespan never runs; `app_state` is populated with fakes per test instead).

**`.github/workflows/ci.yml`** runs on every push/PR: unit tests, `ruff check` + `ruff format
--check`, and a Docker build of both images (API + Streamlit) to catch Dockerfile breakage early.
Note the API image no longer bakes in model weights (`models/` is gitignored and empty on a fresh
checkout) — it creates an empty `models/` dir at build time and relies on `docker-compose.yml`'s
volume mount for real weights at runtime, which also means CI's Docker build doesn't need any
trained checkpoints to succeed.

## What's next

Phase 2 (Anomalib model/backbone comparison) picks up once Phase 1 baseline checkpoints exist.
With the router and YOLO26-seg training now in place too, Phase 3 (the inference API) is the next
big piece: wiring `model_registry.py` + the router into an actual `/predict` endpoint. See the
project plan doc for the full Phase 2–6 roadmap.
