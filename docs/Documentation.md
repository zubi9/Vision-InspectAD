# VisionInspect — Technical Documentation

Reference documentation: exact API contract, every configuration variable, the request lifecycle,
and a map of what lives where. For *why* the system is built this way, see the main
[README](README.md). For phase-by-phase setup commands, see [docs/OPERATIONS.md](docs/OPERATIONS.md).

---

## Table of Contents

1. [Request Lifecycle](#request-lifecycle)
2. [API Reference](#api-reference)
3. [Data Schemas](#data-schemas)
4. [Router Class Taxonomy](#router-class-taxonomy)
5. [Model Registry](#model-registry)
6. [Configuration Reference](#configuration-reference)
7. [Module Map](#module-map)
8. [Serving Modes](#serving-modes)
9. [Test Suite Reference](#test-suite-reference)

---

## Request Lifecycle

What actually happens on a single `POST /predict` call, in order:

1. **Decode.** The uploaded file is validated as an image (`Content-Type` must start with
   `image/`) and opened as an RGB `PIL.Image`. Malformed uploads fail here with `400`, before
   any model runs.
2. **Route.** `app_state.router_model.route(image)` — the YOLO26n-cls router — returns a
   `RoutingDecision`: which of the 18 classes it thinks this is, its confidence, and whether that
   confidence clears `VI_ROUTER_CONFIDENCE_THRESHOLD` (default `0.6`).
3. **Confidence gate.** Below threshold → the request returns an `UnrecognizedResult` immediately.
   No backend runs. This is deliberate: a forced guess here would silently send the image to an
   unrelated specialist model and return a confident-looking, meaningless result.
4. **Resolve.** `model_registry.resolve(registry, router_class)` looks up which backend
   (`anomalib` or `yolo26-seg`) and which checkpoint/URL handles this class. Raises `FileNotFoundError`
   (local checkpoint missing) or `KeyError` (class not in registry) — both become `500`s with the
   specific reason in the response body, not a generic error.
5. **Infer.** The resolved backend (`AnomalibBackend` or `Yolo26SegBackend`) runs. Each backend
   owns its own pre/post-processing — the API layer never touches raw tensors.
6. **Normalize.** The backend's raw dict output is wrapped into a `PredictionResult`, the single
   shape every backend's output gets mapped into regardless of which model actually answered.

```
image bytes -> decode -> route() -> [confident?] -> resolve() -> backend.predict() -> PredictionResult
                             \-- no -> UnrecognizedResult (stop here)
```

---

## API Reference

### `GET /health`

Liveness check. No model access, no dependencies.

**Response `200`:**
```json
{"status": "ok"}
```

### `POST /predict`

**Request:** `multipart/form-data`, single field `file` (image).

```bash
curl -X POST http://localhost:8000/predict -F "file=@sample.png"
```

**Response `200` — recognized (`PredictionResult`):**
```json
{
  "defect_detected": true,
  "image_score": 0.91,
  "regions": [
    {"bbox": [120.0, 84.0, 210.0, 160.0], "mask": null, "score": 0.91, "label": null}
  ],
  "source_model": "anomalib",
  "router_class": "MVTec_bottle",
  "router_confidence": 0.97,
  "model_version": null,
  "heatmap_overlay_base64": "iVBORw0KGgoAAAANSUhEUgAA..."
}
```

- `regions[].label` is always `null` for `source_model: "anomalib"` (one-class, no class head) and
  populated for `source_model: "yolo26-seg"` (e.g. `"scratch"`, `"blowhole"`).
- `heatmap_overlay_base64` is populated only for `source_model: "anomalib"` (base64 PNG, original
  image blended with the anomaly heatmap) — `null` for YOLO results, which return `regions[].mask`
  (polygon points) instead when segmentation masks are available.

**Response `200` — unrecognized (`UnrecognizedResult`):**
```json
{
  "recognized": false,
  "router_raw_class": "MVTec_screw",
  "router_confidence": 0.34,
  "message": "Router confidence below threshold -- image not routed to any model."
}
```

Note this is still HTTP `200`, not an error status — an honest "couldn't confidently route this"
is a valid, expected outcome, not a failure of the API itself. Check the `recognized` field to
distinguish the two response shapes.

**Response `400`:** invalid content-type, or image bytes that fail to decode. `detail` explains which.

**Response `500`:** the resolved model's checkpoint is missing, or the router returned a class
absent from the registry. `detail` includes the specific class and path — check server startup
logs first, since `api/main.py` prints every registry entry pointing at a missing checkpoint at
boot time.

---

## Data Schemas

### `Region` (`src/common/schemas.py`)

| Field | Type | Notes |
|---|---|---|
| `bbox` | `[float, float, float, float]` | `(x_min, y_min, x_max, y_max)`, pixel coordinates |
| `mask` | `list[list[float]] \| null` | Polygon points `[[x, y], ...]`; YOLO only, when segmentation masks are present |
| `score` | `float` | Per-region confidence/anomaly score |
| `label` | `str \| null` | Class name; always `null` for Anomalib (one-class, no class head) |

### `PredictionResult`

| Field | Type | Notes |
|---|---|---|
| `defect_detected` | `bool` | |
| `image_score` | `float` | Anomalib: `pred_score` from the model. YOLO: max region score, `0.0` if no detections |
| `regions` | `list[Region]` | Empty list if nothing detected/flagged |
| `source_model` | `"anomalib" \| "yolo26-seg"` | Which backend actually answered |
| `router_class` | `str` | One of the 18 router classes |
| `router_confidence` | `float` | 0-1 |
| `model_version` | `str \| null` | Reserved, currently always `null` |
| `heatmap_overlay_base64` | `str \| null` | Anomalib only; base64 PNG |

### `UnrecognizedResult`

| Field | Type | Notes |
|---|---|---|
| `recognized` | `Literal[False]` | Always `false` — the discriminator field |
| `router_raw_class` | `str` | What the router guessed, despite low confidence |
| `router_confidence` | `float` | |
| `message` | `str` | Fixed explanatory text |

---

## Router Class Taxonomy

18 classes total. `src/common/paths.py`'s `ROUTER_CLASSES` is the single source of truth — every
other reference to this list (router training, `model_registry.py`, Triton model repo generation)
imports it from there rather than redefining it.

| Router class | Backend | Notes |
|---|---|---|
| `MVTec_bottle` ... `MVTec_zipper` (15 total) | `anomalib` | One PatchCore checkpoint per MVTec AD category |
| `DAGM` | `yolo26-seg` | Specialist model, `models/dagm/` |
| `KolektorSDD2` | `yolo26-seg` | Specialist model, `models/kolektor/` (HF Hub filename is `kolekor-best.onnx`, kept as-is to match the real remote file) |
| `Magnetic_Tile` | `yolo26-seg` | Specialist model, `models/magnetic_tile/` |

`UNKNOWN` is not a registry entry — it's what `model_registry.resolve()` explicitly rejects
(`ValueError`) if ever passed, since it represents "router wasn't confident," which the API layer
already intercepts before `resolve()` is called (Request Lifecycle, step 3).

---

## Model Registry

`src/common/model_registry.py` provides three builder functions, all producing the same 18-key
dict shape (`dict[str, ModelEntry]`), differing only in what each entry's `checkpoint` points at:

| Function | `checkpoint` points at | When used |
|---|---|---|
| `build_registry()` | Local `.ckpt` / `.pt` (training-time formats) | Diagnostics only — not used by the API |
| `build_onnx_registry()` | Local `.onnx` exports | Default API serving mode (`VI_USE_TRITON=false`) |
| `build_triton_registry()` | Anomalib: local `.onnx` (unchanged). YOLO/router: `http://<triton>/<model_name>` | `VI_USE_TRITON=true` |

`ModelEntry.checkpoint` is typed `Path | str` specifically to support both a local file and a
Triton URL through the same field — backends branch on `isinstance(checkpoint, str) and
checkpoint.startswith("http")` rather than needing a separate code path per serving mode.

`build_onnx_registry()` resolves each Anomalib category via `paths.find_anomalib_onnx()`, which
`rglob`s for the filename rather than assuming a fixed nested path — Anomalib's own ONNX exporter
folder structure isn't something this project controls or has found to be stable across versions.

Run `python src/common/model_registry.py [--onnx | --triton]` directly to print every entry and
whether its checkpoint currently resolves — the fastest way to check "is everything wired up"
without starting the API.

---

## Configuration Reference

All environment variables use a `VI_` prefix specifically to avoid collision with generic names
(`DATA_ROOT`, `OUTPUT_ROOT`, etc.) that a different tool in the same environment might also read.
Every path defaults relative to `PROJECT_ROOT`, computed automatically from `src/common/paths.py`'s
own file location — no variable needs to be set for local development; all of these exist purely
as *overrides*.

### Data paths (`src/common/paths.py`)

| Variable | Default (relative to project root) |
|---|---|
| `VI_MVTEC_RAW_ROOT` | `data/raw` |
| `VI_SUPERVISED_RAW_ROOT` | `data/raw_supervised` |
| `VI_ROUTER_SOURCE_ROOT` | `data/router_source` |
| `VI_DOMAIN_ROUTER_DS_ROOT` | `data/domain_router_ds` |
| `VI_YOLO_SEG_DS_ROOT` | `data/yolo-seg-ds` |

### Model paths

| Variable | Default |
|---|---|
| `VI_ANOMALIB_MODELS_DIR` | `models` |
| `VI_ANOMALIB_ONNX_DIR` | `models/onnx` |
| `VI_ANOMALIB_MODEL_NAME` | `patchcore` |
| `VI_ROUTER_RUN_DIR` | `models/runs/classify` |
| `VI_ROUTER_RUN_NAME` | `specialist_router` |
| `VI_ROUTER_ONNX_PATH` | `models/runs/classify/specialist_router/weights/best.onnx` |
| `VI_DAGM_MODEL_DIR` | `models/dagm` |
| `VI_KOLEKTOR_MODEL_DIR` | `models/kolektor` |
| `VI_MAGNETIC_TILE_MODEL_DIR` | `models/magnetic_tile` |
| `VI_DAGM_ONNX_PATH` | `models/dagm/weights/best.onnx` |
| `VI_KOLEKTOR_ONNX_PATH` | `models/kolektor/weights/best.onnx` |
| `VI_MAGNETIC_TILE_ONNX_PATH` | `models/magnetic_tile/weights/best.onnx` |

### Runtime / API behavior (`api/config.py`)

| Variable | Default | Notes |
|---|---|---|
| `VI_ROUTER_CONFIDENCE_THRESHOLD` | `0.6` | Below this, `/predict` returns `UnrecognizedResult` |
| `VI_ANOMALY_REGION_THRESHOLD` | `0.5` | Heatmap threshold for converting Anomalib's anomaly map into bounding-box regions |
| `VI_ANOMALIB_DEVICE` | `AUTO` | Passed to `OpenVINOInferencer` (`AUTO`/`CPU`/`GPU`/`NPU`) |
| `VI_ANOMALIB_INFERENCER_CACHE_SIZE` | `3` | LRU cache size — how many Anomalib category models stay warm in memory at once |
| `VI_USE_TRITON` | `false` | `true`/`1`/`yes` (case-insensitive) switches YOLO/router serving to Triton |
| `VI_TRITON_URL` | `http://triton:8000` | Internal docker-compose service address, not the host-remapped port |

### Other

| Variable | Default | Used by |
|---|---|---|
| `VI_MLFLOW_TRACKING_URI` | `sqlite:///<project_root>/experiments/mlflow.db` | All training scripts |
| `VI_HF_REPO_ID` | *(none - required)* | `scripts/download_models.py` |
| `CI` | *(set by GitHub Actions)* | `scripts/download_models.py` auto-detects this to skip interactive prompts |

---

## Module Map

| Path | Role |
|---|---|
| `src/common/paths.py` | Single source of truth for every path and the 18-class taxonomy. Everything else imports from here. |
| `src/common/model_registry.py` | Router class -> `{backend, checkpoint}`. Three builders for three serving contexts. |
| `src/common/schemas.py` | `Region` / `PredictionResult` / `UnrecognizedResult` — the backend-agnostic response contract. |
| `src/anomalib_pipeline/` | PatchCore training (`train.py`, `train_all_categories.py`), evaluation, dataset structure verification, ONNX export. |
| `src/yolo_pipeline/` | Per-specialist YOLO26-seg training (`train.py` takes `--data`/`--output-dir`; `train_all_specialists.py` wraps all three with correct default paths). |
| `src/router_pipeline/` | Router dataset prep (stratified split, manifest), training, standalone inference/dispatch check. |
| `api/main.py` | FastAPI app + `lifespan` — loads router, builds the registry, constructs both backends once at startup. |
| `api/routes.py` | `/health`, `/predict` — the request lifecycle described above. |
| `api/config.py` | Thin re-export of `paths.py` plus API-only runtime settings (thresholds, cache size, Triton toggle). |
| `api/backends/router.py` | Wraps the YOLO26n-cls router; local `.onnx` or Triton URL through the same `YOLO()` call. |
| `api/backends/anomalib_backend.py` | `OpenVINOInferencer`-based inference, LRU-cached across categories, heatmap-to-region conversion via connected components, heatmap overlay image generation. |
| `api/backends/yolo_backend.py` | Ultralytics-based inference for the 3 specialists; local `.onnx` or Triton URL, same call either way. |
| `serving/triton/generate_model_repository.py` | Builds `serving/triton/model_repository/` from real ONNX exports — auto-complete config, not hand-specified tensor names. |
| `serving/benchmark.py` | Hits a running `/predict` endpoint, reports p50/p95/p99/throughput, grouped by which router class answered. |
| `scripts/download_models.py` | HF Hub model download — ETag version check, interactive/CI-aware prompting. |
| `app/streamlit_app.py` | Demo UI — heatmap overlay for Anomalib results, drawn boxes/masks for YOLO results. |
| `tests/` | pytest suite; `conftest.py` stubs `anomalib`/`ultralytics`/`mlflow`/`huggingface_hub` when not installed. |

---

## Serving Modes

Two interchangeable modes, switched by `VI_USE_TRITON`, both reachable through the identical
`/predict` contract — a client can't tell which mode answered except via the `source_model` field
(which reflects the *backend type*, `anomalib`/`yolo26-seg`, not the *serving mode*, local/Triton).

| | Local (default) | Triton (`VI_USE_TRITON=true`) |
|---|---|---|
| Anomalib | `OpenVINOInferencer`, in-process | **Unchanged** — still local. See `model_registry.py`'s `build_triton_registry()` docstring for why this wasn't migrated. |
| YOLO (3 specialists + router) | `Ultralytics YOLO(local_path)`, in-process | `Ultralytics YOLO(triton_url, task=...)` — same pre/post-processing, tensor compute dispatched to Triton |

Switching modes requires `serving/triton/generate_model_repository.py` to have been run first
(populates `serving/triton/model_repository/`) and `docker compose --profile triton up` to start
the `triton` service — it's opt-in, not part of the default `docker compose up`.

---

## Test Suite Reference

| File | Covers |
|---|---|
| `test_prepare_data.py` | MVTec AD dataset structure verification — valid/invalid splits, missing masks |
| `test_anomalib_backend.py` | `AnomalibBackend`: schema shape, score-threshold fallback, heatmap-to-region conversion |
| `test_yolo_backend.py` | `Yolo26SegBackend`: region extraction, empty detections, mask polygons, caching |
| `test_router_backend.py` | `RouterModel`: confidence gating at/above/below threshold |
| `test_model_registry.py` | All three registry builders, `resolve()`'s error paths (`KeyError`/`FileNotFoundError`/`ValueError` for `UNKNOWN`), Triton URL resolution |
| `test_schemas.py` | `Region`/`PredictionResult`/`UnrecognizedResult` validation, JSON round-trip |
| `test_api.py` | `/health`, `/predict` end-to-end via `TestClient` (no `with` block — lifespan never runs, `app_state` populated with fakes per test) |
| `test_download_models.py` | `HF_TO_LOCAL` filename mapping correctness, version-check decision tree, manifest persistence |

`tests/conftest.py` stubs `anomalib`, `ultralytics`, `mlflow`, and `huggingface_hub` when they
aren't installed, since every test mocks these libraries' public interfaces directly rather than
needing the real (heavy) packages — this is what keeps CI's test job fast (`requirements-test.txt`,
not the full `requirements.txt`).
