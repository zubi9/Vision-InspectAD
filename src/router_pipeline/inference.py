#!/usr/bin/env python3
import argparse
import json
from logging import config
from pathlib import Path

from ultralytics import YOLO

EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path(config.PROJECT_ROOT) / "models" / "runs" / "classify" / "weights" / "best.onnx", required=True)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--imgsz", type=int, default=224)
    p.add_argument("--threshold", type=float, default=25.0)
    p.add_argument("--device", default=None)
    p.add_argument("--output", type=Path(config.PROJECT_ROOT) / "results" / "classify" / "specialist_router_predictions.json")
    p.add_argument("--recursive", action="store_true")
    return p.parse_args()


def main():
    a = args()
    model = YOLO(str(a.model))

    if a.source.is_file():
        imgs = [a.source]
    else:
        iterator = a.source.rglob("*") if a.recursive else a.source.glob("*")
        imgs = sorted(p for p in iterator if p.is_file() and p.suffix.lower() in EXT)

    if not imgs:
        raise RuntimeError("No images found")

    out = []
    for img in imgs:
        res = model.predict(source=str(img), imgsz=a.imgsz, device=a.device, verbose=False)[0]
        idx = int(res.probs.top1)
        conf = float(res.probs.top1conf)
        name = res.names[idx] if isinstance(res.names, dict) else str(idx)

        record = {
            "image": str(img),
            "predicted_specialist": name if conf >= a.threshold else "UNKNOWN",
            "raw_prediction": name,
            "confidence": conf,
            "threshold": a.threshold,
            "accepted": conf >= a.threshold,
        }

        out.append(record)
        print(f"{img} -> {record['predicted_specialist']} ({conf:.4f})")

    if a.output:
        a.output.parent.mkdir(parents=True, exist_ok=True)
        a.output.write_text("\n".join(json.dumps(x) for x in out) + "\n")
        print(f"Wrote {len(out)} predictions to {a.output}")


if __name__ == "__main__":
    main()
