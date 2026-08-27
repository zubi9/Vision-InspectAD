"""
Train all 3 YOLO26-seg specialists (DAGM, Kolektor SDD2, Magnetic Tile) sequentially,
each with its correct data.yaml and output_dir baked in from src/common/paths.py --
avoids retyping (and mistyping) paths across 3 separate manual invocations.

Usage:
    python train_all_specialists.py --epochs 100
    python train_all_specialists.py --specialists dagm kolektor --epochs 50
"""

import argparse

from src.common import paths
from train import train

SPECIALISTS = {
    "dagm": (paths.DAGM_YOLO_DATA_YAML, paths.DAGM_MODEL_DIR),
    "kolektor": (paths.KOLEKTOR_YOLO_DATA_YAML, paths.KOLEKTOR_MODEL_DIR),
    "magnetic_tile": (paths.MAGNETIC_TILE_YOLO_DATA_YAML, paths.MAGNETIC_TILE_MODEL_DIR),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--specialists", nargs="+", default=list(SPECIALISTS), choices=list(SPECIALISTS))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--base-weights", type=str, default="yolo26s-seg.pt")
    args = parser.parse_args()

    results = {}
    for name in args.specialists:
        data_yaml, output_dir = SPECIALISTS[name]
        if not data_yaml.exists():
            print(f"[SKIP] {name}: {data_yaml} not found -- has this specialist's dataset been prepared?")
            continue

        print(f"\n{'=' * 60}\nTraining specialist: {name}\n{'=' * 60}")
        results[name] = train(
            data_yaml=data_yaml,
            epochs=args.epochs,
            image_size=args.image_size,
            base_weights=args.base_weights,
            output_dir=output_dir,
            run_name=f"yolo26seg-{name}",
        )

    print(f"\nDone: {len(results)}/{len(args.specialists)} specialists trained.")
    for name, result in results.items():
        print(f"  {name}: {result['checkpoint']}")


if __name__ == "__main__":
    main()
