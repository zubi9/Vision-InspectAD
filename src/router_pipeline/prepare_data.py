#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.router_pipeline import config



# Configuration
IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}

MASK_NAME_TOKENS = {
    "mask",
    "masks",
    "gt",
    "ground_truth",
    "ground-truth",
    "label",
    "labels",
    "annotation",
    "annotations",
}

MASK_SUFFIXES = (
    "_mask",
    "-mask",
    "_gt",
    "-gt",
    "_label",
    "-label",
    "_labels",
    "-labels",
)

MVTec_CATEGORIES = [
    "bottle",
    "cable",
    "capsule",
    "carpet",
    "grid",
    "hazelnut",
    "leather",
    "metal_nut",
    "pill",
    "screw",
    "tile",
    "toothbrush",
    "transistor",
    "wood",
    "zipper",
]

ROUTER_CLASSES = [
    "DAGM",
    "KolektorSDD2",
    "Magnetic_Tile",
] + [f"MVTec_{x}" for x in MVTec_CATEGORIES]



# Arguments
def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(description="Prepare VisionInspect specialist-router dataset.")

    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(config.RAW_DATA_ROOT),
        help="Root containing the original datasets. Default: datasets",
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(config.CLS_ROUTER_DATA_ROOT),
        help="Router dataset output directory.",
    )

    parser.add_argument("--val-ratio", type=float, default=0.15)

    parser.add_argument("--test-ratio", type=float, default=0.15)

    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--copy-mode",
        choices=["copy", "symlink"],
        default="copy",
        help="Copy images or create symbolic links.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove existing output directory.",
    )

    return parser.parse_args()



# Dataset discovery


def normalize_name(name: str) -> str:
    return "".join(character for character in name.lower() if character.isalnum())


def find_dataset_root(data_root: Path, candidates: list[str]) -> Path | None:
    candidate_keys = {normalize_name(x) for x in candidates}

    for path in data_root.iterdir():
        if not path.is_dir():
            continue
        if normalize_name(path.name) in candidate_keys:
            return path

    return None



# Image filtering


def is_mask_or_label_image(path: Path) -> bool:
    stem = path.stem.lower()

    # Check directory names.
    for part in path.parts:
        if part.lower() in MASK_NAME_TOKENS:
            return True

    # Check filename suffixes.
    if stem.endswith(MASK_SUFFIXES):
        return True

    return False


def find_images(root: Path) -> list[Path]:
    if not root.exists():
        return []

    images: list[Path] = []

    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        if is_mask_or_label_image(path):
            continue
        images.append(path)

    return sorted(images)



# DAGM


def discover_dagm(root: Path) -> list[dict]:
    records: list[dict] = []

    class_dirs = sorted(
        path
        for path in root.iterdir()
        if path.is_dir() and path.name.lower().startswith("class")
    )

    if not class_dirs:
        raise RuntimeError(f"No Class* directories found in DAGM: {root}")

    for class_dir in class_dirs:
        for split in ("train", "val", "test"):
            split_dir = class_dir / split
            if not split_dir.exists():
                continue
            for image in find_images(split_dir):
                records.append(
                    {
                        "router_class": "DAGM",
                        "source_dataset": "DAGM",
                        "source_category": class_dir.name,
                        "source_split": split,
                        "source_path": str(image.resolve()),
                    }
                )

    # Fallback for non-standard DAGM layouts.
    if not records:
        for class_dir in class_dirs:
            for image in find_images(class_dir):
                records.append(
                    {
                        "router_class": "DAGM",
                        "source_dataset": "DAGM",
                        "source_category": class_dir.name,
                        "source_split": "",
                        "source_path": str(image.resolve()),
                    }
                )

    return records



# KolektorSDD2

def discover_kolektor(root: Path) -> list[dict]:
    records: list[dict] = []
    found_split_directory = False

    for split in ("train", "val", "test"):
        split_dir = root / split
        if not split_dir.exists():
            continue
        found_split_directory = True
        for image in find_images(split_dir):
            records.append(
                {
                    "router_class": "KolektorSDD2",
                    "source_dataset": "KolektorSDD2",
                    "source_category": "defect_region",
                    "source_split": split,
                    "source_path": str(image.resolve()),
                }
            )

    # Fallback.
    if not found_split_directory:
        for image in find_images(root):
            records.append(
                {
                    "router_class": "KolektorSDD2",
                    "source_dataset": "KolektorSDD2",
                    "source_category": "defect_region",
                    "source_split": "",
                    "source_path": str(image.resolve()),
                }
            )
    return records



# Magnetic Tile


def discover_magnetic_tile(root: Path) -> list[dict]:
    records: list[dict] = []

    expected_categories = {
        normalize_name("MT_Blowhole"): "Blowhole",
        normalize_name("MT_Break"): "Break",
        normalize_name("MT_Crack"): "Crack",
        normalize_name("MT_Fray"): "Fray",
        normalize_name("MT_Uneven"): "Uneven",
        normalize_name("MT_Free"): "Free",
    }

    category_dirs = [
        path
        for path in root.iterdir()
        if path.is_dir() and normalize_name(path.name) in expected_categories
    ]

    if not category_dirs:
        raise RuntimeError(f"No Magnetic Tile categories found in: {root}")

    for category_dir in sorted(category_dirs):
        category = expected_categories[normalize_name(category_dir.name)]
        for image in find_images(category_dir):
            records.append(
                {
                    "router_class": "Magnetic_Tile",
                    "source_dataset": "Magnetic_Tile",
                    "source_category": category,
                    "source_split": "",
                    "source_path": str(image.resolve()),
                }
            )

    return records

# MVTec AD

def discover_mvtec(root: Path) -> list[dict]:
    """
    Each MVTec category is a separate router class because each category
    has its own PatchCore specialist.
    """
    records: list[dict] = []

    for category in MVTec_CATEGORIES:
        category_root = root / category
        if not category_root.exists():
            raise RuntimeError(f"MVTec category missing: {category_root}")

        found_split_directory = False

        for split in ("train", "val", "test"):
            split_root = category_root / split
            if not split_root.exists():
                continue
            found_split_directory = True
            for image in find_images(split_root):
                records.append(
                    {
                        "router_class": f"MVTec_{category}",
                        "source_dataset": "MVTec_AD",
                        "source_category": category,
                        "source_split": split,
                        "source_path": str(image.resolve()),
                    }
                )

        # Fallback.
        if not found_split_directory:
            for image in find_images(category_root):
                records.append(
                    {
                        "router_class": f"MVTec_{category}",
                        "source_dataset": "MVTec_AD",
                        "source_category": category,
                        "source_split": "",
                        "source_path": str(image.resolve()),
                    }
                )

    return records



# Discover all datasets
def discover_all_datasets(data_root: Path) -> pd.DataFrame:
    dagm = find_dataset_root(data_root, ["DAGM"])

    kolektor = find_dataset_root(
        data_root,
        [
            "KolektorSDD2",
            "Kolektor_SDD2",
        ],
    )

    magnetic = find_dataset_root(
        data_root,
        [
            "magnetic_tile",
            "Magnetic_Tile",
            "Magnetic Tile",
        ],
    )

    mvtec = find_dataset_root(
        data_root,
        [
            "MVTec_AD",
            "MVTec AD",
            "mvtec_ad",
        ],
    )

    found = {
        "DAGM": str(dagm) if dagm else None,
        "KolektorSDD2": str(kolektor) if kolektor else None,
        "Magnetic_Tile": str(magnetic) if magnetic else None,
        "MVTec_AD": str(mvtec) if mvtec else None,
    }

    print("\nFound dataset roots:")
    print(json.dumps(found, indent=2))

    missing = [name for name, path in found.items() if path is None]

    if missing:
        raise RuntimeError("Required datasets missing from " f"{data_root}: {missing}")

    records: list[dict] = []

    print("\n[*] Discovering DAGM...")
    records.extend(discover_dagm(dagm))

    print("[*] Discovering KolektorSDD2...")
    records.extend(discover_kolektor(kolektor))

    print("[*] Discovering Magnetic Tile...")
    records.extend(discover_magnetic_tile(magnetic))

    print("[*] Discovering MVTec AD...")
    records.extend(discover_mvtec(mvtec))

    df = pd.DataFrame(records)

    if df.empty:
        raise RuntimeError("No images were discovered.")

    missing_classes = sorted(set(ROUTER_CLASSES) - set(df["router_class"].unique()))

    if missing_classes:
        raise RuntimeError("Router classes without images: " f"{missing_classes}")

    return df



# Remove duplicates
def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates(subset=["source_path"]).reset_index(drop=True)
    removed = before - len(df)
    if removed:
        print(f"[!] Removed {removed} duplicate source images.")
    return df



# Train / validation / test split


def split_dataset(df: pd.DataFrame, val_ratio: float, test_ratio: float, seed: int) -> pd.DataFrame:
    if not 0 < val_ratio < 1:
        raise ValueError("--val-ratio must be between 0 and 1.")
    if not 0 < test_ratio < 1:
        raise ValueError("--test-ratio must be between 0 and 1.")
    if val_ratio + test_ratio >= 1:
        raise ValueError("--val-ratio + --test-ratio must be < 1.")

    train_df, temp_df = train_test_split(
        df, test_size=val_ratio + test_ratio, stratify=df["router_class"], random_state=seed
    )

    relative_test_ratio = test_ratio / (val_ratio + test_ratio)

    val_df, test_df = train_test_split(
        temp_df, test_size=relative_test_ratio, stratify=temp_df["router_class"], random_state=seed
    )

    return pd.concat(
        [
            train_df.assign(split="train"),
            val_df.assign(split="val"),
            test_df.assign(split="test"),
        ],
        ignore_index=True,
    )



# Build YOLO classification dataset
def transfer_file(source: Path, destination: Path, mode: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists() or destination.is_symlink():
        destination.unlink()

    if mode == "symlink":
        destination.symlink_to(source)
    else:
        shutil.copy2(source, destination)


def build_router_dataset(df: pd.DataFrame, output_root: Path, copy_mode: str, overwrite: bool) -> pd.DataFrame:
    if overwrite and output_root.exists():
        print(f"[*] Removing existing output: {output_root}")
        shutil.rmtree(output_root)

    output_root.mkdir(parents=True, exist_ok=True)

    manifest: list[dict] = []

    for index, row in df.reset_index(drop=True).iterrows():
        source = Path(row["source_path"])

        # Prevent filename collisions.
        digest = hashlib.sha1(str(source).encode("utf-8")).hexdigest()[:10]

        filename = f"{index:08d}_{digest}_{source.name}"

        destination = output_root / row["split"] / row["router_class"] / filename

        transfer_file(source, destination, copy_mode)

        manifest.append(
            {
                **row.to_dict(),
                "router_path": str(destination.resolve()),
            }
        )

    return pd.DataFrame(manifest)



# Reporting


def print_summary(df: pd.DataFrame) -> None:
    print("\n")
    print("=" * 80)
    print("ROUTER DATASET SUMMARY")
    print("=" * 80)

    summary = df.groupby(["split", "router_class"]).size().unstack(fill_value=0)

    print(summary.to_string())

    print("\nTotal images per router class:")

    print(df.groupby("router_class").size().sort_values(ascending=False).to_string())

    print(f"\nTotal images: {len(df)}")



# Main


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    data_root = args.data_root.resolve()
    output_root = args.output_root.resolve()

    print(f"[*] Preparing VisionInspect router dataset\n    Source: {data_root}\n    Output: {output_root}\n")

   
    # 1. Discover original datasets
   
    inventory = discover_all_datasets(data_root)

    print("\n[*] Source image inventory:")
    print(
        inventory.groupby(["source_dataset", "router_class"]).size().to_string()
    )

   
    # 2. Remove duplicates
   
    inventory = remove_duplicates(inventory)

   
    # 3. Create stratified split
   
    split_df = split_dataset(
        inventory, val_ratio=args.val_ratio, test_ratio=args.test_ratio, seed=args.seed
    )

   
    # 4. Build router dataset
   
    router_df = build_router_dataset(
        split_df, output_root=output_root, copy_mode=args.copy_mode, overwrite=args.overwrite
    )

   
    # 5. Save manifest
   
    manifest_path = output_root / "manifest.csv"
    router_df.to_csv(manifest_path, index=False)

   
    # 6. Save metadata
   
    metadata = {
        "router_classes": ROUTER_CLASSES,
        "num_classes": len(ROUTER_CLASSES),
        "mvtec_categories": MVTec_CATEGORIES,
        "source_datasets": ["DAGM", "KolektorSDD2", "Magnetic_Tile", "MVTec_AD"],
        "data_root": str(data_root),
        "output_root": str(output_root),
        "val_ratio": args.val_ratio,
        "test_ratio": args.test_ratio,
        "seed": args.seed,
        "copy_mode": args.copy_mode,
        "total_images": len(router_df),
    }

    metadata_path = output_root / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

   
    # 7. Print summary
   
    print_summary(router_df)

    print("\n[*] Files created:")
    print(f"    {manifest_path}")
    print(f"    {metadata_path}")
    print("\n[✓] Router dataset preparation complete.")


if __name__ == "__main__":
    main()