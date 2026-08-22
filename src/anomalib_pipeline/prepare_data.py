"""
Download and verify the MVTec AD dataset.

MVTec AD's native structure per category is already split correctly for
unsupervised anomaly detection -- we do NOT reshuffle it:

    <category>/
        train/
            good/                  <- defect-free images ONLY
        test/
            good/                  <- defect-free test images
            <defect_type_1>/       <- anomalous test images, per defect type
            <defect_type_2>/
            ...
        ground_truth/
            <defect_type_1>/       <- pixel-level binary masks for the above
            <defect_type_2>/

This script does not invent a train/val/test split -- it verifies the
official structure is present and reports per-category counts, since
downstream Anomalib configs expect this exact layout.

Usage:
    python prepare_data.py --data-root ./data/raw --category bottle
    python prepare_data.py --data-root ./data/raw --category all
"""

import argparse
import sys
from pathlib import Path

MVTEC_CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid", "hazelnut",
    "leather", "metal_nut", "pill", "screw", "tile", "toothbrush",
    "transistor", "wood", "zipper",
]

MVTEC_URL = "https://www.mvtec.com/company/research/datasets/mvtec-ad"
# The official MVTec download endpoint has been intermittently returning 404s
# as of early 2026. If it fails, this HuggingFace mirror is a working fallback:
#   https://huggingface.co/datasets/TheoM55/mvtec_all_objects_split
MVTEC_HF_MIRROR = "https://huggingface.co/datasets/TheoM55/mvtec_all_objects_split"


def verify_category(category_root: Path) -> dict:
    """Verify a single category folder matches the expected MVTec AD layout."""
    issues = []
    counts = {"train_good": 0, "test_good": 0, "test_anomalous": 0, "masks": 0}

    train_good = category_root / "train" / "good"
    if not train_good.is_dir():
        issues.append(f"missing {train_good}")
    else:
        counts["train_good"] = len(list(train_good.glob("*.png")))

    test_dir = category_root / "test"
    if not test_dir.is_dir():
        issues.append(f"missing {test_dir}")
    else:
        for sub in test_dir.iterdir():
            if not sub.is_dir():
                continue
            n = len(list(sub.glob("*.png")))
            if sub.name == "good":
                counts["test_good"] += n
            else:
                counts["test_anomalous"] += n

    gt_dir = category_root / "ground_truth"
    if gt_dir.is_dir():
        counts["masks"] = sum(len(list(sub.glob("*.png"))) for sub in gt_dir.iterdir() if sub.is_dir())
    elif counts["test_anomalous"] > 0:
        issues.append(f"missing {gt_dir} despite anomalous test images present")

    return {"counts": counts, "issues": issues}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", type=Path, default=Path("./data/raw"),
                         help="Root directory containing per-category MVTec AD folders")
    parser.add_argument("--category", type=str, default="all",
                         help=f"One of {MVTEC_CATEGORIES} or 'all'")
    args = parser.parse_args()

    if not args.data_root.exists():
        print(f"[!] {args.data_root} does not exist.")
        print(f"    MVTec AD must be downloaded manually (registration required) from:")
        print(f"    {MVTEC_URL}")
        print(f"    If that endpoint 404s (known intermittent issue as of early 2026),")
        print(f"    try the HuggingFace mirror instead: {MVTEC_HF_MIRROR}")
        print(f"    Extract it so that {args.data_root}/<category>/train/good/*.png exists.")
        sys.exit(1)

    categories = MVTEC_CATEGORIES if args.category == "all" else [args.category]
    all_ok = True

    for cat in categories:
        cat_root = args.data_root / cat
        if not cat_root.is_dir():
            print(f"[!] {cat}: not found at {cat_root}")
            all_ok = False
            continue

        result = verify_category(cat_root)
        c = result["counts"]
        status = "OK" if not result["issues"] else "ISSUES"
        print(f"[{status}] {cat:12s} train_good={c['train_good']:4d}  "
              f"test_good={c['test_good']:4d}  test_anomalous={c['test_anomalous']:4d}  "
              f"masks={c['masks']:4d}")
        for issue in result["issues"]:
            print(f"    - {issue}")
            all_ok = False

    if not all_ok:
        sys.exit(1)
    print("\nAll categories verified against the native MVTec AD split.")


if __name__ == "__main__":
    main()
