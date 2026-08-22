"""
Build data/router_source/, the unified root prepare_data.py's dataset discovery
expects (one folder per dataset, matched by normalized name), from this project's
existing data layout:

    data/raw/<mvtec category>/...          -> data/router_source/MVTec_AD/<category>/...
    data/raw_supervised/dagm/...            -> data/router_source/DAGM/...
    data/raw_supervised/magnetic_tiles/...  -> data/router_source/Magnetic_Tile/...
    data/raw_supervised/kolektor_sdd2/...   -> data/router_source/KolektorSDD2/...

Uses symlinks so nothing is duplicated on disk. Safe to re-run -- existing symlinks
are replaced, not duplicated.

Usage:
    python link_datasets.py
    python link_datasets.py --data-root ./data --output-root ./data/router_source
"""

import argparse
from pathlib import Path
from src.router_pipeline import config


def link_dataset(source: Path, dest: Path) -> bool:
    if not source.is_dir():
        print(f"[!] {source} not found -- skipping (expected if not downloaded yet)")
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_symlink() or dest.exists():
        dest.unlink() if dest.is_symlink() else None
    dest.symlink_to(source.resolve())
    print(f"[OK] {dest} -> {source.resolve()}")
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # use data-root defined in config.py as default, but allow override via CLI for testing
    parser.add_argument("--data-root", type=Path, default=Path(config.RAW_DATA_ROOT), help="Root of the existing data layout")
    parser.add_argument("--output-root", type=Path, default=Path(config.CLS_ROUTER_DATA_ROOT), help="Root of the router_source layout to create")
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)

    mapping = {
        args.data_root / "raw": args.output_root / "MVTec_AD",
        args.data_root / "raw_supervised" / "dagm": args.output_root / "DAGM",
        args.data_root / "raw_supervised" / "magnetic_tiles": args.output_root / "Magnetic_Tile",
        args.data_root / "raw_supervised" / "kolektor_sdd2": args.output_root / "KolektorSDD2",
    }

    n_linked = sum(link_dataset(src, dst) for src, dst in mapping.items())
    print(f"\n{n_linked}/{len(mapping)} datasets linked into {args.output_root}")
    if n_linked < len(mapping):
        print("Missing datasets will cause prepare_data.py to raise -- fill them in under "
              "data/raw_supervised/ before running it.")


if __name__ == "__main__":
    main()
