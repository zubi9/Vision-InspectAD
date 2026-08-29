"""
Download pretrained model weights from the Hugging Face Hub model repo, placing
each into the exact local path model_registry.py / paths.py already expect --
nothing else needs to change after running this.

Checks each file for a newer version before downloading (HTTP ETag comparison
against a local manifest, models/.hf_manifest.json), then prompts before
downloading -- interactively at a real terminal, automatically (yes) when not
attached to one (CI runners, Docker entrypoints, etc.) or when --yes is passed.
hf_hub_download() has its own internal staleness check too (via local_dir's
.cache/huggingface/ bookkeeping), but that skips silently -- the point of the
ETag check here is to actually tell you a newer version exists and let you decide.

HF repo ID must be set via --repo-id or the VI_HF_REPO_ID env var -- no default
baked in here since no actual repo link was provided when this script was
written. Fill it in via either of those once you have it.

Usage:
    python scripts/download_models.py --repo-id your-username/visioninspect-models
    VI_HF_REPO_ID=your-username/visioninspect-models python scripts/download_models.py
    python scripts/download_models.py --repo-id ... --yes           # non-interactive
    python scripts/download_models.py --repo-id ... --only router   # just one file
"""

import argparse
import json
import os
import sys
from pathlib import Path

import requests
from huggingface_hub import hf_hub_download

from src.common import paths

MANIFEST_PATH = paths.PROJECT_ROOT / "models" / ".hf_manifest.json"

# Remote filename (exactly as it exists in the HF repo) -> local destination path.
# Anomalib filenames already match the local naming convention 1:1; the YOLO
# specialists/router don't (flat "<name>-best.onnx" in the repo vs. this project's
# nested models/<name>/weights/best.onnx layout), so those need an explicit rename
# after download. "kolekor-best.onnx" is spelled that way in the source repo --
# kept exactly as given, not "corrected", since it has to match the real filename.
HF_TO_LOCAL = {
    **{
        f"patchcore_{cat}.onnx": paths.ANOMALIB_ONNX_DIR / f"patchcore_{cat}.onnx"
        for cat in paths.MVTEC_CATEGORIES
    },
    "dagm-best.onnx": paths.DAGM_ONNX_PATH,
    "kolekor-best.onnx": paths.KOLEKTOR_ONNX_PATH,
    "magnetic_tile-best.onnx": paths.MAGNETIC_TILE_ONNX_PATH,
    "router-best.onnx": paths.ROUTER_ONNX_PATH,
}


def load_manifest() -> dict:
    if MANIFEST_PATH.exists():
        return json.loads(MANIFEST_PATH.read_text())
    return {}


def save_manifest(manifest: dict):
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))


def remote_etag(repo_id: str, filename: str, revision: str) -> str | None:
    url = f"https://huggingface.co/{repo_id}/resolve/{revision}/{filename}"
    try:
        resp = requests.head(url, allow_redirects=True, timeout=10)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"  [!] Could not check {filename}: {e}")
        return None
    # LFS files report their real content hash via x-linked-etag; small non-LFS
    # files only have a regular etag.
    return resp.headers.get("x-linked-etag") or resp.headers.get("etag")


def should_download(filename: str, local_path: Path, remote_tag: str | None, manifest: dict, auto_yes: bool) -> bool:
    if not local_path.exists():
        return True
    if remote_tag is None:
        print(f"  [!] Couldn't verify remote version for {filename} -- skipping (local file already exists).")
        return False
    if manifest.get(filename) == remote_tag:
        print(f"  [=] {filename}: up to date, skipping.")
        return False

    print(f"  [>] {filename}: newer version available on the Hub.")
    if auto_yes:
        return True
    answer = input("      Download and overwrite local copy? [Y/n] ").strip().lower()
    return answer in ("", "y", "yes")


def download_all(repo_id: str, revision: str, only: list[str] | None, auto_yes: bool):
    manifest = load_manifest()

    targets = HF_TO_LOCAL
    if only:
        targets = {k: v for k, v in HF_TO_LOCAL.items() if k in only or v.stem in only or any(o in k for o in only)}
    if not targets:
        print(f"No matching files for --only {only}. Known filenames: {list(HF_TO_LOCAL)}")
        return

    for filename, local_path in targets.items():
        print(f"\n{filename} -> {local_path}")
        tag = remote_etag(repo_id, filename, revision)

        if not should_download(filename, local_path, tag, manifest, auto_yes):
            continue

        local_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            downloaded_path = Path(hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                revision=revision,
                local_dir=str(local_path.parent),
            ))
        except Exception as e:  # noqa: BLE001 -- surface a clear per-file failure, keep going with the rest
            print(f"  [FAILED] {filename}: {e}")
            continue

        # hf_hub_download always keeps the repo's own filename -- rename to this
        # project's expected local name when they differ (e.g. "dagm-best.onnx"
        # downloaded, but paths.py expects "models/dagm/weights/best.onnx").
        if downloaded_path != local_path:
            downloaded_path.replace(local_path)

        if tag:
            manifest[filename] = tag
        print(f"  [OK] {local_path}")

    save_manifest(manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo-id", default=os.environ.get("VI_HF_REPO_ID"),
                         help="Hugging Face repo ID, e.g. 'username/visioninspect-models'. "
                              "Can also be set via the VI_HF_REPO_ID env var.")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--only", nargs="*", default=None,
                         help="Only these filenames or bare model names (e.g. 'router', 'bottle')")
    parser.add_argument("--yes", "-y", action="store_true", help="Never prompt, always download newer files")
    args = parser.parse_args()

    if not args.repo_id:
        print("No --repo-id given and VI_HF_REPO_ID is not set -- pass the Hugging Face repo ID explicitly.")
        sys.exit(1)

    # Auto-yes if explicitly requested, or if there's no real terminal to prompt at
    # -- an unattended input() call would just hang until the CI job times out.
    auto_yes = args.yes or not sys.stdin.isatty() or os.environ.get("CI", "").lower() == "true"
    if auto_yes and not args.yes:
        print("Non-interactive environment detected -- running with --yes behavior.")

    download_all(args.repo_id, args.revision, args.only, auto_yes)


if __name__ == "__main__":
    main()
