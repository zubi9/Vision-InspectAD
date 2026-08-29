"""
Benchmarks the running API's /predict endpoint. Works against either serving mode
(VI_USE_TRITON=false or true) since it just measures what a real client experiences
-- comparing modes is restart-the-API-with-a-different-env-var, then re-run this
against the same image set.

Reports per-model-class latency (grouped by which router_class actually answered,
since a mixed folder of sample images will hit different backends) plus overall
aggregate stats.

Usage:
    python serving/benchmark.py --images ./sample_images --requests 50
    python serving/benchmark.py --images ./sample_images --requests 100 --concurrency 8
    python serving/benchmark.py --url http://localhost:8000 --images ./one_image.png
"""

import argparse
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}


def collect_images(images_arg: Path) -> list[Path]:
    if images_arg.is_file():
        return [images_arg]
    return sorted(p for p in images_arg.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)


def send_one(url: str, image_path: Path) -> dict:
    with open(image_path, "rb") as f:
        content_type = f"image/{image_path.suffix.lstrip('.').replace('jpg', 'jpeg')}"
        start = time.perf_counter()
        response = requests.post(f"{url}/predict", files={"file": (image_path.name, f, content_type)}, timeout=60)
        elapsed_ms = (time.perf_counter() - start) * 1000

    body = response.json() if response.ok else {}
    return {
        "image": image_path.name,
        "status_code": response.status_code,
        "latency_ms": elapsed_ms,
        "router_class": body.get("router_class") or body.get("router_raw_class") or "unknown",
        "source_model": body.get("source_model", "n/a"),
    }


def summarize(latencies_ms: list[float]) -> dict:
    if not latencies_ms:
        return {}
    sorted_lat = sorted(latencies_ms)
    n = len(sorted_lat)
    return {
        "count": n,
        "mean_ms": statistics.mean(sorted_lat),
        "median_ms": statistics.median(sorted_lat),
        "p95_ms": sorted_lat[min(int(n * 0.95), n - 1)],
        "p99_ms": sorted_lat[min(int(n * 0.99), n - 1)],
        "min_ms": sorted_lat[0],
        "max_ms": sorted_lat[-1],
    }


def run_benchmark(url: str, images: list[Path], n_requests: int, concurrency: int, warmup: int) -> list[dict]:
    request_plan = [images[i % len(images)] for i in range(n_requests)]

    if warmup:
        print(f"Warming up ({warmup} requests)...")
        for img in request_plan[:warmup]:
            send_one(url, img)

    print(f"Running {n_requests} requests at concurrency={concurrency}...")
    results = []
    wall_start = time.perf_counter()

    if concurrency <= 1:
        for img in request_plan:
            results.append(send_one(url, img))
    else:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = [pool.submit(send_one, url, img) for img in request_plan]
            for future in as_completed(futures):
                results.append(future.result())

    wall_elapsed = time.perf_counter() - wall_start
    print(f"Total wall time: {wall_elapsed:.2f}s -> throughput: {n_requests / wall_elapsed:.2f} req/s\n")
    return results


def print_report(results: list[dict]):
    failures = [r for r in results if r["status_code"] != 200]
    successes = [r for r in results if r["status_code"] == 200]

    print("=== Overall ===")
    stats = summarize([r["latency_ms"] for r in successes])
    for k, v in stats.items():
        print(f"  {k}: {v:.2f}" if isinstance(v, float) else f"  {k}: {v}")
    if failures:
        print(f"  FAILURES: {len(failures)}/{len(results)}")

    by_class: dict[str, list[float]] = {}
    for r in successes:
        by_class.setdefault(r["router_class"], []).append(r["latency_ms"])

    print("\n=== By router class ===")
    for cls, latencies in sorted(by_class.items()):
        s = summarize(latencies)
        print(f"  {cls:20s} n={s['count']:4d}  mean={s['mean_ms']:7.2f}ms  "
              f"p95={s['p95_ms']:7.2f}ms  p99={s['p99_ms']:7.2f}ms")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--images", type=Path, required=True, help="Single image file or a directory of images")
    parser.add_argument("--requests", type=int, default=50)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=3)
    args = parser.parse_args()

    images = collect_images(args.images)
    if not images:
        print(f"No images found at {args.images}")
        return

    results = run_benchmark(args.url, images, args.requests, args.concurrency, args.warmup)
    print_report(results)


if __name__ == "__main__":
    main()
