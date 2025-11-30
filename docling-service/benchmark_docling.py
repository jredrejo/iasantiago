# benchmark_docling.py

import time
import httpx
from pathlib import Path


def benchmark_extraction(pdf_path: str, use_gpu: bool = True):
    """Benchmark Docling extraction performance"""

    service_url = "http://localhost:8003" if use_gpu else "http://localhost:8004"

    print(f"\n{'='*60}")
    print(f"Benchmarking: {Path(pdf_path).name}")
    print(f"Mode: {'GPU' if use_gpu else 'CPU'}")
    print(f"{'='*60}\n")

    # Warm-up run
    print("[WARMUP] Running warm-up extraction...")
    with open(pdf_path, "rb") as f:
        files = {"file": (Path(pdf_path).name, f, "application/pdf")}
        httpx.post(f"{service_url}/extract", files=files, timeout=120.0)

    # Actual benchmark (3 runs)
    times = []
    for run in range(3):
        print(f"\n[RUN {run + 1}/3]")

        start = time.time()

        with open(pdf_path, "rb") as f:
            files = {"file": (Path(pdf_path).name, f, "application/pdf")}
            resp = httpx.post(f"{service_url}/extract", files=files, timeout=120.0)

        elapsed = time.time() - start
        times.append(elapsed)

        if resp.status_code == 200:
            data = resp.json()
            stats = data["stats"]
            print(f"  Time: {elapsed:.2f}s")
            print(f"  Elements: {stats['total_elements']}")
            print(f"  Pages: {stats['pages']}")
            print(f"  By type: {stats['by_type']}")
        else:
            print(f"  ERROR: {resp.status_code}")

    # Results
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)

    print(f"\n{'='*60}")
    print(f"RESULTS ({'GPU' if use_gpu else 'CPU'})")
    print(f"{'='*60}")
    print(f"  Average: {avg_time:.2f}s")
    print(f"  Min:     {min_time:.2f}s")
    print(f"  Max:     {max_time:.2f}s")
    print(f"{'='*60}\n")

    return avg_time


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python benchmark_docling.py <pdf_path>")
        sys.exit(1)

    pdf_path = sys.argv[1]

    # Benchmark GPU
    gpu_time = benchmark_extraction(pdf_path, use_gpu=True)

    # Compare
    print("\n" + "=" * 60)
    print("PERFORMANCE SUMMARY")
    print("=" * 60)
    print(f"GPU: {gpu_time:.2f}s")
    print(f"Expected speedup: 3-4x for complex documents")
    print("=" * 60)
