"""
benchmark.py — CPU vs GPU Inference Benchmark
RoadSense AI | SwasthikaSelvakumar
Run this LOCALLY after you have best.pt to generate your paper's benchmark table.
"""

import torch
import time
import numpy as np
from PIL import Image
import requests
from io import BytesIO
from ultralytics import YOLO
import json

# ─────────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────────
MODEL_PATH = "best.pt"
N_RUNS     = 50
IMG_URL    = "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3c/Pothole_in_India.jpg/640px-Pothole_in_India.jpg"

def download_test_image():
    print("Downloading test image...")
    img = Image.open(BytesIO(requests.get(IMG_URL).content)).convert("RGB")
    img.save("test_image.jpg")
    print("Test image saved: test_image.jpg")
    return img

def run_benchmark(model, device_str, use_fp16=False, n_runs=N_RUNS):
    """Run inference N times and return timing stats."""
    times = []
    is_cuda = (device_str != 'cpu')

    # Warm up
    warm_runs = 5 if is_cuda else 2
    for _ in range(warm_runs):
        model.predict("test_image.jpg",
                      device=0 if is_cuda else 'cpu',
                      half=use_fp16, verbose=False)
    if is_cuda:
        torch.cuda.synchronize()

    # Benchmark
    for _ in range(n_runs):
        if is_cuda:
            torch.cuda.synchronize()
        start = time.perf_counter()
        model.predict("test_image.jpg",
                      device=0 if is_cuda else 'cpu',
                      half=use_fp16, verbose=False)
        if is_cuda:
            torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000)

    return {
        "mean_ms"   : round(np.mean(times), 2),
        "std_ms"    : round(np.std(times), 2),
        "min_ms"    : round(min(times), 2),
        "max_ms"    : round(max(times), 2),
        "p95_ms"    : round(np.percentile(times, 95), 2),
        "fps"       : round(1000 / np.mean(times), 1),
    }

def get_gpu_memory():
    if not torch.cuda.is_available():
        return {}
    return {
        "gpu_name"         : torch.cuda.get_device_name(0),
        "total_memory_gb"  : round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2),
        "allocated_mb"     : round(torch.cuda.memory_allocated(0) / 1e6, 1),
        "reserved_mb"      : round(torch.cuda.memory_reserved(0) / 1e6, 1),
    }

def print_table(results: dict):
    w = 70
    print("\n" + "="*w)
    print("   ROADSENSE AI — CUDA BENCHMARK RESULTS")
    print("   (Copy this table into your paper)")
    print("="*w)
    print(f"{'Metric':<30} {'CPU':>10} {'GPU FP32':>10} {'GPU FP16':>10}")
    print("-"*w)

    metrics = [
        ("Mean latency (ms)",   "mean_ms"),
        ("Std deviation (ms)",  "std_ms"),
        ("Min latency (ms)",    "min_ms"),
        ("P95 latency (ms)",    "p95_ms"),
        ("Throughput (FPS)",    "fps"),
    ]
    for label, key in metrics:
        cpu_val = results.get("cpu", {}).get(key, "—")
        f32_val = results.get("gpu_fp32", {}).get(key, "—")
        f16_val = results.get("gpu_fp16", {}).get(key, "—")
        print(f"{label:<30} {str(cpu_val):>10} {str(f32_val):>10} {str(f16_val):>10}")

    print("-"*w)

    if "cpu" in results and "gpu_fp16" in results:
        speedup_fp16 = round(results["cpu"]["mean_ms"] / results["gpu_fp16"]["mean_ms"], 1)
        speedup_fp32 = round(results["cpu"]["mean_ms"] / results["gpu_fp32"]["mean_ms"], 1)
        print(f"{'GPU FP32 speedup over CPU':<30} {speedup_fp32:>10}x")
        print(f"{'GPU FP16 speedup over CPU':<30} {speedup_fp16:>10}x")

    print("="*w)

    # GPU memory
    gpu_mem = get_gpu_memory()
    if gpu_mem:
        print("\n  GPU Memory Profile:")
        print(f"  Device      : {gpu_mem.get('gpu_name')}")
        print(f"  Total       : {gpu_mem.get('total_memory_gb')} GB")
        print(f"  Allocated   : {gpu_mem.get('allocated_mb')} MB")
        print(f"  Reserved    : {gpu_mem.get('reserved_mb')} MB")

    print("="*w)
    print("\n  CUDA Optimizations Applied:")
    print("  ✅ FP16 Mixed Precision (half-precision inference)")
    print("  ✅ cuDNN Benchmark Mode (auto-optimized kernels)")
    print("  ✅ CUDA Synchronization (accurate GPU timing)")
    print("  ✅ GPU Warm-up (eliminates first-call overhead)")
    print("="*w)


def main():
    print("="*70)
    print("  RoadSense AI — CPU vs GPU Benchmark")
    print("="*70)

    # Check CUDA
    cuda_ok = torch.cuda.is_available()
    print(f"\nCUDA available : {cuda_ok}")
    if cuda_ok:
        torch.backends.cudnn.benchmark = True
        print(f"GPU device     : {torch.cuda.get_device_name(0)}")
        print(f"GPU memory     : {torch.cuda.get_device_properties(0).total_memory/1e9:.2f} GB")
        print(f"CUDA version   : {torch.version.cuda}")
    print(f"Running {N_RUNS} inference passes per config...\n")

    download_test_image()
    model = YOLO(MODEL_PATH)

    results = {}

    # CPU
    print("[1/3] Benchmarking CPU...")
    results["cpu"] = run_benchmark(model, "cpu", use_fp16=False)
    print(f"      CPU mean: {results['cpu']['mean_ms']} ms")

    if cuda_ok:
        # GPU FP32
        print("[2/3] Benchmarking GPU FP32...")
        results["gpu_fp32"] = run_benchmark(model, "cuda", use_fp16=False)
        print(f"      GPU FP32 mean: {results['gpu_fp32']['mean_ms']} ms")

        # GPU FP16
        print("[3/3] Benchmarking GPU FP16 (Mixed Precision)...")
        results["gpu_fp16"] = run_benchmark(model, "cuda", use_fp16=True)
        print(f"      GPU FP16 mean: {results['gpu_fp16']['mean_ms']} ms")
    else:
        print("[2/3] Skipping GPU benchmarks (no CUDA)")
        print("[3/3] Skipping GPU benchmarks (no CUDA)")

    print_table(results)

    # Save to JSON for paper
    with open("benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to benchmark_results.json")
    print("Screenshot the table above for your paper!")


if __name__ == "__main__":
    main()
