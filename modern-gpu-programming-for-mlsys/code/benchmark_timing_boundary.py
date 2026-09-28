#!/usr/bin/env python3
"""Demonstrate why a GPU benchmark must declare its timing boundary.

On a CUDA machine this script measures:

1. CUDA-event time around GEMM only.
2. CUDA-event time around GEMM plus ReLU.
3. Synchronized host time around one complete GEMM-plus-ReLU call.

On a machine without CUDA, ``--mode dry-run`` verifies the boundary accounting
with deterministic synthetic durations.
"""

from __future__ import annotations

import argparse
import time
from statistics import median
from typing import Callable, Sequence


def tflops(flops: int, time_us: float) -> float:
    """Convert FLOPs and microseconds to TFLOP/s."""
    return flops / time_us / 1e6


def summarize(samples_us: Sequence[float]) -> dict[str, float]:
    return {
        "median_us": median(samples_us),
        "min_us": min(samples_us),
        "max_us": max(samples_us),
    }


def print_report(
    *,
    mode: str,
    size: int,
    reports: dict[str, dict[str, float]],
) -> None:
    flops = 2 * size * size * size
    print(f"mode: {mode}")
    print(f"problem: 2 * {size}^3 = {flops} FLOP")
    print()
    print(f"{'boundary':<24} {'median_us':>12} {'min_us':>12} {'max_us':>12} {'TFLOP/s':>12}")
    for name, result in reports.items():
        print(
            f"{name:<24} "
            f"{result['median_us']:>12.4f} "
            f"{result['min_us']:>12.4f} "
            f"{result['max_us']:>12.4f} "
            f"{tflops(flops, result['median_us']):>12.4f}"
        )


def run_dry_run(size: int) -> int:
    """Verify boundary accounting without requiring a GPU."""
    reports = {
        "GEMM event only": summarize([100.0]),
        "GEMM+ReLU event": summarize([115.0]),
        "GEMM+ReLU host call": summarize([140.0]),
    }

    # Synthetic numbers are deliberately ordered to expose three nested scopes.
    assert reports["GEMM event only"]["median_us"] < reports["GEMM+ReLU event"]["median_us"]
    assert reports["GEMM+ReLU event"]["median_us"] < reports["GEMM+ReLU host call"]["median_us"]

    flops = 2 * size * size * size
    assert tflops(flops, 100.0) > tflops(flops, 115.0)
    assert tflops(flops, 115.0) > tflops(flops, 140.0)

    print_report(mode="dry-run", size=size, reports=reports)
    print()
    print("Interpretation:")
    print("  GEMM event only excludes ReLU GPU work.")
    print("  GEMM+ReLU event includes both kernels and any stream gap between the events.")
    print("  GEMM+ReLU host call also includes Python dispatch, launches, and the wait.")
    return 0


def run_cuda(warmup: int, samples: int, size: int) -> int:
    try:
        import torch
    except ImportError:
        print("PyTorch is not installed; run --mode dry-run for the accounting check.")
        return 1

    if not torch.cuda.is_available():
        print("CUDA is not available on this machine; run --mode dry-run instead.")
        return 1

    torch.set_float32_matmul_precision("highest")
    torch.manual_seed(0)

    device = "cuda"
    dtype = torch.bfloat16
    a = torch.randn((size, size), device=device, dtype=dtype)
    b = torch.randn((size, size), device=device, dtype=dtype)
    c = torch.empty((size, size), device=device, dtype=dtype)
    output = torch.empty_like(c)

    def gemm_only() -> None:
        torch.mm(a, b, out=c)

    def operation() -> None:
        gemm_only()
        torch.clamp_min(c, 0, out=output)

    # Correctness is checked before timing and does not define the benchmark.
    operation()
    torch.cuda.synchronize()
    expected = torch.mm(a.float(), b.float()).clamp_min(0).to(dtype)
    torch.testing.assert_close(output, expected, rtol=2e-2, atol=1e-2)

    def event_samples(fn: Callable[[], None]) -> list[float]:
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()

        values_us: list[float] = []
        for _ in range(samples):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            fn()
            end.record()
            end.synchronize()
            values_us.append(start.elapsed_time(end) * 1e3)
        return values_us

    def host_samples(fn: Callable[[], None]) -> list[float]:
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()

        values_us: list[float] = []
        for _ in range(samples):
            torch.cuda.synchronize()
            start = time.perf_counter()
            fn()
            torch.cuda.synchronize()
            values_us.append((time.perf_counter() - start) * 1e6)
        return values_us

    reports = {
        "GEMM event only": summarize(event_samples(gemm_only)),
        "GEMM+ReLU event": summarize(event_samples(operation)),
        "GEMM+ReLU host call": summarize(host_samples(operation)),
    }
    print_report(mode="cuda", size=size, reports=reports)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("auto", "cuda", "dry-run"), default="auto")
    parser.add_argument("--size", type=int, default=4096)
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--samples", type=int, default=20)
    args = parser.parse_args()

    if args.mode == "dry-run":
        return run_dry_run(args.size)
    if args.mode == "cuda":
        return run_cuda(args.warmup, args.samples, args.size)

    try:
        import torch
    except ImportError:
        return run_dry_run(args.size)
    if torch.cuda.is_available():
        return run_cuda(args.warmup, args.samples, args.size)
    return run_dry_run(args.size)


if __name__ == "__main__":
    raise SystemExit(main())
