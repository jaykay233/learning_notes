#!/usr/bin/env python3
"""Compare a static tier split with KVStream-style completion routing."""

from __future__ import annotations

import argparse
import heapq
from dataclasses import dataclass
from typing import Sequence

MIB = 1024 * 1024
GBPS = 1_000_000_000


@dataclass(frozen=True)
class StepResult:
    nvme_gbps: float
    pfs_gbps: float
    lower_bound_s: float
    static_s: float
    dynamic_s: float
    dynamic_nvme_share: float


def static_split_time(
    total_chunks: int,
    chunk_bytes: int,
    nvme_bps: float,
    pfs_bps: float,
    nvme_share: float,
) -> float:
    nvme_chunks = round(total_chunks * nvme_share)
    pfs_chunks = total_chunks - nvme_chunks
    nvme_time = nvme_chunks * chunk_bytes / nvme_bps
    pfs_time = pfs_chunks * chunk_bytes / pfs_bps
    return max(nvme_time, pfs_time)


def completion_routed_time(
    total_chunks: int,
    chunk_bytes: int,
    nvme_bps: float,
    pfs_bps: float,
    batch_chunks: int,
) -> tuple[float, float]:
    """Route from both ends; a tier claims more after each batch completes."""
    remaining = total_chunks
    next_batch = 0
    served = {0: 0, 1: 0}  # 0: NVMe, 1: PFS
    busy = {0: False, 1: False}
    finish_time = 0.0
    events: list[tuple[float, int, int, int]] = []

    def claim(tier: int, now: float) -> None:
        nonlocal remaining, next_batch
        if remaining == 0:
            return
        count = min(batch_chunks, remaining)
        remaining -= count
        served[tier] += count
        busy[tier] = True
        bandwidth = nvme_bps if tier == 0 else pfs_bps
        duration = count * chunk_bytes / bandwidth
        heapq.heappush(events, (now + duration, tier, next_batch, count))
        next_batch += 1

    claim(0, 0.0)
    claim(1, 0.0)

    while events:
        now, tier, _, _ = heapq.heappop(events)
        finish_time = max(finish_time, now)
        busy[tier] = False
        claim(tier, now)

        if remaining == 0 and not any(busy.values()):
            break

    nvme_share = served[0] / total_chunks
    return finish_time, nvme_share


def run(
    steps: Sequence[tuple[float, float]],
    total_bytes: int,
    chunk_bytes: int,
    nvme_share: float,
    batch_chunks: int,
) -> list[StepResult]:
    if total_bytes % chunk_bytes:
        raise ValueError("total_bytes must be divisible by chunk_bytes")
    total_chunks = total_bytes // chunk_bytes
    results: list[StepResult] = []

    for nvme_gbps, pfs_gbps in steps:
        nvme_bps = nvme_gbps * GBPS
        pfs_bps = pfs_gbps * GBPS
        bound = total_bytes / (nvme_bps + pfs_bps)
        static = static_split_time(
            total_chunks, chunk_bytes, nvme_bps, pfs_bps, nvme_share
        )
        dynamic, dynamic_share = completion_routed_time(
            total_chunks, chunk_bytes, nvme_bps, pfs_bps, batch_chunks
        )
        results.append(
            StepResult(
                nvme_gbps=nvme_gbps,
                pfs_gbps=pfs_gbps,
                lower_bound_s=bound,
                static_s=static,
                dynamic_s=dynamic,
                dynamic_nvme_share=dynamic_share,
            )
        )
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-mib", type=int, default=1024)
    parser.add_argument("--chunk-mib", type=int, default=16)
    parser.add_argument("--static-nvme-share", type=float, default=0.70)
    parser.add_argument("--batch-chunks", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    steps = (
        (12.0, 6.0),
        (8.0, 2.0),
        (5.0, 5.0),
        (10.0, 1.0),
        (4.0, 7.0),
    )
    results = run(
        steps=steps,
        total_bytes=args.total_mib * MIB,
        chunk_bytes=args.chunk_mib * MIB,
        nvme_share=args.static_nvme_share,
        batch_chunks=args.batch_chunks,
    )

    print(
        "step | NVMe GB/s | PFS GB/s | bound(s) | static(s) | "
        "routed(s) | routed NVMe share"
    )
    print(
        "-----+-----------+----------+----------+-----------+"
        "-----------+------------------"
    )
    for index, result in enumerate(results, start=1):
        print(
            f"{index:4d} | {result.nvme_gbps:9.1f} | {result.pfs_gbps:8.1f} | "
            f"{result.lower_bound_s:8.4f} | {result.static_s:9.4f} | "
            f"{result.dynamic_s:10.4f} | {result.dynamic_nvme_share:17.3f}"
        )

    static_total = sum(item.static_s for item in results)
    dynamic_total = sum(item.dynamic_s for item in results)
    bound_total = sum(item.lower_bound_s for item in results)
    print()
    print(f"static total:  {static_total:.4f} s")
    print(f"routed total:  {dynamic_total:.4f} s")
    print(f"bound total:   {bound_total:.4f} s")
    print(f"static/routed: {static_total / dynamic_total:.3f}x")
    print(f"routed/bound:  {dynamic_total / bound_total:.3f}x")


if __name__ == "__main__":
    main()
