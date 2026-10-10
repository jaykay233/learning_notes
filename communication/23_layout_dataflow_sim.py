#!/usr/bin/env python3
"""Show how the same arrivals become rank-major rows vs expert-major rows."""

from __future__ import annotations


def main() -> None:
    # Arrivals at dst rank2 (local experts 0,1 = global 4,5)
    # Each record: (src_rank, src_token, local_experts_hit)
    arrivals = [
        (0, 5, [0]),       # token5 hits local expert0 only
        (0, 17, [0, 1]),   # token17 hits both local experts
        (1, 3, [1]),
    ]
    num_channels = 2

    print("=== Rank-major (dedupe per src token; LocalMap explicit) ===")
    # Fake: channel = src_token % num_channels; order by src, then channel, then t
    rows = sorted(arrivals, key=lambda a: (a[0], a[1] % num_channels, a[1]))
    for r, (src, t, locals_) in enumerate(rows):
        local_map = (locals_ + [-1, -1])[:2]
        print(f"  r={r}: src={src} recv_src_idx={t} recv_topk_idx={local_map}")

    print("=== Expert-major (one row per (token, expert); LocalMap = axis0) ===")
    for e in range(2):
        j = 0
        for src, t, locals_ in arrivals:
            if e not in locals_:
                continue
            print(f"  e={e} j={j}: src_info={t}  (from src_rank={src})")
            j += 1
        print(f"  layout count for e={e}: {j}")


if __name__ == "__main__":
    main()
