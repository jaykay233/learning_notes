#!/usr/bin/env python3
"""Route Engram global entry indices to RDMA peers (hybrid vs flat)."""

from __future__ import annotations


def route(global_idx: int, entries_per_rank: int, ranks_per_peer: int):
    owner = global_idx // entries_per_rank
    local = global_idx % entries_per_rank
    peer = owner // ranks_per_peer
    intra = owner % ranks_per_peer
    return owner, local, peer, intra


def main() -> None:
    entries_per_rank = 1000
    # Hybrid: 2 scaleout × 4 scaleup
    print("hybrid (peer=scaleout, ranks_per_peer=4):")
    for idx in (0, 999, 1000, 4500, 7999):
        print(f"  entry {idx:4d} -> owner/local/peer/intra={route(idx, entries_per_rank, 4)}")

    print("flat (peer=rank, ranks_per_peer=1):")
    for idx in (0, 1000, 4500):
        print(f"  entry {idx:4d} -> owner/local/peer/intra={route(idx, entries_per_rank, 1)}")


if __name__ == "__main__":
    main()
