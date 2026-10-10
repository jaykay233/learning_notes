#!/usr/bin/env python3
"""Simulate EPLB logical→physical remap vs naive logical dispatch."""

from __future__ import annotations


def rank_of(expert_id: int, experts_per_rank: int) -> int:
    return expert_id // experts_per_rank


def main() -> None:
    # 4 logical experts, +4 redundant → 8 physical slots, 4 ranks → E=2
    experts_per_rank = 2
    # physical slot -> logical expert (hot logical 0 replicated on slots 0 and 4)
    phy2log = [0, 1, 2, 3, 0, 1, 2, 3]
    # pick first physical replica for each logical
    log2phy = {}
    for phy, log in enumerate(phy2log):
        log2phy.setdefault(log, phy)

    logical_topk = [0, 3]  # gate output
    print("naive: pass logical ids to DeepEP (WRONG after move):")
    for lid in logical_topk:
        print(f"  logical {lid} -> rank {rank_of(lid, experts_per_rank)} (thinks weight at old place)")

    print("framework: remap then DeepEP (CORRECT):")
    for lid in logical_topk:
        pid = log2phy[lid]
        print(f"  logical {lid} -> physical {pid} -> rank {rank_of(pid, experts_per_rank)}")

    print("physical layout still symmetric: ranks",
          [rank_of(p, experts_per_rank) for p in range(len(phy2log))])


if __name__ == "__main__":
    main()
