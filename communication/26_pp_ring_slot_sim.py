#!/usr/bin/env python3
"""Simulate PP ring buffer regions and send credit targets."""

from __future__ import annotations


def buffer_offset(src: int, dst: int, num_ranks: int) -> tuple[int, int]:
    nxt = (src + 1) % num_ranks
    return (0, 1) if dst == nxt else (1, 0)


def regions(src: int, dst: int, num_ranks: int, inflight: int, send_count: int):
    local_in_dst, dst_in_local = buffer_offset(src, dst, num_ranks)
    slot = send_count % inflight
    send_region = dst_in_local + 2
    recv_region_on_dst = local_in_dst + 0
    credit_target = send_count - inflight + 1
    arrive_sig = local_in_dst + num_ranks
    release_sig = num_ranks + dst_in_local + 2
    return {
        "pair": buffer_offset(src, dst, num_ranks),
        "slot": slot,
        "send_region": send_region,
        "dst_recv_region": recv_region_on_dst,
        "credit_target": credit_target,
        "arrive_signal": arrive_sig,
        "release_wait_signal": release_sig,
    }


def main() -> None:
    n, inflight = 4, 4
    print("rank2 -> next(3):")
    print(" ", regions(2, 3, n, inflight, send_count=0))
    print(" rank2 -> next, 5th send (count=4):")
    print(" ", regions(2, 3, n, inflight, send_count=4))
    print("rank2 -> prev(1):")
    print(" ", regions(2, 1, n, inflight, send_count=0))


if __name__ == "__main__":
    main()
