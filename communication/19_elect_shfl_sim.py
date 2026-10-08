#!/usr/bin/env python3
"""Simulate elect+shfl lane0 hazard vs lane0-writer pattern."""

from __future__ import annotations


WARP = 32


def shfl_sync(regs: list[int], src_lane: int) -> list[int]:
    src = regs[src_lane]
    return [src for _ in regs]


def dangerous_elect_then_shfl0(true_value: int, elected_lane: int) -> list[int]:
    """if (elect) x = true; x = shfl(x, 0)"""
    regs = [0] * WARP
    regs[elected_lane] = true_value
    return shfl_sync(regs, 0)


def safe_lane0_then_shfl0(true_value: int) -> list[int]:
    """lane0 ? true : 0; shfl(..., 0)"""
    regs = [0] * WARP
    regs[0] = true_value
    return shfl_sync(regs, 0)


def safe_elect_then_shfl_elected(true_value: int, elected_lane: int) -> list[int]:
    """if (elect) x = true; x = shfl(x, elected)"""
    regs = [0] * WARP
    regs[elected_lane] = true_value
    return shfl_sync(regs, elected_lane)


def main() -> None:
    true_value = 42

    # Case A: elect happens to be lane 0 — dangerous pattern looks OK
    a = dangerous_elect_then_shfl0(true_value, elected_lane=0)
    assert a == [42] * WARP, a

    # Case B: elect is lane 7 — dangerous pattern broadcasts 0
    b = dangerous_elect_then_shfl0(true_value, elected_lane=7)
    assert b == [0] * WARP, b
    print(f"dangerous elect=7 -> broadcast {b[0]} (BUG)")

    # Case C: LL-style lane0 writer — always OK
    c = safe_lane0_then_shfl0(true_value)
    assert c == [42] * WARP, c
    print(f"safe lane0 writer -> broadcast {c[0]}")

    # Case D: elect + shfl(elected) — always OK
    d = safe_elect_then_shfl_elected(true_value, elected_lane=7)
    assert d == [42] * WARP, d
    print(f"safe elect=7 shfl(7) -> broadcast {d[0]}")

    print("all checks passed")


if __name__ == "__main__":
    main()
