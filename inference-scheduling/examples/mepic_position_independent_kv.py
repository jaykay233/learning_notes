#!/usr/bin/env python3
"""CPU-only trace of MEPIC reuse exactness and KV block occupancy.

The script models one reusable chunk referenced by several concurrent
requests. It does not execute attention. The vector example isolates two
different effects:

1. fused RoPE can correct the cached key's absolute position;
2. fused RoPE cannot recover K/V changes caused by a different left context.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, cos, sin


Vector = tuple[float, float]


@dataclass(frozen=True)
class ReusePlan:
    chunk_tokens: int
    block_tokens: int
    active_requests: int

    def __post_init__(self) -> None:
        if min(self.chunk_tokens, self.block_tokens, self.active_requests) <= 0:
            raise ValueError("all dimensions must be positive")
        if self.active_requests < 2:
            raise ValueError("the shared-reuse case needs at least two requests")

    @property
    def blocks_per_chunk(self) -> int:
        return ceil(self.chunk_tokens / self.block_tokens)

    @property
    def full_recompute_blocks(self) -> int:
        return self.active_requests * self.blocks_per_chunk

    @property
    def mepic_blocks(self) -> int:
        # One request-specific first block per request, plus one shared tail.
        return self.active_requests + self.blocks_per_chunk - 1

    @property
    def reduction(self) -> float:
        return self.full_recompute_blocks / self.mepic_blocks

    @property
    def savings_ratio(self) -> float:
        return 1.0 - self.mepic_blocks / self.full_recompute_blocks

    def exact_equivalent(self) -> bool:
        # Reusing KV computed under another left context is approximate.
        return self.blocks_per_chunk == 1


def add(a: Vector, b: Vector) -> Vector:
    return a[0] + b[0], a[1] + b[1]


def scale(a: Vector, factor: float) -> Vector:
    return a[0] * factor, a[1] * factor


def contextual_key(token_value: int, left_context: tuple[int, ...]) -> Vector:
    """Synthetic K before RoPE: token content plus left-context contribution."""
    token = (float(token_value), float(token_value * 2 + 1))
    context = (0.0, 0.0)
    for value in left_context:
        context = add(context, (float(value), float(value + 1)))
    return add(token, scale(context, 0.05))


def apply_rope(key: Vector, position: int) -> Vector:
    angle = position * 0.017
    c, s = cos(angle), sin(angle)
    return key[0] * c - key[1] * s, key[0] * s + key[1] * c


def l1_distance(a: Vector, b: Vector) -> float:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def main() -> None:
    plan = ReusePlan(chunk_tokens=256, block_tokens=16, active_requests=8)
    print("== KV block occupancy ==")
    print(f"blocks_per_chunk={plan.blocks_per_chunk}")
    print(f"full_recompute_blocks={plan.full_recompute_blocks}")
    print(f"mepic_blocks={plan.mepic_blocks}")
    print(f"reduction={plan.reduction:.6f}x")
    print(f"hbm_savings={plan.savings_ratio:.2%}")
    print(f"numerically_exact={plan.exact_equivalent()}")

    print("\n== Exactness boundary ==")
    cases = (
        ("first encounter: recompute every block", True),
        ("reuse: first block recomputed, tail shared", False),
        ("all blocks recomputed under current context", True),
    )
    for name, exact in cases:
        print(f"{name}: exact={exact}")

    # Synthetic trace: keep the stale tail K, then rotate it to the new position.
    token_value = 37
    old_context = (1, 2, 3)
    current_context = (2, 5, 7, 11)
    old_position = 16
    new_position = 96

    exact_key = apply_rope(contextual_key(token_value, current_context), new_position)
    position_only = apply_rope(contextual_key(token_value, current_context), old_position)
    position_only = apply_rope(position_only, new_position - old_position)
    reused_tail = apply_rope(contextual_key(token_value, old_context), new_position)

    print("\n== Synthetic key trace ==")
    print(f"exact_vs_position_only_error={l1_distance(exact_key, position_only):.6f}")
    print(f"exact_vs_reused_tail_error={l1_distance(exact_key, reused_tail):.6f}")
    print("position correction removes the RoPE offset only")
    print("the reused tail still carries the old left-context contribution")


if __name__ == "__main__":
    main()
