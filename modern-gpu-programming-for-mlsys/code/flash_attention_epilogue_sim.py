#!/usr/bin/env python3
"""Simulate the FA4 epilogue math: row_sum -> normalize -> fp16 O_smem."""

from __future__ import annotations

import math
import struct
from typing import Any


def f32(value: float) -> float:
    """Round a Python float to IEEE-754 binary32."""
    return struct.unpack("<f", struct.pack("<f", value))[0]


def f16(value: float) -> float:
    """Round a Python float to IEEE-754 binary16."""
    return struct.unpack("<e", struct.pack("<e", value))[0]


def rcp_approx_ftz(value: float) -> float:
    """Reference for PTX rcp.approx.ftz.f32.

    Python has no reciprocal-approximation instruction, so this function uses
    1 / value as the reference result. On Blackwell the PTX instruction is a
    single approximate reciprocal with flush-to-zero handling.
    """
    if value == 0.0:
        return math.inf
    return f32(1.0 / value)


def online_update(
    row_max_old: float,
    row_sum_old: float,
    o_old: list[float],
    s_block: list[float],
    v_block: list[list[float]],
    scale_log2: float = 1.0,
    rescale_threshold: float = 8.0,
) -> tuple[float, list[float], dict[str, Any]]:
    """Apply one FA4 conditional-rescaling update.

    The reference implementation uses base-2 exponentials:
        P = 2 ** ((S - new_ref) * scale_log2)
    """
    candidate_max = max(s_block)
    delta = (row_max_old - candidate_max) * scale_log2

    if delta >= -rescale_threshold:
        new_ref = row_max_old
        acc_scale = 1.0
        kept_old_reference = True
    else:
        new_ref = candidate_max
        acc_scale = 2.0**delta
        kept_old_reference = False

    p = [2.0 ** ((score - new_ref) * scale_log2) for score in s_block]
    block_row_sum = sum(p)
    block_o = [
        sum(p[k] * v_block[k][d] for k in range(len(p)))
        for d in range(len(v_block[0]))
    ]

    row_sum_new = row_sum_old * acc_scale + block_row_sum
    o_new = [
        o_old[d] * acc_scale + block_o[d]
        for d in range(len(o_old))
    ]
    trace = {
        "candidate_max": candidate_max,
        "delta": delta,
        "new_ref": new_ref,
        "acc_scale": acc_scale,
        "kept_old_reference": kept_old_reference,
        "p": p,
        "block_row_sum": block_row_sum,
        "block_o": block_o,
    }
    return row_sum_new, o_new, trace


def normalize_row(
    o_row: list[float],
    row_sum: float,
) -> tuple[list[float], float, bool]:
    """Apply the FA4 epilogue normalization and zero/NaN guard."""
    zero_or_nan = (row_sum == 0.0) or math.isnan(row_sum)
    selected = 1.0 if zero_or_nan else row_sum
    norm_scale = rcp_approx_ftz(selected)
    normalized = [f32(value * norm_scale) for value in o_row]
    return normalized, norm_scale, zero_or_nan


def cast_row_to_f16(o_row: list[float]) -> list[float]:
    """Simulate cast_f32x2_f16x2 for one row."""
    return [f16(value) for value in o_row]


def print_epilogue(title: str, row_sum: float, o_row: list[float]) -> None:
    normalized, norm_scale, zero_or_nan = normalize_row(o_row, row_sum)
    o_f16 = cast_row_to_f16(normalized)
    print(f"== {title} ==")
    print(f"row_sum        = {row_sum!r}")
    print(f"O              = {o_row}")
    print(f"zero_or_nan    = {zero_or_nan}")
    print(f"norm_scale     = {norm_scale}")
    print(f"O * norm_scale = {normalized}")
    print(f"O_smem (fp16)  = {o_f16}")
    print()


def print_update(title: str, s_block: list[float]) -> None:
    row_sum_new, o_new, trace = online_update(
        row_max_old=2.0,
        row_sum_old=3.0,
        o_old=[4.0, 6.0],
        s_block=s_block,
        v_block=[[1.0, 0.0], [0.0, 1.0]],
    )
    print(f"== {title} ==")
    print(f"S                 = {s_block}")
    print(f"candidate_max     = {trace['candidate_max']}")
    print(f"delta             = {trace['delta']}")
    print(f"new_ref           = {trace['new_ref']}")
    print(f"acc_scale         = {trace['acc_scale']}")
    print(f"P                 = {trace['p']}")
    print(f"block_row_sum     = {trace['block_row_sum']}")
    print(f"block_O           = {trace['block_o']}")
    print(f"row_sum_new       = {row_sum_new}")
    print(f"O_new             = {o_new}")
    print()
    print_epilogue(title, row_sum_new, o_new)


def main() -> None:
    print_update("case 1: delta >= -8, keep old reference", [5.0, 4.0])
    print_update("case 2: delta < -8, switch reference", [11.0, 10.0])
    print_epilogue("guard: row_sum == 0", 0.0, [0.0, 0.0])
    print_epilogue("guard: row_sum == NaN", float("nan"), [0.0, 0.0])


if __name__ == "__main__":
    main()
