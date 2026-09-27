#!/usr/bin/env python3
"""Simulate the two-layer causal mask used by the FA4 kernel."""

from __future__ import annotations

import math


NEG_INF = float("-inf")


def ceildiv(x: int, y: int) -> int:
    return (x + y - 1) // y


def max_valid_key(
    query_pos: int,
    seq_len_q: int,
    seq_len_kv: int,
) -> int:
    """Return the largest key visible to a right-aligned query position."""
    return query_pos + seq_len_kv - seq_len_q


def task_n_block_max(
    m_block: int,
    seq_len_q: int,
    seq_len_kv: int,
    block_n: int,
    seq_q_per_tile: int = 128,
    q_stages: int = 2,
) -> int:
    """Mirror flash_attention4.py::n_block_max_of for GQA_RATIO=1."""
    num_kv_blocks = ceildiv(seq_len_kv, block_n)
    m_idx_max = (m_block + 1) * seq_q_per_tile * q_stages
    n_idx = m_idx_max + seq_len_kv - seq_len_q
    return min(num_kv_blocks, ceildiv(n_idx, block_n))


def r2p_keep_column(
    block_col: int,
    col_limit_right: int,
    chunk_size: int = 32,
) -> bool:
    """Reproduce mask_r2p's low-k-bits predicate for one column."""
    chunk = block_col // chunk_size
    lane = block_col % chunk_size
    k_keep = max(col_limit_right - chunk * chunk_size, 0)
    k_keep = min(k_keep, chunk_size)
    mask_inv = (0xFFFFFFFF << k_keep) & 0xFFFFFFFF
    in_bound = ((~mask_inv) & 0xFFFFFFFF) & (1 << lane)
    return bool(in_bound)


def apply_mask_r2p(
    s_chunk: list[float],
    col_limit_right: int,
) -> list[float]:
    """Keep columns with index < col_limit_right and mask the rest."""
    return [
        score if r2p_keep_column(i, col_limit_right) else NEG_INF
        for i, score in enumerate(s_chunk)
    ]


def apply_causal_mask_for_row(
    s_chunk: list[float],
    m_block: int,
    wg_id: int,
    seq_pos_in_wg: int,
    n_block: int,
    seq_len_q: int,
    seq_len_kv: int,
    block_n: int,
    seq_q_per_tile: int = 128,
) -> tuple[list[float], int]:
    """Mirror apply_causal_mask for GQA_RATIO=1 and q_stages=2."""
    row_idx = (
        m_block * seq_q_per_tile * 2
        + wg_id * seq_q_per_tile
        + seq_pos_in_wg
    )
    col_limit_right = (
        row_idx + 1 + seq_len_kv - n_block * block_n - seq_len_q
    )
    return apply_mask_r2p(s_chunk, col_limit_right), col_limit_right


def classify_block(
    query_pos: int,
    n_block: int,
    block_n: int,
    seq_len_q: int,
    seq_len_kv: int,
) -> tuple[str, list[int]]:
    key_limit = max_valid_key(query_pos, seq_len_q, seq_len_kv)
    keys = list(range(n_block * block_n, (n_block + 1) * block_n))
    valid = [key for key in keys if key <= key_limit]
    if not valid:
        return "skip", valid
    if len(valid) == block_n:
        return "full", valid
    return "partial", valid


def softmax2_after_mask(scores: list[float]) -> list[float]:
    finite = [score for score in scores if score != NEG_INF]
    row_max = max(finite)
    numerators = [
        0.0 if score == NEG_INF else math.pow(2.0, score - row_max)
        for score in scores
    ]
    denominator = sum(numerators)
    return [value / denominator for value in numerators]


def print_small_example() -> None:
    seq_len_q = 6
    seq_len_kv = 8
    block_n = 4

    print("== right-aligned block view ==")
    for query_pos in (0, 5):
        key_limit = max_valid_key(query_pos, seq_len_q, seq_len_kv)
        print(f"query {query_pos}: max_valid_key = {key_limit}")
        for n_block in range(ceildiv(seq_len_kv, block_n)):
            kind, valid = classify_block(
                query_pos, n_block, block_n, seq_len_q, seq_len_kv
            )
            print(
                f"  block {n_block} keys "
                f"{n_block * block_n}..{(n_block + 1) * block_n - 1}: "
                f"{kind}, valid={valid}"
            )
    print()


def print_mask_example() -> None:
    scores = [6.0, 5.0, 4.0, 3.0]
    seq_len_q = 6
    seq_len_kv = 8
    block_n = 4
    masked, col_limit = apply_causal_mask_for_row(
        scores,
        m_block=0,
        wg_id=0,
        seq_pos_in_wg=0,
        n_block=0,
        seq_len_q=seq_len_q,
        seq_len_kv=seq_len_kv,
        block_n=block_n,
    )
    probabilities = softmax2_after_mask(masked)

    print("== register-level mask and softmax ==")
    print(f"row_idx        = 0")
    print(f"col_limit_right = {col_limit}")
    print(f"S              = {scores}")
    print(f"masked S       = {masked}")
    print(f"P              = {probabilities}")
    print()


def print_task_bound_example() -> None:
    seq_len_q = 1024
    seq_len_kv = 1024
    block_n = 128
    seq_q_per_tile = 128
    q_stages = 2

    print("== FA4 task-level n_block_max ==")
    for m_block in (0, 1, 2, 3):
        n_block_max = task_n_block_max(
            m_block,
            seq_len_q,
            seq_len_kv,
            block_n,
            seq_q_per_tile,
            q_stages,
        )
        min_query = m_block * seq_q_per_tile * q_stages
        max_query = min_query + seq_q_per_tile * q_stages - 1
        print(
            f"m_block={m_block}, query_rows={min_query}..{max_query}, "
            f"n_block_max={n_block_max}"
        )


def main() -> None:
    print_small_example()
    print_mask_example()
    print_task_bound_example()


if __name__ == "__main__":
    main()
