#!/usr/bin/env python3
"""Trace how FA4 packs GQA query heads into the 128 Q-tile rows."""

from __future__ import annotations

from dataclasses import dataclass


BLK_M = 128
SMEM_PIPE_DEPTH_Q = 2


def ceildiv(x: int, y: int) -> int:
    return (x + y - 1) // y


@dataclass(frozen=True)
class GQAConfig:
    num_qo_heads: int
    num_kv_heads: int
    seq_len_q: int

    @property
    def ratio(self) -> int:
        if self.num_qo_heads % self.num_kv_heads != 0:
            raise ValueError("num_qo_heads must be divisible by num_kv_heads")
        return self.num_qo_heads // self.num_kv_heads

    @property
    def seq_q_per_tile(self) -> int:
        if BLK_M % self.ratio != 0:
            raise ValueError("GQA ratio must divide BLK_M")
        return BLK_M // self.ratio

    @property
    def num_q_blocks_total(self) -> int:
        return ceildiv(self.seq_len_q, self.seq_q_per_tile)

    @property
    def num_m_blocks(self) -> int:
        return ceildiv(self.num_q_blocks_total, SMEM_PIPE_DEPTH_Q)


def decode_packed_row(
    row: int,
    kv_head_idx: int,
    cfg: GQAConfig,
) -> tuple[int, int, int]:
    """Return (sequence_offset, query_head_offset, global_query_head)."""
    if not 0 <= row < BLK_M:
        raise ValueError("row must be in [0, BLK_M)")
    seq_offset = row // cfg.ratio
    query_head_offset = row % cfg.ratio
    query_head = kv_head_idx * cfg.ratio + query_head_offset
    return seq_offset, query_head_offset, query_head


def q_tile_start(
    m_block: int,
    q_stage: int,
    cfg: GQAConfig,
    cta_group: int = 1,
    cta_rank: int = 0,
) -> int:
    """Return the global sequence start of one packed Q stage."""
    if cta_group not in (1, 2) or not 0 <= cta_rank < cta_group:
        raise ValueError("expected cta_group=1 or 2 and 0 <= cta_rank < cta_group")
    return (
        m_block * SMEM_PIPE_DEPTH_Q * cta_group
        + q_stage * cta_group
        + cta_rank
    ) * cfg.seq_q_per_tile


def describe_row(row: int, kv_head_idx: int, cfg: GQAConfig) -> str:
    seq_offset, head_offset, query_head = decode_packed_row(
        row, kv_head_idx, cfg
    )
    owner_kv_head = query_head // cfg.ratio
    return (
        f"row={row:3d} -> sequence={seq_offset:2d}, "
        f"head_offset={head_offset}, qo_head={query_head:2d}, "
        f"owner_kv_head={owner_kv_head}"
    )


def main() -> None:
    cfg = GQAConfig(
        num_qo_heads=32,
        num_kv_heads=8,
        seq_len_q=1024,
    )
    kv_head_idx = 3

    print("== GQA geometry ==")
    print(f"GQA_RATIO       = {cfg.num_qo_heads} / {cfg.num_kv_heads} = {cfg.ratio}")
    print(f"SEQ_Q_PER_TILE  = {BLK_M} / {cfg.ratio} = {cfg.seq_q_per_tile}")
    print(f"num_q_blocks    = ceildiv({cfg.seq_len_q}, {cfg.seq_q_per_tile}) = {cfg.num_q_blocks_total}")
    print(f"num_m_blocks    = ceildiv({cfg.num_q_blocks_total}, 2) = {cfg.num_m_blocks}")
    print()

    print("== packed row decoding for kv_head_idx=3 ==")
    for row in (0, 5, 127):
        print(describe_row(row, kv_head_idx, cfg))
    print()

    print("== one Q stage ==")
    print("SMEM Q tile is viewed as [SEQ_Q_PER_TILE, GQA_RATIO, HEAD_DIM]")
    print(
        "logical tile: "
        f"[{cfg.seq_q_per_tile} sequences, {cfg.ratio} query heads, 128 dims]"
    )
    print("flattened MMA operand: [BLK_M=128 rows, HEAD_DIM=128]")
    print()

    print("== Q stage global starts for m_block=0 ==")
    for q_stage in range(SMEM_PIPE_DEPTH_Q):
        start = q_tile_start(0, q_stage, cfg)
        end = start + cfg.seq_q_per_tile - 1
        print(
            f"stage {q_stage}: sequence {start}..{end}, "
            f"query heads "
            f"{kv_head_idx * cfg.ratio}.."
            f"{(kv_head_idx + 1) * cfg.ratio - 1}"
        )
    print()

    print("== TMA box and coordinates for Q load ==")
    print(
        "box       = (head_dim//2, GQA_RATIO, SEQ_Q_PER_TILE, 1) "
        f"= (64, {cfg.ratio}, {cfg.seq_q_per_tile}, 1)"
    )
    print(
        "coordinates = "
        f"(0, {kv_head_idx * cfg.ratio}, stage_sequence_start, batch*2)"
    )
    print()

    print("== K/V reuse check ==")
    owners = {
        decode_packed_row(row, kv_head_idx, cfg)[2] // cfg.ratio
        for row in range(BLK_M)
    }
    print(f"owner_kv_heads across all packed rows = {sorted(owners)}")
    print("all 128 rows share one K/V tile because every row maps to kv_head_idx=3")


if __name__ == "__main__":
    main()
