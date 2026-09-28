#!/usr/bin/env python3
"""Compare causal attention LPT ordering with a natural m-ascending order."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SchedulerConfig:
    batch_size: int
    num_heads: int
    num_m_blocks: int
    l2_swizzle: int
    num_kv_blocks: int
    seq_q_per_tile: int
    smem_pipe_depth_q: int
    seq_len_q: int
    seq_len_kv: int
    blk_n: int

    @property
    def num_hb(self) -> int:
        return self.batch_size * self.num_heads

    @property
    def l2_major(self) -> int:
        return self.l2_swizzle * self.num_m_blocks

    @property
    def num_hb_quotient(self) -> int:
        return self.num_hb // self.l2_swizzle

    @property
    def num_hb_remainder(self) -> int:
        return max(self.num_hb % self.l2_swizzle, 1)

    @property
    def total_tasks(self) -> int:
        return self.batch_size * self.num_heads * self.num_m_blocks


def ceildiv(x: int, y: int) -> int:
    return (x + y - 1) // y


def n_block_max_of(m_block_idx: int, cfg: SchedulerConfig) -> int:
    m_idx_max = (
        (m_block_idx + 1)
        * cfg.seq_q_per_tile
        * cfg.smem_pipe_depth_q
    )
    n_idx = m_idx_max + cfg.seq_len_kv - cfg.seq_len_q
    return min(cfg.num_kv_blocks, ceildiv(n_idx, cfg.blk_n))


def decode_lpt(
    linear_idx: int,
    cfg: SchedulerConfig,
) -> tuple[int, int, int]:
    """Mirror FlashAttentionLPTScheduler::update_current_m_n_idx."""
    bidhb = linear_idx // cfg.l2_major
    l2_mod = linear_idx % cfg.l2_major

    complete_group = bidhb < cfg.num_hb_quotient
    m_block_raw = (
        l2_mod // cfg.l2_swizzle
        if complete_group
        else l2_mod // cfg.num_hb_remainder
    )
    bidhb_residual = (
        l2_mod % cfg.l2_swizzle
        if complete_group
        else l2_mod % cfg.num_hb_remainder
    )

    bidhb_actual = bidhb * cfg.l2_swizzle + bidhb_residual
    batch_idx = bidhb_actual // cfg.num_heads
    head_idx = bidhb_actual % cfg.num_heads
    m_block_idx = (cfg.num_m_blocks - 1) - m_block_raw
    return batch_idx, head_idx, m_block_idx


def decode_natural(
    linear_idx: int,
    cfg: SchedulerConfig,
) -> tuple[int, int, int]:
    """Natural head-major, m-ascending order used as the comparison."""
    batch_idx = linear_idx // (cfg.num_heads * cfg.num_m_blocks)
    head_idx = (linear_idx // cfg.num_m_blocks) % cfg.num_heads
    m_block_idx = linear_idx % cfg.num_m_blocks
    return batch_idx, head_idx, m_block_idx


def task_cost(task: tuple[int, int, int], cfg: SchedulerConfig) -> int:
    _, _, m_block_idx = task
    return n_block_max_of(m_block_idx, cfg)


def simulate_workers(
    tasks: list[tuple[int, int, int]],
    cfg: SchedulerConfig,
    num_workers: int,
) -> tuple[list[list[int]], list[int]]:
    """Idle worker takes next task; equal loads prefer the smaller id."""
    histories = [[] for _ in range(num_workers)]
    loads = [0] * num_workers
    for task in tasks:
        worker = min(range(num_workers), key=lambda idx: (loads[idx], idx))
        cost = task_cost(task, cfg)
        histories[worker].append(cost)
        loads[worker] += cost
    return histories, loads


def describe_tasks(
    tasks: list[tuple[int, int, int]],
    cfg: SchedulerConfig,
    limit: int | None = None,
) -> None:
    for linear_idx, (batch_idx, head_idx, m_block_idx) in enumerate(tasks):
        if limit is not None and linear_idx >= limit:
            break
        cost = task_cost((batch_idx, head_idx, m_block_idx), cfg)
        print(
            f"linear={linear_idx:2d} -> "
            f"batch={batch_idx}, head={head_idx}, "
            f"m_block={m_block_idx}, cost={cost}"
        )


def main() -> None:
    cfg = SchedulerConfig(
        batch_size=1,
        num_heads=8,
        num_m_blocks=4,
        l2_swizzle=8,
        num_kv_blocks=8,
        seq_q_per_tile=128,
        smem_pipe_depth_q=2,
        seq_len_q=1024,
        seq_len_kv=1024,
        blk_n=128,
    )

    print("== scheduler geometry ==")
    print(f"num_hb          = {cfg.batch_size} * {cfg.num_heads} = {cfg.num_hb}")
    print(f"l2_major        = {cfg.l2_swizzle} * {cfg.num_m_blocks} = {cfg.l2_major}")
    print(f"num_hb_quotient = {cfg.num_hb} // {cfg.l2_swizzle} = {cfg.num_hb_quotient}")
    print(f"num_hb_remainder= max({cfg.num_hb} % {cfg.l2_swizzle}, 1) = {cfg.num_hb_remainder}")
    print(f"total_tasks     = {cfg.total_tasks}")
    print()

    print("== causal task costs ==")
    for m_block_idx in range(cfg.num_m_blocks):
        m_idx_min = m_block_idx * cfg.seq_q_per_tile * cfg.smem_pipe_depth_q
        m_idx_max = (m_block_idx + 1) * cfg.seq_q_per_tile * cfg.smem_pipe_depth_q
        n_idx = m_idx_max + cfg.seq_len_kv - cfg.seq_len_q
        cost = n_block_max_of(m_block_idx, cfg)
        print(
            f"m_block={m_block_idx}: "
            f"m_idx_min={m_idx_min}, m_idx_max={m_idx_max}, n_idx={n_idx}, "
            f"n_block_max={cost}, cost={cost}"
        )
    print()

    lpt_tasks = [decode_lpt(i, cfg) for i in range(cfg.total_tasks)]
    natural_tasks = [decode_natural(i, cfg) for i in range(cfg.total_tasks)]

    print("== LPT launch order (first 16 tasks) ==")
    describe_tasks(lpt_tasks, cfg, limit=16)
    print()

    print("== natural launch order (first 16 tasks) ==")
    describe_tasks(natural_tasks, cfg, limit=16)
    print()

    lpt_histories, lpt_loads = simulate_workers(lpt_tasks, cfg, num_workers=4)
    natural_histories, natural_loads = simulate_workers(
        natural_tasks,
        cfg,
        num_workers=4,
    )

    print("== 4-worker simulation ==")
    print("LPT histories     =", lpt_histories)
    print("LPT loads         =", lpt_loads)
    print("natural histories =", natural_histories)
    print("natural loads     =", natural_loads)
    print("max load          = LPT: 40, natural: 44")


if __name__ == "__main__":
    main()
