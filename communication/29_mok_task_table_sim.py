"""Reconstruct MoK forward task table and tile mapping on CPU.

This mirrors the arithmetic in
  mok/csrc/megakernel/types.cuh          (globals_fwd::grid)
  mok/csrc/megakernel/forward.cuh        (cluster_idx -> task dispatch)
  mok/csrc/megakernel/grouped_gemm.cuh   (expert row segment -> tile_coord)
  mok/csrc/megakernel/swiglu.cuh         (tile -> parent task index)

It does not model timing or memory, only the integer indexing that decides
"which cluster computes which tile of which expert".
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# ---- compile-time constants, copied from csrc/megakernel/types.cuh ----
MLP_Mb = 256
MLP_Nb = 256
SWIGLU_Mb = 128
SWIGLU_Nb = 128
SWIGLU_FWD_PIPE_DEPTH = 3
CLUSTER_SIZE = 2
SUPERGROUP_SIZE = 8
DISPATCH_Mb = 128
DISPATCH_Nb = 512
COMBINE_Mb = 16
COMBINE_Nb = 1024
COMBINE_PIPE_DEPTH = 7

EXPERT_PADDING = 256  # csrc/scheduler.cuh


@dataclass(frozen=True)
class Cfg:
    ep_size: int
    hidden: int
    intermediate: int
    num_local_experts: int
    topk: int
    num_local_tokens: int
    minibatch_size: int
    macrobatch_size: int
    num_comm_sms: int
    schedule_capacity_multiplier: float = 0.5

    @property
    def schedule_capacity_factor(self) -> int:
        return max(2, math.ceil(self.ep_size * self.schedule_capacity_multiplier))

    @property
    def schedule_capacity(self) -> int:
        return self.num_local_tokens * self.topk * self.schedule_capacity_factor


def swizzled_2d_idx(num_rows: int, num_cols: int, linear_idx: int,
                    supergroup: int = SUPERGROUP_SIZE) -> tuple[int, int]:
    """kittens::get_swizzled_2d_idx with ROW_MAJOR = true."""
    supergroup_numel = num_rows * supergroup
    supergroup_idx = linear_idx // supergroup_numel
    supersection_cols = (num_cols // supergroup) * supergroup
    supersection_numel = num_rows * supersection_cols
    if linear_idx < supersection_numel:
        row_idx = (linear_idx % supergroup_numel) // supergroup
        col_idx = supergroup_idx * supergroup + linear_idx % supergroup
    else:
        remainder = linear_idx - supersection_numel
        finalsection_cols = num_cols - supersection_cols
        row_idx = remainder // finalsection_cols
        col_idx = supersection_cols + remainder % finalsection_cols
    if supergroup_idx & 1:
        row_idx = num_rows - row_idx - 1
    return row_idx, col_idx


def balanced_tokens_per_expert(cfg: Cfg, num_tokens_unpadded: int) -> list[int]:
    """Even routing, then pad every expert segment up to 256 rows."""
    base, rest = divmod(num_tokens_unpadded, cfg.num_local_experts)
    per_expert = [base + (1 if i < rest else 0) for i in range(cfg.num_local_experts)]
    return [math.ceil(n / EXPERT_PADDING) * EXPERT_PADDING for n in per_expert]


def fwd_task_table(cfg: Cfg, tokens_per_expert: list[int]) -> dict:
    num_tokens = sum(tokens_per_expert)
    shared_row_blocks = cfg.num_local_tokens // MLP_Mb
    int_col_blocks = cfg.intermediate // MLP_Nb
    hid_col_blocks = cfg.hidden // MLP_Nb

    shared_gate_up = shared_row_blocks * int_col_blocks          # per GEMM
    shared_swiglu_tiles = (cfg.num_local_tokens // SWIGLU_Mb) * (cfg.intermediate // SWIGLU_Nb)
    shared_swiglu = math.ceil(shared_swiglu_tiles / (CLUSTER_SIZE * SWIGLU_FWD_PIPE_DEPTH))
    shared_down = shared_row_blocks * hid_col_blocks
    shared_tasks = 2 * shared_gate_up + shared_swiglu + shared_down

    mb_row_blocks = cfg.minibatch_size // MLP_Mb
    mb_gate_up = mb_row_blocks * int_col_blocks                  # per GEMM
    mb_swiglu_tiles = (cfg.minibatch_size // SWIGLU_Mb) * (cfg.intermediate // SWIGLU_Nb)
    mb_swiglu = math.ceil(mb_swiglu_tiles / (CLUSTER_SIZE * SWIGLU_FWD_PIPE_DEPTH))
    mb_down = mb_row_blocks * hid_col_blocks
    minibatch_tasks = 2 * mb_gate_up + mb_swiglu + mb_down

    # grid() uses schedule_capacity (an upper bound), not the runtime token count
    cap_minibatches = math.ceil(cfg.schedule_capacity / cfg.minibatch_size)
    grid = CLUSTER_SIZE * (shared_tasks + cap_minibatches * minibatch_tasks) + cfg.num_comm_sms

    minibatches_per_mb = cfg.macrobatch_size // cfg.minibatch_size
    num_macrobatches = math.ceil(num_tokens / cfg.macrobatch_size)
    true_minibatches = math.ceil(num_tokens / cfg.minibatch_size)
    last_mb_minibatches = true_minibatches - (num_macrobatches - 1) * minibatches_per_mb

    comm_clusters = cfg.num_comm_sms // CLUSTER_SIZE
    true_clusters = comm_clusters + shared_tasks + true_minibatches * minibatch_tasks

    return dict(
        num_tokens=num_tokens,
        shared_row_blocks=shared_row_blocks,
        shared_gate_up=shared_gate_up,
        shared_swiglu=shared_swiglu,
        shared_down=shared_down,
        shared_tasks=shared_tasks,
        mb_gate_up=mb_gate_up,
        mb_swiglu=mb_swiglu,
        mb_down=mb_down,
        minibatch_tasks=minibatch_tasks,
        cap_minibatches=cap_minibatches,
        grid=grid,
        comm_clusters=comm_clusters,
        true_clusters=true_clusters,
        cancelled_clusters=(grid - cfg.num_comm_sms) // CLUSTER_SIZE - true_clusters,
        num_macrobatches=num_macrobatches,
        true_minibatches=true_minibatches,
        last_mb_minibatches=last_mb_minibatches,
        minibatches_per_mb=minibatches_per_mb,
    )


def resolve_fwd_cluster(cfg: Cfg, tokens_per_expert: list[int], t: dict,
                        compute_cluster_idx: int) -> dict:
    """Map compute_cluster_idx -> (macrobatch, minibatch, family, tile_coord)."""
    if not 0 <= compute_cluster_idx < t["shared_tasks"] + t["true_minibatches"] * t["minibatch_tasks"]:
        return {"kind": "out_of_range"}

    if compute_cluster_idx < t["shared_tasks"]:
        if compute_cluster_idx < t["shared_gate_up"]:
            return {"kind": "shared_gate", "task_idx": compute_cluster_idx}
        if compute_cluster_idx < 2 * t["shared_gate_up"]:
            return {"kind": "shared_up", "task_idx": compute_cluster_idx - t["shared_gate_up"]}
        if compute_cluster_idx < 2 * t["shared_gate_up"] + t["shared_swiglu"]:
            return {"kind": "shared_swiglu",
                    "task_idx": compute_cluster_idx - 2 * t["shared_gate_up"]}
        return {"kind": "shared_down",
                "task_idx": compute_cluster_idx - 2 * t["shared_gate_up"] - t["shared_swiglu"]}

    # Routed: the last macrobatch's minibatches are issued first
    ordered = (compute_cluster_idx - t["shared_tasks"]) // t["minibatch_tasks"]
    task_idx = (compute_cluster_idx - t["shared_tasks"]) - ordered * t["minibatch_tasks"]
    if ordered < t["last_mb_minibatches"]:
        macrobatch_idx = t["num_macrobatches"] - 1
        minibatch_idx = ordered
    else:
        idx = ordered - t["last_mb_minibatches"]
        macrobatch_idx = t["num_macrobatches"] - 2 - idx // t["minibatches_per_mb"]
        minibatch_idx = idx % t["minibatches_per_mb"]

    int_col_blocks = cfg.intermediate // MLP_Nb
    hid_col_blocks = cfg.hidden // MLP_Nb
    if task_idx < t["mb_gate_up"]:
        family = "routed_gate"
    elif task_idx < 2 * t["mb_gate_up"]:
        family = "routed_up"
        task_idx -= t["mb_gate_up"]
    elif task_idx < 2 * t["mb_gate_up"] + t["mb_swiglu"]:
        family = "routed_swiglu"
        task_idx -= 2 * t["mb_gate_up"]
        return {"kind": family, "macrobatch_idx": macrobatch_idx,
                "minibatch_idx": minibatch_idx, "task_idx": task_idx,
                "first_tile_idx": (macrobatch_idx * (cfg.macrobatch_size // SWIGLU_Mb)
                                   + minibatch_idx * (cfg.minibatch_size // SWIGLU_Mb)) * (cfg.intermediate // SWIGLU_Nb)
                                  + task_idx * CLUSTER_SIZE * SWIGLU_FWD_PIPE_DEPTH}
    else:
        family = "routed_down"
        task_idx -= 2 * t["mb_gate_up"] + t["mb_swiglu"]

    col_blocks = int_col_blocks if family != "routed_down" else hid_col_blocks
    global_minibatch_idx = macrobatch_idx * (cfg.macrobatch_size // cfg.minibatch_size) + minibatch_idx
    mb_row_blocks = cfg.minibatch_size // MLP_Mb
    first_rb = global_minibatch_idx * mb_row_blocks
    mb_rb_offset = macrobatch_idx * (cfg.macrobatch_size // MLP_Mb)

    offset = 0
    for expert_idx, n in enumerate(tokens_per_expert):
        expert_rb = n // MLP_Mb
        lo = max(first_rb, offset)
        rb = max(0, min(first_rb + mb_row_blocks, offset + expert_rb) - lo)
        num_tasks = rb * col_blocks
        if task_idx < num_tasks:
            sr, sc = swizzled_2d_idx(rb, col_blocks, task_idx)
            return {"kind": family, "macrobatch_idx": macrobatch_idx,
                    "minibatch_idx": minibatch_idx,
                    "expert_idx": expert_idx, "intersect_row_blocks": rb,
                    "col_blocks": col_blocks,
                    "tile_row_block": lo + sr - mb_rb_offset,
                    "tile_col_block": sc,
                    "first_routed_row": (lo + sr) * MLP_Mb}
        task_idx -= num_tasks
        offset += expert_rb
    return {"kind": "no_tile"}


def comm_task_counts(cfg: Cfg, t: dict) -> dict:
    """Dispatch/combine chunking, from csrc/megakernel/dispatch_combine.cuh."""
    mb_rows = min(cfg.macrobatch_size, t["num_tokens"])
    d_col_blocks = math.ceil(cfg.hidden / DISPATCH_Nb)
    dispatch_tasks = (mb_rows // DISPATCH_Mb) * d_col_blocks
    c_col_blocks = math.ceil(cfg.hidden / COMBINE_Nb)
    combine_tiles = (mb_rows // COMBINE_Mb) * c_col_blocks
    combine_tasks = math.ceil(combine_tiles / COMBINE_PIPE_DEPTH)
    return dict(
        dispatch_tasks=dispatch_tasks,
        dispatch_bytes_per_task=DISPATCH_Mb * DISPATCH_Nb * 2,
        combine_tasks=combine_tasks,
        combine_chunks_per_task=COMBINE_PIPE_DEPTH,
        combine_bytes_per_chunk=COMBINE_Mb * COMBINE_Nb * 2,
        routed_bytes=mb_rows * cfg.hidden * 2,
    )


def counter_contract(cfg: Cfg, t: dict) -> dict:
    """Per-slot required_count for each producer->consumer counter (forward).

    Each entry is the value the consumer compares against at one barrier index,
    i.e. how many `barrier_arrive` calls that slot must collect.
    """
    hid_col_blocks = cfg.hidden // MLP_Nb
    d_col_blocks = math.ceil(cfg.hidden / DISPATCH_Nb)
    c_col_blocks = math.ceil(cfg.hidden / COMBINE_Nb)
    return {
        # one slot per global minibatch; dispatch arrives once per 128x512 chunk
        "x_routed_ready[minibatch]": (cfg.minibatch_size // DISPATCH_Mb) * d_col_blocks,
        # one slot per parent 256x256 tile; gate + up, each 2 CTAs
        "gate_up_tile_ready[tile]": 2 * CLUSTER_SIZE,
        # one slot per 256-row block; SwiGLU arrives once per 128x128 tile
        "hidden_row_block_ready[row_block]": (MLP_Mb // SWIGLU_Mb) * (cfg.intermediate // SWIGLU_Nb),
        # one slot per global minibatch; down GEMM arrives once per output task
        "y_routed_ready[minibatch]": (cfg.minibatch_size // MLP_Mb) * hid_col_blocks * CLUSTER_SIZE,
        # one slot per 128-row block; combine arrives once per 16x1024 tile
        "y_routed_done[row_block_128]": (MLP_Mb // CLUSTER_SIZE // COMBINE_Mb) * c_col_blocks,
    }


def main() -> None:
    cfg = Cfg(ep_size=8, hidden=7168, intermediate=2048, num_local_experts=32,
              topk=8, num_local_tokens=8192, minibatch_size=4096,
              macrobatch_size=131072, num_comm_sms=40)
    unpadded = cfg.ep_size * cfg.num_local_tokens * cfg.topk // cfg.ep_size
    tpe = balanced_tokens_per_expert(cfg, unpadded)
    t = fwd_task_table(cfg, tpe)

    print("== config ==")
    print(f"EP={cfg.ep_size} H={cfg.hidden} I={cfg.intermediate} E_local={cfg.num_local_experts} "
          f"topk={cfg.topk} T_local={cfg.num_local_tokens}")
    print(f"minibatch={cfg.minibatch_size} macrobatch={cfg.macrobatch_size} comm_sms={cfg.num_comm_sms}")
    print(f"schedule_capacity = {cfg.num_local_tokens}*{cfg.topk}*{cfg.schedule_capacity_factor}"
          f" = {cfg.schedule_capacity}")

    print("\n== forward task table ==")
    for k in ("num_tokens", "shared_row_blocks", "shared_gate_up", "shared_swiglu",
              "shared_down", "shared_tasks", "mb_gate_up", "mb_swiglu", "mb_down",
              "minibatch_tasks", "cap_minibatches", "true_minibatches",
              "num_macrobatches", "last_mb_minibatches", "comm_clusters",
              "true_clusters", "grid", "cancelled_clusters"):
        print(f"{k:>20} = {t[k]}")

    print("\n== resolve some compute_cluster_idx ==")
    probes = [
        t["shared_tasks"] + 0,                      # first routed task
        t["shared_tasks"] + 1,
        t["shared_tasks"] + 5 * t["minibatch_tasks"] + 100,   # minibatch 5, 100th task
        t["shared_tasks"] + 5 * t["minibatch_tasks"] + 300,   # minibatch 5, swiglu band
        t["shared_tasks"] + 5 * t["minibatch_tasks"] + 700,   # minibatch 5, down band
    ]
    for c in probes:
        print(f"compute_cluster_idx={c:6d} -> {resolve_fwd_cluster(cfg, tpe, t, c)}")

    print("\n== comm chunks (one macrobatch) ==")
    for k, v in comm_task_counts(cfg, t).items():
        print(f"{k:>28} = {v}")

    print("\n== counter contracts ==")
    for k, v in counter_contract(cfg, t).items():
        print(f"{k:>48} = {v}")

    print("\n== routed issue order with two macrobatches ==")
    cfg2 = Cfg(ep_size=8, hidden=7168, intermediate=2048, num_local_experts=32,
               topk=8, num_local_tokens=8192, minibatch_size=4096,
               macrobatch_size=131072, num_comm_sms=40)
    tpe2 = balanced_tokens_per_expert(cfg2, 150000)
    t2 = fwd_task_table(cfg2, tpe2)
    print(f"num_tokens={t2['num_tokens']} num_macrobatches={t2['num_macrobatches']} "
          f"true_minibatches={t2['true_minibatches']} last_mb_minibatches={t2['last_mb_minibatches']}")
    for ordered in (0, 4, 5, 6, 36):
        c = t2["shared_tasks"] + ordered * t2["minibatch_tasks"]
        r = resolve_fwd_cluster(cfg2, tpe2, t2, c)
        print(f"ordered_minibatch={ordered:2d} -> macrobatch={r['macrobatch_idx']} "
              f"minibatch={r['minibatch_idx']}")


if __name__ == "__main__":
    main()
