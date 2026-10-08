#!/usr/bin/env python3
"""Simulate DeepEP V2 direct-mode dispatch/combine index bookkeeping in pure Python.

Models: sender-side slot counters, buffer[src][slot] staging, rank-major vs
expert-major (expand) recv tensors, recv_src_metadata, and combine write-back
into buffer[contributor][original token_idx] followed by source-side reduce.
"""

from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    num_ranks: int
    num_experts: int
    num_topk: int
    max_tokens: int
    hidden: int = 4
    alignment: int = 4

    @property
    def experts_per_rank(self) -> int:
        return self.num_experts // self.num_ranks


def expert_fn(expert: int, vec: list[int]) -> list[int]:
    return [v * (expert + 1) + expert for v in vec]


def vec_add(a: list[int], b: list[int]) -> list[int]:
    return [x + y for x, y in zip(a, b)]


def align(n: int, a: int) -> int:
    return (n + a - 1) // a * a


def master_lane(values: list[int], k: int) -> bool:
    """ptx::deduplicate: keep the highest lane among lanes with the same value."""
    return max(j for j, v in enumerate(values) if v == values[k]) == k


def make_inputs(cfg: Config, seed: int):
    rng = random.Random(seed)
    xs, topks = [], []
    for _ in range(cfg.num_ranks):
        n = rng.randint(1, cfg.max_tokens)
        xs.append([[rng.randint(-3, 3) for _ in range(cfg.hidden)] for _ in range(n)])
        rows = []
        for _ in range(n):
            row = rng.sample(range(cfg.num_experts), cfg.num_topk)
            rows.append([e if rng.random() > 0.2 else -1 for e in row])
        topks.append(rows)
    return xs, topks


def dispatch(cfg: Config, xs, topks):
    """Return buffer[dst][src][slot] records, sender counters, dst_buffer_slot_idx."""
    R, K, M = cfg.num_ranks, cfg.num_topk, cfg.max_tokens
    buffer = [[[None] * M for _ in range(R)] for _ in range(R)]
    counter = [[0] * R for _ in range(R)]
    dst_buffer_slot_idx = []
    for src in range(R):
        per_src = []
        for t, (x, tk) in enumerate(zip(xs[src], topks[src])):
            dst_ranks = [e // cfg.experts_per_rank if e >= 0 else -1 for e in tk]
            row = [-1] * K
            for k in range(K):
                d = dst_ranks[k]
                if d < 0 or not master_lane(dst_ranks, k):
                    continue
                slot = counter[src][d]
                counter[src][d] += 1
                assert buffer[d][src][slot] is None
                buffer[d][src][slot] = {"x": x, "topk": tk, "src_global": src * M + t}
                row[k] = src * M + slot
            per_src.append(row)
        dst_buffer_slot_idx.append(per_src)
    return buffer, counter, dst_buffer_slot_idx


def copy_epilogue(cfg: Config, buffer, counter, dst: int, expand: bool):
    """Scan buffer[dst][src][0:count] in src order; build recv tensors + metadata."""
    R, K, EPR = cfg.num_ranks, cfg.num_topk, cfg.experts_per_rank
    lo, hi = dst * EPR, (dst + 1) * EPR
    records = [(src, buffer[dst][src][s]) for src in range(R) for s in range(counter[src][dst])]

    expert_count = [0] * EPR
    for _, rec in records:
        for e in rec["topk"]:
            if lo <= e < hi:
                expert_count[e - lo] += 1
    base = [0]
    for c in expert_count:
        base.append(base[-1] + align(c, cfg.alignment))
    cursor = base[:-1]

    recv_x, recv_topk_idx, metadata = [], [], []
    if expand:
        recv_x = [None] * base[-1]
    for src, rec in records:
        local = [e - lo if lo <= e < hi else -1 for e in rec["topk"]]
        master = max(k for k in range(K) if local[k] >= 0)
        meta = [rec["src_global"], src * K + master] + [-1] * K
        if expand:
            for k in range(K):
                if local[k] >= 0:
                    row = cursor[local[k]]
                    cursor[local[k]] += 1
                    recv_x[row] = (rec["x"], local[k])
                    meta[2 + k] = row
        else:
            recv_x.append(rec["x"])
            recv_topk_idx.append(local)
        metadata.append(meta)
    return recv_x, recv_topk_idx, metadata, base


def expert_compute(cfg: Config, dst: int, recv_x, recv_topk_idx, expand: bool):
    lo = dst * cfg.experts_per_rank
    if expand:
        return [None if item is None else expert_fn(lo + item[1], item[0]) for item in recv_x]
    out = []
    for x, local in zip(recv_x, recv_topk_idx):
        acc = [0] * cfg.hidden
        for e in local:
            if e >= 0:
                acc = vec_add(acc, expert_fn(lo + e, x))
        out.append(acc)
    return out


def combine(cfg: Config, y, metadata, me: int, expand: bool, multi_reduce: bool, comb):
    """Write partial results into comb[src][contributor][token_idx]."""
    K, M = cfg.num_topk, cfg.max_tokens
    rank_layout = multi_reduce and cfg.num_ranks <= K
    expanded_send = expand and not multi_reduce
    for i, meta in enumerate(metadata):
        t = meta[0] % M
        src, master = meta[1] // K, meta[1] % K
        slots = meta[2:]
        if expanded_send:
            for k in range(K):
                if slots[k] >= 0:
                    assert comb[src][k][t] is None
                    comb[src][k][t] = y[slots[k]]
            continue
        if expand:
            vec = [0] * cfg.hidden
            for s in slots:
                if s >= 0:
                    vec = vec_add(vec, y[s])
        else:
            vec = y[i]
        strip = me if rank_layout else master
        assert comb[src][strip][t] is None
        comb[src][strip][t] = vec


def source_reduce(cfg: Config, comb, src: int, topks, expand: bool, multi_reduce: bool):
    K = cfg.num_topk
    rank_layout = multi_reduce and cfg.num_ranks <= K
    out = []
    for t, tk in enumerate(topks[src]):
        ranks = [e // cfg.experts_per_rank if e >= 0 else -1 for e in tk]
        if expand and not multi_reduce:
            strips = [k for k in range(K) if ranks[k] >= 0]
        else:
            lanes = [k for k in range(K) if ranks[k] >= 0 and master_lane(ranks, k)]
            strips = [ranks[k] if rank_layout else k for k in lanes]
        acc = [0] * cfg.hidden
        for s in strips:
            acc = vec_add(acc, comb[src][s][t])
        out.append(acc)
    return out


def reference(cfg: Config, xs, topks, src: int):
    out = []
    for x, tk in zip(xs[src], topks[src]):
        acc = [0] * cfg.hidden
        for e in tk:
            if e >= 0:
                acc = vec_add(acc, expert_fn(e, x))
        out.append(acc)
    return out


def run(cfg: Config, seed: int, expand: bool, multi_reduce: bool) -> int:
    xs, topks = make_inputs(cfg, seed)
    buffer, counter, _ = dispatch(cfg, xs, topks)
    rank_layout = multi_reduce and cfg.num_ranks <= cfg.num_topk
    num_strips = cfg.num_ranks if rank_layout else cfg.num_topk
    comb = [[[None] * cfg.max_tokens for _ in range(num_strips)] for _ in range(cfg.num_ranks)]
    for dst in range(cfg.num_ranks):
        recv_x, recv_topk_idx, metadata, _ = copy_epilogue(cfg, buffer, counter, dst, expand)
        y = expert_compute(cfg, dst, recv_x, recv_topk_idx, expand)
        combine(cfg, y, metadata, dst, expand, multi_reduce, comb)
    for src in range(cfg.num_ranks):
        assert source_reduce(cfg, comb, src, topks, expand, multi_reduce) == reference(cfg, xs, topks, src)
    return num_strips


def trace(cfg: Config, seed: int) -> None:
    xs, topks = make_inputs(cfg, seed)
    buffer, counter, slot_idx = dispatch(cfg, xs, topks)
    tk = topks[0][0]
    print(f"src=0 token=0 topk={tk} dst_ranks={[e // cfg.experts_per_rank if e >= 0 else -1 for e in tk]}")
    print(f"  dst_buffer_slot_idx={slot_idx[0][0]}  (value = src*MaxTok + slot, -1 = non-master lane)")
    for dst in range(cfg.num_ranks):
        _, _, metadata, _ = copy_epilogue(cfg, buffer, counter, dst, expand=True)
        for row, meta in enumerate(metadata):
            if meta[0] == 0:
                print(f"  rank{dst}: recv row {row} metadata={meta}")


def main() -> None:
    configs = [
        Config(num_ranks=4, num_experts=8, num_topk=4, max_tokens=8),
        Config(num_ranks=8, num_experts=16, num_topk=2, max_tokens=8),
    ]
    trace(configs[0], seed=0)
    for cfg in configs:
        for expand in (False, True):
            for multi in (True, False):
                strips = 0
                for seed in range(50):
                    strips = run(cfg, seed, expand, multi)
                print(f"R={cfg.num_ranks} K={cfg.num_topk} expand={expand!s:5} multi_reduce={multi!s:5} "
                      f"combine strips={strips} OK")
    print("all checks passed")


if __name__ == "__main__":
    main()
