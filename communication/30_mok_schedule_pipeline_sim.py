#!/usr/bin/env python3
"""
MoK metadata pipeline: count -> pad -> schedule, faithfully re-implemented.

Mirrors:
  csrc/scheduler.cuh:34-63   count_kernel
  csrc/scheduler.cuh:66-77   pad_kernel
  csrc/scheduler.cuh:78-147  schedule_kernel

Run: python3 30_mok_schedule_pipeline_sim.py
"""

from __future__ import annotations

# Stand-ins for the real constants (types.cuh / scheduler.cuh).
EXPERT_PADDING = 256   # == config::MLP_Mb, keeps every GEMM tile inside one expert


# --------------------------------------------------------------------------
# Stage 1: count  ->  cnt[e][r]  (number of *slots*, not tokens)
# --------------------------------------------------------------------------
def count(topk_all, rank, num_local_experts):
    """topk_all[peer][token][k] -> local global-expert id. Returns cnt[local_e][peer]."""
    world = len(topk_all)
    first = rank * num_local_experts
    last = first + num_local_experts
    cnt = [[0] * world for _ in range(num_local_experts)]
    for peer in range(world):
        for row in topk_all[peer]:
            for e in row:
                if first <= e < last:
                    cnt[e - first][peer] += 1
    return cnt


# --------------------------------------------------------------------------
# Stage 2: pad  ->  len[e] (rounded up), num_tokens
# --------------------------------------------------------------------------
def pad(cnt, padding=EXPERT_PADDING):
    lens = []
    for row in cnt:
        raw = sum(row)
        lens.append((raw + padding - 1) // padding * padding)
    return lens, sum(lens)


# --------------------------------------------------------------------------
# Stage 3: schedule  ->  (peer_rank, peer_token_idx) indexed by dst
# --------------------------------------------------------------------------
def schedule(topk_all, rank, num_local_experts, block_dim=None, padding=EXPERT_PADDING):
    """Return (peer_rank, peer_token_idx, len) with dst as the array index."""
    world = len(topk_all)
    num_local_tokens = len(topk_all[0])
    topk = len(topk_all[0][0])
    rank_stride = num_local_tokens * topk
    first = rank * num_local_experts

    cnt = count(topk_all, rank, num_local_experts)
    lens, total = pad(cnt, padding)

    bases, base = [], 0
    for L in lens:
        bases.append(base)
        base += L

    peer_rank = [-1] * total
    peer_token_idx = [-1] * total
    written = [False] * total
    # expected real-slot range per expert: [base_e, base_e + raw_e)
    real = set()
    for le, row in enumerate(cnt):
        real.update(range(bases[le], bases[le] + sum(row)))

    for le in range(num_local_experts):
        cnt_row = cnt[le]          # = tokens_per_peer_rank[] in smem
        for r in range(world):
            # --- enumerate this peer's hits, ordered by (idx % B, idx // B) ---
            hits = [idx for idx in range(rank_stride)
                    if topk_all[r][idx // topk][idx % topk] == first + le]
            B = block_dim or 1024
            hits.sort(key=lambda i: (i % B, i // B))

            for j, idx in enumerate(hits):
                # --- dst formula: scheduler.cuh:134-139 ---
                dst = bases[le]
                for rr in range(world):
                    dst += min(cnt_row[rr], j)
                    dst += 1 if (rr < r and cnt_row[rr] > j) else 0

                assert not written[dst], (dst, (r, idx), "collision")
                written[dst] = True
                peer_rank[dst] = r
                peer_token_idx[dst] = idx

    # real slots covered exactly once; each segment's tail stays -1 (padding)
    assert all(written[i] for i in real), "unfilled slot"
    assert not any(written[i] for i in range(total) if i not in real), "wrote into padding"
    return peer_rank, peer_token_idx, lens, total


# --------------------------------------------------------------------------
# Invariant checks
# --------------------------------------------------------------------------
def check_bijection(cnt_row):
    """dst must cover [0, sum(cnt_row)) contiguously, for any cnt vector."""
    W = len(cnt_row)
    out = []
    for r in range(W):
        for j in range(cnt_row[r]):
            d = 0
            for rr in range(W):
                d += min(cnt_row[rr], j)
                d += 1 if (rr < r and cnt_row[rr] > j) else 0
            out.append(d)
    return sorted(out) == list(range(sum(cnt_row)))


def demo_tiny():
    W, T, K, E_LOCAL, PAD = 2, 4, 2, 2, 4   # PAD shrunk from 256 to 4 for readability
    topk_all = [
        [[0, 3], [0, 1], [2, 3], [1, 0]],   # rank0's tokens
        [[2, 0], [3, 1], [1, 2], [0, 3]],   # rank1's tokens
    ]
    print(f"world={W} tokens/rank={T} topk={K} local_experts={E_LOCAL} "
          f"(EXPERT_PADDING shrunk 256 -> {PAD})")
    print(f"global slots = {W * T * K}\n")
    for r in range(W):
        cnt = count(topk_all, r, E_LOCAL)
        lens, total = pad(cnt, PAD)
        first = r * E_LOCAL
        print(f"rank{r} owns global experts [{first},{first + E_LOCAL})")
        for le in range(E_LOCAL):
            print(f"  e{first + le}: cnt={cnt[le]}  raw={sum(cnt[le])}  len={lens[le]}")
        print(f"  num_tokens={total}")
        pr, pti, _, _ = schedule(topk_all, r, E_LOCAL, block_dim=4, padding=PAD)
        print("  row -> (peer, peer_token_idx, token, k)")
        for d, (p, i) in enumerate(zip(pr, pti)):
            tag = "slot" if p >= 0 else "pad "
            print(f"    row {d}: {tag} peer{p} idx{i} (token{i // K}, k={i % K})")
        print()


def demo_interleave():
    print("interleave structure for cnt = [3, 2, 4] (peer0 has 3, peer1 has 2, peer2 four)")
    cnt = [3, 2, 4]
    W = len(cnt)
    print("        j=0          j=1          j=2          j=3")
    for j in range(max(cnt)):
        row = f"  j={j}  "
        for r in range(W):
            if cnt[r] > j:
                d = 0
                for rr in range(W):
                    d += min(cnt[rr], j)
                    d += 1 if (rr < r and cnt[rr] > j) else 0
                row += f" r{r}->dst{d:<3}  "
            else:
                row += "   absent    "
        print(row)
    print()
    print("contiguous & bijective:", check_bijection(cnt))
    print()


def demo_scale():
    """Kimi-shaped numbers: EP=8, T=7168, topk=8, 48 local experts."""
    W, T, K, E_LOCAL = 8, 7168, 8, 48
    total_slots = W * T * K
    per_rank = total_slots // W
    print("balanced case (EP=8, T=7168, topk=8, 48 local experts)")
    print(f"  global slots          = {total_slots}")
    print(f"  slots per rank        = {per_rank} = {per_rank // 256} x 256")
    print(f"  count launch          = {(total_slots + 1023) // 1024} blocks x 1024 thr")
    print(f"  schedule launch       = {E_LOCAL * W} blocks x 1024 thr")
    print(f"  smem (schedule)       = {W * 4} B")
    print(f"  scan amplification    = {E_LOCAL * W} blocks x {T * K} slots "
          f"= {E_LOCAL * W * T * K / 1e6:.0f}M element reads")
    print(f"  route table size      = {total_slots * 4 / 1e6:.1f} MB (fits L2)")
    cap = T * K * max(2, -(-W // 2))
    print(f"  schedule_capacity     = {cap}  (utilization {per_rank / cap:.0%})")
    print()


def check_random(count_trials=3000):
    import random
    random.seed(0)
    for _ in range(count_trials):
        W = random.randint(1, 6)
        cnt = [random.randint(0, 5) for _ in range(W)]
        assert check_bijection(cnt), cnt
    print(f"bijection/coverage holds for {count_trials} random cnt vectors")


if __name__ == "__main__":
    demo_tiny()
    demo_interleave()
    demo_scale()
    check_random()
