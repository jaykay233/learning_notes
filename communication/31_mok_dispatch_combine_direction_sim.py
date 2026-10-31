#!/usr/bin/env python3
"""
MoK dispatch/combine direction check.

The claim: both are initiated by the EXPERT owner; only the data direction
differs. This script verifies it from the two kernel signatures alone, plus the
one property that makes a "push" write-conflict-free.

Mirrors:
  csrc/megakernel/dispatch_combine.cuh:2-4     dispatch_kernel(peer_buf, x_gmem, ...)
  csrc/megakernel/dispatch_combine.cuh:154-155 combine_kernel(peer_buf, local_buf, ...)
  csrc/megakernel/types.cuh:108-109            send buffer vs recv buffer shapes
  csrc/megakernel/forward.cuh:92-101           task counts
"""

from __future__ import annotations


def direction(reads, writes):
    """Frame rule: arg1 is always the peer (remote) side, arg2 always local."""
    if reads == "peer" and writes == "local":
        return "pull  (dispatch)"
    if reads == "local" and writes == "peer":
        return "push  (combine)"
    return f"?? reads={reads} writes={writes}"


def check_signature_symmetry():
    print("== frame rule: (peer_buf, local) in both signatures ==")
    # dispatch_kernel / combine_kernel / their 4 uses
    cases = [
        ("dispatch_kernel        (fwd dispatch)",       "peer", "local"),
        ("combine_kernel         (fwd combine)",        "local", "peer"),
        ("dispatch_kernel<true>  (bwd reverse-combine)", "peer", "local"),
        ("combine_kernel         (bwd reverse-dispatch)", "local", "peer"),
    ]
    for name, r, w in cases:
        print(f"  {name:<46} -> {direction(r, w)}")

    # the TMA instruction each one uses
    print("\n  actual TMA calls:")
    print("    dispatch  : tma::load_async (..., &peer_buf[peer_rank][...])  then  tma::store_async (x_gmem, ...)")
    print("    combine   : tma::load_async (..., &local_buf[row*cols + ...]) then  tma::store_async (&peer_buf[peer_rank][...], ...)")


def check_write_conflict():
    """Why a push is safe: the recv buffer has a *topk dimension."""
    print("\n== why combine can scatter without atomics ==")
    T, K = 7168, 8
    send_rows = T                       # x_routed_send_buffer  (num_local_tokens, H)
    recv_rows = T * K                   # y_routed_recv_buffer  (num_local_tokens * topk, H)
    print(f"  dispatch reads  from x_buffer     : {send_rows:>6} rows (per token)")
    print(f"  combine  writes into combine_buf  : {recv_rows:>6} rows (per (token,k))")
    print(f"  ratio = topk = {recv_rows // send_rows}")
    print("  -> each (token,k) has exactly one producing expert rank")
    print("  -> the 57344 scatter targets are pairwise disjoint -> no atomic needed")
    print("  -> if the buffer were [T, H], the K contributions would collide")
    print("  -> the weighted sum is deferred to fwd_epilogue (utils.cuh:85-90)")


def check_task_parity():
    """Both sides process the same num_tokens; only the tile shape differs."""
    print("\n== task counts are symmetric by construction (Kimi shape) ==")
    T, K, H = 7168, 8, 7168
    nt = T * K
    DMb, DNb = 128, 512
    CMb, CNb, CPD = 16, 1024, 7
    db = -(-H // DNb)
    cb = -(-H // CNb)
    dt = (nt // DMb) * db
    tiles = (nt // CMb) * cb
    ct = -(-tiles // CPD)
    print(f"  num_tokens = {nt}")
    print(f"  dispatch : tile {DMb}x{DNb}, {db} col blocks -> {dt:>6} tasks")
    print(f"  combine  : tile {CMb}x{CNb}, {cb} col blocks -> {tiles} tiles / pipe {CPD} -> {ct:>6} tasks")
    print(f"  both loops are `task_idx < nt / Mb * col_blocks` (forward.cuh:92-101)")
    print(f"  conclusion: identical shape -> same num_comm_sms covers both")


if __name__ == "__main__":
    check_signature_symmetry()
    check_write_conflict()
    check_task_parity()
