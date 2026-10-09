"""Dense GeMM (bf16)：C = A @ B.T（Triton）

对应 CUDA：operators/code/cuda/bf16_gemm_at_bt.cu
A: (M, K), B: (N, K), C: (M, N)
C[i,j] = sum_k A[i,k] * B[j,k]
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _gemm_bt_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bn,
    stride_bk,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    # B 是 (N, K)：取 B[j,k]，等价于用 B 的行做点积
    b_ptrs = b_ptr + offs_n[None, :] * stride_bn + offs_k[:, None] * stride_bk

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        k_remaining = K - k0
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < k_remaining)
        b_mask = (offs_n[None, :] < N) & (offs_k[:, None] < k_remaining)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        acc += tl.dot(a, b)  # (BLOCK_M, BLOCK_K) @ (BLOCK_K, BLOCK_N)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c = acc.to(tl.bfloat16)
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def run_kernel(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    M: int | None = None,
    N: int | None = None,
    K: int | None = None,
) -> torch.Tensor:
    """
    写入 C = A @ B.T（bf16）。

    A: (M, K), B: (N, K), C: (M, N)，均为 cuda / bfloat16 / contiguous
    """
    assert A.is_cuda and B.is_cuda and C.is_cuda
    assert A.dtype == B.dtype == C.dtype == torch.bfloat16
    assert A.is_contiguous() and B.is_contiguous() and C.is_contiguous()

    M = int(A.shape[0] if M is None else M)
    K = int(A.shape[1] if K is None else K)
    N = int(B.shape[0] if N is None else N)
    assert B.shape == (N, K)
    assert C.shape == (M, N)

    BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    _gemm_bt_kernel[grid](
        A,
        B,
        C,
        M,
        N,
        K,
        A.stride(0),
        A.stride(1),
        B.stride(0),
        B.stride(1),
        C.stride(0),
        C.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )
    return C


if __name__ == "__main__":
    A = torch.tensor([[1, 2, 3], [4, 5, 6]], device="cuda", dtype=torch.bfloat16)
    B = torch.tensor([[7, 8, 9], [10, 11, 12]], device="cuda", dtype=torch.bfloat16)
    C = torch.empty(2, 2, device="cuda", dtype=torch.bfloat16)
    run_kernel(A, B, C)
    print(C)  # expected ≈ [[50, 68], [122, 167]]
