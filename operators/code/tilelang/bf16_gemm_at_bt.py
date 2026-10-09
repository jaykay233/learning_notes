"""Dense GeMM (bf16)：C = A @ B.T（TileLang）

对应 CUDA：operators/code/cuda/bf16_gemm_at_bt.cu
对应 Triton：operators/code/triton/bf16_gemm_at_bt.py

A: (M, K), B: (N, K), C: (M, N)
C[i,j] = sum_k A[i,k] * B[j,k]

依赖：pip install tilelang
API 随 TileLang 版本可能略有差异，以当前安装的文档为准。
"""

from __future__ import annotations

import torch
import tilelang
import tilelang.language as T


def _make_kernel(
    M: int,
    N: int,
    K: int,
    block_M: int = 64,
    block_N: int = 64,
    block_K: int = 32,
    dtype: str = "bfloat16",
):
    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((N, K), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(
            T.ceildiv(N, block_N),
            T.ceildiv(M, block_M),
            threads=128,
        ) as (bx, by):
            A_s = T.alloc_shared((block_M, block_K), dtype)
            B_s = T.alloc_shared((block_N, block_K), dtype)
            C_f = T.alloc_fragment((block_M, block_N), "float32")
            T.clear(C_f)

            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=2):
                T.copy(A[by * block_M : by * block_M + block_M, ko * block_K : ko * block_K + block_K], A_s)
                T.copy(B[bx * block_N : bx * block_N + block_N, ko * block_K : ko * block_K + block_K], B_s)
                # A_s @ B_s.T  => (block_M, block_K) @ (block_K, block_N)
                T.gemm(A_s, B_s, C_f, transpose_B=True)

            T.copy(C_f, C[by * block_M : by * block_M + block_M, bx * block_N : bx * block_N + block_N])

    return main


_compiled = {}


def run_kernel(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    M: int | None = None,
    N: int | None = None,
    K: int | None = None,
) -> torch.Tensor:
    """写入 C = A @ B.T（bf16）。"""
    assert A.is_cuda and B.is_cuda and C.is_cuda
    assert A.dtype == B.dtype == C.dtype == torch.bfloat16
    assert A.is_contiguous() and B.is_contiguous() and C.is_contiguous()

    M = int(A.shape[0] if M is None else M)
    K = int(A.shape[1] if K is None else K)
    N = int(B.shape[0] if N is None else N)
    assert B.shape == (N, K)
    assert C.shape == (M, N)

    key = (M, N, K)
    if key not in _compiled:
        prim = _make_kernel(M, N, K)
        _compiled[key] = tilelang.compile(prim, out_idx=[2])
    _compiled[key](A, B, C)
    return C


if __name__ == "__main__":
    A = torch.tensor([[1, 2, 3], [4, 5, 6]], device="cuda", dtype=torch.bfloat16)
    B = torch.tensor([[7, 8, 9], [10, 11, 12]], device="cuda", dtype=torch.bfloat16)
    C = torch.empty(2, 2, device="cuda", dtype=torch.bfloat16)
    run_kernel(A, B, C)
    print(C)  # expected ≈ [[50, 68], [122, 167]]
