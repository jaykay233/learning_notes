"""FP16 原地加法：A += B（TileLang）

对应 CUDA：operators/code/cuda/fp16_inplace_add.cu
对应 Triton：operators/code/triton/fp16_inplace_add.py

依赖：pip install tilelang
API 随 TileLang 版本可能略有差异，以当前安装的文档为准。
"""

from __future__ import annotations

import torch
import tilelang
import tilelang.language as T


def _make_kernel(numel: int, block_size: int = 256, dtype: str = "float16"):
    @T.prim_func
    def main(A: T.Tensor((numel,), dtype), B: T.Tensor((numel,), dtype)):
        with T.Kernel(T.ceildiv(numel, block_size), threads=block_size) as bx:
            for i in T.Parallel(block_size):
                idx = bx * block_size + i
                if idx < numel:
                    A[idx] = A[idx] + B[idx]

    return main


_compiled = {}


def run_kernel(A: torch.Tensor, B: torch.Tensor, numel: int | None = None) -> torch.Tensor:
    """原地 A += B（fp16）。"""
    assert A.is_cuda and B.is_cuda
    assert A.dtype == torch.float16 and B.dtype == torch.float16
    assert A.is_contiguous() and B.is_contiguous()
    assert A.shape == B.shape

    n = int(A.numel() if numel is None else numel)
    if n <= 0:
        return A

    key = (n, 256)
    if key not in _compiled:
        _compiled[key] = tilelang.compile(_make_kernel(n, block_size=256), out_idx=[])
    _compiled[key](A.view(-1), B.view(-1))
    return A


if __name__ == "__main__":
    a = torch.tensor([1, 2, 3, 4], device="cuda", dtype=torch.float16)
    b = torch.tensor([2, 3, 4, 5], device="cuda", dtype=torch.float16)
    run_kernel(a, b)
    print(a)  # expected: [3, 5, 7, 9]
