"""FP16 原地加法：A += B（Triton）

对应 CUDA：operators/code/cuda/fp16_inplace_add.cu
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _add_inplace_fp16_kernel(
    a_ptr,
    b_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    a = tl.load(a_ptr + offs, mask=mask)
    b = tl.load(b_ptr + offs, mask=mask)
    tl.store(a_ptr + offs, a + b, mask=mask)


def run_kernel(A: torch.Tensor, B: torch.Tensor, numel: int | None = None) -> torch.Tensor:
    """
    原地 A += B。

    参数
    ----
    A, B : torch.float16，同 shape，连续存储；A 会被原地修改
    numel : 可选，默认 A.numel()
    """
    assert A.is_cuda and B.is_cuda
    assert A.dtype == torch.float16 and B.dtype == torch.float16
    assert A.is_contiguous() and B.is_contiguous()
    assert A.shape == B.shape

    n = int(A.numel() if numel is None else numel)
    if n <= 0:
        return A

    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    _add_inplace_fp16_kernel[grid](A, B, n, BLOCK_SIZE=BLOCK_SIZE)
    return A


if __name__ == "__main__":
    a = torch.tensor([1, 2, 3, 4], device="cuda", dtype=torch.float16)
    b = torch.tensor([2, 3, 4, 5], device="cuda", dtype=torch.float16)
    run_kernel(a, b)
    print(a)  # expected: [3, 5, 7, 9]
