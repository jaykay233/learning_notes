"""用 Triton for-range 循环分块求每一行的和。"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _row_sum_kernel(
    x_ptr,
    y_ptr,
    n_cols: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(axis=0)
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    for tile_id in range(tl.cdiv(n_cols, BLOCK_SIZE)):
        cols = tile_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        values = tl.load(
            x_ptr + row * n_cols + cols,
            mask=cols < n_cols,
            other=0.0,
        )
        acc += values

    tl.store(y_ptr + row, tl.sum(acc, axis=0))


def row_sum(x: torch.Tensor, block_size: int = 4) -> torch.Tensor:
    """对二维 CUDA FP32 tensor 的每一行求和。"""
    if not x.is_cuda:
        raise ValueError("x must be a CUDA tensor")
    if x.dtype != torch.float32 or x.ndim != 2 or not x.is_contiguous():
        raise ValueError("x must be a contiguous 2-D torch.float32 tensor")
    if block_size <= 0:
        raise ValueError("block_size must be positive")

    n_rows, n_cols = x.shape
    y = torch.empty((n_rows,), device=x.device, dtype=torch.float32)
    if n_rows == 0:
        return y
    _row_sum_kernel[(n_rows,)](x, y, n_cols, BLOCK_SIZE=block_size)
    return y


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("需要可用的 CUDA GPU 才能运行 Triton kernel")

    x = torch.arange(1, 16, device="cuda", dtype=torch.float32).reshape(3, 5)
    y = row_sum(x, block_size=4)
    result = y.cpu().tolist()
    print(result)
    assert result == [15.0, 40.0, 65.0]


if __name__ == "__main__":
    main()
