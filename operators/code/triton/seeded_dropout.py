"""Triton 官方教程 04 的 low-memory dropout 可运行版本。

核心区别：

- baseline dropout：持久保存一个和输入同 shape 的 keep mask；
- seeded dropout：只保存一个 seed，每个元素按 (seed, global_offset) 现场生成随机数。

运行：
    python operators/code/triton/seeded_dropout.py

需要 CUDA GPU、PyTorch 和 Triton。
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


DEVICE = triton.runtime.driver.active.get_active_torch_device()


@triton.jit
def _dropout(
    x_ptr,
    x_keep_ptr,
    output_ptr,
    n_elements,
    p,
    BLOCK_SIZE: tl.constexpr,
):
    """读取预先生成的 keep mask，并完成 inverted dropout。"""
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    x_keep = tl.load(x_keep_ptr + offsets, mask=mask, other=0).to(tl.int1)
    output = tl.where(x_keep, x / (1 - p), 0.0)
    tl.store(output_ptr + offsets, output, mask=mask)


@triton.jit
def _seeded_dropout(
    x_ptr,
    output_ptr,
    n_elements,
    p,
    seed,
    BLOCK_SIZE: tl.constexpr,
):
    """不读取 mask，按 (seed, global offset) 现场生成随机数。"""
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    # random[i] 是 [0, 1) 上的均匀分布；相同 seed + offset 会得到相同值。
    random = tl.rand(seed, offsets)
    x_keep = random > p

    # inverted dropout：保留时放大 1 / (1 - p)，保持输出的期望不变。
    output = tl.where(x_keep, x / (1 - p), 0.0)
    tl.store(output_ptr + offsets, output, mask=mask)


def dropout_baseline(x: torch.Tensor, x_keep: torch.Tensor, p: float) -> torch.Tensor:
    """使用显式 keep mask 的 baseline。"""
    if not x.is_cuda or not x_keep.is_cuda:
        raise ValueError("x and x_keep must be CUDA tensors")
    if not x.is_contiguous() or not x_keep.is_contiguous():
        raise ValueError("x and x_keep must be contiguous")
    if x_keep.shape != x.shape:
        raise ValueError("x_keep must have the same shape as x")
    if not 0.0 <= p < 1.0:
        raise ValueError("p must satisfy 0 <= p < 1")
    if x.numel() == 0:
        return torch.empty_like(x)

    output = torch.empty_like(x)
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    _dropout[grid](
        x,
        x_keep,
        output,
        n_elements,
        p,
        BLOCK_SIZE=1024,
    )
    return output


def seeded_dropout(x: torch.Tensor, p: float, seed: int) -> torch.Tensor:
    """只保存 seed 的 dropout；同一 seed 会重放同一 mask。"""
    if not x.is_cuda:
        raise ValueError("x must be a CUDA tensor")
    if not x.is_contiguous():
        raise ValueError("x must be contiguous")
    if not 0.0 <= p < 1.0:
        raise ValueError("p must satisfy 0 <= p < 1")
    if p == 0.0:
        return x.clone()
    if x.numel() == 0:
        return torch.empty_like(x)

    output = torch.empty_like(x)
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    _seeded_dropout[grid](
        x,
        output,
        n_elements,
        p,
        seed,
        BLOCK_SIZE=1024,
    )
    return output


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("需要 CUDA GPU 才能运行 Triton dropout kernel")

    torch.manual_seed(0)
    x = torch.tensor(
        [1.48333, -0.239537, -0.640795, 1.62631, 0.263036,
         -0.71516, 1.99474, -1.09546, 1.81107, -0.170083],
        device=DEVICE,
        dtype=torch.float32,
    )
    p = 0.5

    baseline_keep = torch.rand((10,), device=DEVICE) > p
    baseline = dropout_baseline(x, baseline_keep, p)

    output_123 = seeded_dropout(x, p=p, seed=123)
    output_123_repeat = seeded_dropout(x, p=p, seed=123)
    output_512 = seeded_dropout(x, p=p, seed=512)

    same_seed_equal = torch.equal(output_123, output_123_repeat)
    different_seed_equal = torch.equal(output_123, output_512)
    if not same_seed_equal:
        raise AssertionError("same seed must reproduce the same dropout mask")
    if different_seed_equal:
        raise AssertionError("different seeds should normally produce different masks")

    kept = output_123 != 0
    dropped = ~kept
    scale = 1.0 / (1.0 - p)
    if not torch.allclose(output_123[kept], x[kept] * scale):
        raise AssertionError("kept elements must be scaled by 1 / (1 - p)")
    if not torch.all(output_123[dropped] == 0):
        raise AssertionError("dropped elements must become zero")

    print(f"input:              {x.tolist()}")
    print(f"baseline keep:      {baseline_keep.to(torch.int32).tolist()}")
    print(f"baseline output:    {baseline.tolist()}")
    print(f"seed=123 output:    {output_123.tolist()}")
    print(f"seed=123 repeat:    {output_123_repeat.tolist()}")
    print(f"seed=512 output:    {output_512.tolist()}")
    print(f"same_seed_equal:    {same_seed_equal}")
    print(f"different_seed_equal: {different_seed_equal}")


if __name__ == "__main__":
    main()
