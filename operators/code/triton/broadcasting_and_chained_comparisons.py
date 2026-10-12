"""Triton broadcasting and chained-comparison examples.

Run:
    python operators/code/triton/broadcasting_and_chained_comparisons.py

Requires a CUDA GPU, PyTorch, and Triton.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _broadcast_add_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
):
    rows = tl.arange(0, M)[:, None]
    cols = tl.arange(0, N)[None, :]

    a = tl.load(a_ptr + rows)  # [M, 1]
    b = tl.load(b_ptr + cols)  # [1, N]
    c = a + b                  # [M, N]

    tl.store(c_ptr + rows * N + cols, c)


@triton.jit
def _chained_mask_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    chained = 0 <= x <= 8
    explicit = (0 <= x) & (x <= 8)
    tl.static_assert(chained.dtype == tl.int1, "chained comparison must produce int1")
    tl.static_assert(explicit.dtype == tl.int1, "explicit comparison must produce int1")

    result = (chained & explicit).to(tl.int32)
    tl.store(out_ptr + offsets, result, mask=mask)


def broadcast_add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if not a.is_cuda or not b.is_cuda:
        raise ValueError("a and b must be CUDA tensors")
    if a.dtype != torch.float32 or b.dtype != torch.float32:
        raise ValueError("a and b must be float32")
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("a and b must be 2-D")
    if a.shape[1] != 1 or b.shape[0] != 1:
        raise ValueError("expected a=[M, 1] and b=[1, N]")

    m, n = a.shape[0], b.shape[1]
    output = torch.empty((m, n), device=a.device, dtype=torch.float32)
    _broadcast_add_kernel[(1,)](a, b, output, M=m, N=n)
    return output


def chained_mask(x: torch.Tensor) -> torch.Tensor:
    if not x.is_cuda:
        raise ValueError("x must be a CUDA tensor")
    if x.dtype != torch.float32 or x.ndim != 1:
        raise ValueError("x must be a 1-D float32 tensor")

    output = torch.empty_like(x, dtype=torch.int32)
    n_elements = x.numel()
    if n_elements == 0:
        return output

    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    _chained_mask_kernel[grid](x, output, n_elements, BLOCK_SIZE=16)
    return output


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("需要 CUDA GPU 才能运行 Triton broadcasting/comparison kernel")

    device = triton.runtime.driver.active.get_active_torch_device()

    a = torch.tensor([[1.0], [2.0], [3.0]], device=device, dtype=torch.float32)
    b = torch.tensor([[10.0, 20.0, 30.0, 40.0]], device=device, dtype=torch.float32)
    c = broadcast_add(a, b)
    c_ref = a + b
    if not torch.equal(c, c_ref):
        raise AssertionError(f"broadcast output mismatch: {c.tolist()} vs {c_ref.tolist()}")

    x = torch.tensor([-1.0, 0.0, 3.0, 8.0, 9.0, float("nan")], device=device)
    mask = chained_mask(x)
    mask_ref = ((0 <= x) & (x <= 8)).to(torch.int32)
    if not torch.equal(mask, mask_ref):
        raise AssertionError(f"chained mask mismatch: {mask.tolist()} vs {mask_ref.tolist()}")

    print(f"a shape={tuple(a.shape)}, b shape={tuple(b.shape)}, c shape={tuple(c.shape)}")
    print(f"broadcast result: {c.tolist()}")
    print(f"x: {x.tolist()}")
    print(f"0 <= x <= 8: {mask.tolist()}")
    print("all_checks_passed: True")


if __name__ == "__main__":
    main()
