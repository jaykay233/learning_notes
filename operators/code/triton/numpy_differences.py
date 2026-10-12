"""Triton semantics that differ from NumPy/Python.

Run:
    python operators/code/triton/numpy_differences.py

Requires a CUDA GPU, PyTorch, and Triton.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _runtime_divmod_kernel(
    x_ptr,
    y_ptr,
    q_ptr,
    r_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0).to(tl.int32)
    y = tl.load(y_ptr + offsets, mask=mask, other=1).to(tl.int32)

    q = x // y  # Runtime Triton integer tensor: C-style truncation.
    r = x % y   # Defined as x - y * q.
    tl.store(q_ptr + offsets, q, mask=mask)
    tl.store(r_ptr + offsets, r, mask=mask)


@triton.jit
def _constexpr_divmod_kernel(
    q_ptr,
    r_ptr,
    A: tl.constexpr,
    B: tl.constexpr,
):
    # tl.constexpr arithmetic follows Python semantics: -7 // 3 == -3.
    q = A // B
    r = A % B
    tl.store(q_ptr, q)
    tl.store(r_ptr, r)


@triton.jit
def _safe_float_to_int8_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    # NaN has no portable integer result, so define it before casting.
    x = tl.where(x != x, 0.0, x)
    x = tl.clamp(x, -128.0, 127.0)
    out = x.to(tl.int8)
    tl.store(out_ptr + offsets, out, mask=mask)


@triton.jit
def _initialized_loop_value_kernel(
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Declared before the loop so it exists on every path after the loop.
    value = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for _ in range(1):
        value += 1.0

    tl.store(out_ptr + offsets, value, mask=mask)


def runtime_divmod(x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if not x.is_cuda or not y.is_cuda:
        raise ValueError("x and y must be CUDA tensors")
    if x.shape != y.shape or x.ndim != 1:
        raise ValueError("x and y must be same-shape 1-D tensors")

    q = torch.empty_like(x, dtype=torch.int32)
    r = torch.empty_like(x, dtype=torch.int32)
    n_elements = x.numel()
    if n_elements == 0:
        return q, r

    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    _runtime_divmod_kernel[grid](x, y, q, r, n_elements, BLOCK_SIZE=16)
    return q, r


def constexpr_divmod(a: int, b: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    q = torch.empty((1,), device=device, dtype=torch.int32)
    r = torch.empty((1,), device=device, dtype=torch.int32)
    _constexpr_divmod_kernel[(1,)](q, r, A=a, B=b)
    return q, r


def safe_float_to_int8(x: torch.Tensor) -> torch.Tensor:
    if not x.is_cuda:
        raise ValueError("x must be a CUDA tensor")
    if x.dtype != torch.float32 or x.ndim != 1:
        raise ValueError("x must be a 1-D float32 tensor")

    output = torch.empty_like(x, dtype=torch.int8)
    n_elements = x.numel()
    if n_elements == 0:
        return output

    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    _safe_float_to_int8_kernel[grid](x, output, n_elements, BLOCK_SIZE=16)
    return output


def initialized_loop_value(n_elements: int, device: torch.device) -> torch.Tensor:
    output = torch.empty((n_elements,), device=device, dtype=torch.float32)
    if n_elements == 0:
        return output

    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    _initialized_loop_value_kernel[grid](output, n_elements, BLOCK_SIZE=16)
    return output


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("需要 CUDA GPU 才能运行 Triton semantics kernel")

    device = triton.runtime.driver.active.get_active_torch_device()

    x = torch.tensor([-7, 7, -7, 7], device=device, dtype=torch.int32)
    y = torch.tensor([3, -3, -3, 3], device=device, dtype=torch.int32)
    q, r = runtime_divmod(x, y)
    q_ref = torch.trunc(x.to(torch.float32) / y.to(torch.float32)).to(torch.int32)
    r_ref = x - y * q_ref
    if not torch.equal(q, q_ref) or not torch.equal(r, r_ref):
        raise AssertionError("runtime div/mod does not match C-style truncation")

    q_const, r_const = constexpr_divmod(-7, 3, device)
    if q_const.item() != -3 or r_const.item() != 2:
        raise AssertionError("constexpr div/mod must follow Python semantics")

    cast_input = torch.tensor([510.0, -510.0, float("nan"), 12.5], device=device)
    cast_output = safe_float_to_int8(cast_input)
    cast_ref = torch.nan_to_num(cast_input, nan=0.0).clamp(-128.0, 127.0).to(torch.int8)
    if not torch.equal(cast_output, cast_ref):
        raise AssertionError("safe float-to-int8 conversion mismatch")

    loop_output = initialized_loop_value(4, device)
    if not torch.equal(loop_output, torch.ones((4,), device=device, dtype=torch.float32)):
        raise AssertionError("initialized loop value mismatch")

    print(f"runtime x: {x.tolist()}")
    print(f"runtime y: {y.tolist()}")
    print(f"runtime q (trunc): {q.tolist()}")
    print(f"runtime r: {r.tolist()}")
    print(f"constexpr -7 // 3: {q_const.item()}, -7 % 3: {r_const.item()}")
    print(f"cast input: {cast_input.tolist()}")
    print(f"cast output: {cast_output.tolist()}")
    print(f"loop output: {loop_output.tolist()}")
    print("all_checks_passed: True")


if __name__ == "__main__":
    main()
