"""Triton type-promotion examples with static and runtime checks.

Run:
    python operators/code/triton/type_promotion.py

Requires a CUDA GPU, PyTorch, and Triton.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import triton
import triton.language as tl


@dataclass(frozen=True)
class PromotionCase:
    case_id: int
    name: str
    torch_result_dtype: torch.dtype


CASES = (
    PromotionCase(0, "fp16 + fp32 -> fp32", torch.float32),
    PromotionCase(1, "fp16 + bf16 -> fp16", torch.float16),
    PromotionCase(2, "int16 + 4.0 -> fp32", torch.float32),
    PromotionCase(3, "fp16 + 1 -> fp16", torch.float16),
    PromotionCase(4, "fp16 + 1.0 -> fp16", torch.float16),
    PromotionCase(5, "fp16 / fp16 -> fp32", torch.float32),
    PromotionCase(6, "tl.where(cond, fp16, fp32) -> fp32", torch.float32),
)


@triton.jit
def _promotion_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    CASE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    if CASE == 0:
        # Higher kind wins: floating keeps the result out of the integer path.
        x = tl.load(x_ptr + offsets, mask=mask, other=0).to(tl.float16)
        y = tl.load(y_ptr + offsets, mask=mask, other=0).to(tl.float32)
        result = x + y
        tl.static_assert(result.dtype == tl.float32, "fp16 + fp32 must be fp32")
    elif CASE == 1:
        # Same kind and width; the documented pair fp16/bf16 prefers fp16.
        x = tl.load(x_ptr + offsets, mask=mask, other=0).to(tl.float16)
        y = tl.load(y_ptr + offsets, mask=mask, other=0).to(tl.bfloat16)
        result = x + y
        tl.static_assert(result.dtype == tl.float16, "fp16 + bf16 must be fp16")
    elif CASE == 2:
        # A float scalar has a higher kind than an integer tensor, so choose
        # the narrowest fitting scalar dtype: fp32.
        x = tl.load(x_ptr + offsets, mask=mask, other=0).to(tl.int16)
        result = x + 4.0
        tl.static_assert(result.dtype == tl.float32, "int16 + 4.0 must be fp32")
    elif CASE == 3:
        # An integer scalar does not promote a floating tensor.
        x = tl.load(x_ptr + offsets, mask=mask, other=0).to(tl.float16)
        result = x + 1
        tl.static_assert(result.dtype == tl.float16, "fp16 + 1 must be fp16")
    elif CASE == 4:
        # A scalar of the same kind also does not promote a tensor.
        x = tl.load(x_ptr + offsets, mask=mask, other=0).to(tl.float16)
        result = x + 1.0
        tl.static_assert(result.dtype == tl.float16, "fp16 + 1.0 must be fp16")
    elif CASE == 5:
        # Division is an exception: narrow floating division computes in fp32.
        x = tl.load(x_ptr + offsets, mask=mask, other=0).to(tl.float16)
        y = tl.load(y_ptr + offsets, mask=mask, other=1).to(tl.float16)
        result = x / y
        tl.static_assert(result.dtype == tl.float32, "fp16 / fp16 must be fp32")
    elif CASE == 6:
        # The last two arguments of tl.where use the same promotion rules.
        condition = offsets < BLOCK_SIZE // 2
        x = tl.load(x_ptr + offsets, mask=mask, other=0).to(tl.float16)
        y = tl.load(y_ptr + offsets, mask=mask, other=0).to(tl.float32)
        result = tl.where(condition, x, y)
        tl.static_assert(result.dtype == tl.float32, "tl.where fp16/fp32 must be fp32")
    else:
        tl.static_assert(False, "unknown CASE")

    tl.store(out_ptr + offsets, result, mask=mask)


def _reference(case: PromotionCase, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x32 = x.to(torch.float32)
    y32 = y.to(torch.float32)

    if case.case_id == 0:
        result = x32 + y32
    elif case.case_id == 1:
        result = x32 + y32
    elif case.case_id == 2:
        result = x32 + 4.0
    elif case.case_id == 3:
        result = x32 + 1.0
    elif case.case_id == 4:
        result = x32 + 1.0
    elif case.case_id == 5:
        result = x32 / y32
    elif case.case_id == 6:
        condition = torch.arange(x.numel(), device=x.device) < x.numel() // 2
        result = torch.where(condition, x32, y32)
    else:
        raise ValueError(f"unknown case id: {case.case_id}")

    return result.to(case.torch_result_dtype)


def _run_case(case: PromotionCase, device: torch.device) -> None:
    x = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8], device=device, dtype=torch.float32)
    y = torch.tensor([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], device=device, dtype=torch.float32)
    output = torch.empty_like(x, dtype=case.torch_result_dtype)
    n_elements = x.numel()

    _promotion_kernel[(1,)](
        x,
        y,
        output,
        n_elements,
        CASE=case.case_id,
        BLOCK_SIZE=8,
    )

    reference = _reference(case, x, y)
    if output.dtype != case.torch_result_dtype:
        raise AssertionError(f"{case.name}: dtype={output.dtype}, expected={case.torch_result_dtype}")
    if not torch.equal(output, reference):
        raise AssertionError(f"{case.name}: output={output.tolist()}, reference={reference.tolist()}")

    print(f"{case.name}: dtype={output.dtype}, output={output.tolist()}")


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("需要 CUDA GPU 才能运行 Triton type-promotion kernel")

    device = triton.runtime.driver.active.get_active_torch_device()
    for case in CASES:
        _run_case(case, device)
    print("all_checks_passed: True")


if __name__ == "__main__":
    main()
