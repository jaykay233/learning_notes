"""Reference and tolerance validation for FlashAttention-style outputs.

Run with a Python environment that contains PyTorch:

    python3 flash_attention_reference_validation.py

The script does not require CUDA. It compares three things:

1. A fp64 attention oracle, used as the mathematical truth.
2. A simulated kernel path with fp16 inputs, fp32 accumulation, fp16 P, and
   fp16 output, used to model the actual numeric contract.
3. PyTorch SDPA on a small equal-length case, used as a second implementation.

It also injects mask, scale, and head-layout errors and confirms that the same
validator rejects each one.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class Comparison:
    name: str
    max_abs: float
    max_allowed: float
    max_ratio: float
    cosine: float
    passed: bool


def _repeat_kv_heads(x: torch.Tensor, num_q_heads: int) -> torch.Tensor:
    """Expand [B, S, Hkv, D] to [B, S, Hq, D]."""
    num_kv_heads = x.shape[2]
    if num_q_heads % num_kv_heads != 0:
        raise ValueError("num_q_heads must be divisible by num_kv_heads")
    if num_q_heads == num_kv_heads:
        return x
    return x.repeat_interleave(num_q_heads // num_kv_heads, dim=2)


def causal_mask(
    seq_len_q: int,
    seq_len_kv: int,
    *,
    device: torch.device,
) -> torch.Tensor:
    """Bottom-right aligned causal mask for possibly unequal sequence lengths.

    A query row q may attend to key column k iff:

        k <= q + (seq_len_kv - seq_len_q)

    The offset keeps the last query aligned with the last key. This is the
    convention used by the FP64 reference in flash_attention4_fp4.py.
    """
    row = torch.arange(seq_len_q, device=device)[:, None]
    col = torch.arange(seq_len_kv, device=device)[None, :]
    return col <= row + (seq_len_kv - seq_len_q)


def reference_attention_fp64(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale: float,
    *,
    causal: bool,
) -> torch.Tensor:
    """Plain attention in fp64, with GQA expansion by repeating K/V heads."""
    b, sq, hq, _ = q.shape
    k_expanded = _repeat_kv_heads(k, hq)
    v_expanded = _repeat_kv_heads(v, hq)

    q64 = q.double().permute(0, 2, 1, 3)
    k64 = k_expanded.double().permute(0, 2, 1, 3)
    v64 = v_expanded.double().permute(0, 2, 1, 3)

    scores = torch.matmul(q64, k64.transpose(-1, -2)) * softmax_scale
    if causal:
        keep = causal_mask(sq, k.shape[1], device=q.device)
        scores = scores.masked_fill(~keep, float("-inf"))

    probs = torch.softmax(scores, dim=-1)
    out = torch.matmul(probs, v64)
    return out.permute(0, 2, 1, 3).contiguous()


def simulate_kernel_precision(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale: float,
    *,
    causal: bool,
) -> torch.Tensor:
    """Approximate the FA4 numeric path without running a GPU kernel.

    The input tensors are fp16. Scores and accumulation are fp32. P is rounded
    to fp16 before the PV matmul, and the final result is stored as fp16.
    """
    b, sq, hq, _ = q.shape
    k_expanded = _repeat_kv_heads(k, hq).float()
    v_expanded = _repeat_kv_heads(v, hq).float()

    q_fp32 = q.float().permute(0, 2, 1, 3)
    k_fp32 = k_expanded.permute(0, 2, 1, 3)
    v_fp32 = v_expanded.permute(0, 2, 1, 3)

    scores = torch.matmul(q_fp32, k_fp32.transpose(-1, -2)) * softmax_scale
    if causal:
        keep = causal_mask(sq, k.shape[1], device=q.device)
        scores = scores.masked_fill(~keep, float("-inf"))

    probs_fp32 = torch.softmax(scores, dim=-1)
    probs_fp16_for_pv = probs_fp32.half().float()
    out_fp32 = torch.matmul(probs_fp16_for_pv, v_fp32)
    out_fp16 = out_fp32.permute(0, 2, 1, 3).contiguous().half()
    return out_fp16


def compare(
    name: str,
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    rtol: float = 1e-2,
    atol: float = 1e-2,
) -> Comparison:
    """Measure the same pointwise inequality used by assert_close."""
    if actual.shape != expected.shape:
        raise AssertionError(
            f"{name}: shape mismatch, actual={tuple(actual.shape)}, "
            f"expected={tuple(expected.shape)}"
        )

    actual64 = actual.double()
    expected64 = expected.double()
    abs_error = (actual64 - expected64).abs()
    allowed = atol + rtol * expected64.abs()
    max_ratio = (abs_error / allowed).max().item()

    cosine = F.cosine_similarity(
        actual64.flatten(),
        expected64.flatten(),
        dim=0,
        eps=1e-12,
    ).item()

    max_abs = abs_error.max().item()
    max_allowed = allowed.max().item()
    passed = max_ratio <= 1.0

    if not passed:
        raise AssertionError(
            f"{name}: tolerance failed, max_ratio={max_ratio:.6g}, "
            f"max_abs={max_abs:.6g}, max_allowed={max_allowed:.6g}"
        )

    return Comparison(
        name=name,
        max_abs=max_abs,
        max_allowed=max_allowed,
        max_ratio=max_ratio,
        cosine=cosine,
        passed=passed,
    )


def expect_failure(
    name: str,
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    rtol: float = 1e-2,
    atol: float = 1e-2,
) -> str:
    try:
        compare(name, actual, expected, rtol=rtol, atol=atol)
    except AssertionError as exc:
        return str(exc).splitlines()[0]
    raise AssertionError(f"{name}: validator unexpectedly accepted a bad output")


def swap_two_heads(x: torch.Tensor) -> torch.Tensor:
    if x.shape[2] < 2:
        raise ValueError("need at least two heads")
    result = x.clone()
    result[:, :, 0, :], result[:, :, 1, :] = (
        x[:, :, 1, :].clone(),
        x[:, :, 0, :].clone(),
    )
    return result


def print_comparison(label: str, result: Comparison) -> None:
    print(f"{label}:")
    print(f"  max_abs_error   = {result.max_abs:.8e}")
    print(f"  max_allowed     = {result.max_allowed:.8e}")
    print(f"  max_error_ratio = {result.max_ratio:.6f}  (pass iff <= 1)")
    print(f"  cosine_similarity = {result.cosine:.10f}")


def print_tolerance_trace(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    rtol: float,
    atol: float,
) -> None:
    """Print one concrete element and its pointwise tolerance calculation."""
    actual64 = actual.double().flatten()
    expected64 = expected.double().flatten()
    abs_error = (actual64 - expected64).abs()
    index = int(abs_error.argmax().item())
    allowed = atol + rtol * expected64[index].abs()
    print("worst element trace:")
    print(f"  flat_index = {index}")
    print(f"  actual     = {actual64[index].item():.10f}")
    print(f"  expected   = {expected64[index].item():.10f}")
    print(f"  abs_error  = {abs_error[index].item():.10e}")
    print(f"  allowed    = atol + rtol * abs(expected)")
    print(
        f"             = {atol:.1e} + {rtol:.1e} * "
        f"{expected64[index].abs().item():.10f}"
        f" = {allowed.item():.10e}"
    )
    print(f"  ratio      = {abs_error[index].item() / allowed.item():.6f}")


def run_small_trace() -> None:
    """A 1x2 problem that can be followed by hand."""
    q = torch.tensor([[[[1.0, 0.0]], [[0.0, 1.0]]]], dtype=torch.float64)
    k = torch.tensor([[[[1.0, 0.0]], [[0.0, 1.0]]]], dtype=torch.float64)
    v = torch.tensor([[[[10.0, 0.0]], [[0.0, 20.0]]]], dtype=torch.float64)
    scale = 1.0 / math.sqrt(2.0)

    full = reference_attention_fp64(q, k, v, scale, causal=False)
    causal = reference_attention_fp64(q, k, v, scale, causal=True)
    print("manual 2x2 trace:")
    print("  scores = [[1, 0], [0, 1]] / sqrt(2)")
    print(f"  full   = {full.flatten().tolist()}")
    print(f"  causal = {causal.flatten().tolist()}")


def main() -> None:
    torch.manual_seed(0)

    batch_size = 2
    seq_len_q = 6
    seq_len_kv = 8
    num_q_heads = 4
    num_kv_heads = 2
    head_dim = 8
    scale = 1.0 / math.sqrt(head_dim)

    q = torch.randn(
        batch_size,
        seq_len_q,
        num_q_heads,
        head_dim,
        dtype=torch.float16,
    ) * 0.25
    k = torch.randn(
        batch_size,
        seq_len_kv,
        num_kv_heads,
        head_dim,
        dtype=torch.float16,
    ) * 0.25
    v = torch.randn(
        batch_size,
        seq_len_kv,
        num_kv_heads,
        head_dim,
        dtype=torch.float16,
    ) * 0.25

    print("input shapes:")
    print(f"  Q = {tuple(q.shape)}  [B, Sq, Hq, D]")
    print(f"  K = {tuple(k.shape)}  [B, Skv, Hkv, D]")
    print(f"  V = {tuple(v.shape)}  [B, Skv, Hkv, D]")
    print(f"  softmax_scale = {scale:.10f}")

    reference = reference_attention_fp64(q, k, v, scale, causal=True)
    kernel_like = simulate_kernel_precision(q, k, v, scale, causal=True)
    actual_fp32 = kernel_like.float()

    direct_kernel_vs_fp64 = compare(
        "cast(kernel_output, fp32) vs fp64 oracle",
        actual_fp32,
        reference,
    )
    print_comparison("kernel precision path vs fp64 oracle", direct_kernel_vs_fp64)
    print_tolerance_trace(actual_fp32, reference, rtol=1e-2, atol=1e-2)

    # Second implementation check on equal sequence lengths. In this case
    # PyTorch's is_causal convention and the bottom-right mask coincide.
    sdpa_k = _repeat_kv_heads(k[:, :seq_len_q], num_q_heads)
    sdpa_v = _repeat_kv_heads(v[:, :seq_len_q], num_q_heads)
    sdpa_out = F.scaled_dot_product_attention(
        q.transpose(1, 2),
        sdpa_k.transpose(1, 2)[:, :, :seq_len_q, :],
        sdpa_v.transpose(1, 2)[:, :, :seq_len_q, :],
        is_causal=True,
    ).transpose(1, 2).float()

    sdpa_ref = reference_attention_fp64(
        q,
        k[:, :seq_len_q],
        v[:, :seq_len_q],
        scale,
        causal=True,
    )
    sdpa_comparison = compare("PyTorch SDPA vs fp64 oracle", sdpa_out, sdpa_ref)
    print_comparison("PyTorch SDPA vs fp64 oracle", sdpa_comparison)

    bad_mask = simulate_kernel_precision(q, k, v, scale, causal=False)
    bad_scale = simulate_kernel_precision(q, k, v, 1.0, causal=True)
    bad_layout = swap_two_heads(kernel_like)

    failures = [
        expect_failure("mask error", bad_mask.float(), reference),
        expect_failure("scale error", bad_scale.float(), reference),
        expect_failure("head-layout error", bad_layout.float(), reference),
    ]
    print("expected rejected failures:")
    for failure in failures:
        print(f"  {failure}")

    run_small_trace()

    # assert_close checks dtype by default. The FP64 oracle remains the source
    # of truth for error statistics; for this dtype-checked call, compare the
    # rounded kernel output against the oracle rounded to fp32.
    torch.testing.assert_close(
        actual_fp32,
        reference.float(),
        rtol=1e-2,
        atol=1e-2,
        msg="FA-style output must remain inside the reference tolerance",
    )
    print("torch.testing.assert_close: PASS")


if __name__ == "__main__":
    main()
