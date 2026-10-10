"""Triton 官方教程 03 的 FP16 blocked GEMM 可运行版本。

C[M, N] = A[M, K] @ B[K, N]

每个 Triton program 负责一个 BLOCK_SIZE_M x BLOCK_SIZE_N 的 C 输出块，
并沿 K 维分块加载 A/B，使用 FP32 accumulator 做 tl.dot 累加。

运行：
    python operators/code/triton/matmul_tutorial.py
    python operators/code/triton/matmul_tutorial.py --skip-bench

需要 CUDA GPU、PyTorch 和 Triton。
"""

from __future__ import annotations

import argparse

import torch
import triton
import triton.language as tl


def get_autotune_config() -> list[triton.Config]:
    """返回官方教程中的一组代表性 CUDA 配置。"""
    return [
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 8},
            num_stages=3,
            num_warps=8,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8},
            num_stages=5,
            num_warps=2,
        ),
    ]


@triton.autotune(
    configs=get_autotune_config(),
    key=["M", "N", "K"],
)
@triton.jit
def matmul_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    ACTIVATION: tl.constexpr,
):
    """计算 C = A @ B，A 为 (M, K)，B 为 (K, N)，C 为 (M, N)。"""
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    # 在 group 内按列优先排列 program：先走同一组 M 行，再换 N 列。
    # 这样前后启动的 program 更容易复用 A/B 的 L2 cache line。
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # 给整数范围分析提供约束，让后端更积极地优化地址计算。
    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)
    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_cm > 0)
    tl.assume(stride_cn > 0)

    # 初始化 A/B 第一个 K block 的二维指针网格。
    # 对 M/N 取模不会改变有效输出，却能让越界地址仍落在合法内存中。
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (
        offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
    )
    b_ptrs = b_ptr + (
        offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn
    )

    # K 是 reduction 维：program 内顺序循环，tl.dot 把每轮 tile 积累加到 acc。
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        k_remaining = K - k * BLOCK_SIZE_K
        a = tl.load(
            a_ptrs,
            mask=offs_k[None, :] < k_remaining,
            other=0.0,
        )
        b = tl.load(
            b_ptrs,
            mask=offs_k[:, None] < k_remaining,
            other=0.0,
        )
        accumulator = tl.dot(a, b, accumulator)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    if ACTIVATION == "leaky_relu":
        accumulator = leaky_relu(accumulator)
    c = accumulator.to(tl.float16)

    # 写回 C tile；M/N 尾块只在这里阻止越界 store。
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


@triton.jit
def leaky_relu(x):
    return tl.where(x >= 0, x, 0.01 * x)


def matmul(a: torch.Tensor, b: torch.Tensor, activation: str = "") -> torch.Tensor:
    """运行 Triton blocked GEMM，并与官方教程保持相同的输入输出约定。"""
    if not (a.is_cuda and b.is_cuda):
        raise ValueError("a and b must be CUDA tensors")
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("a and b must be 2-D tensors")
    if a.dtype != torch.float16 or b.dtype != torch.float16:
        raise ValueError("a and b must be torch.float16 for this tutorial")
    if not (a.is_contiguous() and b.is_contiguous()):
        raise ValueError("a and b must be contiguous")
    if a.device != b.device:
        raise ValueError("a and b must be on the same device")
    if a.shape[1] != b.shape[0]:
        raise ValueError("Incompatible dimensions")
    if activation not in ("", "leaky_relu"):
        raise ValueError("activation must be '' or 'leaky_relu'")

    M, K = a.shape
    _, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    if M == 0 or N == 0:
        return c

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"])
        * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )
    matmul_kernel[grid](
        a,
        b,
        c,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        ACTIVATION=activation,
    )
    return c


def check_correctness(M: int, N: int, K: int) -> None:
    torch.manual_seed(0)
    a = torch.rand((M, K), device="cuda", dtype=torch.float16) - 0.5
    b = torch.rand((K, N), device="cuda", dtype=torch.float16) - 0.5
    triton_output = matmul(a, b)
    torch_output = torch.matmul(a, b)
    max_abs_error = (triton_output.float() - torch_output.float()).abs().max().item()
    matched = torch.allclose(triton_output, torch_output, atol=1e-2, rtol=0)
    print(
        f"shape=({M}, {N}, {K}) "
        f"max_abs_error={max_abs_error:.6f} matched={matched}"
    )
    if not matched:
        raise AssertionError("Triton output does not match torch.matmul")


def benchmark(size: int) -> None:
    torch.manual_seed(0)
    a = torch.randn((size, size), device="cuda", dtype=torch.float16)
    b = torch.randn((size, size), device="cuda", dtype=torch.float16)
    quantiles = [0.5, 0.2, 0.8]
    cublas_ms, _, _ = triton.testing.do_bench(
        lambda: torch.matmul(a, b),
        quantiles=quantiles,
    )
    triton_ms, _, _ = triton.testing.do_bench(
        lambda: matmul(a, b),
        quantiles=quantiles,
    )

    def tflops(ms: float) -> float:
        return 2 * size * size * size * 1e-12 / (ms * 1e-3)

    print(
        f"size={size} "
        f"cuBLAS={cublas_ms:.3f} ms/{tflops(cublas_ms):.3f} TFLOP/s "
        f"Triton={triton_ms:.3f} ms/{tflops(triton_ms):.3f} TFLOP/s"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench-size", type=int, default=4096)
    parser.add_argument("--skip-bench", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("需要 CUDA GPU 才能运行 Triton matmul kernel")

    check_correctness(512, 512, 512)
    check_correctness(257, 331, 193)
    if not args.skip_bench:
        benchmark(args.bench_size)


if __name__ == "__main__":
    main()
