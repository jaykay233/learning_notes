# Triton 官方教程 03：Blocked Matmul Kernel

## 本次讲解位置

章节：`chapter_triton_gemm`（算子开发 / Triton）<br>
小节：Triton 官方教程 03 的 `matmul_kernel`<br>
知识点：一个 program 如何计算一个 `BLOCK_SIZE_M x BLOCK_SIZE_N` 输出块，并沿 K 维循环累加<br>
上次：Triton 的 `for range` 分块与跨轮累加<br>
下次：`triton.autotune` 的多配置选择与 `num_stages` 软件流水<br>
PTX：不指定具体 PTX；本节停留在 Triton JIT、program grid、tile pointer 与 `tl.dot` 层

## 为什么现在讲这个

上一节的循环只有一个输出行，每轮加载一个一维 tile，最后做 reduction。这个模型不足以解释 GEMM，因为 GEMM 同时有三个关键约束：

1. 一个输出元素不是只读一段输入，而是沿 K 维做点积。
2. 一个 program 不应只算一个元素，否则指针和调度开销远大于计算量。
3. `M/N` 尾块可以只影响输出是否写回，但 `K` 尾块如果读进错误数值，会污染所有有效输出。

本节要建立的核心模型是：**grid 覆盖 C 的二维 tile，program 内部用 K-loop 完成 reduction**。理解这一点后，才能判断某个 program 拥有哪些输出、每轮 A/B 从哪里读、为什么 accumulator 必须是 FP32，以及 mask 应该加在 load 还是 store。

来源：[Triton Matrix Multiplication Tutorial](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html)。本文记录的是基于该教程整理的可复现版本。

## 先建立心智模型

计算定义为：

```text
C[M, N] = A[M, K] @ B[K, N]
```

先把输出切成二维小方块，再把每个方块交给一个 Triton program：

```text
C 的逻辑视图

        N
      ┌──────┬──────┬──────┬──────┐
   M  │ P0   │ P1   │ P2   │ P3   │
      ├──────┼──────┼──────┼──────┤
      │ P4   │ P5   │ P6   │ P7   │
      └──────┴──────┴──────┴──────┘

每个方格: 一个 BLOCK_SIZE_M x BLOCK_SIZE_N 输出 tile
每个方格: 一个 program 负责
方格内部: Triton 把 tile 工作分配到该 program 的 warps / lanes
```

维度职责可以压缩成下表：

| 维度 | Triton 中的角色 | 是否由 grid 切分 | 是否在 program 内循环 |
|---|---|---:|---:|
| `M` | C 的行、A 的行 | 是 | 否 |
| `N` | C 的列、B 的列 | 是 | 否 |
| `K` | A 的列、B 的行，也是 reduction 维 | 否 | 是 |

这个拆分有两个后果：

- `M/N` 的 tile 可以并行。不同 program 写入互不重叠的 C 区域，不需要跨 program 同步。
- `K` 的多个 tile 属于同一个输出元素。每个 program 必须在自己内部依次累加，不能让两个 program 同时写同一个 C 元素。

## 官方 blocked algorithm

教程开头用伪代码表达了整个算法：

```python
# M/N 方向的每个迭代由一个 program 并行执行
for m in range(0, M, BLOCK_SIZE_M):
    for n in range(0, N, BLOCK_SIZE_N):
        acc = zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=float32)
        for k in range(0, K, BLOCK_SIZE_K):
            a = A[m : m + BLOCK_SIZE_M, k : k + BLOCK_SIZE_K]
            b = B[k : k + BLOCK_SIZE_K, n : n + BLOCK_SIZE_N]
            acc += dot(a, b)
        C[m : m + BLOCK_SIZE_M, n : n + BLOCK_SIZE_N] = acc
```

真实 kernel 没有三段 Python 循环，而是：

- 最外层 `m/n` 两维被压成一个一维 program grid。
- program 内只保留 K-loop。
- 每个 K 轮加载一个 A tile `(BLOCK_M, BLOCK_K)` 和一个 B tile `(BLOCK_K, BLOCK_N)`。
- `tl.dot(a, b, acc)` 完成一个矩阵乘并累计到同一个 `(BLOCK_M, BLOCK_N)` accumulator。

## 完整可运行代码

文件：[`code/triton/matmul_tutorial.py`](code/triton/matmul_tutorial.py)

```python
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
```

## 第 1 层：grid 只覆盖 C 的 M/N 两维

启动时，官方代码使用一维 grid：

```python
grid = lambda META: (
    triton.cdiv(M, META["BLOCK_SIZE_M"])
    * triton.cdiv(N, META["BLOCK_SIZE_N"]),
)
```

所以 program 总数是：

```text
num_programs = ceil(M / BLOCK_M) * ceil(N / BLOCK_N)
```

注意这里没有 `ceil(K / BLOCK_K)`。K 循环发生在每个 program 内部。如果把 K 也放进 grid，多个 program 会争着写同一块 C；除非额外做 split-K 和 reduction，否则这不是这个教程采用的算法。

`grid` 写成 lambda 还有一个关键作用：autotune 选中某个配置后，才能用该配置里的 `BLOCK_SIZE_M/N` 计算准确的 grid 大小。因此调用 `matmul_kernel[grid]` 时，不把 `BLOCK_SIZE_M/N` 显式写死在 launch 参数里。

## 第 2 层：从一维 `pid` 还原 `(pid_m, pid_n)`

最直观的 row-major 映射是：

```python
pid_m = pid // num_pid_n
pid_n = pid % num_pid_n
```

这样 program 会按 `(0,0), (0,1), (0,2), ...` 逐列扫描第一行，再进入下一行。官方教程没有直接采用它，而是增加 `GROUP_SIZE_M`，把若干行组成一个 group，并在 group 内先沿着 N 走：

```python
num_pid_in_group = GROUP_SIZE_M * num_pid_n
group_id = pid // num_pid_in_group
first_pid_m = group_id * GROUP_SIZE_M
group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
pid_n = (pid % num_pid_in_group) // group_size_m
```

这不会改变数学结果，只改变 program 的启动顺序。它试图让附近的 program 复用 A 行块和 B 列块，提高 L2 命中率。

当 `GROUP_SIZE_M=1` 时，group 只有一个 M block，映射会退化成按行顺序扫描。`GROUP_SIZE_M` 越大，同一个 group 包含的 M block 越多；太大的 group 会增加 B tile 的复用距离，不一定更快。

## 第 3 层：二维 pointer grid

对 row-major 的二维 tensor，元素地址是：

```text
&X[row, col] = X + row * stride_row + col * stride_col
```

因此 A 的第一个 K block 可以写成：

```python
offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
offs_k = tl.arange(0, BLOCK_SIZE_K)
a_ptrs = a_ptr + (
    offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
)
```

各数组的 shape 和广播过程是：

```text
offs_am                  -> (BLOCK_M,)
offs_am[:, None]         -> (BLOCK_M, 1)
offs_k                   -> (BLOCK_K,)
offs_k[None, :]          -> (1, BLOCK_K)

两者相加并广播         -> (BLOCK_M, BLOCK_K)
a_ptrs                  -> 每个逻辑元素一个地址
```

B 的指针形状不同，因为 B 的逻辑布局是 `(K, N)`：

```python
offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
b_ptrs = b_ptr + (
    offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn
)
```

广播结果是：

```text
offs_k[:, None]          -> (BLOCK_K, 1)
offs_bn[None, :]         -> (1, BLOCK_N)
b_ptrs                   -> (BLOCK_K, BLOCK_N)
```

这样 `tl.load(a_ptrs)` 得到一个 `(BLOCK_M, BLOCK_K)` tile，`tl.load(b_ptrs)` 得到一个 `(BLOCK_K, BLOCK_N)` tile，二者的 inner K 维恰好匹配 `tl.dot`。

### 为什么 M/N 用取模，而 K 用 mask

`offs_am` 和 `offs_bn` 上的 `% M`、`% N` 会让尾块多出来的坐标回绕到合法范围内，避免直接生成越界地址。回绕读出的值只会参与无效 C 行或无效 C 列的计算，最终由 `c_mask` 阻止写回，所以不会污染有效输出。

但 K 维不能这样做。假设有效 K 只有 100，而 `BLOCK_K=32`，最后一轮 `offs_k` 是 100 到 127。如果把 K 坐标回绕到 0 到 27，那么有效输出元素会把本应不存在的 K=100..127 项当成额外乘积加进去。结果地址合法，但数值错误。因此 K 尾块必须在 load 时 mask：

```python
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
```

`other=0.0` 让无效位置参与点积时贡献零，不改变有效结果。

## 第 4 层：K-loop 和 FP32 accumulator

每一轮都执行下面的逐步数据流：

```text
a_ptrs 指向的 A tile
        |
        v
tl.load -> a: (BLOCK_M, BLOCK_K)

b_ptrs 指向的 B tile
        |
        v
tl.load -> b: (BLOCK_K, BLOCK_N)

acc: (BLOCK_M, BLOCK_N)
        |
        + tl.dot(a, b, acc)
        |
        v
新的 acc: (BLOCK_M, BLOCK_N)
```

`tl.dot(a, b, accumulator)` 做的是：

```text
acc_new = a @ b + acc_old
```

而不是覆盖 `acc_old`。这行代码把 K 维的一小段部分积及时并入同一个输出 tile。

accumulator 使用 FP32 有两个原因：

1. FP16 输入经过 K 次乘法累加后，中间误差会快速积累；FP32 accumulator 明显提高数值稳定性。
2. 只有在 K-loop 结束、激活函数完成之后，才执行一次 `accumulator.to(tl.float16)`，把最终输出压回 FP16。

因此 `tl.dot` 的数值语义是“FP16 乘、FP32 累加”。不能因为输入是 FP16，就把 accumulator 也定义成 FP16。

## 第 5 层：激活、cast 和最终 store

教程故意把 `leaky_relu` 放在 cast 之前：

```python
if ACTIVATION == "leaky_relu":
    accumulator = leaky_relu(accumulator)
c = accumulator.to(tl.float16)
```

这样激活函数在 FP32 accumulator 上计算，避免先降到 FP16 再激活带来额外精度损失。这也展示了 Triton 的融合能力：不需要先写一个 matmul kernel 到 global memory，再启动第二个 activation kernel。

最终 store 使用二维 mask：

```python
offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
tl.store(c_ptrs, c, mask=c_mask)
```

当 M 不是 `BLOCK_M` 的整数倍时，`offs_cm` 会包含越界行；当 N 不是 `BLOCK_N` 的整数倍时，`offs_cn` 会包含越界列。两者相与后，只有真实 C 区域会被写入。

## 具体执行轨迹：`pid=5`

使用如下小例子：

```text
M = 256
N = 256
K = 128
BLOCK_M = 128
BLOCK_N = 64
BLOCK_K = 32
GROUP_SIZE_M = 2
pid = 5
```

先计算 program 数量：

```text
num_pid_m = ceil(256 / 128) = 2
num_pid_n = ceil(256 / 64) = 4
num_programs = 2 * 4 = 8
```

再还原 `pid=5` 对应的输出 tile：

```text
num_pid_in_group = GROUP_SIZE_M * num_pid_n = 2 * 4 = 8
group_id = 5 // 8 = 0
first_pid_m = 0 * 2 = 0
group_size_m = min(2 - 0, 2) = 2

pid_m = 0 + ((5 % 8) % 2)
      = 0 + (5 % 2)
      = 1

pid_n = (5 % 8) // 2
      = 5 // 2
      = 2
```

于是该 program 拥有：

```text
C 行：pid_m * BLOCK_M 到 (pid_m + 1) * BLOCK_M
     = 128 到 256

C 列：pid_n * BLOCK_N 到 (pid_n + 1) * BLOCK_N
     = 128 到 192
```

它需要做：

```text
K 轮数 = ceil(K / BLOCK_K)
        = ceil(128 / 32)
        = 4
```

每轮的数据形状和 K 区间如下：

| K 轮次 | A tile | B tile | accumulator 更新 |
|---:|---|---|---|
| 0 | A[128:256, 0:32] | B[0:32, 128:192] | `acc += A0 @ B0` |
| 1 | A[128:256, 32:64] | B[32:64, 128:192] | `acc += A1 @ B1` |
| 2 | A[128:256, 64:96] | B[64:96, 128:192] | `acc += A2 @ B2` |
| 3 | A[128:256, 96:128] | B[96:128, 128:192] | `acc += A3 @ B3` |

这四个 `tl.dot` 不是四个独立输出 tile，而是在更新同一个 `(128, 64)` accumulator。循环结束后，accumulator 的每个元素就是对应 C 元素的完整 K 维点积。

用单个元素表示，例如 C 的行 `130`、列 `135`：

```text
C[130, 135]
  = sum(A[130, k] * B[k, 135], k=0..127)

K 轮 0: k=0..31
K 轮 1: k=32..63
K 轮 2: k=64..95
K 轮 3: k=96..127
```

源码看起来是四个矩阵 tile 运算，实际逻辑就是把这 128 个乘积分四批累加。

这个例子中单个 program 的逻辑资源量约为：

```text
A tile: 128 * 32 * 2 bytes = 8192 bytes
B tile:  32 * 64 * 2 bytes = 4096 bytes
acc:    128 * 64 * 4 bytes = 32768 bytes
```

这些量说明为什么 block size 不能无脑放大。真实资源占用还会受到寄存器分配、指令调度和 `num_stages` 流水线 buffer 数量的影响；但 accumulator 的 `BLOCK_M * BLOCK_N * 4 bytes` 是最直观的压力来源。

## 为什么 `GROUP_SIZE_M` 能改善 L2 复用

仍用 `num_pid_m=2`、`num_pid_n=4` 举例。普通 row-major 顺序是：

| 启动顺序 | `(pid_m, pid_n)` | 复用了什么 |
|---:|---|---|
| 0 | `(0, 0)` | 无 |
| 1 | `(0, 1)` | A row block 0 |
| 2 | `(0, 2)` | A row block 0 |
| 3 | `(0, 3)` | A row block 0 |
| 4 | `(1, 0)` | 无 |
| 5 | `(1, 1)` | A row block 1 |
| 6 | `(1, 2)` | A row block 1 |
| 7 | `(1, 3)` | A row block 1 |

在普通顺序里，A 行块的复用距离比较长，因为要等整个 N 方向扫完，才进入下一个 M 行。官方 grouped 顺序在 `GROUP_SIZE_M=2` 时是：

| 启动顺序 | `(pid_m, pid_n)` | 复用了什么 |
|---:|---|---|
| 0 | `(0, 0)` | 无 |
| 1 | `(1, 0)` | B column block 0 |
| 2 | `(0, 1)` | A row block 0 |
| 3 | `(1, 1)` | A row block 1、B column block 1 |
| 4 | `(0, 2)` | A row block 0 |
| 5 | `(1, 2)` | A row block 1、B column block 2 |
| 6 | `(0, 3)` | A row block 0 |
| 7 | `(1, 3)` | A row block 1、B column block 3 |

这种顺序让相邻 program 更可能在 L2 中复用同一块 A 或 B。它改变的是数据复用路径和 cache 命中率，不改变 `C = A @ B` 的数学结果。

## `tl.assume` 在本 kernel 中做什么

```python
tl.assume(pid_m >= 0)
tl.assume(pid_n >= 0)
tl.assume(stride_am > 0)
# ...
```

这些不是运行时检查，而是给编译器假设。特别是正的 stride 信息可以帮助 integer analysis 简化 offset 计算，减少地址运算指令。

如果传入负 stride 或某种非连续视图，假设就可能不再成立。本脚本在 host wrapper 中要求 `a` 和 `b` 都 contiguous，所以这里的正 stride 假设是成立的。把 wrapper 改成支持任意 stride 时，需要同步检查这些 assume 是否仍然正确。

## `autotune`、`num_warps` 和 `num_stages`

`@triton.autotune` 会在第一次遇到某个 `(M, N, K)` 时，依次测量配置列表，选出较快的配置并缓存。关键字段是：

| 字段 | 作用 | 过大的后果 |
|---|---|---|
| `BLOCK_SIZE_M/N` | 每个 program 拥有多大的输出 tile | 寄存器、accumulator、store 压力增大 |
| `BLOCK_SIZE_K` | 每轮 reduction 的宽度 | 更大的 dot，单轮加载更多数据 |
| `GROUP_SIZE_M` | program 的 L2 访问顺序 | 可能拉长复用距离，未必更快 |
| `num_warps` | 每个 CTA 使用多少 warps | 并行度提高，但资源分配也会变化 |
| `num_stages` | K-loop 软件流水的阶段数 | 增加 shared memory buffer 与流水复杂度 |

`key=["M", "N", "K"]` 表示这些维度变化时重新选配置。autotune 不会因为指针地址或数据内容变化就重新选择，因为性能主要由 shape 和硬件决定。

需要区分编译、autotune 和 steady-state benchmark：

```text
第一次调用某个 shape
  -> 编译多个 config
  -> 实际运行并测量每个 config
  -> 选中一个 config

后续同 shape 再调用
  -> 直接使用缓存配置
  -> 进入 steady-state 执行
```

因此不能把第一次调用的时间当最终性能。教程里的 `triton.testing.do_bench(lambda: matmul(a, b))` 会包含 warmup，然后再取稳态延迟分位数。

## 常见错误和症状

| 错误 | 症状 | 根因 |
|---|---|---|
| A/B stride 传反 | 结果随机错，或某些 tile 对、某些 tile 错 | A 是 `(M, K)`，B 是 `(K, N)`，两者 pointer grid 形状不同 |
| K 尾块忘记 mask | 小 K 或非整除 shape 出现系统性错误，地址可能合法但数值多算 | 回绕或越界位置被当成真实 reduction 项 |
| store 忘记 M/N mask | 尾块越界写、CUDA fault，或相邻内存被破坏 | grid 使用 `cdiv`，尾 tile 超出了真实输出范围 |
| accumulator 使用 FP16 | 误差随 K 增大明显变差 | reduction 中途过早降精度 |
| 把 `tl.arange` 当作固定物理 lane | 对 layout 和寄存器映射作出错误推断 | Triton 的逻辑 tile 到 lane 映射由编译器和 backend 决定 |
| 第一次 autotune 调用就拿来对比性能 | Triton 看起来远慢于 cuBLAS | 时间包含编译和多个候选的测速 |
| 只放大 block size，不看资源 | 编译失败、寄存器溢出、occupancy 下降或速度变差 | accumulator、shared memory pipeline 和寄存器压力同时增加 |
| 把 `GROUP_SIZE_M` 当数学参数 | 改变后结果完全一样，但无法解释性能异常 | 它只改变 program 启动顺序与 L2 复用 |

## 运行与验证边界

静态语法检查：

```bash
python3 -m py_compile operators/code/triton/matmul_tutorial.py
```

有 CUDA GPU、PyTorch 和 Triton 时，运行：

```bash
python operators/code/triton/matmul_tutorial.py --skip-bench
python operators/code/triton/matmul_tutorial.py --bench-size 4096
```

正确性部分会覆盖一个整齐 shape 和一个带尾块的 shape：

```text
shape=(512, 512, 512) max_abs_error=<hardware-dependent> matched=True
shape=(257, 331, 193) max_abs_error=<hardware-dependent> matched=True
```

benchmark 部分会输出类似：

```text
size=4096 cuBLAS=<cublas-ms> ms/<cublas-tflops> TFLOP/s Triton=<triton-ms> ms/<triton-tflops> TFLOP/s
```

教程页面在其测试环境中记录的 4096 方阵结果约为 cuBLAS `221.847 TFLOP/s`、Triton `220.753 TFLOP/s`。这只是该环境的观察结果，不应当作当前机器或所有 GPU 的保证。不同 GPU、Triton 版本、驱动、autotune 配置和功耗状态都会改变结果。

本节只完成静态语法验证。当前环境没有 Triton 和 CUDA GPU，因此不能声称这里的 kernel 已经在本机做过 runtime 验证。

## 自我检查（含答案）

1. **`M=N=256, BLOCK_M=128, BLOCK_N=64, BLOCK_K=32` 时，一维 grid 上有多少 program，K-loop 每轮执行多少次？**
   - **答：** program 数是 `ceil(256/128) * ceil(256/64) = 2 * 4 = 8`。每个 program 的 K-loop 轮数是 `ceil(256/32) = 8`。K 不参与 grid 大小。
2. **为什么 `offs_am` 和 `offs_bn` 可以取模，但 `offs_k` 不能只取模？**
   - **答：** M/N 尾块多出来的行或列最终不会写入 C，回绕不会影响有效输出；K 的无效项参与 reduction 会直接改变有效输出。K 尾块必须用 mask 和 `other=0.0` 排除。
3. **把 accumulator 定义为 `tl.float16` 为什么容易出错？**
   - **答：** K 次乘加会在 FP16 中反复舍入，误差随 K 累积；FP32 accumulator 保留更宽的部分和，最后才 cast 回 FP16。
4. **`pid=5, num_pid_m=2, num_pid_n=4, GROUP_SIZE_M=2` 时，程序拥有哪块 C？**
   - **答：** `group_size_m=2, pid_m=1, pid_n=2`。如果 block 是 `128x64`，它拥有 C 行 `[128, 256)`、列 `[128, 192)`。
5. **`GROUP_SIZE_M` 改变后，C 的结果会变吗？**
   - **答：** 不会。它只改变 program 的启动顺序和附近 program 对 A/B 的 L2 复用，不改变每个 program 负责的数学 tile。

## 进度

已经覆盖：

- [x] Triton `for range` 分块循环、尾块 mask、跨轮次累加与 `tl.static_range`。
- [x] Triton 官方 blocked GEMM：program 到 C tile 的映射、二维 pointer grid、K-loop + `tl.dot`、FP32 accumulator、M/N store mask、`GROUP_SIZE_M` 与 autotune 基本边界。

下一知识点：

- [ ] 单独拆解 `triton.autotune`：多配置选择、`num_stages` 软件流水、第一次编译开销与稳态 benchmark 的边界。
