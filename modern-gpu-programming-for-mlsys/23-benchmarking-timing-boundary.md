# 23. GPU Benchmark 的计时边界

课程位置：

- `appendix/benchmarking_gpu_kernels`
- `Define the Timing Boundary`

本次讲解位置

```text
章节：appendix/benchmarking_gpu_kernels
小节：Define the Timing Boundary
知识点：CUDA event、GPU stream time 与 synchronized host time 的边界
上次：Roofline 用 arithmetic intensity 判断 kernel 的理论瓶颈
下次：warm-up、repeat、rounds 与稳定样本
PTX：无；本节讨论 CUDA event 与 host/device 时间线
```

## 为什么现在讲这个

Roofline 只能回答一个 kernel 在理论上更可能受 compute 还是 memory 限制。
它不能告诉你程序实际花了多久，也不能自动决定应该把哪些操作放进测量区间。

如果两个实现使用不同的计时边界，测出的数字会直接失去可比性。例如：

```text
实现 A：只测 GEMM kernel
实现 B：测 GEMM + ReLU + output 转换
```

即使 B 的 GPU 时间更大，也不能据此说 B 的 GEMM 更慢。它只是做了更多工作。
本节要建立的第一条规则是：

```text
先定义一次 operation 包含什么，再选择计时器，最后才比较数字。
```

## 一、计时边界的心理模型

把 benchmark 想象成在时间线上放两把“切割刀”：

```text
边界外          边界内                         边界外
allocation  ->  GEMM -> ReLU -> synchronization  -> comparison
             ^                                  ^
           start                               end
```

只有位于 start 和 end 之间、或由边界内操作触发的 GPU work，才属于本次测量。

对于同一个 `run()`，至少可以定义三种不同的边界：

| 边界 | 覆盖内容 | 数字回答的问题 |
|---|---|---|
| `torch.mm` 外的 CUDA event | 仅 GEMM 的 GPU-stream 时间 | GEMM kernel 本身多快 |
| 完整 `run()` 外的 CUDA event | GEMM + ReLU，以及事件之间同一 stream 上的空档 | fused operator 的 GPU-stream 时间 |
| 完整 `run()` 外的 synchronized host timer | Python 调用、launch、GPU 执行、等待完成 | 一次 Python API call 的端到端延迟 |

三个边界通常是嵌套关系：

```text
GEMM GPU time
  <= GEMM + ReLU GPU-stream time
  <= 一次同步 host call 的端到端时间
```

但它们不是简单的数学包含关系。CUDA event 记录的是 stream 到达某个位置的设备
时间线；如果事件之间有 idle gap，gap 也会被算进去。

## 二、两个计时器分别在测什么

### CUDA event

```python
start.record()
fn()
end.record()
end.synchronize()
elapsed_ms = start.elapsed_time(end)
```

`record()` 把事件放到当前 CUDA stream。GPU 到达 `start` 时记开始时间，到达
`end` 时记结束时间。结果是设备 stream 上的一段区间。

它适合：

```text
测一个 kernel 或一个由同一 stream 串起来的 GPU operation
排除 Python dispatch 与 host 等待开销
```

它不自动解决：

```text
同一个 stream 上的 idle gap
其他 stream 上的工作
没有建立跨 stream dependency 的 operation
```

多 stream operation 必须显式建立依赖，让所有 branch 在 start 后开始，并在
end 前完成。

### Synchronized wall-clock timer

```python
torch.cuda.synchronize()
t0 = time.perf_counter()
fn()
torch.cuda.synchronize()
t1 = time.perf_counter()
```

第一次 synchronize 是为了清空前一次调用尚未完成的 GPU work。第二次
synchronize 等待本次 `fn()` 提交的 GPU work 全部完成。

结果包含：

```text
Python 调用
框架 dispatch
CUDA launch
GPU 执行
host 等待 GPU 完成
```

它回答的是“用户调用一次这个 API 要等多久”，不是“kernel 在 GPU 上运行多久”。

## 三、完整可运行代码

文件：

```text
modern-gpu-programming-for-mlsys/code/benchmark_timing_boundary.py
```

```python
#!/usr/bin/env python3
"""Demonstrate why a GPU benchmark must declare its timing boundary.

On a CUDA machine this script measures:

1. CUDA-event time around GEMM only.
2. CUDA-event time around GEMM plus ReLU.
3. Synchronized host time around one complete GEMM-plus-ReLU call.

On a machine without CUDA, ``--mode dry-run`` verifies the boundary accounting
with deterministic synthetic durations.
"""

from __future__ import annotations

import argparse
import time
from statistics import median
from typing import Callable, Sequence


def tflops(flops: int, time_us: float) -> float:
    """Convert FLOPs and microseconds to TFLOP/s."""
    return flops / time_us / 1e6


def summarize(samples_us: Sequence[float]) -> dict[str, float]:
    return {
        "median_us": median(samples_us),
        "min_us": min(samples_us),
        "max_us": max(samples_us),
    }


def print_report(
    *,
    mode: str,
    size: int,
    reports: dict[str, dict[str, float]],
) -> None:
    flops = 2 * size * size * size
    print(f"mode: {mode}")
    print(f"problem: 2 * {size}^3 = {flops} FLOP")
    print()
    print(f"{'boundary':<24} {'median_us':>12} {'min_us':>12} {'max_us':>12} {'TFLOP/s':>12}")
    for name, result in reports.items():
        print(
            f"{name:<24} "
            f"{result['median_us']:>12.4f} "
            f"{result['min_us']:>12.4f} "
            f"{result['max_us']:>12.4f} "
            f"{tflops(flops, result['median_us']):>12.4f}"
        )


def run_dry_run(size: int) -> int:
    """Verify boundary accounting without requiring a GPU."""
    reports = {
        "GEMM event only": summarize([100.0]),
        "GEMM+ReLU event": summarize([115.0]),
        "GEMM+ReLU host call": summarize([140.0]),
    }

    # Synthetic numbers are deliberately ordered to expose three nested scopes.
    assert reports["GEMM event only"]["median_us"] < reports["GEMM+ReLU event"]["median_us"]
    assert reports["GEMM+ReLU event"]["median_us"] < reports["GEMM+ReLU host call"]["median_us"]

    flops = 2 * size * size * size
    assert tflops(flops, 100.0) > tflops(flops, 115.0)
    assert tflops(flops, 115.0) > tflops(flops, 140.0)

    print_report(mode="dry-run", size=size, reports=reports)
    print()
    print("Interpretation:")
    print("  GEMM event only excludes ReLU GPU work.")
    print("  GEMM+ReLU event includes both kernels and any stream gap between the events.")
    print("  GEMM+ReLU host call also includes Python dispatch, launches, and the wait.")
    return 0


def run_cuda(warmup: int, samples: int, size: int) -> int:
    try:
        import torch
    except ImportError:
        print("PyTorch is not installed; run --mode dry-run for the accounting check.")
        return 1

    if not torch.cuda.is_available():
        print("CUDA is not available on this machine; run --mode dry-run instead.")
        return 1

    torch.set_float32_matmul_precision("highest")
    torch.manual_seed(0)

    device = "cuda"
    dtype = torch.bfloat16
    a = torch.randn((size, size), device=device, dtype=dtype)
    b = torch.randn((size, size), device=device, dtype=dtype)
    c = torch.empty((size, size), device=device, dtype=dtype)
    output = torch.empty_like(c)

    def gemm_only() -> None:
        torch.mm(a, b, out=c)

    def operation() -> None:
        gemm_only()
        torch.clamp_min(c, 0, out=output)

    # Correctness is checked before timing and does not define the benchmark.
    operation()
    torch.cuda.synchronize()
    expected = torch.mm(a.float(), b.float()).clamp_min(0).to(dtype)
    torch.testing.assert_close(output, expected, rtol=2e-2, atol=1e-2)

    def event_samples(fn: Callable[[], None]) -> list[float]:
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()

        values_us: list[float] = []
        for _ in range(samples):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            fn()
            end.record()
            end.synchronize()
            values_us.append(start.elapsed_time(end) * 1e3)
        return values_us

    def host_samples(fn: Callable[[], None]) -> list[float]:
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()

        values_us: list[float] = []
        for _ in range(samples):
            torch.cuda.synchronize()
            start = time.perf_counter()
            fn()
            torch.cuda.synchronize()
            values_us.append((time.perf_counter() - start) * 1e6)
        return values_us

    reports = {
        "GEMM event only": summarize(event_samples(gemm_only)),
        "GEMM+ReLU event": summarize(event_samples(operation)),
        "GEMM+ReLU host call": summarize(host_samples(operation)),
    }
    print_report(mode="cuda", size=size, reports=reports)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("auto", "cuda", "dry-run"), default="auto")
    parser.add_argument("--size", type=int, default=4096)
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--samples", type=int, default=20)
    args = parser.parse_args()

    if args.mode == "dry-run":
        return run_dry_run(args.size)
    if args.mode == "cuda":
        return run_cuda(args.warmup, args.samples, args.size)

    try:
        import torch
    except ImportError:
        return run_dry_run(args.size)
    if torch.cuda.is_available():
        return run_cuda(args.warmup, args.samples, args.size)
    return run_dry_run(args.size)


if __name__ == "__main__":
    raise SystemExit(main())
```

## 四、代码逐段对应硬件行为

### 1. allocation 在边界外

```python
a = torch.randn(...)
b = torch.randn(...)
c = torch.empty(...)
output = torch.empty_like(c)
```

矩阵分配不在计时区间内。这样测的是 kernel 和调用路径，不把一次性 allocator
行为混进来。若目标本身就是冷启动端到端延迟，应该把 allocation 明确放回边界，
并另报一组数字。

### 2. correctness check 在边界外

```python
operation()
torch.cuda.synchronize()
expected = torch.mm(a.float(), b.float()).clamp_min(0).to(dtype)
torch.testing.assert_close(output, expected, rtol=2e-2, atol=1e-2)
```

先证明结果正确，再计时。否则“优化”可能只是少算、漏 mask 或写错输出。

### 3. `gemm_only` 与 `operation`

```python
def gemm_only():
    torch.mm(a, b, out=c)

def operation():
    gemm_only()
    torch.clamp_min(c, 0, out=output)
```

`gemm_only` 的 event 区间只覆盖 GEMM stream work。
`operation` 的 event 区间同时覆盖 GEMM 与 ReLU。因为两者在同一个 stream，
ReLU 必须等 GEMM 写出的 `c` 准备完成。

### 4. CUDA event 采样

```python
start.record()
fn()
end.record()
end.synchronize()
elapsed_us = start.elapsed_time(end) * 1e3
```

`start.record()` 与 `end.record()` 之间存在什么 GPU work，就测什么。事件之间
如果出现 stream idle gap，也会被包含。

`end.synchronize()` 在 `end.record()` 之后执行。它的作用是让 CPU 等到 GPU
确实到达 end，从而可以安全读取 elapsed time；它不是额外的被测工作。

### 5. Host timer 采样

```python
torch.cuda.synchronize()
t0 = time.perf_counter()
fn()
torch.cuda.synchronize()
t1 = time.perf_counter()
```

第一次 synchronize 排除前一次 sample 的尾部。第二次 synchronize 等待本次
调用完成。因此这个边界包含 launch 与 host wait，通常大于 CUDA event 结果。

## 五、公式必须和边界说同一件事

对 `M x K` 乘 `K x N`，GEMM 的 FLOP 数为：

```text
2 * M * N * K
```

但分母取决于报告的是什么：

| 分子 | 合法分母 | 结果名称 |
|---|---|---|
| GEMM FLOP | 仅 GEMM 的 event 时间 | GEMM kernel TFLOP/s |
| GEMM FLOP | GEMM + ReLU 的 operation 时间 | GEMM 在完整 operation 中的 effective TFLOP/s |
| 整个 operation 的逻辑工作量 | synchronized host timer | operation 端到端吞吐 |

不能只改分母，却继续把结果叫成“GEMM kernel TFLOP/s”。

## 六、具体数字 trace

本机没有 CUDA，所以使用脚本内置的确定性 dry-run 数字来验证 boundary
accounting：

```text
GEMM event only      100 us
GEMM+ReLU event      115 us
GEMM+ReLU host call  140 us
```

问题规模为 `4096 x 4096 x 4096`：

```text
FLOP = 2 * 4096^3
     = 137438953472
```

GEMM-only event 吞吐：

```text
137438953472 / 100 / 10^6
= 1374.3895 TFLOP/s
```

同样的 GEMM FLOP 除以完整 event 时间：

```text
137438953472 / 115 / 10^6
= 1195.1213 TFLOP/s
```

除以 synchronized host time：

```text
137438953472 / 140 / 10^6
= 981.7068 TFLOP/s
```

这三个值的下降不是因为 GEMM 数学变了，而是因为计时边界逐步扩大。

## 七、运行方式与预期输出

本机没有 CUDA，先运行确定性检查：

```bash
python3 modern-gpu-programming-for-mlsys/code/benchmark_timing_boundary.py \
  --mode dry-run
```

预期输出：

```text
mode: dry-run
problem: 2 * 4096^3 = 137438953472 FLOP

boundary                   median_us       min_us       max_us      TFLOP/s
GEMM event only           100.0000     100.0000     100.0000    1374.3895
GEMM+ReLU event           115.0000     115.0000     115.0000    1195.1213
GEMM+ReLU host call       140.0000     140.0000     140.0000     981.7068

Interpretation:
  GEMM event only excludes ReLU GPU work.
  GEMM+ReLU event includes both kernels and any stream gap between the events.
  GEMM+ReLU host call also includes Python dispatch, launches, and the wait.
```

在带 CUDA 和 PyTorch 的机器上运行真实测量：

```bash
python3 modern-gpu-programming-for-mlsys/code/benchmark_timing_boundary.py \
  --mode cuda \
  --size 4096 \
  --warmup 500 \
  --samples 20
```

真实数字依赖 GPU、driver、PyTorch 版本和 clock。必须报告为：

```text
CUDA event GPU time: GEMM only
CUDA event GPU-stream time: GEMM + ReLU
single-call end-to-end time: GEMM + ReLU
```

不要只写一个没有边界的“耗时 105 us”。

## 八、常见错误与可观察症状

| 错误 | 可观察症状 | 修正 |
|---|---|---|
| CPU timer 不加 synchronize | 测得几十微秒甚至更小，像是只有 launch 开销 | 计时末尾 synchronize，并接受 host overhead |
| event 区间包含 idle gap | 结果大于 profiler 中两个 kernel 的 duration 之和 | 用 Nsight Systems 检查 gap，或缩小 event 区间 |
| GEMM-only 与 fused operation 直接比较 TFLOP/s | 错把做更多工作的实现判为 GEMM 更慢 | 先统一 operation boundary，或改名为 effective TFLOP/s |
| 第一个 call 直接计时 | 前几次样本高且漂移 | 先 warm-up，报告 median 和每个 round sample |
| 多 stream 只记录一个 stream 的 event | 报告时间小于实际 operation latency | 用 event dependency 汇合所有 branch |
| allocation 在实现 A 内、实现 B 外 | 两个实现的边界不同 | 明确 allocation 是否属于 operation，并统一处理 |

## 九、自测题与答案

### 1. CUDA event 测出的时间，为什么不等于 profiler 中 kernel duration 的总和？

因为 CUDA event 测的是 stream 上两个事件之间的完整区间。区间内可能有多个
kernel，也可能有 stream idle gap。Profiler 的 kernel duration 只描述每个 kernel
自身的执行区间。

### 2. 为什么 synchronized host timer 通常大于 CUDA event operation time？

Host timer 除了 GPU-stream interval，还包含 Python dispatch、CUDA launch API、
host scheduling 和等待 GPU 完成的时间。CUDA event 只记录设备 stream 上的区间。

### 3. 一个脚本先测完整 GEMM+ReLU 的 event 时间，再用 GEMM 的 `2MNK`
计算并写成 “GEMM kernel TFLOP/s”，问题是什么？

分子只包含 GEMM FLOP，分母却包含 ReLU。结果不是 GEMM kernel TFLOP/s，只能
称为 GEMM 在整个 operation 中的 effective throughput。

### 4. 如果只测 `torch.mm` 外面的 CUDA event，能不能包含另一个 stream 上异步
执行的 fused epilogue？

不能。除非 timing stream 显式等待另一个 stream 的完成事件，并且该 branch 在
start 后开始、在 end 前完成。

### 5. 对比两个实现时，除了 GPU 型号和 shape，还必须对齐哪些边界条件？

至少要记录输入和输出 dtype、layout、accumulation precision、mask / epilogue、
allocation 是否计入、state reset 是否计入、auxiliary kernels 和 communication
是否计入，以及 timer 类型和统计量。

## 十、下一知识点

```text
当前完成：Define the Timing Boundary
下一知识点：warm-up、repeat、rounds 与样本稳定性
```
