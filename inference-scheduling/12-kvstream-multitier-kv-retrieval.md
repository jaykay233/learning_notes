# KVStream 多级 KV 检索：聚合带宽、完成驱动 I/O 池与 TTFT 边界

- 论文：KVStream: Adaptive Parallel Storage Management for KV Caching in Large-Scale LLM Serving
- arXiv：[2610.31827](https://arxiv.org/abs/2610.31827)
- 公众号解读：[VStream：多级存储并行读KV，TTFT降6.1倍](https://mp.weixin.qq.com/s/QJNH4KswAszLrWv60Rgp8g?click_id=368204070)
- 论文单位：Argonne National Laboratory、Rochester Institute of Technology、Illinois Institute of Technology
- 论文版本：arXiv `2610.31827v1`，2026-10-22
- 示例：[kvstream_multitier_retrieval.py](examples/kvstream_multitier_retrieval.py)

```text
本次讲解位置
章节：inference-scheduling
小节：长上下文前缀 KV 的多级存储检索
知识点：step 级 I/O 池、聚合带宽下界与完成驱动路由
上次：P/D 资源分配、有效 Prefill 吞吐与离散事件仿真
下次：把 KVStream 路由接入真实带宽 trace，并比较压缩、重排与 P/D 分离的联合影响
PTX：不适用；本节讨论存储层级、异步 I/O 和请求级队列
```

## 为什么现在讲这个

前面的 IMPRESS 和 MEPIC 说明了一个 KV chunk 什么时候可以复用，以及复用是否精确。它们没有回答另一个问题：命中的 chunk 已经被换到 NVMe 或 PFS 上以后，怎么在 attention 开始前把它读回来。

当长上下文工作集超过 GPU 和 host DRAM，KV 检索会成为 prefill step 的关键路径。此时前缀树匹配得再准，如果取回 40 个 chunk 仍要按请求逐个等待，GPU 也只能空着。KVStream 处理的是这条存储路径，不改 Radix Tree、PagedAttention 或 attention 计算本身。

```text
前缀匹配 / Radix Tree：哪些 token block 可以复用
PagedAttention：这些 block 在 GPU / host 中放在哪里
KVStream：storage miss 的 block 如何从 NVMe / PFS 批量取回
Attention：所有需要的 KV 到齐后执行
```

## 1. NVMe 和 PFS 分别是什么

NVMe 是面向 SSD 的高速接口和协议。论文里的 NVMe 指计算节点本地的 SSD 阵列，延迟低，但单节点容量有限。

PFS 是并行文件系统（Parallel File System），由多台存储服务器组成，多个计算节点可以同时挂载。论文使用 VAST 和 Lustre，PFS 容量达到 4 PB 和 100 PB，带宽在测试中约为每节点 3.6–6.2 GB/s。

| 层级 | 位置 | 优势 | 主要限制 |
|---|---|---|---|
| GPU HBM | GPU 内 | 最低延迟、最高带宽 | 容量最小、成本最高 |
| Host DRAM | 计算节点 | 容量大于 HBM，可做缓冲区 | 仍会溢出 |
| 本地 NVMe | 计算节点 | 延迟低，带宽约 11.6–13.7 GB/s | 单节点只有数 TB |
| PFS | 集群共享 | 容量大，可把空闲带宽加入读路径 | 延迟高，会被其他节点争用 |

KVStream 的关键观察是：本地 NVMe 忙于读取时，PFS 在推理阶段可能几乎没有流量。把同一份 chunk 同时放到两层后，读取可以从两条独立路径并行进行。

## 2. 为什么逐请求读取会浪费带宽

LMCache 的普通路径按请求串行组织检索：

```text
request 0: storage -> host -> GPU
request 1: storage -> host -> GPU
...
request 2: storage -> host -> GPU
```

单个请求内部可以有异步 I/O，但 request 1 仍要等 request 0 的整条流水线完成。这样会让未完成 I/O 数量不足，NVMe 和 PFS 也很难长期保持高队列深度。

一个 forward step 必须等 batch 内所有请求的 KV 到齐，因此 request 0 提前读完并不会让模型更早开始。更合理的做法是把整个 step 的 storage reads 合成一个池：

```text
request 0 ─┐
request 1 ─┼─> [chunk0, chunk1, ..., chunkN] -> forward
request 2 ─┘
```

池中的任务可以乱序完成。每个 chunk 一到 host，就立即发起 H2D，无需等待其他请求。

## 3. 并行 I/O 队列到底长什么样

“一个并行 I/O 队列”是逻辑描述，不代表 Python `queue.Queue`，也不代表一条严格 FIFO 的链表。它是由实现维护的一组未完成读任务：

```text
ReadOp
├── chunk_id
├── tier: NVMe / PFS
├── storage path and offset
├── byte count
├── host buffer
├── destination GPU block
└── state: pending / in-flight / ready
```

每个 tensor-parallel rank 在每次 iteration 开始时调用一次 batch-level API，把所有请求命中的 storage-resident chunks 交给这个读任务池。

底层的 C++ 引擎使用 Linux `io_uring`：

```text
用户线程提交 SQE
        ↓
io_uring Submission Queue
        ↓
NVMe / PFS 异步执行 O_DIRECT 读取
        ↓
io_uring Completion Queue
        ↓
统一完成循环收割 CQE
```

`io_uring` 允许一次提交多个异步请求，完成事件可以乱序返回。论文为读和写使用独立 ring，避免写操作阻塞读操作。`O_DIRECT` 绕过 page cache，数据仍然进入指定的 host buffer，随后通过 H2D 拷到 GPU；这不是 storage-to-GPU 的零拷贝。

调度上，NVMe 从 chunk 列表头部读，PFS 从尾部读，每次领取 `m` 个 chunk。某一层完成当前批次后，立刻领取下一批：

```text
chunk list: [c0, c1, c2, ..., c61, c62, c63]
              ↑                             ↑
             NVMe                          PFS

NVMe 完成 m 个 -> 再领 m 个 ->
PFS 完成 m 个  -> 再领 m 个  -> 两层在中间相遇
```

快的层会自然走得更远，慢的层领取更少。两层接近相遇时允许少量重复读取，第一个到达的副本让 chunk 变为 GPU-ready，后到的重复副本可以无害完成。论文正文没有给出 `m` 的具体取值，也没有公开到数据结构级别的实现；上面的任务是根据算法和 `io_uring` 路径还原出的模型。

## 4. 聚合带宽下界

设第 `i` 个 step 需要从存储层读取 `D_i` 字节，NVMe 与 PFS 在当时的有效读带宽分别是 `b0(i)` 和 `b1(i)`，host-to-device（H2D）带宽为 `b_H2D`。如果两条存储路径独立，那么检索时间不可能低于：

\[
T_i^{ret} \ge
\frac{D_i}{\min\left(b_0(i)+b_1(i),\; b_{H2D}\right)}
\]

测试床中 `b_H2D` 明显高于 `b0+b1`，因此实际观察到的上界主要是：

\[
T_i^{bound}=\frac{D_i}{b_0(i)+b_1(i)}
\]

固定比例的静态切分把 `g` 比例的字节永远交给 NVMe，剩余 `1-g` 交给 PFS：

\[
T_i^{static}
=
\max\left(
\frac{gD_i}{b_0(i)},
\frac{(1-g)D_i}{b_1(i)}
\right)
\]

如果当时两层的速度份额是：

\[
\phi(i)=\frac{b_0(i)}{b_0(i)+b_1(i)}
\]

只有 `g=φ(i)` 时，静态切分才能同时让两层完成。带宽飘移后，固定 `g` 会让其中一层成为 straggler。

完成驱动路由在理想情况下让两层一直工作到任务耗尽，所以动态份额会自动接近 `φ(i)`。这就是论文说它“不需要预测带宽”的原因：它不预测，只根据“谁先完成”继续派活。

## 5. 数值轨迹

示例模拟器使用 1 GiB storage reads、16 MiB chunk、每批领取 2 个 chunk，并让每个 step 的 NVMe/PFS 带宽发生变化。静态策略固定 70% 的 chunk 放 NVMe，完成驱动策略从两端动态领取。

| step | NVMe GB/s | PFS GB/s | 聚合下界 | 固定 70:30 | 完成驱动 | 驱动策略的 NVMe 份额 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 12 | 6 | 0.0597 s | 0.0629 s | 0.0615 s | 0.656 |
| 2 | 8 | 2 | 0.1074 s | 0.1594 s | 0.1174 s | 0.781 |
| 3 | 5 | 5 | 0.1074 s | 0.1510 s | 0.1074 s | 0.500 |
| 4 | 10 | 1 | 0.0976 s | 0.3188 s | 0.1007 s | 0.906 |
| 5 | 4 | 7 | 0.0976 s | 0.1887 s | 0.1007 s | 0.375 |

总计：

```text
static total:  0.8808 s
routed total:  0.4877 s
bound total:   0.4696 s
static/routed: 1.806x
routed/bound:  1.038x
```

第 4 个 step 最能说明问题。此时 NVMe 比 PFS 快 10 倍，理想 NVMe 份额约为 0.909。固定 70:30 仍让 PFS 承担 30%，PFS 成为尾部瓶颈，时间从 `0.3188 s` 降到动态路由的 `0.1007 s`。

动态策略与下界仍差 3.8%，原因是两个固定因素：chunk 只能整块分配，最后一层会剩下不足一批的尾巴；完成事件要等当前批次结束才触发下一次领取。批量越小，负载均衡越细，但提交和调度开销越高。

## 6. 完整模拟器

文件：`inference-scheduling/examples/kvstream_multitier_retrieval.py`

```python
#!/usr/bin/env python3
"""Compare a static tier split with KVStream-style completion routing."""

from __future__ import annotations

import argparse
import heapq
from dataclasses import dataclass
from typing import Sequence

MIB = 1024 * 1024
GBPS = 1_000_000_000


@dataclass(frozen=True)
class StepResult:
    nvme_gbps: float
    pfs_gbps: float
    lower_bound_s: float
    static_s: float
    dynamic_s: float
    dynamic_nvme_share: float


def static_split_time(
    total_chunks: int,
    chunk_bytes: int,
    nvme_bps: float,
    pfs_bps: float,
    nvme_share: float,
) -> float:
    nvme_chunks = round(total_chunks * nvme_share)
    pfs_chunks = total_chunks - nvme_chunks
    nvme_time = nvme_chunks * chunk_bytes / nvme_bps
    pfs_time = pfs_chunks * chunk_bytes / pfs_bps
    return max(nvme_time, pfs_time)


def completion_routed_time(
    total_chunks: int,
    chunk_bytes: int,
    nvme_bps: float,
    pfs_bps: float,
    batch_chunks: int,
) -> tuple[float, float]:
    """Route from both ends; a tier claims more after each batch completes."""
    remaining = total_chunks
    next_batch = 0
    served = {0: 0, 1: 0}  # 0: NVMe, 1: PFS
    busy = {0: False, 1: False}
    finish_time = 0.0
    events: list[tuple[float, int, int, int]] = []

    def claim(tier: int, now: float) -> None:
        nonlocal remaining, next_batch
        if remaining == 0:
            return
        count = min(batch_chunks, remaining)
        remaining -= count
        served[tier] += count
        busy[tier] = True
        bandwidth = nvme_bps if tier == 0 else pfs_bps
        duration = count * chunk_bytes / bandwidth
        heapq.heappush(events, (now + duration, tier, next_batch, count))
        next_batch += 1

    claim(0, 0.0)
    claim(1, 0.0)

    while events:
        now, tier, _, _ = heapq.heappop(events)
        finish_time = max(finish_time, now)
        busy[tier] = False
        claim(tier, now)

        if remaining == 0 and not any(busy.values()):
            break

    nvme_share = served[0] / total_chunks
    return finish_time, nvme_share


def run(
    steps: Sequence[tuple[float, float]],
    total_bytes: int,
    chunk_bytes: int,
    nvme_share: float,
    batch_chunks: int,
) -> list[StepResult]:
    if total_bytes % chunk_bytes:
        raise ValueError("total_bytes must be divisible by chunk_bytes")
    total_chunks = total_bytes // chunk_bytes
    results: list[StepResult] = []

    for nvme_gbps, pfs_gbps in steps:
        nvme_bps = nvme_gbps * GBPS
        pfs_bps = pfs_gbps * GBPS
        bound = total_bytes / (nvme_bps + pfs_bps)
        static = static_split_time(
            total_chunks, chunk_bytes, nvme_bps, pfs_bps, nvme_share
        )
        dynamic, dynamic_share = completion_routed_time(
            total_chunks, chunk_bytes, nvme_bps, pfs_bps, batch_chunks
        )
        results.append(
            StepResult(
                nvme_gbps=nvme_gbps,
                pfs_gbps=pfs_gbps,
                lower_bound_s=bound,
                static_s=static,
                dynamic_s=dynamic,
                dynamic_nvme_share=dynamic_share,
            )
        )
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-mib", type=int, default=1024)
    parser.add_argument("--chunk-mib", type=int, default=16)
    parser.add_argument("--static-nvme-share", type=float, default=0.70)
    parser.add_argument("--batch-chunks", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    steps = (
        (12.0, 6.0),
        (8.0, 2.0),
        (5.0, 5.0),
        (10.0, 1.0),
        (4.0, 7.0),
    )
    results = run(
        steps=steps,
        total_bytes=args.total_mib * MIB,
        chunk_bytes=args.chunk_mib * MIB,
        nvme_share=args.static_nvme_share,
        batch_chunks=args.batch_chunks,
    )

    print(
        "step | NVMe GB/s | PFS GB/s | bound(s) | static(s) | "
        "routed(s) | routed NVMe share"
    )
    print(
        "-----+-----------+----------+----------+-----------+"
        "-----------+------------------"
    )
    for index, result in enumerate(results, start=1):
        print(
            f"{index:4d} | {result.nvme_gbps:9.1f} | {result.pfs_gbps:8.1f} | "
            f"{result.lower_bound_s:8.4f} | {result.static_s:9.4f} | "
            f"{result.dynamic_s:10.4f} | {result.dynamic_nvme_share:17.3f}"
        )

    static_total = sum(item.static_s for item in results)
    dynamic_total = sum(item.dynamic_s for item in results)
    bound_total = sum(item.lower_bound_s for item in results)
    print()
    print(f"static total:  {static_total:.4f} s")
    print(f"routed total:  {dynamic_total:.4f} s")
    print(f"bound total:   {bound_total:.4f} s")
    print(f"static/routed: {static_total / dynamic_total:.3f}x")
    print(f"routed/bound:  {dynamic_total / bound_total:.3f}x")


if __name__ == "__main__":
    main()
```

运行命令：

```bash
python3 inference-scheduling/examples/kvstream_multitier_retrieval.py
```

本次输出：

```text
step | NVMe GB/s | PFS GB/s | bound(s) | static(s) | routed(s) | routed NVMe share
-----+-----------+----------+----------+-----------+-----------+------------------
   1 |      12.0 |      6.0 |   0.0597 |    0.0629 |     0.0615 |             0.656
   2 |       8.0 |      2.0 |   0.1074 |    0.1594 |     0.1174 |             0.781
   3 |       5.0 |      5.0 |   0.1074 |    0.1510 |     0.1074 |             0.500
   4 |      10.0 |      1.0 |   0.0976 |    0.3188 |     0.1007 |             0.906
   5 |       4.0 |      7.0 |   0.0976 |    0.1887 |     0.1007 |             0.375

static total:  0.8808 s
routed total:  0.4877 s
bound total:   0.4696 s
static/routed: 1.806x
routed/bound:  1.038x
```

这个模拟器只验证调度结构，不复现论文的硬件数据。它省略了 host buffer 上限、H2D 竞争、文件元数据、PFS 队列尾延迟和读写干扰。论文在自己的两个测试床上测得完成驱动路由距离实际聚合带宽上界 `4.3%` 以内，与本例的量级一致，但具体百分比不能直接等同。

## 7. 延迟写与写放大

KVStream 还做了延迟写。新生成的 KV 先进入 host buffer，等 forward 执行时再同时刷到 NVMe 和 PFS。此时读路径通常空闲，写入可以躲开下一次检索。

代价是每个 chunk 都要写两份，写放大约为 2 倍。写入能否完全藏在 forward 后面，取决于：

\[
\frac{W_i}{\min_k w_k}
\lesssim
T_{i+1}^{fwd}
\]

`W_i` 是第 `i` 个 step 新生成的 KV 字节数，`w_k` 是各层写带宽。如果新 KV 很多、输入很短、输出很长，或者 host staging buffer 已满，写就会回到关键路径。

这条条件也解释了适用面：

- 多轮对话和 agentic workflow 常常复用很长的旧前缀，只追加少量新 KV，复制写更容易被 forward 摊掉；
- 每次请求都产生大量唯一 KV、复用率很低时，读收益不足以覆盖复制写。

## 8. 6.1 倍 TTFT 应该怎么读

论文的 6.07 倍结果来自 Testbed-2 上的 32B 模型、SWE-bench agentic trace、到达变异系数 `CV=4`，并且比较对象是 LMCache。它指平均 TTFT 缩短为约原来的 `1/6.07`，也就是快 6.07 倍。

这不是“单个 KV chunk 的读取快了 6.1 倍”，也不是所有负载的普遍收益。Qwen-Coder trace 中，每请求检索时间只下降 15%–22%；高负载下，step 变快让等待队列更快排空，TTFT 的队列部分把收益放大到 3.12 倍。

| 场景 | 相对 LMCache |
|---|---:|
| 14B–110B 离线吞吐 | 1.17–1.49 倍 |
| Qwen-Coder 低/中/高负载 | 1.07 / 1.61 / 3.12 倍平均 TTFT |
| Chutes 低/中/高负载 | 1.23 / 1.71 / 2.01 倍 |
| SWE-bench，CV=4 | 6.07 倍平均 TTFT |
| SWE-bench，CV=8 | 1.17 倍平均 TTFT |
| 1 s TTFT SLO 达成率 | KVStream 84%，LMCache 70% |
| host memory 足以容纳工作集 | 约 0.98 倍，基本无收益 |
| 前缀复用率降到 4% | 0.81 倍，慢于 LMCache |

## 9. 常见误区

| 误区 | 现象 | 原因 |
|---|---|---|
| 把 KVStream 当成新的 attention kernel | 以为它会改变 QK 或 softmax 的计算 | 它只改变 KV 从慢层到 GPU 的搬运路径 |
| 把 PFS 当成另一块本地盘 | 认为加一层一定得到两倍带宽 | PFS 是共享资源，可能被其他节点争用 |
| 认为“并行 I/O 队列”是一条 FIFO | 以为所有请求仍要按原顺序逐个完成 | 任务按 chunk 粒度乱序完成，并由完成事件补充 |
| 用固定比例切分代替动态调度 | 静态配置在平均带宽上看似正确，带宽漂移后尾延迟很高 | 读时的实际份额 `φ(i)` 会改变 |
| 只看平均带宽，不看队列深度 | 设备理论带宽很高，实测仍跑不满 | 并行度不足时，NVMe 和 PFS 都不能饱和 |
| 忽略复制写成本 | 高复用场景收益明显，低复用场景反而变慢 | 每份 KV 都要写两个存储层 |
| 把 6.1 倍当作普适加速 | 低负载和 host cache 足够时只有约 1 倍 | 端到端收益还取决于队列、复用距离和 forward 占比 |

## 10. 自检

**Q1：为什么把多个请求的 KV 读取放进一个 step 级任务池能提升带宽？**

因为 GPU 本来就要等整个 batch 的 KV 到齐。按请求串行会让大量存储带宽在等待中空闲；step 级任务池可以同时维持更多未完成读，并把 H2D 与剩余存储读取重叠。

**Q2：完成驱动路由为什么不需要提前知道 NVMe 和 PFS 的带宽？**

某一层完成当前批次后，就去领取下一批任务。更快的层自然领取更多，最终份额由完成速度决定。控制器只需要处理完成事件，不需要预测 `b0(i)` 和 `b1(i)`。

**Q3：为什么 `routed/bound` 不是 1.000？**

chunk 只能整块分给某一层，最后一层还会剩下不足一批的尾巴；每次补充任务也要等当前批次完成。批量 `m` 决定了负载均衡粒度和调度开销之间的折中。

**Q4：为什么 KVStream 在低复用负载上可能输给 LMCache？**

它要为每个新 chunk 写两份存储副本。读请求很少时，复制写成本大于并行读取节省的时间。论文在 4% 复用率下测到 0.81 倍。

**Q5：6.07 倍 TTFT 结果是否说明每个请求都变快 6 倍？**

不是。它是在特定突发到达条件下比较平均 TTFT。单个 step 的检索时间改善只有约 15%–22%，高负载时等待队列被更快排空，才把平均 TTFT 放大到 6 倍左右。

## 结论

KVStream 的工程价值集中在三个条件同时成立时：长上下文工作集放不进 GPU 和 host memory；前缀复用足够高；计算节点旁有一条可用的第二存储带宽来源。满足这些条件后，它把“每个请求各自等待一次慢速存储读取”改成“整个 step 共同使用 NVMe 和 PFS 的聚合带宽”。

它与 RadixAttention、PagedAttention、P/D 分离和 KV 压缩都可以组合。Radix/Paged 决定哪些 block 可复用、放在哪里；P/D 改计算资源的位置；KVStream 改 storage miss 的取回路径。真正需要警惕的是写放大、PFS 争用和低复用负载，而不是把 6.1 倍当成所有场景的固定收益。
