# 二十二、Roofline 与课程收束

### 本次讲解位置

```text
章节：chapter_performance
小节：The Roofline Model / Arithmetic Intensity of Common Workloads
知识点：算术强度、B200 ridge point 与瓶颈分类
上次：chapter_flash_attention 8.4 Reference 与容差验证
下次：教材正文 Parts I-IV 收束；appendix 作为查阅与工具手册
PTX：无直接对应指令；本知识点判断应该优先优化哪类硬件资源
```

### 为什么现在讲这个

前面已经学习了 layout、TMA、mbarrier、Tensor Core、TMEM、pipeline、
warp specialization、Two-CTA cluster 和 Flash Attention。现在遇到的实际
设计问题是：一个 kernel 变慢时，到底应该减少 HBM 流量，还是继续改
pipeline 让 Tensor Core 更少等待？

只看绝对吞吐不能回答这个问题。例如 `330 TFLOP/s` 对 elementwise
kernel 可能已经很高，但对 compute-bound fp16 GEMM 可能只达到峰值的一小
部分。Roofline 提供一个硬件上限。没有这个上限，就容易在 memory-bound
kernel 上反复调整计算调度，或者在 compute-bound kernel 上只做少量访存
优化。

这个知识点防止一种常见误判：看到 kernel 没跑满 Tensor Core，就默认
瓶颈一定在计算路径。真实瓶颈可能只是 HBM 带宽已经到顶。

## 一、心智模型：两个屋顶决定性能上限

kernel 同时受两条路径限制：

```text
compute path:
  CUDA core、Tensor Core、SFU 等计算单元能提供多少 FLOP/s

memory path:
  HBM 每秒能搬运多少 byte
```

计算量相对数据量越大，越可能碰到 compute roof。数据搬运相对计算量
越大，越可能碰到 memory roof。

算术强度定义为：

```text
arithmetic intensity
= compute work / data moved
= FLOP / byte
```

这里的 byte 必须对应明确的内存层级：

```text
HBM byte    -> HBM roofline
L2 byte     -> L2 roofline
SMEM byte   -> SMEM roofline
```

本文默认讨论 HBM roofline。不同层级算出的 AI 不能直接混用。

性能上界是：

```text
attainable performance
<= min(peak compute throughput,
        HBM bandwidth * arithmetic intensity)
```

令两条屋顶线相等，得到 ridge point：

```text
ridge point
= peak compute throughput / HBM bandwidth
```

B200 使用教材中的近似值：

```text
peak compute = 2000 TFLOP/s
HBM bandwidth = 8 TB/s

ridge point
= 2000 / 8
= 250 FLOP/byte
```

判断规则是：

```text
AI < 250   -> 更可能 memory-bound
AI > 250   -> 更可能 compute-bound
AI ~= 250  -> 两侧屋顶接近
```

这只是分类起点，不是最终性能结论。它决定优化方向，实际收益仍要测量。

## 二、把 FLOP 和 byte 写清楚

### 1. 方阵 GEMM 的理想 HBM AI

对于 `M=N=K` 的 fp16 GEMM：

```text
compute work
= 2 * M * N * K
= 2N^3
```

理想 HBM 流量假设 A 和 B 各读一次，C 写一次：

```text
A: N^2 elements * 2 bytes
B: N^2 elements * 2 bytes
C: N^2 elements * 2 bytes
= 6N^2 bytes
```

因此：

```text
AI
= 2N^3 / (6N^2)
= N / 3
```

当 `N=4096`：

```text
AI = 4096 / 3
   = 1365.333333 FLOP/byte
```

这个值高于 ridge point `250`，所以理想 HBM 模型下是 compute-bound。

### 2. 单个 GEMM CTA stage 的 AI

设：

```text
Bm = 128
Bn = 128
Bk = 64
dtype = fp16
```

一个 K-stage 的计算量和 A/B 流量是：

```text
FLOPs
= 2 * Bm * Bn * Bk
= 2 * 128 * 128 * 64
= 2097152

bytes
= 2 * (Bm*Bk + Bk*Bn)
= 2 * (128*64 + 64*128)
= 32768

AI
= 2097152 / 32768
= 64 FLOP/byte
```

`64 < 250`，所以仅看这一层的 HBM 流量，stage 是 memory-bound。这里
不是和“整个 GEMM compute-bound”矛盾，而是说明：

```text
大 GEMM 可以通过全局复用进入 compute-bound
一个孤立 CTA stage 未必能单靠自身复用跨过 ridge point
L2 reuse、cluster tile 和 CTA 间复用会减少有效 HBM byte
```

这也解释了为什么 persistent scheduling、L2 locality group 和 Two-CTA
cluster 不只是“调度技巧”，它们会改变有效内存层级上的 AI。

### 3. materialized attention

对于 sequence length `S`、head dimension `D`，QK^T 和 PV 两部分合计：

```text
FLOPs = 4 * S^2 * D
```

这里用简化模型，假设标准 attention 会写回并重新读取 `S x S` 的 score
matrix，而且中间读写都按 fp16 的 2 bytes 计算：

```text
score traffic
= 8 * S^2 bytes

Q/K/V/O traffic
= 8 * S * D bytes

total bytes
= 8S^2 + 8SD
```

因此：

```text
AI
= 4S^2D / (8S^2 + 8SD)
= SD / (2(S + D))
```

当 `S=4096`、`D=128`：

```text
AI
= 4096 * 128 / (2 * (4096 + 128))
= 62.060606 FLOP/byte
```

`62.060606 < 250`，所以 materialized attention 更容易被 HBM 流量限制。

### 4. Flash Attention prefill

Flash Attention 不把完整 score matrix 写回 HBM。用同一个简化模型，
只计算 Q、K、V、O 各访问一次：

```text
bytes = 8SD

AI
= 4S^2D / 8SD
= S / 2
```

当 `S=4096`：

```text
AI = 2048 FLOP/byte
```

`2048 > 250`，所以长序列 prefill 更接近 compute-bound。

这个结论依赖序列长度和实际 tile reuse。短序列时，AI 可能不足以跨过
ridge point。

### 5. Decode 一个 token

decode 时通常只有一个或很少几个 query token，但 K/V cache 可能已经有
`S=4096` 个 token。

忽略 Q/O 和很小的计算，主要工作与流量为：

```text
FLOPs = 4 * S * D

KV bytes
= K bytes + V bytes
= 2*S*D + 2*S*D
= 4*S*D

AI
= 4SD / 4SD
= 1 FLOP/byte
```

`AI=1` 远低于 `250`。因此 decode attention 典型上是 memory-bound，
优化重点是减少 KV bytes、提高有效带宽或改变算法，而不是单纯把
Tensor Core 使用率继续拉高。

## 三、五个 workload 的结果

| workload | AI (FLOP/byte) | 由 HBM 决定的 roof | 最终 roof | 分类 |
|---|---:|---:|---:|---|
| GEMM `N=4096` ideal | 1365.333333 | 10922.666667 TFLOP/s | 2000 TFLOP/s | compute |
| GEMM CTA stage `128x128x64` | 64.000000 | 512 TFLOP/s | 512 TFLOP/s | memory |
| Attention materialized `S=4096,D=128` | 62.060606 | 496.484848 TFLOP/s | 496.484848 TFLOP/s | memory |
| Flash prefill `S=4096,D=128` | 2048.000000 | 16384 TFLOP/s | 2000 TFLOP/s | compute |
| Decode one token `S=4096,D=128` | 1.000000 | 8 TFLOP/s | 8 TFLOP/s | memory |

这张表说明同一个 attention 算法可以表现出完全不同的瓶颈：

```text
long-sequence prefill
  -> 大矩阵乘为主
  -> compute-bound

single-token decode
  -> 主要读取 KV cache
  -> memory-bound
```

因此推理系统不能只用一个总体“attention 优化方案”覆盖两种阶段。

## 四、完整可运行脚本

文件名：

```text
modern-gpu-programming-for-mlsys/code/roofline_capstone.py
```

```python
from dataclasses import dataclass


PEAK_COMPUTE_TFLOPS = 2000.0
HBM_BANDWIDTH_TBPS = 8.0
TB = 1e12
TFLOP = 1e12
FP16_BYTES = 2


@dataclass(frozen=True)
class RooflineResult:
    name: str
    flops: float
    bytes_moved: float
    arithmetic_intensity: float
    memory_roof_tflops: float
    attainable_tflops: float
    bottleneck: str


def analyze(
    name: str,
    flops: float,
    bytes_moved: float,
    peak_compute_tflops: float = PEAK_COMPUTE_TFLOPS,
    hbm_bandwidth_tbps: float = HBM_BANDWIDTH_TBPS,
) -> RooflineResult:
    if flops <= 0:
        raise ValueError("flops must be positive")
    if bytes_moved <= 0:
        raise ValueError("bytes_moved must be positive")

    arithmetic_intensity = flops / bytes_moved
    memory_roof_tflops = hbm_bandwidth_tbps * TB * arithmetic_intensity / TFLOP
    attainable_tflops = min(peak_compute_tflops, memory_roof_tflops)
    bottleneck = "compute" if peak_compute_tflops <= memory_roof_tflops else "memory"

    return RooflineResult(
        name=name,
        flops=flops,
        bytes_moved=bytes_moved,
        arithmetic_intensity=arithmetic_intensity,
        memory_roof_tflops=memory_roof_tflops,
        attainable_tflops=attainable_tflops,
        bottleneck=bottleneck,
    )


def print_result(result: RooflineResult) -> None:
    print(result.name)
    print(f"  flops              = {result.flops:.6e}")
    print(f"  bytes              = {result.bytes_moved:.6e}")
    print(f"  arithmetic_intensity = {result.arithmetic_intensity:.6f} FLOP/byte")
    print(f"  memory_roof        = {result.memory_roof_tflops:.6f} TFLOP/s")
    print(f"  attainable         = {result.attainable_tflops:.6f} TFLOP/s")
    print(f"  bottleneck         = {result.bottleneck}")
    print()


def main() -> None:
    ridge_point = PEAK_COMPUTE_TFLOPS / HBM_BANDWIDTH_TBPS
    print("B200 rounded roofline inputs")
    print(f"  peak_compute = {PEAK_COMPUTE_TFLOPS:.1f} TFLOP/s")
    print(f"  hbm_bandwidth = {HBM_BANDWIDTH_TBPS:.1f} TB/s")
    print(f"  ridge_point = {ridge_point:.1f} FLOP/byte")
    print()

    gemm_n = 4096
    gemm_flops = 2 * gemm_n**3
    gemm_bytes = 3 * FP16_BYTES * gemm_n**2

    blk_m = 128
    blk_n = 128
    blk_k = 64
    stage_flops = 2 * blk_m * blk_n * blk_k
    stage_bytes = FP16_BYTES * (blk_m * blk_k + blk_k * blk_n)

    seq_len = 4096
    head_dim = 128
    attention_flops = 4 * seq_len**2 * head_dim
    materialized_bytes = 8 * seq_len**2 + 8 * seq_len * head_dim
    flash_prefill_bytes = 8 * seq_len * head_dim

    decode_query_count = 1
    decode_flops = 4 * decode_query_count * seq_len * head_dim
    decode_bytes = 4 * seq_len * head_dim

    results = [
        analyze(
            f"GEMM ideal square: N={gemm_n}, fp16",
            gemm_flops,
            gemm_bytes,
        ),
        analyze(
            f"GEMM CTA stage: Bm={blk_m}, Bn={blk_n}, Bk={blk_k}, fp16",
            stage_flops,
            stage_bytes,
        ),
        analyze(
            f"Attention materialized: S={seq_len}, D={head_dim}, fp16",
            attention_flops,
            materialized_bytes,
        ),
        analyze(
            f"Flash Attention prefill: S={seq_len}, D={head_dim}, fp16",
            attention_flops,
            flash_prefill_bytes,
        ),
        analyze(
            f"Decode one token: S={seq_len}, D={head_dim}, fp16 KV cache",
            decode_flops,
            decode_bytes,
        ),
    ]

    for result in results:
        print_result(result)

    assert results[0].bottleneck == "compute"
    assert results[1].bottleneck == "memory"
    assert results[2].bottleneck == "memory"
    assert results[3].bottleneck == "compute"
    assert results[4].bottleneck == "memory"
    print("classification assertions: PASS")


if __name__ == "__main__":
    main()
```

运行命令：

```bash
'/Users/saboxu/Documents/ChatGPT/mlc学习/mlc/bin/python' \
  modern-gpu-programming-for-mlsys/code/roofline_capstone.py
```

该脚本不需要 GPU，使用纯 Python 标准库。它验证的是性能模型计算和
分类断言，不测量真实硬件性能。

## 五、实测输出

```text
B200 rounded roofline inputs
  peak_compute = 2000.0 TFLOP/s
  hbm_bandwidth = 8.0 TB/s
  ridge_point = 250.0 FLOP/byte

GEMM ideal square: N=4096, fp16
  flops              = 1.374390e+11
  bytes              = 1.006633e+08
  arithmetic_intensity = 1365.333333 FLOP/byte
  memory_roof        = 10922.666667 TFLOP/s
  attainable         = 2000.000000 TFLOP/s
  bottleneck         = compute

GEMM CTA stage: Bm=128, Bn=128, Bk=64, fp16
  flops              = 2.097152e+06
  bytes              = 3.276800e+04
  arithmetic_intensity = 64.000000 FLOP/byte
  memory_roof        = 512.000000 TFLOP/s
  attainable         = 512.000000 TFLOP/s
  bottleneck         = memory

Attention materialized: S=4096, D=128, fp16
  flops              = 8.589935e+09
  bytes              = 1.384120e+08
  arithmetic_intensity = 62.060606 FLOP/byte
  memory_roof        = 496.484848 TFLOP/s
  attainable         = 496.484848 TFLOP/s
  bottleneck         = memory

Flash Attention prefill: S=4096, D=128, fp16
  flops              = 8.589935e+09
  bytes              = 4.194304e+06
  arithmetic_intensity = 2048.000000 FLOP/byte
  memory_roof        = 16384.000000 TFLOP/s
  attainable         = 2000.000000 TFLOP/s
  bottleneck         = compute

Decode one token: S=4096, D=128, fp16 KV cache
  flops              = 2.097152e+06
  bytes              = 2.097152e+06
  arithmetic_intensity = 1.000000 FLOP/byte
  memory_roof        = 8.000000 TFLOP/s
  attainable         = 8.000000 TFLOP/s
  bottleneck         = memory

classification assertions: PASS
```

## 六、代码逐段解释

### `analyze()`

```python
arithmetic_intensity = flops / bytes_moved
memory_roof_tflops = hbm_bandwidth_tbps * TB * arithmetic_intensity / TFLOP
attainable_tflops = min(peak_compute_tflops, memory_roof_tflops)
```

第一行把计算量与 byte 化成 AI。

第二行先计算 `TB/s * FLOP/byte = TFLOP/s`：

```text
8 TB/s * 64 FLOP/byte
= 512 TFLOP/s
```

第三行选择两个屋顶中更低的一个。若 compute roof 是 2000，而 memory
roof 是 512，最终可达到的性能不可能高于 512。

### 方阵 GEMM

```python
gemm_flops = 2 * gemm_n**3
gemm_bytes = 3 * FP16_BYTES * gemm_n**2
```

`2N^3` 对应 `N x N` 矩阵乘的乘加计算量。

`3 * 2 * N^2` 表示：

```text
读取 A：2N^2 bytes
读取 B：2N^2 bytes
写入 C：2N^2 bytes
```

### CTA-stage GEMM

```python
stage_flops = 2 * blk_m * blk_n * blk_k
stage_bytes = FP16_BYTES * (blk_m * blk_k + blk_k * blk_n)
```

这里只计算一个 stage 读取的 A/B tile，不考虑 C 和 L2 reuse。它的
目的是展示“全局 GEMM 是 compute-bound”与“单 CTA stage 的局部流量”
不是一个层级的问题。

### Attention

```python
attention_flops = 4 * seq_len**2 * head_dim
```

QK^T 使用 `2S^2D` FLOP，PV 再使用 `2S^2D` FLOP，因此总共 `4S^2D`。

```python
materialized_bytes = 8 * seq_len**2 + 8 * seq_len * head_dim
flash_prefill_bytes = 8 * seq_len * head_dim
```

第一项包含 `S x S` 中间矩阵的读写，第二项不物化 score matrix。
两者比较可以直接看出 Flash Attention 为什么提高 AI。

### Decode

```python
decode_flops = 4 * decode_query_count * seq_len * head_dim
decode_bytes = 4 * seq_len * head_dim
```

`K` 和 `V` 每个元素占 2 bytes，所以每个 KV token 需要读取
`4D` bytes。计算量近似为 `4D` FLOP，因此 AI 近似为 1。

模型忽略 Q/O 和 cache metadata，是为了突出 decode 的主要矛盾：
每个 token 的计算量太小，却必须把大量 KV cache 从 HBM 搬到片上。

## 七、如何用这张图和前面的章节连接

| 结论 | 优化方向 | 已经学过的机制 |
|---|---|---|
| memory-bound | 减少 HBM bytes、融合、量化、压缩 KV | Flash Attention 避免 materialize、GQA 共享 K/V、低精度 dtype |
| compute-bound | 保持 Tensor Core 忙碌、减少等待 | software pipeline、warp specialization、Two-CTA cluster、TMA + mbarrier |
| 接近 ridge point | 分别测 compute 与 memory 利用率 | Nsight Compute、cache hit、stall reason、实际事务 byte |
| decode KV 流量主导 | 减少每 token 读取的 byte | KV4/GQA/MQA、分页与 cache layout、批量复用权重 |

几条重要推论：

1. 对一个已经真正达到 HBM roof 的 memory-bound kernel，继续优化
   Tensor Core 调度不会提升主循环性能。
2. 对一个 compute-bound kernel，TMA 的主要价值不是减少数学 FLOP，
   而是让数据搬运与 Tensor Core 计算重叠，减少 Tensor Core 等待。
3. 对 GEMM，单看 HBM AI 只能判断全局理论瓶颈。真正 kernel 还会受
   L2 reuse、SMEM bandwidth、TMEM capacity、register pressure 和
   occupancy 影响。
4. occupancy 不是越高越好。warp specialization 或 Two-CTA cluster
   可能降低 occupancy，却提高关键流水线的持续利用率。
5. prefill 与 decode 必须分开分析。长序列 prefill 可能 compute-bound，
   单 token decode 通常 memory-bound。

## 八、常见错误与可观察症状

| 错误 | 后果 | 正确做法 |
|---|---|---|
| 只看 TFLOP/s，不看硬件 roof | 把已经接近 HBM 上限的 kernel 判成“计算效率低” | 先比较 AI 与 ridge point，再测对应 resource |
| 混用 HBM、L2、SMEM 的 byte | AI 和结论互相矛盾 | 每次计算明确 memory level |
| 把理想 GEMM AI 当成实测值 | 忽略 padding、metadata、重复读取和 C write allocate | 用 profiler 中的实际 DRAM bytes 修正 |
| decode 仍按大矩阵 GEMM 方式优化 | Tensor Core 使用率看似提高，端到端延迟不动 | 优先减少 KV bytes 和提高有效带宽 |
| prefill 只看 memory pipe | 忽略 Tensor Core、TMEM 和 barrier stall | 检查 compute roof、pipeline overlap 与 wait reason |
| 认为低 occupancy 必然慢 | 错删有助于 producer-consumer overlap 的结构 | 看 TMA、Tensor Core、store path 是否持续活跃 |

## 九、自测题与答案

### 1. B200 近似 compute roof 为 2000 TFLOP/s，HBM 为 8 TB/s，ridge point 是多少？

```text
ridge point
= 2000 / 8
= 250 FLOP/byte
```

低于 `250` 更可能 memory-bound，高于 `250` 更可能 compute-bound。

### 2. 方阵 GEMM 的理想 HBM AI 是 `N/3`。`N=2048` 时属于哪一侧？

```text
AI = 2048 / 3
   = 682.666667 FLOP/byte
```

`682.666667 > 250`，因此在理想 HBM 模型下是 compute-bound。

### 3. Decode 一个 query token、KV length 为 4096、head dimension 为 128 时，为什么 AI 近似为 1？

计算量近似为：

```text
4 * S * D
```

KV cache 读取量为：

```text
K + V = 2*S*D + 2*S*D = 4*S*D bytes
```

两者相除得到：

```text
AI = 1 FLOP/byte
```

这个值远低于 `250`，所以典型 decode attention 是 memory-bound。

### 4. 为什么同一个 GEMM 可以是 compute-bound，但单个 CTA stage 却是 memory-bound？

因为 AI 依赖计算与 byte 所处的复用范围。整个 GEMM 可以通过 L2 和
片上复用减少有效 HBM byte；而单个 CTA stage 只加载自己的 A/B tile，
没有跨 CTA 的额外复用时更容易受 HBM 流量限制。

### 5. 一个 compute-bound kernel 实测只有 300 TFLOP/s，硬件 roof 是 2000 TFLOP/s，首先应该看什么？

先确认 profiler 显示它没有被 DRAM bandwidth 限制，然后检查：

```text
Tensor Core issue rate
TMA 与 MMA overlap
mbarrier wait reason
epilogue 是否阻塞下一轮
warp specialization 是否让关键路径空转
```

这些正是前面 GEMM advanced 与 Flash Attention 章节反复处理的性能问题。

## 十、课程正文收束

截至这一节，当前笔记已经形成完整主线：

```text
Part I 硬件与性能模型
  layout
  named axes
  swizzle
  TMA
  Tensor Core
  TMEM
  mbarrier
  CLC
  roofline

Part II TIRx
  第一个 kernel
  scope / layout / dispatch
  TileLayout API

Part III GEMM
  tiled baseline
  K loop
  TMA async load
  software pipeline
  persistent scheduler
  warp specialization
  Two-CTA cluster
  multi-consumer

Part IV Flash Attention 4
  online softmax
  conditional rescaling
  QK^T / softmax / PV
  TMEM 复用
  correction 与 epilogue
  causal mask
  GQA
  LPT scheduling
  reference 与 tolerance

性能收束
  arithmetic intensity
  ridge point
  prefill vs decode
  按瓶颈选择优化方向
```

教材的 `appendix` 更适合后续查阅，不应该当作必须顺序啃完的正文：

```text
language reference      查询 TIRx 语法与 API
benchmarking            可复现测时、Proton、Nsight Compute
compiler internals      查询 lowering pipeline
debugging               排查 hang、crash、错误结果和 slowdown
```

### 下一步怎么继续

1. 在真实 NVIDIA GPU 上跑一个小 GEMM 和 attention，记录实测 TFLOP/s、
   DRAM bytes、SM busy、Tensor Core busy 和主要 stall reason。
2. 用自己的 workload 计算 AI 与 ridge point，先判断瓶颈，再选择修改
   TMA、pipeline、layout 还是算法。
3. 对 prefill 和 decode 分开做 benchmark，不要用一个平均值替代两种
   完全不同的性能状态。
4. 学习 PTX/SASS 时，把每条指令放回本文的三个问题：

```text
它增加了多少计算量？
它搬运了多少 byte？
它是否让关键硬件单元更少等待？
```

能把这三个问题稳定回答清楚，就已经具备从推理系统需求走向 GPU
kernel 设计的基本判断力。

## 十一、章节完成清单

```text
chapter_performance
[x] Roofline 模型
[x] arithmetic intensity
[x] B200 ridge point
[x] GEMM、attention prefill、decode 的瓶颈分类
[x] 完整可运行 roofline 计算脚本
[x] 课程正文 Parts I-IV 收束
```

课程正文主线已完成。后续保留 `appendix` 作为按需查阅资料，不再把
每个附录条目当作连续课程继续展开。
