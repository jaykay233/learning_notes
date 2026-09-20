# 06 Warp Tile、K 循环与 Double Buffering

## 目标

第 5 篇解决了单条 `mma.sync.aligned.m16n8k16` 的 fragment 映射。
这一节继续回答：

```text
一个 warp 如何用很多条 MMA 计算更大的 M/N tile
A/B fragment 如何在多个 MMA 之间复用
K 方向为什么要循环累加
为什么单缓冲会在 global memory 延迟上停顿
double buffering 如何重叠搬运和计算
多 stage pipeline 的收益和 shared memory 代价
```

核心数据路径是：

```text
global memory
    -> shared memory
    -> ldmatrix
    -> MMA
    -> accumulator registers
```

## 从单条 MMA 到 Warp Tile

单条 `m16n8k16` 只能计算：

```text
M = 16
N = 8
K = 16
```

完整 GEMM 通常远大于这个形状。一个 warp 会把自己的输出区域划分成
多个 MMA tile。

假设 warp tile 是：

```text
M_w = 64
N_w = 32
```

沿 M 和 N 方向拆分：

```text
M 方向：64 / 16 = 4
N 方向：32 / 8  = 4
```

所以每个 `K=16` 的步骤需要：

```text
4 * 4 = 16 次 MMA
```

可以表示为：

```text
C00 C01 C02 C03
C10 C11 C12 C13
C20 C21 C22 C23
C30 C31 C32 C33
```

每个 `Cij` 都是一个：

```text
16 x 8 fp32 accumulator
```

## Accumulator 的寄存器占用

`64 x 32` 输出 tile 总共有：

```text
64 * 32 = 2048 个 fp32 元素
```

一个 warp 有 32 个 lane：

```text
2048 / 32 = 64 个 fp32/lane
```

因此每个 lane 需要保存：

```text
64 个 fp32 accumulator registers
```

逻辑上可以写成：

```text
C[4][4][4]
```

三个维度的含义是：

```text
第一个 4：M 方向的 MMA tile
第二个 4：N 方向的 MMA tile
第三个 4：单个 m16n8k16 fragment 内的 4 个元素
```

Accumulator 必须持续驻留在寄存器中。K 循环中途如果频繁写回 global
memory，会把 GEMM 变成反复 load/store accumulator 的 kernel。

## Fragment 复用

假设当前 K block 是 `K=16`，A 有四个 M tile：

```text
A0, A1, A2, A3
```

B 有四个 N tile：

```text
B0, B1, B2, B3
```

执行逻辑类似：

```python
for mi in range(4):
    for ni in range(4):
        mma(C[mi][ni], A[mi], B[ni], C[mi][ni])
```

这里会发出 16 次 MMA，但只需要：

```text
4 个 A fragment
4 个 B fragment
```

因为同一个 fragment 会被多个 MMA 复用：

```text
同一份 A[mi] 被 4 个 N tile 使用
同一份 B[ni] 被 4 个 M tile 使用
```

更大的 M/N warp tile 可以提高复用率，减少每次 MMA 对应的数据加载
开销，但 accumulator 数量也会增加。

## K 循环

假设完整 GEMM 的 K 是 128：

```text
K = 128
```

按照 `K=16` 切分：

```text
k0 = 0, 16, 32, 48, 64, 80, 96, 112
```

循环结构类似：

```python
C = 0

for k0 in range(0, K, 16):
    load A fragments at k0
    load B fragments at k0

    for mi in range(4):
        for ni in range(4):
            mma(C[mi][ni], A[mi], B[ni], C[mi][ni])

store C
```

关键区别是：

```text
M/N 方向：产生不同的 C fragment
K 方向：反复累加到同一组 C fragment
```

A/B fragment 会随 `k0` 变化，而 C accumulator 一直保留到 K 循环结束。

因此寄存器占用主要随：

```text
M_w * N_w
```

增长，而不是随 K 增长。

## Warp Tile 越大越好吗

更大的 warp tile 可以提高数据复用，但寄存器压力会快速增加。

例如：

```text
64 x 32 tile -> 64 个 fp32 accumulator/lane
64 x 64 tile -> 128 个 fp32 accumulator/lane
```

除了 accumulator，还需要保存：

```text
A fragment
B fragment
地址
循环变量
pipeline 状态
```

如果寄存器不足，可能出现：

```text
register spill
occupancy 下降
加载延迟无法被其他 warp 隐藏
```

实际 kernel 需要在下面三者之间平衡：

```text
数据复用
寄存器占用
occupancy
```

## 单缓冲为什么会产生停顿

假设 shared memory 只有一个 K tile 缓冲区：

```text
copy tile k from global to shared
wait for copy
ldmatrix + MMA
copy tile k+1
wait for copy
ldmatrix + MMA
```

时间线是：

```text
copy tile 0
compute tile 0
copy tile 1
compute tile 1
copy tile 2
compute tile 2
```

复制和计算严格串行：

```text
搬运数据时，Tensor Core 在等待
计算数据时，global memory 路径可能空闲
```

如果 global memory 延迟大于一次 MMA 的执行时间，Tensor Core 会频繁
空等。

## Double Buffering

Double buffering 使用两块 shared memory buffer：

```text
buffer 0
buffer 1
```

当前 buffer 用于 `ldmatrix + MMA`，下一块 buffer 同时接收下一个 K tile：

```text
时间 ---------------------------------------------->

copy tile 0     [======]
compute tile 0          [============]
copy tile 1            [======]
compute tile 1                    [============]
copy tile 2                       [======]
compute tile 2                              [============]
```

稳定状态下：

```text
compute tile k
   与
copy tile k+1
```

同时发生。

两块 buffer 轮换使用：

```text
当前计算读 buffer[k % 2]
下一份搬运写 buffer[(k + 1) % 2]
```

因此也叫 ping-pong buffering。

## 示意伪代码

下面是 pipeline 结构示意，不是可以直接编译的 CUDA：

```python
prefetch(tile=0, buffer=0)

for k in range(num_k_tiles):
    wait_for_current_tile()

    if k + 1 < num_k_tiles:
        prefetch(
            tile=k + 1,
            buffer=(k + 1) % 2,
        )

    compute(
        tile=k,
        buffer=k % 2,
    )
```

核心是让下面两件事没有数据依赖：

```text
计算正在读取 buffer A
异步复制正在写入 buffer B
```

一旦它们共享同一块 buffer，就必须增加同步，否则复制可能覆盖当前 MMA
仍在读取的数据。

## Ampere 上的 cp.async

Ampere 提供：

```text
cp.async
```

它可以把数据从 global memory 直接复制到 shared memory，而不需要先经过
寄存器：

```text
global memory
    -> cp.async
    -> shared memory
    -> ldmatrix
    -> registers
    -> MMA
```

常见控制指令：

```text
cp.async.commit_group
cp.async.wait_group
```

含义可以概括为：

```text
commit_group：提交当前这一批异步复制
wait_group：等待之前提交的复制组完成
```

于是可以形成：

```text
MMA 计算 tile k
   同时
cp.async 搬运 tile k+1
```

真正实现时还要处理：

```text
barrier
buffer 可复用条件
producer / consumer 的可见性
最后几个 K tile 的 epilogue
```

## Prologue、Steady State、Epilogue

软件 pipeline 通常分成三个阶段：

```text
Prologue：启动流水线，先装入第一批数据
Steady State：当前计算与未来搬运持续重叠
Epilogue：停止预取，计算剩余数据
```

这三个阶段描述的是 pipeline 从“没有数据”到“稳定重叠”，再到“没有后续
数据”的完整生命周期。

### Prologue：填满流水线

刚开始时，shared memory buffer 里没有数据，不能立刻计算。

对于 double buffering，先搬入第一个 tile：

```text
Prologue:
    copy tile 0 -> buffer 0
```

此时还没有重叠：

```text
数据正在搬运
计算单元暂时等待
```

Prologue 的任务是让 pipeline 进入可运行状态。

对于 `S` 个 stage 的 pipeline，通常先预取：

```text
S - 1 个 tile
```

例如：

```text
2 stage -> 预取 1 个 tile
3 stage -> 预取 2 个 tile
4 stage -> 预取 3 个 tile
```

### Steady State：稳定重叠

Prologue 完成后，pipeline 进入效率最高的阶段：

```text
一边计算当前 tile
一边搬运后续 tile
```

假设有四个 K tile：

```text
T0, T1, T2, T3
```

Double buffering 的 steady state 是：

```text
compute T0 || copy T1
compute T1 || copy T2
compute T2 || copy T3
```

这里的 `||` 表示两件事并行发生。

具体 buffer 轮换是：

```text
第 1 轮：
    读取 buffer 0，计算 T0
    同时向 buffer 1 搬运 T1

第 2 轮：
    读取 buffer 1，计算 T1
    同时向 buffer 0 搬运 T2

第 3 轮：
    读取 buffer 0，计算 T2
    同时向 buffer 1 搬运 T3
```

这就是 ping-pong：

```text
buffer 0 -> buffer 1 -> buffer 0 -> buffer 1
```

Steady state 的价值是：

```text
用当前 tile 的 compute 时间覆盖后续 tile 的 memory latency
```

如果 K 很长，大部分执行时间都位于 steady state，因此 pipeline 更容易
带来明显收益。

### Epilogue：排空流水线

搬入最后一个 tile 后，已经没有后续数据可以预取：

```text
Epilogue:
    compute T3
```

此时不会再出现：

```text
compute T3 || copy T4
```

因为 `T4` 不存在。

Epilogue 的任务是把已经搬入但还没有计算的 tile 全部消费掉。对于
`S` 个 stage，通常最后还有：

```text
S - 1 个 tile
```

需要单独计算。

### 完整时间线

四个 tile、double buffering 的完整结构是：

```text
Prologue:
    copy T0

Steady State:
    compute T0 || copy T1
    compute T1 || copy T2
    compute T2 || copy T3

Epilogue:
    compute T3
```

可以画成：

```text
             Prologue     Steady State                  Epilogue
             ---------    --------------------------    --------
copy T0      [====]
compute T0                [===========]
copy T1                    [====]
compute T1                              [===========]
copy T2                                  [====]
compute T2                                            [===========]
copy T3                                                [====]
compute T3                                                            [===========]
```

两种不可避免的空档是：

```text
Prologue：计算单元等待第一份数据
Epilogue：搬运路径已经没有后续工作
```

只有 steady state 持续满足：

```text
搬运和计算同时进行
```

### 多 stage 的推广

```text
S stage pipeline

Prologue:
    预取前 S - 1 个 tile

Steady State:
    compute tile k || prefetch tile k + S - 1

Epilogue:
    计算最后 S - 1 个 tile
```

例如 3-stage pipeline 有六个 tile：

```text
Prologue:
    copy T0
    copy T1

Steady State:
    compute T0 || copy T2
    compute T1 || copy T3
    compute T2 || copy T4
    compute T3 || copy T5

Epilogue:
    compute T4
    compute T5
```

### 对性能的影响

如果 K 很小，例如只有两个 K tile：

```text
T0, T1
```

那么：

```text
Prologue + Epilogue 占比很高
Steady State 很短
```

多 stage pipeline 可能收益不明显，同时浪费 shared memory。

如果 K 很长，例如有：

```text
T0, T1, T2, ..., T63
```

那么：

```text
Steady State 占绝大多数时间
```

Double buffering 或 3/4-stage pipeline 更有机会隐藏 global memory
latency。

## 为什么需要 Multi-stage Pipeline

Double buffering 只有两个 stage。如果 global memory 延迟或 `cp.async`
延迟仍然大于一个 tile 的 compute 时间，计算方还是要等待。

可以使用更多 stage：

```text
stage 0: 当前计算
stage 1: 下一块已经到达
stage 2: 再下一块正在搬运
stage 3: 更早预取
```

常见形式是 3-stage 或 4-stage：

```text
在 stage s 计算
同时预取 s+1、s+2 或更多 tile
```

stage 数增加后，硬件拥有更多独立数据可用，更容易用 compute 覆盖
memory latency。

## Shared Memory 代价

一个 stage 的 A/B tile 大小为：

```text
A stage = BM * BK * bytes_per_element
B stage = BK * BN * bytes_per_element
```

以 fp16 为例：

```text
BM = 128
BN = 128
BK = 32
bytes_per_element = 2
```

则：

```text
A stage = 128 * 32 * 2 = 8192 Byte
B stage =  32 * 128 * 2 = 8192 Byte
```

一个 stage 共：

```text
16 KB
```

不同 stage 数需要的 shared memory：

```text
1 stage -> 16 KB
2 stage -> 32 KB
3 stage -> 48 KB
4 stage -> 64 KB
```

stage 越多通常越容易隐藏延迟，但会增加：

```text
shared memory 占用
同步复杂度
pipeline 状态
可能的 CTA occupancy 压力
```

## 在推理系统中的意义

prefill 阶段的大 GEMM、Attention 的 Q/K/V projection 和 MLP
projection 通常有较长的 K 循环，适合：

```text
double buffering
multi-stage pipeline
warp specialization
```

decode 阶段某些 GEMM 的 M/N 很小，主要瓶颈可能变成：

```text
权重读取带宽
kernel launch
调度开销
KV Cache 访问
```

这时只增加 K pipeline stage 不一定有效，因为瓶颈可能不在 K tile 的
搬运延迟。

## 与第 5 篇的连接

第 5 篇的 `ldmatrix` 负责：

```text
shared memory -> A/B fragment registers
```

这一篇的 pipeline 负责：

```text
让 shared memory -> fragment 这一步永远有已经准备好的数据
```

完整结构可以概括为：

```text
cp.async / TMA
    -> shared memory stage buffers
    -> ldmatrix
    -> A/B fragments
    -> mma.sync
    -> C accumulators
    -> epilogue
```

## 结论

```text
M/N 方向用于覆盖更大的 warp tile
K 方向反复累加到同一组 C fragments
A/B fragment 在多个 MMA 之间复用
单缓冲会串行等待 global memory
double buffering 用两块 buffer 重叠搬运和计算
multi-stage pipeline 用更多 shared memory 换取更高延迟隐藏能力
A/B stage 大小由 BM、BN、BK 和 dtype 决定
```
