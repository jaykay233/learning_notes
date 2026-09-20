# 07 Hopper WGMMA Fragment 与 Blackwell TMEM

## 课程位置

课程：

- <https://github.com/mlc-ai/modern-gpu-programming-for-mlsys>

本章：

- `chapter_layout_generations`
- 中文标题：《Tensor Core 数据布局的演进》

对应章节：

```text
Hopper：从 Shared Memory 直接读取
    └── 累加器仍在寄存器中

Blackwell：累加器进入 TMEM
    └── TMEM 中的累加器
```

补充背景来自同一课程中的：

```text
chapter_data_layout
    └── TMEM 的二维物理空间

chapter_tensor_cores
    └── TMEM 中的 Accumulator
```

## 目标

这一篇连接 Ampere、Hopper 和 Blackwell 的 Tensor Core 数据路径：

```text
Ampere
    A/B register fragment
    -> mma.sync
    -> C/D register fragment

Hopper
    A/B shared-memory descriptor
    -> wgmma.mma_async
    -> C/D register fragment

Blackwell
    A/B shared-memory descriptor
    -> tcgen05.mma
    -> C/D TMEM
    -> tcgen05.ld
    -> register fragment
```

重点回答：

```text
Hopper WGMMA 的 accumulator 为什么仍在 registers 中
每个 thread 为什么持有 M * N / 128 个累加元素
CUTLASS CLayout 的 shape 和 stride 分别描述什么
row / col 映射公式是怎样从 CLayout 推导出来的
Blackwell 为什么把长期存活的 accumulator 移入 TMEM
TMEM 的 TLane / TCol 如何表示二维地址
cta_group::1、M=128 时 C[m,n] 映射到哪里
tcgen05.mma 和 tcgen05.ld 的异步完成条件是什么
```

## 三代架构的数据路径

| 架构 | 主要 MMA 指令 | A/B 主要来源 | 长期累加器位置 | shared memory 布局的表达方式 |
|---|---|---|---|---|
| Ampere | `mma.sync` | registers | registers | kernel 计算地址并应用 swizzle |
| Hopper | `wgmma.mma_async` | A 可来自 registers 或 SMEM；B 来自 SMEM | registers | matrix descriptor 记录 stride 和 swizzle |
| Blackwell | `tcgen05.mma` | 主要来自 SMEM；某些 A 模式可来自 TMEM | TMEM | descriptor 描述 SMEM；TMEM layout 另行描述 accumulator |

可以概括为：

```text
Ampere：先把输入排成 per-lane register fragment
Hopper：让 Tensor Core 通过 descriptor 直接读取 SMEM
Blackwell：把 accumulator 和 scale factors 移入 TMEM
```

## Hopper：输入进入 SMEM 路径，累加器仍在旧位置

Hopper 的 `wgmma.mma_async` 由 128 个 thread 组成的 warpgroup 协同执行。

A/B 有两种常见来源：

```text
SS：A from SMEM, B from SMEM
RS：A from registers, B from SMEM
```

当输入来自 shared memory 时，不需要像 Ampere 那样先使用 `ldmatrix`
构造完整的 A/B register fragment。WGMMA 会直接读取 SMEM，但 matrix
descriptor 必须告诉它：

```text
矩阵起点在哪里
跨到下一组数据时移动多少字节
shared memory 使用了哪种 swizzle
矩阵起点位于 swizzle 重复模式的什么位置
```

关键变化只发生在输入侧。Hopper 的 C/D accumulator 仍然分散在
warpgroup 内各 thread 的寄存器中：

```text
SMEM
    -> wgmma.mma_async
    -> per-thread C/D register fragments
    -> epilogue
```

所以 Hopper kernel 同时存在两种布局：

```text
A/B in SMEM：使用 matrix descriptor 描述
A in registers：使用 per-thread register fragment 描述
C/D in registers：使用 per-thread register fragment 描述
```

## 每个 thread 持有多少 accumulator

WGMMA 的 accumulator fragment 分布在 128 个 thread 上。

如果一个 WGMMA tile 的逻辑形状是 `M x N`，则：

```text
elements_per_thread = M * N / 128
```

例如：

| WGMMA shape | 累加元素总数 | 每 thread 元素数 |
|---|---:|---:|
| `m64n8k16` | `64 * 8 = 512` | `512 / 128 = 4` |
| `m64n16k16` | `64 * 16 = 1024` | `1024 / 128 = 8` |
| `m64n32k16` | `64 * 32 = 2048` | `2048 / 128 = 16` |

如果 accumulator 是 fp32，则每个元素占一个 32-bit register：

```text
m64n8k16, f32 accumulator  -> 4 个 f32 registers/thread
m64n16k16, f32 accumulator -> 8 个 f32 registers/thread
m64n32k16, f32 accumulator -> 16 个 f32 registers/thread
```

如果 accumulator 是 fp16 或 bf16，则两个元素可以打包进一个 32-bit
register。例如：

```text
m64n16k16, f16 accumulator
    8 个 f16 values/thread
    -> 4 个 32-bit registers/thread
```

这也是 Hopper 进一步提高 M/N tile 时的限制之一：输入可以来自 SMEM，
但长期存活的 accumulator 仍会持续消耗 registers。

## CUTLASS 中的 Hopper CLayout

CUTLASS 为 Hopper WGMMA 定义了一组 accumulator layout：

```cpp
template<int N>
using CLayout_64xN = Layout<
    Shape<
        Shape<_4, _8, _4>,
        Shape<_2, _2, Int<N / 8>>
    >,
    Stride<
        Stride<_128, _1, _16>,
        Stride<_64, _8, _512>
    >
>;
```

对应常见实例：

```cpp
using CLayout_64x8  = CLayout_64xN<8>;
using CLayout_64x16 = CLayout_64xN<16>;
using CLayout_64x32 = CLayout_64xN<32>;
```

这里的 `CLayout` 不是直接描述 `(row, col)`，而是描述：

```text
(thread index, value index)
    -> C tile 的线性 offset
```

令：

```text
tid = thread index in warpgroup, 0 ... 127
v   = per-thread value index, 0 ... M*N/128 - 1
```

CUTLASS 使用两个嵌套 shape：

```text
Shape<4, 8, 4>         描述 tid 如何拆分
Shape<2, 2, N / 8>     描述 v 如何拆分
```

### shape 如何拆分 `tid`

先定义：

```text
lane = laneid % 32
warp = laneid // 32, 0 ... 3
```

shape `Shape<4, 8, 4>` 对应：

```text
t0, t1, t2
```

CUTE 的默认拆分顺序是第一个坐标变化最快：

```text
tid = t0 + 4*t1 + 32*t2
```

因此：

```text
t0 = lane % 4
t1 = lane // 4
t2 = warp
```

三部分的含义是：

```text
t0：lane 在一个 4-lane group 内的编号
t1：4-lane group 的编号，每组覆盖 8 个连续逻辑 row
t2：warpgroup 中的 warp 编号
```

### shape 如何拆分 `v`

shape `Shape<2, 2, N / 8>` 对应：

```text
v0, v1, v2
```

同样按第一维最快的方式拆分：

```text
v = v0 + 2*v1 + 4*v2
```

所以：

```text
v0 = v % 2
v1 = (v / 2) % 2
v2 = v / 4
```

这三个坐标选择的是同一个 thread 持有的不同 register slot：

```text
v0：在一对相邻 column 中选择
v1：在一个 8-row 小区域内选择高低两组 4 rows
v2：沿 N 方向选择下一个 8-column group
```

### stride 为什么不能直接解释成 row / col

CLayout 的 stride 是：

```text
t0 stride = 128
t1 stride = 1
t2 stride = 16

v0 stride = 64
v1 stride = 8
v2 stride = 512
```

先计算 C tile 的线性的 row-major offset：

```text
flat =
    128*t0
  +   1*t1
  +  16*t2
  +  64*v0
  +   8*v1
  + 512*v2
```

这里的 C tile 有 64 行，因此：

```text
flat = row + 64*col
```

由于：

```text
0 <= t1 + 16*t2 + 8*v1 <= 7 + 48 + 8 = 63
```

低位部分不会产生进位，所以可以直接拆开：

```text
row = 16*t2 + 8*v1 + t1
col = 8*v2 + 2*t0 + v0
```

代回 thread 和 register 坐标：

```text
row =
    16 * warp
  +  8 * ((v / 2) % 2)
  +      lane / 4
```

```text
col =
    8 * (v / 4)
  + 2 * (lane % 4)
  +     (v % 2)
```

这就是 WGMMA accumulator fragment 的标准映射。

## 一个具体 lane 的例子

取：

```text
laneid = 5
warp   = 1
v      = 7
```

先求 thread 坐标：

```text
lane = 5
t0 = 5 % 4 = 1
t1 = 5 / 4 = 1
t2 = 1
```

再求 register 坐标：

```text
v0 = 7 % 2 = 1
v1 = (7 / 2) % 2 = 3 % 2 = 1
v2 = 7 / 4 = 1
```

代入映射：

```text
row = 16*1 + 8*1 + 1 = 25
col =  8*1 + 2*1 + 1 = 11
```

因此：

```text
thread laneid=5, register slot v=7
    -> C[25, 11]
```

校验线性 offset：

```text
flat = row + 64*col
     = 25 + 64*11
     = 729
```

CLayout 公式得到：

```text
flat =
    128*1
  +   1*1
  +  16*1
  +  64*1
  +   8*1
  + 512*1
  = 729
```

两边一致。

## Blackwell：累加器进入 TMEM

Blackwell 的输入路径延续 Hopper 的思路。`tcgen05.mma` 通过 descriptor
找到 shared memory 中的 A/B，某些模式也允许 A 来自 TMEM。

真正的大变化在 accumulator：

```text
Hopper C/D    -> per-thread registers
Blackwell C/D -> TMEM
```

完整数据路径可以写成：

```text
A/B in SMEM
    -> tcgen05.mma
    -> C/D in TMEM
    -> tcgen05.ld
    -> register fragment
    -> GMEM
```

如果把 TMA 也画进去：

```text
A/B: GMEM -> TMA -> SMEM -> tcgen05.mma -> TMEM

D:   TMEM -> tcgen05.ld -> registers -> epilogue -> GMEM
```

TMEM 的作用是保存主循环中长期存活的 accumulator。这样 accumulator
不再持续占用 register file，但 kernel 必须新增两类工作：

```text
正确分配和寻址 TMEM
保证 MMA 写入布局与 tcgen05.ld 读取布局匹配
```

## TMEM 的物理形状

在 `sm_100a` 上，每个 CTA 拥有一块 TMEM，逻辑上包含：

```text
128 个 Lane rows
每个 Lane row 最多 512 个 columns
每个 cell 为 32 bits
```

容量可以算成：

```text
128 * 512 * 4 Byte
= 262144 Byte
= 256 KiB
```

TMEM 不能只用一个线性地址表示，因为一个位置同时需要：

```text
TLane：选择 128 个 Lane rows 中的一行
TCol：选择该 Lane 上的一个 32-bit column
```

例如一个 `128 x 256` 的 accumulator 可以表示为：

```text
S[(128, 256) : (1@TLane, 1@TCol)]
```

普通 row-major shared memory 只有一个地址轴：

```text
S[(128, 256) : (256@m, 1@m)]
```

而 TMEM 使用两个命名轴：

```text
f(row, col) = row@TLane + col@TCol
```

这使 TMEM layout 天然是二维的，也意味着 `tcgen05.ld` 必须从正确的
`TLane` 和 `TCol` 坐标读取数据。

## `cta_group::1`、`M = 128` 的 accumulator 映射

最简单的 Blackwell accumulator layout 是：

```text
cta_group::1
M = 128
```

一个 CTA 恰好有 128 个 TMEM Lane rows，而 output tile 也恰好有 128
个 M rows，所以可以直接一一对应：

```text
C[m, n] -> (TLane=m, TCol=n)
```

例如：

```text
C[10, 3] -> TLane 10, TCol 3
```

当 accumulator 为 fp32 时，一个逻辑元素正好占一个 32-bit TMEM cell。
对于 fp16/bf16 等打包格式，后续读取时还要结合对应的寄存器打包规则，
不能无条件地把“一个逻辑元素等于一个 32-bit cell”套到所有 dtype。

这个直接映射只在上面这组 layout 条件下成立。其他情况可能采用不同
mapping，例如：

```text
cta_group::1, M=64
cta_group::2, M=256
cta_group::2, M=128
```

因此不能只记住一个公式，还要同时检查：

```text
cta_group
M
instruction kind
dtype
是否使用 .ws
```

## 异步完成：MMA 和 TMEM load

### `tcgen05.mma` 是异步的

发出 `tcgen05.mma` 只表示 MMA 已经启动，不表示 TMEM 中的结果已经
写完。

MMA 发起 thread 通常通过：

```text
tcgen05.commit...mbarrier::arrive
```

把此前发出的异步 `tcgen05` 操作关联到一个 `mbarrier`。MMA 真正完成后，
硬件才会向该 barrier 报告 arrival。

epilogue 的 consumer 必须先等待对应的 barrier，并满足必要的
`tcgen05` 顺序要求，才能读取 TMEM。

错误情况：

```text
发出 tcgen05.mma
    -> 立即 tcgen05.ld
    -> 可能读到尚未写完的 TMEM
```

### `tcgen05.ld` 也是异步的

`tcgen05.ld` 从 TMEM 加载数据到 registers，但加载完成后，目标 registers
才可以安全使用。

所以：

```text
tcgen05.ld
    -> tcgen05.wait::ld
    -> 使用 registers
```

这里的两个等待解决不同问题：

```text
mbarrier wait
    确认 tcgen05.mma 已经完成

tcgen05.wait::ld
    确认该 warp 此前发出的 TMEM load 已经写入目标 registers
```

## 三代架构中的 register fragment 角色

register fragment 在三代架构中都存在，但承担的角色不同。

| 架构 | 计算期间的 accumulator | register fragment 出现的位置 |
|---|---|---|
| Ampere | registers | A/B 输入和 C/D accumulator |
| Hopper | registers | A 的 RS 输入和 C/D accumulator |
| Blackwell | TMEM | TMEM 与 epilogue 的交界处 |

可以写成：

```text
Ampere:
    SMEM -> ldmatrix -> A/B fragments -> MMA -> C/D fragments

Hopper:
    SMEM -> descriptor -> WGMMA -> C/D fragments

Blackwell:
    SMEM -> descriptor -> tcgen05.mma -> TMEM
                                       -> tcgen05.ld
                                       -> register fragment
```

Hopper 主要在“计算结束以后”使用 accumulator register fragment；
Blackwell 主要在“从 TMEM 搬回 epilogue 时”使用 register fragment。

## 对推理系统的意义

prefill 阶段的 GEMM、Q/K/V projection、MLP projection 通常有较长的 K
循环和较大的 M/N tile。

Hopper 的 accumulator register pressure 会限制：

```text
warpgroup 的 M/N tile 大小
同时驻留的 warpgroup 数量
epilogue 可以使用的寄存器数量
```

Blackwell 把 accumulator 移入 TMEM 后，register pressure 降低，但新的
约束变成：

```text
TMEM 容量和分配
MMA 到 epilogue 的同步
tcgen05.ld 的带宽和 load shape
TMEM layout 与 epilogue layout 的一致性
```

这会影响推理 kernel 的优化方向。例如长 K 的大 GEMM 更容易体现
TMEM 的优势；decode 阶段的小 M/N GEMM 可能主要受权重读取、KV Cache
访问、kernel launch 或调度开销限制，不能只根据“用了 TMEM”判断性能。

## 结论

```text
Hopper WGMMA 直接从 SMEM 读取 A/B
Hopper 的 C/D accumulator 仍在 registers
WGMMA 每个 thread 持有 M*N/128 个累加元素
CUTLASS CLayout 的 shape 拆分 tid 和 per-thread value index
CLayout 的 stride 描述映射到 C tile 的线性 offset
Hopper accumulator 的 row/col 公式可由 CLayout 完整推导
Blackwell tcgen05.mma 把 C/D 写入 TMEM
TMEM 是 128 Lane rows 的二维 CTA-scoped memory space
cta_group::1、M=128 时，C[m,n] 直接映射到 TLane=m、TCol=n
tcgen05.mma 完成后 epilogue 才能读取 TMEM
tcgen05.ld 完成后需要 tcgen05.wait::ld 才能使用 registers
register fragment 在 Blackwell 主要位于 TMEM 与 epilogue 的边界
```
