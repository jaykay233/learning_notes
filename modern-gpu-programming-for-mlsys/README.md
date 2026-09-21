# Modern GPU Programming for MLSys 学习笔记

课程：

- <https://github.com/mlc-ai/modern-gpu-programming-for-mlsys>

本目录整理对话中围绕 GPU 数据布局、命名轴、Replication、Offset、
shared memory swizzle、Tensor Core fragment、`ldmatrix`、K 循环
pipeline 和 TMA 异步搬运展开的内容，重点关注这些概念如何影响推理
系统中的 GEMM、Attention 以及 KV Cache 数据路径。

## 文档索引

| 文档 | 内容 |
|---|---|
| [01-data-layout-and-named-axes.md](01-data-layout-and-named-axes.md) | Shape-Stride、Tile Layout、命名轴、Register Fragment |
| [02-replication-and-offset.md](02-replication-and-offset.md) | `R[...]` 副本、`O[...]` 偏移、TMEM 广播、GPU Mesh |
| [03-practice-and-corrections.md](03-practice-and-corrections.md) | 自测题、原始答案、逐题订正 |
| [04-swizzle-layout.md](04-swizzle-layout.md) | shared memory bank conflict、XOR swizzle、完整地址推导 |
| [05-ampere-mma-fragments.md](05-ampere-mma-fragments.md) | `mma.sync`、A/B/C/D fragment、`ldmatrix.x1/x2/x4`、`.trans` |
| [06-warp-tile-and-k-pipeline.md](06-warp-tile-and-k-pipeline.md) | Warp tile、fragment 复用、K 循环、`cp.async`、double buffering |
| [07-hopper-wgmma-and-blackwell-tmem.md](07-hopper-wgmma-and-blackwell-tmem.md) | Hopper WGMMA accumulator fragment、CUTLASS `CLayout`、Blackwell TMEM、`tcgen05`、SFA/SFB 与 `scale_vec` |
| [08-tma-tile-copy-and-synchronization.md](08-tma-tile-copy-and-synchronization.md) | TMA descriptor、128-byte swizzle、3D box、row layout、pipeline、`ldmatrix` 判断、mbarrier load 与 bulk-group store |
| [08-tma-practice-and-solutions.md](08-tma-practice-and-solutions.md) | TMA 10 道自测题、计算推导与答案 |
| [09-tensor-cores-tcgen05.md](09-tensor-cores-tcgen05.md) | Blackwell `tcgen05.mma`、TMEM accumulator、`tcgen05.commit` 与 `cta_group` |
| [10-tmem-allocation-lifecycle.md](10-tmem-allocation-lifecycle.md) | TMEM 容量、按列 allocation、warp Lane 访问窗口、`tcgen05.ld/st`、shape/num、pack/unpack，以及 `wait::ld/st` 的异步完成边界 |
| [11-mbarrier-phase-lifecycle.md](11-mbarrier-phase-lifecycle.md) | `mbarrier` 的 arrival、phase 完成条件、parity 翻转与双 stage pipeline 中的 phase 跟踪 |

## 当前进度

已完成：

```text
Shape-Stride 模型
Tile Layout
命名轴 @m / @TLane / @TCol / @laneid / @reg
Register Fragment 映射
Replication R[...]
Offset O[...]
Shared memory bank conflict
XOR swizzle
mma.sync.aligned
m16n8k16 A/B/C/D fragment
ldmatrix x1 / x2 / x4
ldmatrix.trans
Warp tile M/N 分解
A/B fragment 复用
K 循环累加
Double buffering
cp.async pipeline
Hopper WGMMA accumulator fragment
WGMMA M*N/128 register mapping
CUTLASS CLayout shape / stride 推导
Blackwell TMEM accumulator
TLane / TCol 二维地址
tcgen05.mma completion
tcgen05.ld 与 tcgen05.wait::ld
Block-scaled MMA 的 SFA/SFB
SFA/SFB 的 SMEM -> TMEM 数据路径
tcgen05.cp 与 .warpx4 四分区广播
scale_vec::1X / 2X / 4X
TMEM partition、word byte、K-block reuse 三种复制的区别
Ampere / Hopper / Blackwell 三种数据路径的对比
Producer / Consumer layout 契约检查
布局错误与同步错误的区分方法
chapter_layout_generations 完成
TMA 单 thread 提交整个 tile
tensor map descriptor 与单次 copy 参数
TMA 写入 128-byte swizzled layout
3D TMA 搬运多个 swizzle atoms
TMA box 最内层宽度限制
128-byte grouped layout 与 256-byte row stride
TMA load 的 mbarrier completion
arrival count 与 pending transaction bytes
TMA store 的 commit group / wait group
load / store 同步机制差异
full / empty barrier 与 stage 所有权
双 stage pipeline
prologue / steady state / epilogue
pipeline phase 与 stage reuse
TMA 之后是否需要 `ldmatrix` 的判定
`mma.sync`、WGMMA、`tcgen05` 的 operand 消费差异
TMA tensor map descriptor
WGMMA / Tensor Core matrix descriptor
descriptor 布局 ABI
base offset / phase 与 descriptor 一致性
chapter_tma 完成
tcgen05.mma 的单 thread 发起语义
tcgen05.mma 指令字段
TMEM accumulator
tcgen05.commit 与 mbarrier
tcgen05 与 TMA load / store 的同步机制对比
cta_group::1 与 cta_group::2 基本操作范围
cta_group::2 的 CTA pair 资源访问边界
cta_group::1, M=128 的直接 accumulator 映射
cta_group::1, M=64 的 Layout F
cta_group::2, M=256 的 CTA pair accumulator 切分
cta_group::2, M=128 dense A 的 Layout B
block-scaled MMA 的 SFA/SFB 跨 CTA pair 放置
tcgen05 指令之间的 scope / layout / completion 三层契约
chapter_tensor_cores 完成
TMEM 128 Lanes x 512 Columns 与 256 KiB 容量
TMEM 按 Column allocation 与 warp-collective tcgen05.alloc
tmem_addr 的 SMEM 可见性与 allocated_addr 绑定
allocation size 与连续 allocation 单调不增约束
relinquish_alloc_permit 与 tcgen05.dealloc
cta_group::2 的 CTA pair allocation 契约
warpgroup 内四个 warp 的固定 32-Lane TMEM 访问窗口
CTA allocation 边界与 warp Lane 访问限制的区别
tcgen05.ld/st 的 warp-collective 数据通路
shape 与 num 的 data volume / register count 计算
16-bit pack/unpack 语义
tcgen05.ld/st 的异步 issue / completion 边界
tcgen05.wait::ld/st 的 per-thread 完成语义
tcgen05.wait、tcgen05.fence 与跨线程线程同步的区别
chapter_tmem 完成
chapter_async_barriers 的 mbarrier arrival / pending count
mbarrier phase 完成条件
phase 完成后 parity 的 0 / 1 翻转
复用 barrier 时等待当前 round parity
双 stage pipeline 的 stage barrier 与 phase_tma
```

## 核心主线

GPU layout 不只回答“数据在哪个地址”，还要回答：

```text
数据属于哪个 thread / lane
数据位于哪个 register fragment slot
数据位于哪个 TMEM lane 和 column
同一逻辑数据是否需要复制到多个物理位置
是否需要给整体布局增加固定偏移
```

Tensor Core kernel 出错时，计算表达式本身往往没有问题，问题更常出现在：

```text
元素放错 lane
元素占用错误的 register slot
TMEM lane / column 不匹配
共享内存 swizzle 与 descriptor 不一致
生产者写出的 layout 与消费者读取的 layout 不一致
```

## 环境说明

- 当前机器：Apple M5 Pro
- 本机没有 CUDA / Blackwell 硬件
- Blackwell 的 TMEM、`tcgen05`、WGMMA 和 TMA 示例主要用于理解 layout 和代码路径
- 概念学习与静态代码分析可以在 macOS 上完成，实际性能测量需要 NVIDIA GPU

## 后续可继续整理

```text
WGMMA / tcgen05 matrix descriptor 字段与编码
TMA producer / consumer warp specialization
chapter_async_barriers 的 common handoffs 与 full/empty barrier stage reuse
```
