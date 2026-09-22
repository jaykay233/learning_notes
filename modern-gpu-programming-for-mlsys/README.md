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
| [11-mbarrier-phase-lifecycle.md](11-mbarrier-phase-lifecycle.md) | `mbarrier` 的 arrival、phase 完成条件、parity 翻转、threads 与 TMA 的 fence / sync 交接、`full` / `empty` stage 所有权，以及 `tcgen05.commit` 的完成 arrival |
| [12-clc-dynamic-scheduling.md](12-clc-dynamic-scheduling.md) | 静态 persistent scheduler 的 launch tail、CLC 请求生命周期、请求与当前 tile 的异步重叠、静态与动态调度的适用边界，以及 `sm_100+` 硬件限制 |
| [13-tirx-first-kernel.md](13-tirx-first-kernel.md) | TIRx 单 tile GEMM 数据路径、Scope / Layout / Dispatch、SMEMPool、TMEM allocation、elected-thread MMA 与 warpgroup writeback |
| [14-tirx-layout-api.md](14-tirx-layout-api.md) | TIRx `TileLayout` 的 `S[...]`、`R[...]`、固定 offset、命名轴语义、`apply()` 的三种入口、TMEM accumulator、scale-factor replica、`tmem_datapath_layout` 的 D/F row mapping、`tcgen05_atom_layout` 的 atom / rep / register mapping、`wg_local_layout` 的 warpgroup row-to-thread mapping，以及 `ComposeLayout` 与 shared-memory XOR swizzle |

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
generic proxy 与 async proxy 的可见性边界
threads 写 SMEM 后通过 fence.proxy.async 交接给 TMA
warpgroup_sync 与 TMA store commit_group / wait_group
full / empty barrier 的 stage 所有权协议
empty barrier 初态与 full / empty 独立 phase
tcgen05.commit 将异步完成事件转换为 mbarrier arrival
chapter_async_barriers 完成
静态 persistent scheduler 的 launch tail
worker 延迟启动与 tile 成本不均衡
CTA launch queue 与软件工作队列的区别
CLC 取消 pending launch 并接管 coordinate 的基本模型
clusterlaunchcontrol.try_cancel.async 的 16-byte response
CLC request 的 mbarrier arrival 与 complete-tx 完成条件
clusterlaunchcontrol.query_cancel 的 is_canceled 与 get_first_ctaid
多个 thread 提交 CLC request 时的 response 与 barrier 计数
CLC response 的 async-proxy 写入与 generic-proxy 读取
generic proxy 与 async proxy 的抽象含义
CLC response 的跨 proxy 读写顺序
mbarrier 完成通知与 proxy fence 的分工
在计算当前 tile 前提交 try_cancel
用当前 tile 计算隐藏 grid scheduler 延迟
先请求、再计算、最后等待的单请求软件流水
CLC 单 outstanding request 的 buffer / barrier 复用约束
静态 scheduler 在稳定、均匀、短 tile 场景中的优势
CLC 在 worker 启动与 tile 成本不确定时的收益
CLC 的 request / barrier / fence / query 额外开销
CLC 的 sm_100+ 硬件与编译目标边界
H100/H200 不支持 CLC，但支持 thread block cluster
把 CLC 封装为动态 tile scheduler，并保持 mainloop 不变
chapter_clc 完成
TIRx 是使用 threads、SMEM、TMEM、barrier 与 Tensor Core 概念的 Python DSL
Scope 决定哪些 threads 执行 tile 操作
Layout 决定逻辑 tile 如何映射到 memory、Lane、register 或 TMEM
Dispatch 决定 tile 操作使用哪条硬件路径
单 tile GEMM 计算 D = A × B^T
A/B 数据路径：GMEM -> SMEM -> tcgen05.mma
D 数据路径：tcgen05.mma -> TMEM -> registers -> GMEM
cta_id / warpgroup_id / warp_id_in_wg / lane_id
SMEMPool 的 alloc / move_base_to / commit
mbarrier.init 与 tcgen05.alloc
TileLayout 将逻辑 tile 映射到 TLane / TCol
Tx.cta.copy、Tx.gemm_async 与 Tx.wg.copy_async
tcgen05.commit 与 mbarrier.try_wait
tid_in_wg 将 output row 映射到 thread
tcgen05.wait.ld 与 TMEM -> register writeback
relinquish_alloc_permit 与 tcgen05.dealloc
chapter_intro_tirx 第一个单 tile GEMM Kernel
tvm.compile、IRModule 与 tir_pipeline="tirx"
LowerTIRx 展开 tile-level primitives
检查 TIRx script 与最终 CUDA source
编译后 Executable 直接接收 PyTorch tensors
用 fp32 PyTorch 参考值验证 fp16 kernel 输出
rtol / atol 逐元素误差判断
编译、执行、数值错误的三层排查
chapter_intro_tirx 完成
TileLayout 可以组合 S[...]、R[...] 与固定 offset
S[...] 描述依赖逻辑索引的基础物理映射
R[...] 枚举与逻辑索引无关的额外物理副本
固定 offset 整体平移所有物理坐标但不增加副本
apply() 只返回 D(x) + O，不枚举 replica
用 (1, 3)、shape [8, 16] 完整追踪 flatten、decompose 与坐标合成
R[2 : 4@warpid] 为示例元素生成 warpid 5 和 9 两个位置
chapter_tirx_layout_api 第一个知识点完成：S[...]、R[...] 与 offset
命名轴 laneid / warpid / m / TLane / TCol 的坐标空间语义
laneid 是 warp 内线程坐标，TLane 是 TMEM 存储 Lane 坐标
m 的单位和含义由 buffer scope 决定
TCol 以 buffer element 为单位，dtype 决定 hardware Col 打包
同一个 axis 的多个贡献相加，不同 axis 的相同数值不能合并
chapter_tirx_layout_api 第二个知识点完成：命名轴
apply() 的 logical + shape、linear、shard coordinate 三种入口
logical shape 决定 flatten，shard extents 决定 decompose
TileLayout 不保存 logical shape，同一个 (1, 3) 在 [8, 16] 与 [16, 8] 下映射不同
单参数 apply(coord) 按 linear coordinate 处理
用三入口等价性分层定位 flatten、decompose 与 stride / axis 错误
chapter_tirx_layout_api 第三个知识点完成：apply() 输入形式
(2, 128, 112) accumulator layout 的两个 128 x 112 TMEM 区域
TLane 承载 128 个逻辑 M rows，TCol 覆盖 [0, 224)
TCol = 112 * a + col，非 2 的幂 extent 112 无需补齐
layout 只描述 TMEM 坐标，不负责 allocation、MMA 或 tcgen05.ld
chapter_tirx_layout_api 第四个知识点完成：TMEM accumulator layout
scale-factor atom 的 R[4 : 32@TLane] 在 TLane 上生成四份副本
副本偏移为 0、32、64、96，q=0 就是 base 本身
同一个 scale factor 位于四个 32-lane TMEM partition 的本地窗口
8-bit logical TCol s 打包到 hardware TCol s//4 和 byte s%4
layout.apply() 只返回 base，layout.replica 描述四份副本
chapter_tirx_layout_api 第五个知识点完成：scale-factor replication
tmem_datapath_layout 根据 datapath 返回 tcgen05.mma 的 TMEM row mapping
datapath=D 使用 rows=128，恒等映射 TLane=row
datapath=F 使用 rows=64，四个 16-row slab 分散到 Lane 0/32/64/96
F 的公式 TLane=32*(row//16)+row%16，不复制逻辑数据
F 只使用 64 条 active Lane，span TLane=112 来自最高 Lane 111
chapter_tirx_layout_api 第六个知识点完成：tmem_datapath_layout D/F
tcgen05_atom_layout 将 TMEM fragment 的搬运形状映射到线程寄存器
instr_shape 是 lane x bits 的 atom，tensor_shape 和 dtype 推导 .xN
16x128b 的 fp32 K=128 与 fp16 K=256 都使用 .x32
fp16 的两个相邻元素打包进一个 32-bit register 的低半和高半
chapter_tirx_layout_api 第七个知识点完成：tcgen05_atom_layout
wg_local_layout 把逻辑 row 映射到 tid_in_wg，把同一行的 columns 映射到局部 m
默认 rows=128 时，完整 warpgroup 的每个 thread 负责一行
warp 0/1/2/3 分别负责 rows 0..31 / 32..63 / 64..95 / 96..127
wg_local_layout(rows=128).shard 与 32x32b fp32 fragment 的 shard 相同
wg_local_layout 不校验 atom、不处理 dtype packing，也不执行 allocation 或 copy
rows=64 时只生成 tid_in_wg 0..63，剩下 64 个线程没有该 layout 的数据坐标
chapter_tirx_layout_api 第八个知识点完成：wg_local_layout
ComposeLayout 将 affine TileLayout 与 XOR swizzle 组合起来
M/B/S 是 bit count，不是 bytes
(8,64) fp16 的 128B swizzle 使用 M=B=S=3
low 不参与 XOR，保证连续 16-byte vector 保持成组
x=m>>M，source=(x>>S)&mask，x2=x^source，addr=(x2<<M)|low
(8,64)、i=5、j=0 的逐位路径为 320 -> 40 -> 5 -> 45 -> 360
同一路径的 bank 从 0 变为 20
j=0 时八行地址为 72*i，bank 为 0,4,8,...,28
无 swizzle 时八行都落 bank 0
chapter_tirx_layout_api 第九个知识点完成：ComposeLayout 与 shared-memory swizzle
```

## 下一知识点

```text
chapter_tirx_layout_api 完成
-> chapter_gemm_basics
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
```
