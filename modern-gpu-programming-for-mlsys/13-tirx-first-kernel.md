# TIRx 入门：第一个单 tile GEMM Kernel

本篇整理 `chapter_intro_tirx` 的第一个知识点。目标不是一次学完 TIRx
语法，而是先理解一个可运行 kernel 如何把 threads、SMEM、TMEM、
barrier、Tensor Core 和 copy 组织成同一条数据路径。

## 一、本次讲解位置

```text
本次讲解位置
章节：chapter_intro_tirx
小节：第一个 TIRx Kernel
知识点：单 tile GEMM 的数据路径，以及 Scope、Layout、Dispatch 三个核心概念
上次：chapter_clc 完成
下次：编译并验证 TIRx Kernel
PTX：tcgen05.mma、tcgen05.alloc、tcgen05.commit、mbarrier.try_wait
```

前面几章已经分别学过：

```text
数据如何映射到 Lane / Register / TMEM
TMA、mbarrier 和 tcgen05 如何异步执行
异步操作完成后如何把所有权交还给 thread
```

TIRx 现在要解决的是另一层问题：

```text
如何把这些硬件机制组织成一个结构化、可检查、可 lowering 的 kernel
```

## 二、这个 kernel 计算什么

第一个示例计算：

```text
D = A × B^T
```

Shape 为：

```text
A: 128 × 64
B: 128 × 64
D: 128 × 128
```

这里的 `B^T` 不是先显式生成转置矩阵。Kernel 直接把 B 按
`(N, K)` 读取，再由 MMA 的 operand layout 解释成矩阵乘法需要的
形状。

当前示例只计算一个 output tile：

```text
BLK_M = 128
BLK_N = 128
BLK_K = 64
```

所以 grid 只有一个 CTA：

```text
M // BLK_M = 1
N // BLK_N = 1
grid = 1 × 1
bx = 0
by = 0
```

这不是高性能 GEMM 的最终形态，而是后续 K-loop、多 output tile、
TMA 和 warp specialization 版本的起点。

## 三、完整数据路径

这个 kernel 的数据流可以压缩成：

```text
A/B: GMEM -> SMEM -> tcgen05.mma

D:   tcgen05.mma
     -> TMEM
     -> registers
     -> GMEM
```

展开到具体阶段：

| 阶段 | 数据移动 | 执行者 | 同步方式 |
|---|---|---|---|
| 分配 | 申请 SMEM 和 TMEM | CTA / warp 0 | `cta_sync` |
| Load | A/B: GMEM -> SMEM | 整个 CTA | `Tx.cta.copy` 后 `cta_sync` |
| Compute | SMEM -> Tensor Core -> TMEM | 1 个 elected thread 发起 | `tcgen05.commit` + mbarrier |
| Writeback | TMEM -> registers -> GMEM | 一个 warpgroup | `wait.ld` 后 copy |
| Release | 释放 TMEM | warp 0 | `cta_sync` 后 dealloc |

用一张更直观的图表示：

```text
          GMEM
        A    B
        |    |
        v    v
       SMEM  SMEM
          \  /
           \/
      tcgen05.mma
           |
           v
          TMEM
           |
           v
       registers
           |
           v
          GMEM
           D
```

最重要的观察是：

```text
SMEM 不是 A/B 的最终计算位置
TMEM 也不是 D 的最终用户可见位置
```

A/B 经过 SMEM 是为了让 Tensor Core 按矩阵 descriptor 读取正确布局。
D 先落在 TMEM，是因为 `tcgen05.mma` 的 accumulator 属于 TMEM，最后
再由 `tcgen05.ld` 搬进 registers，完成 cast 和 GMEM writeback。

## 四、TIRx 中一条 tile 操作描述什么

代码中最关键的三项 tile 操作是：

```python
Tx.cta.copy(Asmem[:, :], A[m_st:m_st + BLK_M, :])
Tx.cta.copy(Bsmem[:, :], B[n_st:n_st + BLK_N, :])

Tx.gemm_async(
    tmem[:, :BLK_N],
    Asmem[:, :],
    Bsmem[:, :],
    accum=False,
    dispatch="tcgen05",
    cta_group=1,
)

Tx.wg.copy_async(Dreg_wg[:, :], tmem[:, :BLK_N])
```

它们不是已经展开的 thread-level CUDA 代码，而是 tile-level 操作：

```text
对一整块 tile 描述“搬什么”或“计算什么”
再由编译器和 dispatch 决定“具体由哪些 threads、发哪些指令完成”
```

例如：

```text
Tx.gemm_async(...) 描述的是完整的 128 × 128 × 64 tile GEMM
```

课程代码说明，当前 `tcgen05.mma` 一次处理 16 个 K 元素，因此编译器
会沿 K 维生成：

```text
64 / 16 = 4
```

也就是 4 次 MMA。这里的 4 次拆分来自编译器的 lowering，不需要在
TIRx 源码中手工写一个 K-loop。

## 五、线程坐标系

Kernel 开头建立了几层 thread identity：

```python
bx, by = T.cta_id([M // BLK_M, N // BLK_N])
wg_id = T.warpgroup_id([1])
warp_id = T.warp_id_in_wg([4])
lane_id = T.lane_id([32])
```

它们分别回答：

| 表达式 | 含义 | 当前 kernel 的范围 |
|---|---|---|
| `T.cta_id([...])` | CTA 在 grid 中的坐标 | 1 × 1，只有 `(0, 0)` |
| `T.warpgroup_id([1])` | warpgroup 在 CTA 中的编号 | 只有 warpgroup 0 |
| `T.warp_id_in_wg([4])` | warp 在 warpgroup 中的编号 | 0, 1, 2, 3 |
| `T.lane_id([32])` | lane 在 warp 中的编号 | 0 到 31 |

因此这个只有一个 warpgroup 的 CTA 共有：

```text
4 warps × 32 lanes = 128 threads
```

后面每次判断“谁执行”时，都建立在这套坐标系上。

## 六、申请 SMEM 与 TMEM

### 6.1 SMEMPool

代码使用：

```python
pool = T.SMEMPool()
tmem_addr = pool.alloc((1,), "uint32")
mma_bar = pool.alloc((1,), "uint64", align=8)
pool.move_base_to(1024)
Asmem = pool.alloc((BLK_M, BLK_K), a_type, layout=A_layout)
Bsmem = pool.alloc((BLK_N, BLK_K), b_type, layout=B_layout)
pool.commit()
```

`T.SMEMPool` 是一个 CTA 内部的 shared-memory allocator。可以先把它
理解成一次显式规划：

```text
需要哪些 SMEM 对象
每个对象的 shape、dtype、alignment 和 layout 是什么
它们在共享内存区域中的相对位置如何安排
```

这里分配了三类对象：

| 对象 | 作用 |
|---|---|
| `tmem_addr` | 保存 `tcgen05.alloc` 返回的 TMEM 起始地址 |
| `mma_bar` | MMA 完成事件使用的 mbarrier |
| `Asmem` / `Bsmem` | A/B 在 SMEM 中的 tile |

`tmem_addr` 虽然是 TMEM 地址，但这个地址值本身保存在 SMEM 中，
因此它是 `(1,)` 的 `uint32` buffer。

`pool.move_base_to(1024)` 把后续 allocation 的基准位置移动到至少
1024 bytes 之后。当前代码先为前面的元数据留出空间，再放置 A/B
数据 tile。

最后的：

```python
pool.commit()
```

表示 allocation 规划结束。之后的 buffer 不再依赖 allocator 继续
增长布局。

### 6.2 为什么 A_layout / B_layout 很重要

```python
A_layout = mma_shared_layout(
    a_type,
    SwizzleMode.SWIZZLE_128B_ATOM,
    (BLK_M, BLK_K),
)
```

A/B 在 SMEM 中不是简单的 row-major 平铺，而是采用匹配 MMA operand
要求的 128-byte swizzle layout。

数据路径中的两端必须匹配：

```text
生产者: Tx.cta.copy(...) 按这个 layout 写 SMEM
消费者: tcgen05.mma 按 descriptor 读取 SMEM
```

如果写入时使用一种 swizzle，而 descriptor 按另一种方式解释地址，
矩阵元素会落到错误位置。表现通常不是“程序报越界”，而是结果数值
静静地变错。

## 七、初始化 mbarrier 与 TMEM

### 7.1 mbarrier

```python
if warp_id == 0:
    if lane_id == 0:
        T.ptx.mbarrier.init(mma_bar.ptr_to([0]), 1)
```

这里由 warp 0 的 lane 0 初始化 barrier，arrival count 为 1：

```text
只要 MMA 完成事件产生 1 次 arrival，phase 就可以完成
```

这个 barrier 后面与 `tcgen05.commit` 配对。

### 7.2 tcgen05.alloc

```python
T.ptx.tcgen05.alloc(
    T.address_of(tmem_addr),
    n_cols=512,
    cta_group=1,
)
```

`tcgen05.alloc` 申请 TMEM columns。这里请求 512 columns，
`cta_group=1` 表示 TMEM allocation 只属于当前 CTA。

注意区分：

```text
tmem_addr 是地址存储位置
tcgen05.alloc 写入的是实际 TMEM 起始地址
后面的 tmem decl_buffer 才把这个地址解释成二维 tile
```

### 7.3 fence 与 CTA-wide synchronization

```python
T.ptx.fence.proxy_async("shared::cta")
T.ptx.fence.mbarrier_init()
T.cuda.cta_sync()
```

三个动作解决不同的可见性问题：

| 操作 | 作用 |
|---|---|
| `fence.proxy_async("shared::cta")` | 让 generic proxy 对 shared memory 的写入对 async proxy 可见 |
| `fence.mbarrier_init()` | 发布 barrier 初始化结果 |
| `cta_sync()` | 让 CTA 内所有 threads 等到初始化和分配完成 |

如果没有这些同步，其他 threads 可能在 barrier 尚未初始化，或者
`tcgen05.alloc` 尚未写入有效地址时就开始执行后续操作。

## 八、把 TMEM 地址解释成 tile

```python
tmem = T.decl_buffer(
    (128, 512),
    "float32",
    scope="tmem",
    allocated_addr=tmem_addr[0],
    layout=TileLayout(S[(128, 512) : (1@TLane, 1@TCol)]),
)
```

地址申请完成后，TIRx 需要知道这个地址上的 128 × 512 个 `float32`
元素如何映射到 TMEM：

```text
第 0 维: 1 @ TLane
第 1 维: 1 @ TCol
```

在当前最简单映射下：

```text
tmem[row, col] -> TMEM Lane = row
                 TMEM Column = col
```

也就是说，逻辑坐标：

```text
D[73, 91]
```

在这份 accumulator buffer 中对应：

```text
TMEM Lane = 73
TMEM Column = 91
```

这里的 `row=73` 不是“第 73 个 thread”或“第 73 个寄存器”，而是
TMEM 的逻辑 Lane 坐标。线程和 Lane 的对应关系要到 writeback 时
才由 `tid_in_wg` 建立。

## 九、Load：GMEM -> SMEM

```python
m_st = T.meta_var(bx * BLK_M)
n_st = T.meta_var(by * BLK_N)

Tx.cta.copy(Asmem[:, :], A[m_st:m_st + BLK_M, :])
Tx.cta.copy(Bsmem[:, :], B[n_st:n_st + BLK_N, :])
T.cuda.cta_sync()
```

这里作用域是 `cta`：

```text
整个 CTA 的 128 threads 协作完成 A/B 的 GMEM -> SMEM copy
```

`m_st` 和 `n_st` 是当前 CTA 对应的 output tile origin。当前 grid
只有一个 CTA，所以：

```text
m_st = 0
n_st = 0
```

后面的 `cta_sync()` 保证：

```text
只要 MMA 开始读取 SMEM
A/B tile 就已经由全部 threads 写完
```

当前版本使用普通 threads 搬运。后续引入 TMA 后，同一个
`Tx.cta.copy` 也可能 dispatch 到 TMA 路径，此时同步方式会换成
mbarrier。

## 十、Compute：由 elected thread 发起 MMA

```python
if warp_id == 0:
    if T.ptx.elect_sync():
        Tx.gemm_async(
            tmem[:, :BLK_N],
            Asmem[:, :],
            Bsmem[:, :],
            accum=False,
            dispatch="tcgen05",
            cta_group=1,
        )
        T.ptx.tcgen05.commit(mma_bar.ptr_to([0]), cta_group=1)
```

这段代码有三层“谁执行”的信息。

### 10.1 `warp_id == 0`

选择 warp 0 作为发起 warp。其他 warps 不进入这个分支。

### 10.2 `T.ptx.elect_sync()`

在选中的 warp 内，再选出实际发出指令的单个 thread。`elect_sync`
的结果不是“指定 lane 0”，而是由 warp 内部选出一个 active thread。

所以 MMA 的 scope 可以概括成：

```text
1 个 elected thread 发出 tcgen05.mma
```

这和普通 `mma.sync` 的“整个 warp 一起发”不是同一种 scope。一个
thread 发指令，并不表示只有这个 thread 的数据参与计算；Tensor Core
和 TMEM 会按更宽的硬件 scope 读取 operand 并写 accumulator。

### 10.3 dispatch

```python
dispatch="tcgen05"
cta_group=1
```

`dispatch="tcgen05"` 要求 compiler 选择 Blackwell 的 `tcgen05.mma`
路径。`cta_group=1` 则限定操作只使用当前 CTA 的 SMEM 和 TMEM。

`accum=False` 表示第一次 MMA 不累加已有 accumulator，而是直接写
新的结果。

### 10.4 commit 与 wait

```python
T.ptx.tcgen05.commit(mma_bar.ptr_to([0]), cta_group=1)
```

`tcgen05.commit` 把已经发出的异步 MMA 完成事件关联到
`mma_bar`。这不是由软件手工执行一次普通 barrier arrival，而是：

```text
MMA 完成
-> 硬件产生完成事件
-> 该事件在 mma_bar 上产生 arrival
```

随后：

```python
phase_mma: T.int32 = 0
T.ptx.mbarrier.try_wait(mma_bar.ptr_to([0]), phase_mma)
```

所有需要等待的 threads 检查 phase 0 是否完成。当前 barrier 的
initial phase 为 0，所以第一次等待传入 0。只要 MMA completion
产生约定的 arrival，`try_wait` 就能观察到 phase 完成。

这里把“issue”和“complete”明确分开了：

```text
elect_sync 中的 thread: 发出 MMA
全部需要结果执行者: 等待 mma_bar phase
```

## 十一、Writeback：TMEM -> registers -> GMEM

### 11.1 为每个 thread 建立 register view

```python
Dreg = T.alloc_local((BLK_N,), acc_type)
Dreg_f16 = T.alloc_local((BLK_N,), d_type)

Dreg_wg = Dreg.view(
    128,
    BLK_N,
    layout=TileLayout(S[(128, BLK_N) : (1@tid_in_wg, 1)]),
)
```

`Dreg` 是 per-thread local storage：

```text
每个 thread 有 128 个 float32 registers
```

`Dreg_wg` 把 128 个 thread 的 registers 组织成一个逻辑 tile：

```text
逻辑 row -> tid_in_wg
逻辑 col -> register index
```

因为 CTA 中有 128 threads，所以 128 行刚好一行分配给一个 thread：

| 逻辑 row | warp / lane | 对应 thread |
|---:|---|---|
| 0 | warp 0, lane 0 | `tid_in_wg = 0` |
| 31 | warp 0, lane 31 | `tid_in_wg = 31` |
| 32 | warp 1, lane 0 | `tid_in_wg = 32` |
| 73 | warp 2, lane 9 | `tid_in_wg = 73` |
| 127 | warp 3, lane 31 | `tid_in_wg = 127` |

仍以 `D[73, 91]` 为例：

```text
D[73, 91]
-> TMEM Lane 73, Column 91
-> warp 2, lane 9
-> 该 thread 的 Dreg[91]
```

这正好把前面 TMEM 的二维坐标和 writeback 的 thread/register 坐标
连接起来。

### 11.2 TMEM load 与 wait

```python
Tx.wg.copy_async(Dreg_wg[:, :], tmem[:, :BLK_N])
T.ptx.tcgen05.wait.ld()
```

`Tx.wg.copy_async` 的 scope 是 warpgroup：

```text
一个 warpgroup 的 128 threads 协作
把 TMEM accumulator 分配到各自的 registers
```

`tcgen05.wait.ld()` 等待此前由该 thread 发出的 `tcgen05.ld` 完成。
前面的 copy 是异步 issue，不等待就直接读 `Dreg` 会产生未定义结果。

### 11.3 cast 与写回 GMEM

```python
Tx.cast(Dreg_f16[:], Dreg[:])

m_thr = T.meta_var(m_st + warp_id * 32 + lane_id)
Tx.copy(D[m_thr, n_st : n_st + BLK_N], Dreg_f16[:])
```

公式：

```text
m_thr = m_st + warp_id × 32 + lane_id
```

把 thread 坐标映射成输出矩阵行：

```text
warp 0, lane 0   -> row 0
warp 2, lane 9   -> row 73
warp 3, lane 31  -> row 127
```

每个 thread 负责它那一行的 128 个 columns，因此一次 writeback
就写回完整的 `128 × 128` tile。

## 十二、释放 TMEM

```python
T.cuda.cta_sync()

if warp_id == 0:
    T.ptx.tcgen05.relinquish_alloc_permit(cta_group=1)
    T.ptx.tcgen05.dealloc(
        tmem_addr[0],
        n_cols=512,
        cta_group=1,
    )
```

`cta_sync()` 保证整个 CTA 已经完成 TMEM read，随后 warp 0：

```text
relinquish_alloc_permit: 放弃继续申请 TMEM allocation 的许可
tcgen05.dealloc: 释放此前申请的 512 columns
```

顺序不能反。只要还有 thread 正在读取 TMEM，就不能提前 dealloc。

## 十三、Scope、Layout、Dispatch

现在可以把整段 kernel 放回三个维度中理解。

| 维度 | 回答的问题 | 本 kernel 的例子 | 出错时的典型症状 |
|---|---|---|---|
| Scope | 谁执行这项操作？ | CTA copy、warpgroup TMEM load、elected thread MMA | 少发、多发、重复写或 wait 错对象 |
| Layout | 每个逻辑元素映射到哪里？ | SMEM swizzle、`TLane` / `TCol`、`tid_in_wg` | 数值整体错位、行/列交换、结果不一致 |
| Dispatch | 使用哪条硬件路径？ | `dispatch="tcgen05"` | 生成了不适用于目标架构的代码 |

三者之间不是互相独立的注释，而是共同决定 lowering 结果：

```text
Scope
    -> 生成哪些 thread-level 条件和 collective 操作

Layout
    -> 生成地址计算、Lane/register 映射和 descriptor 约束

Dispatch
    -> 选择 TMA、tcgen05 或其他硬件 primitive
```

可以把它们类比成三个必须同时回答的问题：

```text
Who?    Scope
Where?  Layout
How?    Dispatch
```

任何一项缺失，tile primitive 都无法唯一 lowers 成具体硬件指令。

## 十四、这个版本的简化点

当前 kernel 故意只保留最小结构，因此还没有：

```text
沿 K 维的多轮 accumulation
多个 output tiles
多个 stage 的 software pipeline
TMA load / store
producer / consumer warp specialization
cluster 与 cta_group::2
```

它使用一个 CTA、一次 K-tile、一次 GEMM primitive，因此很适合用来
建立 TIRx 的心智模型：

```text
tile-level primitive
-> 通过 scope / layout / dispatch 补充语义
-> compiler lowering
-> 具体硬件指令序列
```

## 十五、下一步

下一知识点会实际编译这段 kernel，并用 PyTorch 参考结果验证：

```text
D = A × B^T
```

需要记住环境边界：

```text
运行目标需要 Blackwell GPU，例如 B200，目标包括 sm_100a
依赖 apache-tvm==0.26.0 与 cuda-bindings
macOS 上没有 CUDA / Blackwell 硬件，只能做概念和静态代码分析
```

## 十六、当前进度

`chapter_intro_tirx` 的知识点：

```text
[x] 第一个 TIRx Kernel
[ ] 编译并验证结果
```

已经覆盖：

```text
TIRx 是使用硬件概念的 Python DSL
Scope / Layout / Dispatch 是 tile 操作的三项核心语义
单 tile GEMM 计算 D = A × B^T
A/B 数据路径：GMEM -> SMEM -> tcgen05.mma
D 数据路径：tcgen05.mma -> TMEM -> registers -> GMEM
T.cta_id / T.warpgroup_id / T.warp_id_in_wg / T.lane_id
T.SMEMPool 的 alloc / move_base_to / commit
mbarrier.init 与 tcgen05.alloc
fence.proxy_async / fence.mbarrier_init / cta_sync
TileLayout 将逻辑 tile 映射到 TLane / TCol
Tx.cta.copy 的 CTA scope
Tx.gemm_async 的 elected-thread issue scope
tcgen05.commit 将 MMA 完成事件关联到 mbarrier
mbarrier.try_wait 等待 phase 完成
Dreg_wg 使用 tid_in_wg 将 row 分配给 thread
Tx.wg.copy_async 搬 TMEM -> registers
tcgen05.wait.ld 等待异步 load 完成
Tx.cast 与 per-thread row writeback
relinquish_alloc_permit 与 tcgen05.dealloc
```

下一知识点：

```text
chapter_intro_tirx
-> 编译并验证结果
```
