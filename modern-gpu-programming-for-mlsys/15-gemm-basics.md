# 构建 Tiled GEMM：从单 Tile Baseline 开始

这篇笔记开始学习 `chapter_gemm_basics`。这一章不从“高性能 GEMM”直接
起步，而是先建立一个容易验证正确性的 baseline，再逐步加入 K 维累加、
空间 tiling、异步搬运和流水线。

本次先把第 1 步讲完整：

```text
一个 CTA
-> 一个 128 x 128 输出 tile
-> 一个 K=64 tile
-> 一次 GMEM -> SMEM -> TMEM -> registers -> GMEM 数据路径
```

## 本次讲解位置

```text
本次讲解位置
章节：chapter_gemm_basics
小节：第 1 步：顺序执行的单 Tile GEMM
知识点：GEMM 约定、Blackwell 数据路径与 correctness baseline
上次：chapter_tirx_layout_api -> ComposeLayout 与 shared-memory swizzle
下次：第 2 步：K-Loop 累加与 MMA barrier phase
PTX：tcgen05.alloc、tcgen05.mma、tcgen05.commit、tcgen05.ld、mbarrier.try_wait
```

前两章已经分别解释了 TIRx 的 Scope / Layout / Dispatch，以及 tile layout
如何描述 SMEM、TMEM 和 registers。这一章第一次把这些概念组织成可以逐步
扩展的 GEMM 主线。

### 本章知识点清单

```text
[x] GEMM 的 shape 约定与 Blackwell 数据路径
[x] 第 1 步：顺序执行的单 Tile GEMM
[ ] 第 2 步：K-Loop 累加
[ ] 第 3 步：空间 Tiling（Multi-CTA）
```

## 一、这一章为什么要先做慢版本

GEMM 是最常见的计算核心之一。Linear layer、attention projection 和不少
convolution 都可以落到矩阵乘法上。但一个高性能 GEMM kernel 同时包含：

```text
多级存储搬运
K 维分块累加
M/N 维 tiling
Tensor Core 异步计算
barrier 与 pipeline
CTA / cluster 协作
```

如果一开始把这些机制全部放进同一个 kernel，一旦结果错误，很难判断问题
来自数据布局、同步、累加还是 tile 索引。

因此课程采用的演进方式是：

| 步骤 | 新增机制 | 解决的问题 |
|---|---|---|
| 第 1 步 | 单个 `128 x 128 x 64` tile | 建立正确性 baseline |
| 第 2 步 | K-loop | 允许 `K > 64` 并累加 partial sums |
| 第 3 步 | M/N 二维 grid | 用多个 CTAs 覆盖完整输出矩阵 |

这不是性能路线本身，而是让每一步只有一个主要变量变化。后面才会继续加入
TMA、software pipeline、persistent scheduling、warp specialization 和
CTA cluster。

## 二、先固定 GEMM 的数学约定

本章统一使用：

```text
A: (M, K)
B: (N, K)
D: (M, N)

D[m, n] = sum_k A[m, k] * B[n, k]
```

从矩阵形式看，这等价于：

```text
D = A * B^T
```

Kernel 不会显式构造 `B^T`。它直接按 `(N, K)` 读取 B，让 MMA operand
layout 把 `B[n, k]` 解释成右操作数的对应元素。

本次使用：

```text
M = 128
N = 128
K = 64

BLK_M = 128
BLK_N = 128
BLK_K = 64
```

完整输出矩阵只有一个 tile，所以 grid shape 是：

```text
[M / BLK_M, N / BLK_N]
= [128 / 128, 128 / 128]
= [1, 1]
```

当前 CTA 的 tile 起点是：

```text
m_st = bx * 128 = 0
n_st = by * 128 = 0
```

### 一个具体输出元素

考虑：

```text
D[73, 91]
```

它的数学含义是：

```text
D[73, 91]
= sum(k = 0..63) A[73, k] * B[91, k]
```

如果取一个便于手算的例子：

```text
A[73, k] = 1
B[91, k] = k
```

那么：

```text
D[73, 91]
= sum(k = 0..63) k
= 63 * 64 / 2
= 2016
```

这个数不是随机输入下的真实结果，而是一个具体的算术追踪。它能帮助检查
K 维分块的时候，四个 chunk 是否都加到了正确位置。

## 三、Blackwell GEMM 的四段数据路径

本节的 kernel 按下面的路径执行一次：

```text
A/B: GMEM -> SMEM -> tcgen05.mma
D:   tcgen05.mma -> TMEM -> registers -> cast fp16 -> GMEM
```

结构图是：

```text
        GMEM
      A      B
      |      |
      v      v
    Asmem  Bsmem
       \    /
        \  /
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

四个阶段分别是：

| 阶段 | 数据移动 | 执行范围 | 关键同步 |
|---|---|---|---|
| 分配 | 准备 SMEM 与 TMEM | warp 0 / CTA | `cta_sync` 发布初始化结果 |
| Load | GMEM -> SMEM | CTA 的 128 个 threads | `cta_sync` |
| MMA | SMEM -> TMEM | 1 个 elected thread 发起 | `tcgen05.commit` + `mbarrier.try_wait` |
| Epilogue | TMEM -> registers -> GMEM | 一个 warpgroup | `tcgen05.wait.ld` |

最容易记错的两点是：

```text
SMEM 是 A/B 的 operand staging area，不是 D 的最终缓存。
TMEM 是 tcgen05.mma 的 accumulator 位置，不是用户直接可见的输出。
```

## 四、完整可运行代码

下面给出两个文件。放在同一个目录中即可：

```text
tirx_gemm_basics.py
verify_tirx_gemm_basics.py
```

### 文件一：`tirx_gemm_basics.py`

```python
import tvm
from tvm.script import tirx as T
from tvm.script.tirx import tile as Tx
from tvm.backend.cuda.tile_primitive.tma_utils import (
    mma_shared_layout,
    SwizzleMode,
)
from tvm.tirx.layout import TileLayout, S, TLane, TCol, tid_in_wg


def hgemm_v1(M, N, K):
    a_type = tvm.DataType("float16")
    b_type = tvm.DataType("float16")
    d_type = tvm.DataType("float16")
    acc_type = tvm.DataType("float32")

    BLK_M, BLK_N, BLK_K = 128, 128, 64

    A_layout = mma_shared_layout(
        a_type,
        SwizzleMode.SWIZZLE_128B_ATOM,
        (BLK_M, BLK_K),
    )
    B_layout = mma_shared_layout(
        b_type,
        SwizzleMode.SWIZZLE_128B_ATOM,
        (BLK_N, BLK_K),
    )

    @T.prim_func
    def kernel(
        A: T.Buffer((M, K), a_type),
        B: T.Buffer((N, K), b_type),
        D: T.Buffer((M, N), d_type),
    ):
        T.device_entry()

        bx, by = T.cta_id([M // BLK_M, N // BLK_N])
        wg_id = T.warpgroup_id([1])
        warp_id = T.warp_id_in_wg([4])
        lane_id = T.lane_id([32])

        # 前 1024 bytes 留给控制数据，之后分配 A/B operand tiles。
        pool = T.SMEMPool()
        tmem_addr = pool.alloc((1,), "uint32")
        mma_bar = pool.alloc((1,), "uint64", align=8)
        pool.move_base_to(1024)
        Asmem = pool.alloc(
            (BLK_M, BLK_K),
            a_type,
            layout=A_layout,
        )
        Bsmem = pool.alloc(
            (BLK_N, BLK_K),
            b_type,
            layout=B_layout,
        )
        pool.commit()

        # warp 0 初始化 mbarrier，并申请 512 columns 的 TMEM。
        if warp_id == 0:
            if lane_id == 0:
                T.ptx.mbarrier.init(mma_bar.ptr_to([0]), 1)
            T.ptx.tcgen05.alloc(
                T.address_of(tmem_addr),
                n_cols=512,
                cta_group=1,
            )

        T.ptx.fence.proxy_async("shared::cta")
        T.ptx.fence.mbarrier_init()
        T.cuda.cta_sync()

        # 把 tcgen05.alloc 返回的 TMEM address 解释成 128 x 512 的 buffer。
        tmem = T.decl_buffer(
            (128, 512),
            "float32",
            scope="tmem",
            allocated_addr=tmem_addr[0],
            layout=TileLayout(
                S[(128, 512) : (1 @ TLane, 1 @ TCol)]
            ),
        )

        m_st = T.meta_var(bx * BLK_M)
        n_st = T.meta_var(by * BLK_N)
        phase_mma: T.int32 = 0

        # CTA 的 128 个 threads 协作把 A/B 从 GMEM 搬到 SMEM。
        Tx.cta.copy(
            Asmem[:, :],
            A[m_st : m_st + BLK_M, :],
        )
        Tx.cta.copy(
            Bsmem[:, :],
            B[n_st : n_st + BLK_N, :],
        )
        T.cuda.cta_sync()

        # 只由一个 elected thread 发出 tile-level MMA。
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
                T.ptx.tcgen05.commit(
                    mma_bar.ptr_to([0]),
                    cta_group=1,
                )

        T.ptx.mbarrier.try_wait(
            mma_bar.ptr_to([0]),
            phase_mma,
        )

        # 每个 thread hold 一整行，所以用 tid_in_wg 建立 warpgroup view。
        Dreg = T.alloc_local((BLK_N,), acc_type)
        Dreg_f16 = T.alloc_local((BLK_N,), d_type)
        Dreg_wg = Dreg.view(
            128,
            BLK_N,
            layout=TileLayout(
                S[(128, BLK_N) : (1 @ tid_in_wg, 1)]
            ),
        )

        Tx.wg.copy_async(
            Dreg_wg[:, :],
            tmem[:, :BLK_N],
        )
        T.ptx.tcgen05.wait.ld()

        Tx.cast(Dreg_f16[:], Dreg[:])

        m_thr = T.meta_var(
            m_st + warp_id * 32 + lane_id
        )
        Tx.copy(
            D[m_thr, n_st : n_st + BLK_N],
            Dreg_f16[:],
        )

        T.cuda.cta_sync()
        if warp_id == 0:
            T.ptx.tcgen05.relinquish_alloc_permit(
                cta_group=1
            )
            T.ptx.tcgen05.dealloc(
                tmem_addr[0],
                n_cols=512,
                cta_group=1,
            )

    return kernel
```

### 文件二：`verify_tirx_gemm_basics.py`

```python
import torch
import tvm

from tirx_gemm_basics import hgemm_v1


target = tvm.target.Target("cuda")
device = torch.device("cuda")

M, N, K = 128, 128, 64
kernel = hgemm_v1(M, N, K)

with target:
    ex = tvm.compile(
        tvm.IRModule({"main": kernel}),
        target=target,
        tir_pipeline="tirx",
    )

A_tensor = torch.randn(
    M,
    K,
    dtype=torch.float16,
    device=device,
)
B_tensor = torch.randn(
    N,
    K,
    dtype=torch.float16,
    device=device,
)
D_tensor = torch.zeros(
    M,
    N,
    dtype=torch.float16,
    device=device,
)

ex.mod(A_tensor, B_tensor, D_tensor)

D_ref = (
    A_tensor.float() @ B_tensor.float().T
).half()

max_err = float((D_tensor - D_ref).abs().max())
print(f"Max error vs torch reference: {max_err:.6f}")

torch.testing.assert_close(
    D_tensor,
    D_ref,
    rtol=2e-2,
    atol=1e-2,
)
print("PASS")
```

### 运行方式

在装有 Blackwell GPU、TIRx compiler 和 CUDA 版 PyTorch 的环境中：

```bash
python verify_tirx_gemm_basics.py
```

预期输出类似：

```text
Max error vs torch reference: 0.000000
PASS
```

实际误差通常不是严格为零，因为 MMA 的排列和 fp32 累加顺序可能与 PyTorch
参考实现不同。这里使用 `rtol=2e-2`、`atol=1e-2`，判断的是数值是否落在
可接受范围内，而不是要求逐 bit 相同。

## 五、代码逐段映射

### 1. thread 坐标

```python
bx, by = T.cta_id([M // BLK_M, N // BLK_N])
wg_id = T.warpgroup_id([1])
warp_id = T.warp_id_in_wg([4])
lane_id = T.lane_id([32])
```

本次 grid 为 `1 x 1`，所以：

```text
bx = 0
by = 0
```

一个 CTA 含一个 warpgroup：

```text
wg_id = 0
warp_id = 0..3
lane_id = 0..31
```

因此 CTA 中共有：

```text
4 warps * 32 lanes = 128 threads
```

### 2. SMEM allocation

```python
pool = T.SMEMPool()
tmem_addr = pool.alloc((1,), "uint32")
mma_bar = pool.alloc((1,), "uint64", align=8)
pool.move_base_to(1024)
Asmem = pool.alloc((BLK_M, BLK_K), a_type, layout=A_layout)
Bsmem = pool.alloc((BLK_N, BLK_K), b_type, layout=B_layout)
pool.commit()
```

`tmem_addr` 和 `mma_bar` 是小块控制数据。`move_base_to(1024)` 把后续
operand allocation 移到 byte offset 1024，避免和前面区域重叠。

`A_layout` 和 `B_layout` 都是 128-byte swizzle shared-memory layout。
Load 会按这个物理布局写 SMEM，后面的 `tcgen05.mma` 也按相同布局读取。
如果两边不一致，地址本身可能合法，但 Tensor Core 会读到错位的元素。

### 3. mbarrier 与 TMEM 初始化

```python
if warp_id == 0:
    if lane_id == 0:
        T.ptx.mbarrier.init(mma_bar.ptr_to([0]), 1)
    T.ptx.tcgen05.alloc(
        T.address_of(tmem_addr),
        n_cols=512,
        cta_group=1,
    )
```

`mbarrier.init(..., 1)` 表示只需要一次完成 arrival。本轮 arrival 来自
`tcgen05.commit` 关联的 MMA 完成事件。

TMEM allocation 是 CTA-level 操作，但由 warp 0 执行。当前分配 512 个
TMEM columns。

初始化后先执行：

```python
T.ptx.fence.proxy_async("shared::cta")
T.ptx.fence.mbarrier_init()
T.cuda.cta_sync()
```

第一项发布 generic proxy 对 shared memory 的写入，第二项让 barrier
初始化对其他 threads 可见，最后让整个 CTA 在继续执行前对齐。

### 4. 把 TMEM address 变成 tile

```python
tmem = T.decl_buffer(
    (128, 512),
    "float32",
    scope="tmem",
    allocated_addr=tmem_addr[0],
    layout=TileLayout(
        S[(128, 512) : (1 @ TLane, 1 @ TCol)]
    ),
)
```

Layout 的含义是：

```text
logical row -> TLane
logical col -> TCol
```

本次 MMA 写满前 128 个 columns：

```text
tmem[:, :128]
```

后面 epilogue 也从这 128 个 columns 读取。

### 5. GMEM 到 SMEM

```python
Tx.cta.copy(Asmem[:, :], A[m_st : m_st + BLK_M, :])
Tx.cta.copy(Bsmem[:, :], B[n_st : n_st + BLK_N, :])
T.cuda.cta_sync()
```

`Tx.cta.copy` 是 CTA scope 的同步 copy。128 个 threads 共同完成 tile
搬运，但每个 thread 只负责其中的一部分。

`cta_sync()` 同时完成两件事：

```text
wait：等所有 threads 完成 copy
visibility：让 SMEM 写入对后续 MMA 可见
```

如果删除这次同步，MMA 可能读到不完整的 A/B tile。症状通常是少量行或列
不稳定，且结果可能随 launch 改变。

### 6. 发出 MMA

```python
if warp_id == 0:
    if T.ptx.elect_sync():
        Tx.gemm_async(...)
        T.ptx.tcgen05.commit(...)
```

外层 `if` 只保留 warp 0，`elect_sync()` 再从 warp 0 中选出真实发起
指令的一个 lane。

这里只有一个 thread issue，但 MMA 不是“单线程计算”。硬件会按 operand
layout 和 accumulator layout 对整个 tile 执行 Tensor Core 计算。如果
128 个 threads 都发同一条 MMA，反而会重复启动。

`Tx.gemm_async` 是 tile-level operation。当前 K=64，而一条底层 MMA
处理 K=16，所以它会 lower 成 4 条 MMA：

```text
k = 0..15
k = 16..31
k = 32..47
k = 48..63
```

`accum=False` 表示这次 tile operation 不读取更早一次 tile operation
留下的 accumulator。它会在本条 tile operation 的 lowering 内部建立第一份
partial sum，后续 K chunks 继续累加到同一个 TMEM 位置。

`tcgen05.mma` 是异步的。`tcgen05.commit` 把这次 MMA 的完成事件关联到
`mma_bar`：

```python
T.ptx.tcgen05.commit(mma_bar.ptr_to([0]), cta_group=1)
```

随后整个 warpgroup 执行：

```python
T.ptx.mbarrier.try_wait(mma_bar.ptr_to([0]), phase_mma)
```

只有等到 barrier 离开指定 phase，kernel 才能安全读取 TMEM。

### 7. TMEM 到 registers

```python
Dreg = T.alloc_local((BLK_N,), acc_type)
Dreg_f16 = T.alloc_local((BLK_N,), d_type)
Dreg_wg = Dreg.view(
    128,
    BLK_N,
    layout=TileLayout(
        S[(128, BLK_N) : (1 @ tid_in_wg, 1)]
    ),
)
```

`Dreg` 和 `Dreg_f16` 都是每个 thread 私有的 `BLK_N` 元素数组。
`Dreg_wg` 把同一组 registers 解释成一个 warpgroup-wide tile：

```text
logical row -> tid_in_wg
logical col -> 本 thread 的 register index
```

因为 tile 有 128 rows，warpgroup 也有 128 threads，所以映射正好是一行一个
thread：

```text
thread 0  -> row 0
thread 1  -> row 1
...
thread 127 -> row 127
```

读取是异步的：

```python
Tx.wg.copy_async(Dreg_wg[:, :], tmem[:, :BLK_N])
T.ptx.tcgen05.wait.ld()
```

如果没等 `wait.ld()` 就读取 `Dreg`，可能读到旧 register 内容。

### 8. cast 与写回

```python
Tx.cast(Dreg_f16[:], Dreg[:])
m_thr = T.meta_var(m_st + warp_id * 32 + lane_id)
Tx.copy(D[m_thr, n_st : n_st + BLK_N], Dreg_f16[:])
```

TMEM accumulator 是 fp32，输出 D 是 fp16，因此先在 registers 中完成
cast，再写回 GMEM。

`m_thr` 的计算把 warpgroup 内线程还原为全局输出行：

```text
warp 0: rows 0..31
warp 1: rows 32..63
warp 2: rows 64..95
warp 3: rows 96..127
```

### 9. 释放 TMEM

```python
T.cuda.cta_sync()
if warp_id == 0:
    T.ptx.tcgen05.relinquish_alloc_permit(cta_group=1)
    T.ptx.tcgen05.dealloc(tmem_addr[0], n_cols=512, cta_group=1)
```

先同步，确保没有任何 consumer 还在读取 TMEM，再释放 allocation permit
和 TMEM columns。

## 六、完整数值执行追踪

现在追踪：

```text
D[73, 91]
```

### 1. 计算属于哪个 TMEM 位置

MMA accumulator layout 是：

```text
TLane = row
TCol  = col
```

所以：

```text
D[73, 91]
-> TLane 73
-> TCol 91
```

### 2. 计算由哪个 thread 读取

`tid_in_wg` 的映射为：

```text
tid_in_wg = 73
```

拆成 warp 和 lane：

```text
warp = 73 // 32 = 2
lane = 73 % 32 = 9
```

### 3. 计算 register 和 GMEM 坐标

这个 thread 持有 row 73 的完整 register row，因此 `D[73, 91]` 位于：

```text
Dreg[91]
```

写回时：

```text
m_thr = m_st + warp_id * 32 + lane_id
      = 0 + 2 * 32 + 9
      = 73
```

于是最终执行：

```text
D[73, 0:128] <- Dreg_f16[0:128]
```

其中列 91 的值就是：

```text
sum(k = 0..63) A[73, k] * B[91, k]
```

### 4. K=64 如何分成四段累加

由于底层 MMA 的 K atom 是 16：

```text
64 / 16 = 4
```

使用示例 `A[73,k]=1`、`B[91,k]=k`：

| K chunk | 元素范围 | partial sum | 累计结果 |
|---|---:|---:|---:|
| chunk 0 | `k=0..15` | `120` | `120` |
| chunk 1 | `k=16..31` | `376` | `496` |
| chunk 2 | `k=32..47` | `632` | `1128` |
| chunk 3 | `k=48..63` | `888` | `2016` |

四次累计都写在同一个 TMEM accumulator 位置：

```text
TLane = 73
TCol  = 91
```

最终结果再由 thread 73 读入 `Dreg[91]`。

## 七、这一步完成了什么，还没有完成什么

已经完成：

```text
完整的 GMEM -> SMEM -> TMEM -> RF -> GMEM 数据路径
tcgen05.mma 的 elected-thread issue
tcgen05.commit 与 mbarrier 完成等待
TMEM -> register 的 warpgroup readback
fp32 accumulator -> fp16 output
TMEM allocation 与 deallocation
```

还没有完成：

```text
K > 64 的分块累加
M > 128 或 N > 128 的空间 tiling
第二个 K chunk 复用 mma_bar 时的 phase 管理
TMA 异步搬运
搬运与 MMA 的 software pipeline
persistent scheduling
warp specialization
CTA cluster
性能调优
```

这一步像一条完整的“最小闭环”。后续每一版都只替换或扩展其中一段，
数学形式和最终数据路径保持不变。

## 八、常见错误与可观察症状

| 错误 | 可观察症状 | 根因 |
|---|---|---|
| Load 后漏掉 `cta_sync()` | 少量元素错误且可能随运行变化 | MMA 读到未完成的 SMEM |
| 所有 threads 都发 MMA | duplicate issue、结果异常或性能骤降 | MMA issue scope 错误 |
| 漏掉 `tcgen05.commit` | wait 无法得到本轮完成通知，可能挂起 | MMA 与 barrier 未关联 |
| 漏掉 `mbarrier.try_wait` | epilogue 读取未完成的 TMEM | 缺少完成同步 |
| 漏掉 `tcgen05.wait.ld()` | `Dreg` 是旧值或部分旧值 | TMEM load 仍然 pending |
| `Dreg_wg` layout 与 `m_thr` 不一致 | 整行互换或有规律的行错位 | producer / consumer row ownership 不一致 |
| 把 Step 1 直接用于 `K > 64` | copy shape 不匹配或结果只含一部分 K | baseline 只支持一个 K tile |
| 在 epilogue 前提前 dealloc TMEM | invalid access 或不稳定结果 | TMEM lifetime 未覆盖最后 consumer |

## 九、验证命令和预期输出

### 环境要求

课程 kernel 的目标是 Blackwell `sm_100a`。需要：

```text
Blackwell GPU，例如 B200
TIRx compiler
CUDA 版 PyTorch
```

课程给出的最低验证流程是：

```bash
python verify_tirx_gemm_basics.py
```

预期输出：

```text
Max error vs torch reference: ...
PASS
```

### 当前学习机器的执行边界

当前机器是 Apple M5 Pro，没有 CUDA，也没有 Blackwell GPU。因此本机只能
完成：

```text
静态阅读完整源码
检查 kernel 的 scope、layout、dispatch 与同步关系
检查 Python 语法
```

本机不能完成：

```text
编译并运行 sm_100a kernel
执行 tcgen05.mma
测量 TFLOPS
验证真实 GPU 数值
```

可以做不依赖 GPU 的语法检查：

```bash
python -m py_compile \
  tirx_gemm_basics.py \
  verify_tirx_gemm_basics.py
```

预期没有输出，并且命令退出码为 0。

## 十、自测题

### 1. 本章为什么先实现一个只能计算单个 `128 x 128 x 64` tile 的版本？

答：为了建立正确性 baseline。它只包含一次完整数据路径，没有 K-loop、
multi-CTA、TMA 或 pipeline。后续版本每加入一项机制，都可以和它对照，
从而把错误定位到新增机制，而不是在多个变量中猜测。

### 2. A、B、D 的 shape 和数学关系分别是什么？

答：

```text
A: (M, K)
B: (N, K)
D: (M, N)
D[m, n] = sum_k A[m, k] * B[n, k]
```

矩阵形式是 `D = A * B^T`，但 kernel 直接读取 `B[n,k]`，不显式构造
`B^T`。

### 3. 这个 kernel 中 load、MMA issue 和 writeback 分别由谁执行？

答：

```text
Tx.cta.copy:       CTA 的 128 个 threads
Tx.gemm_async:     1 个 elected thread
Tx.wg.copy_async:  一个完整 warpgroup
```

### 4. `D[73, 91]` 最终如何到达 thread 73 的 register？

答：

```text
D[73, 91]
-> TMEM (TLane=73, TCol=91)
-> tid_in_wg=73
-> warp=73//32=2
-> lane=73%32=9
-> Dreg[91]
```

因为 `m_thr = 2*32+9 = 73`，写回时会写回 `D[73,0:128]`。

### 5. K=64 的 MMA 为什么要分成四段，四段累加到哪里？

答：底层 MMA 的 K atom 是 16，所以：

```text
64 / 16 = 4
```

四个 K chunks 都累加到同一个 TMEM accumulator 位置
`(TLane=73, TCol=91)`。这一步还没有 device-side K-loop；K 的拆分发生在
一次 `Tx.gemm_async` tile operation 的 lowering 内部。

## 十一、当前进度

`chapter_gemm_basics` 的知识点：

```text
[x] GEMM 的 shape 约定与 Blackwell 数据路径
[x] 第 1 步：顺序执行的单 Tile GEMM
[ ] 第 2 步：K-Loop 累加
[ ] 第 3 步：空间 Tiling（Multi-CTA）
```

本章第 1 步已经覆盖：

```text
GEMM 使用 A(M,K)、B(N,K)、D(M,N)，计算 D = A * B^T
单 tile baseline 取 M=N=128、K=64，grid shape 为 1x1
Blackwell 数据路径是 GMEM -> SMEM -> TMEM -> registers -> GMEM
A/B 以 128-byte swizzled layout 存入 SMEM
Tx.cta.copy 由 CTA scope 的 threads 协作执行
Load 后的 cta_sync 同时负责完成等待和 SMEM 可见性
Tx.gemm_async 由 elected thread 发起
tcgen05.commit 把 MMA 完成事件关联到 mbarrier
mbarrier.try_wait 等到本轮 MMA 完成后才能读取 TMEM
K=64 的 tile operation 会 lower 成 4 个 K=16 的 MMA
accum=False 表示 tile operation 不继承更早的 TMEM accumulator
Dreg_wg 用 tid_in_wg 将 128 个输出 rows 映射到一个 warpgroup 的 128 threads
Tx.wg.copy_async 读取 TMEM，tcgen05.wait.ld 等待 register load 完成
每个 thread 将自己的 fp32 row cast 为 fp16，再写回 GMEM
TMEM 必须先 cta_sync，再 relinquish permit 和 dealloc
第 1 步的限制是 K<=64、M=N=128、同步 copy、搬运与计算不重叠
chapter_gemm_basics 第 1 个知识点完成：单 Tile Baseline
```

下一知识点：

```text
chapter_gemm_basics
-> 第 2 步：K-Loop 累加与 MMA barrier phase
```
