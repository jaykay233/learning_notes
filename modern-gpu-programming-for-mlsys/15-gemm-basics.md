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

#### 2.1 `pool.commit()` 的精确含义

```text
本次讲解位置
章节：chapter_gemm_basics
小节：第 1 步：顺序执行的单 Tile GEMM
知识点：T.SMEMPool 的 alloc / move_base_to / commit
上次：mma_shared_layout 生成的 SWIZZLE_128B_ATOM
下次：mbarrier 初始化与 tcgen05.alloc 的 TMEM 生命周期
PTX：dynamic shared memory allocation、shared::cta
```

这里先建立一个心智模型：

```text
shared.dyn 是一整块连续内存
SMEMPool 是编译期的 bump allocator
alloc 负责“占位置并把 cursor 向后推”
move_base_to 负责“把 cursor 移到指定位置”
commit 负责“结束规划，报告整块 arena 的最终字节数”
```

因此，`pool.commit()` 不是运行时提交，也不是数据同步。它做的是：

```text
在 lowering 前冻结 allocator 的最终布局信息，
把 pool 的 high-water mark 写成 tirx.pool_max_bytes，
让后续 compiler 知道 shared.dyn 需要分配多少字节。
```

每个 API 的职责可以精确拆成：

| API | 执行阶段 | 作用 | 是否生成 GPU 指令 |
|---|---|---|---|
| `T.SMEMPool()` | Python 解析 / IR 构建 | 创建一个共享内存 bump allocator | 否 |
| `pool.alloc(...)` | Python 解析 / IR 构建 | 对齐 cursor、创建 buffer view、推进 cursor | 否 |
| `pool.move_base_to(...)` | Python 解析 / IR 构建 | 把 cursor 移到绝对 byte offset | 否 |
| `pool.commit()` | Python 解析 / IR 构建 | 结束规划并发出最终 size annotation | 否 |

参数拆开看：

| 表达式 | 含义 |
|---|---|
| `pool.alloc((1,), "uint32")` | 分配 4 bytes，并返回一个 `shared.dyn` view |
| `align=8` | 先把 cursor 向上对齐到 8-byte 边界，再分配 |
| `pool.move_base_to(1024)` | 把 cursor 直接设置为 byte offset 1024 |
| `pool.commit()` | 使用 allocator 记录到的 maximum high-water mark 作为 arena 大小 |

当前这段分配可以逐步追踪为：

| 操作 | 分配起点 | 分配大小 | 操作后的 cursor |
|---|---:|---:|---:|
| `tmem_addr` | 0 | 4 bytes | 4 |
| `mma_bar`，先做 8-byte 对齐 | 8 | 8 bytes | 16 |
| `move_base_to(1024)` | 不分配 | 0 | 1024 |
| `Asmem` | 1024 | `128 x 64 x 2 = 16384` bytes | 17408 |
| `Bsmem` | 17408 | `128 x 64 x 2 = 16384` bytes | 33792 |
| `pool.commit()` | 不分配 | 0 | high-water mark = 33792 |

所以最终布局是：

```text
[0,       4)      tmem_addr
[4,       8)      alignment padding
[8,      16)      mma_bar
[16,   1024)      reserved / alignment gap
[1024, 17408)     Asmem, 16 KiB
[17408,33792)     Bsmem, 16 KiB

shared.dyn total = 33792 bytes = 33 KiB
```

`move_base_to(1024)` 很重要，因为后续两个 operand 需要满足 MMA/TMA
descriptor 的 base alignment。把 cursor 直接移到 1024，可以让：

```text
Asmem base = 1024  bytes = 1 KiB 对齐
Bsmem base = 17408 bytes = 17 KiB 对齐
```

这两个 offset 都能按 1024-byte 对齐。若不做这一步，`Asmem` 会直接接在
`mma_bar` 后面，后续 base 不再满足预期的对齐约束。

下面给出一个不依赖 GPU、可以在当前 Python/TIRx 环境执行的完整检查脚本。

文件：`check_smem_pool_plan.py`

```python
from tvm.script import from_source, tirx as T


SOURCE = """
@T.prim_func
def smem_pool_plan():
    T.device_entry()
    pool = T.SMEMPool()
    tmem_addr = pool.alloc((1,), "uint32")
    mma_bar = pool.alloc((1,), "uint64", align=8)
    pool.move_base_to(1024)
    Asmem = pool.alloc((128, 64), "float16")
    Bsmem = pool.alloc((128, 64), "float16")
    pool.commit()
"""


mod = from_source(SOURCE, {"T": T})
print(mod.script())
```

运行：

```bash
./mlc/bin/python check_smem_pool_plan.py
```

本机实际打印出的关键 lowered IR 是：

```text
tmem_addr = T.decl_scalar(T.uint32, data=pool_buf.data, elem_offset=0,
                          scope="shared.dyn")
mma_bar = T.decl_buffer((1,), "uint64", data=pool_buf.data,
                        elem_offset=1, scope="shared.dyn", align=8)
Asmem = T.decl_buffer((128, 64), "float16", data=pool_buf.data,
                      elem_offset=512, scope="shared.dyn")
Bsmem = T.decl_buffer((128, 64), "float16", data=pool_buf.data,
                      elem_offset=8704, scope="shared.dyn")
with T.attr(pool_buf.data, "tirx.pool_max_bytes", 33792):
    T.evaluate(0)
```

这里的 `elem_offset` 是“按各 buffer 的元素类型计数”，不是统一的 byte
offset：

```text
mma_bar: uint64, 1 element  x 8 bytes = 8
Asmem:   float16, 512 elems x 2 bytes = 1024
Bsmem:   float16, 8704 elems x 2 bytes = 17408
```

`tirx.pool_max_bytes = 33792` 才是整个池最终的 byte size。

这里必须区分三个名字相似的机制：

| 写法 | 发生时间 | 含义 |
|---|---|---|
| `pool.commit()` | 编译期 | 结束 SMEM pool 规划，确定 `shared.dyn` 大小 |
| `T.ptx.tcgen05.alloc(...)` | GPU 运行期 | 真正申请 TMEM columns，并写出 TMEM 地址 |
| `T.ptx.tcgen05.commit(...)` | GPU 运行期 | 把异步 MMA 的完成事件关联到 mbarrier |

它们的关系是：

```text
pool.commit()
  只负责 SMEM 地址空间和大小，不碰 TMEM，也不等待任何任务。

tcgen05.alloc(...)
  才是在 GPU 上申请 TMEM。

tcgen05.commit(...)
  是异步 MMA completion 的同步协议，和 SMEM 分配完全不是一回事。
```

`pool.commit()` 也不是下列任何操作：

```text
不是 __syncthreads() 或 CTA barrier
不是 __threadfence() 或 fence
不是数据 flush
不是 TMEM allocation
不是 tcgen05.commit
不是 kernel launch
```

几个容易踩坑的点：

| 错误 | 可观察症状 |
|---|---|
| 忘记 `pool.commit()` | pool 没有正确的最终 size annotation，lowering/launch 可能得到错误 dynamic shared memory 大小 |
| `commit()` 后继续 `pool.alloc()` | 新 allocation 没被最终 size 覆盖，可能越界、覆盖别的数据或污染 CUDA context |
| 把 `commit()` 当同步 | 误删后面的 `cta_sync`/fence，MMA 读到未完成或不可见的 SMEM 数据 |
| 把 `move_base_to()` 当 free | 向后移动不会减少 high-water mark，也不会自动结束旧 buffer 的生命周期 |
| 反复 `commit()` | 把它当作允许的运行时操作，实际它是 IR 构建阶段的最终化协议 |

其中“`commit()` 后继续 alloc”尤其要注意。`SMEMPool.commit()` 的契约是
“必须在所有 `alloc()` 和 `move_base_to()` 之后调用”。错误的 allocation
顺序通常不会得到友好的 Python 异常，而可能在运行阶段表现为随机错误、
非法地址访问或 context poisoning。

自测：

1. `pool.commit()` 会生成哪条 GPU 指令？
   答：不会生成。它只在编译期写入 pool 最终大小。

2. 为什么 `tmem_addr` 占 4 bytes，却让下一个 buffer 从 offset 8 开始？
   答：`mma_bar` 要求 `align=8`，所以 cursor 从 4 向上对齐到 8。

3. 为什么最终大小是 33792 bytes？
   答：`1024 + 128*64*2 + 128*64*2 = 1024 + 16384 + 16384 = 33792`。

4. `pool.commit()` 和 `tcgen05.commit(...)` 是同一个东西吗？
   答：不是。前者结束 SMEM 编译期分配；后者关联异步 MMA 完成事件与
   mbarrier。

5. 在 `pool.commit()` 后再调用 `pool.alloc()`，还能保证正确吗？
   答：不能。`commit()` 已经把最终大小固定，后续 allocation 不在这个
   大小契约内。

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

#### 7.1 `Tx.wg.copy_async(Dreg_wg, tmem)` 的精确含义

```text
本次讲解位置
章节：chapter_gemm_basics
小节：第 1 步：顺序执行的单 Tile GEMM
知识点：warpgroup 协作的 TMEM -> registers 异步读取
上次：tcgen05.commit 与 mbarrier.try_wait 的 MMA 完成同步
下次：第 2 步：K-Loop 累加与 MMA barrier phase
PTX：tcgen05.ld、tcgen05.wait::ld
```

这一行可以按 API 的三层含义拆开：

```text
Tx           TIRx 的 tile primitive namespace
wg           operation scope 是 warpgroup，本 kernel 中为 128 threads
copy_async   在这些 threads 上协作执行异步 copy
```

因此它不是“一个 thread 把整块 TMEM 搬到自己的 registers”，而是：

```text
一个 warpgroup 共同发出一组 TMEM load
每个 thread 根据 destination layout
只接收属于自己的那一部分元素
```

完整相关代码仍然来自本节的 `hgemm_v1`：

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

Tx.wg.copy_async(Dreg_wg[:, :], tmem[:, :BLK_N])
T.ptx.tcgen05.wait.ld()
Tx.cast(Dreg_f16[:], Dreg[:])
```

参数含义是：

| 参数 | 角色 | 所在空间 | Layout |
|---|---|---|---|
| `Dreg_wg[:, :]` | copy destination | 每线程 registers 的分布式 view | row -> `tid_in_wg`，col -> register index |
| `tmem[:, :BLK_N]` | copy source | TMEM | row -> `TLane`，col -> `TCol` |

`Dreg` 本身只是每个 thread 私有的 128 个 fp32 registers。它的 shape 是
`(BLK_N,)`，从单个 thread 的视角看没有 `128 x 128` 这个 tile 的含义。
`Dreg_wg` 使用 `view` 给同一份 register storage 增加一个 warpgroup-wide
tile layout，所以它没有创建第二份 register buffer。

这个 destination layout 表示：

```text
logical row 0   -> tid_in_wg 0
logical row 1   -> tid_in_wg 1
...
logical row 127 -> tid_in_wg 127

logical col n   -> 该 thread 的 Dreg[n]
```

于是 `Tx.wg.copy_async` 建立的是下面的逐元素对应关系：

```text
tmem(TLane=m, TCol=n)
-> Dreg_wg(row=m, col=n)
-> thread tid_in_wg=m 的 Dreg[n]
```

具体追踪 `D[73, 91]`：

| 步骤 | 结果 |
|---|---|
| 数学元素 | `D[73, 91]` |
| TMEM source | `TLane=73, TCol=91` |
| destination owner | `tid_in_wg=73` |
| thread 73 的物理位置 | warp 2，lane 9 |
| 落入的 register | `Dreg[91]` |

这里最容易混淆的是“一个 warpgroup 共同执行”和“每个 thread 都持有整块
tile”。真实情况是：

```text
warpgroup 共同执行这个 operation
但每个 thread 只持有 layout 分配给它的元素

thread 73 收到 row 73 的 128 个 columns
并不会收到 row 72 或 row 74 的元素
```

`copy_async` 中的 async 表示 load 发出后，执行流可以继续向后走，register
内容不一定已经就绪。因此必须在读取 `Dreg` 前执行：

```python
T.ptx.tcgen05.wait.ld()
```

否则后面的：

```python
Tx.cast(Dreg_f16[:], Dreg[:])
```

可能读到旧值或只完成了一部分的新值。

它在概念上 lower 到 Blackwell 的 TMEM load path，例如
`tcgen05.ld.sync.aligned.*` 形式；具体生成哪一种 atom 和 repeat 次数由
source layout、destination layout、元素类型和 compiler lowering 决定。
学习应该先掌握数据所有权，而不是背死某一条 lowering 后的 opcode。

常见错误及症状：

| 错误 | 可观察症状 |
|---|---|
| 只让一个 thread 执行 `wg.copy_async` | warpgroup copy 不完整，destination layout 无法按预期填充 |
| destination layout 与 `m_thr` 不一致 | 整行数据落到错误 output row |
| 漏掉 `tcgen05.wait.ld()` | 读到 stale registers 或部分旧数据 |
| 把 `Dreg_wg` 当作额外缓冲区 | 误以为有第二份存储，后续更容易出现重复或漏写 |
| copy 尚未完成就复用 `Dreg` | register RAW hazard，结果不稳定 |

自测：

1. `wg` 在 `Tx.wg.copy_async` 中是什么 scope？
   答：warpgroup scope。本 kernel 中一个 warpgroup 有 128 threads。

2. `Dreg_wg` 和 `Dreg` 是两份数据吗？
   答：不是。`Dreg_wg` 是 `Dreg` 的 warpgroup-wide tile view，底层是同一份
   每线程 register storage。

3. `D[73, 91]` 最终落到哪个 thread 的哪个 register？
   答：`tid_in_wg=73` 的 `Dreg[91]`。thread 73 位于 warp 2、lane 9。

4. 为什么 copy 后必须执行 `tcgen05.wait.ld()`？
   答：异步 TMEM load 可能尚未完成；不等待就读取 `Dreg` 会违反数据依赖。

5. 为什么不能假设这个 primitive 的 lowering 固定是一条 `tcgen05.ld`？
   答：它是 tile-level operation，具体 atom 和 repeat 次数由 types、layouts
   与 compiler lowering 决定。

#### 7.2 `tcgen05.wait.ld()`：等待异步 TMEM load 真正完成

```text
本次讲解位置
章节：chapter_gemm_basics
小节：第 1 步：顺序执行的单 Tile GEMM
知识点：tcgen05.ld 的完成等待与 register 可见性
上次：Tx.wg.copy_async 的分布式 destination layout
下次：第 2 步：K-Loop 累加与 MMA barrier phase
PTX：9.7.18.8.5 Tensorcore 5th Generation Instructions: tcgen05.wait
```

上一节已经知道：

```python
Tx.wg.copy_async(Dreg_wg[:, :], tmem[:, :BLK_N])
```

会发出从 TMEM 到 registers 的异步 `tcgen05.ld`。这里的“异步”表示：

```text
发出 load
!=
load 已经完成
!=
destination registers 已经可以安全读取或复用
```

`T.ptx.tcgen05.wait.ld()` 补上的就是这个完成点：

```text
copy_async 发出 TMEM load
    |
    | load 还在进行
    v
T.ptx.tcgen05.wait.ld()
    |
    | 当前线程等待自己此前发出的所有 tcgen05.ld 完成
    v
Dreg 中的 destination registers 可以安全消费
```

可以把整个 writeback 想成下面的顺序：

```text
第一句：把 TMEM 数据订回来，但包裹还在路上
wait.ld：等当前线程订的所有包裹都送到
cast：现在才打开 register 里的包裹
```

它在 TVM 中最终 lower 到：

```cpp
asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
```

逐个 token 解释：

| token | 含义 |
|---|---|
| `tcgen05` | Blackwell 第五代 Tensor Core / Tensor Memory 相关指令族 |
| `wait` | 完成等待，不发起新的数据搬运 |
| `::ld` | 只等待此前异步发出的 `tcgen05.ld` |
| `.sync` | 同一个 warp 中的 threads 都要执行这条 wait，然后才能一起继续 |
| `.aligned` | 同一 warp 的所有 threads 必须执行相同的 wait instruction |
| `"memory"` clobber | 告诉 C++ compiler 这条 asm 会改变内存可见性，不能随意挪动或优化掉 |

最重要、也最容易误解的 scope 是：

```text
tcgen05.wait::ld
-> 等待执行这条指令的 thread 此前发出的所有 tcgen05.ld
-> 不等待其他 thread 发出的 load
-> 不等待 tcgen05.mma
-> 不是 CTA barrier
-> 不是 warpgroup-wide barrier
```

PTX 对 `.sync` 的规定是：执行 wait 的 thread 先等待自己的 prior loads 完成，
然后再等同一个 warp 中的所有 threads 都执行到同一条 wait，之后整段代码才
继续。因此它同时包含两层作用：

```text
thread-local completion：本 thread 的 loads 已完成
warp rendezvous：同一个 warp 的所有 lanes 都到达 wait
```

它不是：

```text
warp 0 等 warp 1 / warp 2 / warp 3
也不是 128-thread warpgroup barrier
```

本 kernel 中四个 warps 都执行：

```python
T.ptx.tcgen05.wait.ld()
```

每个 warp 独立完成自己的 `.sync.aligned` 等待。128 个线程都执行到 wait
以后，各自负责的 register fragment 才是可消费状态。

完整相关代码是：

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

Tx.wg.copy_async(
    Dreg_wg[:, :],
    tmem[:, :BLK_N],
)
T.ptx.tcgen05.wait.ld()

Tx.cast(Dreg_f16[:], Dreg[:])
Tx.copy(
    D[m_st + warp_id * 32 + lane_id, n_st : n_st + BLK_N],
    Dreg_f16[:],
)
```

这里的顺序不能交换：

```text
correct:
    copy_async
    wait.ld
    cast

wrong:
    copy_async
    cast
    wait.ld
```

第二种顺序把 wait 放得太晚。即使某次执行碰巧能读到正确值，也依赖 timing，
不是正确的同步契约。

把 `D[73, 91]` 代入：

| 阶段 | 状态 |
|---|---|
| `Tx.wg.copy_async(...)` | 发出到 `TMEM[TLane=73, TCol=91]` 的异步 load |
| load pending | thread 73 的 `Dreg[91]` 尚不能作为完成后数据使用 |
| `wait.ld()` | thread 73 等待自己的 prior loads 完成；warp 2 内部汇合 |
| wait 返回 | thread 73 的 `Dreg[91]` 达到可消费状态 |
| `Tx.cast(...)` | 从 `Dreg[91]` 读取并转换为 fp16 |
| 后续 `Tx.copy(...)` | 将结果写到 `D[m_thr, 91]`，其中 `m_thr=73` |

需要特别区分三类同步：

| 机制 | 等待什么 | Scope | 典型用途 |
|---|---|---|---|
| `tcgen05.commit` + `mbarrier.try_wait` | `tcgen05.mma` / `cp` / `shift` 的完成事件 | 通过 mbarrier 跨 thread 观察 | 确保 MMA 结果可供后续 consumer 读取 |
| `tcgen05.wait::ld` | 当前 thread 此前发出的 `tcgen05.ld` | thread + warp `.sync.aligned` | 确保 TMEM -> register load 完成 |
| `cta_sync` | CTA 内 thread 到达同步点 | CTA threads | 建立 thread 间控制同步；不会自动等待 async load 完成 |

因此下面这种替代是错的：

```python
Tx.wg.copy_async(Dreg_wg[:, :], tmem[:, :BLK_N])
T.cuda.cta_sync()  # 错：它不会替你等待这个 thread 的 tcgen05.ld
Tx.cast(Dreg_f16[:], Dreg[:])
```

`cta_sync` 只能说明 threads 到达了某个执行点，不能替代 `tcgen05.ld` 的完成
协议。

下面是一个不依赖 GPU 的完成语义模拟，用来区分“发出 load”和“load 完成”：

文件：`check_wait_ld_semantics.py`

```python
OLD_VALUE = 0
NEW_VALUE = 7


class AsyncRegister:
    def __init__(self, value):
        self.value = value
        self.pending_value = None
        self.pending = False

    def issue_ld(self, value):
        self.pending_value = value
        self.pending = True

    def wait_ld(self):
        if self.pending:
            self.value = self.pending_value
            self.pending_value = None
            self.pending = False

    def read(self):
        if self.pending:
            return f"unsafe-before-wait(value={self.value}, pending=True)"
        return f"ready(value={self.value}, pending=False)"


reg = AsyncRegister(OLD_VALUE)
reg.issue_ld(NEW_VALUE)
print(f"after issue: {reg.read()}")

reg.wait_ld()
print(f"after wait:  {reg.read()}")
```

运行：

```bash
python3 check_wait_ld_semantics.py
```

预期输出：

```text
after issue: unsafe-before-wait(value=0, pending=True)
after wait:  ready(value=7, pending=False)
```

这个脚本只模拟 API 的完成顺序，不模拟真实 GPU timing。真实 `tcgen05.ld`
只能在没有分歧的 warp 中执行对应的 `tcgen05.wait::ld`，并且必须运行在
支持该指令的 Blackwell GPU 上。

常见错误和症状：

| 错误 | 可观察症状 |
|---|---|
| 把 `copy_async` 当成同步 copy | 还没完成就读取 `Dreg`，结果依赖 timing |
| 用 `cta_sync` 代替 `wait.ld` | CTA 的线程同步了，但 TMEM load 仍可能 pending |
| 用 `wait.st` 代替 `wait.ld` | 等待了错误的指令类别，不能建立本 load 的完成 |
| 只在部分 lanes 执行 wait | 违反 `.sync.aligned` 的整体执行要求，行为未定义 |
| 提前复用 `Dreg` | 新值可能覆盖尚未完成的旧 load destination，产生 anti-dependency hazard |
| 以为 wait 会等待其他 warps 的 loads | 错误假设跨 warp 已完成；实际每 warp 只负责自己的同步约定 |

自测：

1. `tcgen05.wait.ld()` 等待的是哪一类操作？
   答：当前 thread 此前发出的所有异步 `tcgen05.ld` 操作。

2. 它会不会等待 `tcgen05.mma` 完成？
   答：不会。MMA 的完成通常通过 `tcgen05.commit` 关联到的 mbarrier
   来等待。

3. `.sync.aligned` 的同步范围是整个 CTA 吗？
   答：不是。它约束的是同一个 warp；它不会替代 CTA barrier 或
   warpgroup barrier。

4. 把 `cta_sync()` 放在 load 后、cast 前，能不能保证 `Dreg` 已完成？
   答：不能。`cta_sync` 同步 thread 到达，不负责等待异步 TMEM load。

5. 为什么 `wait.ld()` 必须在 `cast` 之前？
   答：因为 `cast` 要消费 TMEM load 写入的 registers，必须等 prior loads
   完成，建立正确的数据依赖。

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

#### 8.1 Writeback 全链路：TMEM -> RF -> GMEM

```text
本次讲解位置
章节：chapter_gemm_basics
小节：第 1 步：顺序执行的单 Tile GEMM
知识点：TMEM -> per-thread registers -> cast -> GMEM writeback
上次：某一行 TMEM load 如何映射到 thread 和 register
下次：Dreg.view 如何表示 warpgroup distributed tile
PTX：tcgen05.ld、tcgen05.wait::ld、global store
```

7.1 已经解释了 `Tx.wg.copy_async` 如何把 TMEM 的每个元素分配给
warpgroup 中的线程。这里把整段 writeback 连起来：

```text
TMEM accumulator
    |
    | Tx.wg.copy_async + tcgen05.wait.ld
    v
每线程 fp32 registers: Dreg[0:BLK_N]
    |
    | Tx.cast
    v
每线程 fp16 registers: Dreg_f16[0:BLK_N]
    |
    | m_thr 选择全局 row，Tx.copy 写一行列区间
    v
GMEM D[m_thr, n_st:n_st+BLK_N]
```

完整代码块是：

```python
# --- Writeback：TMEM -> RF -> GMEM ---
Dreg = T.alloc_local((BLK_N,), acc_type)
Dreg_f16 = T.alloc_local((BLK_N,), d_type)
Dreg_wg = Dreg.view(
    128,
    BLK_N,
    layout=TileLayout(
        S[(128, BLK_N) : (1 @ tid_in_wg, 1)]
    ),
)
Tx.wg.copy_async(Dreg_wg[:, :], tmem[:, :BLK_N])
T.ptx.tcgen05.wait.ld()
Tx.cast(Dreg_f16[:], Dreg[:])
m_thr = T.meta_var(m_st + warp_id * 32 + lane_id)
Tx.copy(D[m_thr, n_st : n_st + BLK_N], Dreg_f16[:])
```

逐行含义如下。

| 代码 | 作用 | Scope | 数据位置 |
|---|---|---|---|
| `T.alloc_local((BLK_N,), acc_type)` | 每个线程分配 128 个 fp32 accumulator slots | 单 thread | registers，必要时也可能 spill |
| `T.alloc_local((BLK_N,), d_type)` | 每个线程分配 128 个 fp16 output slots | 单 thread | registers |
| `Dreg.view(...)` | 把每线程的局部数组解释成 warpgroup-wide tile | 不执行 | 同一份 register storage |
| `Tx.wg.copy_async(...)` | 异步把 TMEM 元素加载到对应线程的 `Dreg` | warpgroup，128 threads | TMEM -> registers |
| `tcgen05.wait.ld()` | 等待本线程此前发出的 TMEM loads 完成 | 单 thread | register dependency |
| `Tx.cast(...)` | 逐元素 fp32 -> fp16 | 每个线程处理自己的 128 个元素 | registers |
| `m_thr = ...` | 计算当前线程负责的全局 output row | 单 thread | 标量坐标 |
| `Tx.copy(...)` | 把当前线程的 128 个 fp16 写到 GMEM 一行 | 单 thread 一条 row，整体构成 128x128 tile | registers -> GMEM |

`Dreg` 本身是“一个线程的 128 个元素”：

```text
Dreg shape = (128,)
```

它不是整块 `128 x 128` tile。`Dreg_wg` 不分配新存储，而是给同一组 registers
增加一个分布式布局：

```text
logical row -> tid_in_wg
logical col -> 当前线程的 Dreg index
```

因此对于某个逻辑坐标 `(row, col)`：

```text
Dreg_wg[row, col]
-> owner thread = tid_in_wg
-> owner thread 的 Dreg[col]
```

`tmem[:, :BLK_N]` 的 accumulator layout 是：

```text
TLane = row
TCol  = col
```

两边连起来，`Tx.wg.copy_async` 完成的是：

```text
TMEM[TLane, TCol]
-> Dreg_wg[TLane, TCol]
-> thread tid_in_wg=TLane 的 Dreg[TCol]
```

这里的 `async` 只表示 load 已发出，不表示 register 已经可以读取。
`tcgen05.wait.ld()` 是后续 `cast` 能安全读取 `Dreg` 的必要条件：

```text
copy_async 发出 load
    |
    | 此时 Dreg 可能还没有完整的新值
    v
wait.ld()
    |
    | 保证本线程的 TMEM load 已完成
    v
cast 读取 Dreg
```

`Tx.cast(Dreg_f16[:], Dreg[:])` 只改变寄存器中的数值表示：

```text
Dreg[i]     fp32 accumulator
Dreg_f16[i] fp16 output
```

它不写 TMEM，也不写 GMEM。

最后两行决定每个线程把数据写到哪里：

```python
m_thr = T.meta_var(m_st + warp_id * 32 + lane_id)
Tx.copy(D[m_thr, n_st : n_st + BLK_N], Dreg_f16[:])
```

线程编号也可以写成：

```text
tid_in_wg = warp_id * 32 + lane_id
```

所以：

```text
m_thr = m_st + tid_in_wg
```

当前 grid 只有一个 tile，`m_st = 0`、`n_st = 0`，于是：

| `tid_in_wg` | `warp_id` | `lane_id` | `m_thr` | 写往 GMEM |
|---:|---:|---:|---:|---|
| 0 | 0 | 0 | 0 | `D[0, 0:128]` |
| 31 | 0 | 31 | 31 | `D[31, 0:128]` |
| 32 | 1 | 0 | 32 | `D[32, 0:128]` |
| 73 | 2 | 9 | 73 | `D[73, 0:128]` |
| 127 | 3 | 31 | 127 | `D[127, 0:128]` |

以 `D[73, 91]` 为例，完整执行追踪是：

| 阶段 | 坐标或结果 |
|---|---|
| 数学元素 | `D[73, 91]` |
| TMEM source | `TLane=73, TCol=91` |
| owning thread | `tid_in_wg=73` |
| thread 73 的 warp / lane | `warp_id=2, lane_id=9` |
| fp32 register | `Dreg[91]` |
| cast 后 | `Dreg_f16[91]` |
| `m_thr` | `0 + 2*32 + 9 = 73` |
| GMEM destination | `D[73, 91]` |
| 该线程整行写回 | `D[73, 0:128] <- Dreg_f16[0:128]` |

下面是一个不依赖 GPU 的完整坐标检查脚本，用来验证 row ownership 和
register index：

文件：`check_writeback_mapping.py`

```python
BLK_N = 128
m_st = 0
n_st = 0

target_row = 73
target_col = 91

tid_in_wg = target_row
warp_id = tid_in_wg // 32
lane_id = tid_in_wg % 32
register_index = target_col
m_thr = m_st + warp_id * 32 + lane_id

print(f"tid_in_wg={tid_in_wg}")
print(f"warp_id={warp_id}")
print(f"lane_id={lane_id}")
print(f"register_index={register_index}")
print(f"m_thr={m_thr}")
print(f"destination=D[{m_thr}, {n_st + register_index}]")
print(f"row_write=D[{m_thr}, {n_st}:{n_st + BLK_N}]")
```

运行：

```bash
python3 check_writeback_mapping.py
```

预期输出：

```text
tid_in_wg=73
warp_id=2
lane_id=9
register_index=91
m_thr=73
destination=D[73, 91]
row_write=D[73, 0:128]
```

这个脚本只验证坐标，不执行 `tcgen05`。当前 M5 Pro 没有 NVIDIA
Blackwell GPU，因此不能在本机做真实 TMEM load 或 GMEM writeback 的运行期
验证。

常见错误及症状：

| 错误 | 可观察症状 |
|---|---|
| 漏掉 `tcgen05.wait.ld()` | `Dreg` 读到旧值或只完成一部分的新值 |
| `Dreg_wg` 的 row layout 与 `m_thr` 不匹配 | 输出行被置换，部分 row 重复或丢失 |
| 把 `Dreg` 当成整块 tile | 误以为每个线程持有 `128 x 128` 元素，thread 映射完全错 |
| `Dreg_f16` 与 `D` 的 dtype 不一致 | value 被错误解释，lowering 失败或输出编码错误 |
| `n_st` 或 `m_st` 算错 | 合法地址上写入错误 output tile |
| 在 `wait.ld()` 前复用 `Dreg` | register RAW hazard，结果不稳定 |
| 在非整 tile 边界沿用 `n_st + BLK_N` | 最后一次 GMEM store 越界 |

自测：

1. `Dreg` 和 `Dreg_wg` 是两份存储吗？
   答：不是。`Dreg_wg` 是 `Dreg` 的分布式 tile view。

2. `D[73, 91]` 由哪个线程、从哪个 register 写出？
   答：`tid_in_wg=73` 的线程从 `Dreg_f16[91]` 写出。

3. 为什么 `tcgen05.wait.ld()` 必须在 cast 前？
   答：TMEM load 是异步的，不等待就可能读取未完成的 `Dreg`。

4. `Tx.cast` 是否把结果写回 TMEM？
   答：不会。它只在 registers 中把 fp32 转成 fp16。

5. 当前线程的 `Tx.copy` 为什么只写一行？
   答：每个 `tid_in_wg` 通过 layout 拥有一个输出 row，但持有该 row 的
   `BLK_N` 个列元素。

#### 8.2 `Dreg.view` 为什么能表示一个分布式 tile

```text
本次讲解位置
章节：chapter_gemm_basics
小节：第 1 步：顺序执行的单 Tile GEMM
知识点：local buffer 的 warpgroup distributed view 与 TileLayout
上次：TMEM -> RF -> GMEM writeback 全链路
下次：m_thr 如何把线程映射到全局输出行
PTX：tcgen05.ld 的 register destination distribution
```

前面已经知道 `Dreg.view` 后的 `Dreg_wg` 能表示一个 `128 x BLK_N` tile，
但这里最容易卡住的是：

```text
每个 thread 明明只分配了 Dreg[0:BLK_N]
为什么 view 以后会变成一个 128 x BLK_N 的二维 buffer？
```

答案是：`Dreg_wg` 不是一个 thread 私有的 `128 x BLK_N` 连续数组。
它描述的是整个 warpgroup 中 128 个线程共同组成的分布式逻辑 tile。

先建立一个类比：

```text
Dreg
= 一个学生自己的 BLK_N 格答题纸

Dreg_wg
= 老师看到的 128 个学生 x BLK_N 格的成绩册视图

Dreg_wg[row, col]
= 第 row 个学生的第 col 格答案
```

老师没有另造一本包含全部答案的纸；`Dreg_wg` 只是把 128 个学生的本地答题纸
按 row 编号组织成一个二维逻辑视图。

代码是：

```python
Dreg = T.alloc_local((BLK_N,), acc_type)

Dreg_wg = Dreg.view(
    128,
    BLK_N,
    layout=TileLayout(
        S[(128, BLK_N) : (1 @ tid_in_wg, 1)]
    ),
)
```

这里同时存在两个 scope：

| 名称 | 看到的 shape | 实际位置 | 含义 |
|---|---:|---|---|
| `Dreg` | `(BLK_N,)` | 每个 thread 自己的一份 local storage | 当前线程的 `BLK_N` 个 register slots |
| `Dreg_wg` | `(128, BLK_N)` | 同一个 warpgroup 的 128 份 `Dreg` | 128 个线程共同组成的逻辑 tile |

因此：

```text
Dreg_wg 的逻辑元素总数
= 128 * BLK_N

每个 thread 实际持有的元素数
= (128 * BLK_N) / 128
= BLK_N
```

这正好对应：

```text
Dreg.shape = (BLK_N,)
```

`view` 不分配新的 registers，也不复制数据。它只给同一份 local storage
增加一层逻辑 shape 和 layout，告诉后续 tile-level operation：

```text
哪个逻辑元素应该由哪个 thread 持有
该元素又落在那个 thread 的哪个 local index
```

最重要的 layout 是：

```python
S[(128, BLK_N) : (1 @ tid_in_wg, 1)]
```

可以逐项拆成：

| item | extent | stride | layout axis | 作用 |
|---|---:|---:|---|---|
| logical row | `128` | `1 @ tid_in_wg` | thread axis `tid_in_wg` | row 决定 owner thread |
| logical col | `BLK_N` | `1` | 默认 local/register axis | col 决定 owner thread 的 `Dreg` index |

于是坐标为 `(row, col)` 的元素满足：

```text
owner thread = row
local register index = col
```

也就是：

```text
Dreg_wg[row, col]
-> tid_in_wg = row 的线程
-> 该线程的 Dreg[col]
```

举三个具体坐标：

| `Dreg_wg` 坐标 | owner `tid_in_wg` | owner 的 warp / lane | physical register |
|---|---|---|---|
| `[0, 0]` | 0 | warp 0, lane 0 | `Dreg[0]` |
| `[73, 91]` | 73 | warp 2, lane 9 | `Dreg[91]` |
| `[127, 127]` | 127 | warp 3, lane 31 | `Dreg[127]` |

以 `(73, 91)` 为例：

```text
logical row = 73
logical col = 91

73 * (1 @ tid_in_wg) -> tid_in_wg = 73
91 * 1               -> local register = Dreg[91]
```

所以：

```text
线程 73:
    不持有整个 128 x BLK_N tile
    只持有属于自己 row 的 BLK_N 个元素
    Dreg_wg[73, 0:BLK_N] -> Dreg[0:BLK_N]
```

反过来，线程 0 也不会持有 `row=1` 的数据。如果某个操作需要跨 row
访问，必须有 warpgroup scope 的协作机制来路由元素，不能把它当成每线程
都能直接访问的一块连续 memory。

再看它与 TMEM copy 的连接：

```python
Tx.wg.copy_async(
    Dreg_wg[:, :],
    tmem[:, :BLK_N],
)
```

source TMEM 的 layout 是：

```text
TLane = row
TCol  = col
```

destination `Dreg_wg` 的 layout 是：

```text
tid_in_wg = row
register  = col
```

所以 copy 的元素级映射是：

```text
TMEM[TLane, TCol]
-> Dreg_wg[TLane, TCol]
-> thread tid_in_wg=TLane 的 Dreg[TCol]
```

例如：

```text
TMEM[73, 91]
-> Dreg_wg[73, 91]
-> thread 73 的 Dreg[91]
```

`Tx.wg.copy_async` 的 `wg` 很重要。这里的 producer 和 consumer 都是同一个
warpgroup 的 128 个线程：

```text
128 个线程共同发出一次 warpgroup copy
每个线程根据 destination layout 接收自己的元素
thread row 只接收 row=thread row 的那一行
```

这里的 `view` 是编译期抽象的 shape/layout 转换，不会生成一条运行时
`view` 指令。真正执行搬运和路由的是后面的 `Tx.wg.copy_async`；
`view` 提供的是它必须知道的 destination ownership。

下面是一个完整、不依赖 GPU 的语义模拟脚本。它把 128 个线程的本地
`Dreg` 真实存成 128 个一维数组，再通过 layout 规则访问二维逻辑视图。

文件：`check_dreg_distributed_view.py`

```python
ROWS = 128
BLK_N = 128
WARPGROUP_THREADS = ROWS

# 物理存储：每个 thread 只有 BLK_N 个 local slots。
regs = [
    [thread_id * 1000 + col for col in range(BLK_N)]
    for thread_id in range(WARPGROUP_THREADS)
]


def view_read(row, col):
    tid_in_wg = row
    register_index = col
    return regs[tid_in_wg][register_index]


def view_write(row, col, value):
    tid_in_wg = row
    register_index = col
    regs[tid_in_wg][register_index] = value


row = 73
col = 91

print(f"view shape=({ROWS}, {BLK_N})")
print(f"per-thread local shape=({BLK_N},)")
print(f"Dreg_wg[{row}, {col}]={view_read(row, col)}")

view_write(row, col, 123456)
print(f"after write: Dreg_wg[{row}, {col}]={view_read(row, col)}")
print(f"physical owner: regs[{row}][{col}]={regs[row][col]}")
```

运行：

```bash
python3 check_dreg_distributed_view.py
```

预期输出：

```text
view shape=(128, 128)
per-thread local shape=(128,)
Dreg_wg[73, 91]=73091
after write: Dreg_wg[73, 91]=123456
physical owner: regs[73][91]=123456
```

这个脚本模拟的是 ownership 和索引，不执行 `tcgen05.ld`。它的目的是验证：
`Dreg_wg` 的二维坐标最终一定落到“某个 owner thread 的一个 local index”。

常见错误和症状：

| 错误 | 实际发生的错误 | 可观察症状 |
|---|---|---|
| 认为 `Dreg_wg` 是每个线程另分配的 `128 x BLK_N` 数组 | 把分布式 view 当成单线程私有 buffer | register 数量被高估，thread mapping 全错 |
| 认为每个线程拥有完整 tile | 忽略了 128 个 row 分给 128 个线程 | 后续 GMEM writeback 重复写或只写一行 |
| 把 `1 @ tid_in_wg` 写成普通 `1` | row 不再选择 owner thread | 所有逻辑 row 落到同一个 thread 的 register |
| row/col stride 互换 | row 去选 register，col 去选 thread | 输出发生转置、行错位或地址非法 |
| 在 warpgroup 外按 `Dreg_wg` 访问 | 参与操作的 thread set 与 layout 不一致 | copy 不完整或结果未定义 |
| `Dreg_wg` 的 row owner 与 `m_thr` 不一致 | producer 和 consumer 使用不同的 row 映射 | 输出整行互换或有规律缺失 |

最后记成一句话：

```text
Dreg 是“每个线程本地的一行寄存器”
Dreg_wg 是“128 个线程的这些本地行共同组成的分布式二维 tile”
1 @ tid_in_wg 负责选 row owner
后面的 1 负责选该线程的 register index
```

自测：

1. `Dreg` 和 `Dreg_wg` 的 shape 分别是什么？
   答：`Dreg.shape=(BLK_N,)`，`Dreg_wg` 的逻辑 shape 是
   `(128, BLK_N)`；后者是 warpgroup-wide distributed view。

2. `Dreg_wg[73, 91]` 最终落在哪里？
   答：`tid_in_wg=73` 的线程的 `Dreg[91]`。该线程是 warp 2、lane 9。

3. 为什么不能把 `Dreg_wg` 当成每个线程都有的 `128 x BLK_N` buffer？
   答：整个 tile 的元素分散在 128 个线程中。每个线程只持有 `BLK_N`
   个元素，所有线程的本地 storage 合起来才是完整的 `128 x BLK_N`。

4. `1 @ tid_in_wg` 中的 `tid_in_wg` 表示什么？
   答：它标识 warpgroup 内的 owner thread，范围是 0 到 127。logical row
   通过这个轴决定应由哪个线程持有。

5. `view` 是否会生成一条运行时指令来搬运数据？
   答：不会。`view` 是编译期 shape/layout 转换；真正搬运和按 layout
   路由数据的是后面的 `Tx.wg.copy_async`。

#### 8.3 `m_thr` 如何把线程映射到全局输出行

```text
本次讲解位置
章节：chapter_gemm_basics
小节：第 1 步：顺序执行的单 Tile GEMM
知识点：thread id 到 global output row 的 writeback 映射
上次：Dreg_wg 的 warpgroup distributed view
下次：第 2 步：K-Loop 累加与 MMA barrier phase
PTX：global store / vectorized store path
```

先建立一个整行写回的心智模型：

```text
每个 thread 手里有一行结果
-> Dreg_f16[0:BLK_N]

每个 thread 需要知道：
这行结果属于全局 D 的第几行

Tx.copy 负责：
把这一行写到那个全局 row 的连续列区间
```

因此这两行的核心不是重新排列 register，而是回答“当前线程负责哪一行”：

```python
m_thr = T.meta_var(m_st + warp_id * 32 + lane_id)
Tx.copy(D[m_thr, n_st : n_st + BLK_N], Dreg_f16[:])
```

完整上下文仍然是：

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

Tx.wg.copy_async(Dreg_wg[:, :], tmem[:, :BLK_N])
T.ptx.tcgen05.wait.ld()
Tx.cast(Dreg_f16[:], Dreg[:])

m_thr = T.meta_var(m_st + warp_id * 32 + lane_id)
Tx.copy(D[m_thr, n_st : n_st + BLK_N], Dreg_f16[:])
```

五个名字分别表示：

| 名字 | 范围或公式 | 含义 |
|---|---|---|
| `warp_id` | `0..3` | 当前 thread 在 warpgroup 中的 warp 编号 |
| `lane_id` | `0..31` | 当前 thread 在 warp 中的 lane 编号 |
| `tid_in_wg` | `warp_id * 32 + lane_id` | 当前 thread 在 128-thread warpgroup 中的编号 |
| `m_st` | `bx * BLK_M` | 当前 CTA output tile 在全局 D 中的起始 row |
| `n_st` | `by * BLK_N` | 当前 CTA output tile 在全局 D 中的起始 column |
| `m_thr` | `m_st + warp_id * 32 + lane_id` | 当前 thread 负责写回的全局 output row |

先看 `tid_in_wg` 的展开：

```text
warp 0, lane 0  -> tid_in_wg = 0 * 32 + 0  = 0
warp 0, lane 31 -> tid_in_wg = 0 * 32 + 31 = 31
warp 1, lane 0  -> tid_in_wg = 1 * 32 + 0  = 32
warp 2, lane 9  -> tid_in_wg = 2 * 32 + 9  = 73
warp 3, lane 31 -> tid_in_wg = 3 * 32 + 31 = 127
```

所以：

```text
m_thr = m_st + tid_in_wg
```

它只是在 `tid_in_wg` 上增加当前 CTA tile 的 row offset：

```text
tile-local row = tid_in_wg
global row     = m_st + tile-local row
```

当前 baseline 只有一个 tile：

```text
bx = 0
by = 0
m_st = 0
n_st = 0
BLK_N = 128
```

此时 `m_thr` 与 `tid_in_wg` 数值相同：

| `warp_id` | `lane_id` | `tid_in_wg` | `m_thr` | 当前线程写回 |
|---:|---:|---:|---:|---|
| 0 | 0 | 0 | 0 | `D[0, 0:128] <- Dreg_f16[0:128]` |
| 0 | 31 | 31 | 31 | `D[31, 0:128] <- Dreg_f16[0:128]` |
| 2 | 9 | 73 | 73 | `D[73, 0:128] <- Dreg_f16[0:128]` |
| 3 | 31 | 127 | 127 | `D[127, 0:128] <- Dreg_f16[0:128]` |

关键点是这行代码由 warpgroup 的 128 个线程一起执行，但每个线程算出的
`m_thr` 不同：

```text
thread 0   -> m_thr = 0
thread 1   -> m_thr = 1
...
thread 127 -> m_thr = 127
```

因此一条源码语句最终形成 128 个有效的 row store：

```text
thread 0   : D[0,   n_st:n_st+BLK_N]
thread 1   : D[1,   n_st:n_st+BLK_N]
...
thread 127 : D[127, n_st:n_st+BLK_N]
```

128 个线程合起来，正好覆盖整个 `128 x BLK_N` output tile：

```text
128 rows
x
每个线程持有的 BLK_N 个 fp16 columns
=
128 x BLK_N output elements
```

destination 和 source 的形状也需要对应：

| 表达式 | 含义 | shape |
|---|---|---|
| `D[m_thr, n_st:n_st+BLK_N]` | 全局输出中的一行列区间 | `(BLK_N,)` |
| `Dreg_f16[:]` | 当前线程已经 cast 的整行输出 | `(BLK_N,)` |

这里的 `D[...]` 只固定第一个维度 `m_thr`，第二个维度仍然是连续的列区间。
所以它不是每个线程写一个单独元素，而是每个线程写自己那一行的
`BLK_N` 个元素。

再看多 tile 的情况。假设：

```text
BLK_M = 128
BLK_N = 128
bx = 1
by = 2
warp_id = 2
lane_id = 9
```

计算过程是：

```text
m_st = bx * BLK_M
     = 1 * 128
     = 128

tid_in_wg = warp_id * 32 + lane_id
          = 2 * 32 + 9
          = 73

m_thr = m_st + tid_in_wg
      = 128 + 73
      = 201

n_st = by * BLK_N
     = 2 * 128
     = 256
```

因此这个线程执行的是：

```text
D[201, 256:384] <- Dreg_f16[0:128]
```

它不再写 `D[73, ...]`，而是写全局 output tile `(bx=1, by=2)` 内部的第
73 行。少乘或多乘 `BLK_M` 都会表现出“局部结果正确，但落到错误 CTA
tile”的地址上。

##### `T.meta_var` 在这里做什么

`T.meta_var(...)` 是 TIRx script parser 的元值标记。它表示：

```text
m_thr 这个名字在解析脚本时直接绑定到后面的表达式
而不是额外创建一个可变的 local scalar
```

对应源码中的处理是：

```python
if isinstance(value, I.meta_var):
    return value.value
```

所以它不会：

```text
不会生成一条 GPU arithmetic 指令
不会把 warp_id 或 lane_id 变成编译期常量
不会改变 m_st + warp_id * 32 + lane_id 的运行期语义
不会把地址计算变成 per-thread 之外的共享操作
```

使用它之后，`m_thr` 更像一个编译期别名；在
`D[m_thr, n_st:n_st+BLK_N]` 出现的位置，会直接使用等价的表达式。

如果去掉 `T.meta_var`，写成：

```python
m_thr = m_st + warp_id * 32 + lane_id
```

通常仍然能得到相同的结果；区别是 parser 会按普通的 scalar 赋值建立
TIRx 的局部标量绑定。对于这段代码，`T.meta_var` 主要让语义保持为
标量表达式别名，并避免产生没有必要的临时 scalar binding。

下面的完整脚本不依赖 GPU，直接模拟 128 个线程如何覆盖 128 个 output
rows，并检查 `m_thr` 的边界：

文件：`check_m_thr_row_mapping.py`

```python
BLK_M = 128
BLK_N = 128

bx = 0
by = 0
m_st = bx * BLK_M
n_st = by * BLK_N

seen_rows = []

for warp_id in range(4):
    for lane_id in range(32):
        tid_in_wg = warp_id * 32 + lane_id
        m_thr = m_st + tid_in_wg
        seen_rows.append(m_thr)

        if (warp_id, lane_id) in {(0, 0), (0, 31), (2, 9), (3, 31)}:
            print(
                f"warp_id={warp_id}, lane_id={lane_id}, "
                f"tid_in_wg={tid_in_wg}, m_thr={m_thr}, "
                f"row=D[{m_thr}, {n_st}:{n_st + BLK_N}]"
            )

assert len(seen_rows) == 128
assert len(set(seen_rows)) == 128
assert min(seen_rows) == m_st
assert max(seen_rows) == m_st + BLK_M - 1

print(f"written output rows: {m_st}..{m_st + BLK_M - 1}")
print("all 128 rows are covered exactly once")
```

运行：

```bash
python3 check_m_thr_row_mapping.py
```

预期输出：

```text
warp_id=0, lane_id=0, tid_in_wg=0, m_thr=0, row=D[0, 0:128]
warp_id=0, lane_id=31, tid_in_wg=31, m_thr=31, row=D[31, 0:128]
warp_id=2, lane_id=9, tid_in_wg=73, m_thr=73, row=D[73, 0:128]
warp_id=3, lane_id=31, tid_in_wg=127, m_thr=127, row=D[127, 0:128]
written output rows: 0..127
all 128 rows are covered exactly once
```

这个脚本验证的是 row ownership 和覆盖范围，不执行真实的 `Tx.copy`。真实
GMEM store 还需要满足 `D` 的 shape、dtype、地址对齐和边界条件。

常见错误及症状：

| 错误 | 可观察症状 |
|---|---|
| 只写 `lane_id`，漏掉 `warp_id * 32` | warp 0 和 warp 1 都写 `D[0:32]`，后写覆盖先写，结果只保留部分行 |
| 漏掉 `m_st` | 多 CTA 时所有 tile 都写回全局 `D[0:128]`，不同 CTA 互相覆盖 |
| `m_st` 使用错误 tile 起点 | 每个局部 tile 数值可能正确，但整体输出呈现 tile 整体错位 |
| `m_thr` 的 row owner 与 `Dreg_wg` 不一致 | 行内容互换、重复行或有规律缺失 |
| `n_st` 计算错误 | 写入了合法地址但列区间属于相邻 output tile |
| 直接写 `n_st + BLK_N` 而不检查边界 | 当 `N` 不是 `BLK_N` 的整数倍时越界 |
| 让单个线程执行整句 `Tx.copy` | 只覆盖一行，不能形成完整的 128-row tile |

自测：

1. `m_thr` 的完整公式是什么？
   答：`m_thr = m_st + warp_id * 32 + lane_id`，也可写成
   `m_thr = m_st + tid_in_wg`。

2. 当 `m_st=128`、`warp_id=2`、`lane_id=9` 时，`m_thr` 是多少？
   答：`128 + 2*32 + 9 = 201`。

3. 为什么 128 个线程合起来能写完一个 `128 x BLK_N` tile？
   答：每个线程拥有一个 output row 的 `BLK_N` 个 columns；128 个线程
   分别负责 128 个不同的 row。

4. `T.meta_var` 会把 `m_thr` 变成运行期寄存器或指令吗？
   答：不会。它标记 parser-time meta value，让名字直接绑定表达式；运行期
   地址计算仍然由表达式本身描述。

5. 如果第 1 个 warp 和第 2 个 warp 都只使用 `lane_id` 计算 row，会出现
   什么症状？
   答：两个 warp 的 lane 会映射到相同 rows，产生重复写入和错误覆盖，
   最终 output 只有部分 row 来自正确 warp。

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

## 十一、`hgemm_v2` 与 `hgemm_v3` 的核心区别

```text
本次讲解位置
章节：chapter_gemm_basics
小节：从 K-Loop 扩展到 Spatial Tiling
知识点：hgemm_v2 与 hgemm_v3 的区别
上次：m_thr 如何把线程映射到全局输出行
下次：第 2 步：K-Loop 累加与 MMA barrier phase
PTX：global load、global store、tcgen05.mma、mbarrier.try_wait
```

先用一句话区分两个版本：

```text
hgemm_v2 增加 K 方向的 loop
hgemm_v3 在 v2 的基础上，增加 M/N 方向的 CTA spatial tiling
```

它们不是两个互不相关的版本。演进关系是：

```text
v1: 一个 CTA 计算一个 128 x 128 output tile，K 最大为 64
     |
     +-- v2 增加 K-loop，让 K 可以大于 64
              |
              +-- v3 增加 M/N grid，让一个 CTA 负责多个 output tiles 之一
```

因此，从 `v2` 到 `v3` 新增的关键机制只有：

```text
CTA 不再固定处理全局 D 的 (0:128, 0:128)

CTA (bx, by) 负责：
D[bx*BLK_M : (bx+1)*BLK_M,
  by*BLK_N : (by+1)*BLK_N]
```

### 11.1 最关键的代码差异

`v2` 的 load 永远从 A、B 的完整 leading dimension 开始：

```python
Tx.cta.copy(Asmem[:, :], A[:, i*BLK_K:(i+1)*BLK_K])
Tx.cta.copy(Bsmem[:, :], B[:, i*BLK_K:(i+1)*BLK_K])
```

这隐含了：

```text
A.shape[0] == BLK_M
B.shape[0] == BLK_N
```

因此 `v2` 只适合 `M=N=128` 的单 output tile。它虽然写了 `bx, by`，但
grid 的实际 shape 是 `1 x 1`，所以 `m_st=n_st=0`。

`v3` 的 load 增加了当前 CTA tile 的行起点：

```python
Tx.cta.copy(
    Asmem[:, :],
    A[m_st:m_st+BLK_M, i*BLK_K:(i+1)*BLK_K],
)
Tx.cta.copy(
    Bsmem[:, :],
    B[n_st:n_st+BLK_N, i*BLK_K:(i+1)*BLK_K],
)
```

其中：

```text
m_st = bx * BLK_M
n_st = by * BLK_N
```

这正是 `v3` 相对 `v2` 的本质变化。其他核心结构，包括 SMEM 分配、
`tcgen05.mma`、MMA barrier、TMEM layout、writeback 和线程映射，基本不变。

### 11.2 两个版本的完整代码

公共 imports 如下：

```python
import tvm
from tvm.script import tirx as T
from tvm.script.tirx import tile as Tx
from tvm.backend.cuda.tile_primitive.tma_utils import (
    mma_shared_layout,
    SwizzleMode,
)
from tvm.tirx.layout import TileLayout, S, TLane, TCol, tid_in_wg
```

#### `hgemm_v2`：加入 K-loop

```python
def hgemm_v2(M, N, K):
    a_type = tvm.DataType("float16")
    b_type = tvm.DataType("float16")
    d_type = tvm.DataType("float16")
    acc_type = tvm.DataType("float32")

    BLK_M, BLK_N, BLK_K = 128, 128, 64
    K_TILES = K // BLK_K

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

        tmem = T.decl_buffer(
            (128, 512),
            "float32",
            scope="tmem",
            allocated_addr=tmem_addr[0],
            layout=TileLayout(
                S[(128, 512) : (1 @ TLane, 1 @ TCol)]
            ),
        )

        phase_mma: T.int32 = 0
        m_st = T.meta_var(bx * BLK_M)
        n_st = T.meta_var(by * BLK_N)

        for i in T.serial(K_TILES):
            Tx.cta.copy(
                Asmem[:, :],
                A[:, i * BLK_K : (i + 1) * BLK_K],
            )
            Tx.cta.copy(
                Bsmem[:, :],
                B[:, i * BLK_K : (i + 1) * BLK_K],
            )

            T.cuda.cta_sync()

            if warp_id == 0:
                if T.ptx.elect_sync():
                    Tx.gemm_async(
                        tmem[:, :BLK_N],
                        Asmem[:, :],
                        Bsmem[:, :],
                        accum=(i != 0),
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
            phase_mma ^= 1

        Dreg = T.alloc_local((BLK_N,), acc_type)
        Dreg_f16 = T.alloc_local((BLK_N,), d_type)
        Dreg_wg = Dreg.view(
            128,
            BLK_N,
            layout=TileLayout(
                S[(128, BLK_N) : (1 @ tid_in_wg, 1)]
            ),
        )

        Tx.wg.copy_async(Dreg_wg[:, :], tmem[:, :BLK_N])
        T.ptx.tcgen05.wait.ld()
        Tx.cast(Dreg_f16[:], Dreg[:])

        m_thr = T.meta_var(m_st + warp_id * 32 + lane_id)
        Tx.copy(
            D[m_thr, n_st : n_st + BLK_N],
            Dreg_f16[:],
        )

        T.cuda.cta_sync()
        if warp_id == 0:
            T.ptx.tcgen05.relinquish_alloc_permit(cta_group=1)
            T.ptx.tcgen05.dealloc(
                tmem_addr[0],
                n_cols=512,
                cta_group=1,
            )

    return kernel
```

`hgemm_v2` 的合法运行边界是：

```text
M == 128
N == 128
K >= 64
K % 64 == 0
```

#### `hgemm_v3`：加入 M/N spatial tiling

```python
def hgemm_v3(M, N, K):
    a_type = tvm.DataType("float16")
    b_type = tvm.DataType("float16")
    d_type = tvm.DataType("float16")
    acc_type = tvm.DataType("float32")

    BLK_M, BLK_N, BLK_K = 128, 128, 64
    K_TILES = K // BLK_K

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

        tmem = T.decl_buffer(
            (128, 512),
            "float32",
            scope="tmem",
            allocated_addr=tmem_addr[0],
            layout=TileLayout(
                S[(128, 512) : (1 @ TLane, 1 @ TCol)]
            ),
        )

        phase_mma: T.int32 = 0
        m_st = T.meta_var(bx * BLK_M)
        n_st = T.meta_var(by * BLK_N)

        for i in T.serial(K_TILES):
            Tx.cta.copy(
                Asmem[:, :],
                A[
                    m_st : m_st + BLK_M,
                    i * BLK_K : (i + 1) * BLK_K,
                ],
            )
            Tx.cta.copy(
                Bsmem[:, :],
                B[
                    n_st : n_st + BLK_N,
                    i * BLK_K : (i + 1) * BLK_K,
                ],
            )

            T.cuda.cta_sync()

            if warp_id == 0:
                if T.ptx.elect_sync():
                    Tx.gemm_async(
                        tmem[:, :BLK_N],
                        Asmem[:, :],
                        Bsmem[:, :],
                        accum=(i != 0),
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
            phase_mma ^= 1

        Dreg = T.alloc_local((BLK_N,), acc_type)
        Dreg_f16 = T.alloc_local((BLK_N,), d_type)
        Dreg_wg = Dreg.view(
            128,
            BLK_N,
            layout=TileLayout(
                S[(128, BLK_N) : (1 @ tid_in_wg, 1)]
            ),
        )

        Tx.wg.copy_async(Dreg_wg[:, :], tmem[:, :BLK_N])
        T.ptx.tcgen05.wait.ld()
        Tx.cast(Dreg_f16[:], Dreg[:])

        m_thr = T.meta_var(m_st + warp_id * 32 + lane_id)
        Tx.copy(
            D[m_thr, n_st : n_st + BLK_N],
            Dreg_f16[:],
        )

        T.cuda.cta_sync()
        if warp_id == 0:
            T.ptx.tcgen05.relinquish_alloc_permit(cta_group=1)
            T.ptx.tcgen05.dealloc(
                tmem_addr[0],
                n_cols=512,
                cta_group=1,
            )

    return kernel
```

`hgemm_v3` 的合法运行边界是：

```text
M % 128 == 0
N % 128 == 0
K >= 64
K % 64 == 0
```

如果通过 Python 构造两个 TIRx kernel：

```python
kernel_v2 = hgemm_v2(128, 128, 128)
kernel_v3 = hgemm_v3(256, 256, 128)
```

`kernel_v2` 的 grid 是 `1 x 1`，`kernel_v3` 的 grid 是 `2 x 2`。
当前机器没有 Blackwell GPU，因此只能做 TIRx 构造和静态检查，不能真实
执行 `tcgen05.mma`。

### 11.3 逐项比较

| 项目 | `hgemm_v2` | `hgemm_v3` |
|---|---|---|
| 新增机制 | K 方向 loop | M/N 方向 spatial tiling |
| 解决的问题 | `K > 64` | `M > 128` 或 `N > 128` |
| CTA grid | 实际为 1 个 output tile | 每个 CTA 一个 output tile |
| A load | `A[:, k_chunk]` | `A[m_st:m_st+BLK_M, k_chunk]` |
| B load | `B[:, k_chunk]` | `B[n_st:n_st+BLK_N, k_chunk]` |
| `m_st` 使用位置 | 只用于 writeback，实际为 0 | load 和 writeback 都使用 |
| `n_st` 使用位置 | 只用于 writeback，实际为 0 | load 和 writeback 都使用 |
| K 累加 | 有 | 有，沿用 v2 |
| TMEM accumulator | 每个 K chunk 都更新同一个 TMEM tile | 每个 CTA 有独立 TMEM accumulator |
| MMA 与 barrier 协议 | 相同 | 相同 |
| writeback 公式 | 相同 | 相同 |
| 主要限制 | 只支持 `M=N=128` | 要求 M、N 都能被 128 整除 |

可以用一句话记忆：

```text
v2 遍历 K，重复使用当前 output tile 的 accumulator。
v3 遍历 CTA grid，让不同 CTA 计算不同 output tile。
```

### 11.4 用一个具体坐标追踪

取：

```text
M = 256
N = 256
K = 128
BLK_M = 128
BLK_N = 128
BLK_K = 64
```

此时：

```text
grid = [M / BLK_M, N / BLK_N]
     = [256 / 128, 256 / 128]
     = [2, 2]

K_TILES = K / BLK_K
        = 128 / 64
        = 2
```

这个 shape 不能由 `hgemm_v2` 正确处理，因为它会尝试把完整的
`A[:, k_chunk]` 和 `B[:, k_chunk]` 复制到只有 `128 x 64` 的 SMEM tile。

对于 `hgemm_v3` 的：

```text
bx = 1
by = 1
i = 1
```

先算 CTA tile offset：

```text
m_st = bx * BLK_M
     = 1 * 128
     = 128

n_st = by * BLK_N
     = 1 * 128
     = 128
```

第二个 K chunk 的 K 区间是：

```text
k_start = i * BLK_K
        = 1 * 64
        = 64

k_end = (i + 1) * BLK_K
      = 2 * 64
      = 128
```

于是 load 对应：

```text
Asmem[:, :] <- A[128:256, 64:128]
Bsmem[:, :] <- B[128:256, 64:128]
```

对线程 `warp_id=2, lane_id=9`：

```text
m_thr = m_st + warp_id * 32 + lane_id
      = 128 + 2 * 32 + 9
      = 201
```

最终写回：

```text
D[201, 128:256] <- Dreg_f16[0:128]
```

如果 `v3` 漏掉 A、B load 中的 `m_st`/`n_st`，它会错误地读取全局 tile
`(0, 0)`；如果只给 load 加 offset、writeback 仍使用旧的 `m_thr`，它又会
把计算结果写到错误 output tile。

下面是一个不依赖 GPU 的 offset 检查脚本，分别模拟单 tile K-loop 和
multi-CTA spatial tiling：

文件：`check_v2_v3_tile_offsets.py`

```python
BLK_M = 128
BLK_N = 128
BLK_K = 64


def v2_load_slice(bx, by, i):
    m_st = bx * BLK_M
    n_st = by * BLK_N
    return {
        "m_st": m_st,
        "n_st": n_st,
        "A": (slice(None), slice(i * BLK_K, (i + 1) * BLK_K)),
        "B": (slice(None), slice(i * BLK_K, (i + 1) * BLK_K)),
    }


def v3_load_slice(bx, by, i):
    m_st = bx * BLK_M
    n_st = by * BLK_N
    return {
        "m_st": m_st,
        "n_st": n_st,
        "A": (
            slice(m_st, m_st + BLK_M),
            slice(i * BLK_K, (i + 1) * BLK_K),
        ),
        "B": (
            slice(n_st, n_st + BLK_N),
            slice(i * BLK_K, (i + 1) * BLK_K),
        ),
    }


for name, load in [
    ("v2", v2_load_slice(0, 0, 1)),
    ("v3", v3_load_slice(1, 1, 1)),
]:
    print(
        f"{name}: m_st={load['m_st']}, n_st={load['n_st']}, "
        f"A={load['A']}, B={load['B']}"
    )

load = v3_load_slice(1, 1, 1)
warp_id = 2
lane_id = 9
m_thr = load["m_st"] + warp_id * 32 + lane_id
print(f"v3 writeback: D[{m_thr}, 128:256]")

assert load["m_st"] == 128
assert load["n_st"] == 128
assert m_thr == 201
```

运行：

```bash
python3 check_v2_v3_tile_offsets.py
```

预期输出：

```text
v2: m_st=0, n_st=0, A=(slice(None, None, None), slice(64, 128, None)), B=(slice(None, None, None), slice(64, 128, None))
v3: m_st=128, n_st=128, A=(slice(128, 256, None), slice(64, 128, None)), B=(slice(128, 256, None), slice(64, 128, None))
v3 writeback: D[201, 128:256]
```

### 11.5 常见错误及症状

| 错误 | 可观察症状 |
|---|---|
| 把 `v3` 的改动理解成又加了一层 K-loop | 重复处理 K，数值被重复累加 |
| `v3` 的 A load 漏掉 `m_st` | 所有 CTA 都读取 A 的前 128 行 |
| `v3` 的 B load 漏掉 `n_st` | 所有 CTA 都读取 B 的前 128 行 |
| `m_st`/`n_st` 只用于 load，不用于 writeback | 多个 CTA 把结果写到同一个 output tile |
| writeback 只加 `m_st`，漏掉 `n_st` | row 正确但 output columns 都属于第一列 tile |
| 把 `bx` 和 `by` 互换 | M/N tile 完成转置式搬运，边界合法但结果错误 |
| `v2` 直接拿 `M=256` 运行 | A/B 的 source tile shape 与 `Asmem/Bsmem` 不匹配 |
| 没有 tail 处理却使用非整除 M/N/K | 最后一个 tile 越界或结果不完整 |

自测：

1. 从 `hgemm_v2` 到 `hgemm_v3`，新增的是 K 方向还是 M/N 方向？
   答：新增 M/N 方向的 spatial tiling；K-loop 在 `v2` 中已经存在。

2. `v2` 和 `v3` 最直接的 load 代码差异是什么？
   答：`v3` 在 A、B slice 中分别加入 `m_st` 和 `n_st`，`v2` 没有。

3. 当 `bx=1, by=1, BLK_M=BLK_N=128` 时，`m_st` 和 `n_st` 是多少？
   答：`m_st=128`，`n_st=128`。

4. `v3` 中 `warp_id=2, lane_id=9, m_st=128` 时，写回哪一行？
   答：`m_thr = 128 + 2*32 + 9 = 201`，写回 `D[201, ...]`。

5. 为什么 `v3` 的 MMA 和 barrier 协议可以沿用 `v2`？
   答：Spatial tiling 改变的是每个 CTA 读取和写回的全局地址；在一个 CTA
   内部，K 累加和 TMEM accumulator 的生命周期没有改变。

## 十二、当前进度

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
T.SMEMPool 的 alloc 推进 cursor，move_base_to 控制后续 base offset
pool.commit 确定 shared.dyn 的最终 high-water-mark 大小，它不是运行时 barrier
Dreg 是每个 thread 的 (BLK_N,) local storage，分配时不会出现每线程 128 x BLK_N
Dreg_wg 是同一份 local storage 的 warpgroup distributed view，不分配也不复制数据
Dreg_wg 用 tid_in_wg 将 128 个逻辑 rows 映射到 warpgroup 的 128 个 owner threads
view 的 row 决定 owner thread，col 决定该线程的 local register index
Tx.wg.copy_async 读取 TMEM，tcgen05.wait.ld 等待 register load 完成
tcgen05.ld 是异步的，tcgen05.wait::ld 等待当前 thread 的 prior loads 完成
wait::ld.sync.aligned 只做 warp 内汇合，不是 CTA 或 warpgroup barrier
cta_sync 不能替代 wait.ld，因为 thread 到达不等于 TMEM load 已完成
writeback 由 wait.ld、fp32-to-fp16 cast 和按 m_thr 写回 GMEM 组成
每个 thread 将自己的 fp32 row cast 为 fp16，再写回 GMEM
m_thr = m_st + warp_id * 32 + lane_id 选择当前 thread 的全局 output row
一个 thread 写一整行的 BLK_N 个 columns，128 个 threads 合起来覆盖 output tile
T.meta_var 将 m_thr 绑定为 parser-time 表达式别名，不生成额外运行期指令
TMEM 必须先 cta_sync，再 relinquish permit 和 dealloc
第 1 步的限制是 K<=64、M=N=128、同步 copy、搬运与计算不重叠
hgemm_v2 通过 K_TILES serial loop 允许 K 大于 BLK_K
hgemm_v3 在 v2 的 K-loop 上加入 M/N 二维 CTA tiling
v2 只在 writeback 使用 m_st/n_st，实际 tile 仍是 0
v3 在 A/B load 和 writeback 中都使用 m_st/n_st
从 v2 到 v3 不改变 CTA 内部的 MMA 与 mbarrier 协议
chapter_gemm_basics 第 1 个知识点完成：单 Tile Baseline
```

下一知识点：

```text
chapter_gemm_basics
-> 第 2 步：K-Loop 累加与 MMA barrier phase
```
