# TIRx 入门：第一个单 tile GEMM Kernel

这篇笔记只完成一个目标：把一个 `128 x 128 x 64` 的单 tile GEMM
从 TIRx 源码一路讲到 CUDA 执行结果。读完后，你应该能重建完整数据
路径，并解释 Scope、Layout、Dispatch 为什么共同决定程序行为。

## 本次讲解位置

```text
本次讲解位置
章节：chapter_intro_tirx
小节：第一个 TIRx Kernel
知识点：单 tile GEMM 的完整数据路径，以及 Scope / Layout / Dispatch
上次：chapter_clc 完成
下次：chapter_tirx_layout_api -> TileLayout 的 S、R 与 offset
PTX：tcgen05.alloc、tcgen05.mma、tcgen05.commit、tcgen05.ld、mbarrier.try_wait
```

前面已经分别学过数据布局、TMA、mbarrier、TMEM 和 `tcgen05`。这一节
第一次把这些机制放进同一个 kernel，并让编译器从 tile-level 操作
lowering 到具体硬件指令。

## 一、本节要解决的问题

Kernel 计算：

```text
D = A x B^T
```

Shape 和调用参数是：

```text
A: 128 x 64   float16
B: 128 x 64   float16
D: 128 x 128  float16
accumulator: 128 x 128 float32 in TMEM
```

`B` 以 `(N, K)` 存放。Kernel 不显式构造 `B^T`，而是让 MMA 的
operand layout 把 `B[n, k]` 解释成矩阵乘法所需的右操作数。

当前 grid 只有一个 CTA：

```text
M / BLK_M = 128 / 128 = 1
N / BLK_N = 128 / 128 = 1
grid = 1 x 1
```

所以本次执行时：

```text
bx = 0
by = 0
m_st = 0
n_st = 0
```

这不是最终的高性能 GEMM。它暂时没有多 K tile、流水线、TMA load 和
warp specialization。它是最小完整路径，后续版本会在这个骨架上扩展。

### 心智模型

不要先把 TIRx 看成一组 API。可以把它看成为硬件执行计划补齐三类信息：

| 信息 | 回答的问题 | 本 kernel 中的例子 |
|---|---|---|
| Scope | 谁执行这项操作？ | CTA copy、warpgroup copy、单线程 MMA issue |
| Layout | 每个逻辑元素放在哪里？ | SMEM swizzle、`TLane`、`TCol`、`tid_in_wg` |
| Dispatch | 使用哪条硬件路径？ | `dispatch="tcgen05"` |

一项 tile 操作只有在 Scope、Layout、Dispatch 都确定后，编译器才能生成
确定的 thread-level control flow、地址计算和硬件指令。

异步 kernel 还要始终检查三项契约：

```text
1. 生产者写出的布局，等于消费者读取时假设的布局。
2. 读取数据前，producer 到 consumer 的同步已经完成。
3. 释放 TMEM 或复用 SMEM 前，最后一位 consumer 已经完成读取。
```

### 数据路径

```text
A/B: GMEM -> SMEM -> tcgen05.mma
D:   tcgen05.mma -> TMEM -> registers -> cast fp16 -> GMEM
```

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

两个最容易记错的点是：

```text
SMEM 是 A/B 的 operand staging area，不是最终计算缓存。
TMEM 是 tcgen05.mma 的 accumulator 位置，不是用户最终可见的 D。
```

## 二、完整可运行代码

下面给出两个完整文件。把它们放在同一目录中即可：

```text
tirx_hgemm.py
verify_tirx_hgemm.py
```

它们不依赖临时课程 clone，也不依赖未展示的定义。

### 文件一：`tirx_hgemm.py`

```python
import tvm
from tvm.script import tirx as T
from tvm.script.tirx import tile as Tx
from tvm.backend.cuda.tile_primitive.tma_utils import mma_shared_layout, SwizzleMode
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

        # 当前调用中 grid 为 1x1，因此 bx=by=0，两个起点也都是 0。
        bx, by = T.cta_id([M // BLK_M, N // BLK_N])
        wg_id = T.warpgroup_id([1])
        warp_id = T.warp_id_in_wg([4])
        lane_id = T.lane_id([32])

        # 申请 CTA 内部的 SMEM 元数据和 A/B tile。
        pool = T.SMEMPool()
        tmem_addr = pool.alloc((1,), "uint32")
        mma_bar = pool.alloc((1,), "uint64", align=8)
        pool.move_base_to(1024)
        Asmem = pool.alloc((BLK_M, BLK_K), a_type, layout=A_layout)
        Bsmem = pool.alloc((BLK_N, BLK_K), b_type, layout=B_layout)
        pool.commit()

        # warp 0 初始化 barrier 并申请 TMEM。
        if warp_id == 0:
            if lane_id == 0:
                T.ptx.mbarrier.init(mma_bar.ptr_to([0]), 1)
            T.ptx.tcgen05.alloc(
                T.address_of(tmem_addr),
                n_cols=512,
                cta_group=1,
            )

        # 把初始化结果发布给整个 CTA。
        T.ptx.fence.proxy_async("shared::cta")
        T.ptx.fence.mbarrier_init()
        T.cuda.cta_sync()

        tmem = T.decl_buffer(
            (128, 512),
            "float32",
            scope="tmem",
            allocated_addr=tmem_addr[0],
            layout=TileLayout(S[(128, 512) : (1 @ TLane, 1 @ TCol)]),
        )

        m_st = T.meta_var(bx * BLK_M)
        n_st = T.meta_var(by * BLK_N)
        phase_mma: T.int32 = 0

        # CTA scope: 全部 128 threads 协作把 A/B 搬进 SMEM。
        Tx.cta.copy(Asmem[:, :], A[m_st : m_st + BLK_M, :])
        Tx.cta.copy(Bsmem[:, :], B[n_st : n_st + BLK_N, :])
        T.cuda.cta_sync()

        # Elected-thread scope: 只由一个 active thread 发出 tile GEMM。
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

        # 等待 tcgen05 完成事件在 mbarrier 上产生 arrival。
        T.ptx.mbarrier.try_wait(mma_bar.ptr_to([0]), phase_mma)

        # Warpgroup scope: TMEM accumulator -> per-thread registers。
        Dreg = T.alloc_local((BLK_N,), acc_type)
        Dreg_f16 = T.alloc_local((BLK_N,), d_type)
        Dreg_wg = Dreg.view(
            128,
            BLK_N,
            layout=TileLayout(S[(128, BLK_N) : (1 @ tid_in_wg, 1)]),
        )
        Tx.wg.copy_async(Dreg_wg[:, :], tmem[:, :BLK_N])
        T.ptx.tcgen05.wait.ld()
        Tx.cast(Dreg_f16[:], Dreg[:])

        m_thr = T.meta_var(m_st + warp_id * 32 + lane_id)
        Tx.copy(D[m_thr, n_st : n_st + BLK_N], Dreg_f16[:])

        # 所有 consumer 完成后，warp 0 释放 TMEM。
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

### 文件二：`verify_tirx_hgemm.py`

```python
import torch
import tvm

from tirx_hgemm import hgemm_v1


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("This lesson requires a CUDA GPU.")

    device_index = torch.cuda.current_device()
    device_name = torch.cuda.get_device_name(device_index)
    capability = torch.cuda.get_device_capability(device_index)
    print(f"device: {device_name}, capability={capability}")

    if capability[0] != 10:
        raise RuntimeError(
            "This TIRx example requires a Blackwell sm_100a GPU, "
            f"but capability is {capability}."
        )

    torch.manual_seed(0)
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

    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    A_tensor = torch.randn(M, K, dtype=torch.float16, device=device)
    B_tensor = torch.randn(N, K, dtype=torch.float16, device=device)
    D_tensor = torch.zeros(M, N, dtype=torch.float16, device=device)

    ex.mod(A_tensor, B_tensor, D_tensor)

    D_ref = (
        A_tensor.float()
        @ B_tensor.float().transpose(0, 1)
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


if __name__ == "__main__":
    main()
```

## 三、阶段级数据路径

这张表先回答每个阶段“谁做、碰什么、怎么同步、错后会怎样”。后面的
逐段解释会把这些关系映射回源码。

| 阶段 | 执行者 | 资源 | 关键 Layout 或所有权 | 同步 | 典型错误症状 |
|---|---|---|---|---|---|
| thread 坐标 | 每个 thread | 无 | CTA/warpgroup/warp/lane 坐标 | 无 | 写回行号错乱 |
| SMEM 分配 | CTA 源码描述 | SMEM | pool 内偏移与 swizzle | 无 | alias、越界或描述符不匹配 |
| TMEM 分配 | warp 0 | TMEM | 512 columns 归当前 CTA | `cta_sync` | 使用未分配或重复分配 |
| barrier 初始化 | warp 0 lane 0 | mbarrier | arrival count = 1 | fence + `cta_sync` | 等待永不完成 |
| A/B load | 全部 128 threads | GMEM -> SMEM | destination layout 驱动地址 | load 后 `cta_sync` | 偶发读到旧值 |
| MMA issue | warp 0 的 elected thread | SMEM -> TMEM | swizzle 与 descriptor 匹配 | `tcgen05.commit` | 少发、多发或结果整块错 |
| MMA completion | 硬件异步代理 | mbarrier | 完成事件产生 arrival | `mbarrier.try_wait` | 直接读取时拿到未完成结果 |
| TMEM load | 一个 warpgroup | TMEM -> registers | `row -> tid_in_wg` | `tcgen05.wait.ld` | register 已读但 load 未完成 |
| cast 和写回 | 每 thread 一行 | registers -> GMEM | `m_thr -> row` | 依赖 `wait.ld` | 行错位或 dtype 错 |
| TMEM 释放 | warp 0 | TMEM | 所有 reader 已完成 | 先 `cta_sync` | use-after-release |

## 四、代码逐段映射

### 4.1 thread 坐标系

```python
bx, by = T.cta_id([M // BLK_M, N // BLK_N])
wg_id = T.warpgroup_id([1])
warp_id = T.warp_id_in_wg([4])
lane_id = T.lane_id([32])
```

当前 kernel 只有一个 CTA，一个 warpgroup，一个 warpgroup 内有 4 个
warps，每个 warp 有 32 个 lanes：

| 表达式 | 范围 | 本 kernel 中的作用 |
|---|---:|---|
| `bx` | 0 | output tile 在 M 维的 CTA 坐标 |
| `by` | 0 | output tile 在 N 维的 CTA 坐标 |
| `wg_id` | 0 | 当前只有一个 warpgroup |
| `warp_id` | 0..3 | 决定 MMA issue、写回行和释放操作 |
| `lane_id` | 0..31 | warp 内 thread 编号 |

CTA 内总 thread 数：

```text
4 warps x 32 lanes = 128 threads
```

### 4.2 SMEM pool 与 operand layout

```python
pool = T.SMEMPool()
tmem_addr = pool.alloc((1,), "uint32")
mma_bar = pool.alloc((1,), "uint64", align=8)
pool.move_base_to(1024)
Asmem = pool.alloc((BLK_M, BLK_K), a_type, layout=A_layout)
Bsmem = pool.alloc((BLK_N, BLK_K), b_type, layout=B_layout)
pool.commit()
```

`T.SMEMPool` 显式规划 CTA 内的 shared memory：

| 对象 | 大小或作用 |
|---|---|
| `tmem_addr` | 保存 `tcgen05.alloc` 返回的 TMEM 地址，不是 TMEM 本体 |
| `mma_bar` | MMA 完成事件使用的 mbarrier |
| `Asmem` | `128 x 64` fp16 operand |
| `Bsmem` | `128 x 64` fp16 operand |

两个 operand tile 各占：

```text
128 x 64 x 2 bytes = 16384 bytes = 16 KiB
```

两者合计 `32 KiB`。`move_base_to(1024)` 把后续数据 tile 的基础位置
移动到 1024-byte 边界之后，给前面的元数据留出空间。

`A_layout` 和 `B_layout` 使用 128-byte swizzle。生产者
`Tx.cta.copy` 写 SMEM 和消费者 `tcgen05.mma` 读取 descriptor 时必须
使用同一套布局。若只改变一边，程序通常不会报地址越界，而是得到
有规律的错误数值。

### 4.3 barrier 与 TMEM 初始化

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

barrier 的 arrival count 是 1。后续不是由软件线程手工 arrival，而是
让 MMA completion 事件在它上面产生一次 arrival。

`tcgen05.alloc` 申请 512 TMEM columns，`cta_group=1` 表示 allocation
只属于当前 CTA。地址被写入 `tmem_addr`，但该 buffer 位于 SMEM。

```python
T.ptx.fence.proxy_async("shared::cta")
T.ptx.fence.mbarrier_init()
T.cuda.cta_sync()
```

这里解决三种可见性：

| 操作 | 解决的问题 |
|---|---|
| `fence.proxy_async("shared::cta")` | 让 generic proxy 的 SMEM 写入对 async proxy 可见 |
| `fence.mbarrier_init()` | 发布 barrier 初始化 |
| `cta_sync()` | 保证所有 thread 在初始化完成后再继续 |

### 4.4 把 TMEM 地址解释成 tile

```python
tmem = T.decl_buffer(
    (128, 512),
    "float32",
    scope="tmem",
    allocated_addr=tmem_addr[0],
    layout=TileLayout(S[(128, 512) : (1 @ TLane, 1 @ TCol)]),
)
```

当前映射最简单：

```text
tmem[row, col] -> TMEM Lane = row
                 TMEM Column = col
```

例如：

```text
tmem[73, 91] -> TMEM Lane 73, TMEM Column 91
```

虽然 allocation 有 512 columns，这个 kernel 只把前 128 columns 当作
输出 accumulator：

```python
tmem[:, :BLK_N]
```

其中 `BLK_N = 128`。

### 4.5 GMEM 到 SMEM load

```python
m_st = T.meta_var(bx * BLK_M)
n_st = T.meta_var(by * BLK_N)

Tx.cta.copy(Asmem[:, :], A[m_st : m_st + BLK_M, :])
Tx.cta.copy(Bsmem[:, :], B[n_st : n_st + BLK_N, :])
T.cuda.cta_sync()
```

`Tx.cta.copy` 的 scope 是整个 CTA。128 个 threads 一起完成 A/B
搬运。当前示例中 `m_st = 0`、`n_st = 0`。

后面的 `cta_sync()` 是 load 和 MMA 之间的 ownership handoff：

```text
load producers 完成写入
-> CTA 内所有 threads 到达
-> MMA 才可以读取 SMEM
```

如果这个同步缺失，小输入或不稳定调度下可能偶尔看起来正确，但 MMA
可能读取尚未写完的 SMEM，表现为偶发错误。

### 4.6 发出 MMA

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
        T.ptx.tcgen05.commit(
            mma_bar.ptr_to([0]),
            cta_group=1,
        )
```

Scope 有两层：

```text
warp_id == 0
-> 只看 warp 0

elect_sync()
-> 在 warp 0 中选出一个 active thread
```

最终 scope 是：

```text
1 个 elected thread 发出整个 tile GEMM
```

“一个 thread 发指令”不等于“只有一个 thread 的数据参与计算”。Tensor
Core 和 TMEM 会按照 `tcgen05.mma` 的硬件 scope 读取 operand 并写
accumulator。

`Tx.gemm_async` 描述的是完整的 `128 x 128 x 64` tile GEMM。
`dispatch="tcgen05"` 要求编译器选择 Blackwell 路径。当前
`tcgen05.mma` 每次处理 16 个 K 元素，因此 lowering 会生成：

```text
64 / 16 = 4 次 MMA
```

高层只有一个 `gemm_async`，底层是 4 个 K step。第一次写新的
accumulator，后三步继续累加。

### 4.7 等待 MMA 完成

```python
phase_mma: T.int32 = 0
T.ptx.mbarrier.try_wait(mma_bar.ptr_to([0]), phase_mma)
```

`tcgen05.commit` 不是普通的软件 arrival，它把此前发出的异步 MMA
完成事件关联到 `mma_bar`：

```text
tcgen05.mma 完成
-> 硬件 completion event
-> mma_bar 产生 arrival
-> try_wait 观察到 phase 完成
```

issue 和 complete 是两件不同的事：

```text
issue: elected thread 发出 MMA
complete: 全部依赖结果的 thread 等待 mbarrier phase
```

当前 kernel 只执行一次 MMA group，所以使用初始 phase 0。若以后加入
K-loop，每次完成一个新 phase 后必须同步更新 phase 的奇偶值，不能
一直等待同一个 phase。

### 4.8 TMEM 到 registers

```python
Dreg = T.alloc_local((BLK_N,), acc_type)
Dreg_f16 = T.alloc_local((BLK_N,), d_type)
Dreg_wg = Dreg.view(
    128,
    BLK_N,
    layout=TileLayout(S[(128, BLK_N) : (1 @ tid_in_wg, 1)]),
)
Tx.wg.copy_async(Dreg_wg[:, :], tmem[:, :BLK_N])
T.ptx.tcgen05.wait.ld()
```

每个 thread 拥有 128 个 fp32 local registers。`Dreg_wg` 把这些
per-thread registers 组织成一个逻辑 tile：

```text
logical row -> tid_in_wg
logical col -> register index
```

因此：

```text
tid_in_wg = warp_id * 32 + lane_id
```

`Tx.wg.copy_async` 的 scope 是一个完整 warpgroup。它把一个
`128 x 128` accumulator tile 分配到 128 个 threads。它是异步 issue，
所以紧接着必须执行：

```python
T.ptx.tcgen05.wait.ld()
```

否则 `Dreg` 可能在 load 尚未完成时就被读取。

### 4.9 cast 与 GMEM writeback

```python
Tx.cast(Dreg_f16[:], Dreg[:])

m_thr = T.meta_var(m_st + warp_id * 32 + lane_id)
Tx.copy(D[m_thr, n_st : n_st + BLK_N], Dreg_f16[:])
```

计算 precision path 是：

```text
fp16 A/B -> fp32 accumulator in TMEM -> fp16 D in GMEM
```

每个 thread 负责输出矩阵的一行和该行的 128 个 columns：

```text
m_thr = m_st + warp_id * 32 + lane_id
```

这个公式必须与 `Dreg_wg` 的 `tid_in_wg` row mapping 一致。任一边改变
映射，而另一边没改，就会出现行置换或部分行错误。

### 4.10 释放 TMEM

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

`cta_sync()` 保证所有 thread 都已完成 TMEM read。随后 warp 0：

```text
放弃后续 TMEM allocation 许可
释放此前的 512 columns
```

如果 dealloc 提前发生，其他 warps 仍可能访问已经释放的 TMEM，
表现为非法访问、未定义结果，或者错误随调度偶发变化。

## 五、具体执行追踪：`D[73, 91]`

本次执行没有多 CTA：

```text
bx = 0
by = 0
m_st = 0
n_st = 0
```

选择输出元素 `D[73, 91]`。它在当前 TMEM layout 中的位置是：

```text
row 73 -> TMEM Lane 73
col 91 -> TMEM Column 91
```

计算过程可以写成：

```text
D[73, 91] =
    sum(k=0..63) A[73, k] * B[91, k]
```

因为 `tcgen05.mma` 每步处理 16 个 K，所以这一项由 4 个 partial sums
组成：

| MMA step | K range | 累加动作 |
|---:|---:|---|
| 0 | 0..15 | 计算 `S0`，以 `accum=False` 写入 TMEM |
| 1 | 16..31 | 把 `S1` 累加到当前 accumulator |
| 2 | 32..47 | 把 `S2` 累加到当前 accumulator |
| 3 | 48..63 | 把 `S3` 累加到当前 accumulator |

形式化写为：

```text
S0 = sum(k=0..15)  A[73, k] * B[91, k]
S1 = sum(k=16..31) A[73, k] * B[91, k]
S2 = sum(k=32..47) A[73, k] * B[91, k]
S3 = sum(k=48..63) A[73, k] * B[91, k]
```

下面是一组只用于解释累加顺序的合成 partial sums。它们不是随机输入
实际产生的数值：

```text
S0 = 10.0
S1 = 20.0
S2 = 30.0
S3 = 40.0
```

fp32 accumulator 的变化是：

```text
step 0: acc = 10.0
step 1: acc = 10.0 + 20.0 = 30.0
step 2: acc = 30.0 + 30.0 = 60.0
step 3: acc = 60.0 + 40.0 = 100.0
```

最终：

```text
fp32 accumulator = 100.0
cast to fp16   = 100.0
D[73, 91]      = 100.0
```

同一个元素从 TMEM 到 GMEM 的完整坐标路径是：

| 步骤 | 表达式 | 结果 | 物理含义 |
|---|---|---:|---|
| 输出坐标 | `row=73, col=91` | `(73, 91)` | 逻辑 D 坐标 |
| TMEM | `(1 @ TLane, 1 @ TCol)` | `(73, 91)` | TMEM Lane 73, Column 91 |
| warp | `73 // 32` | `2` | warp 2 负责该 row |
| lane | `73 % 32` | `9` | lane 9 负责该 row |
| thread | `2 * 32 + 9` | `73` | `tid_in_wg = 73` |
| register | logical col 91 | `Dreg[91]` | 该 thread 的第 91 个 fp32 register |
| writeback | `m_thr = 0 + 73` | `73` | 写入 `D[73, 0:128]` 中 |

所以 `D[73, 91]` 的最终路径是：

```text
D[73, 91]
-> TMEM[Lane 73, Col 91]
-> warp 2 / lane 9 / tid_in_wg 73
-> Dreg[91]
-> cast 为 fp16
-> D[73, 91] in GMEM
```

## 六、常见错误与可观察症状

| 错误 | 发生的层次 | 可观察症状 | 优先检查 |
|---|---|---|---|
| load 后漏掉 `cta_sync()` | 同步 | 偶发错误，通常在更大 tile 或负载下更明显 | A/B producer 到 MMA 的 handoff |
| SMEM swizzle 与 descriptor 不一致 | Layout | 整块 tile 错位，错误呈现规律性 | `mma_shared_layout`、dispatch operand |
| `elect_sync()` scope 放错 | Scope | 重复 MMA、部分 CTA 未同步或结果不稳定 | MMA 发起条件 |
| barrier 初始化后没有 fence | 同步 | wait 卡住或观察到旧状态 | `fence.mbarrier_init()`、`cta_sync()` |
| `phase_mma` 永远使用 0 | 同步 | 第一轮可能正确，进入下一轮后挂起 | K-loop 中的 phase 奇偶切换 |
| `tcgen05.wait.ld()` 漏掉 | 异步 load | registers 偶尔包含旧值或半完成值 | TMEM -> register 的完成等待 |
| `Dreg_wg` 与 `m_thr` 映射不一致 | Layout | 值基本正确但行被置换或部分行错误 | `tid_in_wg` 和 writeback formula |
| dealloc 早于最后 reader | 生命周期 | illegal access、runner crash 或随机错误 | 最后的 `cta_sync()` |
| `dispatch="tcgen05"` 在错误架构运行 | Dispatch/硬件 | 编译或 launch 失败，非法指令 | CUDA capability 是否满足 `sm_100a` |
| 浮点结果略有不同 | 数值 | 所有元素误差很小，没有规律错位 | `assert_close` 的 `rtol` / `atol` |

排查顺序建议固定为：

```text
1. 先判断是编译问题、执行问题，还是数值问题。
2. 数值整块错位，先看 Layout 和 descriptor。
3. 数值偶发变化，先看同步、phase 和生命周期。
4. 行或列有规律置换，先看 tid_in_wg 与 writeback。
5. 所有元素都只有小误差，最后再考虑浮点累加顺序。
```

## 七、编译、运行与预期输出

### 环境

课程环境是 Blackwell GPU、CUDA、PyTorch 和 TIRx：

```bash
uv venv --python 3.11 .venv
source .venv/bin/activate
uv pip install apache-tvm==0.26.0 cuda-bindings torch
python -c "import torch, tvm, tvm.tirx; print(tvm.__version__)"
```

预期能够打印 TVM 版本：

```text
0.26.0
```

这一步只验证 Python 包可导入，不验证：

```text
机器一定有 CUDA GPU
GPU 架构一定是 sm_100a
tcgen05 指令一定能运行
```

### 运行验证

把两个 Python 文件放在同一目录，然后执行：

```bash
python verify_tirx_hgemm.py
```

在满足要求的 Blackwell 机器上，输出形态应为：

```text
device: NVIDIA B200, capability=(10, 0)
Max error vs torch reference: 0.001953
PASS
```

`Max error` 的精确值会随 GPU、PyTorch 版本和浮点累加顺序变化。
稳定的成功后条件是：

```text
程序打印 PASS
```

如果 Max error 很小、没有行列置换，也没有偶发变化，差异通常来自
fp32 累加顺序和最终 round 到 fp16。不应使用严格逐位相等判断。

### 当前机器的执行边界

当前学习机器是 Apple M5 Pro，没有 CUDA，也没有 Blackwell GPU。因此
本机只能完成：

```text
静态阅读完整源码
执行 Python 语法检查
理解 TIRx lowering 到 tcgen05 的路径
```

本机不能完成：

```text
运行 sm_100a 代码
执行 tcgen05.mma
验证真实 GPU 数值
测量性能
```

可以在本机做无 GPU 的语法检查：

```bash
python -m py_compile tirx_hgemm.py verify_tirx_hgemm.py
```

预期没有输出，并且命令退出码为 0。

## 八、自测题

### 1. 为什么 A/B 要先进入 SMEM，而不能让当前 kernel 直接从 GMEM 做 MMA？

答：`tcgen05.mma` 在本示例中消费的是 SMEM operand，并通过矩阵
descriptor 按所需布局读取数据。SMEM 是 operand staging area，
swizzle 用来满足 Tensor Core 读取形式。GMEM 直接作为该 MMA 的
operand 不属于这条路径。

### 2. 这个 kernel 中三次关键操作的 scope 分别是什么？

答：

```text
Tx.cta.copy:       整个 CTA，128 threads
Tx.gemm_async:     1 个 elected thread issue
Tx.wg.copy_async:  一个完整 warpgroup
```

### 3. `D[73, 91]` 如何到达对应的 register？

答：

```text
D row 73 -> TMEM Lane 73
D col 91 -> TMEM Column 91
warp = 73 // 32 = 2
lane = 73 % 32 = 9
tid_in_wg = 2 * 32 + 9 = 73
register = Dreg[91]
```

### 4. 为什么 `Tx.gemm_async` 描述的 K=64 GEMM 会变成 4 次 MMA？

答：当前 `tcgen05.mma` 每次处理 16 个 K 元素：

```text
64 / 16 = 4
```

第一次用 `accum=False` 写入新的 accumulator，后续三步在相同 TMEM
accumulator 上继续累加。

### 5. 输出数值大多正确，但 row 2 和 row 34 的结果互换，最应该先检查什么？

答：先检查 row 的 ownership mapping。重点是 `Dreg_wg` 中的
`tid_in_wg` layout 和 writeback 的：

```text
m_thr = m_st + warp_id * 32 + lane_id
```

如果 mapping 一边改变而另一边未变，通常会产生有规律的 row 置换；
这通常不是浮点误差。

## 九、当前进度

`chapter_intro_tirx` 的知识点：

```text
[x] 第一个 TIRx Kernel
[x] 编译并验证结果
```

已经覆盖：

```text
TIRx 是使用 threads、SMEM、TMEM、barrier 与 Tensor Core 概念的 Python DSL
Scope / Layout / Dispatch 是 tile 操作的三项核心语义
单 tile GEMM 计算 D = A x B^T
A/B 数据路径：GMEM -> SMEM -> tcgen05.mma
D 数据路径：tcgen05.mma -> TMEM -> registers -> GMEM
T.cta_id / T.warpgroup_id / T.warp_id_in_wg / T.lane_id
T.SMEMPool 的 alloc / move_base_to / commit
mma_shared_layout 与 SWIZZLE_128B_ATOM
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
tvm.compile 与 IRModule 的入口函数绑定
tir_pipeline="tirx" 选择 TIRx lowering pipeline
LowerTIRx 将 tile-level primitive 展开为底层 TIR
kernel.script() 与 inspect_source() 的两层检查
Executable module 直接接收 PyTorch tensors
PyTorch float32 参考结果与 fp16 输出比较
rtol / atol 的逐元素误差判断
编译、执行、数值错误的分层排查
chapter_intro_tirx 完成
```

下一知识点：

```text
chapter_tirx_layout_api
-> TileLayout 的 S、R 与 offset
```
