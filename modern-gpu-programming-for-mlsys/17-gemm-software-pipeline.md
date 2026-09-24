# GEMM Async：Step 5 Software Pipeline（PIPE_DEPTH=2）

这篇笔记继续 `chapter_gemm_async`。Step 4 已经让 TMA 负责
GMEM -> SMEM 搬运，也建立了 `mbarrier` 完成协议，但 SMEM 中只有一组
A/B tiles，因此不能安全地提前加载下一块数据。

Step 5 为 A/B 增加 `PIPE_DEPTH=2` 的 ring buffer。它先预取前两块，
之后每消费完一个 stage，就用这个 stage 加载前方第二块 tile。这个版本
仍然由同一个 warpgroup 顺序 issue、wait 和计算，但它已经具备
prefetch、stage 复用和 phase ring 这三个 pipeline 基础。

## 本次讲解位置

```text
本次讲解位置
章节：chapter_gemm_async
小节：Step 5: Software Pipeline（PIPE_DEPTH=2）
知识点：用多 stage SMEM ring buffer 预取 K tiles，
        并用每个 stage 独立的 TMA barrier 管理 barrier 复用
上次：Step 4: TMA Async Load
下次：Step 6: Persistent Kernel + Tile Scheduler
PTX：cp.async.bulk.tensor、mbarrier.try_wait.parity、
     tcgen05.mma、tcgen05.commit、fence.proxy.async、
     cp.async.bulk.commit_group / wait_group
```

### 本章知识点清单

```text
[x] Step 4：TMA Async Load
[x] Step 5：Software Pipeline（PIPE_DEPTH=2）
[ ] Step 6：Persistent Kernel + Tile Scheduler
[ ] Step 7：Warp Specialization 与完整 Load / Compute overlap
```

## 一、Step 4 为什么还不能预取

Step 4 的 operand 存储只有一对 tiles：

```text
Asmem：[BLK_M, BLK_K]
Bsmem：[BLK_N, BLK_K]
```

K-loop 每一轮执行：

```text
TMA load k -> Asmem/Bsmem
wait TMA k
MMA k 读取 Asmem/Bsmem
wait MMA k
进入 k + 1
```

假设在 MMA k 尚未完成时提前发起 load k+1。此时：

```text
MMA k 仍在异步读取 Asmem/Bsmem
TMA load k+1 正在向同一个 Asmem/Bsmem 写入
```

即使两个操作都通过 mbarrier 正确同步，它们访问的仍是同一块地址。
这不是同步问题，而是 storage hazard：生产者会覆盖消费者仍在读取的
数据。

所以，预取的前提不是“把 TMA 改成异步”这么简单，而是要先提供另一块
可写 storage。

## 二、心智模型：两个 stage，两个邮箱

可以把 SMEM 想成流水线上的两个周转箱：

```text
stage 0: [A0, B0]
stage 1: [A1, B1]
```

每个箱子对应一个 TMA 完成邮箱：

```text
tma_bar[0] 报告 stage 0 的 A/B bytes 是否到齐
tma_bar[1] 报告 stage 1 的 A/B bytes 是否到齐
```

MMA 只有一个累加器：

```text
tmem[:, :BLK_N]
```

因此所有 K tiles 仍共用：

```text
mma_bar[0] 报告当前 MMA 是否已经完成
```

Step 5 的操作顺序是：

```text
先把 k=0、k=1 分别放进 stage 0、stage 1

对每个 k：
  等当前 stage 的 TMA 完成
  用当前 stage 执行 MMA
  等 MMA 完成，证明当前 stage 的数据已被消费
  把 k + PIPE_DEPTH 的 tile 加载到刚释放的 stage
```

这里的 `stage = k % PIPE_DEPTH` 就是周转箱编号。

### 三个容易混淆的词

| 名称 | 含义 | 本例取值 |
|---|---|---|
| `PIPE_DEPTH` | ring buffer 中可复用的 stage 数 | `2` |
| `stage` | 某个 K tile 实际使用的 SMEM slot | `k % 2` |
| `phase` | `mbarrier` 当前 completion round 的 parity | `0` 或 `1` |

`PIPE_DEPTH=2` 表示有两个 SMEM stages，不表示同时执行两个 MMA，也不
表示 K-loop 每次前进两步。

## 三、Step 4 与 Step 5 的结构差异

| 项目 | Step 4 | Step 5 |
|---|---|---|
| Scope | 一个 warpgroup | 不变 |
| A/B SMEM | 一组 `[BLK, BLK_K]` tile | `[PIPE_DEPTH, BLK, BLK_K]` |
| TMA barrier | 一个 `tma_bar[0]` | 每个 stage 一个 `tma_bar[stage]` |
| MMA barrier | 一个 `mma_bar[0]` | 不变 |
| K-loop 前 | 无预取 | 预取前两个 stages |
| Load 目标 | 固定覆盖同一 tiles | 覆盖已消费的 ring slot |
| `phase_tma` | 每轮翻转 | stage 走完整 ring 后翻转 |
| `phase_mma` | 每轮翻转 | 每轮翻转 |
| Load / Compute | 单缓冲，无法预取 | future TMA 可与后续 MMA 部分重叠 |
| 完整 producer / consumer 分工 | 无 | 仍无，留到 Step 7 |

Step 5 已经有了部分时间重叠：

```text
在 k 轮结束时发起 k + 2 的 TMA
下一轮 k + 1 可以继续 wait、issue 和计算
k + 2 的 TMA 在这段时间内异步执行
```

但 TMA issue 和 MMA issue 仍由同一个 warpgroup 顺序完成。完整的
producer warp / consumer warp 分工、独立 full/empty barrier 协议和
更稳定的双路径并发要到 Step 7。

## 四、完整可运行代码

下面给出三个文件。前两个是 Blackwell GPU 上的完整 kernel 与数值验证；
第三个是纯 Python pipeline trace，可在当前 Apple M5 Pro 上运行，用来
核对每个 K iteration 的 stage 和 phase。

```text
tirx_gemm_software_pipeline.py
verify_tirx_gemm_software_pipeline.py
trace_gemm_software_pipeline.py
```

### 文件一：`tirx_gemm_software_pipeline.py`

```python
import tvm
from tvm.script import tirx as T
from tvm.script.tirx import tile as Tx
from tvm.tirx.layout import TileLayout, S, TLane, TCol, tid_in_wg
from tvm.backend.cuda.tile_primitive.tma_utils import (
    mma_shared_layout,
    SwizzleMode,
)


PIPE_DEPTH = 2


def hgemm_v5(M, N, K):
    a_type = tvm.DataType("float16")
    b_type = tvm.DataType("float16")
    d_type = tvm.DataType("float16")
    acc_type = tvm.DataType("float32")
    F16_SIZE = 2
    BLK_M, BLK_N, BLK_K = 128, 128, 64
    K_TILES = K // BLK_K

    # The first dimension is the software-pipeline stage.
    A_layout = mma_shared_layout(
        a_type,
        SwizzleMode.SWIZZLE_128B_ATOM,
        (PIPE_DEPTH, BLK_M, BLK_K),
    )
    B_layout = mma_shared_layout(
        b_type,
        SwizzleMode.SWIZZLE_128B_ATOM,
        (PIPE_DEPTH, BLK_N, BLK_K),
    )
    D_layout = mma_shared_layout(
        d_type,
        SwizzleMode.SWIZZLE_128B_ATOM,
        (BLK_M, BLK_N),
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

        # One TMA barrier per stage; one shared MMA completion barrier.
        pool = T.SMEMPool()
        tmem_addr = pool.alloc((1,), "uint32")
        tma_bar = pool.alloc((PIPE_DEPTH,), "uint64", align=8)
        mma_bar = pool.alloc((1,), "uint64", align=8)
        pool.move_base_to(1024)
        Asmem = pool.alloc(
            (PIPE_DEPTH, BLK_M, BLK_K),
            a_type,
            layout=A_layout,
        )
        Bsmem = pool.alloc(
            (PIPE_DEPTH, BLK_N, BLK_K),
            b_type,
            layout=B_layout,
        )
        Dsmem = pool.alloc(
            (BLK_M, BLK_N),
            d_type,
            layout=D_layout,
        )
        pool.commit()

        if warp_id == 0:
            if lane_id == 0:
                T.ptx.mbarrier.init(mma_bar.ptr_to([0]), 1)
                for s in range(PIPE_DEPTH):
                    T.ptx.mbarrier.init(tma_bar.ptr_to([s]), 1)
        if warp_id == 0:
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
            acc_type,
            scope="tmem",
            allocated_addr=tmem_addr[0],
            layout=TileLayout(
                S[(128, 512) : (1 @ TLane, 1 @ TCol)]
            ),
        )

        m_st = T.meta_var(bx * BLK_M)
        n_st = T.meta_var(by * BLK_N)
        phase_tma: T.int32 = 0
        phase_mma: T.int32 = 0

        @T.inline
        def tma_load(stage, k_offset):
            tma_config = T.meta_var({
                "dispatch": "tma_auto",
                "cta_group": 1,
                "mbar": tma_bar.ptr_to([stage]),
            })
            Tx.copy_async(
                Asmem[stage, :, :],
                A[
                    m_st : m_st + BLK_M,
                    k_offset : k_offset + BLK_K,
                ],
                **tma_config,
            )
            Tx.copy_async(
                Bsmem[stage, :, :],
                B[
                    n_st : n_st + BLK_N,
                    k_offset : k_offset + BLK_K,
                ],
                **tma_config,
            )
            T.ptx.mbarrier.arrive.expect_tx(
                tma_bar.ptr_to([stage]),
                (
                    BLK_M * BLK_K
                    + BLK_N * BLK_K
                ) * F16_SIZE,
            )

        @T.inline
        def mma(stage, accum):
            Tx.gemm_async(
                tmem[:, :BLK_N],
                Asmem[stage, :, :],
                Bsmem[stage, :, :],
                accum=accum,
                dispatch="tcgen05",
                cta_group=1,
            )
            T.ptx.tcgen05.commit(
                mma_bar.ptr_to([0]),
                cta_group=1,
            )

        tid = T.meta_var(warp_id * 32 + lane_id)

        # Prologue: fill the first PIPE_DEPTH stages.
        if tid == 0:
            for s in range(min(PIPE_DEPTH, K_TILES)):
                tma_load(s, s * BLK_K)

        # Main software pipeline.
        for k in range(K_TILES):
            stage = k % PIPE_DEPTH

            T.ptx.mbarrier.try_wait(
                tma_bar.ptr_to([stage]),
                phase_tma,
            )

            if tid == 0:
                mma(stage, accum=(k != 0))

            T.ptx.mbarrier.try_wait(
                mma_bar.ptr_to([0]),
                phase_mma,
            )
            phase_mma ^= 1

            next_k = k + PIPE_DEPTH
            if next_k < K_TILES:
                if tid == 0:
                    tma_load(stage, next_k * BLK_K)

            if stage == PIPE_DEPTH - 1:
                phase_tma ^= 1

        # Epilogue: TMEM -> RF -> Dsmem -> TMA -> GMEM.
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
        T.cuda.cta_sync()

        Tx.cast(Dreg_f16[:], Dreg[:])
        Tx.copy(
            Dsmem[warp_id * 32 + lane_id, 0:BLK_N],
            Dreg_f16[:],
        )
        T.ptx.fence.proxy_async("shared::cta")
        T.cuda.warpgroup_sync(10)

        if tid == 0:
            Tx.copy_async(
                D[
                    m_st : m_st + BLK_M,
                    n_st : n_st + BLK_N,
                ],
                Dsmem[:, :],
                dispatch="tma_auto",
            )
            T.ptx.cp_async.bulk.commit_group()
            T.ptx.cp_async.bulk.wait_group(0)

        T.cuda.warpgroup_sync(10)

        T.cuda.cta_sync()
        if warp_id == 0:
            T.ptx.tcgen05.relinquish_alloc_permit(
                cta_group=1,
            )
            T.ptx.tcgen05.dealloc(
                tmem_addr[0],
                n_cols=512,
                cta_group=1,
            )

    return kernel
```

### 文件二：`verify_tirx_gemm_software_pipeline.py`

```python
import torch
import tvm

from tirx_gemm_software_pipeline import hgemm_v5


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

    M, N, K = 4096, 4096, 4096
    kernel = hgemm_v5(M, N, K)

    with target:
        ex = tvm.compile(
            tvm.IRModule({"main": kernel}),
            target=target,
            tir_pipeline="tirx",
        )

    torch.cuda.empty_cache()
    torch.cuda.synchronize()

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

    max_err = float(
        (D_tensor - D_ref).abs().max()
    )
    print(
        f"Max error vs torch reference: "
        f"{max_err:.6f}"
    )

    torch.testing.assert_close(
        D_tensor,
        D_ref,
        rtol=2e-2,
        atol=5e-2,
    )
    print("PASS")


if __name__ == "__main__":
    main()
```

### 文件三：`trace_gemm_software_pipeline.py`

这个脚本不导入 TVM，也不执行 GPU kernel。它只复现 Step 5 的
`stage = k % PIPE_DEPTH`、`phase_mma ^= 1` 和 ring wrap 时的
`phase_tma ^= 1`，用于核对同步状态。

```python
def trace_pipeline(pipe_depth=2, k_tiles=5):
    phase_tma = 0
    phase_mma = 0

    print(
        "k | stage | wait_tma | prefetch | "
        "wait_mma | next_phase_mma | next_phase_tma"
    )
    for k in range(k_tiles):
        stage = k % pipe_depth
        wait_tma = phase_tma
        wait_mma = phase_mma

        next_k = k + pipe_depth
        prefetch = (
            f"k{next_k}->stage{stage}"
            if next_k < k_tiles
            else "-"
        )

        phase_mma ^= 1
        if stage == pipe_depth - 1:
            phase_tma ^= 1

        print(
            f"{k} | {stage} | {wait_tma} | {prefetch} | "
            f"{wait_mma} | {phase_mma} | {phase_tma}"
        )


if __name__ == "__main__":
    trace_pipeline(pipe_depth=2, k_tiles=5)
```

## 五、运行方式和预期结果

### 1. 当前 Apple M5 Pro 上运行 trace

```bash
python3 trace_gemm_software_pipeline.py
```

预期输出：

```text
k | stage | wait_tma | prefetch | wait_mma | next_phase_mma | next_phase_tma
0 | 0 | 0 | k2->stage0 | 0 | 1 | 0
1 | 1 | 0 | k3->stage1 | 1 | 0 | 1
2 | 0 | 1 | k4->stage0 | 0 | 1 | 1
3 | 1 | 1 | - | 1 | 0 | 0
4 | 0 | 0 | - | 0 | 1 | 0
```

这张表说明：

```text
stage 0 的等待 phase 依次是 0, 1, 0
stage 1 的等待 phase 依次是 0, 1
MMA barrier 的等待 phase 依次是 0, 1, 0, 1, 0
只有 k=0,1,2 还有 future tile 可预取
```

### 2. Blackwell GPU 上运行 kernel

在支持 `sm_100a` 的 NVIDIA GPU、TIRx compiler 和 CUDA 版 PyTorch
环境中：

```bash
python verify_tirx_gemm_software_pipeline.py
```

预期输出类似：

```text
device: NVIDIA B200, capability=(10, 0)
Max error vs torch reference: 0.031250
PASS
```

最大误差不要求逐 bit 相同，因为 MMA 的 K 累加顺序可能与 PyTorch
参考实现不同，最后还有 fp16 rounding。

### 3. 只做 Python 语法检查

```bash
python -m py_compile \
  tirx_gemm_software_pipeline.py \
  verify_tirx_gemm_software_pipeline.py \
  trace_gemm_software_pipeline.py
```

该命令只能说明 Python 语法可被解析，不会编译 TIRx，也不会执行 GPU
kernel。

## 六、完整代码逐段讲解

### 1. 把 stage 维写进 SMEM layout

```python
A_layout = mma_shared_layout(
    a_type,
    SwizzleMode.SWIZZLE_128B_ATOM,
    (PIPE_DEPTH, BLK_M, BLK_K),
)
```

Step 4 的逻辑形状是：

```text
Asmem：[128, 64]
Bsmem：[128, 64]
```

Step 5 变成：

```text
Asmem：[2, 128, 64]
Bsmem：[2, 128, 64]
```

第一维不是矩阵 M 或 N，而是 pipeline stage。`Asmem[0, :, :]` 和
`Asmem[1, :, :]` 是两个彼此独立的 operand tile。

这个 stage 维必须同时出现在：

```text
pool.alloc 的 buffer shape
mma_shared_layout 的 layout shape
TMA destination slice
tcgen05.mma 的 operand slice
```

任何一处漏掉这一维，都会让 stage 选择与 descriptor / pointer offset
不一致。常见结果是 stage 1 覆盖 stage 0，或者 MMA 读取错误基址。

### 2. 双缓冲 SMEM 的容量

单个 A tile：

```text
128 * 64 * 2 bytes = 16384 bytes = 16 KiB
```

单个 B tile：

```text
128 * 64 * 2 bytes = 16384 bytes = 16 KiB
```

一个 stage 的 A/B payload：

```text
16 KiB + 16 KiB = 32 KiB
```

两个 stages：

```text
2 * 32 KiB = 64 KiB
```

epilogue 的 `Dsmem`：

```text
128 * 128 * 2 bytes = 32768 bytes = 32 KiB
```

因此 A/B ring buffer 加 Dsmem 的 payload 为：

```text
64 KiB + 32 KiB = 96 KiB
```

还要加上 barriers、TMEM address、对齐和 `move_base_to(1024)` 留出的
空间。因此不能只看到“多了一个维度”就以为没有成本；增加
`PIPE_DEPTH` 会线性增加每个 CTA 的 SMEM 占用。

### 3. 为什么 TMA barrier 要按 stage 分开

```python
tma_bar = pool.alloc((PIPE_DEPTH,), "uint64", align=8)
mma_bar = pool.alloc((1,), "uint64", align=8)
```

Stage 0 和 stage 1 是两个独立生命周期：

```text
stage 0 may be filled again while stage 1 still contains old data
stage 1 may be filled again while stage 0 is being consumed
```

如果两个 stages 共用一个 `tma_bar`，barrier 的 phase 将无法区分：

```text
stage 0 当前完成事件
stage 1 当前完成事件
```

一个 barrier 的完成相位只能表示它自己的第几轮完成。让两个 stage 共用
它会丢失“哪个 stage 已经 ready”的信息，并让 `try_wait` 可能通过错误
的一代完成事件。

MMA barrier 则可以只有一个，因为所有 K tiles 都在更新同一个 TMEM
累加器，MMA completion 是一条串行 completion stream。

### 4. barrier 初始化和阶段

```python
if warp_id == 0:
    if lane_id == 0:
        T.ptx.mbarrier.init(mma_bar.ptr_to([0]), 1)
        for s in range(PIPE_DEPTH):
            T.ptx.mbarrier.init(tma_bar.ptr_to([s]), 1)
```

每个 TMA barrier 的 expected arrival count 是 1：

```text
tid == 0 调用 arrive.expect_tx
每次发布 A/B tile 时只产生一个 thread arrival
```

每个 stage 还有独立的 pending transaction byte count：

```text
A bytes = 128 * 64 * 2 = 16384
B bytes = 128 * 64 * 2 = 16384
pending = 32768
```

因此 `tma_bar[stage]` 的完成条件仍然是：

```text
1 thread arrival 已发生
AND
32768 transaction bytes 已由 TMA complete_tx 清零
```

### 5. `tma_load(stage, k_offset)` 同时选择 storage 和 barrier

```python
@T.inline
def tma_load(stage, k_offset):
    tma_config = T.meta_var({
        "dispatch": "tma_auto",
        "cta_group": 1,
        "mbar": tma_bar.ptr_to([stage]),
    })
    Tx.copy_async(
        Asmem[stage, :, :],
        A[m_st:m_st+BLK_M, k_offset:k_offset+BLK_K],
        **tma_config,
    )
    Tx.copy_async(
        Bsmem[stage, :, :],
        B[n_st:n_st+BLK_N, k_offset:k_offset+BLK_K],
        **tma_config,
    )
    T.ptx.mbarrier.arrive.expect_tx(
        tma_bar.ptr_to([stage]),
        (BLK_M * BLK_K + BLK_N * BLK_K) * F16_SIZE,
    )
```

这里有一个很重要的不变量：

```text
TMA destination stage
==
mbar stage
==
expect_tx stage
```

三者必须完全相同。若 A 写到 stage 1，但 completion 报告给
`tma_bar[0]`，consumer 会在 stage 1 数据未到齐时放行 stage 0，或者
在错误 phase 上永久等待。

`@T.inline` 表示 helper 在 TIRx lowering 时展开，不保留普通 Python
函数调用边界。它让“选择 stage、issue A、issue B、登记 byte count”
成为一个统一操作。

### 6. `mma(stage, accum)` 只改变 operand 来源

```python
@T.inline
def mma(stage, accum):
    Tx.gemm_async(
        tmem[:, :BLK_N],
        Asmem[stage, :, :],
        Bsmem[stage, :, :],
        accum=accum,
        dispatch="tcgen05",
        cta_group=1,
    )
    T.ptx.tcgen05.commit(
        mma_bar.ptr_to([0]),
        cta_group=1,
    )
```

`tmem` destination 不随 stage 改变，因为 K-loop 的所有 partial products
累加到同一个结果区域。变化的只有 A/B operand slice：

```text
k=0 -> Asmem[0], Bsmem[0]
k=1 -> Asmem[1], Bsmem[1]
k=2 -> Asmem[0], Bsmem[0]
k=3 -> Asmem[1], Bsmem[1]
```

`accum=(k != 0)` 仍沿用 Step 3 的规则：

```text
k == 0：accum=False，用第一块 K 覆盖未初始化的 TMEM
k != 0：accum=True，把后面的 K tile 累加进去
```

### 7. Prologue：先填满两个 stages

```python
if tid == 0:
    for s in range(min(PIPE_DEPTH, K_TILES)):
        tma_load(s, s * BLK_K)
```

当 `PIPE_DEPTH=2`、`K_TILES >= 2` 时，启动映射为：

```text
s=0 -> A/B tile k=0 -> stage 0
s=1 -> A/B tile k=1 -> stage 1
```

`min(PIPE_DEPTH, K_TILES)` 是边界保护。如果 K 只有一块 tile，循环只
加载 stage 0，不会去访问不存在的 stage 1。

Prologue 只是 issue，不会立即等到两块都完成。进入 main loop 后，第一个
`try_wait(tma_bar[0], 0)` 才等待 stage 0 可用。

### 8. Main loop：等待、消费、再复用

```python
for k in range(K_TILES):
    stage = k % PIPE_DEPTH

    T.ptx.mbarrier.try_wait(
        tma_bar.ptr_to([stage]),
        phase_tma,
    )

    if tid == 0:
        mma(stage, accum=(k != 0))

    T.ptx.mbarrier.try_wait(
        mma_bar.ptr_to([0]),
        phase_mma,
    )
    phase_mma ^= 1

    next_k = k + PIPE_DEPTH
    if next_k < K_TILES:
        if tid == 0:
            tma_load(stage, next_k * BLK_K)

    if stage == PIPE_DEPTH - 1:
        phase_tma ^= 1
```

逐句理解：

1. `stage = k % PIPE_DEPTH` 选出当前 k tile 所在 slot。
2. `try_wait(tma_bar[stage], phase_tma)` 证明这个 slot 已经是 full。
3. `mma(stage, accum=(k != 0))` 让 Tensor Core 消费这个 slot。
4. `try_wait(mma_bar[0], phase_mma)` 证明 MMA 已完成，slot 不再被
   Tensor Core 使用。
5. 只有证明 MMA 完成后，才允许
   `tma_load(stage, next_k * BLK_K)` 覆盖该 slot。
6. `next_k = k + PIPE_DEPTH` 让新 load 写回刚释放的 slot。
7. `phase_mma ^= 1` 为下一次 MMA completion 准备新的 parity。
8. `stage == PIPE_DEPTH - 1` 表示 ring 走完一整圈，准备让下一圈的
   stage 0 使用新的 TMA phase。

第 4 步是 Step 5 的 empty 条件。这里没有单独的 `empty_bar`，因为当前
单 warpgroup 设计直接把“MMA 完成”当作“该 stage 可以被 TMA 覆盖”。

### 9. `phase_mma` 为什么每轮翻转

所有 K iterations 都使用同一个 barrier：

```text
mma_bar[0]
```

barrier 每完成一轮，下一轮使用另一个 parity。因此 K-loop 必须记录：

```text
k=0 wait phase 0
k=1 wait phase 1
k=2 wait phase 0
k=3 wait phase 1
后续按同样规律交替
```

若忘记 `phase_mma ^= 1`，第 k=1 轮可能继续等待 phase 0。phase 0 早已
完成，等待可能立即通过，导致误把旧完成状态当成当前 MMA 已完成。

### 10. `phase_tma` 为什么不每轮翻转

TMA barrier 是每个 stage 一个：

```text
stage 0：第一次 load phase 0，第二次 load phase 1，第三次 load phase 0
stage 1：第一次 load phase 0，第二次 load phase 1
```

在同一个 ring round 中，stage 0 和 stage 1 都使用相同起始 parity。
因此可以用一个 `phase_tma` 同时跟踪两个 stage；只有当整个 ring 回到
最后一个 stage 后，才翻转给下一轮使用。

这就是下面代码只在 stage 1 执行的语义：

```python
if stage == PIPE_DEPTH - 1:
    phase_tma ^= 1
```

### 11. Epilogue 与 Step 4 完全相同

```python
Tx.wg.copy_async(Dreg_wg[:, :], tmem[:, :BLK_N])
T.ptx.tcgen05.wait.ld()
T.cuda.cta_sync()
Tx.cast(Dreg_f16[:], Dreg[:])
Tx.copy(Dsmem[warp_id * 32 + lane_id, 0:BLK_N], Dreg_f16[:])
T.ptx.fence.proxy_async("shared::cta")
T.cuda.warpgroup_sync(10)
```

这里没有新的 pipeline stage。输出 tile 仍然只有一个 `Dsmem`：

```text
TMEM -> registers
-> fp32 to fp16 cast
-> Dsmem
-> TMA store
-> GMEM
```

K-loop 的 pipeline 只作用于 A/B operand 的加载和消费；它不会改变
epilogue 的 ownership、layout 或 store synchronization。

## 七、具体执行推演：`PIPE_DEPTH=2`、`K_TILES=5`

为了让 ring wrap 发生两次，取一个比真实 K-loop 更小的例子：

```text
PIPE_DEPTH = 2
K_TILES = 5
```

### 1. Prologue

```text
s=0: load k=0 -> stage 0
s=1: load k=1 -> stage 1
```

此时：

```text
phase_tma = 0
phase_mma = 0
```

### 2. Main loop trace

| k | stage | `try_wait(tma_bar[stage], phase_tma)` | 本轮回填 | `try_wait(mma_bar[0], phase_mma)` | 下一轮 `phase_mma` | 本轮末尾 `phase_tma` |
|---:|---:|---:|---|---:|---:|---:|
| 0 | 0 | phase 0 | k=2 -> stage 0 | phase 0 | 1 | 0 |
| 1 | 1 | phase 0 | k=3 -> stage 1 | phase 1 | 0 | 1 |
| 2 | 0 | phase 1 | k=4 -> stage 0 | phase 0 | 1 | 1 |
| 3 | 1 | phase 1 | 无 | phase 1 | 0 | 0 |
| 4 | 0 | phase 0 | 无 | phase 0 | 1 | 0 |

Stage 0 的 TMA barrier 观察到：

```text
k=0: wait phase 0
k=2: wait phase 1
k=4: wait phase 0
```

Stage 1 的 TMA barrier 观察到：

```text
k=1: wait phase 0
k=3: wait phase 1
```

MMA barrier 观察到：

```text
k=0: wait phase 0
k=1: wait phase 1
k=2: wait phase 0
k=3: wait phase 1
k=4: wait phase 0
```

### 3. 为什么最后两块不再预取

`next_k = k + 2`：

```text
k=0: next_k=2 < 5，预取
k=1: next_k=3 < 5，预取
k=2: next_k=4 < 5，预取
k=3: next_k=5，不再预取
k=4: next_k=6，不再预取
```

最后两个 iterations 是在 drain。ring 里已经没有未来的 K tile，继续
issue 会访问越界或写入不存在的 tile。

### 4. 真实课程参数的 barrier 使用次数

课程使用：

```text
K = 4096
BLK_K = 64
K_TILES = 4096 / 64 = 64
PIPE_DEPTH = 2
```

因此每个 stage 被使用：

```text
64 / 2 = 32 次
```

`phase_tma` 在 K-loop 中翻转 32 次，最终回到 0；`phase_mma` 翻转 64
次，也回到 0。这个偶数性质很重要：下一步做 persistent kernel、复用
同一个 tile 状态的 barriers 时，才能把下一块 output tile 从 phase 0
重新开始。若改变 K 或 `PIPE_DEPTH` 使次数为奇数，必须继承当前 parity，
不能盲目重置。

## 八、为什么这不是完整的 producer / consumer pipeline

Step 5 的 ring buffer 已经让“未来 TMA”与“当前后续 MMA”有时间重叠的
可能，但仍然是单 warpgroup 顺序 issue：

```text
same warpgroup:
  wait TMA current stage
  issue MMA current stage
  wait MMA current stage
  issue TMA k + PIPE_DEPTH
```

第 7 步才会把角色拆成：

```text
producer warps:
  wait stage empty
  TMA global -> shared
  arrive stage full

consumer warps:
  wait stage full
  ldmatrix / MMA
  arrive stage empty
```

Step 7 的关键改进不是“凭空让搬运更快”，而是让生产者和消费者的
推进可以独立调度，减少单 warpgroup 在 issue、wait、MMA 之间串行
等待造成的空档。

## 九、常见错误与可观察症状

| 错误 | 失效原因 | 可观察症状 |
|---|---|---|
| 只把 Asmem/Bsmem 扩成两维，`tma_bar` 仍只有一个 | 两个 stage 的 completion 被混为一代 | TMA wait 偶发提前通过或长期卡住 |
| `tma_bar` 分开了，但 `phase_tma` 每轮翻转 | stage 第二次使用时 parity 错误 | k=2/k=3 以后开始死锁或读到旧数据 |
| 回填前不等 `mma_bar` | TMA 覆盖 MMA 仍在读取的 stage | 数值偶发错误、与调度有关 |
| `next_k` 判断写成 `<=` | 最后一个不存在的 tile 也发起 load | 越界地址、illegal memory access 或脏数据 |
| layout / allocation / TMA slice 三者 shape 不一致 | descriptor 与 buffer 物理地址不匹配 | 固定 pattern 的错位或 lowering 报错 |
| `phase_mma` 不翻转 | 使用上一轮 completion parity | MMA wait 立即返回或永久等待 |
| 以为 `PIPE_DEPTH=2` 等于两个 MMA 同时运行 | 把 storage stages 与 compute concurrency 混淆 | 错误估算 TMEM、依赖与性能 |
| `@T.inline` helper 漏掉 `expect_tx` | barrier 不知道 transaction bytes | `try_wait` 无法在正确时间完成 |

如果错误表现为“每次运行不同、和 wait 时序有关”，优先检查 barrier
arrival / phase / stage 选择。若表现为“从某一列、某一行开始固定错位”，
优先检查 layout、descriptor 和 stage 基址。

## 十、与推理系统的联系

LLM 推理中的 GEMM 通常有很深的 K：

```text
prefill GEMM：K 可能达到 hidden size 或更大的 reduction
decode GEMM：K 也可能是 hidden / intermediate size
```

如果每个 `BLK_K` tile 都等 TMA 完全到齐再计算，Tensor Core 会周期性
空闲。软件 pipeline 的目标是让：

```text
TMA tile k + 2
和
MMA tile k + 1
在时间上重叠
```

这样可以在不改变数学结果的前提下，用 SMEM 空间换取更连续的 Tensor
Core 利用率。代价是：

```text
更多 SMEM
更多 barrier
更复杂的 phase 管理
```

Attention、KV cache 和 MoE 中的矩阵乘法同样依赖这条数据路径。理解
ring stage 与 phase 后，后续 producer / consumer specialization 才能
被看作“谁在什么时间拥有 stage”，而不是背 API。

## 十一、硬件和验证边界

这段 kernel 使用：

```text
TMA
mbarrier
tcgen05.mma
TMEM
```

因此完整验证需要 Blackwell `sm_100a` GPU。H100 / H200 支持 TMA，但
不能执行这里的 `tcgen05` TMEM MMA 路径。当前 Apple M5 Pro 只能运行
纯 Python trace、检查语法和推导 phase，不能验证真实 TMA/MMA 时序，
也不能测量 overlap 收益。

```text
可以静态验证：
  Python/TIRx 源码结构
  stage 与 phase 的数值 trace
  每轮 transaction byte count
  layout shape、stage 基址和 barrier index

必须到目标 GPU 才能验证：
  TMA 实际完成顺序
  mbarrier 没有死锁或错误 phase
  tcgen05.mma 的数值结果
  pipeline 的实际吞吐和 overlap
```

## 十二、自测

### 1. 为什么需要每个 SMEM stage 对应一个 `tma_bar`？

答：不同 stage 可以在不同时间被填充和复用。一个 barrier 只能通过
自己的 phase 表示自己的完成轮次；如果多个 stage 共用一个 barrier，
consumer 无法区分“stage 0 的这一轮完成”和“stage 1 的下一轮完成”。
每个 stage 独立 barrier 后，`try_wait(tma_bar[stage], phase_tma)` 才能
准确等待那个 slot。

### 2. 为什么 `phase_tma` 只在 stage 1 翻转，而不是每轮翻转？

答：两个 stage 在同一个 ring round 中先使用 parity 0，下一轮再使用
parity 1。只有当 ring 走完最后一个 stage、准备重新从 stage 0 开始时，
等待 parity 才需要整体翻转。因此条件写成
`if stage == PIPE_DEPTH - 1`。

### 3. 为什么 `phase_mma` 必须每轮翻转？

答：所有 K iterations 共用同一个 `mma_bar[0]`。每次 MMA completion
都会让这个 barrier 进入下一轮，所以第 k 轮必须等待当前 round 的
parity，不能重复使用上一轮的完成状态。

### 4. 在 `PIPE_DEPTH=2`、`K_TILES=5` 时，k=2 回填到哪里？k=3 和
k=4 为什么没有回填？

答：k=2 时 `next_k = 4`，而 `stage = 2 % 2 = 0`，所以 k=4 的 tile
回填到 stage 0。k=3 的 `next_k=5`，k=4 的 `next_k=6`，都超过
`K_TILES - 1`，因此最后两个 iterations 只 drain，不再预取。

### 5. 为什么不能把 load k+2 放在 wait MMA k 之前？

答：load k+2 会写入 stage k 所使用的 SMEM。此时 MMA k 可能仍在
异步读取该 stage，提前覆盖会让它读到混合的旧数据和新数据。必须先用
`mma_bar` 等 MMA k 完成，证明该 stage 已被消费，才能回填。
