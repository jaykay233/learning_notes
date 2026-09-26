# GEMM Async：Step 6 Persistent Kernel 与 Tile Scheduler

这篇笔记继续 `chapter_gemm_async`。Step 5 已经用
`PIPE_DEPTH=2` 把一块 output tile 内部的 K-loop 做了软件流水，
但 launch grid 仍然是二维的：每个 output tile 对应一个 CTA。

Step 6 把调度粒度从“一个 CTA 一块 tile”改成“固定数量的持久
worker，每个 worker 顺序处理多块 tile”。MMA、TMA、SMEM layout 和
epilogue 数据路径都不变，变化集中在 launch grid 和 tile 的来源。

## 本次讲解位置

```text
本次讲解位置
章节：chapter_gemm_async
小节：Step 6: Persistent Kernel + Tile Scheduler
知识点：固定 persistent CTA 数量，在一个 CTA 内循环领取多个
        output tiles，并复用 TMEM、mbarrier 和 SMEM allocation
上次：Step 5: Software Pipeline（PIPE_DEPTH=2）
下次：Step 7: Warp Specialization 与完整 Load / Compute overlap
PTX：TMA、mbarrier、tcgen05.mma、tcgen05.commit、
     cp.async.bulk.commit_group / wait_group
```

### 本章知识点清单

```text
[x] Step 4：TMA Async Load
[x] Step 5：Software Pipeline（PIPE_DEPTH=2）
[x] Step 6：Persistent Kernel + Tile Scheduler
[ ] Step 7：Warp Specialization 与完整 Load / Compute overlap
```

## 一、Step 5 留下了什么开销

课程固定输出大小：

```text
M = 4096
N = 4096
BLK_M = 128
BLK_N = 128

M tiles = 4096 / 128 = 32
N tiles = 4096 / 128 = 32
total tiles = 32 * 32 = 1024
```

Step 5 的 launch grid 是：

```python
bx, by = T.cta_id([M // BLK_M, N // BLK_N])
```

因此一共启动：

```text
32 * 32 = 1024 个 CTA
```

每个 CTA 只处理一块 `128 x 128` output tile，然后退出。问题不是
这些初始化操作本身很慢，而是它们被重复了 1024 次：

```text
初始化 mbarriers
分配 TMEM columns
建立 SMEM allocation
构造 TMA descriptor / address state
执行固定 prologue
```

二维 grid 还让 tile 到硬件的分配主要交给 grid scheduler。程序虽然
可以控制 `bx/by` 与坐标的对应关系，却不能直接控制哪些 CTA 更接近
同时运行，也不容易围绕 L2 访问顺序组织 tile。

Step 6 要解决的不是 MMA 计算本身，而是：

```text
一个 CTA 能否处理多块 output tile
多块 tile 应该按什么顺序分给 CTA
CTA 复用时哪些资源可以保留
barrier 的 phase 在 tile 边界如何继续
```

## 二、心智模型：worker 和 task

Persistent kernel 把执行分成两个概念：

| 概念 | Step 6 对应物 |
|---|---|
| worker | 一个长期运行的 CTA |
| task | 一块 `128 x 128` output tile |
| task id | scheduler 的 `linear_idx` |
| task 坐标 | scheduler 的 `m_idx`、`n_idx` |
| worker 身份 | `bx = T.cta_id([SM_COUNT])` |
| 领取下一任务 | `tile_scheduler.next_tile()` |

Step 5 的结构是：

```text
CTA 0 -> tile (0, 0) -> 退出
CTA 1 -> tile (0, 1) -> 退出
CTA 2 -> tile (0, 2) -> 退出
...
CTA 1023 -> tile (31, 31) -> 退出
```

Step 6 的结构是：

```text
CTA 0 -> tile A -> tile B -> tile C -> ...
CTA 1 -> tile D -> tile E -> tile F -> ...
...
CTA 147 -> tile ... -> 退出
```

在课程参数中：

```text
SM_COUNT = 148
```

因此 launch grid 从 `1024` 个 CTA 缩小到 `148` 个 CTA。每个 CTA
仍然根据自己的资源占用和硬件 residency 情况运行。`SM_COUNT` 表示
“打算接近每个 SM 放一个 persistent CTA”，不表示某个 CTA 被永久
绑定到某个具体 SM。

还要区分两件事：

| 机制 | Step 6 | `chapter_clc` |
|---|---|---|
| tile 来源 | 程序内的静态公式 | 硬件动态返回 pending launch |
| scheduling 信息 | 只在 CTA 本地 | 通过 CLC response 获取 |
| 负载均衡方式 | 固定 stride | 先完成的 worker 继续领取 |
| 额外开销 | 几次整数运算 | request、barrier、proxy fence |

Step 6 仍然是静态 persistent scheduler，不是动态 work stealing。

## 三、Step 5 与 Step 6 的结构差异

| 项目 | Step 5 | Step 6 |
|---|---|---|
| launch grid | `32 x 32`，1024 个 CTA | 1D，148 个 CTA |
| 每个 CTA 的 tile 数 | 1 | 6 或 7，取决于 `bx` |
| tile 坐标来源 | `bx`、`by` | `tile_scheduler.m_idx/n_idx` |
| SMEM allocation | 每个 tile 重新做 | CTA 生命周期内只做一次 |
| mbarrier init | 每个 tile 重新做 | CTA 生命周期内只做一次 |
| TMEM allocation | 每个 tile 重新做 | CTA 生命周期内只做一次 |
| `tmem` buffer 绑定 | 每个 tile 一次 | CTA 生命周期内一次 |
| K-loop | `PIPE_DEPTH=2` | 完全不变 |
| TMA/MMA layout | 不变 | 不变 |
| barrier phase 管理 | 单块 tile 内 | 块内 reset，跨 tile 复用 |
| tile 顺序 | 主要交给 grid scheduler | scheduler 明确编号 |

最重要的结论是：

```text
Step 6 不是一个新的 MMA 算法。
它改变的是 tile 的所有权，不是 tile 内部的数据路径。
```

## 四、完整 Kernel

下面代码和课程 Step 6 对应。不同 TVM/TIRx wheel 对
`tma_utils` 的模块位置有过调整，因此这里加了一个兼容 import：

```python
try:
    from tvm.backend.cuda.tile_primitive.tma_utils import (
        mma_shared_layout,
        SwizzleMode,
    )
except ModuleNotFoundError:
    from tvm.backend.cuda.operator.tile_primitive.tma_utils import (
        mma_shared_layout,
        SwizzleMode,
    )
```

课程源码所在版本如果能直接导入第一路径，就使用第一路径；本地
`tvm 0.26.dev246` 需要第二路径。

### 文件一：`tirx_gemm_persistent.py`

```python
import tvm
from tvm.script import tirx as T
from tvm.script.tirx import tile as Tx
from tvm.tirx.layout import TileLayout, S, TLane, TCol, tid_in_wg
from tvm.backend.cuda.lang.tile_scheduler import (
    ClusterPersistentScheduler2D,
)

try:
    from tvm.backend.cuda.tile_primitive.tma_utils import (
        mma_shared_layout,
        SwizzleMode,
    )
except ModuleNotFoundError:
    from tvm.backend.cuda.operator.tile_primitive.tma_utils import (
        mma_shared_layout,
        SwizzleMode,
    )


SM_COUNT = 148
PIPE_DEPTH = 2


def hgemm_v6(M, N, K):
    a_type = tvm.DataType("float16")
    b_type = tvm.DataType("float16")
    d_type = tvm.DataType("float16")
    acc_type = tvm.DataType("float32")
    F16_SIZE = 2
    BLK_M, BLK_N, BLK_K = 128, 128, 64

    assert K % BLK_K == 0, "K must be divisible by BLK_K"
    K_TILES = K // BLK_K
    assert K_TILES % (2 * PIPE_DEPTH) == 0, (
        "K_TILES must be divisible by 2 * PIPE_DEPTH"
    )

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

        # 1D persistent grid: one CTA per hardware worker.
        bx = T.cta_id([SM_COUNT])
        wg_id = T.warpgroup_id([1])
        warp_id = T.warp_id_in_wg([4])
        lane_id = T.lane_id([32])

        # Allocate CTA resources once for all output tiles.
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

        # Initialize barriers and TMEM once per persistent CTA.
        if warp_id == 0 and lane_id == 0:
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

        # The scheduler assigns output tiles to persistent CTAs.
        tile_scheduler = ClusterPersistentScheduler2D(
            "ts",
            num_m_tiles=M // BLK_M,
            num_n_tiles=N // BLK_N,
            l2_group_size=8,
            num_clusters=SM_COUNT,
        )
        tile_scheduler.init(bx)

        tid = T.meta_var(warp_id * 32 + lane_id)

        @T.inline
        def tma_load(stage, k_offset, m_st, n_st):
            tma_config = T.meta_var(
                {
                    "dispatch": "tma_auto",
                    "cta_group": 1,
                    "mbar": tma_bar.ptr_to([stage]),
                }
            )
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
                (BLK_M * BLK_K + BLK_N * BLK_K) * F16_SIZE,
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

        # Outer loop: claim one output tile at a time.
        while tile_scheduler.valid():
            m_st = T.meta_var(tile_scheduler.m_idx * BLK_M)
            n_st = T.meta_var(tile_scheduler.n_idx * BLK_N)

            # A tile-local phase trace starts at 0 because each barrier
            # completes an even number of rounds per tile.
            phase_tma: T.int32 = 0
            phase_mma: T.int32 = 0

            # Prologue: fill the first PIPE_DEPTH stages.
            if tid == 0:
                for s in range(min(PIPE_DEPTH, K_TILES)):
                    tma_load(
                        s,
                        s * BLK_K,
                        m_st,
                        n_st,
                    )

            # The same staged K-loop as Step 5.
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
                        tma_load(
                            stage,
                            next_k * BLK_K,
                            m_st,
                            n_st,
                        )

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

            # All threads must finish the tile before advancing the
            # shared scheduler state.
            T.cuda.cta_sync()
            tile_scheduler.next_tile()

        # Release TMEM after all tiles assigned to this CTA are done.
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

### 文件二：`verify_tirx_gemm_persistent.py`

```python
import torch
import tvm

from tirx_gemm_persistent import hgemm_v6


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
    kernel = hgemm_v6(M, N, K)

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
    print(f"Max error vs torch reference: {max_err:.6f}")

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

### 文件三：`trace_persistent_scheduler.py`

这个脚本不导入 TVM，也不执行 GPU kernel。它复现
`ClusterPersistentScheduler2D` 在当前参数下的静态编号和每个 CTA
的 tile 序列。

```python
def decode_group_major(
    work_idx,
    num_m_tiles=32,
    num_n_tiles=32,
    l2_group_size=8,
):
    grouped_m_rows = (
        num_m_tiles // l2_group_size
    ) * l2_group_size
    tail_rows = num_m_tiles - grouped_m_rows
    group_span = l2_group_size * num_n_tiles

    if work_idx < grouped_m_rows * num_n_tiles:
        group_id = work_idx // group_span
        within_group = work_idx % group_span
        m_idx = (
            group_id * l2_group_size
            + within_group % l2_group_size
        )
        n_idx = within_group // l2_group_size
        return m_idx, n_idx

    remainder = work_idx - grouped_m_rows * num_n_tiles
    m_idx = grouped_m_rows + remainder % tail_rows
    n_idx = remainder // tail_rows
    return m_idx, n_idx


def cta_sequence(
    cta_id,
    num_m_tiles=32,
    num_n_tiles=32,
    num_clusters=148,
):
    total_tiles = num_m_tiles * num_n_tiles
    work_idx = cta_id
    sequence = []

    while work_idx < total_tiles:
        m_idx, n_idx = decode_group_major(
            work_idx,
            num_m_tiles,
            num_n_tiles,
        )
        sequence.append((work_idx, m_idx, n_idx))
        work_idx += num_clusters

    return sequence


def main():
    print("global work id -> tile coordinate")
    for work_idx in range(16):
        m_idx, n_idx = decode_group_major(work_idx)
        print(f"{work_idx:3d} -> (m={m_idx}, n={n_idx})")

    print()
    print("persistent CTA sequences")
    for cta_id in [0, 1, 2, 135, 136, 147]:
        sequence = cta_sequence(cta_id)
        formatted = " -> ".join(
            f"{work}:(m{m},n{n})"
            for work, m, n in sequence
        )
        print(f"CTA {cta_id:3d} [{len(sequence)}] {formatted}")

    counts = {}
    for cta_id in range(148):
        count = len(cta_sequence(cta_id))
        counts[count] = counts.get(count, 0) + 1
    print()
    print(f"tile-count histogram: {counts}")


if __name__ == "__main__":
    main()
```

## 五、Scheduler 如何编码 2D tile

`ClusterPersistentScheduler2D` 这里虽然名字里有 cluster，但课程使用：

```text
cluster_m = 1
cluster_n = 1
num_clusters = SM_COUNT = 148
```

所以这里可以把它理解成“148 个 persistent CTA 的 2D tile scheduler”。

内部最关键的状态是：

```text
linear_idx = 当前 work item 编号
tile_count = 当前 CTA 已处理多少块 tile
m_idx      = 当前 tile 的 M 方向编号
n_idx      = 当前 tile 的 N 方向编号
```

初始化：

```python
tile_scheduler.init(bx)
```

等价于：

```python
linear_idx = bx
tile_count = 0
m_idx, n_idx = decode(linear_idx)
```

领取下一块 tile：

```python
tile_scheduler.next_tile()
```

等价于：

```python
linear_idx += num_clusters
tile_count += 1
m_idx, n_idx = decode(linear_idx)
```

因此 CTA 的 work id 序列是：

```text
bx, bx + 148, bx + 296, bx + 444, ...
```

它不是：

```text
bx * 7, bx * 7 + 1, bx * 7 + 2, ...
```

也就是说，scheduler 先定义一个全局 tile 编号顺序，再让 148 个
worker 以固定 stride 从这个顺序中取编号。

## 六、`l2_group_size=8` 的实际顺序

当前参数：

```text
num_m_tiles = 32
num_n_tiles = 32
l2_group_size = 8

full M groups = 32 / 8 = 4
group span = 8 * 32 = 256 work ids
tail rows = 32 - 4 * 8 = 0
```

每一组覆盖 8 行 M tiles。组内按列扫描：

```text
group 0: M rows 0..7
  n=0:  (m=0,n=0) ... (m=7,n=0)
  n=1:  (m=0,n=1) ... (m=7,n=1)
  ...
  n=31: (m=0,n=31) ... (m=7,n=31)

group 1: M rows 8..15
  按同样方式扫描 n=0..31

group 2: M rows 16..23
group 3: M rows 24..31
```

全局 work id 的前 16 项是：

```text
 0 -> (m=0, n=0)
 1 -> (m=1, n=0)
 2 -> (m=2, n=0)
 3 -> (m=3, n=0)
 4 -> (m=4, n=0)
 5 -> (m=5, n=0)
 6 -> (m=6, n=0)
 7 -> (m=7, n=0)
 8 -> (m=0, n=1)
 9 -> (m=1, n=1)
10 -> (m=2, n=1)
11 -> (m=3, n=1)
12 -> (m=4, n=1)
13 -> (m=5, n=1)
14 -> (m=6, n=1)
15 -> (m=7, n=1)
```

这里为什么对 L2 有利？

同一列 `n` 的 8 个 tile 使用相同的 `B[n_start:n_start+128, :]`。它们
在全局编号中连续出现，因此更可能在相近时间被执行，B tile 有更高
机会在 L2 中复用。

这 8 个 tile 使用不同的 A rows：

```text
A rows 0..1023
```

当编号从 `n=0` 移到 `n=1` 时，同一个 `m` 会再次出现，只是 N 坐标
不同。一组内也会重新使用对应的 A tile 范围。

`l2_group_size=8` 不是：

```text
每个 CTA 只处理 8 块 tile
8 个 CTA 是一个固定 cluster
输出 tile 的 BLK_M 变成 8
```

它只是 scheduler 的全局编号分组大小。

## 七、每个 CTA 实际拿到什么

总 tile 数：

```text
1024
```

平均分给 148 个 CTA：

```text
1024 = 148 * 6 + 136
```

所以：

```text
CTA 0..135   : 7 块 tile
CTA 136..147 : 6 块 tile
```

这正是：

```text
tile-count histogram: {7: 136, 6: 12}
```

几个实际序列：

```text
CTA   0 [7]
0:(m0,n0) -> 148:(m4,n18) -> 296:(m8,n5)
-> 444:(m12,n23) -> 592:(m16,n10) -> 740:(m20,n28)
-> 888:(m24,n15)

CTA   1 [7]
1:(m1,n0) -> 149:(m5,n18) -> 297:(m9,n5)
-> 445:(m13,n23) -> 593:(m17,n10) -> 741:(m21,n28)
-> 889:(m25,n15)

CTA 135 [7]
135:(m7,n16) -> 283:(m11,n3) -> 431:(m15,n21)
-> 579:(m19,n8) -> 727:(m23,n26) -> 875:(m27,n13)
-> 1023:(m31,n31)

CTA 136 [6]
136:(m0,n17) -> 284:(m12,n3) -> 432:(m8,n22)
-> 580:(m20,n8) -> 728:(m16,n27) -> 876:(m28,n13)

CTA 147 [6]
147:(m3,n18) -> 295:(m15,n4) -> 443:(m11,n23)
-> 591:(m23,n9) -> 739:(m19,n28) -> 887:(m31,n14)
```

需要注意：

```text
全局 work id 相邻，不代表同一个 CTA 的下一次任务相邻。
每个 CTA 的序列由 +148 stride 决定。
```

静态 stride 会尽量均衡 tile 数量，但不会根据每块 tile 的真实耗时
动态抢任务。如果 tile 成本差异很大，仍可能出现尾部空闲；那属于
`chapter_clc` 讨论的动态 scheduler 问题。

## 八、哪些资源只初始化一次

### 1. SMEM allocation

`pool.alloc` 和 `pool.commit` 在 outer while 之前执行。整个 CTA 生命
周期内只建立一次：

```text
tma_bar
mma_bar
Asmem
Bsmem
Dsmem
```

### 2. mbarrier

```python
if warp_id == 0 and lane_id == 0:
    T.ptx.mbarrier.init(mma_bar.ptr_to([0]), 1)
    for s in range(PIPE_DEPTH):
        T.ptx.mbarrier.init(tma_bar.ptr_to([s]), 1)
```

这些 barrier 不是在每块 tile 之前重新 init，而是在 tile 之间继续
使用新的一轮 phase。重新初始化一个正在复用的 barrier 会破坏它的
completion/parity 状态。

### 3. TMEM

```python
T.ptx.tcgen05.alloc(
    T.address_of(tmem_addr),
    n_cols=512,
    cta_group=1,
)
```

TMEM 也在 outer loop 之前分配一次。每块 tile 继续复用同一个
`tmem[:, :BLK_N]` accumulator。

### 4. Scheduler state

```python
tile_scheduler = ClusterPersistentScheduler2D(...)
tile_scheduler.init(bx)
```

Scheduler 的 local scalar 也建立一次；每块 tile 结束时只推进
`linear_idx`。

### 每个 tile 仍会做什么

虽然资源可以复用，每块 tile 内部仍必须完整执行：

```text
从 scheduler 读取 m_st / n_st
重置本 tile 的 phase 变量
K-loop 的 64 次 TMA/MMA 流水
TMEM -> register -> Dsmem -> GMEM writeback
CTA 同步
推进下一块 tile
```

Persistent kernel 省掉的是固定初始化，不是 tile 的计算和写回。

## 九、barrier phase 为什么可以每块 tile 重置为 0

当前参数：

```text
K_TILES = 4096 / 64 = 64
PIPE_DEPTH = 2
```

每个 tile 的 K-loop：

```text
mma_bar 每轮完成一次
每个 tma_bar[stage] 每两轮完成一次
```

因此每块 tile 中：

```text
mma_bar[0] 完成次数 = 64
tma_bar[0] 完成次数 = 64 / 2 = 32
tma_bar[1] 完成次数 = 64 / 2 = 32
```

64 和 32 都是偶数。mbarrier 的 completion parity 在偶数轮后回到
初始状态：

```text
第 1 次完成：phase 0
第 2 次完成：phase 1
第 3 次完成：phase 0
...
第 64 次完成：phase 1
下一次期待：phase 0
```

所以下一块 tile 可以重新写：

```python
phase_tma: T.int32 = 0
phase_mma: T.int32 = 0
```

源码中的 assertion 就是在限制这个条件：

```python
assert K_TILES % (2 * PIPE_DEPTH) == 0
```

它保证：

```text
K_TILES 能被 PIPE_DEPTH 整除
每个 stage 的使用次数 K_TILES / PIPE_DEPTH 是偶数
```

如果修改参数后每个 stage 使用 33 次，那么完成 parity 会落在 0，
下一块 tile 应该继承这个 parity，不能盲目重置为 0。症状通常是：

```text
第一块 tile 正确
第二块 tile 开始等待立即通过或永久卡住
偶发读到旧 SMEM
```

## 十、完整执行路径

一个 persistent CTA 的生命周期是：

```text
设备启动 148 个 CTA
    |
    v
初始化 barrier、TMEM、SMEM allocation
    |
    v
tile_scheduler.init(bx)
    |
    v
while scheduler.valid():
    decode m_idx, n_idx
    m_st = m_idx * 128
    n_st = n_idx * 128
    phase_tma = 0
    phase_mma = 0
    prologue TMA load
    staged K-loop: TMA wait -> MMA -> MMA wait -> refill
    TMEM -> register -> Dsmem
    TMA store -> GMEM
    cta_sync
    scheduler.next_tile()
    |
    v
没有更多 tile
    |
    v
cta_sync
dealloc / relinquish TMEM
```

这个顺序里的每一个 `cta_sync` 都有明确边界：

| 同步点 | 保护什么 |
|---|---|
| 初始化后的 `cta_sync` | barrier / TMEM 初始化对所有线程可见 |
| writeback 中的 `cta_sync` | 等所有线程完成 TMEM load |
| epilogue 的两个 `warpgroup_sync(10)` | Dsmem 写完后才提交 TMA store |
| outer loop 末尾的 `cta_sync` | 所有线程完成 epilogue 后才推进 scheduler |

最后一个 `cta_sync` 很关键。`tile_scheduler` 的 local scalar 被整个
CTA 共用，如果某些线程提前执行 `next_tile()`，不同线程可能使用不同
的 `m_st/n_st`，从而把不同 tile 的 A/B/D 地址混在一起。

## 十一、运行和验证

### 1. 在当前 Apple M5 Pro 上运行 scheduler trace

```bash
python3 trace_persistent_scheduler.py
```

预期输出：

```text
global work id -> tile coordinate
  0 -> (m=0, n=0)
  1 -> (m=1, n=0)
  2 -> (m=2, n=0)
  3 -> (m=3, n=0)
  4 -> (m=4, n=0)
  5 -> (m=5, n=0)
  6 -> (m=6, n=0)
  7 -> (m=7, n=0)
  8 -> (m=0, n=1)
  9 -> (m=1, n=1)
 10 -> (m=2, n=1)
 11 -> (m=3, n=1)
 12 -> (m=4, n=1)
 13 -> (m=5, n=1)
 14 -> (m=6, n=1)
 15 -> (m=7, n=1)

persistent CTA sequences
CTA   0 [7] 0:(m0,n0) -> 148:(m4,n18) -> 296:(m8,n5) -> 444:(m12,n23) -> 592:(m16,n10) -> 740:(m20,n28) -> 888:(m24,n15)
CTA   1 [7] 1:(m1,n0) -> 149:(m5,n18) -> 297:(m9,n5) -> 445:(m13,n23) -> 593:(m17,n10) -> 741:(m21,n28) -> 889:(m25,n15)
CTA   2 [7] 2:(m2,n0) -> 150:(m6,n18) -> 298:(m10,n5) -> 446:(m14,n23) -> 594:(m18,n10) -> 742:(m22,n28) -> 890:(m26,n15)
CTA 135 [7] 135:(m7,n16) -> 283:(m11,n3) -> 431:(m15,n21) -> 579:(m19,n8) -> 727:(m23,n26) -> 875:(m27,n13) -> 1023:(m31,n31)
CTA 136 [6] 136:(m0,n17) -> 284:(m12,n3) -> 432:(m8,n22) -> 580:(m20,n8) -> 728:(m16,n27) -> 876:(m28,n13)
CTA 147 [6] 147:(m3,n18) -> 295:(m15,n4) -> 443:(m11,n23) -> 591:(m23,n9) -> 739:(m19,n28) -> 887:(m31,n14)

tile-count histogram: {7: 136, 6: 12}
```

### 2. 只做 Python 语法检查

```bash
python -m py_compile \
  tirx_gemm_persistent.py \
  verify_tirx_gemm_persistent.py \
  trace_persistent_scheduler.py
```

这不会导入 CUDA，也不会执行 GPU kernel。

### 3. 在支持 TIRx 的环境中解析 PrimFunc

如果文件和 `python` 来自同一个 TIRx wheel 环境：

```bash
python -c \
  'from tirx_gemm_persistent import hgemm_v6; \
   k = hgemm_v6(4096, 4096, 4096); \
   print(type(k).__name__)'
```

预期输出：

```text
PrimFunc
```

这一步验证的是 Python import 和 TIRx parser 能成功构造 PrimFunc，
不表示 CUDA 编译器或 Blackwell 硬件已经执行。

### 4. 在 Blackwell GPU 上运行

```bash
python verify_tirx_gemm_persistent.py
```

预期输出类似：

```text
device: NVIDIA B200, capability=(10, 0)
Max error vs torch reference: 0.031250
PASS
```

最大误差不要求逐 bit 相同，因为 MMA 的 K 累加顺序可能与 PyTorch
参考实现不同，最后还有 fp16 rounding。

## 十二、常见错误与可观察症状

| 错误 | 失效原因 | 可观察症状 |
|---|---|---|
| 仍使用 `bx, by = T.cta_id([32, 32])` | scheduler 收到的 `bx` 不是 0..147 的 worker id | scheduler 分配错误的 tile 集合或发生越界 |
| 忘记 `tile_scheduler.init(bx)` | `m_idx/n_idx` 未初始化 | 第一块 tile 地址随机或错误 |
| 忘记 `next_tile()` | worker 永远处理同一块 tile | 重复写同一输出，其他 tile 未初始化 |
| 在 outer loop 内重新 init barriers | 破坏正在复用的 phase | 第二块 tile 开始挂起、读到旧数据或行为随机 |
| tile 边界不重置 phase，但实际完成次数是偶数 | 等待的 parity 落后一轮 | 每块 tile 都多等一轮，正确但明显变慢 |
| 奇数次完成却把 phase 重置为 0 | 等待了已经过去的 phase | 立即通过并读旧数据，或永久等待 |
| `next_tile()` 放在 `cta_sync()` 之前 | 各线程 scheduler 坐标不一致 | A/B/D 使用不同 tile offset，结果固定或随机错位 |
| 忘记在下一 tile 前等待 TMA store 完成 | Dsmem 被下一块 tile 覆盖 | 输出尾部出现旧 tile 数据 |
| 以为 148 等于硬绑定到 148 个固定 SM | 混淆 launch count 与硬件 affinity | 对 occupancy 和 tail 的判断错误 |
| 修改 `K/BLK_K/PIPE_DEPTH` 但删除 assertion | parity 条件失效 | 第一 tile 通过，后续 tile 开始异常 |

排查顺序可以固定为：

```text
先看 scheduler:
  bx 范围、linear_idx、m_idx/n_idx、valid、next_tile

再看 phase:
  K_TILES、每 stage 使用次数、tile 边界是否需要继承 parity

最后看数据路径:
  m_st/n_st 是否同时用于 A/B load 和 D store
  所有线程是否在 next_tile 前完成 epilogue
```

## 十三、与推理系统的联系

在推理系统中，persistent GEMM 常见于适合长期占用 CTA 的大矩阵乘：

```text
prefill 的大 GEMM
MoE 中 grouped GEMM 的多个小专家任务
batch 内多个独立输出的 tile 集合
```

Persistent 结构适合把固定初始化成本摊到多块 tile，也能通过
scheduler 控制 L2 locality。

但是静态 persistent scheduler 不是万能方案：

```text
tile 成本接近、SM 可用量稳定：
  Step 6 的静态 stride 通常足够

tile 成本差异大、worker 启动时间不确定：
  chapter_clc 的动态 scheduler 更合适
```

这两种实现可以共用同一个 tile mainloop。变化的只是
`m_idx/n_idx` 从哪里来，以及下一块 tile 由谁决定。

## 十四、硬件和验证边界

这段 kernel 仍使用：

```text
TMA
mbarrier
tcgen05.mma
TMEM
```

因此完整验证需要 Blackwell `sm_100a` GPU。H100 / H200 支持 TMA，
但不能执行这里的 `tcgen05` TMEM MMA 路径。当前 Apple M5 Pro 可以：

```text
运行纯 Python scheduler trace
检查 Python 语法
解析 TIRx PrimFunc
推导 tile 编号、phase 和资源生命周期
```

当前 Apple M5 Pro 不能：

```text
执行 tcgen05.mma
验证 TMA engine 的完成时序
测量 persistent kernel 的吞吐
观察真实 SM residency 和 launch tail
```

## 十五、自测

### 1. `M=N=4096`、`BLK_M=BLK_N=128` 时，总共有多少块 tile？

答：

```text
M tiles = 4096 / 128 = 32
N tiles = 4096 / 128 = 32
total = 32 * 32 = 1024
```

### 2. `SM_COUNT=148` 时，每个 CTA 处理几块 tile？

答：

```text
1024 = 148 * 6 + 136

CTA 0..135   -> 7 块
CTA 136..147 -> 6 块
```

### 3. `l2_group_size=8` 在当前网格中表示什么？

答：表示全局 work id 每 8 行 M tile 分成一组。组内按固定 N column
连续扫描 8 个 M tiles。它不是每个 CTA 的 tile 数，也不是 CTA 数量。

### 4. CTA 0 为什么处理 `(m4,n18)`，而不是 `(m0,n1)`？

答：CTA 0 第一次拿到 work id 0，对应 `(m0,n0)`。下一次执行
`next_tile()` 后：

```text
linear_idx = 0 + 148 = 148
group 0 的 within_group = 148
m = 148 % 8 = 4
n = 148 // 8 = 18
```

所以第二块是 `(m4,n18)`。

### 5. 为什么每块 tile 可以重置 phase？persistent kernel 复用和保留了哪些工作？

答：`K_TILES=64`，`PIPE_DEPTH=2`：

```text
mma_bar 每 tile 完成 64 次，64 是偶数
每个 tma_bar 每 tile 完成 32 次，32 是偶数
```

偶数次 completion 后，barrier 的下一期待 parity 回到 0，所以下一块
tile 可以从 phase 0 开始。该性质由
`K_TILES % (2 * PIPE_DEPTH) == 0` 的 assertion 保证。

复用的资源是 CTA 生命周期内的 barrier、TMEM allocation、SMEM
allocation 和 scheduler state。没有省掉每块 tile 的 TMA load、MMA、
K-loop、TMEM load 与 GMEM writeback；persistent 改变的是 tile
所有权和调度，不是 tile 内部的数据路径。

## 十六、Scheduler 追问补充：work_id、L2 locality 与 CTA 分配

这一节把 Step 6 中容易混在一起的几个概念拆开：`tile_scheduler.init(bx)`、
`work_id`、`num_clusters`、SM 与 CTA 的关系，以及
`l2_group_size` 对任务编号顺序的影响。

### 本次讲解位置

```text
本次讲解位置
章节：chapter_gemm_async
小节：Step 6: Persistent Kernel + Tile Scheduler
知识点：scheduler 状态初始化、work_id、静态 stride 与 L2 locality
上次：Step 6 的主 kernel 与资源复用
下次：Step 7: Warp Specialization 与完整 Load / Compute overlap
PTX：本补充不引入新的 PTX 指令，讨论 TMA / MMA 之前的 tile 调度
```

### 1. `tile_scheduler.init(bx)` 到底做了什么

`init` 不是初始化 CUDA，也不是把 CTA 绑定到某个 SM。它只初始化
scheduler 的逻辑状态，并立即计算当前 CTA 的第一块 output tile。

概念上等价于：

```python
linear_idx = bx
tile_count = 0
m_idx, n_idx = decode_group_major(linear_idx)
```

其中：

```text
bx          = 当前 CTA 编号，范围 0..147
linear_idx  = 当前 CTA 对应的全局 work id
tile_count  = 当前 CTA 已经领取多少块 tile
m_idx       = 当前 tile 的 M 方向编号
n_idx       = 当前 tile 的 N 方向编号
```

例如：

```text
CTA 0:
  init(0)
  linear_idx = 0
  (m_idx, n_idx) = (0, 0)

CTA 136:
  init(136)
  linear_idx = 136
  (m_idx, n_idx) = (0, 17)

CTA 147:
  init(147)
  linear_idx = 147
  (m_idx, n_idx) = (3, 18)
```

第一块 tile 的坐标已经由 `init` 计算出来，所以 `while` 第一次进入
循环时不需要先调用 `next_tile()`。

### 2. `work_id` 的角色

`work_id` 有时也叫 `linear_idx` 或 `work_idx`。它是：

```text
二维 tile 任务在一维调度顺序中的编号
```

它不是：

```text
SM 编号
CTA 编号
m_idx
n_idx
A/B/D 的内存地址
```

当前共有：

```text
32 * 32 = 1024 个 output tiles
```

所以：

```text
work_id = 0..1023
```

`work_id` 先经过 scheduler 解码，再变成二维坐标：

```text
work_id -> decode(work_id) -> (m_idx, n_idx)
```

三个编号的职责可以对齐如下：

| 编号 | 范围 | 角色 |
|---|---:|---|
| `bx` | `0..147` | worker 身份，也就是 persistent CTA 编号 |
| `work_id` | `0..1023` | 全局任务编号，决定任务顺序 |
| `(m_idx,n_idx)` | `0..31` | output tile 的二维逻辑坐标 |

物理地址来自 `(m_idx,n_idx)`，不直接来自 `work_id`：

```python
m_st = tile_scheduler.m_idx * BLK_M
n_st = tile_scheduler.n_idx * BLK_N
```

例如 CTA 0 第二步拿到：

```text
work_id = 148
decode(148) = (m_idx=4, n_idx=18)
```

于是：

```text
m_st = 4 * 128 = 512
n_st = 18 * 128 = 2304
```

`148` 本身不会作为矩阵下标使用。

### 3. 为什么 `next_tile()` 让 work id 加 148

当前 scheduler 有 148 个逻辑 worker。每个 worker 从自己的 `bx`
开始，然后按 worker 数量向前走：

```text
work_id = bx + k * 148
k = 0, 1, 2, ...
```

因此：

```text
CTA 0   -> 0,   148, 296, 444, ...
CTA 1   -> 1,   149, 297, 445, ...
CTA 2   -> 2,   150, 298, 446, ...
...
CTA 147 -> 147, 295, 443, 591, ...
```

这些序列的余数互不相同：

```text
CTA 0   -> work_id % 148 = 0
CTA 1   -> work_id % 148 = 1
...
CTA 147 -> work_id % 148 = 147
```

因此不会出现两个 CTA 领取同一个 `work_id`，也不会漏掉中间的任务。
所有序列合起来正好覆盖：

```text
0, 1, 2, ..., 1023
```

这里的 148 来自 worker 数量：

```text
num_clusters = SM_COUNT = 148
```

它不是 tile 数量，也不是 K 维参数。当前 cluster 配置是：

```text
cluster_m = 1
cluster_n = 1
cta_group = 1
```

所以一个 scheduler cluster 正好对应一个 CTA。

如果以后一个 cluster 包含两个 CTA，那么 scheduler 的 stride 应该按
cluster 数量计算，而不是继续使用 CTA 数量。否则两个 CTA 会错误地
领取不同的 scheduler 任务。

### 4. `num_clusters`、SM、CTA、tile 不是一回事

| 概念 | 当前值 | 含义 |
|---|---:|---|
| SM 数量 | 约 148 | 硬件计算单元数量 |
| CTA 数量 | 148 | kernel 实际启动的 persistent worker |
| scheduler cluster 数量 | 148 | 当前一个 cluster 对应一个 CTA |
| output tile 数量 | 1024 | 总任务数量 |
| 每 CTA tile 数量 | 6 或 7 | 静态分配结果 |

因此：

```text
一个 SM 不是一个 tile。
一个 CTA 也不只处理一个 tile。
一个 CTA 在任一时刻处理一块 tile，完成后继续领取下一块。
```

`SM_COUNT=148` 的设计目标是在 B200 上近似做到“一个 SM 一个
persistent CTA”。但语言层面没有把某个 CTA 永久绑定到某个 SM。
CTA 实际放在哪个 SM、何时被调度，仍由硬件决定。

当前 CTA 又分配了：

```python
n_cols=512
```

而 TMEM 的一颗 SM 总布局是：

```text
128 lanes x 512 columns
```

所以该 kernel 的资源占用会强烈接近“一 SM 一 CTA”的执行模型。
这是资源 residency 的结果，不是 CTA 与 SM 的固定 affinity。

### 5. `l2_group_size` 改变的是什么

`l2_group_size` 不改变：

```text
CTA 数量
每个 CTA 的 stride
总 tile 数量
```

它只改变：

```text
work_id -> (m_idx,n_idx)
```

也就是全局任务顺序。

为了把三种顺序完整看清，使用一个小网格：

```text
M tiles = 16
N tiles = 8
l2_group_size = 8
```

#### M 优先

M 优先表示外层遍历 `m`，内层遍历 `n`：

```text
0   -> (m0,n0)
1   -> (m0,n1)
...
7   -> (m0,n7)
8   -> (m1,n0)
...
127 -> (m15,n7)
```

矩阵形式：

| m\n | n0 | n1 | n2 | n3 | n4 | n5 | n6 | n7 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
| 1 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 |
| 2 | 16 | 17 | 18 | 19 | 20 | 21 | 22 | 23 |
| 3 | 24 | 25 | 26 | 27 | 28 | 29 | 30 | 31 |
| 4 | 32 | 33 | 34 | 35 | 36 | 37 | 38 | 39 |
| 5 | 40 | 41 | 42 | 43 | 44 | 45 | 46 | 47 |
| 6 | 48 | 49 | 50 | 51 | 52 | 53 | 54 | 55 |
| 7 | 56 | 57 | 58 | 59 | 60 | 61 | 62 | 63 |
| 8 | 64 | 65 | 66 | 67 | 68 | 69 | 70 | 71 |
| 9 | 72 | 73 | 74 | 75 | 76 | 77 | 78 | 79 |
| 10 | 80 | 81 | 82 | 83 | 84 | 85 | 86 | 87 |
| 11 | 88 | 89 | 90 | 91 | 92 | 93 | 94 | 95 |
| 12 | 96 | 97 | 98 | 99 | 100 | 101 | 102 | 103 |
| 13 | 104 | 105 | 106 | 107 | 108 | 109 | 110 | 111 |
| 14 | 112 | 113 | 114 | 115 | 116 | 117 | 118 | 119 |
| 15 | 120 | 121 | 122 | 123 | 124 | 125 | 126 | 127 |

特征是：

```text
A_m 连续复用
B_n 的复用距离更大
```

例如 `A0` 被 work id `0..7` 连续使用，而 `B0` 下一次出现在
work id `8`。

#### N 优先

N 优先表示外层遍历 `n`，内层遍历 `m`：

```text
0   -> (m0,n0)
1   -> (m1,n0)
...
15  -> (m15,n0)
16  -> (m0,n1)
...
127 -> (m15,n7)
```

矩阵形式：

| m\n | n0 | n1 | n2 | n3 | n4 | n5 | n6 | n7 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 16 | 32 | 48 | 64 | 80 | 96 | 112 |
| 1 | 1 | 17 | 33 | 49 | 65 | 81 | 97 | 113 |
| 2 | 2 | 18 | 34 | 50 | 66 | 82 | 98 | 114 |
| 3 | 3 | 19 | 35 | 51 | 67 | 83 | 99 | 115 |
| 4 | 4 | 20 | 36 | 52 | 68 | 84 | 100 | 116 |
| 5 | 5 | 21 | 37 | 53 | 69 | 85 | 101 | 117 |
| 6 | 6 | 22 | 38 | 54 | 70 | 86 | 102 | 118 |
| 7 | 7 | 23 | 39 | 55 | 71 | 87 | 103 | 119 |
| 8 | 8 | 24 | 40 | 56 | 72 | 88 | 104 | 120 |
| 9 | 9 | 25 | 41 | 57 | 73 | 89 | 105 | 121 |
| 10 | 10 | 26 | 42 | 58 | 74 | 90 | 106 | 122 |
| 11 | 11 | 27 | 43 | 59 | 75 | 91 | 107 | 123 |
| 12 | 12 | 28 | 44 | 60 | 76 | 92 | 108 | 124 |
| 13 | 13 | 29 | 45 | 61 | 77 | 93 | 109 | 125 |
| 14 | 14 | 30 | 46 | 62 | 78 | 94 | 110 | 126 |
| 15 | 15 | 31 | 47 | 63 | 79 | 95 | 111 | 127 |

特征是：

```text
B_n 连续复用
A_m 的复用距离更大
```

例如 `B0` 被 work id `0..15` 连续使用，而 `A0` 下一次出现在
work id `16`。

#### `l2_group_size=8`

每 8 行 M tile 组成一组：

```text
group 0: m0..m7
group 1: m8..m15
```

组内固定 `n`，扫描 8 个 `m`：

```text
0  -> (m0,n0)
1  -> (m1,n0)
...
7  -> (m7,n0)
8  -> (m0,n1)
9  -> (m1,n1)
...
63 -> (m7,n7)
64 -> (m8,n0)
...
127 -> (m15,n7)
```

矩阵形式：

| m\n | n0 | n1 | n2 | n3 | n4 | n5 | n6 | n7 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 8 | 16 | 24 | 32 | 40 | 48 | 56 |
| 1 | 1 | 9 | 17 | 25 | 33 | 41 | 49 | 57 |
| 2 | 2 | 10 | 18 | 26 | 34 | 42 | 50 | 58 |
| 3 | 3 | 11 | 19 | 27 | 35 | 43 | 51 | 59 |
| 4 | 4 | 12 | 20 | 28 | 36 | 44 | 52 | 60 |
| 5 | 5 | 13 | 21 | 29 | 37 | 45 | 53 | 61 |
| 6 | 6 | 14 | 22 | 30 | 38 | 46 | 54 | 62 |
| 7 | 7 | 15 | 23 | 31 | 39 | 47 | 55 | 63 |
| 8 | 64 | 72 | 80 | 88 | 96 | 104 | 112 | 120 |
| 9 | 65 | 73 | 81 | 89 | 97 | 105 | 113 | 121 |
| 10 | 66 | 74 | 82 | 90 | 98 | 106 | 114 | 122 |
| 11 | 67 | 75 | 83 | 91 | 99 | 107 | 115 | 123 |
| 12 | 68 | 76 | 84 | 92 | 100 | 108 | 116 | 124 |
| 13 | 69 | 77 | 85 | 93 | 101 | 109 | 117 | 125 |
| 14 | 70 | 78 | 86 | 94 | 102 | 110 | 118 | 126 |
| 15 | 71 | 79 | 87 | 95 | 103 | 111 | 119 | 127 |

特征是：

```text
B_n 在连续 8 个 work id 中复用
A_m 每隔 8 个 work id 再次出现
```

例如：

```text
B0：work id 0..7
A0：work id 0, 8, 16, 24, ...
```

### 6. 为什么这样对 L2 有帮助

每个 output tile `(m,n)` 都需要：

```text
A_m：128 x K
B_n：128 x K
```

不同 tile 会共享 A 或 B。如果两个 tile 的 work id 离得很近，它们
更可能在相近时间访问同一份 A/B 数据，也就更有机会命中 L2。

当前 `32 x 32` tile 网格在第一个 wave 有 148 个 work id。粗看三个
顺序的工作集：

| 顺序 | 第一个 wave 的 A tile | 第一个 wave 的 B tile | 粗略工作集 |
|---|---:|---:|---:|
| M 优先 | 5 | 32 | `5 + 32 = 37` |
| N 优先 | 32 | 5 | `32 + 5 = 37` |
| `l2_group_size=8` | 8 | 19 | `8 + 19 = 27` |

每个完整 A/B tile 在 `K=4096`、fp16 下约为：

```text
128 * 4096 * 2 bytes = 1 MiB
```

所以这三种顺序的粗略 working set 大约分别是：

```text
M 优先：37 MiB
N 优先：37 MiB
group=8：27 MiB
```

这个计算没有考虑 pipeline stage、L2 容量和 CTA 实际推进速度，只
用于解释编号顺序为什么会影响缓存压力。

需要强调：

```text
l2_group_size 只提高 L2 复用的概率。
它不保证每个请求都命中 L2。
它也不保证 CTA 按照全局 work id 完全同步执行。
```

### 7. scheduler 绑定后，后面是不是普通流水线

对于当前 kernel，可以这样理解：

```text
scheduler 外层：
  决定当前 CTA 处理哪块 tile

K-loop 内层：
  仍然是 Step 5 的 TMA + MMA 软件流水线
```

结构是：

```python
tile_scheduler.init(bx)

while tile_scheduler.valid():
    m_st = tile_scheduler.m_idx * BLK_M
    n_st = tile_scheduler.n_idx * BLK_N

    # Step 5 的完整 tile 计算流水线
    # prologue TMA
    # K-loop MMA
    # epilogue writeback

    T.cuda.cta_sync()
    tile_scheduler.next_tile()
```

“逻辑分配固定”和“物理执行固定”要分开：

| 项目 | 是否固定 |
|---|---|
| CTA 的 work id 序列 | 固定 |
| work id 到 `(m_idx,n_idx)` 的映射 | 固定 |
| 每块 tile 的 `m_st/n_st` | 固定 |
| CTA 运行在哪颗 SM | 硬件决定，不固定 |
| CTA 何时开始执行 | 硬件决定，不保证 |
| 各 CTA 是否同步推进 | 不保证 |

所以后面大部分确实只是原有 pipeline，但外层 loop 仍有两个硬性
约束：

```text
所有线程完成当前 tile 后，才能 next_tile()
每块 tile 重置 phase 前，必须满足偶数次 completion 的 parity 条件
```

### 8. 常见错误

| 错误理解或实现 | 后果 |
|---|---|
| 把 `work_id` 当成 A/B/D 地址 | tile 坐标和内存 offset 混乱 |
| `next_tile()` 只加 1 | 多个 CTA 重复处理相同 tile |
| 把 `SM_COUNT` 当成 tile 数量 | 误判每个 CTA 的任务数 |
| 认为 CTA 永久绑定某个 SM | 错误推断 residency 和负载 |
| 在 `cta_sync()` 前调用 `next_tile()` | 不同线程使用不同 `m_st/n_st` |
| 认为 `l2_group_size` 保证 L2 命中 | 用逻辑顺序代替性能测量 |
| 多 CTA cluster 仍使用 CTA 数量作为 stride | cluster 之间任务划分错误 |

### 9. 补充自测

#### 1. `tile_scheduler.init(bx)` 会得到什么状态？

答：得到当前 worker 的初始 `linear_idx=bx`、`tile_count=0`，并
立即解码出第一块 tile 的 `m_idx/n_idx`。它不会调用
`next_tile()`，也不会绑定物理 SM。

#### 2. `work_id` 和 `(m_idx,n_idx)` 的区别是什么？

答：`work_id` 是任务在全局调度顺序中的一维编号；`(m_idx,n_idx)`
是任务对应的 output tile 二维坐标。地址由 `(m_idx,n_idx)` 计算，
而不是直接使用 `work_id`。

#### 3. 为什么 `next_tile()` 让 work id 增加 148？

答：当前有 148 个 scheduler cluster，每个 cluster 对应一个 CTA。
每个 worker 从 `bx` 开始，按 worker 数量做 stride，保证 148 个
序列互不重叠并覆盖所有 tile。

#### 4. `SM_COUNT=148` 是否表示一个 SM 一块 tile？

答：不是。它启动 148 个 CTA，每个 CTA 处理 6 或 7 块 tile。设计
目标是接近一 SM 一个 persistent CTA，但 CTA 与 SM 没有固定绑定。

#### 5. 为什么 `l2_group_size=8` 对 L2 locality 有好处？

答：它让固定 N column 的 8 个 M tile 连续出现，使 `B_n` 在很短
的 work id 窗口中复用；同时 `A_m` 每隔 8 个 work id 再次出现。
相比 M 优先或 N 优先，它把 A/B 工作集控制在更平衡、通常更小的
范围内，从而提高 L2 命中的概率。
