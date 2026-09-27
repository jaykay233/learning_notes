# GEMM Advanced Step 8-9：Two-CTA Cluster 与 Multi-Consumer Warp Specialization

这篇笔记进入 `chapter_gemm_advanced` 的 Step 8。Step 7 已经在一个 CTA
内部把 TMA producer、MMA consumer 和 writeback 拆成三个角色；Step 8
把协作范围继续扩大到一个包含两个 CTA 的 cluster。

本文先讲 Step 8 的五个小知识点：

```text
1. 两个 CTA 为什么只各自加载一半 A/B，却能共同计算一个
   256 x 256 的 output tile。
2. m_st / n_st / n_st_epi 分别属于哪条数据路径，以及 epilogue
   为什么要把 256 columns 分成两段 128-column 写回。
3. 两个 CTA 的 TMA transaction 为什么集中完成到 CTA0 的
   tma2mma barrier，以及为什么每个 stage 登记 65536 bytes。
4. cta_group=2 如何让一次 MMA 读取 CTA pair 的 SMEM，并更新两侧
   TMEM；cta_mask=3 又如何把同一个 MMA 的完成事件通知给两侧。
5. 两个 writeback warpgroup 的 256 个 arrivals 如何通过 CTA0 的
   ld2mma barrier，保证下一块 tile 不会过早覆盖 TMEM。
6. 如何增加第二个 MMA consumer，让两个 consumers 共享同一份 staged B，
   把 cluster output tile 从 256 x 256 扩展到 512 x 256，并把 TMEM
   和 barrier 从按 stage 索引改为按 consumer 索引。
```

## 本次讲解位置

```text
本次讲解位置
章节：chapter_gemm_advanced
小节：Step 8: Two-CTA Cluster
知识点：CTA0/CTA1 的 A/B slice 所有权，以及 cooperative MMA 产生的
        256 x 256 output tile 与 TMEM 切分
上次：Step 7: Warp Specialization 与四条 barrier 交接
下次：Step 8.4：cta_group=2 cooperative MMA 与 cta_mask=3 completion；
      本文末尾继续 Step 9
PTX：tcgen05.mma cta_group::2、cp.async.bulk.tensor、
     mbarrier remote arrive、tcgen05.commit cta_mask
```

### Step 8 知识点清单

```text
[x] 8.1 Two-CTA cluster 的 A/B 所有权与 256 x 256 output tile
[x] 8.2 Tile 地址计算与 epilogue 的两段 128-column 写回
[x] 8.3 CTA0 集中式 tma2mma barrier 与 65536-byte transaction
[x] 8.4 cta_group=2 cooperative MMA 与 cta_mask=3 completion
[x] 8.5 ld2mma 的 256 arrivals 与跨 CTA TMEM 复用
```

## 一、问题：Step 7 每个 CTA 仍然重复加载 operand

Step 7 以 `128 x 128` output tile 为调度单位。对相邻的 `2 x 2` tile 区域：

```text
D00 = A0 @ B0.T
D01 = A0 @ B1.T
D10 = A1 @ B0.T
D11 = A1 @ B1.T
```

如果四块 tile 完全独立执行，每个 CTA 都会加载自己需要的 A/B slice：

```text
CTA D00: A0 + B0
CTA D01: A0 + B1
CTA D10: A1 + B0
CTA D11: A1 + B1
```

于是：

```text
A0 被加载 2 次
A1 被加载 2 次
B0 被加载 2 次
B1 被加载 2 次
合计 8 个 128 x K slice
```

Step 8 的目标是让两个 CTA 一起覆盖一个 `256 x 256` output tile：

```text
CTA0 只加载 A0 和 B0
CTA1 只加载 A1 和 B1

cluster 合计只加载 4 个 128 x K slice
```

对于同一个 `256 x 256` output 区域，operand traffic 从 8 个 slice
降到 4 个 slice，也就是减半。这是 Two-CTA Cluster 的主要收益。

## 二、心智模型：一份 operand 跨 CTA 参与四次计算

先不要从 barrier 或 `cta_group=2` 开始。先用数据所有权理解：

```text
A 的两半决定 output 的两组 rows
B 的两半决定 output 的两组 columns
```

因此：

| CTA | 加载的 A slice | 加载的 stored-B slice | 自己持有的 TMEM rows |
|---|---|---|---|
| CTA 0 (`cbx=0`) | `A[m: m + 128, :]` | `B[n: n + 128, :]` | output rows `m : m + 128` |
| CTA 1 (`cbx=1`) | `A[m + 128 : m + 256, :]` | `B[n + 128 : n + 256, :]` | output rows `m + 128 : m + 256` |

注意 `B` 的 stored shape 是 `N x K`，kernel 计算的是：

```python
D = A @ B.T
```

所以 `B[n : n + 128, :]` 对应的是 output 的 columns
`n : n + 128`，不是 output rows。

两个 CTA 的 SMEM operand 组成一个完整的 `A/B` 对：

```text
CTA0 SMEM: A0, B0
CTA1 SMEM: A1, B1
```

cooperative MMA 会跨 CTA 读取两边 SMEM，于是产生四个 subproduct：

| 计算 | output quadrant | 写入哪个 CTA 的 TMEM |
|---|---|---|
| `A0 @ B0.T` | `D[m:m+128, n:n+128]` | CTA0，columns `0:128` |
| `A0 @ B1.T` | `D[m:m+128, n+128:n+256]` | CTA0，columns `128:256` |
| `A1 @ B0.T` | `D[m+128:m+256, n:n+128]` | CTA1，columns `0:128` |
| `A1 @ B1.T` | `D[m+128:m+256, n+128:n+256]` | CTA1，columns `128:256` |

每个 CTA 的 TMEM accumulator 因此是：

```text
128 rows x 256 columns
```

CTA0 拥有 output 的前 128 rows；CTA1 拥有 output 的后 128 rows。
两边合起来就是完整的 `256 x 256` output tile。

## 三、完整 Kernel

下面代码与课程 Step 8 对应，只增加了不同 TVM wheel 下
`tma_utils` 的兼容 import。

### 文件一：`tirx_gemm_two_cta_cluster.py`

```python
import tvm
from tvm.script import tirx as T
from tvm.script.tirx import tile as Tx
from tvm.tirx.layout import TileLayout, S, TLane, TCol, tid_in_wg

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

from tvm.backend.cuda.lang.pipeline import (
    TMABar,
    TCGen05Bar,
    MBarrier,
    PipelineState,
)
from tvm.backend.cuda.lang.tile_scheduler import (
    ClusterPersistentScheduler2D,
)


SM_COUNT = 148
F16_SIZE = 2


def hgemm_v8(M, N, K):
    a_type = tvm.DataType("float16")
    b_type = tvm.DataType("float16")
    d_type = tvm.DataType("float16")
    acc_type = tvm.DataType("float32")

    CTA_GROUP = 2
    BLK_M, BLK_N, BLK_K = 128, 128, 64
    MMA_M, MMA_N = 256, 256
    K_TILES = K // BLK_K
    PIPE_DEPTH = 4
    WG_NUMBER = 2

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
        (BLK_M, 128),
    )

    @T.prim_func
    def kernel(
        A: T.Buffer((M, K), a_type),
        B: T.Buffer((N, K), b_type),
        D: T.Buffer((M, N), d_type),
    ):
        T.device_entry()
        bx = T.cta_id([SM_COUNT])
        cbx, cby = T.cta_id_in_cluster([CTA_GROUP, 1])
        wg_id = T.warpgroup_id([WG_NUMBER])
        warp_id = T.warp_id_in_wg([4])
        lane_id = T.lane_id([32])

        pool = T.SMEMPool()
        tmem_addr = pool.alloc((1,), "uint32")
        tma2mma = TMABar(pool, PIPE_DEPTH)
        mma2tma = TCGen05Bar(pool, PIPE_DEPTH)
        mma2ld = TCGen05Bar(pool, 1)
        ld2mma = MBarrier(pool, 1)
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
            (BLK_M, 128),
            d_type,
            layout=D_layout,
        )

        tma2mma.init(1)
        mma2tma.init(1)
        mma2ld.init(1)
        ld2mma.init(128 * CTA_GROUP)
        pool.commit()

        if wg_id == 0:
            if warp_id == 0:
                T.ptx.tcgen05.alloc(
                    T.address_of(tmem_addr),
                    n_cols=512,
                    cta_group=CTA_GROUP,
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

        tile_scheduler = ClusterPersistentScheduler2D(
            "ts",
            num_m_tiles=M // 256,
            num_n_tiles=N // 256,
            l2_group_size=8,
            num_clusters=SM_COUNT // CTA_GROUP,
        )
        tile_scheduler.init(bx // CTA_GROUP)

        m_idx = T.meta_var(tile_scheduler.m_idx)
        n_idx = T.meta_var(tile_scheduler.n_idx)
        m_st = T.meta_var(
            (m_idx * CTA_GROUP + cbx) * BLK_M
        )
        n_st = T.meta_var(
            (n_idx * CTA_GROUP + cbx) * BLK_N
        )

        tma2mma_cta0 = tma2mma.remote_view(0)
        ld2mma_cta0 = ld2mma.remote_view(0)

        if wg_id == 1:
            if warp_id == 3:
                tma_ps = PipelineState(PIPE_DEPTH, phase=1)

                @T.inline
                def tma_load(k_offset):
                    Tx.copy_async(
                        Asmem[tma_ps.stage, :, :],
                        A[
                            m_st : m_st + BLK_M,
                            k_offset : k_offset + BLK_K,
                        ],
                        dispatch="tma_auto",
                        cta_group=CTA_GROUP,
                        mbar=tma2mma_cta0.ptr_to(
                            [tma_ps.stage]
                        ),
                    )
                    Tx.copy_async(
                        Bsmem[tma_ps.stage, :, :],
                        B[
                            n_st : n_st + BLK_N,
                            k_offset : k_offset + BLK_K,
                        ],
                        dispatch="tma_auto",
                        cta_group=CTA_GROUP,
                        mbar=tma2mma_cta0.ptr_to(
                            [tma_ps.stage]
                        ),
                    )

                if T.filter(lane_id, T.ptx.elect_sync()):
                    while tile_scheduler.valid():
                        for k in range(K_TILES):
                            mma2tma.wait(
                                tma_ps.stage,
                                tma_ps.phase,
                            )
                            tma_load(k * BLK_K)
                            if cbx == 0:
                                tma2mma_cta0.arrive(
                                    tma_ps.stage,
                                    CTA_GROUP
                                    * (
                                        BLK_M * BLK_K
                                        + BLK_N * BLK_K
                                    )
                                    * F16_SIZE,
                                )
                            tma_ps.advance()
                        tile_scheduler.next_tile()

            elif warp_id == 0:
                mma_ps = PipelineState(PIPE_DEPTH, phase=0)
                ld_ps = PipelineState(1, phase=1)

                if cbx == 0:
                    if T.filter(lane_id, T.ptx.elect_sync()):
                        while tile_scheduler.valid():
                            ld2mma.wait(
                                ld_ps.stage,
                                ld_ps.phase,
                            )
                            ld_ps.advance()

                            for k in range(K_TILES):
                                tma2mma.wait(
                                    mma_ps.stage,
                                    mma_ps.phase,
                                )
                                Tx.gemm_async(
                                    tmem[:, :MMA_N],
                                    Asmem[mma_ps.stage, :, :],
                                    Bsmem[mma_ps.stage, :, :],
                                    accum=(k != 0),
                                    dispatch="tcgen05",
                                    cta_group=CTA_GROUP,
                                )
                                mma2tma.arrive(
                                    mma_ps.stage,
                                    cta_group=CTA_GROUP,
                                    cta_mask=3,
                                )
                                mma_ps.advance()

                            mma2ld.arrive(
                                0,
                                cta_group=CTA_GROUP,
                                cta_mask=3,
                            )
                            tile_scheduler.next_tile()

        elif wg_id == 0:
            wb_ps = PipelineState(1, phase=0)
            reg_f16 = T.alloc_local((128,), d_type)

            while tile_scheduler.valid():
                mma2ld.wait(
                    wb_ps.stage,
                    wb_ps.phase,
                )
                wb_ps.advance()
                T.ptx.tcgen05.fence.after_thread_sync()

                for no in T.unroll(2):
                    reg = T.alloc_local(
                        (128,),
                        acc_type,
                    )
                    reg_wg = reg.view(
                        128,
                        128,
                        layout=TileLayout(
                            S[
                                (128, 128)
                                : (1 @ tid_in_wg, 1)
                            ]
                        ),
                    )
                    Tx.wg.copy_async(
                        reg_wg[:],
                        tmem[
                            :,
                            no * 128 : (no + 1) * 128,
                        ],
                    )
                    T.ptx.tcgen05.wait.ld()
                    Tx.cast(reg_f16[:], reg[:])
                    Tx.copy(
                        Dsmem[
                            warp_id * 32 + lane_id,
                            :,
                        ],
                        reg_f16[:],
                    )
                    T.ptx.fence.proxy_async("shared::cta")
                    T.cuda.warpgroup_sync(10)
                    if warp_id == 0:
                        if lane_id == 0:
                            n_st_epi = T.meta_var(
                                n_idx * 256 + no * 128
                            )
                            Tx.copy_async(
                                D[
                                    m_st : m_st + BLK_M,
                                    n_st_epi : n_st_epi + 128,
                                ],
                                Dsmem[:, :],
                                dispatch="tma_auto",
                            )
                            T.ptx.cp_async.bulk.commit_group()
                            T.ptx.cp_async.bulk.wait_group(0)
                    T.cuda.warpgroup_sync(10)

                ld2mma_cta0.arrive(0)
                tile_scheduler.next_tile()

        T.cuda.cluster_sync()
        if warp_id == 0:
            T.ptx.tcgen05.relinquish_alloc_permit(
                cta_group=CTA_GROUP
            )
            T.ptx.tcgen05.dealloc(
                tmem_addr[0],
                n_cols=512,
                cta_group=CTA_GROUP,
            )

    return kernel
```

## 四、按数据所有权拆解

### 1. Cluster tile 是 `256 x 256`

```python
tile_scheduler = ClusterPersistentScheduler2D(
    "ts",
    num_m_tiles=M // 256,
    num_n_tiles=N // 256,
    l2_group_size=8,
    num_clusters=SM_COUNT // CTA_GROUP,
)
```

这里的 scheduler 不再返回 `128 x 128` tile coordinate，而是返回
`256 x 256` cluster tile coordinate：

```text
m_idx, n_idx
```

两个 CTA 共享同一个 `(m_idx, n_idx)`，因为它们在合作计算同一块
`256 x 256` output。

### 2. `cbx` 决定当前 CTA 的 slice

```python
cbx, cby = T.cta_id_in_cluster([CTA_GROUP, 1])

m_st = (m_idx * CTA_GROUP + cbx) * BLK_M
n_st = (n_idx * CTA_GROUP + cbx) * BLK_N
```

当 `m_idx=0`、`n_idx=0` 时：

```text
cbx = 0:
    m_st = 0
    n_st = 0
    A slice = A[0:128, :]
    B slice = B[0:128, :]

cbx = 1:
    m_st = 128
    n_st = 128
    A slice = A[128:256, :]
    B slice = B[128:256, :]
```

`m_st` 同时也是 CTA 最终写回的 output row 起点。

`n_st` 只用于选择当前 CTA 提供给 cooperative MMA 的 stored-B slice。
它不是当前 CTA 最终写回的 column 起点。Epilogue 会把 256 columns
拆成两个 128-column chunk，所以写回位置使用：

```python
n_st_epi = n_idx * 256 + no * 128
```

### 3. 四个 quadrant 来自同一个 cooperative MMA

Step 8 仍然只发起一条 MMA dispatch：

```python
Tx.gemm_async(
    tmem[:, :MMA_N],
    Asmem[mma_ps.stage, :, :],
    Bsmem[mma_ps.stage, :, :],
    accum=(k != 0),
    dispatch="tcgen05",
    cta_group=CTA_GROUP,
)
```

区别是：

```text
cta_group=2
```

它会覆盖 CTA pair 的两份 SMEM 和两份 TMEM：

```text
读取 CTA0.Asmem 和 CTA1.Asmem
读取 CTA0.Bsmem 和 CTA1.Bsmem
更新 CTA0.tmem 和 CTA1.tmem
```

最终得到：

```text
CTA0 TMEM:
    row 0:128，column 0:256

CTA1 TMEM:
    row 0:128，column 0:256
```

这里的 TMEM row 是每个 CTA 的本地 accumulator row；映射回全局
output 时，CTA0 对应前 128 rows，CTA1 对应后 128 rows。

### 4. TMA completion 集中记在 CTA0

两个 CTA 的 TMA load 都传入同一个 remote barrier：

```python
tma2mma_cta0 = tma2mma.remote_view(0)
```

每次 load 时：

```python
mbar=tma2mma_cta0.ptr_to([tma_ps.stage])
```

因此两侧的 TMA transaction 都更新 CTA0 的 `tma2mma`。

CTA0 中选出的 producer lane 只执行一次：

```python
if cbx == 0:
    tma2mma_cta0.arrive(
        tma_ps.stage,
        CTA_GROUP
        * (BLK_M * BLK_K + BLK_N * BLK_K)
        * F16_SIZE,
    )
```

这里的第二项是 `tx_count`，不是普通 arrival count。对于
`BLK_M=BLK_N=128`、`BLK_K=64`、fp16：

```text
单个 CTA 的 A bytes:
128 * 64 * 2 = 16384

单个 CTA 的 B bytes:
128 * 64 * 2 = 16384

单个 CTA subtotal:
32768 bytes

两个 CTA total:
32768 * 2 = 65536 bytes
```

所以 phase 完成条件是：

```text
CTA0 的 1 次 arrival
+
CTA0 和 CTA1 的 65536 个 TMA bytes 全部完成
```

这保证 CTA0 的 MMA consumer 不会在另一侧 operand 尚未到时开始
cooperative MMA。

### 5. Completion 通知两侧

cooperative MMA 只由 CTA0 发起一次，但结果会更新两侧 TMEM。因此
每次 MMA 完后需要通知两侧：

```python
mma2tma.arrive(
    mma_ps.stage,
    cta_group=CTA_GROUP,
    cta_mask=3,
)
```

`cta_mask=3` 的二进制是：

```text
11
```

bit 0 表示 cluster rank 0，bit 1 表示 cluster rank 1：

```text
mma2tma[stage] phase complete
-> CTA0 producer may reuse SMEM stage
-> CTA1 producer may reuse SMEM stage
```

整个 K-loop 结束后：

```python
mma2ld.arrive(
    0,
    cta_group=CTA_GROUP,
    cta_mask=3,
)
```

两侧 writeback 随后分别读取自己的 TMEM rows。

### 6. TMEM 的释放也必须等两个 CTA

每个 CTA 的 writeback warpgroup 有 128 threads：

```python
ld2mma_cta0.arrive(0)
```

两个 CTA 总共产生：

```text
128 * 2 = 256 arrivals
```

因此初始化必须是：

```python
ld2mma.init(128 * CTA_GROUP)
```

只有两个 CTA 的 writeback 都读完各自 TMEM，CTA0 才能允许下一块
cluster tile 的 cooperative MMA 覆盖 accumulator。

## 五、具体执行 trace

设：

```text
M = N = 4096
K = 128
m_idx = 0
n_idx = 0
cluster tile = D[0:256, 0:256]
BLK_M = BLK_N = 128
BLK_K = 64
```

### 1. 两个 CTA 的 slice

| `cbx` | A slice | stored-B slice | 本地 TMEM rows | 全局 output rows |
|---:|---|---|---|---|
| 0 | `A[0:128, :]` | `B[0:128, :]` | `0:128` | `D[0:128, :]` |
| 1 | `A[128:256, :]` | `B[128:256, :]` | `0:128` | `D[128:256, :]` |

两个本地 TMEM row 编号都是 `0:128`，因为它们属于不同的 CTA
allocation。映射回全局 output 时加上各自的 `m_st`。

### 2. Cooperative tile 的四个 quadrant

| Quadrant | 使用的 A | 使用的 stored-B | 最终全局位置 |
|---|---|---|---|
| `D00` | `A[0:128]` | `B[0:128]` | `D[0:128, 0:128]` |
| `D01` | `A[0:128]` | `B[128:256]` | `D[0:128, 128:256]` |
| `D10` | `A[128:256]` | `B[0:128]` | `D[128:256, 0:128]` |
| `D11` | `A[128:256]` | `B[128:256]` | `D[128:256, 128:256]` |

### 3. K tile 的 byte count

第一次 K iteration，`BLK_K=64`：

| CTA | A bytes | B bytes | subtotal |
|---:|---:|---:|---:|
| 0 | `128*64*2 = 16384` | `128*64*2 = 16384` | `32768` |
| 1 | `128*64*2 = 16384` | `128*64*2 = 16384` | `32768` |
| 合计 |  |  | `65536` |

所以每个 K stage 的 `tma2mma` phase 需要等待 `65536` TMA bytes。

### 4. Epilogue 的两段写回

CTA0 的 TMEM 是：

```text
128 rows x 256 columns
```

Epilogue 分成：

```text
no=0:
    D[m_st:m_st+128, n_base:n_base+128]

no=1:
    D[m_st:m_st+128, n_base+128:n_base+256]
```

CTA0 和 CTA1 的 `m_st` 分别是 0 和 128，因此最终覆盖：

```text
CTA0:
    D[0:128, 0:128]
    D[0:128, 128:256]

CTA1:
    D[128:256, 0:128]
    D[128:256, 128:256]
```

四块 `128 x 128` 合并成完整 `256 x 256` output tile。

## 六、可运行的纯 Python trace

### 文件二：`trace_two_cta_cluster_ownership.py`

```python
CTA_GROUP = 2
BLK_M = 128
BLK_N = 128
BLK_K = 64
F16_SIZE = 2

m_idx = 0
n_idx = 0

print("cluster tile:", (m_idx, n_idx))

for cbx in range(CTA_GROUP):
    m_st = (m_idx * CTA_GROUP + cbx) * BLK_M
    n_st = (n_idx * CTA_GROUP + cbx) * BLK_N
    print()
    print(f"CTA {cbx}")
    print(f"  A slice: A[{m_st}:{m_st + BLK_M}]")
    print(f"  B slice: B[{n_st}:{n_st + BLK_N}]")
    print(
        "  output rows: "
        f"D[{m_st}:{m_st + BLK_M}, :]"
    )

tx_count = (
    CTA_GROUP
    * (BLK_M * BLK_K + BLK_N * BLK_K)
    * F16_SIZE
)
print()
print("TMA bytes per K stage:", tx_count)
print("ld2mma arrivals:", 128 * CTA_GROUP)

quadrants = [
    ("D00", "A0", "B0", "D[0:128, 0:128]"),
    ("D01", "A0", "B1", "D[0:128, 128:256]"),
    ("D10", "A1", "B0", "D[128:256, 0:128]"),
    ("D11", "A1", "B1", "D[128:256, 128:256]"),
]

print()
for name, a_name, b_name, output in quadrants:
    print(
        f"{name}: {a_name} @ {b_name}.T -> {output}"
    )
```

预期输出：

```text
cluster tile: (0, 0)

CTA 0
  A slice: A[0:128]
  B slice: B[0:128]
  output rows: D[0:128, :]

CTA 1
  A slice: A[128:256]
  B slice: B[128:256]
  output rows: D[128:256, :]

TMA bytes per K stage: 65536
ld2mma arrivals: 256

D00: A0 @ B0.T -> D[0:128, 0:128]
D01: A0 @ B1.T -> D[0:128, 128:256]
D10: A1 @ B0.T -> D[128:256, 0:128]
D11: A1 @ B1.T -> D[128:256, 128:256]
```

## 七、常见错误与可观察症状

| 错误 | 本质 | 症状 |
|---|---|---|
| 两个 CTA 都使用 `n_st` 做 epilogue column | `n_st` 选择 B slice，不是最终 output column | 输出列重复或缺失 |
| 单个 CTA 独立计算自己的 128 rows 和 128 columns | 只得到 `D00` 和 `D11`，没有跨 CTA 的 `D01` / `D10` | output tile 一半 quadrant 错误 |
| `tma2mma` 没有 remote view 到 CTA0 | MMA consumer 等待不到另一侧 TMA completion | CTA0 deadlock 或错误 phase |
| `tx_count` 只统计单个 CTA | 65536 被错写成 32768 | full phase 提前完成，MMA 读取不完整 B1 或 A1 |
| `mma2tma` / `mma2ld` 没有 `cta_mask=3` | 只有一侧收到 completion | 另一侧 producer 或 writeback 永久等待 |
| `ld2mma.init(128)` | 只统计一个 writeback warpgroup | 另一侧仍读 TMEM 时下一 tile 就开始覆盖 |
| MMA 或 alloc 使用 `cta_group=1` | 没有建立 CTA pair 资源语义 | 编译失败、非法访问或结果错误 |
| epilogue 直接读取 256 columns | register 用量和 copy 形状不匹配 | register pressure 升高或 copy 形状错误 |

## 八、GPU 运行边界与验证

当前 Apple M5 Pro 可以运行上面的 ownership trace，也可以静态阅读
和检查 TIRx。完整 kernel 需要支持 CTA cluster、TMEM 和
`tcgen05.mma cta_group::2` 的 NVIDIA `sm_100a` GPU，例如 B200。

### 文件三：`verify_tirx_gemm_two_cta_cluster.py`

```python
import torch
import tvm

from tirx_gemm_two_cta_cluster import hgemm_v8


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("This lesson requires a CUDA GPU.")

    device_index = torch.cuda.current_device()
    device_name = torch.cuda.get_device_name(device_index)
    capability = torch.cuda.get_device_capability(device_index)
    print(
        f"device: {device_name}, capability={capability}"
    )

    if capability[0] != 10:
        raise RuntimeError(
            "This TIRx example requires a Blackwell sm_100a GPU, "
            f"but capability is {capability}."
        )

    torch.manual_seed(0)
    target = tvm.target.Target("cuda")
    device = torch.device("cuda")

    M, N, K = 4096, 4096, 4096
    kernel = hgemm_v8(M, N, K)

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
        f"Max error vs torch reference: {max_err:.6f}"
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

在 Blackwell 机器上运行：

```bash
python trace_two_cta_cluster_ownership.py
python verify_tirx_gemm_two_cta_cluster.py
```

预期第二项输出类似：

```text
device: NVIDIA B200, capability=(10, 0)
Max error vs torch reference: 0.000000
PASS
```

## 九、自测题与答案

### 1. 为什么 stored-B 的 rows 对应 output columns？

答：kernel 计算 `D = A @ B.T`。`B` 的 stored shape 是 `N x K`，
`B[n, :]` 转置后成为 output 的第 `n` 个 column。因此 CTA0 加载
`B[0:128]`，对应 output columns `0:128`。

### 2. 为什么只由 CTA0 发起 cooperative MMA？

答：`cta_group=2` 的一条 MMA 已经覆盖两个 CTA 的 SMEM 和 TMEM。
如果两个 CTA 各发一次，就会重复计算并产生重复 completion。代码用
`if cbx == 0` 保留 cluster 中的单次 issue。

### 3. 两个 CTA 每轮 K stage 一共登记多少 TMA bytes？

答：

```text
CTA_GROUP * (BLK_M * BLK_K + BLK_N * BLK_K) * F16_SIZE
= 2 * (128 * 64 + 128 * 64) * 2
= 65536 bytes
```

### 4. 为什么 `ld2mma` 的 arrival count 是 256？

答：两个 CTA 各有 128 个 writeback threads，每个 thread 都读完自己
负责的 TMEM fragment 后执行一次 `arrive`。因此需要
`128 * CTA_GROUP = 256` 次 arrival，下一块 tile 才能复用 TMEM。

### 5. 为什么 epilogue 要拆成两段 128-column copy？

答：每个 CTA 的 TMEM accumulator 是 128 rows x 256 columns。拆成
两个 128-column chunk 后，每轮只需要一组 `128 x 128` register
fragment 和 `Dsmem`，降低 register pressure，并复用同一份
writeback / TMA store 流程。

## 十、Step 8.2：Tile 地址计算与 Epilogue 的两段 128-column 写回

## 本次讲解位置

```text
本次讲解位置
章节：chapter_gemm_advanced
小节：Step 8: Two-CTA Cluster / Tile 地址计算
知识点：m_st / n_st / n_st_epi 分别属于哪条数据路径，以及
        256-column epilogue 为什么拆成两段 128-column 写回
上次：Step 8.1：CTA0/CTA1 的 A/B slice 所有权与 256 x 256 output tile
下次：Step 8.3：CTA0 集中式 tma2mma barrier 与 65536-byte transaction
PTX：cp.async.bulk.tensor、tcgen05.ld、tcgen05.wait::ld、
     cp.async.bulk.commit_group / wait_group
```

### 1. 学习目标：不要把“输入切片地址”和“输出切片地址”混在一起

Step 8.1 已经说明，一个 cluster 共同计算 `256 x 256` output tile：

```text
CTA0 加载 A0 和 B0
CTA1 加载 A1 和 B1
```

这里容易出现一个很自然、但错误的推断：

```text
CTA0 加载 B0，所以 CTA0 只写 output columns 0:128
CTA1 加载 B1，所以 CTA1 只写 output columns 128:256
```

这不是 cooperative MMA 的结果所有权。

真正的关系是：

```text
输入 B 的所有权是斜对角切分：
    CTA0 只加载 B0
    CTA1 只加载 B1

输出 D 的所有权是行带切分：
    CTA0 写前 128 rows 的全部 256 columns
    CTA1 写后 128 rows 的全部 256 columns
```

本节要解决的核心问题就是：

```text
哪两个地址可以用 cbx，
哪一个 epilogue 地址不能用 cbx。
```

### 2. 心智模型：输入所有权是斜对角，输出所有权是行带

把一个 `256 x 256` cluster tile 看成四个 `128 x 128` quadrant：

```text
                output columns
              C0 = 0:128    C1 = 128:256
            +-------------+-------------+
R0 = 0:128  |    D00      |    D01      |
            +-------------+-------------+
R1 = 128:256|    D10      |    D11      |
            +-------------+-------------+
```

每个 quadrant 需要的 A/B 和最终持有它的 CTA 是：

| Quadrant | 使用 A | 使用 stored-B | 输出 rows | 输出 columns | TMEM 所在 CTA |
|---|---|---|---|---|---|
| `D00` | A0，CTA0 加载 | B0，CTA0 加载 | R0 | C0 | CTA0 |
| `D01` | A0，CTA0 加载 | B1，CTA1 加载 | R0 | C1 | CTA0 |
| `D10` | A1，CTA1 加载 | B0，CTA0 加载 | R1 | C0 | CTA1 |
| `D11` | A1，CTA1 加载 | B1，CTA1 加载 | R1 | C1 | CTA1 |

所以 `B` 的加载所有权决定“谁把哪份 B 放进 SMEM”，并不决定“谁最终
写哪几列 D”。CTA0 读取两份 B 的结果 `D00` 和 `D01` 都在自己的 TMEM
中，因此它必须写完整的两段 columns。

### 3. 三个地址的名字、公式和使用路径

先看源码里的三行：

```python
m_st = T.meta_var((m_idx * CTA_GROUP + cbx) * BLK_M)
n_st = T.meta_var((n_idx * CTA_GROUP + cbx) * BLK_N)
n_st_epi = T.meta_var(n_idx * 256 + no * 128)
```

其中：

```text
CTA_GROUP = 2
BLK_M = BLK_N = 128
cbx in {0, 1}
no  in {0, 1}
```

展开后得到：

```text
m_st     = m_base + cbx * 128
n_st     = n_base + cbx * 128
n_st_epi = n_base + no * 128
```

这里的 `m_base` 和 `n_base` 是 cluster tile 的左上角：

```text
m_base = m_idx * 256
n_base = n_idx * 256
```

| 地址 | 是否使用 `cbx` | 公式 | 用于哪条路径 |
|---|---:|---|---|
| `m_st` | 是 | `(m_idx * 2 + cbx) * 128` | A 的 row 起点；D 的 row 起点 |
| `n_st` | 是 | `(n_idx * 2 + cbx) * 128` | 当前 CTA 加载的 stored-B row 起点 |
| `n_st_epi` | 否 | `n_idx * 256 + no * 128` | D 的 column 起点；与 `no` 选择的 TMEM chunk 对齐 |

最重要的不变量是：

```text
m_st 同时属于 A load 和 D store。
n_st 只属于 B load。
n_st_epi 只属于 D store。
```

`n_st` 和 `n_st_epi` 在外观上相似，但生命周期完全分开。只要把
`n_st` 误用到 epilogue，CTA1 就会从 output 的中线开始写，导致前半列
缺失、后半列重复，甚至写到相邻 cluster tile。

### 4. 完整 Epilogue 路径：一次处理 128 columns，执行两次

每个 CTA 的 TMEM accumulator 在逻辑上是：

```text
128 local rows x 256 columns
```

写回时只取其中一段：

```python
tmem[:, no * 128 : (no + 1) * 128]
```

完整 epilogue 如下。代码与前面 Step 8 kernel 的 `wg_id == 0` 分支一致：

```python
elif wg_id == 0:
    wb_ps = PipelineState(1, phase=0)
    reg_f16 = T.alloc_local((128,), d_type)

    while tile_scheduler.valid():
        mma2ld.wait(
            wb_ps.stage,
            wb_ps.phase,
        )
        wb_ps.advance()
        T.ptx.tcgen05.fence.after_thread_sync()

        for no in T.unroll(2):
            reg = T.alloc_local(
                (128,),
                acc_type,
            )
            reg_wg = reg.view(
                128,
                128,
                layout=TileLayout(
                    S[
                        (128, 128)
                        : (1 @ tid_in_wg, 1)
                    ]
                ),
            )
            Tx.wg.copy_async(
                reg_wg[:],
                tmem[
                    :,
                    no * 128 : (no + 1) * 128,
                ],
            )
            T.ptx.tcgen05.wait.ld()
            Tx.cast(reg_f16[:], reg[:])
            Tx.copy(
                Dsmem[
                    warp_id * 32 + lane_id,
                    :,
                ],
                reg_f16[:],
            )
            T.ptx.fence.proxy_async("shared::cta")
            T.cuda.warpgroup_sync(10)
            if warp_id == 0:
                if lane_id == 0:
                    n_st_epi = T.meta_var(
                        n_idx * 256 + no * 128
                    )
                    Tx.copy_async(
                        D[
                            m_st : m_st + BLK_M,
                            n_st_epi : n_st_epi + 128,
                        ],
                        Dsmem[:, :],
                        dispatch="tma_auto",
                    )
                    T.ptx.cp_async.bulk.commit_group()
                    T.ptx.cp_async.bulk.wait_group(0)
            T.cuda.warpgroup_sync(10)

        ld2mma_cta0.arrive(0)
        tile_scheduler.next_tile()
```

数据路径是：

```text
TMEM 128 x 128 chunk
    --tcgen05.ld--> reg: 128 fp32 values per thread
    --cast-------> reg_f16: 128 fp16 values per thread
    --copy-------> Dsmem[128, 128]
    --TMA store--> D[m_st:m_st+128, n_st_epi:n_st_epi+128]
```

每次只保留一段 128-column fragment：

| 方案 | 每线程需要同时保持的 fp32 accumulator | 代价 |
|---|---:|---|
| 一次处理 256 columns | 256 registers | 接近或超过 CUDA 255-register 上限，容易 spill 或编译失败 |
| 每次处理 128 columns | 128 registers | 连续执行两次，使用同一份 `reg` / `Dsmem` 流程 |

这不是为了改变 output tile，而是为了让寄存器压力可控。最终仍然由
两个 CTA 各写两段，合计覆盖完整的 `256 x 256` tile。

### 5. 具体例子：`m_idx=5, n_idx=7`

cluster tile 的起点是：

```text
m_base = 5 * 256 = 1280
n_base = 7 * 256 = 1792
```

因此它覆盖：

```text
D[1280:1536, 1792:2048]
```

两个 CTA 的地址如下：

| CTA | `cbx` | `m_st` | `n_st` | D rows | `n_st_epi` 取值 |
|---|---:|---:|---:|---|---|
| CTA0 | 0 | 1280 | 1792 | `1280:1408` | 1792, 1920 |
| CTA1 | 1 | 1408 | 1920 | `1408:1536` | 1792, 1920 |

注意两个关键对照：

```text
n_st 在 CTA0/CTA1 之间不同：
    CTA0 加载 B[1792:1920]
    CTA1 加载 B[1920:2048]

n_st_epi 在 CTA0/CTA1 之间相同：
    两边都写 D[:, 1792:1920]
    两边都写 D[:, 1920:2048]
```

它们不会冲突，因为 row 起点不同：

```text
CTA0:
    D[1280:1408, 1792:1920]
    D[1280:1408, 1920:2048]

CTA1:
    D[1408:1536, 1792:1920]
    D[1408:1536, 1920:2048]
```

四块 `128 x 128` 结果正好拼成：

```text
D[1280:1536, 1792:2048]
```

### 6. 完整可运行的地址 trace

下面是只依赖 Python 标准库的完整脚本。它不依赖 CUDA，可以在 macOS
上直接执行，并检查四个 quadrant 是否恰好被覆盖一次。

#### 文件：`trace_two_cta_tile_address.py`

```python
CTA_GROUP = 2
BLK_M = 128
BLK_N = 128

m_idx = 5
n_idx = 7

m_base = m_idx * CTA_GROUP * BLK_M
n_base = n_idx * CTA_GROUP * BLK_N
covered = set()

print(f"m_idx={m_idx} n_idx={n_idx}")
print(
    f"cluster output rows: "
    f"D[{m_base}:{m_base + CTA_GROUP * BLK_M}]"
)
print(
    f"cluster output cols: "
    f"D[{n_base}:{n_base + CTA_GROUP * BLK_N}]"
)

for cbx in range(CTA_GROUP):
    m_st = (m_idx * CTA_GROUP + cbx) * BLK_M
    n_st = (n_idx * CTA_GROUP + cbx) * BLK_N

    print()
    print(f"CTA cbx={cbx}")
    print(f"  m_st={m_st} -> A rows / D rows")
    print(f"  n_st={n_st} -> stored-B rows only")

    for no in range(2):
        n_st_epi = n_idx * CTA_GROUP * BLK_N + no * BLK_N
        row_tile = (m_st // BLK_M, n_st_epi // BLK_N)
        covered.add(row_tile)
        print(
            f"  no={no} -> n_st_epi={n_st_epi} -> "
            f"D[{m_st}:{m_st + BLK_M}, "
            f"{n_st_epi}:{n_st_epi + BLK_N}]"
        )

expected = {
    (
        m_base // BLK_M + row,
        n_base // BLK_N + col,
    )
    for row in range(CTA_GROUP)
    for col in range(CTA_GROUP)
}

print()
print(f"covered 128x128 tiles: {len(covered)}")
assert covered == expected
print("PASS: all four output quadrants are covered exactly once")
```

运行：

```bash
python3 trace_two_cta_tile_address.py
```

预期输出：

```text
m_idx=5 n_idx=7
cluster output rows: D[1280:1536]
cluster output cols: D[1792:2048]

CTA cbx=0
  m_st=1280 -> A rows / D rows
  n_st=1792 -> stored-B rows only
  no=0 -> n_st_epi=1792 -> D[1280:1408, 1792:1920]
  no=1 -> n_st_epi=1920 -> D[1280:1408, 1920:2048]

CTA cbx=1
  m_st=1408 -> A rows / D rows
  n_st=1920 -> stored-B rows only
  no=0 -> n_st_epi=1792 -> D[1408:1536, 1792:1920]
  no=1 -> n_st_epi=1920 -> D[1408:1536, 1920:2048]

covered 128x128 tiles: 4
PASS: all four output quadrants are covered exactly once
```

### 7. 常见错误与可观察症状

| 错误 | 错误地址 | 可观察症状 |
|---|---|---|
| epilogue 复用 `n_st` | CTA1 从 `n_base+128` 写 column | 前半列缺失、后半列重复，第二段甚至越界到邻居 tile |
| `n_st_epi` 加 `cbx` | CTA1 只写后半段 | CTA1 的 `C0` quadrant 没有被写；如果再加第二段会越界 |
| `m_st` 使用 `m_idx * BLK_M` | cluster row 间距从 256 变成 128 | 相邻 cluster tile 的 rows 重叠，部分输出永远不被覆盖 |
| 把 `n_st` 当成 output column base | 混淆“谁加载 B”和“谁写 D” | 列所有权完全错位，结果依赖 CTA 顺序 |
| epilogue 一次读取 256 columns | 每线程同时持有 256 个 fp32 | register pressure 过高、spill 或编译失败 |
| 忘记按两段切换 TMEM column | 两段都读 `tmem[:, 0:128]` | 后半列没有被读取，输出右半部分保持旧值 |

### 8. 自测题与答案

#### 1. `m_idx=5, n_idx=7` 时，CTA1 的 `m_st`、`n_st` 和两个
`n_st_epi` 分别是多少？

答：

```text
m_st = (5 * 2 + 1) * 128 = 1408
n_st = (7 * 2 + 1) * 128 = 1920
n_st_epi(no=0) = 7 * 256 + 0 * 128 = 1792
n_st_epi(no=1) = 7 * 256 + 1 * 128 = 1920
```

#### 2. 为什么 `n_st_epi` 不能包含 `cbx`？

答：`cbx` 表示当前 CTA 加载哪一份 stored-B，不表示它最终拥有哪几列
output。cooperative MMA 已把两段 columns 的结果都放进每个 CTA 的
TMEM，所以两个 CTA 都要写完整的 `n_base:n_base+256`。它们通过不同的
`m_st` 避免写同一行。

#### 3. CTA0 没有加载 B1，为什么可以写 `D[:, n_base+128:n_base+256]`？

答：B1 由 CTA1 加载，但 `cta_group=2` 的 cooperative MMA 可以跨 CTA
读取两边 SMEM。`A0 @ B1.T` 的结果仍写入 CTA0 所属的 TMEM row band，
所以 CTA0 最终负责写 `D01`。

#### 4. 为什么 epilogue 要拆成两次 128-column copy？

答：一次处理 256 columns 会让每个 writeback thread 同时保留 256 个
fp32 accumulator，接近或超过线程寄存器上限。两段处理将 live
accumulator 降到 128 个，并复用同一份 `reg`、`reg_f16` 和 `Dsmem`。

#### 5. 如果把 `n_st` 用到 TMA store 的 column 参数上，会发生什么？

答：CTA0 会写 `0:128` 和 `128:256`，CTA1 会写 `128:256` 和
`256:384`。结果是前半列缺失、后半列被重复写，第二段还侵入相邻
cluster tile 的 columns。

## 十一、Step 8.3：CTA0 集中式 `tma2mma` barrier 与 65536-byte transaction

## 本次讲解位置

```text
本次讲解位置
章节：chapter_gemm_advanced
小节：Step 8: Two-CTA Cluster / Cross-CTA Data Handoff
知识点：CTA0 集中式 tma2mma barrier、remote_view(0) 与
        65536-byte transaction
上次：Step 8.2：Tile 地址计算与两段 128-column epilogue
下次：Step 8.4：cta_group=2 cooperative MMA 与 cta_mask=3 completion
PTX：cp.async.bulk.tensor、mbarrier.arrive.expect_tx、
     mbarrier.try_wait.parity、remote CTA barrier
```

### 1. 学习目标：让 CTA0 一次等到两个 CTA 的四笔 TMA

Step 8 的 cooperative MMA 只由 CTA0 中一个 elected thread 发起。但是这次
MMA 读取的 operand 分布在两个 CTA 的 SMEM 中：

```text
CTA0 SMEM: A0, B0
CTA1 SMEM: A1, B1
```

因此 CTA0 的 MMA consumer 不能只等待自己的 A0、B0。它必须等到：

```text
CTA0 的 A0 load 完成
CTA0 的 B0 load 完成
CTA1 的 A1 load 完成
CTA1 的 B1 load 完成
```

最直接的想法是让每个 CTA 先等待自己的两笔 TMA，再执行一次 cluster-wide
同步。但这个做法会把两侧的 TMA latency 串成两段，并且破坏 per-stage
pipeline。

Step 8.3 采用集中式完成协议：

```text
两个 CTA 的 TMA 完成事件
-> 都报告到 CTA0 的 tma2mma[stage]

CTA0 中唯一的 producer thread
-> 登记本轮需要完成的 65536 bytes

CTA0 的 MMA consumer
-> 只等待这一份 barrier
```

学习目标不是背出 `65536`，而是能回答下面三个问题：

```text
为什么 init 的数量是 1，而不是 2？
为什么一个 K stage 要登记 65536 bytes，而不是 32768？
remote_view(0) 把什么变成 remote，什么仍然保持 local？
```

### 2. 心智模型：一个 CTA0 barrier 汇总四笔 transaction

先把 `tma2mma` 想成 CTA0 中每 stage 一个“到货看板”。它同时记录两类工作：

```text
software arrivals
以及尚未完成的 TMA bytes
```

一个 phase 完成的条件是：

```text
待到达的 arrival 次数 == 0
并且
待完成的 transaction bytes == 0
```

K stage 0 的完整关系可以画成：

```text
CTA0 TMA engine
    A0: 16384 bytes --------+
    B0: 16384 bytes --------+----> CTA0 tma2mma[0]
                                 |
CTA1 TMA engine                  |
    A1: 16384 bytes --------+----> 同一份 barrier
    B1: 16384 bytes --------+

CTA0 elected producer
    1 arrival + expect 65536 ----^

CTA0 MMA consumer
    wait(tma2mma[0], phase) <----+
```

四条 `complete_tx` 路径只减少 pending bytes，不会各自产生一次 software
arrival。software arrival 只有 CTA0 中一个 elected producer 提供。

因此，`tma2mma.init(1)` 的含义是：

```text
每个 phase 只等待 1 次软件 arrival
```

不是：

```text
每个 CTA 各自 arrive 一次
每个 TMA engine arrive 一次
每个 A/B operand arrive 一次
```

TMA 的到货通过 byte count 报告，不是通过普通 thread arrival 报告。

### 3. `tma2mma`、`init` 与 `remote_view(0)` 的作用范围

完整 kernel 中与本节直接相关的初始化代码是：

```python
pool = T.SMEMPool()
tmem_addr = pool.alloc((1,), "uint32")
tma2mma = TMABar(pool, PIPE_DEPTH)
mma2tma = TCGen05Bar(pool, PIPE_DEPTH)
mma2ld = TCGen05Bar(pool, 1)
ld2mma = MBarrier(pool, 1)
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
    (BLK_M, 128),
    d_type,
    layout=D_layout,
)

tma2mma.init(1)
mma2tma.init(1)
mma2ld.init(1)
ld2mma.init(128 * CTA_GROUP)
pool.commit()

if wg_id == 0:
    if warp_id == 0:
        T.ptx.tcgen05.alloc(
            T.address_of(tmem_addr),
            n_cols=512,
            cta_group=CTA_GROUP,
        )

T.ptx.fence.proxy_async("shared::cta")
T.ptx.fence.mbarrier_init()
T.cuda.cta_sync()
```

这里每个 CTA 都会执行 `tma2mma = TMABar(pool, PIPE_DEPTH)` 和
`tma2mma.init(1)`。每个 CTA 的 shared memory 中都有一组本地 barrier
slots，但是 Step 8 只把 CTA0 的那组当作聚集点。

两边得到同一个远程引用的方式是：

```python
tma2mma_cta0 = tma2mma.remote_view(0)
```

它的语义可以展开为：

| 调用所在位置 | `remote_view(0)` 解析结果 | 结果 |
|---|---|---|
| CTA0 | CTA0 的本地 `tma2mma` | 仍然是 local barrier |
| CTA1 | cluster rank 0 的 `tma2mma` | 通过 DSMEM 访问 CTA0 的 barrier |

这里“remote”的只有 barrier 的 completion target。A1、B1 的数据并不是写到
CTA0 的 SMEM，它们仍分别搬进 CTA1 自己的：

```text
CTA1.Asmem
CTA1.Bsmem
```

需要区分三类 object：

| Object | 数据或状态所在位置 | 谁写 | 谁读 |
|---|---|---|---|
| `Asmem`, `Bsmem` | 当前 CTA 的 shared memory | 当前 CTA 的 TMA load | cooperative MMA |
| CTA0 的 `tma2mma[stage]` | CTA0 的 shared memory | 两个 CTA 的 TMA engine，以及 CTA0 的 producer | CTA0 的 MMA consumer |
| TMEM accumulator | 两个 CTA 各自的 TMEM | cooperative MMA | 各自的 writeback warpgroup |

### 4. 完整交接路径

以下是完整 kernel 中实际执行的四段路径：

```python
if wg_id == 1:
    if warp_id == 3:
        tma_ps = PipelineState(PIPE_DEPTH, phase=1)

        @T.inline
        def tma_load(k_offset):
            Tx.copy_async(
                Asmem[tma_ps.stage, :, :],
                A[
                    m_st : m_st + BLK_M,
                    k_offset : k_offset + BLK_K,
                ],
                dispatch="tma_auto",
                cta_group=CTA_GROUP,
                mbar=tma2mma_cta0.ptr_to([tma_ps.stage]),
            )
            Tx.copy_async(
                Bsmem[tma_ps.stage, :, :],
                B[
                    n_st : n_st + BLK_N,
                    k_offset : k_offset + BLK_K,
                ],
                dispatch="tma_auto",
                cta_group=CTA_GROUP,
                mbar=tma2mma_cta0.ptr_to([tma_ps.stage]),
            )

        if T.filter(lane_id, T.ptx.elect_sync()):
            while tile_scheduler.valid():
                for k in range(K_TILES):
                    mma2tma.wait(
                        tma_ps.stage,
                        tma_ps.phase,
                    )
                    tma_load(k * BLK_K)
                    if cbx == 0:
                        tma2mma_cta0.arrive(
                            tma_ps.stage,
                            CTA_GROUP
                            * (
                                BLK_M * BLK_K
                                + BLK_N * BLK_K
                            )
                            * F16_SIZE,
                        )
                    tma_ps.advance()
                tile_scheduler.next_tile()

    elif warp_id == 0:
        mma_ps = PipelineState(PIPE_DEPTH, phase=0)
        ld_ps = PipelineState(1, phase=1)

        if cbx == 0:
            if T.filter(lane_id, T.ptx.elect_sync()):
                while tile_scheduler.valid():
                    ld2mma.wait(
                        ld_ps.stage,
                        ld_ps.phase,
                    )
                    ld_ps.advance()

                    for k in range(K_TILES):
                        tma2mma.wait(
                            mma_ps.stage,
                            mma_ps.phase,
                        )
                        Tx.gemm_async(
                            tmem[:, :MMA_N],
                            Asmem[mma_ps.stage, :, :],
                            Bsmem[mma_ps.stage, :, :],
                            accum=(k != 0),
                            dispatch="tcgen05",
                            cta_group=CTA_GROUP,
                        )
                        mma2tma.arrive(
                            mma_ps.stage,
                            cta_group=CTA_GROUP,
                            cta_mask=3,
                        )
                        mma_ps.advance()

                    mma2ld.arrive(
                        0,
                        cta_group=CTA_GROUP,
                        cta_mask=3,
                    )
                    tile_scheduler.next_tile()
```

完整 kernel 位于本文件的“三、完整 Kernel”，因此不需要打开临时课程
clone。这里把每个 action 的 scope 再列一次：

| 代码 | 执行者 | 作用 |
|---|---|---|
| `tma_load(k_offset)` | 两边各自的 producer elected lane | 每个 CTA 发起自己的 A、B TMA load |
| `mbar=tma2mma_cta0.ptr_to([stage])` | 两边各自的 TMA copy | completion bytes 都报告到 CTA0 的 stage barrier |
| `if cbx == 0: arrive(stage, 65536)` | 只有 CTA0 的 producer elected lane | 提供 1 次 arrival，并登记两个 CTA 的总 bytes |
| `tma2mma.wait(stage, phase)` | 只有 CTA0 的 MMA consumer elected lane | 等完整 A0/B0/A1/B1 集合 |
| `Tx.gemm_async(..., cta_group=2)` | CTA0 的 MMA consumer | 跨 CTA 读取两侧 SMEM |

CTA1 不执行 `tma2mma.wait`，因为 CTA1 不单独发起 cooperative MMA。
CTA1 的 TMA engine 只负责搬运自己的 SMEM，并把“搬完了多少 bytes”报告给
CTA0 的 barrier。

### 5. 为什么是 1 次 arrival，不是 2 次

`tma2mma.init(1)` 和 `arrive(stage, ...)` 的第二项是两个不同的计数器。

| 项目 | `init(1)` | `arrive(stage, 65536)` |
|---|---|---|
| 作用 | 设置每个 phase 的 expected software arrivals | 执行 1 次 arrival并登记 tx bytes |
| 物理含义 | 需要多少个普通 arrival 事件 | 需要多少 TMA bytes 完成 |
| 谁提供 | CTA0 的 elected producer | 两个 CTA 的四笔 TMA transaction |
| 数量来源 | 协调者只有一个 | `2 * (A + B)` 的总字节数 |

如果写成：

```python
tma2mma.init(2)
```

但只有一个 CTA0 producer 执行 arrive，那么每个 phase 都会缺少一次
arrival。表现通常是：

```text
TMA 数据已经全部搬完
pending bytes 已经变成 0
但 pending arrivals 仍然为 1
CTA0 MMA consumer 一直等不到 phase completion
kernel hang 或 timeout
```

反过来，如果没有 `if cbx == 0`，让两个 CTA 都执行 arrive，而 barrier 仍然
初始化为 1，也会破坏协议。这里的 `init` 统计的是 software arrival 数量，
不是 CTA 数量，也不是 TMA engine 数量。

可以用一句话记住：

```text
两个 CTA 产生 bytes
一个 CTA0 协调者产生 arrival
```

### 6. 为什么每个 K stage 是 65536 bytes

当前配置是：

```python
CTA_GROUP = 2
BLK_M = 128
BLK_N = 128
BLK_K = 64
F16_SIZE = 2
```

先算一个 CTA：

```text
A bytes
= BLK_M * BLK_K * F16_SIZE
= 128 * 64 * 2
= 16384

B bytes
= BLK_N * BLK_K * F16_SIZE
= 128 * 64 * 2
= 16384

单个 CTA 的 A + B
= 16384 + 16384
= 32768
```

再算两个 CTA：

```text
cluster bytes
= CTA_GROUP * (A bytes + B bytes)
= 2 * (16384 + 16384)
= 2 * 32768
= 65536
```

源码公式可以逐项展开：

```python
CTA_GROUP * (BLK_M * BLK_K + BLK_N * BLK_K) * F16_SIZE
= 2 * (128 * 64 + 128 * 64) * 2
= 2 * (8192 + 8192) * 2
= 2 * 16384 * 2
= 65536
```

注意这里第一个 `2` 表示两个 CTA，最后一个 `2` 表示 fp16 的 2 bytes。
两者数量相同，但物理含义不同。

如果只登记 `32768`，barrier 可能在一个 CTA 的两笔 load 完成后就认为
phase 已经完成，CTA0 会过早发起 MMA，读到另一侧尚未完成的 SMEM。

### 7. 具体状态 trace：一个 stage 从 phase 0 到 phase 1

以 K stage 0 为例。barrier 初始化后：

```text
phase = 0
pending arrivals = 1
pending tx bytes = 0
```

两个 CTA 各自发出 A、B 两笔 TMA copy。仅“发出 copy”不会改变 software
arrival count，也不会立刻完成 phase。

当 CTA0 的 producer 执行：

```python
tma2mma_cta0.arrive(0, 65536)
```

状态变为：

```text
phase = 0
pending arrivals = 0
pending tx bytes = 65536
```

四笔 transaction 的完成顺序不固定。下面故意使用与发射顺序不同的顺序：

| 完成事件 | 完成 bytes | arrival 余量 | tx bytes 余量 | phase |
|---|---:|---:|---:|---:|
| 初始状态 | 0 | 1 | 0 | 0 |
| CTA0 执行 `arrive.expect_tx(65536)` | 0 | 0 | 65536 | 0 |
| CTA1 B1 完成 | 16384 | 0 | 49152 | 0 |
| CTA1 A1 完成 | 16384 | 0 | 32768 | 0 |
| CTA0 B0 完成 | 16384 | 0 | 16384 | 0 |
| CTA0 A0 完成 | 16384 | 0 | 0 | 0 -> 1 |

当 arrival 和 bytes 同时归零时：

```text
phase 0 完成
phase parity 翻转为 1
barrier 自动开始准备下一轮 arrival
CTA0 的 tma2mma.wait(0, phase=0) 返回
CTA0 可以发起 cooperative MMA
```

完成顺序交换不会改变最终结果。barrier 关心的是总 bytes 是否归零，不是
四个 transaction 按什么顺序完成。

### 8. 完整可运行的 barrier 状态模拟

下面的脚本不依赖 GPU，只模拟 `tma2mma` 的 arrival 与 tx-count 协议。
它用相反顺序完成四笔 transaction，验证 phase 仍然正确完成：

```bash
python3 - <<'PY'
from dataclasses import dataclass, field


@dataclass
class PhaseBarrier:
    expected_arrivals: int
    pending_arrivals: int = field(init=False)
    tx_pending: int = 0
    phase: int = 0
    completed_phases: int = 0

    def __post_init__(self):
        self.pending_arrivals = self.expected_arrivals

    @property
    def complete(self):
        return self.pending_arrivals == 0 and self.tx_pending == 0

    def arrive_expect_tx(self, tx_count):
        assert self.pending_arrivals > 0, "no arrival is pending"
        self.pending_arrivals -= 1
        self.tx_pending += tx_count
        self._finish_if_ready()
        print(
            f"arrive.expect_tx({tx_count}): "
            f"arrivals={self.pending_arrivals}, tx={self.tx_pending}, "
            f"phase={self.phase}, completed={self.completed_phases}"
        )

    def complete_tx(self, byte_count):
        assert self.tx_pending >= byte_count, (
            "complete-tx exceeds pending bytes"
        )
        self.tx_pending -= byte_count
        self._finish_if_ready()
        print(
            f"complete_tx({byte_count}): "
            f"arrivals={self.pending_arrivals}, tx={self.tx_pending}, "
            f"phase={self.phase}, completed={self.completed_phases}"
        )

    def _finish_if_ready(self):
        if self.complete:
            self.pending_arrivals = self.expected_arrivals
            self.phase ^= 1
            self.completed_phases += 1


CTA_GROUP = 2
BLK_M = 128
BLK_N = 128
BLK_K = 64
F16_SIZE = 2

a_bytes = BLK_M * BLK_K * F16_SIZE
b_bytes = BLK_N * BLK_K * F16_SIZE
local_bytes = a_bytes + b_bytes
cluster_bytes = CTA_GROUP * local_bytes

barrier = PhaseBarrier(expected_arrivals=1)

transactions = [
    ("CTA0 A", a_bytes),
    ("CTA0 B", b_bytes),
    ("CTA1 A", a_bytes),
    ("CTA1 B", b_bytes),
]

print(f"A per CTA: {a_bytes}")
print(f"B per CTA: {b_bytes}")
print(f"local subtotal: {local_bytes}")
print(f"cluster tx_count: {cluster_bytes}")
print()

for name, nbytes in transactions:
    print(f"issue {name}: {nbytes} bytes -> CTA0 barrier")

barrier.arrive_expect_tx(cluster_bytes)

for name, nbytes in reversed(transactions):
    print(f"finish {name}")
    barrier.complete_tx(nbytes)

assert barrier.completed_phases == 1
assert barrier.phase == 1
assert barrier.pending_arrivals == 1
assert barrier.tx_pending == 0
print("\nPASS: one phase completed after 1 arrival and 65536 bytes")
PY
```

本机静态运行得到的预期输出是：

```text
A per CTA: 16384
B per CTA: 16384
local subtotal: 32768
cluster tx_count: 65536

issue CTA0 A: 16384 bytes -> CTA0 barrier
issue CTA0 B: 16384 bytes -> CTA0 barrier
issue CTA1 A: 16384 bytes -> CTA0 barrier
issue CTA1 B: 16384 bytes -> CTA0 barrier
arrive.expect_tx(65536): arrivals=0, tx=65536, phase=0, completed=0
finish CTA1 B
complete_tx(16384): arrivals=0, tx=49152, phase=0, completed=0
finish CTA1 A
complete_tx(16384): arrivals=0, tx=32768, phase=0, completed=0
finish CTA0 B
complete_tx(16384): arrivals=0, tx=16384, phase=0, completed=0
finish CTA0 A
complete_tx(16384): arrivals=1, tx=0, phase=1, completed=1

PASS: one phase completed after 1 arrival and 65536 bytes
```

### 9. 常见错误与可观察症状

| 错误 | 协议发生了什么 | 可观察症状 |
|---|---|---|
| CTA1 的 TMA 使用本地 `tma2mma`，没有走 `remote_view(0)` | CTA0 只收到 32768 bytes，却登记了 65536 | CTA0 的 `tma2mma.wait` 永远不返回，kernel hang |
| `tma2mma.init(2)`，但只有 CTA0 执行 arrive | 每轮少一次 software arrival | bytes 已归零但 phase 不完成 |
| 两个 CTA 都执行 arrive，但 init 仍是 1 | 多出一次 arrival，可能提前推进 phase | barrier 记账错位、随机读到未完成数据或后续 phase 挂起 |
| tx_count 只写 32768 | 一个 CTA 完成后就被误判为完整 | cooperative MMA 读取另一侧尚未完成的 SMEM，结果随机错误 |
| `PIPE_DEPTH` 个 stages 共用一份 barrier | 新旧 K stage 的完成事件混在同一 phase | stage 交错时随机 hang 或读到旧 tile |
| 把 `remote_view(0)` 理解成 remote copy | 误以为 A1、B1 应写进 CTA0 SMEM | layout 和 local allocation 全错，MMA 读取错误地址 |
| barrier init 后缺少 fence 和 CTA sync | 其他线程可能观察不到初始化结果 | barrier 行为未定义、首轮随机失败 |

### 10. 自测题与答案

#### 1. 两个 CTA 都有 TMA producer，为什么 `tma2mma.init(1)` 仍然正确？

答：TMA completion 通过 `complete_tx` 减少 pending bytes，不会产生
software arrival。两个 CTA 的四笔 transaction 都由 CTA0 唯一的 elected
producer 通过一次 `arrive` 统一登记，所以 expected arrival count 是 1。

#### 2. 为什么一个 K stage 的 `tx_count` 是 65536？

答：

```text
A per CTA = 128 * 64 * 2 = 16384 bytes
B per CTA = 128 * 64 * 2 = 16384 bytes
one CTA = 32768 bytes
two CTAs = 65536 bytes
```

#### 3. `remote_view(0)` 让哪部分变成 remote？

答：它让 barrier 的 arrival 和 transaction completion 落到 cluster rank
0，也就是 CTA0 的 `tma2mma`。A1、B1 的数据仍然写进 CTA1 自己的 SMEM，
数据目的地没有变成 remote。

#### 4. 如果 CTA1 的 TMA completion 报告到本地 barrier，会发生什么？

答：CTA0 只能看见自己的 32768 bytes，无法让登记的 65536 bytes 归零。
CTA0 的 MMA consumer 会一直等待，典型症状是 kernel hang 或 timeout。

#### 5. 为什么不能直接用 `cluster_sync()` 代替 `tma2mma`？

答：`cluster_sync()` 只能说明两个 CTA 的线程到达了同步点，不能说明各自
发出的异步 TMA 已经把 A、B bytes 全部写入 SMEM。即使两侧都调用
`cluster_sync()`，TMA 仍可能尚未完成。完成协议必须等 TMA 的 bytes，
`tma2mma` 正是这个 per-stage byte-completion barrier。

## 十二、Step 8.4：`cta_group=2` cooperative MMA 与
`cta_mask=3` completion

### 本次讲解位置

```text
本次讲解位置
章节：chapter_gemm_advanced
小节：Step 8: Two-CTA Cluster / Cooperative MMA and Completion
知识点：cta_group=2 的 CTA-pair operand/TMEM scope，以及
        cta_mask=3 的 MMA completion multicast
上次：Step 8.3：CTA0 集中式 tma2mma barrier
下次：Step 8.5：ld2mma 的 256 arrivals 与跨 CTA TMEM 复用
PTX：tcgen05.mma.cta_group::2、tcgen05.commit.cta_group::2、
     mbarrier completion arrival
```

### 1. 学习目标：一条 MMA 为什么能覆盖四个象限

Step 8.3 已经保证 CTA0 能看到四个 operand slice：

```text
CTA0.SMEM: A0, B0
CTA1.SMEM: A1, B1
```

接下来要解决的问题不是“把四条 TMA 都发出去”，而是：

```text
怎样让一个 MMA dispatch 同时消费 CTA0 和 CTA1 的 SMEM
怎样让它的结果同时写进 CTA0 和 CTA1 的 TMEM
MMA 完成后，怎样通知两个 CTA 各自的 producer 和 writeback
```

Step 8.4 的核心结论是：

```text
cta_group=2
    决定 MMA 的操作范围

cta_mask=3
    决定 MMA completion 要通知哪些 CTA 的 barrier
```

两者相关，但不是同一个参数。前者决定计算范围，后者决定完成通知范围。

### 2. 心智模型：CTA pair 是两半 SMEM 和两半 TMEM

CTA pair 是同一 cluster 中 cluster rank 只在最低位不同的两个 CTA：

```text
pair member 0 -> CTA0
pair member 1 -> CTA1
```

物理上，两个 CTA 仍然各自拥有自己的 SMEM 和 TMEM：

```text
CTA0 SMEM: A0, B0
CTA1 SMEM: A1, B1

CTA0 TMEM: 128 local rows x 256 columns
CTA1 TMEM: 128 local rows x 256 columns
```

从 cooperative MMA 的视角看，它们组成两块逻辑 operand 和一块逻辑 output：

```text
A = [A0]
    [A1]        logical 256 x K

B = [B0]
    [B1]        logical 256 x K

D = [A0 @ B0.T   A0 @ B1.T]
    [A1 @ B0.T   A1 @ B1.T]   logical 256 x 256
```

四个 subproduct 不是由四条 MMA 指令分别完成。源码只发出一次：

```python
Tx.gemm_async(
    tmem[:, :MMA_N],
    Asmem[mma_ps.stage, :, :],
    Bsmem[mma_ps.stage, :, :],
    accum=(k != 0),
    dispatch="tcgen05",
    cta_group=CTA_GROUP,
)
```

其中 `CTA_GROUP=2`。硬件负责跨 CTA pair 读取对应 SMEM，并把 logical
rows 0:128 写入 CTA0 TMEM、rows 128:256 写入 CTA1 TMEM。

### 3. 为什么代码仍然写在 `if cbx == 0`

PTX 只要求 CTA pair 中的一个 thread 发出 `tcgen05.mma`；发射 CTA 可以是
偶数 CTA，也可以是奇数 CTA，但 peer CTA 必须保持 active。本书统一采用
CTA0 作为 issuing CTA：

```python
if cbx == 0:
    if T.filter(lane_id, T.ptx.elect_sync()):
        while tile_scheduler.valid():
            ...
            Tx.gemm_async(
                tmem[:, :MMA_N],
                Asmem[mma_ps.stage, :, :],
                Bsmem[mma_ps.stage, :, :],
                accum=(k != 0),
                dispatch="tcgen05",
                cta_group=CTA_GROUP,
            )
```

执行者一共收窄了两层：

| 条件 | 剩余执行者 | 原因 |
|---|---|---|
| `cbx == 0` | CTA0 的 MMA 路径 | 一次 cooperative MMA 只需 issue 一次 |
| `elect_sync()` | 该 warp 中一个 lane | 避免多条重复 MMA 和重复 commit |

`cta_group=2` 不是“让两个 CTA 各调用一次 `gemm_async`”。恰恰相反：

```text
一个 issuing thread
-> 一条 cooperative MMA
-> 覆盖 CTA pair
```

### 4. TMEM allocation 也必须使用相同的 cta_group

cooperative MMA 会访问两侧 TMEM，因此 TMEM allocation 和 deallocation
也要使用 `cta_group=2`：

```python
if wg_id == 0:
    if warp_id == 0:
        T.ptx.tcgen05.alloc(
            T.address_of(tmem_addr),
            n_cols=512,
            cta_group=CTA_GROUP,
        )
```

同一个 kernel 中，`tcgen05` 的 allocation、MMA、commit 和 deallocation
必须遵守一致的 `cta_group` 契约。不能先用 `cta_group=2` 申请 pair TMEM，
再用 `cta_group=1` 的 MMA 或 commit 访问它。

释放前还需要：

```python
T.cuda.cluster_sync()
if warp_id == 0:
    T.ptx.tcgen05.relinquish_alloc_permit(
        cta_group=CTA_GROUP
    )
    T.ptx.tcgen05.dealloc(
        tmem_addr[0],
        n_cols=512,
        cta_group=CTA_GROUP,
    )
```

`cluster_sync()` 保证两个 CTA 都已经停止访问 TMEM，避免其中一侧先退出
或先 dealloc，造成 peer 仍在执行 collective `tcgen05` 操作。

### 5. `cta_mask=3`：把 completion 通知给两侧

MMA 是异步的。`Tx.gemm_async` 返回时，Tensor Core 可能仍在更新两侧
TMEM。因此 issuing thread 紧接着发出 commit：

```python
mma2tma.arrive(
    mma_ps.stage,
    cta_group=CTA_GROUP,
    cta_mask=3,
)
```

`cta_mask=3` 的二进制是：

```text
3 = 0b11
bit 0 -> CTA0
bit 1 -> CTA1
```

它表示 completion event 要同时更新两个 CTA 中对应的
`mma2tma[stage]` barrier：

```text
MMA 完成
  -> CTA0.mma2tma[stage] 收到一次 completion arrival
  -> CTA1.mma2tma[stage] 收到一次 completion arrival
```

每个 CTA 的 `mma2tma[stage]` 都由：

```python
mma2tma.init(1)
```

初始化。这里不是 `init(2)`，因为 mask 的两个 bit 是把同一个 completion
分发到两个 barrier，不是在一个 barrier 上累加两次 arrival。

completion 的两条消费路径是：

| Barrier | 等待者 | 完成后允许什么 |
|---|---|---|
| `mma2tma[stage]` | 两个 CTA 各自的 TMA producer | 重新加载并覆盖该 SMEM stage |
| `mma2ld[0]` | 两个 CTA 各自的 writeback warpgroup | 从各自 TMEM 读取最终 accumulator |

`mma2ld` 的 commit 也使用相同的 mask：

```python
mma2ld.arrive(
    0,
    cta_group=CTA_GROUP,
    cta_mask=3,
)
```

### 6. `cta_group` 与 `cta_mask` 的区别

| 参数 | 作用对象 | 解决的问题 | 本课取值 |
|---|---|---|---|
| `cta_group` | MMA、TMEM allocation、commit 硬件作用域 | 哪些 CTA 参与同一次 cooperative 操作 | `2` |
| `cta_mask` | `tcgen05.commit` 的 mbarrier completion multicast | completion 到达哪些 CTA 的 barrier | `3` |

错误的替换方式：

```text
用 cta_mask=3 代替 cta_group=2
-> commit 可以通知两侧，但 MMA 本身仍然不是 pair-scope

只写 cta_group=2，不写 cta_mask=3
-> MMA 覆盖 pair，但另一侧的 producer 或 writeback 收不到完成通知
```

### 7. 具体执行 trace

配置：

```text
M = N = 256
K = 64
BLK_M = BLK_N = 128
BLK_K = 64
K_TILES = 1
```

只考虑 K stage 0：

| 顺序 | 动作 | 作用范围 | 结果 |
|---:|---|---|---|
| 1 | 两侧 TMA 加载 A0/B0/A1/B1 | 两个 CTA | 65536 bytes 到齐 |
| 2 | CTA0 的 `tma2mma.wait` 返回 | CTA0 consumer lane | 四个 operand slice 可读 |
| 3 | 一个 lane 发出 `gemm_async(..., cta_group=2)` | CTA pair | 一条 cooperative MMA |
| 4 | Tensor Core 读取两侧 SMEM | CTA pair | 计算四个 subproduct |
| 5 | Tensor Core 更新两侧 TMEM | CTA pair | CTA0 rows 0:128，CTA1 rows 128:256 |
| 6 | commit `cta_mask=3` | 两侧 barrier | 两个 `mma2tma[0]` 各收到一次完成 |
| 7 | K-loop 结束，commit `mma2ld` | 两侧 writeback | 两个 CTA 各自读取本侧 TMEM |

具体 quadrant 与 TMEM 落点：

| Global rows | Global cols | A slice | B slice | TMEM owner |
|---|---:|---|---|---|
| `0:128` | `0:128` | A0 | B0 | CTA0 lanes 0:128, TCol 0:128 |
| `0:128` | `128:256` | A0 | B1 | CTA0 lanes 0:128, TCol 128:256 |
| `128:256` | `0:128` | A1 | B0 | CTA1 lanes 0:128, TCol 0:128 |
| `128:256` | `128:256` | A1 | B1 | CTA1 lanes 0:128, TCol 128:256 |

这里的 CTA0/CTA1 TMEM 是两块物理 128-lane TMEM；逻辑 output 的 256 rows
由它们拼接组成。

### 8. 静态验证脚本

下面的脚本验证 quadrant 映射和 `cta_mask=3` 的 bit 含义：

```bash
python3 - <<'PY'
CTA_GROUP = 2
BLK_M = 128
BLK_N = 128
cta_mask = 3

recipients = [
    cta
    for cta in range(CTA_GROUP)
    if cta_mask & (1 << cta)
]

assert recipients == [0, 1]
print(f"cta_mask={cta_mask:02b} -> completion recipients={recipients}")

for cta in recipients:
    for mb in range(2):
        for nb in range(2):
            global_m = mb * BLK_M + cta * BLK_M
            global_n = nb * BLK_N
            print(
                f"cta={cta}, A{mb}, B{nb}: "
                f"D[{global_m}:{global_m + BLK_M}, "
                f"{global_n}:{global_n + BLK_N}]"
            )
PY
```

预期输出：

```text
cta_mask=11 -> completion recipients=[0, 1]
cta=0, A0, B0: D[0:128, 0:128]
cta=0, A0, B1: D[0:128, 128:256]
cta=1, A1, B0: D[128:256, 0:128]
cta=1, A1, B1: D[128:256, 128:256]
```

这个脚本只验证静态映射和 mask，不执行 GPU MMA。实际运行需要支持
`sm_100a` 与 `tcgen05` 的 Blackwell GPU。

### 9. 常见错误与可观察症状

| 错误 | 协议破坏点 | 可观察症状 |
|---|---|---|
| 两个 CTA 都执行 `gemm_async` | 同一 accumulator 被两条 MMA 重复更新 | 结果偏大、偶发错误或 barrier 数量翻倍 |
| 只用 `cta_group=2`，但 commit 不设 `cta_mask=3` | 另一侧收不到完成通知 | 一侧 producer 永久等待，或 writeback 不启动 |
| 把 `cta_mask` 当成 operand 选择器 | 误以为它控制 A/B slice | operand 映射完全错误，结果与 mask 没有直接关系 |
| TMEM allocation 使用 `cta_group=1` | allocation 与 MMA/commit scope 不一致 | 编译/lowering 报错，或硬件访问越界/未定义 |
| commit 放在 `elect_sync()` 外 | 32 个 lanes 都创建 commit group | 31 个空 group 提前通知，TMA 覆盖尚未读完的 SMEM |
| 一侧 CTA 提前退出 | CTA pair peer inactive | collective `tcgen05` 操作挂起或行为未定义 |

### 10. 自测题与答案

#### 1. `cta_group=2` 改变的是什么？

答：它把 MMA 和对应 `tcgen05` 操作的范围扩大到一个 CTA pair。指令只需
由 pair 中一个 thread 发出，但 operand 读取和 TMEM 更新覆盖两侧 CTA。

#### 2. `cta_mask=3` 是否表示要发起三次或两次 MMA？

答：不是。它只控制 completion multicast。MMA 仍然只发起一次，
mask 的 bit 0 和 bit 1 分别让 completion 到达 CTA0 和 CTA1 的 barrier。

#### 3. 为什么 `mma2tma` 仍然可以每侧初始化为 1？

答：一次 cooperative MMA 的每个 commit 会在每个目标 CTA 的对应 barrier
上产生一次 arrival。两侧 barrier 各自期望一次，而不是同一 barrier 期望
两次。

#### 4. 如果只由 CTA0 发起 MMA，为什么 CTA1 也需要 `mma2tma`？

答：CTA1 的 TMA producer 需要知道 CTA1 的 SMEM stage 何时已经被 pair
MMA 读完，之后才能覆盖它。`cta_mask=3` 正是把 completion 送到 CTA1
本地 barrier 的机制。

#### 5. 为什么 allocation/deallocation 也必须使用 `cta_group=2`？

答：MMA 会访问 pair 两侧的 TMEM。allocation、访问和 deallocation 必须
遵守同一个 CTA-pair 契约，并在释放前用 cluster sync 保证两侧都已经停止
访问。

## 十三、Step 8.5：`ld2mma` 的 256 arrivals 与跨 CTA
TMEM 复用

### 本次讲解位置

```text
本次讲解位置
章节：chapter_gemm_advanced
小节：Step 8: Two-CTA Cluster / TMEM Reuse Across the Pair
知识点：两个 writeback warpgroup 的 256 个 arrivals，以及
        CTA0 集中式 ld2mma barrier
上次：Step 8.4：cta_group=2 cooperative MMA 与 cta_mask=3 completion
下次：Step 9：Multi-Consumer Warp Specialization
PTX：mbarrier.init、mbarrier.arrive、mbarrier.try_wait.parity、
     tcgen05.commit、tcgen05.ld、tcgen05.wait::ld
```

### 1. 学习目标：什么时候才能覆盖下一块 tile 的 TMEM

`mma2ld` 解决的是：

```text
MMA 写完了吗？
writeback 可以开始读 TMEM 了吗？
```

`ld2mma` 解决的是相反方向：

```text
writeback 已经读完 TMEM 了吗？
下一块 tile 的 MMA 可以覆盖它了吗？
```

在同一块 persistent output tile 完成后，CTA0 的 MMA issue path 会继续处理
下一块 tile。下一块 tile 的 cooperative MMA 会再次更新两个 CTA 的 TMEM。
如果它开始得太早，就会覆盖：

```text
CTA0 尚未读完的 TMEM
CTA1 尚未读完的 TMEM
```

因此不能让 CTA0 只等自己的 writeback。必须等待两个 CTA 的 writeback 都
完成读取。

### 2. 心智模型：`ld2mma` 是两路 reader 的汇合点

每个 CTA 的 writeback warpgroup 有 128 个 threads：

```text
CTA0 WG0: 128 threads
CTA1 WG0: 128 threads
```

每个 thread 在完成自己负责的 TMEM -> register -> SMEM -> GMEM 写回后，
都向 CTA0 的 `ld2mma[0]` 报告一次 arrival：

```python
ld2mma_cta0 = ld2mma.remote_view(0)
...
ld2mma_cta0.arrive(0)
```

CTA0 的 barrier 初始化必须是：

```python
ld2mma.init(128 * CTA_GROUP)
```

代入 `CTA_GROUP=2`：

```text
128 * 2 = 256 arrivals
```

这里统计的是 thread arrivals：

```text
CTA0 贡献 128
CTA1 贡献 128
合计 256
```

不是：

```text
2 次 CTA-level 汇报
128 次单一 warpgroup 汇报
```

### 3. 完整交接代码

MMA 侧在完成整个 K-loop 后发出 `mma2ld`：

```python
for k in range(K_TILES):
    tma2mma.wait(
        mma_ps.stage,
        mma_ps.phase,
    )
    Tx.gemm_async(
        tmem[:, :MMA_N],
        Asmem[mma_ps.stage, :, :],
        Bsmem[mma_ps.stage, :, :],
        accum=(k != 0),
        dispatch="tcgen05",
        cta_group=CTA_GROUP,
    )
    mma2tma.arrive(
        mma_ps.stage,
        cta_group=CTA_GROUP,
        cta_mask=3,
    )
    mma_ps.advance()

mma2ld.arrive(
    0,
    cta_group=CTA_GROUP,
    cta_mask=3,
)
```

两个 CTA 的 writeback warpgroup 等待各自的本地 `mma2ld[0]`：

```python
while tile_scheduler.valid():
    mma2ld.wait(
        wb_ps.stage,
        wb_ps.phase,
    )
    wb_ps.advance()
    T.ptx.tcgen05.fence.after_thread_sync()

    for no in T.unroll(2):
        ...
        Tx.wg.copy_async(
            reg_wg[:],
            tmem[
                :,
                no * 128 : (no + 1) * 128,
            ],
        )
        T.ptx.tcgen05.wait.ld()
        ...

    ld2mma_cta0.arrive(0)
    tile_scheduler.next_tile()
```

每个 CTA 的 128 个 writeback threads 都执行最后一次 arrive。即使它们
读取的是各自 CTA 的 TMEM，arrival 仍然通过 `remote_view(0)` 汇聚到
CTA0 的同一份 barrier。

### 4. 为什么 barrier 初始 phase 是 1

MMA consumer 使用：

```python
ld_ps = PipelineState(1, phase=1)
...
ld2mma.wait(
    ld_ps.stage,
    ld_ps.phase,
)
ld_ps.advance()
```

`ld2mma` barrier 初始处于 phase 0。第一次 `wait(phase=1)` 直接通过，
因为还没有上一块 tile 需要回收：

```text
初始 phase = 0
第一次 wait 使用 parity = 1
-> parity 已经不等于当前 phase，立即通过
```

随后 `ld_ps.advance()` 把下一次 wait parity 改成 0，进入真正等待状态。

一次 tile 的生命周期：

| 阶段 | `ld2mma` phase | 含义 |
|---|---:|---|
| kernel 初始 | 0 | 没有旧 TMEM 需要等待 |
| 第一次 `wait(phase=1)` | 0 | 直接通过 |
| `advance()` 后 | 0 | 下一块 tile 要等本次 reader 完成 |
| 256 arrivals 全部到达 | 0 -> 1 | 两侧 writeback 都已经释放 TMEM |
| 下一轮 `wait(phase=0)` | 1 | 成功返回，可以覆盖 TMEM |

### 5. 具体 execution trace

以一个 persistent worker 连续处理 tile A 和 tile B 为例：

| 顺序 | 事件 | `ld2mma` arrivals | phase | 是否允许覆盖 TMEM |
|---:|---|---:|---:|---|
| 1 | 开始处理 tile A | 0 / 256 | 0 | 是，因为是首次使用 |
| 2 | tile A MMA 完成，写满两侧 TMEM | 0 / 256 | 0 | 否，正在读取 |
| 3 | CTA0 的 128 threads 完成写回 | 128 / 256 | 0 | 否，CTA1 仍可能读取 |
| 4 | CTA1 的 128 threads 完成写回 | 256 / 256 | 0 -> 1 | 是，两侧均已完成 |
| 5 | tile B 的 MMA 开始前执行 wait | 重置为 0 / 256 | 1 | 安全覆盖两侧 TMEM |
| 6 | tile B 的 writer 再次 arrive 256 次 | 0 -> 256 | 1 -> 0 | 开始下一轮 |

最关键的是第 3 步：

```text
CTA0 自己已经完成，但 phase 仍然没有翻转
```

这正是跨 CTA fan-in 的意义。如果 barrier 只等待 128 次，CTA0 会在 CTA1
仍读 TMEM 时启动下一块 tile，形成难以稳定复现的结果错误。

### 6. 完整可运行的 arrival 协议模拟

下面的脚本模拟两个 CTA 的 writeback 线程汇聚到 CTA0 barrier：

```bash
python3 - <<'PY'
from dataclasses import dataclass


@dataclass
class ArrivalBarrier:
    expected: int
    arrivals: int = 0
    phase: int = 0

    def arrive(self, count=1):
        self.arrivals += count
        if self.arrivals == self.expected:
            self.arrivals = 0
            self.phase ^= 1
            return True
        if self.arrivals > self.expected:
            raise AssertionError("too many arrivals")
        return False


CTA_GROUP = 2
THREADS_PER_WG = 128
barrier = ArrivalBarrier(
    expected=THREADS_PER_WG * CTA_GROUP
)

print(
    f"expected={barrier.expected}, "
    f"phase={barrier.phase}, arrivals={barrier.arrivals}"
)

cta0_complete = barrier.arrive(THREADS_PER_WG)
print(
    f"after CTA0: complete={cta0_complete}, "
    f"phase={barrier.phase}, arrivals={barrier.arrivals}"
)
assert not cta0_complete
assert barrier.phase == 0

cta1_complete = barrier.arrive(THREADS_PER_WG)
print(
    f"after CTA1: complete={cta1_complete}, "
    f"phase={barrier.phase}, arrivals={barrier.arrivals}"
)
assert cta1_complete
assert barrier.phase == 1
print("PASS: TMEM is released only after 128 + 128 arrivals")
PY
```

预期输出：

```text
expected=256, phase=0, arrivals=0
after CTA0: complete=False, phase=0, arrivals=128
after CTA1: complete=True, phase=1, arrivals=0
PASS: TMEM is released only after 128 + 128 arrivals
```

### 7. `mma2ld` 与 `ld2mma` 的对称关系

| Barrier | 生产者 | 消费者 | 保护的资源 | 典型 count |
|---|---|---|---|---:|
| `mma2ld` | Tensor Core / MMA commit | 两侧 writeback | TMEM 结果已经可读 | 1 per consumer |
| `ld2mma` | 两侧所有 writeback threads | CTA0 的下一轮 MMA | TMEM 可以被覆盖 | `128 * CTA_GROUP` |

二者不能互相替代：

```text
只等 mma2ld
-> 知道结果写完了，但不知道 reader 是否读完

只等 ld2mma
-> 知道 reader 读完旧 tile，但不知道新 MMA 是否完成
```

### 8. 常见错误与可观察症状

| 错误 | 协议发生了什么 | 可观察症状 |
|---|---|---|
| `ld2mma.init(128)` | CTA0 单独到达 128 就错误释放 | 下一块 tile 覆盖 CTA1 尚未读完的 TMEM，输出偶发错误 |
| CTA1 使用本地 `ld2mma`，不走 `remote_view(0)` | CTA0 永远只收到 128 arrivals | 第二块 tile 前永久等待，kernel hang |
| 只有一个 thread 执行 arrive | 距离 256 相差 255 | 所有 tile 在复用 TMEM 前挂起 |
| 取消 `mma2ld.wait` | writeback 在 MMA 完成前读取 TMEM | 读到旧值、部分结果或 NaN/异常数值 |
| 取消 `ld2mma` | 下一块 MMA 早于 writeback 完成 | 随机数据竞争，错误依赖时序和 tile 顺序 |
| 把 256 理解成 256 bytes 或 256 events | 错误修改 init count | barrier 提前完成或永远无法完成 |

### 9. 自测题与答案

#### 1. `ld2mma.init(128 * CTA_GROUP)` 为什么等于 256？

答：每个 CTA 的 writeback warpgroup 有 128 个线程，两个 CTA 各报告一次
thread arrival，因此 128 + 128 = 256。

#### 2. 为什么 CTA1 的 writeback 必须通知 CTA0？

答：下一块 tile 的 cooperative MMA 由 CTA0 发起，但会同时覆盖 CTA0 和
CTA1 的 TMEM。CTA0 只有确认两侧 reader 都完成后，才能安全启动下一轮。

#### 3. 为什么 CTA0 完成 128 arrivals 后 phase 还不能翻转？

答：CTA1 的 TMEM 仍然可能正在被读取。barrier 必须等到 256 arrivals，
代表两侧都已释放 TMEM。

#### 4. `mma2ld` 和 `ld2mma` 各自代表什么方向？

答：`mma2ld` 表示 MMA 向 writeback 交付可读结果；`ld2mma` 表示
writeback 把 TMEM 所有权交还给下一轮 MMA。

#### 5. `ld2mma_cta0.arrive(0)` 是普通 arrival 还是
`tcgen05.commit`？

答：它是普通 `mbarrier.arrive`。Tensor Core completion 使用
`mma2ld.arrive()`；writeback thread completion 使用普通的
`ld2mma_cta0.arrive()`。

## 十四、Step 9：Multi-Consumer Warp Specialization

### 本次讲解位置

```text
本次讲解位置
章节：chapter_gemm_advanced
小节：Step 9: Multi-Consumer Warp Specialization
知识点：两个 MMA consumers 沿 M 维扩展 cluster tile，并共享同一份
        staged B；TMEM 与 result-reuse barriers 改为按 consumer 索引
上次：Step 8.5：ld2mma 的 256 arrivals 与跨 CTA TMEM 复用
下次：chapter_flash_attention
PTX：tcgen05.mma.cta_group::2、tcgen05.commit、
     mbarrier.arrive / try_wait、cp.async.bulk.tensor、
     cp.async.bulk.commit_group / wait_group、bar.sync named barrier
```

### 1. 学习目标：让同一份 B 参与两倍计算

Step 8 的一个 cluster tile 是 `256 x 256`：

```text
CTA0 提供 A0、B0
CTA1 提供 A1、B1

一次 cooperative MMA
计算 256 x 256 output
```

每个 K stage 在两个 CTA 中一共搬运：

```text
A blocks：2
B blocks：2
合计：4 个 128 x 64 fp16 block
```

Step 9 不改变一次 cooperative MMA 的 `256 x 256` 形状，而是发起两次
cooperative MMA：

```text
consumer 0 计算 output rows m_base : m_base + 256
consumer 1 计算 output rows m_base + 256 : m_base + 512

两者都计算 output columns n_base : n_base + 256
```

因此每个 K stage 变为：

```text
A blocks：4
B blocks：2
合计：6 个 128 x 64 fp16 block

贡献 output：512 x 256
```

最重要的变化是：

```text
B 没有增加
```

同一对 CTA 中的两份 B block 先服务 consumer 0，再服务 consumer 1。
因为两个 consumers 拥有不同的 output rows，却覆盖相同的 output
columns。B row 在本题的 `D = A @ B.T` 约定中决定 output column，
所以两份 output 可以共同消费同一份 stored-B。

Step 9 的优化结果应理解为：

```text
B 的搬运量没有增加
512 x 256 output 仍然只搬运两份 B

因此每单位 output 的 staged-B traffic 减半
```

它没有让 B 在 SMEM 内被物理复制成两份，也没有让两个 consumers 各自
再发一轮 TMA。两个 consumers 的 MMA 分别读取同一份
`Bsmem[stage, :, :]`。

### 2. 心智模型：两个 consumer 是两对沿 M 方向堆叠的 A

先不要看 barrier。把 cluster tile 看成一堵 `512 x 256` 的结果墙：

```text
                 B columns
            n_base : n_base + 256
          +------------------------+
consumer0 |  256 rows from A rows 0 |
          +------------------------+
consumer1 |  256 rows from A rows 1 |
          +------------------------+
```

每个 consumer 内部仍然是 Step 8 的 Two-CTA cooperative MMA。因此一个
consumer 实际由两个 CTA 各提供一半 A rows：

| Consumer | CTA0 A slice | CTA1 A slice | 共同使用的 B | 每个 CTA 的 TMEM 区域 |
|---|---|---|---|---|
| 0 | `A[m_base:m_base+128]` | `A[m_base+128:m_base+256]` | `B[n_base:n_base+256]` | TCol `[0:256]` |
| 1 | `A[m_base+256:m_base+384]` | `A[m_base+384:m_base+512]` | `B[n_base:n_base+256]` | TCol `[256:512]` |

注意表中的 B slice 是逻辑范围。物理上仍然是：

```text
CTA0 SMEM: B[n_base:n_base+128]
CTA1 SMEM: B[n_base+128:n_base+256]
```

硬件跨 CTA pair 读取这份 B，因为 MMA 使用 `cta_group=2`。

每个 CTA 的 TMEM 仍是 `128 x 512`，现在被切成两个独立的
accumulator range：

```text
TMEM TCol [0:256]    -> consumer 0 accumulator
TMEM TCol [256:512]  -> consumer 1 accumulator
```

两个 ranges 对应不同的逻辑 output rows，而不是同一组 output 的两个
副本。不能把它们看成同一个 accumulator 的两半列。

### 3. 角色分配：3 个 warpgroup、12 个 warp 中只使用 5 个计算角色

Step 9 设置：

```python
NUM_CONSUMER = 2
WG_NUMBER = 3
```

角色表如下：

| Warpgroup | Warp | 角色 | 访问的资源 |
|---|---:|---|---|
| WG 2 | 0 | MMA consumer 0 | `Asmem[stage, 0]` + 共享 `Bsmem[stage]` -> TMEM `[0:256]` |
| WG 2 | 1 | MMA consumer 1 | `Asmem[stage, 1]` + 共享 `Bsmem[stage]` -> TMEM `[256:512]` |
| WG 2 | 2 | 空闲 | 代码中没有使用该 warp 的独立角色 |
| WG 2 | 3 | TMA producer | 每 stage 加载 2 个 A block 和 1 个 B block |
| WG 0 | 全部 4 warps | writeback consumer 0 | 读取 TMEM `[0:256]` |
| WG 1 | 全部 4 warps | writeback consumer 1 | 读取 TMEM `[256:512]` |

两个 MMA issue warps 都在 CTA0 中调用 `elect_sync()`。真正发出
`gemm_async` 的仍然只是一个 elected lane：

```text
warp 0 lane elect -> consumer 0 的一条 cooperative MMA
warp 1 lane elect -> consumer 1 的一条 cooperative MMA
```

这里两个 consumers 是两条独立的 MMA issue path，不是 64 个 lanes
同时扫描两个 accumulator range。

### 4. Layout 变化：只有 A 增加 consumer 维

三个关键 buffer 的 shape 是：

```python
Asmem = pool.alloc(
    (
        PIPE_DEPTH,
        NUM_CONSUMER,
        BLK_M,
        BLK_K,
    ),
    a_type,
    layout=A_layout,
)

Bsmem = pool.alloc(
    (
        PIPE_DEPTH,
        BLK_N,
        BLK_K,
    ),
    b_type,
    layout=B_layout,
)

Dsmem = pool.alloc(
    (
        NUM_CONSUMER,
        BLK_M,
        EPI_N,
    ),
    d_type,
    layout=D_layout,
)
```

为什么 A 要多一个轴，而 B 不需要：

| Buffer | 是否增加 consumer 轴 | 原因 |
|---|---|---|
| `Asmem` | 是 | consumer 0 和 consumer 1 需要不同的 A rows |
| `Bsmem` | 否 | 两个 consumers 读取完全相同的 B rows |
| `Dsmem` | 是 | WG0 和 WG1 可能同时处于 epilogue，不能覆盖同一块临时区 |

`Dsmem` 增加 consumer 维并不是为了让 TMEM 复制数据，而是为了让两个
writeback warpgroups 的 `TMEM -> register -> SMEM -> GMEM` 路径拥有
互不冲突的临时 buffer。

### 5. 四组 barrier 的索引语义

Step 8 中四条 barrier 的方向没有变化，但 Step 9 的索引改变了：

| Barrier | 索引 | 计数 | 保护对象 |
|---|---|---:|---|
| `tma2mma` | pipeline stage | `init(1)` | 一个 stage 的全部 A/B 已经到达 |
| `mma2tma` | pipeline stage | `init(NUM_CONSUMER)` | 两个 consumers 都已读完该 stage |
| `mma2ld` | consumer | `init(1)` per slot | consumer 的 TMEM accumulator 已可读 |
| `ld2mma` | consumer | `init(128 * CTA_GROUP)` per slot | consumer 的 TMEM 已被两侧 writer 释放 |

先看正向路径：

```text
TMA producer
  加载 A[stage, 0]、A[stage, 1]、B[stage]
  最后一次 software arrival 登记 tx_count
        |
        v
tma2mma[stage]
        |
        +--> consumer 0 wait
        +--> consumer 1 wait
```

两个 consumers 可以等待同一个 `tma2mma[stage]`。一个 barrier phase
完成之后，两个等待者都观察到这个 phase，并不会把等待者数量写成
barrier 的 expected arrival count。

再看反向 SMEM 释放路径：

```text
consumer 0 完成 MMA 并 commit
        |
        +--> mma2tma[stage] arrival

consumer 1 完成 MMA 并 commit
        |
        +--> mma2tma[stage] arrival

mma2tma[stage] 到达 2 次
        |
        v
TMA producer 才能覆盖 stage
```

因此：

```python
mma2tma.init(NUM_CONSUMER)
```

如果仍然写 `init(1)`，producer 会在其中一个 consumer 尚未读完 B 时
覆盖 stage。不同 consumer 可能按不同速度发出 MMA，所以症状通常不是
固定错误，而是偶发数值错误或随调度变化的错误。

### 6. `mma2ld` 和 `ld2mma` 为什么改为按 consumer 索引

一个 consumer 的全部 K tiles 结束后，才产生完整的 TMEM output。两个
consumers 使用不同的 TMEM column ranges，所以它们不能共用同一个
`PipelineState(1, ...)` slot：

```text
consumer 0 -> mma2ld[0] -> WG0
consumer 1 -> mma2ld[1] -> WG1
```

MMA 侧：

```python
mma2ld.arrive(
    warp_id,
    cta_group=CTA_GROUP,
    cta_mask=3,
)
```

其中：

```python
warp_id == consumer_id
```

因为 WG2 中只剩 warp 0、1 执行 consumer 路径。

Writeback 侧：

```python
mma2ld.wait(
    wg_id,
    wb_ps.phase,
)
```

其中：

```python
wg_id == consumer_id
```

WG0 读取 TMEM `[0:256]`，WG1 读取 TMEM `[256:512]`。

反向释放也一样：

```python
ld2mma_cta0.arrive(wg_id)
```

每个 CTA 的 WG0 128 个 threads 汇聚到 CTA0 的 `ld2mma[0]`；
每个 CTA 的 WG1 128 个 threads 汇聚到 CTA0 的 `ld2mma[1]`。每个
consumer slot 都需要：

```text
CTA0 的 128 threads
+ CTA1 的 128 threads
= 256 arrivals
```

所以初始化仍可写成对每个 slot 使用：

```python
ld2mma.init(128 * CTA_GROUP)
```

`MMA consumer` 和 `writeback warpgroup` 通过同一个 consumer slot 建立
一一对应关系：

```text
consumer 0 path: A[..., 0] -> TMEM[0:256]    -> WG0 -> ld2mma[0]
consumer 1 path: A[..., 1] -> TMEM[256:512] -> WG1 -> ld2mma[1]
```

### 7. Scheduler 为什么变成 `M // 512`、`N // 256`

Step 8 的 cluster tile 是 `256 x 256`：

```python
num_m_tiles = M // 256
num_n_tiles = N // 256
```

Step 9 的 cluster tile 是 `512 x 256`。其中 512 来自：

```text
NUM_CONSUMER      = 2
CTA_GROUP         = 2
BLK_M             = 128

2 * 2 * 128 = 512
```

所以：

```python
num_m_tiles = M // 512
num_n_tiles = N // 256
```

源码写成等价形式：

```python
tile_scheduler = ClusterPersistentScheduler2D(
    "ts",
    num_m_tiles=M // 256 // NUM_CONSUMER,
    num_n_tiles=N // 256,
    l2_group_size=8,
    num_clusters=SM_COUNT // CTA_GROUP,
)
```

CTA-local A row 起点是：

```python
m_st = (
    m_idx * NUM_CONSUMER * CTA_GROUP
    + cbx
) * BLK_M
```

consumer 1 的 A rows 要从 consumer 0 之后再移动：

```text
CTA_GROUP * BLK_M = 2 * 128 = 256 rows
```

所以源码中有：

```python
m_st_c1 = T.meta_var(
    m_st + CTA_GROUP * BLK_M
)
```

这不是两个 CTA 之间的距离。同一 consumer 中 CTA0 到 CTA1 的距离是
`BLK_M=128`；两个 consumers 之间的距离才是 `CTA_GROUP*BLK_M=256`。

### 8. 具体坐标 trace：`m_idx=1, n_idx=0`

设：

```text
M, N = 1024, 512
K = 128
BLK_M = BLK_N = 128
BLK_K = 64
NUM_CONSUMER = 2
CTA_GROUP = 2
m_idx = 1
n_idx = 0
```

当前 cluster tile 是：

```text
D[512:1024, 0:256]
```

CTA0 的 `m_st`：

```text
m_st = (1 * 2 * 2 + 0) * 128
     = 512
```

CTA1 的 `m_st`：

```text
m_st = (1 * 2 * 2 + 1) * 128
     = 640
```

A block 映射如下：

| CTA | Consumer | A row range | B row range |
|---:|---:|---|---|
| 0 | 0 | `A[512:640]` | `B[0:128]` |
| 1 | 0 | `A[640:768]` | `B[128:256]` |
| 0 | 1 | `A[768:896]` | `B[0:128]` |
| 1 | 1 | `A[896:1024]` | `B[128:256]` |

两次 cooperative MMA 的输出范围：

| Consumer | CTA0 本地 TMEM | CTA1 本地 TMEM | 全局 output |
|---:|---|---|---|
| 0 | TCol `[0:256]` | TCol `[0:256]` | `D[512:768, 0:256]` |
| 1 | TCol `[256:512]` | TCol `[256:512]` | `D[768:1024, 0:256]` |

最终 `D[512:1024, 0:256]` 由两个 consumers 的四块 `128 x 128`
writeback 合并：

```text
consumer 0, CTA0 -> D[512:640, 0:128]
consumer 0, CTA0 -> D[512:640, 128:256]
consumer 0, CTA1 -> D[640:768, 0:128]
consumer 0, CTA1 -> D[640:768, 128:256]

consumer 1, CTA0 -> D[768:896, 0:128]
consumer 1, CTA0 -> D[768:896, 128:256]
consumer 1, CTA1 -> D[896:1024, 0:128]
consumer 1, CTA1 -> D[896:1024, 128:256]
```

### 9. 每个 K stage 的 TMA byte count

单个 CTA、单个 stage 的 TMA traffic：

```text
consumer 0 的 A block:
128 * 64 * 2 = 16384 bytes

consumer 1 的 A block:
128 * 64 * 2 = 16384 bytes

本 CTA 的 B block:
128 * 64 * 2 = 16384 bytes

单个 CTA subtotal:
16384 * 3 = 49152 bytes
```

两个 CTA 合计：

```text
49152 * 2 = 98304 bytes
```

所以源码中的 `tx_count` 是：

```python
CTA_GROUP * (
    NUM_CONSUMER * BLK_M * BLK_K
    + BLK_N * BLK_K
) * F16_SIZE
```

代入数值：

```text
2 * (2 * 128 * 64 + 128 * 64) * 2
= 2 * (16384 + 8192) * 2
= 2 * 24576 * 2
= 98304 bytes
```

与 Step 8 的 `65536 bytes` 相比，多出的正好是一份 A block 的两倍：

```text
98304 - 65536 = 32768 bytes
32768 = 2 个 CTA * 128 * 64 * 2 bytes
```

证据是代码中的 `tx_count` 公式只新增了 `NUM_CONSUMER * BLK_M * BLK_K`，
B 项没有乘 `NUM_CONSUMER`。

### 10. Epilogue 为什么把 256 columns 分成四个 64-column 回合

每个 consumer 在自己的 256-column TMEM range 中读取结果。Step 9 设置：

```python
EPI_N = 64
```

因此：

```text
MMA_N / EPI_N = 256 / 64 = 4
```

每个 writeback warpgroup 执行四轮：

```text
i=0 -> local TCol c*256 + 0   : c*256 + 64
i=1 -> local TCol c*256 + 64  : c*256 + 128
i=2 -> local TCol c*256 + 128 : c*256 + 192
i=3 -> local TCol c*256 + 192 : c*256 + 256
```

每轮的处理链路：

```text
tcgen05.ld
TMEM -> register

cast
fp32 -> fp16

register -> Dsmem

fence.proxy.async
TMA async proxy 可以观察 Dsmem

warpgroup_sync(wg_id + 10)
确保本 WG 的 Dsmem 写入完成

TMA store
Dsmem -> GMEM

commit_group / wait_group(0)
确保 TMA store 已读完 Dsmem

warpgroup_sync(wg_id + 10)
确保下一轮可以覆盖 Dsmem
```

两个 writeback warpgroups 必须使用不同的 named barrier slot：

```text
WG0 -> barrier 10
WG1 -> barrier 11
```

这就是：

```python
T.cuda.warpgroup_sync(wg_id + 10)
```

如果两个 WGs 都使用 barrier 10，它们会错误地互相等待或提前释放同一块
SMEM，可能出现 hang 或写回数据竞争。

### 11. 完整 Kernel

下面是 Step 9 的完整文件内容。它包含所有 import、常量、helper
layout 和 `hgemm_v9` 本体，不依赖本文前面临时定义的变量。

#### 文件一：`tirx_gemm_multi_consumer.py`

```python
import tvm
from tvm.script import tirx as T
from tvm.script.tirx import tile as Tx
from tvm.tirx.layout import TileLayout, S, TLane, TCol, tid_in_wg

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

from tvm.backend.cuda.lang.pipeline import (
    TMABar,
    TCGen05Bar,
    MBarrier,
    PipelineState,
)
from tvm.backend.cuda.lang.tile_scheduler import (
    ClusterPersistentScheduler2D,
)


SM_COUNT = 148
F16_SIZE = 2


def hgemm_v9(M, N, K):
    a_type = tvm.DataType("float16")
    b_type = tvm.DataType("float16")
    d_type = tvm.DataType("float16")
    acc_type = tvm.DataType("float32")

    CTA_GROUP = 2
    NUM_CONSUMER = 2
    BLK_M, BLK_N, BLK_K = 128, 128, 64
    MMA_N = BLK_N * CTA_GROUP
    K_TILES = K // BLK_K
    PIPE_DEPTH = 4
    EPI_N = 64
    WG_NUMBER = 3

    A_layout = mma_shared_layout(
        a_type,
        SwizzleMode.SWIZZLE_128B_ATOM,
        (
            PIPE_DEPTH,
            NUM_CONSUMER,
            BLK_M,
            BLK_K,
        ),
    )
    B_layout = mma_shared_layout(
        b_type,
        SwizzleMode.SWIZZLE_128B_ATOM,
        (
            PIPE_DEPTH,
            BLK_N,
            BLK_K,
        ),
    )
    D_layout = mma_shared_layout(
        d_type,
        SwizzleMode.SWIZZLE_128B_ATOM,
        (
            NUM_CONSUMER,
            BLK_M,
            EPI_N,
        ),
    )

    @T.prim_func
    def kernel(
        A: T.Buffer((M, K), a_type),
        B: T.Buffer((N, K), b_type),
        D: T.Buffer((M, N), d_type),
    ):
        T.device_entry()
        bx = T.cta_id([SM_COUNT])
        cbx, cby = T.cta_id_in_cluster([CTA_GROUP, 1])
        wg_id = T.warpgroup_id([WG_NUMBER])
        warp_id = T.warp_id_in_wg([4])
        lane_id = T.lane_id([32])

        pool = T.SMEMPool()
        tmem_addr = pool.alloc((1,), "uint32")
        tma2mma = TMABar(pool, PIPE_DEPTH)
        mma2tma = TCGen05Bar(pool, PIPE_DEPTH)
        mma2ld = TCGen05Bar(pool, NUM_CONSUMER)
        ld2mma = MBarrier(pool, NUM_CONSUMER)
        pool.move_base_to(1024)
        Asmem = pool.alloc(
            (
                PIPE_DEPTH,
                NUM_CONSUMER,
                BLK_M,
                BLK_K,
            ),
            a_type,
            layout=A_layout,
        )
        Bsmem = pool.alloc(
            (
                PIPE_DEPTH,
                BLK_N,
                BLK_K,
            ),
            b_type,
            layout=B_layout,
        )
        Dsmem = pool.alloc(
            (
                NUM_CONSUMER,
                BLK_M,
                EPI_N,
            ),
            d_type,
            layout=D_layout,
        )

        tma2mma.init(1)
        mma2tma.init(NUM_CONSUMER)
        mma2ld.init(1)
        ld2mma.init(128 * CTA_GROUP)
        pool.commit()

        if wg_id == 0:
            if warp_id == 0:
                T.ptx.tcgen05.alloc(
                    T.address_of(tmem_addr),
                    n_cols=512,
                    cta_group=CTA_GROUP,
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

        tile_scheduler = ClusterPersistentScheduler2D(
            "ts",
            num_m_tiles=M // 256 // NUM_CONSUMER,
            num_n_tiles=N // 256,
            l2_group_size=8,
            num_clusters=SM_COUNT // CTA_GROUP,
        )
        tile_scheduler.init(bx // CTA_GROUP)

        m_idx = T.meta_var(tile_scheduler.m_idx)
        n_idx = T.meta_var(tile_scheduler.n_idx)
        m_st = T.meta_var(
            (
                m_idx * NUM_CONSUMER * CTA_GROUP
                + cbx
            )
            * BLK_M
        )
        n_st = T.meta_var(
            (
                n_idx * CTA_GROUP
                + cbx
            )
            * BLK_N
        )

        tma2mma_cta0 = tma2mma.remote_view(0)
        ld2mma_cta0 = ld2mma.remote_view(0)

        if wg_id == 2:
            if warp_id == 3:
                tma_ps = PipelineState(
                    PIPE_DEPTH,
                    phase=1,
                )

                @T.inline
                def tma_load(k_offset):
                    m_st_c1 = T.meta_var(
                        m_st + CTA_GROUP * BLK_M
                    )
                    Tx.copy_async(
                        Asmem[tma_ps.stage, 0, :, :],
                        A[
                            m_st : m_st + BLK_M,
                            k_offset : k_offset + BLK_K,
                        ],
                        dispatch="tma_auto",
                        cta_group=CTA_GROUP,
                        mbar=tma2mma_cta0.ptr_to(
                            [tma_ps.stage]
                        ),
                    )
                    Tx.copy_async(
                        Asmem[tma_ps.stage, 1, :, :],
                        A[
                            m_st_c1 : m_st_c1 + BLK_M,
                            k_offset : k_offset + BLK_K,
                        ],
                        dispatch="tma_auto",
                        cta_group=CTA_GROUP,
                        mbar=tma2mma_cta0.ptr_to(
                            [tma_ps.stage]
                        ),
                    )
                    Tx.copy_async(
                        Bsmem[tma_ps.stage, :, :],
                        B[
                            n_st : n_st + BLK_N,
                            k_offset : k_offset + BLK_K,
                        ],
                        dispatch="tma_auto",
                        cta_group=CTA_GROUP,
                        mbar=tma2mma_cta0.ptr_to(
                            [tma_ps.stage]
                        ),
                    )

                if T.filter(lane_id, T.ptx.elect_sync()):
                    while tile_scheduler.valid():
                        for k in range(K_TILES):
                            mma2tma.wait(
                                tma_ps.stage,
                                tma_ps.phase,
                            )
                            tma_load(k * BLK_K)
                            if cbx == 0:
                                tma2mma_cta0.arrive(
                                    tma_ps.stage,
                                    CTA_GROUP
                                    * (
                                        NUM_CONSUMER
                                        * BLK_M
                                        * BLK_K
                                        + BLK_N * BLK_K
                                    )
                                    * F16_SIZE,
                                )
                            tma_ps.advance()
                        tile_scheduler.next_tile()

            elif warp_id < NUM_CONSUMER:
                mma_ps = PipelineState(
                    PIPE_DEPTH,
                    phase=0,
                )
                ld_ps = PipelineState(
                    1,
                    phase=1,
                )

                if cbx == 0:
                    if T.filter(
                        lane_id,
                        T.ptx.elect_sync(),
                    ):
                        while tile_scheduler.valid():
                            ld2mma.wait(
                                warp_id,
                                ld_ps.phase,
                            )
                            ld_ps.advance()

                            for k in range(K_TILES):
                                tma2mma.wait(
                                    mma_ps.stage,
                                    mma_ps.phase,
                                )
                                Tx.gemm_async(
                                    tmem[
                                        :,
                                        warp_id
                                        * MMA_N : warp_id
                                        * MMA_N
                                        + MMA_N,
                                    ],
                                    Asmem[
                                        mma_ps.stage,
                                        warp_id,
                                        :,
                                        :,
                                    ],
                                    Bsmem[
                                        mma_ps.stage,
                                        :,
                                        :,
                                    ],
                                    accum=(k != 0),
                                    dispatch="tcgen05",
                                    cta_group=CTA_GROUP,
                                )
                                mma2tma.arrive(
                                    mma_ps.stage,
                                    cta_group=CTA_GROUP,
                                    cta_mask=3,
                                )
                                mma_ps.advance()

                            mma2ld.arrive(
                                warp_id,
                                cta_group=CTA_GROUP,
                                cta_mask=3,
                            )
                            tile_scheduler.next_tile()

        elif wg_id < NUM_CONSUMER:
            wb_ps = PipelineState(
                1,
                phase=0,
            )
            reg_f16 = T.alloc_local(
                (EPI_N,),
                d_type,
            )

            while tile_scheduler.valid():
                mma2ld.wait(
                    wg_id,
                    wb_ps.phase,
                )
                wb_ps.advance()
                T.ptx.tcgen05.fence.after_thread_sync()

                for i in T.unroll(MMA_N // EPI_N):
                    reg = T.alloc_local(
                        (EPI_N,),
                        acc_type,
                    )
                    reg_wg = reg.view(
                        128,
                        EPI_N,
                        layout=TileLayout(
                            S[
                                (128, EPI_N)
                                : (1 @ tid_in_wg, 1)
                            ]
                        ),
                    )
                    col_st = T.meta_var(
                        wg_id * MMA_N + i * EPI_N
                    )
                    col_end = T.meta_var(
                        wg_id * MMA_N
                        + i * EPI_N
                        + EPI_N
                    )
                    Tx.wg.copy_async(
                        reg_wg[:],
                        tmem[:, col_st:col_end],
                    )
                    T.ptx.tcgen05.wait.ld()
                    Tx.cast(
                        reg_f16[:],
                        reg[:],
                    )
                    Tx.copy(
                        Dsmem[
                            wg_id,
                            warp_id * 32 + lane_id,
                            :,
                        ],
                        reg_f16[:],
                    )
                    T.ptx.fence.proxy_async(
                        "shared::cta"
                    )
                    T.cuda.warpgroup_sync(wg_id + 10)
                    if warp_id == 0:
                        if lane_id == 0:
                            m_st_epi = T.meta_var(
                                (
                                    m_idx
                                    * NUM_CONSUMER
                                    * CTA_GROUP
                                    + wg_id * CTA_GROUP
                                    + cbx
                                )
                                * BLK_M
                            )
                            n_st_epi = T.meta_var(
                                n_idx * MMA_N
                                + i * EPI_N
                            )
                            Tx.copy_async(
                                D[
                                    m_st_epi : m_st_epi
                                    + BLK_M,
                                    n_st_epi : n_st_epi
                                    + EPI_N,
                                ],
                                Dsmem[wg_id, :, :],
                                dispatch="tma_auto",
                            )
                            T.ptx.cp_async.bulk.commit_group()
                            T.ptx.cp_async.bulk.wait_group(0)
                    T.cuda.warpgroup_sync(wg_id + 10)

                ld2mma_cta0.arrive(wg_id)
                tile_scheduler.next_tile()

        T.cuda.cluster_sync()
        if warp_id == 0:
            T.ptx.tcgen05.relinquish_alloc_permit(
                cta_group=CTA_GROUP
            )
            T.ptx.tcgen05.dealloc(
                tmem_addr[0],
                n_cols=512,
                cta_group=CTA_GROUP,
            )

    return kernel
```

### 12. 逐块解释完整数据路径

#### TMA producer

`warp_id == 3` 的 elected lane 每个 K stage 发三笔 TMA：

```text
Asmem[stage, 0] <- 本 CTA 的 consumer 0 A block
Asmem[stage, 1] <- 本 CTA 的 consumer 1 A block
Bsmem[stage]    <- 本 CTA 的共享 B block
```

只有 CTA0 的 elected lane 登记总 byte count：

```text
98304 bytes
```

这一步没有把 B 加载两次，这是 Step 9 的全部收益来源。

#### MMA consumers

WG2 的 warp 0、1 各自维护独立的：

```text
mma_ps
ld_ps
```

`warp_id` 同时承担三件事：

```text
选择 Asmem consumer axis
选择 TMEM column range
选择 mma2ld / ld2mma slot
```

每个 K tile：

```text
等待同一个 tma2mma[stage]
发起自己的 cooperative MMA
发一次 mma2tma[stage]
推进自己的 pipeline stage
```

`mma2tma[stage]` 只有在两个 warps 都执行 arrive 后才完成 phase。

#### Writeback

WG0、WG1 各自拥有独立 `wb_ps` 和 named barrier slot。每轮只读取
64 columns，避免 register fragment 一次膨胀到 256 columns。

`Dsmem[wg_id]` 隔离两个 writeback groups 的临时 tile。TMA store 由
WG 内 `warp_id == 0, lane_id == 0` 的线程提交。

#### Cleanup

整个 persistent loop 结束后，两个 CTA 先 `cluster_sync()`，再释放
CTA-pair TMEM。顺序不能反过来，否则 peer CTA 可能仍在使用 collective
TMEM allocation。

### 13. K stage 与 consumer slot 的交错 trace

设：

```text
PIPE_DEPTH = 4
K_TILES = 4
NUM_CONSUMER = 2
```

为了便于观察，把 TMA 用 `T0...T3`、consumer 0 MMA 用 `M0_0...M0_3`、
consumer 1 MMA 用 `M1_0...M1_3` 表示。

| 时间片 | TMA producer | Consumer 0 | Consumer 1 | `mma2tma[stage]` |
|---:|---|---|---|---|
| 0 | load T0 | 等 `tma2mma[0]` | 等 `tma2mma[0]` | 0 arrivals |
| 1 | load T1 | issue `M0_0` | issue `M1_0` | 到达 2，释放 stage 0 |
| 2 | load T2 | issue `M0_1` | issue `M1_1` | 到达 2，释放 stage 1 |
| 3 | load T3 | issue `M0_2` | issue `M1_2` | 到达 2，释放 stage 2 |
| 4 | 等 stage 3 | issue `M0_3` | issue `M1_3` | 到达 2，释放 stage 3 |
| 5 | 下一 tile | commit `mma2ld[0]` | commit `mma2ld[1]` | accumulator 可读 |
| 6 | 下一 tile | WG0 writeback | WG1 writeback | 每 slot 256 arrivals 后才能复用 |

注意 `tma2mma[stage]` 只完成一次 phase，但两个 consumers 都在等它。
barrier 的 count 统计完成条件，不统计 waiter 数。

### 14. 完整可运行的静态验证脚本

当前 Apple M5 Pro 没有 CUDA/Blackwell，不能执行 `tcgen05` kernel。
下面的脚本只验证 Step 9 的坐标、TMEM ownership、TMA bytes 和 barrier
count，这些检查不需要 GPU。

#### 文件二：`trace_multi_consumer_gemm.py`

```python
CTA_GROUP = 2
NUM_CONSUMER = 2
BLK_M = 128
BLK_N = 128
BLK_K = 64
EPI_N = 64
F16_SIZE = 2

M, N = 1024, 512
m_idx = 1
n_idx = 0

num_m_tiles = M // (NUM_CONSUMER * CTA_GROUP * BLK_M)
num_n_tiles = N // (CTA_GROUP * BLK_N)
assert num_m_tiles == 2
assert num_n_tiles == 2

print(
    "cluster tile shape: "
    f"{NUM_CONSUMER * CTA_GROUP * BLK_M}"
    f" x {CTA_GROUP * BLK_N}"
)
print(
    f"scheduler tiles: "
    f"m={num_m_tiles}, n={num_n_tiles}"
)

for consumer in range(NUM_CONSUMER):
    tmem_start = consumer * BLK_N * CTA_GROUP
    tmem_end = tmem_start + BLK_N * CTA_GROUP
    print(
        f"consumer {consumer}: "
        f"TMEM TCol [{tmem_start}:{tmem_end}]"
    )

print("CTA-local A/B and output ownership:")
for consumer in range(NUM_CONSUMER):
    for cbx in range(CTA_GROUP):
        m_st = (
            m_idx * NUM_CONSUMER * CTA_GROUP
            + consumer * CTA_GROUP
            + cbx
        ) * BLK_M
        n_b = (
            n_idx * CTA_GROUP + cbx
        ) * BLK_N
        global_out_start = m_st
        global_out_end = m_st + BLK_M
        print(
            f"consumer={consumer}, cbx={cbx}: "
            f"A[{m_st}:{m_st + BLK_M}], "
            f"B[{n_b}:{n_b + BLK_N}], "
            f"D[{global_out_start}:{global_out_end}, "
            f"{n_idx * BLK_N * CTA_GROUP}:"
            f"{n_idx * BLK_N * CTA_GROUP + BLK_N * CTA_GROUP}]"
        )

expected_rows = {
    0: [512, 640],
    1: [768, 896],
}
for consumer, rows in expected_rows.items():
    actual = [
        (
            m_idx * NUM_CONSUMER * CTA_GROUP
            + consumer * CTA_GROUP
            + cbx
        )
        * BLK_M
        for cbx in range(CTA_GROUP)
    ]
    assert actual == rows, (consumer, actual, rows)

tx_bytes = CTA_GROUP * (
    NUM_CONSUMER * BLK_M * BLK_K
    + BLK_N * BLK_K
) * F16_SIZE
assert tx_bytes == 98304

mma2tma_expected = NUM_CONSUMER
ld2mma_expected_per_consumer = 128 * CTA_GROUP
assert mma2tma_expected == 2
assert ld2mma_expected_per_consumer == 256

print(
    f"tma2mma tx_count per stage: {tx_bytes} bytes"
)
print(
    f"mma2tma arrivals per stage: {mma2tma_expected}"
)
print(
    "ld2mma arrivals per consumer slot: "
    f"{ld2mma_expected_per_consumer}"
)
print(
    "epilogue rounds per consumer: "
    f"{BLK_N * CTA_GROUP // EPI_N}"
)
print("PASS: multi-consumer ownership and counts are consistent")
```

本地运行：

```bash
python3 trace_multi_consumer_gemm.py
```

预期输出：

```text
cluster tile shape: 512 x 256
scheduler tiles: m=2, n=2
consumer 0: TMEM TCol [0:256]
consumer 1: TMEM TCol [256:512]
CTA-local A/B and output ownership:
consumer=0, cbx=0: A[512:640], B[0:128], D[512:640, 0:256]
consumer=0, cbx=1: A[640:768], B[128:256], D[640:768, 0:256]
consumer=1, cbx=0: A[768:896], B[0:128], D[768:896, 0:256]
consumer=1, cbx=1: A[896:1024], B[128:256], D[896:1024, 0:256]
tma2mma tx_count per stage: 98304 bytes
mma2tma arrivals per stage: 2
ld2mma arrivals per consumer slot: 256
epilogue rounds per consumer: 4
PASS: multi-consumer ownership and counts are consistent
```

这个脚本只证明协议和坐标一致，不会证明 GPU 上的 MMA 指令、layout
descriptor 或 barrier lowering 正确。后三者必须在 Blackwell 硬件上
运行验证。

### 15. 完整 GPU 编译与数值验证

下面文件需要在支持 `sm_100a`、TMEM 和 `tcgen05` 的 NVIDIA GPU 上
执行。

#### 文件三：`verify_tirx_gemm_multi_consumer.py`

```python
import torch
import tvm

from tirx_gemm_multi_consumer import hgemm_v9


def main():
    if not torch.cuda.is_available():
        raise RuntimeError(
            "This lesson requires a CUDA GPU."
        )

    device_index = torch.cuda.current_device()
    device_name = torch.cuda.get_device_name(
        device_index
    )
    capability = torch.cuda.get_device_capability(
        device_index
    )
    print(
        f"device: {device_name}, "
        f"capability={capability}"
    )

    if capability[0] != 10:
        raise RuntimeError(
            "This TIRx example requires a Blackwell "
            "sm_100a GPU, "
            f"but capability is {capability}."
        )

    torch.manual_seed(0)
    target = tvm.target.Target("cuda")
    device = torch.device("cuda")

    M, N, K = 4096, 4096, 4096
    kernel = hgemm_v9(M, N, K)

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

Blackwell 机器上依次运行：

```bash
python3 trace_multi_consumer_gemm.py
python3 -c "import torch; print(torch.cuda.get_device_name(), torch.cuda.get_device_capability())"
python3 verify_tirx_gemm_multi_consumer.py
```

预期硬件信息类似：

```text
NVIDIA B200 (10, 0)
```

预期数值验证结尾：

```text
device: NVIDIA B200, capability=(10, 0)
Max error vs torch reference: 0.000000
PASS
```

课程在 B200、`M=N=K=4096`、fp16、固定 clocks、1000 次计时条件下给出
约 `0.094 ms`，与同条件 cuBLAS reference 的约 `0.094 ms` 接近。
Step 8 约 `0.104 ms`，所以 Step 9 的增量收益在这组测量中约为
`10%`。这些数字不能直接外推到其他 GPU、尺寸或 clock 设置。

### 16. 常见错误与可观察症状

| 错误 | 破坏的契约 | 可观察症状 |
|---|---|---|
| `Asmem` 不增加 consumer 轴 | 两个 consumers 读取同一块 A | 第二个 consumer 的 rows 错误，或两个结果范围互相覆盖 |
| `Bsmem` 也增加 consumer 轴并加载两次 | B 复用消失 | 结果可能正确，但 TMA traffic 回升，Step 9 收益消失 |
| `mma2tma.init(1)` | producer 只等一个 consumer | stage 可能在另一个 consumer 读 B 时被覆盖，出现偶发数值错误 |
| `mma2ld` 按 stage 而不是 consumer 索引 | WG 与 TMEM range 错配 | 一个 WG 读到另一个范围内的结果，或等待永远不会到来的 phase |
| 两个 consumers 共用 `ld2mma[0]` | release count 与 ownership 混淆 | barrier 计数翻倍或提前完成，kernel hang/数据竞争 |
| `ld2mma.init(128)` | 只统计一个 CTA 的 writer | CTA0 或 CTA1 尚未读完 TMEM 时下一轮 MMA 开始 |
| scheduler 仍使用 `M // 256` | cluster tile 被误报为 256 rows | 输出 tile 重叠、越界，M 不能被 512 整除时尤其明显 |
| consumer 1 的 A 起点少加 256 rows | 两个 consumers 计算相同 A rows | `D[512:768]` 与 `D[768:1024]` 中一组结果错误 |
| `Dsmem` 不按 consumer 隔离 | 两个 WGs 同时覆盖临时 SMEM | epilogue 输出交错或损坏 |
| 两个 WGs 都使用 `warpgroup_sync(10)` | named barrier 身份冲突 | hang，或 TMA store 尚未完成就覆盖 Dsmem |
| 一次读取 256 columns register fragment | register pressure 过高 | 编译失败、spill 严重，或性能明显下降 |
| cleanup 缺少 `cluster_sync()` | 一侧提前释放 pair TMEM | collective `tcgen05` 挂起或未定义行为 |

### 17. 自测题与答案

#### 1. Step 9 为什么可以复用 B，而不能直接复用 A？

答：两个 consumers 沿 M 维扩展，计算不同 output rows，所以需要不同
A rows；它们的 output columns 相同，因此都使用同一组 stored-B。

#### 2. `Asmem`、`Bsmem`、`Dsmem` 各自是否需要 consumer 维？

答：`Asmem` 需要，因为两个 consumers 使用不同 A block；`Bsmem`
不需要，因为 B 被共享；`Dsmem` 需要，因为两个 writeback warpgroups
需要互不冲突的临时写回区。

#### 3. 为什么 `mma2tma.init(1)` 在 Step 8 可以，到 Step 9 必须改成
`mma2tma.init(NUM_CONSUMER)`？

答：Step 8 每个 stage 只有一个 consumer 使用 A/B。Step 9 有两个
consumers 共用同一份 staged B，producer 必须等到两个 consumers 都
完成读取后才覆盖 stage。

#### 4. `mma2ld` 和 `ld2mma` 按 consumer 索引后，每个 slot 分别等待
什么？

答：`mma2ld[c]` 等待 consumer `c` 的完整 TMEM accumulator 可读；
`ld2mma[c]` 等待两个 CTA 的 WG `c` 共 256 个 threads 都完成 TMEM
读取，之后 consumer `c` 才能开始下一块 tile。

#### 5. Step 9 的 cluster tile 为什么是 `512 x 256`？

答：一个 consumer 覆盖 256 rows，两个 consumers 沿 M 叠加得到 512
rows；N 方向仍由 CTA pair 的 2 个 128-column B slice 组成，所以是
256 columns。

#### 6. 每个 stage 新增的 TMA bytes 是多少，为什么 B 部分不乘 2？

答：新增两个 CTA 各一份 128 x 64 fp16 A block，共 32768 bytes，
因此每个 stage 从 65536 变为 98304 bytes。B 部分仍只从两个 CTA 各
加载一份，被两个 consumers 共享，所以不乘 `NUM_CONSUMER`。

#### 7. 为什么两个 writeback warpgroups 使用 barrier 10 和 11？

答：named barrier 要求参与线程使用同一个 slot。WG0 和 WG1 是两组
独立的 128-thread group，使用不同 slot 才能分别同步自己的 Dsmem
写入和 TMA store 完成，不会互相卷入同一个 barrier。

## 本章完成与下一章

到这里，`chapter_gemm_advanced` 的 Step 1 到 Step 9 已经完整覆盖：

```text
GEMM baseline
K-loop accumulation
spatial tiling
TMA async load
software pipeline
persistent scheduling
warp specialization
two-CTA cooperative MMA
multi-consumer B reuse
```

下一章进入：

```text
chapter_flash_attention
```

学习顺序会从 FlashAttention 的 tile 分解、online softmax、running
maximum / denominator、QK^T 与 PV 的数据复用开始，再把本章的
multi-stage pipeline 和 warp specialization 映射到 attention kernel。
