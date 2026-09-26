# GEMM Advanced Step 8.1：Two-CTA Cluster 的 Tile 所有权与数据复用

这篇笔记进入 `chapter_gemm_advanced` 的 Step 8。Step 7 已经在一个 CTA
内部把 TMA producer、MMA consumer 和 writeback 拆成三个角色；Step 8
把协作范围继续扩大到一个包含两个 CTA 的 cluster。

本文只讲 Step 8 的第一个小知识点：

```text
两个 CTA 为什么只各自加载一半 A/B，
却能共同计算一个 256 x 256 的 output tile。
```

## 本次讲解位置

```text
本次讲解位置
章节：chapter_gemm_advanced
小节：Step 8: Two-CTA Cluster
知识点：CTA0/CTA1 的 A/B slice 所有权，以及 cooperative MMA 产生的
        256 x 256 output tile 与 TMEM 切分
上次：Step 7: Warp Specialization 与四条 barrier 交接
下次：Step 8 的 tile 地址计算：cta_id_in_cluster、m_st / n_st 与
      256-column epilogue 的两段写回
PTX：tcgen05.mma cta_group::2、cp.async.bulk.tensor、
     mbarrier remote arrive、tcgen05.commit cta_mask
```

### Step 8 知识点清单

```text
[x] 8.1 Two-CTA cluster 的 A/B 所有权与 256 x 256 output tile
[ ] 8.2 Tile 地址计算与 epilogue 的两段 128-column 写回
[ ] 8.3 CTA0 集中式 tma2mma barrier 与 65536-byte transaction
[ ] 8.4 cta_group=2 cooperative MMA 与 cta_mask=3 completion
[ ] 8.5 ld2mma 的 256 arrivals 与跨 CTA TMEM 复用
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

## 下一知识点

下一步只讲 Step 8 的地址计算：

```text
m_idx / n_idx 如何表示 256 x 256 cluster tile
cbx 如何选择 A/B slice
m_st / n_st / n_st_epi 分别用在哪个数据路径
为什么 epilogue 不能复用 n_st
```
