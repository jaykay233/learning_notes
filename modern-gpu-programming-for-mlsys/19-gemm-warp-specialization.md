# GEMM Advanced：Step 7 Warp Specialization 与四条 Barrier 交接

这篇笔记继续 `chapter_gemm_advanced`。Step 6 已经让固定数量的
persistent CTA 连续处理多块 output tile，也复用了 SMEM、TMEM 和
barrier，但一个 warpgroup 仍然同时控制 TMA issue、MMA issue 和
writeback。

Step 7 把这份控制权拆给三个固定角色：

```text
TMA Producer  负责 GMEM -> SMEM
MMA Consumer  负责 SMEM -> TMEM
Writeback     负责 TMEM -> RF -> SMEM -> GMEM
```

角色之间不再靠“谁先走到下一行代码”碰运气，而是通过四条 barrier
交接 SMEM stage 和 TMEM accumulator 的所有权。

## 本次讲解位置

```text
本次讲解位置
章节：chapter_gemm_advanced
小节：Step 7: Warp Specialization
知识点：TMA producer / MMA consumer / writeback 的角色拆分，
        以及 tma2mma / mma2tma / mma2ld / ld2mma 四条 barrier 交接
上次：chapter_gemm_async Step 6: Persistent Kernel + Tile Scheduler
下次：Step 8: Two-CTA Cluster
PTX：tcgen05.mma、tcgen05.commit、mbarrier、
     cp.async.bulk.tensor、cp.async.bulk.commit_group /
     wait_group、bar.sync named barrier
```

### 本章知识点清单

```text
[x] Step 7：Warp Specialization 与完整 Load / Compute / Writeback overlap
[ ] Step 8：Two-CTA Cluster（进行中；8.1 见
    `20-gemm-two-cta-cluster.md`）
[ ] Step 9：Multi-Consumer Warp Specialization
```

## 一、为什么一个 warpgroup 仍然会互相等待

Step 5 和 Step 6 已经有 double buffering、prefetch 和 persistent
scheduling，但控制路径仍类似：

```text
同一个 warpgroup
  等待 TMA
  发起 MMA
  等待 MMA
  回填下一个 stage
  ...
  读取 TMEM
  转换 dtype
  写 Dsmem
  发起 TMA store
  等待 store
```

这些操作由不同硬件单元执行，但 issue 和等待仍集中在同一组线程。
如果一个 warpgroup 正在完成 writeback，它不能同时在下一次 K-loop
中推进 TMA producer；如果所有线程都在 producer 分支里，也没有另一组
固定线程专门等待 MMA 结果。

Warp specialization 的优化目标并不是让一条指令“同时做两件事”，
而是建立三个可以独立推进的软件角色：

```text
Producer 关心：
  当前 SMEM stage 是否为空
  TMA 是否已经完成

Consumer 关心：
  当前 SMEM stage 是否已装满
  MMA 是否已经读完该 stage
  writeback 是否已经释放 TMEM

Writeback 关心：
  TMEM 中的完整 accumulator 是否已经可读
  自己是否已经读完 TMEM
  Dsmem 与 TMA store 是否已经完成
```

## 二、心智模型：三个角色和两个资源环

当前 Step 7 使用 `WG_NUMBER=2`，也就是两个 warpgroups、八个 warps。
分工如下：

| 角色 | 执行者 | 工作 | 使用的硬件路径 |
|---|---|---|---|
| TMA Producer | `wg_id == 1`，`warp_id == 3` | 加载 A/B stage | TMA engine |
| MMA Consumer | `wg_id == 1`，`warp_id == 0` | 发出并等待 MMA | `tcgen05.mma` |
| Writeback | `wg_id == 0`，全部 128 threads | TMEM -> RF -> Dsmem -> GMEM | `tcgen05.ld`、TMA store |

Producer 和 MMA consumer 只由各自 warp 中一个 elected lane 实际执行
`T.filter(lane_id, T.ptx.elect_sync())` 内的代码。Writeback 则需要
整个 warpgroup 的 128 个 threads，因为 TMEM fragment 会分散到这些
threads 的 registers。

整个 kernel 有两个资源环：

```text
SMEM ring:
  stage 0, stage 1, ... 被 TMA 填，被 MMA 消费，再被 TMA 重用

TMEM ring:
  一块 accumulator 被 MMA 填，被 writeback 读，再被下一块 tile 的
  MMA 重用
```

它们不能共用一个 barrier，因为资源对象、完成事件和消费者都不同。

## 三、四条 Barrier 的方向和所有权

Barrier 名称采用 `source2destination`：

| Barrier | 类型 | 保护对象 | `wait` 方 | `arrive` / completion 方 |
|---|---|---|---|---|
| `tma2mma[stage]` | `TMABar` | 当前 SMEM stage 已装满 | MMA consumer | TMA transaction 完成 |
| `mma2tma[stage]` | `TCGen05Bar` | 当前 SMEM stage 可复用 | TMA producer | 当前 MMA 完成 |
| `mma2ld[0]` | `TCGen05Bar` | 整块 TMEM accumulator 已就绪 | writeback | K-loop 最后一个 MMA 完成 |
| `ld2mma[0]` | `MBarrier` | TMEM 已被 writeback 读完，可复用 | MMA consumer | 128 个 writeback threads |

可以把它压缩成两条协议：

```text
SMEM:
  TMA fill complete -> tma2mma -> MMA may read
  MMA read complete -> mma2tma -> TMA may overwrite

TMEM:
  final MMA complete -> mma2ld -> writeback may read
  writeback read complete -> ld2mma -> next MMA may overwrite
```

`tma2mma` 和 `mma2tma` 是每个 SMEM stage 各一份；`mma2ld` 和
`ld2mma` 只保护一块当前 TMEM accumulator，所以各有一份。

`tma2mma` 的 arrival count 初始化为 1。Producer 中选出的线程在发起
TMA 时登记预计传输的字节数：

```text
(BLK_M * BLK_K + BLK_N * BLK_K) * 2
= (128 * 64 + 128 * 64) * 2
= 32768 bytes
```

只有 thread arrival 和 bytes transaction 都满足，这个 phase 才完成。

`mma2tma` 与 `mma2ld` 的完成来源不是普通线程，而是
`tcgen05.commit` 关联的 MMA completion。`ld2mma` 则是普通
`mbarrier`，`init(128)` 表示每块 tile 必须收到 writeback warpgroup
全部 128 个 arrivals。

## 四、完整 Kernel

下面代码与课程 Step 7 对应，只增加了不同 TVM wheel 下
`tma_utils` 的兼容 import。

### 文件一：`tirx_gemm_warp_specialization.py`

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


def hgemm_v7(M, N, K):
    a_type = tvm.DataType("float16")
    b_type = tvm.DataType("float16")
    d_type = tvm.DataType("float16")
    acc_type = tvm.DataType("float32")

    BLK_M, BLK_N, BLK_K = 128, 128, 64
    K_TILES = K // BLK_K
    PIPE_DEPTH = 2
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
        (BLK_M, BLK_N),
    )

    @T.prim_func
    def kernel(
        A: T.Buffer((M, K), a_type),
        B: T.Buffer((N, K), b_type),
        D: T.Buffer((M, N), d_type),
    ):
        T.device_entry()

        bx = T.cta_id([SM_COUNT])
        wg_id = T.warpgroup_id([WG_NUMBER])
        warp_id = T.warp_id_in_wg([4])
        lane_id = T.lane_id([32])

        # --- Allocation ---
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
            (BLK_M, BLK_N),
            d_type,
            layout=D_layout,
        )

        # --- Barrier init ---
        tma2mma.init(1)
        mma2tma.init(1)
        mma2ld.init(1)
        ld2mma.init(128)
        pool.commit()

        # --- TMEM alloc + fence ---
        if wg_id == 0:
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

        # --- Tile scheduler ---
        tile_scheduler = ClusterPersistentScheduler2D(
            "ts",
            num_m_tiles=M // BLK_M,
            num_n_tiles=N // BLK_N,
            l2_group_size=8,
            num_clusters=SM_COUNT,
        )
        tile_scheduler.init(bx)

        m_st = T.meta_var(tile_scheduler.m_idx * BLK_M)
        n_st = T.meta_var(tile_scheduler.n_idx * BLK_N)

        # =============================================
        # Warpgroup 1:
        #   warp 3 -> TMA producer
        #   warp 0 -> MMA consumer
        # =============================================
        if wg_id == 1:
            if warp_id == 3:
                # === TMA Producer ===
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
                        cta_group=1,
                        mbar=tma2mma.ptr_to([tma_ps.stage]),
                    )
                    Tx.copy_async(
                        Bsmem[tma_ps.stage, :, :],
                        B[
                            n_st : n_st + BLK_N,
                            k_offset : k_offset + BLK_K,
                        ],
                        dispatch="tma_auto",
                        cta_group=1,
                        mbar=tma2mma.ptr_to([tma_ps.stage]),
                    )

                if T.filter(lane_id, T.ptx.elect_sync()):
                    while tile_scheduler.valid():
                        for k in range(K_TILES):
                            mma2tma.wait(
                                tma_ps.stage,
                                tma_ps.phase,
                            )
                            tma_load(k * BLK_K)
                            tma2mma.arrive(
                                tma_ps.stage,
                                (
                                    BLK_M * BLK_K
                                    + BLK_N * BLK_K
                                )
                                * F16_SIZE,
                            )
                            tma_ps.advance()
                        tile_scheduler.next_tile()

            elif warp_id == 0:
                # === MMA Consumer ===
                mma_ps = PipelineState(PIPE_DEPTH, phase=0)
                ld_ps = PipelineState(1, phase=1)

                if T.filter(lane_id, T.ptx.elect_sync()):
                    while tile_scheduler.valid():
                        # Wait until the previous tile's writeback
                        # releases TMEM.
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
                                tmem[:, :BLK_N],
                                Asmem[mma_ps.stage, :, :],
                                Bsmem[mma_ps.stage, :, :],
                                accum=(k != 0),
                                dispatch="tcgen05",
                                cta_group=1,
                            )
                            mma2tma.arrive(
                                mma_ps.stage,
                                cta_group=1,
                                cta_mask=0,
                            )
                            mma_ps.advance()

                        mma2ld.arrive(
                            0,
                            cta_group=1,
                            cta_mask=0,
                        )
                        tile_scheduler.next_tile()

        # =============================================
        # Warpgroup 0: all threads perform writeback
        # =============================================
        elif wg_id == 0:
            wb_ps = PipelineState(1, phase=0)
            reg_f16 = T.alloc_local((BLK_N,), d_type)

            while tile_scheduler.valid():
                mma2ld.wait(
                    wb_ps.stage,
                    wb_ps.phase,
                )
                wb_ps.advance()
                T.ptx.tcgen05.fence.after_thread_sync()

                reg = T.alloc_local((BLK_N,), acc_type)
                reg_wg = reg.view(
                    128,
                    BLK_N,
                    layout=TileLayout(
                        S[(128, BLK_N) : (1 @ tid_in_wg, 1)]
                    ),
                )
                Tx.wg.copy_async(
                    reg_wg[:],
                    tmem[:, :BLK_N],
                )
                T.ptx.tcgen05.wait.ld()

                # All 128 writeback threads report that they are
                # done reading TMEM.
                ld2mma.arrive(0)

                Tx.cast(reg_f16[:], reg[:])
                Tx.copy(
                    Dsmem[warp_id * 32 + lane_id, :],
                    reg_f16[:],
                )
                T.ptx.fence.proxy_async("shared::cta")
                T.cuda.warpgroup_sync(10)

                if warp_id == 0:
                    if lane_id == 0:
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
                tile_scheduler.next_tile()

        # --- Cleanup ---
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

## 五、代码按角色拆解

### 1. Producer：先等 empty，再填 stage

```python
mma2tma.wait(tma_ps.stage, tma_ps.phase)
tma_load(k * BLK_K)
tma2mma.arrive(tma_ps.stage, 32768)
tma_ps.advance()
```

执行顺序是：

```text
当前 SMEM stage 可以被覆盖吗
  -> 发起 A/B TMA load
  -> 登记本次 TMA 需要完成的 32768 bytes
  -> 推进到下一个 stage
```

`mma2tma.wait` 不是等待 TMA，它等待的是 MMA 读完这个 stage。第一次
进入 stage 0、stage 1 时还没有旧 MMA，因此 producer 的初始
`phase=1` 表示“初始 empty credit 已经存在”，前两次 wait 直接通过。

从 k=2 开始，stage 0 已经被 k=0 使用过。Producer 必须等待：

```text
MMA(k0) completion -> mma2tma[stage 0]
```

才能覆盖 stage 0。同理，TMA(k3) 要等 `mma2tma[stage 1]`。

### 2. Consumer：先等 full，再发 MMA

```python
tma2mma.wait(mma_ps.stage, mma_ps.phase)
Tx.gemm_async(..., accum=(k != 0))
mma2tma.arrive(
    mma_ps.stage,
    cta_group=1,
    cta_mask=0,
)
```

`tma2mma.wait` 表示“这个 stage 的 A/B bytes 是否已经到齐”。只有
到齐后才能读取 `Asmem[stage]`、`Bsmem[stage]`。

刚刚发出的 MMA 是异步的，因此 consumer 立即通过
`mma2tma.arrive` 把这次 MMA 的完成事件关联到 barrier。硬件在 MMA
真正完成后产生 arrival。这个 completion 同时告诉 producer：

```text
这个 SMEM stage 已经读完，可以准备 k + PIPE_DEPTH 的 load
```

每次开始新 output tile 前，consumer 还要先等上一次 writeback
释放 TMEM：

```python
ld2mma.wait(ld_ps.stage, ld_ps.phase)
ld_ps.advance()
```

第一块 tile 的 TMEM 本来就是空闲的，所以 `ld_ps` 初始
`phase=1`，第一次 wait 直接通过。

### 3. Writeback：等结果就绪，读完再释放 TMEM

```python
mma2ld.wait(wb_ps.stage, wb_ps.phase)
T.ptx.tcgen05.fence.after_thread_sync()
Tx.wg.copy_async(reg_wg[:], tmem[:, :BLK_N])
T.ptx.tcgen05.wait.ld()
ld2mma.arrive(0)
```

这里有两个不同的完成条件：

| 操作 | 等待什么 |
|---|---|
| `mma2ld.wait` | 整个 K-loop 的最后一个异步 MMA 已经完成 |
| `tcgen05.wait.ld` | `tcgen05.ld` 已经把自己的结果写入 registers |

`mma2ld.wait` 是跨线程的 barrier completion，
`tcgen05.fence.after_thread_sync()` 用来把后续 `tcgen05.ld` 排在这次
completion 之后。否则线程即使观察到 barrier，也仍可能缺少正确的
`tcgen05` 执行顺序。

读 TMEM 的 128 个 threads 随后都执行 `ld2mma.arrive(0)`。因为这个
barrier 初始化为 128，只有整个 writeback warpgroup 都到达后，下一块
tile 的 MMA 才能覆盖 TMEM。

### 4. Writeback 的两次 named-barrier sync

```python
T.cuda.warpgroup_sync(10)

if warp_id == 0:
    if lane_id == 0:
        Tx.copy_async(..., Dsmem[:, :], dispatch="tma_auto")
        T.ptx.cp_async.bulk.commit_group()
        T.ptx.cp_async.bulk.wait_group(0)

T.cuda.warpgroup_sync(10)
```

它 lower 为：

```text
bar.sync 10, 128
```

这里的 `10` 是 CTA 的 named-barrier slot，不是 mbarrier phase。
第一次 sync 保证 128 个线程都已经写完 `Dsmem`；第二次 sync 保证
选出的 thread 已经等到 TMA store 完成，其他线程不会在 store 尚未
结束时复用 `Dsmem`。

这里不能把 `cta_sync()` 放进 `wg_id == 0` 分支，因为
`wg_id == 1` 的 producer / consumer 不会到达同一个 CTA 同步点。
使用整个 CTA 的 barrier 会把 kernel 直接锁死。

## 六、`PIPE_DEPTH=2` 的具体交接

### 1. Producer 的 stage / phase

```text
k0: stage 0, phase 1
k1: stage 1, phase 1
k2: stage 0, phase 0
k3: stage 1, phase 0
k4: stage 0, phase 1
k5: stage 1, phase 1
```

stage 按 ring buffer 循环，phase 只在 stage 绕回 0 时翻转。

### 2. TMA 覆盖 stage 的前置条件

| TMA load | 目标 stage | 覆盖前等待的 empty 事件 |
|---|---:|---|
| `k0` | 0 | 初始为空，producer `phase=1` 直接通过 |
| `k1` | 1 | 初始为空，producer `phase=1` 直接通过 |
| `k2` | 0 | `MMA(k0)` 完成并到达 `mma2tma[0]` |
| `k3` | 1 | `MMA(k1)` 完成并到达 `mma2tma[1]` |
| `k4` | 0 | `MMA(k2)` 完成并到达 `mma2tma[0]` |

最关键的一行是：

```text
MMA(k0) -> mma2tma[stage 0] -> TMA(k2)
```

不是 `TMA(k1)`。因为 k1 使用的是 stage 1；只有 `k0 + PIPE_DEPTH`
才回到 stage 0。

### 3. 可运行的纯 Python trace

下面的脚本不依赖 CUDA 或 TVM，可以在当前 Apple M5 Pro 上运行。

#### 文件二：`trace_warp_specialization_v7.py`

```python
from dataclasses import dataclass


@dataclass
class PipelineState:
    depth: int
    stage: int = 0
    phase: int = 0

    def advance(self):
        self.stage += 1
        if self.stage == self.depth:
            self.stage = 0
            self.phase ^= 1


def states(depth, phase, count):
    state = PipelineState(depth, phase=phase)
    result = []
    for _ in range(count):
        result.append((state.stage, state.phase))
        state.advance()
    return result


def main():
    producer = states(2, phase=1, count=6)
    consumer = states(2, phase=0, count=6)
    writeback = states(1, phase=0, count=3)

    assert [stage for stage, _ in producer] == [
        0, 1, 0, 1, 0, 1,
    ]
    assert [phase for _, phase in producer] == [
        1, 1, 0, 0, 1, 1,
    ]
    assert [stage for stage, _ in consumer] == [
        0, 1, 0, 1, 0, 1,
    ]
    assert [phase for _, phase in consumer] == [
        0, 0, 1, 1, 0, 0,
    ]
    assert [phase for _, phase in writeback] == [0, 1, 0]

    print("producer k -> (stage, phase)")
    for k, (stage, phase) in enumerate(producer):
        print(f"k{k}: s{stage}, phase={phase}")

    print()
    print("SMEM stage reuse dependency")
    for k in range(5):
        stage = k % 2
        if k < 2:
            reason = "initial empty"
        else:
            reason = (
                f"MMA(k{k - 2}) completes "
                f"mma2tma[s{stage}]"
            )
        print(
            f"TMA(k{k}) -> smem stage {stage}: "
            f"wait {reason}"
        )

    print()
    print("TMEM tile handoff")
    print(
        "MMA(tile 0) -> mma2ld phase 0 -> "
        "writeback wait 0"
    )
    print(
        "128 writeback arrivals -> ld2mma phase 0 -> "
        "next tile MMA may reuse TMEM"
    )
    print(
        "MMA(tile 1) -> mma2ld phase 1 -> "
        "writeback wait 1"
    )


if __name__ == "__main__":
    main()
```

运行：

```bash
python trace_warp_specialization_v7.py
```

预期输出：

```text
producer k -> (stage, phase)
k0: s0, phase=1
k1: s1, phase=1
k2: s0, phase=0
k3: s1, phase=0
k4: s0, phase=1
k5: s1, phase=1

SMEM stage reuse dependency
TMA(k0) -> smem stage 0: wait initial empty
TMA(k1) -> smem stage 1: wait initial empty
TMA(k2) -> smem stage 0: wait MMA(k0) completes mma2tma[s0]
TMA(k3) -> smem stage 1: wait MMA(k1) completes mma2tma[s1]
TMA(k4) -> smem stage 0: wait MMA(k2) completes mma2tma[s0]

TMEM tile handoff
MMA(tile 0) -> mma2ld phase 0 -> writeback wait 0
128 writeback arrivals -> ld2mma phase 0 -> next tile MMA may reuse TMEM
MMA(tile 1) -> mma2ld phase 1 -> writeback wait 1
```

### 4. TMEM 的 tile 级交接

TMEM 只保护两块 tile 的生命周期，而不是每个 K tile 都重新建立
barrier：

```text
第一块 output tile:
  consumer: ld2mma wait phase 1 -> 直接通过
  128 次 MMA 完成
  consumer: mma2ld arrival phase 0
  writeback: mma2ld wait phase 0 -> 通过
  writeback: 读取 TMEM
  writeback: 128 个 threads 都 ld2mma arrival

第二块 output tile:
  consumer: ld2mma wait phase 0 -> 通过
  128 次 MMA 完成
  consumer: mma2ld arrival phase 1
  writeback: mma2ld wait phase 1 -> 通过
```

`ld2mma` 初始为 128，是因为每块 tile 的 release 都来自 writeback
warpgroup 的全体 threads。如果把它错误地初始化成 1，第一个到达的
thread 就会让 barrier 完成，consumer 可能在其余 127 个 threads 还在
读 TMEM 时开始覆盖结果。

## 七、为什么 producer 和 consumer 的初始 phase 相反

把 `phase` 理解成“当前要等待的已经发生的 round”：

| 角色 | 初始 phase | 第一次 wait 的语义 | 结果 |
|---|---:|---|---|
| Producer | 1 | 等待 empty barrier 离开 phase 1 | 初始没有历史 MMA，empty credit 已存在，直接通过 |
| MMA consumer | 0 | 等待 full barrier 完成 phase 0 | 初始没有数据，必须等待 TMA load |
| Writeback | 0 | 等待 `mma2ld` 完成 phase 0 | 初始没有结果，必须等待 MMA |

如果把 producer 也初始化成 0，它第一次 `mma2tma.wait` 会等待一个
尚未发生的“旧 MMA 完成”，于是不会发第一块 TMA，consumer 也永远
等不到 `tma2mma`，整个 kernel deadlock。

如果 consumer 的 `phase` 错误地设置成 1，第一次 wait 可能直接通过，
MMA 会读取尚未完成的 SMEM stage，通常表现为随机错误结果、非法
memory access 或后续 barrier phase 全部错位。

## 八、SMEM 成本与 `PIPE_DEPTH`

每个 stage 的 A/B 数据量：

```text
A: 128 * 64 * 2 bytes = 16384 bytes
B: 128 * 64 * 2 bytes = 16384 bytes
stage total = 32768 bytes = 32 KiB
```

`PIPE_DEPTH=2` 时：

```text
Asmem + Bsmem = 2 * 32 KiB = 64 KiB
Dsmem = 128 * 128 * 2 bytes = 32 KiB
合计 = 96 KiB
```

还没有计算 barrier 和 allocation metadata。`PIPE_DEPTH=4` 时 operand
stage 变为 128 KiB，再加 Dsmem 32 KiB，约 160 KiB；深度越大，越容易
触及单个 SM 的 SMEM 容量上限。

Warp specialization 的收益也不是“depth 越大越好”。角色拆分减少了
控制路径串行，但 stage 仍然必须满足：

```text
TMA 不能覆盖 MMA 尚未读完的 stage
MMA 不能覆盖 writeback 尚未读完的 TMEM
下一块 tile 不能被调度到尚未释放的资源上
```

## 九、常见错误和可观察症状

| 错误 | 本质 | 可观察症状 |
|---|---|---|
| Producer 初始 phase 用 0 | 首次 wait 等一个不存在的 empty completion | 第一个 TMA 不发，kernel deadlock |
| Consumer 初始 phase 用 1 | 首次 full wait 被跳过 | MMA 读未完成 SMEM，结果随机错误 |
| `ld2mma.init(1)` | TMEM release 只统计一个 thread | 下一块 tile MMA 过早覆盖 TMEM，尾部元素错误 |
| `mma2ld` 后漏 fence | `tcgen05.ld` 没有跨线程 completion 顺序 | 读取 TMEM 时结果尚未稳定 |
| `ld2mma.arrive` 漏 executing | TMEM 永远不被释放 | consumer 下一块 tile 永久等待 |
| `tma2mma` bytes 多报 | phase 永远等不到 | deadlock |
| `tma2mma` bytes 少报 | phase 提前完成 | MMA 读部分未到数据 |
| WG 分支中使用 `cta_sync()` | 另一 warpgroup 不会到达 | CTA 全体 deadlock |
| 第二次 `warpgroup_sync(10)` 漏掉 | 下一轮覆盖尚未完成的 TMA store 源 | 输出 tile 片段错乱 |
| 忘记 `wait_group(0)` | store 未完成就进入下一轮 | `Dsmem` 被覆盖，尾部写入旧数据 |

排查时先从资源所有权入手，不要只看“计算表达式是否相同”：

```text
SMEM stage 的 owner 现在是谁
TMEM accumulator 的 owner 现在是谁
下一条 load / MMA 需要等哪条 barrier
barrier 的 arrival count 与实际线程数是否一致
```

## 十、GPU 运行边界与验证

当前 Apple M5 Pro 可以运行上面的 Python trace，也可以静态阅读和
检查 TIRx 代码，但不能执行 `tcgen05.mma`、TMEM、TMA 或
`bar.sync` 的 Blackwell 数据路径。完整 kernel 需要
NVIDIA `sm_100a` GPU，例如 B200。

### 文件三：`verify_tirx_gemm_warp_specialization.py`

```python
import torch
import tvm

from tirx_gemm_warp_specialization import hgemm_v7


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
    kernel = hgemm_v7(M, N, K)

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

在 Blackwell 机器上运行：

```bash
python verify_tirx_gemm_warp_specialization.py
```

预期输出类似：

```text
device: NVIDIA B200, capability=(10, 0)
Max error vs torch reference: 0.000000
PASS
```

具体 error 由随机数据和 Tensor Core 累加顺序决定，但应满足
`rtol=2e-2`、`atol=5e-2`。

## 十一、自测题与答案

### 1. 为什么 producer 初始 phase 是 1，而 consumer 初始 phase 是 0？

答：producer 第一次面临的是“初始 empty stage”，没有历史 MMA
需要等待，所以用 phase 1 表示 empty credit 已存在。consumer 第一次
面临的是“初始 empty data stage”，必须等 TMA 完成 phase 0，所以从
phase 0 开始等待。

### 2. `PIPE_DEPTH=2` 时，什么事件允许 TMA load k=2 覆盖 stage 0？

答：不是 TMA k=1，也不是普通 `cta_sync`，而是 `MMA(k0)` 完成并通过
`mma2tma[0]` 通知 producer。具体路径是：

```text
MMA(k0) -> mma2tma[stage 0] -> TMA(k2)
```

### 3. 为什么 `ld2mma` 的 arrival count 是 128，而不是 1？

答：TMEM 的读取由 writeback warpgroup 的 128 个 threads 共同完成。
只统计其中一个 thread 会让 barrier 过早完成，下一块 tile 的 MMA
可能覆盖其他 threads 尚未读完的数据。`init(128)` 要求整个
warpgroup 都完成 release。

### 4. `mma2ld.wait` 和 `tcgen05.wait.ld()` 分别等待什么？

答：`mma2ld.wait` 等到 K-loop 最后一个 MMA 完成，保证 TMEM 中已经有
完整结果；`tcgen05.wait.ld()` 等待当前线程自己的 `tcgen05.ld` 把数据
写入 registers。

### 5. `warpgroup_sync(10)` 和 mbarrier 的区别是什么？

答：`warpgroup_sync(10)` lower 为 `bar.sync 10, 128`，让执行它的
128 个 threads 在 named barrier slot 10 上停下来，用于同步 Dsmem
写入和 TMA store 完成。mbarrier 用于记录 arrival count、TMA bytes
或 `tcgen05.commit` 产生的异步完成事件。两者的用途和硬件对象不同。

## 下一知识点

`chapter_gemm_advanced` 的 Step 8.1-8.2 已经记录在
[`20-gemm-two-cta-cluster.md`](20-gemm-two-cta-cluster.md)：

```text
两个 CTA 的 A/B slice 所有权
256 x 256 output tile 与 128 x 256 per-CTA TMEM
m_st / n_st / n_st_epi 的三条地址生命周期
epilogue 为什么拆成两段 128-column 写回
```

下一步进入 Step 8.3：CTA0 集中式 `tma2mma` barrier 与
65536-byte K-stage transaction。
