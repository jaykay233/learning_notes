# GEMM Async：Step 4 TMA Async Load

这篇笔记进入 `chapter_gemm_async`。本章继续沿用前面的
`hgemm_v1 -> v2 -> v3` 主线，先只把 A/B 的 GMEM -> SMEM 搬运从
CTA 线程协作的 `Tx.cta.copy` 换成 TMA。

本节只讲 Step 4。它让 TMA 负责搬运，并让 `mbarrier` 负责报告“整块数据
什么时候真正到达 SMEM”，但尚未加入双缓冲，也尚未让 Load 和 MMA 重叠。

## 本次讲解位置

```text
本次讲解位置
章节：chapter_gemm_async
小节：Step 4: TMA Async Load
知识点：用单线程 TMA + mbarrier 替换 CTA 协作 copy
上次：hgemm_v3 的 Multi-CTA spatial tiling 与 cta_sync 同步范围
下次：Step 5: Software Pipeline（PIPE_DEPTH=2）
PTX：cp.async.bulk.tensor、mbarrier.arrive.expect_tx、
     mbarrier.try_wait.parity、cp.async.bulk.commit_group、
     cp.async.bulk.wait_group、fence.proxy.async
```

### 本章知识点清单

```text
[x] Step 4：TMA Async Load
[ ] Step 5：Software Pipeline（PIPE_DEPTH=2）
[ ] Step 6：Persistent Kernel + Tile Scheduler
[ ] Step 7：Warp Specialization 与完整 Load / Compute overlap
```

## 一、这一步解决什么问题

`hgemm_v3` 的每次 K-loop Load 都由 CTA 的 128 个线程共同完成：

```python
Tx.cta.copy(
    Asmem[:, :],
    A[m_st : m_st + BLK_M, k_st : k_st + BLK_K],
)
Tx.cta.copy(
    Bsmem[:, :],
    B[n_st : n_st + BLK_N, k_st : k_st + BLK_K],
)
T.cuda.cta_sync()
```

这里的执行逻辑是：

```text
128 个线程都参与地址计算
128 个线程共同发出 load / store 指令
cta_sync 等到所有线程都到这里
MMA 才能读取 SMEM
```

`cta_sync` 能保证“线程都执行到了这里”，但它不知道 TMA 这种异步硬件
搬运是否已经完成。Step 4 因此换成：

```text
一个 thread 发起整块 TMA 搬运
TMA engine 自己生成 tile 内地址并搬运数据
硬件用 complete_tx 报告已经搬完多少 byte
mbarrier 在 arrival 和 transaction byte 都归零时完成当前 phase
consumer 用 try_wait(phase) 等到数据真的可用
```

### 心智模型

把一次 A tile 搬运想成快递：

```text
thread 0       = 下单的人
descriptor     = 地址与分块说明
TMA engine     = 真正的运输车队
SMEM           = 收件仓库
mbarrier       = 带“预计到货字节数”的签收计数器
try_wait       = 收件方等待本轮全部货物签收
```

“一个 thread 发起”不等于“一个 thread 搬运”。线程只负责提交硬件任务，
每个 byte 仍由 TMA engine 搬运，所有 CTA 线程都可以做别的工作。

不过 Step 4 紧接着就执行 `try_wait`，所以此时 Load 与 MMA 还没有真正
重叠。它先建立正确的异步搬运和完成协议，Step 5 才增加第二个 SMEM stage
来支持 prefetch。

## 二、Step 3 与 Step 4 的执行结构

| 项目 | Step 3 | Step 4 |
|---|---|---|
| Scope | 一个 warpgroup | 不变 |
| Layout | 相同的 SMEM / TMEM / register tiles | 不变 |
| GMEM -> SMEM | 128 threads 执行 `Tx.cta.copy` | 1 thread 执行 `Tx.copy_async(..., dispatch="tma_auto")` |
| Load 完成判断 | Load 后 `cta_sync` | `mbarrier.arrive.expect_tx` + `try_wait` |
| MMA consume | 读取同步 copy 后的 SMEM | 读取 TMA 完成后的 SMEM |
| Load / Compute overlap | 无 | 无，仍然立即 wait |
| Store | registers -> GMEM | registers -> SMEM -> TMA -> GMEM |

真正发生变化的是 Dispatch：

```text
GMEM -> SMEM 的搬运动作
从 thread-driven copy
变成 TMA engine dispatch
```

### 最关键的代码差异

Before：

```python
Tx.cta.copy(
    Asmem[:, :],
    A[m_st : m_st + BLK_M, i * BLK_K : (i + 1) * BLK_K],
)
Tx.cta.copy(
    Bsmem[:, :],
    B[n_st : n_st + BLK_N, i * BLK_K : (i + 1) * BLK_K],
)
T.cuda.cta_sync()
```

After：

```python
tid = warp_id * 32 + lane_id

if tid == 0:
    Tx.copy_async(
        Asmem[:, :],
        A[m_st : m_st + BLK_M, k_st : k_st + BLK_K],
        dispatch="tma_auto",
        cta_group=1,
        mbar=tma_bar.ptr_to([0]),
    )
    Tx.copy_async(
        Bsmem[:, :],
        B[n_st : n_st + BLK_N, k_st : k_st + BLK_K],
        dispatch="tma_auto",
        cta_group=1,
        mbar=tma_bar.ptr_to([0]),
    )
    T.ptx.mbarrier.arrive.expect_tx(
        tma_bar.ptr_to([0]),
        (BLK_M * BLK_K + BLK_N * BLK_K) * 2,
    )

T.ptx.mbarrier.try_wait(
    tma_bar.ptr_to([0]),
    phase_tma,
)
```

在本节完整代码中，相同的参数会收进 `tma_config`，避免在两个
`copy_async` 调用里重复书写。

## 三、同步对象分别保证什么

### 1. `cta_sync`

保证的是 CTA 内线程执行汇合：

```text
所有参与线程已经到达这个 barrier
```

它不会等待 TMA engine 的异步 transaction。即使所有线程都到了
`cta_sync`，最后一块 TMA 数据仍可能正在路上。

### 2. `mbarrier.arrive.expect_tx`

这一步同时做两件事：

```text
arrive：
  当前 thread 向 mbarrier 报告一次 arrival

expect_tx：
  声明当前 phase 还预期接收指定数量的 transaction bytes
```

当 mbarrier 被初始化为：

```python
T.ptx.mbarrier.init(tma_bar.ptr_to([0]), 1)
```

表示当前 phase 只需要一个 thread arrival。当前代码只由 `tid == 0`
调用 `arrive.expect_tx`，所以 arrival 条件正好匹配。

### 3. `complete_tx`

TMA engine 每完成一部分 transaction bytes，就通过硬件侧
`complete_tx` 减少 pending byte count。它不是一条需要程序员手写的
软件函数，而是异步代理的完成事件。

mbarrier 当前 phase 完成需要同时满足：

```text
pending arrivals == 0
AND
pending transaction bytes == 0
```

### 4. `mbarrier.try_wait`

`try_wait(tma_bar, phase_tma)` 检查当前 phase 是否完成。完成以后，
后续 MMA 读取 SMEM 才满足数据可见性契约。

同一个 mbarrier 每一轮完成后会翻转 phase：

```text
round 0: phase 0
round 1: phase 1
round 2: phase 0
...
```

所以 Step 4 每个 K iteration 都要：

```python
phase_tma ^= 1
```

如果忘记翻转，下一轮可能错误地接受上一轮已经完成的旧 phase，
也可能一直等一个永远不会完成的 phase。

## 四、完整可运行代码

课程 Step 4 使用完整问题尺寸 `M=N=K=4096`。下面给出两个文件：

```text
tirx_gemm_async.py
verify_tirx_gemm_async.py
```

放在同一个目录中即可。

### 文件一：`tirx_gemm_async.py`

```python
import tvm
from tvm.script import tirx as T
from tvm.script.tirx import tile as Tx
from tvm.tirx.layout import TileLayout, S, TLane, TCol, tid_in_wg
from tvm.backend.cuda.tile_primitive.tma_utils import (
    mma_shared_layout,
    SwizzleMode,
)


def hgemm_v4(M, N, K):
    a_type = tvm.DataType("float16")
    b_type = tvm.DataType("float16")
    d_type = tvm.DataType("float16")
    acc_type = tvm.DataType("float32")

    BLK_M, BLK_N, BLK_K = 128, 128, 64
    K_TILES = K // BLK_K
    F16_SIZE = 2

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

        pool = T.SMEMPool()
        tmem_addr = pool.alloc((1,), "uint32")
        tma_bar = pool.alloc((1,), "uint64", align=8)
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
        Dsmem = pool.alloc(
            (BLK_M, BLK_N),
            d_type,
            layout=D_layout,
        )
        pool.commit()

        if warp_id == 0 and lane_id == 0:
            T.ptx.mbarrier.init(mma_bar.ptr_to([0]), 1)
            T.ptx.mbarrier.init(tma_bar.ptr_to([0]), 1)

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
        def tma_load(k_st):
            tma_config = T.meta_var({
                "dispatch": "tma_auto",
                "cta_group": 1,
                "mbar": tma_bar.ptr_to([0]),
            })
            Tx.copy_async(
                Asmem[:, :],
                A[
                    m_st : m_st + BLK_M,
                    k_st : k_st + BLK_K,
                ],
                **tma_config,
            )
            Tx.copy_async(
                Bsmem[:, :],
                B[
                    n_st : n_st + BLK_N,
                    k_st : k_st + BLK_K,
                ],
                **tma_config,
            )
            T.ptx.mbarrier.arrive.expect_tx(
                tma_bar.ptr_to([0]),
                (
                    BLK_M * BLK_K
                    + BLK_N * BLK_K
                ) * F16_SIZE,
            )

        @T.inline
        def mma(accum):
            Tx.gemm_async(
                tmem[:, :BLK_N],
                Asmem[:, :],
                Bsmem[:, :],
                accum=accum,
                dispatch="tcgen05",
                cta_group=1,
            )
            T.ptx.tcgen05.commit(
                mma_bar.ptr_to([0]),
                cta_group=1,
            )

        tid = T.meta_var(warp_id * 32 + lane_id)

        for k in range(K_TILES):
            k_st = T.meta_var(k * BLK_K)

            if tid == 0:
                tma_load(k_st)

            T.ptx.mbarrier.try_wait(
                tma_bar.ptr_to([0]),
                phase_tma,
            )

            if tid == 0:
                mma(accum=k != 0)

            T.ptx.mbarrier.try_wait(
                mma_bar.ptr_to([0]),
                phase_mma,
            )
            phase_tma ^= 1
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

### 文件二：`verify_tirx_gemm_async.py`

```python
import torch
import tvm

from tirx_gemm_async import hgemm_v4


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
    kernel = hgemm_v4(M, N, K)

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

### 运行方式

在装有 Blackwell GPU、TIRx compiler 和 CUDA 版 PyTorch 的环境中：

```bash
python verify_tirx_gemm_async.py
```

预期输出类似：

```text
device: NVIDIA B200, capability=(10, 0)
Max error vs torch reference: 0.031250
PASS
```

这里的误差不是要求逐 bit 相同。K=4096 时，MMA 的累加顺序和 PyTorch
参考实现可能不同，最后的 fp16 round 也可能不同。脚本检查的是误差是否
落在可接受范围内。

## 五、完整代码逐段映射

### 1. SMEM 中新增 Dsmem

```python
Asmem = pool.alloc((BLK_M, BLK_K), a_type, layout=A_layout)
Bsmem = pool.alloc((BLK_N, BLK_K), b_type, layout=B_layout)
Dsmem = pool.alloc((BLK_M, BLK_N), d_type, layout=D_layout)
```

Step 4 的 store 也改成 TMA，所以 epilogue 不再由寄存器直接写 GMEM，
而是：

```text
registers
-> SMEM 的 Dsmem
-> TMA store
-> GMEM
```

A/B 的 layout 仍必须同时满足：

```text
TMA 写入 SMEM 的物理地址描述
tcgen05.mma 读取 SMEM 的 operand descriptor
```

两边都使用同一个 `A_layout` / `B_layout`，把 layout 一致性收在一个
定义点。

### 2. 两个 mbarrier 的职责

```python
if warp_id == 0 and lane_id == 0:
    T.ptx.mbarrier.init(mma_bar.ptr_to([0]), 1)
    T.ptx.mbarrier.init(tma_bar.ptr_to([0]), 1)
```

| barrier | 谁触发完成 | 等什么 |
|---|---|---|
| `tma_bar` | `tid == 0` 的 arrival + TMA `complete_tx` | A/B tile 是否搬完 |
| `mma_bar` | `tcgen05.commit` 关联的 MMA 完成 | TMEM accumulator 是否写完 |

它们不能混用。TMA load 完成不代表 MMA 完成，MMA 完成也不代表下一块
TMA load 已完成。

### 3. `tma_load` 为什么放在 `@T.inline`

这里使用 `@T.inline` 标记 `tma_load(k_st)`。它表示 helper 在编译期展开，
而不是保留成普通的 Python 函数边界。

`@T.inline` 在编译期展开，不会生成一次普通函数调用。它让 A/B 的 issue
和 `expect_tx` 保持为一个原子语义单元：

```text
issue A
issue B
登记两块 tile 的总 transaction bytes
```

如果只复制两条 `Tx.copy_async`，却漏掉 `expect_tx`，mbarrier 不会有
正确的 pending byte count，consumer 可能过早通过或一直等待。

### 4. `tid == 0` 到底选中了谁

```python
tid = T.meta_var(warp_id * 32 + lane_id)
if tid == 0:
    tma_load(k_st)
```

warpgroup 中有：

```text
warp_id = 0..3
lane_id = 0..31
tid     = warp_id * 32 + lane_id = 0..127
```

只有：

```text
warp_id = 0, lane_id = 0 -> tid = 0
```

会执行 TMA issue。不能在每个 warp 里直接调用 `elect_sync()` 后发 TMA，
因为那会选出一个 elected lane per warp，总共四个 thread 提交同一块
tile。

### 5. `tma_config` 的四个语义

```python
tma_config = T.meta_var({
    "dispatch": "tma_auto",
    "cta_group": 1,
    "mbar": tma_bar.ptr_to([0]),
})
```

| 字段 | 含义 |
|---|---|
| `dispatch="tma_auto"` | 让 LowerTIRx 选择 TMA bulk tensor 路径并生成必要 descriptor |
| `cta_group=1` | 搬运范围属于单个 CTA |
| `mbar=...` | TMA transaction 完成后向哪个 mbarrier 报告 `complete_tx` |
| 两个 tile 共用同一个 `mbar` | consumer 一次等待 A/B 两块数据都完成 |

`cta_group` 在这里不是 TMEM allocation 的 group，也不是 MMA 的
shape；它描述这条异步搬运操作所使用的 CTA 范围。

### 6. 为什么 byte count 是 32768

```python
(
    BLK_M * BLK_K
    + BLK_N * BLK_K
) * F16_SIZE
```

代入课程参数：

```text
BLK_M = 128
BLK_N = 128
BLK_K = 64
F16_SIZE = 2 bytes

A bytes = 128 * 64 * 2 = 16384
B bytes = 128 * 64 * 2 = 16384
total   = 16384 + 16384 = 32768
```

这个数必须和本轮实际 TMA transaction bytes 一致。少算会造成等待条件
提前满足，多算会造成等待永远不满足。

### 7. K-loop 中的顺序

```python
if tid == 0:
    tma_load(k_st)

T.ptx.mbarrier.try_wait(tma_bar.ptr_to([0]), phase_tma)

if tid == 0:
    mma(accum=k != 0)

T.ptx.mbarrier.try_wait(mma_bar.ptr_to([0]), phase_mma)

phase_tma ^= 1
phase_mma ^= 1
```

这是一个严格顺序链：

```text
TMA A/B
-> wait TMA
-> MMA
-> wait MMA
-> 下一轮 K
```

因此 Step 4 虽然使用了异步硬件，但没有 overlap。`try_wait` 把异步操作
立即同步回顺序执行，先保证正确性。Step 5 才会拆出多个 SMEM stage，
让下一轮 Load 可以提前发。

### 8. Epilogue 为什么先写 Dsmem

```python
Tx.copy(
    Dsmem[warp_id * 32 + lane_id, 0:BLK_N],
    Dreg_f16[:],
)
T.ptx.fence.proxy_async("shared::cta")
T.cuda.warpgroup_sync(10)
```

TMEM 先读进每个 thread 的 registers，再 cast fp16，然后每个 thread
写 Dsmem 的对应一行：

```text
warp 0 / lane 0..31 -> rows 0..31
warp 1 / lane 0..31 -> rows 32..63
warp 2 / lane 0..31 -> rows 64..95
warp 3 / lane 0..31 -> rows 96..127
```

`fence.proxy_async` 把 generic proxy 对这些 SMEM bytes 的写入发布给
async proxy，使 TMA engine 能看到完整数据。`warpgroup_sync(10)` 则保证
所有 128 个 writer 都已经完成写入和 fence，之后 `tid == 0` 才发 store。

### 9. TMA store 为什么不用 mbarrier

```python
if tid == 0:
    Tx.copy_async(
        D[m_st : m_st + BLK_M, n_st : n_st + BLK_N],
        Dsmem[:, :],
        dispatch="tma_auto",
    )
    T.ptx.cp_async.bulk.commit_group()
    T.ptx.cp_async.bulk.wait_group(0)

T.cuda.warpgroup_sync(10)
```

Store 使用 bulk async group：

```text
commit_group：
  把此前已 issue 但尚未 commit 的 TMA store 组织成一个 group

wait_group(0)：
  允许 pending 的 group 数量为 0
  因此它返回时，之前 commit 的 store 已经完成
```

第二条 `warpgroup_sync(10)` 让其他 thread 也等 `tid == 0` 完成
`wait_group(0)`。这样 Dsmem 在本次 store 完成前不会被复用。

## 六、一次具体的 mbarrier 执行追踪

使用本节参数：

```text
A tile = 128 x 64 fp16 = 16384 bytes
B tile = 128 x 64 fp16 = 16384 bytes
total = 32768 bytes
```

假设当前是第一次 K iteration，`phase_tma = 0`：

| 步骤 | 执行者 | mbarrier 状态 | 结果 |
|---|---|---|---|
| 初始化 | warp 0 lane 0 | pending arrivals = 1，pending bytes = 0 | 准备进入 phase 0 |
| issue A | `tid == 0` | 不变 | TMA engine 接受 A transaction |
| issue B | `tid == 0` | 不变 | TMA engine 接受 B transaction |
| `arrive.expect_tx(32768)` | `tid == 0` | arrivals = 0，bytes = 32768 | arrival 条件已满足，但 phase 尚未完成 |
| A `complete_tx(16384)` | TMA engine | arrivals = 0，bytes = 16384 | phase 仍未完成 |
| B `complete_tx(16384)` | TMA engine | arrivals = 0，bytes = 0 | phase 0 完成 |
| `try_wait(tma_bar, 0)` | 所有需要等待的线程 | phase 0 已完成 | 返回，MMA 可以读取 SMEM |
| 循环末尾 | `tid` 逻辑 | 下一次使用 phase 1 | `phase_tma ^= 1` |

如果只看线程：

```text
tid 0:
  issue A -> issue B -> arrive.expect_tx(32768)

TMA engine:
  complete_tx(16384) -> complete_tx(16384)

consumer:
  try_wait(phase=0) passes
```

如果 B 还差 1 byte：

```text
arrivals = 0
pending bytes = 1
phase not complete
try_wait 仍然不能通过
```

这解释了为什么只做 `cta_sync` 不够：线程汇合并不能把
`pending bytes = 1` 变成 0。

## 七、常见错误和可观察症状

| 错误 | 发生原因 | 症状 |
|---|---|---|
| 用 `cta_sync` 代替 TMA wait | 线程同步不能等待 async transaction | 偶发读到旧 SMEM，正确性随调度变化 |
| 漏掉 `arrive.expect_tx` | mbarrier 不知道预期 bytes | phase 可能提前完成或 consumer 卡住 |
| byte count 少算 | 实际 transaction 大于声明值 | phase 可能在实际数据到齐前完成 |
| byte count 多算 | 声明值大于实际 transaction | pending bytes 永远不为 0，kernel hang |
| 每个 warp 都 selected lane 发 TMA | 4 个 thread 重复 issue | 重复搬运、mbarrier 状态混乱或结果错误 |
| 忘记 `phase_tma ^= 1` | 复用 barrier 时 parity 不对 | 等待上一轮或永远等不到本轮 |
| Store 前漏 `fence.proxy_async` | TMA async proxy 看不到 thread writes | TMA 读到部分旧 Dsmem，输出局部错误 |
| Store 前漏 `warpgroup_sync` | `tid == 0` 太早发 store | 其他 thread 还没写完 Dsmem |
| 以为 Step 4 已经 overlap | 每轮 issue 后立刻 wait | 正确但性能提升有限 |

### 一个排查顺序

```text
先确认 mbarrier 协议：
  init count、arrival thread、expect_tx bytes、phase flip

再确认数据路径：
  TMA destination layout 与 MMA operand layout 是否一致

最后看 epilogue：
  Dsmem writer 是否全部完成，proxy fence 与 warpgroup sync 是否齐全
```

如果错误呈现为“偶发、和运行次数有关”，优先怀疑同步。
如果错误呈现为“固定行列错位”，优先怀疑 layout 和 descriptor。

## 八、硬件与运行边界

本节完整 kernel 使用：

```text
TMA
mbarrier
tcgen05.mma
TMEM
```

因此它需要支持 Blackwell `sm_100a` 的 NVIDIA GPU。H100 / H200 有 TMA，
但没有本代码使用的 `tcgen05` TMEM 路径；在当前 Apple M5 Pro 上也无法
执行这段 CUDA/TIRx kernel。

当前机器可以做静态检查，不能做真实运行验证：

```text
可以做：
  阅读 TIRx 源码
  检查 Python 语法
  推导 layout、byte count 与 phase
  阅读 lowering 后的 CUDA source

不能做：
  执行 tcgen05.mma
  验证 TMA engine 的真实完成时序
  测量 Load / Compute overlap 前后的性能
```

如果没有目标 GPU，只保存两个 Python 文件并做语法检查：

```bash
python -m py_compile \
  tirx_gemm_async.py \
  verify_tirx_gemm_async.py
```

该命令只检查 Python 语法和文件能否编译，不导入 TVM/CUDA，也不代表
kernel 已在 GPU 上验证。

## 九、自测

### 1. 为什么 Step 4 可以由一个 thread 发起整块 A/B 搬运？

答：因为 `dispatch="tma_auto"` 把 GMEM -> SMEM 的搬运交给 TMA engine。
线程只提交操作和 descriptor，TMA engine 负责剩余地址生成与 tile
transaction。所谓“单线程”只描述 issue scope，不描述实际搬运范围。

### 2. 为什么 `cta_sync` 不能保证 TMA load 已经完成？

答：`cta_sync` 只汇集 CTA 内线程，不能观察 TMA engine 的异步
transaction。TMA 完成由 `complete_tx` 更新 mbarrier 的 pending bytes，
consumer 必须通过 `mbarrier.try_wait` 等待它。

### 3. 当前 A/B tile 的 `expect_tx` byte count 是多少？

答：

```text
A = 128 * 64 * 2 = 16384 bytes
B = 128 * 64 * 2 = 16384 bytes
total = 32768 bytes
```

### 4. `arrive.expect_tx(32768)` 后为什么 barrier 还没完成？

答：它把 pending arrivals 从 1 减到 0，同时把 pending transaction bytes
设置为 32768。phase 完成还需要 TMA engine 的全部 `complete_tx` 把
pending bytes 减到 0。

### 5. Step 4 为什么还没有真正 overlap Load 和 Compute？

答：SMEM 仍只有一组 A/B tile，而且每轮 `tma_load` 后立即执行
`try_wait`。下一轮 TMA load 不能安全覆盖当前 MMA 可能仍在读取的 SMEM。
Step 5 增加 `PIPE_DEPTH=2` 的多 stage SMEM，才具备 prefetch 和后续
overlap 的存储条件。
