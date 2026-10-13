# Triton Gluon 09–13：Blackwell 数据流与卷积

## 本次讲解位置

章节：`chapter_gluon_blackwell_dataflow`（算子开发 / Triton Gluon）<br>
小节：TMA gather/scatter、`tcgen05_copy`、block-scaled MMA、CLC、TMA im2col<br>
知识点：把不规则访存、SMEM→TMEM 搬运、低精度反量化、动态调度和卷积地址生成交给 Blackwell 硬件<br>
上次：persistent kernel 与 warp specialization 负责让不同角色跨 tile 重叠工作<br>
下次：CGA、TMA multicast、cluster barrier 与 2CTA tcgen05 的端到端 capstone<br>
PTX：TMA gather/scatter、`tcgen05.copy`、`tcgen05.mma` scaled、`clusterlaunchcontrol.try_cancel`、TMA im2col

## 为什么现在讲这个

到这里，普通 TMA、WGMMA/tcgen05、persistent loop 和 warp specialization 已经能组成一条高性能 GEMM 主线。但不同模型还留下几个很贵的“缝”：

1. MoE、稀疏 attention 或 embedding 访问的行不规则，普通 block TMA 只能搬连续矩形。
2. TMEM 是 tensor core 的工作区，但把数据从 SMEM 搬到 TMEM 如果先经过寄存器，会同时消耗寄存器、指令带宽和同步机会。
3. 低比特推理把 FP8/FP4 和 block scale 分开存；若先软件反量化再 MMA，会在寄存器里多做一遍乘法，吞吐被压低。
4. persistent kernel 的静态 tile 分配遇到 cache miss 或 ragged K，慢 tile 会把整个 SM 尾部拖住。
5. 卷积的 im2col 矩阵重叠度很高；显式物化会复制数据，地址计算也会回到普通 load。

这一节把这五个问题连成一张数据通路图：

```text
global memory
  |-- 规则 tile ----------> TMA async_load --------------------> SMEM
  |-- 不规则 rows --------> TMA gather ------------------------> SMEM
  |-- 卷积窗口 -----------> TMA im2col ------------------------> SMEM
                                                                    |
                                                                    +--> tcgen05_copy --> TMEM
                                                                    |                     |
                                                                    +--> tcgen05_mma_scaled
                                                                                          |
                                                                                       TMEM
                                                                                          |
                                  TMA store / scatter <-- SMEM <----------------------+

CLC：不改变数据通路，只让空闲 CTA 动态接管尚未启动的 cluster。
```

关键不是“这些指令都更底层”，而是让硬件承担它最擅长的索引、搬运和格式解释，同时把真正的所有权和完成同步留在程序员手里。

## 五个能力先做一次总览

| 能力 | 解决的问题 | 实际搬移 | 硬件限制最重要的一条 | 同步对象 |
|---|---|---|---|---|
| TMA gather/scatter | 不连续的若干行 | HBM/global ↔ SMEM | 2D descriptor，block 第 0 维必须是 1 | TMA mbarrier / `store_wait` |
| `tcgen05_copy` | SMEM → TMEM 不走寄存器 | SMEM → TMEM | 常见路径是 swizzled NVMMA、32-bit、非转置 | `tcgen05_commit` + mbarrier |
| `tcgen05_mma_scaled` | 反量化与 MMA 融合 | SMEM/TMEM → TMEM | scale 必须在 TMEM，B 必须是转置视图 | tensor pipe 顺序 + mbarrier |
| CLC | 动态接管未启动的 cluster | 只传 tile id | 只能取消尚未启动的 cluster | CLC mbarrier |
| TMA im2col | 隐式构造卷积 A 矩阵 | HBM/global → SMEM | `pixel_box`、stride 与 padding 必须一致 | TMA mbarrier |

## TMA gather/scatter：让规则 DMA 处理不规则行

### 心智模型

普通 TMA block load 是“从二维矩形左上角取一个矩形 tile”：

```text
desc[0:BLOCK_X, y:y+BLOCK_Y]
```

在 MoE 路由、稀疏行选择、embedding lookup 或某些 attention 变体中，行号不在连续区间里，列却是连续的：

```text
row 3, columns 0:16
row 1, columns 0:16
row 7, columns 0:16
row 0, columns 0:16
```

TMA gather 把“行号列表”和“公共列起点”交给硬件，等价的 Python 语义就是：

```python
out = tensor_desc[x_offsets, y_offset:y_offset + BLOCK_Y]
```

其中 `x_offsets` 是一维 tensor，`out.shape == (BLOCK_X, BLOCK_Y)`。scatter 反向写回：

```python
tensor_desc[x_offsets, y_offset:y_offset + BLOCK_Y] = src
```

### PTX 为什么要求特殊的 offsets layout

底层不是每行发一条普通 TMA，而是围绕 `gather4`/`scatter4` 组织。一个 warp 以 4 行为一组处理，因此 offsets 的寄存器分配必须满足：

```text
连续的 4 个 offset
  -> 落在同一个 thread 的连续寄存器中；

同一个 warp 的所有 lane
  -> 必须广播同一份 offset 数据，lane bases 不能引入新的 index bit。
```

官方给出一个通用有效布局：

```python
offsets_layout: gl.constexpr = gl.SliceLayout(
    0,
    gl.BlockedLayout(
        size_per_thread=[1, 4],
        threads_per_warp=[32, 1],
        warps_per_cta=[1, gl.num_warps()],
        order=[1, 0],
    ),
)
```

它的线性布局应满足两个可检查条件：

```text
register bases 的前两个必须是 [1] 和 [2]
lane bases 必须全部是 [0]
```

这意味着 warp 内的 lane 不负责区分 offset；offset 的 1、2 两个最低位由同一线程的寄存器承担。若 lane bases 不是零，编译器可能生成非法或低效的 gather4 分组。

### 硬约束

| 约束 | 原因 | 违反后的表现 |
|---|---|---|
| descriptor 必须是 2D | gather/scatter 的块形状按行组织 | 编译期拒绝或描述符无效 |
| descriptor block shape 必须是 `[1, BLOCK_Y]` | `BLOCK_X` 来自 offsets 长度，不来自 block 第 0 维 | 看错 shape 会误以为 block 能大于 1 |
| offsets 至少有 8 个元素 | 4 行一组的 warp 搬运需要足够并行度 | 无法形成有效 gather4/scatter4 |
| `BLOCK_Y >= (32 / bitwidth) * 8` | 每行至少要有足够连续的字节 | fp16 时 `BLOCK_Y >= 16` |
| `y_offset` 对齐 16 bytes | 硬件访问粒度和对齐要求 | CUDA context 报 `illegal instruction` |
| scatter 不接受负 row/column offset | 写回需要明确的合法地址 | 运行时非法指令；gather 可以读负数并补零 |
| 越界 gather 读 | 由硬件补零 | 输出对应行为 0 |
| 越界 scatter 写 | 由硬件忽略 | 不会写到 descriptor 之外 |

### 完整 gather 内核

下面是官方最小 gather kernel。注意 descriptor 的 block shape 是 `[1, BLOCK_Y]`，但共享内存目标可以是 `[BLOCK_X, BLOCK_Y]`，因为第一维由 offsets tensor 的长度决定：

```python
@gluon.jit
def async_gather_kernel(out_ptr, out_stride_x, out_stride_y, tensor_desc,
                        x_offsets_ptr, y_offset, BLOCK_X: gl.constexpr):
    BLOCK_Y: gl.constexpr = tensor_desc.block_type.shape[1]

    coalesced_1d_layout: gl.constexpr = gl.BlockedLayout(
        [1], [32], [gl.num_warps()], [0]
    )
    x_offsets = gl.load(
        x_offsets_ptr + gl.arange(0, BLOCK_X, coalesced_1d_layout)
    )

    offsets_layout: gl.constexpr = gl.SliceLayout(
        0,
        gl.BlockedLayout([1, 4], [32, 1], [1, gl.num_warps()], [1, 0]),
    )
    x_offsets = gl.convert_layout(x_offsets, offsets_layout)

    smem_dest = gl.allocate_shared_memory(
        tensor_desc.dtype, [BLOCK_X, BLOCK_Y], tensor_desc.layout
    )

    bar = gl.allocate_shared_memory(
        gl.int64, [1], mbarrier.MBarrierLayout()
    )
    mbarrier.init(bar, count=1)
    mbarrier.expect(bar, BLOCK_X * tensor_desc.block_type.nbytes)

    tma.async_gather(
        tensor_desc, x_offsets, y_offset, barrier=bar, result=smem_dest
    )
    mbarrier.wait(bar, phase=0)
    mbarrier.invalidate(bar)

    coalesced_2d_layout: gl.constexpr = gl.BlockedLayout(
        [1, 1], [1, 32], [1, gl.num_warps()], [1, 0]
    )
    out = smem_dest.load(coalesced_2d_layout)
    indices_x = gl.arange(
        0, BLOCK_X, gl.SliceLayout(1, coalesced_2d_layout)
    )[:, None] * out_stride_x
    indices_y = gl.arange(
        0, BLOCK_Y, gl.SliceLayout(0, coalesced_2d_layout)
    )[None, :] * out_stride_y
    gl.store(out_ptr + indices_x + indices_y, out)
```

### 数值轨迹

设一个 `8 x 4` 输入，`BLOCK_Y=4`，行数据如下：

```text
X = [[0, 1, 2, 3],
     [4, 5, 6, 7],
     [8, 9,10,11],
     [12,13,14,15],
     [16,17,18,19],
     [20,21,22,23],
     [24,25,26,27],
     [28,29,30,31]]
```

执行：

```python
x_offsets = [3, 1, 7, 0, -1, 6, 2, 5]
out = desc[x_offsets, 0:4]
```

结果逐行是：

| output row | offset | value |
|---:|---:|---|
| 0 | 3 | `[12, 13, 14, 15]` |
| 1 | 1 | `[4, 5, 6, 7]` |
| 2 | 7 | `[28, 29, 30, 31]` |
| 3 | 0 | `[0, 1, 2, 3]` |
| 4 | -1 | `[0, 0, 0, 0]`，负行号由 gather 补零 |
| 5 | 6 | `[24, 25, 26, 27]` |
| 6 | 2 | `[8, 9, 10, 11]` |
| 7 | 5 | `[20, 21, 22, 23]` |

### 完整 scatter 内核

scatter 先从普通 global memory 读 source 到寄存器，再写入共享内存，最后让 TMA 从共享内存按行散播。两者之间必须有 `fence_async_shared()`，因为写入使用 generic proxy，TMA 读取使用 async proxy。

```python
@gluon.jit
def async_scatter_kernel(tensor_desc, x_offsets_ptr, y_offset,
                         src_ptr, src_stride_x, src_stride_y,
                         BLOCK_X: gl.constexpr):
    BLOCK_Y: gl.constexpr = tensor_desc.block_type.shape[1]

    coalesced_2d_layout: gl.constexpr = gl.BlockedLayout(
        [1, 1], [1, 32], [1, gl.num_warps()], [1, 0]
    )
    indices_x = gl.arange(
        0, BLOCK_X, gl.SliceLayout(1, coalesced_2d_layout)
    )[:, None] * src_stride_x
    indices_y = gl.arange(
        0, BLOCK_Y, gl.SliceLayout(0, coalesced_2d_layout)
    )[None, :] * src_stride_y
    src = gl.load(src_ptr + indices_x + indices_y)

    coalesced_1d_layout: gl.constexpr = gl.BlockedLayout(
        [1], [32], [gl.num_warps()], [0]
    )
    x_offsets = gl.load(
        x_offsets_ptr + gl.arange(0, BLOCK_X, coalesced_1d_layout)
    )
    offsets_layout: gl.constexpr = gl.SliceLayout(
        0,
        gl.BlockedLayout([1, 4], [32, 1], [1, gl.num_warps()], [1, 0]),
    )
    x_offsets = gl.convert_layout(x_offsets, offsets_layout)

    smem_src = gl.allocate_shared_memory(
        tensor_desc.dtype, [BLOCK_X, BLOCK_Y], tensor_desc.layout
    )
    smem_src.store(src)
    fence_async_shared()
    tma.async_scatter(tensor_desc, x_offsets, y_offset, smem_src)
    tma.store_wait(0)
```

### `async_scatter` 和普通 store 的选择

官方结论不是“永远用 gather/scatter”。三种路径的边界是：

```text
连续矩形：
  TMA async_load / async_store

若干任意行、列区间连续：
  TMA async_gather / async_scatter

完全任意元素地址、mask 复杂：
  gl.load / gl.store 或 async_copy
```

gather/scatter 仍然经过共享内存，且需要 offsets 布局转换。当不规则度很低时，这种转换和 mbarrier 开销可能超过收益；应当在目标 shape 上测，而不是只看“用了 TMA”这个标签。

## Fused gather–GEMM–scatter：把不规则行选择接进持久化 GEMM

### 本次讲解位置

章节：`chapter_gluon_blackwell_dataflow`（算子开发 / Triton Gluon）<br>
小节：TMA Gather and Scatter → Fused Gather/Scatter GEMM<br>
知识点：`out[out_scatter_indx, :] = X[X_gather_indx, :] @ W` 的多缓冲数据流<br>
上次：单独的 `async_gather` / `async_scatter`：offsets layout、shared memory 和完成同步<br>
下次：`tcgen05_copy`：为什么 SMEM→TMEM 值得再开一条异步通路<br>
PTX：Blackwell TMA `gather4` / `scatter4` + `tcgen05.mma`

### 为什么现在讲这个

前面只把若干行搬进/搬出 SMEM；真实算子不会为了搬运而搬运。MoE token 重排、稀疏行选择等场景要做的是“按索引取 X 的行 → 乘 W → 按另一组索引写回”。如果拆成 gather kernel、GEMM kernel、scatter kernel，中间矩阵要落到 global memory，再被重新读取；如果把行号误当连续 tile 的起点，结果又会悄悄算错。本知识点把 TMA gather/scatter 放进同一个 tcgen05 GEMM，并解释它怎样用持久 CTA 和缓冲环把输入加载、MMA、输出散写串成可重叠的数据流。

### 先看数学语义，再看硬件

设 `X.shape=(M,K)`、`W.shape=(K,N)`，两组索引长度都是 `M`：

```text
Y[i, n] = Σ(k=0..K-1) X[X_gather_indx[i], k] * W[k, n]
out[out_scatter_indx[i], n] = Y[i, n]
```

`i` 是“逻辑行号”。它不是 X 的物理行号，也不是 out 的物理行号：

- 从 X 读取时用 `X_gather_indx[i]`；
- GEMM 的 M 维仍按逻辑行 `i` 分块；
- 写回时用 `out_scatter_indx[i]`。

所以每个输出 tile `(pid_m,pid_n)` 的三块数据是：

```text
X tile:  X[X_gather_indx[pid_m*BM : ...], k : k+BK]  -- TMA gather
W tile:  W[k : k+BK, pid_n*BN : ...]                 -- TMA block load
Y tile:  acc[0:BM, 0:BN]                              -- tcgen05 MMA
out:     out[out_scatter_indx[pid_m*BM : ...], ...]   -- TMA scatter
```

X 的 M 行不连续，但每行 K 列连续，因此正好符合 gather 的“索引行 + 连续列段”。W 仍是普通矩形，不需要 gather。

### 完整核心实现（Blackwell）

下面保留教程的核心路径，并把它依赖的 `MMAv5` accumulator 和 grouped persistent scheduler 一并列出，便于看清 `issue_loads → issue_mma → epilogue`。将代码保存成独立的 `fused_gather_scatter_demo.py`，不要覆盖仓库内固定版本归档的官方脚本；即可使用文末命令在 Blackwell 上跑正确性检查。

```python
import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor
from triton.experimental.gluon.language.nvidia.blackwell import (
    TensorMemoryLayout,
    tensor_memory_descriptor,
    allocate_tensor_memory,
    tcgen05_mma,
    tcgen05_commit,
    tma,
    mbarrier,
    fence_async_shared,
)


@gluon.aggregate
class MMAv5:
    use_acc: gl.tensor
    acc_tmem: tensor_memory_descriptor
    bar: gl.shared_memory_descriptor
    counter: gl.tensor

    @gluon.jit
    def initialize(dtype: gl.constexpr, BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr,
                   num_warps: gl.constexpr):
        layout: gl.constexpr = TensorMemoryLayout([BLOCK_M, BLOCK_N], col_stride=1)
        acc_tmem = allocate_tensor_memory(gl.float32, [BLOCK_M, BLOCK_N], layout)
        bar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
        mbarrier.init(bar, count=1)
        return MMAv5(gl.to_tensor(False), acc_tmem, bar, gl.to_tensor(0))

    @gluon.jit
    def issue_async_mma(self, a, b):
        tcgen05_mma(a, b, self.acc_tmem, use_acc=self.use_acc)
        tcgen05_commit(self.bar)
        return MMAv5(gl.to_tensor(True), self.acc_tmem, self.bar, self.counter + 1)

    @gluon.jit
    def wait_num_outstanding(self, num_outstanding: gl.constexpr):
        mbarrier.wait(self.bar, (self.counter - 1 - num_outstanding) & 1)
        return self

    @gluon.jit
    def take_result(self, splitn: gl.constexpr = False):
        next_mma = MMAv5(gl.to_tensor(False), self.acc_tmem, self.bar, self.counter)
        return self.acc_tmem.load(), next_mma


def GroupedPersistentTileScheduler(GROUP_SIZE_M):
    GROUP_SIZE_M = gl.constexpr(GROUP_SIZE_M)

    @gluon.aggregate
    class Scheduler:
        start_pid: gl.tensor
        num_pid_m: gl.tensor
        num_pid_in_group: gl.tensor
        num_pid: gl.tensor

        @gluon.jit
        def initialize(M, N, BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr):
            start_pid = gl.program_id(axis=0)
            num_pid_m = gl.cdiv(M, BLOCK_M)
            num_pid_n = gl.cdiv(N, BLOCK_N)
            num_pid_in_group = GROUP_SIZE_M * num_pid_n
            num_pid = num_pid_m * num_pid_n
            return Scheduler(start_pid, num_pid_m, num_pid_in_group, num_pid)

        @gluon.jit
        def get_num_tiles(self):
            return gl.cdiv(self.num_pid - self.start_pid, gl.num_programs(axis=0))

        @gluon.jit
        def get_tile(self, idx):
            tile_id = self.start_pid + idx * gl.num_programs(axis=0)
            group_id = tile_id // self.num_pid_in_group
            first_pid_m = group_id * GROUP_SIZE_M
            group_size_m = min(self.num_pid_m - first_pid_m, GROUP_SIZE_M)
            pid_m = first_pid_m + (tile_id % group_size_m)
            pid_n = (tile_id % self.num_pid_in_group) // group_size_m
            return pid_m, pid_n

    return Scheduler


@gluon.jit
def issue_loads(producer, X_desc, W_desc, X_gather_indx_ptr, off_m, off_n, k,
                bars, x_bufs, w_bufs, BLOCK_M: gl.constexpr,
                num_buffers: gl.constexpr, pred=True):
    gather_indx_layout: gl.constexpr = gl.SliceLayout(
        0, gl.BlockedLayout([1, 4], [32, 1], [1, gl.num_warps()], [1, 0])
    )
    offs_x_m = gl.load(
        X_gather_indx_ptr + off_m + gl.arange(0, BLOCK_M, gather_indx_layout)
    )

    index = producer % num_buffers
    producer += 1
    bar = bars.index(index)

    mbarrier.expect(
        bar, W_desc.block_type.nbytes + BLOCK_M * X_desc.block_type.nbytes
    )
    tma.async_gather(X_desc, offs_x_m, k, bar, x_bufs.index(index), pred)
    tma.async_load(W_desc, [k, off_n], bar, w_bufs.index(index), pred)
    return producer


@gluon.jit
def issue_mma(consumer, mma, bars, x_bufs, w_bufs,
              num_buffers: gl.constexpr):
    index = consumer % num_buffers
    b_index = consumer % num_buffers
    phase = consumer // num_buffers & 1
    consumer += 1
    mbarrier.wait(bars.index(index), phase)
    mma = mma.wait_num_outstanding(0)
    mma = mma.issue_async_mma(x_bufs.index(index), w_bufs.index(b_index))
    return consumer, mma


@gluon.jit
def matmul_fused_gather_scatter_kernel(
    X_desc, W_desc, out_desc, X_gather_indx_ptr, out_scatter_indx_ptr,
    BLOCK_M: gl.constexpr, SchedulerImpl: gl.constexpr,
    num_buffers: gl.constexpr,
):
    BLOCK_N: gl.constexpr = W_desc.block_type.shape[1]
    BLOCK_K: gl.constexpr = W_desc.block_type.shape[0]
    dtype: gl.constexpr = X_desc.dtype
    M = X_desc.shape[0]
    N = W_desc.shape[1]
    K = X_desc.shape[1]

    x_bufs = gl.allocate_shared_memory(
        dtype, [num_buffers, BLOCK_M, BLOCK_K], X_desc.layout
    )
    w_bufs = gl.allocate_shared_memory(
        dtype, [num_buffers, BLOCK_K, BLOCK_N], W_desc.layout
    )
    out_smem = gl.allocate_shared_memory(
        dtype, [BLOCK_M, BLOCK_N], out_desc.layout
    )

    bars = gl.allocate_shared_memory(
        gl.int64, [num_buffers, 1], mbarrier.MBarrierLayout()
    )
    for i in gl.static_range(num_buffers):
        mbarrier.init(bars.index(i), count=1)
    producer = 0
    consumer = 0

    mma = MMAv5.initialize(dtype, BLOCK_M, BLOCK_N, gl.num_warps())
    scheduler = SchedulerImpl.initialize(M, N, BLOCK_M, BLOCK_N)
    num_tiles = scheduler.get_num_tiles()

    # Prologue: 先把可用的 load slots 填起来。
    idx = 0
    pid_m, pid_n = scheduler.get_tile(idx)
    off_m = pid_m * BLOCK_M
    off_n = pid_n * BLOCK_N
    for ki in gl.static_range(0, BLOCK_K * (num_buffers - 2), BLOCK_K):
        producer = issue_loads(
            producer, X_desc, W_desc, X_gather_indx_ptr, off_m, off_n, ki,
            bars, x_bufs, w_bufs, BLOCK_M, num_buffers
        )
    k = BLOCK_K * (num_buffers - 2)
    producer = issue_loads(
        producer, X_desc, W_desc, X_gather_indx_ptr, off_m, off_n, k,
        bars, x_bufs, w_bufs, BLOCK_M, num_buffers
    )

    # Persistent outer loop: 每轮产出一个 (BM, BN) 输出 tile。
    for _ in range(num_tiles):
        consumer, mma = issue_mma(consumer, mma, bars, x_bufs, w_bufs, num_buffers)
        for k in range(BLOCK_K * (num_buffers - 1), K, BLOCK_K):
            producer = issue_loads(
                producer, X_desc, W_desc, X_gather_indx_ptr, off_m, off_n, k,
                bars, x_bufs, w_bufs, BLOCK_M, num_buffers
            )
            consumer, mma = issue_mma(
                consumer, mma, bars, x_bufs, w_bufs, num_buffers
            )

        epilogue_off_m = off_m
        epilogue_off_n = off_n
        scatter_indx_layout: gl.constexpr = gl.SliceLayout(
            0, gl.BlockedLayout([1, 4], [32, 1], [1, gl.num_warps()], [1, 0])
        )
        out_offs_m = gl.load(
            out_scatter_indx_ptr + epilogue_off_m
            + gl.arange(0, BLOCK_M, scatter_indx_layout)
        )

        # 下一 tile 的 prologue 与当前 tile 的 K-pipeline drain 重叠。
        # 只在确实有 next tile 时才求 scheduler 坐标、读 offsets、发起 TMA。
        idx += 1
        has_next = idx < num_tiles
        if has_next:
            next_pid_m, next_pid_n = scheduler.get_tile(idx)
            next_off_m = next_pid_m * BLOCK_M
            next_off_n = next_pid_n * BLOCK_N

        for ki in gl.static_range(0, BLOCK_K * (num_buffers - 2), BLOCK_K):
            if has_next:
                producer = issue_loads(
                    producer, X_desc, W_desc, X_gather_indx_ptr,
                    next_off_m, next_off_n, ki, bars, x_bufs, w_bufs,
                    BLOCK_M, num_buffers
                )
            # 即使没有 next tile，也要消费完当前 tile pipeline 中的最后一个 K slice。
            consumer, mma = issue_mma(
                consumer, mma, bars, x_bufs, w_bufs, num_buffers
            )
        if has_next:
            k = BLOCK_K * (num_buffers - 2)
            producer = issue_loads(
                producer, X_desc, W_desc, X_gather_indx_ptr,
                next_off_m, next_off_n, k, bars, x_bufs, w_bufs,
                BLOCK_M, num_buffers
            )

        mma = mma.wait_num_outstanding(0)
        out, mma = mma.take_result()
        out = out.to(dtype)
        tma.store_wait(pendings=0)
        out_smem.store(out)
        fence_async_shared()
        tma.async_scatter(out_desc, out_offs_m, epilogue_off_n, out_smem)

    tma.store_wait(pendings=0)


def matmul_fused_gather_scatter(X, X_gather_indx, W, out_scatter_indx,
                                BLOCK_M=128, BLOCK_N=128, BLOCK_K=64,
                                GROUP_SIZE_M=8, num_buffers=3):
    M, K = X.shape
    N = W.shape[1]
    if M % BLOCK_M or N % BLOCK_N or K % BLOCK_K:
        raise ValueError("This teaching kernel requires M/N/K divisible by their block sizes")
    if K < (num_buffers - 1) * BLOCK_K:
        raise ValueError("K must cover the initial num_buffers-1 stage prefetch")
    if X_gather_indx.numel() != M or out_scatter_indx.numel() != M:
        raise ValueError("Both row-index arrays must have length M")
    out = torch.empty((M, N), dtype=X.dtype, device=X.device)
    dtype = getattr(gl, str(X.dtype).split('.')[1])

    x_layout = gl.NVMMASharedLayout.get_default_for([BLOCK_M, BLOCK_K], dtype)
    w_layout = gl.NVMMASharedLayout.get_default_for([BLOCK_K, BLOCK_N], dtype)
    out_layout = gl.NVMMASharedLayout.get_default_for([BLOCK_M, BLOCK_N], dtype)
    X_desc = TensorDescriptor.from_tensor(X, [1, BLOCK_K], x_layout)
    W_desc = TensorDescriptor.from_tensor(W, [BLOCK_K, BLOCK_N], w_layout)
    out_desc = TensorDescriptor.from_tensor(out, [1, BLOCK_N], out_layout)

    num_sms = torch.cuda.get_device_properties(X.device).multi_processor_count
    num_pid = triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N)
    grid = (min(num_sms, num_pid),)
    SchedulerImpl = GroupedPersistentTileScheduler(GROUP_SIZE_M)
    matmul_fused_gather_scatter_kernel[grid](
        X_desc, W_desc, out_desc, X_gather_indx, out_scatter_indx,
        BLOCK_M, SchedulerImpl, num_buffers
    )
    return out


def check_fused_gather_scatter():
    torch.manual_seed(0)
    M, N, K = 1024, 1024, 2048
    X = torch.randn((M, K), device="cuda", dtype=torch.bfloat16)
    W = torch.randn((K, N), device="cuda", dtype=torch.bfloat16)
    X_gather_indx = torch.randperm(M, device="cuda", dtype=torch.int32)
    out_scatter_indx = torch.randperm(M, device="cuda", dtype=torch.int32)

    out = matmul_fused_gather_scatter(
        X, X_gather_indx, W, out_scatter_indx,
        BLOCK_M=128, BLOCK_N=128, BLOCK_K=64, num_buffers=3
    )
    out_ref = torch.empty_like(out)
    out_ref[out_scatter_indx, :] = X[X_gather_indx, :] @ W
    torch.testing.assert_close(out, out_ref, atol=1e-3, rtol=1e-3)
    max_error = (out.float() - out_ref.float()).abs().max().item()
    print(f"check passed; max_abs_error={max_error:.6g}")


if __name__ == "__main__":
    check_fused_gather_scatter()
```

> 上面 `MMAv5` 和 scheduler 使样例能独立阅读；它保留官方教程的流水结构，并把最后一次 peeled prologue 写成显式 `has_next` 分支。这样当 persistent CTA 没有下一个 tile 时，不会再计算越界 scheduler 坐标或读取不存在的 offsets；但它仍假设 M/N/K 可被相应 block 整除、索引有效，而且 K 至少能覆盖初始的 `num_buffers-1` 级预取。若支持不整除尾块，还必须给当前 tile 的 offsets 和 MMA 有效区域增加 mask，不能把这段示例直接当成任意 shape 的完整生产 kernel。

### 从代码连回线程、内存和同步

| 阶段 | 谁执行 / 读写哪里 | 关键不变量与同步 | 漏掉之后的典型问题 |
|---|---|---|---|
| offsets 的 `gl.load` | 当前 CTA 的线程从 global 读 `BM` 个行号 | `SliceLayout` 将 gather4/scatter4 所需的 4 个 offset 放到同线程连续寄存器；一个 warp 的 lanes 持有相同索引组 | 编译拒绝，或 offsets layout conversion/指令生成不满足约束 |
| `tma.async_gather` + `tma.async_load` | TMA 从 global 搬 X 的不规则行和 W 的规则 tile 到选中的 SMEM slot | 两个传输共用该 slot 的 mbarrier；`expect` 字节数必须等于两者总量；消费前 `wait` | barrier bytes 错会卡住或过早放行；先读会拿到旧数据/半块数据 |
| `issue_mma` | CTA 发 tcgen05 MMA，操作数来自刚完成的两个 SMEM tile，accumulator 在 TMEM | 等对应 buffer 的 barrier phase；`wait_num_outstanding(0)` 后再发；`use_acc` 由 MMAv5 状态区分首个 K slice 与累加 | phase 错导致挂起；buffer 提前覆盖或 accumulator 初始化错会产生随机/系统性错误 |
| `take_result` → `out_smem.store` | 等 MMA 完成，将 TMEM 结果取回寄存器，再写 SMEM | accumulator 完成后才能读取；写后必须 `fence_async_shared()`，因为下一步 TMA 走 async proxy | 没有等待时读到未完成 accumulator；没 fence 时 scatter 可能看到旧 SMEM |
| `tma.async_scatter` | TMA 从 SMEM 将 tile 写到 `out_scatter_indx` 指定的行 | 复用单个 `out_smem` 前 `store_wait(0)`；kernel 末再 wait 一次 | 过早覆盖 source SMEM 或 kernel 返回前写未完成 |

### 多缓冲流水逐拍追踪：`num_buffers=3, BLOCK_K=64`

令一个输出 tile 的 `K=256`，因此有 4 个 K-slice。producer 与 consumer 是单调递增的逻辑序号，物理槽位由 `%3` 决定；barrier generation/phase 由 `floor(i/3) & 1` 决定：

| 动作顺序 | producer / consumer | 物理 SMEM slot | 当前 tile 的 K slice | phase | 说明 |
|---:|---|---:|---:|---:|---|
| 1 | producer 0 | 0 | 0 | — | 预取 K `[0,64)` |
| 2 | producer 1 | 1 | 1 | — | 预取 K `[64,128)` |
| 3 | consumer 0 | 0 | 0 | 0 | barrier 0 完成后发 MMA |
| 4 | producer 2 | 2 | 2 | — | 在 MMA 运行时预取 K `[128,192)` |
| 5 | consumer 1 | 1 | 1 | 0 | 发第二个 K-slice MMA |
| 6 | producer 3 | 0 | 3 | — | 槽 0 绕回；这是新一代传输 |
| 7 | consumer 2 | 2 | 2 | 0 | 消费 slot 2 |
| 8 | producer 4 | 1 | 下一输出 tile 的 0 | — | 外层循环边界提前预取下一 tile |
| 9 | consumer 3 | 0 | 当前 tile 的 3 | 1 | `3 % 3=0` 仍是 slot 0，但 phase 翻成 1，避免把旧完成误认成新完成 |
| 10 | producer 5 | 2 | 下一输出 tile 的 1 | — | 新 tile 继续填 buffer；之后取回当前 tile 的 accumulator 并 scatter |

注意序号 8/9 的“交错”：next-tile 的输入加载可以和 current-tile 的最后一个 MMA/drain 重叠；这正是 persistent outer loop 不只是“把普通 for loop 放在 kernel 里”的原因。

### 用真实索引走一行，再手算一个输出元素

假设逻辑行 `i=0`：`X_gather_indx[0]=13`，`out_scatter_indx[0]=5`。取一段小的 K 数据演示数学语义（这是地址/数值示意，不是 Blackwell 的硬件 tile 大小）：

```text
X[13, :] = [1, 2, 0, 1]
W[:, 2]  = [3,-1, 4, 2]
Y[0,2]   = 1*3 + 2*(-1) + 0*4 + 1*2 = 3
最终写入 out[5,2] = 3
```

实际 kernel 在一个 `(BM,BN)` tile 上并行算这 `BM×BN` 个逻辑输出；`out_scatter_indx` 只改变输出行地址，不改变 MMA 的逻辑矩阵坐标。

### 选择这条路的条件

| 数据形态 | 典型做法 | 付出的代价 |
|---|---|---|
| M 行连续，N/K 为规则块 | 普通 TMA `async_load/store` + GEMM | 最简单、通常首选 |
| M 行不连续、每行 K 连续；写回也是按行置换 | TMA gather/scatter + GEMM 融合 | offset 布局、mbarrier、SMEM buffer 与尾块处理更复杂 |
| 元素地址任意、行内也不连续或 mask 特别复杂 | `gl.load/store` / async copy | 索引灵活，但可能失去 TMA 的搬运效率 |

### 容易踩的坑

1. **把两个索引数组合成一个。** gather index 决定“从 X 的哪行读”，scatter index 决定“Y 的逻辑行写到 out 哪行”；它们是两个独立映射。误用会得到形状正确但行语义错位的结果。
2. **`BLOCK_M` 和 descriptor 首维混为一谈。** X/out descriptor 分别是 `[1,BLOCK_K]`、`[1,BLOCK_N]`；一次 gather/scatter 的多行数来自 offsets 长度 `BLOCK_M`，不是 descriptor block 首维。
3. **producer/consumer 用同一个随手取的 slot 或不翻 phase。** 多缓冲循环会重用 SMEM barrier；相同 slot 的后续一代必须区分 phase，否则可能等到上一代就误消费。
4. **scatter 没有完成就覆盖 `out_smem`。** TMA scatter 是异步读取 SMEM 的；`fence_async_shared()` 解决 generic/async proxy 可见性，`store_wait` 解决生命周期/完成，两者不互相替代。
5. **scatter 行索引不是排列。** 官方测试用 permutation，保证 out 的每行都被唯一写一次。重复索引会有并发写冲突；漏掉某些行则这些位置仍是未初始化值。这不是普通 scatter 自动定义的归约或填零。
6. **把示例当成可直接处理任意 shape。** 官方测试 shape 的 M/N/K 对 block 整除且 indices 合法；加尾块时 offsets 的 `gl.load` 也必须 mask，且所有 peeled load 都要正确 predicate。BF16 的 gather/scatter 还须满足行段最小宽度及列偏移 16-byte 对齐等前置条件。

### 验证命令与预期

在安装了匹配版本 Triton、CUDA，并运行于 Blackwell GPU 的环境：

```bash
cd /Users/saboxu/Downloads/learning_notes/operators/code/triton/gluon/tutorials
python fused_gather_scatter_demo.py
```

预期：reference 对比断言通过，最后打印 `check passed; max_abs_error=...`。因 BF16 舍入，最大误差是小的非负数，不应要求 bitwise 相同。仓库中完整官方 pytest（包含 gather、scatter、融合 GEMM 参数组）可用：

```bash
cd /Users/saboxu/Downloads/learning_notes
pytest -q operators/code/triton/gluon/tutorials/09-tma-gather-scatter.py
```

本机没有 CUDA/Blackwell 和 Triton GPU runtime，不能在此验证运行时数值或性能；只能做源码静态检查。命令可在目标机器执行。

### 自我检查

1. **为什么 `X_gather_indx` 和 `out_scatter_indx` 不能互换？**
   - 答：前者映射逻辑输入行到 X 的物理行，后者映射逻辑输出行到 out 的物理行；矩阵乘法在逻辑行空间进行。
2. **`num_buffers=3` 时 consumer 序号 3 读哪个槽、等哪个 phase？**
   - 答：`3 % 3 = 0`，读 slot 0；`(3 // 3) & 1 = 1`，等新一代 phase 1。
3. **为什么 barrier 的 expected bytes 要把 gather 与 W load 两部分相加？**
   - 答：同一个 slot 的 barrier 表示“这对 MMA 操作数都可用”；只等其中一个传输结束就可能让 MMA 读到另一个尚未到齐的 tile。
4. **`fence_async_shared()` 和 `tma.store_wait(0)` 各解决什么？**
   - 答：fence 建立 generic proxy 写 SMEM 到 async proxy 读取 SMEM 的可见性；store_wait 等待异步 scatter 完成，以便安全复用 SMEM/结束 kernel。

## `tcgen05_copy`：SMEM 到 TMEM 的异步通道

### 为什么需要它

MMA 的操作数可以从 SMEM 读，但 accumulator 和 block scales 常常必须位于 TMEM。如果路径是：

```text
SMEM -> registers -> TMEM
```

就会多消耗寄存器、寄存器和 TMEM 之间的 load/store 指令，以及两次可见性同步。`tcgen05_copy` 把它变成：

```text
SMEM --tcgen05_copy--> TMEM
```

它仍然是异步的，完成信号由 `tcgen05_commit(bar)` 跟踪：

```python
smem.store(value)
fence_async_shared()
tcgen05_copy(smem, tmem)
tcgen05_commit(bar)
mbarrier.wait(bar, phase)
```

关键顺序保证是：`tcgen05_copy`、`tcgen05_mma` 和 `tcgen05_commit` 都在 SM 的 tensor pipe 上按发射顺序执行。因此下面的代码不需要在 copy 和 mma 之间额外等待：

```python
tcgen05_copy(smem, tmem)
tcgen05_mma(tmem, rhs_smem, acc_tmem, use_acc=False)
tcgen05_commit(bar)
```

MMA 天然排在 copy 后面。barrier 是给普通线程或 epilogue 用的，不是 tensor pipe 之间的全局串行锁。

### 常见布局约束

官方当前教程给出的 `NVMMASharedLayout -> TensorMemoryLayout` 限制：

| 要求 | 值 | 原因 |
|---|---|---|
| shared layout | `swizzle_byte_width > 0` | `tcgen05.copy` 只接受支持 swizzle 的 atom |
| dtype | 32-bit，例如 `gl.float32` | 该路径的 instruction atom 限制 |
| `TensorMemoryLayout.blockM` | `128` | TMEM atom 的行组织 |
| shared layout | 非 transposed | 转置路径不在该基础 atom 中 |

注意这些限制属于当前 `TensorMemoryLayout` 路径；其它 TMEM layout 或内部 atom 可能有不同形状，不能把表当成所有 TMEM 操作的全局规则。

### 完整最小 kernel

```python
@gluon.jit
def tcgen05_copy_kernel(in_ptr, in_stride0, in_stride1,
                        out_ptr, out_stride0, out_stride1,
                        M: gl.constexpr, N: gl.constexpr,
                        smem_layout: gl.constexpr,
                        tmem_layout: gl.constexpr):
    coalesced_2d_layout: gl.constexpr = gl.BlockedLayout(
        [1, 1], [1, 32], [1, gl.num_warps()], [1, 0]
    )
    offs_m = gl.arange(0, M, gl.SliceLayout(1, coalesced_2d_layout))
    offs_n = gl.arange(0, N, gl.SliceLayout(0, coalesced_2d_layout))
    input = gl.load(
        in_ptr + offs_m[:, None] * in_stride0
        + offs_n[None, :] * in_stride1
    )

    smem = gl.allocate_shared_memory(input.dtype, (M, N), smem_layout)
    tmem = allocate_tensor_memory(input.dtype, (M, N), tmem_layout)

    bar = gl.allocate_shared_memory(
        gl.int64, [1], gl.constexpr(mbarrier.MBarrierLayout())
    )
    mbarrier.init(bar, count=1)

    smem.store(input)
    fence_async_shared()
    tcgen05_copy(smem, tmem)
    tcgen05_commit(bar)
    mbarrier.wait(bar, phase=0)
    mbarrier.invalidate(bar)

    output = tmem.load()
    output = gl.convert_layout(output, coalesced_2d_layout)
    gl.store(
        out_ptr + offs_m[:, None] * out_stride0
        + offs_n[None, :] * out_stride1,
        output,
    )
```

一次 `M x N` copy 的执行轨迹：

```text
thread 侧：global -> registers -> SMEM
generic/async proxy fence
tensor pipe：tcgen05.copy(SMEM -> TMEM)
tensor pipe：tcgen05.commit -> mbarrier arrival
thread 侧：wait 完成
thread 侧：TMEM -> registers -> global
```

如果把 `wait` 删掉，常见症状不是稳定的错误，而是随机读到旧 TMEM 或半完成数据。因为 tensor pipe 是异步的，代码布局正确不代表执行顺序正确。

### accumulate GEMM 中的用法

`tcgen05_copy` 很适合把已有的 C tile 预先送进 TMEM，然后执行：

```text
D = A @ B + C
```

数据通路是：

```text
TMA(C) -> C_smem
tcgen05_copy(C_smem -> acc_tmem)
tcgen05_mma(A_smem, B_smem, acc_tmem, use_acc=True)
```

本教程还给出了更通用的 scales 路径：先 TMA 把 scale block 加载到 SMEM，再用 `tcgen05_copy` 搬到 `TensorMemoryScalesLayout` 的 TMEM，最后直接喂给 `tcgen05_mma_scaled`。

### 本次讲解位置

章节：`Triton Gluon 09–13：Blackwell 数据流与卷积`<br>
小节：TCGen05 Copy Instruction → `D = A @ B + C` 的 TMEM accumulator 初始化<br>
知识点：先将 C tile 用 `tcgen05_copy` 放进 TMEM，再用 `tcgen05_mma(use_acc=True)` 累加乘积<br>
上次：融合 gather → tcgen05 GEMM → scatter：persistent tile 与多缓冲流水<br>
下次：Block-scaled MMA：把低精度块的 scale 搬进 TMEM 并融合计算<br>
PTX：`tcgen05.copy`、`tcgen05.mma`、`tcgen05.commit`

### 为什么现在讲这个

上一课解决的是“矩阵行不连续时，如何把 gather/GEMM/scatter 接成流水”。但即使 A、B 已经进入 SMEM，`D=A@B+C` 仍有一个关键选择：C 是不是先被搬进 tensor core 的 TMEM accumulator？若让 MMA 从零开始，最后再做一次加法，就多出一次 accumulator 读写和独立加法；若把 C tile 预置到 TMEM，MMA 可直接在它上面累加。这个选择也决定 `use_acc`、TMEM 可见时机以及 C 的 SMEM 何时能复用，能避免“结果少了 C”或“异步 copy 尚未结束就覆盖 C buffer”的隐蔽错误。

### 心智模型：三个异步角色，不是一条普通 Python 顺序

```text
普通线程 / generic proxy：准备 SMEM 数据、最后读回 TMEM
                         │ fence_async_shared()
                         ▼
SM async proxy：TMA 或 tensor-core 指令访问 SMEM
                         │ mbarrier 告知完成 / buffer 可复用
                         ▼
tensor pipe：tcgen05.copy(C_smem → acc_tmem)
             tcgen05.mma(A_smem, B_smem, acc_tmem, use_acc=True)
```

把它拆成三个不同问题，就不会把同步 API 混为一谈：

| 问题 | 机制 | 它保证什么 | 它不保证什么 |
|---|---|---|---|
| generic 线程写进 SMEM，tensor pipe 能否看见？ | `fence_async_shared()` | generic proxy 的 SMEM 写对 async proxy 可见 | 不等 copy/MMA 完成，也不允许覆盖仍在使用的 buffer |
| copy 和 MMA 谁先执行？ | tensor pipe 上的发射顺序 | 前面的 `tcgen05_copy` 完成后，后面的 MMA 才消费其 TMEM 结果 | 普通线程不能因此立刻安全读取 TMEM |
| 何时可让普通线程读取结果或复用 SMEM？ | `tcgen05_commit(bar)` + `mbarrier.wait` | 对应 tensor-core 工作完成，并通过 barrier 建立跨线程/代理的完成点 | 不能修复错误的布局、shape 或错误的 barrier phase |

**关键结论：** copy 与紧随其后的 MMA 之间通常不用插入一个“等 copy 完成”的 barrier；但在普通线程读 accumulator、或生产者重写异步操作正在读取的 SMEM 前，必须经过正确的完成/所有权同步。

### 术语与不变量

以 `A[M,K]`、`B[K,N]`、`C[M,N]` 为例：

- `A`、`B`：本例为 FP16，供 tensor core 做乘法；TMA/共享内存 layout 要符合 `tcgen05_mma` 的要求。
- `C`、累加器和 `D`：本例为 FP32。`C` 的每个输出 tile 经 `tcgen05_copy` 进入 TMEM 中的 accumulator tile。
- `BLOCK_M=128, BLOCK_N=128`：一个 accumulator tile 的二维形状。当前教程中的 `NVMMASharedLayout → TensorMemoryLayout` copy 路径是 **2D、32-bit、swizzled、非转置**，并要求 TMEM 的 `blockM=128`；这是这条 copy 路径的限制，不应泛化成所有 TMEM 操作的规则。
- `use_acc=True`：这次 MMA 要读取 accumulator 中原有的值，再加上 `A_tile @ B_tile`。C 已经是初值，所以第一次就必须为真；后续 K tile 也继续累加。
- barrier phase：同一个 barrier 被循环复用时用 phase 区分第几代事件。phase 跟着 producer/consumer 的协议翻转，不能只因循环回到某个 buffer index 就固定写 `0`。

### 完整最小示例：一个 tile 计算 `D=A@B+C`

这是可单独保存运行的教学代码，对应仓库文件 [`10-tcgen05-copy-accumulate-demo.py`](code/triton/gluon/tutorials/10-tcgen05-copy-accumulate-demo.py)。为聚焦 tensor-pipe 依赖，它用普通 `gl.load` 读取一个完整 tile，再写 SMEM；**这不是性能实现**。官方教程的生产型版本把 global→SMEM 改成 TMA，并用 warp specialization 重叠 load、compute、epilogue。

```python
"""Minimal Blackwell demo: D = A @ B + C using tcgen05.copy.

This pedagogical single-CTA kernel keeps one 128x128 output tile and one K tile.
It deliberately uses ordinary global loads and is not a performance kernel.
"""

import torch
import triton.experimental.gluon as gluon
import triton.experimental.gluon.language as gl
from triton.experimental.gluon.language.nvidia.blackwell import (
    TensorMemoryLayout,
    allocate_tensor_memory,
    fence_async_shared,
    mbarrier,
    tcgen05_commit,
    tcgen05_copy,
    tcgen05_mma,
)


@gluon.jit
def accumulate_one_tile_kernel(a_ptr, b_ptr, c_ptr, d_ptr):
    BM: gl.constexpr = 128
    BN: gl.constexpr = 128
    BK: gl.constexpr = 64

    # 4 contiguous values/thread, a 4x8 lane tile, and four warps over M.
    reg_layout: gl.constexpr = gl.BlockedLayout(
        [1, 4], [4, 8], [4, 1], [1, 0]
    )
    rm = gl.arange(0, BM, gl.SliceLayout(1, reg_layout))
    rn = gl.arange(0, BN, gl.SliceLayout(0, reg_layout))
    rk_a = gl.arange(0, BK, gl.SliceLayout(0, reg_layout))
    rk_b = gl.arange(0, BK, gl.SliceLayout(1, reg_layout))

    a = gl.load(a_ptr + rm[:, None] * BK + rk_a[None, :])
    b = gl.load(b_ptr + rk_b[:, None] * BN + rn[None, :])
    c = gl.load(c_ptr + rm[:, None] * BN + rn[None, :])

    a_layout: gl.constexpr = gl.NVMMASharedLayout.get_default_for(
        [BM, BK], gl.float16
    )
    b_layout: gl.constexpr = gl.NVMMASharedLayout.get_default_for(
        [BK, BN], gl.float16
    )
    c_layout: gl.constexpr = gl.NVMMASharedLayout.get_default_for(
        [BM, BN], gl.float32
    )
    a_smem = gl.allocate_shared_memory(gl.float16, [BM, BK], a_layout)
    b_smem = gl.allocate_shared_memory(gl.float16, [BK, BN], b_layout)
    c_smem = gl.allocate_shared_memory(gl.float32, [BM, BN], c_layout)
    acc_tmem = allocate_tensor_memory(
        gl.float32, [BM, BN], TensorMemoryLayout([BM, BN], col_stride=1)
    )

    done = gl.allocate_shared_memory(
        gl.int64, [1], gl.constexpr(mbarrier.MBarrierLayout())
    )
    mbarrier.init(done, count=1)

    # Generic-proxy SMEM stores become visible to the async tensor pipe.
    a_smem.store(a)
    b_smem.store(b)
    c_smem.store(c)
    fence_async_shared()

    # One ordered tensor-pipe sequence: seed accumulator with C, then add A@B.
    tcgen05_copy(c_smem, acc_tmem)
    tcgen05_mma(a_smem, b_smem, acc_tmem, use_acc=True)
    tcgen05_commit(done)
    mbarrier.wait(done, phase=0)
    mbarrier.invalidate(done)

    # Only after the wait may ordinary threads read the completed accumulator.
    out = acc_tmem.load()
    out = gl.convert_layout(out, reg_layout)
    gl.store(d_ptr + rm[:, None] * BN + rn[None, :], out)


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("This demo requires a CUDA GPU with Blackwell support")
    if torch.cuda.get_device_capability()[0] < 10:
        raise RuntimeError("This demo requires NVIDIA compute capability 10.x or newer")

    torch.manual_seed(0)
    a = torch.randn((128, 64), device="cuda", dtype=torch.float16)
    b = torch.randn((64, 128), device="cuda", dtype=torch.float16)
    c = torch.randn((128, 128), device="cuda", dtype=torch.float32)
    d = torch.empty_like(c)

    accumulate_one_tile_kernel[(1,)](a, b, c, d, num_warps=4)
    reference = a.float() @ b.float() + c
    torch.testing.assert_close(d, reference, atol=5e-3, rtol=1e-2)
    max_abs_error = (d - reference).abs().max().item()
    print(f"D = A @ B + C: check passed; max_abs_error={max_abs_error:.6g}")


if __name__ == "__main__":
    main()
```

保存/运行与可观察结果：

```bash
cd /Users/saboxu/Downloads/learning_notes
python3 -m py_compile operators/code/triton/gluon/tutorials/10-tcgen05-copy-accumulate-demo.py
python3 operators/code/triton/gluon/tutorials/10-tcgen05-copy-accumulate-demo.py
```

第一条只检查 Python 语法；第二条要求 NVIDIA Blackwell（compute capability 10.x）及匹配版本的 Triton。成功时输出类似：

```text
D = A @ B + C: check passed; max_abs_error=<一个小的非负数>
```

### 逐段读代码：谁执行、触碰什么、由谁同步

1. **构造坐标与读 A/B/C。** 这个教学 kernel 的一个 CTA 负责整个 `128×128` 输出 tile；CTA 中的线程根据 `BlockedLayout` 分摊元素，普通 `gl.load` 把 global 数据读到寄存器。真实教程的 load partition 则由 TMA producer 搬 tile 到 SMEM。
2. **分配 SMEM 与 TMEM。** `a_smem`、`b_smem` 是 tcgen MMA 操作数；`c_smem` 是待复制的初始 accumulator；`acc_tmem` 是 `128×128` FP32 TMEM。C 的 shared layout 必须与 copy atom 匹配，不能拿任意普通 shared layout 代替。
3. **写 SMEM 后 fence。** `.store()` 由普通线程通过 generic proxy 写共享内存；`fence_async_shared()` 使之后 tensor pipe 通过 async proxy 发出的 copy/MMA 看见这些值。若数据来自 TMA，则 TMA completion barrier 的 wait 已是 producer-consumer 边界，不能机械地在每次 TMA 后照抄 generic-store fence。
4. **发出 copy，再发出 MMA。** 两条 `tcgen05_*` 指令排入同一 tensor pipe。MMA 消费这个 accumulator 时，硬件保证先前 copy 已完成；`use_acc=True` 保留 C 并向其加上乘积。
5. **commit/wait 后读 TMEM。** `tcgen05_commit(done)` 将已发出的 tensor-core 工作关联到 barrier；普通线程 `mbarrier.wait` 成功后才 `acc_tmem.load()`。接着把 TMEM 结果转换成适合合并 global store 的线程布局，再写 D。

### 官方 warp-specialized 版本如何把单 tile 扩展成流水

官方 `matmul_accumulate_kernel` 把同一条依赖拆给三个 partition，而不改变数学含义：

| Partition | 主要工作 | 等待/发信号 | buffer 所有权 |
|---|---|---|---|
| load | TMA 把 C tile 以及每个 K 片段的 A/B 放进 SMEM | 等 `c_empty` / `load_empty` 才写；TMA 完成后 `expect` 对应字节并 signal `c_ready` / `load_ready` | 只能写已经被消费者归还的 slot |
| MMA/compute（default partition） | `copy(C_smem, acc_tmem)`，逐段发 `mma(A,B,acc,use_acc=True)` | 等 `c_ready`、`load_ready`、`acc_empty`；copy 完成后归还 C；每个 MMA 完成后归还对应 A/B；整 tile 完成后 signal `acc_ready` | C 单 buffer、A/B 环形多 buffer、accumulator 双 buffer |
| epilogue | 等 accumulator ready，读 TMEM、转布局，用 `gl.store` 写 D | 读完并取回寄存器结果后归还 `acc_empty` | 归还后 compute 才能复用该 accumulator slot |

因此源码中的 `gl.warp_specialize([...], [1, 1], [24, 24])` 是一种角色切分：compute 是 default partition；另外两个 worker partition 分别做 epilogue 和 load，各 1 个 warp、各分配 24 个寄存器。这里列表顺序很关键：第一项 epilogue 是 default partition，接收未分配给 worker 的 warps；第二项 MMA 与第三项 load 是 worker，各分配 1 个 warp、各 24 个寄存器。它不是“自动把同一段代码复制到 3 组 warp”，而是让不同 partition 执行不同函数。C 只有一个 buffer，A/B 有 `num_buffers` 个槽，accumulator 有两个槽，分别用 barrier 管理各自的生命周期。

为什么 epilogue 用普通 `gl.store` 而不用 TMA store？C 本身是大块 FP32 数据。例：`BM=BN=128` 时，一个 C tile 是 `128×128×4 = 65,536` 字节，即 **64 KiB**。若另分同样大的 shared output staging buffer，会显著增加 CTA shared-memory 占用；TMA store 还会与 load 争用 TMA pipe。用 `gl.store` 省掉该 buffer/争用，代价是 TMEM→寄存器布局转换可能较贵——官方注释也指出它会用不少 shared memory，可通过 subtile epilogue 权衡。

### 数字执行轨迹：`M=N=256, K=192` 的一个输出 tile

取 `BM=BN=128, BK=64`，看 `(pid_m,pid_n)=(1,0)`：

| 计算项 | 代入 | 结果 / 物理含义 |
|---|---|---|
| C/D 行起点 | `off_m = 1×128` | 行 `128..255` |
| C/D 列起点 | `off_n = 0×128` | 列 `0..127` |
| K tile 数 | `192/64` | 3 次 MMA：`k=0,64,128` |
| 一个 A tile 字节数 | `128×64×2`（FP16） | 16 KiB |
| 一个 B tile 字节数 | `64×128×2`（FP16） | 16 KiB |
| C / accumulator tile 字节数 | `128×128×4`（FP32） | 64 KiB |

数学上，这个 tile 的累加历史是：

```text
初值:   acc = C[128:256, 0:128]
k=0:   acc = C_tile + A[128:256,   0:64] @ B[  0:64, 0:128]
k=64:  acc = 前值  + A[128:256,  64:128] @ B[ 64:128, 0:128]
k=128: acc = 前值  + A[128:256, 128:192] @ B[128:192, 0:128]
输出:   D[128:256, 0:128] = acc
```

每一行的 `A_tile @ B_tile` 形状都是 `(128×64) @ (64×128) → (128×128)`。第一次乘加之前先 copy 64 KiB 的 C 到 accumulator；三个 K 片段共享这个 accumulator，因而不产生三个独立的局部和，也不需要末尾再把 C 加一次。若 A/B 的 load ring 有 3 槽，则第 4 个 K 片段（若存在）回到 slot 0，但必须等新一代 barrier phase，而不是误把旧的 ready 信号当成新数据。

### 常见错误与症状

1. **第一次 MMA 把 `use_acc=False`。** 这表示不要把 TMEM 里的 C 当作累加初值；结果就像 `A@B`，C 被丢掉。形状都对，数值却整体少了 C。
2. **copy 后立刻普通线程 `acc_tmem.load()`，没有等 commit barrier。** copy 是异步的；可能随机读到旧 C 或半完成 tile。tensor pipe 内 copy→MMA 的顺序不等于普通线程已能读取。
3. **generic `.store()` 后省略 fence。** async proxy 可能观察到旧 SMEM 内容，表现为偶发数据错误；这与 barrier phase 错误不同，增加 wait 不能代替 proxy fence。
4. **TMA load 未完成就启动 MMA，或 MMA 未完成就覆盖 A/B slot。** 前者读到不完整 tile；后者更常是间歇性损坏，且可能只在多 buffer/高并发时出现。ready 和 empty 是 producer/consumer 的两方向信号。
5. **把每个 barrier 都用 phase 0，或让一方独自翻 phase。** 第一次迭代正常，barrier 复用后卡住或误判上一轮完成。phase 是每个 barrier 循环世代的一部分，producer 与 consumer 必须遵守同一协议。
6. **C copy 用错布局或 dtype。** 可能编译时报 unsupported layout/shape，也可能没有对应 copy atom。先核对 2D、FP32、swizzle、非转置和 `blockM=128`，不要把限制误套到 scales 等其它 TMEM layout。

### 自我检查

1. **为什么 `tcgen05_copy(C_smem, acc_tmem)` 后可以立刻发 MMA，却不能立刻用普通线程读取 `acc_tmem`？**
   - 答：copy 与 MMA 都进入按序执行的 tensor pipe；普通线程不在这个隐式序列里，必须等 commit 对应的 mbarrier。
2. **`D=A@B+C` 的第一次 MMA 为什么要 `use_acc=True`？**
   - 答：TMEM 在 MMA 前已由 C 初始化，第一次 MMA 必须读取并保留它，再加上第一个 K 片段的乘积。
3. **什么时候需要 `fence_async_shared()`？TMA 路径总要加吗？**
   - 答：普通线程 generic proxy 写 SMEM 后、async proxy 读取前需要；TMA 写入由 TMA async proxy 完成并通过它的 barrier wait 建立可用边界，不能不分路径地机械添加。
4. **为什么官方 accumulator 要双缓冲？**
   - 答：epilogue 正在读取/写回上一个输出 tile 时，compute 可在另一个 TMEM accumulator 上做下一 tile；empty/ready barrier 管理两者的交接。
5. **为什么官方 epilogue 选择 `gl.store` 而非 TMA store？**
   - 答：避免为 64 KiB FP32 tile 另占 shared staging buffer，并降低 TMA pipe 与输入 load 的争用；代价是 TMEM→线程布局转换可能较贵。

## Block-scaled MMA：硬件内反量化

### 量化的块结构

block scaling 把 K 方向切成固定长度 `VEC_SIZE`，每个块只有一个 scale：

```text
X[m, k] ~= Q[m, k] * S[m, k // VEC_SIZE]
```

对矩阵：

```text
C = (A * A_scale) @ (B * B_scale)
```

形状约定为：

```text
A.shape       = (M, K)
B.shape       = (N, K)             # 注意这里是转置后的 B
A_scale.shape = (M, K // VEC_SIZE)
B_scale.shape = (N, K // VEC_SIZE)
```

硬件把每 `VEC_SIZE` 个低精度元素与同一个 scale 解释在一起，不需要在寄存器中显式生成完整反量化矩阵。

### 支持的格式

| 格式 | `VEC_SIZE` | scale dtype | 元素表示 | 关键限制 |
|---|---:|---|---|---|
| nvfp4 | 16 | `float8_e4m3fn` | 2 个 fp4 打包进一个 `uint8` | A/B 必须同为 nvfp4 |
| mxfp4 | 32 | `fp8e8m0` | 2 个 fp4 打包进一个 `uint8` | B 走转置路径 |
| mxfp8 | 32 | `fp8e8m0` | 一个 `float8_e4m3fn` 元素 | B 走转置路径 |
| mxfp8 × mxfp4 | 32 | `fp8e8m0` | mixed precision | fp4 侧 padded；`BLOCK_K >= 128` |

MX 的 `e8m0` 是只有 exponent、没有 mantissa 的 scale，表示的数值是 2 的整数次幂。`VEC_SIZE=32` 意味着每 32 个 K 元素共享一个二进制指数。

### fp4 的坐标轨迹

设 `K=128`、`VEC_SIZE=32`，逻辑元素按 K 方向每两个打包一个 `uint8`：

```text
logical k:   0   1   2   3  ... 13  14  15
byte index:  0   0   1   1  ...  6   7   7
scale index: 0                 ...       0
```

再取 `k=64` 的逻辑位置：

```text
global load offset for fp4 = k // 2 = 32
scale offset              = k // VEC_SIZE = 2
```

这就是 `off_k_a = k // A_ELEM_PER_BYTE` 和 `k // VEC_SIZE` 两个计算分别存在的原因。

### 完整 `simple_mma_scaled_kernel` 核心路径

下面保留完整的搬移、scale 装载、TMEM 写入和 scaled MMA；省略的只是外层测试参数生成：

```python
@gluon.jit
def simple_mma_scaled_kernel(a_desc, b_desc, c_desc,
                             a_scale_ptr, a_scale_stride_m, a_scale_stride_k,
                             b_scale_ptr, b_scale_stride_n, b_scale_stride_k,
                             VEC_SIZE: gl.constexpr):
    A_IS_FP4: gl.constexpr = a_desc.dtype == gl.uint8
    B_IS_FP4: gl.constexpr = b_desc.dtype == gl.uint8
    A_ELEM_PER_BYTE: gl.constexpr = 2 if A_IS_FP4 else 1
    B_ELEM_PER_BYTE: gl.constexpr = 2 if B_IS_FP4 else 1

    BLOCK_M: gl.constexpr = c_desc.block_type.shape[0]
    BLOCK_N: gl.constexpr = c_desc.block_type.shape[1]
    BLOCK_K: gl.constexpr = a_desc.block_type.shape[1] * A_ELEM_PER_BYTE
    K = a_desc.shape[1] * A_ELEM_PER_BYTE

    a_smem = gl.allocate_shared_memory(
        a_desc.dtype, a_desc.block_type.shape, a_desc.layout
    )
    b_smem = gl.allocate_shared_memory(
        b_desc.dtype, b_desc.block_type.shape, b_desc.layout
    )

    scale_layout: gl.constexpr = TensorMemoryScalesLayout()
    a_scale_tmem = allocate_tensor_memory(
        a_scale_ptr.dtype.element_ty,
        [BLOCK_M, BLOCK_K // VEC_SIZE],
        scale_layout,
    )
    b_scale_tmem = allocate_tensor_memory(
        b_scale_ptr.dtype.element_ty,
        [BLOCK_N, BLOCK_K // VEC_SIZE],
        scale_layout,
    )

    tmem_layout: gl.constexpr = TensorMemoryLayout(
        [BLOCK_M, BLOCK_N], col_stride=1
    )
    acc_tmem = allocate_tensor_memory(
        gl.float32, [BLOCK_M, BLOCK_N], tmem_layout
    )
    use_acc = False

    bar = gl.allocate_shared_memory(
        gl.int64, [1], mbarrier.MBarrierLayout()
    )
    mma_bar = gl.allocate_shared_memory(
        gl.int64, [1], mbarrier.MBarrierLayout()
    )
    mbarrier.init(bar, count=1)
    mbarrier.init(mma_bar, count=1)
    phase = 0

    pid_m = gl.program_id(0)
    pid_n = gl.program_id(1)
    off_m = pid_m * BLOCK_M
    off_n = pid_n * BLOCK_N

    for k in range(0, K, BLOCK_K):
        off_k_a = k // A_ELEM_PER_BYTE
        off_k_b = k // B_ELEM_PER_BYTE

        mbarrier.expect(
            bar, a_desc.block_type.nbytes + b_desc.block_type.nbytes
        )
        tma.async_load(a_desc, [off_m, off_k_a], bar, a_smem)
        tma.async_load(b_desc, [off_n, off_k_b], bar, b_smem)
        mbarrier.wait(bar, phase)

        coalesced_2d_layout: gl.constexpr = gl.BlockedLayout(
            [1, 1], [1, 32], [1, gl.num_warps()], [1, 0]
        )
        a_scale_offs_m = off_m + gl.arange(
            0, BLOCK_M, layout=gl.SliceLayout(1, coalesced_2d_layout)
        )
        a_scale_offs_k = k // VEC_SIZE + gl.arange(
            0, BLOCK_K // VEC_SIZE,
            layout=gl.SliceLayout(0, coalesced_2d_layout),
        )
        a_scale = gl.load(
            a_scale_ptr
            + a_scale_offs_m[:, None] * a_scale_stride_m
            + a_scale_offs_k[None, :] * a_scale_stride_k
        )

        b_scale_offs_n = off_n + gl.arange(
            0, BLOCK_N, layout=gl.SliceLayout(1, coalesced_2d_layout)
        )
        b_scale_offs_k = k // VEC_SIZE + gl.arange(
            0, BLOCK_K // VEC_SIZE,
            layout=gl.SliceLayout(0, coalesced_2d_layout),
        )
        b_scale = gl.load(
            b_scale_ptr
            + b_scale_offs_n[:, None] * b_scale_stride_n
            + b_scale_offs_k[None, :] * b_scale_stride_k
        )

        a_scale_layout: gl.constexpr = a_scale_tmem.get_reg_layout()
        b_scale_layout: gl.constexpr = b_scale_tmem.get_reg_layout()
        a_scale = gl.convert_layout(a_scale, a_scale_layout)
        b_scale = gl.convert_layout(b_scale, b_scale_layout)
        a_scale_tmem.store(a_scale)
        b_scale_tmem.store(b_scale)

        a_format: gl.constexpr = "e2m1" if A_IS_FP4 else "e4m3"
        b_format: gl.constexpr = "e2m1" if B_IS_FP4 else "e4m3"
        tcgen05_mma_scaled(
            a_smem,
            b_smem.permute((1, 0)),
            acc_tmem,
            a_scale_tmem,
            b_scale_tmem,
            a_format,
            b_format,
            use_acc=use_acc,
        )
        tcgen05_commit(mma_bar)
        mbarrier.wait(mma_bar, phase)
        use_acc = True
        phase ^= 1

    mbarrier.invalidate(bar)
    mbarrier.invalidate(mma_bar)

    acc = acc_tmem.load()
    acc = acc.to(c_desc.dtype)
    acc_smem = gl.allocate_shared_memory(
        c_desc.dtype, c_desc.block_type.shape, c_desc.layout
    )
    acc_smem.store(acc)
    fence_async_shared()
    tma.async_store(c_desc, [off_m, off_n], acc_smem)
    tma.store_wait(0)
```

### 为什么要改 scale 的全局布局

最直观的 scale 排布是：

```text
[MN, K // VEC_SIZE]
```

但一个 program 一次只使用一个 `[BLOCK_MN, BLOCK_K // VEC_SIZE]` scale 子块，列方向跨 K 时地址并不连续，TMA/普通 load 的 sector 利用率很差。官方 profile 显示每个 thread 只利用 sector 中很少的字节，benchmark 只有约 `33.41 TFLOPS`（mxfp8）和 `70.84 TFLOPS`（nvfp4）。

改成按 program block 聚簇的布局：

```text
[MN/BLOCK_MN, BLOCK_MN, SCALE_K/SCALE_BLOCK_K, SCALE_BLOCK_K]
permute(0, 2, 1, 3)
contiguous()
```

等价于把某个 program 的 `[BLOCK_MN, BLOCK_K // VEC_SIZE]` scale block 放在连续内存中。这样 scale load 才能合并，随后再通过寄存器或 `tcgen05_copy` 送入 TMEM。

### 性能轨迹说明

官方教程的 scale 布局优化与 pipeline 版本分别报告了很高的吞吐：

| 版本 | mxfp8 | mxfp4 | mxfp8 × mxfp4 | nvfp4 |
|---|---:|---:|---:|---:|
| contiguous scale | `900.97` | `2081.76` | `1000.48` | `2002.05` TFLOPS |
| contiguous + `tcgen05_copy` | `929.07` | `2147.76` | `1035.60` | `2092.39` TFLOPS |
| pipelined / warp-specialized | `2018.58` | `3916.62` | `2144.05` | `3842.19` TFLOPS |

这些数字的顺序不是“换一条指令就从 33 变成 2000”。第一跳来自 scale 访存和布局优化，第二跳来自多 buffer、异步 tensor pipe 和 warp specialization。所有数字都是上游指定 GPU 上的结果，不是本机实测。

### 常见错误

| 错误 | 症状 | 修复 |
|---|---|---|
| scale 放在寄存器或 SMEM 直接传给 `tcgen05_mma_scaled` | 编译期类型/布局错误 | 分配 `TensorMemoryScalesLayout`，先写 TMEM |
| B 不转置 | API 拒绝或结果形状错误 | 构造 `[BLOCK_N, BLOCK_K]`，调用 `b_smem.permute((1, 0))` |
| fp4 的 K 偏移没除以 2 | TMA 读到错误地址，结果整体错位 | 用 `A_ELEM_PER_BYTE`/`B_ELEM_PER_BYTE` 换算 |
| mixed precision 的 `BLOCK_K < 128` | 描述符 swizzle 无匹配 atom | 提高 `BLOCK_K`，fp4 padded 路径需要足够 interior extent |
| scale 沿用 `[MN, K/VEC]` | 吞吐异常低，profile 报 sector utilization 差 | 按 `[MN/BLOCK_MN, SCALE_K/SCALE_BLOCK_K, BLOCK_MN, SCALE_BLOCK_K]` 重排 |
| 第一轮 `use_acc=True` | 结果带旧 TMEM 值 | 第一轮 `False`，后续 `True` |

## CLC：把“下一个未启动 cluster”交给空闲 CTA

### 静态 persistent 的尾部问题

普通 persistent matmul 的 tile 分配是：

```text
grid = SM 数
program i -> tile i, i + num_programs, i + 2 * num_programs, ...
```

如果每个 tile 完全等成本，这很稳定；真实 workload 可能有：

```text
某些 tile 的 K 更长；
某些 tile 的 A/B 已经在 L2；
某些 tile 的 TMA 遇到 cache miss；
某些 tile 的 epilogue 更慢。
```

最后会剩下一条最慢的 program 拖住 kernel。CLC 允许一个早完成的 CTA 取消一个尚未启动的 cluster，并接管它的 work。

### API 和隐藏 latency 的位置

关键 API：

```python
clc.try_cancel(result_buffer, barrier)
clc_result = clc.load_result(result_buffer)
has_work = clc_result.is_canceled()
tile_id = clc_result.program_id(0)
```

教程的技巧是把 `try_cancel` 放在当前 tile 的 TMA prologue 附近，完成这个 tile 后再读取结果：

```text
issue CLC request
  -> 当前 tile 的 load / MMA / epilogue
  -> wait CLC result
  -> 成功则接管返回的 tile id
```

这样 CLC 的往返延迟被当前 tile 的计算盖住。如果等 tile 完成才发请求，CLC 延迟就会直接暴露在主循环尾部。

### 完整 scheduler

```python
@gluon.aggregate
class ClcTileScheduler:
    has_work: gl.tensor
    tile_id: gl.tensor
    clc_result_buf: gl.shared_memory_descriptor
    barrier: gl.shared_memory_descriptor
    phase: gl.tensor

    @gluon.jit
    def initialize(M, N, BLOCK_M, BLOCK_N):
        has_work = gl.to_tensor(True)
        starting_tile_id = gl.program_id(0)
        barrier = mbarrier.allocate_mbarrier()
        result = gl.allocate_shared_memory(
            gl.int64, [2], gl.SwizzledSharedLayout(1, 1, 1, [0])
        )
        mbarrier.init(barrier, count=1)
        phase = gl.to_tensor(0)
        return ClcTileScheduler(
            has_work, starting_tile_id, result, barrier, phase
        )

    @gluon.jit
    def try_cancel(self) -> None:
        clc.try_cancel(self.clc_result_buf, self.barrier)
        mbarrier.expect(self.barrier, 16)

    @gluon.jit
    def advance(self):
        mbarrier.wait(self.barrier, self.phase)
        clc_res = clc.load_result(self.clc_result_buf)
        return ClcTileScheduler(
            clc_res.is_canceled(),
            clc_res.program_id(0),
            self.clc_result_buf,
            self.barrier,
            self.phase ^ 1,
        )
```

静态 scheduler 也实现相同的 `has_work / try_cancel / advance` 接口，因此主 kernel 不需要为两种策略复制两套控制流：

```python
@gluon.aggregate
class StaticTileScheduler:
    has_work: gl.tensor
    tile_id: gl.tensor
    num_tiles: gl.tensor

    @gluon.jit
    def initialize(M, N, BLOCK_M, BLOCK_N):
        tile_id = gl.program_id(0)
        num_tiles = gl.cdiv(M, BLOCK_M) * gl.cdiv(N, BLOCK_N)
        has_work = tile_id < num_tiles
        return StaticTileScheduler(has_work, tile_id, num_tiles)

    @gluon.jit
    def try_cancel(self) -> None:
        pass

    @gluon.jit
    def advance(self):
        next_tile_id = self.tile_id + gl.num_programs(0)
        has_work = next_tile_id < self.num_tiles
        return StaticTileScheduler(has_work, next_tile_id, self.num_tiles)
```

### grid 的选择

```python
if use_clc:
    num_pid_m = triton.cdiv(M, BLOCK_M)
    num_pid_n = triton.cdiv(N, BLOCK_N)
    grid = num_pid_m * num_pid_n
    SchedulerImpl = ClcTileScheduler
else:
    grid = torch.cuda.get_device_properties(A.device).multi_processor_count
    SchedulerImpl = StaticTileScheduler
```

CLC 版本启动全部 tile 的 cluster；一部分 cluster 已启动，一部分等待硬件调度。早完成的 CTA 通过 `try_cancel` 取消尚未启动的 cluster，然后处理它的 tile id。静态版本只启动 SM 数量个 program，每个 program 顺序处理一批 tile。

这里要区分“取消 cluster”和“取消正在执行的 CTA”：CLC 只能取消尚未启动的 cluster，已经运行的 work 不会被抢走。

### 性能轨迹

官方 `8192 x 8192 x 8192`、fp16、Blackwell 样本：

| 调度 | 吞吐 |
|---|---:|
| static | `1040.13 TFLOPS` |
| CLC | `1080.74 TFLOPS`，约 `103.9%` |

这不是让 MMA 本身更快，而是让每条 SM 的任务长度更接近。对于 K 固定且完全均匀的 GEMM，收益可能很小；对 ragged matmul、MoE 或多阶段 pipeline，动态分配的收益更明显。

### 常见错误

| 错误 | 症状 | 修复 |
|---|---|---|
| CLC 完成后才发起 `try_cancel` | 没有加速，甚至更慢 | 在 prologue 发起，在 tile 末尾读取 |
| 把 `is_canceled()` 当成“当前 CTA 是否还需要工作” | 可能过早退出或错误接管 | 只有成功取消时才使用返回 tile id；失败时按调度协议结束或继续 |
| 未等待 CLC mbarrier 就读 result | 读到旧 result buffer | `mbarrier.wait` 后再 `clc.load_result` |
| CLC grid 只启动 SM 数 | 没有未启动 cluster 可接管 | CLC 使用总 tile 数启动 |
| phase 不翻转 | 第二轮等待旧 parity 或挂起 | 每次 round `phase ^= 1` |

## TMA im2col：不物化矩阵的卷积

### 显式 im2col 的内存浪费

对 `R x S` 卷积，输入空间位置会被最多 `R*S` 个输出窗口重复使用。显式 im2col 会为每个输出位置复制一个 patch：

```text
input NHWC  [N, H, W, Ci]
                     |
             materialize A
                     v
A [N*out_h*out_w, R*S*Ci]   # 有大量重复元素
```

隐式 GEMM 把 A 的构造留在 TMA 地址生成里：

```text
for r in R:
  for s in S:
    for ci_block in Ci:
      TMA im2col load -> A_smem [BLOCK_M, BLOCK_K]
      TMA load weight  -> B_smem [BLOCK_N, BLOCK_K]
      acc += A_smem @ B_smem^T
```

### GEMM 维度映射

```text
M_gemm = N * out_h * out_w    # 输出空间位置数
N_gemm = Co                    # 输出通道数
K_gemm = R * S * Ci            # reduction 维度
```

每个 `(r, s, ci_block)` 迭代加载 A 的一列块和 B 的一行块：

```text
A[m, k] = input[batch, out_y*stride + r - pad,
                  out_x*stride + s - pad,
                  ci_block*BLOCK_K + k_within]
B[n, k] = weight[co, r, s, ci]
```

### descriptor 的五个关键参数

```python
in_desc = TensorDescriptorIm2Col.from_tensor(
    input_nhwc,
    block_shape=[BLOCK_M, BLOCK_K],
    layout=input_layout,
    padding="zero",
    element_strides=[1, stride, stride, 1],
    pixel_box_lower_corner=[-padding, -padding],
    pixel_box_upper_corner=[upper_h, upper_w],
)
```

含义：

| 参数 | 作用 | 常见错误 |
|---|---|---|
| `block_shape=[BLOCK_M, BLOCK_K]` | 一次搬 M 个输出位置、K 个通道元素 | 把它当成图像 H/W tile |
| `element_strides` | 从当前像素走到下一个输出位置的空间步长 | 只填一个 stride，NHWC 需要 `[1, h_stride, w_stride, 1]` |
| `padding="zero"` | 越界读取补零 | 忘记它会让卷积边界错误 |
| `pixel_box_lower_corner` | TMA 可访问窗口下界 | 负 padding 与 coordinate 符号混用 |
| `pixel_box_upper_corner` | TMA 可访问窗口上界 | 没按输出大小和有效输入范围计算 |

### 边界公式

给定 TMA load 的 offsets `[off_h, off_w]`，访问窗口是：

```text
lower_h = pixel_box_lower_corner[0] + off_h
upper_h = (H - 1) + pixel_box_upper_corner[0] + off_h

lower_w = pixel_box_lower_corner[1] + off_w
upper_w = (W - 1) + pixel_box_upper_corner[1] + off_w
```

教程中的 `4 x 4` 例子 `pixel_box_lower_corner=[-1,-1]`、`pixel_box_upper_corner=[-1,-1]`、`offsets=[0,0]` 得到：

```text
H in [-1, 2]
W in [-1, 2]
```

这与图像坐标的八邻域窗口不是一回事；它只是描述 TMA 可以解释的 padding/裁剪窗口。加载时，窗口外的值是 0，窗口内但原 tensor 外的位置也由 padding 语义补 0。

### 4 x 4 数值轨迹

输入使用 `1..16` 填充：

```text
 1  2  3  4
 5  6  7  8
 9 10 11 12
13 14 15 16
```

当 `coord=[0,0,0,0]`、`offsets=[0,0]` 时，TMA 按 `[BLOCK_M=16, BLOCK_K=32]` 读取，每行第一个通道是：

```text
[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
```

当窗口是 `H in [-1,2]`、`W in [-1,2]` 时，读取到的 `BLOCK_M=16` 行值是：

```text
[0, 0, 0, 0,
 0, 1, 2, 3,
 0, 5, 6, 7,
 0, 9,10,11]
```

如果把 offsets 改成 `[1,1]`，窗口整体右移和下移：

```text
H in [0,3]
W in [0,3]
```

这时重新得到原始 `1..16`，因为 access window 已从 padded 区域平移到原 tensor。

### 完整 TMA im2col micro-kernel

下面是一个最小可读版本，完整版把 MMA 和 epilogue 接在后面：

```python
@gluon.jit
def tma_im2col_kernel(in_desc, out_desc,
                      coord_n: int, coord_h: int, coord_w: int, coord_c: int,
                      offset_h: gl.constexpr, offset_w: gl.constexpr):
    smem = gl.allocate_shared_memory(
        in_desc.dtype, in_desc.block_shape, in_desc.layout
    )
    bar = mbarrier.allocate_mbarrier()
    mbarrier.init(bar, count=1)
    mbarrier.expect(bar, in_desc.block_type.nbytes)

    tma.async_load_im2col(
        in_desc,
        [coord_n, coord_h, coord_w, coord_c],
        [offset_h, offset_w],
        bar,
        smem,
    )
    mbarrier.wait(bar, phase=0)
    mbarrier.invalidate(bar)

    tma.async_store(out_desc, [0, 0], smem)
    tma.store_wait(pendings=0)
```

host 侧的 descriptor 构造如下：

```python
def run_tma_im2col(pixel_box_lower_corner,
                   pixel_box_upper_corner,
                   coord, offsets):
    inp = torch.arange(1, 17, device="cuda", dtype=torch.float32)
    inp = inp.unsqueeze(1).repeat(1, 32).reshape(1, 4, 4, 32)
    out = torch.zeros(16, 32, device="cuda", dtype=torch.float32)

    block_shape = [16, 32]
    layout = gl.NVMMASharedLayout(
        swizzle_byte_width=128, element_bitwidth=32, rank=2
    )
    in_desc = TensorDescriptorIm2Col.from_tensor(
        inp,
        block_shape,
        layout,
        padding="zero",
        element_strides=[1, 1, 1, 1],
        pixel_box_lower_corner=pixel_box_lower_corner,
        pixel_box_upper_corner=pixel_box_upper_corner,
    )
    out_desc = TensorDescriptor.from_tensor(out, block_shape, layout)
    tma_im2col_kernel[(1,)](in_desc, out_desc, *coord, *offsets, num_warps=1)
    return inp, out
```

### 卷积端到端形状与验证

官方完整 convolution 的 shapes 是：

```text
input:  [N, H, W, Ci]       = [4, 16, 16, 64] 或 [4,16,16,96]
weight: [Co, R, S, Ci]      = [64, 3, 3, 64] 或 [96,3,3,96]
output: [N, out_h, out_w, Co]
```

测试参数覆盖：

```text
N in {1, 4}
R,S in {(3,3), (1,1)}
stride in {1, 2}
padding in {0, 1}
```

核心 K-loop 的坐标是：

```python
for k_iter in range(R * S * ci_num_blocks):
    ci_block, rs_idx = k_iter % ci_num_blocks, k_iter // ci_num_blocks
    r, s = rs_idx // S, rs_idx % S

    tma.async_load_im2col(
        in_desc,
        [batch_id,
         out_y * stride_h - pad_h,
         out_x * stride_w - pad_w,
         ci_block * BLOCK_K],
        [r.to(tl.int16), s.to(tl.int16)],
        tma_bar,
        a_smem,
    )
    k_offset = r * S * Ci + s * Ci + ci_block * BLOCK_K
    tma.async_load(
        weight_desc,
        [pid_n * BLOCK_N, k_offset],
        tma_bar,
        b_smem,
    )
    mbarrier.wait(tma_bar, phase=phase)
    mma = mma.issue_async_mma(a_smem, b_smem.permute((1, 0)))
    phase ^= 1
```

官方测试使用 `torch.nn.functional.conv2d` 作为 reference，误差阈值是 `atol=1e-2, rtol=1e-2`，可观察结果是 `max error` 在该量级内；教程中部分小例子报告 `0.000000`。

### 常见错误

| 错误 | 症状 | 修复 |
|---|---|---|
| `coord` 和 `offsets` 的符号弄反 | 读到的图像窗口偏移一个 stride | 把 coordinate 当连续起点，offset 当离散 filter 位移 |
| `pixel_box` 没覆盖完整输出范围 | 边界 tile 结果错误 | 按 `window_h = (out_h-1)*stride+1` 计算 |
| Ci 不是 `BLOCK_K` 倍数 | 最后一个 K block 读出越界数据 | 用 `tma.cdiv` 循环并依赖 TMA zero pad |
| weight K 偏移按 `BLOCK_K` 而不是 Ci 计算 | 权重错位 | 使用 `k_offset = (r*S+s)*Ci + ci_block*BLOCK_K` |
| 把 B 按 `[BLOCK_K, BLOCK_N]` 直接喂给转置接口 | 形状错误或 MMA 语义不对 | 统一使用 `b_smem.permute((1, 0))` 的约定 |

## 本节的数据流总览

```text
              +---------------------- TMA gather/scatter ----------------------+
              |                                                                |
HBM ---> TMA block load ---> SMEM ---> tcgen05_copy ---> TMEM ---> MMA ---> TMEM
              |                 |                                      |
              |                 +---- TMA im2col ---> A_smem -----------+
              |                                                   |
              +---------------------- TMA multicast --------------+
                                                                  |
CLC scheduler -> tile_id ----------------------------------------> epilogue TMA store
```

把这条链拆开后，后面的 CGA/2CTA 就只是把“一个 CTA 内的数据流”升级成“一个 cluster 内的数据流”：TMA 可以 multicast，barrier 可以跨 CTA，TMEM 可以被两个 CTA 共享，MMA 可以形成 2CTA 的外积。

## 硬件与验证边界

- `09`、`10`、`11`、`12`、`13` 中的 runtime 检查分别要求 Blackwell 或 Hopper 以上；本机没有 CUDA GPU 和对应 Triton runtime，因此这里只做静态源码核对。
- `pytest` 的完全执行需要 `triton.experimental.gluon`、CUDA driver、目标 GPU 和对应 Triton revision；当前环境无法复现 benchmark。
- 本节的 benchmark 数字均来自官方教程的指定硬件结果，只能用作机制对比，不应解释成本机性能承诺。
- 静态检查命令：

```bash
cd /Users/saboxu/Downloads/learning_notes
python3 -m py_compile operators/code/triton/gluon/tutorials/*.py
git diff --check
```

若要在有 GPU 的机器上运行：

```bash
pytest -q operators/code/triton/gluon/tutorials/09-tma-gather-scatter.py
pytest -q operators/code/triton/gluon/tutorials/10-tcgen05-copy.py
pytest -q operators/code/triton/gluon/tutorials/11-tcgen05-mma-scaled.py
pytest -q operators/code/triton/gluon/tutorials/12-cluster-launch-control.py
pytest -q operators/code/triton/gluon/tutorials/13-conv-im2col.py
```

## 自我检查

1. **为什么 TMA gather 的 descriptor block 第 0 维必须是 1，而 offsets 可以有 8 行以上？**
   - 答：descriptor 只描述一行内的连续列块 `[1, BLOCK_Y]`；行号不在 descriptor 的矩形里，而由一维 `x_offsets` 提供。至少 8 行是为了形成有效的 gather4/scatter4 warp 访问。
2. **`tcgen05_copy` 后面紧接 `tcgen05_mma`，为什么可以不中间等一次 barrier？**
   - 答：两者都进入 tensor pipe 并按发射顺序执行；barrier 用于通知普通线程何时可以安全读取 TMEM 或复用 SMEM，而不是限制 tensor pipe 自身顺序。
3. **block-scaled MMA 中 `A_scale.shape=(M,K/VEC_SIZE)` 的 scale 是怎么应用的？**
   - 答：scale 沿 K 方向每个 `VEC_SIZE` 元素广播一次，硬件在 MMA 前把该块乘到 A 的对应元素块上，不需要物化完整反量化矩阵。
4. **CLC 为什么要把 `try_cancel` 提前到 prologue？**
   - 答：CLC 请求本身有延迟。提前发起后，当前 tile 的 TMA/MMA/epilogue 可以与请求重叠；完成 tile 后读取结果，隐藏请求延迟。
5. **TMA im2col 的 `coord` 和 `offsets` 分别表示什么？**
   - 答：`coord` 是 NHWC 上连续起点的坐标，`offsets` 是 `[r,s]` 形式的 filter 位移；两者相加后决定实际的 TMA 访问地址。
6. **为什么 scale 的全局布局会影响 scaled MMA 的吞吐？**
   - 答：scale block 必须被 program 顺序、合并地读入并写入 TMEM。若原始 `[MN,K/VEC]` 的列方向步长导致分散访问，scale load 会成为主瓶颈；按 program block 聚簇后，多个 scale 元素才能连续进入一个 cache sector。

## 本章学习进度

- [x] TMA gather/scatter 单独搬运：offsets layout、descriptor 约束与边界行为
- [x] 融合 gather → tcgen05 GEMM → scatter：persistent tile 与多缓冲流水
- [x] `tcgen05_copy`：SMEM → TMEM 的异步搬运、累加式 GEMM 与同步契约

### 下一知识点

`tcgen05_mma_scaled`。在已掌握 C accumulator 的 SMEM→TMEM 初始化后，继续看低精度数据块如何通过 TMEM scales 参与 MMA，而不显式物化反量化矩阵。
