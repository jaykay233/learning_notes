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
