# Triton Gluon 14：CGA、multicast 与 2CTA Capstone

## 本次讲解位置

章节：`chapter_gluon_multicta_capstone`（算子开发 / Triton Gluon）<br>
小节：CGA、TMA multicast、cluster barrier 与 2CTA tcgen05 capstone<br>
知识点：`cga_layout` 的分片/广播语义、跨 CTA `mbarrier`、`two_ctas` 与四角色 GEMM<br>
上次：gather/scatter、TMEM copy、scaled MMA、CLC 与 TMA im2col<br>
下次：把单例 kernel 参数化，接入 autotune，并用真实模型 shape 做端到端 profile<br>
PTX：cluster barrier、TMA multicast、`tcgen05.mma.cta_group::2`

## 为什么现在讲这个

前面 01 到 13 的所有优化都还局限在一个 CTA 内部。单 CTA 的做法有三个硬上限：

1. 一个 CTA 的 shared memory 装不下过大的 A/B tile；
2. 一个 CTA 的 TMEM 和 tensor core 资源有限，M=256 这类大 tile 无法由一组 `tcgen05.mma` 独吞；
3. 两个相邻 CTA 常常需要同一份 A 或 B，普通 load 会重复走 HBM/L2，再各自写入自己的 SMEM。

CGA（Cooperative Grid Array，也常直接叫 cluster）把这几个上限变成协作机会：

```text
单 CTA 数据流：
  global -> TMA -> SMEM -> tcgen05 -> TMEM -> epilogue

CGA 数据流：
  global -> TMA multicast -> 多个 CTA 的 SMEM
                              |
                              +-> DSMEM/cross-CTA barrier
                              |
                              +-> 2CTA tcgen05 -> 两块 TMEM 组成一个更大 accumulator
```

这一步不是为了“多启动几个 CTA”。真正的目标是：让同一 cluster 内的 CTA 共享搬运、共享同一个更大的矩阵乘结果，同时不给每一次数据交换都付 full-cluster barrier 的代价。学完本节，你应该能判断一个布局应该在 CTA 维度上分片还是广播、何时需要 `two_ctas=True`、以及为什么 `warp_specialize` 里不能随意放跨 CTA 的 `reduce`。

## CGA 的基本约束

CGA 是 Hopper 引入的线程组层级：

```text
thread -> warp -> CTA -> CGA(cluster) -> grid
```

| 属性 | 规则 |
|---|---|
| 引入架构 | Hopper 起步，更完整能力在 Blackwell `SM100+` |
| cluster 大小 | `num_ctas` 必须是 2 的幂，且 `1 <= num_ctas <= 16` |
| CTA 所在 SM | 不保证 cluster 内 CTA 落在同一个 SM |
| 共享方式 | TMA multicast、distributed shared memory、跨 CTA mbarrier |
| 代价 | 跨 CTA 同步和访问远慢于 CTA 内 `bar.sync` 或 SMEM 访问 |
| 设计原则 | 最大化一次协作带来的复用，同时减少不必要的 cluster barrier |

最后一条很重要。CGA 不是把 SMEM 变成一块免费的大 SMEM。假设两个 CTA 每次 K-loop 都为同一个 A tile 做一次 full-cluster barrier，barrier 成本可能超过省下来的 HBM 流量。TMA multicast 和 2CTA MMA 的价值在于：它们把协作放进硬件数据通路，而不是让所有线程反复手工同步。

## `cga_layout`：把 CTA 维度写进线性布局

### 心智模型

一个逻辑 tensor 原本只回答“元素坐标是什么”。加入 CGA 后，还要回答“这个元素由哪个 CTA 持有”。`cga_layout` 就是这层 CTA 维度映射。

它表示为若干个 basis 向量。每个 basis 对应 CTA ID 的一个二进制位：

```text
CTA bit 0 -> basis[0]
CTA bit 1 -> basis[1]
CTA bit n -> basis[n]
```

basis 中的 0 表示这个 CTA 位不改变该 tensor 维度，也就是做广播；非 0 表示沿该维度切成连续 chunk。

### 两个 CTA 沿 M 分片

```python
cga_layout: gl.constexpr = ((1, 0),)
```

这里只有一个 CTA bit，因此通常配 `num_ctas=2`：

| CTA ID | `cga_layout` 贡献 | 含义 |
|---:|---|---|
| 0 | `0 * (1,0)` | 第 0 个 M chunk |
| 1 | `1 * (1,0)` | 第 1 个 M chunk |

在 M 方向上，两个 CTA 各拿一半连续行；N 方向不分片。

### 8 个 CTA 沿 M 和 N 联合分片

```python
cga_layout = ((1, 0), (2, 0), (0, 1))
```

三个 basis 对应三个 CTA bit：

| CTA bit | basis | 作用 |
|---:|---|---|
| bit 0 | `(1, 0)` | M 方向的第一级分片 |
| bit 1 | `(2, 0)` | M 方向的第二级分片 |
| bit 2 | `(0, 1)` | N 方向的一层分片 |

所以这是一个 `4 x 2` 的 CTA grid 语义：

```text
CTA ID  bits(b2 b1 b0)   M shard   N shard
0       000               0         0
1       001               1         0
2       010               2         0
3       011               3         0
4       100               0         1
5       101               1         1
6       110               2         1
7       111               3         1
```

这不是“每个 CTA 拿一个元素”。basis 的单位由对应 layout 的 chunk 大小决定。对官方示例而言，它表示先沿第 0 维切 4 个连续子张量，再沿第 1 维切 2 个连续子张量。

### `cga_layout` 中零向量的意义

```python
cga_layout = ((0, 0),)
```

两个 CTA 的映射都是 `(0, 0)`，因此两个 CTA 拿到相同的 tile。这个布局主要用于 TMA multicast：多个 CTA 需要同一份 A 或 B，就让一次 TMA 把同一片数据广播到它们的 SMEM。

### `get_split_dim` 的数值例子

官方 capstone 中有一个辅助函数：

```python
@gluon.constexpr_function
def get_split_dim(cga_layout, dim):
    return 1 << sum(b[dim] != 0 for b in cga_layout)
```

它数的是“有多少个非零 basis 落在某个维度上”：

| layout | dim | 非零 basis 数 | 返回 |
|---|---:|---:|---:|
| `((1, 0),)` | 0 | 1 | 2 |
| `((1, 0),)` | 1 | 0 | 1 |
| `((1, 0), (2, 0), (0, 1))` | 0 | 2 | 4 |
| `((1, 0), (2, 0), (0, 1))` | 1 | 1 | 2 |

capstone 里：

```python
tile_m = block_size_m * get_split_dim(cga_layout, 0)
```

当 `cga_layout=((1,0),)`、`block_size_m=128` 时：

```text
tile_m = 128 * 2 = 256
num_ctas = 2**len(cga_layout) = 2
```

也就是说，host 看到的 tile 是 `256 x N`，每个 CTA 本地负责其中 128 行。

## 跨 CTA 同步：哪些自动，哪些必须手写

Gluon 自动插入跨 CTA 同步的范围很保守：

| 操作 | 是否自动处理 CTA 分片差异 | 说明 |
|---|---|---|
| `gl.convert_layout` | 是 | 源和目标的 CTA 分片不同时插入 cluster 同步 |
| `gl.reduce` / `gl.sum` / `gl.max` | 是 | 结果可能需要跨 CTA 组合时插入 |
| TMA | 否 | 必须自己用 `mbarrier` 管 `expect -> async_load -> wait` |
| WGMMA | 否 | 仍需 barrier 保护操作数和完成 |
| `tcgen05_mma` | 否 | 仍需 load-ready、accumulator-ready、empty 等协议 |

因此 multi-CTA softmax 可以写得像单 CTA 一样：

```python
row_max = gl.max(x, axis=0)
y = gl.exp(x - row_max)
row_sum = gl.sum(y, axis=0)
```

只要 `x` 的 layout 把一个 row 切到多个 CTA，Gluon 会在 `gl.max` 和 `gl.sum` 前后补跨 CTA 规约所需的同步。

### multi-CTA mbarrier 的布局

最简单的 mbarrier layout 是“每个 CTA 一个独立 barrier”：

```python
MBarrierLayout(cga_layout=[[1], [2], [4]])
```

这组 basis 构成一个 3-bit identity，所以 8 个 CTA 各有一个 barrier。这个模式太常见，Gluon 提供：

```python
bar = mbarrier.allocate_mbarrier()
```

也可以让奇数 CTA 和偶数 CTA 两两共享 barrier：

```python
MBarrierLayout(cga_layout=[[0], [1], [2]])
```

这里 bit 0 的 basis 是 0，意味着 `CTA0/CTA1`、`CTA2/CTA3` 等共享一个 barrier。每个 group 中 CTA ID 最小的那个是 lead CTA。

### 跨 CTA mbarrier 的四条语义

对共享一个 barrier 的 CTA group：

| API | 语义 |
|---|---|
| `mbarrier.init` | count 自动乘以 group 内的 CTA 数；只在 lead CTA 初始化 |
| `mbarrier.expect` | `size_per_cta` 自动乘以 CTA 数；只在 lead CTA expect；非 lead CTA 也会贡献一次 arrival |
| `mbarrier.arrive` | group 内每个 CTA 都向 lead CTA 的 barrier arrival |
| `mbarrier.wait` | 只有 lead CTA 等待该 barrier |

`mbarrier.expect` 的费用也很具体：

```text
group 内 2 个 CTA
每 CTA 需要搬运 4096 bytes

lead CTA 执行 expect(bar, 4096)
  -> 最终等待的字节数是 2 * 4096 = 8192
```

但代码里仍然传 per-CTA 的 `4096`，因为 API 自己乘 group size。

### 初始化顺序

只要 barrier 会被跨 CTA 使用，或者会被 multicast/2CTA 消费，初始化顺序就有额外要求：

```text
每个相关 mbarrier.init
  -> fence.mbarrier_init.release.cluster
  -> relaxed cluster barrier
  -> 第一次使用 barrier
```

Gluon 会在顶层 init 序列后自动插入 fence 和 relaxed cluster barrier。因此 capstone 把 barrier 分配与初始化放在 `gl.warp_specialize` 外面：

```text
kernel entry
  -> allocate SMEM/TMEM
  -> init all barriers
  -> mandatory cluster sync
  -> gl.warp_specialize(partitions, warp_groups, reg_budgets)
```

cluster barrier 必须由 kernel 中所有线程执行，所以它和可能跨 CTA 的 `convert_layout/reduce/sum/max` 都不能放进 `warp_specialize` 内。

## 完整可运行样例：2CTA tcgen05 GEMM

下面是官方最小 2CTA 例程的完整展开。它只做一个 `256 x 64 @ 64 x 128` 的 GEMM，便于把布局和 barrier 关系看清楚。

文件可以保存为 `two_cta_tcgen05_example.py`：

```python
import pytest
import torch

from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.blackwell import (
    TensorMemoryLayout,
    allocate_tensor_memory,
    tcgen05_mma,
)
from triton.experimental.gluon.language.nvidia.hopper import mbarrier, tma
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor


def is_blackwell():
    if not torch.cuda.is_available():
        return False
    cc = torch.cuda.get_device_capability()
    return cc[0] == 10


@gluon.jit
def two_cta_tcgen05_kernel(a_desc, b_desc, c_desc):
    gl.static_assert(gl.num_ctas() == 2)

    cluster_m: gl.constexpr = a_desc.block_shape[0]
    tile_n: gl.constexpr = b_desc.block_shape[1]
    cta_m: gl.constexpr = cluster_m // 2
    cga_layout: gl.constexpr = c_desc.layout.cga_layout

    smem_a = gl.allocate_shared_memory(
        a_desc.dtype,
        a_desc.block_shape,
        a_desc.layout,
    )
    smem_b = gl.allocate_shared_memory(
        b_desc.dtype,
        b_desc.block_shape,
        b_desc.layout,
    )

    tma_bar = mbarrier.allocate_mbarrier(two_ctas=True)
    mma_bar = mbarrier.allocate_mbarrier()
    mbarrier.init(tma_bar, count=1)
    mbarrier.init(mma_bar, count=1)

    mbarrier.expect(
        tma_bar,
        a_desc.nbytes_per_cta + b_desc.nbytes_per_cta,
    )
    tma.async_load(a_desc, [0, 0], tma_bar, smem_a)
    tma.async_load(b_desc, [0, 0], tma_bar, smem_b)
    mbarrier.wait(tma_bar, phase=0, deps=[smem_a, smem_b])
    mbarrier.invalidate(tma_bar)

    acc_layout: gl.constexpr = TensorMemoryLayout(
        block=(cta_m, tile_n),
        col_stride=1,
        cga_layout=cga_layout,
        two_ctas=True,
    )
    acc = allocate_tensor_memory(
        gl.float32,
        [cluster_m, tile_n],
        acc_layout,
    )

    tcgen05_mma(
        smem_a,
        smem_b,
        acc,
        use_acc=False,
        mbarriers=[mma_bar],
    )
    mbarrier.wait(mma_bar, phase=0, deps=[smem_a, smem_b])
    mbarrier.invalidate(mma_bar)

    c_smem = gl.allocate_shared_memory(
        c_desc.dtype,
        c_desc.block_shape,
        c_desc.layout,
    )
    c_smem.store(acc.load().to(c_desc.dtype))
    tma.async_store(c_desc, [0, 0], c_smem)


def run_two_cta_tcgen05(a, b, c):
    M, N, K = a.shape[0], b.shape[1], a.shape[1]
    a_layout = gl.NVMMASharedLayout.get_default_for(
        [M, K],
        gl.float16,
        cga_layout=[(1, 0)],
    )
    b_layout = gl.NVMMASharedLayout.get_default_for(
        [K, N],
        gl.float16,
        cga_layout=[(0, 1)],
    )
    c_layout = gl.NVMMASharedLayout.get_default_for(
        [M, N],
        gl.float16,
        cga_layout=[(1, 0)],
    )

    a_desc = TensorDescriptor.from_tensor(a, [M, K], a_layout)
    b_desc = TensorDescriptor.from_tensor(b, [K, N], b_layout)
    c_desc = TensorDescriptor.from_tensor(c, [M, N], c_layout)

    two_cta_tcgen05_kernel[(1,)](
        a_desc,
        b_desc,
        c_desc,
        num_warps=4,
        num_ctas=2,
    )


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
def test_two_cta_tcgen05():
    M, N, K = 256, 128, 64
    a = torch.randn((M, K), device="cuda", dtype=torch.float16)
    b = torch.randn((K, N), device="cuda", dtype=torch.float16)
    c = torch.empty((M, N), device="cuda", dtype=torch.float16)

    run_two_cta_tcgen05(a, b, c)
    torch.testing.assert_close(c, torch.matmul(a, b), atol=1e-1, rtol=1e-2)
```

运行方式：

```bash
pytest -q two_cta_tcgen05_example.py
```

在 Blackwell 机器上，测试通过时没有额外输出；断言失败会显示 `c` 与 `a @ b` 的最大误差。当前笔记所在机器没有 CUDA GPU 和可执行的 Triton Gluon runtime，只能做 Python 静态编译检查，不能把 pytest 通过当作本机验证结果。

## 2CTA MMA 的布局和数据契约

这个例子的 shape 是：

```text
A: [256, 64] fp16
B: [64, 128] fp16
C: [256, 128] fp16
num_ctas = 2
```

A、B、C 的 CTA 分片必须是：

| tensor | `cga_layout` 的第一 basis | 含义 |
|---|---|---|
| LHS A | `(1, 0)` | 两个 CTA 沿 M 分片 |
| RHS B | `(0, 1)` | 两个 CTA 沿 N 分片 |
| accumulator C | `(1, 0)` | 两个 CTA 沿 M 分片，存回各自 TMEM |

2CTA `tcgen05.mma.cta_group::2` 接收一对 CTA 的数据，做一种协作外积式计算：

```text
CTA0 提供 A 的上半 M 行和 B 的左半边 N 列
CTA1 提供 A 的下半 M 行和 B 的右半边 N 列
硬件组合两边操作数，使每个 CTA 都得到自己负责的 M 行、完整 N 列结果
```

用本地 tile 表示：

```text
cluster A block: [256, 64]
CTA0 local A:    [128, 64]
CTA1 local A:    [128, 64]

cluster B block: [64, 128]
CTA0 local B:    [64, 64]
CTA1 local B:    [64, 64]

cluster C block: [256, 128]
CTA0 local C:    [128, 128]
CTA1 local C:    [128, 128]
```

这里的 `TensorMemoryLayout(block=(cta_m, tile_n), ...)` 是关键。`block=(cta_m, tile_n)` 表示单 CTA 的 TMEM 指令形状是 `128 x 128`；而 `allocate_tensor_memory(..., [cluster_m, tile_n])` 申请的是 cluster 视角下的完整 accumulator。

### 必须改的三处代码

从单 CTA `tcgen05` 切到 2CTA，至少有三处契约变化：

| 位置 | 单 CTA | 2CTA |
|---|---|---|
| accumulator layout | `two_ctas=False` | `TensorMemoryLayout(..., two_ctas=True)` |
| pre-MMA barrier | 普通 `allocate_mbarrier()` | `allocate_mbarrier(two_ctas=True)` |
| MMA completion count | `tcgen05_mma_barrier_count(..., two_ctas=False)` | 传入真实 `two_ctas=...` |

`tma_bar` 在最小样例里就是 pre-MMA barrier。它必须让 lead CTA 等两边的 A/B row 都 ready，才能发 collective MMA。

`mma_bar` 自己不需要 `two_ctas=True`，因为 `tcgen05_mma` 会把 completion signal multicast 给 pair 内 CTA。真正需要 `two_ctas=True` 的是 `tcgen05_mma_barrier_count` 的参数，用来算正确的 arrival 数。

还有两个容易踩的规则：

```text
一旦 kernel 中任何一条 tcgen05_mma 使用 2CTA 模式，
该 kernel 中所有 tcgen05_mma 都必须是 2CTA 模式。
```

```text
2CTA 的 mbarrier.expect 仍然传 per-CTA 字节数，
不是 pair 的总字节数。
```

### 数值轨迹

总 FLOP：

```text
2 * M * N * K
= 2 * 256 * 128 * 64
= 4,194,304 FLOP
```

如果只看 CTA0 负责的输出区域：

```text
C[0:128, 0:128]
```

它需要：

```text
A[0:128, 0:64] @ B[0:64, 0:128]
```

但 2CTA 硬件不是让 CTA0 自己再搬完整 B。B 在 CTA0/CTA1 间各有一半列，硬件用 pair 内交换完成整个 N 方向的结果。因此 CTA0 本地 TMEM 是 `128 x 128`，而不是 `128 x 64`。

### `tcgen05_mma_barrier_count` 为什么必须和 multicast 配合

在一个 K-loop 里，SMEM buffer 会被反复复用：

```text
k=0: TMA -> smem_a[stage0]
     MMA 读 smem_a[stage0]
k=1: TMA -> smem_a[stage1]
k=2: TMA -> smem_a[stage2]
k=STAGES: TMA 想重新写 smem_a[stage0]
```

如果下一轮 TMA 在 MMA 还没读完 `stage0` 时就覆盖它，结果就会错。因此需要让 TMA 等 `load_empty`，而 `load_empty` 的 arrival 由 MMA completion 产生。

当 TMA 或 MMA 带 multicast 时，完成信号可能来自多个 CTA，所以 barrier 的 count 不能写死为 1，要用：

```python
mma_barrier_count = tcgen05_mma_barrier_count(
    [a_bufs.index(0), b_bufs.index(0)],
    multicast=True,
    two_ctas=acc_bufs.index(0).type.layout.two_ctas,
)
```

它把参与一次 collective MMA 的 CTA 数和 multicast group 一起算进去。写错时最典型的现象是 kernel hang 或 barrier phase 永远不翻转，而不是简单的数值偏差。

## TMA multicast 的最小完整例子

当两个 CTA 需要同一片数据时，可以把 shared-memory layout 的第一个 CTA basis 设成 `(0, 0)`：

```python
layout = gl.NVMMASharedLayout.get_default_for(
    inp.shape,
    gl.float16,
    cga_layout=[[0, 0]],
)
```

完整 kernel：

```python
@gluon.jit
def tma_multicast_copy_kernel(in_desc, out_desc):
    gl.static_assert(gl.num_ctas() == 2)

    smem = gl.allocate_shared_memory(
        in_desc.dtype,
        in_desc.block_shape,
        in_desc.layout,
    )

    bar = mbarrier.allocate_mbarrier()
    mbarrier.init(bar, count=1)

    mbarrier.expect(bar, in_desc.nbytes_per_cta)
    tma.async_load(
        in_desc,
        [0, 0],
        bar,
        smem,
        multicast=True,
    )
    mbarrier.wait(bar, phase=0, deps=[smem])

    tma.async_store(out_desc, [0, 0], smem)


def run_tma_multicast_copy(inp, out):
    layout = gl.NVMMASharedLayout.get_default_for(
        inp.shape,
        gl.float16,
        cga_layout=[[0, 0]],
    )
    in_desc = TensorDescriptor.from_tensor(inp, inp.shape, layout)
    out_desc = TensorDescriptor.from_tensor(out, inp.shape, layout)

    tma_multicast_copy_kernel[(1,)](
        in_desc,
        out_desc,
        num_warps=4,
        num_ctas=2,
    )
```

host 侧验证：

```python
M, N = 128, 128
inp = torch.randn((M, N), device="cuda", dtype=torch.float16)
out = torch.empty_like(inp)
run_tma_multicast_copy(inp, out)
torch.testing.assert_close(out, inp, atol=0, rtol=0)
```

这里要区分两种 barrier：

```text
TMA barrier：
  每个 CTA 自己的 1D barrier。
  这个 kernel 没有用 2CTA MMA，所以不需要 two_ctas=True。

MMA completion barrier：
  如果进入 2CTA 模式，pre-MMA barrier 必须 two_ctas=True；
  其 completion count 要用 helper 算。
```

### TMA multicast 的收益模型

假设两个 CTA 都需要同一块 `128 x 64` fp16 tile：

```text
tile bytes = 128 * 64 * 2 = 16,384 bytes
```

没有 multicast：

```text
CTA0 读 16,384 bytes
CTA1 读 16,384 bytes
合计 L2/HBM 侧请求 32,768 bytes
```

有 multicast：

```text
一次 TMA 请求广播到两个 CTA 的 SMEM
L2/HBM 侧只需要取 16,384 bytes
```

实际收益取决于 A/B 的复用方向和 L2 命中率。对 GEMM 来说，A 常沿 M 分片、B 常沿 N 分片；把 K 方向广播掉的部分做成 multicast，通常比让每个 CTA 单独发 TMA 更省 L2。

## TMA multicast 与 tcgen05 的流水线契约

把 multicast 放进 K-loop 后，有三个地方要同时打开：

```python
tma.async_load(a_desc, [off_m, k], tma_bar, a_buf, multicast=True)
tma.async_load(b_desc, [k, off_n], tma_bar, b_buf, multicast=True)
tcgen05_mma(
    a_buf,
    b_buf,
    acc_buf,
    use_acc=use_acc,
    multicast=True,
    mbarriers=[mma_bar],
)
mbarrier.init(
    load_empty_bars.index(0),
    count=tcgen05_mma_barrier_count(
        [a_bufs.index(0), b_bufs.index(0)],
        multicast=True,
        two_ctas=acc_bufs.index(0).type.layout.two_ctas,
    ),
)
```

它们分别管：

| 位置 | 作用 |
|---|---|
| TMA load | 一次搬入，广播到 group 内每个 CTA 的 SMEM |
| tcgen05_mma | 完成时把 arrival multicast 给 group 内所有相关 CTA |
| count helper | 按实际参与 CTA 数初始化 barrier count |

这是为了阻止“下一次 TMA 覆盖上一轮尚未消费的 SMEM”：

```text
load_empty 的 arrival 必须来自本轮所有消费该 SMEM 的 MMA
  -> 所有 MMA 都完成后，load_empty 才 flip phase
  -> 下一轮 TMA 才能复用这个 stage
```

如果只在 TMA 上写 `multicast=True`，却忘了 MMA 的 multicast 和 count，最常见结果是多 CTA 之间 phase 不一致：一个 CTA 已经进入下一轮，另一个还在等上一轮，最后表现为 hang。

## Capstone：CLC 加四角色 warp specialization

官方最后的大例子把以前的优化合成一个 kernel。它的四个 partition 是：

| partition | 角色 | 读什么 | 写什么 | 关键 barrier |
|---|---|---|---|---|
| `matmul_clc_partition` | 动态取 tile | CLC result | 往 SMEM 写 packed pid | `clc_barriers`、`clc_planar_ready_bars`、`clc_consumed_bars` |
| `matmul_load_partition` | 发 TMA | CLC schedule | A/B SMEM stage | `load_empty_bars`、`load_ready_bars` |
| `matmul_mma_partition` | 发 tcgen05 | A/B SMEM | accumulator TMEM | `load_ready_bars`、`acc_empty_bars`、`acc_ready_bars` |
| `matmul_epilogue_partition` | 存结果 | accumulator TMEM | C SMEM 再 TMA store | `acc_ready_bars`、`acc_empty_bars` |

角色之间的关系不是单向直线，而是一个有反压的环：

```text
CLC producer
  -> tile schedule
  -> load partition
       -> load_ready
       -> MMA partition
            -> acc_ready
            -> epilogue partition
                 -> acc_empty
                 -> MMA partition

MMA partition also returns load_empty -> load partition
epilogue/load/MMA each consume the CLC ring slot
  -> clc_consumed
  -> CLC producer can reuse the slot
```

### CLC 为什么可以放进 warp_specialize

前面 12-cluster-launch-control 的 CLC 是单个 CTA 里的调度器。capstone 把它拆成一个专门的 warp group：

```python
gl.warp_specialize([
    (matmul_epilogue_partition, (p,)),
    (matmul_load_partition, (p,)),
    (matmul_mma_partition, (p,)),
    (matmul_clc_partition, (p,)),
], [1, 1, 1], [24, 24, 24])
```

这里的参数含义是：

| 参数 | 含义 |
|---|---|
| 函数列表 | 四个 partition |
| `[1, 1, 1]` | 每个 partition 分到 1 个 warp group |
| `[24, 24, 24]` | 每个 partition 的寄存器预算 |

注意 CLC 是跨 CTA cluster 调度，但 CLC 自己通过普通 SMEM buffer 和 mbarrier 广播 schedule。真正会产生 cluster barrier 的 `convert_layout/reduce` 仍然放在顶层，或者避免在 partition 内跨 CTA 调用。

### `_planar_snake`：不改变数学，只改变 tile 访问顺序

capstone 用 `_planar_snake` 做 L2 友好的 tile swizzle。它不改变输出数值，只改变 tile 的执行顺序：

```text
straight order:
  (0,0) (0,1) (0,2) (0,3)
  回到下一 row 时，相邻 tile 复用率下降

snake order:
  row 0 从左到右
  row 1 从右到左
  相邻 row 在边界处继续访问物理上较近的 tile
```

一个小的 `3 x 4` tile grid，`grid_tile_width=3` 时可以看到这个思想：

```text
线性 id    snake 后的 (m,n)
0          (0,0)
1          (1,0)
2          (2,0)
3          (2,1)
4          (1,1)
5          (0,1)
6          (0,2)
7          (1,2)
```

它的意义在 K 较大时更明显：相邻 tile 之间共享 A/B 的 L2 行，访问顺序越连续，越容易命中缓存。

### CLC 状态对象

CLC producer 每轮做四件事：

```text
1. 等 clc_consumed：这个 ring slot 已被三个 consumer 释放
2. try_cancel：向硬件请求一个尚未启动的 cluster
3. load_result：取回 tile id，判断是否 canceled
4. planar_snake：把 tile id 映射为 pid_m/pid_n
5. 打包并写入 planar pid，arrive planar_ready
```

核心代码形态是：

```python
mbarrier.wait(
    p.clc_consumed_bars.index(consumed_state.index),
    consumed_state.phase,
    pred=(i >= acc_stages),
)
barrier = p.clc_barriers.index(state.index)
result = p.clc_result_buffers.index(state.index)
mbarrier.expect(barrier, 16, from_cta=0x0)
clc.try_cancel(result, barrier)
mbarrier.wait(barrier, state.phase)
clc_res = clc.load_result(result)
has_work = clc_res.is_canceled()
```

`try_cancel` 的返回条件不是“拿到一个合法 tile”，而是“是否成功取消了尚未启动的 cluster”：

```text
is_canceled() == True  -> 有工作，使用 clc_res.program_id(0)
is_canceled() == False -> 没有更多 cluster，可以结束
```

### 四个 partition 的 barrier 合约

#### load partition

```text
等 load_empty[stage] 可以写
  -> expect A/B 字节数
  -> 发两条 TMA multicast load
  -> 推进 stage counter
```

具体代码：

```python
mbarrier.wait(
    p.load_empty_bars.index(state.index),
    state.phase,
    pred=pred,
)
bar = p.load_ready_bars.index(state.index)
mbarrier.expect(bar, p.a_desc.nbytes_per_cta + p.b_desc.nbytes_per_cta)
tma.async_load(
    p.a_desc,
    [off_m, k],
    bar,
    p.a_bufs.index(state.index),
    multicast=True,
)
tma.async_load(
    p.b_desc,
    [k, off_n],
    bar,
    p.b_bufs.index(state.index),
    multicast=True,
)
```

`pred` 处理的是第一次流水填充：前几个 stage 还没有历史 MMA，不需要等 empty。

#### MMA partition

```text
等 acc_empty[acc_stage] 可以累加
  -> 对 K 的每个 block 等 load_ready
  -> 发 tcgen05_mma
  -> 把 load_empty[load_stage] 作为 MMA completion barrier
  -> K-loop 完成后 tcgen05_commit acc_ready
```

`use_acc` 的控制很关键：

```text
第一个 K tile: use_acc=False，覆盖 accumulator
后续 K tile:    use_acc=True，继续累加
```

#### epilogue partition

```text
等 acc_ready[acc_stage]
  -> 从 TMEM load accumulator
  -> subtile 到多个 SMEM buffer
  -> TMA async_store 到 C
  -> arrive acc_empty，允许 MMA 复用 accumulator
```

epilogue 的 subtile 设计是为了让 TMEM 到 SMEM 再到 HBM 的搬运重叠起来。`subtile_stages` 控制 SMEM 使用量，`subtile_factor` 由 tile N 和 epilogue N 决定：

```text
subtile_factor = tile_n // split_tile_n
```

例如：

```text
tile_n = 256
split_tile_n = 32
subtile_factor = 8
```

如果一次性把 `256 x M` 全写进 SMEM，SMEM 会爆；拆成 8 个 `32 x M` subtile 后，可以在有限的 SMEM 预算里让 store 与下一段加载重叠。

### 顶层 barrier 初始化

capstone 的 kernel entry 先做一次性资源分配：

```python
two_ctas: gl.constexpr = gl.num_ctas() > 1

a_bufs = gl.allocate_shared_memory(
    dtype,
    [STAGES] + a_desc.block_shape,
    a_desc.layout,
)
b_bufs = gl.allocate_shared_memory(
    dtype,
    [STAGES] + b_desc.block_shape,
    b_desc.layout,
)

tmem_layout: gl.constexpr = TensorMemoryLayout(
    [BLOCK_SIZE_M, block_n // get_split_dim(CGA_LAYOUT, 1)],
    col_stride=1,
    cga_layout=CGA_LAYOUT,
    two_ctas=two_ctas,
)
acc_bufs = allocate_tensor_memory(
    gl.float32,
    [ACC_STAGES, block_m, block_n],
    tmem_layout,
)

mma_barrier_count: gl.constexpr = tcgen05_mma_barrier_count(
    [a_bufs.index(0), b_bufs.index(0)],
    multicast=True,
    two_ctas=acc_bufs.index(0).type.layout.two_ctas,
)
```

然后初始化所有 stage barrier：

```python
load_empty_bars = mbarrier.allocate_mbarrier(batch=STAGES)
load_ready_bars = mbarrier.allocate_mbarrier(
    batch=STAGES,
    two_ctas=two_ctas,
)
for i in gl.static_range(STAGES):
    mbarrier.init(load_empty_bars.index(i), count=mma_barrier_count)
    mbarrier.init(load_ready_bars.index(i), count=1)

acc_empty_bars = mbarrier.allocate_mbarrier(
    batch=ACC_STAGES,
    two_ctas=two_ctas,
)
acc_ready_bars = mbarrier.allocate_mbarrier(batch=ACC_STAGES)
for i in gl.static_range(ACC_STAGES):
    mbarrier.init(acc_empty_bars.index(i), count=1)
    mbarrier.init(acc_ready_bars.index(i), count=mma_barrier_count)
```

这些初始化之所以必须放在 `gl.warp_specialize` 之前，是因为：

1. 跨 CTA barrier 需要 `fence.mbarrier_init.release.cluster`；
2. 这个 fence 后面需要 relaxed cluster barrier；
3. cluster barrier 必须所有线程执行，不能藏在某个 warp-specialized partition 里。

### host 侧参数

capstone 的默认配置：

```python
cfg = {
    "block_size_m": 128,
    "block_size_n": 256,
    "block_size_k": 64,
    "grid_minor_dim": 0,
    "grid_tile_width": 16,
    "stages": 6,
    "acc_stages": 2,
    "cga_layout": ((1, 0),),
    "epilogue_size_n": 32,
    "subtile_stages": 4,
}
```

它对应的含义：

| 参数 | 值 | 作用 |
|---|---:|---|
| `block_size_m` | 128 | 单 CTA M tile |
| `block_size_n` | 256 | cluster N tile |
| `block_size_k` | 64 | 每轮 K block |
| `stages` | 6 | A/B SMEM 流水级数 |
| `acc_stages` | 2 | accumulator 双缓冲 |
| `epilogue_size_n` | 32 | TMEM 每次搬出的 N 子块 |
| `subtile_stages` | 4 | epilogue SMEM 级数 |
| `cga_layout` | `((1,0),)` | 2 个 CTA，沿 M 分片 |

### 上游 benchmark 结果

这些是官方教程在 Blackwell 上给出的结果，不是本机测量：

```text
M = 8192, N = 8192
    K         multi-CTA    cublas
  512           1096.31   1190.98
 1024           1306.07   1344.48
 2048           1379.80   1374.48
 4096           1444.26   1431.93
 8192           1302.33   1347.82
16384           1292.40   1371.82
```

单位是 TFLOPS。可以看到：

- K 太小时，TMA pipeline 和 launch/epilogue 的成本占比高，capstone 仍落后 cuBLAS；
- K 在 2048 到 4096 附近，本配置与 cuBLAS 接近甚至反超；
- K 很大时，accumulator 复用、TMEM 带宽或 tile scheduling 会重新成为瓶颈。

这也说明“2CTA + CLC + warp specialization”不是无条件更快。它们的价值需要满足：tile 足够大、K 足够长、复用足够高，才能覆盖更复杂的同步和资源调度成本。

## 跨 CTA 通信路径对照

| 路径 | 发起者 | 数据位置 | 典型同步 | 使用场景 |
|---|---|---|---|---|
| TMA block load | 每个 CTA 自己发 | HBM -> 本 CTA SMEM | TMA mbarrier | 普通单 CTA 数据加载 |
| TMA multicast | CTA group | HBM -> 多个 CTA SMEM | 广播 arrival 的 TMA mbarrier | A/B tile 被多个 CTA 复用 |
| distributed shared memory | 目标 CTA 访问别的 CTA SMEM | SMEM -> SMEM/cross-CTA | cluster barrier 或 mbarrier | 小规模交换、reduce 中间量 |
| 2CTA tcgen05 | pair 内 lead CTA 发 collective MMA | pair 的 SMEM/TMEM | `two_ctas` barrier + completion multicast | M/N 分片的大 GEMM tile |
| CLC | producer warp group | 只传 tile id | CLC result barrier | cluster 间动态工作分配 |

## 常见错误与症状

| 错误 | 典型症状 | 根因 | 修复 |
|---|---|---|---|
| `num_ctas` 不是 2 的幂或大于 16 | launch 失败 | CGA 硬件约束 | 选合法 cluster size |
| 用 `cga_layout=((1,0),)` 但 host 仍按 `[128,N]` 建 descriptor | 形状或越界错误 | tile M 实际是 `128*2=256` | 用 `get_split_dim` 扩展 descriptor tile |
| 2CTA accumulator 没设 `two_ctas=True` | 编译失败或 MMA 结果错 | TMEM 子块分配契约不成立 | 在 `TensorMemoryLayout` 打开 |
| pre-MMA barrier 没设 `two_ctas=True` | hang 或读到半边数据 | lead CTA 没等 pair 内另一路 ready | 给 TMA handoff barrier 打开 |
| `tcgen05_mma_barrier_count` 的 `two_ctas` 写错 | barrier phase 不翻转或提前翻转 | arrival 数不匹配 | 从 accumulator layout 读取实际值 |
| kernel 一部分 MMA 用 2CTA，一部分不用 | 编译期拒绝 | 同一 kernel 的 `tcgen05` group 必须一致 | 全部统一 |
| TMA 带 multicast，但 MMA 不带完成 multicast | 多 CTA 间 phase 漂移，最终 hang | 上一轮 SMEM 被提前覆盖 | 三个 multicast 位一起检查 |
| 在 `warp_specialize` 内做跨 CTA `gl.sum` | 编译失败 | `reduce` 会发 cluster barrier，cluster barrier 不能放在 partition 内 | 提前规约或移到顶层 |
| 先进入 `warp_specialize` 再初始化跨 CTA barrier | 非法或 race | init fence 和 cluster barrier 顺序不满足 | 全部 init 放在顶层 |
| `mbarrier.expect` 传 pair 总字节数 | hang 或提前完成 | API 会再乘 group size | 传 per-CTA 字节数 |
| epilogue 一次性把整个 TMEM tile 写进 SMEM | SMEM 超限或占用率骤降 | subtile factor 没算清 | 按 `subtile_stages` 分块 |

## 硬件与验证边界

本机没有 CUDA GPU，也没有与固定 revision匹配的 Triton Gluon runtime。因此：

- `num_ctas`、cluster barrier、TMA multicast 和 `tcgen05.mma.cta_group::2` 都不能在本机运行时验证；
- benchmark 表全部来自官方教程，不代表本机性能；
- 笔记可以做静态检查，包括 Python 语法、Markdown 结构和源码路径一致性；
- 真正的 pytest 和 benchmark 需要在 Hopper 或 Blackwell 机器上执行。

静态检查：

```bash
cd /Users/saboxu/Downloads/learning_notes
python3 -m py_compile operators/code/triton/gluon/tutorials/*.py
git diff --check
```

有 CUDA GPU 时，优先按顺序运行：

```bash
pytest -q operators/code/triton/gluon/tutorials/14-multicta.py
```

如果出现 hang，先检查所有 barrier 的 `two_ctas`、`multicast` 和 `Counter` phase，再用 GPU Nsight Compute / Nsight Systems 看 kernel 是否停在 `mbarrier.try_wait`，以及哪个 partition 没有 arrival。

## 这一章真正要掌握的能力链

把 07 到 12 串起来，Gluon 的能力链是：

```text
layout
  让每个元素明确落到某个 thread/warp/CTA/register/Shared/TMEM 位置
    |
    +-> async copy / TMA
    |     把 global <-> shared 的搬运从普通 load/store 变成异步描述符操作
    |
    +-> WGMMA / tcgen05
    |     让 tensor core 直接从 SMEM/TMEM 取操作数并异步完成
    |
    +-> persistent / warp specialization
    |     让 load、MMA、epilogue 用不同角色并行，不再串行执行
    |
    +-> gather/scatter / tcgen05_copy / scaled MMA / CLC / im2col
    |     把不规则访问、格式转换、低精度反量化和动态调度交给专用硬件
    |
    +-> CGA / TMA multicast / 2CTA
          让多个 CTA 共享一次搬运、共享一个大 accumulator，协同跑一个 tile
```

最终最重要的不是记住每个 API 的名字，而是每一步都回答四个问题：

1. **数据在哪里？** register、SMEM、TMEM 还是 HBM。
2. **谁拥有它？** 哪个 thread、warp、CTA 或 CTA pair。
3. **谁生产、谁消费？** producer 和 consumer 的边界在哪里。
4. **什么证明它 ready / free？** mbarrier phase、commit group、cluster sync 还是访存顺序。

能完整回答这四个问题，Gluon 的 layout 和同步错误就会从“玄学 hang”变成可定位的状态机问题。

## 自我检查

1. **`cga_layout=((1,0),)` 和 `((0,0),)` 的核心区别是什么？**
   - 答：`((1,0),)` 让两个 CTA 沿第 0 维切连续 chunk；`((0,0),)` 让两个 CTA 得到同一片数据，用于 TMA multicast 之类的广播。
2. **为什么 2CTA 的 pre-MMA barrier 需要 `two_ctas=True`，而 `mma_bar` 自己通常不需要？**
   - 答：pre-MMA barrier 要让 lead CTA 等到 pair 内两边的 A/B 都 ready；`mma_bar` 的 completion 由 `tcgen05_mma` 自动 multicast 到 pair 内 CTA。关键是 completion count 的计算要传对 `two_ctas`。
3. **为什么 barrier 初始化必须放在 `gl.warp_specialize` 之前？**
   - 答：跨 CTA barrier init 后需要 `fence.mbarrier_init.release.cluster` 和 relaxed cluster barrier，而 cluster barrier 必须由所有线程执行，不能放进某个 warp-specialized partition。
4. **TMA multicast 已经省了一次 HBM/L2 读取，为什么还需要 MMA completion multicast？**
   - 答：TMA 解决的是“怎么把数据送进多个 CTA”；MMA completion multicast 解决的是“何时允许下一轮 TMA 覆盖已经被所有 CTA 消费过的 SMEM”。没有后者，多 CTA 的 stage 复用会出现提前覆盖或 hang。
5. **capstone 的四个 partition 之间，哪一个负责把 CLC tile id 变成可执行的 `pid_m/pid_n`？**
   - 答：CLC producer 负责 `try_cancel`、取 tile id、`_planar_snake` 映射和写入 packed pid；load/MMA/epilogue partition 通过 `ClcTileSchedulerConsumer` 读取同一份 schedule。
6. **为什么官方 benchmark 在 K=512 时落后 cuBLAS，而 K=2048/4096 时又能接近或反超？**
   - 答：K 很小时，TMA pipeline、CLC、epilogue 和同步开销占比高，协作收益覆盖不了固定成本；K 适中时，2CTA 的 shared load 和 tensor core 利用率收益占主导。K 过大时 accumulator/TMEM 压力和 tile 调度又可能重新成为瓶颈。
