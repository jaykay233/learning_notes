# Triton Gluon 05–06：WGMMA 与 tcgen05

## 本次讲解位置

章节：`chapter_gluon_tensor_cores`（算子开发 / Triton Gluon）<br>
小节：Hopper 的 warp-group MMA 与 Blackwell 的 tensor memory MMA<br>
知识点：`warpgroup_mma`、异步 WGMMA、`TensorMemoryLayout`、`allocate_tensor_memory`、`tcgen05_mma`、`tcgen05_commit`<br>
上次：TMA 和 mbarrier 把 global memory 与 shared memory 的传输显式流水化<br>
下次：让 kernel 常驻 SM，并把 load、MMA、epilogue 拆到不同 warp 角色<br>
PTX：`wgmma.mma_async`、`tcgen05.alloc`、`tcgen05.st/ld`、`tcgen05.mma`、`tcgen05.commit`

## 为什么现在讲这个

前两节解决了两个问题：tile 里的元素由谁拥有，以及数据怎样异步进入 shared memory。Tensor Core 是下一步：要让整块 A、B 参与矩阵乘，并让累加过程与下一块数据的装载重叠。

Hopper 和 Blackwell 的核心差异可以先用一句话概括：

```text
Hopper:   accumulator 在 register，WGMMA 按 commit group 等待
Blackwell: accumulator 在 TMEM，tcgen05 MMA 按 mbarrier 等待
```

如果只记 API 名称，很容易把两套完成语义混在一起。正确的心智模型是把一次 MMA 拆成三个契约：

```text
operand contract   A/B 放在哪里，layout 是否合法
execution contract MMA 是异步的，不能提前读结果或覆盖输入
completion contract 用哪种对象判断“已经做完”
```

## 01 到 06 的数据通路

把官方前六课连起来看，数据通路逐渐变得更显式：

```text
global memory
    |
    | TMA, cp.async
    v
shared memory
    |
    | WGMMA 直接读 SMEM
    | tcgen05 MMA 读 SMEM/TMEM
    v
Hopper: registers  /  Blackwell: TMEM
    |
    | epilogue, TMA store
    v
global memory
```

Gluon 并没有让数据通路变得更短。它让每一段搬移、布局和同步都成为源码的一部分，因此可以针对具体硬件做精细调度。

## Warp-group MMA：Hopper 的异步矩阵乘

`warpgroup_mma` 执行：

```text
D = A @ B + C
```

它不是一个单独的硬件指令，而是把多个 `wgmma.mma_async` 指令平铺到 block shape。Gluon 根据 accumulator layout 推导怎样切分到 warp group。

### 操作数位置

| 操作数 | 允许的位置 | 关键约束 |
|---|---|---|
| A | register 或 shared memory | register 路径使用 `DotOperandLayout` |
| B | 只能是 shared memory | 需要 `NVMMASharedLayout` |
| C/D | register | 使用 `NVMMADistributedLayout` |

因此下面两种写法都存在：

```python
acc = warpgroup_mma(a_reg, b_smem, acc, is_async=True)
acc = warpgroup_mma(a_smem, b_smem, acc, is_async=True)
```

### 指令形状

WGMMA 的 instruction shape 写作 `[m, n, k]`：

```text
m = 16
k = 256 / A.bitwidth
n = 浮点类型时，是 8 的倍数且不超过 256
```

例如 fp16 的 K 元素宽度是 16 bit：

```text
k = 256 / 16 = 16
```

fp8 则是：

```text
k = 256 / 8 = 32
```

`warps_per_cta` 的作用和 `BlockedLayout` 类似，但最小 atom 是 `[4, 1]`。这意味着：

- 至少需要 4 warps，也就是一个 warp group。
- M 维度至少有 4 个 warp，而 `m=16`，所以 BLOCK_M 至少是 64。
- 8 warps 可以选择 `[4, 2]` 或 `[8, 1]`，两种切分会改变 reduction 和访存成本。

### dependency 不能省

WGMMA 的结果虽然在 register 中，但异步指令完成前不可读。`warpgroup_mma_wait` 必须接收并返回 accumulator：

```python
acc = warpgroup_mma(a, b, acc, is_async=True)
acc = warpgroup_mma_wait(num_outstanding=0, deps=(acc,))
```

`deps` 不是装饰参数。它把后续使用和这一次异步写建立数据依赖。若丢掉返回值，只调用：

```python
warpgroup_mma_wait(num_outstanding=0, deps=(acc,))
# acc 仍是旧依赖
```

编译器无法知道你真正想等待哪一组 MMA，后续读结果可能提前。

### shared store 到 WGMMA 的 fence

TMA 与 WGMMA 都走 async proxy，因此它们之间不需要额外 fence。但普通 `smem.store` 走 generic proxy：

```python
b_smem.store(b)
fence_async_shared()
acc = warpgroup_mma(a, b_smem, acc, is_async=True)
```

`fence_async_shared()` 建立 generic store 到 async WGMMA 的可见性顺序。

## WGMMA 完整可运行例程

下面是从固定版本官方教程摘出的最小内核，并补齐 host 启动和验证。它演示 A 在 register/SMEM 两种路径，以及 `D = A @ B + C`。

```python
from __future__ import annotations

import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor
from triton.experimental.gluon.language.nvidia.hopper import (
    tma,
    mbarrier,
    fence_async_shared,
    warpgroup_mma,
    warpgroup_mma_wait,
)


@gluon.jit
def small_mma_kernel(
    a_desc,
    b_desc,
    c_desc,
    d_desc,
    LHS_IN_REG: gl.constexpr,
    INSTR_SHAPE_N: gl.constexpr,
    num_warps: gl.constexpr,
):
    bar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(bar, count=1)

    a_smem = gl.allocate_shared_memory(
        a_desc.dtype, a_desc.block_type.shape, a_desc.layout
    )
    b_smem = gl.allocate_shared_memory(
        b_desc.dtype, b_desc.block_type.shape, b_desc.layout
    )
    c_smem = gl.allocate_shared_memory(
        c_desc.dtype, c_desc.block_type.shape, c_desc.layout
    )

    mbarrier.expect(
        bar,
        a_desc.block_type.nbytes
        + b_desc.block_type.nbytes
        + c_desc.block_type.nbytes,
    )
    tma.async_load(a_desc, [0, 0], bar, a_smem)
    tma.async_load(b_desc, [0, 0], bar, b_smem)
    tma.async_load(c_desc, [0, 0], bar, c_smem)
    mbarrier.wait(bar, phase=0)
    mbarrier.invalidate(bar)

    m: gl.constexpr = 16
    k: gl.constexpr = 256 // a_desc.dtype.primitive_bitwidth
    n: gl.constexpr = INSTR_SHAPE_N
    warps_per_cta: gl.constexpr = [num_warps, 1]

    c_layout: gl.constexpr = gl.NVMMADistributedLayout(
        version=[3, 0],
        warps_per_cta=warps_per_cta,
        instr_shape=[m, n, k],
    )
    a_reg_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=0,
        parent=c_layout,
        k_width=32 // a_desc.dtype.primitive_bitwidth,
    )

    if LHS_IN_REG:
        a = a_smem.load(a_reg_layout)
    else:
        a = a_smem

    c = c_smem.load(c_layout)
    d = warpgroup_mma(a, b_smem, c, is_async=True, use_acc=True)
    d = warpgroup_mma_wait(num_outstanding=0, deps=(d,))

    d_smem = gl.allocate_shared_memory(
        d_desc.dtype, d_desc.block_type.shape, d_desc.layout
    )
    d_smem.store(d)
    fence_async_shared()
    tma.async_store(d_desc, [0, 0], d_smem)
    tma.store_wait(pendings=0)


def small_mma(A, B, C, D, instr_shape_n, lhs_in_reg=False, num_warps=4):
    a_layout = gl.NVMMASharedLayout.get_default_for(A.shape, gl.float16)
    b_layout = gl.NVMMASharedLayout.get_default_for(B.shape, gl.float16)
    cd_layout = gl.NVMMASharedLayout.get_default_for(C.shape, gl.float32)

    a_desc = TensorDescriptor.from_tensor(A, A.shape, a_layout)
    b_desc = TensorDescriptor.from_tensor(B, B.shape, b_layout)
    c_desc = TensorDescriptor.from_tensor(C, C.shape, cd_layout)
    d_desc = TensorDescriptor.from_tensor(D, D.shape, cd_layout)

    small_mma_kernel[(1,)](
        a_desc,
        b_desc,
        c_desc,
        d_desc,
        lhs_in_reg,
        instr_shape_n,
        num_warps=num_warps,
    )


if __name__ == "__main__":
    M, N, K = 64, 64, 32
    A = torch.randn(M, K, device="cuda", dtype=torch.float16)
    B = torch.randn(K, N, device="cuda", dtype=torch.float16)
    C = torch.randn(M, N, device="cuda", dtype=torch.float32)
    D = torch.empty_like(C)
    small_mma(A, B, C, D, instr_shape_n=16, lhs_in_reg=False)
    torch.testing.assert_close(A @ B + C, D, atol=1e-3, rtol=1e-1)
    print("wgmma_ok", D.sum().item())
```

运行要求 Hopper。官方文件的完整参数矩阵和 benchmark harness 位于 [`05-wgmma.py`](code/triton/gluon/tutorials/05-wgmma.py)。

## WGMMA 操作数、累加与 epilogue 逐行解释

### 本次讲解位置

章节：`chapter_gluon_tensor_cores`（算子开发 / Triton Gluon）<br>
小节：Warp-Group MMA / operand layout、accumulator 与 epilogue<br>
知识点：`DotOperandLayout`、`NVMMASharedLayout`、`LHS_IN_REG`、`use_acc`、`is_async`、register accumulator 到 TMA store<br>
上次：`NVMMADistributedLayout` 如何描述 accumulator 的 warp/register 分布<br>
下次：在 K-loop 中复用 accumulator，并区分首轮清零、后续累加和 WGMMA pipeline 等待<br>
PTX：`wgmma.mma_async`、`fence.proxy.async.shared::cta`、TMA store commit group<br>
完整代码：见上文 [WGMMA 完整可运行例程](#wgmma-完整可运行例程)

### 为什么现在讲这个

前一段已经知道 `c_layout` 决定 accumulator 在 4 个 warp 之间怎样分布，但只有 accumulator layout 还不能调用 WGMMA。A 还有两条完全不同的数据路径：

```text
LHS_IN_REG=False: A 留在 SMEM，WGMMA 直接读 SMEM
LHS_IN_REG=True:  A 先按 DotOperandLayout 装进 register，WGMMA 从 register 读
```

这两条路径不是语法糖。选择错误会多走一次 SMEM-to-register 搬运，或者根本无法构造出合法的 WGMMA operand。本节还要解决另一个容易混淆的问题：accumulator 异步完成后，怎样从 register 安全地放进 SMEM，再交给 TMA 写回 GMEM。

### 心智模型：A、B、C 各自遵守不同的位置契约

一次 `warpgroup_mma` 可以写成：

```text
D = A @ B + C
```

但这里的 A、B、C 不是三个普通 PyTorch tensor，它们分别处在不同存储和 layout 契约下：

| 对象 | 允许位置 | Gluon 中的表示 |
|---|---|---|
| A / LHS | SMEM 或 register | `a_smem` 或 `a_smem.load(a_reg_layout)` |
| B / RHS | 只能是 SMEM | `b_smem` |
| C / accumulator 输入 | register | `c_smem.load(c_layout)` |
| D / accumulator 输出 | register | `warpgroup_mma` 返回的异步依赖 |

因此这段代码首先不是在“读参数”，而是在把三块数据转换到 WGMMA 唯一接受的位置和 layout。

### `DotOperandLayout`：A 从 SMEM 搬到 register 时的分布契约

```python
a_reg_layout: gl.constexpr = gl.DotOperandLayout(
    operand_index=0,
    parent=c_layout,
    k_width=32 // a_desc.dtype.primitive_bitwidth,
)
```

三个参数的含义是：

```text
operand_index=0
    这是矩阵乘的第 0 个操作数，也就是 A / LHS

parent=c_layout
    A 的 register 分布必须和 accumulator 的 warp/instruction layout 相容

k_width=32 / dtype.bitwidth
    每个 32-bit register 在 K 方向装多少个元素
```

例如 A 是 fp16：

```text
k_width = 32 / 16 = 2
```

所以一个 32-bit register 可以装两个相邻的 fp16 元素：

```text
register r0: A[i, 0], A[i, 1]
register r1: A[i, 2], A[i, 3]
register r2: A[i, 4], A[i, 5]
```

如果 A 是 fp8：

```text
k_width = 32 / 8 = 4
register r0: A[i, 0], A[i, 1], A[i, 2], A[i, 3]
```

`DotOperandLayout` 不只是决定每行怎样打包，还决定这个行块属于哪个 warp、哪个 lane 的哪些 register。因此它必须由 `parent=c_layout` 派生，不能随便写成一个普通 `BlockedLayout`。

### `LHS_IN_REG`：两种写法的真实差别

```python
if LHS_IN_REG:
    a = a_smem.load(a_reg_layout)
else:
    a = a_smem
```

当 `LHS_IN_REG=True` 时，执行：

```text
SMEM A --load(a_reg_layout)--> register A --WGMMA--> register D
```

当 `LHS_IN_REG=False` 时，执行：

```text
SMEM A --WGMMA 直接读取----------------------> register D
```

什么时候选寄存器路径？

- A 本来就在 register，或者需要在 register 中做变换、缩放或复用：寄存器路径避免再绕回 SMEM。
- A 刚由 TMA 搬进 SMEM，且没有 register 侧计算：直接读 SMEM 通常更便宜，因为省掉一次 load。

WGMMA 的 B 没有这个选择，因为 RHS 必须是 SMEM。代码把 `b_smem` 直接传给 `warpgroup_mma`，而 A 根据 `LHS_IN_REG` 选择实际传入对象。

### `static_assert`：为什么 SMEM operand 必须是 `NVMMASharedLayout`

```python
gl.static_assert(isinstance(a_smem.type.layout, gl.NVMMASharedLayout))
gl.static_assert(isinstance(b_smem.type.layout, gl.NVMMASharedLayout))
```

这两个判断在编译期执行，不产生运行时分支。它们检查 A/B 的 SMEM 物理布局是否满足两个要求：

1. TMA 能把 tile 正确写入该 SMEM。
2. WGMMA 能按特定 stride、swizzle 和 bank 访问模式读取该 SMEM。

如果用了普通 row-major SMEM layout，即使数学上元素都正确，WGMMA 也可能无法形成合法的 operand view。把问题尽早变成编译错误，比运行到一半才得到错误结果更安全。

### `c_smem.load(c_layout)`：C 从 SMEM 进入 register accumulator

```python
c = c_smem.load(c_layout)
```

C 的物理 tile 在 SMEM 中，但 WGMMA 的 accumulator 加法发生在 register 中。`load(c_layout)` 的作用是：

```text
SMEM 中的 C tile
    --按 c_layout 分布--> 每个 warp/lane 的 register
```

这里的 `c_layout` 必须和后面的 accumulator layout 一致。否则 Gluon 需要插入 `convert_layout`，轻则增加 shuffle，重则因为 WGMMA 不接受该 operand layout 而编译失败。

### `use_acc`：首轮清零与后续累加的关键开关

```python
d = warpgroup_mma(a, b_smem, c, is_async=True, use_acc=True)
```

`use_acc` 决定 C/旧 accumulator 是否参与计算：

| `use_acc` | 数学语义 | 典型场景 |
|---|---|---|
| `True` | `D = A @ B + C` | C 已经有值，或需要继续累加 |
| `False` | 忽略 C，效果相当于 `D = A @ B + 0` | K-loop 第一轮，省掉显式清空 accumulator |

这里传 `True`，因为 host 端准备了一个真实的 C tile，测试验证的是 `A @ B + C`。如果改成 K-loop，通常写成：

```python
use_acc=(k != 0)
```

第一轮忽略旧 accumulator，后续轮次才继续累加。要注意：`use_acc=False` 不会把 C 的 SMEM 或 register 内容物理清零，它只是让 WGMMA 不读取旧值，语义上相当于从零开始累加。

### `is_async`：同步语义和异步依赖的区别

```python
d = warpgroup_mma(a, b_smem, c, is_async=True, use_acc=True)
```

`is_async=True` 表示 WGMMA 发起后立即返回一个“尚未完成”的 accumulator 依赖。代码不能在这个点读取 `d` 的数值，因为 tensor core 可能还在写它。

如果使用默认的：

```python
d = warpgroup_mma(a, b_smem, c, use_acc=True)
```

语义上等价于马上等待 0 个 outstanding WGMMA，也就是“发起后立即等完”。它不是另一种同步硬件指令，而是把 WGMMA 的异步机制包成了同步调用。需要与 TMA 或其他 MMA overlap 时，必须使用 `is_async=True`，并在真正读取结果前显式 wait。

### `warpgroup_mma_wait`：结果依赖必须穿过 wait

```python
d = warpgroup_mma_wait(num_outstanding=0, deps=(d,))
```

这里同时做了两件事：

```text
deps=(d,)                告诉编译器：代码依赖的是这次 WGMMA 产生的 d
num_outstanding=0        等到在途 WGMMA 不大于 0，也就是全部完成
返回的 d                 现在才可以当作完成的 register accumulator
```

如果只写：

```python
warpgroup_mma_wait(num_outstanding=0, deps=(d,))
d_smem.store(d)
```

没有把返回值重新赋给 `d`，编译器看到的 `d` 仍然可能是旧依赖。更安全的写法始终是：

```python
d = warpgroup_mma(..., is_async=True)
d = warpgroup_mma_wait(num_outstanding=0, deps=(d,))
d_smem.store(d)
```

### Epilogue：register accumulator 到 GMEM 的完整路径

WGMMA 完成后，代码先把 D 放进 SMEM：

```python
d_smem = gl.allocate_shared_memory(
    d_desc.dtype,
    d_desc.block_type.shape,
    d_desc.layout,
)
d_smem.store(d)
fence_async_shared()
tma.async_store(d_desc, [0, 0], d_smem)
tma.store_wait(pendings=0)
```

逐步看数据通路：

```text
WGMMA accumulator D
    |
    | generic register -> SMEM store
    v
d_smem
    |
    | fence_async_shared
    | 让 generic store 对 async proxy 可见
    v
TMA async_store
    |
    | store_wait(pendings=0)
    v
global memory
```

为什么这里必须有 `fence_async_shared()`？

```text
d_smem.store(d)      走 generic proxy
tma.async_store(...) 走 async proxy，从 SMEM 读数据
```

`fence_async_shared()` 建立两个 proxy 之间的可见性顺序。少了它，TMA 可能在 `d_smem.store` 尚未对 async proxy 可见时就开始读取，看到旧值或部分写入的 tile。

`tma.store_wait(pendings=0)` 等待在途 TMA store 降到 0。它保证内核结束前，SMEM 已经完成向 GMEM 的提交，不能因为 kernel 退出而让尚未完成的 TMA store 丢失。

### 一个具体执行轨迹

以 fp16、`M=N=64`、`K=32`、`LHS_IN_REG=True` 为例：

```text
1. TMA: GMEM A/B/C -> a_smem/b_smem/c_smem
2. a = a_smem.load(a_reg_layout)
3. c = c_smem.load(c_layout)
4. d = warpgroup_mma(a, b_smem, c, is_async=True, use_acc=True)
5. d = warpgroup_mma_wait(0, deps=(d,))
6. d_smem.store(d)
7. fence_async_shared()
8. tma.async_store(d_desc, [0, 0], d_smem)
9. tma.store_wait(0)
10. 回到 host，验证 D == A @ B + C
```

如果 `LHS_IN_REG=False`，只删除第 2 步，并把第 4 步的 A operand 改成 `a_smem`；这就是同一段代码性能参数化的来源。

### 常见错误与症状

| 错误 | 症状 | 原因 |
|---|---|---|
| A 放 register 但不用 `DotOperandLayout` | layout 校验失败，或自动插入昂贵转换 | WGMMA 要求 operand 分布与 accumulator 相容 |
| B 从 register 传入 | 编译失败 | WGMMA 的 B/RHS 必须来自 SMEM |
| A/B 用普通 SMEM layout | `static_assert` 失败 | TMA 与 WGMMA operand 需要 `NVMMASharedLayout` |
| 首轮使用 `use_acc=True` 且没有初始化 accumulator | 结果包含旧值或未定义值 | 旧 accumulator 被纳入加法 |
| 异步 WGMMA 后直接读 `d` | 读到旧值或竞态 | 没有经过 `warpgroup_mma_wait` |
| wait 后不接收返回值 | 依赖链可能仍然指向旧 accumulator | 没有把完成依赖重新绑定给 `d` |
| `d_smem.store` 后直接 TMA store | SMEM 结果偶发错误 | 缺少 generic-to-async proxy fence |
| 不执行 `tma.store_wait` | kernel 结束时可能存在未完成 store | TMA store 未等待完成 |

### 自我检查

1. **`DotOperandLayout` 为什么需要 `parent=c_layout`？**
   - 答：A 的 register 分布必须与 accumulator 的 warp tiling 和 WGMMA instruction shape 相容，不能独立选择。
2. **`k_width=32 // primitive_bitwidth` 的物理含义是什么？**
   - 答：每个 32-bit register 在 K 方向能装几个 A 元素。fp16 是 2 个，fp8 是 4 个。
3. **`use_acc=False` 与先把 C 手动清零有什么不同？**
   - 答：`use_acc=False` 让 WGMMA 忽略旧 accumulator，语义上从零开始，不要求先把 C 的存储物理清零。
4. **为什么 TMA 写完 A/B 后直接接 WGMMA 不需要 fence，而 `d_smem.store` 后接 TMA store 需要 fence？**
   - 答：前者的生产者和消费者都在 async proxy；后者是 generic proxy store 到 async proxy read，需要显式建立可见性顺序。
5. **`tma.store_wait(pendings=0)` 为什么不能省？**
   - 答：TMA store 是异步的，必须等待在途 store 完成，才能安全结束内核并保证 SMEM tile 已提交到 GMEM。

## 用数值轨迹理解 K-loop

设 accumulator 的某个元素是 `c[i, j]`，说明：

```text
c[i, j] += sum_k A[i, k] * B[k, j]
```

若 BLOCK_K=2，K=6，则 WGMMA 版本的累加轨迹是：

```text
iteration 0:
  c = C
  c += A[:, 0:2] @ B[0:2, :]

iteration 1:
  c += A[:, 2:4] @ B[2:4, :]

iteration 2:
  c += A[:, 4:6] @ B[4:6, :]
```

数学上可以这样看，但硬件执行不是三次同步乘加。`warpgroup_mma` 返回的是一个“尚未完成的 accumulator 依赖”。在每轮 K 开始前等待上一轮完成，是为了保证复用 operand buffer 安全；在循环结束后再等一次，才能读最终值。

官方 pipelined kernel 的核心节奏是：

```text
load(k+1) -> wait TMA -> wait previous WGMMA -> issue WGMMA(k) -> next buffer
```

注意这里等待的是上一轮 WGMMA，不是刚刚发出的 WGMMA。这样 TMA 可以继续跑，WGMMA 也可以留在 tensor pipe 中。

## WGMMA 的 pipeline 为什么有效

没有流水时，TMA 和 WGMMA 串行：

```text
TMA(k) -> MMA(k) -> TMA(k+1) -> MMA(k+1)
```

双缓冲后，TMA 读取下一块，而 tensor core 处理当前块：

```text
TMA(k+1) || MMA(k)
```

官方在没有 WGMMA pipeline 时的最佳结果约为 466 TFLOPS；加入异步 WGMMA pipeline 后最佳结果约为 495.31 TFLOPS。数值不是本机测量，重点是异步指令把 tensor core 从等待 operand 的空档中释放出来。

## Blackwell Tensor Memory

Blackwell 的 tensor core 不把 accumulator 放在普通 register file。每个 SM 有一块 tensor memory，组织为：

```text
128 rows x 512 columns x 32-bit cells
```

它不是 shared memory 的另一块别名，而是与 register file 并列的低延迟存储区域。限制决定了 Gluon API 的形状：

| 限制 | 后果 |
|---|---|
| 每个 warp 只能访问由其 warp ID 决定的 32 行 | 4 warps 才能覆盖 128 行 |
| TMEM 按列分配，大小必须是 `[32, 512]` 内 2 的幂 | 不能任意按字节分配 |
| Gluon 的 TMEM load/store 只支持 4 或 8 warps | 小 CTA 不能单独访问完整 TMEM |
| Gluon 只支持二维 load/store | 三维操作需要拆成 descriptor 索引 |
| 分配会阻塞但通常不直接降低 occupancy | 分配失败时 kernel 会等待资源 |

`TensorMemoryLayout(block=(blockM, blockN), ...)` 中的：

```text
blockM = 64 或 128
blockN = 1 到 256 的 2 的幂
```

当 `blockM=64` 且有多个 block 时，两行 block 可以打包到 128 行中，slicing descriptor 因此更复杂。

## tcgen05 MMA 的操作数契约

`tcgen05_mma` 同样计算矩阵乘加，但位置约束不同：

| 操作数 | 允许位置 |
|---|---|
| LHS A | SMEM 或 TMEM |
| RHS B | SMEM |
| accumulator C/D | TMEM |

SMEM operand 需要 `NVMMASharedLayout`。TMEM operand 是 `tensor_memory_descriptor`，不是普通 tensor。accumulator 也始终是 TMEM descriptor。

完成跟踪使用 mbarrier：

```python
tcgen05_mma(a, b_smem, acc_tmem)
tcgen05_commit(bar)
mbarrier.wait(bar, phase=0)
```

也可以直接传入 barrier：

```python
tcgen05_mma(a, b_smem, acc_tmem, mbarriers=[bar], mbarrier_preds=[True])
```

两种方式表达的是同一件事：把“此前发出的 tcgen05 MMA 已完成”变成一次 mbarrier arrival。

## TMEM 完整可运行例程

下面的程序只测试 TMEM store/load，不计算矩阵乘。它把 global memory 中的 `M x N` tile 转到 TMEM，再取回 register 并写回。这样可以先隔离地理解 TMEM layout 和 register layout 的关系。

```python
from __future__ import annotations

import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.blackwell import (
    TensorMemoryLayout,
    allocate_tensor_memory,
)


@gluon.jit
def tmem_example_kernel(
    in_ptr,
    out_ptr,
    M: gl.constexpr,
    N: gl.constexpr,
    num_warps: gl.constexpr,
):
    global_layout: gl.constexpr = gl.BlockedLayout(
        [1, 1], [1, 32], [1, num_warps], [1, 0]
    )
    offs_m = gl.arange(0, M, layout=gl.SliceLayout(1, global_layout))
    offs_n = gl.arange(0, N, layout=gl.SliceLayout(0, global_layout))
    offs = offs_m[:, None] * N + offs_n[None, :]

    value = gl.load(in_ptr + offs)
    tmem_layout: gl.constexpr = TensorMemoryLayout(
        block=(64, 64),
        col_stride=32 // in_ptr.dtype.element_ty.primitive_bitwidth,
    )
    tmem = allocate_tensor_memory(
        element_ty=in_ptr.dtype.element_ty,
        shape=[M, N],
        layout=tmem_layout,
    )

    tmem_reg_layout: gl.constexpr = tmem.get_reg_layout()
    value = gl.convert_layout(value, tmem_reg_layout)
    tmem.store(value)
    value = tmem.load()
    value = gl.convert_layout(value, global_layout)
    gl.store(out_ptr + offs, value)


def run_tmem(M=64, N=64, num_warps=4):
    x = torch.randn(M, N, device="cuda", dtype=torch.float32)
    y = torch.empty_like(x)
    tmem_example_kernel[(1,)](x, y, M, N, num_warps=num_warps)
    torch.testing.assert_close(x, y, atol=0, rtol=0)
    return y


if __name__ == "__main__":
    y = run_tmem()
    print("tmem_copy_ok", y.sum().item())
```

运行要求 Blackwell。完整示例见 [`06-tcgen05.py`](code/triton/gluon/tutorials/06-tcgen05.py)。

### 一个 64 x 64 tile 的存储轨迹

`block=(64, 64)` 时，一个 block 正好覆盖 64 行和 64 列：

```text
(0, 0)   -> TMEM row 0,  col 0
(0, 63)  -> TMEM row 0,  col 63
(63, 0)  -> TMEM row 63, col 0
(63, 63) -> TMEM row 63, col 63
```

如果 `M=128`，同样的 layout 会形成两个 `64 x 64` block。因为每个 warp 只能寻址 32 行，最终 register layout 不是可以随意指定的；`tmem.get_reg_layout()` 给出的是能够完成 `tcgen05.st` 和 `tcgen05.ld` 的合法 register 分布。

## tcgen05 MMA 的 K-loop 轨迹

Blackwell 的 blocked matmul 可以简化为：

```text
for k:
    TMA load A[k], B[k] -> SMEM
    wait TMA mbarrier
    tcgen05_mma(A_smem, B_smem, acc_tmem, use_acc=(k != 0))
    tcgen05_commit(mma_bar)
    wait MMA mbarrier
```

第一次使用 `use_acc=False`，硬件忽略 TMEM 中原有的值，相当于高效清零 accumulator。后续使用 `use_acc=True`，把新的 `A @ B` 累加到当前 TMEM 值。

异步 pipeline 则把“等待当前 MMA”改为等待更早发出的 MMA：

```text
issue MMA(k)
while MMA(k) runs:
    TMA load operands(k+1)
wait enough earlier MMA
issue MMA(k+1)
```

Gluon 对 tcgen05 的隐式顺序保证包括：

- 两个 shape 和 accumulator dtype 相同的 `tcgen05_mma` 按发射顺序执行。
- `tcgen05_mma` 与紧随其后的 `tcgen05_commit` 保序。
- `tcgen05_copy` 与 `tcgen05_mma` 之间保序。

因此不需要在每一对 MMA 之间都插入显式 barrier。barrier 用来告诉普通线程“什么时候可以安全读取累加器或复用 operand buffer”，而不是限制 tensor pipe 自身。

## 官方 06 pipeline 完整拆解：U/V 双 tile、双缓冲 TMA 与 tcgen05 commit

### 本次讲解位置

章节：`chapter_gluon_tensor_cores`（算子开发 / Triton Gluon）<br>
小节：The 5th Generation TensorCore / Pipeline MMA<br>
知识点：U/V 两个 M tile、双缓冲 TMA、load barrier 与 MMA barrier 分离、`tcgen05_commit` 驱动 buffer 复用和 epilogue<br>
上次：非流水 tcgen05 blocked matmul 的 K-loop<br>
下次：persistent kernel 与 warp specialization<br>
PTX：`tcgen05.mma`、`tcgen05.commit`、`cp.async.bulk.tensor`、`mbarrier.try_wait.parity`<br>
完整代码：[官方归档脚本](code/triton/gluon/tutorials/06-tcgen05.py)

### 为什么现在讲这个

前一节的非流水 tcgen05 K-loop 已经能算对，但它的节奏是：

```text
load(k) -> wait load -> MMA(k) -> wait MMA -> load(k+1)
```

每次等 MMA 完成才发下一次 load，tensor core 和 TMA 仍然有大量时间互相等待。官方流水版本要解决两个具体问题：

1. 一个 CTA 只算一个 `128 x 128` 输出 tile 时，K-loop 中每轮只有一次 MMA，单次操作的等待粒度太粗。
2. 如果为了提前加载而直接覆盖 U、V 或 B buffer，TMA 可能覆盖 tensor core 仍在读取的 SMEM，结果会随机错误。

因此它一次计算同一 N 列上的两个 M tile，并为三块输入各准备两份 SMEM buffer。这样既能同时保住两个 accumulator 的吞吐，又能用 MMA barrier 精确判断某个 buffer 何时可以复用。

### 心智模型：一个 B tile 服务两个输出 tile

设 CTA 的 `off_m` 指向一对 M tile：

```text
U = A[off_m           : off_m + BLOCK_M,     k]
V = A[off_m + BLOCK_M : off_m + 2 * BLOCK_M, k]
B = B[k               : k + BLOCK_K,         off_n : off_n + BLOCK_N]
```

一次 K stage 产生两个独立的矩阵乘：

```text
UB_tile += U @ B
VB_tile += V @ B
```

这两个乘法共享 B。B 是公共操作数，不应加载两次。每个 K stage 需要三次 TMA load：

```text
U tile
V tile
B tile
```

却可以执行两次 MMA，所以官方注释把它概括为 `3:2 ratio of loads to MMAs`。代价是同一份 B buffer 有两个消费者，复用 B 之前必须同时确认 U MMA 和 V MMA 都读完了。

### 完整可运行示例：双 tile、双缓冲 tcgen05 pipeline

下面的文件 `tcgen05_pipeline_demo.py` 把官方 pipelined kernel 缩成一个可直接运行的最小程序。它保留了完整 kernel、host 侧 descriptor、launch 和数值验证。

```python
from __future__ import annotations

import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor
from triton.experimental.gluon.language.nvidia.blackwell import (
    TensorMemoryLayout,
    allocate_tensor_memory,
    fence_async_shared,
    mbarrier,
    tcgen05_commit,
    tcgen05_mma,
    tma,
)


def is_blackwell():
    target = triton.runtime.driver.active.get_current_target()
    return target.backend == "cuda" and torch.cuda.get_device_capability()[0] == 10


@gluon.jit
def get_and_increment(counter):
    return counter % 2, counter // 2 & 1, counter + 1


@gluon.jit
def blocked_matmul_pipelined_kernel(a_desc, b_desc, c_desc, num_warps: gl.constexpr):
    BLOCK_M: gl.constexpr = c_desc.block_type.shape[0]
    BLOCK_N: gl.constexpr = c_desc.block_type.shape[1]
    BLOCK_K: gl.constexpr = a_desc.block_type.shape[1]
    dtype: gl.constexpr = a_desc.dtype
    K = a_desc.shape[1]

    pid_m = gl.program_id(axis=0)
    pid_n = gl.program_id(axis=1)
    off_m = pid_m * (2 * BLOCK_M)
    off_n = pid_n * BLOCK_N

    u_bufs = gl.allocate_shared_memory(
        dtype, [2] + a_desc.block_type.shape, a_desc.layout
    )
    v_bufs = gl.allocate_shared_memory(
        dtype, [2] + a_desc.block_type.shape, a_desc.layout
    )
    b_bufs = gl.allocate_shared_memory(
        dtype, [2] + b_desc.block_type.shape, b_desc.layout
    )

    tmem_layout: gl.constexpr = TensorMemoryLayout([BLOCK_M, BLOCK_N], col_stride=1)
    ub_tmem = allocate_tensor_memory(gl.float32, [BLOCK_M, BLOCK_N], tmem_layout)
    vb_tmem = allocate_tensor_memory(gl.float32, [BLOCK_M, BLOCK_N], tmem_layout)

    mma_ub_bars = gl.allocate_shared_memory(
        gl.int64, [2, 1], mbarrier.MBarrierLayout()
    )
    mma_vb_bars = gl.allocate_shared_memory(
        gl.int64, [2, 1], mbarrier.MBarrierLayout()
    )
    load_ub_bars = gl.allocate_shared_memory(
        gl.int64, [2, 1], mbarrier.MBarrierLayout()
    )
    load_v_bars = gl.allocate_shared_memory(
        gl.int64, [2, 1], mbarrier.MBarrierLayout()
    )
    for i in gl.static_range(2):
        mbarrier.init(mma_ub_bars.index(i), count=1)
        mbarrier.init(mma_vb_bars.index(i), count=1)
        mbarrier.init(load_ub_bars.index(i), count=1)
        mbarrier.init(load_v_bars.index(i), count=1)

    load_counter = 0
    mma_counter = 0
    k = 0
    ub_acc = False
    vb_acc = False

    load_index, load_phase, load_counter = get_and_increment(load_counter)
    load_ub_bar = load_ub_bars.index(load_index)
    mbarrier.expect(
        load_ub_bar,
        a_desc.block_type.nbytes + b_desc.block_type.nbytes,
    )
    tma.async_load(a_desc, [off_m, k], load_ub_bar, u_bufs.index(load_index))
    tma.async_load(b_desc, [k, off_n], load_ub_bar, b_bufs.index(load_index))

    load_v_bar = load_v_bars.index(load_index)
    mbarrier.expect(load_v_bar, a_desc.block_type.nbytes)
    tma.async_load(
        a_desc,
        [off_m + BLOCK_M, k],
        load_v_bar,
        v_bufs.index(load_index),
    )
    k += BLOCK_K

    load_index, load_phase, load_counter = get_and_increment(load_counter)
    load_ub_bar = load_ub_bars.index(load_index)
    mbarrier.expect(
        load_ub_bar,
        a_desc.block_type.nbytes + b_desc.block_type.nbytes,
    )
    tma.async_load(a_desc, [off_m, k], load_ub_bar, u_bufs.index(load_index))
    tma.async_load(b_desc, [k, off_n], load_ub_bar, b_bufs.index(load_index))

    load_v_bar = load_v_bars.index(load_index)
    mbarrier.expect(load_v_bar, a_desc.block_type.nbytes)
    tma.async_load(
        a_desc,
        [off_m + BLOCK_M, k],
        load_v_bar,
        v_bufs.index(load_index),
    )
    k += BLOCK_K

    for _ in range(gl.cdiv(K, BLOCK_K) - 2):
        mma_index, mma_phase, mma_counter = get_and_increment(mma_counter)

        mbarrier.wait(load_ub_bars.index(mma_index), mma_phase)
        tcgen05_mma(
            u_bufs.index(mma_index),
            b_bufs.index(mma_index),
            ub_tmem,
            use_acc=ub_acc,
        )
        tcgen05_commit(mma_ub_bars.index(mma_index))
        ub_acc = True

        mbarrier.wait(load_v_bars.index(mma_index), mma_phase)
        tcgen05_mma(
            v_bufs.index(mma_index),
            b_bufs.index(mma_index),
            vb_tmem,
            use_acc=vb_acc,
        )
        tcgen05_commit(mma_vb_bars.index(mma_index))
        vb_acc = True

        load_index, load_phase, load_counter = get_and_increment(load_counter)
        mbarrier.wait(mma_ub_bars.index(mma_index), mma_phase)
        load_ub_bar = load_ub_bars.index(load_index)
        mbarrier.expect(
            load_ub_bar,
            a_desc.block_type.nbytes + b_desc.block_type.nbytes,
        )
        tma.async_load(
            a_desc,
            [off_m, k],
            load_ub_bar,
            u_bufs.index(load_index),
        )

        mbarrier.wait(mma_vb_bars.index(mma_index), mma_phase)
        tma.async_load(
            b_desc,
            [k, off_n],
            load_ub_bar,
            b_bufs.index(load_index),
        )
        load_v_bar = load_v_bars.index(load_index)
        mbarrier.expect(load_v_bar, a_desc.block_type.nbytes)
        tma.async_load(
            a_desc,
            [off_m + BLOCK_M, k],
            load_v_bar,
            v_bufs.index(load_index),
        )
        k += BLOCK_K

    mma_index, mma_phase, mma_counter = get_and_increment(mma_counter)
    ub_bar = mma_ub_bars.index(mma_index)
    vb_bar = mma_vb_bars.index(mma_index)
    epilogue_phase = mma_phase

    mbarrier.wait(load_ub_bars.index(mma_index), mma_phase)
    tcgen05_mma(
        u_bufs.index(mma_index),
        b_bufs.index(mma_index),
        ub_tmem,
        use_acc=True,
    )

    mbarrier.wait(load_v_bars.index(mma_index), mma_phase)
    tcgen05_mma(
        v_bufs.index(mma_index),
        b_bufs.index(mma_index),
        vb_tmem,
        use_acc=True,
    )

    mma_index, mma_phase, mma_counter = get_and_increment(mma_counter)
    mbarrier.wait(load_ub_bars.index(mma_index), mma_phase)
    tcgen05_mma(
        u_bufs.index(mma_index),
        b_bufs.index(mma_index),
        ub_tmem,
        use_acc=True,
    )
    tcgen05_commit(ub_bar)

    mbarrier.wait(load_v_bars.index(mma_index), mma_phase)
    tcgen05_mma(
        v_bufs.index(mma_index),
        b_bufs.index(mma_index),
        vb_tmem,
        use_acc=True,
    )
    tcgen05_commit(vb_bar)

    mbarrier.wait(ub_bar, epilogue_phase)
    c_smem = gl.allocate_shared_memory(
        dtype, c_desc.block_type.shape, c_desc.layout
    )
    ub = ub_tmem.load()
    c_smem.store(ub.to(dtype))
    fence_async_shared()
    tma.async_store(c_desc, [off_m, off_n], c_smem)

    mbarrier.wait(vb_bar, epilogue_phase)
    vb = vb_tmem.load()
    tma.store_wait(pendings=0)
    c_smem.store(vb.to(dtype))
    fence_async_shared()
    tma.async_store(c_desc, [off_m + BLOCK_M, off_n], c_smem)
    tma.store_wait(pendings=0)


def blocked_matmul_pipelined(A, B, C, BLOCK_M, BLOCK_N, BLOCK_K, num_warps):
    M, N = C.shape
    a_layout = gl.NVMMASharedLayout.get_default_for(
        [BLOCK_M, BLOCK_K], gl.float16
    )
    b_layout = gl.NVMMASharedLayout.get_default_for(
        [BLOCK_K, BLOCK_N], gl.float16
    )
    c_layout = gl.NVMMASharedLayout.get_default_for(
        [BLOCK_M, BLOCK_N], gl.float16
    )
    a_desc = TensorDescriptor.from_tensor(A, [BLOCK_M, BLOCK_K], a_layout)
    b_desc = TensorDescriptor.from_tensor(B, [BLOCK_K, BLOCK_N], b_layout)
    c_desc = TensorDescriptor.from_tensor(C, [BLOCK_M, BLOCK_N], c_layout)

    grid = (triton.cdiv(M, 2 * BLOCK_M), triton.cdiv(N, BLOCK_N))
    blocked_matmul_pipelined_kernel[grid](
        a_desc,
        b_desc,
        c_desc,
        num_warps=num_warps,
    )


def main():
    if not is_blackwell():
        raise RuntimeError("This example requires a Blackwell NVIDIA GPU")

    torch.manual_seed(0)
    M, N, K = 256, 256, 256
    BLOCK_M, BLOCK_N, BLOCK_K = 128, 128, 64
    A = torch.randn(M, K, device="cuda", dtype=torch.float16)
    B = torch.randn(K, N, device="cuda", dtype=torch.float16)
    C = torch.empty(M, N, device="cuda", dtype=torch.float16)

    blocked_matmul_pipelined(
        A,
        B,
        C,
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        num_warps=8,
    )
    torch.testing.assert_close(A @ B, C, rtol=1e-3, atol=1e-1)
    print("tcgen05 pipelined matmul ok")


if __name__ == "__main__":
    main()
```

运行：

```bash
python tcgen05_pipeline_demo.py
```

在符合条件的 Blackwell GPU 上，数值检查通过后输出：

```text
tcgen05 pipelined matmul ok
```

### 为什么一个 CTA 处理两个 M tile

grid 是：

```python
grid = (triton.cdiv(M, 2 * BLOCK_M), triton.cdiv(N, BLOCK_N))
```

若 `M=N=256`、`BLOCK_M=BLOCK_N=128`，则：

```text
grid = (ceil(256 / 256), ceil(256 / 128))
     = (1, 2)
```

第二个 CTA 的 `off_n=128`。该 CTA 同时计算：

```text
C[0:128,   128:256] = A[0:128,   :] @ B[:, 128:256]
C[128:256, 128:256] = A[128:256, :] @ B[:, 128:256]
```

这两个输出 tile 使用完全相同的 B。一次 load B，两次 MMA，是这套 schedule 的第一个收益来源。它也解释为什么要有两个 TMEM accumulator：

```text
ub_tmem: C[off_m           : off_m + BLOCK_M,     off_n : off_n + BLOCK_N]
vb_tmem: C[off_m + BLOCK_M : off_m + 2 * BLOCK_M, off_n : off_n + BLOCK_N]
```

### 双缓冲的对象到底有哪些

源码中有三组输入 buffer：

```python
u_bufs: [2, BLOCK_M, BLOCK_K]
v_bufs: [2, BLOCK_M, BLOCK_K]
b_bufs: [2, BLOCK_K, BLOCK_N]
```

每个 slot 的生命周期分成两个问题：

| 问题 | 需要等待 | 原因 |
|---|---|---|
| TMA 是否已经写完 slot？ | 对应的 `load_*_bars[slot]` | MMA 不能读半块数据 |
| MMA 是否已经读完 slot？ | 对应的 `mma_*_bars[slot]` | TMA 不能覆盖仍在读的 SMEM |

因此一个 slot 的完整生命周期是：

```text
empty
  -> expect + TMA load
  -> load barrier 完成
  -> MMA 读取
  -> commit
  -> MMA barrier 完成
  -> 可以重新 TMA load
```

`load` barrier 和 `mma` barrier 不能合并。前者回答“数据到了吗”，后者回答“消费者走了吗”；它们是相反方向的所有权。即便都使用 `count=1` 的 mbarrier，也必须分别初始化和分别翻转 phase。

### `get_and_increment`：slot 与 phase 的唯一分配器

```python
@gluon.jit
def get_and_increment(counter):
    return counter % 2, counter // 2 & 1, counter + 1
```

返回值的顺序是：

```text
counter % 2        -> slot index: 0, 1, 0, 1, ...
counter // 2 & 1   -> phase: 0, 0, 1, 1, 0, 0, ...
counter + 1        -> 下一次的 counter
```

具体轨迹：

| counter | slot | phase | next | 对应 K tile |
|---:|---:|---:|---:|---|
| 0 | 0 | 0 | 1 | K1 |
| 1 | 1 | 0 | 2 | K2 |
| 2 | 0 | 1 | 3 | K3 |
| 3 | 1 | 1 | 4 | K4 |
| 4 | 0 | 0 | 5 | K5 |
| 5 | 1 | 0 | 6 | K6 |
| 6 | 0 | 1 | 7 | K7 |

同一个 slot 每被复用一次，phase 就翻转一次。slot 0 的 phase 序列是 `0, 1, 0, 1, ...`；slot 1 也是同样规律，只是第一次使用从 phase 0 开始。

prologue 中返回的 `load_phase` 没有参与 wait，因为 prologue 的职责只是发起前两次 load。主循环中的 `mma_phase` 同时用于等待同名 slot 的 load barrier。这里能这样写，是因为 `load_counter` 和 `mma_counter` 在这个固定 schedule 中同步推进，index 与 phase 一一对齐；换成更复杂的 producer/consumer 结构后，两个 counter 应分别维护。

### Prologue：先把两个 slot 都填满

第一段加载 K1：

```text
slot 0: U1, B1, V1
```

`load_ub_bars[0]` 由 U1 和 B1 共同完成，因此 `expect` 的字节数是：

```text
U tile bytes + B tile bytes
```

`load_v_bars[0]` 只跟踪 V1，因此只 expect V tile bytes。

第二段加载 K2：

```text
slot 1: U2, B2, V2
```

此时两个 slot 都有可消费的数据。主循环每轮先消费已经加载的 tile，再为两轮之前的 slot 发起新 load。

### 主循环：每轮同时推进 MMA 和下一轮 load

主循环可以压缩成下面的顺序：

```text
wait load_ub[slot]
issue U MMA
commit mma_ub[slot]

wait load_v[slot]
issue V MMA
commit mma_vb[slot]

wait mma_ub[slot]
load U(next), but do not overwrite B yet

wait mma_vb[slot]
load B(next), V(next)
```

前两步是消费者路径，后两步是生产者路径。它们在同一轮里交错出现，因此没有独立的 producer warpgroup，也没有额外的 block-level barrier。

这里最容易看漏的是 B 的共享关系：

```text
slot s 的 B
  -> U MMA 读取
  -> V MMA 读取
```

所以 `wait(mma_ub[slot])` 之后只能覆盖 `u_bufs[slot]`，不能立刻覆盖 `b_bufs[slot]`。必须等 `wait(mma_vb[slot])` 之后，才能安全写 B，同时写 V。代码正是按这个顺序拆开两次 wait 的。

### K_TILES=4 的完整时间线

取 `K=256`、`BLOCK_K=64`，于是 `K_TILES=4`：

```text
prologue:
  load K1 -> slot 0
  load K2 -> slot 1

main iteration 0:
  slot 0, phase 0
  MMA U1 and V1
  commit 到 mma_ub[0], mma_vb[0]
  wait mma_ub[0] and mma_vb[0]
  load K3 -> slot 0

main iteration 1:
  slot 1, phase 0
  MMA U2 and V2
  commit 到 mma_ub[1], mma_vb[1]
  wait mma_ub[1] and mma_vb[1]
  load K4 -> slot 1

final tile K3:
  slot 0, phase 1
  wait load[0] phase 1
  MMA U3 and V3, use_acc=True

final tile K4:
  slot 1, phase 1
  wait load[1] phase 1
  MMA U4 and V4, use_acc=True
  commit 到最终 ub_bar = mma_ub[0]
  commit 到最终 vb_bar = mma_vb[0]
  wait 最终 barrier, phase 1
  TMEM -> register -> SMEM -> TMA store
```

最后两次 commit 的 barrier 使用的是前面保存的 `ub_bar`、`vb_bar`。它们对应 slot 0，但 commit 完成事件覆盖此前的 U/V MMA 序列，因此最后等待 `ub_bar` 和 `vb_bar` 时，最终累加器已经可以读取。这里不能只等 K4 那一次 MMA，因为 K3 的结果也必须在 epilogue 读取前完成。

教程中的这个 schedule 默认 `K_TILES >= 3`，并且只对 `BLOCK_M = BLOCK_N = 128` 做专门优化。若要推广到任意 K tile 数，必须把 final 阶段的 `use_acc` 和最后一个未提交 MMA 的处理重新设计，不能只改循环上界。

### Epilogue：为什么 TMEM 不能直接被 TMA store

最后一段路径是：

```text
ub_tmem / vb_tmem
    -> TMEM load 到 register
    -> cast 到 fp16
    -> store 到 SMEM c_smem
    -> fence_async_shared()
    -> TMA store 到 GMEM
```

TMA store 读取的是 SMEM，不是 TMEM，所以不能直接把 TMEM descriptor 交给 TMA。TMEM 的结果必须先经 `tcgen05.ld` 进入 register，再由普通 store 写进 SMEM。写完 SMEM 后需要 `fence_async_shared()`，让后续 async proxy 的 TMA store 看见这笔普通写入。

第一次 TMA store 与第二个 TMEM load 的等待顺序也值得注意：

```text
load UB -> store UB 到 c_smem -> launch TMA store UB
wait VB MMA -> load VB -> tma.store_wait(0) -> store VB 到 c_smem -> launch TMA store VB
```

`tma.store_wait(pendings=0)` 在复用同一块 `c_smem` 之前等待前一个 TMA store 读完它。少了这一步，第二个输出 tile 的 SMEM 写入可能覆盖第一个 store 仍在读取的数据。

### Barrier 所有权表

| Barrier | 谁让它完成 | 谁等待它 | 等待后能做什么 |
|---|---|---|---|
| `load_ub_bars[s]` | U、B 两次 TMA 的 transaction bytes | MMA 前 | 读取 U 和 B |
| `load_v_bars[s]` | V 的 TMA transaction bytes | MMA 前 | 读取 V |
| `mma_ub_bars[s]` | `tcgen05_commit` 对应的 U MMA | 下一轮 TMA 前 | 覆盖 `u_bufs[s]` |
| `mma_vb_bars[s]` | `tcgen05_commit` 对应的 V MMA | 下一轮 TMA 前 | 覆盖 `b_bufs[s]` 和 `v_bufs[s]` |

`mbarrier.init(..., count=1)` 只定义“多少个 arrival 完成当前 phase”。它不表示 barrier 只能使用一次。每次 phase 完成后，下一次 arrival 进入新 phase，所以同一个 slot 可以反复复用，但等待方必须使用与本次使用对应的 parity。

### 常见错误与可观察症状

| 错误 | 典型症状 | 根因与修复 |
|---|---|---|
| B 只等 U MMA 完成就覆盖 | 结果随机错误，单独跑小矩阵有时能过 | B 还被 V MMA 读取；先等 `mma_vb` 再覆盖 B |
| 复用 load 和 MMA 的同一个 barrier | 首次可能正常，后续永久卡住或随机乱序 | 两类完成的 phase 不同；分开 barrier 并分别 init |
| 忘记翻转 slot phase | 第二个 K tile 永远等不到，或提前通过 | `get_and_increment` 必须同时推进 index 和 phase |
| 主循环只 load 下一块，不等待旧 buffer 的 MMA | 数据竞争，结果随运行次数波动 | `mma_*_bars` 是 buffer 可复用的许可证 |
| Epilogue 在 commit 后不 wait | 读到旧 accumulator，最终结果偶尔少一项 | `tcgen05_commit` 是异步完成信号，不是同步点 |
| TMA store 前少 `fence_async_shared()` | 输出旧 SMEM 或部分新数据 | 普通 store 到 async proxy 之间需要 fence |
| 第二个输出覆盖 `c_smem` 前不 `store_wait(0)` | 第一个输出 tile 随机损坏 | TMA store 仍在读 SMEM |

### 自测

1. **为什么每个 K stage 是 3 次 TMA load、2 次 MMA？**
   - 答：需要加载 U、V、B 三块输入；U 和 V 分别与同一个 B 做 MMA，因此产生 UB 和 VB 两个输出 tile。
2. **`counter=2` 时 `get_and_increment` 返回什么？**
   - 答：返回 `slot=0, phase=1, next_counter=3`。slot 0 第二次被使用，所以 phase 从 0 翻到 1。
3. **为什么等待 `mma_ub` 后不能马上覆盖 B？**
   - 答：B 还被 V MMA 读取。必须先等待 `mma_vb`，才能安全覆盖 B。
4. **为什么 load barrier 和 MMA barrier 不能复用？**
   - 答：一个表示 TMA 已把数据写进 SMEM，另一个表示 tensor core 已读完 SMEM。完成来源、phase 推进和消费者都不同，混用会让后续 wait 的 parity 失去意义。
5. **Epilogue 为什么必须经过 register 和 SMEM？**
   - 答：TMA store 的源是 SMEM，不能直接读 TMEM；最终 accumulator 需要先由 `tcgen05.ld` 从 TMEM 取到 register，再存入 SMEM 交给 TMA。

### 验证与运行边界

静态检查：

```bash
python3 -m py_compile operators/code/triton/gluon/tutorials/06-tcgen05.py
git diff --check
```

运行检查要求 Blackwell 计算能力 10.x 和同等版本的 Triton Gluon：

```bash
python tcgen05_pipeline_demo.py
```

预期输出：

```text
tcgen05 pipelined matmul ok
```

当前文档环境只做静态验证。非 Blackwell GPU 会在 `is_blackwell()` 处失败，这表示硬件边界不符合教程要求，不是 kernel 的数值结果。

## 官方性能轨迹

| 版本 | Hopper WGMMA | Blackwell tcgen05 |
|---|---:|---:|
| 非异步或无流水起点 | 约 `466 TFLOPS` | `1019.46 TFLOPS` |
| 最佳流水版本 | 约 `495.31 TFLOPS` | `1113.49 TFLOPS` |

这些是上游教程在对应 GPU 上给出的结果。它们不能横向解释成“Blackwell 比 Hopper 快两倍”，因为矩阵 shape、block 配置和 pipeline 深度不同。

## 常见错误与症状

| 错误 | 典型症状 | 根因与修复 |
|---|---|---|
| WGMMA 的返回依赖没有穿过 `warpgroup_mma_wait` | 随机旧值、race、结果偶尔错误 | 使用 `acc = warpgroup_mma_wait(..., deps=(acc,))` |
| B 放在 register | 编译失败或 API 拒绝 | WGMMA 的 B 必须来自 SMEM |
| accumulator 不用 `NVMMADistributedLayout` | layout 校验失败 | accumulator layout 决定 instruction shape |
| BLOCK_M 小于 64 | 无法满足 warp atom | WGMMA 最少 4 warps，M 至少 64 |
| 普通 SMEM store 后直接 WGMMA | 读到旧数据 | 插入 `fence_async_shared()` |
| TMEM 按元素而不是按列理解 | allocation size 错误 | 分配单位是列，且为限定范围内的 2 的幂 |
| 用普通 tensor 当 accumulator | 类型错误 | `tcgen05_mma` 的 accumulator 必须是 TMEM descriptor |
| TMEM 写完立即读 | 读到旧 accumulator | `tcgen05_commit` 后用 mbarrier 等完成 |
| TMA 与 MMA 复用同一个 mbarrier 而不重新初始化 | undefined behavior | 分开 barrier，或 invalidate 后重新 init |
| `use_acc` 第一轮设成 True | 结果包含未定义初值 | 第一轮用 False，后续用 True |

## 硬件与验证边界

- [`05-wgmma.py`](code/triton/gluon/tutorials/05-wgmma.py) 只适用于 Hopper。
- [`06-tcgen05.py`](code/triton/gluon/tutorials/06-tcgen05.py) 只适用于 Blackwell 计算能力 10.x。
- 本机没有 Triton runtime 和 CUDA GPU，只做了源码静态检查；上文 benchmark 均是上游结果。真正的 warpgroup 数量、TMEM 分配、mbarrier phase 和吞吐必须在目标 GPU 上运行验证。

## 自我检查

1. **WGMMA 的 A、B、C 分别可以在哪里？**
   - 答：A 可以在 register 或 SMEM，B 必须在 SMEM，C/D 在 register。
2. **为什么 `warpgroup_mma_wait` 必须返回 accumulator？**
   - 答：异步写需要通过 `deps` 和返回值建立数据依赖，后续读才能看见完成后的结果。
3. **TMEM 的 128 x 512 是什么单位？**
   - 答：每个 SM 有 128 行、512 列 32-bit cell；分配按列，Gluon load/store 需要 4 或 8 warps。
4. **`tcgen05_commit(bar)` 的语义是什么？**
   - 答：此前发出的 tcgen05 操作全部完成后，让指定 mbarrier 发生一次 arrival。
5. **为什么 tcgen05 与 WGMMA 都需要“异步完成等待”，但等待对象不同？**
   - 答：两者都异步访问 operand 并写 accumulator；WGMMA 用 commit group 跟踪 register accumulator，tcgen05 用 mbarrier 跟踪 TMEM accumulator 和 tensor-pipe 完成。

## 官方 05 blocked matmul：一个 CTA 如何完成一个二维输出 tile

### 本次讲解位置

章节：`chapter_gluon_tensor_cores`（算子开发 / Triton Gluon）<br>
小节：Official Tutorial 05 / Blocked Matmul<br>
知识点：从二维 grid、K-loop、TMA 尾块、mbarrier parity 到 WGMMA store 的完整 baseline<br>
上次：单个输出 tile 上的 WGMMA、instruction shape 与 `warpgroup_mma_wait`<br>
下次：为 A/B 增加多阶段 buffer，让下一轮 TMA load 与当前 WGMMA overlap<br>
PTX：`cp.async.bulk.tensor.*`、`mbarrier.arrive.expect_tx`、`wgmma.mma_async`<br>
官方源码：[`05-wgmma.py`](code/triton/gluon/tutorials/05-wgmma.py)<br>
官方教程：[Warp-Group MMA](https://triton-lang.org/main/getting-started/tutorials/gluon/wgmma.html)

### 为什么现在讲这个

上一节能在固定的一块 `BLOCK_M x BLOCK_N` 上发出 WGMMA，但真实 GEMM 的 C 是二维矩阵。一个 CTA 只负责其中一个输出 tile，因此必须先回答三个具体问题：

```text
1. 当前 CTA 应该处理 C 的哪一块？
2. 这一块需要沿 K 方向读取哪些 A/B slice？
3. 当 M、N 或 K 不能被 block size 整除时，多出来的尾块由谁清零或裁剪？
```

如果 grid、坐标或 phase 任意一项写错，症状不是简单的“结果差一点”，而是多个 CTA 写错位置、第二个 K iteration 提前读取、或者最后一次 load 永久等不到 completion。本节把官方 baseline 当作一条完整生产路径来走，并为下一节的双缓冲流水线建立坐标系。

### 心智模型：CTA 是二维 tile worker

把 C 想成由 `BLOCK_M x BLOCK_N` 小方格组成的大棋盘：

```text
grid = (ceil(M / BLOCK_M), ceil(N / BLOCK_N))

pid_m = gl.program_id(axis=0)  -> 输出 tile 在 M 方向的编号
pid_n = gl.program_id(axis=1)  -> 输出 tile 在 N 方向的编号

off_m = pid_m * BLOCK_M
off_n = pid_n * BLOCK_N
```

一个 CTA 的主循环不是“把整个 K 乘完才产生中间结果”，而是：

```text
for k in range(0, K, BLOCK_K):
    加载 A[off_m : off_m + BLOCK_M, k : k + BLOCK_K]
    加载 B 的对应 tile
    等 TMA 完成
    acc += A_tile @ B_tile

把 fp32 acc 转成输出 dtype
通过 TMA 将 C[off_m, off_n] tile 写回
```

这里的 `acc` 一直是寄存器中的 `BLOCK_M x BLOCK_N` 累加器。K 方向被切成若干段，但 CTA 始终只负责同一个 C tile。

### 完整可运行代码

下面是可以单独保存为 `blocked_matmul_wgmma_demo.py` 的完整版本，包含 imports、helper、kernel、host 启动和数值校验。它需要 Hopper GPU。

```python
from __future__ import annotations

import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor
from triton.experimental.gluon.language.nvidia.hopper import (
    fence_async_shared,
    mbarrier,
    tma,
    warpgroup_mma,
    warpgroup_mma_wait,
)


def is_hopper() -> bool:
    target = triton.runtime.driver.active.get_current_target()
    return target.backend == "cuda" and torch.cuda.get_device_capability()[0] == 9


@gluon.constexpr_function
def get_warps_per_cta(BLOCK_M, BLOCK_N, num_warps):
    warps_per_cta = [4, 1]
    m = 16
    while warps_per_cta[0] * warps_per_cta[1] != num_warps:
        if BLOCK_M > m * warps_per_cta[0]:
            warps_per_cta[0] *= 2
        else:
            warps_per_cta[1] *= 2
    return warps_per_cta


@gluon.constexpr_function
def get_instr_shape_n(BLOCK_M, BLOCK_N, num_warps):
    m = 16
    m_reps = triton.cdiv(BLOCK_M, m)
    n_reps = triton.cdiv(num_warps, m_reps)
    max_n = max(BLOCK_N // n_reps, 8)
    n = 256
    while n > max_n or BLOCK_N % n != 0:
        n -= 8
    assert n >= 8, "expected to find a valid n"
    return n


@gluon.constexpr_function
def pick_wgmma_layout(dtype, BLOCK_M, BLOCK_N, num_warps):
    m = 16
    k = 256 // dtype.primitive_bitwidth
    n = get_instr_shape_n(BLOCK_M, BLOCK_N, num_warps)
    warps_per_cta = get_warps_per_cta(BLOCK_M, BLOCK_N, num_warps)
    return gl.NVMMADistributedLayout(
        version=[3, 0],
        warps_per_cta=warps_per_cta,
        instr_shape=[m, n, k],
    )


@gluon.jit
def blocked_matmul_kernel(
    a_desc,
    b_desc,
    c_desc,
    TRANSPOSE_B: gl.constexpr,
    num_warps: gl.constexpr,
):
    BLOCK_M: gl.constexpr = c_desc.block_type.shape[0]
    BLOCK_N: gl.constexpr = c_desc.block_type.shape[1]
    BLOCK_K: gl.constexpr = a_desc.block_type.shape[1]
    dtype: gl.constexpr = a_desc.dtype
    K = a_desc.shape[1]

    a_smem = gl.allocate_shared_memory(dtype, a_desc.block_type.shape, a_desc.layout)
    b_smem = gl.allocate_shared_memory(dtype, b_desc.block_type.shape, b_desc.layout)

    pid_m = gl.program_id(axis=0)
    pid_n = gl.program_id(axis=1)
    off_m = pid_m * BLOCK_M
    off_n = pid_n * BLOCK_N

    mma_layout: gl.constexpr = pick_wgmma_layout(
        dtype, BLOCK_M, BLOCK_N, num_warps
    )
    acc = gl.zeros((BLOCK_M, BLOCK_N), dtype=gl.float32, layout=mma_layout)

    bar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(bar, count=1)
    phase = 0

    for k in range(0, K, BLOCK_K):
        mbarrier.expect(
            bar,
            a_desc.block_type.nbytes + b_desc.block_type.nbytes,
        )
        tma.async_load(a_desc, [off_m, k], bar, a_smem)
        if TRANSPOSE_B:
            tma.async_load(b_desc, [off_n, k], bar, b_smem)
        else:
            tma.async_load(b_desc, [k, off_n], bar, b_smem)

        mbarrier.wait(bar, phase=phase)
        phase ^= 1

        if TRANSPOSE_B:
            b = b_smem.permute((1, 0))
        else:
            b = b_smem

        acc = warpgroup_mma(a_smem, b, acc, is_async=True)
        acc = warpgroup_mma_wait(num_outstanding=0, deps=(acc,))

    mbarrier.invalidate(bar)

    c_smem = gl.allocate_shared_memory(dtype, c_desc.block_type.shape, c_desc.layout)
    c_smem.store(acc.to(dtype))
    fence_async_shared()
    tma.async_store(c_desc, [off_m, off_n], c_smem)
    tma.store_wait(pendings=0)


def blocked_matmul(
    A,
    B,
    C,
    BLOCK_M,
    BLOCK_N,
    BLOCK_K,
    TRANSPOSE_B,
    num_warps,
):
    M, N = C.shape

    a_layout = gl.NVMMASharedLayout.get_default_for(
        [BLOCK_M, BLOCK_K], gl.float16
    )
    a_desc = TensorDescriptor.from_tensor(A, [BLOCK_M, BLOCK_K], a_layout)

    b_block_shape = [BLOCK_N, BLOCK_K] if TRANSPOSE_B else [BLOCK_K, BLOCK_N]
    b_layout = gl.NVMMASharedLayout.get_default_for(b_block_shape, gl.float16)
    b_desc = TensorDescriptor.from_tensor(B, b_block_shape, b_layout)

    c_layout = gl.NVMMASharedLayout.get_default_for(
        [BLOCK_M, BLOCK_N], gl.float16
    )
    c_desc = TensorDescriptor.from_tensor(C, [BLOCK_M, BLOCK_N], c_layout)

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    blocked_matmul_kernel[grid](
        a_desc,
        b_desc,
        c_desc,
        TRANSPOSE_B,
        num_warps=num_warps,
    )


if __name__ == "__main__":
    if not is_hopper():
        raise RuntimeError("This demo requires a Hopper NVIDIA GPU")

    M, N, K = 208, 416, 304
    BLOCK_M, BLOCK_N, BLOCK_K = 128, 128, 64
    torch.manual_seed(0)

    for transpose_b in (False, True):
        A = torch.randn(M, K, device="cuda", dtype=torch.float16)
        B = torch.randn(
            (N, K) if transpose_b else (K, N),
            device="cuda",
            dtype=torch.float16,
        )
        C = torch.empty(M, N, device="cuda", dtype=torch.float16)

        blocked_matmul(
            A,
            B,
            C,
            BLOCK_M,
            BLOCK_N,
            BLOCK_K,
            transpose_b,
            num_warps=4,
        )

        C_ref = A @ (B.T if transpose_b else B)
        torch.testing.assert_close(C_ref, C, rtol=1e-3, atol=1e-1)
        print(f"TRANSPOSE_B={transpose_b} ok, sum={C.sum().item()}")
```

预期输出形如：

```text
TRANSPOSE_B=False ok, sum=...
TRANSPOSE_B=True ok, sum=...
```

### 三个 constexpr helper 在选什么

这三个 helper 都发生在编译期。它们不是“调优建议”，而是在构造一个合法的 `NVMMADistributedLayout`。

#### `get_warps_per_cta`

WGMMA 的最小 warp 切分 atom 是：

```text
[4, 1]
```

也就是至少 4 个 warp 沿 M 方向覆盖 `m=16` 的 instruction row。以 `BLOCK_M=64, BLOCK_N=64, num_warps=8` 为例：

```text
初始 [4, 1]
BLOCK_M > 16 * 4 ? 64 > 64 -> False
因此第二次沿 N 翻倍，得到 [4, 2]
```

如果 `BLOCK_M=128`，则第一步得到 `[8, 1]`。代码只在 M 方向还能容纳新的 warp 分区时增加 M；否则增加 N，避免让两个 warp 覆盖同一组 M 数据而产生广播式冗余工作。

#### `get_instr_shape_n`

FP16 的 `k` 固定为：

```text
256 / 16 = 16
```

但每个 `wgmma.mma_async` 的 `n` 不能随便取。`get_instr_shape_n` 用两个边界缩小它：

```text
m_reps = ceil(BLOCK_M / 16)
n_reps = ceil(num_warps / m_reps)
max_n  = max(BLOCK_N / n_reps, 8)
```

然后从 256 开始每次减 8，直到同时满足：

```text
n <= max_n
BLOCK_N % n == 0
```

例如 `BLOCK_M=128, BLOCK_N=128, num_warps=4`：

```text
m_reps = ceil(128 / 16) = 8
n_reps = ceil(4 / 8) = 1
max_n  = max(128 / 1, 8) = 128
n      从 256 递减到 128
```

最终得到 `instr_shape=[16, 128, 16]`。如果选择 `n=256`，它大于 `BLOCK_N`，一个 instruction 就会越过 tile 边界。

#### `pick_wgmma_layout`

最后把三个量放进 accumulator layout：

```python
gl.NVMMADistributedLayout(
    version=[3, 0],
    warps_per_cta=[4, 1],
    instr_shape=[16, 128, 16],
)
```

`acc` 是 `[128, 128]` 的 fp32 tensor，但物理上分散在 warp group 的寄存器里。`mma_layout` 规定的正是哪一条 instruction 结果由哪些 lane 持有。

### 从 host descriptor 到 CTA grid

host 侧构造三个 descriptor：

| Descriptor | block shape | 全局 tensor |
|---|---|---|
| A | `[BLOCK_M, BLOCK_K]` | `[M, K]` |
| B，非转置 | `[BLOCK_K, BLOCK_N]` | `[K, N]` |
| B，转置 | `[BLOCK_N, BLOCK_K]` | `[N, K]` |
| C | `[BLOCK_M, BLOCK_N]` | `[M, N]` |

grid 是：

```python
grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
```

`program_id(axis=0)` 对应 M 方向，`program_id(axis=1)` 对应 N 方向。这里不能把两个 axis 交换后仍然使用原来的 `off_m/off_n`，否则 CTA 会从错误的 A/B 区域取数，甚至把不同计算结果写到相同 C tile。

### TMA load 如何制造 barrier completion

每次 K iteration 做四步：

```text
1. mbarrier.expect(bar, A_bytes + B_bytes)
2. tma.async_load(A slice, bar, a_smem)
3. tma.async_load(B slice, bar, b_smem)
4. mbarrier.wait(bar, phase)
```

`mbarrier.expect` 同时完成两件事：先为当前 phase 记下一笔预期传输字节数，并消耗 barrier 初始 count 的一次 arrival；之后每次 TMA byte completion 都递减这笔 tx count。A、B 都传完后，phase 才能完成。

若 expect 的字节数小于实际传输量，barrier 可能过早完成，WGMMA 读到尚未写完的 SMEM；若大于实际传输量，最后一次 wait 会长期等待。这里的正确总量是：

```text
A bytes = BLOCK_M * BLOCK_K * 2
B bytes = BLOCK_N * BLOCK_K * 2
```

### phase 为什么要每次翻转

`mbarrier` 不是一次性锁。它会沿 phase parity `0 -> 1 -> 0 -> ...` 循环复用。`wait` 时要传入“我正在等待的那个 phase 的 parity”。

以 `K=304, BLOCK_K=64` 为例，循环有 5 次：

| iteration | `k` | wait phase | 完成后 `phase` |
|---:|---:|---:|---:|
| 0 | 0 | 0 | 1 |
| 1 | 64 | 1 | 0 |
| 2 | 128 | 0 | 1 |
| 3 | 192 | 1 | 0 |
| 4 | 256 | 0 | 1 |

如果循环里永远写 `phase=0`，第一次会等待正确的 phase；从第二次开始，可能会命中已经完成的旧 phase 而立即返回。这正是“偶发读旧数据”的根源之一。

### 具体执行轨迹：M=208, N=416, K=304

使用：

```text
BLOCK_M = 128
BLOCK_N = 128
BLOCK_K = 64
num_warps = 4
```

得到：

```text
grid_m = ceil(208 / 128) = 2
grid_n = ceil(416 / 128) = 4
grid   = (2, 4)
K iterations = ceil(304 / 64) = 5
```

取 `pid_m=1, pid_n=3`：

```text
off_m = 1 * 128 = 128
off_n = 3 * 128 = 384
```

这个 CTA 的职责是：

```text
C tile: rows 128..255, cols 384..511 理论上
实际有效区域: rows 128..207, cols 384..415
```

每个 K iteration 的 A slice 是：

```text
A[128:256, k:k+64]
```

当 `k=256` 时，全局 K 只剩 48 个有效元素；`k=304..319` 越界。TMA 根据 descriptor 中保存的全局 shape 将越界读填零，因此 WGMMA 仍可计算完整的 `128 x 128 x 64` block，只是多乘了零。

`TRANSPOSE_B=False` 时：

```text
B global shape = [K, N]
B tile         = B[k:k+64, 384:512]
逻辑 MMA B    = [64, 128]
```

`TRANSPOSE_B=True` 时：

```text
B global shape = [N, K]
B tile         = B[384:512, k:k+64]
permute view   = [64, 128]
```

写回时 TMA store 同样根据 `c_desc.shape=[M, N]` 裁剪边界，所以完整 tile 中越界的 rows 208..255 和 cols 416..511 不会写到 C 之外，真正的有效写回大小是 `80 x 32`。

### `TRANSPOSE_B` 为什么不需要复制数据

TMA 只负责把一块数据按描述符搬进 SMEM，它不会自动替 WGMMA 做矩阵转置。代码通过两步把语义补齐：

```python
if TRANSPOSE_B:
    tma.async_load(b_desc, [off_n, k], bar, b_smem)
    b = b_smem.permute((1, 0))
else:
    tma.async_load(b_desc, [k, off_n], bar, b_smem)
    b = b_smem
```

非转置路径的 SMEM layout 是 `[BLOCK_K, BLOCK_N]`；转置路径装入的是 `[BLOCK_N, BLOCK_K]`，`permute((1, 0))` 只创建一个逻辑转置 view，不搬运元素。WGMMA 接收这个 view 时，会使用相应的 SMEM descriptor 读取成 `[BLOCK_K, BLOCK_N]`。

因此 `TRANSPOSE_B` 改变的是三个地方，必须同时一致：

```text
host descriptor block shape
TMA load 坐标
WGMMA 之前是否 permute
```

只改其中一处通常不会产生“稍微不对”的结果，而是直接读错矩阵，数值测试会大面积失败。

### 这个 baseline 与下一节 pipeline 的差别

| 项目 | 本节 blocked baseline | 下一节 double-buffered pipeline |
|---|---|---|
| A/B SMEM buffer | 各 1 份 | 各 2 份，轮流使用 |
| 下一轮 TMA load | 必须等当前 WGMMA 完成后覆盖同一 buffer | 可写另一 buffer，与当前 WGMMA overlap |
| WGMMA 发起位置 | TMA wait 后立即发起并等待 | 构造长 dependency chain，延后统一 wait |
| 额外状态 | `phase` | `phase`、buffer index、MMA dependency |
| 结果正确性核心 | TMA 完成、MMA 完成后才能覆盖 | 前一轮 MMA 完成才能覆盖对应 buffer |

baseline 不是“错误版本”，它更容易验证坐标、layout 和尾块是否正确。确认 baseline 后再加入双缓冲，才能把正确性问题和调度性能问题分开定位。

### 常见错误与可观察症状

| 错误 | 可观察症状 | 根本原因 |
|---|---|---|
| grid axes 与 `off_m/off_n` 交换 | 大面积 wrong answer，某些 tile 随机覆盖 | CTA 取错 A/B 区域或写错 C tile |
| `phase` 不翻转 | 从第二个 K iteration 开始偶发读到旧 SMEM | `wait` 命中了旧的已 complete phase |
| `mbarrier.expect` 字节数算错 | 偶尔提前完成或永久 hang | tx count 与实际 TMA bytes 不匹配 |
| TMA 后不 `mbarrier.wait` | 数值不稳定、首次运行就可能错 | WGMMA 读到未完成的 SMEM tile |
| `warpgroup_mma_wait` 前就开始下一轮 TMA | 偶发 race、尤其大 K 时明显 | 单 buffer 被仍在读的 WGMMA 覆盖 |
| `TRANSPOSE_B` 只改 descriptor 没改坐标 | 编译通过但结果错误 | TMA 读取区域方向不一致 |
| `c_smem.store` 后缺少 `fence_async_shared` | C 偶尔是旧值或部分值 | generic proxy 写入尚未对 TMA async proxy 可见 |
| TMA store 后缺少 `store_wait` | kernel 提前结束或后续读 C 不稳定 | TMA store 尚未完成全局写回 |

### 自我检查

1. **`M=208, N=416, BLOCK_M=BLOCK_N=128` 的 grid 是什么？**
   - 答：`(ceil(208 / 128), ceil(416 / 128)) = (2, 4)`，共 8 个 CTA，每个 CTA 负责一个 C tile。
2. **`pid_m=1, pid_n=3` 处理 C 的哪个区域？**
   - 答：逻辑 tile 从 `(128, 384)` 开始，范围是 `[128:256, 384:512]`；因为全局 C 只有 `[208, 416]`，实际有效写回是 `[128:208, 384:416]`。
3. **为什么同一个 mbarrier 在每次 K iteration 都要翻转 `phase`？**
   - 答：barrier 的 completion parity 每完成一个 phase 就翻转。复用 barrier 时，下一次 wait 必须等待新的 parity，否则会命中旧的完成状态。
4. **`TRANSPOSE_B=True` 时，为什么 `b_smem.permute((1, 0))` 不增加数据搬运？**
   - 答：它只改变 SMEM tensor 的逻辑 view 和 layout，让 WGMMA 按转置后的索引读取，不生成新的 SMEM buffer。
5. **为什么这个 single-buffer baseline 必须在每轮 `warpgroup_mma` 后立即 `warpgroup_mma_wait`？**
   - 答：因为 A/B 只有一份 SMEM buffer；只有确认 WGMMA 已完成读取，下一轮 TMA 才能安全覆盖它。

### 验证边界

本机没有 Hopper runtime，因此只能做静态归档校验。实际运行需要 Hopper GPU 和对应版本的 Triton Gluon：

```bash
python blocked_matmul_wgmma_demo.py
```

静态检查命令：

```bash
python3 -m py_compile operators/code/triton/gluon/tutorials/05-wgmma.py
git diff --check
```

## 官方 05 完整拆解：WGMMA asynchronous pipeline

### 本次讲解位置

章节：`chapter_gluon_tensor_cores`（算子开发 / Triton Gluon）<br>
小节：Warp-Group MMA / WGMMA asynchronous pipeline<br>
知识点：`warpgroup_mma_init`、`warpgroup_mma`、`warpgroup_mma_wait`、`deps`、双缓冲 operand、WGMMA 与 TMA overlap<br>
上次：官方 blocked matmul 的二维 tile grid、K-loop、TMA 尾块与 mbarrier phase<br>
下次：在 WGMMA 流水中同时异步化 TMA 的多阶段加载，并进一步做 warp specialization<br>
PTX：`wgmma.mma_async`、commit-group wait<br>
官方源码：[`05-wgmma.py`](code/triton/gluon/tutorials/05-wgmma.py)<br>
官方教程：[Warp-Group MMA](https://triton-lang.org/main/getting-started/tutorials/gluon/wgmma.html)

### 为什么现在讲这个

上一节已经知道一次 WGMMA 计算 `D = A @ B + C`，也知道 A 可以在 register 或 SMEM、B 必须在 SMEM、accumulator 在 register。但只知道“发出一次 WGMMA 再等它结束”还不够，因为这样 tensor core 会在等待 operand 时空转：

```text
TMA(k) -> WGMMA(k) -> TMA(k+1) -> WGMMA(k+1)
```

这一节要解决的是一个新的具体问题：MMA 已经把第 `k` 块 A/B 放进 tensor pipe 后，能不能立刻用另一块 SMEM buffer 接收第 `k+1` 块数据？答案可以，但前提是分清两个不同的完成事件：

```text
TMA 完成  -> 可以开始 WGMMA 读这块 SMEM
WGMMA 完成 -> 可以覆盖这块 SMEM，也可以读取 accumulator
```

如果不理解这两个事件的区别，就会出现典型的随机错误：TMA 尚未来得及写完，WGMMA 已经读到旧数据；或者上一轮 WGMMA 还在读 buffer，下一轮 TMA 已经把它覆盖。

### 心智模型：WGMMA 是“异步块运算”，不是立即得到结果的函数

把 `warpgroup_mma` 看成向 tensor pipe 投递一个任务，而不是普通 Python 函数调用更接近硬件：

```text
acc 是一个异步依赖
    |
    | warpgroup_mma(..., is_async=True)
    v
tensor core 正在计算
    |
    | warpgroup_mma_wait(..., deps=(acc,))
    v
acc 变成“结果已可安全读取”的依赖
```

`warpgroup_mma_wait(num_outstanding=0)` 的含义是：等到在途 WGMMA 不超过 0 个，也就是这一条依赖链上的 WGMMA 全部完成。它不是 mbarrier，也不是 CTA 同步；它只约束 WGMMA commit group 以及依赖它的 register/SMEM 使用。

`deps=(acc,)` 是数据依赖，不是可选提示。编译器要靠它知道“后面读取的 acc 属于哪一组异步写”。因此必须写成：

```python
acc = warpgroup_mma(a, b_smem, acc, is_async=True)
acc = warpgroup_mma_wait(num_outstanding=0, deps=(acc,))
```

不能再写：

```python
warpgroup_mma(a, b_smem, acc, is_async=True)
warpgroup_mma_wait(num_outstanding=0, deps=(acc,))
```

第二种写法里 `acc` 仍然指旧依赖，后续读取没有正确挂到本次 MMA 上。结果可能偶尔正确、偶尔读到旧值，属于最隐蔽的 race。

### 完整可运行示例：双缓冲 WGMMA pipeline

下面是一个完整 Hopper 示例。它用两个 SMEM buffer 轮流接收 TMA 数据；每一轮先等待上一轮 WGMMA 完成，再发出当前 WGMMA。TMA 下一轮加载和当前 tensor pipe 中的 WGMMA 因此可以重叠。

文件可以保存为 `wgmma_pipeline_demo.py`：

```python
from __future__ import annotations

import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor
from triton.experimental.gluon.language.nvidia.hopper import (
    fence_async_shared,
    mbarrier,
    tma,
    warpgroup_mma,
    warpgroup_mma_init,
    warpgroup_mma_wait,
)


def is_hopper() -> bool:
    target = triton.runtime.driver.active.get_current_target()
    return target.backend == "cuda" and torch.cuda.get_device_capability()[0] == 9


@gluon.constexpr_function
def get_warps_per_cta(BLOCK_M, BLOCK_N, num_warps):
    warps_per_cta = [4, 1]
    m = 16
    while warps_per_cta[0] * warps_per_cta[1] != num_warps:
        if BLOCK_M > m * warps_per_cta[0]:
            warps_per_cta[0] *= 2
        else:
            warps_per_cta[1] *= 2
    return warps_per_cta


@gluon.constexpr_function
def get_instr_shape_n(BLOCK_M, BLOCK_N, num_warps):
    m = 16
    m_reps = triton.cdiv(BLOCK_M, m)
    n_reps = triton.cdiv(num_warps, m_reps)
    max_n = max(BLOCK_N // n_reps, 8)
    n = 256
    while n > max_n or BLOCK_N % n != 0:
        n -= 8
    assert n >= 8, "expected to find a valid n"
    return n


@gluon.constexpr_function
def pick_wgmma_layout(dtype, BLOCK_M, BLOCK_N, num_warps):
    m = 16
    k = 256 // dtype.primitive_bitwidth
    n = get_instr_shape_n(BLOCK_M, BLOCK_N, num_warps)
    warps_per_cta = get_warps_per_cta(BLOCK_M, BLOCK_N, num_warps)
    return gl.NVMMADistributedLayout(
        version=[3, 0],
        warps_per_cta=warps_per_cta,
        instr_shape=[m, n, k],
    )


@gluon.jit
def wgmma_pipeline_kernel(a_desc, b_desc, c_desc, num_warps: gl.constexpr):
    BLOCK_M: gl.constexpr = c_desc.block_type.shape[0]
    BLOCK_N: gl.constexpr = c_desc.block_type.shape[1]
    BLOCK_K: gl.constexpr = a_desc.block_type.shape[1]
    dtype: gl.constexpr = a_desc.dtype
    K = a_desc.shape[1]

    # 每块 operand 有 2 份，TMA 写下一份时 WGMMA 可以读另一份。
    a_smem = gl.allocate_shared_memory(
        dtype, [2] + a_desc.block_type.shape, a_desc.layout
    )
    b_smem = gl.allocate_shared_memory(
        dtype, [2] + b_desc.block_type.shape, b_desc.layout
    )
    index = 0

    pid_m = gl.program_id(axis=0)
    pid_n = gl.program_id(axis=1)
    off_m = pid_m * BLOCK_M
    off_n = pid_n * BLOCK_N

    mma_layout: gl.constexpr = pick_wgmma_layout(
        dtype, BLOCK_M, BLOCK_N, num_warps
    )
    acc = warpgroup_mma_init(
        gl.zeros((BLOCK_M, BLOCK_N), dtype=gl.float32, layout=mma_layout)
    )

    bar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(bar, count=1)
    phase = 0

    for k in range(0, K, BLOCK_K):
        a = a_smem.index(index)
        b = b_smem.index(index)

        mbarrier.expect(
            bar,
            a_desc.block_type.nbytes + b_desc.block_type.nbytes,
        )
        tma.async_load(a_desc, [off_m, k], bar, a)
        tma.async_load(b_desc, [k, off_n], bar, b)
        mbarrier.wait(bar, phase=phase)
        phase ^= 1

        # 没有在途 WGMMA 时，第一次 wait 是 no-op；之后它等待上一轮 MMA。
        acc = warpgroup_mma_wait(num_outstanding=0, deps=(acc,))
        acc = warpgroup_mma(a, b, acc, is_async=True)

        # 下一轮 TMA 写另一份 buffer；当前 WGMMA 仍在读这一份。
        index ^= 1

    # 最后一轮没有下一次循环来替它 wait，因此必须显式等待。
    acc = warpgroup_mma_wait(num_outstanding=0, deps=(acc,))
    mbarrier.invalidate(bar)

    c_smem = gl.allocate_shared_memory(
        dtype, c_desc.block_type.shape, c_desc.layout
    )
    c_smem.store(acc.to(dtype))
    fence_async_shared()
    tma.async_store(c_desc, [off_m, off_n], c_smem)
    tma.store_wait(pendings=0)


def wgmma_pipeline(A, B, C, BLOCK_M=128, BLOCK_N=128, BLOCK_K=64, num_warps=4):
    M, N = C.shape

    a_layout = gl.NVMMASharedLayout.get_default_for(
        [BLOCK_M, BLOCK_K], gl.float16
    )
    b_layout = gl.NVMMASharedLayout.get_default_for(
        [BLOCK_K, BLOCK_N], gl.float16
    )
    c_layout = gl.NVMMASharedLayout.get_default_for(
        [BLOCK_M, BLOCK_N], gl.float16
    )

    a_desc = TensorDescriptor.from_tensor(A, [BLOCK_M, BLOCK_K], a_layout)
    b_desc = TensorDescriptor.from_tensor(B, [BLOCK_K, BLOCK_N], b_layout)
    c_desc = TensorDescriptor.from_tensor(C, [BLOCK_M, BLOCK_N], c_layout)

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    wgmma_pipeline_kernel[grid](a_desc, b_desc, c_desc, num_warps=num_warps)


if __name__ == "__main__":
    if not is_hopper():
        raise RuntimeError("This tutorial requires a Hopper NVIDIA GPU")

    M, N, K = 512, 512, 512
    torch.manual_seed(0)
    A = torch.randn(M, K, device="cuda", dtype=torch.float16)
    B = torch.randn(K, N, device="cuda", dtype=torch.float16)
    C = torch.empty(M, N, device="cuda", dtype=torch.float16)

    wgmma_pipeline(A, B, C)
    torch.testing.assert_close(A @ B, C, rtol=1e-3, atol=1e-1)
    print("wgmma pipeline ok", C.sum().item())
```

预期输出形如：

```text
wgmma pipeline ok <一个有限的 float32/float16 标量和>
```

本例最后写回的是 fp16，因此会对 `A @ B` 的 fp32 结果做舍入；`rtol=1e-3, atol=1e-1` 与官方测试一致。

### `warpgroup_mma_init` 到底做了什么

它不是一次矩阵计算，也不是 `wgmma.mma_async`。它的作用是给循环建立一个可沿着流水线继续传递的 accumulator 依赖起点：

```text
gl.zeros(...)                 -> 一块普通 register tensor
warpgroup_mma_init(...)       -> 可以被 WGMMA 依赖链携带的 accumulator
```

为什么不能直接让第一个 `warpgroup_mma_wait` 等一个普通 tensor？因为循环必须先执行“wait 上一轮”再执行“issue 当前轮”，而第一轮还没有上一轮 WGMMA。`warpgroup_mma_init` 让这个统一循环有合法的初始依赖；此时没有在途 WGMMA，所以第一次 `wait` 是 no-op。

### 双缓冲时间线：为什么 buffer 不会被覆盖

以 `BLOCK_K=64, K=256` 为例，循环有四轮。`index` 的轨迹是：

```text
k=0: load buf0 -> wait TMA -> wait old MMA(no-op) -> issue MMA0(buf0) -> index=1
k=64: load buf1 -> wait TMA -> wait MMA0       -> issue MMA1(buf1) -> index=0
k=128: load buf0 -> wait TMA -> wait MMA1       -> issue MMA2(buf0) -> index=1
k=192: load buf1 -> wait TMA -> wait MMA2       -> issue MMA3(buf1) -> index=0
```

关键点不是“有两个 buffer 就自动安全”，而是每个 buffer 在被复用前都经历了一次 WGMMA wait：

```text
k=64 时 wait MMA0
    -> 确认 MMA0 已读完 buf0
    -> k=128 的 TMA 才可以覆盖 buf0
```

所以正确顺序必须保持为：

```text
wait TMA(current buffer)
wait previous WGMMA
issue current WGMMA
```

如果为了“更激进的 overlap”删掉 `wait previous WGMMA`，可能会出现 TMA 已经覆盖 SMEM，而上一轮 tensor core 还在读它。其症状不是稳定地算错，而是结果随运行时机波动。

### `num_outstanding` 的准确含义

WGMMA 的异步完成由 commit group 跟踪，类似 TMA store：

```text
issue WGMMA0 -> group 0
issue WGMMA1 -> group 1
wait(num_outstanding=0) -> 这两组都完成
wait(num_outstanding=1) -> 允许 1 组在途，只保证更早的组已完成
```

本教程用 `num_outstanding=0`，因为它要同时保证两件事：

1. 后续可以读取 accumulator 的最终值。
2. 后续可以复用 WGMMA 刚刚读取的 SMEM buffer。

若以后做成更深的 tensor-core pipeline，可以允许更多 outstanding 组，但必须用额外 buffer 隔离仍在读取的 SMEM。否则“等待更少”带来的不是重叠，而是数据竞争。

### fence：什么时候需要 `fence_async_shared`

WGMMA 和 TMA 都属于 async proxy，因此 TMA 写完 SMEM 后直接发起 WGMMA 不需要额外 fence：

```python
tma.async_load(a_desc, [off_m, k], bar, a)
mbarrier.wait(bar, phase=phase)
acc = warpgroup_mma(a, b, acc, is_async=True)
```

但普通 generic store 写 SMEM 时，WGMMA 看不到一个自动跨 proxy 的可见顺序：

```python
b_smem.store(b)
fence_async_shared()
warpgroup_mma(a, b_smem, acc, is_async=True)
```

这里少了 `fence_async_shared()`，编译器可能把普通写入和异步读取排成错误顺序，表现为 WGMMA 读到旧数据或部分旧数据。TMA store 之前也要在处理 `d_smem.store(d)` 后插入 fence，因为 TMA store 通过 async proxy 读取 SMEM。

### 转置 B 为什么不用复制数据

官方 blocked matmul 额外展示了 `TRANSPOSE_B`。如果 B 以 `[N, K]` 形式存入 SMEM，则不需要手工交换元素，直接创建一个转置 view：

```python
if TRANSPOSE_B:
    b = b_smem.permute((1, 0))
else:
    b = b_smem

acc = warpgroup_mma(a_smem, b, acc, is_async=True)
```

`permute` 改变的是 WGMMA 看到 B 的逻辑索引顺序，并把转置要求传递给 MMA 逻辑。这样仍然遵循“B 必须来自 SMEM”的约束，不需要先把 B 复制到一块新的 `[K, N]` buffer。

### instruction shape 与 block shape 不是同一件事

例如一个 tile 是：

```text
BLOCK_M = 128
BLOCK_N = 128
num_warps = 4
dtype = fp16
```

有：

```text
m = 16
k = 256 / 16 = 16
mReps = ceil(128 / 16) = 8
nReps = ceil(4 / 8) = 1
maxN = max(128 / 1, 8) = 128
```

因此 `get_instr_shape_n` 从 256 向下找到 128，得到：

```text
instr_shape = [16, 128, 16]
```

一个 block tile `128 x 128 x 64` 会由多组较小的 `wgmma.mma_async` 覆盖。`warps_per_cta` 也不是 `num_warps` 的同义词：4 warps 的基础 atom 是 `[4, 1]`；8 warps 可能选 `[4, 2]` 或 `[8, 1]`，后者改变 M/N 方向上的切分和后续 reduction 成本。

### 官方参数搜索为什么不是随便 hardcode

官方 `find_configs` 先用资源约束缩小候选，再按 instruction shape 和 tile 面积筛选。核心约束可以直接算：

```text
SMEM per CTA ~= 228 KB / occupancy - 1 KB
register per CTA ~= 64K registers / occupancy - baseline
A/B SMEM = (BLOCK_M * BLOCK_K + BLOCK_N * BLOCK_K) * dtype_bytes * num_buffers
acc registers = BLOCK_M * BLOCK_N
```

在双缓冲变体中，A/B 需要两份；C 的 SMEM 与 A/B 生命周期不重叠。筛选逻辑优先保留最大的 `instr_shape_n`，再在同样 instruction shape 下保留最大的 `BLOCK_M * BLOCK_K`。这只是减少搜索空间的启发式，不是硬件保证，最终仍要在目标 GPU 上 benchmark。

### 官方性能结果

单次小 WGMMA 的参数实验结果：

| A 路径 | instr_shape_n=16 | 32 | 64 | 128 |
|---|---:|---:|---:|---:|
| SMEM | 9.47 us | 8.48 us | 8.32 us | 8.32 us |
| register | 9.32 us | 8.60 us | 8.37 us | 8.36 us |

更大的 `n` 减少了指令数量，所以通常更快。register 路径没有自动胜出：如果 A 原本就在 SMEM，先 load 到 register 反而增加了搬运成本；如果 A 已经在 register，它的价值就不同。

官方 blocked matmul 的展示结果：

| 版本 | 最佳展示值 |
|---|---:|
| 非 WGMMA-pipeline | 约 466 TFLOPS（候选表中最佳约 474.03） |
| WGMMA-pipeline | 495.31 TFLOPS |

这些是教程在 Hopper 上的数字，不是本机实测。关键是 pipeline 只带来约 5% 的提升，但最佳 block 配置发生了变化：非流水候选的最佳 `BLOCK_N=256` 在流水版本里不再最好，最佳变为 `BLOCK_M=128, BLOCK_N=128, BLOCK_K=64, num_warps=4, instr_shape_n=128`。因此“流水化”和“重新调参”必须一起做。

### 常见错误与可观察症状

| 错误写法 | 症状 | 原因 |
|---|---|---|
| 丢掉 `warpgroup_mma_wait` 的返回值 | 偶发旧 accumulator、race | `deps` 没有连到真正要等待的 MMA |
| 只保留两个 buffer，却删除循环内的 previous-MMA wait | 偶发错误或数值不稳定 | TMA 覆盖了 WGMMA 仍在读的 SMEM |
| 循环结束后不补最后一次 wait | 最终 tile 偶发错误 | 最后一个 MMA 仍在途就被读取 |
| 用 generic `store` 后直接 WGMMA | 读到旧 B | 缺少 `fence_async_shared()` |
| B 放在 register | 编译或 layout 校验失败 | WGMMA 的 B 必须来自 SMEM |
| `instr_shape_n` 大于 `maxN` | 编译期失败或非法布局 | 指令 shape 与 warp 切分不匹配 |
| 把 `num_warps` 直接当成 `warps_per_cta` | 布局/WGMMA atom 错误 | 8 warps 需要合法的 `[4,2]` 或 `[8,1]` 切分 |
| 把 `num_outstanding` 调大却没有更多 buffer | 随机错误 | 允许更多 in-flight MMA，却提前复用了 SMEM |

### 自我检查

1. **为什么 `warpgroup_mma_wait` 必须在第一轮也可以调用？**
   - 答：因为循环采用统一顺序 `wait previous -> issue current`。`warpgroup_mma_init` 建立了初始依赖；没有在途 WGMMA 时，wait 是 no-op。
2. **`BLOCK_K=64` 的双缓冲循环为什么不会在 `k=128` 覆盖 `buf0`？**
   - 答：`k=64` 的迭代在发出 MMA1 前先 wait 了 MMA0；MMA0 已完成读取 buf0，所以 `k=128` 的 TMA 可以安全写入 buf0。
3. **`num_outstanding=0` 保证哪两件事？**
   - 答：在途 WGMMA 全部完成，因此 accumulator 可安全读，且 WGMMA 所读的 SMEM 可以安全覆盖。
4. **TMA 写入 SMEM 后接 WGMMA，为什么通常不需要 `fence_async_shared`？**
   - 答：TMA 和 WGMMA 都走 async proxy；但普通 generic SMEM store 接 WGMMA 时仍需要 fence。
5. **为什么官方说流水化之后还需要重新调参？**
   - 答：WGMMA 流水改变了 buffer、register、occupancy 和延迟隐藏方式；非流水时最优的 block/instruction shape 不一定是流水后的最优配置。

### 验证边界

静态检查命令：

```bash
python3 -m py_compile operators/code/triton/gluon/tutorials/05-wgmma.py
git diff --check
```

运行时要求 Hopper GPU 和对应版本的 Triton Gluon。本机没有 Hopper runtime，因此上面的 benchmark 数值来自官方教程，不是本机实测；真实验证应在 Hopper 机器上运行完整归档脚本，或把本节的 `wgmma_pipeline_demo.py` 保存后执行。
