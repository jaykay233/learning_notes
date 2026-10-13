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
