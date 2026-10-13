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

## 官方 05 完整拆解：WGMMA asynchronous pipeline

### 本次讲解位置

章节：`chapter_gluon_tensor_cores`（算子开发 / Triton Gluon）<br>
小节：Warp-Group MMA / WGMMA asynchronous pipeline<br>
知识点：`warpgroup_mma_init`、`warpgroup_mma`、`warpgroup_mma_wait`、`deps`、双缓冲 operand、WGMMA 与 TMA overlap<br>
上次：WGMMA 的 A/B/C 位置约束、instruction shape 与一次同步 MMA<br>
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
