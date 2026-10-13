# Triton Gluon 03–04：异步拷贝、TMA 与流水线

## 本次讲解位置

章节：`chapter_gluon_async_dataflow`（算子开发 / Triton Gluon）<br>
小节：从 `cp.async` 到 TMA 的 global ↔ shared 数据通道<br>
知识点：commit group、mbarrier、double/triple buffering、tensor descriptor<br>
上次：layout 决定元素由谁拥有，以及 global/shared 访问是否高效<br>
下次：用 WGMMA 和 tcgen05 把 shared-memory tile 送入 tensor core<br>
PTX：涉及 `cp.async`、TMA 异步代理以及 mbarrier 的 completion 语义；Gluon API 对它们做了封装。

## 为什么现在讲这个

前一节只解决了“元素应该由谁负责”。如果每个元素仍按 `global -> register -> compute -> register -> global` 走，64 位地址、mask、中间值和结果会同时占用寄存器。对于 GEMM，这种压力会直接限制 block size、stage 数和 occupancy。

异步拷贝把 transfer 从主线程的执行流中移走：先把 tile 发到 shared memory，再继续准备后续工作。这里的关键不是“有一条更快的 copy 指令”，而是建立三个可以重叠的通道：

```text
global memory transfers
        |
        v
shared-memory buffers
        |
        v
register computation and stores
```

`cp.async` 是这条路上的通用 Ampere 工具；TMA 是 Hopper 以后由专用 descriptor 驱动的硬件路径。两者都必须回答同一个问题：**消费者怎么知道生产者已经完成？**

## 两个同步对象：commit group 与 mbarrier

| 对象 | 典型用途 | 完成方式 | 主要风险 |
|---|---|---|---|
| commit group | `cp.async`、部分 TMA store、WGMMA 等按组等待 | `commit_group` 后 `wait_group(N)` | 等待数量算错会读到未完成数据或过早复用 buffer |
| mbarrier | TMA read、`tcgen05_mma`、`tcgen05_copy` 等 | arrive count 或字节数归零，phase 翻转 | phase parity 错、复用 barrier 太早会 race |

`cp.async` 的典型序列是：

```text
issue async_load
commit_group
做一些不依赖该 buffer 的工作
wait_group(N)
从 shared memory 读
```

`wait_group(N)` 表示最多允许 N 个已提交 group 尚未完成。不是“等待第 N 个 group”，而是“保留 N 个在途”。例如 `wait_group(0)` 要等所有已提交 group 完成；`wait_group(1)` 表示还允许一个 group 未完成，通常对应刚刚提交的最新 group，而更早提交的 group 已经完成。

## 多缓冲为什么能形成流水线

单缓冲时，load 和 compute 必须串行：

```text
load B0 -> wait B0 -> compute B0 -> load B1 -> wait B1 -> compute B1
```

双缓冲把 shared memory 拆成两个物理 slot。因为 B0 和 B1 不共用存储，B1 的 load 可以与 B0 的 compute 同时进行：

```text
time ->
load B0  | done
         | compute B0 | done
load B1  | waiting... | done
                     | compute B1 | done
load B2              | waiting...  | done
```

这就是 software pipelining。`num_buffers` 不是“把 CTA 分成几组”，而是 shared-memory tile 的物理缓冲数。缓冲越多，允许在途的 load 越多，但每个 slot 都占用 shared memory；超过访存延迟所需要的深度后，吞吐不会继续上升。

官方 elementwise-add 结果很直接：

| 版本 | 吞吐 |
|---|---:|
| 普通 `gl.load`/`gl.store` | `1.48 TB/s` |
| `cp.async` | `3.97 TB/s` |
| 双缓冲 `cp.async` pipeline | `4.20 TB/s` |
| 三缓冲 `cp.async` pipeline | `4.20 TB/s` |

双缓冲到三缓冲不再变快，通常说明内存带宽已经接近瓶颈，继续增加 buffer 只增加寄存器和地址计算压力。

### 具体循环轨迹

官方 `num_buffers=2` 的 pipeline 可以简化为：

```text
prefetch: load B0 -> commit group 0

steady:
  issue load B1 -> commit group 1
  wait_group(1)                  # group 0 已完成，group 1 可以在途
  compute B0

  issue load B2 -> commit group 2
  wait_group(1)                  # group 1 已完成
  compute B1

drain:
  wait_group(0)
  compute B2
```

注意 `wait_group(1)` 并不表示 `B0` 或某个固定 buffer 已完成；它表示“老的 group 已完成，最新的一个可以继续在途”。实际 buffer 通过 `read_idx % num_buffers` 与 `copy_idx % num_buffers` 对应。

## 完整可运行例程：`cp.async` elementwise add 软件流水线

### 本次讲解位置

章节：`chapter_gluon_async_dataflow`（算子开发 / Triton Gluon）<br>
小节：`03-async-copy.py` 的 software pipeline<br>
知识点：`issue_loads`、`perform_add`、prologue/steady/drain、`wait_group(num_buffers - 2 - i)`<br>
上次：`wait_group(N)` 表示最多允许 N 个已提交 group 未完成<br>
下次：TMA tensor descriptor 与 mbarrier 生命周期<br>
PTX：`cp.async`、`cp.async.commit_group`、`cp.async.wait_group`

### 为什么现在讲这个

前面的轨迹给出了双缓冲的直觉，但还不足以解释完整 kernel 为什么要把循环拆成三段，也不足以解释 drain 的等待深度为什么每轮减一。具体风险是：少做一次 drain 会丢掉流水线里最后 `num_buffers-1` 个 tile；把 drain 的 `wait_group` 参数写成固定值，则可能读取尚未完成的数据，或者让已经完成的 buffer 继续等待。

这一节把完整数据流落实为可运行代码：

```text
prologue: 预取前 num_buffers-1 个 tile
steady:   发出未来 copy，等待最老 copy，计算当前 tile
drain:    不再发新 copy，将剩余 num_buffers-1 个 tile 收尾
```

归档源码：[`03-async-copy.py`](code/triton/gluon/tutorials/03-async-copy.py)。完整的最小 Ampere `cp.async` 版本如下。它包含 imports、三个核心函数、host wrapper、运行入口和正确性校验，使用 `(1000, 2000)` 的输入分别验证 `num_buffers=1,2,3`：

```python
from __future__ import annotations

import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.ampere import async_copy as cp


XBLOCK = 32
YBLOCK = 64


@gluon.jit
def issue_loads(copy_idx, a_smem, b_smem, a_ptrs, ystride_a, b_ptrs, xmask, ynumel, y_idx, ystride_b,
                YBLOCK: gl.constexpr, num_buffers: gl.constexpr):
    yoffs = copy_idx * YBLOCK + y_idx
    mask = xmask & (yoffs < ynumel)[None, :]
    cp.async_load(a_smem.index(copy_idx % num_buffers),
                  a_ptrs + ystride_a * yoffs[None, :], mask)
    cp.async_load(b_smem.index(copy_idx % num_buffers),
                  b_ptrs + ystride_b * yoffs[None, :], mask)
    cp.commit_group()
    return copy_idx + 1


@gluon.jit
def perform_add(read_idx, a_smem, b_smem, c_ptrs, ynumel, ystride_c, y_idx, xmask,
                YBLOCK: gl.constexpr, num_buffers: gl.constexpr, layout: gl.constexpr):
    a_val = a_smem.index(read_idx % num_buffers).load(layout)
    b_val = b_smem.index(read_idx % num_buffers).load(layout)
    c_val = a_val + b_val
    yoffs = read_idx * YBLOCK + y_idx
    mask = xmask & (yoffs < ynumel)[None, :]
    gl.store(c_ptrs + ystride_c * yoffs[None, :], c_val, mask=mask)
    return read_idx + 1


@gluon.jit
def elementwise_add_pipelined_kernel(
        a_ptr, b_ptr, c_ptr, xnumel, ynumel,
        xstride_a, ystride_a, xstride_b, ystride_b, xstride_c, ystride_c,
        XBLOCK: gl.constexpr, YBLOCK: gl.constexpr,
        smem_layout: gl.constexpr, num_buffers: gl.constexpr,
):
    pid = gl.program_id(0)
    layout: gl.constexpr = gl.BlockedLayout([1, 1], [1, 32], [1, 4], [1, 0])
    xoffs = pid * XBLOCK + gl.arange(0, XBLOCK, gl.SliceLayout(1, layout))
    a_ptrs = a_ptr + xstride_a * xoffs[:, None]
    b_ptrs = b_ptr + xstride_b * xoffs[:, None]
    c_ptrs = c_ptr + xstride_c * xoffs[:, None]

    y_idx = gl.arange(0, YBLOCK, gl.SliceLayout(0, layout))
    xmask = (xoffs < xnumel)[:, None]

    dtype: gl.constexpr = a_ptr.dtype.element_ty
    a_smem = gl.allocate_shared_memory(dtype, [num_buffers, XBLOCK, YBLOCK], layout=smem_layout)
    b_smem = gl.allocate_shared_memory(dtype, [num_buffers, XBLOCK, YBLOCK], layout=smem_layout)
    copy_idx = 0
    read_idx = 0

    for _ in gl.static_range(num_buffers - 1):
        copy_idx = issue_loads(copy_idx, a_smem, b_smem, a_ptrs, ystride_a, b_ptrs,
                               xmask, ynumel, y_idx, ystride_b, YBLOCK, num_buffers)

    for _ in range(gl.cdiv(ynumel, YBLOCK) - (num_buffers - 1)):
        copy_idx = issue_loads(copy_idx, a_smem, b_smem, a_ptrs, ystride_a, b_ptrs,
                               xmask, ynumel, y_idx, ystride_b, YBLOCK, num_buffers)
        cp.wait_group(num_buffers - 1)
        read_idx = perform_add(read_idx, a_smem, b_smem, c_ptrs, ynumel, ystride_c,
                               y_idx, xmask, YBLOCK, num_buffers, layout)

    for i in gl.static_range(num_buffers - 1):
        cp.wait_group(num_buffers - 2 - i)
        read_idx = perform_add(read_idx, a_smem, b_smem, c_ptrs, ynumel, ystride_c,
                               y_idx, xmask, YBLOCK, num_buffers, layout)


def elementwise_add_pipelined(A, B, C, xblock=XBLOCK, yblock=YBLOCK, num_buffers=2):
    assert A.shape == B.shape == C.shape
    xnumel, ynumel = A.shape
    grid = (triton.cdiv(xnumel, xblock),)
    smem_layout = gl.SwizzledSharedLayout(vec=1, per_phase=1, max_phase=1, order=[1, 0])
    return elementwise_add_pipelined_kernel[grid](
        A, B, C, xnumel, ynumel,
        *A.stride(), *B.stride(), *C.stride(),
        xblock, yblock, smem_layout, num_buffers,
    )


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("This example requires an NVIDIA CUDA GPU")
    if torch.cuda.get_device_capability()[0] < 8:
        raise RuntimeError("This example requires Ampere or newer")

    xnumel, ynumel = 1000, 2000
    a = torch.randn(xnumel, ynumel, device="cuda")
    b = torch.randn(xnumel, ynumel, device="cuda")
    c = torch.empty_like(a)

    for num_buffers in (1, 2, 3):
        elementwise_add_pipelined(a, b, c, num_buffers=num_buffers)
        torch.testing.assert_close(a + b, c, atol=0, rtol=0)
        print(f"num_buffers={num_buffers}: ok")


if __name__ == "__main__":
    main()
```

运行方式与预期结果：

```bash
python 03-async-copy-pipeline.py
```

```text
num_buffers=1: ok
num_buffers=2: ok
num_buffers=3: ok
```

原始教程归档同时给出了 benchmark 输出：双缓冲和三缓冲都约为 `4.20 TB/s`。这组数值来自上游特定机器，不是当前环境的实测值。

### 环形缓冲的两个坐标

shared memory 的第一维不是矩阵数据维度，而是 pipeline slot 维度：

```text
a_smem.shape = [num_buffers, XBLOCK, YBLOCK]
```

`index(k)` 沿第一维取 view，因此：

```text
a_smem.index(k).shape = [XBLOCK, YBLOCK]
```

两个逻辑下标分工不同：

| 变量 | 含义 | shared memory slot |
|---|---|---:|
| `copy_idx` | 下一个要发出的逻辑 tile | `copy_idx % num_buffers` |
| `read_idx` | 下一个要消费的逻辑 tile | `read_idx % num_buffers` |

在三段循环的边界上，不变量是：

```text
copy_idx - read_idx = num_buffers - 1
```

生产者比消费者提前 `num_buffers-1` 个 tile。因此刚写入的 slot 和当前读取的 slot 不是同一个物理 slot。

### 一个 tile 对应一个 async-copy group

`issue_loads` 对同一个 tile 先后发出 A、B 两个 `cp.async_load`，然后只调用一次：

```python
cp.commit_group()
```

所以这里数 group 就是数 tile：

```text
group k = {A tile k, B tile k}
```

如果把 A、B 分别 commit 成两个 group，`wait_group` 的参数必须整体换算；当前代码没有这样做。

### prologue 为什么运行 `num_buffers-1` 次

prologue 先把 pipeline 填满，但必须至少留一个空 slot 给 steady state 的新生产者，因此预取数量不是 `num_buffers`，而是：

```text
num_buffers - 1
```

例如三缓冲的初始状态是：

```text
B0, B1 已提交
slot 2 仍为空
```

进入 steady state 后，每次发出一块，再消费最老的一块，slot 数始终保持能容纳“在途 copy + 当前 compute”。

### steady state 为什么等待 `num_buffers-1`

每轮先发出一个新 group。假设发出前最多有 `num_buffers-1` 个 group 未完成，那么发出后有：

```text
至多 num_buffers 个 group 未完成
```

调用：

```python
cp.wait_group(num_buffers - 1)
```

会把未完成上限降到 `num_buffers-1`。由于 group 按提交顺序完成，被排在最老位置的 `read_idx` tile 已经可读，而最新的 `num_buffers-1` 个 copy 可以继续在途。

### drain 为什么是 `num_buffers - 2 - i`

steady state 结束时，没有新的 `issue_loads` 再执行，但 pipeline 中还剩 `num_buffers-1` 个已提交 group 没有 `perform_add`。drain 的执行次数因此固定为：

```python
for i in gl.static_range(num_buffers - 1):
```

每消费一个最老 group，剩余可允许在途的 group 上限就应该少一个，所以等待深度从 `num_buffers-2` 依次降到 `0`：

```python
cp.wait_group(num_buffers - 2 - i)
```

`gl.static_range` 会在编译期展开，使每个 `wait_group` 参数都是常量。具体值如下：

| `num_buffers` | `i=0` | `i=1` | `i=2` | 最后一次 |
|---:|---:|---:|---:|---:|
| 1 | 不进入 drain |  |  |  |
| 2 | `wait_group(0)` |  |  | 所有 copy 完成 |
| 3 | `wait_group(1)` | `wait_group(0)` |  | 所有 copy 完成 |
| 4 | `wait_group(2)` | `wait_group(1)` | `wait_group(0)` | 所有 copy 完成 |

### 具体轨迹：四块 tile，三缓冲

令 `num_buffers=3`，共有 `B0, B1, B2, B3` 四块：

```text
prologue:
  issue B0 -> group 0
  issue B1 -> group 1

steady:
  issue B2 -> group 2
  wait_group(2)   # group 0 完成，group 1/2 可以在途
  compute B0       # read_idx = 0 -> 1

  issue B3 -> group 3
  wait_group(2)   # group 1 完成，group 2/3 可以在途
  compute B1       # read_idx = 1 -> 2

drain:
  i=0: wait_group(1)  # group 2 完成，group 3 可以在途
       compute B2      # read_idx = 2 -> 3

  i=1: wait_group(0)  # group 3 完成
       compute B3      # read_idx = 3 -> 4
```

如果删掉 drain，`B2`、`B3` 虽然已经拷贝到 shared memory，却没有任何代码把它们加到 C。这是“结果尾部缺失”，不是偶发 race。

如果始终用 `wait_group(0)` 做 drain，正确性仍然成立，但每轮都会把全部在途 copy 等完，收尾阶段失去重叠。当前公式在保证依赖满足的前提下，让尚未轮到消费的更新 group 继续在途。

### 边界与常见错误

| 错误 | 症状 | 修复方向 |
|---|---|---|
| 少做一次 drain | C 的最后若干列块没有写回 | drain 次数必须是 `num_buffers-1` |
| `wait_group` 参数固定写成 `0` | 结果可能正确，但收尾完全串行 | 使用 `num_buffers-2-i` 逐级收干 |
| `wait_group` 参数比应有值大 | 读取尚未完成的 shared memory，数值随机错误 | 明确“最多允许在途 group 数”的语义 |
| `issue_loads` 的 A/B 分开 commit | group 数量与 tile 数量不再一一对应 | A/B 放在同一个 `commit_group` 前，或者同步换算等待深度 |
| prologue 超过真实 tile 数 | 越界读或垃圾写回 | 保留 `xmask & (yoffs < ynumel)`，并验证 `(4000, 120)` 这类小列数 shape |
| `num_buffers=1` 时误以为仍会重叠 | 性能退回非流水线版本 | 接受退化为 `issue -> wait(0) -> compute`，或用测试确认正确性 |

### 静态验证与运行边界

当前机器没有可用的 Triton CUDA runtime，因此本节只能完成源码级静态检查。真机验证需要在 Ampere 或更新 NVIDIA GPU 上运行上面的 `main()`，并确认三个 `num_buffers` 都打印 `ok`。这台机器若只有 CPU 或较旧 CUDA GPU，会在 `main()` 开始时明确失败，不应把源码阅读当成 runtime 验证。

### 自我检查：pipeline drain

1. **为什么 prologue 和 drain 都是 `num_buffers-1` 次？**
   - 答：prologue 预取 `num_buffers-1` 个 tile，steady 每轮新发一个并消费一个，结束时仍剩 `num_buffers-1` 个已发出但未消费的 tile，drain 正好逐一消费。
2. **`wait_group(num_buffers - 2 - i)` 为什么每轮减一？**
   - 答：drain 不再发新 group，每消费一个最老 group，剩余允许在途的 group 上限就减少一个，最后一次必须降到 0。
3. **`num_buffers=1` 时 drain 为什么为空？**
   - 答：没有预取，steady 每轮执行 `issue -> wait(0) -> compute`，循环结束时没有遗留 tile。
4. **为什么 `copy_idx` 和 `read_idx` 不能只保留一个下标？**
   - 答：生产者需要知道下一块写到哪里，消费者需要知道下一块从哪里读；两者之间必须稳定相差 `num_buffers-1`。
5. **把 drain 完全删掉会发生什么？**
   - 答：最后 `num_buffers-1` 个 tile 不会执行 `perform_add` 和写回，C 的对应列块保持未写状态。

## TMA：把地址计算交给 tensor descriptor

TMA 不替代 global memory，也不让数据凭空进入寄存器。它改变的是地址生成方式：

```text
普通 load:
  每个 lane 计算地址和 mask
  load 到 register

TMA:
  host 端创建 TensorDescriptor
  kernel 只给 block 坐标
  硬件把 global tile 搬到 shared memory
```

descriptor 中包含 base pointer、shape、stride、dtype、block shape 和 shared-memory layout。kernel 中常见的调用是：

```python
tma.async_load(in_desc, [pid * XBLOCK], bar, smem)
tma.async_store(out_desc, [pid * XBLOCK], smem)
```

TMA 的坐标表示 block 起点。对于二维 descriptor，TMA 从坐标指定的左上角开始拷贝 `block_shape`。越界 read/write 由 descriptor 的 shape 自动处理，减少 kernel 内 mask。

### mbarrier 的生命周期

官方 1D TMA 例子把读路径分成六步：

```text
1. 分配 64-bit mbarrier，初始化 count=1
2. mbarrier.expect(bar, bytes)
3. tma.async_load(desc, coord, bar, smem)
4. mbarrier.wait(bar, phase=0)
5. 使用 smem 中的数据
6. mbarrier.invalidate(bar)
```

`expect` 登记 TMA 将完成多少字节；每次 transaction 完成都会减少 outstanding byte count。字节数归零时，mbarrier 完成当前 phase。

phase 只有 0/1 两种 parity。初始化时 barrier 的布局和相位决定第一次要等哪个 parity。后续每次完成都会翻转 phase。下面是一个双缓冲时序：

| 迭代 | 使用 buffer | 等待 phase | 完成后 | 下一次等待 |
|---:|---|---:|---|---:|
| 0 | B0 | 0 | phase 0 完成 | 1 |
| 1 | B1 | 1 | phase 1 完成 | 0 |
| 2 | B0 | 0 | phase 0 完成 | 1 |

如果 phase 算错，常见症状不是“偶尔慢”，而是：

- 读到旧 tile，数值结果随机不正确；
- wait 永远不返回，kernel 挂起；
- barrier 被复用得太快，后续 TMA 与消费者重叠，出现 data race。

## 完整可运行例程：TMA 1D 拷贝

归档源码：[`04-tma.py`](code/triton/gluon/tutorials/04-tma.py)。以下是去掉测试框架后的完整脚本，要求 Hopper 或更新 GPU：

```python
from __future__ import annotations

import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor
from triton.experimental.gluon.language.nvidia.hopper import tma, mbarrier


@gluon.jit
def copy_1d_tma_kernel(in_desc, out_desc, XBLOCK: gl.constexpr):
    pid = gl.program_id(0)
    smem = gl.allocate_shared_memory(in_desc.dtype, [XBLOCK], in_desc.layout)
    bar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(bar, count=1)

    gl.static_assert(in_desc.block_type == out_desc.block_type)
    gl.static_assert(in_desc.layout == out_desc.layout)

    mbarrier.expect(bar, in_desc.block_type.nbytes)
    tma.async_load(in_desc, [pid * XBLOCK], bar, smem)
    mbarrier.wait(bar, phase=0)
    mbarrier.invalidate(bar)

    tma.async_store(out_desc, [pid * XBLOCK], smem)
    tma.store_wait(pendings=0)


def copy_1d_tma(x: torch.Tensor, y: torch.Tensor, block: int = 256) -> None:
    assert x.is_cuda and y.is_cuda and x.shape == y.shape
    layout = gl.NVMMASharedLayout.get_default_for([block], x.dtype)
    in_desc = TensorDescriptor.from_tensor(x, [block], layout)
    out_desc = TensorDescriptor.from_tensor(y, [block], layout)
    grid = (triton.cdiv(x.numel(), block),)
    copy_1d_tma_kernel[grid](in_desc, out_desc, block)


if __name__ == "__main__":
    x = torch.arange(1024, device="cuda", dtype=torch.float32)
    y = torch.empty_like(x)
    copy_1d_tma(x, y)
    torch.testing.assert_close(x, y, atol=0, rtol=0)
    print("tma_copy_ok:", y.sum().item())
```

预期输出：

```text
tma_copy_ok: 523776.0
```

这里 `1024 * 1023 / 2 = 523776`。如果没有 Hopper 或更新的 GPU，脚本会在构造 descriptor 或启动 TMA 时失败；在只有 CUDA 的旧设备上，应该使用 `03-async-copy.py` 的 `cp.async` 路径。

## TMA 的对齐与布局限制

官方教程特别列出两条硬限制：

1. 最内层坐标必须按 16 字节对齐。
2. 如果 shared layout 是 `fp4_padded`，最内层坐标必须按 128 字节对齐。

例如 `float16` 每个元素 2 bytes，则 8 个元素是 16 bytes：

```text
float16 坐标 [4, 8]：
  内层坐标 8 * 2 = 16 bytes，合法

float16 坐标 [8, 4]：
  内层坐标 4 * 2 = 8 bytes，不合法
```

这不是“建议向量化”的软约束，而是 TMA descriptor/指令的要求。错误坐标可能在 descriptor 构造、编译或 hardware 检查时报 illegal instruction。

## 代理与 fence

TMA 访问 shared memory 时走 async proxy，普通 `smem.store` 走 generic proxy。两者之间需要 `fence_async_shared()` 建立可见性顺序：

```text
generic proxy store -> fence_async_shared() -> TMA read/store
TMA read/write -> 等待完成 -> generic proxy 复用 buffer
```

如果只靠 mbarrier 保证“数据已经到 shared memory”，却忘记 producer 侧的 generic-to-async fence，可能看到旧数据。反过来，如果 TMA 完成等待写得正确，等待本身可以充当后续 generic 访问的同步边界。

## 常见错误与症状

| 错误 | 症状 | 修复方向 |
|---|---|---|
| `wait_group(N)` 理解成等 N 个 group | pipeline 提前读或长期等待 | 把它当成“最多允许 N 个未完成 group” |
| buffer 数与 producer/consumer 索引不匹配 | 结果随迭代错位或数据被覆盖 | 同时维护 `copy_idx` 与 `read_idx`，用 `% num_buffers` |
| mbarrier phase 不翻转 | 挂起、旧数据或随机错误 | 用 `phase ^= 1`，并明确初始化 parity |
| TMA 读取后没有 `invalidate`/复用规则错误 | 下一次 transaction 与旧等待混淆 | 明确 barrier 生命周期；不要同时复用未清理的 barrier |
| 忘记 `fence_async_shared` | shared memory 可见性错误 | 在 generic store 与 TMA/WGMMA/tcgen05 之间插入 fence |
| 坐标未对齐 | illegal instruction 或 descriptor 检查失败 | 内层按 16 字节，fp4 padded 按 128 字节 |
| 只增加 buffers 不看 shared memory/register 容量 | occupancy 下降或没有收益 | 用 Nsight/profiler 判断实际瓶颈 |

## 硬件与验证边界

`03-async-copy.py` 要求 Ampere 或更新；`04-tma.py` 要求 Hopper 或更新。教程中的 `4.20 TB/s`、`5.50 TB/s`、`5.90 TB/s` 是上游 benchmark 的示例值，不是本机实测。当前环境缺少 Triton runtime 和 CUDA GPU，无法在本机证明 TMA 的 runtime 对齐、barrier phase 或吞吐；只能完成源码静态检查。

## 自我检查

1. **`wait_group(0)` 和 `wait_group(1)` 的语义差异是什么？**
   - 答：0 表示不允许任何已提交 group 未完成；1 表示允许一个 group 在途，通常用来保留刚刚提交的下一轮 load，同时确认更早 group 已完成。
2. **TMA read 的完成为什么用 mbarrier，而 TMA store 通常用 commit group？**
   - 答：TMA read 需要让消费者知道数据已经落到 shared memory；TMA store 的完成由 commit group/`store_wait` 管理，并且还可能区分“读完 shared”与“写到 global”两个阶段。
3. **为什么双缓冲能重叠 load 与 compute？**
   - 答：两个 buffer 的 load 和 compute 访问不同共享内存槽，使异步 transfer 与当前 tile 计算同时进行。
4. **TMA 的 `[4,8]` 和 `[8,4]` 坐标在 float16 下哪个合法？**
   - 答：`[4,8]` 的内层偏移是 16 bytes，合法；`[8,4]` 只有 8 bytes，不满足最内层 16-byte 对齐。
5. **为什么 TMA 与普通 shared store 之间可能需要 fence？**
   - 答：TMA 属于 async proxy，普通 shared store 属于 generic proxy；跨代理访问 shared memory 需要建立顺序和可见性。
