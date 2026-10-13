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

## 官方教程完整拆解：descriptor、read/store 完成点与三缓冲流水线

### 本次讲解位置

章节：`chapter_gluon_async_dataflow`（算子开发 / Triton Gluon）<br>
小节：官方 `TMA in Gluon` 的 pipelined elementwise add<br>
知识点：`TensorDescriptor`、每个 buffer 一个 mbarrier、TMA store 的两个完成点、三缓冲流水线<br>
上次：`cp.async` 的 commit group，以及 1D TMA memcpy 的 read/store 基本路径<br>
下次：WGMMA / `tcgen05` 如何消费已经落到 shared memory 的 tile<br>
PTX：TMA async proxy、`mbarrier` phase/byte completion、async-store commit group、generic/async proxy fence

### 为什么现在讲这个

前一节的 1D TMA 例程虽然把数据搬对了，但它仍是一条串行路径：

```text
TMA 读 A/B -> 等 TMA -> 计算 -> 写 C -> 等 store -> 下一轮
```

这样 TMA 的异步能力没有真正发挥出来。要形成流水线，必须同时维护两件事：让后续 tile 的 TMA read 提前发出，并且判断当前 buffer 什么时候可以被计算或覆盖。只加多缓冲但不维护 barrier phase，会出现“等待永远不返回”或“结果偶尔用到旧 tile”；只提前发 store 但不等待 store 读完 `c_smem`，又会覆盖正在被 TMA 读取的数据。

这一节用一个完整的 `a + b -> c` 例子说明三件事：

| 问题 | 本节机制 |
|---|---|
| 谁负责把 global tile 搬到 shared memory | `TensorDescriptor` + `tma.async_load` |
| 消费者怎么知道 read 已完成 | 每个 buffer 一个 `mbarrier`，按 byte 数完成 |
| 何时能复用 store staging buffer | `tma.store_wait(pendings=0)` 等待 TMA 已读完 shared memory |

### 心智模型：descriptor 是“搬运合同”，不是数据本身

普通 global load 需要 lane 自己计算地址和 mask：

```text
lane -> 计算 64-bit 地址 -> 计算越界 mask -> LDG -> register
```

TMA 把地址生成从 kernel 移到 descriptor：

```text
TensorDescriptor:
  base pointer
  global shape / strides
  dtype
  block_shape
  shared-memory layout

kernel:
  只给 block 的起始坐标
  TMA 硬件按 block_shape 搬运
```

例如 descriptor 的 global shape 是 `[xnumel, ynumel]`，`block_shape=[32, 64]`。当请求坐标是 `[xoff, yoff]` 时，TMA 搬的是：

```text
global[xoff : xoff + 32, yoff : yoff + 64]
```

坐标是元素坐标，不是 byte offset。descriptor 的 shape 会自动处理边界：越界 read 不会把非法地址变成访问，越界 write 也不会写出 tensor 外。这样 kernel 不需要为每个元素生成 mask。

TMA 的代价是地址灵活性下降。它要求 global tensor 的 base/stride、block shape 和 shared-memory layout 满足硬件约束；如果 descriptor 的布局条件不满足，就不能把普通 load 机械替换成 TMA。

### 完整可运行代码：三缓冲 elementwise add

下面的代码是官方 04 示例中与 TMA 流水线有关部分的完整版本，移到单个文件即可运行。它不依赖前一节 03 的辅助函数；末尾使用确定性的输入做正确性检查。

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
)


@gluon.jit
def issue_loads(
    copy_index,
    a_desc,
    b_desc,
    a_smem,
    b_smem,
    bars,
    xoff,
    YBLOCK: gl.constexpr,
    num_buffers: gl.constexpr,
):
    yoff = copy_index * YBLOCK
    bar = bars.index(copy_index % num_buffers)

    # 两次 TMA read 共用同一个 barrier，所以 expect 的是两块 tile 的总字节数。
    mbarrier.expect(
        bar,
        a_desc.block_type.nbytes + b_desc.block_type.nbytes,
    )
    tma.async_load(
        a_desc,
        [xoff, yoff],
        bar,
        a_smem.index(copy_index % num_buffers),
    )
    tma.async_load(
        b_desc,
        [xoff, yoff],
        bar,
        b_smem.index(copy_index % num_buffers),
    )
    return copy_index + 1


@gluon.jit
def perform_add(
    read_index,
    bars,
    a_smem,
    b_smem,
    c_smem,
    c_desc,
    xoff,
    layout: gl.constexpr,
    YBLOCK: gl.constexpr,
    num_buffers: gl.constexpr,
):
    # 同一个 barrier slot 每被环形复用一次，phase 翻转一次。
    read_phase = read_index // num_buffers & 1
    mbarrier.wait(
        bars.index(read_index % num_buffers),
        read_phase,
    )

    a_val = a_smem.index(read_index % num_buffers).load(layout)
    b_val = b_smem.index(read_index % num_buffers).load(layout)
    c_val = a_val + b_val

    yoff = read_index * YBLOCK

    # 保证上一笔 TMA store 已经读完 c_smem，才允许覆盖它。
    tma.store_wait(pendings=0)
    c_smem.store(c_val)

    # generic proxy 的 shared store 必须先对 async proxy 可见，
    # TMA store 才能安全地从 c_smem 读取。
    fence_async_shared()
    tma.async_store(c_desc, [xoff, yoff], c_smem)
    return read_index + 1


@gluon.jit
def elementwise_add_tma_kernel(
    a_desc,
    b_desc,
    c_desc,
    xnumel,
    ynumel,
    XBLOCK: gl.constexpr,
    YBLOCK: gl.constexpr,
    num_buffers: gl.constexpr,
):
    pid = gl.program_id(0)
    layout: gl.constexpr = gl.BlockedLayout(
        [1, 1], [1, 32], [1, 4], [1, 0]
    )
    xoff = pid * XBLOCK

    dtype: gl.constexpr = a_desc.type.block_type.element_ty
    a_smem = gl.allocate_shared_memory(
        dtype,
        [num_buffers, XBLOCK, YBLOCK],
        a_desc.layout,
    )
    b_smem = gl.allocate_shared_memory(
        dtype,
        [num_buffers, XBLOCK, YBLOCK],
        b_desc.layout,
    )
    c_smem = gl.allocate_shared_memory(
        dtype,
        [XBLOCK, YBLOCK],
        c_desc.layout,
    )

    bars = gl.allocate_shared_memory(
        gl.int64,
        [num_buffers, 1],
        mbarrier.MBarrierLayout(),
    )
    for i in gl.static_range(num_buffers):
        mbarrier.init(bars.index(i), count=1)

    copy_index = 0
    read_index = 0

    # Prologue：先发 num_buffers - 1 个 read，不等它们完成。
    for _ in gl.static_range(num_buffers - 1):
        copy_index = issue_loads(
            copy_index,
            a_desc,
            b_desc,
            a_smem,
            b_smem,
            bars,
            xoff,
            YBLOCK,
            num_buffers,
        )

    # Steady state：发出一块新的 read，同时消费最早发出去的那一块。
    for _ in range(
        gl.cdiv(ynumel, YBLOCK) - (num_buffers - 1)
    ):
        copy_index = issue_loads(
            copy_index,
            a_desc,
            b_desc,
            a_smem,
            b_smem,
            bars,
            xoff,
            YBLOCK,
            num_buffers,
        )
        read_index = perform_add(
            read_index,
            bars,
            a_smem,
            b_smem,
            c_smem,
            c_desc,
            xoff,
            layout,
            YBLOCK,
            num_buffers,
        )

    # Drain：清空仍在流水线里的最后 num_buffers - 1 个 tile。
    for _ in gl.static_range(num_buffers - 1):
        read_index = perform_add(
            read_index,
            bars,
            a_smem,
            b_smem,
            c_smem,
            c_desc,
            xoff,
            layout,
            YBLOCK,
            num_buffers,
        )

    for i in gl.static_range(num_buffers):
        mbarrier.invalidate(bars.index(i))

    # 等待最后一个 TMA store 读完 c_smem，之后 kernel 才能安全退出。
    tma.store_wait(pendings=0)


def elementwise_add_tma(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    XBLOCK: int = 32,
    YBLOCK: int = 64,
    num_buffers: int = 3,
) -> None:
    assert a.shape == b.shape == c.shape
    xnumel, ynumel = a.shape
    grid = (triton.cdiv(xnumel, XBLOCK),)

    block_shape = [XBLOCK, YBLOCK]
    layout = gl.NVMMASharedLayout.get_default_for(
        block_shape,
        gl.float32,
    )

    a_desc = TensorDescriptor.from_tensor(a, block_shape, layout)
    b_desc = TensorDescriptor.from_tensor(b, block_shape, layout)
    c_desc = TensorDescriptor.from_tensor(c, block_shape, layout)

    elementwise_add_tma_kernel[grid](
        a_desc,
        b_desc,
        c_desc,
        xnumel,
        ynumel,
        XBLOCK,
        YBLOCK,
        num_buffers,
    )


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise RuntimeError("this example requires CUDA")
    if torch.cuda.get_device_capability()[0] < 9:
        raise RuntimeError("TMA requires Hopper or newer")

    xnumel, ynumel = 256, 128
    a = torch.arange(xnumel * ynumel, dtype=torch.float32, device="cuda")
    a = a.reshape(xnumel, ynumel)
    b = torch.ones_like(a)
    c = torch.empty_like(a)

    elementwise_add_tma(a, b, c, XBLOCK=32, YBLOCK=64, num_buffers=3)
    torch.cuda.synchronize()
    torch.testing.assert_close(c, a + b, atol=0, rtol=0)
    print("tma_add_ok:", tuple(c.shape), c[0, :4].tolist())
```

预期输出：

```text
tma_add_ok: (256, 128) [1.0, 2.0, 3.0, 4.0]
```

当前工作机如果没有 Hopper 或更新 GPU，只能做源码静态检查；不能声称已经完成 TMA runtime 验证。

### 每个 buffer 为什么需要一个 mbarrier

假设 `num_buffers=3`：

```text
a_smem shape = [3, 32, 64]
b_smem shape = [3, 32, 64]
bars shape   = [3, 1]
```

第 `n` 次 read 使用 slot：

```text
slot = n % 3
```

如果三个 slot 共用一个 barrier，就无法区分“slot 0 已到达”和“slot 2 已到达”。TMA read 完成顺序也不保证严格等于 issue 顺序，所以必须让 buffer 与完成事件一一对应。

每个 barrier 的初始化计数是 `1`，但 TMA read 并不是简单调用一次 `arrive`。`mbarrier.expect(bar, bytes)` 先登记本次 transaction 一共会完成多少字节；TMA 完成一个 transaction 就减少对应 outstanding bytes。总字节数归零后，barrier 才完成当前 phase。

在 `issue_loads` 中，A 和 B 共用本次 slot 的 barrier：

```text
expect_bytes = bytes(A tile) + bytes(B tile)
```

如果只写 A 的字节数，B 可能还没到齐就完成 phase；如果字节数写多了，消费者会永远等不到完成。

### phase 公式怎么推出来

代码中的公式是：

```python
read_phase = read_index // num_buffers & 1
```

拆成两个动作：

```text
slot      = read_index % num_buffers
generation = read_index // num_buffers
phase      = generation & 1
```

`slot` 决定操作哪块 shared memory；`generation` 表示同一个物理 slot 第几次被复用。mbarrier 的 phase 只保存 0/1 parity，所以每复用一次翻转一次。

以 `num_buffers=3` 为例：

| `read_index` | slot | generation | 等待 phase |
|---:|---:|---:|---:|
| 0 | 0 | 0 | 0 |
| 1 | 1 | 0 | 0 |
| 2 | 2 | 0 | 0 |
| 3 | 0 | 1 | 1 |
| 4 | 1 | 1 | 1 |
| 5 | 2 | 1 | 1 |
| 6 | 0 | 2 | 0 |
| 7 | 1 | 2 | 0 |

这不是“phase 0 和 phase 1 交替代表两个不同 buffer”。phase 是同一个 barrier 的代际 parity；buffer 由 `read_index % num_buffers` 决定，二者是两个维度。

### 三缓冲的完整执行轨迹

假设 CTA 负责 `ynumel / YBLOCK = 8` 个 tile，`num_buffers=3`：

```text
copy_index = 0
read_index = 0
```

先做 prologue，发出 `3 - 1 = 2` 次 read：

```text
issue copy 0
issue copy 1
```

之后 6 轮 steady state：

| 轮次 | 新发 read | 同时消费 | 消费 buffer | 等待 phase |
|---:|---:|---:|---:|---:|
| 0 | copy 2 | read 0 | slot 0 | 0 |
| 1 | copy 3 | read 1 | slot 1 | 0 |
| 2 | copy 4 | read 2 | slot 2 | 0 |
| 3 | copy 5 | read 3 | slot 0 | 1 |
| 4 | copy 6 | read 4 | slot 1 | 1 |
| 5 | copy 7 | read 5 | slot 2 | 1 |

最后 drain 两次：

```text
consume read 6，等待 slot 0 phase 0
consume read 7，等待 slot 1 phase 0
```

所以三缓冲的含义不是“同时计算三个 tile”，而是“最多提前发出 `num_buffers - 1 = 2` 个尚未消费的 read”。这样当前 tile 做 load 到 register、add、store 时，后面的 TMA read 已经可以在硬件 async proxy 上并行进行。

### TMA store 的两个完成点

TMA store 和 TMA read 的等待方式不同。TMA read 通过 mbarrier 告诉消费者“数据已经写进 shared memory”；TMA store 通过隐式 async-store commit group 跟踪自己。

默认：

```python
tma.store_wait(pendings=0)
```

只保证 TMA 已经把源数据从 shared memory 读完，因此可以安全复用这块 `c_smem`。它不保证 tile 已经写到 global memory 并被其他 CTA 观察到。

如果生产者 CTA 要通过 global memory 通知消费者 CTA，语义需要升级为：

```python
tma.store_wait(pendings=0, read_only=False)
```

两者差别是：

```text
默认 store_wait:
  TMA 已读完 shared memory
  -> 可以覆盖 c_smem
  -> 不能据此宣称 global write 对其他 CTA 可见

read_only=False:
  再等待 global write 完成
  -> 可以跨 CTA 用 global memory 传递数据
```

官方 message-passing 例程之所以在 `atomic_xchg(..., sem="release")` 前使用 `read_only=False`，就是因为接收 CTA 之后要通过 TMA 从 global memory 读回消息。如果生产 CTA 只等默认完成点就发布 ready 标志，接收 CTA 可能与尚未落盘的 TMA store 竞争。

### 为什么 `c_smem` 复用前要先 `store_wait`

`perform_add` 中是这个顺序：

```text
tma.store_wait(pendings=0)   # 上一笔 TMA store 已读完 c_smem
c_smem.store(c_val)          # generic proxy 覆盖 shared memory
fence_async_shared()         # 让 async proxy 看见 generic store
tma.async_store(...)         # 发出新的 TMA store
```

如果去掉第一个 `store_wait`，一轮新的 `c_smem.store` 可能覆盖上一轮 TMA 还没读走的数据。症状不是稳定报错，而是某些输出 tile 混入后一轮内容。

如果去掉 `fence_async_shared()`，顺序会反过来：CPU 视角看到 shared store 已经执行，但 TMA async proxy 没有顺序保证，可能读到的仍是旧数据。这里不能用 TMA read 的 mbarrier 代替，因为没有等待这笔 generic store 的 mbarrier。

### shared memory 容量计算

官方教程给出的上限是每个 SM 约 `228 KB` shared memory。对 `64 x 128` 的 `float32` tile：

```text
单块 A tile = 64 * 128 * 4 = 32768 bytes = 32 KiB
单块 B tile = 32 KiB
A + B 一个 stage = 64 KiB
3 个 stage 的 A/B = 3 * 64 = 192 KiB
C staging tile = 64 * 128 * 4 = 32 KiB
合计 = 192 + 32 = 224 KiB
```

224 KiB 已经接近 228 KiB，因此这组配置能容纳三缓冲，但剩余余量很小。若改成 `128 x 128`，A+B 一个 stage 就是：

```text
2 * (128 * 128 * 4) = 128 KiB
```

双缓冲两个 stage 就要 256 KiB，还没算 C staging，已经超过 shared-memory 上限。所以 block size 不能只看 register pressure；TMA pipeline 往往先把 shared memory 吃满。

### descriptor 与坐标的硬约束

TMA descriptor 不是任意 strided tensor 都能接受，教程明确列出：

1. global tensor 的 stride 需要 16-byte aligned。
2. TMA 最内层坐标必须 16-byte aligned。
3. 如果 shared layout 是 `fp4_padded`，最内层坐标必须 128-byte aligned。

`float16` 下检查坐标 `[8, 4]`：

```text
最内层坐标 = 4
byte offset = 4 * 2 = 8 bytes
8 % 16 != 0
-> 非法
```

坐标 `[4, 8]`：

```text
最内层坐标 = 8
byte offset = 8 * 2 = 16 bytes
16 % 16 == 0
-> 合法
```

这里的“对齐”检查的是 descriptor 指令实际访问的地址，不是“最好能向量化”的性能建议。错误配置可能在构造 descriptor、编译或硬件执行时失败。

### 常见错误与可观察症状

| 错误 | 症状 | 修复 |
|---|---|---|
| `expect` 字节数少写 | 消费者提前通过 barrier，读到尚未完成的 tile | 对齐 `desc.block_type.nbytes` 与 TMA transaction 总字节 |
| `expect` 字节数多写 | `mbarrier.wait` 永久不返回 | 确保所有 TMA 都到达同一 barrier，且字节数精确 |
| phase 不按 slot generation 翻转 | 旧 tile 数据、随机数值错误或 kernel 挂起 | 使用 `read_index // num_buffers & 1` |
| 覆盖 `c_smem` 前不 `store_wait` | 输出 tile 混有相邻轮次内容 | 在覆盖 staging buffer 前等待上一笔 TMA store 读完 |
| generic shared store 后不加 fence | TMA 读到旧值，结果稳定或随机错误 | 调用 `fence_async_shared()` 后再发起 TMA store |
| 把默认 `store_wait` 当作 global 可见 | 跨 CTA 消费时偶发读到未落盘数据 | 使用 `read_only=False`，再发布 release 标志 |
| 盲目增加 buffer | 编译失败、occupancy 下降或性能没有提升 | 先算 shared memory 与 latency hiding 需求 |

### 自我检查

1. **为什么每个 pipeline buffer 要各自有一个 mbarrier？**
   - 答：TMA read 完成顺序不保证等于 issue 顺序；buffer slot 与完成事件必须一一对应，否则消费者可能等到错误 slot 的完成状态，或覆盖仍在传输的 buffer。
2. **`read_phase = read_index // num_buffers & 1` 为什么等价于“按 slot 的代际翻转”？**
   - 答：`read_index % num_buffers` 是 slot，`read_index // num_buffers` 是这个 slot 第几次被复用；mbarrier 只有 0/1 parity，所以对 generation 取最低位。
3. **TMA store 的默认 `store_wait(pendings=0)` 和 `read_only=False` 分别代表什么完成点？**
   - 答：默认只等到 TMA 已从 shared memory 读完源数据，可安全复用 shared buffer；`read_only=False` 还等到 global write 完成，适合跨 CTA 通过 global memory 传递数据。
4. **`c_smem.store(c_val)` 后为什么需要 `fence_async_shared()`？**
   - 答：普通 shared store 走 generic proxy，TMA store 从 async proxy 读取 shared memory；跨 proxy 没有隐式顺序，必须先 fence，否则 TMA 可能读到旧值。
5. **`XBLOCK=64, YBLOCK=128, float32, num_buffers=3` 为什么几乎吃满 228 KB shared memory？**
   - 答：A/B 每 stage 各 32 KiB，合计 64 KiB；三 stage 是 192 KiB；再加 32 KiB 的 C staging，共 224 KiB。
