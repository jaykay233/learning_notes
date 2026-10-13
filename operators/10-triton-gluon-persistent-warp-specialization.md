# Triton Gluon 07–08：Persistent Kernel 与 Warp Specialization

## 本次讲解位置

章节：`chapter_gluon_scheduling`（算子开发 / Triton Gluon）<br>
小节：静态 persistent tile scheduling，以及 load、MMA、epilogue 的 warp 角色拆分<br>
知识点：`@gluon.aggregate`、persistent outer loop、`PersistentTileScheduler`、`gl.warp_specialize`、跨 partition mbarrier<br>
上次：WGMMA 与 tcgen05 都是异步 tensor-core 操作，必须显式跟踪完成<br>
下次：进入 Blackwell 专属 TMA gather/scatter、TMEM copy、block-scaled MMA、CLC 与 im2col<br>
PTX：`bar.sync`、warpgroup 资源划分、TMA/WGMMA/tcgen05 与 mbarrier 接力

## 为什么现在讲这个

传统 Triton kernel 通常采用“一个 program 一个 tile”的写法：

```text
grid = (num_tiles,)
program i -> tile i -> exit
```

GPU 硬件动态调度这些 program。好处是负载均衡由硬件负责，代价是每个 tile 结束后 program 会退出，新 tile 重新启动，tile 之间很难保留 accumulator、barrier 和 pipeline 状态，也很难让上一个 tile 的 epilogue 与下一个 tile 的 load 重叠。

Persistent kernel 把“谁来处理哪些 tile”变成 kernel 自己的循环：

```text
grid = (min(num_sms, num_tiles),)
program i -> tile i, tile i + P, tile i + 2P, ... -> exit
```

Warp specialization 再向前走一步：不是所有 warp 都执行相同代码，而是把线程分成 load、MMA、epilogue 等角色，通过 shared memory 和 mbarrier 接力。

## Persistent 不等于普通的 grid-stride loop

普通 CUDA grid-stride loop 也可以让一个 block 处理多个 tile，但 Gluon persistent kernel 强调三件事：

1. tile 到 program 的映射由 `SchedulerImpl` 明确表达，可切换 row-major、grouped 或动态策略。
2. barrier、TMEM、部分 pipeline 状态跨 outer-loop iteration 保留。
3. load、MMA、epilogue 可以跨 tile 边界重叠，而不是每个 tile 开始时重新暖机。

区别可以画成：

```text
non-persistent:
prologue | load K | MMA K | epilogue | exit
prologue | load K | MMA K | epilogue | exit

persistent:
prologue | load tile0 K | MMA tile0 K | epilogue tile0 | \
                         load tile1 K | MMA tile1 K | epilogue tile1 | ...
```

真正获得收益的通常不是减少一次 launch，而是把不同 tile 的不同阶段重叠起来。

## PersistentTileScheduler

官方的基础 scheduler 使用连续区间分配：

```python
@gluon.aggregate
class PersistentTileScheduler:
    pid_start: gl.tensor
    pid_end: gl.tensor
    num_pid_m: gl.tensor

    @gluon.jit
    def initialize(M, N, BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr):
        kernel_id = gl.program_id(axis=0)
        num_kernels = gl.num_programs(axis=0)
        num_pid_m = gl.cdiv(M, BLOCK_M)
        num_pid_n = gl.cdiv(N, BLOCK_N)
        num_pid = num_pid_m * num_pid_n

        pid_per_kernel = gl.cdiv(num_pid, num_kernels)
        pid_start = kernel_id * pid_per_kernel
        pid_end = min(pid_start + pid_per_kernel, num_pid)
        return PersistentTileScheduler(pid_start, pid_end, num_pid_m)

    @gluon.jit
    def get_num_tiles(self):
        return self.pid_end - self.pid_start

    @gluon.jit
    def get_tile(self, idx):
        pid = self.pid_start + idx
        pid_m = pid % self.num_pid_m
        pid_n = pid // self.num_pid_m
        return pid_m, pid_n
```

例如有 10 个 tile、4 个 persistent program：

```text
program 0: tiles 0, 1, 2
program 1: tiles 3, 4, 5
program 2: tiles 6, 7
program 3: tiles 8, 9
```

这种静态分配简单且局部性好，但如果 tile 成本差异很大，某些 program 可能提前完成，另一些尾部拖长。后续 CLC 解决的问题正是这种静态分配尾部。

## Grouped scheduler 为什么能改善 L2 复用

`GroupedPersistentTileScheduler(GROUP_SIZE_M)` 改变 tile 顺序，使一个 program 每轮跨过 `num_programs` 个 tile，并在 M 方向按 group 聚类：

```python
pid = start_pid + idx * num_programs
group_id = pid // num_pid_in_group
first_pid_m = group_id * GROUP_SIZE_M
pid_m = first_pid_m + (pid % group_size_m)
pid_n = (pid % num_pid_in_group) // group_size_m
```

对于 GEMM，这意味着多个 concurrent program 更可能同时读取相同或相邻的 M tile，帮助 B 的 tile 命中 L2。官方 Hopper 结果里，`GROUP_SIZE_M=8` 常比 1 更快。

## @gluon.aggregate 的状态模型

Persistent 代码通常需要把 descriptor、barrier、accumulator、phase、counter 等打包成一个状态对象。Gluon 用：

```python
@gluon.aggregate
class WGMMA:
    acc: ...
    use_acc: gl.tensor
```

关键规则是 **aggregate 不支持原地 mutation**。方法不能写成：

```python
self.use_acc = gl.to_tensor(True)  # 错误思路
```

而必须返回新实例：

```python
@gluon.jit
def issue_async_mma(self, a, b):
    acc = warpgroup_mma(a, b, self.acc, is_async=True, use_acc=self.use_acc)
    return WGMMA(acc, gl.to_tensor(True))
```

调用处再重新赋值：

```python
mma = mma.issue_async_mma(a, b)
```

这不是烦琐的对象风格，而是在编译期明确状态依赖。每次状态变化都有新的 SSA 值，MMA 与等待之间的顺序可以被正确追踪。

## 统一 Hopper 与 Blackwell 的 MMA 接口

官方把 WGMMA 和 tcgen05 包装成相同的方法集合：

```text
initialize(dtype, BLOCK_M, BLOCK_N, num_warps)
issue_async_mma(a, b)
wait_num_outstanding(n)
take_result()
```

Hopper 的 `WGMMA` 直接保存 register accumulator，并用 `warpgroup_mma_wait` 等待。

Blackwell 的 `MMAv5` 保存 TMEM descriptor、barrier 和 issued-MMA counter，并用 mbarrier phase 等待。这样上层 persistent pipeline 不必写两套调度逻辑，只需要在 host 侧通过 `select_mma_impl()` 选择实现。

```python
def select_mma_impl():
    if torch.cuda.get_device_capability()[0] == 9:
        return WGMMA
    elif torch.cuda.get_device_capability()[0] == 10:
        return MMAv5
    return None
```

## Persistent 内核的完整结构

持久化不是简单地在最外层加 `for`。官方实现保留了 barrier 与 MMA 状态，只把 operand buffer 和 C store buffer 放在内层，让 shared-memory allocator 知道它们的生命周期不会与 epilogue buffer 冲突。

核心结构是：

```python
bars = allocate_shared_memory(...)
for i in static_range(num_buffers):
    mbarrier.init(bars.index(i), count=1)

producer = 0
consumer = 0
mma = MMAImpl.initialize(...)
scheduler = SchedulerImpl.initialize(M, N, BLOCK_M, BLOCK_N)

for idx in range(scheduler.get_num_tiles()):
    pid_m, pid_n = scheduler.get_tile(idx)

    a_bufs = allocate_shared_memory(...)
    b_bufs = allocate_shared_memory(...)

    # prefetch
    for k in static_range(0, BLOCK_K * (num_buffers - 2), BLOCK_K):
        producer = issue_loads(...)

    # steady state
    for k in range(...):
        producer = issue_loads(...)
        consumer, mma = issue_mma(...)

    # drain
    for _ in static_range(num_buffers - 2):
        consumer, mma = issue_mma(...)

    mma = mma.wait_num_outstanding(0)
    c, mma = mma.take_result()
    tma.async_store(...)
```

### 三段循环的意义

| 阶段 | 作用 |
|---|---|
| prefetch | 先把 `num_buffers - 2` 个 load 放入在途，保留 2 个 buffer 给 MMA 和下一轮 load 重叠 |
| steady | 每轮先发新 load，再消费旧 buffer 并发射异步 MMA |
| drain | 数据全部装载后，继续消费剩余的 buffer，等待最后一个 MMA 完成 |

`issue_mma` 在发射新的 MMA 前调用 `wait_num_outstanding(0)`，保证 `acc_tmem` 或 register accumulator 已被合理接管。对 Blackwell 而言，它还负责让 TMEM 的旧依赖有明确完成点。

## Persistent 的数值与性能轨迹

官方数据中，Hopper `K=8192`：

| 版本 | TFLOPS |
|---|---:|
| 非 persistent | `629.44` |
| persistent | `646.22` |
| persistent + pipeline | `661.57` |
| cuBLAS | `661.11` |

pipeline persistent 版本大致追平 cuBLAS。这里的收益来自跨 tile 的 load/MMA/epilogue 重叠和更少的状态重建，不只是“少 launch 几个 program”。

## Warp specialization 的心智模型

普通 block-level pipeline 是所有 warp 都循环执行 load、MMA、store：

```text
all warps: load -> wait -> mma -> wait -> epilogue -> load ...
```

Warp specialization 把角色固定下来：

```text
load partition:  专做 TMA load
MMA partition:   专做 tensor-core issue
epilogue partition: 专做 TMEM/register -> SMEM -> global store
default partition: 初始化或处理标量控制流
```

角色之间不通过函数调用返回数据，而是通过 shared-memory buffer 和 mbarrier 交接：

```text
load_empty  --load partition--> load_ready
load_ready  --MMA partition--> load_empty
acc_empty   --MMA partition--> acc_ready
acc_ready   --epilogue------> acc_empty
```

这使不同 warp 可以同时执行，而不是全 block 在同一个 `bar.sync` 上等待。

## gl.warp_specialize 的调用形式

官方 elementwise-add 示例：

```python
gl.warp_specialize(
    [
        (compute_partition, (barriers, buffers, ynumel, YBLOCK, layout)),
        (load_partition, (descs, barriers, buffers, xoff, numel, YBLOCK)),
        (store_partition, (descs, barriers, buffers, xoff, numel, YBLOCK)),
    ],
    [1, 1],
    [24, 24],
)
```

参数有三层：

| 位置 | 含义 |
|---|---|
| 第一个列表 | 每个 partition 的函数和参数 |
| `[1, 1]` | 两个 worker partition 的 warpgroup 数 |
| `[24, 24]` | 两个 worker partition 的 register budget |

### 约束

- 所有 worker partition 的函数签名必须一致；只有 default partition 可以接 tensor 参数。
- 高级 IR 中的 worker partition 参数不能包含分布式 tensor，避免隐含的跨 partition 数据依赖。
- warp 与 register 分配按 warpgroup 进行，即 4 个连续 warp 一组。
- kernel 的 warp 数会向上取整到 4 的倍数。
- default partition 得到剩余 register；host 侧 `maxnreg` 会影响这个预算。
- cluster barrier 必须由 kernel 中所有 thread 执行，因此不能放进 `warp_specialize`。
- 跨 CTA 的 `convert_layout`、reduce、sum、max 会插入 cluster barrier，也不能出现在可能跨 CTA 的 warp-specialized block 中。

## Warp-specialized elementwise add 的完整代码

下面保留官方 08 教程的三角色结构。它计算 `C = A + B`，但把 load、compute、store 放入不同 partition。运行需要 Hopper 或更新 GPU。

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
)


@gluon.jit
def load_partition(descs, barriers, buffers, xoff, numel, YBLOCK: gl.constexpr):
    a_desc, b_desc, c_desc = descs
    load_empty_bars, load_ready_bars, c_empty_bars, c_ready_bars = barriers
    a_bufs, b_bufs, c_bufs = buffers
    _, ynumel = numel
    num_buffers: gl.constexpr = a_bufs.type.shape[0]

    for i in range(gl.cdiv(ynumel, YBLOCK)):
        index = i % num_buffers
        phase = i // num_buffers & 1
        a_buf = a_bufs.index(index)
        b_buf = b_bufs.index(index)
        empty_bar = load_empty_bars.index(index)
        ready_bar = load_ready_bars.index(index)

        # mbarrier 初始 phase 为 1 complete，因此 producer 从 phase ^ 1 开始等待。
        mbarrier.wait(empty_bar, phase ^ 1)
        yoff = i * YBLOCK
        mbarrier.expect(
            ready_bar,
            a_desc.block_type.nbytes + b_desc.block_type.nbytes,
        )
        tma.async_load(a_desc, [xoff, yoff], ready_bar, a_buf)
        tma.async_load(b_desc, [xoff, yoff], ready_bar, b_buf)


@gluon.jit
def compute_partition(
    barriers,
    buffers,
    ynumel,
    YBLOCK: gl.constexpr,
    layout: gl.constexpr,
):
    load_empty_bars, load_ready_bars, c_empty_bars, c_ready_bars = barriers
    a_bufs, b_bufs, c_bufs = buffers
    num_load_buffers: gl.constexpr = a_bufs.type.shape[0]
    num_store_buffers: gl.constexpr = c_bufs.type.shape[0]

    for i in range(gl.cdiv(ynumel, YBLOCK)):
        load_index = i % num_load_buffers
        load_phase = i // num_load_buffers & 1
        a_buf = a_bufs.index(load_index)
        b_buf = b_bufs.index(load_index)
        ready_bar = load_ready_bars.index(load_index)
        empty_bar = load_empty_bars.index(load_index)

        mbarrier.wait(ready_bar, load_phase)
        a_val = a_buf.load(layout)
        b_val = b_buf.load(layout)
        fence_async_shared()
        mbarrier.arrive(empty_bar, count=1)
        c_val = a_val + b_val

        store_index = i % num_store_buffers
        store_phase = i // num_store_buffers & 1
        c_buf = c_bufs.index(store_index)
        c_empty_bar = c_empty_bars.index(store_index)
        c_ready_bar = c_ready_bars.index(store_index)

        mbarrier.wait(c_empty_bar, store_phase ^ 1)
        c_buf.store(c_val)
        fence_async_shared()
        mbarrier.arrive(c_ready_bar, count=1)


@gluon.jit
def store_partition(descs, barriers, buffers, xoff, numel, YBLOCK: gl.constexpr):
    a_desc, b_desc, c_desc = descs
    load_empty_bars, load_ready_bars, c_empty_bars, c_ready_bars = barriers
    a_bufs, b_bufs, c_bufs = buffers
    _, ynumel = numel
    num_buffers: gl.constexpr = c_bufs.type.shape[0]
    outstanding_stores: gl.constexpr = num_buffers - 1

    for i in range(gl.cdiv(ynumel, YBLOCK)):
        index = i % num_buffers
        phase = i // num_buffers & 1
        c_buf = c_bufs.index(index)
        c_ready_bar = c_ready_bars.index(index)
        c_empty_bar = c_empty_bars.index((i - outstanding_stores) % num_buffers)

        mbarrier.wait(c_ready_bar, phase)
        yoff = i * YBLOCK
        tma.async_store(c_desc, [xoff, yoff], c_buf)
        tma.store_wait(outstanding_stores)
        mbarrier.arrive(c_empty_bar, count=1, pred=i >= outstanding_stores)

    tma.store_wait(0)


@gluon.jit
def elementwise_add_ws_kernel(
    a_desc,
    b_desc,
    c_desc,
    xnumel,
    ynumel,
    XBLOCK: gl.constexpr,
    YBLOCK: gl.constexpr,
    num_load_buffers: gl.constexpr,
    num_store_buffers: gl.constexpr,
    num_warps: gl.constexpr,
):
    layout: gl.constexpr = gl.BlockedLayout(
        [1, 1], [1, 32], [1, num_warps], [1, 0]
    )
    a_bufs = gl.allocate_shared_memory(
        a_desc.dtype, [num_load_buffers] + a_desc.block_type.shape, a_desc.layout
    )
    b_bufs = gl.allocate_shared_memory(
        b_desc.dtype, [num_load_buffers] + b_desc.block_type.shape, b_desc.layout
    )
    c_bufs = gl.allocate_shared_memory(
        c_desc.dtype,
        [num_store_buffers] + c_desc.block_type.shape,
        c_desc.layout,
    )

    load_empty_bars = gl.allocate_shared_memory(
        gl.int64, [num_load_buffers, 1], mbarrier.MBarrierLayout()
    )
    load_ready_bars = gl.allocate_shared_memory(
        gl.int64, [num_load_buffers, 1], mbarrier.MBarrierLayout()
    )
    c_empty_bars = gl.allocate_shared_memory(
        gl.int64, [num_store_buffers, 1], mbarrier.MBarrierLayout()
    )
    c_ready_bars = gl.allocate_shared_memory(
        gl.int64, [num_store_buffers, 1], mbarrier.MBarrierLayout()
    )

    for i in gl.static_range(num_load_buffers):
        mbarrier.init(load_empty_bars.index(i), count=1)
        mbarrier.init(load_ready_bars.index(i), count=1)
    for i in gl.static_range(num_store_buffers):
        mbarrier.init(c_empty_bars.index(i), count=1)
        mbarrier.init(c_ready_bars.index(i), count=1)

    descs = (a_desc, b_desc, c_desc)
    barriers = (load_empty_bars, load_ready_bars, c_empty_bars, c_ready_bars)
    buffers = (a_bufs, b_bufs, c_bufs)
    numel = (xnumel, ynumel)
    xoff = gl.program_id(0) * XBLOCK

    gl.warp_specialize(
        [
            (compute_partition, (barriers, buffers, ynumel, YBLOCK, layout)),
            (load_partition, (descs, barriers, buffers, xoff, numel, YBLOCK)),
            (store_partition, (descs, barriers, buffers, xoff, numel, YBLOCK)),
        ],
        [1, 1],
        [24, 24],
    )


def elementwise_add_ws(a, b, c, XBLOCK=32, YBLOCK=64, num_warps=4):
    xnumel, ynumel = a.shape
    block_shape = [XBLOCK, YBLOCK]
    layout = gl.NVMMASharedLayout.get_default_for(block_shape, gl.float32)
    a_desc = TensorDescriptor.from_tensor(a, block_shape, layout)
    b_desc = TensorDescriptor.from_tensor(b, block_shape, layout)
    c_desc = TensorDescriptor.from_tensor(c, block_shape, layout)

    grid = (triton.cdiv(xnumel, XBLOCK),)
    elementwise_add_ws_kernel[grid](
        a_desc,
        b_desc,
        c_desc,
        xnumel,
        ynumel,
        XBLOCK,
        YBLOCK,
        2,
        2,
        num_warps=num_warps,
        maxnreg=128,
    )


if __name__ == "__main__":
    xnumel, ynumel = 1000, 2000
    a = torch.randn(xnumel, ynumel, device="cuda")
    b = torch.randn(xnumel, ynumel, device="cuda")
    c = torch.empty_like(a)
    elementwise_add_ws(a, b, c)
    torch.testing.assert_close(a + b, c, atol=0, rtol=0)
    print("warp_specialized_add_ok", c.sum().item())
```

完整 benchmark 和 matrix specialization 版本见 [`08-warp-specialization.py`](code/triton/gluon/tutorials/08-warp-specialization.py)。

## Barrier phase 的具体轨迹

设每个 buffer 的 barrier 初始为 phase 1 complete。两 buffer 的 load 通道可以简化为：

```text
iteration 0:
  load waits empty[0] phase 1 -> 立即通过
  TMA(0) completes -> ready[0] phase 0 -> phase 1
  compute waits ready[0] phase 0
  compute releases empty[0]

iteration 1:
  load reuses buffer 1, phase 1
  compute waits ready[1] phase 0

iteration 2:
  load reuses buffer 0, now waits empty[0] phase 0
```

这里的 `phase = i // num_buffers & 1` 决定每一轮 barrier 应该等待哪一个 parity。两 buffer 时每两次访问翻转一次。若把 phase 写成 `i & 1` 或忘记翻转，常见症状是第一次能跑、第二轮挂起，或消费者读到上一轮的数据。

## 性能位置

官方结果：

| 版本 | 吞吐 |
|---|---:|
| elementwise TMA | `5.89 TB/s` |
| elementwise warp-specialized | `5.98 TB/s` |

Blackwell GEMM 与 cuBLAS 的差距则随 K 变化：

| K | warp specialized | cuBLAS |
|---:|---:|---:|
| 512 | `1004.18` | `1191.77` |
| 1024 | `1182.61` | `1334.85` |
| 8192 | `1291.56` | `1301.11` |

短 K 时 cuBLAS 优势明显，长 K 时调度和 pipeline 开销被摊薄，差距减小。Warp specialization 不是自动加速，它需要 work 足够长，能够抵消额外 barrier 和角色划分的成本。

## 常见错误与症状

| 错误 | 典型症状 | 根因与修复 |
|---|---|---|
| persistent grid 不是 `min(num_sms, num_tiles)` | 每个 program tile 太少或空闲 SM 过多 | 先限制 program 数，再让 scheduler 分配多个 tile |
| static scheduler 负载不均 | 尾部一条 SM 拖慢 | 改 grouped 调度，或使用 CLC 动态接管 |
| aggregate 原地改状态 | 编译错误或状态依赖丢失 | 返回新实例并在 call site 重新赋值 |
| partition 函数签名不一致 | `warp_specialize` 编译失败 | 所有 worker 参数列表保持一致 |
| worker partition 传 tensor | 编译失败或 IR 不合法 | worker 只传 descriptor、barrier、buffer 和标量 |
| phase 不随 round 翻转 | 第二轮挂起或读旧数据 | 使用 `i // num_buffers & 1` 与 `phase ^ 1` |
| TMA 与泛型 SMEM load 之间缺 fence | 数据竞争或看到旧值 | 在 producer/consumer 交接处使用 `fence_async_shared` |
| `maxnreg` 设置过大 | 默认 partition 挤压、occupancy 下降 | 按 `maxnreg * warps * 32` 估算并实测 |
| 在 warp specialization 中做跨 CTA reduce | cluster barrier 非法 | 把跨 CTA layout 操作移到 top-level |
| 试图用 WS 解决短 kernel | 开销大于收益 | 保持普通 kernel，或减少 partition 数 |

## 硬件与验证边界

`07-persistence.py` 要求 Hopper 或更新；`08-warp-specialization.py` 的基础 elementwise 版本也要求 Hopper 或更新。TMEM 和 tcgen05 分支只在 Blackwell 上执行。当前工作区没有 Triton runtime 与 CUDA GPU，本节只做源码静态检查；phase race、warpgroup 资源分配和吞吐必须上目标 GPU 验证。

## 自我检查

1. **Persistent kernel 比普通 tile-per-program kernel 的主要收益是什么？**
   - 答：减少 program launch 与状态重建开销，并允许跨 tile 重叠 load、MMA 和 epilogue；如果调度不均，也可能引入尾部问题。
2. **`@gluon.aggregate` 为什么不能原地修改字段？**
   - 答：Gluon 用返回值表达新的状态依赖；原地修改会让编译期无法准确追踪 accumulator 和 phase 的 SSA 关系。
3. **`gl.warp_specialize` 的三个列表分别表示什么？**
   - 答：partition 函数及参数、各 worker 的 warpgroup 数、各 worker 的 register budget。
4. **load 和 compute partition 为什么需要 empty/ready 两组 barrier？**
   - 答：ready 表示数据已可消费，empty 表示 buffer 已可复用；没有 empty 侧，producer 会覆盖仍被消费者读取的 buffer。
5. **为什么短 K GEMM 的 warp specialization 不保证快？**
   - 答：角色划分、barrier 交接和额外寄存器预算有固定成本，K 太短时摊不薄，cuBLAS 的成熟调度反而更接近天花板。
