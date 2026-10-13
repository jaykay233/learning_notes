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

---

## 官方 Persistent Kernels 全链路：持久化调度、K 流水与借用 B buffer 写回

### 本次讲解位置

章节：`operators / Triton Gluon`（第 10 章）<br>
小节：官方 `07-persistence.py`：`persistent_matmul` → `persistent_matmul_pipelined`<br>
知识点：一个 CTA 静态处理多个输出 tile；复用 MMA/TMA 状态；将下一个 tile 的预取与当前 tile 的 epilogue 重叠；用额外 B-buffer 容纳异步输出<br>
上次：本章已学 `PersistentTileScheduler`、K 方向 load/MMA 双缓冲，以及 `gl.warp_specialize` 的角色分工<br>
下次：Blackwell CLC 动态领活、TMA gather/scatter 与多 CTA/cluster 路径<br>
PTX：教程没有列 PTX 清单；对应硬件路径是 TMA、mbarrier、Hopper WGMMA 或 Blackwell tcgen05

官方教程：[Persistent Kernels](https://triton-lang.org/main/getting-started/tutorials/gluon/persistence.html)。完整官方源码（包含 imports、Hopper/Blackwell MMA adapter、scheduler、kernel、host wrapper、pytest 和 benchmark）：[07-persistence.py](code/triton/gluon/tutorials/07-persistence.py)。这份归档源码与学习笔记一起保存在仓库中；下文聚焦它的核心数据流和容易读错的最终优化。

### 为什么现在讲这个

前面学到的 K-loop pipeline 只回答了“同一个输出 tile 的多个 K-slice 如何重叠加载与 MMA”。官方 persistence 教程继续追问两个更具体的问题：

1. **同一 CTA 算完一个输出 tile 后，能否保留 barrier、MMA accumulator 状态和预取节奏，继续算下一个 tile？**
2. **当前 tile 的结果正通过 TMA 写回时，能否提前为下一个 tile 加载 A/B，而不额外分配一整块输出 SMEM？**

第一问引入 persistent outer loop；第二问导致最终版本的 `STEALB`：临时借用 B 的共享内存 buffer 作为 epilogue 的 TMA-store source。若只记住“persistent = 少启动 CTA”，就会漏掉本教程真正的资源设计，也会错误地以为 persistent 一定更快。

### 心智模型：CTA 变成常驻的 tile worker

普通 GEMM kernel 通常是：

```text
grid = 输出 tile 总数
每个 program：算一个 (pid_m, pid_n) → 写回 → 结束
```

Persistent kernel 则把 grid 缩小到约一个 program/SM，再在每个 program 内循环：

```text
grid = min(SM 数, 输出 tile 数)
program p：tile p → tile p+P → tile p+2P → ... → 结束
```

其中 `P = gl.num_programs(axis=0)`。CTA 是“常驻工人”，tile scheduler 是它的静态派工表；CTA 的异步状态可跨多个输出 tile 保留。

要非常精确地理解收益：普通 GPU grid 中，一个 CTA 结束后，硬件仍可把同一 kernel 的其他 CTA 调度到空出的 SM；persistent 并不是“普通 kernel 必须等全部 CTA 结束才会继续派活”。Persistent 的不同在于**把 tile 指派、跨 tile 状态复用和流水安排交给 kernel 显式控制**。代价是静态派工失去硬件动态负载均衡，尾部可能不均。

### 1. Host grid：先限制常驻 CTA 数

官方 wrapper 的关键计算是：

```python
num_sms = torch.cuda.get_device_properties("cuda").multi_processor_count
num_pid = triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N)
grid = (min(num_sms, num_pid),)
```

- `num_pid_m = ceil(M / BLOCK_M)`，`num_pid_n = ceil(N / BLOCK_N)`。
- 输出 tile 总数 `num_pid = num_pid_m * num_pid_n`。
- grid 一维，只有 `P=min(num_sms,num_pid)` 个 CTA。
- `num_warps` 是每个 CTA 的 warp 配置，不是 tile 数，也不是 SM 数。

例如有 10 个输出 tile、4 个常驻 program，基础 scheduler 用 `ceil(10/4)=3` 个 tile 作为每个 program 的连续区间：

| CTA/program | 负责的线性 tile ID |
|---|---|
| 0 | 0、1、2 |
| 1 | 3、4、5 |
| 2 | 6、7、8 |
| 3 | 9 |

这很简单，但最后一个 CTA 只有一个 tile。tile 成本若不同，其他 CTA 会先空闲，最后变成静态调度的尾部。

### 2. `PersistentTileScheduler`：tile ID 展开成 M/N 坐标

基础 scheduler 把矩阵分成二维 tile 网格，再把线性 ID 展开：

```python
pid = pid_start + idx
pid_m = pid % num_pid_m
pid_n = pid // num_pid_m
```

M 方向变化最快。若 `num_pid_m=5`，线性 tile 0、1、2、3、4 对应 `(m,n)=(0,0),(1,0),(2,0),(3,0),(4,0)`；tile 5 回到 `(0,1)`。

program 的 `pid_start/pid_end` 在初始化时计算一次，`get_tile(idx)` 则在 outer loop 中给出此 program 的第 `idx` 个 tile。最关键的生命周期安排是：

- `bars`、`producer/consumer`、`mma`、`scheduler` 建立在输出 tile 循环外，允许复用状态。
- plain persistent 版本的 `a_bufs/b_bufs` 建立在 tile 循环内，缩短其 SMEM live range，避免与输出 store buffer 同时占用空间。
- 每一个输出 tile 结束后，`mma.take_result()` 返回结果并把 accumulator 的 `use_acc` 状态重置；否则下一个输出 tile 会把上一个 tile 的 accumulator 当作初值，算错。

### 3. `GroupedPersistentTileScheduler`：改变顺序是为了数据复用

行列 tile 顺序会影响同时运行的 CTA 是否能共享 L2 中的数据。官方 grouped scheduler 使用：

```python
tile_id = start_pid + idx * gl.num_programs(axis=0)
group_id = tile_id // (GROUP_SIZE_M * num_pid_n)
first_pid_m = group_id * GROUP_SIZE_M
group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
pid_m = first_pid_m + tile_id % group_size_m
pid_n = (tile_id % (GROUP_SIZE_M * num_pid_n)) // group_size_m
```

以 `num_pid_m=5`、`num_pid_n=3`、`GROUP_SIZE_M=2` 为例，按 scheduler 的线性 tile 顺序：

| tile ID | `(pid_m, pid_n)` | tile ID | `(pid_m, pid_n)` |
|---:|---:|---:|---:|
| 0 | (0,0) | 8 | (2,1) |
| 1 | (1,0) | 9 | (3,1) |
| 2 | (0,1) | 10 | (2,2) |
| 3 | (1,1) | 11 | (3,2) |
| 4 | (0,2) | 12 | (4,0) |
| 5 | (1,2) | 13 | (4,1) |
| 6 | (2,0) | 14 | (4,2) |
| 7 | (3,0) |  |  |

对于 GEMM，固定 `pid_n` 代表多个输出 tile 共用同一块 B（同一个 N tile、同一 K-slice）。按 M 分组可让邻近 CTA 更可能重复使用 B tile，提高 L2 命中；这不是数学变化，只是输出 tile 的访问顺序变化。

分组不是免费午餐。官方这次测试里，Blackwell 的普通 persistent 路径 L2 hit rate 是 52.93%，`GROUP_SIZE_M=8` 的测试可升到 70%；但 Hopper 上 grouped 调度会增加负载不均。最佳分组依 GPU、形状和 tile 大小而变，不能把 8 当普适常数。

下面是一个可直接用 Python 运行的 scheduler 轨迹器，复现 `(num_pid_m=5, num_pid_n=3, GROUP_SIZE_M=2)`，再把 15 个 tile 轮转分给 4 个常驻 CTA：

```python
def grouped_order(num_pid_m, num_pid_n, group_size_m):
    num_pid_in_group = group_size_m * num_pid_n
    order = []
    for tile_id in range(num_pid_m * num_pid_n):
        group_id = tile_id // num_pid_in_group
        first_pid_m = group_id * group_size_m
        group_m = min(num_pid_m - first_pid_m, group_size_m)
        pid_m = first_pid_m + tile_id % group_m
        pid_n = (tile_id % num_pid_in_group) // group_m
        order.append((pid_m, pid_n))
    return order

order = grouped_order(5, 3, 2)
num_programs = 4
assignments = {
    pid: [order[tile_id] for tile_id in range(pid, len(order), num_programs)]
    for pid in range(num_programs)
}
assert assignments == {
    0: [(0, 0), (0, 2), (2, 1), (4, 0)],
    1: [(1, 0), (1, 2), (3, 1), (4, 1)],
    2: [(0, 1), (2, 0), (2, 2), (4, 2)],
    3: [(1, 1), (3, 0), (3, 2)],
}
for pid, tiles in assignments.items():
    print(f"CTA {pid}: {tiles}")
```

预期输出：

```text
CTA 0: [(0, 0), (0, 2), (2, 1), (4, 0)]
CTA 1: [(1, 0), (1, 2), (3, 1), (4, 1)]
CTA 2: [(0, 1), (2, 0), (2, 2), (4, 2)]
CTA 3: [(1, 1), (3, 0), (3, 2)]
```


### 4. 同一个 persistent CTA 内，K 方向如何流水？

以 `BLOCK_K=64`、`num_buffers=3`、`K=256` 为例，共有 4 个 K-slice：`k=0,64,128,192`。双层逻辑是：

```text
prefetch：先装入 num_buffers-2 = 1 个 slice，即 K0
steady：每次发出下一个 TMA load，同时消费前一个已就绪 slice：
        load K1 + MMA K0
        load K2 + MMA K1
        load K3 + MMA K2
drain：把最后留在 pipeline 中的 K3 送入 MMA
```

`issue_loads()` 的 producer 序号和 `issue_mma()` 的 consumer 序号分别按 buffer 数取模，形成环形缓冲。consumer 对应的 `mbarrier` phase 为：

```python
phase = consumer // num_buffers & 1
```

当同一 barrier 绕环复用时，phase 翻转，防止消费者把上一轮已经完成的 barrier 状态误认成新一轮完成。读取 MMA accumulator 前，adapter 的 `wait_num_outstanding(0)` 还会等待所有尚未完成的 MMA：Hopper 用 `warpgroup_mma_wait`，Blackwell 用 tcgen05 的 mbarrier/counter。

`num_buffers` 同时受共享内存、MMA 吞吐、TMA 吞吐与 accumulator 占用影响。官方测试里 Hopper 从 3 增加到 4 个 buffer 基本不再收益；Blackwell 在指定形状上 4 个 buffer 有收益。这说明它是测量参数，不是 buffer 越多越快。

### 5. 最终难点：让当前 epilogue 与下一 tile 的预取重叠

没有额外设计时，输出 epilogue 需要把 accumulator 放进 `c_smem`，再由 TMA 写回：

```text
TMEM/register accumulator → c_smem → TMA store → global C
```

若要让 TMA store 尚未完成时就开始下一 tile 的 TMA load，`c_smem` 的源数据仍必须保持不变；同时 operand buffers 也不能被提前覆盖。额外单独分配输出 buffer 会让 SMEM 占用和 live range 增大。因此官方给出三种方向：普通 `gl.store`（布局转换仍可能要 SMEM，且不方便重叠）、拆分 store，或借用一个/多个 B buffer。

最终版本选第三种，`STEALB` 在 `num_buffers==4` 时启用：

```python
b_bufs = gl.allocate_shared_memory(
    dtype,
    [num_buffers + STEALB] + b_desc.block_type.shape,
    b_desc.layout,
)
```

对 `BLOCK_M=128, BLOCK_N=256, BLOCK_K=64, FP16`：

- 一个 B tile 是 `64×256×2 = 32 KiB`。
- 一个输出 tile 是 `128×256×2 = 64 KiB`。
- 输出正好是一个 B tile容量的两倍，所以把输出沿 N 切成两半；每半 `128×128×2 = 32 KiB`，刚好塞进一个 B tile 大小的 buffer。
- `STEALB=1` 将 B buffer 数从 4 增至 5；steady-state 仍按需要消费 operand buffer，额外的可用槽参与保存 epilogue。输出结果通过 `c_half_desc` 描述成两个 `(BLOCK_M, BLOCK_N/2)` 的 TMA store。

`BLOCK_M == BLOCK_K` 时，一个 B buffer 的元素数与完整输出 tile 相同，能直接容纳输出；`BLOCK_M == 2*BLOCK_K` 时需要两个 B buffer，源码里的 `static_assert` 正是保证这两种容量关系成立。这里的 `reinterpret(...)` 是把同一片共享内存按输出 half-tile 的 shape/layout 重新解释，不是复制出一块新 buffer。

producer 对 A/barrier 使用 `% num_buffers`，而 B 使用 `% (num_buffers + STEALB)`；消费者同样采用对应的 B 索引。也就是说，A 的环和 barrier 环仍是 4 格，B 的环扩成 5 格，因为只有 B 侧被借去充当 store source。

store source 的同步契约不能省：

1. `c_buf.store(...)` 把 accumulator 内容写入共享内存。
2. `fence_async_shared()` 让之后的 TMA async proxy 看见这次共享内存写入。
3. `tma.async_store(...)` 发起写回。
4. 下一轮真正复用同一 B buffer 前，`tma.store_wait(pendings=0)` 保证上一个 store 已不再读取该 buffer。

这就是为什么最终版本把 tile prologue 剥离出来（peeled prologue），并用谓词 `pred = idx < num_tiles` 处理最后一次“预取下一个 tile”：它把下一个 tile 的 TMA load 安排在当前 tile 的计算/epilogue 交界处，同时保证最后一轮不会访问不存在的 tile。

注意，**这份 persistence 教程的 GEMM kernel 没有调用 `gl.warp_specialize`**。它用普通 CTA 控制流发起 TMA/MMA 异步工作，并让硬件异步单元与 CTA 执行重叠；warp specialization 是同一章另一教程的技术，不要因为 persistent 和 warp-specialized 经常组合就把两者当同义词。

### 6. 用一个完整形状串起来

取 `M=256, N=512, K=256`，`BLOCK_M=128, BLOCK_N=256, BLOCK_K=64`：

1. 输出网格是 `ceil(256/128) × ceil(512/256) = 2×2 = 4` 个 tile：`(0,0),(1,0),(0,1),(1,1)`（按基础 scheduler 的线性展开顺序）。
2. 若机器有至少 2 个 SM，则 `P=min(num_sms,4)`。例如 P=2 时，每个 persistent CTA 处理 2 个输出 tile；如果 P=4，则每个只处理 1 个 tile，持久化的跨 tile 复用空间较少。
3. 每个输出 tile 的 K 方向有 `256/64=4` 次 MMA。用 4 个 buffer 时，prologue 先发若干 TMA load，steady state 一边加载下一 K-slice，一边消费已 ready 的 slice，之后 drain。
4. 最后一轮 K-slice 完成后，`mma.wait_num_outstanding(0)` 保证 accumulator 可读；`take_result()` 把本 tile 累加值拿出并重置下一 tile 的累加状态。
5. epilogue 将 `128×256` 输出分成左右两个 `128×128` half，写入两个 32 KiB shared-memory 槽，fence 后发出两次 TMA store。下一 tile 的 load 只能复用已经确认 store 完成的槽。

输出数学仍然只是：

```text
C[pid_m*128 : (pid_m+1)*128,
  pid_n*256 : (pid_n+1)*256]
  = A_tile @ B_tile
```

Persistent 调度、buffer 轮转和 epilogue 拆分都不得改变这个结果。

### 7. 性能结果该怎样读

官方 benchmark 使用 `M=N=8192`，改变 K，对比非 persistent、persistent、pipelined persistent 和 cuBLAS。Blackwell 官方表格（TFLOP/s，属于教程作者的测试设备，不是当前机器保证值）：

| K | 非 persistent | persistent | persistent + pipeline | cuBLAS |
|---:|---:|---:|---:|---:|
| 512 | 615.86 | 828.70 | 993.50 | 1108.11 |
| 2,048 | 1152.74 | 1190.55 | 1133.37 | 1435.01 |
| 4,096 | 1164.05 | 1120.92 | 1143.47 | 1563.98 |
| 16,384 | 1185.62 | 1096.34 | 1296.93 | 1548.42 |

从表里不要得出“persistent 总是快”或“pipeline 总是快”：K=2048 时 pipelined persistent 反而比 persistent 单独版本低；K=4096 时 persistent 版本都比 non-persistent 低。短 K 时跨 tile 管线更可能摊薄 epilogue 占比；长 K 时更容易被 MMA 吞吐主导。静态负载不均、L2 局部性、SMEM/TMEM 资源、instruction shape 和实现成熟度都会改变结论。

教程解释 Blackwell 版仍落后 cuBLAS 的原因包括 cuBLAS 使用 2-CTA MMA/分布式共享内存获得更大的矩阵指令形状，以及其更成熟的 warp specialization；示例 Gluon 实现也没有双缓冲 accumulator，并留下部分 TMEM 列空间未用。故 persistent 是一种**调度与资源控制工具**，不是自动优化开关。

另外，教程开头关于常规 kernel 调度的文字不要误读成“普通 kernel 的 CTA 必须全部退出后，GPU 才能派发下一个 CTA”。准确边界是：普通 CUDA scheduler 能在资源释放后继续安排同一个 grid 的 CTA；persistent 让 kernel 自己拥有 tile 分派权并跨 tile 保留状态，利于特定的复用/重叠，但承担静态负载均衡风险。

### 易错点与症状

| 误解/错误 | 为什么错 | 可能表现 |
|---|---|---|
| `grid=min(num_sms,num_tiles)` 就保证每个 SM 一定满载 | 它只限制常驻 CTA 数；资源占用、尾部剩余和 tile 数仍决定利用率 | 部分 SM 提前空闲，吞吐不稳定 |
| 用 persistent 就一定减少总开销 | 每个 CTA 要串行承担多个 tile；静态区间可能不均 | 某些 shape 比普通 grid-stride kernel 慢 |
| `GROUP_SIZE_M` 越大越好 | L2 复用与负载均衡存在取舍 | 一个 GPU 变快，另一个变慢 |
| 环形 buffer 只需对 producer 取模 | 复用 mbarrier 时还必须区分 phase；producer/consumer 索引必须对应同一代数据 | 卡死、读到旧 slice 或偶发错值 |
| accumulator 状态跨 tile 保留不重置 | 下一个输出 tile 会错误累加上一个 tile 的结果 | 每个 tile 起点后续输出系统性错误 |
| TMA store 发出后马上覆盖 store source | TMA 仍可能在读这片 SMEM | epilogue/下一 tile 偶发数据损坏 |
| 认为此教程实现了 warp specialization | persistence 和 warp specialization 是不同维度；这份主 GEMM 没有 `gl.warp_specialize` | 错把 CTA 内异步 overlap 解释成专属 warp partition |

### 完整代码与验证边界

完整实现不是只有 `persistent_matmul_pipelined_kernel`：还依赖 `WGMMA`/`MMAv5` aggregate、`issue_loads_stealb`、`issue_mma_stealb`、两种 scheduler、TMA descriptor wrapper、pytest 参数组和 Hopper/Blackwell 专属导入。为避免把省略 import 的片段误当作可运行程序，本章保留完整源文件在仓库内：

- 源码：[code/triton/gluon/tutorials/07-persistence.py](code/triton/gluon/tutorials/07-persistence.py)
- 归档 revision 与硬件范围：[code/triton/gluon/PROVENANCE.md](code/triton/gluon/PROVENANCE.md)
- 在装好匹配版本的 Triton、CUDA 和 Hopper/Blackwell GPU 的环境里运行：

  ```bash
  cd ~/Downloads/learning_notes/operators/code/triton/gluon/tutorials
  python -m pytest -q 07-persistence.py -k persistent_matmul_pipelined
  ```

  预期：48 组参数化测试通过，且每组的 `torch.testing.assert_close(A @ B, C, rtol=1e-3, atol=1e-1)` 成立；48 组来自 2 shapes × 2 tile shapes × 2 buffer counts × 2 warp counts × 3 schedulers。benchmark 则直接执行脚本入口，打印不同 K 下的 TFLOP/s，数值随 GPU、驱动和 Triton revision 变化。

当前机器 `torch.cuda.is_available() == False` 且没有安装 Triton，因此本课只能做文档/源码静态讲解，不能声称执行了 GPU 测试或复现了官方 benchmark。

### 自测（附答案）

1. **普通 kernel 的 scheduler 与 persistent scheduler 最大的不同是什么？**
   答：前者由硬件动态派发 CTA；persistent kernel 把一个较小 CTA 集合常驻起来，由 CTA 内 outer loop 和 scheduler 静态分配多个 tile，可复用状态，但更容易负载不均。
2. **为什么 producer/consumer 除了 buffer index 还需要 phase？**
   答：同一个 mbarrier 槽会循环复用；phase 区分该槽的第几代完成，避免把旧完成状态当成新数据 ready。
3. **在 BM=128、BK=64、BN=256 的例子里，为什么输出能放进两个 B buffer？**
   答：FP16 输出是 128×256=32768 元素；一个 B tile 是 64×256=16384 元素。将 C 沿 N 对半切，每半 128×128=16384 元素，恰好等于一个 B tile 的容量。
4. **`fence_async_shared()` 与 `tma.store_wait()` 分别保证什么？**
   答：fence 排序普通 SMEM 写与异步 proxy 的可见性；store_wait 等待异步 TMA store 不再读取 source buffer，之后才可复用该槽。
5. **哪些情况下 persistent 版本可能更慢？**
   答：静态 tile 分配负载不均、Grouped 顺序降低局部性、tile 数太少、SMEM/TMEM 或寄存器占用压低利用率，以及 epilogue/load overlap 不足以抵消调度成本时。

### 本节进度

- [x] 一个 CTA 处理多个输出 tile 的 persistent grid 与静态 scheduler。
- [x] K-slice 的 TMA producer / MMA consumer 环形缓冲与 mbarrier phase。
- [x] grouped scheduler 的坐标映射、L2 复用动机与负载均衡代价。
- [x] persistent accumulator/barrier 生命周期，以及 `take_result()` 的跨 tile 重置。
- [x] 用额外 B-buffer 承接 epilogue TMA store，并在 buffer 复用前等待 store 完成。
- [x] 性能数据的适用边界：persistent/pipeline 并非对所有 shape 单调变快。

下一知识点：Blackwell Cluster Launch Control 如何在 persistent 静态派工与硬件动态领活之间折中；随后学习 TMA gather/scatter 与 multi-CTA GEMM。
