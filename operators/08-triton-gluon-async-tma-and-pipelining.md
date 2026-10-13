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
