# Triton 官方教程 04：Low-Memory Dropout

## 本次讲解位置

章节：`chapter_triton_randomness`（算子开发 / Triton）<br>
小节：Triton 官方教程 04 的 `_seeded_dropout`<br>
知识点：用 `(seed, global offset)` 并行生成随机数，现场构造 dropout mask，而不是保存同 shape 的 mask tensor<br>
上次：Triton blocked GEMM 的 program/tile 映射与 K-loop 累加<br>
下次：把 dropout 扩展到矩阵输入、逐行 seed 和 strided layout<br>
PTX：不指定具体 PTX；本节停留在 Triton JIT、`tl.rand`、Philox PRNG 与 elementwise operator 层

## 为什么现在讲这个

前面的 GEMM 是确定性计算：同样的 A/B 一定得到同样的 C。Dropout 多了一个关键状态：**每个元素是否保留**。传统实现把这个决定保存成一个和输入同 shape 的 mask。训练使用 recompute/checkpoint 时，mask 还需要跟着 RNG state 一起保存和恢复；在显存和反复读写上都很重。

本节的 low-memory 不是指随机数完全不占空间，而是指：

```text
传统状态：一个和输入同 shape 的 mask
seeded 状态：一个 int32 seed
```

元素是否 dropout 不再从全局 mask 读取，而是用元素的全局 offset 现场计算。这样做换来了三个结果：

1. 持久状态从 `O(N)` 降到 `O(1)`。
2. 少一次 mask 的 global memory read。
3. 用 Philox 随机数计算替代这次访存，因此是明确的 memory/compute trade-off。

它同时引入一个新的正确性条件：同一个 seed 和同一个 element offset 必须得到同一个随机值；不同 block 不能各自从 `0..BLOCK_SIZE-1` 重新生成，否则每块都会重复同样的 dropout 图案。

来源：[Triton Low-Memory Dropout Tutorial](https://triton-lang.org/main/getting-started/tutorials/04-low-memory-dropout.html)。

## 先建立心智模型

### Dropout 在做什么

设 dropout 概率为 `p`，keep 概率就是：

```text
keep_probability = 1 - p
```

对每个元素 `x[i]`：

```text
以概率 p 变成 0
以概率 1-p 保留 x[i] / (1-p)
```

为什么保留时要除以 `1-p`？因为保留事件 `K` 是 Bernoulli 变量：

```text
P(K=1) = 1-p
P(K=0) = p

output = K * x / (1-p)

E[output] = (1-p) * x / (1-p) + p * 0
          = x
```

所以 scale `1/(1-p)` 是为了保持输出的期望不变。这个技巧叫 inverted dropout，训练时完成缩放，推理时直接使用完整输入。

### baseline 和 seeded 的状态区别

```text
baseline:
state = x_keep[N]

per element:
load x[i]
load x_keep[i]
output[i] = x_keep[i] ? x[i] / (1-p) : 0
```

```text
seeded:
state = seed

per element:
load x[i]
r[i] = rand(seed, i)
keep[i] = r[i] > p
output[i] = keep[i] ? x[i] / (1-p) : 0
```

seeded 版没有全局 mask。`keep[i]` 是临时值，由当前 program 现场计算，计算完即可丢弃。

### 并行 PRNG 的关键：随机数不是顺序“下一个”

CPU 上常见 RNG 模型是维护一个全局 state：

```text
r0 = next(global_state)
r1 = next(global_state)
r2 = next(global_state)
```

这种模型很难直接并行，因为 `r1` 依赖 `r0` 已消耗 state。

Triton 的 `tl.rand(seed, offsets)` 使用 Philox 风格的可并行思路：每个元素带着自己的全局 offset，直接计算：

```text
r[i] = PRNG(seed, global_offset=i)
```

因此：

```text
相同 (seed, offset) -> 相同随机值
不同 offset          -> 可并行独立计算
不同 seed            -> 一般得到不同随机序列
```

这里不存在“其它线程先取走了几个随机数”的问题。随机值绑定在元素坐标上，而不是绑定在某个线程的执行顺序上。

## 完整可运行代码

文件：[`code/triton/seeded_dropout.py`](code/triton/seeded_dropout.py)

```python
"""Triton 官方教程 04 的 low-memory dropout 可运行版本。

核心区别：

- baseline dropout：持久保存一个和输入同 shape 的 keep mask；
- seeded dropout：只保存一个 seed，每个元素按 (seed, global_offset) 现场生成随机数。

运行：
    python operators/code/triton/seeded_dropout.py

需要 CUDA GPU、PyTorch 和 Triton。
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


DEVICE = triton.runtime.driver.active.get_active_torch_device()


@triton.jit
def _dropout(
    x_ptr,
    x_keep_ptr,
    output_ptr,
    n_elements,
    p,
    BLOCK_SIZE: tl.constexpr,
):
    """读取预先生成的 keep mask，并完成 inverted dropout。"""
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    x_keep = tl.load(x_keep_ptr + offsets, mask=mask, other=0).to(tl.int1)
    output = tl.where(x_keep, x / (1 - p), 0.0)
    tl.store(output_ptr + offsets, output, mask=mask)


@triton.jit
def _seeded_dropout(
    x_ptr,
    output_ptr,
    n_elements,
    p,
    seed,
    BLOCK_SIZE: tl.constexpr,
):
    """不读取 mask，按 (seed, global offset) 现场生成随机数。"""
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    # random[i] 是 [0, 1) 上的均匀分布；相同 seed + offset 会得到相同值。
    random = tl.rand(seed, offsets)
    x_keep = random > p

    # inverted dropout：保留时放大 1 / (1 - p)，保持输出的期望不变。
    output = tl.where(x_keep, x / (1 - p), 0.0)
    tl.store(output_ptr + offsets, output, mask=mask)


def dropout_baseline(x: torch.Tensor, x_keep: torch.Tensor, p: float) -> torch.Tensor:
    """使用显式 keep mask 的 baseline。"""
    if not x.is_cuda or not x_keep.is_cuda:
        raise ValueError("x and x_keep must be CUDA tensors")
    if not x.is_contiguous() or not x_keep.is_contiguous():
        raise ValueError("x and x_keep must be contiguous")
    if x_keep.shape != x.shape:
        raise ValueError("x_keep must have the same shape as x")
    if not 0.0 <= p < 1.0:
        raise ValueError("p must satisfy 0 <= p < 1")
    if x.numel() == 0:
        return torch.empty_like(x)

    output = torch.empty_like(x)
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    _dropout[grid](
        x,
        x_keep,
        output,
        n_elements,
        p,
        BLOCK_SIZE=1024,
    )
    return output


def seeded_dropout(x: torch.Tensor, p: float, seed: int) -> torch.Tensor:
    """只保存 seed 的 dropout；同一 seed 会重放同一 mask。"""
    if not x.is_cuda:
        raise ValueError("x must be a CUDA tensor")
    if not x.is_contiguous():
        raise ValueError("x must be contiguous")
    if not 0.0 <= p < 1.0:
        raise ValueError("p must satisfy 0 <= p < 1")
    if p == 0.0:
        return x.clone()
    if x.numel() == 0:
        return torch.empty_like(x)

    output = torch.empty_like(x)
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    _seeded_dropout[grid](
        x,
        output,
        n_elements,
        p,
        seed,
        BLOCK_SIZE=1024,
    )
    return output


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("需要 CUDA GPU 才能运行 Triton dropout kernel")

    torch.manual_seed(0)
    x = torch.tensor(
        [1.48333, -0.239537, -0.640795, 1.62631, 0.263036,
         -0.71516, 1.99474, -1.09546, 1.81107, -0.170083],
        device=DEVICE,
        dtype=torch.float32,
    )
    p = 0.5

    baseline_keep = torch.rand((10,), device=DEVICE) > p
    baseline = dropout_baseline(x, baseline_keep, p)

    output_123 = seeded_dropout(x, p=p, seed=123)
    output_123_repeat = seeded_dropout(x, p=p, seed=123)
    output_512 = seeded_dropout(x, p=p, seed=512)

    same_seed_equal = torch.equal(output_123, output_123_repeat)
    different_seed_equal = torch.equal(output_123, output_512)
    if not same_seed_equal:
        raise AssertionError("same seed must reproduce the same dropout mask")
    if different_seed_equal:
        raise AssertionError("different seeds should normally produce different masks")

    kept = output_123 != 0
    dropped = ~kept
    scale = 1.0 / (1.0 - p)
    if not torch.allclose(output_123[kept], x[kept] * scale):
        raise AssertionError("kept elements must be scaled by 1 / (1 - p)")
    if not torch.all(output_123[dropped] == 0):
        raise AssertionError("dropped elements must become zero")

    print(f"input:              {x.tolist()}")
    print(f"baseline keep:      {baseline_keep.to(torch.int32).tolist()}")
    print(f"baseline output:    {baseline.tolist()}")
    print(f"seed=123 output:    {output_123.tolist()}")
    print(f"seed=123 repeat:    {output_123_repeat.tolist()}")
    print(f"seed=512 output:    {output_512.tolist()}")
    print(f"same_seed_equal:    {same_seed_equal}")
    print(f"different_seed_equal: {different_seed_equal}")


if __name__ == "__main__":
    main()
```

## 第 1 层：grid 和 global offset

kernel 仍然是标准的一维 elementwise grid：

```python
pid = tl.program_id(axis=0)
block_start = pid * BLOCK_SIZE
offsets = block_start + tl.arange(0, BLOCK_SIZE)
```

`tl.arange(0, BLOCK_SIZE)` 产生 program 内的局部位置：

```text
[0, 1, 2, ..., BLOCK_SIZE-1]
```

再加上 `block_start` 后，得到整个 tensor 上的全局位置。例如 `BLOCK_SIZE=4`：

| `pid` | 局部 `tl.arange` | `offsets` |
|---:|---|---|
| 0 | `[0, 1, 2, 3]` | `[0, 1, 2, 3]` |
| 1 | `[0, 1, 2, 3]` | `[4, 5, 6, 7]` |
| 2 | `[0, 1, 2, 3]` | `[8, 9, 10, 11]` |

`tl.rand` 必须接收全局 `offsets`。如果错误地写成：

```python
offset = tl.arange(0, BLOCK_SIZE)
random = tl.rand(seed, offset)
```

那么每个 program 都会生成同样的一组随机数，dropout 图案会按 block 周期重复。这种错误通常不会让结果变成非法数值，但会让随机性严重退化，统计分布不再接近独立 dropout。

`offsets` 同时用于数据 load 和 PRNG。也就是说，元素 `x[i]` 的随机值由 `(seed, i)` 决定，元素地址和随机 key 保持一致。

## 第 2 层：现场生成 random 和 keep

```python
random = tl.rand(seed, offsets)
x_keep = random > p
```

`tl.rand` 返回 `[0, 1)` 上的 uniform float32。对每个元素：

```text
P(random <= p) = p
P(random > p)  = 1-p
```

所以 `random > p` 是一个 keep 概率为 `1-p` 的布尔 mask。和 baseline 不同的是，这个布尔 mask 只在寄存器和临时 tile 中存在，不会写成大小为 `N` 的 global tensor。

### 固定随机值的手工例子

为了让计算关系更直观，先假设随机数已经给定：

```text
x      = [2, 4, 6, 8]
p      = 0.25
random = [0.8, 0.1, 0.7, 0.2]
```

判定 `random > p`：

| 索引 | `x[i]` | `random[i]` | `random[i] > 0.25` | `output[i]` |
|---:|---:|---:|---|---:|
| 0 | 2 | 0.8 | true | `2 / 0.75 = 2.6667` |
| 1 | 4 | 0.1 | false | `0` |
| 2 | 6 | 0.7 | true | `6 / 0.75 = 8.0` |
| 3 | 8 | 0.2 | false | `0` |

最终输出：

```text
[2.6667, 0, 8.0, 0]
```

这里的 `random` 只是为了演示；真实值由 Triton 的 `tl.rand(seed, offsets)` 产生。

## 第 3 层：inverted dropout 缩放

```python
output = tl.where(x_keep, x / (1 - p), 0.0)
```

这行同时完成两个操作：

```text
keep 为 true  -> output = x / (1-p)
keep 为 false -> output = 0
```

如果忘记除以 `1-p`，训练时输出期望会变成：

```text
E[output] = (1-p) * x
```

每一层都缩水后，网络激活值的尺度会随 dropout 概率改变。除以 `1-p` 后：

```text
E[output] = (1-p) * x / (1-p) = x
```

这就是 inverted dropout 保持期望的核心。

`p == 0.0` 时，脚本直接在 host 侧返回 `x.clone()`，不启动 kernel。`p == 1.0` 会让 `1-p` 变成 0，并且所有元素都必须被删除，没有可缩放的有效输出，因此脚本要求 `0 <= p < 1`。

## 具体执行轨迹：官方示例

取官方页面中的输入：

```text
input = [
    1.48333, -0.239537, -0.640795, 1.62631, 0.263036,
    -0.71516, 1.99474, -1.09546, 1.81107, -0.170083
]
p = 0.5
seed = 123
```

页面中的 seeded 输出是：

```text
output(seed=123) = [
    0, -0.479074, 0, 0, 0,
    -1.43032, 0, 0, 3.62215, -0.340165
]
```

逐项解释：

| 索引 | 输入 | 输出 | 发生了什么 |
|---:|---:|---:|---|
| 0 | `1.48333` | `0` | drop |
| 1 | `-0.239537` | `-0.479074` | keep，乘 2 |
| 2 | `-0.640795` | `0` | drop |
| 3 | `1.62631` | `0` | drop |
| 4 | `0.263036` | `0` | drop |
| 5 | `-0.71516` | `-1.43032` | keep，乘 2 |
| 6 | `1.99474` | `0` | drop |
| 7 | `-1.09546` | `0` | drop |
| 8 | `1.81107` | `3.62215` | keep，乘 2 |
| 9 | `-0.170083` | `-0.340165` | keep，乘 2 |

再运行一次同样的 `seed=123`，输出逐元素完全相同。把 seed 改成 `512`，keep/drop 图案改变，说明不同 PRNG key 产生了不同 mask。

注意：seed 相同不代表“随机数固定等于某个常量”，而是代表随机序列可以重放。相同的 `(seed, offset)` 组合在同一个 Triton 实现和目标上会得到相同的 PRNG 输出。

## 第 4 层：显存和访存到底省在哪里

设元素个数是 `N`，输入和输出都是 float32。baseline 使用 bool mask 时，粗略内存足迹是：

```text
baseline 持久状态：N bytes 的 bool mask
baseline 每次执行：
  read  x        : 4N bytes
  read  mask     : N bytes
  write output   : 4N bytes
```

seeded 版本：

```text
seeded 持久状态：4 或 8 bytes 的 seed
seeded 每次执行：
  read  x        : 4N bytes
  generate rand  : 计算，不读全局 mask
  write output   : 4N bytes
```

于是节省的是：

```text
O(N) mask 状态 -> O(1) seed 状态
少一次 N bytes 的 global mask read
```

但这不是零成本：

- `tl.rand` 要为每个元素执行 Philox 随机数计算。
- `random` tile 仍然占用寄存器。
- 如果 `BLOCK_SIZE` 很大，临时 random 的 live range 会影响寄存器和 occupancy。
- 需要显式管理 seed，保证 backward/recompute 能恢复同一个 mask。

所以它把一部分 HBM 带宽和状态管理成本，换成了 GPU 计算与 seed 生命周期管理成本。具体是否更快，取决于 `N`、mask dtype、访存带宽、随机数计算吞吐和 kernel 是否已被带宽 bound。

## 第 5 层：为什么这对 checkpoint / recompute 有意义

普通 dropout 的 backward 需要知道哪些元素被保留。传统做法要么保存 mask，要么保存 RNG state。seeded dropout 使状态变成一个 seed：

```text
forward:
  seed -> generate mask -> output

recompute:
  same seed -> generate same mask -> backward
```

这让“随机状态”更容易跟着每个算子或每个层保存。实际训练框架还要解决 seed 的分配和记录，例如不同 step、不同 layer、不同 invocation 不能意外共用同一个 seed；否则它们会得到完全相同的 dropout 图案。

另外，seed 绑定的是 element offset。如果 kernel 改成支持 strided layout，需要明确随机 key 使用逻辑坐标还是物理线性 offset。这个选择会影响相同逻辑 tensor 在不同 layout 下是否得到相同 mask。

## 常见错误和症状

| 错误 | 症状 | 根因 |
|---|---|---|
| 每个 program 都从 `0..BLOCK_SIZE-1` 生成 random | dropout 图案按 block 重复，随机性明显退化 | 没有使用全局 `offsets` 作为 PRNG key |
| 忘记 `1/(1-p)` 缩放 | 激活尺度越来越小，训练统计和预期不一致 | inverted dropout 的期望补偿缺失 |
| 用 `1/p` 而不是 `1/(1-p)` | 保留元素的幅度完全错误 | 把 drop 概率当成 keep 概率 |
| 同一 seed 被所有层复用 | 不同层的 dropout 图案相同 | seed 生命周期设计错误 |
| 允许 `p=1` | 除以 0，输出无法定义 | 没有合法 keep 概率 |
| 输入非 contiguous 但仍按线性 offset 生成 | mask 与逻辑元素错位 | PRNG key 和真实地址布局不一致 |
| 把 random 当成全局顺序状态 | 并发执行时结果不可复现 | Triton 的 `tl.rand` 是 offset-based PRNG，不是单线程 RNG |
| 假设跨所有 Triton 版本和硬件逐 bit 相同 | 升级后 mask 可能变化 | 可复现边界首先是在某个实现与 target 内成立 |

## 运行与验证边界

静态语法检查：

```bash
python3 -m py_compile operators/code/triton/seeded_dropout.py
```

有 CUDA GPU、PyTorch 和 Triton 时，运行：

```bash
python operators/code/triton/seeded_dropout.py
```

脚本会检查：

```text
same_seed_equal:      True
different_seed_equal: False
kept elements are exactly x / (1-p)
dropped elements are exactly 0
```

由于本机没有 CUDA GPU 和 Triton runtime，这里只完成静态语法检查，不声称已经在本机运行过 kernel。真机输出会随输入、seed、GPU 和 Triton 版本变化，但同一实现内的重复 seed 语义和缩放关系仍应成立。

## 自我检查（含答案）

1. **baseline 和 seeded dropout 的持久状态分别是什么？**
   - **答：** baseline 需要一个和输入同 shape 的 keep mask；seeded 只需要一个 seed，mask 在 kernel 内按 `(seed, offset)` 临时计算。
2. **为什么 `offsets` 必须包含 `pid * BLOCK_SIZE`？**
   - **答：** `tl.rand` 用 offset 作为元素级随机 key。如果每个 program 都从 0 开始，所有 block 会得到相同的随机图案。
3. **`output = x / (1-p)` 为什么不改变训练时的期望？**
   - **答：** 元素以 `1-p` 的概率被保留，所以 `E[output] = (1-p) * x / (1-p) = x`。
4. **相同 seed 的两次 seeded dropout 会发生什么？**
   - **答：** 在同一个 Triton 实现和目标上，相同 offset 得到相同随机数，因此会重放同一个 keep/drop mask。
5. **seeded dropout 是否一定比 baseline 快？**
   - **答：** 不一定。它省掉全局 mask read 和 `O(N)` 状态，但增加了随机数计算和寄存器压力；是否更快取决于 workload 和硬件瓶颈。

## 进度

已经覆盖：

- [x] Triton 官方 low-memory dropout：baseline mask、seeded PRNG、global offset、inverted dropout 缩放、`tl.rand` 与状态/访存权衡。

下一知识点：

- [ ] Dropout 扩展练习：矩阵输入、逐行 seed 与 strided layout 下 PRNG key 的选择。
