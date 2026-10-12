# Triton Broadcasting 与 Chained Comparisons

## 本次讲解位置

章节：`chapter_triton_semantics`<br>
小节：Triton Semantics 的 `Broadcasting` 与 `Chained Comparisons`<br>
知识点：shape 左补一、维度为 1 的虚拟扩展，以及链式比较展开为逐对比较的按位 AND<br>
上次：Type Promotion 的 kind/width/signedness、tensor-scalar 与除法/取模规则<br>
下次：Triton 与 NumPy 的差异：整数除法取整、越界 float-to-int cast、变量作用域<br>
PTX：不指定具体 PTX；本节停留在 Triton tensor shape 语义与表达式 lowering 层

## 为什么现在讲这个

上一节已经解决了 dtype 问题：两个操作数即使类型不同，Triton 也能选出一个公共计算 dtype。但表达式还必须解决第二个问题：两个 tensor 的 shape 不同，能否直接运算？如果不能，就必须手写索引、扩展维度或 mask，这会增加大量重复代码。

Broadcasting 的价值是让你用形状更自然的表达方式写 kernel。例如一个 `[M, 1]` 的列向量加一个 `[1, N]` 的行向量，可以直接得到 `[M, N]`，不需要在 host 侧真的创造一个 `M x N` 的输入。它降低的是代码复杂度，也减少了显存中的重复数据。

链式比较解决的是另一类容易误读的问题。`0 <= x <= 8` 看起来像一次三操作数比较，实际语义是：

```text
(0 <= x) & (x <= 8)
```

其中 `x` 只求值一次。如果误以为它在 runtime tensor 上也像 Python 一样会短路，就可能假设后半段在条件不成立时不会执行，进而在后面放不该执行的 load 或其他有副作用的表达式。Triton 的实际规则正好相反：runtime tensor 比较不会短路。

官方规则来源：[Triton Semantics](https://triton-lang.org/main/python-api/triton-semantics.html)。

## 心智模型：broadcast 是无拷贝的虚拟扩展

把 tensor 想象成带 shape 和 stride 的逻辑视图。broadcast 不要求把 `[1, N]` 真正扩展成 `[M, N]` 的 global tensor，而是给缺失或为 1 的维度提供一个“可重复使用”的虚拟步长；概念上可以把它想成 stride 为 0 的视图。

```text
logical shape: [M, N]
virtual source: [1, N] -> 沿行方向重复 M 次
virtual source: [M, 1] -> 沿列方向重复 N 次
```

但这不等于完全零成本。Triton 仍然要把参与运算的值放进 tile、寄存器和指令中。需要区分两点：

```text
不发生 global memory 复制：不会为 [1, N] 额外写一份 [M, N] 数据
不需要任何执行成本：不成立；lowering 仍可能复制寄存器值或使用更复杂的 lane mapping
```

所以 broadcast 首先是一个正确、简洁的 shape 语义，不是无条件免费的性能优化。

## 规则一：shape 维度不足时，左边补 1

如果两个 tensor 的维数不同，维度少的那个在左侧补 1，直到两者维数相同：

```text
(3, 4), (5, 3, 4) -> (1, 3, 4), (5, 3, 4)
```

左侧对齐是广播最容易出错的地方。它不会在右侧补 1：

```text
错误联想： (3, 4) -> (3, 4, 1)
正确规则： (3, 4) -> (1, 3, 4)
```

例如常见矩阵和 bias 的形状：

```text
A: [B, M, K]
bias: [K]
对齐后: [B, M, K] 与 [1, 1, K]
```

这样 bias 沿最后的 K 维对齐，而不是沿最前面的 batch 维对齐。

## 规则二：每一维必须相等，或者其中一维是 1

对齐到相同维数后，从最后一维逐维比较：

```text
相等       -> 保持该维度
其中一维是 1 -> 扩展为另一维的长度
其它情况    -> 不兼容，编译失败
```

示例：

| lhs shape | rhs shape | 对齐过程 | 结果 shape |
|---|---|---|---|
| `[3, 4]` | `[5, 3, 4]` | `[1, 3, 4]` + `[5, 3, 4]` | `[5, 3, 4]` |
| `[128, 1]` | `[1, 256]` | 两个为 1 的维度分别扩展 | `[128, 256]` |
| `[5, 1, 3]` | `[4, 3]` | `[5, 1, 3]` + `[1, 4, 3]` | `[5, 4, 3]` |
| `[2, 3]` | `[4, 5]` | 最后一维 `3 != 5` 且都不为 1 | 编译失败 |

这里的关键不是“shape 完全一样”，而是每一条从右向左对齐后的维度都满足兼容条件。只要有一个维度既不相等，也没有 1，就无法形成统一的输出 shape。

## 完整可运行代码

文件：[`code/triton/broadcasting_and_chained_comparisons.py`](code/triton/broadcasting_and_chained_comparisons.py)

```python
"""Triton broadcasting and chained-comparison examples.

Run:
    python operators/code/triton/broadcasting_and_chained_comparisons.py

Requires a CUDA GPU, PyTorch, and Triton.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _broadcast_add_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
):
    rows = tl.arange(0, M)[:, None]
    cols = tl.arange(0, N)[None, :]

    a = tl.load(a_ptr + rows)  # [M, 1]
    b = tl.load(b_ptr + cols)  # [1, N]
    c = a + b                  # [M, N]

    tl.store(c_ptr + rows * N + cols, c)


@triton.jit
def _chained_mask_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    chained = 0 <= x <= 8
    explicit = (0 <= x) & (x <= 8)
    tl.static_assert(chained.dtype == tl.int1, "chained comparison must produce int1")
    tl.static_assert(explicit.dtype == tl.int1, "explicit comparison must produce int1")

    result = (chained & explicit).to(tl.int32)
    tl.store(out_ptr + offsets, result, mask=mask)


def broadcast_add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if not a.is_cuda or not b.is_cuda:
        raise ValueError("a and b must be CUDA tensors")
    if a.dtype != torch.float32 or b.dtype != torch.float32:
        raise ValueError("a and b must be float32")
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("a and b must be 2-D")
    if a.shape[1] != 1 or b.shape[0] != 1:
        raise ValueError("expected a=[M, 1] and b=[1, N]")

    m, n = a.shape[0], b.shape[1]
    output = torch.empty((m, n), device=a.device, dtype=torch.float32)
    _broadcast_add_kernel[(1,)](a, b, output, M=m, N=n)
    return output


def chained_mask(x: torch.Tensor) -> torch.Tensor:
    if not x.is_cuda:
        raise ValueError("x must be a CUDA tensor")
    if x.dtype != torch.float32 or x.ndim != 1:
        raise ValueError("x must be a 1-D float32 tensor")

    output = torch.empty_like(x, dtype=torch.int32)
    n_elements = x.numel()
    if n_elements == 0:
        return output

    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    _chained_mask_kernel[grid](x, output, n_elements, BLOCK_SIZE=16)
    return output


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("需要 CUDA GPU 才能运行 Triton broadcasting/comparison kernel")

    device = triton.runtime.driver.active.get_active_torch_device()

    a = torch.tensor([[1.0], [2.0], [3.0]], device=device, dtype=torch.float32)
    b = torch.tensor([[10.0, 20.0, 30.0, 40.0]], device=device, dtype=torch.float32)
    c = broadcast_add(a, b)
    c_ref = a + b
    if not torch.equal(c, c_ref):
        raise AssertionError(f"broadcast output mismatch: {c.tolist()} vs {c_ref.tolist()}")

    x = torch.tensor([-1.0, 0.0, 3.0, 8.0, 9.0, float("nan")], device=device)
    mask = chained_mask(x)
    mask_ref = ((0 <= x) & (x <= 8)).to(torch.int32)
    if not torch.equal(mask, mask_ref):
        raise AssertionError(f"chained mask mismatch: {mask.tolist()} vs {mask_ref.tolist()}")

    print(f"a shape={tuple(a.shape)}, b shape={tuple(b.shape)}, c shape={tuple(c.shape)}")
    print(f"broadcast result: {c.tolist()}")
    print(f"x: {x.tolist()}")
    print(f"0 <= x <= 8: {mask.tolist()}")
    print("all_checks_passed: True")


if __name__ == "__main__":
    main()
```

这个脚本同时验证两件事：

1. `[3, 1]` 和 `[1, 4]` 的 broadcast 加法得到 `[3, 4]`。
2. 链式比较 `0 <= x <= 8` 与显式写法 `(0 <= x) & (x <= 8)` 得到相同的 `int1` mask。

## 按执行阶段拆解 broadcast

### 第 1 层：用 `[: , None]` 和 `[None, :]` 建立二维坐标

```python
rows = tl.arange(0, M)[:, None]  # [M, 1]
cols = tl.arange(0, N)[None, :]  # [1, N]
```

`tl.arange(0, M)` 本身是一维 `[M]`。`[:, None]` 在末尾增加一个长度为 1 的维度，得到 `[M, 1]`；`[None, :]` 在开头增加一个长度为 1 的维度，得到 `[1, N]`。

### 第 2 层：load 的 block shape 已经携带广播信息

```python
a = tl.load(a_ptr + rows)  # [M, 1]
b = tl.load(b_ptr + cols)  # [1, N]
```

`a` 的逻辑 shape 是 `[M, 1]`，`b` 是 `[1, N]`。它们来自长度分别为 M 和 N 的连续输入，不需要先用 host 代码复制成二维大矩阵。

### 第 3 层：加法执行广播

```python
c = a + b  # [M, N]
```

对齐规则从右向左看：

```text
a: [M, 1]
b: [1, N]
       ^ 最后一维：1 与 N 兼容，扩展为 N
    ^    倒数第二维：M 与 1 兼容，扩展为 M
结果: [M, N]
```

概念上：

```text
a 沿列方向重复 N 次
b 沿行方向重复 M 次
对应位置相加
```

### 第 4 层：用广播后的指针写回

```python
tl.store(c_ptr + rows * N + cols, c)
```

`rows * N + cols` 的 shape 是 `[M, N]`，`c` 也是 `[M, N]`。因此每个逻辑元素都有独立的 global memory 地址。broadcast 只发生在输入阶段，输出仍然是普通的 M x N tile，不能把结果写回一个只有 M 个元素或 N 个元素的 buffer。

## 具体执行轨迹：`[3, 1] + [1, 4]`

示例输入：

```text
a = [[1.0],
     [2.0],
     [3.0]]       # shape [3, 1]

b = [[10.0, 20.0, 30.0, 40.0]]  # shape [1, 4]
```

广播后的逻辑矩阵是：

```text
a_broadcast:
[[1, 1, 1, 1],
 [2, 2, 2, 2],
 [3, 3, 3, 3]]

b_broadcast:
[[10, 20, 30, 40],
 [10, 20, 30, 40],
 [10, 20, 30, 40]]
```

逐元素相加：

| 行 \ 列 | 0 | 1 | 2 | 3 |
|---:|---:|---:|---:|---:|
| 0 | `1+10=11` | `1+20=21` | `1+30=31` | `1+40=41` |
| 1 | `2+10=12` | `2+20=22` | `2+30=32` | `2+40=42` |
| 2 | `3+10=13` | `3+20=23` | `3+30=33` | `3+40=43` |

输出：

```text
[[11, 21, 31, 41],
 [12, 22, 32, 42],
 [13, 23, 33, 43]]
```

这里没有创建 `a_broadcast` 和 `b_broadcast` 两个真实的 global tensor。上面的矩阵只是帮助理解逻辑 shape 和元素映射。

## Chained Comparisons 的真实语义

### 链式比较是逐对比较后的 elementwise AND

```python
0 <= x <= 8
```

等价于：

```python
(0 <= x) & (x <= 8)
```

每一步都会使用普通的 type promotion 和 broadcasting：

```text
0 <= x       -> scalar 0 与 tensor x 对齐
x <= 8       -> tensor x 与 scalar 8 对齐
两个 int1 结果 & -> 最终 int1 mask
```

更复杂的链也遵循同样规则：

```python
0 <= x < y != 8
```

展开为：

```python
(0 <= x) & (x < y) & (y != 8)
```

三个比较条件必须同时为真。各个比较可以有不同的 shape，只要每一对操作数本身能完成 promotion 和 broadcast，最终结果再按 elementwise 逻辑 AND 合并。

### 每个操作数只求值一次

链式比较不会因为 `x` 出现在两个相邻比较里，就执行两次 `x` 的表达式。概念上：

```text
先求值一次 x
复用同一个 x 结果做 0 <= x
复用同一个 x 结果做 x <= 8
```

这对含有复杂 load 或计算的表达式很重要，因为重复求值可能意味着重复访存或重复计算。不过“只求值一次”不等于“runtime 会短路”。它是两件不同的事。

## 具体执行轨迹：`0 <= x <= 8`

示例输入：

```text
x = [-1, 0, 3, 8, 9, nan]
```

逐项展开：

| 索引 | `x` | `0 <= x` | `x <= 8` | `(0 <= x) & (x <= 8)` |
|---:|---:|---|---|---:|
| 0 | `-1` | false | true | 0 |
| 1 | `0` | true | true | 1 |
| 2 | `3` | true | true | 1 |
| 3 | `8` | true | true | 1 |
| 4 | `9` | true | false | 0 |
| 5 | `nan` | false | false | 0 |

因此最终 mask 是：

```text
[0, 1, 1, 1, 0, 0]
```

`nan` 与任何值比较都得不到 true，所以两个比较结果都是 false，最终逻辑 AND 也是 false。

## 短路规则：编译期和 runtime 要分开看

### 编译期条件为 false 时可以跳过

如果某个比较在编译期已经确定是 false，后续比较和操作数可以不再处理：

```python
if COMPILE_TIME_FALSE:
    # 这段可能不会进入 lowering
    ...
```

这与 Triton 的 `and` 编译期规则一致。它依赖的是编译器已经知道结果，而不是 GPU 运行时某个 lane 的布尔值。

### runtime tensor 比较不会短路

对于 runtime tensor，包括 runtime scalar tensor：

```text
第一个比较为 false
-> 后续比较仍然可能被求值
-> 后续操作数中的 load / 计算也要按表达式语义处理
```

链式比较最终是布尔 tensor 的逐元素运算，不是每个元素各自控制后面的 Python 执行流程。你不能写出“如果这个元素不满足前半段，就不执行后半段的 load”的依赖，除非使用显式 control flow、mask 或提前构造安全指针。

这和 `tl.where` 的一个重要性质一致：`tl.where` 也会求值两个候选分支，只是选择最终写出的值。不要把 `where` 或比较链当作 runtime 短路工具。

## 常见错误和可观察症状

| 错误 | 症状 | 根因 |
|---|---|---|
| 把缺维补在右边 | 得到错误的维度组合或直接报 shape 不兼容 | broadcasting 只在左侧补 1 |
| 期待 `[2, 3]` 和 `[4, 5]` 可以广播 | 编译失败 | 最后一维 `3` 与 `5` 不相等且都不为 1 |
| 把 `[1, N]` load 成 `[N]` 后忘记增维 | 与 `[M, 1]` 无法得到期望的 `[M, N]` | source tensor 的逻辑维度不匹配 |
| 认为 broadcast 一定零指令成本 | 寄存器压力或 layout 成本高于预期 | 不复制 global 数据不等于 lowering 不产生重复 |
| 把链式比较当成三操作数指令 | 不清楚中间产生两个 `int1` mask | 它实际是逐对比较再逐元素 AND |
| 认为 runtime 链式比较会短路 | 后续 load 仍发生，可能出现无效访问或额外带宽 | runtime tensor 比较不做短路控制流 |
| 认为 `nan` 会满足区间条件 | `0 <= nan <= 8` 得到 false | NaN 的所有有序比较均为 false |

## 运行与验证边界

静态语法检查：

```bash
python3 -m py_compile operators/code/triton/broadcasting_and_chained_comparisons.py
```

有 CUDA GPU、PyTorch 和 Triton 时运行：

```bash
python operators/code/triton/broadcasting_and_chained_comparisons.py
```

预期会看到类似：

```text
a shape=(3, 1), b shape=(1, 4), c shape=(3, 4)
broadcast result: [[11.0, 21.0, 31.0, 41.0], [12.0, 22.0, 32.0, 42.0], [13.0, 23.0, 33.0, 43.0]]
x: [-1.0, 0.0, 3.0, 8.0, 9.0, nan]
0 <= x <= 8: [0, 1, 1, 1, 0, 0]
all_checks_passed: True
```

当前开发环境没有 Triton runtime 和 CUDA GPU，因此只完成 Python 静态语法检查，没有声称在本机运行过 Triton kernel。真机运行还需要检查 Triton 版本、目标后端以及编译器对 broadcast layout 的选择。

## 自我检查（含答案）

1. **`[3, 4]` 与 `[5, 3, 4]` 广播后的 shape 是什么？**
   - **答：** 先把短 shape 左补为 `[1, 3, 4]`，再与 `[5, 3, 4]` 比较，得到 `[5, 3, 4]`。
2. **`[128, 1]` 与 `[1, 256]` 广播时，两个长度 1 的维度分别扩成多少？**
   - **答：** 第一维从 1 扩成 128，第二维从 1 扩成 256，结果是 `[128, 256]`。
3. **`0 <= x <= 8` 实际展开成什么表达式？**
   - **答：** `(0 <= x) & (x <= 8)`。先做两个逐元素比较，再对两个 `int1` 结果做 elementwise AND。
4. **runtime tensor 链式比较会像 Python `and` 一样短路吗？**
   - **答：** 不会。编译期确定为 false 的比较可以跳过后续处理，但 runtime tensor 比较不会短路，后续操作数仍可能被求值。
5. **broadcast 的“without copying the data”是否意味着绝对没有额外成本？**
   - **答：** 不意味着。它不会创建同 shape 的 global 副本，但 lowering 仍可能需要寄存器复制、特殊 layout 和其它执行成本。

## 进度

已经覆盖：

- [x] Triton broadcasting：左补一、维度相等或为 1、虚拟扩展与显存/寄存器成本边界。
- [x] Triton chained comparisons：逐对比较、type promotion/broadcasting、操作数只求值一次，以及 runtime 不短路和 NaN 行为。

下一知识点：

- [ ] Triton 与 NumPy 的差异：整数除法/取模的 C 取整、越界 float-to-int cast，以及控制流后的变量定义要求。
