# Triton Semantics：Type Promotion（类型提升）

## 本次讲解位置

章节：`chapter_triton_semantics`<br>
小节：Triton Semantics 的 `Type Promotion`<br>
知识点：tensor-tensor、tensor-scalar 的公共 dtype 推导，以及除法/取模例外<br>
上次：Triton 官方教程 04 的 low-memory dropout 与 seeded PRNG<br>
下次：Broadcasting 与 chained comparisons<br>
PTX：不指定具体 PTX；本节停留在 Triton 编译期 dtype legalization 与转换层

## 为什么现在讲这个

前面的 dropout kernel 里有这样一行：

```python
output = tl.where(x_keep, x / (1 - p), 0.0)
```

如果 `x` 是 FP16，而 `1 - p` 是 Python float，Triton 到底是先算 FP16 除法，还是先转换成 FP32？这个选择会直接改变精度、溢出边界和性能。Type promotion 回答的就是“两个不同 dtype 的操作数，在执行运算前会被转换成哪一种公共 dtype”。

如果只凭语言直觉写 kernel，最容易出现两类错误。第一类是**精度被意外丢掉**：本来希望 `fp16` 输入在 `fp32` 中累加，但某个标量把中间结果拉回了低精度。第二类是**结果 dtype 和显存 dtype 不一致**：运算返回 `fp32`，写进 `fp16` 输出时又发生一次隐式转换；如果只看输出 buffer，会误以为整个计算都在 FP16 中进行。

所以本节的边界是：

```text
Type promotion 决定“用哪个 dtype 做运算”
Broadcasting 决定“哪些 shape 可以对齐”
Store / load 决定“结果如何进出显存”
```

这三件事经常一起出现，但不是同一个问题。

官方规则来源：[Triton Semantics: Type Promotion](https://triton-lang.org/main/python-api/triton-semantics.html#type-promotion)。

## 心智模型：先把“计算 dtype”和“存储 dtype”分开

考虑：

```python
a = tl.load(a_ptr + offsets)  # fp16
b = tl.load(b_ptr + offsets)  # fp32
c = a + b
tl.store(c_ptr + offsets, c)  # c_ptr 指向 fp16 buffer
```

不要把它理解成一次没有中间类型的操作。更准确的执行模型是：

```text
load a              -> a: fp16
load b              -> b: fp32
promote a to fp32   -> a_fp32: fp32
compute a_fp32 + b  -> c: fp32
store c to fp16 ptr -> 写回时再 cast 成 fp16
```

因此要看两个不同的问题：

| 问题 | 由什么决定 | 本例结果 |
|---|---|---|
| `a + b` 的运算 dtype 是什么？ | type promotion | `fp32` |
| 写进 `c_ptr` 后存储 dtype 是什么？ | store 的目标元素类型 | `fp16` |

这个区分对 attention、softmax、layer norm 等 kernel 很重要。输入可能是 FP16/BF16，但 accumulation、division、reshape 前的中间值是否在 FP32，要由 promote 规则和显式 cast 共同决定。

## 规则一：kind 决定第一条分界线

Triton 把 dtype 先按 kind 分层：

```text
bool < integral < floating
```

这里的 `integral` 包括 `int8`、`uint8`、`int16`、`uint16`、`int32`、`uint32`、`int64`、`uint64` 等；`floating` 包括 `float16`、`bfloat16`、`float32`、`float64` 和各种 FP8。

当一个 tensor 的 kind 更高时，另一个 tensor 先提升到高 kind：

```text
int32 + bfloat16 -> bfloat16
```

原因是浮点 kind 高于整数 kind，所以整数操作数先被转换成浮点，再参与加法。这里得到的是公共操作 dtype，不等于最终一定以这个 dtype 写回显存。

## 规则二：同 kind 时比较宽度

如果两个 dtype 属于同一个 kind，宽度更高的 dtype 胜出：

```text
float16 + float32 -> float32
int16 + int32     -> int32
```

这是最常见也最符合直觉的一类 promotion。它解释了为什么混合使用低精度输入和 FP32 accumulator 时，中间加法通常会留在 FP32。

## 规则三：同宽度、不同浮点类型时偏向 FP16

对于同宽度且 signedness 相同的不同浮点类型，Triton 的文档规则是优先选择 `float16`：

```text
float16 + bfloat16 -> float16
float8_e4m3 + float8_e5m2 -> float16
```

不要把这个例子理解成“所有浮点类型都按数值范围选择最宽的类型”。这里的“同宽度”是规则的一部分，而不同浮点格式具有不同的指数位和尾数位，不能只靠一个位宽数字判断数值精度。

### 规则四：同宽度、不同 signedness 时偏向 unsigned

如果两个整数 dtype 位宽相同但 signedness 不同：

```text
int32 + uint32 -> uint32
```

这来自整数提升里的 unsigned 优先规则。它不是“所有整数都转 int64”，也不是“保留左边操作数的类型”。操作数顺序一般不改变公共 dtype 的规则，但显式写出两边 dtype 仍然比依赖隐式推导更安全。

### tensor-tensor promotion 汇总

| 输入组合 | 公共 dtype | 主要规则 |
|---|---|---|
| `int32, bfloat16` | `bfloat16` | floating kind 更高 |
| `float16, float32` | `float32` | 同 kind，宽度更高者胜 |
| `int16, int32` | `int32` | 同 kind，宽度更高者胜 |
| `float16, bfloat16` | `float16` | 同宽度、不同浮点类型时偏向 FP16 |
| `int32, uint32` | `uint32` | 同宽度、不同 signedness 时偏向 unsigned |

## 规则五：scalar 不按 tensor 的方式参与 promotion

这里的 scalar 不是“一个 shape 为 `[]` 的 tensor”，而是：

```text
数值字面量，例如 1、1.0、4.0
tl.constexpr 包装的数值
这些值参与组成的常量表达式
```

当操作数一边是 tensor、另一边是 scalar 时，规则变成两步。

### 情况 A：scalar 的 kind 小于或等于 tensor

scalar 不参与 promotion，结果继续使用 tensor 的 dtype：

```text
uint8_tensor + 3   -> uint8
float16_tensor + 1 -> float16
float16_tensor + 1.0 -> float16
```

这解释了为什么 `fp16 tensor + 1.0` 不一定自动得到 FP32。`1.0` 在 Python 里是 float，但在这个规则里它仍是 scalar；scalar 的 floating kind 与 tensor 的 floating kind 相同，所以它不把 tensor 拉宽。

如果希望强制 FP32，不要猜 scalar 会不会自动帮助，直接写：

```python
x = x.to(tl.float32)
result = x + 1.0
```

### 情况 B：scalar 的 kind 高于 tensor

scalar 先选择一个“能装下它的最窄 dtype”，然后 tensor 和 scalar 一起提升到该 dtype。对于整数 tensor 加浮点 scalar，最常见的例子是：

```text
int16_tensor + 4.0 -> float32
```

过程可以拆成：

```text
1. tensor kind = integral，scalar kind = floating
2. floating 更高，所以 scalar 参与 promotion
3. Python 的 4.0 能装进 float32，因此选择 float32
4. int16 tensor 和 4.0 都提升到 float32
```

如果把 scalar 换成超出 FP32 表示范围的大浮点数，则可能选择 `float64`。但对于普通整数 scalar，Triton 不会因为“字面量看起来更宽”就无条件把所有 int tensor 变成 int64；这正是最容易和直觉不一致的地方。当前实现还会做 representability 检查，例如把负数加到 unsigned tensor，或者把超出目标范围的值转成窄整数，可能直接报错，而不是静默 wrap。生产 kernel 中遇到这种边界，应显式 cast 并写清语义。

### tensor-scalar promotion 汇总

| 表达式 | 结果 dtype | 认知重点 |
|---|---|---|
| `uint8_tensor + 3` | `uint8` | scalar kind 低于/等于 tensor，不参与 promotion |
| `float16_tensor + 1` | `float16` | 整数 scalar 不会自动变成 FP32 |
| `float16_tensor + 1.0` | `float16` | 同 floating kind 的 scalar 也不自动拉宽 |
| `int16_tensor + 4.0` | `float32` | 浮点 scalar kind 更高，选择能装下它的最窄浮点 dtype |
| `uint8_tensor + (-1)` | 可能报错 | 负数不能安全表示在 unsigned 目标类型中 |

## 除法与取模：不要套用普通加法规则

除法 `/` 和取模 `%` 有一个重要例外：对于宽度小于 `float32` 的浮点类型，硬件或 Triton 语义可能没有原生的窄浮点除法/取模，因此只要任一操作数是浮点，通常会把两边提升到 `float32`：

```text
float16 / float16 -> float32
bfloat16 % bfloat16 -> float32
```

这并不表示结果一定以 `float32` 写回显存。如果输出指针指向 FP16 buffer，store 仍会把它转回 FP16。它只表示运算和中间的求值发生在 FP32。

整数除法和取模不走这条浮点例外，而是继续使用整数 promotion。混合 signedness 的整数 `/`、`//` 或 `%` 在 Triton 中可能被拒绝，因为结果符号和取整方向容易产生不直观的行为。对于这类表达式，显式把两边转成同一种 signed integer dtype，通常比依赖隐式规则更清楚。

浮点 `float64` 以及不同 Triton 版本、不同目标后端上的混合浮点边界，可能比文档中的基础规则更细。对精度或可移植性有硬要求的 kernel，应把关键操作数显式转换到目标计算 dtype，并在测试中检查结果 dtype。

## 类型提升不等于 shape broadcast

这两个规则经常在同一条表达式里发生，但负责的事情不同：

```text
type promotion: 选择公共 dtype
broadcasting:   选择公共 shape
```

例如：

```text
A: float16, shape [128, 1]
B: float32, shape [1, 256]
C: A + B
```

先做 dtype promotion：

```text
float16 + float32 -> float32
```

再做 shape broadcast：

```text
[128, 1] + [1, 256] -> [128, 256]
```

最终：

```text
C: float32, shape [128, 256]
```

如果只看到 `A` 是 FP16，会误以为整个结果仍在 FP16 中；如果只看到 `B` 是 `[1, 256]`，会误以为输出还是 `[1, 256]`。两个独立规则必须同时看。

## 完整可运行代码

文件：[`code/triton/type_promotion.py`](code/triton/type_promotion.py)

```python
"""Triton type-promotion examples with static and runtime checks.

Run:
    python operators/code/triton/type_promotion.py

Requires a CUDA GPU, PyTorch, and Triton.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import triton
import triton.language as tl


@dataclass(frozen=True)
class PromotionCase:
    case_id: int
    name: str
    torch_result_dtype: torch.dtype


CASES = (
    PromotionCase(0, "fp16 + fp32 -> fp32", torch.float32),
    PromotionCase(1, "fp16 + bf16 -> fp16", torch.float16),
    PromotionCase(2, "int16 + 4.0 -> fp32", torch.float32),
    PromotionCase(3, "fp16 + 1 -> fp16", torch.float16),
    PromotionCase(4, "fp16 + 1.0 -> fp16", torch.float16),
    PromotionCase(5, "fp16 / fp16 -> fp32", torch.float32),
    PromotionCase(6, "tl.where(cond, fp16, fp32) -> fp32", torch.float32),
)


@triton.jit
def _promotion_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    CASE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    if CASE == 0:
        # Higher kind wins: floating keeps the result out of the integer path.
        x = tl.load(x_ptr + offsets, mask=mask, other=0).to(tl.float16)
        y = tl.load(y_ptr + offsets, mask=mask, other=0).to(tl.float32)
        result = x + y
        tl.static_assert(result.dtype == tl.float32, "fp16 + fp32 must be fp32")
    elif CASE == 1:
        # Same kind and width; the documented pair fp16/bf16 prefers fp16.
        x = tl.load(x_ptr + offsets, mask=mask, other=0).to(tl.float16)
        y = tl.load(y_ptr + offsets, mask=mask, other=0).to(tl.bfloat16)
        result = x + y
        tl.static_assert(result.dtype == tl.float16, "fp16 + bf16 must be fp16")
    elif CASE == 2:
        # A float scalar has a higher kind than an integer tensor, so choose
        # the narrowest fitting scalar dtype: fp32.
        x = tl.load(x_ptr + offsets, mask=mask, other=0).to(tl.int16)
        result = x + 4.0
        tl.static_assert(result.dtype == tl.float32, "int16 + 4.0 must be fp32")
    elif CASE == 3:
        # An integer scalar does not promote a floating tensor.
        x = tl.load(x_ptr + offsets, mask=mask, other=0).to(tl.float16)
        result = x + 1
        tl.static_assert(result.dtype == tl.float16, "fp16 + 1 must be fp16")
    elif CASE == 4:
        # A scalar of the same kind also does not promote a tensor.
        x = tl.load(x_ptr + offsets, mask=mask, other=0).to(tl.float16)
        result = x + 1.0
        tl.static_assert(result.dtype == tl.float16, "fp16 + 1.0 must be fp16")
    elif CASE == 5:
        # Division is an exception: narrow floating division computes in fp32.
        x = tl.load(x_ptr + offsets, mask=mask, other=0).to(tl.float16)
        y = tl.load(y_ptr + offsets, mask=mask, other=1).to(tl.float16)
        result = x / y
        tl.static_assert(result.dtype == tl.float32, "fp16 / fp16 must be fp32")
    elif CASE == 6:
        # The last two arguments of tl.where use the same promotion rules.
        condition = offsets < BLOCK_SIZE // 2
        x = tl.load(x_ptr + offsets, mask=mask, other=0).to(tl.float16)
        y = tl.load(y_ptr + offsets, mask=mask, other=0).to(tl.float32)
        result = tl.where(condition, x, y)
        tl.static_assert(result.dtype == tl.float32, "tl.where fp16/fp32 must be fp32")
    else:
        tl.static_assert(False, "unknown CASE")

    tl.store(out_ptr + offsets, result, mask=mask)


def _reference(case: PromotionCase, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x32 = x.to(torch.float32)
    y32 = y.to(torch.float32)

    if case.case_id == 0:
        result = x32 + y32
    elif case.case_id == 1:
        result = x32 + y32
    elif case.case_id == 2:
        result = x32 + 4.0
    elif case.case_id == 3:
        result = x32 + 1.0
    elif case.case_id == 4:
        result = x32 + 1.0
    elif case.case_id == 5:
        result = x32 / y32
    elif case.case_id == 6:
        condition = torch.arange(x.numel(), device=x.device) < x.numel() // 2
        result = torch.where(condition, x32, y32)
    else:
        raise ValueError(f"unknown case id: {case.case_id}")

    return result.to(case.torch_result_dtype)


def _run_case(case: PromotionCase, device: torch.device) -> None:
    x = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8], device=device, dtype=torch.float32)
    y = torch.tensor([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], device=device, dtype=torch.float32)
    output = torch.empty_like(x, dtype=case.torch_result_dtype)
    n_elements = x.numel()

    _promotion_kernel[(1,)](
        x,
        y,
        output,
        n_elements,
        CASE=case.case_id,
        BLOCK_SIZE=8,
    )

    reference = _reference(case, x, y)
    if output.dtype != case.torch_result_dtype:
        raise AssertionError(f"{case.name}: dtype={output.dtype}, expected={case.torch_result_dtype}")
    if not torch.equal(output, reference):
        raise AssertionError(f"{case.name}: output={output.tolist()}, reference={reference.tolist()}")

    print(f"{case.name}: dtype={output.dtype}, output={output.tolist()}")


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("需要 CUDA GPU 才能运行 Triton type-promotion kernel")

    device = triton.runtime.driver.active.get_active_torch_device()
    for case in CASES:
        _run_case(case, device)
    print("all_checks_passed: True")


if __name__ == "__main__":
    main()
```

这个脚本包含三层验证：

1. kernel 内的 `tl.static_assert` 检查编译器推导出的 `result.dtype`。
2. host 侧检查输出 buffer 的 `torch.dtype`。
3. host 侧逐元素对比 Torch 参考值。

这里的 `tl.static_assert` 在 Triton 编译期执行，不是 GPU 运行到某一 lane 时才判断的 runtime assert。它检查的是 Triton 对 `result` 的类型结果，适合用来固定本节要讲的 promotion 规则。

## 按执行阶段拆解代码

### 第 1 层：每个 case 是一个编译期分支

`CASE: tl.constexpr` 让 `if CASE == 0` 等分支在 Triton 追踪和编译阶段就变成静态选择。每次 launch 只保留一个 case 的运算路径，不会真的让 GPU 在运行时逐个比较七个 case。

这样做的好处是每个分支都可以用静态断言验证一种 promotion 规则，而且输入 shape 保持简单，避免把 broadcasting 和 dtype 推导混在一起。

### 第 2 层：load 只是把值带进来，cast 才决定操作 dtype

例如 case 0：

```python
x = tl.load(x_ptr + offsets, mask=mask, other=0).to(tl.float16)
y = tl.load(y_ptr + offsets, mask=mask, other=0).to(tl.float32)
result = x + y
```

虽然 host 侧 `x` 和 `y` 的实际 tensor 都是 FP32，kernel 在 load 之后显式地把 `x` 转成 FP16、把 `y` 转成 FP32。`x + y` 看到的是 FP16 tensor 和 FP32 tensor，因此公共运算 dtype 是 FP32。

`tl.static_assert(result.dtype == tl.float32, ...)` 证明的不是“输出 buffer 是 FP32”，而是“这步加法的结果类型是 FP32”。这两个检查层要分开理解。

### 第 3 层：scalar 分支验证“不要按直觉猜”

case 2：

```python
x = tl.load(x_ptr + offsets, mask=mask, other=0).to(tl.int16)
result = x + 4.0
```

`x` 是 integral kind，`4.0` 是 floating scalar。floating kind 更高，scalar 会参与 promotion；`4.0` 可以用 FP32 表示，因此两者提升到 FP32。

case 3 和 case 4 反过来：

```python
result = x + 1
result = x + 1.0
```

`x` 已经是 FP16 tensor，`1` 的 integral kind 更低，`1.0` 的 floating kind 与 tensor 相同。按照 scalar 规则，它们都不把 tensor 提升到 FP32，所以结果分别是 FP16。

这说明 scalar 规则不是简单的“更宽的类型永远胜出”，而是先比较 kind，再决定 scalar 是否参与；只有 scalar kind 更高时，才选择能容纳该 scalar 的类型。

### 第 4 层：`tl.where` 只提升后两个参数

case 6：

```python
condition = offsets < BLOCK_SIZE // 2
result = tl.where(condition, x, y)
```

`condition` 是布尔 mask，负责选择元素；`x` 和 `y` 分别来自 FP16 和 FP32 tensor。promotion 作用于 `tl.where` 的最后两个参数，因此公共 dtype 是 FP32，结果也是 FP32。

不要把 condition 和 value 混为一谈：

```text
condition dtype control: bool / int1
value dtype promotion:   fp16 + fp32 -> fp32
```

## 具体执行轨迹：用一组输入把规则走完

脚本使用：

```text
x = [1, 2, 3, 4, 5, 6, 7, 8]
y = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
BLOCK_SIZE = 8
```

### case 0：FP16 + FP32

`x` 在 kernel 内先变成 FP16，`y` 是 FP32：

```text
[1, 2, 3, 4, 5, 6, 7, 8] + [0.5, ...]
公共 dtype: fp32
结果: [1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5]
输出 buffer dtype: float32
```

### case 1：FP16 + BF16

两种输入都是 16-bit 浮点，文档规则选择 FP16：

```text
公共 dtype: fp16
结果: [1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5]
输出 buffer dtype: float16
```

### case 2：INT16 + 4.0

整型 tensor 遇到浮点 scalar，scalar kind 更高：

```text
[1, 2, 3, 4, 5, 6, 7, 8] + 4.0
公共 dtype: fp32
结果: [5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
```

### case 5：FP16 / FP16

除法的浮点例外把运算拉到 FP32：

```text
x = [1, 2, 3, 4, 5, 6, 7, 8]
y = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
公共运算 dtype: fp32
结果: [2, 4, 6, 8, 10, 12, 14, 16]
输出 buffer dtype: float32
```

### case 6：`tl.where(cond, FP16, FP32)`

前四个位置选择 FP16 分支，后四个位置选择 FP32 分支：

| offset | condition | 选中值 | promotion 后 | 输出 |
|---:|---|---:|---:|---:|
| 0 | true | `1` | fp32 | `1.0` |
| 1 | true | `2` | fp32 | `2.0` |
| 2 | true | `3` | fp32 | `3.0` |
| 3 | true | `4` | fp32 | `4.0` |
| 4 | false | `0.5` | fp32 | `0.5` |
| 5 | false | `0.5` | fp32 | `0.5` |
| 6 | false | `0.5` | fp32 | `0.5` |
| 7 | false | `0.5` | fp32 | `0.5` |

最终结果是 `[1.0, 2.0, 3.0, 4.0, 0.5, 0.5, 0.5, 0.5]`，dtype 为 FP32。

## 为什么它对算子开发很关键

### 1. 用 FP16/BF16 做存储，不等于用 FP16/BF16 做 accumulation

在 GEMM、attention 和 reduction 中，低精度输入很常见，但长时间累加通常需要 FP32。Type promotion 可以解释一部分隐式行为，但不要把关键精度交给隐式行为：

```python
acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
acc += tl.dot(a, b)  # 输入可以是 fp16，accumulator 显式保持 fp32
```

显式 accumulator dtype 比“希望 promotion 帮我自动变宽”更可检查，也更容易跨版本保持稳定。

### 2. 除法经常在 FP32 中发生

`fp16 / fp16` 的公共运算 dtype 是 FP32，这会改变中间求值精度，也可能影响性能和寄存器压力。不要把除法一律当成低精度 elementwise op。若确实需要别的行为，应先写清数值需求，再做显式 cast。

### 3. 输出 dtype 不能反推中间 dtype

如果中间结果是 FP32，最后存到 FP16 buffer，检查输出时只能看到 FP16。要确认计算路径，必须同时看 load 后的 dtype、每条关键表达式的结果 dtype 和 store 目标 dtype。`tl.static_assert` 正是用来把中间 dtype 固定下来。

## 常见错误和可观察症状

| 错误理解或写法 | 可观察症状 | 根因 |
|---|---|---|
| 认为 `fp16 + 1.0` 一定变成 FP32 | 结果保持 FP16，累加精度不如预期 | scalar 的 kind 不高于 tensor kind，不参与 promotion |
| 认为 `int16 + 4.0` 仍按 INT16 运算 | 结果变成 FP32，数值路径与预期不同 | 浮点 scalar kind 高于整数 tensor，选择 FP32 |
| 在 FP16 buffer 上验证后就说“全程 FP16” | 输出 dtype 是 FP16，但中间表达式其实在 FP32 | store 会把运算结果 cast 回目标 buffer |
| 用 `fp16 / fp16` 期待 FP16 除法 | 运算 dtype 为 FP32，资源和精度表现与预期不同 | 除法对窄浮点有 FP32 例外 |
| 把 promotion 和 broadcasting 混为一谈 | 以为 shape 对齐会顺便决定 dtype，或反过来 | 一个决定公共 dtype，一个决定公共 shape |
| 依赖隐式的 `bf16`/整数混合规则 | 不同 Triton 版本或目标上结果 dtype 变化 | 规则边界与实现细节相关；关键路径应显式 cast |
| 把 unsigned tensor 和负 scalar 混用 | 编译期报错或得到不安全结果 | 负值无法安全表示在 unsigned 目标类型中 |

## 运行与验证边界

静态语法检查：

```bash
python3 -m py_compile operators/code/triton/type_promotion.py
```

在带有 CUDA GPU、PyTorch 和 Triton 的环境中运行：

```bash
python operators/code/triton/type_promotion.py
```

预期输出会逐行打印每个 case 的 dtype 和值，最后打印：

```text
all_checks_passed: True
```

其中 dtype 检查的预期边界是：

```text
fp16 + fp32 -> fp32
fp16 + bf16 -> fp16
int16 + 4.0 -> fp32
fp16 + 1 -> fp16
fp16 + 1.0 -> fp16
fp16 / fp16 -> fp32
tl.where(cond, fp16, fp32) -> fp32
```

当前仓库所在的开发环境没有安装 Triton runtime，也没有可用的 CUDA GPU。因此这里只能完成 Python 静态语法检查，不能声称已经在真机执行过该 kernel。真机验证时，具体 Triton 版本和目标后端仍应作为测试矩阵的一部分；本节通过显式 `.to(...)` 和静态断言把规则固定下来，避免把版本差异混进教学结论。

## 自我检查（含答案）

1. **`float16` tensor 加 Python scalar `1.0`，为什么结果通常不是 FP32？**
   - **答：** 因为 scalar 的 floating kind 与 tensor 的 floating kind 相同，scalar 不参与 promotion；运算继续使用 tensor 的 FP16 dtype。要强制 FP32，应显式 `.to(tl.float32)`。
2. **`int16` tensor 加 `4.0` 的结果 dtype 是什么？**
   - **答：** FP32。整数 tensor 的 kind 是 integral，浮点 scalar 的 kind 更高；`4.0` 可以用 FP32 表示，因此两边提升到 FP32。
3. **`fp16 + fp32 -> fp32` 与 `[128, 1] + [1, 256] -> [128, 256]` 分别由什么规则控制？**
   - **答：** 前者由 type promotion 控制，选择公共 dtype；后者由 broadcasting 控制，选择公共 shape。两者可以同时发生，但不是同一个机制。
4. **为什么 `fp16 / fp16` 不能简单按“两边都是 FP16”理解？**
   - **答：** 除法/取模对窄浮点有例外，通常把浮点操作数提升到 FP32 进行运算；最终写回 FP16 只是 store 时的转换。
5. **如果只想确认 kernel 中间加法是 FP32，应该怎么做？**
   - **答：** 在关键操作数上显式 `.to(tl.float32)`，并在编译期用 `tl.static_assert(result.dtype == tl.float32, ...)` 固定结果 dtype；不要只检查输出 buffer 的 dtype。

## 进度

已经覆盖：

- [x] Triton type promotion：kind 层级、tensor-tensor 宽度与 signedness、tensor-scalar 参与规则、除法/取模例外，以及 promotion 与 broadcasting 的区别。

下一知识点：

- [ ] Triton Broadcasting 与 chained comparisons：shape 如何按维度对齐，以及比较表达式为什么不能按普通 Python 链式语义理解。
