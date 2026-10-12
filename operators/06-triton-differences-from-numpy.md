# Triton 与 NumPy 的差异：整数除法、越界 Cast 与变量作用域

## 本次讲解位置

章节：`chapter_triton_semantics`<br>
小节：Triton Semantics 的 `Differences with NumPy`<br>
知识点：整数除法/取模的 C 取整、越界 float-to-int cast、控制流后的变量定义要求<br>
上次：Broadcasting 与 chained comparisons<br>
下次：`tl.where` 的 eager evaluation、显式 mask 与 control flow 的选择边界<br>
PTX：不指定具体 PTX；本节停留在 Triton source semantics、compile-time/runtime 分界和 cast 定义域

## 为什么现在讲这个

前面几节一直强调 Triton 大体遵循 NumPy 语义，但“大体遵循”不等于所有表达式都可以直接按 NumPy 或 Python 的直觉理解。下面三个差异最容易在普通测试里藏起来：

1. 正整数除法看起来完全一样，只有混合符号的整数才会暴露 C 取整和 Python floor 取整的区别。
2. 正常范围内的 float-to-int cast 看起来稳定，只有 NaN 或越界值才会进入 undefined 区域。
3. 变量在普通 Python 里可能在代码路径结束后仍然可见，但 Triton 的 kernel 作用域要求变量在所有路径上都有定义。

这些问题危险的地方不是“偶尔算错一个数”，而是测试集如果只覆盖正数、正常值或所有分支都赋值的路径，代码可能长期显示为正确，直到推理或边界数据出现。本节的目标不是记住三个孤立 warning，而是把三类边界固定成可以检查的写法。

官方规则来源：[Triton Semantics: Differences with NumPy](https://triton-lang.org/main/python-api/triton-semantics.html#differences-with-numpy)。

## 心智模型：Triton 同时存在编译期语义和 runtime 语义

同一个符号在 Triton kernel 中不一定是同一种东西：

| 对象 | 求值位置 | 语义来源 | 本节的坑 |
|---|---|---|---|
| `tl.constexpr` 数值 | Python 追踪/编译期 | Python 语义 | `-7 // 3` 按 floor，结果为 `-3` |
| runtime int tensor | GPU warning/runtime | C/Triton 语义 | `-7 // 3` 向零截断，结果为 `-2` |
| float-to-int cast | 编译后的硬件指令 | 只在定义域内有保证 | 越界和 NaN 结果是 undefined |
| kernel 内变量定义 | Triton AST/编译期作用域分析 | Triton 作用域规则 | 变量必须在所有路径上赋值 |

因此不要只问“Triton 和 Python 哪个对”，而要问：

```text
这个值现在还是 constexpr，还是已经变成 runtime tensor？
这个 cast 的输入是否在目标整数类型的可表示范围内？
这个变量在控制流结束后，编译器是否能证明每个路径都定义过它？
```

## 差异一：整数除法采用 C 的向零取整

### Python 的 `//` 是 floor division

Python 对整数除法的结果是向负无穷方向取整：

```text
-7 // 3 = -3
 7 // -3 = -3
-7 // -3 = 2
 7 // 3 = 2
```

因为 `-7 / 3 = -2.333...`，向负无穷取整得到 `-3`。Python 的 `%` 与这个除法配套，所以：

```text
-7 % 3 = 2
 7 % -3 = -2
```

### Triton runtime 整数 tensor 是 truncation toward zero

Triton 为了贴近 C 和硬件效率，对 runtime `int // int` 使用向零截断：

```text
-7 // 3 = -2
 7 // -3 = -2
-7 // -3 = 2
 7 // 3 = 2
```

对于 `-7 / 3 = -2.333...`，向零截断得到 `-2`。对应的 remainder 定义为：

```text
r = a - b * (a // b)
```

两种语义的完整对比如下：

| `a` | `b` | Python `a // b` | Triton runtime `a // b` | Python `a % b` | C 语义 `a % b` |
|---:|---:|---:|---:|---:|---:|
| `-7` | `3` | `-3` | `-2` | `2` | `-1` |
| `7` | `-3` | `-3` | `-2` | `-2` | `1` |
| `-7` | `-3` | `2` | `2` | `-1` | `-1` |
| `7` | `3` | `2` | `2` | `1` | `1` |

只有同号时，两种规则恰好一致。混合符号时才暴露差异。

### `tl.constexpr` 仍然走 Python 语义

最容易踩的坑不是“runtime 是 C 语义”，而是同一个数学表达式在变成 runtime 值前后会换语义：

```python
CONST_Q = -7 // 3
# CONST_Q == -3

x = tl.full((1,), -7, tl.int32)
y = tl.full((1,), 3, tl.int32)
RUNTIME_Q = x // y
# 概念上 RUNTIME_Q == -2
```

也就是说，下面两段代码不等价：

```text
constexpr:  -7 // 3
runtime:    tl.full(..., -7) // tl.full(..., 3)
```

当常量被参数化、从 host 传入或被提升成 tensor 时，取整规则就会从 Python 切到 runtime。对混合符号数据有明确要求时，应先定义想要的舍入方向，再选择算法或显式使用对应的除法实现，不要依赖“常量看起来没问题”来推断 runtime 行为。

## 差异二：越界 float-to-int cast 没有可移植结果

### cast 的定义域

浮点值转整数类型时，只有当“向零取整后的值”能装进目标整数范围，结果才有定义。对于 `int8`：

```text
可表示范围: [-128, 127]
```

以下输入不属于定义域：

```text
510.0  -> 超出 int8 上界 127
-510.0 -> 超出 int8 下界 -128
inf    -> 无法映射到有限整数
NaN    -> 没有整数对应值
```

规则不是“自动 saturate 到 127/-128”，也不是“所有后端都返回 0”。结果是不确定的，可能在编译器与 `TRITON_INTERPRET=1` 解释器之间不同，也可能在不同硬件、驱动和工具链版本之间不同。

### 安全写法：先定义 NaN，再 clamp，最后 cast

如果业务语义希望“越界值接近边界”且 NaN 映射到 0，可以写成：

```python
x = tl.where(x != x, 0.0, x)
x = tl.clamp(x, -128.0, 127.0)
out = x.to(tl.int8)
```

顺序很重要：

1. 先把 NaN 替换成明确值，防止 NaN 穿过 clamp 或 cast。
2. 再把有限值 clamp 到 `int8` 可表示范围。
3. 最后做 cast，此时每个输入都在定义域内。

安全转换后的例子：

| 输入 | NaN 处理 | clamp 后 | cast 后 |
|---:|---:|---:|---:|
| `510.0` | `510.0` | `127.0` | `127` |
| `-510.0` | `-510.0` | `-128.0` | `-128` |
| `NaN` | `0.0` | `0.0` | `0` |
| `12.5` | `12.5` | `12.5` | `12` |

`12.5` 转整数时向零取整为 12。这个行为在输入已处于定义域时是稳定的；真正不稳定的部分是越界和 NaN。

## 差异三：变量必须在控制流的所有路径上定义

### Python 作用域里的“看似可见”

普通 Python 函数中，循环体只绑定一个变量时，循环结束后这个名字通常仍然存在：

```python
def f(out_ptr):
    for _ in range(0, 1):
        value = 1.0
    store(out_ptr, value)
```

这段代码在 Python 层面可以运行，因为函数局部变量表在循环结束后仍然保留 `value`。

### Triton 要求所有路径都有定义

同样的结构在 Triton kernel 中会报 `NameError: 'value' is not defined`：

```python
@triton.jit
def kernel(out_ptr):
    for _ in range(0, 1):
        value = 1.0
    tl.store(out_ptr, value)  # error: value is not defined here
```

Triton 不按“动态变量是否存在”分析控制流，而是要求变量在退出 block 时，对该 block 之后的每一条路径都已经定义。这个规则即使循环范围写成 `range(0, 1)` 也不例外，因为编译器不会用“循环一定执行一次”来替代静态定义要求。

### 正确写法：在控制流之前先初始化

```python
value = 0.0
for _ in range(0, 1):
    value = 1.0
tl.store(out_ptr, value)
```

对 `if` 语句也一样：

```python
value = 0.0
if condition:
    value = 1.0
else:
    value = 2.0
tl.store(out_ptr, value)
```

关键不是“分支最后一定会执行哪个”，而是编译器能看到所有路径上的定义。对 tensor 变量还应注意两条路径赋出的 shape 和 dtype 要能满足后续合并；否则即使名字定义了，类型也可能无法统一。

## 完整可运行代码

文件：[`code/triton/numpy_differences.py`](code/triton/numpy_differences.py)

```python
"""Triton semantics that differ from NumPy/Python.

Run:
    python operators/code/triton/numpy_differences.py

Requires a CUDA GPU, PyTorch, and Triton.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _runtime_divmod_kernel(
    x_ptr,
    y_ptr,
    q_ptr,
    r_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0).to(tl.int32)
    y = tl.load(y_ptr + offsets, mask=mask, other=1).to(tl.int32)

    q = x // y  # Runtime Triton integer tensor: C-style truncation.
    r = x % y   # Defined as x - y * q.
    tl.store(q_ptr + offsets, q, mask=mask)
    tl.store(r_ptr + offsets, r, mask=mask)


@triton.jit
def _constexpr_divmod_kernel(
    q_ptr,
    r_ptr,
    A: tl.constexpr,
    B: tl.constexpr,
):
    # tl.constexpr arithmetic follows Python semantics: -7 // 3 == -3.
    q = A // B
    r = A % B
    tl.store(q_ptr, q)
    tl.store(r_ptr, r)


@triton.jit
def _safe_float_to_int8_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    # NaN has no portable integer result, so define it before casting.
    x = tl.where(x != x, 0.0, x)
    x = tl.clamp(x, -128.0, 127.0)
    out = x.to(tl.int8)
    tl.store(out_ptr + offsets, out, mask=mask)


@triton.jit
def _initialized_loop_value_kernel(
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Declared before the loop so it exists on every path after the loop.
    value = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for _ in range(1):
        value += 1.0

    tl.store(out_ptr + offsets, value, mask=mask)


def runtime_divmod(x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if not x.is_cuda or not y.is_cuda:
        raise ValueError("x and y must be CUDA tensors")
    if x.shape != y.shape or x.ndim != 1:
        raise ValueError("x and y must be same-shape 1-D tensors")

    q = torch.empty_like(x, dtype=torch.int32)
    r = torch.empty_like(x, dtype=torch.int32)
    n_elements = x.numel()
    if n_elements == 0:
        return q, r

    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    _runtime_divmod_kernel[grid](x, y, q, r, n_elements, BLOCK_SIZE=16)
    return q, r


def constexpr_divmod(a: int, b: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    q = torch.empty((1,), device=device, dtype=torch.int32)
    r = torch.empty((1,), device=device, dtype=torch.int32)
    _constexpr_divmod_kernel[(1,)](q, r, A=a, B=b)
    return q, r


def safe_float_to_int8(x: torch.Tensor) -> torch.Tensor:
    if not x.is_cuda:
        raise ValueError("x must be a CUDA tensor")
    if x.dtype != torch.float32 or x.ndim != 1:
        raise ValueError("x must be a 1-D float32 tensor")

    output = torch.empty_like(x, dtype=torch.int8)
    n_elements = x.numel()
    if n_elements == 0:
        return output

    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    _safe_float_to_int8_kernel[grid](x, output, n_elements, BLOCK_SIZE=16)
    return output


def initialized_loop_value(n_elements: int, device: torch.device) -> torch.Tensor:
    output = torch.empty((n_elements,), device=device, dtype=torch.float32)
    if n_elements == 0:
        return output

    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    _initialized_loop_value_kernel[grid](output, n_elements, BLOCK_SIZE=16)
    return output


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("需要 CUDA GPU 才能运行 Triton semantics kernel")

    device = triton.runtime.driver.active.get_active_torch_device()

    x = torch.tensor([-7, 7, -7, 7], device=device, dtype=torch.int32)
    y = torch.tensor([3, -3, -3, 3], device=device, dtype=torch.int32)
    q, r = runtime_divmod(x, y)
    q_ref = torch.trunc(x.to(torch.float32) / y.to(torch.float32)).to(torch.int32)
    r_ref = x - y * q_ref
    if not torch.equal(q, q_ref) or not torch.equal(r, r_ref):
        raise AssertionError("runtime div/mod does not match C-style truncation")

    q_const, r_const = constexpr_divmod(-7, 3, device)
    if q_const.item() != -3 or r_const.item() != 2:
        raise AssertionError("constexpr div/mod must follow Python semantics")

    cast_input = torch.tensor([510.0, -510.0, float("nan"), 12.5], device=device)
    cast_output = safe_float_to_int8(cast_input)
    cast_ref = torch.nan_to_num(cast_input, nan=0.0).clamp(-128.0, 127.0).to(torch.int8)
    if not torch.equal(cast_output, cast_ref):
        raise AssertionError("safe float-to-int8 conversion mismatch")

    loop_output = initialized_loop_value(4, device)
    if not torch.equal(loop_output, torch.ones((4,), device=device, dtype=torch.float32)):
        raise AssertionError("initialized loop value mismatch")

    print(f"runtime x: {x.tolist()}")
    print(f"runtime y: {y.tolist()}")
    print(f"runtime q (trunc): {q.tolist()}")
    print(f"runtime r: {r.tolist()}")
    print(f"constexpr -7 // 3: {q_const.item()}, -7 % 3: {r_const.item()}")
    print(f"cast input: {cast_input.tolist()}")
    print(f"cast output: {cast_output.tolist()}")
    print(f"loop output: {loop_output.tolist()}")
    print("all_checks_passed: True")


if __name__ == "__main__":
    main()
```

脚本有四个验证点：

1. runtime int tensor 的 `//` 和 `%` 匹配 C-style truncation。
2. `tl.constexpr` 的 `-7 // 3`、`-7 % 3` 匹配 Python 的 `-3`、`2`。
3. 越界 float 和 NaN 经过显式处理后再转 `int8`，结果有定义。
4. 循环前先初始化 `value`，循环后可以安全使用。

## 按执行阶段拆解 runtime div/mod

### 第 1 层：load 成 int32 tensor

```python
x = tl.load(x_ptr + offsets, mask=mask, other=0).to(tl.int32)
y = tl.load(y_ptr + offsets, mask=mask, other=1).to(tl.int32)
```

这里把输入显式转成 runtime `int32`。一旦进入 tensor 表达式，`//` 和 `%` 就不再使用 `tl.constexpr` 的 Python 语义。

### 第 2 层：除法先得到向零截断的商

```python
q = x // y
```

当 `x=-7, y=3` 时：

```text
精确商 = -2.333...
Q = -2
```

### 第 3 层：余数由商反推

```python
r = x % y
```

语义是：

```text
r = x - y * q
  = -7 - 3 * (-2)
  = -1
```

所以 C 语义下得到 `(-7 // 3, -7 % 3) = (-2, -1)`，而 Python 得到 `(-3, 2)`。这不是符号“算错了”，而是两组不同的定义在混合符号时选择了不同方向。

## 具体执行轨迹：混合符号整数

脚本输入：

```text
x = [-7, 7, -7, 7]
y = [ 3, -3, -3, 3]
```

runtime 结果：

| 索引 | `x` | `y` | `x // y` | `x % y` |
|---:|---:|---:|---:|---:|
| 0 | -7 | 3 | -2 | -1 |
| 1 | 7 | -3 | -2 | 1 |
| 2 | -7 | -3 | 2 | -1 |
| 3 | 7 | 3 | 2 | 1 |

constexpr 结果：

```text
-7 // 3 = -3
-7 % 3 = 2
```

两者放在一起就能看到：不是因为 data type 都是 int，所以结果一定相同；还需要知道这个 int 是编译期常量还是 runtime tensor。

## 具体执行轨迹：安全 cast

输入：

```text
[510.0, -510.0, nan, 12.5]
```

处理过程：

| 输入 | `x != x` | NaN 处理后 | clamp 后 | int8 |
|---:|---:|---:|---:|---:|
| `510.0` | false | `510.0` | `127.0` | `127` |
| `-510.0` | false | `-510.0` | `-128.0` | `-128` |
| `nan` | true | `0.0` | `0.0` | `0` |
| `12.5` | false | `12.5` | `12.5` | `12` |

如果把 `510.0` 或 `nan` 直接 cast 到 `int8`，结果没有可移植保证。脚本没有依赖那段 undefined 行为，而是在 cast 前显式定义了业务想要的边界。

## 具体执行轨迹：循环变量作用域

合法版本：

```python
value = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
for _ in range(1):
    value += 1.0
tl.store(out_ptr + offsets, value, mask=mask)
```

对 BLOCK_SIZE 个位置：

```text
初始 value = [0, 0, 0, 0]
循环执行一次 value += 1
循环后 value = [1, 1, 1, 1]
store 写出 [1, 1, 1, 1]
```

如果删掉循环前的 `value = tl.zeros(...)`，Triton 在编译期检查循环后的 `value` 时无法认为它在所有路径上都有定义，会产生 `NameError`。这个错误是编译期作用域错误，不是 GPU runtime 数值错误。

## 常见错误和可观察症状

| 错误 | 症状 | 根因 |
|---|---|---|
| 用 Python `//` 结果推断 runtime Triton 结果 | 负数输入时商和余数与参考不同 | runtime 整数除法向零截断，Python 向负无穷取整 |
| 用同号样例验证 div/mod | 测试全部通过，混合符号上线后出错 | 同号时两种舍入规则恰好一致 |
| 把 `tl.constexpr` 常量参数化后继续假设旧语义 | 常量版和 runtime 版输出不同 | constexpr 走 Python，runtime tensor 走 Triton/C |
| 把 `510.0` 直接转 `int8` 并期待 127 | 编译器、解释器或后端结果可能不同 | 越界 float-to-int cast 是 undefined |
| 先 cast 再处理 NaN | NaN 已经产生不可移植结果 | 必须在 cast 前替换或拒绝 NaN |
| 认为 clamp 会自动处理 NaN | NaN 可能穿过 min/max 选择 | 应先显式检测 `x != x` |
| 只在循环内给 `value` 赋值 | Triton 编译时报 `NameError` | 变量必须在控制流后的每一条路径上定义 |
| `if/else` 两条路径变量 dtype/shape 不一致 | 合并变量时类型错误或后续广播失败 | 作用域可见不等于类型可合并 |

## 运行与验证边界

静态语法检查：

```bash
python3 -m py_compile operators/code/triton/numpy_differences.py
```

有 CUDA GPU、PyTorch 和 Triton 时运行：

```bash
python operators/code/triton/numpy_differences.py
```

预期关键输出：

```text
runtime q (trunc): [-2, -2, 2, 2]
runtime r: [-1, 1, -1, 1]
constexpr -7 // 3: -3, -7 % 3: 2
cast output: [127, -128, 0, 12]
loop output: [1.0, 1.0, 1.0, 1.0]
all_checks_passed: True
```

当前开发环境没有 Triton runtime 和 CUDA GPU，因此只完成 Python 静态语法检查，没有声称在本机运行过 kernel。尤其要注意“越界 cast”本身不能靠真机跑一次就证明可移植；本节示例故意先把输入规范化到 `int8` 定义域，再执行 cast。

## 自我检查（含答案）

1. **Python 的 `-7 // 3` 和 runtime Triton `int32` tensor 的 `-7 // 3` 分别是多少？**
   - **答：** Python 是 `-3`，向负无穷取整；runtime Triton 是 `-2`，向零截断。
2. **为什么 `-7 % 3` 在 Python 和 C 语义下不同？**
   - **答：** remainder 由商定义，`r = a - b * (a // b)`。Python 使用 floor 商 `-3`，所以余数 `2`；C 使用 trunc 商 `-2`，所以余数 `-1`。
3. **`tl.constexpr` 的整数除法跟哪套规则？**
   - **答：** 跟 Python 语义，不跟 runtime tensor 的 C 语义。混合符号常量计算可能与运行时计算得到不同舍入结果。
4. **为什么 `510.0.to(int8)` 不能直接依赖？**
   - **答：** `int8` 只能表示 `[-128, 127]`，`510.0` 越界；越界 float-to-int cast 的结果 undefined，可能随编译器、解释器和后端变化。应先 clamp 到范围。
5. **Triton 为什么不允许只在 `for` 循环里给变量第一次赋值，然后在循环外使用？**
   - **答：** Triton 要求变量在控制流退出后的每条路径上都有定义。它不会因为 `range(0, 1)` 看起来一定会执行，就把循环体内第一次赋值视为循环外已定义。

## 进度

已经覆盖：

- [x] Triton 与 NumPy 的差异：runtime 整数除法/取模向零截断、`tl.constexpr` 使用 Python 取整。
- [x] 越界和 NaN float-to-int cast 的 undefined 边界，以及先处理 NaN、再 clamp、最后 cast 的安全模式。
- [x] Triton 控制流后的变量作用域要求，以及所有路径初始化的写法。

下一知识点：

- [ ] `tl.where` 的 eager evaluation、显式 mask 与 control flow 的取舍：如何避免无意中执行昂贵的 load 或无效地址访问。
