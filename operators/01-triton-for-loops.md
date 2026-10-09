# Triton 的 `for` 循环：分块遍历与累加

## 本次讲解位置

章节：`chapter_triton_control_flow`（算子开发 / Triton）<br>
小节：Triton kernel 中的 `for range` 与 `tl.static_range`<br>
知识点：用循环逐块覆盖较长维度，并跨迭代累加<br>
上次：`tl.program_id` 与 `tl.arange` 组成一个 elementwise tile（[原地加法示例](code/triton/fp16_inplace_add.py)）<br>
下次：GEMM 的 K-loop：每轮 `tl.dot` 如何累加到同一 `acc`（[现有 Triton GEMM 示例](code/triton/bf16_gemm_at_bt.py)）<br>
PTX：不适用；本节停留在 Triton JIT / Triton language 层，不指定某条 PTX 指令。

## 为什么现在讲这个

`tl.arange(0, BLOCK_SIZE)` 只描述一个 tile。若一行有 5 个元素而 tile 宽度是 4，单独一次 load 会漏掉第 5 个元素；若把 tile 一味扩大，又可能浪费寄存器并降低并行度。循环让一个 program 分多轮处理 tile，并用累加器保留前几轮结果。先弄清楚循环变量是“第几个 tile”、向量索引是“tile 内的哪些元素”，后面读 GEMM 的 K-loop 时就不容易把顺序累加误认为一次覆盖整个 K 维。

## 心智模型：CTA 顺序走 tile，tile 内元素并行

把一个 Triton program（通常对应一个 CTA）想成一个处理小组：

1. `tl.program_id(0)` 选中该 CTA 负责的行。
2. `for tile_id in range(...)` 依次选择这行的第 0、1、……个 tile；`tile_id` 是循环控制用的标量。
3. 每轮里的 `tl.arange(0, BLOCK_SIZE)` 生成 tile 内的向量坐标。向量元素由 Triton 映射到该 program 的 lanes / warps 并行处理。
4. `acc` 在轮次之间保留部分和；循环结束后再归约成一个数。

所以，**循环的轮次是顺序的，单轮 tile 内的向量工作仍是并行的**。它不是让 Python 在 GPU 上逐元素调用 kernel，也不等于每个 lane 都各自启动一个 Python 循环。

## 两种常见循环写法

| 写法 | 含义 | 适合场景 | 要留意什么 |
|---|---|---|---|
| `for i in range(n)` | 普通循环；Triton JIT 会把它编译进 kernel。`n` 可以是编译期常量，也可以来自运行时标量。 | 循环较长，或希望保持循环结构 | 即便边界可知，也不要把“编译器可能优化”当成“源码必定完全展开” |
| `for i in tl.static_range(n)` | 静态循环，按编译期已知的轮数展开 | 轮数少且固定的小循环 | 展开会复制循环体；轮数过大可能增加代码体积和编译时间 |
| `tl.arange(...)` | 不是循环，而是产生一个向量化的坐标范围 | 一个 tile 内的并行元素坐标 | 只覆盖指定 tile，不会自动遍历后续 tile |

`static_range` 的迭代边界必须能在编译时确定。是否展开、展开多少，不只是语法问题，也影响编译时间与生成代码大小；刚开始写 kernel 时，优先用普通 `range` 表达算法，再在有测量依据时调整。

## 完整可运行例子：循环分块求每行和

文件：[`code/triton/for_loop_row_sum.py`](code/triton/for_loop_row_sum.py)

```python
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _row_sum_kernel(
    x_ptr,
    y_ptr,
    n_cols: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(axis=0)
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    for tile_id in range(tl.cdiv(n_cols, BLOCK_SIZE)):
        cols = tile_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        values = tl.load(
            x_ptr + row * n_cols + cols,
            mask=cols < n_cols,
            other=0.0,
        )
        acc += values

    tl.store(y_ptr + row, tl.sum(acc, axis=0))


def row_sum(x: torch.Tensor, block_size: int = 4) -> torch.Tensor:
    if not x.is_cuda:
        raise ValueError("x must be a CUDA tensor")
    if x.dtype != torch.float32 or x.ndim != 2 or not x.is_contiguous():
        raise ValueError("x must be a contiguous 2-D torch.float32 tensor")
    if block_size <= 0:
        raise ValueError("block_size must be positive")

    n_rows, n_cols = x.shape
    y = torch.empty((n_rows,), device=x.device, dtype=torch.float32)
    if n_rows == 0:
        return y
    _row_sum_kernel[(n_rows,)](x, y, n_cols, BLOCK_SIZE=block_size)
    return y


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("需要可用的 CUDA GPU 才能运行 Triton kernel")

    x = torch.arange(1, 16, device="cuda", dtype=torch.float32).reshape(3, 5)
    y = row_sum(x, block_size=4)
    result = y.cpu().tolist()
    print(result)
    assert result == [15.0, 40.0, 65.0]


if __name__ == "__main__":
    main()
```

运行命令：

```bash
python operators/code/triton/for_loop_row_sum.py
```

期望输出：

```text
[15.0, 40.0, 65.0]
```

这个例子需要 PyTorch、Triton 和可用的 CUDA GPU。若当前环境缺少任一依赖或 CUDA 设备，代码可以做静态检查，但不能据此声称 kernel 已经运行验证。

## 按执行顺序追踪代码

### 1. Grid 与 CTA 的职责

调用处使用 grid `(n_rows,)`。输入有 3 行，所以启动 3 个 program：`program_id(0)` 分别是 0、1、2；每个 program 只负责一行。循环不会让一个 program 接管其他行。

### 2. `acc` 是跨轮次保留的向量累加器

`acc` 的逻辑 shape 是 `(4,)`，dtype 是 FP32。它不是输入行的完整副本，而是每轮 tile 在对应向量位置上的部分和。kernel 的状态关系为：

```text
acc 初值 -> 加上第 0 个 tile -> 加上第 1 个 tile -> tl.sum -> 写出该行结果
```

### 3. 循环次数与坐标公式

输入每行 5 个元素，tile 宽度为 4：

```text
循环次数 = ceil(5 / 4) = 2
cols     = tile_id * 4 + [0, 1, 2, 3]
```

以第 1 行（从 0 开始计数，即数值 `[6, 7, 8, 9, 10]`）为例：

| `tile_id` | `cols` | 有效输入 | 加入 `acc` 后 |
|---:|---|---|---|
| 初值 | — | — | `[0, 0, 0, 0]` |
| 0 | `[0, 1, 2, 3]` | `[6, 7, 8, 9]` | `[6, 7, 8, 9]` |
| 1 | `[4, 5, 6, 7]` | `[10, 0, 0, 0]` | `[16, 7, 8, 9]` |
| 结束 | — | `tl.sum(acc, 0)` | `16 + 7 + 8 + 9 = 40` |

这里 `mask=cols < n_cols` 把最后一个 tile 越界的三个位置屏蔽掉，`other=0.0` 让它们对累加没有贡献。其余两行同理得到 `1+2+3+4+5=15` 和 `11+12+13+14+15=65`。

### 4. 作用域与资源

- `row`、`tile_id` 是 program 级别的控制值；同一 program 在同一轮使用同一个 tile 编号。
- `cols`、`values`、`acc` 是 tile 级向量。编译器把向量计算分配到该 program 的线程 / lane 与寄存器中；具体 lane 布局由编译器决定，不应把 `tl.arange` 的每个元素硬编码成某个 lane。
- `tl.load` 读取全局内存，`acc` 保存部分和，`tl.sum` 完成最终归约，`tl.store` 写全局内存。
- 源码不需要手写 CTA 间同步：各行输出彼此独立；循环里的累加和最后归约属于 Triton 表达的依赖，归约实现所需的底层协作由编译器处理。

## 常见错误与可观察症状

1. **只写一次 `tl.arange`，忘了循环或扩大覆盖范围。** 当 `N > BLOCK_SIZE` 时，输出只含前一个 tile 的贡献，后面的数据被漏算。
2. **循环偏移漏乘 tile 宽度。** 若写成 `cols = tile_id + tl.arange(...)`，相邻轮会重叠大部分元素；结果重复计数，并可能仍然看起来像一个数值输出。
3. **尾块忘记 mask。** 当 `N` 不是 `BLOCK_SIZE` 的整数倍时，最后一轮会访问逻辑范围外的地址，可能报错或产生错误结果。mask 要按实际列坐标 `cols < n_cols` 判断。
4. **把 `range` 当作自动展开。** 正确性通常不因此改变，但误以为一定展开会导致对代码体积、编译时间或性能做出错误判断。需要显式静态展开时再考虑 `tl.static_range`，并比较编译与运行成本。
5. **`acc` 每轮重新初始化。** 如果把 `acc = tl.zeros(...)` 放进循环体，循环每轮都会抹掉此前累加，最后只剩最后一个 tile 的部分和。

## 自我检查（含答案）

1. `N=10, BLOCK_SIZE=4` 时循环几轮？
   - **答：** `ceil(10/4)=3` 轮，tile 起始列分别是 0、4、8；最后一轮只有前两个元素有效。
2. `range` 和 `tl.static_range` 最大的语义区别是什么？
   - **答：** `range` 保留普通循环结构；`tl.static_range` 要求编译期已知轮数并静态展开循环体。
3. 为什么每轮的 `cols` 要加 `tile_id * BLOCK_SIZE`？
   - **答：** 把本轮 tile 平移到对应的列区间，保证 tile 之间连续、不重叠、不遗漏。
4. 本例的三个 program 各自负责什么？
   - **答：** 每个 program 负责一整行；每行内部再由循环顺序遍历多个 tile。

## 进度

已经覆盖：

- [x] Triton `for range` 分块循环、尾块 mask 与跨轮次累加；`tl.static_range` 的适用边界。

下一知识点：

- [ ] GEMM K-loop：`tl.dot` 每轮的 tile 计算和 `acc` 累加，以及 K 尾块 mask。
