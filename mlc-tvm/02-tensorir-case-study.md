# 02 TensorIR Case Study

对应 notebook:
`3_TensorIR_Tensor_Program_Abstraction_Case_Study_Action.ipynb`

## 本课目标

把 TensorIR 拆成几个基本构件，并用矩阵乘法作为案例：

```text
函数参数
Buffer
循环
Block
Block Axis
函数属性
调度变换
编译执行
```

## TVMScript 基本结构

```python
@I.ir_module
class MyModule:
    @T.prim_func(s_tir=True)
    def main(
        A: T.Buffer((1024, 1024), "float32"),
        B: T.Buffer((1024, 1024), "float32"),
        C: T.Buffer((1024, 1024), "float32"),
    ) -> None:
        ...
```

当前 TVM 开发版中：

- `I.ir_module` 定义一个 IRModule。
- `T.prim_func(s_tir=True)` 定义一个调度级 TensorIR 函数。
- `T.Buffer` 声明有形状和 dtype 的参数 Buffer。
- 函数返回 `None` 并不表示没有输出，输出通过 Buffer 写回。

## Buffer

```python
A: T.Buffer((1024, 1024), "float32")
```

表示 A 是一个二维 Buffer：

```text
shape = (1024, 1024)
dtype = float32
```

Buffer 是索引和依赖分析的基础。TVM 通过它判断两个访问是否重叠、是否存在读写冲突，以及能否进行重排和向量化。

## 循环

```python
for i, j, k in T.grid(1024, 1024, 1024):
    ...
```

等价于三层嵌套循环。`T.grid` 只是简洁的语法糖。

在矩阵乘法中：

```text
i: 输出行，spatial
j: 输出列，spatial
k: reduction
```

## Block

```python
with T.sblock("matmul"):
    vi, vj, vk = T.axis.remap("SSR", [i, j, k])
    with T.init():
        C[vi, vj] = 0.0
    C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]
```

`T.sblock` 是当前开发版对旧 `T.block` 的名称。Block 是一个计算语义单元，不只是语法括号。

Block 中显式包含：

- 它写哪些 Buffer 区域
- 它读哪些 Buffer 区域
- 哪些轴是空间轴
- 哪些轴是 reduction 轴
- reduction 的初始化如何完成

Block 内部负责“怎么算”，Block 外部负责“循环如何组织”，调度主要作用于两者之间的映射。

## Axis 与 SSR

```python
vi, vj, vk = T.axis.remap("SSR", [i, j, k])
```

`SSR` 表示：

| 字符 | 含义 | 对 matmul 的对应 |
|---|---|---|
| `S` | spatial | `i` 输出行 |
| `S` | spatial | `j` 输出列 |
| `R` | reduction | `k` 累加轴 |

如果 block 轴和循环没有通过 block 的迭代空间正确绑定，访问会错位。Block 中的额外 axis 信息也让 TVM 能验证循环变换是否保持 block 语义。

## Reduction Init

```python
with T.init():
    C[vi, vj] = 0.0
```

reduction 需要一个初始值。矩阵乘法的累加初值是 0。

这个初始化区域和后续更新区域可以分别处理：

- `init`: 初始化 accumulator
- update block: 对每个 `k` 执行 `C += A * B`

后续 `decompose_reduction` 会利用这个结构化信息。

## 函数属性

```python
T.func_attr({
    "global_symbol": "main",
    "tirx.noalias": True,
})
```

### `global_symbol`

指定外部可见的函数符号。常见的 `"main"` 让运行时通过：

```python
lib["main"](...)
```

调用它。

### `tirx.noalias`

承诺不同 Buffer 不互相重叠。它允许 TVM 更积极地进行：

- 循环重排
- 并行化
- 向量化
- 读写分析

这不是运行时的强制检查，而是给编译器的语义保证。属性通常可以省略，但某些优化会变保守，或者模块导出行为会不同。

## 初次变换：Split 与 Reorder

```python
sch = s_tir.Schedule(MyModule)
block = sch.get_sblock("matmul")
i, j, k = sch.get_loops(block)

i0, i1 = sch.split(i, factors=[None, 32])
sch.reorder(i0, j, i1, k)
```

变化后的结构可以理解为：

```text
i0: 外层 tile
i1: tile 内行
j:  输出列
k:  reduction
```

变换不改变计算结果，只是改变执行顺序。

## 常见高级原语

### `cache_read`

把输入搬运到局部缓存：

```python
A_local = sch.cache_read(block, 0, "local")
```

`0` 表示 block 读取的第 0 个 Buffer。

### `cache_write`

为输出或中间结果创建临时写入缓存：

```python
C_local = sch.cache_write(block, 0, "local")
```

### `compute_at`

把生产者移动到某个循环层内部计算：

```python
sch.compute_at(A_local, k)
```

它常用于把数据搬运放在合适循环层，以避免重复加载或减少缓存压力。

### `reverse_compute_at`

它从输出写回方向调整 producer 的位置。常见于 reduction 的 accumulator：

```python
sch.reverse_compute_at(C_local, j)
```

含义可以理解为：在 `j` 这一层完成输出累加区域的收口，然后把结果写回。

### `decompose_reduction`

```python
sch.decompose_reduction(block, k)
```

把 reduction 拆成：

```text
init block
update block
```

GPU shared memory、局部 blocking 和 TensorIntrin 都经常需要这个拆分。

## Build 与 Run

```python
lib = tvm.build(sch.mod, target="llvm")
lib["main"](A, B, C)
```

调度结束后，`sch.mod` 是变换后的 IRModule。它仍然是中间表示，需要经过 lowering 和 build 才能执行。

## 关键 API

| API | 说明 |
|---|---|
| `I.ir_module` | IRModule 定义 |
| `T.prim_func(s_tir=True)` | TensorIR 函数 |
| `T.Buffer` | 有形状和 dtype 的 Buffer |
| `T.grid` | 多层循环语法糖 |
| `T.sblock` | 当前版本的 Block |
| `T.axis.remap` | 绑定空间轴与 reduction 轴 |
| `T.init` | reduction 初始化 |
| `T.func_attr` | 函数属性 |
| `get_sblock` | 获取 Block |
| `get_loops` | 获取周围循环 |
| `split` | 拆分循环 |
| `reorder` | 重排循环 |
| `cache_read/cache_write` | 创建缓存 |
| `compute_at/reverse_compute_at` | 定位 producer 计算位置 |
| `decompose_reduction` | 分离 reduction 初始化与更新 |

## 本课结论

TensorIR 的价值不只是“可以写循环”，而是把计算语义、循环结构和读写依赖同时表示出来，使调度能够安全地重写程序。

```text
Buffer 告诉编译器数据在哪里
Block 告诉编译器一次计算是什么
Loop 告诉编译器计算怎样重复
Schedule 告诉编译器如何重新组织执行
```
