# 01 Tensor Program Abstraction

对应 notebook:
`2_tensor_program_abstraction.ipynb`

## 本课目标

理解机器学习编译中的两层核心抽象：

1. 先用 Tensor Expression 描述“要算什么”。
2. 再用 Schedule 描述“循环如何组织、如何映射到硬件”。

同一份计算结果可以对应很多不同的循环实现，调度不会改变数学语义，只改变执行方式。

## Tensor Expression

先用 `te.placeholder` 声明输入张量：

```python
A = te.placeholder((1024, 1024), name="A")
B = te.placeholder((1024, 1024), name="B")
```

声明 reduction 轴：

```python
k = te.reduce_axis((0, A.shape[1]), name="k")
```

计算定义只描述每个输出元素是什么：

```python
C = te.compute(
    (A.shape[0], B.shape[1]),
    lambda i, j: te.sum(A[i, k] * B[k, j], axis=k),
    name="C",
)
```

`te.compute` 的 lambda 表示：

```text
C[i, j] = sum_k A[i, k] * B[k, j]
```

此时还没有决定循环顺序、并行方式、缓存和向量化。

## 从 Tensor Expression 到 PrimFunc

```python
func = te.create_prim_func([A, B, C])
mod = tvm.IRModule({"main": func})
```

`PrimFunc` 是 TensorIR 层的函数表示，包含：

- 参数 Buffer
- 循环
- block
- 读写关系

可以直接查看生成的 TVMScript：

```python
print(mod.script())
```

三种常见创建方式：

```text
TVMScript      -> 手写 TensorIR
Tensor Expression -> 声明式生成 PrimFunc
Schedule 变换  -> 对已有 PrimFunc/IRModule 继续改写
```

## Build 与 Run

```python
lib = tvm.build(mod, target="llvm")

a_nd = tvm.runtime.tensor(a_np)
b_nd = tvm.runtime.tensor(b_np)
c_nd = tvm.runtime.tensor(c_np)
lib["main"](a_nd, b_nd, c_nd)
```

角色分别是：

| 对象 | 作用 |
|---|---|
| `IRModule` | 编译前的计算和调度 IR |
| `tvm.build` | 把 IR 降低并编译为目标代码 |
| `runtime.Module` | 可调用的编译产物 |
| `tvm.runtime.tensor` | 运行时张量容器 |

## Schedule

从调度视角看，矩阵乘法的关键循环是：

```text
i: 输出行
j: 输出列
k: reduction
```

取得 block 和循环：

```python
sch = s_tir.Schedule(mod)
block_c = sch.get_sblock("C")
i, j, k = sch.get_loops(block_c)
```

常见变换：

### Split

```python
i0, i1 = sch.split(i, factors=[None, 32])
```

把长度为 `N` 的循环拆成：

```text
i = i0 * 32 + i1
```

`None` 表示该维度自动推导。

### Reorder

```python
sch.reorder(i, j, k)
```

改变循环顺序。循环顺序会显著影响：

- 缓存复用
- 内存访问局部性
- 并行度

### Parallel

```python
sch.parallel(i)
```

提示该循环可以在 CPU 上并行执行。

### Unroll 与 Vectorize

```python
sch.unroll(k)
sch.vectorize(j)
```

- `unroll`: 展开小循环，减少循环控制开销并暴露指令级并行。
- `vectorize`: 把连续元素组成 SIMD 操作。

## 同一计算，不同实现

逻辑公式始终是：

```python
C[i, j] = sum_k A[i, k] * B[k, j]
```

但调度可以产生：

```text
普通三重循环
tiled matmul
CPU 并行 matmul
GPU thread-block matmul
shared-memory matmul
tensorized matmul
```

这就是 MLC 的核心分层思想：

```text
计算语义保持不变
执行结构通过 schedule 搜索和变换
```

## 关键 API

| API | 作用 |
|---|---|
| `te.placeholder` | 声明输入张量 |
| `te.reduce_axis` | 声明 reduction 轴 |
| `te.compute` | 定义输出张量的逐元素表达式 |
| `te.sum` | 表达 reduction |
| `te.create_prim_func` | 把 Tensor Expression 转成 TensorIR PrimFunc |
| `tvm.IRModule` | 管理一个或多个 IR 函数 |
| `tvm.build` | 针对 target 编译 IRModule |
| `s_tir.Schedule` | 获取和修改调度 |
| `get_sblock` | 获取 block handle |
| `get_loops` | 获取 block 周围的循环 |
| `split` | 拆分循环 |
| `reorder` | 改变循环顺序 |
| `parallel` | 标记 CPU 并行 |
| `unroll` | 展开循环 |
| `vectorize` | 生成向量操作 |

## 需要记住的结论

- Tensor Expression 描述数学计算，不直接描述存储访问和循环细节。
- PrimFunc/TensorIR 是后续调度和 lowering 的中间表示。
- Schedule 通过在 loop 和 block 上施加变换生成不同实现。
- 编译和运行分成 `IRModule -> tvm.build -> runtime.Module` 三步。
