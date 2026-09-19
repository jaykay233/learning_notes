# 03 End-to-End Model Execution

对应 notebook:
`4_Build_End_to_End_Model.ipynb`

## 本课目标

把单个 TensorIR 算子组合成一个可以执行完整模型的 `IRModule`，并用 Relax VM 运行。

整体结构：

```text
输入张量
  -> Relax 计算图
  -> call_tir 调用 TensorIR 算子
  -> 调度和 build
  -> Relax VirtualMachine 执行
  -> 输出张量
```

## 从算子视角到图视角

一个模型可以表示成多个算子组成的计算图，例如：

```text
X
  -> Matmul
  -> Add
  -> ReLU
  -> Matmul
  -> Output
```

每个节点有两种身份：

- 高层数据流操作，例如 `matmul`、`relu`
- 可以进一步 lower 成 TensorIR PrimFunc 的可调度算子

Relax 负责图级控制流和数据流，TensorIR 负责单个张量计算块的循环级实现。

## 用 TVMScript 构造 IRModule

典型结构：

```python
@I.ir_module
class Model:
    @R.function
    def main(x: R.Tensor((1, 784), "float32")):
        with R.dataflow():
            ...
            R.output(output)
        return output
```

`R.function` 是图级函数，`R.dataflow` 表示数据流区域。

数据流区域中的表达式主要描述“生产者和消费者关系”，而不是底层循环。

## `call_tir`

`call_tir` 用于在 Relax 图中调用一个 TensorIR 函数：

```python
output = R.call_tir(
    matmul_func,
    (x, weight),
    out_sinfo=R.Tensor((1, 128), "float32"),
)
```

三个关键部分：

| 参数 | 作用 |
|---|---|
| 被调用的 PrimFunc | 具体 TensorIR 实现 |
| 输入表达式 | Relax 层数据流输入 |
| `out_sinfo` | 输出形状和 dtype |

`call_tir` 是 Relax 和 TensorIR 之间的桥。

## DPS：Destination Passing Style

DPS 表示 Destination Passing Style，即“目标地址传递风格”。

普通函数式写法可能是：

```python
output = matmul(input_a, input_b)
```

DPS 写法把输出 Buffer 也作为参数传进去：

```python
matmul(input_a, input_b, output)
```

函数通过写 `output` 返回结果，而不是在语言层返回一个新值。TensorIR 的 PrimFunc 通常使用这种形式：

```python
@T.prim_func(s_tir=True)
def matmul(
    A: T.Buffer(...),
    B: T.Buffer(...),
    C: T.Buffer(...),
) -> None:
    ...
```

`C` 是 destination。函数返回 `None` 并不矛盾，因为结果已经写到了 `C` 中。

DPS 的优点：

- 明确表达输出内存
- 便于复用 Buffer 和内存规划
- 容易连接外部库
- 方便图级编译做内存分配和生命周期分析

## 数据流 Block

数据流区域中的节点可以并排描述：

```text
x -> linear1 -> relu -> linear2 -> output
```

Relax 通过数据流 block 表达节点依赖，并在后续 pass 中完成：

- 常量折叠
- 算子融合
- 内存规划
- 执行顺序安排

## `call_dps_packed` 与外部库

某些算子不注册为 TensorIR，而是调用已有运行时库：

```python
y = R.call_dps_packed(
    "my_runtime_function",
    (x, weight, y),
    out_sinfo=R.Tensor(...),
)
```

名字 `"my_runtime_function"` 需要在运行环境中由 `tvm.register_func` 提供：

```python
@tvm.register_func("my_runtime_function")
def my_runtime_function(...):
    ...
```

这使 Relax 图可以混合：

```text
Relax 图和 TensorIR 算子
Relax 图和已有 C++/CUDA/框架函数
Relax 图和厂商库
```

## Build 与 VirtualMachine

构造完 Relax IRModule 后：

```python
executable = relax.build(module, target="llvm")
vm = relax.VirtualMachine(executable, tvm.cpu())
output = vm["main"](input_tensor)
```

流程是：

```text
IRModule
  -> relax.build
  -> executable
  -> VirtualMachine
  -> 运行函数
```

TensorIR 部分会被降低成目标代码，Relax 部分负责图级调度、内存和运行时调用。

## 绑定参数

模型参数可以作为：

- `relax.const`
- 函数参数
- 外部运行时对象
- 状态交给 VM 管理的变量

绑定参数的目标是让 graph 中的常量数据拥有明确的存储和生命周期，并能被后续的布局转换、设备和内存规划正确处理。

## 混合 TensorIR 与已有库

完整模型不一定全部由 TensorIR 实现。实际编译流程通常是混合的：

```text
可优化的部分 -> TensorIR + schedule
已有高性能库 -> call_dps_packed
无法下沉的节点 -> Relax runtime function
```

这也是工业 MLC 系统常见的工作方式。

## 关键 API

| API | 作用 |
|---|---|
| `R.function` | 图级函数 |
| `R.dataflow` | 数据流区域 |
| `R.Tensor` | 图级张量类型 |
| `R.call_tir` | 从 Relax 调用 TensorIR |
| `R.call_dps_packed` | 调用 packed runtime 函数 |
| `R.output` | 指定数据流块的输出 |
| `relax.build` | 编译 Relax IRModule |
| `relax.VirtualMachine` | 执行编译后的模块 |
| `relax.const` | 图内常量 |
| `tvm.register_func` | 注册运行时函数 |

## 本课结论

端到端模型执行不是“一个巨大的 TensorIR 循环”，而是两层 IR 协同：

```text
Relax：图级数据流、控制、内存和库调用
TensorIR：算子内部循环、依赖和调度
```

DPS 是两层之间重要的连接约定：显式传入输出，让数据流、内存规划和外部库调用都能保持清晰。
