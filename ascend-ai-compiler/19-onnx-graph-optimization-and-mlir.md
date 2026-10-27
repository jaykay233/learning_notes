# 19 · ONNX 图优化与 MLIR：Pass 应该写在哪一层

```text
本次讲解位置
章节：ascend-ai-compiler
小节：Frontend 图优化与 MLIR lowering 的边界
知识点：自定义 ONNX graph pass、ONNX 与 MLIR 的关系、是否需要 round-trip
上次：08 Frontend Parser 把框架模型变成统一的中间图
下次：Lab3 跟踪 ONNX → ATC → om 的完整阶段
PTX：不涉及，本讲停留在图 IR 层
```

## 为什么现在讲这个

前面已经知道 Parser 会把 ONNX 导入中间图，也知道 MLIR 用 Pass 和 Pattern Rewrite 做优化。接下来要解决一个实际选择：自定义优化应该直接改 `.onnx`，还是先转到 MLIR，再优化到硬件代码。

这个选择会直接影响优化能保留多少。只做高层融合时，转到 MLIR 再转回 ONNX 往往增加复杂度；如果已经 lowering 到 `linalg`、`memref` 或 GPU dialect，再试图无损转回 ONNX，则会遇到表达能力不足和语义丢失。先把层级边界分清，后面的 Pass 才不会写错位置。

## 1. 先把三个东西分开

ONNX、ONNX Runtime 和 MLIR 不是同一个层次：

| 名称 | 它是什么 | 它负责什么 |
|---|---|---|
| ONNX | Protobuf 模型格式 | 定义 Op、输入输出、属性、opset 和权重 |
| ONNX Runtime | 推理运行时 | 导入 ONNX，做图优化并调用 Execution Provider |
| MLIR | 编译器基础设施 | 用 Dialect、Operation、Pass、PatternRewrite 做多层 lowering |

ONNX 中也会使用“IR”这个说法，它表示 ONNX 自己的计算图表示，不是 MLIR 的 IR。

```text
PyTorch / TensorFlow
        ↓ 导出
ONNX 图（Protobuf）
        ↓ importer
MLIR 的 onnx dialect
        ↓ lowering
linalg / tensor / arith / scf / memref
        ↓
LLVM IR、GPU 代码或 NPU 后端格式
```

ONNX 不基于 MLIR。`onnx-mlir`、IREE、torch-mlir 等项目会把 ONNX 或其他前端模型导入 MLIR，但这一步是可选前端，不是 ONNX 格式的组成条件。

## 2. ONNX 可以写自己的图优化

可以。直接遍历和修改 `ModelProto.graph` 就是自定义 graph pass。常见的变换包括：

- 消除 `Identity`、冗余 `Cast` 和死节点。
- 做常量折叠、CSE 和 shape inference。
- 匹配 `MatMul + Add`、LayerNorm 等子图并替换。
- 把 `Transpose(weight) + MatMul` 改成预转置权重。
- 改写 QDQ 量化模式。
- 把一段算子子图替换成目标后端支持的融合算子。

现成工具可以先试：

| 工具 | 适合做什么 |
|---|---|
| `onnxoptimizer` | 使用已有的 ONNX pass，如消除 Identity、死节点 |
| `onnxsim` | 常量折叠、冗余节点清理、形状简化 |
| ONNX GraphSurgeon | 按节点和 Tensor 连接做图手术 |
| ONNX Script / ONNX IR API | 用 Python API 构造和重写图 |
| ONNX Runtime optimizer | Runtime 自己的融合、QDQ 和 EP 优化 |

这些工具解决的是“如何改图”，第 3 节用一段完整代码说明 pass 的核心动作。

## 3. 完整例子：消除图内部的 Identity

下面的代码构造：

```text
X → MatMul → Identity → Relu → Y
```

然后删除图内部的 `Identity`，改写消费者输入，检查模型并比较输出。创建两个独立模型对象是为了避免 pass 原地修改后影响 baseline。

文件名为 `onnx_graph_pass_demo.py`：

```python
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from onnx.reference import ReferenceEvaluator


def build_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 3])
    y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 2])
    w = numpy_helper.from_array(
        np.array(
            [[1.0, -2.0], [0.5, 1.5], [2.0, -1.0]],
            dtype=np.float32,
        ),
        name="W",
    )
    nodes = [
        helper.make_node("MatMul", ["X", "W"], ["mm"]),
        helper.make_node("Identity", ["mm"], ["same"]),
        helper.make_node("Relu", ["same"], ["Y"]),
    ]
    graph = helper.make_graph(nodes, "identity_demo", [x], [y], [w])
    model = helper.make_model(
        graph,
        producer_name="learning_notes",
        opset_imports=[helper.make_opsetid("", 18)],
    )
    onnx.checker.check_model(model)
    return model


def eliminate_internal_identity(model: onnx.ModelProto) -> onnx.ModelProto:
    graph = model.graph
    output_names = {value.name for value in graph.output}
    aliases: dict[str, str] = {}
    dead_node_ids: set[int] = set()

    def resolve(name: str) -> str:
        while name in aliases:
            name = aliases[name]
        return name

    for node_id, node in enumerate(graph.node):
        if node.op_type != "Identity" or len(node.input) != 1:
            continue

        source = resolve(node.input[0])
        target = node.output[0]

        # 这个教学 pass 只处理内部值。图输出 Identity 要额外改写 graph.output。
        if target in output_names:
            continue

        aliases[target] = source
        dead_node_ids.add(node_id)

    kept_nodes = []
    for node_id, node in enumerate(graph.node):
        if node_id in dead_node_ids:
            continue

        rewritten_inputs = [resolve(name) for name in node.input]
        del node.input[:]
        node.input.extend(rewritten_inputs)
        kept_nodes.append(node)

    del graph.node[:]
    graph.node.extend(kept_nodes)
    onnx.checker.check_model(model)
    return model


original = build_model()
optimized = eliminate_internal_identity(
    onnx.ModelProto.FromString(original.SerializeToString())
)

x = np.array(
    [[1.0, 2.0, -1.0], [0.0, -0.5, 3.0]],
    dtype=np.float32,
)
before = ReferenceEvaluator(original).run(None, {"X": x})[0]
after = ReferenceEvaluator(optimized).run(None, {"X": x})[0]

print("before:", [node.op_type for node in original.graph.node])
print("after: ", [node.op_type for node in optimized.graph.node])
print("max_abs_diff:", float(np.max(np.abs(before - after))))
```

运行：

```bash
python3 -m pip install onnx numpy
python3 onnx_graph_pass_demo.py
```

预期输出：

```text
before: ['MatMul', 'Identity', 'Relu']
after:  ['MatMul', 'Relu']
max_abs_diff: 0.0
```

### 3.1 按阶段看这段代码

1. `build_model()` 创建 `MatMul → Identity → Relu`。`Identity` 的输出名是 `same`，也是 `Relu` 的输入名。
2. `aliases["same"] = "mm"` 表示以后看到 `same`，就沿着别名找到真正的生产者输出 `mm`。
3. 遍历剩余节点时，把 `Relu` 的输入从 `same` 改成 `mm`。这一步不能省，否则删除 `Identity` 后会出现悬空引用。
4. 重建 `graph.node`，移除已经失效的 Identity。
5. `onnx.checker.check_model()` 检查节点、类型和连接是否合法。
6. `ReferenceEvaluator` 对同一输入 `x` 分别运行优化前后模型，最大绝对误差应为零。

输入为：

```text
X = [[ 1.0,  2.0, -1.0],
     [ 0.0, -0.5,  3.0]]
```

`X @ W` 得到：

```text
MM = [[0.0,  2.0],
      [5.75, -3.75]]
```

`Relu(MM)` 得到：

```text
Y = [[0.0, 2.0],
     [5.75, 0.0]]
```

Identity 不改变数值，所以删除它之后，输出逐元素不变。这个例子没有线程、shared memory、同步或硬件资源访问；它只在编译期修改图结构。

### 3.2 为什么不能只删节点

如果直接删除 Identity，却没有把消费者输入从 `same` 改为 `mm`，Relu 仍会引用一个不存在的值，`onnx.checker` 会报图连接错误。

如果 Identity 的输出同时是 `graph.output`，还要同步处理输出名，否则图输出将引用已删除的节点。别名链、图输入输出、initializer 和 `value_info` 都要一起检查。

## 4. 它确实很像 AI 编译器的 Pass

ONNX 自定义图优化就是 AI 编译器前端的一类 pass：

| ONNX graph pass | 对应编译器概念 |
|---|---|
| 删除 Identity、死节点 | DCE |
| 两个常量先算掉 | Constant Folding |
| 相同子图只算一次 | CSE |
| `MatMul + Add` 替换为 `Gemm` | Pattern Rewrite / Fusion |
| 权重预转置 | Layout Optimization |
| QDQ 重组 | Quantization Pass |
| 子图替换为后端算子 | Legalization / Lowering |

但 ONNX 层通常没有完整的 PassManager、AnalysisManager、Dialect Conversion 和多层合法性约束。很多 ONNX 优化工具本质上是在 GraphProto 上做节点级修改；MLIR 则把不同阶段拆成不同 Dialect，并为每一层提供 verifier 和 rewrite 基础设施。

```text
ONNX graph
  → 高层算子图优化
  → Tensor / Linalg IR
  → tiling、fusion、vectorization
  → Memref / GPU / NPU IR
  → 内存规划、软件流水、代码生成
```

ONNX pass 决定“图里有哪些算子、它们怎样连接”；MLIR 后续 pass 继续决定“算子怎样切、怎样排、怎样放到硬件上”。

## 5. 要不要转成 MLIR 再转回 ONNX

通常不建议把它作为默认闭环。先看优化目标和最终产物：

### 5.1 只做高层图重写：留在 ONNX

```text
ONNX
  → shape inference / constant folding / DCE / fusion
  → 优化后的 ONNX
  → ONNX Runtime / 目标后端
```

适合删除节点、子图融合、QDQ 重写和算子替换。模型仍可被其他 ONNX Runtime 或后端加载。

### 5.2 需要编译级优化：进入 MLIR 后继续 lowering

```text
ONNX
  → MLIR onnx dialect
  → tensor/linalg
  → tiling + fusion + vectorization
  → memref / GPU / NPU dialect
  → LLVM 或目标机器码
```

适合 tiling、循环融合、shared memory 规划、software pipeline、线程映射和自定义硬件 codegen。最终产物通常是可执行库、目标代码或特定 runtime 格式，不需要再回到 ONNX。

### 5.3 必须输出 ONNX：只做可导出范围内的优化

可以在 `onnx` dialect 仍能表达完整语义时做高层 rewrite，然后导出 ONNX：

```text
ONNX → MLIR onnx dialect → 高层 rewrite → 导出 ONNX
```

不要等 lowering 到 `linalg`、`memref`、`scf`、`vector`、GPU dialect 后再转回 ONNX。这些层中的 tile 大小、buffer 分配、线程映射和调度策略，ONNX 没有通用表达方式。强行回写通常只能变成目标后端不认识的 custom op，或者丢失原优化。

## 6. 推荐的实际工作流

如果目标后端仍接收 ONNX：

1. 固定 opset 和输入 shape。
2. 运行 shape inference。
3. 做常量折叠和 DCE。
4. 匹配业务子图，做目标后端支持的融合。
5. 如果涉及量化，保持 Q/DQ 的 scale、zero-point 和数据类型语义。
6. 执行 `onnx.checker.check_model()`。
7. 用相同输入比较数值误差。
8. 在目标 Runtime 上测延迟、显存和带宽。

不要只看“模型能加载”。一个 pass 生成了后端不认识或性能更差的算子时，ONNX 仍然可能完全合法。

## 7. 常见错误

**把 ONNX 当成 MLIR。**

ONNX 是模型格式，MLIR 是编译器基础设施。两者可以连接，但没有继承关系。

**lowering 到底层后再导回 ONNX。**

tiling、memref、线程映射和 pipeline 往往没有 ONNX 等价语义，round-trip 会丢失优化或产生不兼容 custom op。

**先改图，后补节点连接。**

修改节点输出后必须同步所有消费者，否则 checker 会看到悬空值。

**只检查图合法，不检查数值。**

量化、常量折叠和算子融合都可能改变精度语义，必须跑数值回归。

**认为 MLIR 一定比 ONNX 优化更好。**

如果后端最终只接受 ONNX，或者 Runtime 已经能完成同样的高层融合，额外 round-trip 只会增加复杂度和兼容风险。

## 下一知识点

[Lab3：ONNX → om](./labs/lab3-onnx-to-om.md) 会把这里看到的图优化放回真实 ATC 管道，用日志区分 Parser、图准备、优化和编译阶段。

## 8. 自检

**1. ONNX 是基于 MLIR 的吗？**

不是。ONNX 是模型交换格式，MLIR 是编译器基础设施。`onnx-mlir` 等工具可以把 ONNX 导入 MLIR，这是工具链选择，不是格式依赖。

**2. ONNX 自定义图优化为什么算 Pass？**

它读取计算图，匹配节点或子图，改写连接和算子，再检查合法性。这与编译器的常量折叠、DCE、Pattern Rewrite 和 Fusion pass 属于同一类操作，只是停留在高层 graph IR。

**3. 什么时候适合 ONNX → MLIR → ONNX？**

只在 MLIR 中仍使用可导出的高层 `onnx` dialect、且所有优化都能由 ONNX 表达时适合。已经 lowering 到 `linalg`、`memref` 或硬件 dialect 后，不应期待通用、无损地导回 ONNX。

**4. 删除 Identity 时最少要改哪两处？**

第一，删除 Identity 节点；第二，把所有消费者对 Identity 输出的引用改成它真正输入的生产者值。若 Identity 的输出是 graph output，还要同步更新图输出。

**5. 为什么 pass 前后都要做数值和执行验证？**

`onnx.checker` 只能证明图结构合法，不能证明量化、constant folding 或融合后的语义完全一致；最终性能也必须在目标 Runtime 和硬件上测量，ONNX 图合法不等于 kernel 更快。
