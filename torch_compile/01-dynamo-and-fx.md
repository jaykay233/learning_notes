# TorchDynamo 调用过程，以及和 Torch FX 的关系

> 对应 `torch.compile` / `torch._dynamo` 前端。示例来自 `@dynamo.optimize(my_compiler)` 自定义 backend 的最小写法。

---

## 1. 最小例子在干什么

```python
from typing import List
import torch
import torch._dynamo as dynamo

def my_compiler(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
    print("my_compiler() called with FX graph:")
    gm.graph.print_tabular()
    return gm.forward  # 真 backend 会在这里换成编译后的可执行物

@dynamo.optimize(my_compiler)
def foo(a, b):
    x = a / (torch.abs(a) + 1)
    if b.sum() < 0:
        b = b * -1
    return x * b

foo(torch.randn(10), torch.randn(10))
```

- `@dynamo.optimize(my_compiler)`：给 `foo` 挂上「先抓图、再交 backend」的入口。  
- `my_compiler`：自定义 backend；收到的 `gm` 已经是 **FX `GraphModule`**。  
- `foo` 里的 `if b.sum() < 0`：数据依赖控制流，容易触发 **graph break**。

---

## 2. 调用过程（时间线）

```text
1. @dynamo.optimize(my_compiler)
   └─ 包装 foo：调用时先走 Dynamo，不直接跑原函数体

2. foo(randn, randn) 第一次进来
   └─ Dynamo 接管（拦截入口 / 相关字节码）

3. 字节码级追踪
   └─ 遇到可图化的 torch 算子 → 记进 FX Graph
   └─ 遇到不好进图的东西 → graph break

4. 对本例 foo，大致在 if 处断开：

   段 A（可图化）:
     x = a / (abs(a) + 1)
     → GraphModule gm1
     → my_compiler(gm1, example_inputs)
     → 得到 callable（这里是 gm.forward）并执行 → 得到 x

   缝（eager / 解释器）:
     cond = (b.sum() < 0)   # 依赖真实数据
     if cond: b = b * -1

   段 B（可图化）:
     return x * b
     → 再 capture → my_compiler(gm2, ...) → 执行

5. 之后若 guard 仍满足（shape / dtype / 设备等）
   └─ 可走缓存，不必每次重 trace
   └─ guard 失败 → 再编译一条路径（或再 break）
```

因此 `my_compiler() called with FX graph:` **可能打印多次**——每捕获到一段可编译图，就调一次 backend。

---

## 3. 三层角色（对照编译课）

| 角色 | 在这例子里 | 类比 |
|---|---|---|
| Frontend | Dynamo：Python/字节码 → FX | Parser / 抓中间图 |
| 交接物 | `GraphModule` + `example_inputs` | 中间 IR |
| Backend | `my_compiler`（真场景常是 Inductor） | GE / lowering；`return` 的 callable ≈ 编好的核 |

`print_tabular()` 是在交接处把 FX IR 摊开看。

和 CUDA Graph 世界的直觉也通：Dynamo 的 graph break ≈「能抓的抓成段，不能抓的回 eager」；SGLang Breakable Graph 是在 GPU capture 层做类似的事。

---

## 4. 和 Torch FX：很像，但不是同一个东西

**像是故意的**：Dynamo 抓出来的中间结果就是 FX 图。

| | **Torch FX** | **TorchDynamo** |
|---|---|---|
| 是什么 | **IR + 改图 API**（`Graph` / `Node` / `GraphModule`、Pass、`print_tabular`） | **前端捕获器**：从 Python/字节码自动抓出可编译段 |
| 产出 | 你手里的那张图 | 通常也是 FX `GraphModule`，再交给 backend |
| 怎么得到图 | 经典：`symbolic_trace`（要 trace 友好）；或手建/改图 | 跑起来时按字节码抓，更能打真实代码 |
| 控制流 | 纯 `symbolic_trace` 对数据依赖 `if` 很吃力 | 可 graph break，段与段之间仍是 FX |

链路：

```text
Dynamo（怎么抓） → FX GraphModule（抓成什么样） → my_compiler / Inductor（怎么降）
```

一句话：

- **FX**：图长什么样、怎么变图（中端 IR）。  
- **Dynamo**：怎么从 eager Python **自动弄出** 这些 FX 图（前端；比老式 symbolic trace 更能打）。  

类比：MLIR 的 Dialect/IR vs Parser 怎么进到这张 IR。

---

## 5. TorchScript / FX / LazyTensor / Dynamo 对照

PyTorch 里「把 eager 代码变成图再优化」试过好几代入口。四者常被混谈，按 **在哪抓、抓成什么、能不能改原代码、控制流怎么处理** 区分：

| | **TorchScript** | **Torch FX** | **Lazy Tensor** | **TorchDynamo** |
|---|---|---|---|---|
| **一句话** | 把模型变成可序列化的 Script 模块 | 中端 **图 IR + 变换 API** | 在 **ATen dispatcher** 推迟执行、攒图 | 在 **Python 帧/字节码** 抓图，交给 backend |
| **典型 API** | `torch.jit.trace` / `torch.jit.script` | `symbolic_trace`、`GraphModule`、自定义 Pass | PyTorch/XLA 等路径上的 lazy 执行 | `torch.compile` / `dynamo.optimize` |
| **抓图位置** | C++ dispatcher（trace）或 Script 语言子集（script） | Python 层用 Proxy 跑一遍 | C++ dispatcher：每次 op 先记账不马上算 | CPython Frame Evaluation：改字节码执行 |
| **IR / 产物** | TorchScript IR / `.pt` 一类 | FX Graph（也可再 lowering） | Lazy 图 → 常交给 XLA 等编译 | 多为 **FX GraphModule** → Inductor 等 |
| **改代码？** | script 常要大改、加注解；trace 少改但不安全 | `symbolic_trace` 要求代码「可静默跑通」 | 目标是少改，但生态/开销因路径而异 | **尽量零改** 吃下真实 PyTorch 代码 |
| **数据依赖 `if`** | trace：**只录走到的支**，可能静默错；script：能表达，但语言子集受限 | 经典 symbolic_trace **基本搞不定** | 可整段攒图，但图一变就要重编译/开销大 | **graph break**：回 Python 判分支，再抓下一段（正确优先） |
| **抓图频率** | 一次 trace/script（导出导向） | 一次 trace（或手改图） | 历史上常 **每 iteration 再攒**（开销来源之一） | 按 guard：同 shape 等可缓存；变了再编 |
| **和另外几者关系** | 老部署/部分厂商仍吃 Script IR | Dynamo **产出** FX；FX 自己不是「怎么从任意 Python 抓」的完整答案 | 和 Dynamo 目标像（少改代码动态抓图），手法在 dispatcher；现主推路径多是 Dynamo | PT2 默认前端；可与 AOTAutograd 等组合 |

### 5.1 各自最容易踩的坑

- **TorchScript**  
  - `trace`：控制流/动态 shape → **静默错**（只留下实际走过的路径）。  
  - `script`：不是完整 Python；一碰不支持的语法就整段失败，大模型「torchscript 化」成本高。

- **Torch FX（symbolic_trace）**  
  - 适合「已经比较静态、可 Proxy 跑」的模块变换（量化插桩、图改写）。  
  - 不是 `torch.compile` 的完整前端；复杂控制流别指望它单独吃下。

- **Lazy Tensor**  
  - 在 dispatcher 记账，语义贴近 eager aten 流，适合接 XLA 等。  
  - 每步攒图 / 图 hash 一变就重编译 → 曾出现相对 eager 明显变慢；后来有和 Dynamo 的混合（用 Dynamo 决定何时需要再抓，减轻每 iter 开销）。

- **TorchDynamo**  
  - 正确性靠 **guard + graph break**，不是「一张整图或失败」。  
  - break 多 → 优化空间变碎；要极致整图可用 `fullgraph=True`（遇 break 直接报错，逼你改代码）。

### 5.2 怎么记（选型直觉）

```text
要导出/老 TorchScript 生态     → TorchScript（清楚风险）
要改图、插 Pass、看 tabular    → FX（IR 层）
要接 XLA/TPU 一类 lazy 栈     → Lazy Tensor（或现成 XLA 集成）
要加速且尽量不改训练/推理代码 → Dynamo / torch.compile（默认选这个）
```

产品拼装仍是：

```text
用户代码
  → Dynamo（抓）
  → FX 图（表示）
  → Inductor / 其它 backend（降）
  → （可选）CUDA Graph 等
```

TorchScript / LazyTensor 是 **另一条或上一代的抓图哲学**，不是 FX 的别名，也不等于 Dynamo。

---

## 6. 和 `torch.compile` 的关系（定位）

常见产品入口：

```text
torch.compile(fn, backend="inductor", ...)
  ≈ Dynamo 抓 FX（可多段）
  + Inductor（或其它 backend）降低 / 生成核
  + （可选）CUDA Graph Trees 等运行时包装
```

本页钉住 **Dynamo 调用过程**、**FX 交接**，以及 **Script / FX / Lazy / Dynamo 分工**；  
AOTAutograd 详见 [02-aot-autograd.md](./02-aot-autograd.md)；Inductor / CUDAGraph Trees 见 [cuda-graph/](../cuda-graph/)。
