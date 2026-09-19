# MLC / TVM 学习笔记

本目录整理 MLC 课程 notebook 2 到 8 的学习内容，以及对话中围绕
TensorIR、调度原语和 tensorization 展开的补充问答。

## 学习环境

- Python: 3.11
- 虚拟环境: `uv` 创建的 `mlc`
- TVM: `0.26.dev246`
- 本地机器: Apple M5 Pro
- 默认验证 target: `llvm`
- CPU 端示例可以直接运行；CUDA 示例在无 CUDA runtime 的环境下只能检查生成的代码

当前 TVM 开发版与旧版 notebook 有以下主要 API 差异：

| 旧版 API | 当前版本 |
|---|---|
| `@tvm.script.ir_module` | `@I.ir_module` |
| `from tvm.script import tir as T` | `from tvm.script import tirx as T` |
| `@T.prim_func` | `@T.prim_func(s_tir=True)` |
| `T.block` | `T.sblock` |
| `"tir.noalias"` | `"tirx.noalias"` |
| `tvm.tir.Schedule` | `tvm.s_tir.Schedule` |
| `get_block` | `get_sblock` |
| `tvm.tir.TensorIntrin` | `tvm.s_tir.TensorIntrin` |
| `tvm.nd.array` | `tvm.runtime.tensor` |

## 文档索引

| 文档 | 对应内容 | 核心主题 |
|---|---|---|
| [01-tensor-program-abstraction.md](01-tensor-program-abstraction.md) | notebook 2 | Tensor Expression、IRModule、Schedule、build/run |
| [02-tensorir-case-study.md](02-tensorir-case-study.md) | notebook 3 | Buffer、循环、Block、axis、函数属性、变换 |
| [03-end-to-end-model.md](03-end-to-end-model.md) | notebook 4 | Relax 计算图、`call_tir`、DPS、端到端执行 |
| [04-automatic-optimization.md](04-automatic-optimization.md) | notebook 5 | 随机调度、Trace、搜索、AutoScheduler |
| [05-framework-integration.md](05-framework-integration.md) | notebook 6 | TorchFX、PyTorch 图、BlockBuilder、算子映射 |
| [06-gpu-hardware-part1.md](06-gpu-hardware-part1.md) | notebook 7 | GPU 层级、thread binding、shared/local memory、Matmul |
| [07-gpu-hardware-part2-tensorization.md](07-gpu-hardware-part2-tensorization.md) | notebook 8 | specialized hardware、blockize、TensorIntrin、tensorize |
| [08-tensorization-faq.md](08-tensorization-faq.md) | 课程问答 | `T.sblock`、`blockize`、`tensorize`、`match_buffer`、LLVM pragma |

## 推荐的复习顺序

1. 先读 01 和 02，建立“计算怎么写”和“调度怎么改”的基础。
2. 再读 03，理解 TensorIR 如何嵌入完整模型并被 Relax VM 执行。
3. 读 04 和 05，掌握自动调度与机器学习框架前端。
4. 读 06，理解 GPU 的线程层级、内存层级和 Matmul 调度。
5. 最后读 07 和 08，理解 TensorIntrin 与 tensorization。

## 贯穿整套课程的主线

```text
数学计算
  -> Tensor Expression / TVMScript
  -> TensorIR PrimFunc
  -> Schedule 变换
  -> 面向 target 的 lowering
  -> 可执行 Module
```

硬件加速部分进一步把主线扩展为：

```text
普通标量循环
  -> blockize
  -> tensorized block
  -> TensorIntrin + tensorize
  -> 硬件指令或 extern micro-kernel
```
