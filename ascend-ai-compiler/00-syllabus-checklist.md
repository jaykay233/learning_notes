# 00 · 大纲校对清单（原课 → 本仓库课件）

原课以「通用 NPU + MLIR / tpu-mlir」为平台；本仓库全部改为 **昇腾 CANN** 表述，但 **知识点一条不漏**。

> 读法：与加厚后的讲义一致，每条对应课件均含「为什么 / 怎样 / 效果 / 用在哪」展开（见各章正文）。
>
> 图例：✅ 已有独立课件 · 🔧 算能专有名词已替换为昇腾等价物

---

## AI 编译器基础

| 原课条目 | 课件 | 状态 |
|---|---|---|
| SmallVector / StringRef / ArrayRef、BumpPtrAllocator、RTTI、CRTP、Visitor、TableGen | [01-cpp-foundations.md](./01-cpp-foundations.md) | ✅ |
| 计算图表示、算子融合、AOT vs JIT；框架图前后端 | [02-compiler-overview.md](./02-compiler-overview.md) | ✅ |
| Dialect / Op / Type、MLIR 语法、Builtin Dialect | [03-mlir-dialect-op-type.md](./03-mlir-dialect-op-type.md) | ✅ |
| Pass Manager、GreedyPatternRewrite | [04-mlir-pass-patterns.md](./04-mlir-pass-patterns.md) | ✅ |
| Lab1：冗余算子消除 Pass | [labs/lab1-redundant-op-pass.md](./labs/lab1-redundant-op-pass.md) | ✅ |

## 硬件与环境

| 原课条目 | 课件 | 状态 |
|---|---|---|
| NPU 结构、DMA、局部存储 | [05-ascend-npu-hardware.md](./05-ascend-npu-hardware.md) 🔧 昇腾 AI Core/UB 等 | ✅ |
| WSL2+Docker + 工具链 | [06-environment-setup.md](./06-environment-setup.md) 🔧 Linux/Docker+CANN | ✅ |
| Lab2：ResNet50 CPU/NPU | [labs/lab2-resnet50.md](./labs/lab2-resnet50.md) | ✅ |

## 工具链实战（原「MLIR 实战剖析」）

| 原课条目 | 课件 | 状态 |
|---|---|---|
| Transform / Deploy 工作流 | [07-toolchain-workflow.md](./07-toolchain-workflow.md) 🔧 ATC+AscendCL | ✅ |
| Frontend：TopDialect / converter | [08-frontend-parser.md](./08-frontend-parser.md) 🔧 Parser→GE 图 | ✅ |
| Backend：TpuDialect / lowering | [09-backend-ge.md](./09-backend-ge.md) 🔧 GE 优化编译→om | ✅ |
| LayerGroup 调度/内存 | [10-memory-schedule-aoe.md](./10-memory-schedule-aoe.md) 🔧 内存/stream/AOE | ✅ |
| Lab3：ONNX→MLIR→BModel | [labs/lab3-onnx-to-om.md](./labs/lab3-onnx-to-om.md) 🔧 ONNX→om | ✅ |

## 量化与性能优化

| 原课条目 | 课件 | 状态 |
|---|---|---|
| INT8 对称/非对称、per-channel、误差 | [11-int8-quant-math.md](./11-int8-quant-math.md) | ✅ |
| MinMax / KL / percentile | [12-calibration.md](./12-calibration.md) | ✅ |
| Lab4：精度对比与回退 | [labs/lab4-quant-accuracy.md](./labs/lab4-quant-accuracy.md) | ✅ |
| Profiling | [13-profiling.md](./13-profiling.md) | ✅ |

## CV 部署

| 原课条目 | 课件 | 状态 |
|---|---|---|
| YOLOv8：letterbox / NMS | [14-yolov8-deploy.md](./14-yolov8-deploy.md) | ✅ |
| Lab5：Stream 并行 | [labs/lab5-yolo-stream.md](./labs/lab5-yolo-stream.md) | ✅ |

## LLM

| 原课条目 | 课件 | 状态 |
|---|---|---|
| Attention / MHA / Softmax·Matmul | [15-attention-on-npu.md](./15-attention-on-npu.md) | ✅ |
| KV Cache 管理 | [16-kv-cache.md](./16-kv-cache.md) | ✅ |
| Qwen 量化部署 W8A8/W4A16 | [17-qwen-quant-deploy.md](./17-qwen-quant-deploy.md) | ✅ |
| Lab6：聊天机器人 | [labs/lab6-chatbot.md](./labs/lab6-chatbot.md) | ✅ |

## 求职与大纲其它块

| 原课条目 | 课件 | 状态 |
|---|---|---|
| 求职辅导 / 岗位方向 | [18-interview-career.md](./18-interview-career.md) | ✅ |
| 课程框架图（Frontend/Backend 分块说明） | [02](./02-compiler-overview.md) §框架图 | ✅ |
| 适合学员 / 培养目标 | [18](./18-interview-career.md) + 本清单 | ✅ |

---

## 名词替换速查

| 原课 | 本课 |
|---|---|
| TopDialect / converter | Framework **Parser** → GE Graph |
| TpuDialect / TopToTpu | **GE** 图优化 + 编译 |
| LayerGroup | 内存分配 / stream / **AOE** |
| BModel | **`.om`** |
| tpu-mlir 板端 | **AscendCL** / msame |
| 算能 NPU 课板 | **Atlas / 昇腾云** |
