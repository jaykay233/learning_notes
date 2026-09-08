# 昇腾版 AI 编译器课程讲义

> 按原「AI 编译器」大纲 **逐条覆盖**，硬件/工具链从算能 tpu-mlir 改为 **昇腾 CANN（ATC / GE / om）**。
> 校对表见 [00-syllabus-checklist.md](./00-syllabus-checklist.md)。
>
> **讲义深度约定**：各章按「为什么 → 怎样工作/实现（概念）→ 效果 → 用在哪 → 与前后章怎么连」展开（[01](./01-cpp-foundations.md) 为范例；**02–18 与 labs 已按同一标准加厚**）。

## 学习顺序（与大纲一致）

| 阶段 | 讲义 / Lab |
|---|---|
| **AI 编译器基础** | [01](./01-cpp-foundations.md) → [02](./02-compiler-overview.md) → [03](./03-mlir-dialect-op-type.md) → [04](./04-mlir-pass-patterns.md) → [Lab1](./labs/lab1-redundant-op-pass.md) |
| **硬件与环境** | [05](./05-ascend-npu-hardware.md) → [06](./06-environment-setup.md) → [Lab2](./labs/lab2-resnet50.md) |
| **CANN 实战** | [07](./07-toolchain-workflow.md) → [08](./08-frontend-parser.md) → [09](./09-backend-ge.md) → [10](./10-memory-schedule-aoe.md) → [Lab3](./labs/lab3-onnx-to-om.md) |
| **量化与性能** | [11](./11-int8-quant-math.md) → [12](./12-calibration.md) → [Lab4](./labs/lab4-quant-accuracy.md) → [13](./13-profiling.md) |
| **CV 部署** | [14](./14-yolov8-deploy.md) → [Lab5](./labs/lab5-yolo-stream.md) |
| **LLM** | [15](./15-attention-on-npu.md) → [16](./16-kv-cache.md) → [17](./17-qwen-quant-deploy.md) → [Lab6](./labs/lab6-chatbot.md) |
| **求职** | [18](./18-interview-career.md) |

## 文件树

```
ascend-ai-compiler/
├── 00-syllabus-checklist.md
├── 01-cpp-foundations.md
├── 02-compiler-overview.md
├── 03-mlir-dialect-op-type.md
├── 04-mlir-pass-patterns.md
├── 05-ascend-npu-hardware.md
├── 06-environment-setup.md
├── 07-toolchain-workflow.md
├── 08-frontend-parser.md
├── 09-backend-ge.md
├── 10-memory-schedule-aoe.md
├── 11-int8-quant-math.md
├── 12-calibration.md
├── 13-profiling.md
├── 14-yolov8-deploy.md
├── 15-attention-on-npu.md
├── 16-kv-cache.md
├── 17-qwen-quant-deploy.md
├── 18-interview-career.md
└── labs/
    ├── lab1-redundant-op-pass.md
    ├── lab2-resnet50.md
    ├── lab3-onnx-to-om.md
    ├── lab4-quant-accuracy.md
    ├── lab5-yolo-stream.md
    └── lab6-chatbot.md
```

## 与 `models/` 的关系

`models/` 讲模型结构（Qwen/GDN/QSA…）；本目录讲 **编译与上板部署**。LLM 阶段可交叉阅读。
