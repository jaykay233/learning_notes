# 算子开发

集中记录高性能算子（CUDA / Triton / TileLang 等）从问题建模到上板验证的知识点。
与 [`ascendc/`](../ascendc/)（AscendC CPU 孪生工程环境）、[`modern-gpu-programming-for-mlsys/`](../modern-gpu-programming-for-mlsys/)（GPU layout / Tensor Core / GEMM）互补：那边偏硬件与编程模型，这边偏**算子工程闭环**。

## 目录

```text
operators/
├── README.md
└── code/
    ├── cuda/       # 原生 CUDA / WMMA
    ├── triton/     # OpenAI Triton
    └── tilelang/   # TileLang DSL
```

| 后端 | FP16 原地加 `A += B` | BF16 `C = A @ B.T` |
|---|---|---|
| CUDA | [code/cuda/fp16_inplace_add.cu](code/cuda/fp16_inplace_add.cu) | [code/cuda/bf16_gemm_at_bt.cu](code/cuda/bf16_gemm_at_bt.cu) |
| Triton | [code/triton/fp16_inplace_add.py](code/triton/fp16_inplace_add.py) | [code/triton/bf16_gemm_at_bt.py](code/triton/bf16_gemm_at_bt.py) |
| TileLang | [code/tilelang/fp16_inplace_add.py](code/tilelang/fp16_inplace_add.py) | [code/tilelang/bf16_gemm_at_bt.py](code/tilelang/bf16_gemm_at_bt.py) |

> 评测平台通常只收 CUDA 的 `extern "C" run_kernel`；Triton / TileLang 便于本地对照与学习。

## 主题路线

1. **算子是什么**：host / device 边界、dispatch、kernel launch、输入输出约定。
2. **正确性闭环**：参考实现、数值误差、边界 shape、CPU 孪生 / 仿真 vs 真机。
3. **性能模型**：算力 / 带宽 bound、访存合并、占用率、流水与双缓冲。
4. **典型算子族**：Elementwise → Reduce → GEMM → Attention / Softmax → 融合算子。
5. **工程交付**：接口设计、dtype / layout、版本与平台差异、profiling 与回归。

## 阅读提示

- Ascend 工程环境与 `add_custom` 示例见 [`ascendc/`](../ascendc/)。
- GPU layout、Tensor Core、pipeline 细节见 [`modern-gpu-programming-for-mlsys/`](../modern-gpu-programming-for-mlsys/)。
- 量化相关 GEMM / Attention 算子见 [`quantization/`](../quantization/)。
- 笔记以「能复现」为准：完整代码、具体数值轨迹、可执行验证命令；缺硬件时写明静态校验与运行时校验的边界。
