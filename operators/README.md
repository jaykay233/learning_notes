# 算子开发

集中记录高性能算子（CUDA / Triton / TileLang 等）从问题建模到上板验证的知识点。
与 [`ascendc/`](../ascendc/)（AscendC CPU 孪生工程环境）、[`modern-gpu-programming-for-mlsys/`](../modern-gpu-programming-for-mlsys/)（GPU layout / Tensor Core / GEMM）互补：那边偏硬件与编程模型，这边偏**算子工程闭环**。

## 目录

```text
operators/
├── README.md
├── 01-triton-for-loops.md  # Triton 分块循环与累加
├── 02-triton-matmul-blocked-kernel.md  # Triton blocked GEMM 与 K-loop
├── 03-triton-low-memory-dropout.md  # Triton seeded low-memory dropout
├── 04-triton-type-promotion.md  # Triton type promotion
├── 05-triton-broadcasting-and-chained-comparisons.md  # Broadcasting 与链式比较
├── 06-triton-differences-from-numpy.md  # 与 NumPy/Python 的语义差异
├── 07-triton-gluon-programming-model-and-layouts.md  # Gluon 编程模型与 layouts
├── 08-triton-gluon-async-tma-and-pipelining.md  # async copy、TMA 与流水线
├── 09-triton-gluon-wgmma-and-tcgen05.md  # WGMMA、TMEM 与 tcgen05
├── 10-triton-gluon-persistent-warp-specialization.md  # persistent 与 warp specialization
├── 11-triton-gluon-blackwell-dataflow-and-convolution.md  # gather/scatter、scaled MMA、CLC、im2col
├── 12-triton-gluon-multicta-and-capstone.md  # CGA、multicast、2CTA 与 capstone GEMM
├── 13-flashmla-sparse-attention-reading-guide.md  # FlashMLA sparse decode：API、paged KV、split-KV 与 softmax merge
├── 14-flashkda-kda-dataflow-reading-guide.md  # FlashKDA KDA：chunkwise delta rule、K1/K2 与 recurrent state
└── code/
    ├── cuda/       # 原生 CUDA / WMMA
    ├── triton/     # OpenAI Triton；含固定 revision 的 gluon 教程归档
    ├── tilelang/   # TileLang DSL
    ├── flashmla/   # FlashMLA 数学 trace 示例
    └── flashkda/   # FlashKDA chunk recurrence 数学 trace 示例
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

## Triton 学习笔记

| 文档 | 内容 |
|---|---|
| [01-triton-for-loops.md](01-triton-for-loops.md) | `for range` 分块遍历、尾块 mask、跨轮累加与 `tl.static_range` |
| [02-triton-matmul-blocked-kernel.md](02-triton-matmul-blocked-kernel.md) | Triton 官方 blocked GEMM：program/tile 映射、二维 pointer grid、`tl.dot` K-loop、L2 分组与 autotune |
| [03-triton-low-memory-dropout.md](03-triton-low-memory-dropout.md) | Triton 官方 low-memory dropout：`(seed, global offset)` 现场生成 mask、Philox PRNG、inverted dropout 与 recompute 状态管理 |
| [04-triton-type-promotion.md](04-triton-type-promotion.md) | Triton type promotion：kind/width/signedness、tensor-scalar 特殊规则、除法取模例外与 broadcasting 的区别 |
| [05-triton-broadcasting-and-chained-comparisons.md](05-triton-broadcasting-and-chained-comparisons.md) | Triton broadcasting：左补一、维度相等或为 1、虚拟扩展成本边界；chained comparisons：逐对比较、elementwise AND、操作数只求值一次与 runtime 不短路 |
| [06-triton-differences-from-numpy.md](06-triton-differences-from-numpy.md) | Triton 与 NumPy/Python 的差异：runtime 整数除法/取模向零截断、`tl.constexpr` 的 Python 取整、越界 float-to-int cast 与变量作用域 |
| [07-triton-gluon-programming-model-and-layouts.md](07-triton-gluon-programming-model-and-layouts.md) | Gluon 编程模型与 layouts：`BlockedLayout`、`SliceLayout`、`NVMMASharedLayout`、distributed layout 与显式 layout 控制 |
| [08-triton-gluon-async-tma-and-pipelining.md](08-triton-gluon-async-tma-and-pipelining.md) | Gluon async copy、TMA 与软件流水：`cp.async`、commit group、`mbarrier`、tensor descriptor、代理 fence、TMA store 两个完成点与三缓冲 elementwise add |
| [09-triton-gluon-wgmma-and-tcgen05.md](09-triton-gluon-wgmma-and-tcgen05.md) | WMMA/WGMMA/tcgen05 操作数驻留对照；Hopper WGMMA 与 Blackwell tcgen05：二维 blocked tile/K-loop、TMA 尾块与 mbarrier parity、A 的 register/SMEM 路径、`DotOperandLayout`、异步等待、双缓冲流水、Tensor Memory 容量、TMEM→register→GMEM epilogue |
| [10-triton-gluon-persistent-warp-specialization.md](10-triton-gluon-persistent-warp-specialization.md) | Persistent kernel 与 `gl.warp_specialize`：grouped scheduler、`@gluon.aggregate`、producer/MMA/epilogue 分工与 barrier phase |
| [11-triton-gluon-blackwell-dataflow-and-convolution.md](11-triton-gluon-blackwell-dataflow-and-convolution.md) | Blackwell 数据流与卷积：TMA gather/scatter、`tcgen05_copy`、block-scaled MMA、CLC、TMA im2col 与 implicit GEMM |
| [12-triton-gluon-multicta-and-capstone.md](12-triton-gluon-multicta-and-capstone.md) | CGA/multi-CTA：`cga_layout` 分片与广播、cluster barrier、TMA multicast、2CTA tcgen05、CLC+四角色 capstone GEMM；kernel 开发流程：correctness baseline、profile、逐层引入硬件数据流 |
| [13-flashmla-sparse-attention-reading-guide.md](13-flashmla-sparse-attention-reading-guide.md) | FlashMLA sparse decode：DSA indices 与 paged quantized KV 的接口契约、Python→C++→kernel dispatch、split-KV 稳定 softmax 合并 |
| [14-flashkda-kda-dataflow-reading-guide.md](14-flashkda-kda-dataflow-reading-guide.md) | FlashKDA：逐 token 公式拆解与 LaTeX 数据流图、数值轨迹、chunkwise delta rule、三角矩阵求解、K1/K2 与 TMA/warp specialization |

示例脚本：[sparse_attention_split_merge.py](code/flashmla/sparse_attention_split_merge.py)、[kda_scalar_trace.py](code/flashkda/kda_scalar_trace.py)、[for_loop_row_sum.py](code/triton/for_loop_row_sum.py)、[matmul_tutorial.py](code/triton/matmul_tutorial.py)、[seeded_dropout.py](code/triton/seeded_dropout.py)、[type_promotion.py](code/triton/type_promotion.py)、[broadcasting_and_chained_comparisons.py](code/triton/broadcasting_and_chained_comparisons.py)、[numpy_differences.py](code/triton/numpy_differences.py)。Gluon 官方教程按固定 revision 归档在 [code/triton/gluon](code/triton/gluon/)，provenance 与运行范围见 [PROVENANCE.md](code/triton/gluon/PROVENANCE.md)。
