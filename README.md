# learning_notes

AI 相关学习笔记：`models/` 偏模型架构；`ascend-ai-compiler/` 偏 **昇腾版 AI 编译器 / 部署**；`cuda-graph/` 偏推理 CUDA Graph；`speculative-decoding/` 偏投机解码；`torch_compile/` 偏 `torch.compile` / Dynamo / FX。

## 目录

```
ascend-ai-compiler/          # 原「AI 编译器」课 → 昇腾 CANN 全量改写（大纲逐条覆盖）
├── 00-syllabus-checklist.md # 校对清单
├── glossary.md              # 缩写英中对照
├── 01 … 18                  # 讲义
└── labs/lab1…lab6
codes/lab1/                  # Lab1：inputs / expected / Pass 骨架
cuda-graph/                  # CUDA Graph、memory-saver、SGLang vs Inductor Trees
speculative-decoding/        # 投机解码收益（GPU vs LPU 等）
torch_compile/               # Dynamo 调用过程、与 FX 的关系
```

完整目录与学习顺序见 [ascend-ai-compiler/README.md](ascend-ai-compiler/README.md)。

```
cuda-graph/
├── 01-basics-and-memory-saver.md              # Graph 是什么；region / pause / resume
├── 02-sglang-cudagraph-vs-inductor-trees.md   # 录图粒度；CudaGraphRunner vs CUDAGraph Trees
└── 03-flavors-vmm-and-hijack.md               # 几种外壳对照；VMM；劫持换分配器
```

```
speculative-decoding/
└── 01-gpu-vs-lpu-sram.md   # verify(K)≪K×decode(1)；大 SRAM/低算力为何吃不满收益
```

```
torch_compile/
├── 01-dynamo-and-fx.md           # Dynamo 调用过程；FX；Script/LazyTensor 对照
├── 02-aot-autograd.md            # AOTAutograd：joint 图、functionalize、partition
└── 03-dispatcher-and-modes.md    # DispatchKey / vtable / Mode / TLS / AOT 衔接
```

```
models/
├── deepseek_v4/
│   ├── architecture.md   # DeepSeek-V4：MQA / CSA / HCA、MoE、mHC
│   ├── mhc.md            # mHC（流形约束超连接）详解与伪代码
│   └── mqa.md            # MQA / Attention Layer 详解
├── glm-5.3/
│   ├── architecture.md   # GLM-5.3 text-only：MLA + DSA + MoE
│   ├── mla.md            # MLA latent attention
│   └── dsa.md            # DSA 动态稀疏注意力
├── glm-5.3-flash/
│   ├── architecture.md            # GLM-5.3-Flash / Glm5Next：KDA / MLA / mHC
│   └── radix_linear_attention.md  # RadixLinearAttention / KDA gated delta
├── kimi-k3/
│   └── architecture.md   # Kimi-K3：MLA / KDA / MoE / AttnRes
├── qwen-3.8/
│   └── architecture.md   # Qwen3.8-2.4T-A95B：GDN + Gated GQA + MoE（无 QSA）
└── qwen3.8-flash-next/
    ├── architecture.md   # Qwen3.8-Flash-Next 总览（Qwen4 预览）
    ├── qsa.md            # Qwen Sparse Attention（c4 indexer → 稀疏 GQA）
    └── ple.md            # N-gram / PLE（hash 查表、门控注入 HC）
```

## 昇腾 AI 编译器讲义

原课大纲 **全部条目** 均有对应课件（算能工具名 → ATC/GE/om）。入口与校对表：

- [ascend-ai-compiler/README.md](ascend-ai-compiler/README.md)
- [00-syllabus-checklist.md](ascend-ai-compiler/00-syllabus-checklist.md)
- [glossary.md](ascend-ai-compiler/glossary.md)（缩写英中对照）

| 阶段 | 内容 |
|---|---|
| 基础 | C++ 基石、编译器概论、MLIR Dialect/Pass、Lab1 |
| 硬件环境 | 昇腾存储层次、CANN 环境、Lab2 ResNet50 |
| CANN | 工作流、Parser、GE、AOE/内存调度、Lab3 ONNX→om |
| 量化性能 | INT8、校准、Lab4、Profiling |
| CV / LLM | YOLO+Stream、Attention、KV、Qwen、Lab5/6 |
| 求职 | 岗位与面试题方向 |

## 模型对照（速查）

| 目录 | 模型 | 主干注意力 | 残差 / 额外容量 | 备注 |
|---|---|---|---|---|
| `deepseek_v4/` | DeepSeek-V4 | MQA + CSA/HCA | **mHC** | 压缩稀疏检索 |
| `glm-5.3/` | GLM-5.3 text | **MLA + DSA** | 单流 | 与 Flash 不是同一套 |
| `glm-5.3-flash/` | GLM-5.3-Flash | **KDA + MLA**（hybrid） | **mHC** | `Glm5Next*` |
| `kimi-k3/` | Kimi-K3 | **KDA + MLA** | **AttnRes** | 对称 KDA 等 |
| `qwen-3.8/` | Qwen3.8-2.4T-A95B | **GDN + Gated GQA** | 单流 | **无 QSA / 无 PLE** |
| `qwen3.8-flash-next/` | Qwen3.8-Flash-Next | **GDN + QSA** | **HC 4 路 + PLE ~51B** | `qwen4_exp`；≠ 旧 Qwen3-Next-80B |

易混：

- **Qwen3.8** ≠ Qwen3-**8B** dense；本仓是 **2.4T-A95B** MoE。
- **Qwen3.8-Flash-Next** ≠ 旧 `Qwen3-Next-80B-A3B`；是 Qwen4 架构预览。
- **GLM-5.3**（MLA+DSA text）≠ **GLM-5.3-Flash**（KDA hybrid + mHC）。

## 专题索引

按机制跨模型跳转：

| 主题 | 文档 |
|---|---|
| CUDA Graph / memory-saver / SGLang Runner | [01](cuda-graph/01-basics-and-memory-saver.md)、[02](cuda-graph/02-sglang-cudagraph-vs-inductor-trees.md)、[03 · 几种外壳与 VMM](cuda-graph/03-flavors-vmm-and-hijack.md) |
| 投机解码（GPU vs LPU） | [speculative-decoding/01](speculative-decoding/01-gpu-vs-lpu-sram.md) |
| torch.compile / Dynamo / FX / AOTAutograd / Dispatcher | [01](torch_compile/01-dynamo-and-fx.md)、[02](torch_compile/02-aot-autograd.md)、[03 · Dispatcher/Mode](torch_compile/03-dispatcher-and-modes.md) |
| MLA | [glm-5.3/mla.md](models/glm-5.3/mla.md)、[glm-5.3-flash/architecture.md](models/glm-5.3-flash/architecture.md)、[kimi-k3/architecture.md](models/kimi-k3/architecture.md) |
| DSA（token/KPool 稀疏） | [glm-5.3/dsa.md](models/glm-5.3/dsa.md) |
| QSA（block 粗选 → token 展开） | [qwen3.8-flash-next/qsa.md](models/qwen3.8-flash-next/qsa.md) |
| Linear / gated delta（KDA / GDN / Radix） | [glm-5.3-flash/radix_linear_attention.md](models/glm-5.3-flash/radix_linear_attention.md)、[qwen-3.8/architecture.md](models/qwen-3.8/architecture.md)、[kimi-k3/architecture.md](models/kimi-k3/architecture.md) |
| mHC / HC / Gated Residual | [deepseek_v4/mhc.md](models/deepseek_v4/mhc.md)、[qwen3.8-flash-next/architecture.md](models/qwen3.8-flash-next/architecture.md)（HC；无 Sinkhorn） |
| AttnRes | [kimi-k3/architecture.md](models/kimi-k3/architecture.md) |
| PLE / N-gram embedding | [qwen3.8-flash-next/ple.md](models/qwen3.8-flash-next/ple.md) |
| MQA / CSA / HCA | [deepseek_v4/mqa.md](models/deepseek_v4/mqa.md)、[deepseek_v4/architecture.md](models/deepseek_v4/architecture.md) |

---

## DeepSeek-V4

基于 [SGLang](https://github.com/sgl-project/sglang) 源码梳理。

| 文档 | 内容 |
|---|---|
| [architecture.md](models/deepseek_v4/architecture.md) | 模块树、MQA / CSA / HCA、MoE、KV 与压缩状态 |
| [mhc.md](models/deepseek_v4/mhc.md) | mHC 动机、公式、Sinkhorn、层内数据流与伪代码 |
| [mqa.md](models/deepseek_v4/mqa.md) | MQA 投影形状、单 KV head、RoPE/nope、CSA/HCA 与 Attention 数据流 |

## GLM-5.3

Text-only 主干（`GlmMoeDsaForCausalLM` 一脉）。**不要**和 `glm-5.3-flash/` 混读。

| 文档 | 内容 |
|---|---|
| [architecture.md](models/glm-5.3/architecture.md) | MLA / DSA / MoE 主干结构 |
| [mla.md](models/glm-5.3/mla.md) | MLA 投影、latent KV cache、RoPE/nope |
| [dsa.md](models/glm-5.3/dsa.md) | DSA indexer、top-k、skip 复用、KPool 与 MLA 关系 |

## GLM-5.3-Flash

`Glm5Next*`：hybrid **KDA + MLA**，可选 mHC / 多模态。

| 文档 | 内容 |
|---|---|
| [architecture.md](models/glm-5.3-flash/architecture.md) | KDA、MLA、DSA、mHC、多模态入口与 DFlash capture |
| [radix_linear_attention.md](models/glm-5.3-flash/radix_linear_attention.md) | `RadixLinearAttention` 层壳、KDA gated delta 递推、prefill chunk / decode、conv/SSM cache |

## Kimi-K3

| 文档 | 内容 |
|---|---|
| [architecture.md](models/kimi-k3/architecture.md) | 配置一览、MLA/KDA 混排、MoE、AttnRes（bank + prefix）、forward / logits |

## Qwen3.8

**Qwen3.8-2.4T-A95B**（`qwen3_5_moe_text`）。Hybrid = GDN + **Gated GQA**，**没有** QSA / PLE。

| 文档 | 内容 |
|---|---|
| [architecture.md](models/qwen-3.8/architecture.md) | 配置、GDN/Gated GQA 混排、Sparse MoE |

## Qwen3.8-Flash-Next

Qwen4 架构预览（`qwen4_exp`）：GDN + **QSA**，**Gated Residual（HC×4）**，**N-gram / PLE ~51B**。

| 文档 | 内容 |
|---|---|
| [architecture.md](models/qwen3.8-flash-next/architecture.md) | 配置总览、GDN/QSA 混排、HC Mix/Combine、PLE 入口 |
| [qsa.md](models/qwen3.8-flash-next/qsa.md) | c4 压缩 indexer、`update_compressed_index_cache`、expand_blocks / unfinished 尾巴、稀疏 GQA、MTP IndexShare |
| [ple.md](models/qwen3.8-flash-next/ple.md) | PLE 动机、可学习 N-gram 表、int64 hash、`gate(norm(Q),norm(K))`、注入 HC、host offload |

要点备忘：

- **QSA**：先 top-k **block** → `expand_blocks` 成 token index → 读**原始** K/V；未满 4-token 尾巴强制并入。
- **PLE**：预训练学好的查表（非 token emb 拷贝）；推理冻结，可放 host；hash = 乘数 + XOR + 素数模。

---

## License

[MIT](LICENSE)
