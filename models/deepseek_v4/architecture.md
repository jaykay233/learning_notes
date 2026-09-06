# SGLang 中 DeepSeek-V4 模型结构梳理

> 基于本地仓库 `~/Downloads/sglang` 中的实现整理，主入口：
> `python/sglang/srt/models/deepseek_v4.py`、`python/sglang/srt/configs/deepseek_v4.py`。

## 1. 一句话总览

DeepSeek-V4 在 SGLang 里是一个 **带 mHC（流形约束超连接）的 MoE Transformer**，注意力侧是 **单 KV head 的 MQA + 低秩 Q/O 投影**，并按层配置 **混合压缩注意力**：

| `compress_ratio` | 名称 | 含义 |
|---|---|---|
| `0` | SWA / 纯窗口 | 不做压缩，主要走局部窗口注意力 |
| `4` | **CSA**（Compressed Sparse Attention） | 4× 压缩状态 + **C4Indexer** 做 top-k 检索 |
| `128` | **HCA**（Highly Compressed Attention） | 128× 强压缩，长上下文主路径 |

官方描述（相对 DSv3.2 @ 1M context）：混合 CSA+HCA 大约 **~27% 推理 FLOPs / ~10% KV cache**。

变体规模（HF）：

| 变体 | 总参 | 激活 |
|---|---:|---:|
| Flash / Flash-0731 | ~284–304B | ~13B |
| Pro / Pro-0813 | ~1.6–1.65T | ~49B |

Instruct 权重常见混合精度：**FP4 routed experts + FP8 attention/dense**。

### 1.1 各版本主干结构参数

下面参数来自各 checkpoint 的 `config.json`。`layers` 指主干 `DeepseekV4DecoderLayer` 数量；每层 `hidden_size` 固定不变。因为 V4 使用 `hc_mult=4`，层间 mHC residual 的形状是 `[T, 4, hidden_size]`，等价扁平宽度是 `4 * hidden_size`。

| Variant / HF repo | 总参 | 激活参数 | layers | 每层 `hidden_size` | mHC residual 扁平宽度 | routed experts / 层 | 每 token 激活专家 | shared experts | expert hidden (`moe_intermediate_size`) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `DeepSeek-V4-Flash` | 284B | 13B | 43 | 4096 | 16384 | 256 | 6 | 1 | 2048 |
| `DeepSeek-V4-Flash-0731` | 304B | 13B | 43 | 4096 | 16384 | 256 | 6 | 1 | 2048 |
| `DeepSeek-V4-Flash-Vision-Exp` | 305B | 13B | 43 | 4096 | 16384 | 256 | 6 | 1 | 2048 |
| `DeepSeek-V4-Pro` | 1.6T | 49B | 61 | 7168 | 28672 | 384 | 6 | 1 | 3072 |
| `DeepSeek-V4-Pro-0813` | 1.65T | 49B | 61 | 7168 | 28672 | 384 | 6 | 1 | 3072 |

读表要点：

1. **Flash 系列主干相同**：43 层、`hidden_size=4096`、每层 256 个 routed experts，token 级 top-6。
2. **Pro 系列主干更宽更深**：61 层、`hidden_size=7168`、每层 384 个 routed experts，token 级仍然 top-6。
3. **0731 / 0813 Official**：主干结构分别沿用 Flash / Pro，但 checkpoint 带 DSpark draft head，因此总参略高。
4. **Flash Vision Exp**：语言主干沿用 Flash-0731 形态，额外加 vision encoder / aligner，所以总参到约 305B；SGLang support 走 preview 路径。

### 1.2 Attention 与压缩层分布差异

| Variant | attention heads | KV heads | head dim | Q LoRA rank | C4 index top-k | `compress_ratio` 分布 |
|---|---:|---:|---:|---:|---:|---|
| Flash | 64 | 1 | 512 | 1024 | 512 | `0×3, 4×21, 128×20` |
| Flash-0731 / Flash-Vision | 64 | 1 | 512 | 1024 | 512 | `0×5, 4×21, 128×20` |
| Pro | 128 | 1 | 512 | 1536 | 1024 | `0×1, 4×30, 128×31` |
| Pro-0813 | 128 | 1 | 512 | 1536 | 1024 | `0×3, 4×30, 128×31` |

这里的 `compress_ratio` 统计来自 HF config。`0/4/128` 分别对应 SWA / CSA / HCA；`*-0731`、`*-0813` 多出来的 `0` 主要和 official checkpoint 里的 draft/附加头配置有关，主干 decoder layer 数仍看 `num_hidden_layers`。

---

## 2. SGLang 模块层级

```
DeepseekV4ForCausalLM
└── DeepseekV4Model
    ├── embed_tokens          # VocabParallelEmbedding
    ├── layers[i]             # DeepseekV4DecoderLayer × num_hidden_layers
    │   ├── self_attn         # MQALayer
    │   │   ├── wqkv_a / (wq_a + wkv)   # 低秩 Q + 共享 KV 投影
    │   │   ├── q_norm, wq_b            # Q LoRA-B → multi-head
    │   │   ├── kv_norm
    │   │   ├── wo_a, wo_b              # 分组 O 投影（o_groups）
    │   │   ├── attn_sink
    │   │   ├── rotary_emb / freqs_cis
    │   │   ├── compressor?             # ratio∈{4,128}
    │   │   ├── indexer?                # 仅 ratio==4 → C4Indexer
    │   │   └── attn_mqa                # RadixAttention (MQA, kv_heads=1)
    │   ├── mlp                         # DeepseekV2MoE(is_deepseek_v4=True)
    │   │   ├── gate (+ HashTopK on early layers)
    │   │   ├── experts (routed)
    │   │   └── shared_experts
    │   ├── input_layernorm / post_attention_layernorm
    │   └── mHC params (attn/ffn 两套 mixing)
    ├── norm
    └── hc_head_*                       # 最后把 hc_mult 路合并回 hidden
```

相关衍生入口：

| 类 / 文件 | 用途 |
|---|---|
| `DeepseekV4ForCausalLM` | 主模型 |
| `deepseek_v4_nextn.py` | NextN / MTP draft |
| `deepseek_v4_dspark.py` | DSpark speculative draft head（0731/0813） |
| `deepseek_v4_backend.py` | NVIDIA attention backend |
| `mem_cache/deepseek_v4_memory_pool.py` | 混合 KV + C4/C128 压缩状态池 |

HF architecture 识别：`DeepseekV4ForCausalLM` / `...NextN` / `...DSpark`。

---

## 3. Decoder Layer 数据流（mHC 是核心差异）

普通 Transformer 是 `x → norm → attn → +x → norm → ffn → +x`。

V4 把 residual 扩成 **`hc_mult` 路并行流**（默认 `hc_mult=4`），用 **mHC（manifold-constrained hyper-connections）** 做混合：

```
hidden [T, H] 或 [T, hc_mult, H]
        │
        ▼
   hc_pre(attn)  ──► Sinkhorn 约束的 mixing
        │              +（可融合）input_layernorm
        ▼
   MQALayer (attention)
        │
        ▼
   hc_post(attn) → hc_pre(ffn)  ──►（可跨层融合）
        │                           + post_attention_layernorm
        ▼
   DeepseekV2MoE
        │
        ▼
   hc_post(ffn)  ──► 交给下一层 / 最后 hc_head 收束
```

要点：

1. **`hc_pre`**：对 `hc_mult * hidden` 做线性混合，RMS 归一后经 Sinkhorn（`hc_sinkhorn_iters`，默认 20）得到组合权重。
2. **`hc_post`**：把子模块输出写回多路 residual。
3. 实现里有大量融合路径（TileLang / FlashInfer / HIP aiter / NPU），以及 **跨层 `mhc_post_pre` fusion**，把「上一层 ffn 的 post + 本层 attn 的 pre」并成一次 kernel。
4. 最后一层用 **`hc_head`** 把 `[T, hc_mult, H]` 压回 `[T, H]`，再进 lm_head。

---

## 4. Attention：`MQALayer`

### 4.1 投影形状（相对 MLA/标准 MHA）

配置默认（Flash 量级；Pro 见 §1.1 / §1.2）：

| 字段 | 默认 | 说明 |
|---|---:|---|
| `hidden_size` | 4096 | |
| `num_attention_heads` | 64 | |
| `num_key_value_heads` | **1** | 真·MQA |
| `qk_nope_head_dim` | 448 | 非 RoPE 部分 |
| `qk_rope_head_dim` | 64 | RoPE 部分 |
| `head_dim` | ≈512 | `nope + rope`（HF 可显式给出） |
| `v_head_dim` | 512 | |
| `q_lora_rank` | 1024 | Q 低秩中间维 |
| `o_lora_rank` | 1024 | O 低秩中间维 |
| `o_groups` | 8 | O 投影分组 |
| `window_size` | 128 | 局部窗口 |

投影路径（可融合 `wqkv_a`）：

```
x ─► wq_a / wqkv_a ─► q_lora ─► q_norm ─► wq_b ─► Q [T, H_local, D]
x ─► wkv            ─► kv_norm (± RoPE) ─► 单路 KV cache
attn(Q, K, V) ─► wo_a (按 o_groups) ─► wo_b ─► out [T, H]
```

还有 **`attn_sink`**（per-head float 参数），用于 attention sink。

### 4.2 按层 `compress_ratio`

`config.compress_ratios[layer_id] ∈ {0, 4, 128}`：

```
ratio == 0:
  纯 SWA / 主 RoPE，无 compressor / indexer

ratio == 4 (CSA):
  Compressor(4×, overlap=True) + C4Indexer
  YaRN / compress_rope_theta 用于压缩分支

ratio == 128 (HCA):
  Compressor(128×)，无 C4Indexer
  长上下文高度压缩 KV 状态
```

RoPE 策略（代码注释）：

- 纯 SWA 层：主 RoPE（未缩放）
- C4 / C128 层：压缩 YaRN RoPE（HF checkpoint 常见 `compress_rope_theta=160000`；SGLang dataclass 默认 40000）用于 Q、SWA 分支与压缩 KV

### 4.3 `Compressor`

把若干 token 的 KV 压成一个压缩状态，写入 `CompressStatePool`：

- 权重：`ape`（绝对位置偏置）、`wkv_gate`、`norm`
- CSA(4)：**overlap**（窗口重叠）
- HCA(128)：可用 **online compress**（ring 塌成 1，decode 更省）

### 4.4 `C4Indexer`（仅 CSA 层）

用 compressed index 对历史做 **top-k**，再驱动稀疏 MQA：

| 字段 | 默认 |
|---|---:|
| `index_n_heads` | 64 |
| `index_head_dim` | 128 |
| `index_topk` | 512 |

结构大致：`wq_b`（从 q_lora 出 indexer Q）+ 内嵌 `Compressor(is_in_indexer=True)` + `weights_proj` + top-k / paged MQA logits kernel。

可选用实验开关 `--enable-deepseek-v4-fp4-indexer` 走 FP4 indexer。

---

## 5. FFN / MoE

V4 **复用** `DeepseekV2MoE`，构造时 `is_deepseek_v4=True`。

Flash 量级 MoE 配置（Pro 对照见 §1.1）：

| 字段 | 默认 | 说明 |
|---|---:|---|
| `n_routed_experts` | 256 | 路由专家数 |
| `num_experts_per_tok` | 6 | 每 token 激活专家 |
| `n_shared_experts` | 1 | 共享专家 |
| `moe_intermediate_size` | 2048 | |
| `n_group` / `topk_group` | 8 | 分组路由 |
| `topk_method` | `noaux_tc` | |
| `scoring_func` | `sqrtsoftplus` | |
| `routed_scaling_factor` | 1.5 | |
| `n_hash_layers` / `num_hash_layers` | 3 | 前几层用 **HashTopK** |

行为要点：

1. 前 `num_hash_layers` 层 gate 走 **HashTopK**（与 Vision 等场景下 shared-experts fusion 互斥时有关）。
2. 其余层走常规 DeepSeek 风格 top-k。
3. Shared expert 可与 routed experts **fusion**（`--enforce-shared-experts-fusion`，Blackwell MXFP4 路径常见）。
4. 服务侧可接 DeepEP / MegaMoE / EPLB 等并行后端（属 runtime，不是结构本身）。

Dense 前缀：`first_k_dense_replace`（默认 0）——若 >0，前几层可仍是 dense MLP（配置项保留，具体 checkpoint 以 HF `config.json` 为准）。

---

## 6. 整网默认超参速查（`DeepSeekV4Config`）

SGLang dataclass 默认值基本对应 Flash 量级 schema；真实部署以 checkpoint `config.json` 为准。

| 字段 | Flash 系列 | Pro 系列 | 说明 |
|---|---:|---:|---|
| `model_type` | `deepseek_v4` | `deepseek_v4` | HF / SGLang 识别名 |
| `vocab_size` | 129280 | 129280 | tokenizer 词表 |
| `num_hidden_layers` | 43 | 61 | 主干 decoder block 数 |
| `hidden_size` | 4096 | 7168 | 每层 residual / hidden 维度 |
| `hc_mult` | 4 | 4 | mHC residual 路数 |
| `hc_mult * hidden_size` | 16384 | 28672 | mHC 内部扁平 residual 宽度 |
| `num_attention_heads` | 64 | 128 | Q heads |
| `num_key_value_heads` | 1 | 1 | MQA 单 KV head |
| `head_dim` | 512 | 512 | attention head dim |
| `q_lora_rank` | 1024 | 1536 | Q 低秩中间维 |
| `o_lora_rank` | 1024 | 1024 | O 低秩中间维 |
| `n_routed_experts` | 256 | 384 | 每个 MoE 层的 routed expert 总数 |
| `num_experts_per_tok` | 6 | 6 | 每 token 激活 routed experts |
| `n_shared_experts` | 1 | 1 | 共享专家数 |
| `moe_intermediate_size` | 2048 | 3072 | 单个 expert 的中间维 |
| `num_hash_layers` | 3 | 3 | 前几层 gate 走 HashTopK |
| `index_topk` | 512 | 1024 | CSA indexer top-k |
| `max_position_embeddings` | 65536 | 65536 | 配置基准；服务目标可到 1M context |
| `compress_rope_theta` | 160000 | 160000 | HF checkpoint 值；SGLang dataclass 默认是 40000 |
| `hc_sinkhorn_iters` | 20 | 20 | mHC Sinkhorn 迭代数 |
| `hc_eps` | 1e-6 | 1e-6 | mHC 数值稳定项 |

**注意**：`intermediate_size` 在 HF checkpoint 里通常不用作普通 dense FFN 主路径；MoE 重点看 `moe_intermediate_size`、`n_routed_experts`、`n_shared_experts` 和 `num_experts_per_tok`。

---

## 7. KV / 压缩状态内存形态

`DeepSeekV4TokenToKVPool` 不是单一 dense KV，而是混合池：

1. **主 KV**：MQA，nope 部分常 FP8/量化存储，rope 部分 BF16。
2. **C4 / C128 CompressStatePool**：按 `compress_ratio` 分 ring；dtype 可用 `SGLANG_DSV4_COMPRESS_STATE_DTYPE`（默认 fp32，可 bf16）。
3. **Indexer cache**（CSA 层）：按 `index_head_dim` 存 FP8 或 FP4 索引向量。

这解释了长上下文时 KV 明显小于「全量 dense MHA / 甚至 MLA」。

---

## 8. 与 DeepSeek-V2/V3 的对比（实现视角）

| 维度 | V2/V3（SGLang） | V4 |
|---|---|---|
| Attention | MLA（低秩 KV）为主 | **MQA + 压缩状态（CSA/HCA）+ Indexer** |
| Residual | 标准残差 | **mHC 多路超连接** |
| MoE 模块 | `DeepseekV2MoE` | **同一套 MoE**，加 HashTopK / fusion 策略 |
| 长上下文 | YaRN / DSA 等演进 | **C4 indexer + C128 压缩** 为主路径 |
| Speculative | EAGLE 等 | 0731/0813 捆绑 **DSpark**；另有 NextN |

---

## 9. 源码地图（继续深挖时看这些）

```
srt/models/deepseek_v4.py              # 主模型 / Decoder / MQA / mHC
srt/configs/deepseek_v4.py             # Config schema
srt/layers/attention/dsv4/             # compressor / indexer / metadata
srt/layers/attention/deepseek_v4_backend.py
srt/mem_cache/deepseek_v4_memory_pool.py
srt/mem_cache/deepseek_v4_compress_state.py
srt/models/deepseek_v2.py              # 复用的 MoE / Gate / HashTopK
srt/models/deepseek_v4_dspark.py       # DSpark draft
srt/models/deepseek_v4_nextn.py        # NextN
kernels/ops/attention/dsv4/            # CUDA/HIP kernels
docs/cookbook/.../DeepSeek-V4.mdx      # 部署与变体说明
```

---

## 10. 读代码时的推荐顺序

1. `DeepseekV4DecoderLayer.forward` —— 先建立 mHC → attn → MoE 顺序  
2. `MQALayer.__init__` —— 看 `compress_ratio` 如何挂 Compressor / Indexer  
3. `Compressor` + `C4Indexer` —— 理解 CSA 稀疏检索  
4. `DeepseekV2MoE(..., is_deepseek_v4=True)` —— 路由与 HashTopK  
5. `DeepSeekV4TokenToKVPool` —— 对照服务时的显存结构  

---

*整理日期：2026-09-06，对应本地 sglang 树中的 DeepSeek-V4 实现。*
