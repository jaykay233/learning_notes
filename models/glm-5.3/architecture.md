# GLM-5.3 架构梳理

> 本文先整理 **GLM-5.3 text-only / `GlmMoeDsaForCausalLM`** 主干，基于本地 SGLang 实现阅读：
> `python/sglang/srt/models/glm4_moe.py`、`python/sglang/srt/models/deepseek_v2.py`、`python/sglang/srt/layers/attention/dsa/`。
> 注意：`GLM-5.3-Flash` 是另一套新训练的 320B / 18B active 多模态混合注意力架构，包含 KDA、KPool-DSA、mHC 等；不要和本文的 GLM-5.3 BF16 / FP8 text-only 主干混在一起。

---

## 1. 一句话总览

GLM-5.3 text-only 可以先理解成一个 **MoE + MLA + DSA 的大规模 Decoder-only Transformer**：

```text
Token Embedding
  │
  ▼
78 层 GLM decoder blocks
  ├─ Attention: DeepSeek-style MLA + DSA 稀疏检索
  ├─ FFN: 前 3 层 dense，后续 MoE
  └─ Norm: RMSNorm
  │
  ▼
Final RMSNorm
  │
  ▼
LM Head

可选：第 78 层之后还有 MTP / next-token draft layer，用于 speculative decoding。
```

核心心智模型：

1. **MLA** 负责把 KV cache 压到 latent 表示，降低长上下文 KV 成本。
2. **DSA** 负责在长上下文中做动态稀疏选择，避免每层都全量看所有历史。
3. **MoE** 负责扩大总参数量，但每个 token 只激活少量专家。
4. **前 3 层 dense**，从第 3 层之后大多是 routed MoE。

---

## 2. 关键结构参数

常见 GLM-5.3 / GLM-5.2 同构配置：

| 字段 | 值 | 含义 |
|---|---:|---|
| `architectures` | `GlmMoeDsaForCausalLM` | HF / runtime 识别的模型类 |
| `model_type` | `glm_moe_dsa` | 配置类型 |
| `vocab_size` | 154880 | tokenizer 词表大小 |
| `num_hidden_layers` | 78 | 主干 decoder layers |
| `hidden_size` | 6144 | residual / hidden 宽度 |
| `intermediate_size` | 12288 | dense FFN 中间维 |
| `moe_intermediate_size` | 2048 | 单个 expert 的中间维 |
| `num_attention_heads` | 64 | Q heads |
| `num_key_value_heads` | 64 | 注意：MLA 下不是普通 GQA 的主要省 cache 手段 |
| `q_lora_rank` | 2048 | Q 低秩投影 rank |
| `kv_lora_rank` | 512 | MLA KV latent rank |
| `qk_nope_head_dim` | 192 | Q/K 非 RoPE 部分 |
| `qk_rope_head_dim` | 64 | Q/K RoPE 部分 |
| `v_head_dim` | 256 | V head dim |
| `index_topk` | 2048 | DSA indexer top-k |
| `index_head_dim` | 128 | DSA indexer head dim |
| `index_n_heads` | 32 | DSA indexer heads |
| `index_kpool` | checkpoint dependent | `>1` 时用 `IndexerKPool`，缺省或 `1` 时用普通 `Indexer` |
| `max_position_embeddings` | 1048576 | 1M context |
| `rms_norm_eps` | 1e-5 | RMSNorm eps |
| `tie_word_embeddings` | false | lm head 不绑 embedding |

一些派生尺寸：

```text
Q/K head_dim = qk_nope_head_dim + qk_rope_head_dim
             = 192 + 64
             = 256

Attention output per token before O projection:
  num_attention_heads * v_head_dim
  = 64 * 256
  = 16384

KV latent cache rank:
  kv_lora_rank = 512
```

---

## 3. 模块树

SGLang 里的关键点：`GlmMoeDsaForCausalLM` 定义在 `glm4_moe.py`，但它本身几乎是一个入口类：

```python
class GlmMoeDsaForCausalLM(DeepseekV2ForCausalLM):
    fused_shared_experts_architecture = "GlmMoeDsaForCausalLM"
```

也就是说，GLM-5.3 text-only 在 SGLang 里主要复用 `deepseek_v2.py` 的 MLA / MoE / DSA 服务实现，而不是走同文件里的 `Glm4MoeAttention` 那套普通 QKV attention。

教学版模块树：

```text
GlmMoeDsaForCausalLM
└── DeepseekV2ForCausalLM runtime path
    ├── model                         # DeepseekV2Model 风格
    ├── embed_tokens
    ├── layers[0..77]                 # DeepseekV2DecoderLayer × 78
    │   ├── input_layernorm
    │   ├── self_attn                 # DeepseekV2AttentionMLA
    │   │   ├── q_a_proj / q_a_layernorm / q_b_proj
    │   │   ├── kv_a_proj_with_mqa    # hidden -> KV latent + RoPE part
    │   │   ├── kv_a_layernorm
    │   │   ├── kv_b_proj             # latent -> K/V heads
    │   │   ├── o_proj
    │   │   ├── indexer?              # DSA Indexer；index_kpool > 1 时为 IndexerKPool
    │   │   └── rotary_emb
    │   ├── post_attention_layernorm
    │   └── mlp
    │       ├── DeepseekV2MLP         # 前 first_k_dense_replace 层
    │       └── DeepseekV2MoE         # 后续层
    │           ├── gate / router
    │           ├── routed experts    # 256 experts
    │           └── shared experts    # 1 shared expert
    ├── norm
    └── lm_head

可选:
└── GlmMoeDsaForCausalLMNextN          # speculative draft
```

命名在不同 runtime / checkpoint 里会略有变化，但读 SGLang 代码时重点看三块：

1. `DeepseekV2AttentionMLA`：MLA、RoPE/nope 拆分、DSA indexer。
2. `DeepseekV2DecoderLayer`：layer norm、attention、dense/MoE 分支、DSA communicator。
3. `DeepseekV2MoE`：router、top-k experts、shared experts。
4. `GlmMoeDsaForCausalLMNextN`：如果 checkpoint 带 speculative decoding draft layer，要把它和 78 层主干分开看。

---

## 4. Decoder Layer 数据流

单层可以抽象成：

```text
x [T,H]
  │
  ├─► RMSNorm
  │     │
  │     ▼
  │   MLA / DSA Attention
  │     │
  │     ▼
  │   attn_out [T,H]
  │
  ├─ residual add
  │
  ▼
x1 [T,H]
  │
  ├─► RMSNorm
  │     │
  │     ▼
  │   Dense MLP or MoE
  │     │
  │     ▼
  │   ffn_out [T,H]
  │
  └─ residual add
        ▼
      x2 [T,H]
```

层类型由两条配置决定：

```text
Attention:
  MLA 是基础；
  DSA / indexer 根据 indexer_types / index_topk_pattern 等配置启用。

FFN:
  layer_id < first_k_dense_replace -> dense MLP
  其余层 -> MoE
```

常见：

```text
first_k_dense_replace = 3

layers 0,1,2:
  dense FFN

layers 3..77:
  MoE FFN
```

---

## 5. 整网伪代码（SGLang text-only 路径）

下面是教学版伪代码，目标是帮你把 SGLang 里的 `GlmMoeDsaForCausalLM -> DeepseekV2ForCausalLM` 路径串起来，不是逐行复刻 kernel。

形状约定：

```text
T      = 当前 batch 展平后的 token 数
H      = hidden_size = 6144
L      = num_hidden_layers = 78
A      = num_attention_heads = 64
D_nope = qk_nope_head_dim = 192
D_rope = qk_rope_head_dim = 64
D_qk   = D_nope + D_rope = 256
D_v    = v_head_dim = 256
R_q    = q_lora_rank = 2048
R_kv   = kv_lora_rank = 512
E      = n_routed_experts = 256
K_exp  = num_experts_per_tok = 8
K_dsa  = index_topk = 2048
```

### 5.1 顶层 `ForCausalLM`

```python
class GlmMoeDsaForCausalLM(DeepseekV2ForCausalLM):
    """
    SGLang 入口类。GLM-5.3 text-only 主要复用 DeepseekV2ForCausalLM 的
    Model / DecoderLayer / MLA / MoE / DSA 实现。
    """
    pass


def causal_lm_forward(input_ids, positions, forward_batch):
    """
    input_ids    : [T]
    positions    : [T]
    forward_batch: runtime metadata，包括 kv cache、DSA metadata、prefill/decode 模式等
    """
    hidden = model_forward(input_ids, positions, forward_batch)   # [T, H]
    logits = lm_head(hidden)                                      # [T, vocab_size]
    return logits
```

### 5.2 `DeepseekV2Model` 风格主干

```python
def model_forward(input_ids, positions, forward_batch):
    """
    return hidden: [T, H]
    """
    hidden = embed_tokens(input_ids)                              # [T, H]
    residual = None

    for layer_id in range(L):                                     # L = 78
        hidden, residual = decoder_layer(
            layer_id=layer_id,
            hidden=hidden,                                        # [T, H]
            residual=residual,                                    # None or [T, H]
            positions=positions,                                  # [T]
            forward_batch=forward_batch,
        )

    if residual is None:
        hidden = final_norm(hidden)                               # [T, H]
    else:
        hidden = final_norm(hidden, residual)                     # [T, H]

    return hidden
```

### 5.3 Decoder Layer

```python
def decoder_layer(layer_id, hidden, residual, positions, forward_batch):
    """
    hidden  : [T, H]
    residual: None or [T, H]
    """

    # ---- Attention branch ----
    attn_in, residual = input_layernorm(hidden, residual)
    # attn_in : [T, H]
    # residual: [T, H]

    attn_out = mla_dsa_attention(
        layer_id=layer_id,
        x=attn_in,
        positions=positions,
        forward_batch=forward_batch,
    )                                                             # [T, H]

    hidden = attn_out                                             # [T, H]

    # ---- FFN branch ----
    ffn_in, residual = post_attention_layernorm(hidden, residual)
    # ffn_in  : [T, H]
    # residual: [T, H]

    if layer_id < first_k_dense_replace:                          # 常见 0,1,2
        ffn_out = dense_mlp(ffn_in)                               # [T, H]
    else:
        ffn_out = moe_ffn(layer_id, ffn_in, forward_batch)        # [T, H]

    hidden = ffn_out
    return hidden, residual
```

### 5.4 MLA + DSA Attention

```python
def mla_dsa_attention(layer_id, x, positions, forward_batch):
    """
    x: [T, H]
    return: [T, H]
    """

    # 1. Q low-rank path
    q_lora = q_a_proj(x)                                          # [T, R_q]
    q_lora = q_a_layernorm(q_lora)                                # [T, R_q]
    q = q_b_proj(q_lora)                                          # [T, A * D_qk]
    q = q.reshape(T, A, D_qk)                                     # [T, A, D_qk]

    q_nope, q_rope = split(q, [D_nope, D_rope], dim=-1)
    # q_nope: [T, A, D_nope]
    # q_rope: [T, A, D_rope]

    # 2. KV latent path
    kv_and_rope = kv_a_proj_with_mqa(x)                           # [T, R_kv + D_rope]
    kv_latent, k_rope = split(kv_and_rope, [R_kv, D_rope], dim=-1)
    # kv_latent: [T, R_kv]
    # k_rope   : [T, D_rope]

    kv_latent = kv_a_layernorm(kv_latent)                         # [T, R_kv]
    kv_cache.write_mla_latent(layer_id, kv_latent, k_rope, positions)

    # 3. 展开当前 token 的 K/V，用于 prefill 或当前步 attention 计算。
    kv_full = kv_b_proj(kv_latent)                                # [T, A * (D_nope + D_v)]
    k_nope, v = split_kv_b(kv_full)
    # k_nope: [T, A, D_nope]
    # v     : [T, A, D_v]

    q_rope = apply_rope(q_rope, positions)                        # [T, A, D_rope]
    k_rope = apply_rope(k_rope, positions)                        # [T, D_rope]

    q = concat(q_nope, q_rope, dim=-1)                             # [T, A, D_qk]

    # 4. DSA 选择历史。部分层 / 模式可跳过 top-k，退化成更 dense 的 MLA 读取。
    if dsa_enabled(layer_id, forward_batch):
        selected = dsa_indexer(
            layer_id=layer_id,
            q_lora=q_lora,                                        # [T, R_q]
            positions=positions,
            forward_batch=forward_batch,
        )                                                         # [T, ..., K_dsa]
    else:
        selected = None

    # 5. MLA attention backend：根据 selected 只读相关历史 latent/KV。
    ctx = mla_attention_backend(
        q=q,                                                       # [T, A, D_qk]
        current_k_nope=k_nope,                                    # [T, A, D_nope]
        current_k_rope=k_rope,                                    # [T, D_rope]
        current_v=v,                                               # [T, A, D_v]
        kv_cache=kv_cache,
        selected_indices=selected,
        forward_batch=forward_batch,
    )                                                             # [T, A, D_v]

    ctx = ctx.reshape(T, A * D_v)                                 # [T, A * D_v]
    out = o_proj(ctx)                                             # [T, H]
    return out
```

### 5.5 DSA Indexer

```python
def dsa_indexer(layer_id, q_lora, positions, forward_batch):
    """
    q_lora: [T, R_q]

    return:
      selected_indices: [T, index_heads or A, K_dsa]
    """
    index_q = indexer_q_proj(q_lora)                              # [T, index_n_heads * index_head_dim]
    index_q = index_q.reshape(T, index_n_heads, index_head_dim)   # [T, index_heads, index_D]
    index_q = apply_indexer_rope(index_q, positions)

    index_k_cache = dsa_index_k_cache[layer_id]
    logits = paged_mqa_logits(index_q, index_k_cache)             # [T, history_len]
    selected_indices = topk(logits, k=K_dsa)                      # [T, K_dsa] 或带 heads 维
    return selected_indices
```

### 5.6 Dense MLP / MoE

```python
def dense_mlp(x):
    """
    x: [T, H]
    """
    gate_up = gate_up_proj(x)                                     # [T, 2 * intermediate_size]
    gate, up = gate_up.chunk(2, dim=-1)
    y = silu(gate) * up
    return down_proj(y)                                           # [T, H]


def moe_ffn(layer_id, x, forward_batch):
    """
    x: [T, H]
    """
    router_logits = gate(x)                                       # [T, E], E=256
    expert_ids, expert_w = topk(router_logits, k=K_exp)           # [T, 8], [T, 8]
    expert_w = normalize_topk_prob(expert_w)

    routed = fused_experts(
        x,
        expert_ids=expert_ids,
        expert_weights=expert_w,
    )                                                             # [T, H]

    shared = shared_experts(x)                                    # [T, H]
    out = routed_scaling_factor * routed + shared                 # [T, H]
    return out
```

---

## 6. Attention：MLA + DSA

### 6.1 MLA 是什么

MLA = Multi-head Latent Attention。它的核心不是减少 KV heads，而是把 KV cache 存成低秩 latent：

```text
x [T,H]
  │
  ▼
kv_a_proj_with_mqa
  │
  ├─► kv_latent [T, kv_lora_rank]
  └─► k_rope    [T, qk_rope_head_dim]

kv_latent
  │
  ▼
kv_a_layernorm
  │
  ▼
kv_b_proj
  │
  ├─► K_nope [T, n_heads, qk_nope_head_dim]
  └─► V      [T, n_heads, v_head_dim]

K = concat(K_nope, K_rope)
```

形状约定：

```text
T = token 数
H = hidden_size = 6144
A = num_attention_heads = 64
R_kv = kv_lora_rank = 512
D_nope = qk_nope_head_dim = 192
D_rope = qk_rope_head_dim = 64
D_qk = D_nope + D_rope = 256
D_v = v_head_dim = 256
```

直观形状：

```text
x          : [T, H]
kv_latent  : [T, R_kv]
k_rope     : [T, D_rope]
K_nope     : [T, A, D_nope]
K_rope     : [T, 1, D_rope] 或 broadcast 到 heads
K          : [T, A, D_qk]
V          : [T, A, D_v]
```

decode 时，runtime 可以缓存更小的 latent / RoPE 表示，而不是普通 MHA 的完整 K/V。

### 6.2 Q 路径

Q 也走低秩路径：

```text
x [T,H]
  │
  ▼
q_a_proj
  │
  ▼
q_lora [T,q_lora_rank]
  │
  ▼
q_a_layernorm
  │
  ▼
q_b_proj
  │
  ▼
Q [T,A,D_qk]
```

其中：

```text
q_lora_rank = 2048
D_qk = 256
A = 64
Q 展开后总宽度 = 64 * 256 = 16384
```

### 6.3 RoPE / nope 拆分

和 DeepSeek MLA 系模型类似，Q/K 分成 nope 与 RoPE 两部分：

```text
Q = concat(Q_nope, Q_rope)
K = concat(K_nope, K_rope)

Q_nope / K_nope: 192 dims
Q_rope / K_rope: 64 dims
```

注意力分数可以理解成：

```text
score = dot(Q_nope, K_nope) + dot(RoPE(Q_rope), RoPE(K_rope))
```

`nope` 保留内容相似度，`rope` 注入位置信息。

### 6.4 DSA：动态稀疏注意力

DSA = Dynamic Sparse Attention。它的目标是长上下文时不让每个 query 都全量扫完整历史。

教学版流程：

```text
当前 query
  │
  ▼
DSA indexer
  │
  ├─ 生成 indexer query
  ├─ 和历史 indexer key 打分
  └─ top-k 选择历史位置 / block
        │
        ▼
sparse attention 只读被选中的历史
```

关键参数：

| 字段 | 值 | 含义 |
|---|---:|---|
| `index_topk` | 2048 | 每个 query 最多选择的历史候选 |
| `index_head_dim` | 128 | indexer key/query 维度 |
| `index_n_heads` | 32 | indexer heads |

注意：DSA indexer 是“找哪些历史值得看”的轻量检索结构；MLA latent KV 是“被 attention 真正读取的内容表示”。两者角色不同。

`Indexer` 和 `IndexerKPool` 的选择只看 config：

```python
indexer_cls = IndexerKPool if getattr(config, "index_kpool", 1) > 1 else Indexer
```

所以 GLM-5.3 text-only 路径是否实际用 KPool，取决于 checkpoint `config.json` 里的 `index_kpool`。没有该字段或值为 `1` 时，就是普通 DSA `Indexer`。

---

## 7. FFN / MoE

GLM-5.3 的 FFN 侧是 dense 前缀 + MoE 主体：

```text
first_k_dense_replace = 3
```

| 层范围 | FFN 类型 |
|---|---|
| `0..2` | dense MLP |
| `3..77` | MoE |

MoE 常见配置：

| 字段 | 值 | 含义 |
|---|---:|---|
| `n_routed_experts` | 256 | routed experts 总数 |
| `num_experts_per_tok` | 8 | 每 token 激活 experts |
| `n_shared_experts` | 1 | 共享专家 |
| `moe_intermediate_size` | 2048 | 单 expert 中间维 |
| `routed_scaling_factor` | 2.5 | routed 输出缩放 |
| `norm_topk_prob` | true | 归一化 top-k gate 概率 |
| `n_group` / `topk_group` | 1 / 1 | expert group 配置 |

教学版 MoE：

```python
def moe_ffn_teaching(x):
    """
    x: [T,H]
    """
    scores = router(x)                         # [T, 256]
    topk_ids, topk_w = topk(scores, k=8)        # [T, 8]
    topk_w = normalize(topk_w)                  # norm_topk_prob=True

    routed = 0
    for expert_id, weight in zip(topk_ids, topk_w):
        routed += weight * expert[expert_id](x) # each expert: [T,H]

    shared = shared_expert(x)                   # [T,H]
    return routed_scaling_factor * routed + shared
```

---

## 8. MTP / NextN

不少 GLM-5.3 serving checkpoint 会带一个额外 MTP / next-token prediction draft layer。它通常不算在 78 层主干里，可以理解成 speculative decoding 用的辅助头：

```text
主干 decoder layers: 0..77
MTP draft layer    : layer 78 / nextn
```

作用：

1. 预测未来多个 token 草稿。
2. 给 target model 验证，提高 decode 吞吐。
3. 结构上可能有自己的 attention / MLP / norms / shared head。

读 architecture 时建议先忽略 MTP，把 78 层主干看清楚，再单独看 draft layer。

---

## 9. 和 GLM-5.3-Flash 的区别

这点很重要：

| 项 | GLM-5.3 text-only | GLM-5.3-Flash |
|---|---|---|
| 主架构 | `GlmMoeDsaForCausalLM` | `Glm5NextForConditionalGeneration` |
| 总参数 | 约 753B | 320B |
| 激活参数 | MoE top-8，具体随实现统计 | 18B |
| 层数 | 78 main layers | 45 main layers |
| Attention | MLA + DSA | KDA + KPool-DSA hybrid |
| hidden size | 6144 | 4096 |
| routed experts | 256 | 288 |
| 模态 | text-only | natively multimodal |
| mHC | 本文主干不按 mHC 梳理 | 4-stream mHC |

所以：

```text
models/glm-5.3/architecture.md      -> 本文，GLM-5.3 text-only
如果后面写 GLM-5.3-Flash        -> 应单独建 flash 架构笔记
```

---

## 10. 读代码 / 配置推荐顺序

1. `config.json`
   - 先确认 `architectures`、`num_hidden_layers`、`hidden_size`、`first_k_dense_replace`。
2. `GlmMoeDsaForCausalLM`
   - 在 `glm4_moe.py` 里只是 GLM DSA 入口，实际继承 `DeepseekV2ForCausalLM`。
3. `DeepseekV2DecoderLayer`
   - 看 layer 内 attention / MoE / residual / norm 顺序。
4. `DeepseekV2AttentionMLA`
   - 重点看 `q_lora_rank`、`kv_lora_rank`、RoPE/nope、`Indexer` / `IndexerKPool`。
5. `DeepseekV2MoE`
   - 看 router、top-8、shared expert、dense 前缀层。
6. `layers/attention/dsa/`
   - 看 DSA indexer、top-k、KPool、prefill / decode metadata。
7. MTP / NextN
   - 最后再看 speculative draft layer。

---

## 11. 最小心智模型

只记一张图：

```text
GLM-5.3 text-only
  = 78-layer Decoder-only Transformer
  + MLA latent KV cache
  + DSA sparse historical retrieval
  + 3 dense FFN layers
  + 75 MoE layers, 256 routed experts, top-8 active
  + optional MTP draft layer
```

只记一句话：

> GLM-5.3 用 MLA 把 KV 表示压低，用 DSA 在长上下文里稀疏选择历史，再用大 MoE 提供总容量。

---

*整理日期：2026-09-07。*
