# Qwen3.8-Flash-Next Architecture Notes

> 本目录对应 **Qwen3.8-Flash-Next**（`Qwen/Qwen3.8-Flash-Next`），是 Qwen4 架构预览，**不是** 旧的 `Qwen3-Next-80B-A3B`。
>
> 相对 Qwen3.5→Qwen3.8 一脉的「GDN + Gated Attention」，Flash-Next 把 full-attention 槽位换成 **QSA（Qwen Sparse Attention）**，并加上 **Gated Residual（HC）** 与 **N-gram / PLE**。
>
> 专题：[qsa.md](./qsa.md) · [ple.md](./ple.md)
>
> HF：`model_type=qwen4_exp`，`architectures=Qwen4ExpForConditionalGeneration`。SGLang 有 day-0 支持（Cookbook / LMSYS 博客）；本地树若尚未合入，以 HF config + 公开架构说明为准。

## 入口

```text
Qwen4ExpForConditionalGeneration
├── vision_config / vision tower          # 多模态
└── text_config (qwen4_exp_text)
    ├── embed + PLE (n-gram, 常在 layer 2)
    ├── layers[48]: 3×GDN + 1×QSA 循环
    ├── Gated Residual / HyperConnection（hc_count=4）
    ├── Sparse MoE（每层）
    └── lm_head + MTP
```

## Config（官方 `Qwen3.8-Flash-Next`）

数值来自 HuggingFace `config.json` 的 `text_config`。

### 一览

| 项 | 值 |
|---|---|
| 主干参数 / 激活 | ~125B / ~6B（另有 ~51B N-gram 表，常放 host） |
| `num_hidden_layers` | 48 |
| `hidden_size` | 2560 |
| Hybrid | 36 GDN + 12 QSA（`full_attention_interval=4`） |
| `vocab_size` | 248320 |
| `max_position_embeddings` | 262144 |
| `model_type`（text） | `qwen4_exp_text` |

### Full / Sparse Attention（QSA 槽位）

`layer_types` 里仍标 `"full_attention"`，实现上是 **QSA**：indexer 选 block，再对选出的原始 K/V 做稀疏 GQA。

| 字段 | 值 | 含义 |
|---|---|---|
| `num_attention_heads` | 24 | Q heads |
| `num_key_value_heads` | 2 | KV heads |
| `head_dim` | 256 | |
| `partial_rotary_factor` | 0.25 | RoPE 维 64 |
| `output_gate_type` | sigmoid | |
| `indexer_n_heads` | 4 | indexer query heads |
| `indexer_kv_heads` | 1 | indexer 共享 key |
| `indexer_head_dim` | 128 | |
| `indexer_compress_ratio` | 4 | c4：每 4 token 压成 1 个 index key |
| `indexer_budget` | 2048 | 展开后最多约 2048+残块 token |

### Linear Attention（GDN）

| 字段 | 值 |
|---|---|
| `linear_num_key_heads` | 16 |
| `linear_num_value_heads` | 48 |
| `linear_key_head_dim` / `linear_value_head_dim` | 128 |
| `linear_conv_kernel_dim` | 4 |
| `mamba_ssm_dtype` | float32 |

### MoE

| 字段 | 值 |
|---|---|
| `num_experts` | 512 |
| `num_experts_per_tok` | 10 |
| `moe_intermediate_size` | 640 |
| `shared_expert_intermediate_size` | 640 |

### Gated Residual / HyperConnection

| 字段 | 值 | 含义 |
|---|---|---|
| `hc_count` | 4 | 残差支路数 |
| `hc_lowrank` | 320 | Mix 低秩投影维 |

每层 token-mix / MoE 前 **Mix**（4 路 → 1 个 hidden），后 **Combine**（写回 4 路）。

### N-gram / PLE

| 字段 | 值 | 含义 |
|---|---|---|
| `ngram_size` | 3 | 最高 3-gram |
| `ngram_vocab_size_base` | 20000000 | 表规模基数 |
| `heads_per_ngram` | 8 | 每类 n-gram 的 hash head 数 |
| `ple_layer_ids` | [2] | 插入位置（配置层号） |
| `ple_embed_dim` | 2560 | 与 H 对齐 |
| `ple_conv_kernel_size` | 4 | PLE 短卷积 |
| `split_ngram_parts` | 128 | |
| `make_ngram_vocab_size_divisible_by` | 128 | |

### MTP / RoPE / Vision（壳）

| 字段 | 值 |
|---|---|
| `mtp_num_hidden_layers` | 1 |
| `mtp.hybrid` | true（draft 可配独立 `layer_types`） |
| `rope_parameters.mrope_section` | [11,11,10]（多模态 RoPE） |
| `rope_theta` | 1e7 |
| vision `hidden_size` / `depth` | 1152 / 27 |
| vision `out_hidden_size` | 2560（对齐 text H） |

## 架构要点

### 1. GDN + QSA（3:1）

```text
[GDN → MoE] ×3 → [QSA → MoE] ×1   重复 12 组 → 48 层
```

- **GDN**：固定大小 conv + SSM 状态，长度线性。
- **QSA**：indexer 在压缩 key（c4）上打分 → top blocks → 对**原始** K/V 做稀疏 GQA；budget≈2048。

### 2. Gated Residual（HC）

```text
R ∈ [T, 4, H]
  Mix(R) → h ∈ [T, H]     # 低秩门控读
  block(h) → y
  Combine(R, y) → R'       # 写回 4 支路
```

**`hc_mix(R)`** = 从 4 支路 **读出** 一个 `[T,H]`：

```python
def hc_mix(R, hc_count=4, hc_lowrank=320):
    """
    R: [T, 4, H]
    return h: [T, H]   # 本层 token-mix（GDN/QSA）的输入
    """
    # 低秩投影得到门控（及混合系数）；官方 hc_lowrank=320
    # 概念上：对 4 路做内容相关的加权，再收成一路
    gates = sigmoid(W_gate_down(R))           # 经 lowrank 再展开，示意
    # 或：lowrank → 每路 / 逐维 gate，实现里常融进单个 GEMM epilogue
    h = (gates * R).sum(dim=1)                # [T, H]
    # 真实 kernel：SiLU/Sigmoid + 门控 + reduce 可融在 Mix GEMM 里
    return h
```

对称的是 **`hc_combine(R, y)`**：本层输出 `y` 经注入系数写回 4 支路，得到下一层的 `R'`。一层 = `h=Mix(R)` → block → `R=Combine(R,y)`。

### 3. PLE / N-gram Embedding

详解见 [ple.md](./ple.md)。要点：

- 约 **51B 预训练学好的**独立查表（非 token emb 拷贝）；推理冻结，常 pinned host + 异步 gather。
- 每 token：raw id 做 int64 混合+XOR，8 bigram + 8 trigram head 取模 → 16 行拼成 `[H]`。
- `Gate(Norm(Q), Norm(K))` ≈ scaled dot → signed-sqrt → sigmoid，再乘 `V` 写入 HC；详见 ple.md。
- 作用：主干外加局部短语容量，几乎不涨 per-token FLOPs。

## 整网伪代码（概念）

```python
def qwen38_flash_next_forward(input_ids, positions, mm_inputs, forward_batch):
    x = embed_tokens(input_ids)                 # [T, H]
    R = init_hc_branches(x)                     # [T, hc_count, H]

    for layer_idx in range(48):
        if layer_idx in ple_layer_ids:
            R = inject_ple(R, ngram_lookup(input_ids))

        h = hc_mix(R)                           # [T, H]
        if layer_types[layer_idx] == "linear_attention":
            y = gated_delta_net(h, forward_batch)
        else:
            y = qsa_attention(h, positions, forward_batch)
            # indexer → top blocks → sparse GQA on full K/V
        y = sparse_moe(y)
        R = hc_combine(R, y)

    x = hc_readout(R)
    x = final_norm(x)
    return logits_processor(...)
```

## 和 Qwen3.8 / 旧 Qwen3-Next 的区别

| | Qwen3-Next-80B-A3B | Qwen3.8-2.4T-A95B | **Qwen3.8-Flash-Next** |
|---|---|---|---|
| 本目录？ | 否 | 见 `qwen-3.8/` | **是** |
| Attention 槽 | Gated GQA | Gated GQA | **QSA** |
| Residual | 单流 | 单流 | **HC 4 支路** |
| 额外容量 | 无 | 无 | **N-gram/PLE ~51B** |
| H / L | 2048 / 48 | 8192 / 92 | **2560 / 48** |
| 激活 | ~3B | ~95B | **~6B** |

## 一句话总结

Qwen3.8-Flash-Next 是 Qwen4 预览：在 3:1 GDN 混排上，把全局层换成 QSA，用 4 路 Gated Residual 加宽信息流，并用 host 侧 N-gram embedding 换容量；推理栈按 `qwen4_exp` 多模态壳 + text hybrid 来接。
