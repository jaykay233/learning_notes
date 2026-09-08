# Qwen3.8 Architecture Notes

> 本目录对应 **Qwen3.8**（旗舰开源 MoE，如 `Qwen/Qwen3.8-2.4T-A95B`），**不是** Qwen3-8B dense。
>
> 架构上与 Qwen3.5 MoE text 同源：HF `model_type=qwen3_5_moe_text`，类名 `Qwen3_5MoeForCausalLM`。SGLang 实现见 `qwen3_5.py` / `qwen3_5_text.py`（复用 Qwen3-Next 的 hybrid 骨架再放大）。

## 入口类

```python
# 多模态壳（若 checkpoint 带 vision）
Qwen3_5MoeForConditionalGeneration
# text-only
Qwen3_5MoeForCausalLM / Qwen3_5ForCausalLM（text 包装）
```

模块树（text backbone）：

```text
Qwen3_5MoeForCausalLM
├── embed_tokens
├── layers[L]
│   ├── linear_attention 层: Qwen3_5LinearDecoderLayer
│   │   ├── linear_attn: Qwen3_5GatedDeltaNet
│   │   └── mlp: Sparse MoE
│   └── full_attention 层: Qwen3_5AttentionDecoderLayer
│       ├── self_attn: Gated GQA (+ Q/K norm, partial RoPE)
│       └── mlp: Sparse MoE
├── norm
└── lm_head
```

层类型由 `layer_types` / `full_attention_interval` 决定：**每 4 层里 3 个 GDN + 1 个 Gated Attention**，每层后都接 MoE。

## Config（官方 `Qwen3.8-2.4T-A95B`）

数值来自 HuggingFace `Qwen/Qwen3.8-2.4T-A95B` 的 `config.json`。

### 一览

| 项 | 值 |
|---|---|
| 总参数 / 激活 | ~2.4T / ~95B（官方 A95B） |
| `num_hidden_layers` | 92 |
| `hidden_size` | 8192 |
| Hybrid | 69 GDN + 23 Gated Attention（`full_attention_interval=4`） |
| `vocab_size` | 248320 |
| `max_position_embeddings` | 262144（可扩展到约 1M） |
| `model_type` | `qwen3_5_moe_text` |
| `architectures` | `Qwen3_5MoeForCausalLM` |
| MTP | `mtp_num_hidden_layers=1` |

### Full Attention（Gated GQA）

| 字段 | 值 | 含义 |
|---|---|---|
| `num_attention_heads` | 64 | Q heads |
| `num_key_value_heads` | 4 | KV heads（GQA） |
| `head_dim` | 256 | |
| `partial_rotary_factor` | 0.25 | RoPE 维 = 64 |
| `attn_output_gate` | true | `ctx * gate` 再 `o_proj` |
| `output_gate_type` | swish | |
| `rope_parameters.rope_theta` | 1e7 | |
| `attention_bias` | false | |
| `rms_norm_eps` | 1e-6 | 含 Q/K norm |

### Linear Attention（Gated DeltaNet）

| 字段 | 值 | 含义 |
|---|---|---|
| `linear_num_key_heads` | 16 | Q/K heads |
| `linear_num_value_heads` | 128 | V heads |
| `linear_key_head_dim` | 128 | |
| `linear_value_head_dim` | 128 | |
| `linear_conv_kernel_dim` | 4 | short conv |
| `mamba_ssm_dtype` | float32 | SSM 状态精度 |

### MoE

| 字段 | 值 | 含义 |
|---|---|---|
| `num_experts` | 512 | routed |
| `num_experts_per_tok` | 10 | top-k |
| `moe_intermediate_size` | 2048 | expert FFN 中间维 |
| `shared_expert_intermediate_size` | 2048 | shared expert |
| `router_aux_loss_coef` | 0.001 | |
| `output_router_logits` | false | |

### 层类型

```python
# 官方 layer_types：重复 [linear, linear, linear, full] × 23
# 等价：
for l in range(92):
    if (l + 1) % 4 == 0:
        "full_attention"      # 23 层
    else:
        "linear_attention"    # 69 层
```

形状速记：

```text
H=8192, L=92
GQA:  64 Q / 4 KV / D=256；RoPE 64 维；output gate
GDN:  K_heads=16, V_heads=128, dim=128, conv=4
MoE:  512 experts, top-10, intermediate=2048
```

## 整网伪代码

```python
def qwen3_8_text_forward(input_ids, positions, forward_batch):
    x = embed_tokens(input_ids)                 # [T, H]
    residual = None

    for layer_idx in range(L):
        kind = layer_types[layer_idx]           # linear_attention / full_attention
        if kind == "full_attention":
            x, residual = attention_decoder_layer(
                x, residual, positions, forward_batch
            )
        else:
            x, residual = linear_decoder_layer(
                x, residual, forward_batch
            )
        # 两种层内部都是：token-mix → Sparse MoE

    x, _ = final_norm(x, residual)
    return logits_processor(input_ids, x, lm_head, forward_batch)
```

## Full Attention 层

与 Qwen3-Next / Qwen3.5 同族：GQA + Q/K RMSNorm + partial RoPE + **attn output gate**。

```text
x → RMSNorm → qkv(+gate) → q/k norm → RoPE(部分维)
  → RadixAttention → ctx * gate → o_proj → MoE
```

cache：普通 paged KV（仅 full attention 层增长）。

## Linear Attention 层 / GDN

```text
x → RMSNorm → in_proj_qkvz / in_proj_ba
  → short conv → RadixLinearAttention(q,k,v,a,b)
  → gated RMSNorm(z) → out_proj → MoE
```

cache：conv window + recurrent SSM 状态（大小与序列长度无关）。

## MoE

每层（含 GDN 与 full attn）后接 sparse MoE：512 experts、top-10、另有 shared expert。路由与 FFN 中间维见上表。

## 和易混模型的区别

| | Qwen3-8B dense | Qwen3-Next-80B-A3B | **Qwen3.8-2.4T-A95B** |
|---|---|---|---|
| 本目录？ | 否 | 否（见 `qwen3.8-next` 的前身） | **是** |
| 结构 | 全层 GQA + dense MLP | 3:1 GDN/GQA + MoE | 同 hybrid，但规模更大 |
| H / L | 4096 / 36 | 2048 / 48 | **8192 / 92** |
| 激活 | ~8B | ~3B | **~95B** |

## 一句话总结

Qwen3.8 是 Qwen3.5 MoE text 骨架上的旗舰放大版：69 层 Gated DeltaNet + 23 层 Gated GQA，层层 Sparse MoE（512/top-10），HF 仍挂在 `qwen3_5_moe_text` / `Qwen3_5MoeForCausalLM` 下。
