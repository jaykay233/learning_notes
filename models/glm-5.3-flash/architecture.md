# GLM-5.3-Flash Architecture Notes

> 这份笔记按 SGLang 的 `glm5_next.py` / `glm5_next.py config` 来读 GLM-5.3-Flash 路径。SGLang 类名使用 `Glm5Next*`，可以理解为 GLM5 Next/Flash 这一套 hybrid 架构；它和前面 `glm-5.3/architecture.md` 里的 text-only `GlmMoeDsaForCausalLM -> DeepseekV2ForCausalLM` 路径不是同一个主干。

## 入口类

SGLang 入口：

```python
EntryClass = [Glm5NextForConditionalGeneration]
```

整体模块树：

```text
Glm5NextForConditionalGeneration
├── visual: Glm5NextVisionModel                 # 可选，language_only 时没有
├── model: Glm5NextModel                        # text backbone
│   ├── embed_tokens
│   ├── layers[N]: Glm5NextDecoderLayer
│   │   ├── self_attn
│   │   │   ├── Glm5NextLinearAttention         # KDA / linear attention layer
│   │   │   └── DeepseekV2AttentionMLA          # MLA full attention layer
│   │   ├── input_layernorm / post_attention_layernorm
│   │   ├── layer_communicator
│   │   │   ├── LayerCommunicator
│   │   │   └── MHCLayerCommunicator            # mhc=True 时
│   │   └── mlp
│   │       ├── Glm5NextMoE
│   │       └── Glm5NextMLP
│   └── norm
├── lm_head
└── logits_processor
```

入口同时支持：

- `language_only=True`：只跑 text backbone。
- `encoder_only=True`：只跑视觉 encoder。
- 默认多模态：图片/视频特征经 `Glm5NextVisionModel` 对齐后混入语言模型 token embedding。

## Text Config 关键字段

`Glm5NextTextConfig` 默认值里能看出这套结构的重点：

```python
hidden_size = 4096
num_hidden_layers = 45
num_attention_heads = 64
head_dim = hidden_size // num_attention_heads or config.head_dim

# MLA
q_lora_rank = 1536
kv_lora_rank = 512
qk_nope_head_dim = 256
qk_rope_head_dim = 0
v_head_dim = 256

# MoE
n_routed_experts = 288
num_experts_per_tok = 7
n_shared_experts = 1
first_k_dense_replace = 3
moe_layer_freq = 1

# KDA / linear attention
linear_head_dim = 128
linear_num_heads = 64
linear_conv_kernel_dim = 4

# mHC
mhc = False or True
hc_mult = 4
hc_sinkhorn_iters = 20
hc_eps = 1e-6
```

真实 checkpoint 是否开启 `mhc`、具体 `layer_types`、`index_topk`、`index_kpool` 等字段，以 checkpoint `config.json` 为准。

## Attention Layer 怎么选

`Glm5NextTextConfig` 会构造 `linear_attn_config`：

```python
if layer_types is None:
    kda_layers = [
        layer_idx
        for layer_idx in range(num_hidden_layers)
        if layer_idx % 4 != 3
    ]
else:
    kda_layers = [
        layer_idx
        for layer_idx, layer_type in enumerate(layer_types)
        if layer_type == "linear_attention"
    ]

full_attn_layers = [
    layer_idx
    for layer_idx in range(num_hidden_layers)
    if layer_idx not in kda_layers
]
```

也就是默认模式下：

```text
layer 0: KDA
layer 1: KDA
layer 2: KDA
layer 3: MLA full attention
layer 4: KDA
layer 5: KDA
layer 6: KDA
layer 7: MLA full attention
...
```

`Glm5NextDecoderLayer` 里实际选择：

```python
self.is_linear_attn = config.is_kda_layer(layer_id)

if self.is_linear_attn:
    self.self_attn = Glm5NextLinearAttention(...)
else:
    self.self_attn = DeepseekV2AttentionMLA(...)
```

## MLP / MoE 怎么选

和 DeepSeek/GLM text-only 类似，前若干层可以是 dense MLP，之后按 MoE 频率启用 MoE：

```python
def is_layer_sparse(layer_id, is_nextn):
    return is_nextn or (
        config.n_routed_experts is not None
        and layer_id >= config.first_k_dense_replace
        and layer_id % config.moe_layer_freq == 0
    )

if is_layer_sparse:
    mlp = Glm5NextMoE(...)
else:
    mlp = Glm5NextMLP(...)
```

默认字段下，`first_k_dense_replace=3`，所以前 3 层是 dense MLP，后面基本是 MoE。

## 整网伪代码

设：

- `T`：本次 forward token 数。
- `H`：hidden size。
- `L`：层数。
- `x: [T, H]`。
- `positions: [T]`，启用 mRoPE 时可能来自 `forward_batch.mrope_positions`。

```python
def glm5_flash_forward(input_ids, positions, forward_batch, input_embeds=None):
    if is_mrope_enabled:
        positions = forward_batch.mrope_positions

    # 多模态入口会先把 image/video embedding 合并进 input_embeds
    if input_embeds is None:
        x = embed_tokens(input_ids)             # [T, H]
    else:
        x = input_embeds                        # [T, H]

    residual = None
    topk_indices = None                         # DSA / indexer 可复用 topk

    for layer_id in range(L):
        x, residual, topk_indices = decoder_layer(
            positions=positions,
            hidden_states=x,
            forward_batch=forward_batch,
            residual=residual,
            prev_topk_indices=topk_indices,
        )

    if residual is None:
        x = norm(x)                             # [T, H]
    else:
        x, _ = norm(x, residual)                # [T, H]

    logits = lm_head(x)                         # [T, vocab]
    return logits
```

## Decoder Layer 伪代码

```python
def glm5_next_decoder_layer(x, residual, positions, forward_batch, prev_topk_indices):
    # x: [T, H]

    x, residual = layer_communicator.prepare_attn(
        x,
        residual,
        forward_batch,
    )

    if is_linear_attn:
        attn_out = glm5_kda_attention(x, forward_batch)        # [T, H]
        topk_indices = None
    else:
        attn_out = mla_attention(
            positions=positions,
            hidden_states=x,
            forward_batch=forward_batch,
            prev_topk_indices=prev_topk_indices,
        )
        if returns_tuple(attn_out):
            attn_out, topk_indices = attn_out
        else:
            topk_indices = None

    x, residual = layer_communicator.prepare_mlp(
        attn_out,
        residual,
        forward_batch,
    )

    if is_sparse_moe_layer:
        x = Glm5NextMoE(x, forward_batch)                      # [T, H]
    else:
        x = Glm5NextMLP(x)                                     # [T, H]

    x, residual = layer_communicator.postprocess_layer(
        x,
        residual,
        forward_batch,
    )

    return x, residual, topk_indices
```

如果 `config.mhc=True`，`layer_communicator` 换成 `MHCLayerCommunicator`，prepare/postprocess 里会多做 mHC 的 pre/post。

## KDA / Glm5NextLinearAttention

> `RadixLinearAttention` 层壳、KDA gated delta rule 递推与 cache 形态，详见同目录 [radix_linear_attention.md](./radix_linear_attention.md)。

GLM5-Flash 的 KDA 层和 Kimi-K3 很像，走 `RadixLinearAttention`：

```text
hidden [T, H]
├── qkv_proj or fused_qkvbfg_a_proj
│   ├── q [T, A * D]
│   ├── k [T, A * D]
│   └── v [T, A * D]
├── b_proj / f_a_proj / f_b_proj
│   ├── beta        [T, A]
│   └── forget_gate [T, A * D]
├── g_a_proj / g_b_proj
│   └── output_gate [T, A * D]
├── qkv_conv1d state
├── RadixLinearAttention(mixed_qkv, a=forget_gate, b=beta)
├── FusedRMSNormGated(core, output_gate)
└── o_proj -> [T, H]
```

伪代码：

```python
def glm5_kda_attention(x, forward_batch):
    # x: [T, H]

    if do_fuse_qkvbfg:
        mixed_qkv, beta, forget_gate, gate = fused_qkvbfg(x)
    else:
        mixed_qkv = qkv_proj(x)
        beta = b_proj(x)
        forget_gate = f_b_proj(f_a_proj(x))
        gate = g_b_proj(g_a_proj(x))

    if not forward_batch.forward_mode.is_decode():
        forget_gate = forget_gate.unsqueeze(0)

    beta = beta.unsqueeze(0)

    core = RadixLinearAttention(
        forward_batch,
        mixed_qkv=mixed_qkv,
        a=forget_gate,
        b=beta,
    )                                           # [1, T, A, D] or equivalent

    gate = gate.unflatten(-1, (-1, D))          # [T, A, D]
    core = FusedRMSNormGated(core, gate)
    core = core.squeeze(0).flatten(-2)          # [T, A * D]

    return o_proj(core)                         # [T, H]
```

KDA 层保存的是 linear attention state / conv state，不是 dense KV cache。

## MLA Full Attention

full attention 层直接使用 `DeepseekV2AttentionMLA`：

```text
hidden [T, H]
├── q_a_proj -> q_a_layernorm -> q_b_proj
│   └── q_nope / q_rope
├── kv_a_proj_with_mqa -> kv_a_layernorm -> kv_b_proj
│   └── kv_latent / k_rope -> k_nope / v
├── optional DSA indexer / KPool
├── MLA attention
└── o_proj -> [T, H]
```

这里是 MLA latent KV cache，不是 Qwen3 dense 那种每个 KV head 完整缓存。

形状：

```python
q_latent = q_a_proj(x)                         # [T, q_lora_rank]
q_latent = q_a_layernorm(q_latent)
q_full = q_b_proj(q_latent)                    # [T, A * (D_nope + D_rope)]

kv_pack = kv_a_proj_with_mqa(x)                # [T, kv_lora_rank + D_rope]
kv_latent = kv_pack[..., :kv_lora_rank]        # [T, kv_lora_rank]
k_rope = kv_pack[..., kv_lora_rank:]           # [T, D_rope]

ctx = mla_attention(q_full, kv_latent, k_rope)
out = o_proj(ctx)                              # [T, H]
```

## DSA / Indexer

`Glm5NextForConditionalGeneration` 会通过：

```python
self.use_dsa = is_deepseek_dsa(text_config)
get_attn_tp_context().init_context(q_lora_rank, self.use_dsa, text_config.mhc)
```

来告诉 MLA attention 是否启用 DSA/indexer 路径。相关 config 字段包括：

```python
index_head_dim
index_topk
index_kpool
index_kpool_always_select_tail
index_kpool_compress
index_n_heads
index_topk_freq
index_topk_pattern
index_skip_topk_offset
index_share_for_mtp_iteration
indexer_rope_interleave
```

概念伪代码：

```python
def mla_with_dsa(q, kv_latent_cache, k_rope_cache, prev_topk_indices):
    if should_recompute_topk(layer_id, index_topk_freq):
        index_q = make_index_query(q)                   # [T, index_n_heads, index_head_dim]
        index_k = read_index_k_cache()                  # [S, index_n_heads, index_head_dim]
        topk_indices = topk(index_q @ index_k.T, k=index_topk)
    else:
        topk_indices = reuse_or_shift(prev_topk_indices, index_skip_topk_offset)

    selected_kv = gather(kv_latent_cache, topk_indices)
    selected_rope = gather(k_rope_cache, topk_indices)

    ctx = mla_attention_over_selected_blocks(q, selected_kv, selected_rope)
    return ctx, topk_indices
```

这里的 DSA/indexer 只服务 full MLA attention 层；KDA linear attention 层走 recurrent state，不做这种 KV top-k 检索。

## mHC

`mhc=True` 时，每个 decoder layer 上会多出两组 mHC 参数：

```python
hc_attn_base, hc_attn_scale, hc_attn_fn
hc_ffn_base,  hc_ffn_scale,  hc_ffn_fn
```

并把普通 `LayerCommunicator` 换成 `MHCLayerCommunicator`：

```python
if config.mhc:
    layer_communicator = MHCLayerCommunicator(
        input_layernorm=input_layernorm,
        post_attention_layernorm=post_attention_layernorm,
        hc_attn_pre=hc_attn_pre,
        hc_ffn_pre=hc_ffn_pre,
        hc_post=hc_post,
        hc_mult=config.hc_mult,
    )
else:
    layer_communicator = LayerCommunicator(...)
```

层内直觉：

```python
def mhc_prepare_attn(x, residual):
    # 普通路径是 norm(x + residual)
    # mHC 路径是把 residual/hidden 映射到 hc_mult 倍宽的 manifold 表示
    h, h_res, h_post = hc_pre(
        x,
        hc_fn=hc_attn_fn,
        hc_scale=hc_attn_scale,
        hc_base=hc_attn_base,
    )
    return h, (h_res, h_post)

def mhc_post(x, residual, h_res, h_post):
    return hc_post_fn(
        x=x,
        residual=residual,
        h_post=h_post,
        h_res=h_res,
        hc_mult=hc_mult,
    )
```

`hc_mult=4` 时，内部 hidden 会临时扩到 `4H` 相关空间，最后再 contract 回普通 hidden stream。

## 多模态路径

默认不是 `language_only` 时：

```python
visual = Glm5NextVisionModel(config.vision_config)
```

forward 里使用 `general_mm_embed_routine`：

```python
def glm5_flash_multimodal_forward(input_ids, images_or_videos, positions, forward_batch):
    image_embeds = get_image_feature(images)       # [T_img, H]
    video_embeds = get_video_feature(videos)       # [T_vid, H]

    input_embeds = merge_mm_embeds(
        input_ids,
        image_embeds,
        video_embeds,
    )                                              # [T_total, H]

    hidden = Glm5NextModel(
        input_ids=input_ids,
        positions=positions,
        input_embeds=input_embeds,
        forward_batch=forward_batch,
    )

    return logits_processor(hidden)
```

启用 mRoPE 时：

```python
if is_mrope_enabled:
    positions = forward_batch.mrope_positions
```

也就是图片/视频 token 的位置信息不再只是 1D position，而是多维 RoPE position。

## DFlash / Aux Hidden Capture

SGLang 里还有：

```python
set_dflash_layers_to_capture(layer_ids)
```

它会：

```python
self.capture_aux_hidden_states = True
self.model.dflash_capture = True
self.model.layers_to_capture = [layer_id + 1 for layer_id in layer_ids]
```

如果 `dflash_capture=True` 且 `config.mhc=True`，捕获 aux hidden 前会先：

```python
aux_hidden_state = hc_contract(aux_hidden_state, hc_mult)
```

意思是：mHC 内部可能还在 widened hidden 表示里，给 draft/aux head 用之前要 contract 回普通 hidden 语义。

## 和 GLM-5.3 Text-Only 的差异

| 项 | GLM-5.3 text-only | GLM-5.3-Flash / Glm5Next |
|---|---|---|
| SGLang 入口 | `GlmMoeDsaForCausalLM` | `Glm5NextForConditionalGeneration` |
| attention | MLA + DSA，继承 DeepSeekV2 路径 | KDA linear attention + MLA full attention 混排 |
| linear attention | 无独立 KDA 层 | 默认大多数层是 `Glm5NextLinearAttention` |
| mHC | text-only 路径不作为核心出现 | `mhc=True` 时由 `MHCLayerCommunicator` 接管 |
| 多模态 | text-only | 可接 `Glm5NextVisionModel` |
| cache | MLA latent KV cache | KDA state + MLA latent KV cache 混合 |
| draft/aux | 普通 EAGLE/DSA capture | 有 `set_dflash_layers_to_capture` / `dflash_capture` |

## 一句话总结

GLM-5.3-Flash 在 SGLang 里是 `Glm5Next` 路径：多数层用 KDA linear attention 承担长上下文状态，周期性 full MLA attention 层配合 DSA/indexer 做精确检索，后接 dense/MoE FFN；开启 `mhc` 后，residual/norm 通道还会进入 mHC manifold 通信器。它和普通 GLM-5.3 text-only 路径不是小修小补，而是主干层类型和 cache 形态都变了。
