# Kimi-K3 Architecture Notes

> 这份笔记按 SGLang 的 `kimi_k3.py` / `kimi_linear.py` 来读 Kimi-K3。重点是 text backbone 的层内结构；多模态 wrapper 只作为入口关系说明。

## 入口类

SGLang 里 Kimi-K3 暴露两个入口：

```python
EntryClass = [KimiK3ForConditionalGeneration, KimiK3LinearForCausalLM]
```

- `KimiK3ForConditionalGeneration`：多模态入口，外面包一层视觉塔和 projector。
- `KimiK3LinearForCausalLM`：text-only causal LM，真正的语言模型主干。

整体可以先按下面这棵树理解：

```text
KimiK3ForConditionalGeneration
├── vision_tower: KimiK3VisionTower        # MoonViT3d
├── mm_projector: KimiK3MultiModalProjector
└── language_model: KimiK3LinearForCausalLM
    ├── model: KimiK3LinearModel
    │   ├── embed_tokens
    │   ├── layers[N]: KimiK3DecoderLayer
    │   └── norm
    └── lm_head
```

如果只研究 LLM 结构，可以从 `KimiK3LinearForCausalLM -> KimiK3LinearModel -> KimiK3DecoderLayer` 这条线往下看。

## Config

配置分两层：

- 多模态壳：`KimiK3Config`（`text_config` + `vision_config`）
- Text 主干：`KimiLinearConfig`（`text_config` 本体；SGLang 里 text-only 也直接用它）

源码：`sglang/srt/configs/kimi_k3.py`、`kimi_linear.py`。下面「官方 checkpoint」数值来自 HuggingFace `moonshotai/Kimi-K3` 的 `config.json`。

### 官方 Kimi-K3 一览（text）

| 项 | 值 |
|---|---|
| 总参数 / 激活 | ~2.8T / ~104B（官方宣称） |
| `num_hidden_layers` | 93 |
| `hidden_size` | 7168 |
| `num_attention_heads` | 96 |
| Attention 组成 | 69 KDA + 24 Gated MLA |
| `attn_res_block_size` | 12 → bank 行数 `cdiv(93,12)=8` |
| `vocab_size` | 163840 |
| `max_position_embeddings` | 1048576（1M） |
| `hidden_act` | `situ`（SiTU-GLU；`activation_situ_beta=4`，`activation_situ_linear_beta=25`） |
| MoE | 896 experts，top-16，2 shared；`first_k_dense_replace=1` |
| Latent MoE | `routed_expert_hidden_size=3584`，`moe_intermediate_size=3072` |
| 量化 | MXFP4 weights（compressed-tensors）；attn / shared / dense mlp / lm_head / vision 等 ignore |

层类型规律（1-based 列表）：

- **MLA（full）**：每 4 层一个，加最后一层 → `4,8,…,92,93`（24 层）
- **KDA**：其余 → `1,2,3,5,…,91`（69 层）
- 代码判断：`(layer_idx + 1) in kda_layers`（checkpoint 1-based，`layer_idx` 0-based）

### Text：`KimiLinearConfig` 字段分组

#### 主干尺寸

| 字段 | 官方值 | 含义 |
|---|---|---|
| `hidden_size` | 7168 | 残差流宽度 H |
| `intermediate_size` | 33792 | dense MLP 中间维 |
| `num_hidden_layers` | 93 | decoder 层数 L |
| `num_attention_heads` | 96 | MLA/通用 head 数（与 KDA `num_heads` 对齐） |
| `num_key_value_heads` | 96 | |
| `rms_norm_eps` | 1e-5 | |
| `rope_theta` / `rope_scaling` | 10000 / null | K3 MLA 开了 `mla_use_nope`，位置更多靠 KDA conv/decay |

#### MLA（full attn 层）

| 字段 | 官方值 | 含义 |
|---|---|---|
| `q_lora_rank` | 1536 | Q 低秩 |
| `kv_lora_rank` | 512 | KV latent |
| `qk_nope_head_dim` | 128 | |
| `qk_rope_head_dim` | 64 | |
| `v_head_dim` | 128 | |
| `mla_use_nope` | true | MLA 侧 NoPE |
| `mla_use_output_gate` | true | attn 输出 × sigmoid(gate) 再 `o_proj` |

#### KDA / `linear_attn_config`

| 字段 | 官方值 | 含义 |
|---|---|---|
| `kda_layers` | 69 个 1-based id | 走 `KimiK3DeltaAttention` |
| `full_attn_layers` | 24 个 1-based id | 走 `KimiK3MLAAttention` |
| `num_heads` | 96 | KDA head 数 A |
| `head_dim` | 128 | D；状态约 `A×D×D` / 层 |
| `short_conv_kernel_size` | 4 | causal conv 核宽 C |
| `use_full_rank_gate` | true | 全秩 output gate；fused `[q,k,v,g]` |
| `gate_lower_bound` | -5.0 | forget gate 下界 |

#### MoE / Latent MoE

| 字段 | 官方值 | 含义 |
|---|---|---|
| `num_experts` | 896 | routed experts |
| `num_experts_per_token` | 16 | top-k |
| `num_shared_experts` | 2 | |
| `moe_intermediate_size` | 3072 | 每个 expert FFN 中间维 |
| `routed_expert_hidden_size` | 3584 | Latent MoE：expert 在更窄空间算 |
| `latent_moe_use_norm` | true | |
| `first_k_dense_replace` | 1 | 前 1 层 dense MLP，之后按 freq 上 MoE |
| `moe_layer_freq` | 1 | 每层都是 MoE（在 dense replace 之后） |
| `moe_router_activation_func` | sigmoid | |
| `topk_method` | noaux_tc | |
| `routed_scaling_factor` | 1.0 | |
| `use_grouped_topk` / `num_expert_group` / `topk_group` | true / 1 / 1 | |

#### Attention Residual / 激活 / 其它

| 字段 | 官方值 | 含义 |
|---|---|---|
| `attn_res_block_size` | 12 | 非 null 即开启 AttnRes；每 12 层写一行 bank |
| `activation_situ_beta` | 4.0 | SiTU |
| `activation_situ_linear_beta` | 25.0 | |
| `num_nextn_predict_layers` | 0 | MTP 等；官方 checkpoint 为 0 |

### Vision：`KimiK3VisionConfig`（官方）

| 字段 | 值 | 含义 |
|---|---|---|
| `vt_hidden_size` | 1024 | MoonViT 隐层 |
| `vt_num_hidden_layers` | 27 | |
| `vt_num_attention_heads` | 12 | |
| `vt_intermediate_size` | 4096 | |
| `qkv_hidden_size` | 1536 | ViT QKV 宽度（可 ≠ vt_hidden） |
| `patch_size` | 14 | |
| `merge_kernel_size` | (2,2) | |
| `merge_type` | sd2_tpool | |
| `mm_projector_type` | patchmergerv2 | |
| `mm_hidden_size` | 1024 | projector 输入侧 |
| `text_hidden_size` | 7168 | 对齐到 LLM H |

多模态壳还有：`media_placeholder_token_id=163605`，`image_placeholder="<|kimi_image_placeholder|>"`。

### 和层选择相关的判定（复述）

```python
def is_kda_layer(layer_idx):
    return (
        linear_attn_config is not None
        and (layer_idx + 1) in linear_attn_config["kda_layers"]
    )

is_moe_layer = (
    config.is_moe
    and layer_idx >= config.first_k_dense_replace
    and layer_idx % config.moe_layer_freq == 0
)
```

## Decoder Layer

每一层的主结构：

```text
KimiK3DecoderLayer
├── input_layernorm
├── self_attn
│   ├── KimiK3DeltaAttention   # KDA / linear attention layer
│   └── KimiK3MLAAttention     # full attention / MLA layer
├── post_attention_layernorm
└── mlp
    ├── KimiK3MoE              # MoE layer
    └── KimiK3MLP              # dense MLP layer
```

attention 类型由 `config.is_kda_layer(layer_idx)` 决定：

```python
if config.is_kda_layer(layer_idx):
    self_attn = KimiK3DeltaAttention(...)
else:
    self_attn = KimiK3MLAAttention(...)
```

MLP 类型由 MoE predicate 决定：

```python
is_moe_layer = (
    config.is_moe
    and config.num_experts is not None
    and layer_idx >= config.first_k_dense_replace
    and layer_idx % config.moe_layer_freq == 0
)

if is_moe_layer:
    mlp = KimiK3MoE(...)
else:
    mlp = KimiK3MLP(...)
```

所以 Kimi-K3 的一层不是固定一种 attention，而是按照 config 混排：

```text
layer 0:  MLA or KDA + dense MLP / MoE
layer 1:  MLA or KDA + dense MLP / MoE
...
layer L:  MLA or KDA + dense MLP / MoE
```

## Text Forward 伪代码

对应 `KimiK3LinearForCausalLM.forward` → `KimiK3LinearModel.forward`（PP=1、忽略 SP/DSPARK 细节）。

设：

- `T`：本次 forward 的 token 数，prefill 时可以是多 token，decode 时通常是 batch 内各请求的新 token 总数。
- `H`：hidden size。
- `L`：decoder 层数（PP 下实际是 `end_layer - start_layer`）。
- `x: [T, H]`。
- `positions: [T]`。

```python
def kimi_k3_text_forward(input_ids, positions, forward_batch, inputs_embeds=None):
    # input_ids: [T]；多模态可直接传 inputs_embeds，跳过 embed
    # positions: [T]

    if inputs_embeds is not None:
        x = inputs_embeds                 # [T, H]
    else:
        x = embed_tokens(input_ids)       # [T, H]
    residual = None

    attn_res = None
    if config.attn_res_block_size is not None:
        # 真实：AttnResidual(x, cdiv(end_layer, attn_res_block_size), block_residual=residual)
        attn_res = AttnResidual(
            x,
            block_num=cdiv(L, attn_res_block_size),
            block_residual=residual,
        )
        residual = None                   # 交给 bank；后面层走 attn_res 路径

    sp_sharded = False
    for layer_idx in range(L):            # 真实：range(start_layer, end_layer)
        # 每层内部：norm → MLA/KDA → norm → MLP/MoE
        # （标准 residual / attn_res 两条路径见下文）
        # 注意：不要写成 layers[i](…)，部分 Markdown 预览会把 [i]( 当成链接，出现大块空白
        layer = layers[layer_idx]
        x, residual, sp_sharded = layer(
            positions=positions,
            hidden_states=x,              # [T, H]
            forward_batch=forward_batch,
            residual=residual,
            attn_res=attn_res,
            # 另有 zero_allocator / input_sharded / keep_sharded（SP-MoE）
        )

    # ---- final norm / aggregation（仅 last PP rank）----
    if attn_res is None:
        if residual is None:
            x = norm(x)                   # [T, H]
        else:
            x, _ = norm(x, residual)      # fused add+RMSNorm
    else:
        # 不是另写一个 final_attn_res_aggregate；就是再调一次 attn_res.forward，
        # 用 output 侧的 proj/norm，out_norm=self.norm（把 delayed add 折进去）
        x, _ = attn_res.forward(
            hidden_states=x,
            prefix_sum=residual,
            score_proj=output_attn_res_proj,
            score_norm=output_attn_res_norm,
            out_norm=norm,
        )                                 # [T, H]

    # 真实不是裸 lm_head(x)，而是 LogitsProcessor(input_ids, x, lm_head, forward_batch)
    logits = logits_processor(input_ids, x, lm_head, forward_batch)
    return logits
```

和源码对齐时注意：

| 笔记简化 | 源码 |
|---|---|
| `range(L)` | `range(start_layer, end_layer)`（pipeline parallel） |
| `final_attn_res_aggregate(...)` | `attn_res.forward(..., output_attn_res_proj/norm, self.norm)` |
| `lm_head(x)` | `LogitsProcessor` + `ParallelLMHead` |
| 只写 `embed_tokens` | 还可 `inputs_embeds`（多模态 merge 后） |
| 忽略空 batch | `if hidden_states.shape[0] != 0` 才做 final norm |

主干数据流（embed → 可选 AttnResidual → L 层 → final norm → logits）是对的。

## 标准 Residual 路径

不开 `attn_res_block_size` 时，一层的 forward 接近普通 pre-norm decoder：

```python
def decoder_layer_forward(x, residual):
    # x: [T, H]

    if residual is None:
        residual = x
        x = input_layernorm(x)                 # [T, H]
    else:
        x, residual = input_layernorm(x, residual)

    attn_out = self_attn(x)                    # [T, H]
    attn_out = finish_attn_reduce(attn_out)    # [T, H]

    x, residual = post_attention_layernorm(attn_out, residual)
    x = mlp(x)                                 # [T, H]
    return x, residual
```

这条路径里，attention 输出和 MLP 输出通过 `RMSNorm(hidden, residual)` 这类 fused norm/add 方式维持 residual stream。

## KimiK3MLAAttention

`KimiK3MLAAttention` 继承 `DeepseekV2AttentionMLA`，也就是 MLA 的 latent KV 路径：

```text
hidden [T, H]
├── q_a_proj / q_a_layernorm / q_b_proj
│   └── q_nope + q_rope
├── kv_a_proj_with_mqa / kv_a_layernorm / kv_b_proj
│   └── kv_latent + k_rope -> k_nope, v
├── RadixAttention / MLA backend
└── o_proj -> [T, H]
```

关键差异是 K3 可以带 output gate：

```python
if config.mla_use_output_gate:
    gate = g_proj(hidden_states)              # [T, num_heads * v_head_dim]
    attn_out = attn_out * sigmoid(gate)       # TP-local attention output
    out = o_proj(attn_out)                    # [T, H]
```

形状可以这样记：

```python
q_lora      = q_a_proj(x)                     # [T, q_lora_rank]
q_lora      = q_a_layernorm(q_lora)           # [T, q_lora_rank]
q_full      = q_b_proj(q_lora)                # [T, A * (D_nope + D_rope)]

kv_pack     = kv_a_proj_with_mqa(x)           # [T, R_kv + D_rope]
kv_latent   = kv_pack[..., :R_kv]             # [T, R_kv]
k_rope      = kv_pack[..., R_kv:]             # [T, D_rope]

attn_out    = mla_attention(q_full, kv_latent, k_rope)
out         = o_proj(attn_out)                # [T, H]
```

这里的 KV cache 主要缓存 latent KV 和 rope K，而不是每个 head 完整展开后的 dense K/V。

## KimiK3DeltaAttention / KDA

KDA 是 Kimi-K3 的 linear attention 层。它不是普通 softmax attention，而是维护线性注意力状态。

核心输入输出：

```python
x: [T, H]

mixed_qkv   = qkv_proj_or_fused(x)
beta        = b_proj(x)
forget_gate = f_b_proj(f_a_proj(x))
gate        = g_proj(x) or g_b_proj(g_a_proj(x))

core = RadixLinearAttention(
    mixed_qkv=mixed_qkv,
    a=forget_gate,
    b=beta,
)

core = gated_rms_norm(core, gate)
out  = o_proj(core)                           # [T, H]
```

更具体一点：

```python
def kda_forward(x):
    # x: [T, H]

    q, k, v = project_qkv(x)                  # each roughly [T, A_kda * D]
    beta = project_beta(x)                    # [T, A_kda]
    forget_gate = project_forget_gate(x)      # [T, A_kda * D]
    output_gate = project_output_gate(x)      # [T, A_kda * Dv]

    # RadixLinearAttention 内部会结合短卷积状态和 recurrent state
    y = radix_linear_attention(
        q=q,
        k=k,
        v=v,
        a=forget_gate,
        b=beta,
        forward_batch=forward_batch,
    )                                         # [1, T, A_kda, Dv] 或等价布局

    y = gated_rms_norm(y, output_gate)
    y = flatten_heads(y)                      # [T, A_kda * Dv]
    return o_proj(y)                          # [T, H]
```

KDA 层的 cache 不是 dense KV cache，而是 linear attention state / short conv state。`KimiLinearConfig.mamba2_cache_params` 会按 `linear_layer_ids` 为 KDA 层创建对应状态形状。

## Attention Residual

开启 `config.attn_res_block_size` 后，不再只用一条「当前 residual」；而是维护一个 **跨层 snapshot bank**，在 attention 前 / MLP 前各做一次 **softmax 加权聚合**。源码：`sglang/srt/layers/attn_residual.py`。

### prefix 是什么

**`prefix`（代码里常叫 `prefix_sum`）= 当前 block 里、还没写进 bank 的那条 running residual 流。**

对比：

```text
标准 Transformer：
  residual ← residual + attn_out / mlp_out   （一条一直加下去）

Attn Residual：
  每隔 B 层把当时的 residual 冻进 bank 一行
  冻进去之后，本 block 重新从空开始累加 → 这条正在累加的就是 prefix
```

在一次 Agg 里：

```text
候选行 = [ bank[0], bank[1], ..., bank[nvb-1],  prefix ]
              ↑ 历史冻结的快照                    ↑ 当前还在长的这条
       —— softmax 加权混合 ——→ 送给 Attn 或 MLP
```

所以：

- **bank 行**：过去某个 write 层拍下的旧 prefix（只读）
- **prefix**：这一刻的「活」residual；write 层会把它写入 `bank[:, nvb]`，然后常把 `prefix_sum = None`，新 block 重开

和 `hidden` 的关系——Agg 时经常是挂起加法：

```text
prefix = prefix_sum + hidden
```

- `prefix_sum`：上一截已经攒好的流
- `hidden`：上一模块吐出的 delta（例如 attn 输出）
- 二者相加才是完整 prefix，再拿去和 bank 一起打分混合

`prefix_sum is None` 时，说明 `hidden` 本身已经是完整 head（block 边界刚 write 完、或 PP 入口）。

**一句话：prefix = 当前 block 尚未入库的 residual 前缀；bank = 历史 prefix 的快照本。**

### 角色与张量

```text
block_residual: [T, NB, H]   # bank；NB = cdiv(L, attn_res_block_size)，K3 通常 NB≤8
num_valid_blocks (nvb)       # 已写入的 snapshot 行数
prefix_sum: [T, H] | None    # 当前 block 内「还没加进 bank」的 running prefix
score_proj: H → 1            # 每层两套：attn 侧 / mlp 侧
score_norm / out_norm: RMSNorm
```

每层：

```python
is_block_write_layer = (layer_idx % attn_res_block_size == 0)
# layer 0, B, 2B, ... 在 attn 侧聚合时把当前 prefix 写入下一行 bank
```

### 核心：`aggregate`（数学参考 = `aggregate_stream_torch`）

对每个 token，把 **已冻结的 bank 行 + 当前 prefix** 当成 `nvb+1` 条候选流，打分后 softmax 混合，再 `out_norm`：

```python
def aggregate(prefix, bank, nvb, score_proj, score_norm, out_norm):
    # prefix: [T, H]   bank: [T, NB, H]  只用前 nvb 行
    if nvb == 0:
        return out_norm(prefix)

    # rows[j] = bank[:, j]  (j = 0..nvb-1)
    # rows[nvb] = prefix      ← 当前 running prefix 也参与竞争
    rows = cat(bank[:, :nvb, :], prefix[:, None, :], dim=1)   # [T, nvb+1, H]

    flat = rows.reshape(T * (nvb + 1), H)
    scores = score_proj(score_norm(flat)).reshape(T, nvb + 1)  # [T, R]
    probs = softmax(scores, dim=-1)                            # [T, R]
    mixed = (probs[:, :, None] * rows).sum(dim=1)              # [T, H]
    return out_norm(mixed)
```

实现里用 Triton/TMA 融合，不显式 materialize `[T,R,H]`；语义同上。`cw = score_norm.weight ⊙ score_proj.weight` 把 score 侧 RMSNorm+线性压成一次点积。

### `AttnResidual` 对象（一次 forward 建一次）

```python
class AttnResidual:
    def __init__(self, x, block_num, block_residual=None):
        self.bank = empty(T, block_num, H)     # block_residual
        self.nvb = 0
        if block_residual is not None:         # PP 从上一段继承
            self.bank[:, :NB_in] = block_residual
            self.nvb = NB_in

    def write(self, prefix):
        self.bank[:, self.nvb, :] = prefix
        self.nvb += 1

    def forward(self, hidden, prefix_sum, score_proj, score_norm, out_norm,
                write=False):
        """
        返回 (normed, prefix)：
          normed  — 给下游模块用（已 out_norm）
          prefix  — 本步 materialize 后的完整 prefix（可能写入 bank）
        """
        if self.nvb == 0:
            # 第 0 层 attn 侧：bank 还空，只有当前 hidden
            assert prefix_sum is None
            if write:
                self.write(hidden)
            return out_norm(hidden), hidden

        if prefix_sum is None:
            # hidden 已是完整 head（block 边界重启 / PP 入口）
            prefix = hidden
            normed = aggregate(prefix, self.bank, self.nvb,
                               score_proj, score_norm, out_norm)
        else:
            # 挂起的 add：先拼出完整 prefix，再聚合
            prefix = prefix_sum + hidden
            normed = aggregate(prefix, self.bank, self.nvb,
                               score_proj, score_norm, out_norm)

        if write:
            self.write(prefix)                 # snapshot 进 bank 下一行
        return normed, prefix
```

### 层内怎么用（`_forward_attn_residual`，忽略 SP）

层与层之间约定：上一层 MLP 返回的 `hidden` 是 **未加进 prefix 的 delta**，`prefix_sum` 常为 `None`（MLP 已把 prefix 折进输出，或 write 后清掉）。注释原文：*hidden_states carries the previous layer's un-added MLP delta*。

```python
def decoder_layer_with_attn_res(x, prefix_sum, attn_res, layer_idx):
    # x: [T, H]     prefix_sum: [T, H] | None

    # ---- Agg1：attention 侧 ----
    # out_norm = input_layernorm；write 层顺带把 prefix 写入 bank
    x, prefix_sum = attn_res.forward(
        hidden=x,
        prefix_sum=prefix_sum,
        score_proj=self_attention_res_proj,
        score_norm=self_attention_res_norm,
        out_norm=input_layernorm,
        write=is_block_write_layer,
    )
    if is_block_write_layer:
        prefix_sum = None                      # 本 block 的 prefix 已进 bank，重新开跑

    # ---- Self-Attn（MLA 或 KDA）----
    attn_out = self_attn(x)                    # [T, H]
    attn_out = finish_o_proj_reduce(attn_out)

    # ---- Agg2：MLP 侧 ----
    x, prefix_sum = attn_res.forward(
        hidden=attn_out,
        prefix_sum=prefix_sum,
        score_proj=mlp_res_proj,
        score_norm=mlp_res_norm,
        out_norm=post_attention_layernorm,
        write=False,
    )

    # ---- MLP：把 prefix_sum 折进尾部 add（dense: down_proj 后加；MoE: 三路尾加）----
    out = mlp(x, prefix_sum=prefix_sum)        # [T, H]
    return out, None                           # 上层再进来时 prefix_sum 从 None 起
```

### 整网时间线（`attn_res_block_size = B` 示意）

```text
layer 0 (write):  Agg1 写 bank[0]=embed_prefix → Attn → Agg2 → MLP
layer 1..B-1:     Agg1 用 bank[0]+prefix 混合 → Attn → Agg2 → MLP
layer B (write):  Agg1 写 bank[1]=当前 prefix → Attn → ...
...
final:            attn_res.forward(x, residual, output_attn_res_proj/norm, out_norm=norm)
                  （logits 前再聚合一次，语义同 Agg，权重用 output 侧）
```

### 和标准 residual 的对比

| | 标准路径 | Attn Residual |
|---|---|---|
| 流 | 单条 `residual`，fused add+RMSNorm | bank 多条历史 prefix + 当前 prefix |
| 进 Attn/MLP 前 | `RMSNorm(x, residual)` | `softmax` 混合后再 `out_norm` |
| 跨层记忆 | 只有最近 residual | 每隔 `B` 层冻结一行，后续层可再读到 |
| 层返回 | `(x, residual)` | 通常 `(mlp_out, None)`，prefix 在层内消化或写入 bank |

**一句话：** Attention Residual = 把「每隔 B 层的 prefix 快照」存进 bank，之后每个 Agg 点用可学习的 score 对「历史快照 + 当前 prefix」做 softmax 路由，再 RMSNorm 送给 Attn 或 MLP。

## 多模态路径

多模态入口大致是：

```python
def kimi_k3_multimodal_forward(input_ids, images, positions, forward_batch):
    text_embeds = language_model.embed_tokens(input_ids)       # [T_text, H]

    vision_features = vision_tower(images)                     # [T_img, H_v]
    image_embeds = mm_projector(vision_features)               # [T_img, H]

    inputs_embeds = merge_text_and_image_embeds(
        input_ids,
        text_embeds,
        image_embeds,
    )                                                          # [T_total, H]

    return language_model(
        input_ids=None_or_ids,
        positions=positions,
        inputs_embeds=inputs_embeds,
        forward_batch=forward_batch,
    )
```

也就是说，图片先进 `MoonViT3d`，再经 projector 对齐到语言模型 hidden size，最后作为 embedding token 混入 text backbone。后面的 decoder 层仍然是同一套 `KimiK3LinearModel`。

## 一句话总结

Kimi-K3 的主干可以理解为：一个支持多模态 embedding 输入的 decoder-only LM；每层在 MLA full attention 和 KDA linear attention 之间按 config 混排，MLP 又可以按 config 在 dense MLP 和 MoE 之间切换；如果开启 `attn_res_block_size`，还会多一条跨 block 的 residual bank 来辅助层间聚合。
