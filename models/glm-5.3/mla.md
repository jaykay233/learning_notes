# GLM-5.3 MLA 详解

> 基于 SGLang text-only 路径整理：
> `GlmMoeDsaForCausalLM -> DeepseekV2ForCausalLM -> DeepseekV2AttentionMLA`。
> 主源码：`python/sglang/srt/models/deepseek_v2.py`。

---

## 1. 一句话总览

GLM-5.3 text-only 的 attention 主体是 **MLA（Multi-head Latent Attention）**：

```text
Q: 低秩 q_lora -> 多头 Q
KV: hidden -> kv_latent + k_rope
Cache: 主要缓存 kv_latent + k_rope
Attention: 从 latent cache 重建/吸收 K/V 语义，得到 ctx
```

MLA 的重点不是简单把普通 K/V head 数改小，而是把历史 KV 存成更低维的 latent；在 SGLang 的 absorb 路径里，它会以 MQA 形式执行：**query heads 仍然很多，但 KV latent head 是 1 个，被所有 query heads 共享**。

```text
普通 MHA cache: [T, heads, K_dim + V_dim]
MLA latent cache: [T, kv_lora_rank + qk_rope_head_dim]
```

GLM-5.3 常见尺寸：

```text
heads = 64
qk_nope_head_dim = 192
qk_rope_head_dim = 64
v_head_dim = 256
kv_lora_rank = 512
```

所以每个 token 的 latent cache 约是：

```text
kv_lora_rank + qk_rope_head_dim = 512 + 64 = 576
```

而不是把完整 K/V 展开成：

```text
64 * (256 + 256) = 32768
```

---

## 2. SGLang 模块位置

GLM-5.3 在 SGLang 里不是单独实现一套 `GlmAttention`，而是复用 DeepSeekV2 风格实现：

```python
class GlmMoeDsaForCausalLM(DeepseekV2ForCausalLM):
    ...
```

真正的 attention 类：

```text
DeepseekV2AttentionMLA
```

核心子模块：

| 子模块 | 作用 |
|---|---|
| `fused_qkv_a_proj_with_mqa` | 一次投影出 `q_lora + kv_latent + k_rope` |
| `q_a_layernorm` | 归一化 Q 低秩表示 |
| `q_b_proj` | `q_lora -> Q heads` |
| `kv_a_layernorm` | 归一化 KV latent |
| `kv_b_proj` | `kv_latent -> K_nope + V` |
| `rotary_emb` | 给 Q/K 的 rope 部分加位置 |
| `attn_mqa` | MLA absorb 路径使用的 RadixAttention，KV head 为 1，V dim 为 `kv_lora_rank` |
| `attn_mha` | fallback / normal 展开路径 |
| `o_proj` | attention context -> hidden |

`attn_mqa` 的构造很容易看出这个点：

```python
self.attn_mqa = RadixAttention(
    self.num_local_heads,                         # Q heads，仍然是多头
    self.kv_lora_rank + self.qk_rope_head_dim,    # latent K head dim
    self.scaling,
    num_kv_heads=1,                               # KV heads = 1
    v_head_dim=self.kv_lora_rank,
)
```

所以不要把 `num_kv_heads=1` 理解成 “attention 没有 heads”。准确形状是：

```text
Q heads:  A = num_attention_heads
KV heads: 1

q_nope_out: [T, A, R_kv]
k_nope:     [T, 1, R_kv]
v_latent:   [T, 1, R_kv]
```

每个 query head 都有自己的 `q_nope_out[:, h, :]`，但它们和同一份 latent K cache 打分：

```python
score_h = q_nope_out[:, h, :] @ k_nope[:, 0, :].T
```

SGLang 还会根据 backend 选择不同 forward method：

```text
MLA / MLA_ROCM / MLA_FUSED_ROPE_ROCM / MLA_FUSED_ROPE_CPU / MLA_NPU / DSA_NPU ...
```

这些是后端实现差异，数学语义仍围绕 MLA latent cache。

---

## 3. 关键形状

```text
T      = token 数
H      = hidden_size = 6144
A      = num_attention_heads = 64
D_nope = qk_nope_head_dim = 192
D_rope = qk_rope_head_dim = 64
D_qk   = D_nope + D_rope = 256
D_v    = v_head_dim = 256
R_q    = q_lora_rank = 2048
R_kv   = kv_lora_rank = 512
```

核心张量：

| 张量 | Shape | 含义 |
|---|---|---|
| `x` | `[T,H]` | attention 输入 hidden |
| `q_lora` | `[T,R_q]` | Q 低秩表示 |
| `q` | `[T,A,D_qk]` | 多头 Q |
| `kv_latent` | `[T,R_kv]` | MLA KV latent |
| `k_rope` | `[T,D_rope]` | K 的 RoPE 部分 |
| `latent_cache` | `[T_cache,R_kv + D_rope]` | decode cache 里的历史 latent |
| `q_nope_out` | `[T,A,R_kv]` | Q 内容部分吸收到 latent 维后的多头 query |
| `k_nope` absorb 路径 | `[T,1,R_kv]` | 单 KV head 的 latent K，被所有 Q heads 共享 |
| `k_nope` | `[T,A,D_nope]` | 从 latent 展开的 K 内容部分 |
| `v` | `[T,A,D_v]` | 从 latent 展开的 V |
| `ctx` | `[T,A,D_v]` | attention 输出 |
| `out` | `[T,H]` | O projection 后输出 |

---

## 4. 投影路径

### 4.1 fused A 投影

当 `q_lora_rank` 存在时，SGLang 会把 Q 的 A 投影和 KV 的 A 投影合并：

```text
x [T,H]
  │
  ▼
fused_qkv_a_proj_with_mqa
  │
  ├─ q_lora   [T,R_q]
  ├─ kv_latent[T,R_kv]
  └─ k_rope   [T,D_rope]
```

对应输出总宽度：

```text
R_q + R_kv + D_rope
= 2048 + 512 + 64
= 2624
```

教学版：

```python
fused = fused_qkv_a_proj_with_mqa(x)              # [T, R_q + R_kv + D_rope]
q_lora, kv_latent, k_rope = split(
    fused,
    [R_q, R_kv, D_rope],
    dim=-1,
)
```

### 4.2 Q 路径

```python
q_lora = q_a_layernorm(q_lora)                    # [T, R_q]
q = q_b_proj(q_lora)                              # [T, A * D_qk]
q = q.reshape(T, A, D_qk)                         # [T, A, D_qk]

q_nope, q_rope = split(q, [D_nope, D_rope], dim=-1)
q_rope = apply_rope(q_rope, positions)            # [T, A, D_rope]
q = concat(q_nope, q_rope, dim=-1)                # [T, A, D_qk]
```

### 4.3 KV latent 路径

```python
kv_latent = kv_a_layernorm(kv_latent)             # [T, R_kv]
k_rope = apply_rope(k_rope, positions)            # [T, D_rope]

latent_cache.write(
    concat(kv_latent, k_rope, dim=-1)
)                                                 # [T, R_kv + D_rope]
```

当前 token 或 prefill 阶段需要展开内容 K/V：

```python
kv_full = kv_b_proj(kv_latent)                    # [T, A * (D_nope + D_v)]
k_nope, v = split_kv_b(kv_full)
# k_nope: [T, A, D_nope]
# v     : [T, A, D_v]
```

但在 SGLang 的 MLA absorb 推理路径里，通常不会把历史 cache 先完整展开成 `[S,A,D_nope]` / `[S,A,D_v]` 再算 attention，而是：

```python
q_nope_out = q_nope @ w_kc        # [T, A, R_kv]
k_nope     = kv_latent            # [S, 1, R_kv]

attn_latent = attn_mqa(
    q=q_nope_out,
    k=k_nope,
    v=k_nope,
    q_rope=q_rope,
    k_rope=k_rope,
)                                 # [T, A, R_kv]

ctx = attn_latent @ w_vc           # [T, A, D_v]
```

这就是前面说的 “Q heads 很多，但 KV latent head 只有 1 个”。

---

## 5. Latent Cache 怎么省

普通完整 K/V 如果直接缓存，可以粗略看成：

```text
K: [T,A,D_qk] = [T,64,256]
V: [T,A,D_v]  = [T,64,256]
合计每 token: 64 * (256 + 256) = 32768 scalar
```

MLA cache：

```text
kv_latent: [T,512]
k_rope   : [T,64]
合计每 token: 576 scalar
```

这就是 MLA 的核心收益：历史 token 不必把每个 head 的 K/V 都完整落 cache。

注意：计算时仍然可以恢复出多头语义：

```text
kv_latent -> kv_b_proj -> K_nope / V
k_rope 负责位置部分
```

---

## 6. Attention 语义

教学版完整 attention：

```python
def mla_attention(x, positions, kv_cache):
    """
    x: [T,H]
    return: [T,H]
    """
    fused = fused_qkv_a_proj_with_mqa(x)            # [T, R_q + R_kv + D_rope]
    q_lora, kv_latent, k_rope = split(fused, [R_q, R_kv, D_rope], dim=-1)

    q_lora = q_a_layernorm(q_lora)                  # [T,R_q]
    q = q_b_proj(q_lora).reshape(T, A, D_qk)        # [T,A,D_qk]
    q_nope, q_rope = split(q, [D_nope, D_rope], dim=-1)

    kv_latent = kv_a_layernorm(kv_latent)           # [T,R_kv]
    q_rope = apply_rope(q_rope, positions)          # [T,A,D_rope]
    k_rope = apply_rope(k_rope, positions)          # [T,D_rope]

    kv_cache.write_latent(kv_latent, k_rope)        # [T,R_kv + D_rope]

    selected_history = kv_cache.visible_indices()   # dense MLA 或 DSA top-k 后的历史
    hist_latent, hist_k_rope = kv_cache.read(selected_history)
    # hist_latent: [S,R_kv]
    # hist_k_rope: [S,D_rope]

    hist_kv = kv_b_proj(hist_latent)                # [S,A*(D_nope+D_v)]
    hist_k_nope, hist_v = split_kv_b(hist_kv)
    # hist_k_nope: [S,A,D_nope]
    # hist_v     : [S,A,D_v]

    q = concat(q_nope, q_rope, dim=-1)              # [T,A,D_qk]
    k = concat(hist_k_nope, broadcast_rope(hist_k_rope), dim=-1)
    # k: [S,A,D_qk]

    ctx = attention(q, k, hist_v)                   # [T,A,D_v]
    out = o_proj(ctx.reshape(T, A * D_v))           # [T,H]
    return out
```

SGLang 热路径通常不会真的为所有历史显式 materialize `hist_k_nope/hist_v` 大张量；backend 会用 absorb / fused kernels 做等价计算。

---

## 7. MLA 和 DSA 的关系

MLA 和 DSA 不是二选一：

```text
MLA: 历史内容怎么存、怎么读
DSA: 长历史里先选哪些位置值得读
```

放在一起：

```text
q_lora / q
  ├─► DSA indexer 选 selected_indices
  │
  └─► MLA attention backend 只读 selected_indices 对应的 latent cache
```

没有 DSA 时：

```text
selected_indices = 全部可见历史 / backend 决定的 dense 历史范围
```

有 DSA 时：

```text
selected_indices = top-k 历史 token/block
```

---

## 8. 读源码顺序

1. `DeepseekV2AttentionMLA.__init__`
   - 看 `fused_qkv_a_proj_with_mqa`、`q_b_proj`、`kv_b_proj`、`attn_mqa`。
2. `prepare_qkv_latent`
   - 看 fused A 投影如何产出 `q_lora + kv_latent + k_rope`。
3. `q_b_proj_forward`
   - 看 `q_lora -> Q heads`。
4. `forward_prepare` / `forward_core`
   - 看 backend 如何分派到 MLA / DSA / NPU / ROCm 路径。
5. `deepseek_common/attention_forward_methods`
   - 看 MLA absorb 路径如何避免显式展开完整历史 K/V。

---

## 9. 最小心智模型

```text
MLA = 把历史 K/V 存成 latent cache。

当前 token:
  q_lora -> Q heads
  hidden -> kv_latent + k_rope

历史 token:
  cache 里主要保存 kv_latent + k_rope

attention:
  从 latent cache 读出 / 吸收出 K/V 语义
  输出 ctx -> o_proj -> hidden
```
