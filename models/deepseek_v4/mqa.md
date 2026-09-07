# DeepSeek-V4 MQA / Attention Layer 详解

> 基于 SGLang DeepSeek-V4 实现和本目录 `architecture.md` 拆解整理。建议先读：
> 1. `architecture.md` 的 §4 Attention：`MQALayer`
> 2. `mhc.md` 的 §2 / §7，理解 Attention 外层的 mHC residual 边界

本文只聚焦 **Attention layer 内部**：MQA 是什么、DeepSeek-V4 为什么用单 KV head、Q/K/V 怎么投影、RoPE/nope 如何拆分，以及 CSA / HCA 压缩状态如何接入注意力。

---

## 1. 一句话总览

DeepSeek-V4 的 attention 可以概括为：

```text
mHC 给出单路 hidden [T,H]
      │
      ▼
低秩 Q 投影 + 单 KV 投影
      │
      ▼
Q 多头，K/V 单头共享，也就是 MQA
      │
      ▼
按层选择 SWA / CSA / HCA
      │
      ▼
低秩 O 投影回 hidden [T,H]
      │
      ▼
mHC 写回多路 residual
```

这里的关键点：

1. **Q 是多头**：`num_attention_heads = 64 / 128`，每个 query head 都能学不同的查询子空间。
2. **K/V 是单头**：`num_key_value_heads = 1`，所有 Q heads 共享同一套 K/V cache。
3. **Q 和 O 走低秩投影**：用 `q_lora_rank` / `o_lora_rank` 控制中间维，降低投影成本。
4. **长上下文不只靠 MQA**：V4 还按层混入 CSA / HCA，把历史 KV 压缩成更小状态。

---

## 2. MQA vs MHA vs GQA

标准 Multi-Head Attention（MHA）里，Q/K/V 都有很多头：

```text
Q: [T, n_heads, D]
K: [T, n_heads, D]
V: [T, n_heads, D]
```

Grouped-Query Attention（GQA）减少 KV 头数：

```text
Q: [T, n_q_heads,  D]
K: [T, n_kv_heads, D]   # n_kv_heads < n_q_heads
V: [T, n_kv_heads, D]
```

Multi-Query Attention（MQA）是 GQA 的极限：**只有 1 个 KV head**。

```text
Q: [T, n_q_heads, D]
K: [T, 1,         D]
V: [T, 1,         D]
```

DeepSeek-V4 采用的是 **真 MQA**：

| 字段 | Flash 系列 | Pro 系列 | 含义 |
|---|---:|---:|---|
| `num_attention_heads` | 64 | 128 | Q head 数 |
| `num_key_value_heads` | 1 | 1 | KV head 数 |
| `head_dim` | 512 | 512 | 单个 head 的 Q/K 维度 |
| `v_head_dim` | 512 | 512 | V head 维度 |

为什么这件事重要？因为 decode 阶段每生成一个 token，都要反复读取历史 K/V cache。KV head 从 `n_heads` 降到 `1` 后，KV cache 近似按 head 数缩小：

```text
MHA KV cache  ≈ T * n_heads * (Dk + Dv)
MQA KV cache  ≈ T * 1       * (Dk + Dv)
```

以 Flash 的 64 个 Q heads 为例，单从 KV head 数看，MQA 的 dense KV cache 约是 MHA 的 `1/64`。这不是全部显存账本，因为 V4 还有压缩状态、indexer cache、量化 dtype 等细节，但直觉上可以先这样记：

> MQA 保留多头查询能力，但把历史记忆 K/V 变成一份共享副本。

---

## 3. Attention Layer 在 Decoder 里的位置

在 DeepSeek-V4 里，Attention 不是直接吃标准 residual。外面还有 mHC：

```text
R [T,hc,H]
   │
   ▼
hc_pre(attn) + input_layernorm
   │
   ▼
y [T,H]
   │
   ▼
MQALayer
   │
   ▼
o_attn [T,H]
   │
   ▼
hc_post(attn)
   │
   ▼
R' [T,hc,H]
```

所以读 attention 源码时要分清：

| 层级 | 张量形状 | 负责什么 |
|---|---|---|
| mHC residual | `[T,hc,H]` | 多路残差流混合与写回 |
| Attention 输入 | `[T,H]` | 进入 `MQALayer` 的普通 hidden |
| Q | `[T,n_q_heads,D]` | 多头查询 |
| K/V | `[T,1,D]` | 单头共享历史记忆 |
| Attention 输出 | `[T,H]` | 交回 mHC 写回 residual |

`MQALayer` 内部基本不用关心 `[T,hc,H]`，它只处理 mHC 给它合成好的 `[T,H]`。

---

## 4. 关键配置与形状

设：

```text
T = token 数
H = hidden_size
A = num_attention_heads
D = head_dim
R_q = q_lora_rank
R_o = o_lora_rank
G_o = o_groups
```

Flash 量级常见：

| 字段 | 值 | 说明 |
|---|---:|---|
| `hidden_size` | 4096 | Decoder hidden 宽度 |
| `num_attention_heads` | 64 | Q heads |
| `num_key_value_heads` | 1 | MQA 单 KV head |
| `qk_nope_head_dim` | 448 | 不做 RoPE 的 Q/K 部分 |
| `qk_rope_head_dim` | 64 | 做 RoPE 的 Q/K 部分 |
| `head_dim` | 512 | `448 + 64` |
| `v_head_dim` | 512 | V 维度 |
| `q_lora_rank` | 1024 | Q 低秩中间维 |
| `o_lora_rank` | 1024 | O 低秩中间维 |
| `o_groups` | 8 | 输出投影分组 |

Pro 量级常见差异：

| 字段 | Flash | Pro |
|---|---:|---:|
| `hidden_size` | 4096 | 7168 |
| `num_attention_heads` | 64 | 128 |
| `q_lora_rank` | 1024 | 1536 |
| `o_lora_rank` | 1024 | 1024 |
| `head_dim` | 512 | 512 |
| `num_key_value_heads` | 1 | 1 |

直观形状：

```text
y      : [T, H]
q_lora : [T, R_q]
Q      : [T, A, D]
K      : [T, 1, D]
V      : [T, 1, D]
attn   : [T, A, D]
out    : [T, H]
```

---

## 5. Q 投影：低秩但仍是多头

DeepSeek-V4 不直接做一个巨大的 `H -> A*D` Q 投影，而是分成两段：

```text
y [T,H]
   │
   ▼
wq_a / wqkv_a
   │
   ▼
q_lora [T,R_q]
   │
   ▼
q_norm
   │
   ▼
wq_b
   │
   ▼
Q [T,A,D]
```

可以把它理解成 LoRA 风格的低秩分解，但这里是模型结构本身的一部分：

```text
直接投影: y @ W_q
低秩投影: y @ Wq_a @ Wq_b
```

收益：

1. `Wq_a` 先把 hidden 压到 `R_q`。
2. `q_norm` 稳定中间表示。
3. `Wq_b` 再展开成所有 Q heads。

注意：**低秩的是投影路径，不是 Q heads 数**。Q 最终仍然是多头的 `[T,A,D]`。

---

## 6. K/V 投影：单 KV head

K/V 共享一条投影路径：

```text
y [T,H]
   │
   ▼
wkv
   │
   ▼
kv_norm
   │
   ▼
K/V 单路状态
```

从 attention 语义看，所有 query heads 都去和同一份 K 做相似度，再读取同一份 V：

```text
score[h, t, s] = dot(Q[t,h,:], K[s,0,:]) / sqrt(D)
out[t,h,:]     = Σ_s softmax(score[h,t,s]) * V[s,0,:]
```

其中 `h` 是 Q head，`s` 是历史 token。虽然 K/V 只有一份，但每个 Q head 的查询向量不同，所以每个 head 的 attention 分布仍然可以不同：

```text
Q head 0 ─┐
Q head 1 ─┼──► shared K/V cache
...       │
Q head A ─┘
```

这就是 MQA 的核心折中：

| 保留 | 压缩 |
|---|---|
| 多个 Q heads 的查询能力 | 多份 K/V 历史记忆 |
| 每个 Q head 不同的 attention logits | KV cache head 数 |

---

## 7. RoPE / nope 拆分

DeepSeek-V4 的 Q/K head dim 拆成两部分：

```text
D = qk_nope_head_dim + qk_rope_head_dim
  = 448              + 64
  = 512
```

可以概念化为：

```text
Q = concat(Q_nope, Q_rope)
K = concat(K_nope, K_rope)
```

| 部分 | 是否加 RoPE | 作用直觉 |
|---|---|---|
| `nope` | 否 | 保存内容/语义相似度 |
| `rope` | 是 | 注入相对位置信息 |

注意力分数本质上还是完整 Q/K 的点积：

```text
dot(Q, K)
= dot(Q_nope, K_nope) + dot(Q_rope_after_rope, K_rope_after_rope)
```

为什么只给一小部分加 RoPE？

1. 位置编码只占 head dim 的一部分，给模型留出大量纯内容维度。
2. 长上下文下，RoPE 相关维度可以单独配合 YaRN / compressed rope theta 调整。
3. 对压缩状态来说，位置分支和内容分支可以更清晰地被缓存、量化、压缩。

---

## 8. `compress_ratio`：同一个 MQA Layer 的三种历史读法

DeepSeek-V4 按层配置：

```text
compress_ratio ∈ {0, 4, 128}
```

它决定 attention 读历史的方式。

### 8.1 每一层怎么选 Attention 类型

DeepSeek-V4 不是运行时动态给每个 token 选择 attention 类型，而是 **每个 decoder layer 在 config 里预先指定一个 `compress_ratio`**：

```python
compress_ratio = config.compress_ratios[layer_id]
```

然后 `MQALayer.__init__` 根据这个值决定本层挂哪些模块：

```python
class MQALayer:
    def __init__(self, config, layer_id, ...):
        self.compress_ratio = config.compress_ratios[layer_id]

        self.compressor = None
        self.indexer = None

        if self.compress_ratio in (4, 128):
            self.compressor = Compressor(
                compress_ratio=self.compress_ratio,
                is_in_indexer=False,
                head_dim=head_dim,
            )

            if self.compress_ratio == 4:
                self.indexer = C4Indexer(...)

        self.attn_mqa = RadixAttention(
            num_kv_heads=1,
            layer_id=layer_id,
        )
```

所以三种层的模块配置是：

| `compress_ratio` | attention 类型 | `self.compressor` | `self.indexer` | RoPE |
|---:|---|---|---|---|
| `0` | SWA / 纯窗口 | 无 | 无 | 普通 RoPE |
| `4` | CSA | `Compressor(4)` | `C4Indexer` | compressed YaRN RoPE |
| `128` | HCA | `Compressor(128)` | 无 | compressed YaRN RoPE |

forward 时，这个 `compress_ratio` 会继续传给 backend：

```python
o = attn_backend.forward(
    q=q,
    k=attn_k,
    v=attn_k,
    layer=self.attn_mqa,
    forward_batch=forward_batch,
    compress_ratio=self.compress_ratio,
    attn_sink=attn_sink,
)
```

同时，attention 前的准备阶段会按本层模块做额外工作：

```python
if self.indexer is not None:
    self.indexer(...)                 # 只在 ratio=4 时产生 top-k sparse indices

if self.compressor is not None:
    forward_core_compressor(...)      # ratio=4 写 C4，ratio=128 写 C128
```

整体可以记成：

```text
layer_id
  └─► config.compress_ratios[layer_id]
        ├─ 0   -> SWA 层
        ├─ 4   -> CSA 层 = MQA + C4 compressor + C4Indexer top-k
        └─ 128 -> HCA 层 = MQA + C128 compressor
```

常见分布在 `architecture.md` 里也列过：

| Variant | `compress_ratio` 分布 |
|---|---|
| Flash | `0×3, 4×21, 128×20` |
| Flash-0731 / Flash-Vision | `0×5, 4×21, 128×20` |
| Pro | `0×1, 4×30, 128×31` |
| Pro-0813 | `0×3, 4×30, 128×31` |

注意：表里的 `0×3, 4×21, 128×20` 是数量统计，不保证真实列表一定是“先 3 层 0，再 21 层 4，再 20 层 128”。真正的堆叠顺序要看 checkpoint `config.json` 里的完整 `compress_ratios`：

```python
for layer_id in range(num_hidden_layers):
    ratio = config.compress_ratios[layer_id]
    layers.append(DeepseekV4DecoderLayer(..., compress_ratio=ratio))
```

层间堆叠的心智图：

```text
token hidden
  │
  ▼
Layer 0  : ratio = compress_ratios[0]   -> SWA / CSA / HCA 之一
  │
  ▼
Layer 1  : ratio = compress_ratios[1]   -> SWA / CSA / HCA 之一
  │
  ▼
Layer 2  : ratio = compress_ratios[2]   -> SWA / CSA / HCA 之一
  │
  ▼
...
  │
  ▼
Layer N-1: ratio = compress_ratios[N-1] -> SWA / CSA / HCA 之一
```

所以：

```text
层间：SWA / CSA / HCA 按 config 列表堆叠。
层内：一层只按自己的 ratio 走一种主 attention 路径。
```

#### Sliding window 是不是每个 attention 里都有？

从工程实现和心智模型上，可以这样理解：

> **每层都有局部 SWA dense KV ring 作为近邻窗口基础；但只有 `ratio=0` 层是“纯 sliding window attention”。**

也就是说：

| 层类型 | 是否有局部 SWA dense KV | 是否额外读压缩历史 |
|---|---|---|
| `ratio=0` SWA | 有 | 没有 |
| `ratio=4` CSA | 有 | 有，读 C4 + top-k 稀疏历史 |
| `ratio=128` HCA | 有 | 有，读 C128 高压缩历史 |

可以画成：

```text
ratio=0:
  Q -> 最近 window 的 dense MQA KV

ratio=4:
  Q -> 最近 window 的 dense MQA KV
    -> C4Indexer 选中的 C4 压缩历史

ratio=128:
  Q -> 最近 window 的 dense MQA KV
    -> C128 压缩历史
```

所以不要把 `compress_ratio=4/128` 理解成“完全没有 sliding window”。更准确是：

```text
SWA 负责局部细节，压缩分支负责更远历史。
ratio=0 只有 SWA；
ratio=4/128 是 SWA + 对应压缩历史路径。
```

#### Dense window 和压缩历史怎么融合？

CSA / HCA 层最终不会给下游返回两个 attention 输出。它们会融合成 **一个** attention context：

```text
ctx: [T, A, D]
```

直觉上，不是这样：

```text
ctx = dense_window_attention(q) + compressed_history_attention(q)
```

而更像这样：

```text
候选 K/V = 最近 window 的 dense MQA KV
        + 远处的 compressed KV 候选

ctx = attention(q, 候选 K/V)
```

也就是说，局部 dense token 和远处 compressed block 会进入同一个 attention 归一化语义里，让 softmax 自己决定概率质量分给谁：

```python
def hybrid_attention(q, dense_window_kv, compressed_kv, compressed_indices=None):
    """
    q              : [T, A, D]
    dense_window_kv: [W, 1, D]，W 通常是 window_size=128
    compressed_kv  : [N_selected, 1, D]，教学抽象；真实实现是 packed/compressed 可读布局

    return:
      ctx: [T, A, D]
    """

    # 1. 局部窗口 logits
    dense_logits = q @ dense_window_kv.K.transpose(-1, -2)
    # dense_logits: [T, A, W]

    # 2. 压缩历史 logits
    # CSA: compressed_indices 来自 C4Indexer top-k，只读被选中的 C4 blocks
    # HCA: 读 C128 高压缩历史
    compressed_logits = q @ read_compressed_keys(compressed_kv).transpose(-1, -2)
    # compressed_logits: [T, A, N_selected]

    # 3. 合在同一个 softmax 里竞争注意力质量
    logits = concat([dense_logits, compressed_logits], dim=-1)
    # logits: [T, A, W + N_selected]

    weights = softmax(logits, dim=-1)
    # weights: [T, A, W + N_selected]

    # 4. 同一个 weights 同时读 dense V 和 compressed V
    compressed_values = read_compressed_values(compressed_kv)
    values = concat([dense_window_kv.V, compressed_values], dim=0)
    # values: [W + N_selected, 1, D]

    ctx = weights @ values
    # ctx: [T, A, D]
    return ctx
```

上面是教学版。真实 SGLang backend 不一定真的把 dense / compressed logits 先 `concat` 成一个大张量；为了省显存和带宽，kernel 可能分段算 logits、维护 running max / sum，再得到等价的 softmax 结果。但语义上可以先按“同一个 attention 归一化空间”理解。

三种层可以对比成：

```text
ratio=0 SWA:
  candidates = dense_window
  ctx = attention(q, candidates)

ratio=4 CSA:
  selected_c4 = C4Indexer(q_lora, c4_indexer_k_cache).topk()
  candidates = dense_window + main_c4_compressed_kv[selected_c4]
  ctx = attention(q, candidates)

ratio=128 HCA:
  candidates = dense_window + c128_compressed_kv
  ctx = attention(q, candidates)
```

最终下游只看到：

```text
ctx [T,A,D] -> wo_a/wo_b -> out [T,H] -> mHC 写回 residual
```

### 8.2 `ratio == 0`：SWA / 纯窗口注意力

```text
Q 当前 token
   │
   ▼
读取最近窗口内的 dense MQA KV cache
```

特点：

1. 不创建 compressor / indexer。
2. 主要看局部窗口，默认 `window_size = 128`。
3. 适合保留短程细节。

### 8.3 `ratio == 4`：CSA

CSA = Compressed Sparse Attention。

```text
dense KV ─► Compressor(4x, overlap=True) ─► C4 compressed states
                                            │
Q ─► C4Indexer ─► top-k 历史块              │
                                            ▼
                              稀疏 MQA 读取被选中的历史
```

关键组件：

| 组件 | 作用 |
|---|---|
| `Compressor(4)` | 把历史 KV 按约 4 倍压缩成 compressed states |
| `C4Indexer` | 用压缩索引向量做 top-k 检索 |
| `index_topk` | Flash 常见 512，Pro 常见 1024 |
| `overlap=True` | 压缩窗口有重叠，减少边界信息损失 |

直觉：

> CSA 不是让每个 token 看完整历史，而是先用压缩索引找「可能相关」的历史块，再做稀疏读取。

### 8.4 `ratio == 128`：HCA

HCA = Highly Compressed Attention。

```text
dense KV ─► Compressor(128x) ─► 高压缩历史状态
Q ───────────────────────────► 读取压缩历史
```

特点：

1. 压缩比很高，服务长上下文主路径。
2. 没有 C4Indexer 那样的 top-k indexer。
3. 可配合 online compress，让 decode 阶段更省 cache 和读带宽。

### 8.5 三类层的分工

| 类型 | 看什么 | 长处 |
|---|---|---|
| SWA | 最近窗口 dense KV | 局部细节 |
| CSA | 4x 压缩 + top-k 稀疏历史 | 中长程相关片段 |
| HCA | 128x 高压缩历史 | 超长上下文粗粒度记忆 |

所以 V4 的 attention 不是单一机制，而是：

```text
MQA 负责减少 KV head 数
CSA/HCA 负责减少长历史状态
SWA 负责保住局部细节
```

### 8.6 KV cache / C4 / C128 / C4Indexer 写入逻辑

SGLang 里不要把 `kv_cache.write_dense(k, v)` 理解成一个真实存在的简单 Python 调用。DeepSeek-V4 的热路径做了大量 fused 写入：

| 状态 | 由谁产生 | 写到哪里 | 什么时候有 |
|---|---|---|---|
| SWA dense MQA KV | `wkv + kv_norm + RoPE` | `swa_kv_pool` 或 unified KV 的 SWA ring | 所有层 |
| C4 attention KV | `Compressor(ratio=4, is_in_indexer=False)` | `c4_kv_pool` / extra key buffer | `compress_ratio == 4` |
| C128 attention KV | `Compressor(ratio=128, is_in_indexer=False)` | `c128_kv_pool` / extra key buffer | `compress_ratio == 128` |
| C4 indexer K | `C4Indexer.compressor(ratio=4, is_in_indexer=True)` | `c4_indexer_kv_pool` | 只有 CSA 层 |
| C4/C128 raw compress state | `Compressor` fused kernel 的中间 `kv_score_buffer` | `CompressStatePool` | 只有压缩层 |
| top-k 稀疏索引 | `C4Indexer.forward_c4_indexer` | `core_metadata.c4_sparse_page_indices` | 只有 CSA 层 |

核心数据流可以画成：

```text
所有层:
  hidden y
    └─► wkv / kv_norm / RoPE
          └─► SWA dense MQA KV ring

CSA 层 ratio=4:
  hidden y
    ├─► main Compressor(4)
    │     ├─► attention CompressStatePool
    │     └─► C4 extra KV pages
    │
    └─► C4Indexer
          ├─► indexer Compressor(4)
          │     ├─► indexer CompressStatePool
          │     └─► C4 indexer K pages
          ├─► q_lora -> indexer Q
          ├─► weights_proj(y)
          └─► paged MQA logits -> top-k -> c4_sparse_page_indices

HCA 层 ratio=128:
  hidden y
    └─► main Compressor(128)
          ├─► attention CompressStatePool
          └─► C128 extra KV pages
```

几个实现细节很容易混：

1. **SWA KV 写入总是存在**：即使当前层是 C4 / C128，局部窗口仍需要 dense MQA KV。源码注释里写得很直白：KV cache write fused into K kernel。
2. **C4/C128 extra KV 不是同一个池**：`DeepSeekV4TokenToKVPool` 会按层的 `compress_ratio` 建 `c4_kv_pool` / `c128_kv_pool`，并用 `layer_mapping[layer_id]` 把真实 layer id 映射到压缩池里的局部 layer id。
3. **C4 有两套压缩写入**：
   - main compressor：给 attention 读历史用。
   - indexer compressor：给 top-k 检索用，写的是 indexer K cache。
4. **C128 没有 C4Indexer**：它只走 main compressor，不产生 `c4_sparse_page_indices`。
5. **`CompressStatePool` 保存的是压缩过程的 kv+score 状态**：普通离线模式下最后一维是 `2 * (1 + overlap) * head_dim`；C4 因为 `overlap=True`，所以比 C128 多一份 overlap 状态。online C128 会变成请求级的 compact state，ring size 可塌到 1。

block / page / ring 的关系可以先这样记：

```text
full token loc
  └─► SWA ring loc
        ├─► SWA dense KV: 直接按窗口 ring 写
        └─► C4 state loc: 先映射到 SWA page/ring，再除以 4 得到压缩块 loc

request position
  └─► C128 state loc: req_pool_idx * ring_size + position % ring_size
        └─► 再除以 128 得到压缩块 loc
```

默认 ring 大小：

| ratio | 普通 decode ring | speculative ring | online |
|---|---:|---:|---:|
| C4 `ratio=4` | 8 | 16 | 不适用 |
| C128 `ratio=128` | 128 | 256 | 1 |

extra KV pool 的 page size 会按压缩比缩小：

```text
c4_page_size   = global_page_size / 4
c128_page_size = global_page_size / 128
```

所以 C4 / C128 的写入不是“往 token 序列 append 一个 token”，而是按 `out_loc` 写入压缩后的 page/block 位置；`out_loc` 由 backend metadata 预先根据 prefill/decode、请求长度、SWA ring 映射等算好。

### 8.7 C4 overlap 伪代码

先写一个不 overlap 的压缩。假设 `ratio=4`，每 4 个 token 压成 1 个 block：

```python
def compress_no_overlap(kv_score, ratio=4):
    """
    kv_score: [T, 2 * D]
      - 前 D 维可以理解成待压缩 KV-like 内容
      - 后 D 维可以理解成 score / gate，用来做加权压缩

    return:
      blocks: [ceil(T / 4), D]
    """
    blocks = []
    for start in range(0, T, ratio):
        window = kv_score[start : start + ratio]       # [<=4, 2D]
        block = weighted_compress(window)              # [D]
        blocks.append(block)
    return stack(blocks)                               # [ceil(T / 4), D]
```

这种切法的问题是边界很硬：

```text
B0 = compress(tokens 0,1,2,3)
B1 = compress(tokens 4,5,6,7)

跨边界片段 2,3,4,5 会被拆成两半。
```

C4 的 `overlap=True` 可以概念化成：每个 4-token block 除了自己的主窗口，还维护一份相邻窗口相关的压缩通道。源码里的关键开关是：

```python
overlap = ratio == 4
coff = 1 + overlap      # C4: 2, C128: 1
```

所以 C4 的中间状态最后一维更宽：

```text
C4 state   : 2 * coff * D = 4D
C128 state : 2 * coff * D = 2D

其中:
  外层 2   = kv 部分 + score 部分
  coff=2  = main 通道 + overlap 通道
```

教学版可以这样写：

```python
def compress_c4_with_overlap(kv_score, ratio=4):
    """
    kv_score: [T, 4D]
      因为 C4 overlap=True，所以可以拆成:
        main_kv      : [T, D]
        overlap_kv   : [T, D]
        main_score   : [T, D]
        overlap_score: [T, D]

    return:
      main_blocks   : [ceil(T / 4), D]
      overlap_blocks: [ceil(T / 4), D]
    """
    main_kv, overlap_kv, main_score, overlap_score = split_4xD(kv_score)

    main_blocks = []
    overlap_blocks = []

    for block_id, start in enumerate(range(0, T, ratio)):
        end = start + ratio

        # 主窗口：当前 4 个 token
        main_window_kv = main_kv[start:end]             # [<=4, D]
        main_window_score = main_score[start:end]       # [<=4, D]
        main_block = weighted_compress(
            main_window_kv, main_window_score
        )                                               # [D]

        # overlap 窗口：教学版写成“向前看一个 4-token 邻居”
        # 真实 kernel 用 write_overlap_loc / extra_data 在 ring 中处理边界位置。
        overlap_start = max(0, start - ratio)
        overlap_end = end
        overlap_window_kv = overlap_kv[overlap_start:overlap_end]        # [<=8, D]
        overlap_window_score = overlap_score[overlap_start:overlap_end]  # [<=8, D]
        overlap_block = weighted_compress(
            overlap_window_kv, overlap_window_score
        )                                                               # [D]

        main_blocks.append(main_block)
        overlap_blocks.append(overlap_block)

    return stack(main_blocks), stack(overlap_blocks)
```

上面把两条通道分开返回，只是为了看清 overlap。真实 kernel 会把它们**融合成一个最终 block 向量**，给 main compressor 时叫 `new_compressed_kv`，给 C4Indexer compressor 时就是 `index_k`。教学版合并可以这样写：

```python
def merge_main_overlap_blocks(main_blocks, overlap_blocks, main_scores, overlap_scores):
    """
    main_blocks   : [N, D]
    overlap_blocks: [N, D]
    main_scores   : [N, D] 或 [N, 1]，表示主通道可信度 / gate
    overlap_scores: [N, D] 或 [N, 1]，表示 overlap 通道可信度 / gate

    return:
      merged_blocks: [N, D]

    注意：这是教学版。源码里的 fused compressor kernel 会把 kv/score/APE/loc
    一起算掉，不会真的在 Python 里先生成这些 blocks 再 merge。
    """

    # 1. 把 main / overlap 的 score 变成两路权重。
    # 如果 score 是 [N,D]，就是逐维 gate；如果是 [N,1]，就是整块 gate。
    two_scores = stack([main_scores, overlap_scores], dim=1)      # [N, 2, D] or [N, 2, 1]
    weights = softmax(two_scores, dim=1)                          # [N, 2, D] or [N, 2, 1]

    w_main = weights[:, 0]                                        # [N, D] or [N, 1]
    w_overlap = weights[:, 1]                                     # [N, D] or [N, 1]

    # 2. 用 gate 把两条通道融合成一个 D 维 block。
    merged = w_main * main_blocks + w_overlap * overlap_blocks    # [N, D]

    # 3. Compressor 后处理：归一化 + RoPE；indexer 路径还会量化打包。
    merged = rms_norm(merged)                                     # [N, D]
    merged = apply_rope_to_compressed_block(merged)               # [N, D]
    return merged
```

如果放到 C4Indexer 里，语义就是：

```python
def build_index_k_from_overlap(kv_score, positions):
    """
    kv_score : [T, 4 * index_D]
    positions: [T]

    return:
      index_k: [N_new_c4_blocks, index_D]
    """
    main_blocks, overlap_blocks = compress_c4_with_overlap(kv_score, ratio=4)
    # main_blocks / overlap_blocks: [N_new_c4_blocks, index_D]

    main_scores, overlap_scores = compute_block_scores(kv_score, ratio=4)
    # main_scores / overlap_scores: [N_new_c4_blocks, index_D] or [N_new_c4_blocks, 1]

    index_k = merge_main_overlap_blocks(
        main_blocks,
        overlap_blocks,
        main_scores,
        overlap_scores,
    )                                                             # [N_new_c4_blocks, index_D]

    index_k = quantize_for_indexer_cache(index_k)                 # FP8 or FP4 packed
    return index_k
```

所以“合并”不是简单 `concat([main, overlap])`，否则维度会变成 `2D`。它更像：

```text
main / overlap 两个 D 维候选
  └─► 根据 score/gate 做加权融合
        └─► 得到一个 D 维 compressed block
              └─► 写入 c4_kv_pool 或 c4_indexer_kv_pool
```

上面是教学版，真实 decode 更像“边走边写 ring”。核心位置计算可以简化成：

```python
def c4_decode_write_locs(seq_lens, req_pool_indices, req_to_token, token_to_kv_pool):
    """
    seq_lens        : [B]，每个 request 当前长度
    req_pool_indices: [B]

    return:
      write_loc        : [B]，当前 C4 block 的压缩写入位置
      write_overlap_loc: [B]，前一个 C4 block 的 overlap 写入位置
    """

    def clip_down(pos, ratio=4):
        # 把 token position 对齐到 4-token block 起点
        return pos // ratio * ratio

    def get_raw_loc(positions):
        # C4 先走 full token loc -> SWA loc -> state loc
        full_loc = req_to_token[req_pool_indices, positions]      # [B]
        swa_loc = token_to_kv_pool.translate_loc_from_full_to_swa(full_loc)
        swa_pages = swa_loc // token_to_kv_pool.swa_page_size
        state_loc = (
            swa_pages * token_to_kv_pool.get_ring_size(4)
            + swa_loc % token_to_kv_pool.get_ring_size(4)
        )
        return state_loc // 4                                     # [B]

    # 当前 token 落在哪个 4-token block
    write_positions = clip_down(seq_lens - 1)                     # [B]
    write_loc = get_raw_loc(write_positions)                      # [B]

    # overlap 额外关联前一个 4-token block
    write_overlap_positions = write_positions - 4                 # [B]
    write_overlap_loc = get_raw_loc(write_overlap_positions)      # [B]

    return write_loc, write_overlap_loc
```

读源码时，对应关系是：

```text
is_overlap_compress(4) = True
create_paged_compressor_data(...)
  write_loc         = get_raw_loc(clip_down(seq_lens - 1))
  write_overlap_loc = get_raw_loc(clip_down(seq_lens - 1) - 4)
  extra_data        = write_overlap_loc.view(-1, 1)

Compressor.forward_compress(...)
  用 write_loc + extra_data 驱动 fused compressor kernel
```

所以 overlap 的一句话直觉是：

> C4 不只维护“当前 4-token block 的压缩表示”，还给相邻 block 边界多留一条压缩通道；这样 top-k 和 sparse attention 读 C4 block 时，边界语义不至于被 4-token 分块硬切断。

---

## 9. Output Projection：低秩 O 投影

Attention 得到每个 Q head 的输出：

```text
attn_out: [T,A,D]
```

展平后理论上是：

```text
[T, A*D]
```

DeepSeek-V4 不直接做一个大的 `A*D -> H`，而是用 `wo_a / wo_b` 两段，并按 `o_groups` 分组：

```text
attn_out [T,A,D]
      │
      ▼
wo_a，按 o_groups 分组
      │
      ▼
o_lora [T,R_o]
      │
      ▼
wo_b
      │
      ▼
out [T,H]
```

和 Q 低秩投影类似，O 投影也用低秩中间维控制参数量和计算量。`o_groups` 的存在说明输出投影不是完全一整块 dense 矩阵，而是先分组压缩，再汇总回 hidden。

---

## 10. Attention Sink

`MQALayer` 里还有 `attn_sink`，通常是 per-head 的浮点参数。

直觉上，它给每个 head 一个额外的「稳定吸收位」或偏置项，让注意力在长上下文 / 流式 decode 时不必把概率质量全压到真实历史 token 上。你可以把它当成 attention 分布里的稳定锚点：

```text
softmax([QK logits, sink logit])
```

学习时模型会决定每个 head 多依赖这个 sink。对读代码来说，重点不是把它当成普通 token，而是注意它会改变 attention logits / softmax 的归一化空间。

---

## 11. Decode 阶段为什么省

自回归 decode 每一步大致要做：

```text
新 token 的 Q
  ×
所有历史 token 的 K
  →
attention weights
  ×
所有历史 token 的 V
```

瓶颈常常不是单个新 token 的 Q 投影，而是反复读历史 K/V。MQA 的收益集中在这里：

```text
历史 K/V 从 [T,A,D] 变成 [T,1,D]
```

而 CSA/HCA 进一步处理长历史：

```text
短历史：dense MQA KV
中长历史：C4 compressed states + top-k
超长历史：C128 compressed states
```

所以可以分两层理解：

1. **MQA 先让每个 token 的 KV cache 变薄**。
2. **压缩注意力再让长上下文历史变短 / 变稀疏**。

---

## 12. 教学版伪代码

下面不是源码逐行翻译，而是为了学习 attention layer 的语义。

形状约定：

```text
T       = 当前参与计算的 token 数
T_cache = 当前层 dense KV cache 里的历史 token 数
N_c4    = 当前层 C4 cache 里的压缩块数
N_c128  = 当前层 C128 cache 里的压缩块数
N_new_c4_blocks   = 本次 forward 新产生的 C4 压缩块数
N_new_c128_blocks = 本次 forward 新产生的 C128 压缩块数
H       = hidden_size
A       = num_attention_heads
D       = head_dim = qk_nope_head_dim + qk_rope_head_dim
D_nope  = qk_nope_head_dim，Flash 常见 448
D_rope  = qk_rope_head_dim，Flash 常见 64
R_q     = q_lora_rank
R_o     = o_lora_rank
K_top   = layer_cfg.index_topk，仅 CSA 分支使用
index_heads = config.index_n_heads
index_D     = config.index_head_dim
```

```python
def write_dense_swa_kv(k, v, positions, kv_cache, layer_id):
    """
    语义：写局部窗口 dense MQA KV。

    k/v       : [T, 1, D]
    positions : [T]

    逻辑结果:
      SWA_K_cache: [num_swa_slots, 1, D]
      SWA_V_cache: [num_swa_slots, 1, D]

    SGLang 实现提示:
      - 热路径不是 Python append，而是 fused 到 K kernel / attention backend。
      - 文档里按 K/V 逻辑拆开；源码里常叫 key buffer，实际是 FlashMLA packed cache。
      - 常见入口是 set_swa_key_buffer_radix_fused_norm_rope 或 unified_kv scatter。
      - 物理位置是 SWA ring loc，约等于 req_slot * window + position % window。
    """
    swa_loc = kv_cache.make_swa_loc(positions)        # [T]
    kv_cache.write_swa_ring(layer_id, swa_loc, k, v)  # semantic write


def run_main_compressor(y, positions, ratio, kv_cache, layer_id):
    """
    语义：C4/C128 attention 主压缩器。

    y         : [T, H]
    positions : [T]
    ratio     : 4 or 128

    内部近似:
      kv_score = wkv_gate(y)       # [T, 2 * coff * D]
      state    = fused_compress(kv_score, ape, plan)
      state    = norm + RoPE       # compressed KV-like vectors

    coff = 2 if ratio == 4 else 1
    state buffer last dim:
      ratio=4   : 2 * coff * D = 4D   # overlap=True
      ratio=128 : 2 * coff * D = 2D

    SGLang 实现提示:
      - Compressor.forward_native 先写/读 attention CompressStatePool.kv_score_buffer。
      - backend.forward_core_compressor 再把 new_compressed_kv 写入 extra KV pool。
    """
    compressor = kv_cache.get_main_compressor(layer_id, ratio)
    compressed_kv = compressor(y, positions)
    # [N_new_blocks, D]，其中:
    #   ratio=4   -> N_new_blocks = N_new_c4_blocks
    #   ratio=128 -> N_new_blocks = N_new_c128_blocks

    out_loc = kv_cache.get_compress_out_loc(layer_id, ratio)
    # ratio=4   -> core_metadata.c4_out_loc
    # ratio=128 -> core_metadata.c128_out_loc
    # out_loc: [N_new_blocks]

    kv_cache.write_extra_kv(layer_id, ratio, out_loc, compressed_kv)
    # ratio=4   -> c4_kv_pool.set_key_buffer_fused(...)
    # ratio=128 -> c128_kv_pool.set_key_buffer_fused(...)

    return compressed_kv, out_loc


def run_c4_indexer(y, q_lora, positions, kv_cache, layer_id):
    """
    语义：CSA 层专用的 C4Indexer。

    y         : [T, H]
    q_lora    : [T, R_q]
    positions : [T]

    输出:
      selected_blocks / c4_sparse_page_indices: [T, index_heads or A, K_top]

    SGLang 实现提示:
      C4Indexer 有自己的 Compressor(is_in_indexer=True)，和 main Compressor(4)
      是两套写入：一个服务检索，一个服务 attention 读取。
    """

    # A. 写 indexer 的压缩 K cache
    indexer_compressor = kv_cache.get_indexer_compressor(layer_id)
    index_k = indexer_compressor(y, positions)       # [N_new_c4_blocks, index_D]
    c4_out_loc = kv_cache.get_compress_out_loc(layer_id, ratio=4)
    kv_cache.write_index_k(layer_id, c4_out_loc, index_k)
    # FP8 路径: set_index_k_fused(...)
    # FP4 路径: set_index_k_fp4(...)

    # B. 生成 indexer query 和 per-head weights
    q_indexer = indexer_wq_b(q_lora)                 # [T, index_heads * index_D]
    q_indexer = q_indexer.reshape(T, index_heads, index_D)
    q_indexer = apply_rope_or_hadamard(q_indexer, positions)
    # q_indexer: [T, index_heads, index_D]

    weights = weights_proj(y)                        # [T, index_heads]

    # C. 对 indexer K cache 做 paged MQA logits
    index_k_cache = kv_cache.get_index_k_cache(layer_id)
    # index_k_cache: paged [N_c4_pages, c4_page_size, 1, index_D + scale]

    logits = paged_mqa_logits(
        q_indexer,                                   # [T, index_heads, index_D]
        index_k_cache,
        weights,                                     # [T, index_heads]
    )
    # logits: [T, N_c4_visible_blocks]

    # D. top-k raw block index -> physical sparse page index
    selected_blocks = topk_transform_paged(
        logits,
        page_table=kv_cache.c4_page_table(layer_id),
        topk=K_top,
    )
    # selected_blocks: [T, index_heads or A, K_top]
    # 实现里写入 core_metadata.c4_sparse_page_indices
    return selected_blocks


def dsv4_mqa_layer(y, positions, kv_cache, layer_cfg):
    """
    y         : [T, H]，已经经过 mHC pre + RMSNorm
    positions : [T]
    kv_cache  : 当前层已有历史，逻辑上保存 dense KV / C4 / C128 等状态
    """

    # 1. Q: low-rank multi-head query
    q_lora = wq_a(y)                 # [T, H] -> [T, R_q]
    q_lora = q_norm(q_lora)          # [T, R_q]
    q = wq_b(q_lora)                 # [T, R_q] -> [T, A * D]
    q = q.reshape(T, A, D)           # [T, A, D]

    # 2. KV: single-head key/value
    kv = wkv(y)                      # [T, H] -> single KV path
    kv = kv_norm(kv)                 # [T, kv_inner_dim]
    k, v = split_kv(kv)              # k: [T, 1, D], v: [T, 1, D]

    # 3. Split nope / rope and apply RoPE only to rope part
    q_nope, q_rope = split(q, [D_nope, D_rope], dim=-1)
    # q_nope: [T, A, D_nope], q_rope: [T, A, D_rope]

    k_nope, k_rope = split(k, [D_nope, D_rope], dim=-1)
    # k_nope: [T, 1, D_nope], k_rope: [T, 1, D_rope]

    q_rope = apply_rope(q_rope, positions)  # [T, A, D_rope]
    k_rope = apply_rope(k_rope, positions)  # [T, 1, D_rope]

    q = concat(q_nope, q_rope, dim=-1)   # [T, A, D]
    k = concat(k_nope, k_rope, dim=-1)   # [T, 1, D]

    # 4. SWA dense KV write: all layers have this local-window cache.
    write_dense_swa_kv(k, v, positions, kv_cache, layer_cfg.layer_id)
    # Logical dense cache after write:
    #   SWA K/V window: [min(T_cache + T, window_size), 1, D]

    if layer_cfg.compress_ratio == 4:
        # 5a. CSA main compressed KV write.
        c4_kv, c4_out_loc = run_main_compressor(
            y, positions, ratio=4, kv_cache=kv_cache, layer_id=layer_cfg.layer_id
        )
        # c4_kv     : [N_new_c4_blocks, D] 语义形状
        # c4_out_loc: [N_new_c4_blocks]
        # C4 extra KV cache after write: roughly [N_c4 + N_new_c4_blocks, D]

        # 5b. CSA indexer writes its own C4 index K, then produces top-k blocks.
        selected_blocks = run_c4_indexer(
            y, q_lora, positions, kv_cache, layer_id=layer_cfg.layer_id
        )
        # selected_blocks / c4_sparse_page_indices: [T, A or index_heads, K_top]

        # 5c. Sparse MQA reads SWA + selected C4 pages.
        ctx = sparse_mqa(q, kv_cache, selected_blocks, attn_sink=attn_sink)
        # q: [T, A, D], selected history is sparse
        # ctx: [T, A, D]

    elif layer_cfg.compress_ratio == 128:
        # 5. HCA compressed KV write.
        c128_kv, c128_out_loc = run_main_compressor(
            y, positions, ratio=128, kv_cache=kv_cache, layer_id=layer_cfg.layer_id
        )
        # c128_kv     : [N_new_c128_blocks, D] 语义形状
        # c128_out_loc: [N_new_c128_blocks]
        # C128 extra KV cache after write: roughly [N_c128 + N_new_c128_blocks, D]

        ctx = compressed_mqa(q, kv_cache, c128_kv, attn_sink=attn_sink)
        # q: [T, A, D], compressed history is highly compact
        # ctx: [T, A, D]

    else:
        # 5. Pure SWA layer: no Compressor, no C4Indexer.
        ctx = sliding_window_mqa(q, kv_cache, window_size=layer_cfg.window_size,
                                 attn_sink=attn_sink)
        # Reads recent dense K/V window:
        #   K_window/V_window: [min(T_cache + T, window_size), 1, D]
        # ctx: [T, A, D]

    # 6. Low-rank output projection
    ctx = ctx.reshape(T, A * D)       # [T, A, D] -> [T, A * D]
    out_lora = wo_a_grouped(ctx)      # [T, A * D] -> [T, R_o]
    out = wo_b(out_lora)             # [T, R_o] -> [T, H]
    return out
```

读这段伪代码时，重点看三条线：

1. `q_lora -> q -> attention`：Q 是多头查询。
2. `k/v -> kv_cache`：KV 是单头共享缓存。
3. `compress_ratio`：决定历史怎么被读。

---

## 13. 和 MLA 的区别

DeepSeek-V2/V3 常见关键词是 MLA（Multi-head Latent Attention），DeepSeek-V4 这里变成 MQA + compressed attention。

非常粗略地对比：

| 维度 | MLA | DeepSeek-V4 MQA |
|---|---|---|
| KV 压缩思路 | 把 K/V 存成 latent，再恢复/参与计算 | 直接使用单 KV head，并叠加压缩状态 |
| KV heads | 不是简单单头共享 | `num_key_value_heads = 1` |
| 长上下文机制 | 依赖 latent KV / RoPE / 后端优化 | SWA + CSA + HCA |
| 读代码关注点 | latent cache 如何生成和还原 | `MQALayer`、`Compressor`、`C4Indexer`、KV pool |

一句话：

> MLA 是「把 KV 表示低秩化」；V4 的 MQA 是「把 KV head 数压到 1」，再用 CSA/HCA 处理长历史。

---

## 14. 读源码顺序

建议按这个顺序读：

1. `DeepseekV4DecoderLayer.forward`
   - 先看 attention 外层的 mHC 调用边界。
2. `MQALayer.__init__`
   - 找 `num_attention_heads`、`num_key_value_heads`、`q_lora_rank`、`o_lora_rank`。
3. `MQALayer.forward`
   - 跟 `wq_a / wqkv_a`、`q_norm`、`wq_b`、`wkv`、`kv_norm`、`wo_a`、`wo_b`。
4. `Compressor`
   - 看 `compress_ratio == 4 / 128` 时压缩状态如何生成。
   - 重点函数：`forward_native`、`forward_core_compressor`、`forward_indexer_compressor`。
5. `C4Indexer`
   - 只在 CSA 层重点看，理解 top-k 检索怎么驱动稀疏 attention。
   - 重点函数：`compute_q`、`compute_weights`、`forward_c4_indexer`、`topk_transform_paged`。
6. `DeepSeekV4TokenToKVPool`
   - 对照 KV cache、compress state、indexer cache 的真实内存布局。
   - 重点看：`swa_kv_pool`、`c4_kv_pool`、`c128_kv_pool`、`c4_indexer_kv_pool`、`compress_state_pools`、`indexer_compress_state_pools`。
7. `create_paged_compressor_data`
   - 看 C4 / C128 的 `out_loc` 如何从 prefill/decode、SWA ring、request position 推出来。

---

## 15. 常见误解

### 15.1 MQA 不是只有一个 attention head

MQA 只有一个 **KV head**，不是只有一个 **Q head**。V4 仍然有 64 / 128 个 Q heads。

```text
错误：MQA = 单头注意力
正确：MQA = 多个 Q heads 共享一组 K/V
```

### 15.2 单 KV head 不代表所有 head 输出相同

因为每个 Q head 不同：

```text
softmax(Q_head_0 @ K) != softmax(Q_head_1 @ K)
```

共享的是 K/V，不是 attention 分布。

### 15.3 `q_lora_rank` 不是 LoRA 微调开关

这里的 `q_lora_rank` 是结构里的低秩投影维度，不是外部 PEFT LoRA adapter。

### 15.4 CSA/HCA 不是 MQA 的替代

它们是叠加关系：

```text
MQA: KV head 数变少
CSA/HCA: 历史状态被压缩 / 稀疏读取
```

### 15.5 mHC 不是 Attention 内部机制

mHC 包住 Attention layer，负责 residual 多路混合。`MQALayer` 本体仍然吃 `[T,H]`，输出 `[T,H]`。

---

## 16. 最小心智模型

如果只想先记住一张图：

```text
                 ┌──────────── Q: many heads ────────────┐
hidden [T,H] ────┤                                         ├──► attention ─► out [T,H]
                 └──────────── K/V: one shared head ──────┘

短历史：窗口 dense MQA
中历史：C4 压缩 + indexer top-k
长历史：C128 高压缩状态
```

如果只想先记住一句话：

> DeepSeek-V4 的 attention layer 用 MQA 把 KV cache 变薄，用 CSA/HCA 把长历史变短，再用低秩 Q/O 投影控制计算和参数量。

---

*整理日期：2026-09-07。*
