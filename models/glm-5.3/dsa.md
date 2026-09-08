# GLM-5.3 DSA 详解

> 基于 SGLang text-only 路径整理：
> `GlmMoeDsaForCausalLM -> DeepseekV2AttentionMLA -> Indexer / IndexerKPool`。
> 主源码：`python/sglang/srt/models/deepseek_v2.py`、`python/sglang/srt/layers/attention/dsa/`。

---

## 1. 一句话总览

DSA = Dynamic Sparse Attention。它解决的问题是：1M context 下，如果每个 query 都全量扫所有历史，即使有 MLA latent cache，attention 计算也会很重。

DSA 做的事：

```text
当前 query
  │
  ▼
轻量 indexer
  │
  ▼
对历史 index key 打分
  │
  ▼
top-k 选出相关历史位置 / block
  │
  ▼
MLA attention backend 只读这些 selected 历史
```

一句话：

> MLA 负责“历史怎么压缩存储”，DSA 负责“长历史里先读哪些”。

---

## 2. SGLang 里怎么挂上 DSA

在 `DeepseekV2AttentionMLA.__init__` 里：

```python
self.use_dsa = is_deepseek_dsa(config)
self.indexer = None

if self.use_dsa:
    self.skip_topk = dsa_layer_skips_topk(config, layer_id)
    self.next_skip_topk = dsa_layer_skips_topk(config, layer_id + 1)

    if not self.skip_topk or is_nextn:
        indexer_cls = IndexerKPool if get_dsa_index_kpool(config) > 1 else Indexer
        self.indexer = indexer_cls(...)
```

几个关键词：

| 字段 / 函数 | 含义 |
|---|---|
| `is_deepseek_dsa(config)` | 判断该模型是否启用 DSA 路径 |
| `dsa_layer_skips_topk(config, layer_id)` | 某层是否跳过 top-k，复用上一层结果 |
| `Indexer` | 基础 DSA indexer |
| `IndexerKPool` | KPool 版本 indexer，适合复用/池化 index keys |
| `index_topk` | 每个 query 选多少历史候选 |
| `index_n_heads` | indexer heads |
| `index_head_dim` | indexer 每头维度 |
| `index_kpool` | 是否启用 KPool indexer；不存在或等于 1 时不用 KPool |

源码里的 KPool 判定非常直接：

```python
def get_dsa_index_kpool(config):
    return getattr(config, "index_kpool", 1)

indexer_cls = (
    IndexerKPool if get_dsa_index_kpool(config) > 1 else Indexer
)
```

所以对 GLM-5.3 text-only 来说：

```text
config 里没有 index_kpool
  -> getattr(..., 1)
  -> 普通 Indexer

config.index_kpool = 1
  -> 普通 Indexer

config.index_kpool > 1
  -> IndexerKPool
```

也就是说，SGLang 的 GLM-5.3 DSA 路径“支持 KPool”，但实际是不是 KPool indexer，不能只看模型名字，要看 checkpoint 的 `config.json` 里 `index_kpool` 的值。

GLM-5.3 常见参数：

```text
index_topk = 2048
index_n_heads = 32
index_head_dim = 128
index_kpool = 未设置或 1 时为普通 Indexer；大于 1 时为 IndexerKPool
```

---

## 3. DSA 和 MLA 的分工

不要把 DSA index key 和 MLA latent KV 混成一件事。

| 机制 | 角色 | 典型维度 |
|---|---|---|
| MLA latent cache | 真正承载 attention 内容 | `kv_lora_rank + rope_dim = 512 + 64` |
| DSA index cache | 用来快速检索哪些历史值得读 | `index_head_dim = 128` |

流程上：

```text
DSA indexer cache
  └─ 负责 top-k 检索

MLA latent cache
  └─ 被 top-k 选中后，真正参与 attention 内容读取
```

所以 DSA 更像搜索系统里的“召回索引”，MLA latent cache 更像被召回后的“内容库”。

---

## 4. Indexer 的输入输出

形状约定：

```text
T       = 当前 query token 数
S       = 历史 index key 数
R_q     = q_lora_rank = 2048
I_heads = index_n_heads = 32
I_dim   = index_head_dim = 128
K_dsa   = index_topk = 2048
```

教学版：

```python
def dsa_indexer(q_lora, positions, index_cache):
    """
    q_lora     : [T, R_q]
    positions  : [T]
    index_cache: [S, I_heads, I_dim] 或 paged/packed 等价布局

    return:
      selected_indices: [T, I_heads, K_dsa] 或 backend 压缩后的 page/raw indices
    """
    index_q = indexer_q_proj(q_lora)                         # [T, I_heads * I_dim]
    index_q = index_q.reshape(T, I_heads, I_dim)             # [T, I_heads, I_dim]
    index_q = apply_indexer_rope(index_q, positions)         # [T, I_heads, I_dim]

    logits = paged_mqa_logits(index_q, index_cache)          # [T, I_heads, S]
    selected_indices = topk(logits, k=K_dsa)                 # [T, I_heads, K_dsa]
    return selected_indices
```

真实实现里，`logits` 可能不会以完整 `[T,I_heads,S]` 大矩阵长期保留；kernel 会用 paged layout、top-k transform、CUDA graph 友好的 buffer 来节省内存。

---

## 5. Index Key 从哪里来

DSA 需要历史 index key cache。它不是 MLA 的 `kv_latent` 原样复用，而是 indexer 自己维护的轻量检索表示。

概念流程：

```text
hidden / q_lora / KV latent side information
  │
  ▼
indexer key projection
  │
  ▼
index_k [T, index_heads, index_head_dim]
  │
  ▼
写入 DSA index cache
```

然后未来 token 来时：

```text
index_q @ historical index_k -> logits -> top-k
```

对于 `IndexerKPool`，历史 key 还会按 KPool 方式组织，让长上下文索引更省。注意：这只在 `index_kpool > 1` 时发生；普通 GLM-5.3 DSA config 如果没有这个字段，就会按默认值 1 走基础 `Indexer`。

---

## 6. skip_topk / top-k 复用

SGLang 里有两个很重要的字段：

```python
self.skip_topk = dsa_layer_skips_topk(config, layer_id)
self.next_skip_topk = dsa_layer_skips_topk(config, layer_id + 1)
```

直觉：

```text
有些层不重新算 top-k，而是复用上一层 / 相邻层的 top-k 结果。
```

为什么要这样做？

1. top-k 检索本身也有成本。
2. 相邻层关注的长程历史常常相似。
3. 复用 top-k 能减少 indexer 计算和 metadata 传递成本。

教学版：

```python
def maybe_run_dsa(layer, q_lora, prev_topk_indices):
    """
    prev_topk_indices: 上一层传来的 top-k 结果
    """
    if layer.skip_topk:
        selected = prev_topk_indices
    else:
        selected = layer.indexer(q_lora)

    if layer.next_skip_topk:
        next_topk_indices = selected
    else:
        next_topk_indices = None

    return selected, next_topk_indices
```

这解释了为什么 `DeepseekV2AttentionMLA.forward` 里有 `prev_topk_indices`，以及 `forward_core` 可能返回 `(hidden_states, topk_indices)`。

---

## 7. DSA 如何接入 MLA Attention

DSA 本身不产出最终 `ctx`。它只产出选择结果：

```text
selected_indices
```

然后 MLA attention backend 用这些 indices 去读 latent cache：

```python
def mla_with_dsa(q, kv_cache, selected_indices):
    """
    q: [T, A, D_qk]
    selected_indices: [T, ..., K_dsa]
    """
    hist_latent, hist_rope = kv_cache.read(selected_indices)
    # hist_latent: [selected, R_kv]
    # hist_rope  : [selected, D_rope]

    ctx = mla_attention(q, hist_latent, hist_rope)     # [T, A, D_v]
    return ctx
```

所以：

```text
DSA output -> attention 的读取范围
MLA output -> 真正传给 o_proj 的 context
```

---

## 8. KPool 是什么直觉

`IndexerKPool` 是 DSA indexer 的一个变体。名字里的 KPool 可以先理解成：历史 index key 不一定逐 token 平铺检索，而是可以组织成池化/分组结构。

SGLang 选择它的条件只有一个：

```python
config.index_kpool > 1
```

否则就是正常 DSA indexer：

```python
config.index_kpool <= 1 or config has no index_kpool
=> Indexer
```

心智模型：

```text
普通 Indexer:
  index_q 对大量 historical index_k 逐项打分

IndexerKPool:
  historical index_k 被组织进 K pools
  先在 pool / block 结构上更高效地定位候选
```

这和 DeepSeek-V4 里 C4Indexer 的“低维检索 cache”有点像，但 GLM-5.3 text-only 这里还是 DSA/MLA 路径，不是 C4/C128 压缩注意力。

---

## 9. 和 DeepSeek-V4 C4Indexer 的区别

| 项 | GLM-5.3 DSA | DeepSeek-V4 C4Indexer |
|---|---|---|
| 主 attention | MLA | MQA + C4 compressed attention |
| 检索对象 | DSA index cache / KPool | C4 indexer K cache |
| 选出结果 | top-k 历史 token/block indices | `c4_sparse_page_indices` |
| 被读取内容 | MLA latent KV cache | main C4 compressed KV |
| 压缩比例 | 不是 C4/C128 机制 | 明确 C4 4x |

两者共同点：

```text
都先用轻量检索表示做 top-k，再让主 attention 读取更重的内容表示。
```

---

## 10. 读源码顺序

1. `DeepseekV2AttentionMLA.__init__`
   - 看 `self.use_dsa`、`skip_topk`、`next_skip_topk`、`Indexer / IndexerKPool`。
2. `DeepseekV2AttentionMLA.forward`
   - 看 `prev_topk_indices` 如何传入。
3. `DeepseekV2AttentionMLA.op_core`
   - 看 DSA 模型可能返回 `(hidden_states, topk_indices)`。
4. `layers/attention/dsa/dsa_indexer.py`
   - 看基础 indexer 的 query/key/logits/top-k。
5. `layers/attention/dsa/dsa_indexer_kpool.py`
   - 看 KPool 版本。
6. `layers/attention/dsa/dsa_backend.py`
   - 看 prefill/decode 里 selected indices 如何喂给 attention backend。

---

## 11. 最小心智模型

```text
DSA = 长上下文 attention 的检索器。

它不直接生成最终 hidden；
它先用低维 index cache 选 top-k 历史；
然后 MLA attention 只读这些历史 latent；
最终 ctx -> o_proj -> hidden。
```
