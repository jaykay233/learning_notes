# Qwen Sparse Attention (QSA) 详解

> 针对 **Qwen3.8-Flash-Next**。总览见同目录 [architecture.md](./architecture.md)。
>
> 依据：HF `text_config`（`indexer_*` 字段）+ [LMSYS Day-0 博客](https://www.lmsys.org/blog/2026-08-26-qwen-flash-next/)。本地 SGLang 树若尚未合入 `qwen4_exp`，以下以公开说明为准。

---

## 1. 一句话

QSA = **先粗后细的稀疏 attention**：轻量 indexer 在 **压缩后的 micro-block** 上挑重要上下文，再对选出的位置做 **原始 K/V 上的稀疏 GQA**。

Flash-Next 里，hybrid 的「每 4 层第 4 个槽」不再是普通 Gated GQA，而是 QSA。

```text
[GDN → MoE] ×3 → [QSA → MoE] ×1   （重复 12 组 → 48 层）
```

---

## 2. 为什么要 QSA


| 路径      | 长序列代价                                           |
| ------- | ----------------------------------------------- |
| 全量 GQA  | 算力 + KV 访存都 ∝ 长度 L                              |
| 只有 GDN  | 状态固定，但缺精确 token 级检索                             |
| **QSA** | indexer 扫 ~L/4 个压缩 key；attention 只读 ~2K 个原始 K/V |


模型级 KV 省主要靠 hybrid（仅 12/48 层存增长 KV），不是靠丢掉 QSA 层里的原始 K/V——**原始 K/V 仍进 paged pool**；压缩 index key 只服务检索。

---



## 3. 官方相关配置（Flash-Next）


| 字段                       | 值    | 含义                                 |
| ------------------------ | ---- | ---------------------------------- |
| `num_attention_heads`    | 24   | 稀疏 GQA 的 Q heads                   |
| `num_key_value_heads`    | 2    | KV heads                           |
| `head_dim`               | 256  | GQA head 维                         |
| `partial_rotary_factor`  | 0.25 | RoPE 64 维                          |
| `indexer_n_heads`        | 4    | indexer query heads                |
| `indexer_kv_heads`       | 1    | indexer 共享 key                     |
| `indexer_head_dim`       | 128  | indexer 维                          |
| `indexer_compress_ratio` | 4    | **c4**：每 4 token → 1 个压缩 index key |
| `indexer_budget`         | 2048 | 展开后最多约 2048 逻辑位置（再加当前未满 block）     |


`layer_types` 里仍写 `"full_attention"`，实现上该槽是 QSA。

---



## 4. 两段式结构

```text
hidden h
  │
  ├─► Indexer 路径（轻）
  │     q^I (4×128), k^I (共享 1 head)
  │     每 4 个 raw index key → 平均 + norm + MRoPE → 压缩 block key
  │     score 所有可见 block → top blocks
  │
  └─► Sparse GQA 路径（重，但定长 budget）
        用选出的逻辑位置读 **原始** K/V
        标准（gated）GQA softmax attention
        → o_proj / output gate
```

要点：**压缩 key 只用于打分；最终 attention 的 K/V 不压缩。**

---



## 5. Indexer 打分（概念）

对 token `t`、**已完成**压缩 block `b`：

```text
s_{t,b} = 1/√128 · Σ_{h=1..4} ReLU( ⟨ q^I_{t,h} , k̄^I_b ⟩ )
```

- `q^I`：4 个 128 维 indexer query（当前 token）
- `k̄^I_b`：block `b` 的压缩 key（见下节）
- **block-causal**：只对「已满 4 token、整块可见」的 block 打分；当前未满块不进 `scores`，而是硬选进 attention（见 `unfinished_block_tokens`）

然后：

```python
def qsa_select(scores, compress_ratio=4, budget=2048):
    # scores: [T, num_complete_blocks]，num ≈ floor(L / 4)
    n_keep = budget // compress_ratio          # 2048/4 = 512 blocks
    top_blocks = topk(scores, k=n_keep)        # block ids
    positions = expand_blocks(top_blocks, ratio=4)
    positions = union(positions, unfinished_block_tokens)
    return positions                           # ≤ 2048 + 0..3 ≈ 2051
```

---



## 5.1 `update_compressed_index_cache` / `all_compressed_keys`

序列按 **非重叠 micro-block** 切：`r = indexer_compress_ratio = 4`。

```text
token:  0  1  2  3 | 4  5  6  7 | 8  9 10 11 | 12 13 ...
block:     b=0     |    b=1     |    b=2     |  未满尾块
```



### 压缩一个满块

对 block `b`（起始位置 `p_b = b · 4`），四个 raw indexer key：

```python
def compress_block(k_I[p : p+4], rope_pos=p):
    # 1) FP32 平均（先压内容，再统一位置 —— 避免不同 RoPE 相位相加）
    k_bar = mean(k_I[p:p+4], dim=0, dtype=fp32)   # [128]
    k_bar = rms_norm(k_bar)
    # 2) 用该块**首 token** 的 MRoPE / partial RoPE（64/128 维）
    k_bar = partial_rope(k_bar, position=p)
    return k_bar.to(bf16)                         # 写入压缩 index cache
```



### `update_compressed_index_cache(k_I, forward_batch)`

推理侧维护两类状态（每层 QSA、每请求）：

```python
# 持久：已完成块的压缩 key（paged，约 L/4 个，BF16）
compressed_index_cache[b] = k̄_b

# 请求本地：当前未满块的 raw key 暂存（固定 4 slot ring）
pending_ring: list[k_I]   # 长度 0..3；满 4 就 flush 成一个 k̄
```

伪代码：

```python
def update_compressed_index_cache(k_I_new, forward_batch):
    """
    k_I_new: 本步新产生的 raw indexer key（prefill 可一次多 token；decode 通常 1）
    """
    for k in k_I_new:                          # 按时间序追加
        pending_ring.append(k)                 # 未满块 ring，最多 4
        if len(pending_ring) == 4:
            b = next_block_id(forward_batch)   # 下一个完整 block 编号
            k_bar = compress_block(pending_ring, rope_pos=b * 4)
            write_compressed_cache(b, k_bar)   # → compressed_index_cache
            pending_ring.clear()
    # pending_ring 里剩下的 0..3 个 raw key 不写压缩表，留给 unfinished
```

直觉：


| 存什么                  | 何时写           | 用途                           |
| -------------------- | ------------- | ---------------------------- |
| **压缩 key** `k̄`      | 每满 4 token 一次 | indexer 打分扫这些                |
| **ring 里 raw** `k_I` | 当前尾巴未满 4      | 不参与打分；对应 token **必选**进稀疏 GQA |


相对「整段存 raw index key」，只长期存 `L/4` 个压缩 key + 每请求 4 slot → 博客称 index-cache 开销约降 **80%**。

### `all_compressed_keys` 是什么

打分时读的对象，不是 ring：

```python
all_compressed_keys = compressed_index_cache[0 : num_complete_blocks]
# num_complete_blocks = floor(prefix_len / 4)
# 形状概念上 [B, 128]，B ≈ L/4
#
# 因果：query 在位置 t 只能看到「已完全落入过去」的块
# （块尾 ≤ t；实现里用 block-causal mask）
```

所以 `indexer_score(q_I, all_compressed_keys)` 扫的是 **~L/4 个小向量**，不是全长 L 个 raw key。

Decode 一步：通常往 ring 塞 1 个 `k_I`；每 4 步才多出一个 `k̄` 进 `all_compressed_keys`。Prefill：整段连续 flush，多数块立刻变成压缩 key。

---



## 5.2 `expand_blocks` / `unfinished_block_tokens`



### `expand_blocks(top_blocks, ratio=4)`

Indexer 选出的是 **block id**，稀疏 GQA 要的是 **原始 token 位置**（去读主路径的 K/V，不是压缩 key）：

```python
def expand_blocks(block_ids, ratio=4):
    # block b → 逻辑位置 {b*4, b*4+1, b*4+2, b*4+3}
    positions = []
    for b in block_ids:
        positions.extend(range(b * ratio, (b + 1) * ratio))
    return positions[:budget]   # 截断到 K=2048（训练/实现可再 truncate）
```

例：`top_blocks = {0, 7}` → 位置 `{0,1,2,3, 28,29,30,31}`。

### `unfinished_block_tokens` 是什么

序列长度 `L` 往往 **不是 4 的倍数**。最后不够一整块的尾巴：

```text
L = 14 → 完整块 b=0,1,2（token 0..11）已压缩并可被 top-k
         未满块 token 12,13（2 个）→ unfinished_block_tokens
```

```python
def unfinished_block_tokens(prefix_len, ratio=4):
    rem = prefix_len % ratio          # 0..3
    if rem == 0:
        return []                     # 刚好整除：没有「未完成块」
    start = prefix_len - rem          # 当前尾巴起点
    return list(range(start, prefix_len))
```

这些位置：

- **不进** `scores` / top-k（块还没压完，没有合法 `k̄`）
- **始终并入** `sel`（论文：*tokens in the final incomplete block are always included*）
- 对应 ring 里那几个 raw index key 的 token；attention 仍读它们的 **原始** K/V

因此：

```text
|sel| ≤ 512×4 + (0..3) = 2048 + 0..3 ≤ 2051
```

若 `L % 4 == 0`，未满块为空，上限就是 2048。

---



## 5.3 串起来的完整 Indexer 伪代码

```python
def qsa_indexer_select(h, positions, forward_batch, r=4, budget=2048):
    q_I, k_I = project_indexer(h)                 # Q: 4×128, K: 1×128
    # k_I 可先不 RoPE；压缩时对 k̄ 用块首位置 RoPE（见 compress_block）

    update_compressed_index_cache(k_I, forward_batch)
    # 副作用：满块 → compressed_index_cache；尾巴 → pending_ring

    keys = all_compressed_keys(forward_batch)     # [B, 128], B=floor(L/r)
    scores = indexer_score(q_I, keys)             # [T, B]；block-causal
    # s[t,b] = 1/sqrt(128) * sum_h ReLU(<q_I[t,h], keys[b]>)

    n_blocks = budget // r                        # 512
    top_b = topk(scores, k=min(n_blocks, B))      # 每 query 一套（实现可批）

    sel = expand_blocks(top_b, ratio=r)           # block → token ids
    sel = unique_sorted(sel + unfinished_block_tokens(L, r))
    return sel                                    # gather 原始 K/V 用
```

和主路径关系：

```text
Indexer 路径：q_I / k_I → 压缩 cache → 选出 token 下标 sel
主 GQA 路径：q / k / v → 原始 KV cache
稀疏 GQA：   用 sel gather 原始 k,v，对 q 做 softmax（压缩 key 不参与 softmax）
```

---



## 6. 端到端伪代码

```python
def qsa_forward(h, positions, forward_batch):
    # h: [T, H=2560]

    # ---- 主路径投影（可与 indexer 并行流）----
    q, k, v, gate = project_gqa_and_gate(h)     # GQA: 24Q / 2KV / D=256
    q, k = partial_rope(q, k, positions)        # 64 维 RoPE

    # 原始 K/V 写入普通 paged KV pool（与 dense GQA 相同）
    write_kv_cache(k, v, forward_batch.out_cache_loc)

    # ---- Indexer（细节见 §5.1–5.3）----
    sel = qsa_indexer_select(h, positions, forward_batch,
                             r=4, budget=2048)  # → ≤2051 token ids

    # ---- 稀疏 GQA：只读 sel 上的原始 K/V ----
    k_s, v_s = gather_kv(sel, forward_batch)
    ctx = sparse_gqa(q, k_s, v_s)               # softmax over |sel|
    if gate is not None:
        ctx = ctx * sigmoid(gate)               # output_gate_type=sigmoid
    return o_proj(ctx)                          # [T, H]
```

---



## 7. Cache 布局（推理侧）

```text
每层 QSA：
  ├─ 原始 K/V          — 标准 paged pool（Radix Cache 可复用）
  ├─ 压缩 index key    — 每 4 token 一个 BF16；page 对齐 full_slot/4
  └─ 未满 block ring   — 每请求 4 slot，暂存尚未压缩的 raw index key
```

相对「整段存 raw index key」，压缩 + ring 可把 index-cache 开销压低约 80%（博客数字）。

---



## 8. Prefill / Decode / MTP


| 阶段        | 行为                                                                                                                      |
| --------- | ----------------------------------------------------------------------------------------------------------------------- |
| Prefill   | GPU 打分 → 快 top-k → 展开 index → Triton/稀疏 GQA                                                                             |
| Decode    | paged indexer；compact 选出的 K/V；再 FA / TRTLLM-Gen                                                                         |
| MTP draft | **IndexShare**：target 接受后的 draft-extend 跑一次 indexer，整轮 draft decode **复用**该 selection（再补上 in-flight draft 位），避免每步重扫 L/4 |


---



## 9. 和 DSA / 普通 GQA 对比


|                | 全量 GQA | GLM DSA 等      | **QSA**               |
| -------------- | ------ | -------------- | --------------------- |
| 选位粒度           | 全序列    | 常 token/latent | **micro-block（c4）**   |
| 打分用什么          | —      | 轻量 index       | 压缩 index key          |
| Attention 读什么  | 全部 K/V | selected 历史    | selected 的 **原始** K/V |
| Flash-Next 中位置 | —      | —              | 每 4 层 1 个槽（共 12）      |


---



## 10. 一句话

QSA：**c4 压缩 indexer 粗选 block → 在原始 KV 上精算稀疏 GQA**；长上下文省的是扫描量与 attention 算力，不是丢掉主 KV cache。