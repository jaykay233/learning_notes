# PLE / N-gram Embedding 详解

> 针对 **Qwen3.8-Flash-Next**。总览见同目录 [architecture.md](./architecture.md)。
>
> PLE 英文全称：**Per-Layer Embedding**（复数常写 Per-Layer Embeddings）。
> 实现上是 **hash 寻址的 N-gram embedding memory**，插在指定 decoder 层（官方 `ple_layer_ids=[2]`），不是「每一层各一张表」——名字里的 Layer 指挂在某一层上的 per-layer 模块。
>
> 依据：HF `text_config`（`ngram_*` / `ple_*`）+ [LMSYS Day-0 博客](https://www.lmsys.org/blog/2026-08-26-qwen-flash-next/)。

---

## 1. 一句话

PLE 用「当前 token + 前面几个 token」做 **N-gram hash 查表**，给模型加一块很大的 **局部短语记忆**；参数量极大（约 **51B**），但每 token 只取十几行，算力几乎可忽略。表常放 **pinned host**，异步 gather 进 GPU。

---

## 2. 它解决什么问题 / 为什么要 `ngram_lookup`

| 手段 | 作用 |
|---|---|
| 加深/加宽主干 | 涨算力与激活显存 |
| MoE | 涨专家容量，仍要算路由+FFN |
| **N-gram / PLE** | 把容量放在 **查表** 上：常见局部模式直接召回向量，几乎不涨 per-token FLOPs |

Flash-Next：主干约 125B（激活 ~6B）+ **额外 ~51B N-gram 表**（可不常驻 GPU）。

官方动机（Qwen / LMSYS）：在 Transformer 主干之外用 N-gram embedding **扩容量**（思路接近 Gemma PLE / DeepSeek Engram）——搭配、固定说法、局部句法不必每次用矩阵「算」出来，按局部上下文召回即可。

设计取舍：

| 设计点 | 原因 |
|---|---|
| 用 **2/3-gram** 而非单 token | 输入 embedding 已覆盖「这个词是谁」；N-gram 管「这几个词连在一起」 |
| **hash 查表** 而非真词表 | 全部 bigram/trigram 爆炸；hash 进固定大表（基数 ~2e7） |
| **多 head（8+8）** | 缓解碰撞：同一 n-gram 多路寻址，拼成 2560 维 |
| **只插很浅一层**（layer 2） | 早期注入局部先验，后面层接着用；整网只付一次 gather |
| **再 gate 进 HC** | 查到的不一定有用；与当前残差对齐才写入 |

一句话：**参数堆在可离线、可 offload 的查表上，算力仍花在稀疏激活的主干上。**

---

## 3. 表是学出来的吗

**是。** N-gram table 是预训练一起学的 **独立参数**，不是手工规则，也不是从 token embedding 拷贝。

| | 含义 |
|---|---|
| **训练** | 每行是可学习向量；hash 命中哪些行，反传更新这些行 |
| **与 token emb** | **另一套参数**；键是 n-gram hash，不是单 token id |
| **推理** | **冻结权重**，只 gather；可放 host，不是 KV cache，也不是在线动态记忆 |

约 51.2B 参数（约 320M 行 × 160 维，16 head packed）；推理侧看到的「固定表」= 训好后的权重。

---

## 4. 官方相关配置

| 字段 | 值 | 含义 |
|---|---|---|
| `ngram_size` | 3 | 最高用到 3-gram |
| `ngram_vocab_size_base` | 2e7 | 表规模基数 |
| `heads_per_ngram` | 8 | 每种 n-gram 的 hash head 数 |
| `ple_layer_ids` | `[2]` | 插入位置（配置层号；约第 2 个 decoder block） |
| `ple_embed_dim` | 2560 | 与 `hidden_size` 对齐 |
| `ple_conv_kernel_size` | 4 | PLE 短 depthwise conv |
| `split_ngram_parts` | 128 | 表切分 |
| `make_ngram_vocab_size_divisible_by` | 128 | 对齐 |
| `hc_count` | 4 | 注入目标：Gated Residual 四支路 |

整网 **只在少数层**（官方就一层）插 PLE，不是每层一张表。

---

## 5. 查表：每 token 取什么

对 token `x_t`：

```text
8 个 2-gram head：hash(x_t, x_{t-1})           → 8 个 row id
8 个 3-gram head：hash(x_t, x_{t-1}, x_{t-2})   → 8 个 row id
合计 16 行；每行 160 维 → 拼成 E_t ∈ R^{2560}
```

### `bigram_hashes` / `trigram_hashes` 怎么算

不是字符串 hash，而是对 **原始 tokenizer id** 做定点数混合（与 NeMo / SGLang / checkpoint 对齐）：

1. **取上下文**（遇 EOS 重置段；缺历史则填 `eos_token_id`）  
   `u0 = x_t`，`u1 = x_{t-1}`，`u2 = x_{t-2}`（跨 EOS 不算）
2. **线性混合 + XOR**（乘法用 **signed int64**，故意溢出）：

```text
# 固定乘数（checkpoint）
m0 = 23703573157769
m1 = 20109073645365
m2 = 8052911324071

# bigram（8 个 head）
mixed2 = (u0 * m0) XOR (u1 * m1)

# trigram（另 8 个 head）
mixed3 = (u0 * m0) XOR (u1 * m1) XOR (u2 * m2)
```

3. **每 head 一个素数模 + 全局 offset**（表是 16 段竖着拼成一张）：

```text
# 每 head 模数 ≈ 2e7 的不同素数，例如
# bigram heads:  20000003, 20000023, …, 20000077
# trigram heads: 20000081, 20000093, …, 20000171
# offset_h = 前面各 head 模数之和（packed table）

row_id[h] = (mixed % vocab_size[h]) + offset[h]   # 正余数
```

多 head + 不同素数：同一 n-gram 在某一 head 碰撞时，其它 head 几乎不一起撞。

```python
def ngram_lookup(input_ids, t):
    u0, u1, u2 = context_ids(input_ids, t)  # EOS 段内；缺则 eos_id
    mixed2 = (u0 * m0) ^ (u1 * m1)          # int64
    mixed3 = mixed2 ^ (u2 * m2)
    ids_2 = [mixed2 % p + off for p, off in bigram_heads]   # 8
    ids_3 = [mixed3 % p + off for p, off in trigram_heads]  # 8
    rows = gather(ngram_table, cat(ids_2, ids_3))           # 16 × 160
    return concat(rows)                                     # [2560]
```

---

## 6. `inject_ple` 伪代码

`inject_ple` / `ple_inject_into_hc` 是教学名：把查表向量 `E` **门控注入**当前 HC 状态（实现里常先 `h = h + ple(...)`，再进 HC Mix）。

### `gate(norm(Q), norm(K))` 是什么

博客写成 `Gate(Norm(Q), Norm(K)) → [4,1]`，**不是**另学一个大 MLP。对照开源推理实现（与 HF 对齐的 MLX port），展开是：

```python
# Q, K: [T, 4, H]  已按支路做 group-RMSNorm（每路各自归一化）
# 1) 支路内缩放点积（像单 head attention score）
s = (Q * K).sum(dim=-1, keepdim=True) / sqrt(H)   # [T, 4, 1]

# 2) 保号压幅：sign(s) * sqrt(|s|)
s = sign(s) * sqrt(max(|s|, 1e-6))

# 3) 压到 (0,1) 当写入强度
g = sigmoid(s)                                    # [T, 4, 1]
```

直觉：先看「当前 HC 支路 Q」和「N-gram 投出的 K」有多对齐，再决定这条支路吃多少 `V`。`Norm` = 按支路的 RMSNorm；`Gate` = **scaled dot → signed-sqrt → sigmoid**。

### 完整 `inject_ple`

```python
def inject_ple(R, E, hc_count=4):
    """
    R: [T, 4, H] 或等价 flatten [T, 4H]（实现里常 flatten 后 group-norm）
    E: [T, H]     ngram_lookup 结果
    return delta: [T, 4H] 再 reshape 加回残差 / HC
    """
    K = rms_norm_per_branch(W_k(E))           # [T, 4, H]
    V = W_v(E)                                # [T, H]
    Q = rms_norm_per_branch(R)                # [T, 4, H]  # 博客里的 Norm(Q)

    g = gate(Q, K)                            # 上节；[T, 4, 1]
    U = g * V[:, None, :]                     # [T, 4, H]
    U = U.reshape(T, hc_count * H)

    # 短卷积残差（ple_conv_kernel_size=4，可带 dilation）
    U = U + silu(dwconv(rms_norm(U)))
    return U                                  # 调用方：R_flat = R_flat + U
```

对应公式：

```text
g = σ( sign(s) · √|s| ),   s = ⟨Norm(Q), Norm(K)⟩ / √H
U = g ⊙ V
Δ = U + SiLU(DWConv(RMSNorm(U)))
R ← R + Δ
```

---

## 7. 层内位置伪代码

```python
def decoder_block_with_ple(R, input_ids, layer_idx, forward_batch):
    # R: [T, hc_count=4, H]

    if layer_idx in ple_layer_ids:          # 官方: 2
        E = ngram_lookup_batch(input_ids)  # [T, H]
        R = inject_ple(R, E)                   # 上节：门控写入 HC 四支路

    h = hc_mix(R)                          # [T, H]
    if is_gdn_layer(layer_idx):
        y = gated_delta_net(h, forward_batch)
    else:
        y = qsa_attention(h, ..., forward_batch)
    y = sparse_moe(y)
    R = hc_combine(R, y)
    return R
```

MTP：**target** 在 prefill/decode/verify 保留 PLE；**单层 draft** 常关掉 PLE 以降延迟。

---

## 8. 请求本地状态

除大表外，每请求还要：

| 状态 | 形状/内容 | 用途 |
|---|---|---|
| 最近 token id | 2 个 | 2/3-gram hash |
| 短卷积历史 | 约与 `ple_conv` 相关 | DWConv 跨步 |

大表本身是 **训好后的冻结权重**（见 §3），不是 KV cache。

---

## 9. Host offload（SGLang 实践）

```text
N-gram 表（~95 GiB BF16 量级）
  → 按 TP 切分放 pinned host
  → 每 token 算 16 个 row id
  → 专用 CUDA stream Triton UVA gather → 小块 BF16 GPU buffer
  → 与前一层计算重叠
```

效果（博客 H200 TP4 量级）：GPU 上 target 权重明显下降，同 memory fraction 下 KV 容量上升；吞吐基本不掉。Offload 改的是 **存放位置**，不改表归属与数学。

---

## 10. 和 embedding / MoE 对比

| | Token embedding | MoE expert | **PLE N-gram** |
|---|---|---|---|
| 键 | 单 token id | hidden 路由 | **局部 n-gram hash** |
| 每 token 触达 | 1 行 | top-k 专家整段 FFN | **16 行查表 + 小门控** |
| 是否学习 | 预训练学 | 预训练学 | **预训练学（独立表）** |
| 推理时 | 冻结 | 冻结 | **冻结；可 host offload** |
| 参数位置 | GPU 常驻 | GPU | **可 host** |
| 插入点 | 网络最前 | 几乎每层 | **极少数层（官方 1 层）** |

---

## 11. 一句话

PLE：**用巨大的、预训练学好的 N-gram 查表换局部模式容量**；算在「16 行 gather + 门控注入 HC」，表可以放 host，不和 KV 抢同一类显存。
