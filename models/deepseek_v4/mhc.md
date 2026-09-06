# DeepSeek-V4 mHC 详解（Manifold-Constrained Hyper-Connections）

> 基于 SGLang 实现梳理：
> - `python/sglang/srt/models/deepseek_v4.py`（`hc_pre` / `hc_post` / `hc_head`）
> - `python/sglang/kernels/ops/layernorm/mhc.py`（Sinkhorn / combine / fused kernels）
> - `python/sglang/kernels/ops/layernorm/mhc_head.py`

默认超参（Flash 量级常见）：`hc_mult = 4`，`hc_sinkhorn_iters = 20`，`hc_eps = 1e-6`，`mhc_post_mult = 2.0`。

---

## 1. 为什么需要 mHC？

标准 Transformer residual：

```
x ──► SubLayer ──► + ──► x'
 ▲_________________|
```

只有 **1 条** 残差高速公路。信息流、梯度路径都挤在同一条通道上。

**Hyper-Connections（HC）** 的想法：把 residual 扩成 **`hc_mult` 条并行流**：

```
R ∈ R^{T × hc × H}     # T=token, hc=流数, H=hidden
```

每个子模块（Attn / MoE）前，用一组可学习混合系数把多路合成 **一层输入**；子模块算完后，再用另一组系数写回多路 residual。

**mHC（manifold-constrained HC）** 进一步约束混合矩阵，使跨流混合落在更「好」的流形上——实现上通过 **Sinkhorn 迭代**，把 `comb` 矩阵推向近似 **双随机**（行/列和接近 1），训练更稳、推理数值更可控。

SGLang cookbook 一句话：DeepSeek-V4 使用 **manifold-constrained hyper-connections (mHC)**。

---

## 2. 一张图看清层内数据流

```
                    ┌─────────────────────────────────────────────┐
 residual R         │              DeepseekV4DecoderLayer         │
 [T, hc, H]         │                                             │
        │           │   hc_pre(attn)                              │
        │           │     mixes = RMSNormLinear(flatten(R))       │
        │           │     (pre, post, comb) = split_sinkhorn(...) │
        │           │     y = Σ_k pre[k] * R[k]     → [T, H]      │
        │           │     (+ input_layernorm，可融合)              │
        │           │              │                              │
        │           │              ▼                              │
        │           │         MQALayer / Attn                     │
        │           │              │ o_attn [T, H]                │
        │           │              ▼                              │
        │           │   hc_post(attn):                            │
        │           │     R ← post⊙o + comb·R                     │
        │           │              │                              │
        │           │   hc_pre(ffn) → y_ffn [T, H]                │
        │           │     (+ post_attn_layernorm)                 │
        │           │              ▼                              │
        │           │         DeepseekV2MoE                       │
        │           │              │ o_ffn [T, H]                 │
        │           │              ▼                              │
        │           │   hc_post(ffn) → R' [T, hc, H]              │
        └───────────┴─────────────────────────────────────────────┘
                                      │
                                      ▼  (最后一层)
                                 hc_head(R) → [T, H] → lm_head
```

注意：

- **子模块始终吃 `[T, H]`**（单路），多路只活在 residual / mHC 边界。
- Attn 与 FFN **各有一套** mixing 参数：`hc_attn_*` 与 `hc_ffn_*`。
- 实现里常把「上一层 ffn 的 `hc_post` + 本层 attn 的 `hc_pre`」融成一次 **cross-layer `mhc_post_pre`**。

---

## 3. 参数与张量形状

设 `hc = hc_mult`（默认 4），`H = hidden_size`。

### 3.1 每层 Attn / FFN 各一套

`make_hc_mixing_params(hc, H)` 返回：

| 参数 | Shape | 含义 |
|---|---|---|
| `hc_fn` | `((2+hc)*hc , hc*H)` = `(24, 4H)` | 从 flatten residual 预测 mixes |
| `hc_base` | `((2+hc)*hc,)` = `(24,)` | mixes 的 bias |
| `hc_scale` | `(3,)` | 分别缩放 pre / post / comb 三段 |

同一层还有第二套：`hc_ffn_fn / hc_ffn_base / hc_ffn_scale`。

`mixes` 的扁平布局（长度 `(2+hc)*hc`）：

```
[ pre_logits (hc) | post_logits (hc) | comb_logits (hc*hc) ]
   indices 0..hc-1    hc .. 2hc-1         2hc .. end
```

### 3.2 运行时中间量

| 符号 | Shape | 角色 |
|---|---|---|
| `R` / `residual` | `[T, hc, H]` | 多路残差 |
| `mixes` | `[T, (2+hc)*hc]` | 未约束的混合 logits |
| `pre` | `[T, hc]` | 把多路压成子模块输入的权重 |
| `post` | `[T, hc]` | 子模块输出写回各路的门控 |
| `comb` | `[T, hc, hc]` | 路与路之间的线性混合（Sinkhorn 后） |
| `y` | `[T, H]` | 子模块输入 |
| `o` | `[T, H]` | 子模块输出 |

### 3.3 网络末端 `hc_head`

| 参数 | Shape |
|---|---|
| `hc_head_fn` | `(hc, hc*H)` |
| `hc_head_base` | `(hc,)` |
| `hc_scale` | `(1,)` |

把最终 `[T, hc, H]` 收成 `[T, H]`。

---

## 4. 数学形式

> 下面公式用**纯文本代码块**书写，GitHub / Cursor 预览都能直接看，不依赖 LaTeX 渲染。

### 4.1 `hc_pre`：多路 → 单路输入

对 token 维省略，记残差 `R ∈ R^{hc × H}`，以及 `flatten(R) ∈ R^{hc·H}`。

**Step A — RMS + 线性得到 mixes**

```text
r = rsqrt( mean(R_flat²) + ε_rms )

m = W_fn · R_flat · r
  ∈ R^{(2+hc)·hc}
```

（实现里是 `F.linear(x_flat, hc_fn) * rsqrt`，等价于对 flatten 后做 RMS 再投影。）

**Step B — 切成 pre / post / comb，并约束**

```text
pre_j  = σ(m_j · s0 + b_j) + ε_hc

post_j = 2 · σ(m_{hc+j} · s1 + b_{hc+j})

C̃_jk  = m_{2hc + j·hc + k} · s2 + b_{2hc + j·hc + k}
```

然后对 `C̃` 做 **Sinkhorn**（见 §5）得 `C = comb`。

**Step C — 合成子模块输入**

```text
y = Σ_{k=0}^{hc-1}  pre_k · R_k
  ∈ R^{H}
```

代码里叫 `hc_combine`：`y[h] = Σ_k pre[k] * R[k, h]`。

之后通常再接 `RMSNorm`（input / post-attn），可与 `hc_pre` 融进同一 kernel。

### 4.2 子模块

```text
o = Attn(y)    或    o = MoE(y)
```

### 4.3 `hc_post`：单路输出 → 写回多路

```text
R'_j = post_j · o + Σ_{k=0}^{hc-1} C_jk · R_k
```

向量形式：

```text
R' = (post ⊙ o) + C R
```

其中 `post ⊙ o` 表示把子模块输出按 `post` 门控广播到各路。

Torch 参考实现（与 kernel 一致）：

```python
# post: [T, hc], o: [T, H], comb: [T, hc, hc], R: [T, hc, H]
R_new = post.unsqueeze(-1) * o.unsqueeze(1) + (comb.unsqueeze(-1) * R.unsqueeze(2)).sum(dim=1)
#           [T,hc,1]*[T,1,H]                  sum over k of [T,hc,hc,1]*[T,1,hc,H]
```

直觉：

- `post`：本层输出 **灌进每一路** 的强度（带 `×2·sigmoid`，幅度可 >1）
- `comb`：旧残差在各路之间的 **线性重组**（Sinkhorn 后更接近置换/双随机混合）

### 4.4 `hc_head`：多路 → 最终 hidden

```text
m = W_head · R_flat · r_rms

α_j = σ(m_j · s + b_j) + ε

z = Σ_j α_j · R_j
```

---

## 5. Sinkhorn：把 `comb` 约束到流形上

这是 **m** 的来源（manifold-constrained）。

对每个 token 的 `comb` 矩阵 `C ∈ R^{hc × hc}`：

1. **初始化**：对行做稳定 softmax，再加 `eps`，再按列归一化
2. **迭代 `sinkhorn_iters-1` 次**（默认总共约 20 轮量级）：交替
   - 行归一：`C_jk ← C_jk / (Σ_{k'} C_jk' + ε)`
   - 列归一：`C_jk ← C_jk / (Σ_{j'} C_j'k + ε)`

效果：`C` 接近 **双随机矩阵**（行和、列和都 ≈ 1）。  
跨流混合因此是一种「质量守恒」式的重分配，而不是任意爆炸的线性层——这就是 manifold 约束的工程落地。

对比：Hunyuan 的 iHC 等变体「有门控但无 combination + Sinkhorn」，与 DSV4 mHC **不可互换**（SGLang docs 也强调这一点）。

---

## 6. 伪代码（教学版，对齐 SGLang 数值）

### 6.1 工具函数

```python
# hc = hc_mult, 默认 4
# POST_MULT = 2.0   # _MHC_POST_MULT_VALUE

def rms_rsqrt(x_flat, eps):
    # x_flat: [T, hc*H]
    return rsqrt(mean(x_flat ** 2, dim=-1, keepdim=True) + eps)


def split_sinkhorn(mixes, scale, base, hc, iters, eps):
    """
    mixes: [T, (2+hc)*hc]
    scale: [3]
    base : [(2+hc)*hc]
    returns pre[T,hc], post[T,hc], comb[T,hc,hc]
    """
    pre  = sigmoid(mixes[:, :hc]            * scale[0] + base[:hc]) + eps
    post = POST_MULT * sigmoid(mixes[:, hc:2*hc] * scale[1] + base[hc:2*hc])

    comb = mixes[:, 2*hc:] * scale[2] + base[2*hc:]
    comb = comb.reshape(T, hc, hc)

    # init: row softmax + eps, then col normalize
    comb = exp(comb - comb.amax(dim=-1, keepdim=True))
    comb = comb / comb.sum(dim=-1, keepdim=True) + eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)

    for _ in range(iters - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)  # row
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)  # col

    return pre, post, comb


def hc_combine(R, pre):
    # R: [T, hc, H], pre: [T, hc] → y: [T, H]
    return (pre.unsqueeze(-1) * R).sum(dim=1)


def hc_post(o, R, post, comb):
    # o: [T,H], R:[T,hc,H], post:[T,hc], comb:[T,hc,hc] → R_new:[T,hc,H]
    return post.unsqueeze(-1) * o.unsqueeze(1) + einsum("tjk,tkh->tjh", comb, R)
```

### 6.2 `hc_pre`

```python
def hc_pre(R, hc_fn, hc_scale, hc_base, rms_eps, hc_eps, sinkhorn_iters, hc):
    """
    R: [T, hc, H]  (第一层前可能从 [T,H] unsqueeze/repeat 而来)
    """
    T, hc, H = R.shape
    x_flat = R.reshape(T, hc * H).float()

    r = rms_rsqrt(x_flat, rms_eps)
    mixes = linear(x_flat, hc_fn) * r          # [T, (2+hc)*hc]

    pre, post, comb = split_sinkhorn(
        mixes, hc_scale, hc_base, hc, sinkhorn_iters, hc_eps
    )

    y = hc_combine(R, pre).to(R.dtype)         # [T, H]
    return y, post, comb
    # 调用方再可选: y = RMSNorm(y)
```

### 6.3 `hc_head`

```python
def hc_head(R, hc_fn, hc_scale, hc_base, rms_eps, hc_eps):
    # R: [T, hc, H] → z: [T, H]
    T, hc, H = R.shape
    x_flat = R.reshape(T, hc * H).float()
    r = rms_rsqrt(x_flat, rms_eps)
    mixes = linear(x_flat, hc_fn) * r          # [T, hc]
    alpha = sigmoid(mixes * hc_scale + hc_base) + hc_eps
    return (alpha.unsqueeze(-1) * R).sum(dim=1).to(R.dtype)
```

### 6.4 完整 Decoder Layer（非融合版）

```python
def decoder_layer(R, positions, batch, layer):
    """
    R: [T, hc, H]
    返回 R_out: [T, hc, H]
    """

    # ---- Attention branch ----
    y, post_a, comb_a = hc_pre(
        R,
        layer.hc_attn_fn, layer.hc_attn_scale, layer.hc_attn_base,
        rms_eps, hc_eps, sinkhorn_iters, hc,
    )
    y = RMSNorm(y, layer.input_layernorm)      # 可与 hc_pre 融合
    o = layer.self_attn(y, positions, batch)   # [T, H]
    R = hc_post(o, R, post_a, comb_a)          # [T, hc, H]

    # ---- FFN / MoE branch ----
    y, post_f, comb_f = hc_pre(
        R,
        layer.hc_ffn_fn, layer.hc_ffn_scale, layer.hc_ffn_base,
        rms_eps, hc_eps, sinkhorn_iters, hc,
    )
    y = RMSNorm(y, layer.post_attention_layernorm)
    o = layer.mlp(y, batch)                    # DeepseekV2MoE
    R = hc_post(o, R, post_f, comb_f)

    return R
```

### 6.5 整网骨架

```python
def deepseek_v4_forward(input_ids):
    h = embed(input_ids)                       # [T, H]

    # 进入 mHC：复制到 hc 路（具体初始化以权重/实现为准；
    # 之后每层维护 [T, hc, H]）
    R = expand_to_hc_streams(h, hc)            # [T, hc, H]

    for layer in layers:
        R = decoder_layer(R, ...)

    z = hc_head(R, hc_head_fn, hc_head_scale, hc_head_base, ...)
    z = RMSNorm(z)
    logits = lm_head(z)
    return logits
```

> 实现细节：SGLang 在 fused 路径下，层与层之间可能 **延迟** 执行 `hc_post(ffn)`，把 `(residual, post, comb)` 传给下一层，与下一层 `hc_pre(attn)` 合成 `mhc_post_pre` 一次算完。语义仍等价于上面的非融合伪代码。

---

## 7. 和「普通 residual / 朴素 HC」对比

| | 标准 Residual | 朴素 HC（无约束） | **mHC（V4）** |
|---|---|---|---|
| 残差维度 | `[T,H]` | `[T,hc,H]` | `[T,hc,H]` |
| 进子模块前 | identity / 选一路 | 任意线性混合 | `pre` 门控加权和 |
| 出子模块后 | `x+o` | 任意写回 | `post⊙o + comb·R` |
| 混合矩阵 | — | 无结构 | **Sinkhorn → 近双随机** |
| 稳定性 | 高 | 多路易飘 | 约束后更稳 |
| 子模块接口 | `[T,H]` | `[T,H]` | 仍是 `[T,H]`（Attn/MoE 不用改内部） |

---

## 8. SGLang 实现要点（读代码时）

1. **参考数值路径**就看：
   - `DeepseekV4DecoderLayer.hc_pre` / `hc_post` 里的 torch fallback
   - `kernels/.../mhc.py` 的 `_hc_split_sinkhorn_torch`
2. **热路径**用 TileLang / FlashInfer / HIP aiter / NPU 融合 kernel；还有 DeepGEMM 做大 batch 的 `hc_prenorm` GEMM。
3. **`post` 乘 2**：kernel 与 torch 都是 `2 * sigmoid(...)`，对应 `_MHC_POST_MULT_VALUE = 2.0`。
4. **跨层融合**：`use_fused_mhc_post_pre` 时，layer `forward` 返回 `(hidden, residual, post, comb)`，下一层入口消费。
5. **Speculative（DSpark / NextN）**：draft 侧也要维护 `hc_mult` 维；`resolve_spec_hidden_size` 会把边界 hidden 扩成 `hidden * hc_mult`。

---

## 9. 最小数值例子（`hc=2` 示意）

假设某 token 上：

```
R = [[1, 0],      # stream 0, H=2
     [0, 1]]      # stream 1

pre  = [0.7, 0.3]
post = [1.2, 0.8]
comb ≈ [[0.9, 0.1],
        [0.2, 0.8]]   # 已近双随机
```

则：

```
y = 0.7*[1,0] + 0.3*[0,1] = [0.7, 0.3]

假设 Attn(y) → o = [0.5, 0.5]

R'_0 = 1.2*o + 0.9*R0 + 0.1*R1 = 1.2*[0.5,0.5] + 0.9*[1,0] + 0.1*[0,1]
     = [0.6,0.6] + [0.9,0] + [0,0.1] = [1.5, 0.7]

R'_1 = 0.8*o + 0.2*R0 + 0.8*R1 = ...
```

可以看出：每一路既吃到了本层输出，也按 `comb` 重新分配了旧信息——这就是 hyper-connection；`comb` 被 Sinkhorn 管住，就是 manifold constraint。

---

## 10. 阅读顺序建议

1. 本文 §4–§6（先建立公式与伪代码）
2. `deepseek_v4.py` → `DeepseekV4DecoderLayer.forward`（非 fused 分支）
3. `mhc.py` → `_hc_split_sinkhorn_torch` + `hc_combine`
4. `mhc_head.py` → `fused_hc_head` 注释里的 shape
5. 再回头看 fused `mhc_post_pre`（优化，不改语义）

---

*与 `architecture.md` 配套：那份讲整网结构，本份只深入 mHC。*
