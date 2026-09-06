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

## 5. 源码函数速查

这一节按源码里的函数名解释作用。可以先看这里建立索引，再去读后面的数学和伪代码。

### 5.1 参数与后端选择

| 函数 / 类 | 位置 | 作用 |
|---|---|---|
| `MhcOps` | `srt/models/deepseek_v4.py` | 把当前平台可用的 mHC kernel 打包成一个 NamedTuple，后面 Decoder 只通过这个对象调用。 |
| `_get_mhc_ops()` | `srt/models/deepseek_v4.py` | 根据 CUDA/NPU/可用扩展选择具体实现：TileLang / Triton / FlashInfer / NPU / torch fallback。 |
| `make_hc_mixing_params()` | `srt/models/deepseek_v4.py` | 为每层 attn 或 ffn 创建一套 mHC mixing 参数：`hc_fn`、`hc_base`、`hc_scale`。 |
| `make_hc_head_params()` | `srt/models/deepseek_v4.py` | 为模型末端 `hc_head` 创建收束多路 residual 的参数。 |
| `_is_fused_mhc_post_pre_enabled_xpu()` | `srt/models/deepseek_v4.py` | 判断 XPU 后端是否启用跨层 `hc_post + hc_pre` 融合。 |
| `_flashinfer_mhc_pre_num_splits()` | `srt/models/deepseek_v4.py` | 给 FlashInfer 的 fused pre kernel 选择 split-K 数量，用于适配 token 数和 `hc*H` 大小。 |
| `_flashinfer_hc_pre()` | `srt/models/deepseek_v4.py` | FlashInfer 路径的 `hc_pre` 包装：做 pre 的大融合计算，并返回子模块输入、`post`、`comb`。 |

### 5.2 核心语义函数

| 函数 | 位置 | 输入 → 输出 | 作用 |
|---|---|---|---|
| `hc_expand(x, n)` | `kernels/ops/layernorm/mhc.py` | `[T,H] → [T,n,H]` | 模型第一层前，把普通 hidden 复制/扩成 `hc_mult` 路 residual。 |
| `hc_contract(x, n)` | `kernels/ops/layernorm/mhc.py` | `[T,n,H] → [T,H]` | 把多路 residual 临时压回一路，主要用于兼容路径或调试。 |
| `hc_split_sinkhorn()` | `kernels/ops/layernorm/mhc.py` | `mixes → pre, post, comb` | 把线性层产生的 logits 切成 `pre/post/comb`，并对 `comb` 做 Sinkhorn 约束。 |
| `hc_combine()` | `kernels/ops/layernorm/mhc.py` | `pre + residual → y` | 执行 `y = Σ pre[k] * R[k]`，也就是 `hc_pre` 的“多路读成一路”最后一步。 |
| `hc_pre()` | `kernels/ops/layernorm/mhc.py` | `R + hc_fn/base/scale → y, comb, post` | 非 fused 的 pre 语义入口：从多路 residual 预测 mixing，并产出子模块输入。 |
| `hc_post()` | `kernels/ops/layernorm/mhc.py` | `o + R + post + comb → R'` | 非 fused 的 post 语义入口：把 Attention/MoE 输出写回多路 residual。 |
| `hc_head_torch()` | `srt/models/deepseek_v4.py` | `R + head params → z` | torch fallback 版本的末端收束，把最后的多路 residual 合成 `[T,H]`。 |
| `fused_hc_head()` | `kernels/ops/layernorm/mhc_head.py` | `R + head params → z` | Triton fused 版本的 `hc_head`，语义与 `hc_head_torch()` 一致但更快。 |

### 5.3 `hc_split_sinkhorn` 相关实现

| 函数 | 位置 | 作用 |
|---|---|---|
| `hc_split_sinkhorn_kernel()` | `kernels/ops/layernorm/mhc.py` | TileLang kernel 版本：在 GPU 上完成 split、sigmoid、Sinkhorn。 |
| `_hc_split_sinkhorn_torch()` | `kernels/ops/layernorm/mhc.py` | PyTorch 参考实现，最容易对照数学公式阅读。 |
| `_hc_split_sinkhorn_triton_kernel()` | `kernels/ops/layernorm/mhc.py` | Triton kernel 版本，处理小维度 `hc×hc` 的 Sinkhorn。 |
| `_hc_split_sinkhorn_triton()` | `kernels/ops/layernorm/mhc.py` | Triton kernel 的 Python 包装，负责分配输出和调 kernel。 |
| `hc_split_sinkhorn()` | `kernels/ops/layernorm/mhc.py` | 对外统一入口：优先走可用高性能 kernel，不合适时回退。 |

#### 常见误解：`hc_split_sinkhorn` 不是注意力系数

笔记伪代码里的 `split_sinkhorn` = 源码 `hc_split_sinkhorn`。它产出的确实是一组**归一化混合权重**，所以容易联想到「分配注意力」；但**不宜直接理解成 Transformer Attention**。

**像的地方**

- `pre`：各条残差流合成子模块输入时的比重 → 有点像对 `hc` 路做 soft 加权 / pooling
- `comb`：流与流之间的混合矩阵 → 有点像「路与路」的转移权重

**不像的地方**

| | Attention | `hc_split_sinkhorn` |
|---|---|---|
| 作用对象 | token / 位置之间（Q·K） | **残差流（stream）** 之间 |
| 输入 | Q、K 相似度 | `mixes`（由 `flatten(R)` 线性投影得到） |
| 输出 | 通常一套权重去乘 V | 同时产出 **`pre` / `post` / `comb` 三套**，用途不同 |
| `post` | 没有对应物 | 写回强度门控，且是 `2σ`，**不是**对旧残差的注意力 |
| `comb` | 一般一次 softmax | Sinkhorn → **近双随机**（行和、列和都 ≈ 1） |

更准确的说法：

> `hc_split_sinkhorn` = **残差流上的混合系数生成器**：把 `mixes` 切成三份，并对 `comb` 做流形投影。  
> - `pre`：进子模块的聚合权重  
> - `post`：出子模块的注入门控  
> - `comb`：旧残差的流间重分配矩阵  

一句话：它是 **mHC 边界上的系数拆分 + Sinkhorn**，不是 Transformer 里那种注意力。

调用关系（非融合 fallback）：只在 `DeepseekV4DecoderLayer.hc_pre` 里，先算 `mixes`，再调 `hc_split_sinkhorn`，最后 `hc_combine`；TileLang / FlashInfer / HIP 等 fused `mhc_pre` 则把同等逻辑内联进 kernel，不一定单独调这个函数。

### 5.4 `hc_pre` / prenorm 融合实现

| 函数 | 位置 | 作用 |
|---|---|---|
| `mhc_pre_big_fuse_tilelang()` | `kernels/ops/layernorm/mhc.py` | 大融合 pre kernel：把 RMS、GEMM、split/Sinkhorn、combine、可选 norm 尽量合在一起。 |
| `mhc_pre_gemm_sqrsum_tilelang()` | `kernels/ops/layernorm/mhc.py` | 同时算 `hc_fn @ R_flat` 和 `sum(R_flat²)`，为 pre 的 RMS + 线性准备中间量。 |
| `_mhc_pre_gemm_sqrsum_dispatch()` | `kernels/ops/layernorm/mhc.py` | 选择并缓存上面的 TileLang dispatch。 |
| `mhc_pre_gemm_sqrsum_splitk_kernel()` | `kernels/ops/layernorm/mhc.py` | split-K 版本的 prenorm GEMM，适合更大的 `hc*H` 或 token 数。 |
| `_compute_num_split_for_mhc_pre()` | `kernels/ops/layernorm/mhc.py` | 根据 token 数和 hidden 大小估算 split-K 分片数。 |
| `get_mhc_pre_token_count_representatives()` | `kernels/ops/layernorm/mhc.py` | 给预热/编译缓存准备代表性的 token 数。 |
| `prewarm_mhc_pre()` | `kernels/ops/layernorm/mhc.py` | 预热 mHC pre kernels，减少首次请求编译开销。 |
| `mhc_pre_big_fuse_with_norm_tilelang()` | `kernels/ops/layernorm/mhc.py` | 在 big fuse 基础上把 RMSNorm 也融合进去，用于 `hc_pre + input_layernorm/post_attention_layernorm`。 |
| `mhc_pre()` | `kernels/ops/layernorm/mhc.py` | fused pre 的统一入口，产出 norm 后的子模块输入、`post`、`comb` 等。 |
| `npu_hc_pre()` | `kernels/ops/layernorm/mhc.py` | NPU 自定义算子版本的 `hc_pre`，语义相同，但调用 `torch.ops.custom.npu_hc_pre`。 |

### 5.5 `hc_post` 与跨层融合

| 函数 | 位置 | 作用 |
|---|---|---|
| `mhc_post_tilelang()` | `kernels/ops/layernorm/mhc.py` | TileLang 版本的 post kernel，实现 `R' = post⊙o + comb·R`。 |
| `mhc_post()` | `kernels/ops/layernorm/mhc.py` | fused post 的统一入口，用于可直接写回多路 residual 的路径。 |
| `mhc_fused_post_pre_fma_tilelang()` | `kernels/ops/layernorm/mhc.py` | 跨层融合核心 kernel：把上一子层 `hc_post` 和下一子层 `hc_pre` 合并，避免把中间 `R'` 完整写回再读出。 |
| `mhc_fused_post_pre()` | `kernels/ops/layernorm/mhc.py` | 跨层融合 Python 入口：输入上一段的 `post/comb/o/R` 和下一段的 `hc_fn`，直接得到下一子模块输入。 |
| `_mhc_post_torch()` | `kernels/ops/layernorm/mhc.py` | PyTorch 参考 post，实现最接近公式，便于验证 kernel 数值。 |
| `_mhc_post_dispatch()` | `kernels/ops/layernorm/mhc.py` | 根据设备、dtype、shape 选择 post 实现。 |

### 5.6 fallback / dispatch 函数

| 函数 | 位置 | 作用 |
|---|---|---|
| `_mhc_pre_torch()` | `kernels/ops/layernorm/mhc.py` | PyTorch 参考 pre：线性预测 mixing，split/Sinkhorn，再 combine。 |
| `_mhc_pre_dispatch()` | `kernels/ops/layernorm/mhc.py` | 为 `hc_pre()` 选择 TileLang / torch 等实现。 |
| `_mhc_post_dispatch()` | `kernels/ops/layernorm/mhc.py` | 为 `hc_post()` 选择 TileLang / torch 等实现。 |
| `_hc_combine_kernel()` | `kernels/ops/layernorm/mhc.py` | Triton kernel，专门执行 `pre` 加权求和。 |
| `_hc_head_kernel()` | `kernels/ops/layernorm/mhc_head.py` | Triton kernel，专门执行末端 `hc_head` 的 RMS、线性、sigmoid 和 combine。 |

### 5.7 Decoder 里怎么用这些函数

| 位置 | 作用 |
|---|---|
| `DeepseekV4DecoderLayer` | 持有 attn/ffn 两套 mHC 参数，并在每个子层前后调用 `hc_pre` / `hc_post` 或融合版本。 |
| `DeepseekV4Model` | 在第一层前调用 `hc_expand` 建立多路 residual，在最后一层后调用 `hc_head` 收束。 |
| `DeepseekV4ForCausalLM` | 接上最终 `lm_head`，mHC 本身已经在 `DeepseekV4Model` 内部完成。 |

---

## 6. Sinkhorn：把 `comb` 约束到流形上

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

## 7. 伪代码（教学版，对齐 SGLang 数值）

约定：`T` = token 数，`hc = hc_mult`（默认 4），`H = hidden_size`。  
Flash 量级常见：`hc=4` 时 `(2+hc)*hc = 24`，`hc*H = 4H`。

### 7.1 工具函数

```python
# POST_MULT = 2.0   # _MHC_POST_MULT_VALUE

def rms_rsqrt(x_flat, eps):
    """
    作用：计算 RMSNorm 里的倒数 RMS 系数。
    它只返回缩放因子 r，不直接改写输入；后面用 `linear(x) * r` 等价于先 RMS 再线性。
    """
    # x_flat : [T, hc*H]
    # return : [T, 1]
    return rsqrt(mean(x_flat ** 2, dim=-1, keepdim=True) + eps)


def split_sinkhorn(mixes, scale, base, hc, iters, eps):
    """
    对应源码 hc_split_sinkhorn。把 `hc_fn` 预测出的 mixing logits 拆成三类权重。
    `pre` 负责读多路 residual，`post` 负责写回子模块输出，`comb` 负责旧 residual 的跨路重组；
    其中 `comb` 会经过 Sinkhorn，变成近似双随机矩阵。

    注意：这是残差「流」上的混合系数，不是 token 维 Attention。
    详见上文「常见误解：hc_split_sinkhorn 不是注意力系数」。

    mixes : [T, (2+hc)*hc]     # 例如 hc=4 → [T, 24]
    scale : [3]                # s0=pre, s1=post, s2=comb
    base  : [(2+hc)*hc]        # 例如 [24]
    → pre : [T, hc]
      post: [T, hc]
      comb: [T, hc, hc]
    """
    # mixes 布局: [ pre_logits | post_logits | comb_logits ]
    #              [0, hc)       [hc, 2hc)     [2hc, (2+hc)*hc)

    pre  = sigmoid(mixes[:, :hc]            * scale[0] + base[:hc]) + eps
    # pre : [T, hc]

    post = POST_MULT * sigmoid(mixes[:, hc:2*hc] * scale[1] + base[hc:2*hc])
    # post: [T, hc]

    comb = mixes[:, 2*hc:] * scale[2] + base[2*hc:]   # [T, hc*hc]
    comb = comb.reshape(T, hc, hc)                    # [T, hc, hc]

    # init: row softmax + eps, then col normalize
    comb = exp(comb - comb.amax(dim=-1, keepdim=True))          # [T, hc, hc]
    comb = comb / comb.sum(dim=-1, keepdim=True) + eps          # 行归一
    comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)        # 列归一

    for _ in range(iters - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)    # row  → [T, hc, hc]
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)    # col  → [T, hc, hc]

    return pre, post, comb   # [T,hc], [T,hc], [T,hc,hc]


def hc_combine(R, pre):
    """
    作用：执行 `hc_pre` 的最后一步，把多路 residual 加权合成一路子模块输入。
    Attention / MoE 只看这一条 `[T,H]` 输入，不直接处理 `[T,hc,H]`。
    """
    # R   : [T, hc, H]
    # pre : [T, hc]
    # → y : [T, H]
    #
    # pre.unsqueeze(-1) * R  →  [T, hc, 1] * [T, hc, H] = [T, hc, H]
    # .sum(dim=1)            →  [T, H]
    return (pre.unsqueeze(-1) * R).sum(dim=1)


def hc_post(o, R, post, comb):
    """
    作用：把子模块输出 `o` 写回多路 residual，并同时重组旧 residual。
    这是 mHC 里对应普通 Transformer `x = x + sublayer(x)` 的写回阶段。
    """
    # o    : [T, H]
    # R    : [T, hc, H]
    # post : [T, hc]
    # comb : [T, hc, hc]
    # → R' : [T, hc, H]
    #
    # term1 = post.unsqueeze(-1) * o.unsqueeze(1)
    #       = [T, hc, 1] * [T, 1, H]  →  [T, hc, H]
    # term2 = einsum("tjk,tkh->tjh", comb, R)
    #       = [T, hc, hc] @ [T, hc, H]（对 k 求和）→ [T, hc, H]
    return (
        post.unsqueeze(-1) * o.unsqueeze(1)
        + einsum("tjk,tkh->tjh", comb, R)
    )
```

### 7.2 `hc_pre`

```python
def hc_pre(R, hc_fn, hc_scale, hc_base, rms_eps, hc_eps, sinkhorn_iters, hc):
    """
    作用：一个子层入口的 mHC 读阶段。
    它从当前多路 residual 预测 `pre/post/comb`，返回单路 `y` 给 Attention / MoE，
    同时把 `post/comb` 留给该子层输出后的 `hc_post` 使用。

    入参 / 权重:
      R        : [T, hc, H]                 # 多路 residual
      hc_fn    : [(2+hc)*hc, hc*H]          # 例如 [24, 4H]
      hc_scale : [3]
      hc_base  : [(2+hc)*hc]                # 例如 [24]
    返回:
      y        : [T, H]                     # 子模块输入
      post     : [T, hc]                    # 留给 hc_post
      comb     : [T, hc, hc]                # 留给 hc_post
    """
    T, hc, H = R.shape                      # R: [T, hc, H]
    x_flat = R.reshape(T, hc * H).float()   # [T, hc*H]

    r = rms_rsqrt(x_flat, rms_eps)          # [T, 1]
    mixes = linear(x_flat, hc_fn) * r       # [T, hc*H] @ [(2+hc)*hc, hc*H]^T
                                            # → [T, (2+hc)*hc]

    pre, post, comb = split_sinkhorn(
        mixes, hc_scale, hc_base, hc, sinkhorn_iters, hc_eps
    )
    # pre: [T, hc],  post: [T, hc],  comb: [T, hc, hc]

    y = hc_combine(R, pre).to(R.dtype)      # [T, H]
    return y, post, comb
    # 调用方再可选: y = RMSNorm(y)          # 仍为 [T, H]
```

### 7.3 `hc_head`

```python
def hc_head(R, hc_fn, hc_scale, hc_base, rms_eps, hc_eps):
    """
    作用：模型末端的 mHC 收束阶段。
    它不再产生 `post/comb`，只预测每一路 residual 的最终权重，把 `[T,hc,H]` 合成 `[T,H]`，
    再交给最后的 RMSNorm 和 `lm_head`。

    入参 / 权重:
      R        : [T, hc, H]
      hc_fn    : [hc, hc*H]                 # 注意比层内 hc_fn 窄：只出 hc 维
      hc_scale : [1] 或 scalar
      hc_base  : [hc]
    返回:
      z        : [T, H]
    """
    T, hc, H = R.shape                      # [T, hc, H]
    x_flat = R.reshape(T, hc * H).float()   # [T, hc*H]
    r = rms_rsqrt(x_flat, rms_eps)          # [T, 1]
    mixes = linear(x_flat, hc_fn) * r       # [T, hc]
    alpha = sigmoid(mixes * hc_scale + hc_base) + hc_eps
    # alpha: [T, hc]

    # alpha.unsqueeze(-1) * R → [T, hc, 1] * [T, hc, H] = [T, hc, H]
    # .sum(dim=1)             → [T, H]
    return (alpha.unsqueeze(-1) * R).sum(dim=1).to(R.dtype)
```

### 7.4 完整 Decoder Layer（非融合版）

```python
def decoder_layer(R, positions, batch, layer):
    """
    作用：展示一个 DeepSeek-V4 DecoderLayer 的非融合语义。
    一层里有两次 mHC：Attention 前后一次，MoE 前后一次；真实实现可把相邻 post/pre 融合，
    但数学结果等价于这里的顺序写法。

    R_in / R_out : [T, hc, H]

    层内权重形状（Attn / FFN 各一套）:
      hc_*_fn    : [(2+hc)*hc, hc*H]
      hc_*_scale : [3]
      hc_*_base  : [(2+hc)*hc]
    """

    # ---- Attention branch ----
    y, post_a, comb_a = hc_pre(
        R,                                              # [T, hc, H]
        layer.hc_attn_fn, layer.hc_attn_scale, layer.hc_attn_base,
        rms_eps, hc_eps, sinkhorn_iters, hc,
    )
    # y: [T, H],  post_a: [T, hc],  comb_a: [T, hc, hc]

    y = RMSNorm(y, layer.input_layernorm)               # [T, H]（可与 hc_pre 融合）
    o = layer.self_attn(y, positions, batch)            # [T, H]
    R = hc_post(o, R, post_a, comb_a)                   # [T, hc, H]

    # ---- FFN / MoE branch ----
    y, post_f, comb_f = hc_pre(
        R,                                              # [T, hc, H]
        layer.hc_ffn_fn, layer.hc_ffn_scale, layer.hc_ffn_base,
        rms_eps, hc_eps, sinkhorn_iters, hc,
    )
    # y: [T, H],  post_f: [T, hc],  comb_f: [T, hc, hc]

    y = RMSNorm(y, layer.post_attention_layernorm)      # [T, H]
    o = layer.mlp(y, batch)                             # [T, H]  DeepseekV2MoE
    R = hc_post(o, R, post_f, comb_f)                   # [T, hc, H]

    return R                                            # [T, hc, H]
```

### 7.5 整网骨架

```python
def deepseek_v4_forward(input_ids):
    """
    作用：展示整网里 mHC 的生命周期。
    普通 embedding 先扩成多路 residual；中间每层保持 `[T,hc,H]`；最后 `hc_head` 收回一路。
    """
    # input_ids : [T] 或 [B, S]（此处按展平后的 token 维 T 叙述）

    h = embed(input_ids)                        # [T, H]

    # 进入 mHC：把单路 hidden 扩成 hc 路
    # （具体是 repeat / 学到的投影，以 checkpoint 为准）
    R = expand_to_hc_streams(h, hc)             # [T, H] → [T, hc, H]

    for layer in layers:                        # num_hidden_layers 次
        R = decoder_layer(R, ...)               # [T, hc, H] → [T, hc, H]

    z = hc_head(
        R,                                      # [T, hc, H]
        hc_head_fn,                             # [hc, hc*H]
        hc_head_scale,                          # [1]
        hc_head_base,                           # [hc]
        ...
    )                                           # → [T, H]
    z = RMSNorm(z)                              # [T, H]
    logits = lm_head(z)                         # [T, vocab]
    return logits
```

> 实现细节：SGLang 在 fused 路径下，层与层之间可能 **延迟** 执行 `hc_post(ffn)`，把 `(residual [T,hc,H], post [T,hc], comb [T,hc,hc])` 传给下一层，与下一层 `hc_pre(attn)` 合成 `mhc_post_pre` 一次算完。语义仍等价于上面的非融合伪代码。

---

## 8. 和「普通 residual / 朴素 HC」对比

| | 标准 Residual | 朴素 HC（无约束） | **mHC（V4）** |
|---|---|---|---|
| 残差维度 | `[T,H]` | `[T,hc,H]` | `[T,hc,H]` |
| 进子模块前 | identity / 选一路 | 任意线性混合 | `pre` 门控加权和 |
| 出子模块后 | `x+o` | 任意写回 | `post⊙o + comb·R` |
| 混合矩阵 | — | 无结构 | **Sinkhorn → 近双随机** |
| 稳定性 | 高 | 多路易飘 | 约束后更稳 |
| 子模块接口 | `[T,H]` | `[T,H]` | 仍是 `[T,H]`（Attn/MoE 不用改内部） |

---

## 9. SGLang 实现要点（读代码时）

1. **参考数值路径**就看：
   - `DeepseekV4DecoderLayer.hc_pre` / `hc_post` 里的 torch fallback
   - `kernels/.../mhc.py` 的 `_hc_split_sinkhorn_torch`
2. **热路径**用 TileLang / FlashInfer / HIP aiter / NPU 融合 kernel；还有 DeepGEMM 做大 batch 的 `hc_prenorm` GEMM。
3. **`post` 乘 2**：kernel 与 torch 都是 `2 * sigmoid(...)`，对应 `_MHC_POST_MULT_VALUE = 2.0`。
4. **跨层融合**：`use_fused_mhc_post_pre` 时，layer `forward` 返回 `(hidden, residual, post, comb)`，下一层入口消费。
5. **Speculative（DSpark / NextN）**：draft 侧也要维护 `hc_mult` 维；`resolve_spec_hidden_size` 会把边界 hidden 扩成 `hidden * hc_mult`。

---

## 10. 最小数值例子（`hc=2` 示意）

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

## 11. 问答与理解补充

这一节整理实际阅读代码时最容易卡住的几个问题。

### 11.1 `hc_post` 是什么意思？

`hc_post` 不是 post-attention norm 里的 “post”，而是 **mHC 的写回阶段**。

在普通 Transformer 里，一个子层大致是：

```text
x = x + SubLayer(norm(x))
```

DeepSeek-V4 把单路 residual `x` 换成多路 residual `R ∈ [T,hc,H]` 后，就不能直接 `x + o` 了。子模块 Attention / MoE 仍然只输出一路 `o ∈ [T,H]`，所以需要 `hc_post` 把它写回多路：

```text
R'_j = post_j · o + Σ_k comb_jk · R_k
```

可以把它理解成两件事同时发生：

- `post_j · o`：把当前子模块输出灌进第 `j` 路 residual。
- `Σ_k comb_jk · R_k`：把旧的多路 residual 重新混合后保留下来。

所以 `hc_pre` 是“从多路读出一路给子模块”，`hc_post` 是“把子模块输出写回多路”。

### 11.2 `hc_head` 在哪里被调用？

`hc_head` 不在单个 Decoder Layer 内部调用，而是在 **所有 layers 跑完之后**，在整网末端调用。

伪代码里对应 §7.5：

```python
for layer in layers:
    R = decoder_layer(R, ...)               # [T, hc, H] → [T, hc, H]

z = hc_head(
    R,                                      # [T, hc, H]
    hc_head_fn,                             # [hc, hc*H]
    hc_head_scale,                          # [1]
    hc_head_base,                           # [hc]
    ...
)                                           # → [T, H]
z = RMSNorm(z)
logits = lm_head(z)
```

它的作用是 **最后收束**：中间所有层都保持 `[T,hc,H]` 的多路 residual，直到模型末端才通过 `hc_head` 合成普通 hidden `[T,H]`，再接最终 RMSNorm 和 `lm_head`。

因此：

- `hc_pre` / `hc_post`：每个 Decoder Layer 内反复使用。
- `hc_head`：整网最后只用一次，用来从 mHC 世界回到普通 hidden。

### 11.3 为什么一个 Decoder Layer 里要做两次 `hc_pre`？

因为一个 Decoder Layer 本来就有两个子模块：

```text
Attention
FFN / MoE
```

标准 Transformer 一层也有两次 residual 更新：

```text
x = x + Attention(norm(x))
x = x + FFN(norm(x))
```

DeepSeek-V4 只是把普通 residual `x` 换成多路 residual `R`，所以对应变成：

```text
R = hc_post(Attention(hc_pre(R)), R)
R = hc_post(MoE(hc_pre(R)), R)
```

也就是一层内的实际语义：

```text
R
│
├─ hc_pre(attn)  → y_attn
│                 ↓
│              Attention
│                 ↓
├─ hc_post(attn) → R_after_attn
│
├─ hc_pre(ffn)   → y_ffn
│                 ↓
│              MoE / FFN
│                 ↓
└─ hc_post(ffn)  → R_after_ffn
```

第一段 `hc_pre`：

```python
y, post_a, comb_a = hc_pre(
    R,
    layer.hc_attn_fn, layer.hc_attn_scale, layer.hc_attn_base,
    rms_eps, hc_eps, sinkhorn_iters, hc,
)
```

意思是：从当前多路 residual `R` 里混出一路 `y`，给 Attention 用。这里用 `layer.hc_attn_*`，因为 Attention 有自己的一套 mixing 参数。

第二段 `hc_pre`：

```python
y, post_f, comb_f = hc_pre(
    R,
    layer.hc_ffn_fn, layer.hc_ffn_scale, layer.hc_ffn_base,
    rms_eps, hc_eps, sinkhorn_iters, hc,
)
```

意思是：Attention 已经更新过 `R` 之后，再从新的 `R` 混出一路 `y`，给 FFN / MoE 用。这里用 `layer.hc_ffn_*`，因为 FFN / MoE 也有自己的一套 mixing 参数。

这两次不是重复，而是 **Attention 和 MoE 各自都有独立的 mHC 读写门控**：

- Attention 前的 `hc_pre`：决定哪些 residual 流的信息送去做注意力。
- Attention 后的 `hc_post`：决定注意力结果怎么写回多路流。
- MoE 前的 `hc_pre`：基于更新后的 residual，再决定哪些信息送去专家网络。
- MoE 后的 `hc_post`：把专家输出写回多路流。

### 11.4 `post_a/comb_a` 和 `post_f/comb_f` 为什么跟着 `hc_pre` 返回？

`hc_pre` 不只是生成子模块输入 `y`，它还一次性生成后面写回要用的 `post` 和 `comb`。

这是因为 `pre/post/comb` 都来自同一个 `mixes`：

```text
mixes = RMSNormLinear(flatten(R))

mixes → pre, post, comb
```

所以一次 `hc_pre` 会同时决定：

- 这次子模块要从多路 residual 里读什么：`pre`
- 子模块输出之后怎么灌回每一路：`post`
- 旧 residual 怎么跨路重组：`comb`

Attention 分支返回的是 `post_a/comb_a`，只给这次 Attention 的 `hc_post` 用；FFN 分支返回的是 `post_f/comb_f`，只给这次 MoE 的 `hc_post` 用。它们不能混用，因为参数不同、输入 residual 也不同。

---

## 12. 阅读顺序建议

1. 本文 §4–§6（先建立公式与伪代码）
2. `deepseek_v4.py` → `DeepseekV4DecoderLayer.forward`（非 fused 分支）
3. `mhc.py` → `_hc_split_sinkhorn_torch` + `hc_combine`
4. `mhc_head.py` → `fused_hc_head` 注释里的 shape
5. 再回头看 fused `mhc_post_pre`（优化，不改语义）

---

*与 `architecture.md` 配套：那份讲整网结构，本份只深入 mHC。*
