# RadixLinearAttention 详解（GLM-5.3-Flash / KDA 路径）

> 基于 SGLang 实现梳理：
>
> - `python/sglang/srt/layers/radix_linear_attention.py`（层壳 / dispatch）
> - `python/sglang/srt/models/glm5_next.py`（`Glm5NextLinearAttention` 投影与调用）
> - `python/sglang/srt/layers/attention/linear/kda_backend.py`（conv + KDA backend）
> - `python/sglang/kernels/ops/attention/fla/`（`kda.py` / `fused_sigmoid_gating_recurrent.py` 等 kernel）

整体架构见同目录 [architecture.md](./architecture.md)。这里只讲 **RadixLinearAttention 是什么、怎么接到 KDA、递推状态怎么更新**。

---

## 1. 先把名字拆开

在 GLM-5.3-Flash 里：

```text
Glm5NextLinearAttention          # 模型层：投影 + 门控 + o_proj
        │
        ▼
RadixLinearAttention             # 公共层壳：保存 head 维 / A_log / conv 权重，转发到 backend
        │
        ▼
KDAAttnBackend / TritonKDAKernel # 真正算：causal conv1d + gated delta rule
```

所以：

- **RadixLinearAttention** 本身几乎不算数学，主要是统一接口。
- Flash 这层的数学主体是 **KDA（Kimi Delta Attention）**：带 sigmoid / softplus 门控的 **gated delta rule** 线性注意力。
- 和 full MLA 不同：它缓存的是 **conv window + SSM 状态 `h`**，不是 dense / latent KV cache。

SGLang 注释里也写了：`TritonKDAKernel` = *Kimi Delta Attention linear attention*。

---

## 2. KDA 输入 / 输出（带维度）

下面默认按 **Flash 常见配置、单卡 / 未切 TP** 写。

### 2.0 变量 / 符号对照（先看这里）

形状里的字母是**缩写**，不是随便起的：


| 缩写   | 全称 / 配置字段                                           | Flash 常见值 | 中文含义                                                     |
| ---- | --------------------------------------------------- | --------- | -------------------------------------------------------- |
| `T`  | tokens / seq_len                                    | 变长        | 本步处理的 token 数。prefill=序列长；decode 时常每请求 1 个，batch 时 `T≈B` |
| `B`  | batch / num_seqs                                    | 变长        | 请求条数（decode 时常用）                                         |
| `H`  | `hidden_size`                                       | 4096      | 隐藏层宽度，残差流维度                                              |
| `A`  | `linear_num_heads` / `num_heads`                    | 64        | **linear / KDA 的 head 数**（Attention heads）。不是 `A_log`    |
| `D`  | `linear_head_dim` / `head_dim`                      | 128       | **每个 head 的维度**。Q/K/V 每个 head 都是 `D`                     |
| `C`  | `short_conv_kernel_size` / `linear_conv_kernel_dim` | 4         | short causal conv 的 kernel 长度                            |
| `AD` | `A * D`                                             | 8192      | 所有 head 拼起来的宽度：`num_heads * head_dim`                    |
| `P`  | `attn_tp_size`                                      | 1         | attention tensor parallel 切分数；切了以后本卡 head 数变成 `A/P`      |


容易混的几个名字：


| 名字                  | 是什么                                            | 不是什么          |
| ------------------- | ---------------------------------------------- | ------------- |
| `A`                 | head **个数**（64）                                | 不是参数 `A_log`  |
| `A_log`             | 可学习衰减参数，shape 约 `[1,1,A,1]`                    | 不是 head 数     |
| `a` / `forget_gate` | 本步算出的 forget **logits**，约 `[T, A*D]`           | 不是 `A`        |
| `b` / `beta`        | 本步算出的写入强度；logits 为 `[T,A]`，sigmoid 后仍是 `[T,A]` | 不是 batch `B`  |
| `D`                 | head_dim                                       | 不是 depth / 层数 |
| `h`（小写）             | SSM 矩阵状态，每 head `[D, D]`                       | 不是 hidden `H` |


读形状时建议默念中文，例如：

```text
[T, A, D]     = [token数, head数, head维]
[T, A*D]      = [token数, head数×head维]
[T, 3*A*D]    = [token数, 3组(q/k/v)拼在一起]
[n_slots, A, D, D] = [cache槽位, head数, V维, K维]  # V=K=D
```

若开了 attention TP（`attn_tp_size = P`），文中的 `A` 表示**本 rank 的 head 数** `A/P`。

### 2.1 整层 `Glm5NextLinearAttention`（最外层）

```text
输入:  x            [T, H]           # [token, hidden]
输出:  y            [T, H]           # [token, hidden]
```

### 2.2 `RadixLinearAttention` / KDA core（中间算子）

```text
输入:
  mixed_qkv         [T, 3*A*D]       # [token, 3*num_heads*head_dim]，q||k||v，尚未 short conv
  a = forget_gate   [1, T, A*D]      # [1, token, num_heads*head_dim]；decode 也可能是 [T, A*D]
  b = beta          [1, T, A]        # [1, token, num_heads]；或 [T, A] / [T, 1, A]

输出:
  core              [1, T, A, D]     # [1, token, num_heads, head_dim]
```

挂在 layer 上、随 forward 一起用的参数 / cache（不是本步从 `x` 新算出来的）：

```text
A_log               [1, 1, A, 1]     # 可学习衰减底数（按 head）；别和缩写 A 搞混
dt_bias             [A*D]            # [num_heads*head_dim]，加在 forget logits 上
conv_weights        [3*A*D, C]       # [3*AD, kernel]；depthwise causal conv 权重
conv_state          [n_slots, 3*A*D, C-1]   # 短卷积滑窗 cache
ssm_state h         [n_slots, A, D, D]      # [槽位, head, V=D, K=D]；每 head 一个 DxD 矩阵
```

### 2.3 端到端伪代码（维度标全）

```python
# ============================================================
# Glm5NextLinearAttention + RadixLinearAttention(KDA)
#
# 变量含义（Flash 常见数值）:
#   T  = token 数
#   H  = hidden_size           = 4096
#   A  = linear_num_heads      = 64     # head 个数
#   D  = linear_head_dim       = 128    # 每个 head 的维
#   C  = short_conv_kernel     = 4
#   AD = A * D                 = 8192   # num_heads * head_dim
# ============================================================

def glm5_kda_layer(x, forward_batch, params, cache):
    """
    x:              [T, H]              # [token, hidden]
    return y:       [T, H]              # [token, hidden]
    """

    # ---------- 1) 投影（或 fused_qkvbfg）----------
    mixed_qkv = qkv_proj(x)                 # [T, 3*AD] = [T, 3*A*D] = [T, 24576]
    # 拆开的语义（尚未 conv）:
    #   q_raw, k_raw, v_raw = split(mixed_qkv, [AD, AD, AD], dim=-1)
    #   各 [T, AD] = [T, A*D] = [token, num_heads*head_dim]

    beta = b_proj(x)                        # [T, A] = [T, 64] = [token, num_heads]
    forget = f_b_proj(f_a_proj(x))          # [T, AD] = [T, 8192]；per-(head,K) forget logits
    out_gate = g_b_proj(g_a_proj(x))        # [T, AD] = [T, 8192]；输出 RMSNorm 门控

    # RadixLinearAttention 入口约定的 batch 维
    a = forget                              # [T, AD]；这里的 a 是 forget logits，不是 head 数 A
    b = beta.unsqueeze(0)                   # [1, T, A] = [1, token, num_heads]
    if not forward_batch.is_decode():
        a = a.unsqueeze(0)                  # [1, T, AD]

    # ---------- 2) RadixLinearAttention / KDA backend ----------
    core = radix_kda_forward(
        mixed_qkv=mixed_qkv,                # [T, 3*AD]
        a=a,                                # forget logits
        b=b,                                # beta logits
        A_log=params.A_log,                 # [1, 1, A, 1]；参数名，不是缩写 A
        dt_bias=params.dt_bias,             # [AD]
        conv_w=params.conv_weights,         # [3*AD, C]
        conv_state=cache.conv_state,        # [n_slots, 3*AD, C-1]
        ssm_state=cache.ssm_state,          # [n_slots, A, D, D]
        scale=D ** -0.5,
    )
    # core: [1, T, A, D] = [1, token, num_heads, head_dim]

    # ---------- 3) 输出门控 + 投影 ----------
    gate = out_gate.view(T, A, D)           # [T, A, D] = [token, num_heads, head_dim]
    core = FusedRMSNormGated(core, gate)    # 仍约 [1, T, A, D]
    core = core.squeeze(0).flatten(-2)      # [T, A*D] = [T, AD]
    y = o_proj(core)                        # [T, H] = [token, hidden]
    return y


def radix_kda_forward(mixed_qkv, a, b, A_log, dt_bias, conv_w, conv_state, ssm_state, scale):
    """
    mixed_qkv: [T, 3*AD]           # [token, 3*num_heads*head_dim]
    a:         [1, T, AD]          # forget logits；[1, token, num_heads*head_dim]
    b:         [1, T, A]           # beta logits；[1, token, num_heads]
    return:    [1, T, A, D]        # [1, token, num_heads, head_dim]
    """

    # short causal conv（depthwise，对 3*AD 通道）
    qkv = causal_conv1d(mixed_qkv, conv_w, conv_state)   # [T, 3*AD]
    q, k, v = split(qkv, [AD, AD, AD], dim=-1)           # 各 [T, AD]
    q = q.view(T, A, D)                                  # [T, A, D] = [token, head, head_dim]
    k = k.view(T, A, D)
    v = v.view(T, A, D)

    # 门控：a/b 是本步 logits；A_log/dt_bias 是层参数
    a = a.reshape(T, A, D)                               # [T, A, D]
    b = b.reshape(T, A)                                  # [T, A]
    g = -exp(A_log.reshape(A, 1)) * softplus(a + dt_bias.view(A, D))  # [T, A, D]
    beta = sigmoid(b)                                    # [T, A]；写入强度 ∈ (0,1)

    # 逐步 / chunk 递推（写法按 decode 逐步；prefill 用等价 chunk kernel）
    # h: [A, V, K] = [A, D, D] = [num_heads, head_dim, head_dim]
    h = ssm_state[slot]                                  # 由 cache_indices 选槽
    o = empty(T, A, D)
    for t in range(T):
        qt = l2norm(q[t]) * scale           # [A, D]
        kt = l2norm(k[t])                   # [A, D]
        vt = v[t]                           # [A, D]
        gt = g[t]                           # [A, D]
        bt = beta[t]                        # [A] = [num_heads]

        # KDA gated delta rule（逐 head）
        # h[a]: [V, K] = [D, D]
        h = h * exp(gt)[:, None, :]         # 按 K 维衰减: [A, V, K] * [A, 1, K]
        vt = vt - einsum("avk,ak->av", h, kt)
        vt = vt * bt[:, None]               # [A, D]
        h = h + einsum("av,ak->avk", vt, kt)
        o[t] = einsum("avk,ak->av", h, qt)  # [A, D]

    return o.unsqueeze(0)                   # [1, T, A, D]
```

### 2.4 一张表对照


| 张量                | Shape                 | 展开读法                              | 含义                                |
| ----------------- | --------------------- | --------------------------------- | --------------------------------- |
| `x`               | `[T, H]`              | `[token, hidden]`                 | 层输入                               |
| `mixed_qkv`       | `[T, 3AD]`            | `[token, 3*num_heads*head_dim]`   | 投影后的 qkv 拼接                       |
| `a` / forget      | `[1,T,AD]` 或 `[T,AD]` | `[…, num_heads*head_dim]`         | forget **logits**（不是 head 数 `A`）  |
| `b` / beta logits | `[1,T,A]` 或 `[T,A]`   | `[…, num_heads]`                  | 写入强度 logits                       |
| `out_gate`        | `[T, AD]`             | `[token, num_heads*head_dim]`     | 输出 RMSNorm 门控                     |
| `q,k,v`（conv 后）   | `[T, A, D]`           | `[token, num_heads, head_dim]`    | 真正进递推的 QKV                        |
| `g`               | `[T, A, D]`           | `[token, num_heads, head_dim]`    | `-exp(A_log)*softplus(a+dt_bias)` |
| `beta`            | `[T, A]`              | `[token, num_heads]`              | `sigmoid(b)`                      |
| `h`（状态）           | `[n_slots, A, D, D]`  | `[槽位, num_heads, V, K]`           | 每 head 的矩阵记忆，`V=K=D`              |
| `core`            | `[1, T, A, D]`        | `[1, token, num_heads, head_dim]` | KDA 输出                            |
| `y`               | `[T, H]`              | `[token, hidden]`                 | `o_proj` 后层输出                     |


Decode 时常见 `T = B`（batch 里每请求 1 token），状态通过 `cache_indices` 索引到对应 `n_slots` 行；prefill 则对一条序列的 `T` 做 chunked 递推，最后写回该请求的 `h`。

---

## 3. `RadixLinearAttention` 层壳做什么

源码类很小，核心字段：

```python
class RadixLinearAttention(nn.Module):
    def __init__(..., conv_weights, bias, A_log, dt_bias, lower_bound=None):
        self.num_q_heads / num_k_heads / num_v_heads
        self.head_q_dim / head_k_dim / head_v_dim
        self.conv_weights, self.bias, self.activation   # short conv
        self.A_log, self.dt_bias, self.lower_bound     # KDA / GDN 共享门控参数
```

`forward(forward_batch, mixed_qkv, a, b)`：


| 参数          | 在 GLM5-Flash 里的含义                                 |
| ----------- | ------------------------------------------------- |
| `mixed_qkv` | `qkv_proj(x)` 后的拼接向量，稍后会过 causal conv，再拆成 q/k/v   |
| `a`         | forget gate 原始 logits（再经 softplus / `A_log` 变成衰减） |
| `b`         | beta logits（sigmoid 后变成写入强度）                      |


转发逻辑（简化）：

```python
def forward(self, forward_batch, mixed_qkv, a, b):
    # 1) extend + piecewise CUDA graph：走 unified custom op，写到预分配 output
    # 2) DP padded extend：只在真实 token 前缀上算，padding 位置清零
    # 3) 常规路径：get_attn_backend().forward(layer=self, mixed_qkv, a, b)
    return get_attn_backend().forward(...)
```

要点：

- **不在这一层里写循环递推**；递推在 KDA kernel。
- `A_log` / `dt_bias` / `conv_weights` 挂在 layer 上，backend 用 `layer.A_log` 取。
- 还有一套 `unified_linear_attention_with_output` custom op，给 piecewise CUDA graph / breakable graph 用，避免 graph 里直接抓 Python 上下文。

`RadixLinearAttention` 构造时 Flash 一般设：

```python
RadixLinearAttention(
    num_q_heads=local_A, num_k_heads=local_A, num_v_heads=local_A,
    head_q_dim=D, head_k_dim=D, head_v_dim=D,
    conv_weights=qkv_conv1d.weight.squeeze(1),
    A_log=A_log, dt_bias=dt_bias,
)
```

---

## 4. Radix 之外：整层还在做什么

`Glm5NextLinearAttention` / `glm5_kda_layer` 里，**RadixLinearAttention 只负责中间核心**。外面还有两段。

### 4.1 之前：投影造料

从 `x [T, H]` 造出 KDA 需要的输入（或 fused 一次算完）：


| 产物                         | 作用                                             | Shape（常见）  |
| -------------------------- | ---------------------------------------------- | ---------- |
| `mixed_qkv`                | 投成 q||k||v                                     | `[T, 3AD]` |
| `beta`（`b_proj`）           | 写入强度 logits → 进 Radix 变 `sigmoid(b)`           | `[T, A]`   |
| `forget_gate`（`f_a`→`f_b`） | 遗忘 logits → 进 Radix 和 `A_log`/`dt_bias` 变成 `g` | `[T, AD]`  |
| `out_gate`（`g_a`→`g_b`）    | **输出门控，不进 Radix**，留给最后的 RMSNorm                | `[T, AD]`  |


非 decode 时还会给 `forget` / `beta` 加一维，对齐 backend 接口。

> **short causal conv 不在这一层外面单独做**，是进 Radix / KDA backend 之后才做。

### 4.2 之后：门控归一化 + 投回 hidden


| 步骤                                  | 作用                              |
| ----------------------------------- | ------------------------------- |
| `FusedRMSNormGated(core, out_gate)` | 对每个 head 的 `D` 维做 gated RMSNorm |
| `squeeze + flatten`                 | `[1,T,A,D] → [T, AD]`           |
| `o_proj`                            | 线性投回 `[T, H]`                   |


### 4.3 分工一句话

```text
x [T,H]
  │
  ├─ qkv / beta / forget / out_gate   ← 投影造料（Radix 外）
  │
  ▼
RadixLinearAttention(KDA)             ← conv + 递推（Radix 内）
  │ core [1,T,A,D]
  ▼
RMSNormGated(out_gate) → o_proj       ← 收成 y [T,H]（Radix 外）
```

**外层 = hidden 的读入/写出接口与门控投影；Radix = 线性注意力递推核心。**

---

## 5. Backend：从 `mixed_qkv` 到递推

`KDA` backend 的典型 decode 路径：

```text
mixed_qkv
   │
   ▼
causal_conv1d_update(mixed_qkv, conv_state, conv_weights)   # depthwise short conv + SiLU
   │
   ▼
split -> q, k, v     # 各 [..., A, D]
   │
   ▼
fused_sigmoid_gating_delta_rule_update / packed_decode
   │  用 a, b, A_log, dt_bias 算 g, beta
   │  更新 SSM state h
   ▼
core_attn_out [1, B, A, D]
```

Prefill / extend 类似，只是：

- conv 用 `causal_conv1d_fn`（varlen）
- 递推常用 **chunked KDA**（`chunk_kda`）而不是逐步 decode kernel
- 仍读写同一套 `conv_state` + `ssm_states`（mamba pool 的 temporal）

状态形状（概念上）：

```text
conv_state  ≈ [n_slots, 3*A*D, C-1]         # 短卷积滑窗，C=kernel
ssm_state h ≈ [n_slots, A, D, D]            # 每个 head 一个 V×K 矩阵，V=K=D
```

### 5.1 `causal_conv1d` 怎么做

给 `mixed_qkv` 的每个通道做 **depthwise 因果短卷积**，再过 **SiLU**（在递推之前）：

```text
C = 4（Flash 常见 short_conv_kernel）
对通道 c、时刻 t：

y[t,c] = SiLU(
    w[c,0]*x[t-3,c] + w[c,1]*x[t-2,c]
  + w[c,2]*x[t-1,c] + w[c,3]*x[t,  c]
  + bias[c]
)
```


| 性质        | 含义                                  |
| --------- | ----------------------------------- |
| Depthwise | 每个通道自己卷，通道间不混                       |
| Causal    | 只用当前和过去，不看未来                        |
| 短         | 只看最近 `C` 个 token，所以 cache 长度是 `C-1` |



| 模式             | API                    | 行为                                        |
| -------------- | ---------------------- | ----------------------------------------- |
| Prefill/Extend | `causal_conv1d_fn`     | 整段一次卷；可接已有 `conv_state`；末尾写回              |
| Decode         | `causal_conv1d_update` | 每步 1 token：用 state 的 `C-1` 历史 + 当前算输出，再滑窗 |


Decode 一步：

```text
# state 已有 [x_{t-3}, x_{t-2}, x_{t-1}]，各 [3AD]
y_t = SiLU( w · concat(state, x_t) + bias )
state <- [x_{t-2}, x_{t-1}, x_t]
```

作用：KDA 的 `h` 是长期压缩记忆；short conv 先给 q/k/v 加一点**局部邻域混合**，再进 delta rule（和 Mamba 前面短卷积同类）。

### 5.2 `conv_state` / `h` 能不能当 cache 复用

**可以，而且本来就是 cache。**

- **同一次生成**：decode 必须逐步复用 `conv_state`（和 `h`），否则短卷积 / 递推会断。
- **跨请求 prefix cache**：前缀一致时，可以从已算好的 conv 窗口 + SSM 终态恢复（KDA backend 有 track / restore 相关逻辑）；`**conv_state` 与 `h` 通常要一起对齐恢复**。
- **不能乱复用**：后缀变了、中间 token 变了，就要重算或回滚。
- **体积**：每层每请求大约 `O((C-1)·3AD)` 的 conv 窗 + `O(A·D·D)` 的 `h`，相对序列长度恒定，远小于完整 KV。

---

## 6. 核心数学：KDA = gated delta rule

把 kernel 里的更新写成逐步形式（`IS_KDA=True`）：

### 6.1 门控怎么算

```text
# a: forget logits, 形状约 [B, HV, K]（KDA per-K）
# b: beta logits,   形状约 [B, HV]

u  = a + dt_bias
Δ  = softplus(u)                         # 恒 > 0，像步长 / 衰减强度
A  = -exp(A_log)                         # 恒 < 0，稳定衰减底数（参数重参数化）
g  = A * Δ = -exp(A_log) * softplus(u)   # 恒 ≤ 0，log 域门控
beta = sigmoid(b)                        # 写入强度，(0, 1)
```

可选 `lower_bound`：限制 gate，避免衰减过狠（safe gate）。

#### 为什么像“两次 exp”，其实不是

看到：

```text
g = -exp(A_log) * softplus(a + dt_bias)
h ← h * exp(g)
```

容易以为对同一个量做了两次 `exp`。两层 `exp` 干的是**不同的事**：


| 出现位置         | 作用                                                |
| ------------ | ------------------------------------------------- |
| `exp(A_log)` | 把可学习参数变成恒正尺度，再取负得到 `A=-exp(A_log)`，保证衰减底数恒负、训练不发散 |
| `exp(g)`     | 把 log 域门控 `g≤0` 变成乘法遗忘系数 `exp(g)∈(0,1]`           |


等价于 SSM / Mamba 离散化：

```text
h_new = exp(A · Δ) · h_old
```

这里 `g = A·Δ`，`exp(g)` 才是真正乘到 `h` 上的因子。  
**不是 `exp(exp(...))` 叠在同一物理量上。**

### 6.2 Q/K 归一化与缩放

```text
q <- l2_normalize(q)
k <- l2_normalize(k)
q <- q * scale                  # scale = 1/sqrt(D)
```

### 6.3 一步递推（delta rule）

对每个 head，状态 `h ∈ R^{V × K}`：

```text
# 1) 遗忘 / 衰减
h <- h * exp(g)                 # KDA: g 是 per-K 向量，按 K 维广播

# 2) delta 修正：先去掉当前 k 方向上已有内容，再按 beta 写入新 v
v <- v - h @ k                  # 等价于 v -= sum_k h[:,k] * k[k]
v <- v * beta

# 3) 外积写入
h <- h + outer(v, k)            # h[v,k] += v[v] * k[k]

# 4) 读出
o <- h @ q                      # o[v] = sum_k h[v,k] * q[k]
```

对应 Triton 注释里的顺序：

```text
h *= exp(g)
v -= sum(h * k, dim=K)
v *= beta
h += k[:, None] * v[None, :]
o  = sum(h * q, dim=K)
```

直觉：

- `**g**`：控制旧记忆衰减多快（forget）；本身是 log 域，`exp(g)` 才是乘子。
- `**beta**`：控制这一步新信息写进 `h` 的强度。
- **delta**：先从 `h` 读出对当前 `k` 的预测并减掉，再写入，接近 gated delta-rule 思路。
- **输出**：用归一化后的 `q` 去读矩阵状态 `h`，复杂度对序列长度是递推的，不显式存整段 KV。

---

## 7. 和 Softmax Attention / MLA 的对比


|             | Softmax / MLA  | RadixLinearAttention (KDA)         |
| ----------- | -------------- | ---------------------------------- |
| 交互形式        | token 两两相似度    | 固定大小状态 `h` 上的递推读写                  |
| Cache       | KV / latent KV | conv window + SSM `h`（可 prefix 复用） |
| Prefill 复杂度 | 通常对长度二次或稀疏近似   | chunk 线性注意力，近似线性                   |
| Decode      | 读增长的 cache     | 单步更新 `h`，cache 大小恒定                |
| Flash 中的位置  | 周期性 full MLA 层 | 默认大多数层                             |


Flash hybrid 的设计意图：多数层用 KDA 扛长上下文状态，少数 MLA 层做精确检索（可再配 DSA/indexer）。

---

## 8. Prefill chunk 递推 vs Decode 单步递推

**数学上同一套 KDA delta rule**；差别在实现怎么扫序列、怎么读写 `h`。


|                      | Decode 单步                                    | Prefill chunk                   |
| -------------------- | -------------------------------------------- | ------------------------------- |
| 一次 forward 的 token 数 | 通常每请求 `T=1`                                  | 一整段 `T`（可变长）                    |
| 递推方式                 | 对当前 1 个 token 更新 `h`                         | 按 `chunk_size`（常见 **64**）切块处理   |
| `h` 读写               | 每步读 slot → 更新 → 写回                           | chunk 间传递 `h`；整段结束写终态           |
| 为何不逐步扫 prefill       | —                                            | 逐步扫正确但太慢；chunk 内可并行/矩阵化         |
| 典型 kernel            | `packed_decode` / `fused_sigmoid_gating_...` | `chunk_kda`（或 cutedsl/flashkda） |
| conv                 | `causal_conv1d_update`                       | `causal_conv1d_fn`（整段 + state）  |


下面伪代码共用同一步更新函数（与 §6.3 一致）：

```python
def kda_step(h, q, k, v, g, beta, scale):
    """
    h: [A, D, D]   # 当前 SSM 状态
    q,k,v,g: [A, D]
    beta: [A]
    return o [A, D], h_new [A, D, D]
    """
    q = l2norm(q) * scale
    k = l2norm(k)

    h = h * exp(g)[:, None, :]              # 遗忘
    v = v - einsum("avk,ak->av", h, k)      # delta
    v = v * beta[:, None]
    h = h + einsum("av,ak->avk", v, k)      # 写入
    o = einsum("avk,ak->av", h, q)          # 读出
    return o, h
```

### 8.1 Decode：单步递推

```python
def kda_decode_one_token(
    mixed_qkv_t,      # [1, 3AD]  当前 token 的投影
    a_t, b_t,         # forget / beta logits
    A_log, dt_bias,
    conv_state,       # [3AD, C-1]
    h,                # [A, D, D]  该请求的 SSM 状态
    conv_w, scale,
):
    # 1) 短卷积：用滑窗 cache 只产出 1 个 token
    qkv_t, conv_state = causal_conv1d_update(
        mixed_qkv_t, conv_state, conv_w, activation="silu"
    )                                       # qkv_t: [1, 3AD]
    q, k, v = split_heads(qkv_t)            # 各 [A, D]

    # 2) 门控（本步）
    g = -exp(A_log) * softplus(a_t + dt_bias)   # [A, D]
    beta = sigmoid(b_t)                         # [A]

    # 3) 单步递推，立刻写回 h
    o, h = kda_step(h, q, k, v, g, beta, scale)

    return o, conv_state, h                 # o: [A, D]
```

批量 decode 时：每个请求各自一个 `(conv_state, h)` slot，互不影响；每步 `T≈B` 个 token 并行，但**每条序列长度仍是 1**。

### 8.2 Prefill：若逐步扫（语义参考，实际不用）

```python
def kda_prefill_naive(mixed_qkv, a, b, A_log, dt_bias, h0, conv_w, scale):
    """
    mixed_qkv: [T, 3AD]
    语义上 = 把 decode 重复 T 次。正确，但 prefill 太慢。
    """
    qkv = causal_conv1d_fn(mixed_qkv, conv_w, ...)   # [T, 3AD]，整段因果 conv
    q, k, v = split_heads(qkv)                       # [T, A, D]
    g = -exp(A_log) * softplus(a + dt_bias)          # [T, A, D]
    beta = sigmoid(b)                                # [T, A]

    h = h0                                           # [A, D, D]；可来自 prefix cache
    outs = []
    for t in range(T):                               # 串行 T 步
        o_t, h = kda_step(h, q[t], k[t], v[t], g[t], beta[t], scale)
        outs.append(o_t)
    return stack(outs), h                            # [T, A, D], 终态 h
```

### 8.3 Prefill：chunk 递推（实际路径）

把长度 `T` 切成宽度 `BT=64` 的块。块与块之间仍要按时间传递 `h`；**块内部**用 chunk 公式一次性算出该块所有输出（等价于块内逐步 `kda_step`，但可并行/矩阵化）。

```python
def kda_prefill_chunk(
    mixed_qkv,        # [T, 3AD]
    a, b,             # forget / beta
    A_log, dt_bias,
    h0,               # [A, D, D]  初始状态（无 prefix 则为 0）
    conv_w, scale,
    BT=64,            # chunk_size，FLA/Triton KDA 常见 64
):
    # 1) 整段 short conv（一次），得到全部 q,k,v
    qkv = causal_conv1d_fn(mixed_qkv, conv_w, initial_state=..., ...)
    q, k, v = split_heads(qkv)                # [T, A, D]
    # 2) 门控：可与 chunk-local cumsum 融合（实现里 kda_gate_chunk_cumsum）
    g = -exp(A_log) * softplus(a + dt_bias)   # [T, A, D]
    beta = sigmoid(b)                         # [T, A]

    # 3) 真实实现不是 Python for-chunk 调 block，而是下面 §8.3.2 的整段流水
    return chunk_kda(q, k, v, g, beta, scale, h0, ...)
```

概念上仍等价于「对每个 chunk 做一块递推」；代码里把「所有 chunk 的块内工作」和「跨 chunk 传 `h`」拆开成全局 kernel，见 §8.3.2。

### 8.3.1 拆分 chunk /「块内并行」是什么意思

**拆分 chunk**：把长度 `T` 切成宽度 `BT=64` 的时间块，例如 `T=200` → chunk0=`[0,64)`、chunk1=`[64,128)`、…。块与块之间必须靠状态 `h` 接力（因果），不能乱序。

**「块内并行」不是说块内 token 互不依赖。** 块内仍是因果的：token `t` 只看同块里 `≤t` 的写入，外加块首 `h`。

这里的「并行」指**算子形态**，相对逐步 `for t: kda_step`：

```text
逐步（串行递推）          chunk（矩阵化）
for t in chunk:           一次算 BT×BT 下三角 Akk / Aqk
  h ← forget/delta/write  再 solve + GEMM 得到该块全部 w,u,o 相关量
  o_t ← read(h)           GPU 上对 (chunk, head) 开大量 CTA 一起算
```

所以有两层：


| 说法             | 实际含义                                                                                     |
| -------------- | ---------------------------------------------------------------------------------------- |
| 跨 chunk 并行（②④） | 各 chunk 的 `Akk/Aqk/w/u` 或最终 `o` **只依赖本块 qkvg +（④还要）本块入口 h**；入口 h 已知后，不同 chunk 的 CTA 可同时跑 |
| 块内并行 / 矩阵化     | 一块 64 token 的相互作用打成矩阵乘 + 三角求解，而不是 64 次标量递推循环                                             |
| 跨 chunk 串行（③）  | `h` 必须 chunk0 → chunk1 → … 传，这一段不能并行扫时间                                                  |


直觉：chunk 把「长串行」收成「短串行（NT 段）+ 每段里一大坨可并行线性代数」。

### 8.3.2 `chunk_kda` 实际实现（= 教学里的 `chunk_kda_block` 全集）

源码：`fla/kda.py` → `chunk_kda` → `chunk_kda_fwd`。`BT=64`，块内再切 `BC=16` 四个子块。

```text
q,k,v,g_raw,beta, h0
        │
        ▼
① kda_gate_chunk_cumsum / chunk_local_cumsum
   g ← activate(+ chunk 内 cumsum)；log2 空间，后续用 exp2
        │
        ▼
② chunk_kda_fwd_intra          【所有 chunk 并行】
   建块内下三角 Akk、Aqk；solve → Akk⁻¹
   再算 w, u, kg（WY / recompute）
        │
        ▼
③ chunk_gated_delta_rule_fwd_h 【按 chunk 串行扫】
   用 w,u,kg 把 h 跨 chunk 接力；写出每块入口状态 h[c]、v_new
   （可选 inplace 写回终态到 state pool）
        │
        ▼
④ chunk_gla_fwd_o_gk           【所有 chunk 并行】
   o ← scale·q 读 h[c] + 块内 Aqk @ v_new
```

总控（对齐 `chunk_kda_fwd`）：

```python
def chunk_kda_fwd(q, k, v, g, beta, scale, h0, BT=64):
    g = gate_and_chunk_cumsum(g, A_log, dt_bias, BT)                 # ①
    w, u, kg, Aqk = chunk_kda_fwd_intra(q, k, v, g, beta, scale, BT) # ②
    h_chunk, v_new, h = chunk_gated_delta_rule_fwd_h(                # ③
        kg, w, u, g, h0, BT
    )
    o = chunk_gla_fwd_o_gk(q, v_new, g, Aqk, h_chunk, scale, BT)     # ④
    return o, h
```

下面把 ②③④ 展开成可读伪代码（**单 head、一块/整段**视角；真实是 Triton 对 `(chunk, head)` 开 grid）。约定：传入的 `g` 已是 **chunk 内 cumsum 后的 log 门控**（log2 空间，用 `exp2`）。

#### A. `chunk_kda_fwd_intra`（`chunk_intra.py`）

块内只看本块 `BT` 个 token，**不读 `h`**。产出后面要用的 `w/u/kg/Aqk`。所有 chunk 可并行。

```python
def chunk_kda_fwd_intra(q, k, v, g, beta, scale, BT=64):
    """
    单 head、单 chunk 局部下标 0..L-1（末块 L 可能 < BT）。
    q,k,v,g: [L, D]   beta: [L]
    返回 w,kg: [L,D]；u: [L,D]；Aqk: [L,L]
    """
    L = len(beta)
    Akk = zeros(L, L)   # 严格下三角（对角 0）
    Aqk = zeros(L, L)   # 含对角的下三角

    # 1) 建块内因果矩阵（实现里按 BC=16 子块 GEMM，这里写成双重循环）
    for i in range(L):
        for j in range(i + 1):          # j ≤ i
            # 相对门控：真实用 exp2(g_i − g_ref) 等拼，此处示意
            decay = exp2(g[i] - g[j])   # 逐通道
            if i > j:
                Akk[i, j] = beta[i] * dot(k[i], k[j] * decay)
            Aqk[i, j] = scale * dot(q[i] * decay_q(i), k[j] * decay_k(j))

    # 2) 解 (I − Akk) → Akk_inv（对角前代换 + 子块合并 = inter_solve_fused）
    Akk_inv = solve_tril(eye(L) - Akk)  # [L, L]

    # 3) recompute（对齐 fused 注释）
    #    u  = Akk_inv @ (v * beta)
    #    w  = Akk_inv @ (k * beta * exp2(g))
    #    kg = k * exp2(g_end - g)
    g_end = g[L - 1]
    u  = Akk_inv @ (v * beta[:, None])
    w  = Akk_inv @ (k * beta[:, None] * exp2(g))
    kg = k * exp2(g_end[None, :] - g)

    # Aqk 原样留给 ④；右乘 Akk_inv 的效果由 v_new ≈ Akk_inv @ … 吸收
    return w, u, kg, Aqk
```

直觉：`Akk` = 块内后面 token 对前面写入的 delta 耦合；解一次 ≈ 消掉块内多次 `kda_step` 的相互影响。

#### B. `update_h_across_chunk`（真实函数：`chunk_gated_delta_rule_fwd_h`）

教学名 `update_h_across_chunk`；源码在 `chunk_delta_h.py`，对 `NT` 个 chunk **串行**扫。

```python
def chunk_gated_delta_rule_fwd_h(kg, w, u, g, h0, BT=64):
    """
    kg,w,u,g: [T, D]   h0: [D, D]   # 单 head；状态为 V×K，此处 V=K=D
    返回 h_chunk[NT,D,D], v_new[T,D], h_final[D,D]
    """
    T, D = kg.shape
    NT = cdiv(T, BT)
    h = h0
    h_chunk = empty(NT, D, D)
    v_new = empty(T, D)

    for c in range(NT):                         # 唯一时间串行
        s, e = c * BT, min((c + 1) * BT, T)
        h_chunk[c] = h                          # 先存入口，给 ④

        # delta：v_new = u − w @ hᵀ
        v_new[s:e] = u[s:e] - (w[s:e] @ h.T)

        # 等价于 update_h_across_chunk(...)
        h = update_h_across_chunk(h, kg[s:e], v_new[s:e], g[s:e])

    # 真实还可 inplace 把 h 写回 state pool
    return h_chunk, v_new, h


def update_h_across_chunk(h, kg, v_new, g):
    """一块：入口 h → 出口 h（下一块入口）。"""
    g_end = g[-1]
    h = h * exp2(g_end)[None, :]                # 按 K 通道乘块末门控
    h = h + einsum("ld,lk->dk", v_new, kg)      # Σ_t v_new_t ⊗ kg_t
    return h
```

#### C. `chunk_gla_fwd_o_gk`（`kda.py` → `chunk_gla_fwd_kernel_o`）

每块入口 `h` 已知后，**所有 chunk 并行**。

```python
def chunk_gla_fwd_o_gk(q, v_new, g, Aqk, h_chunk, scale, BT=64):
    """
    q,v_new,g: [T,D]  Aqk: [T,BT]（每行本 chunk 相对列）  h_chunk: [NT,D,D]
    """
    T, D = q.shape
    o = empty(T, D)
    for c in range(cdiv(T, BT)):                # 实现里是 grid 并行，不是 Python for
        s, e = c * BT, min((c + 1) * BT, T)
        L = e - s
        # ① 读块首状态：o += (scale·q·exp2(g)) @ hᵀ
        o[s:e] = ((q[s:e] * scale) * exp2(g[s:e])) @ h_chunk[c].T
        # ② 块内：因果 Aqk @ v_new
        A = tril(Aqk[s:e, :L])                  # 保留 i ≥ j
        o[s:e] = o[s:e] + A @ v_new[s:e]
    return o
```

| 教学名 | 真实位置 |
|---|---|
| `chunk_kda_fwd_intra` | `chunk_intra.py` |
| `update_h_across_chunk` | `chunk_gated_delta_rule_fwd_h` 循环体（`chunk_delta_h.py`） |
| `chunk_gla_fwd_o_gk` | `kda.py` |

`intra` 再拆：对角子块 (`BC=16`) → `inter_solve_fused`（off-diagonal + `solve_tril` + 可选 fused 写 `w/u/kg`）。

### 8.4 对照小结

```text
Decode:
  token_t + (conv_state, h)  --单步-->  o_t + (conv_state', h')
  调用次数 ∝ 生成长度；每次 T=1

Prefill naive:
  for t in 1..T: 同上          # 语义清晰，带宽/启动开销炸

Prefill chunk:
  先整段 conv 得到 [T] 的 qkv
  for chunk in chunks:         # 约 ceil(T/64) 次块间串行
      块内并行/矩阵化算出 BT 个输出，并得到块末 h
  最后写回 h_final
```

**一句话：decode 是“来一个更新一次”；prefill chunk 是“同一递推，按 64 个一组打包算，块间传 `h`”，结果应与逐步递推一致，只是更快。**

Speculative / target_verify 另说：多 token 草案路径会 checkpoint 中间 `h`，且常 `disable_state_update`，接受后再提交——和普通 prefill/decode 的“提交式写回”不同。

backend 选择由 `--linear-attn-*-backend` 一类 flag 决定（triton / helion / flashinfer / cutedsl …）。**数学是同一套 KDA，实现换成不同 kernel。**

---

## 9. 一张图串起来

```text
token x
  │
  ├─► qkv / beta / forget / out_gate 投影     ← Radix 外
  │
  ▼
mixed_qkv ──► short causal conv (stateful) ──► q, k, v   ← Radix/backend 内
a=forget ─┐
b=beta   ─┼─► g = A·Δ = -e^{A_log} * softplus(a+dt_bias)  # g≤0，log 域
          │   β = σ(b)
          ▼
     h ← exp(g)⊙h          # 这里的 exp 把 g 变成 (0,1] 乘子
     v ← β ⊙ (v - h k)
     h ← h + v kᵀ
     o ← h q
          │
          ▼
   RMSNormGated(o, out_gate) → o_proj → 残差/mHC   ← Radix 外
```

---

## 10. 读源码顺序建议

1. `Glm5NextLinearAttention.forward`：看清投影（Radix 外）和 `a/b/mixed_qkv`。
2. `RadixLinearAttention.forward`：确认只是 backend dispatch。
3. `kda_backend.forward_decode`：看 `causal_conv1d_update` + packed/non-packed。
4. `fused_sigmoid_gating_recurrent.py`：对着 `g / exp(g) / h/v/o` 读一遍。
5. `fla/kda.py` 的 `chunk_kda`：理解 prefill 如何把同一递推 chunk 化。

---

## 11. 一句话总结

`RadixLinearAttention` 是 SGLang 给 linear / KDA / GDN 共用的层接口；在 GLM-5.3-Flash 上，外层投影造出 `mixed_qkv/forget/beta/out_gate`，Radix/KDA backend 用 short conv（`conv_state` 可 cache）+ gated delta rule（`g=A·Δ` 再 `h*=exp(g)`）维护固定大小矩阵状态 `h`，最后 gated RMSNorm + `o_proj` 写回 hidden，从而在多数层用线性复杂度注意力替代完整 Softmax/MLA KV cache。