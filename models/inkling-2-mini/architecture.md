# Inkling 2 Mini (26.4B) 架构笔记：looped 主干

> **依据**：一张架构图（水印 Sebastian Raschka，属其 LLM architecture gallery 风格），逐框解读 + 用图右侧三个参数数字反推。
>
> **注意**：本仓其他 `models/` 笔记都对照了开源源码（SGLang / Transformers）。这一篇**没有源码和权重可查**，凡是标「推断」的结论都是从 26.4B / 3.4B / 5.4B 三个数字反解出来的，尚未与官方技术报告核对。图上直接写着的数值视为事实。

配套：

- [looped-transformer.md](looped-transformer.md)：把 looped 这个机制单拎出来讲，含族谱、设计空间与消融顺序
- [code/mini_looped.py](code/mini_looped.py)：同骨架的可运行 numpy 实现（小尺寸，验证 24 份权重 / 48 次应用的记账）

## 一眼看懂

```text
token → token embedding (2304)
      → [prelude 1–6]            ┐
      → [core  7–18]×3  ←──┐     │ 24 份独立权重
      → [coda  19–24]      │     │ 48 次层应用
      → Final Block AttnRes│     │
      → Final RMSNorm      │     ┘
      → Linear (200,058)   │
      → logit softcap 30   │
                           │
      prev-pass block 输出 ─┘（gate 注入，第 2、3 个 pass）
```

整个模型只有一个骨架级事实：**中间的 12 层参数只有一份，被跑 3 遍**。其余（GQA、MoE、conv、各种 clip / softcap）都是「单个 block 内部怎么做得更省更稳」，属于可插拔的部件。

## 1. 分段预算：24 份权重，48 次应用

| 区段 | 层号 | 参数 | 跑几次 | 累计应用 |
|---|---|---|---|---|
| prelude | 1–6 | 6 份，各自独立 | 1 | 6 |
| **core** | **7–18** | **12 份，三趟共用** | **3** | 6 + 36 = 42 |
| coda | 19–24 | 6 份，各自独立 | 1 | 48 |

\[
N_{\text{distinct}} = 6 + 12 + 6 = 24,\qquad
N_{\text{applications}} = 6 + 12\times 3 + 6 = 48
\]

两条容易看漏的连线：

- 图上 **「Layers before the loop 1–6 all feed all three passes」**：prelude 的输出 \(h_0\) 是三个 pass 的**同一个起点**，不是第 2 个 pass 接着第 1 个 pass 的输出往下走。
- 图上 **「Layers after the loop 19–24 run once after pass 3」**：coda 只吃最后一轮的输出。

即 \(h^{(p)}_{0} = h_0,\ p=1,2,3\)，三个 pass 是**并行同源的三次精炼**，不是三段串联的深网络。

## 2. 尺寸常量（照图抄，可直接当代码常量）

| 符号 | 值 | 备注 |
|---|---|---|
| `d_model` | 2,304 | embedding width / 残差流宽度 |
| `vocab` | 200,058 | |
| `n_q_heads` | 24 | head_dim = 2304 / 24 = **96** |
| `n_kv_heads` | 8 | GQA 分组 24:8 = **3:1** |
| `window` | 1,024 | local attention 窗口 |
| `context` | 1,000,000 | native 1M |
| `local:global` | 5 : 1 | 24 层 → **20 local + 4 global** |
| `n_experts` | 128 | |
| `top_k` | 8 | 每 token 激活 8 个，**0 shared** |
| `d_ff` (expert) | 1,152 | SwiGLU 中间维 |
| `conv_k` | 4 | short conv 核长 |
| `logit_softcap` | 30 | 输出层 |

## 3. 状态：不止 hidden

这个网络里真正在流动的东西有三条，混淆这三条基本就读不懂图了：

| 状态 | 内容 | 谁写 | 谁读 |
|---|---|---|---|
| `h` | 当前层/当前 pass 的 hidden | 每层 | 下一层 |
| `srcs` | **深度源**：token embedding + 历史 block 输出 | 每层写完 append | AttnRes（每层 + Final） |
| `prev[i]` | **上一 pass 第 i 层的输出** | core 第 i 层 | 同一层在下一个 pass 的 gate |

`srcs` 对应图左侧 `Depth sources`（`Token embeddings` + `Earlier block sums`）；`prev` 对应橙色箭头 `Gated message on passes 2–3`。

## 4. 前向骨架

```python
def forward(tok):
    x0 = embed(tok)                     # (B, T, 2304)
    srcs = [x0]                         # 深度源 #0

    # prelude：1–6，各自权重，跑一次
    for layer in prelude:
        x = layer(x, srcs)
        srcs.append(x)
    h0 = x                              # 三个 pass 的共同起点

    # core：7–18，12 份权重，跑 3 遍
    prev = [None] * N_CORE
    for p in range(3):
        h = h0
        for i in range(N_CORE):
            h = core[i](h, prev[i], srcs, p)
            prev[i] = h                 # 写给下一次 pass
            srcs.append(h)

    # coda：19–24，跑一次
    for layer in coda:
        x = layer(x, srcs)

    x = final_attn_res(x, srcs)         # 和全部深度源再融合一次
    x = final_rmsnorm(x)
    logits = lm_head(x)                 # → 200,058
    return 30.0 * np.tanh(logits / 30.0)  # logit softcap
```

## 5. core 的一层（7–18）拆解

从下往上，图上画的是：

```text
h ──► RMSNorm 1 ──► GQA(local/global, QK L2 clip, V norm, rel-pos bias)
                    └─► short conv (k=4) ──► ⊕ (gate, W/g)
   ──► RMSNorm 2 ──► MoE(SwiGLU, top-8/128)
                    └─► short conv (k=4) ──► ⊕ (gate, W/g)
```

写成伪代码，并标出**哪一行是 looped 特有的**：

```python
def core_layer(h, msg, srcs, p):
    # ① 跨 pass 门控注入：只有 p>=1 才有 msg        ← looped 特有
    if msg is not None:
        h = h + sigmoid(W_msg @ [h, msg]) * msg

    # ② 深度残差：在“深度轴”上做注意力，读回历史 block
    h = h + attn_res(h, srcs, split_qkv=is_global_layer)

    # ③ 注意力子层
    u = short_conv(rmsnorm1(h), k=4)                 # 先做局部混合
    u = gqa(u, window=(1024 if is_local else 1_000_000),
            qk_l2_clip=True, v_norm=True, rel_pos_bias=True)
    h = h + sigmoid(W1 @ h) * u                      # 门控残差（图上的 W / g）

    # ④ FFN 子层
    v = moe_swiglu(rmsnorm2(h))                      # router → top-8/128
    h = h + sigmoid(W2 @ h) * v
    return h
```

**只有 ① 和 `srcs` 的用法是 looped 架构带来的**；③④ 是任何现代 block 都能有的东西。

### 5.1 跨 pass 门控消息（全图最重要的一根箭头）

图上原文：*On later passes, each layer gates the previous pass's block outputs.*

\[
h^{(p)}_i \leftarrow h^{(p)}_i + \sigma\big(W [\,h^{(p)}_i;\, h^{(p-1)}_i\,]\big)\odot h^{(p-1)}_i
\]

没有这一项，三个 pass 就是把同一个函数套三遍，等价于「一个 12 层网络重复前向」——表达力上并没有变深。加了这一项，第 2、3 遍变成**对上一遍表示的增量修正**，整组 core 层退化成一个小型 RNN cell 的展开，深度才真正被"用起来"。

它也解释了为什么图上把第 2、3 个 pass 画成带橙色箭头的回边，而不是三条平行的箭头。

### 5.2 深度残差 AttnRes

图上左侧：

- `Depth sources`：`Token embeddings`、`Earlier block sums`
- `4 AttnRes blocks, 6 layers per block`（4 × 6 = 24 层全覆盖）
- `Global layers mix Q, K, and V inputs separately`

含义是：残差流不只是 \(h \leftarrow h + f(h)\) 的累加，而是每层在**深度轴**上做一次注意力，自己决定从「哪些历史层」读回多少：

\[
h \leftarrow h + \mathrm{softmax}_j\big(w(h)_j\big)\sum_j \alpha_j\, s_j ,\qquad s_j \in \texttt{srcs}
\]

每个 AttnRes block 覆盖 6 层、共享一套读权重；global 层因为要跨 1M 上下文，Q/K/V 三个输入分别混合（而不是只混一个 stream）。

对循环结构来说这是**必需件**而不是装饰：core 每转一圈表示都会漂移，如果没有一条随时能回看原始 embedding 和早期 block 的通路，几轮之后早期信息就被残差稀释掉了。它相当于给循环网络留了一条「回到起点再问一次」的查询通道。

### 5.3 注意力子层

- **GQA** 24 Q heads / 8 KV heads，head_dim 96；L2 clip 作用在 q、k 的范数上，V 额外做 norm。
- **local : global = 5 : 1**：24 层里 20 层是 1024 窗口的 local，只有 4 层是 global。这是 1M 上下文能在显存里活下来的关键（见 §7）。
- **learned relative-position bias**：给 local attention 一张与相对距离相关的可学偏置表。
- **short conv (k=4)**：在注意力 / FFN 输出侧各挂一条极窄的深度可分离卷积，做窗口内的局部混合；比全注意力便宜得多，适合 1M 上下文里"近处信息必须精确、远处可以稀疏"的分布。

### 5.4 MoE 子层

- 128 个专家，每 token **top-8**，**0 个 shared 专家**；
- 单专家 SwiGLU，中间维 1,152，gate 上再加一层 softcap（`Softcapped SiLU ⊗ Linear + softcap`），防止门控饱和后梯度消失；
- Router 在多专家扇出的**底部**（图上 MoE 框里最下面那个方块），即先路由再算。

## 6. 参数账反推：MoE 铺满 24 层（推断）

图上给了三个数：`26.4B total`、`3.4B unique active parameters per token`、`5.4B weight activations per token`。用它们可以反推 MoE 铺在哪。

单专家参数量（SwiGLU 三矩阵）：

\[
3 \times 2304 \times 1152 \approx 7.96\text{M}
\]

假设 MoE 铺满全部 24 层：

| 项 | 计算 | 结果 |
|---|---|---|
| 24 层专家总参数 | \(24 \times 128 \times 7.96\text{M}\) | ≈ 24.5B |
| 稠密部分（embed + attn + head + AttnRes…） | 26.4 − 24.5 | ≈ 1.9B |
| 活跃参数 | \(1.9 + 24 \times \frac{8}{128} \times 1.02\) | ≈ **3.47B** ✓ |

活跃参数与图上的 3.4B 吻合（误差 2%），且 24 层 × 128 专家恰好吃掉 93% 的参数预算。**反过来说**：如果 MoE 只铺在中间 12 层，稠密部分会膨胀到 ~14B、活跃参数会到十几 B，与 3.4B 差一个数量级。

所以：**图上把 MoE 只画在 core 框里，是放大展示，不代表只有 core 有 MoE。**

第三个数 `5.4B weight activations` 对不上唯一口径，两种记法各落一边：

| 口径 | 含义 | 结果 |
|---|---|---|
| A | 稠密权重只计一次，专家权重按 48 次应用计 | ≈ 5.0B |
| B | 稠密权重也按 48 次应用计 | ≈ 6.9B |

图上的 5.4B 落在两者之间的近端。所以只能拿到**定性**结论：多出来的那部分，就是**中间 12 层的权重被重复读了两次**；官方具体怎么记账（是否含 gate/conv/norm、dense 部分是否也摊销到每个 pass）图上没说，别拿 5.4 去卡第三位小数。跑 `code/mini_looped.py --real` 会把这两个口径一起打出来。

## 7. 推理侧的四个后果

**① prefill 算力按 48 层付，权重按 24 层存。** 这是典型的「用算力换参数」：同等参数预算下表达力更强，代价是 FLOPs 直接 ×2（相对同参数、24 次应用的模型）。

**② KV cache 只按 24 层存，但中间 12 层的 KV 每个 pass 都要重算。** 三个 pass 的 hidden 不同，KV 不能复用；所以 KV 显存 = 24 层的量，KV 计算 = 36 层的量。prefix cache 也省不掉这部分——它省的是跨请求的重复计算，省不了同一次前向内部的三个 pass。

**③ 1M 上下文靠「只有 4 层 global」撑住。** 按 fp16、8 KV heads、head_dim 96 估：

- global 层 KV：\(4 \times 10^6 \times 8 \times 96 \times 2(\text{K,V}) \times 2\text{B} \approx 12\text{ GB}\)
- local 20 层 KV：\(20 \times 1024 \times 8 \times 96 \times 2 \times 2\text{B} \approx 63\text{ MB}\)

即 1M 上下文时**几乎全部 KV 显存都花在那 4 层 global 上**（local 只占不到 1%）。这是「5:1 local/global」真正的用意所在。

**④ 三个 pass 串行，流水线并行会被打薄。** 单个 token 的前向是一条 48 段的长链，中间 12 段还要重复走，PP 的 stage 划分和 bubble 都会比同规模普通模型更难排。

## 8. 易错点

- **「等效 48 层」≠ 48 层网络。** 是 24 份权重、48 次应用；参数效率与算力成本要分开算。
- **三个 pass 不是串联的深网络。** prelude 输出同时喂三个 pass（图上明写），第 2 个 pass 不是从第 1 个 pass 的 coda 接着往下走。
- **AttnRes 不是普通残差。** 它在深度方向上做 softmax 选择，能跨层读 `srcs`；和「残差相加」不是一回事。
- **MoE 不止在 core。** 见 §6 的反推。
- **`5 local : 1 global` 描述的是注意力范围，"GQA 3:1" 描述的是头分组。** 两个比例都出现在这张图上，别混。
- **图上那些 clip / softcap / norm 不是表达力来源，是可收敛性来源。** 换成普通 24 层网络可以省掉大半；在循环结构里它们是必需品（见 [looped-transformer.md §4](looped-transformer.md)）。

## 9. 待核实

- 官方技术报告 / 权重是否公开，prelude 与 coda 是否也用 top-8 MoE；
- `12 × per pass` 标注的确切所指（12 层 core 每 pass 走一遍，还是 12 只在某个三维度重复）；
- AttnRes 的 `4 blocks × 6 layers` 里 global / local 的读权重是否真的不共享；
- short conv 的确切位置（图中画在子层输出侧，是否等同于「注意力前的 QKV 卷积」需看源码）。
