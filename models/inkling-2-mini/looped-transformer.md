# Looped Transformer（循环深度 / 层复用）笔记

> 这篇只讲机制，不绑某个具体模型。用例：Inkling 2 Mini（见 [architecture.md](architecture.md)）。
> 机制层面的结论有公开工作可对照（Universal Transformer、DEQ、Ouro/LoopLM 等）；具体到某个模型的设计选择，标「推断」的未与官方文档核对。

## 1. 定义：一句话

**只保留 \(N\) 份层权重，但把其中一段连续层在同一个 token 上跑 \(K\) 遍。**

\[
N_{\text{distinct}} = N_{\text{pre}} + N_{\text{core}} + N_{\text{coda}},\qquad
N_{\text{applications}} = N_{\text{pre}} + K \cdot N_{\text{core}} + N_{\text{coda}}
\]

普通 Transformer 是 \(N_{\text{applications}} = N_{\text{distinct}}\)；looped 把这两者解耦，于是「深度」变成一个可以单独调的旋钮，而参数量不跟着涨。

### 与几个名字的边界

| 说法 | 指什么 | 与本文关系 |
|---|---|---|
| **layer sharing / 层复用** | ALBERT 式：所有层共用一套参数 | \(N_{\text{core}} = N,\ K = N\) 的极端情形 |
| **looped / recurrent depth** | 一个 block 段重复 K 次，可带跨轮耦合 | 本文主体 |
| **Universal Transformer (ACT)** | 每步同一层 + 动态停机 | 是 looped 的一种，多一个「什么时候停」的控制器 |
| **DEQ（deep equilibrium）** | 循环到不动点、隐式反传 | looped 的极限形式：\(K \to \infty\) 且用不动点代替展开 |
| **looped 的「隐式推理」叙事** | 多轮 = 潜空间里多做几步计算 | 解释动机，不是架构约束 |

## 2. 族谱：五种「循环」的差别

| 工作 | 循环什么 | 轮间耦合 | 停不停 | 反传 |
|---|---|---|---|---|
| ALBERT | 整段层 | 无（纯重复） | 固定 | 展开 BPTT |
| Universal Transformer | 单层 | 有（每步同一函数） | **ACT 动态** | 展开 |
| DEQ | 整段层 | 有 | 求**不动点** | 隐式微分 |
| Ouro / LoopLM 一类 | 中间若干层 | 有 | 固定或学习 | 展开 |
| **Inkling 2 Mini**（推断） | 中间 12 层 ×3 | **门控注入上轮 block 输出** | 固定 3 轮 | 展开 |

一句话概括分野：**UT/DEQ 是「反复思考直到收敛」，Ouro/Inkling 这类是「同一组层走三趟，每趟带上一趟的草稿」。** 前者像迭代求根，后者像改稿。

## 3. 设计空间：绕不开的四个问题

任何 looped 架构都要回答这四个，缺一个就退化：

**Q1 循环哪一段？**
常见是「首尾各留几层独立、中间循环」。首尾独立的价值：入口处做 token → 表示的提升（不同层做不同的事），出口处做任务特定的收敛。中间循环段才是「反复精炼」的部分。

**Q2 循环里共享什么？**
- 权重（必须）
- norm 参数（通常共享）
- KV cache（**不**共享：每轮 hidden 不同，KV 必须重算）
- 位置编码（通常注入一次，循环轮次不重新编码位置）

**Q3 轮与轮之间怎么耦合？** ← **最关键**

三种做法，表达力递增：

1. **完全不耦合**：\(h^{(p)}_0 = h_0\)。等价于把同一个函数套 K 遍，没有真正的"更深"。
2. **串联耦合**：\(h^{(p)}_0 = h^{(p-1)}_{\text{last}}\)。像深网络，但会漂移、难训。
3. **门控回注**（Inkling 走这条）：每个 pass 的每层都能拿到**上一 pass 同层**的输出，用 gate 融合：

\[
h \leftarrow h + \sigma(W[h;\ m])\odot m,\qquad m = h^{(p-1)}_i
\]

这让「上一轮的想法」和「这一轮的输入」在同一层被比较、被加权——整组层从「重复的函数」变成「展开的 RNN cell」。

**Q4 怎么保证不发散？**
见 §4。这是 looped 架构真正的工程难点。

## 4. 为什么必须堆稳定器

同一个函数被反复施加，最怕两件事：

- **发散**：范数逐轮放大。残差流本身就在累加，再叠 K 轮，logits 很快爆掉。
- **塌缩**：多轮之后表示挤到低维子空间，\(h^{(1)} \approx h^{(2)} \approx h^{(3)}\)，白跑。

所以这类模型普遍带一整套「数值保险」，它们的共同作用是**给每一轮的增益加一个上界**：

| 手段 | 作用对象 | 防什么 |
|---|---|---|
| QK L2 clip（裁 q、k 的范数） | attention logits | logits 爆炸 → softmax 饱和 → 梯度消失 |
| V norm | attention 输出 | 单 token 值域失控 |
| RMSNorm（每子层前） | hidden | 逐轮范数漂移 |
| logit softcap（\(c\tanh(z/c)\)） | 输出 logits | 训练早期大 logit 主导梯度 |
| 门控残差 \(h+\sigma(W h)\odot f(h)\) | 子层增益 | 让模型自己学会「这一层要不要用力」 |
| SwiGLU gate softcap | MoE 中间激活 | 专家门控饱和 |

**判断标准**：把这些删掉后能否训练成功，是 looped 与普通 Transformer 最实际的分水岭。普通 24 层网络通常能省掉一半；循环结构里它们几乎都是必需品。

## 5. 成本模型：省参数、费算力、不省 KV

设普通模型有 \(L\) 层、每层活跃权重 \(w\)、每 token FLOPs \(f\)：

| 量 | 普通 \(L\) 层 | looped（\(N_{\text{core}}\) 层 × K，其余各一次） |
|---|---|---|
| 权重显存 | \(L\cdot w\) | \((N_{\text{pre}} + N_{\text{core}} + N_{\text{coda}})\cdot w\) |
| prefill FLOPs | \(L\cdot f\) | \((N_{\text{pre}} + K N_{\text{core}} + N_{\text{coda}})\cdot f\) |
| 权重读取（带宽） | \(L\cdot w_{\text{active}}\) | 同上按应用次数 |
| KV 显存 | \(L\) 层 KV | **仍按 distinct 层数**，不乘 K |
| KV 计算 | \(L\) 层 | 按应用次数（中间段每轮重算） |
| 串行链长 | \(L\) | 应用次数（PP 更难排） |

结论：**looped 严格是「参数 ↔ 算力」的兑换**，只有在参数/显存是瓶颈、算力相对富裕时才划算。它不会省 KV 显存，也不会缩短串行链。

一个推论：looped 模型天然适合「参数预算卡死」的场合（端侧、单卡、显存受限的 serving），不适合「算力卡死」的场合。

## 6. AttnRes：把残差从「累加」改成「按深度选择」

普通残差 \(h_i = h_{i-1} + f_i(h_{i-1})\) 是**加法累积**：信息只能沿深度一层层往前挤，深度越深，早期信号被稀释得越狠。循环结构把这个缺陷放大 K 倍——每转一圈再稀释一遍。

AttnRes 的做法：维护一个深度源的集合 `srcs`（token embedding + 每个 block 的输出），每层在**深度轴**上做一次 softmax，自己决定从哪些源读回多少。

\[
\alpha_j = \operatorname{softmax}_j\big(\mathrm{score}(h, s_j)\big),\qquad
\mathrm{out} = \sum_j \alpha_j\, s_j
\]

可运行验证见 [code/attn_res_demo.py](code/attn_res_demo.py)（三种打分方式 + bank 记账 + 凸组合范数验证）。

### 6.1 三种打分方式

| 变体 | score | 参数量 | 用在哪 |
|---|---|---|---|
| **A 无 query** | \(w^\top\mathrm{RMSNorm}(s_j)\)，只依赖源自己 | \(H\) | **Kimi-K3**（`score_proj: H→1` + `score_norm`） |
| **B 带 query** | \(\langle hW_q,\ s_j\rangle/\sqrt{H}\)，当前 hidden 做 query | \(H\times H\) | 直觉版；`mini_looped.py` 用的是简化形 |
| **C per-stream** | 每个 stream（Q/K/V / 每 head）一套 \(w\) | \(n_{\text{stream}}\times H\) | **global 层**，对应图上 `Global layers mix Q, K, and V inputs separately` |

A 与 B 的差别值得记一句：**B 能表达「我这一层现在想找什么」，A 只能表达「哪个源本身更值得读」**。A 参数少一个数量级，且 score 侧可压成一次点积（`cw = score_norm.weight ⊙ score_proj.weight`），推理更便宜——Kimi-K3 选 A 是工程取舍。

`mini_looped.py` 里最短的实现（B 的简化形，query 是每 token 一个标量）：

```python
def attn_res(h, srcs, p):
    S = h @ p["ar_w"]                                   # (T,1) 每 token 一个标量
    s = np.concatenate(srcs, axis=0).reshape(len(srcs), *h.shape)
    scores = np.einsum("std,t->st", s, S[:, 0])         # (S,T)：每个源对每个 token 的分
    scores -= scores.max(0, keepdims=True)
    a = np.exp(scores); a /= a.sum(0, keepdims=True)    # 沿深度轴 softmax
    return np.einsum("st,std->td", a, s)                # 凸组合
```

四个要点：

1. **softmax 沿深度轴做，不是沿 token 轴。** 它不在时间上混合，在层与层之间混合；每个 token 独立算自己的 \(\alpha\)。这和 self-attention 正好互补。
2. **输出是源的凸组合**，\(\alpha_j \ge 0,\ \sum_j \alpha_j = 1\)，所以 \(\|\mathrm{out}\| \le \max_j \|s_j\|\)。范数不会随深度放大——这是它能在循环结构里稳住的原因之一（demo 里两项都验证了）。
3. **反传多了一条「深度高速公路」。** \(\partial\,\mathrm{out}/\partial s_j = \alpha_j > 0\)，梯度直接落到任意历史源，不必穿过中间几十层。**AttnRes 买的主要不是表达力，是深度方向的梯度路径。**
4. **`srcs` 的存储不能省**，每层 append 一次是 \(O(L\cdot T\cdot H)\)。Kimi-K3 用 bank 把这份开销压到 \(O(\frac{L}{12}\cdot T\cdot H)\)，见下。

### 6.2 Kimi-K3 的 bank 记账（怎么把写放大压到 1/12）

朴素做法每层都要存一份 `[T,H]` 快照，24 层就是 24 份。Kimi-K3 的 `AttnResidual` 改成：

| 量 | 形状 | 含义 |
|---|---|---|
| `bank` | `[T, NB, H]` | 冻结的 snapshot，**每 12 层才写一行** |
| `nvb` | scalar | 已写入行数（softmax 只在 `[:nvb]` 上做） |
| `prefix_sum` | `[T,H]` 或 `None` | 当前 block 内还没入 bank 的 running prefix |

```python
def aggregate(prefix, bank, nvb, w):
    if nvb == 0:
        return prefix, None                      # 第 0 层：bank 还空，直通
    rows = cat(bank[:, :nvb, :], prefix[:, None, :], dim=1)   # bank 行 + 当前 prefix 一起竞争
    a = softmax(rms_norm(rows) @ w, dim=1)        # 沿 bank 行轴
    return (a[:, :, None] * rows).sum(1), a
```

关键设计有两条：

- **当前 `prefix` 也参与 softmax**，与冻结的 bank 行竞争。否则 block 中间的那些层就"看不见"自己了。
- **只有在 `layer_idx % block_size == 0` 时才写 bank**，其余层的读取发生在 `prefix` 上，不新增快照。`mini_looped`（玩具）里把 12 压成 3，跑出来是：

```
layer  0: 写入 bank 第 0 行，nvb=1，alpha 形状 (退化为直通)
layer  3: 写入 bank 第 1 行，nvb=2，alpha 形状 (2, 3)
layer  6: 写入 bank 第 2 行，nvb=3，alpha 形状 (3, 3)
layer  9: 写入 bank 第 3 行，nvb=4，alpha 形状 (4, 3)
```

Inkling 图上写 `4 AttnRes blocks, 6 layers per block`，就是同一手法：24 层 / 6 = 4 行 bank。

### 6.3 两个必须做对的细节

- **`srcs` 必须含 token embedding。** 不含的话模型断了「回到输入」的通道，循环几轮后原始 token 信息就彻底没了。图上左侧 `Depth sources` 第一项正是 `Token embeddings`。
- **打分前要归一（RMSNorm 或减 max）。** 深度源的量级差异很大（embedding 和深层 hidden 不是一个尺度）。demo 里量化过：不做 norm 时 \(\alpha\) 的熵会掉到 0.29（几乎只读一个源，退化成普通残差），做了之后回到 1.19——softmax 不再饱和。Kimi-K3 挂 `score_norm` 就是为这个。

## 7. 最小实现骨架与消融顺序

可运行的玩具实现见 [code/mini_looped.py](code/mini_looped.py)（numpy，无依赖，打印 distinct / applications / 参数量三本账）。

骨架（与 [architecture.md §4](architecture.md) 同构，去掉 MoE 与 GQA 的细节）：

```python
def forward(x):
    srcs = [x]
    for layer in prelude:            # 一次
        x = layer(x, srcs); srcs.append(x)
    h0 = x                            # 三个 pass 同一起点

    prev = [None] * len(core)
    for p in range(K):
        h = h0
        for i, layer in enumerate(core):   # 同一批权重，跑 K 遍
            h = layer(h, prev[i], srcs, p) # ① 门控回注 + ② 深度残差
            prev[i] = h; srcs.append(h)

    for layer in coda:                # 一次
        x = layer(x, srcs)
    return head(final_attn_res(x, srcs))
```

**学习顺序**（每一步只引入一个变量，才能看清每根箭头为什么存在）：

1. 先写 12 层重复 3 次、**不加**任何轮间耦合与 clip 的版本，跑小语料 → 观察 loss 抖动/发散，理解 §4 的必要性；
2. 只加 ①「跨 pass 门控回注」，再跑 → 对比收敛速度与最终 loss，理解 §3-Q3；
3. 只加 `srcs` 深度残差（先退化成加权求和，不做到 softmax），看长依赖任务的变化；
4. 最后加 MoE 与 GQA——它们与 looped 正交，是「同一骨架上的效率插件」，可以独立评估。

## 8. 检查清单（读别人的 looped 模型时问这七句）

1. 首/中/尾各几层，中间段跑几轮？→ 算出 24 / 48 这类数
2. 中间段权重在哪几轮共享？norm / 位置编码是否也共享？
3. 轮间有没有耦合？没有的话它只是「变相更贵的前向」
4. 耦合是回注、串联还是注意力式？gate 的输入里有没有上一轮的量？
5. 残差是纯加法还是带深度选择（AttnRes / 跨层注意力）？
6. 稳定器有哪几个？缺哪个最可能炸？
7. 轮数固定还是可调？固定轮数时，test-time compute 的旋钮是关着的

## 9. 参考

- Universal Transformer（Dehghani et al., 2018）：ACT 动态深度
- Deep Equilibrium Models（Bai et al., 2019）：不动点 + 隐式微分
- ALBERT（Lan et al., 2019）：跨层参数共享的早期工程验证
- recurrent-depth / looped LM 一脉（Ouro / LoopLM 等）：把循环深度用在生成模型上
