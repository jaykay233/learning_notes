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

## 6. 最小实现骨架与消融顺序

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

## 7. 检查清单（读别人的 looped 模型时问这七句）

1. 首/中/尾各几层，中间段跑几轮？→ 算出 24 / 48 这类数
2. 中间段权重在哪几轮共享？norm / 位置编码是否也共享？
3. 轮间有没有耦合？没有的话它只是「变相更贵的前向」
4. 耦合是回注、串联还是注意力式？gate 的输入里有没有上一轮的量？
5. 残差是纯加法还是带深度选择（AttnRes / 跨层注意力）？
6. 稳定器有哪几个？缺哪个最可能炸？
7. 轮数固定还是可调？固定轮数时，test-time compute 的旋钮是关着的

## 8. 参考

- Universal Transformer（Dehghani et al., 2018）：ACT 动态深度
- Deep Equilibrium Models（Bai et al., 2019）：不动点 + 隐式微分
- ALBERT（Lan et al., 2019）：跨层参数共享的早期工程验证
- recurrent-depth / looped LM 一脉（Ouro / LoopLM 等）：把循环深度用在生成模型上
