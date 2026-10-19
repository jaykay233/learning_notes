# FlashKDA：从 Delta Rule 到 K1/K2 数据流

> 本笔记对应本机 `/Users/saboxu/Downloads/FlashKDA` 的源码快照 `7afb9f4`（2026-10-19 阅读）。实现会变化；下面的 chunk 大小、数据类型和 warp 映射均以该快照为准。

## 本次讲解位置

```text
章节：算子开发 / 状态空间与线性注意力算子
小节：FlashKDA v1：chunkwise KDA 的 K1 prepare → K2 recurrence
知识点：KDA chunk 内的 L / INV / Mqk，以及跨 chunk recurrent state 如何推进
上次：13-flashmla-sparse-attention-reading-guide.md 的 sparse decode、split-KV 与 softmax merge
下次：沿 K2 MMA warp 的 16×16 tile 深挖 state update、寄存器转置和 TMA pipeline
PTX：不是单条 PTX 指令专题；实现重点是 SM80 16×8×16 BF16 MMA atom 与 SM90 TMA
```

## 为什么现在讲这个

前一节的 FlashMLA 可以把长 KV 分给多个 split，再合并 partial softmax；每个 split 主要是在一组给定 KV 上做 attention。FlashKDA 的困难不同：**序列的状态有时间依赖**，后一块必须接着前一块产生的 state 算，不能把所有 chunk 都当成彼此独立的 attention split。与此同时，chunk 内仍可把 token 维矩阵化并并行计算。

FlashKDA 因此把工作切成两种并行度：K1 让每个 chunk 独立预处理，尽量铺开 token 并行；K2 让每个 `(sequence, head)` CTA 按 chunk 顺序递推 state。理解这项切分，就能避免把它误读成 softmax attention 或“多个 chunk 各算一份最后取平均”，也能解释为什么它用了 TMA / warp specialization，却不需要 FlashMLA 那种 split-softmax 合并。

## 先建立心智模型

把 KDA 想成“每个 head 持有一张可更新的记忆矩阵”，而不是把所有历史 K/V token 永久堆在一个 cache 里：

```text
输入 token (q, k, v, gate, beta)
       │
       ├─ K1：每个 16-token chunk 独立做 gate/decay、构造小矩阵并求 INV
       │       token/chunk 并行；把中间结果写到 workspace
       ▼
       workspace ───────────────┐
                               ▼
初始 state ───────────→ K2：每个 (sequence, head) 按 chunk 串行递推
                               ├─ 用当前 state 算本 chunk 的输出
                               ├─ 更新 state，交给下一个 chunk
                               └─ 写 output / 可选 final_state
```

- **KDA（Kimi Delta Attention）属于 gated delta-rule / linear attention 路线。** 它不是对所有历史 token 做 softmax；长历史的信息由 state 汇总。
- 每个 head 的 state 形状是 `[V, K]`。当前 FlashKDA 限定 `K = V = 128`，所以是 `[128,128]`。
- 一个 CTA 处理 `CHUNK=16` 个连续 token。chunk 内先变成小的 16×16 下三角线性系统；chunk 与 chunk 之间仍然保留 recurrent 顺序。
- `beta` 输入是 sigmoid **之前的 logit**；kernel 内部做 sigmoid。`g` 输入也处于 gate activation 之前，门控还结合 `A_log`、`dt_bias` 和 `lower_bound`。

## API 与边界

仓库 README / `flash_kda/__init__.py` 的接口语义：

| 张量 | 形状 | 类型 / 语义 |
|---|---|---|
| `q, k, v` | `[B,T,H,128]` | BF16；Q/K 会做 L2 normalize |
| `g` | `[B,T,H,128]` | BF16，gate activation 前的输入 |
| `beta` | `[B,T,H]` | BF16 logits，内部 sigmoid |
| `A_log` | `[H]` | FP32 |
| `dt_bias` | `[H,128]` | FP32 |
| `out` | `[B,T,H,128]` | BF16，调用方提供的输出 buffer |
| `initial_state/final_state` | `[B,H,V,K]`，变长时 `[N,H,V,K]` | 可 BF16 / FP32；内部 state 运算驻留 BF16 |
| `cu_seqlens` | `[N+1]` | 可选；提供时要求 `B=1`，定义 N 个变长序列 |

当前代码还要求 CUDA contiguous 张量、CUDA 12.9+、PyTorch 2.4+ 和 SM90+；README 的编译目标例子为 `90a,100a`。**注意**：K1/K2 的 MMA 使用 SM80 风格的 MMA atom，不代表整套 kernel 可在 SM80 运行；当前实现依赖 SM90 TMA 路径。

## 逐 token 公式逐项拆解：从记忆矩阵到当前输出

### 本次讲解位置

```text
章节：算子开发 / 状态空间与线性注意力算子
小节：KDA 原始逐 token 递归
知识点：逐 Key 遗忘、预测残差、门控修正、秩一写入与 Query 读出
上次：K1 prepare → K2 recurrence 的整体数据流与公式图
下次：把逐 token 递归对应到 chunk 内的下三角系统
PTX：本节为数学数据流，不涉及单条 PTX 指令
```

### 为什么现在讲这个

只知道 K1 准备 workspace、K2 推进 state，还无法解释为什么需要严格下三角矩阵，或为什么当前输出包含当前 token 的贡献。先逐项拆开原始递归，明确“先遗忘、用 Key 预测、写入残差、用 Query 读新状态”的顺序，才能区分写入门控与 attention 权重，也能避免把状态的 Key 列衰减写成 Value 行衰减。

> 同步来源：[飞书中的逐 token 公式讲解](https://pcng0ddyhlxs.feishu.cn/wiki/VralwO1dmi4mzAk0Z9kcDdT3nzt#doxcnBrzXyx3EfNAHxjGWiExLvf)，2026-10-19，revision 116。以下图为飞书最终画板的导出版本；正文保留原生 LaTeX 公式。

![KDA 原始逐 token 递归：遗忘、检索与修正、状态写入与读出](assets/flashkda/kda-token-recurrence-latex.jpg)

这一节只解释上图中的原始逐 token 递归，不展开 chunk 并行和 CUDA 优化。核心问题是：当前 token 到来后，KDA 怎样遗忘旧信息、修正记忆，再从记忆里读出结果？

### 1. 先建立直觉：一张从 Key 空间映射到 Value 空间的记忆矩阵

普通 softmax attention 保留历史 Key、Value，让当前 Query 匹配它们。KDA 则把历史信息逐步汇总进固定大小的状态矩阵，而不是把历史 token 原封不动地存下来：

$$
S_{t-1}\in\mathbb{R}^{V\times K}
$$

暂时把它理解成一个从 Key 空间映射到 Value 空间的记忆系统：

$$
\underbrace{S_{t-1}}_{V\times K}\underbrace{k}_{K\times 1}=\underbrace{\text{记忆对这个 Key 的 Value 响应}}_{V\times 1}
$$

当前 token 带来一个新关联：$k_t\longleftrightarrow v_t$。KDA 希望让记忆对当前 Key 的响应更接近当前 Value。这张矩阵是压缩记忆，不是保证精确保存所有历史关联的字典。

### 2. 符号与维度：只看一个 sequence 的一个 head

省略 batch、head 下标；以下向量都按列向量理解。$K$ 和 $V$ 是维度大小，不是历史 Key、Value 矩阵。

| 符号 | 形状 | 含义 |
| --- | --- | --- |
| $t$ | — | 当前 token 的位置 |
| $q_t$ | $K\times 1$ | 当前 Query，用于最终读出记忆 |
| $k_t$ | $K\times 1$ | 当前 Key，用于检索和定位写入方向 |
| $v_t$ | $V\times 1$ | 当前 Value，希望记住的目标内容 |
| $S_{t-1}$ | $V\times K$ | 处理当前 token 之前的旧记忆 |
| $\boldsymbol{g}_t$ | $K\times 1$ | 已激活、处于 base-2 指数域的逐 Key 维衰减控制量 |
| $D_t$ | $K\times K$ | 由保留比例组成的对角矩阵 |
| $\bar{S}_t$ | $V\times K$ | 已衰减、尚未写入当前信息的记忆 |
| $p_t,e_t,u_t$ | $V\times 1$ | 分别是预测 Value、预测误差和实际修正向量 |
| $\beta_t$ | 标量 | 当前 token 的写入／修正强度 |
| $S_t$ | $V\times K$ | 处理当前 token 后的新记忆 |
| $\alpha$ | 标量 | Query 的缩放系数，对应 API 的 scale |
| $o_t$ | $V\times 1$ | 当前 token 的最终输出 |

### 3. 第一步：决定旧记忆的各个 Key 维度保留多少

$$
D_t=\operatorname{diag}\!\left(2^{\boldsymbol{g}_t}\right)
$$

先对门控向量逐元素取指数，得到每个 Key 维度的保留比例：

$$
d_{t,j}=2^{g_{t,j}}
$$

| $g_{t,j}$ | $d_{t,j}$ | 含义 |
| --- | --- | --- |
| $0$ | $1$ | 完全保留 |
| $-1$ | $0.5$ | 保留一半 |
| $-2$ | $0.25$ | 保留四分之一 |

**这里的门控量已经完成激活和 base-2 指数域转换，不是 API 直接收到的原始 g 输入。**在这条衰减路径中，激活后的门控量非正，保留比例因而不大于 1。自然指数写法与 base-2 写法可以换算，不能把不同指数域中的数值直接混用。

例如，两个 Key 维度的保留比例分别为 0.5、0.25，则 diag 把它们放在对角线上：

$$
2^{\boldsymbol{g}_t}=\begin{bmatrix}0.5\\0.25\end{bmatrix},\qquad D_t=\begin{bmatrix}0.5&0\\0&0.25\end{bmatrix}
$$

所以每个 Key 维度可以有不同的遗忘速度。

### 4. 第二步：先衰减旧记忆

$$
\bar{S}_t=S_{t-1}D_t
$$

$S_{t-1}$ 是旧记忆；$D_t$ 是当前 token 决定的保留比例；$\bar{S}_t$ 是已经衰减、尚未写入当前信息的记忆。上面的横线只是区分这个中间状态。

因为对角矩阵乘在右侧，所以它缩放状态矩阵的列，即 Key 维度：

$$
\bar{S}_t[i,j]=S_{t-1}[i,j]d_{t,j}
$$

$$
\begin{bmatrix}a&b\\c&d\end{bmatrix}\begin{bmatrix}0.5&0\\0&0.25\end{bmatrix}=\begin{bmatrix}0.5a&0.25b\\0.5c&0.25d\end{bmatrix}
$$

**在本文的状态布局约定下，这是逐 Key 列衰减，不是逐 Value 行衰减，也不是逐条删除历史 token。**若其他资料使用转置后的状态布局，衰减乘法的左右位置也会相应变化。

### 5. 第三步：用当前 Key 检查记忆已经知道什么

$$
p_t=\bar{S}_t k_t
$$

$p_t$ 是记忆对当前 Key 的预测 Value。这一步是在问：“拿当前 Key 查询这张记忆，它会给我什么 Value？”

$$
(V\times K)(K\times 1)=V\times 1,\qquad p_t[i]=\sum_{j=1}^{K}\bar{S}_t[i,j]k_t[j]
$$

**Key 用来检查当前要写入的关联是否已被记住；Query 用来产生最终输出。**因此这一步是写入前的检索／预测，还不是当前 token 的最终 attention 输出。

### 6. 第四步：计算记忆还缺少什么

$$
e_t=v_t-p_t
$$

$v_t$ 是当前想写入的目标；$p_t$ 是旧记忆已经预测出的内容；$e_t$ 是两者的误差，也叫残差。

$$
v_t=\begin{bmatrix}3\\2\end{bmatrix},\quad p_t=\begin{bmatrix}1\\0\end{bmatrix}\quad\Longrightarrow\quad e_t=\begin{bmatrix}2\\2\end{bmatrix}
$$

**KDA 写入的是修正量，而不是把整个 Value 再加一遍。**这就是 delta rule 的核心直觉：先预测，再根据预测误差修正记忆。

如果旧记忆已经预测正确，即 $p_t=v_t$，则 $e_t=0$，无需额外修正；但此前的状态衰减仍然已经发生。

### 7. 第五步：决定这次修正要做多大

$$
\beta_t=\sigma\!\left(\beta_{\mathrm{logit},t}\right),\qquad \sigma(x)=\frac{1}{1+\exp(-x)}
$$

sigmoid 把模型输出的原始 logit 转换为介于 0 和 1 之间的写入强度。随后：

$$
u_t=\beta_t e_t
$$

实际修正向量等于修正强度乘以预测误差。写入强度接近 0 时只做很小的修正；等于 0.5 时修正一半；接近 1 时做接近完整的修正。

**这是写入门控，不是对历史 token 做 softmax 得到的 attention 权重。**FlashKDA API 接收的是 beta logit，内部执行 sigmoid，不应把已经 sigmoid 的值再次当作 logit 传入。

### 8. 第六步：沿当前 Key 的方向修正记忆

$$
S_t=\bar{S}_t+u_t k_t^{\mathsf{T}}
$$

修正向量只有 Value 维度，但我们要更新一张矩阵，因此需要外积：

$$
\underbrace{u_t}_{V\times 1}\underbrace{k_t^{\mathsf{T}}}_{1\times K}=\underbrace{\Delta S_t}_{V\times K},\qquad \Delta S_t[i,j]=u_t[i]k_t[j]
$$

$$
u_t=\begin{bmatrix}a\\b\end{bmatrix},\quad k_t=\begin{bmatrix}c\\d\end{bmatrix}\quad\Longrightarrow\quad u_tk_t^{\mathsf{T}}=\begin{bmatrix}ac&ad\\bc&bd\end{bmatrix}
$$

$u_t$ 决定修正什么 Value 内容；$k_t$ 决定沿哪个 Key 方向写入。外积的秩最多为 1，所以图中称为“秩一状态写入”。

为什么这种写入能修正预测？更新后，再用同一个 Key 检索：

$$
S_tk_t=p_t+\beta_t(v_t-p_t)(k_t^{\mathsf{T}}k_t)
$$

若当前 Key 理想地归一化为单位向量，即 $k_t^{\mathsf{T}}k_t=1$，则：

$$
S_tk_t=p_t+\beta_t(v_t-p_t)=(1-\beta_t)p_t+\beta_t v_t
$$

**更新后的预测，是旧预测和新目标之间的一次插值。**写入强度为 0 时保持衰减后的旧预测；为 0.5 时向目标移动一半；为 1 时，理想单位 Key 条件下对当前 Key 的预测直接变成目标 Value。0 和 1 是理解门控的极限情形，不是有限 sigmoid logit 的精确输出。

实际实现的归一化含 epsilon 且使用有限精度，以上是理想数学解释。不同 Key 方向不一定互相正交，更新也可能影响其他 Query／Key 的响应，不能理解成一份互不干扰的精确字典。

### 9. 第七步：用当前 Query 从新记忆中读出结果

$$
o_t=S_t(\alpha q_t)
$$

$S_t$ 是已经写入当前 token 后的记忆；$q_t$ 是当前 Query；$\alpha$ 是缩放系数；$o_t$ 是最终输出。图中顺序是**先更新状态，再读出当前输出**，因此当前输出可以包含当前 token 刚写入的信息。

把状态更新代入读出式：

$$
o_t=\alpha\bar{S}_tq_t+\alpha u_t(k_t^{\mathsf{T}}q_t)
$$

$$
o_t=\underbrace{\alpha\bar{S}_tq_t}_{\text{从衰减后的历史记忆读取}}+\underbrace{\alpha\beta_t e_t(k_t^{\mathsf{T}}q_t)}_{\text{当前修正对输出的贡献}}
$$

这里的 $k_t^{\mathsf{T}}q_t$ 是熟悉的 Key–Query 匹配项：匹配程度决定当前修正会多大程度影响输出。但这里没有对所有历史 token 做 softmax。

### 10. 完整数值例子：衰减、预测、修正、读出

为方便手算，取 $K=V=2$、$\alpha=1$。这是说明公式的玩具例子，不是把当前 FlashKDA kernel 的维度限制改成了 2；实际实现仍限定 128 维，scale 也应按调用方设定。

$$
S_{t-1}=\begin{bmatrix}2&0\\0&4\end{bmatrix},\qquad k_t=q_t=\begin{bmatrix}1\\0\end{bmatrix},\qquad v_t=\begin{bmatrix}3\\2\end{bmatrix}
$$

$$
\boldsymbol{g}_t=\begin{bmatrix}-1\\-2\end{bmatrix},\qquad \beta_{\mathrm{logit},t}=0
$$

先得到保留比例和修正强度：

$$
D_t=\begin{bmatrix}0.5&0\\0&0.25\end{bmatrix},\qquad \beta_t=\sigma(0)=0.5
$$

| 步骤 | 计算 | 结果 |
| --- | --- | --- |
| 衰减 | $\bar{S}_t=S_{t-1}D_t$ | $\begin{bmatrix}1&0\\0&1\end{bmatrix}$ |
| 预测 | $p_t=\bar{S}_tk_t$ | $\begin{bmatrix}1\\0\end{bmatrix}$ |
| 误差 | $e_t=v_t-p_t$ | $\begin{bmatrix}2\\2\end{bmatrix}$ |
| 加权 | $u_t=0.5e_t$ | $\begin{bmatrix}1\\1\end{bmatrix}$ |
| 外积 | $u_tk_t^{\mathsf{T}}$ | $\begin{bmatrix}1&0\\1&0\end{bmatrix}$ |
| 更新 | $S_t=\bar{S}_t+u_tk_t^{\mathsf{T}}$ | $\begin{bmatrix}2&0\\1&1\end{bmatrix}$ |
| 读出 | $o_t=S_tq_t$ | $\begin{bmatrix}2\\1\end{bmatrix}$ |

旧预测是 $[1,0]^{\mathsf{T}}$，目标是 $[3,2]^{\mathsf{T}}$。修正强度为 0.5，更新后的预测刚好走到中间：$[2,1]^{\mathsf{T}}$。而当前 Key 是 $[1,0]^{\mathsf{T}}$，因此此次写入只修正状态矩阵第一列。

### 11. 把整套公式压缩成一句话

$$
S_t=S_{t-1}D_t+\beta_t\left(v_t-S_{t-1}D_tk_t\right)k_t^{\mathsf{T}}
$$

**新记忆 = 衰减后的旧记忆 + 沿当前 Key 方向写入的、经过门控的预测误差。**随后用当前 Query 从新记忆中读出结果：

$$
o_t=S_t(\alpha q_t)
$$

### 12. 三个自检题与答案

**问题 1：衰减门控和写入门控有什么区别？**

答案：$D_t$ 控制旧记忆如何遗忘；$\beta_t$ 控制当前预测误差的修正幅度。两者不能互相替代。

**问题 2：为什么不直接写入完整的 Value 外积？**

答案：KDA 要修正已有预测，写入的是 $\beta_t(v_t-p_t)k_t^{\mathsf{T}}$，不是 $\beta_t v_tk_t^{\mathsf{T}}$。如果目标已经被预测正确，当前修正项为零。

**问题 3：预测 Value 和最终输出有什么区别？**

答案：$p_t$ 用 Key 从衰减后的旧状态中预测 Value；$o_t$ 用 Query 从更新后的新状态中产生输出。它们的输入向量和读取的状态都不同。

## 数学：chunk 内如何变成矩阵乘法

以下把 gate 写成已经激活后的 $g_{\mathrm{act},t,j}$，使用 base-2 指数。源码中 gate 预处理等价于（省略 head 下标）：

$$
g_{\mathrm{act},t,j}=g_{\min}\log_2(e)\,\sigma\!\left(e^{A_{\mathrm{log}}}\left(g_{\mathrm{in},t,j}+b_{\mathrm{dt},j}\right)\right)
$$

其中 $g_{\min}$ 对应 API 的 `lower_bound`，$b_{\mathrm{dt},j}$ 对应 `dt_bias`。它是逐 token、逐 Key 维度计算，通常为非正数。于是 $2^{g_{\mathrm{act},t,j}}$ 表示 state 在 Key 维度上的保留比例。$G$ 表示 chunk 内沿 token 轴的前缀和，$C=16$：

$$
G_{t,j}=\sum_{\tau=0}^{t}g_{\mathrm{act},\tau,j},\qquad g_{\mathrm{total},j}=G_{C-1,j}
$$

以下乘法和指数都按元素计算，$\alpha$ 对应 API 的 `scale`，$K_{t,j}$、$Q_{t,j}$ 已做 L2 normalize：

$$
\begin{aligned}
(K_{\mathrm{decay}})_{t,j}&=K_{t,j}\,2^{G_{t,j}},\\
(Q_{\mathrm{decay}})_{t,j}&=\alpha Q_{t,j}\,2^{G_{t,j}},\\
(K_{\mathrm{inv}})_{t,j}&=K_{t,j}\,2^{-G_{t,j}},\\
(K_{\mathrm{restored}})_{t,j}&=(K_{\mathrm{inv}})_{t,j}\,2^{g_{\mathrm{total},j}}.
\end{aligned}
$$

其中 $K_{\mathrm{decay}}$、$Q_{\mathrm{decay}}$、$K_{\mathrm{inv}}$、$K_{\mathrm{restored}}$ 形状都是 $C\times K=16\times128$，$\boldsymbol g_{\mathrm{total}}$ 是 $K$ 维向量。$K_{\mathrm{restored}}$ 把 chunk 内逆衰减的 Key 换算到 chunk 末尾 state 的时间坐标。

接着构造两个 chunk 矩阵。定义 $\boldsymbol\beta=(\beta_0,\ldots,\beta_{C-1})^{\mathsf T}$，其中 $\beta_t=\sigma(\beta_{\mathrm{logit},t})$，左乘对角矩阵表示逐 token 行缩放，而非普通矩阵与向量相乘：

$$
\begin{aligned}
L&=\operatorname{diag}(\boldsymbol\beta)\,\operatorname{tril}\!\left(K_{\mathrm{decay}}K_{\mathrm{inv}}^{\mathsf T},-1\right),\\
M_{qk}&=\operatorname{tril}\!\left(Q_{\mathrm{decay}}K_{\mathrm{inv}}^{\mathsf T},0\right),\\
\mathrm{INV}&=(I+L)^{-1}.
\end{aligned}
$$

- $L$ 只保留**严格下三角**，因此 token $t$ 只依赖 chunk 内更早的 token；它编码了 delta update 的因果依赖。
- $\mathrm{INV}$ 把 chunk 内顺序依赖改写成一次小型三角系统求解：$U=\mathrm{INV}\,\widetilde V$。$I+L$ 是单位下三角矩阵，可以用 forward substitution 求逆／求解。
- $M_{qk}$ 保留下三角（含对角），把 chunk 中各 token 的 update 投影到各个 Query 上。

对输入 state $S$，参考实现中的核心数据流可以写成：

$$
\begin{aligned}
\widetilde V&=\operatorname{diag}(\boldsymbol\beta)\left(V-K_{\mathrm{decay}}S^{\mathsf T}\right),\\
U&=\mathrm{INV}\,\widetilde V,\\
O&=Q_{\mathrm{decay}}S^{\mathsf T}+M_{qk}U,\\
\Delta S&=\left(K_{\mathrm{restored}}^{\mathsf T}U\right)^{\mathsf T},\\
S_{\mathrm{new}}&=S\operatorname{diag}\!\left(2^{\boldsymbol g_{\mathrm{total}}}\right)+\Delta S.
\end{aligned}
$$

这里 $V\in\mathbb R^{C\times V}$ 是当前 chunk 的 Value 矩阵，$\widetilde V$ 对应下面 Python 代码的 `v_adjusted`，$O$ 对应 `out_chunk`，$\Delta S$ 对应 `delta_state`。

矩阵维度逐项核对：

| 运算 | 左侧形状 | 右侧形状 | 结果形状 |
|---|---|---|---|
| $K_{\mathrm{decay}}S^{\mathsf T}$ | $C\times K$ | $K\times V$ | $C\times V$ |
| $\mathrm{INV}\,\widetilde V$ | $C\times C$ | $C\times V$ | $C\times V$ |
| $Q_{\mathrm{decay}}S^{\mathsf T}$ | $C\times K$ | $K\times V$ | $C\times V$ |
| $M_{qk}U$ | $C\times C$ | $C\times V$ | $C\times V$ |
| $(K_{\mathrm{restored}}^{\mathsf T}U)^{\mathsf T}$ | $K\times C$ | $C\times V$ | 转置后 $V\times K$ |

所以 $O$ 是 $C\times V$，更新后的 state 仍为 $V\times K$。注意 state 更新里的 $2^{\boldsymbol g_{\mathrm{total}}}$ 按 Key 维度缩放，即沿 $S$ 的列维广播。

### K1 如何求 16×16 的 INV

当前源码不是旧版 Neumann 级数。它保持 `L` 的 fp32 精度，把 16×16 的 `I+L` 分为两个 8×8 对角块做 FP32 forward substitution，再通过两个 BF16 HMMA 矩阵乘把非对角块合并；最后得到 BF16 `INV`。这样既利用下三角结构，也把矩阵尺寸限制在 16×16。仓库 deep-dive 说明，较大的 chunk 会加重 `exp(cumsum(g))` 的动态范围与矩阵求逆成本；`CHUNK=16` 是本版本的设计选择，不是 KDA 数学要求永远固定 16。

## 从源码看 K1 → workspace → K2

入口在 `csrc/smxx/fwd_launch.cu`：K1 grid 为 `(total_tiles, H)`，K2 grid 为 `(N,H)`；两个 kernel 按顺序在同一 CUDA stream launch。每个 kernel 的 grid 维度就是理解并行度的第一线索。

### K1：每个 chunk 独立 prepare

在 `csrc/smxx/fwd_kernel1.cuh`：

1. **CTA 找到 sequence / chunk。** 定长 batch 直接算映射；变长模式先用 tile prefix，把 `global_tile_idx` 二分映射到 sequence 和局部 chunk。prefix 统计每条序列的 `ceil(seq_len/16)`；`total_tiles` 按上界分配，落到有效 chunk 之外的 CTA 会提前 return。
2. **TMA 把输入 tile 搬到 shared memory。** 一个 K1 CTA 准备一个 head 的一个 16-token 块：Q、K、g、beta 和该 head 的 `dt_bias`。发起 TMA 的是 CTA 线程 0；其余线程可在 TMA 传输期间先算 `exp(A_log)`，随后用 transaction barrier 等待数据可见。
3. **线程做 Q/K L2 normalize 与 gate 前缀和。** K1 的 256 threads 将工作分配到行/列；对每个 key feature 得到长度 16 的 gate prefix `G` 和 `g_total`，尾块无效 token 的 gate 设 0、K 清零。
4. **形成 decay 后的矩阵。** 算出 `K_decay`、`Q_decay`、`K_inv`、`K_restored`。随后一个 warp 算 fp32 `L`，另一个 warp 算 bf16 `Mqk`；按三角形 mask 处理后求 `INV`。
5. **写中间 workspace。** 用 TMA store 输出六项：`K_decay`、`Q_decay`、`K_restored`、`g_total`、`INV`、`Mqk`。K2 不需要重新做门控累积和矩阵逆。

K1 的 shared storage 用 union 覆盖生命周期不重叠的缓冲区：先存 `q/k/g`，随后复用空间存 decay 矩阵和 `L/INV/Mqk`。这是用更复杂的生命周期管理换 shared-memory 容量与 occupancy。

### K2：每个 sequence/head 顺序推进 state

`csrc/smxx/fwd_kernel2.cuh` 的 K2 grid 是 `(N,H)`，每个 CTA 只持有一份 `[128,128]` state，并在 `for (t=0; t<t_tiles; ++t)` 中逐 chunk 循环。当前 chunk 的 state 更新完成前，下个 chunk 不可开始，因此这部分不能简单拆成独立 chunk CTAs。

默认 192 threads 分成三种角色：

| Warp | 数量 | 角色 | 工作 |
|---|---:|---|---|
| MMA | 4 | consumer / compute | 读 shared-memory tiles，做矩阵乘、输出和 state update |
| LOAD_QKG | 1 | producer | 用 TMA 加载 V、beta 和 K1 workspace；初始化时还加载 state |
| STORE | 1 | output consumer | 把 output tile 写回 global memory，并在需要时写 final state |

实际采用 3 个 input stages、2 个 output stages。LOAD warp 通过 `producer_acquire` 取得空 pipeline stage，发起 TMA；MMA warp `consumer_wait` 确认对应数据到 shared memory 后再读，用完后 `consumer_release` 释放 stage。MMA 计算后的 output stage 通过 store pipeline 交给 STORE warp 写回。也就是说，warp specialization 不改变 state 的依赖顺序，而是在一条 CTA 的 chunk 循环里重叠“加载下一份 tile”和“计算当前 tile”。

每个 chunk 在数学上的计算次序是：

```text
读取 V 与 K1 workspace
→ 用当前 state 算旧状态对 V 的预测，并形成 residual
→ 乘 beta，再用 INV 得到 U
→ Q_decay @ state.T + Mqk @ U 得到当前 chunk output
→ K_restored.T @ U 并加上衰减后的 state，得到下一份 state
```

U 会在 MMA 路径的寄存器片段之间转置传递，源码注释说明通过 `MOVM_T` 避免将 U 写回 shared memory 再读出。完整块输出用 TMA store；最后一个不满 16 token 的尾块走手动 store，只写有效行，避免覆盖变长 batch 中下一个 sequence 的数据。若需要 `final_state`，CTA 在所有 chunk 后写最终 state。

### 变长序列的 tile 映射

假设 `cu_seqlens=[0,19,24]`，`CHUNK=16`：第一条序列 19 个 token 对应 2 个 tile，第二条 5 个 token 对应 1 个 tile；prefix 是 `[0,2,3]`。K1 的 `global_tile_idx=2` 落在第二条序列：`seq_idx=1`，`local_t=2-2=0`，`bos=19`，所以它读第二条序列的第一个 chunk `[19,24)`。此例 `total_tiles=ceil(24/16)+2=4` 是上界，`global_tile_idx=3` 映射到第二条序列的局部 chunk 1 后发现越界并退出。K2 则直接由 `blockIdx.x=seq_idx` 选择序列，并对该序列的 chunk 做顺序循环。

## 和 FlashMLA、Triton / TMA 概念对照

| 项目 | FlashKDA（本快照） | FlashMLA sparse decode（前一节） |
|---|---|---|
| 算法 | gated delta-rule / recurrent linear attention | 给定稀疏索引的 softmax attention |
| 长历史表示 | 每个 head 的 `[V,K]` state | KV cache 中显式 token（可 paged/量化） |
| chunk / split | chunk 间有 state 依赖，K2 顺序递推 | KV 可 split，partial softmax 需要合并 |
| 数学核心 | 下三角 `INV`、`U`、state update | softmax score、value reduction、LSE |
| 本仓库实现 | CUTLASS/CuTe CUDA，不是 Triton kernel | 多种 CUDA / 其他平台的 kernel family |

| 硬件机制 | 在 FlashKDA 里是否用到？ | 应如何理解 |
|---|---|---|
| TMA | **是** | K1/K2 对输入、workspace、output、state 建了 TMA descriptor；K2 用 TMA pipeline |
| Warp specialization | **是，K2 明确使用** | 4 MMA + 1 LOAD + 1 STORE warp；K1 则以一个 CTA 内的线程/warp 分工完成 prepare |
| Block cluster / CGA 多 CTA | **没有看到** | grid 中 CTA 彼此独立；源码出现 `ClusterTransactionBarrier` 类型不等于创建了多-CTA cluster |
| PDL | **没有看到** | 两个 kernel 在同一 stream 顺序 launch；本路径未用 programmatic dependent launch |
| TMEM / tcgen05 | **没有看到** | 不要把 Blackwell tcgen05 示例套过来；K2 明确使用 `SM80_16x8x16_F32BF16BF16F32_TN` MMA atom，累加片段走 CUTLASS/CuTe 路径 |

TMA 本身是 SM90 能力，但使用 TMA 不代表一定要 cluster、PDL 或 TMEM。FlashKDA 的重点是：**SM90 TMA 搬运 + shared-memory pipeline + 常规 warp MMA/state recurrence**，而不是 Blackwell multi-CTA/TMEM 数据流。

## 两 token 数值轨迹：亲手算出 output 与新 state

为了能手算，把不支持的真实维度缩成 $K=V=1$、$C=2$。这里只验证公式的数据流，**不代表 FlashKDA kernel 接受该 shape**。Q/K 已归一化，取：

$$
Q=K=\begin{bmatrix}1\\1\end{bmatrix},\qquad
V=\begin{bmatrix}3\\4\end{bmatrix},\qquad S=\begin{bmatrix}2\end{bmatrix}
$$

$$
\boldsymbol g_{\mathrm{act}}=\begin{bmatrix}-1\\-1\end{bmatrix},\qquad
\boldsymbol\beta_{\mathrm{logit}}=\begin{bmatrix}0\\0\end{bmatrix},\qquad \alpha=1
$$

先算门控、衰减和 Key 变换：$\sigma(0)=0.5$，累计 gate $G=[-1,-2]^{\mathsf T}$，$g_{\mathrm{total}}=-2$。

| token $t$ | $G_t$ | $2^{G_t}$ | $2^{-G_t}$ | $(K_{\mathrm{decay}})_t$ | $(Q_{\mathrm{decay}})_t$ | $(K_{\mathrm{inv}})_t$ | $(K_{\mathrm{restored}})_t$ |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | -1 | 0.5 | 2 | 0.5 | 0.5 | 2 | 0.5 |
| 1 | -2 | 0.25 | 4 | 0.25 | 0.25 | 4 | 1 |

接着逐项做小矩阵：

$$
\begin{aligned}
K_{\mathrm{decay}}K_{\mathrm{inv}}^{\mathsf T}&=\begin{bmatrix}1&2\\0.5&1\end{bmatrix},\\
L&=\operatorname{diag}(\boldsymbol\beta)\operatorname{tril}\!\left(K_{\mathrm{decay}}K_{\mathrm{inv}}^{\mathsf T},-1\right)=\begin{bmatrix}0&0\\0.25&0\end{bmatrix},\\
I+L&=\begin{bmatrix}1&0\\0.25&1\end{bmatrix},\qquad \mathrm{INV}=(I+L)^{-1}=\begin{bmatrix}1&0\\-0.25&1\end{bmatrix},\\
M_{qk}&=\operatorname{tril}\!\left(Q_{\mathrm{decay}}K_{\mathrm{inv}}^{\mathsf T},0\right)=\begin{bmatrix}1&0\\0.5&1\end{bmatrix}.
\end{aligned}
$$

对旧 state 做 prediction／residual：

$$
\begin{aligned}
K_{\mathrm{decay}}S^{\mathsf T}&=\begin{bmatrix}0.5\\0.25\end{bmatrix}\cdot 2=\begin{bmatrix}1\\0.5\end{bmatrix},\\
V-K_{\mathrm{decay}}S^{\mathsf T}&=\begin{bmatrix}3\\4\end{bmatrix}-\begin{bmatrix}1\\0.5\end{bmatrix}=\begin{bmatrix}2\\3.5\end{bmatrix},\\
\widetilde V&=\operatorname{diag}(\boldsymbol\beta)\left(V-K_{\mathrm{decay}}S^{\mathsf T}\right)=\begin{bmatrix}1\\1.75\end{bmatrix},\\
U&=\mathrm{INV}\,\widetilde V=\begin{bmatrix}1\\-0.25\cdot1+1.75\end{bmatrix}=\begin{bmatrix}1\\1.5\end{bmatrix}.
\end{aligned}
$$

最后分别得到输出与状态：

$$
\begin{aligned}
Q_{\mathrm{decay}}S^{\mathsf T}&=\begin{bmatrix}0.5\\0.25\end{bmatrix}\cdot2=\begin{bmatrix}1\\0.5\end{bmatrix},\\
M_{qk}U&=\begin{bmatrix}1\\0.5\cdot1+1.5\end{bmatrix}=\begin{bmatrix}1\\2\end{bmatrix},\\
O&=\begin{bmatrix}1\\0.5\end{bmatrix}+\begin{bmatrix}1\\2\end{bmatrix}=\begin{bmatrix}2\\2.5\end{bmatrix},\\
\Delta S&=\left(K_{\mathrm{restored}}^{\mathsf T}U\right)^{\mathsf T}=\begin{bmatrix}0.5\cdot1+1\cdot1.5\end{bmatrix}=\begin{bmatrix}2\end{bmatrix},\\
S_{\mathrm{new}}&=2^{-2}\begin{bmatrix}2\end{bmatrix}+\begin{bmatrix}2\end{bmatrix}=\begin{bmatrix}2.5\end{bmatrix}.
\end{aligned}
$$

这条轨迹强调：chunk 第二个 token 的结果已经包含第一个 token 的因果影响（由 $L$、$\mathrm{INV}$ 编码），而 chunk 结束后产生的 $S_{\mathrm{new}}=2.5$ 才交给下一个 chunk。

## 完整可运行示例

脚本 `operators/code/flashkda/kda_scalar_trace.py` 只使用 Python 标准库，可直接运行。以下是完整内容：

```python
#!/usr/bin/env python3
"""A tiny, standard-library-only trace of one FlashKDA chunk recurrence.

This uses K=V=1 and CHUNK=2 to make every intermediate value visible. It is a
mathematical teaching example, not a supported FlashKDA kernel shape.
"""
from math import exp
from typing import List, Sequence

Matrix = List[List[float]]


def transpose(a: Matrix) -> Matrix:
    return [list(row) for row in zip(*a)]


def matmul(a: Matrix, b: Matrix) -> Matrix:
    if not a or not b or len(a[0]) != len(b):
        raise ValueError("incompatible matrix shapes")
    bt = transpose(b)
    return [[sum(x * y for x, y in zip(row, col)) for col in bt] for row in a]


def lower_triangle(a: Matrix, include_diagonal: bool) -> Matrix:
    result: Matrix = []
    for i, row in enumerate(a):
        result.append([
            value if (j <= i if include_diagonal else j < i) else 0.0
            for j, value in enumerate(row)
        ])
    return result


def inverse_unit_lower_from_l(l: Matrix) -> Matrix:
    """Solve (I + L) X = I by forward substitution; L is strict-lower."""
    n = len(l)
    a = [[l[i][j] + float(i == j) for j in range(n)] for i in range(n)]
    inverse = [[0.0 for _ in range(n)] for _ in range(n)]
    for col in range(n):
        for row in range(n):
            rhs = float(row == col) - sum(
                a[row][k] * inverse[k][col] for k in range(row)
            )
            inverse[row][col] = rhs / a[row][row]
    return inverse


def scale_rows(a: Matrix, factors: Sequence[float]) -> Matrix:
    return [[value * factors[i] for value in row] for i, row in enumerate(a)]


def add(a: Matrix, b: Matrix) -> Matrix:
    return [[x + y for x, y in zip(ra, rb)] for ra, rb in zip(a, b)]


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + exp(-x))


def show(name: str, value: object) -> None:
    print(f"{name} = {value}")


def main() -> None:
    # Two tokens, one key/value feature. q and k are already L2-normalized.
    q: Matrix = [[1.0], [1.0]]
    k: Matrix = [[1.0], [1.0]]
    v: Matrix = [[3.0], [4.0]]
    g_after_activation = [-1.0, -1.0]  # base-2 log decay per token
    beta_logits = [0.0, 0.0]
    state: Matrix = [[2.0]]  # [V, K]
    scale = 1.0

    beta = [sigmoid(x) for x in beta_logits]
    g_cumsum: List[float] = []
    running = 0.0
    for gate in g_after_activation:
        running += gate
        g_cumsum.append(running)
    g_total = g_cumsum[-1]

    decay = [2.0 ** x for x in g_cumsum]
    inverse_decay = [2.0 ** (-x) for x in g_cumsum]
    k_decayed = scale_rows(k, decay)
    q_decayed = [[row[0] * factor * scale] for row, factor in zip(q, decay)]
    k_inv = scale_rows(k, inverse_decay)
    k_restored = scale_rows(k_inv, [2.0 ** g_total] * len(k_inv))

    raw_k_interaction = matmul(k_decayed, transpose(k_inv))
    l = lower_triangle(raw_k_interaction, include_diagonal=False)
    l = scale_rows(l, beta)
    mqk = lower_triangle(matmul(q_decayed, transpose(k_inv)), include_diagonal=True)
    inv = inverse_unit_lower_from_l(l)

    old_state_prediction = matmul(k_decayed, transpose(state))
    residual = [[x - y for x, y in zip(rv, pv)] for rv, pv in zip(v, old_state_prediction)]
    v_adjusted = scale_rows(residual, beta)
    u = matmul(inv, v_adjusted)

    direct_output = matmul(q_decayed, transpose(state))
    correction = matmul(mqk, u)
    output = add(direct_output, correction)

    delta_state_kv = matmul(transpose(k_restored), u)  # [K, V]
    delta_state = transpose(delta_state_kv)             # [V, K]
    decayed_state = scale_rows(state, [2.0 ** g_total])
    next_state = add(decayed_state, delta_state)

    show("beta", [round(x, 6) for x in beta])
    show("g_cumsum", g_cumsum)
    show("k_decayed", k_decayed)
    show("k_inv", k_inv)
    show("k_restored", k_restored)
    show("L", l)
    show("INV", inv)
    show("Mqk", mqk)
    show("v_adjusted", v_adjusted)
    show("U", u)
    show("output", output)
    show("delta_state", delta_state)
    show("next_state", next_state)

    assert output == [[2.0], [2.5]]
    assert next_state == [[2.5]]
    print("trace check: PASS")


if __name__ == "__main__":
    main()
```

运行：

```bash
python3 operators/code/flashkda/kda_scalar_trace.py
```

预期末尾：

```text
output = [[2.0], [2.5]]
delta_state = [[2.0]]
next_state = [[2.5]]
trace check: PASS
```

## 常见误读与可观察后果

1. **把它当作 softmax attention。** FlashKDA 没有对历史 token 的 `QK^T` 做 softmax；它用 `INV` 解 chunk 内 delta-rule 依赖。如果照 softmax 写 reference，输出会在整个序列上系统性不同。
2. **把 chunk 当成可独立的 split-K。** K1 的 prepare 可以并行，但 K2 处理下个 chunk 要等当前 `state_new`。若各 chunk 使用相同初始 state，跨 chunk 输出和 final_state 都会错。
3. **忘记 beta 是 logit。** API 输入 `beta`，内部才 sigmoid；调用前先 sigmoid 会变成 `sigmoid(sigmoid(x))`，更新幅度通常偏小。
4. **把 `g` 当作直接乘法 gate 或自然指数。** 源码先把 gate 参数变换到 base-2 的指数域并沿 token 轴累加。把 `2**G` 写成 `exp(G)` 会改变 decay 强度；不累计则会丢失 chunk 内时间位置。
5. **忽略 state 的转置与维度。** 存储为 `[V,K]`，预测要乘 `S.T`，而 `(K_restored.T @ U)` 结果需转回 `[V,K]`。错了通常会 shape error；非方阵泛化时尤其容易暴露。
6. **认为 TMA 就意味着 block cluster、PDL 或 TMEM。** 这些是不同机制。源码中的 CTA grid 和普通同 stream kernel launches 没有形成多-CTA cluster/PDL；K2 的 MMA 也不是 tcgen05/TMEM。按错误架构去找 barrier 或 profiler 指标，会对不上源码和 Nsight 轨迹。
7. **以为 K1/K2 能合并一定更快。** 仓库作者报告拆分后比早期融合版至少快 15%，原因是 K1 的 chunk 并行度远高于 state recurrence；但这是项目 deep-dive 的 benchmark 结论，不是本机验证结果，也不保证所有 shape 都相同。

## 如何验证与本次边界

在任何有 Python 3 的环境运行上面的纯 Python 标量 trace：

```bash
python3 operators/code/flashkda/kda_scalar_trace.py
```

这只验证本文缩维数学流程的中间结果与自检，不验证 CUDA kernel。FlashKDA 的真实回归入口是其仓库 `bash tests/test.sh`，需要 CUDA 12.9+、PyTorch 2.4+ 和 SM90+ NVIDIA GPU。本次环境为 macOS arm64，且没有 PyTorch、`nvcc` 或 `nvidia-smi`，因此这里完成的是源码静态阅读与 CPU 数学示例验证，**没有声称编译或运行过 FlashKDA GPU kernel**。

主要源码锚点（FlashKDA `7afb9f4`）：

- `flash_kda/__init__.py`：Python API 文档与 workspace 调用。
- `csrc/flash_kda.cpp`：dtype、contiguous、shape、K=128 与 varlen 校验。
- `csrc/smxx/fwd_launch.cu`：workspace / TMA descriptor、tile prefix、K1/K2 grid 与 launch 顺序。
- `csrc/smxx/fwd_kernel1.cuh`：K1 的 varlen tile 映射、TMA 输入、Q/K normalize、gate scan、L/INV/Mqk 和 workspace store。
- `csrc/smxx/fwd_kernel2.cuh`：K2 的 warp roles、state load、TMA pipeline、chunk recurrence、output/state store。
- `tests/torch_ref.py`：可读的 chunk 参考计算与数值约定。
- `docs/20260420-flashkda-v1-deep-dive.md`：CHUNK、K1/K2 分工、精度与性能设计说明。

## 自我检查（含答案）

1. K1 和 K2 分别沿什么维度并行？
   - **答：** K1 的 CTA 对应 head × chunk，可同时 prepare 多个 chunk；K2 的 CTA 对应 sequence × head，chunk 维在 CTA 内顺序递推。
2. `L` 为什么严格下三角，而 `Mqk` 包含对角线？
   - **答：** `L` 表示一个 token 对更早 token 更新的因果依赖；`Mqk` 还要包含当前 token 自身对当前输出的贡献。
3. `INV` 在 `U = INV @ v_adjusted` 里做什么？
   - **答：** 它是 `(I+L)^-1`，将一个 chunk 内的因果三角耦合转换为矩阵形式的解。
4. FlashKDA 是否像 FlashMLA 那样对 split 的 softmax partial 做 merge？
   - **答：** 不做。它不是 softmax attention；K2 通过 state 把 chunk 依赖接起来。
5. 代码中有 TMA、warp specialization，就能推断有 cluster / PDL / TMEM 吗？
   - **答：** 不能。当前路径明确有 TMA 与 K2 warp roles，但没有看到多-CTA cluster、PDL 或 tcgen05/TMEM 使用。

## 进度

已经覆盖：

- [x] KDA 原始逐 token 公式逐项拆解、统一 LaTeX 记号与递归数据流图、2×2 遗忘／残差／秩一更新数值例子。
- [x] FlashKDA 的 KDA 数学、16-token chunk 内三角系统、K1 prepare → workspace → K2 recurrent state 的执行数据流及 TMA/warp specialization 边界。

下一知识点：

- [ ] 逐段映射 K2 中 4 个 MMA warps 的 16×16 tile：U 如何留在寄存器里、state 如何分块读改写，以及 input/output pipeline 如何重叠。
