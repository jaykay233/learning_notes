# TP 与 SP 配合时的前向数据流

## 本次讲解位置

- 章节：`training` / Megatron-LM 张量并行与序列并行
- 小节：Megatron-LM 02《张量并行（Tensor Parallelism）》Part 4，4.1「序列并行（SP）与 TP 的配合」
- 知识点：一个 Transformer 子层前向中，激活如何在 SP 序列分片布局与 TP 矩阵乘布局之间转换
- 上次：Reduce-Scatter 是跨 rank 求和后按某个维度分片；SP 下 Vocab Embedding 也可用它直接产出序列分片
- 下次：沿 Megatron 的 `ColumnParallelLinear` / `RowParallelLinear` 源码逐段核对这些布局转换

## 为什么现在讲这个

只知道“SP 把序列切开”以及“TP 把权重切开”，还看不出两种切分怎样在一次前向中接上。关键设计问题是：LayerNorm、残差等地方希望每卡只存一段序列来节省激活显存；但 TP 的列并行、行并行计算与归约需要各 rank 对齐同一批 token。若不追踪张量布局，就会把 all-gather 误解为“SP 失效了”，或以为 reduce-scatter 只是普通 all-reduce 的别名。

## 先记住一个画面

把 TP 组里的每张卡想成负责一部分 token 的工位：

1. **SP 区域**：每卡只拿自己那段 token，但每个 token 的完整 hidden 向量都在本卡。
2. **进入 TP 的矩阵乘区域**：all-gather 把 token 段拼起来，让每个 TP rank 都有完整序列；与此同时，每卡仍只持有自己那片权重。
3. **离开 TP 矩阵乘区域**：row-parallel 产生的局部结果先跨卡求和，再 reduce-scatter 把不同 token 段分回各卡。
4. 回到 LayerNorm、残差等 SP 区域时，每卡又只需保留本地 token 段。

因此可以先记成：

```text
SP 序列分片
  → all-gather 序列
  → TP 列并行矩阵乘 + 激活 + TP 行并行矩阵乘
  → reduce-scatter（求和并切序列）
  → SP 序列分片
```

这里的 all-gather / reduce-scatter 是布局转换边界；实际代码可能把通信封装在并行线性层里，也可能使用融合或异步实现。

## TP 和 SP 分别切什么

设 hidden state 的形状为 `[S, B, H]`，`S` 是序列长度、`B` 是 batch、`H` 是 hidden size，TP 组大小为 `P`。

| 机制 | 切分对象 | 每个 rank 大致持有什么 |
|---|---|---|
| TP | 通常切线性层权重的输入/输出 hidden 维 | 不同的权重分片；特定阶段的激活可能按 TP 规则复制或部分持有 |
| SP | 非矩阵乘区域激活的 sequence 维 | `[S/P, B, H]`，即本地 token 段的完整 hidden 向量 |

SP 不是新加一组 rank，也不是把 TP 替换掉；在这里它复用同一 TP group，只改变一些阶段中激活由“完整序列复制”变成“序列分片”的布局。

## 一层 MLP 的前向逐步看

以常见的两层 MLP 为例：第一层是 ColumnParallelLinear，第二层是 RowParallelLinear。忽略 bias、dropout 和具体 fused kernel，关注张量的形状与归属。

### 1. SP 区域：本地归一化

输入 `X` 是 `[S, B, H]`。开 SP 后，TP 组大小为 `P` 时，每个 rank 只保留 `S/P` 个 token：

```text
rank 0: X[0 : S/P]       → [S/P, B, H]
rank 1: X[S/P : 2S/P]   → [S/P, B, H]
...
```

LayerNorm 对每个 token 的 hidden 维独立计算，因此可以对本地序列片段做，不必先把全序列 gather 到每张卡。

### 2. all-gather：为标准 TP 矩阵乘恢复完整序列

进入 ColumnParallelLinear 前，沿 sequence 维 all-gather：每个 TP rank 都得到 `[S, B, H]` 的完整输入。所有 rank 此时看到相同的 token 坐标，但各 rank 的权重输出通道分片不同。

### 3. ColumnParallelLinear：切 hidden 输出维

假设 MLP 中间维是 `4H`。权重沿输出维分到 `P` 个 rank：

```text
每 rank 的局部权重: [H, 4H/P]
每 rank 的输出:     [S, B, 4H/P]
```

所以每个 rank 都计算完整序列，但只计算中间 hidden 的一部分。逐元素激活函数（例如 GELU）可直接作用于这份局部输出。

### 4. RowParallelLinear：各算一份 partial，再对齐求和

第二层权重沿输入维切分。每个 rank 用自己的中间 hidden 分片计算：

```text
每 rank 的局部输入:    [S, B, 4H/P]
每 rank 的局部权重:    [4H/P, H]
每 rank 的 partial 输出: [S, B, H]
```

这些 `[S, B, H]` 是同一批 token 的**部分和**：rank 0、rank 1……分别贡献不同 hidden 分片乘出来的结果。必须对同一 token 的 partial 求和，才得到完整的 MLP 输出。

### 5. reduce-scatter：求和的同时切回序列

SP 开启时，reduce-scatter 把所有 rank 的 partial 输出按元素求和，再沿 sequence 维切成 `P` 份分给各 rank：

```text
reduce:  对同一 token 的各 rank partial 相加，恢复完整 H 维输出
scatter: rank 0 留前段 token，rank 1 留后段 token，……
```

结果每卡回到 `[S/P, B, H]`。接下来的残差、dropout 等可以在本地序列片段上继续做。

## 一个两卡、四 token 的具体例子

令 `P=2`、`S=4`、`B=1`，为了容易看，把 `H=2`：

| 阶段 | rank 0 | rank 1 |
|---|---|---|
| SP 输入 | token `t0,t1`，形状 `[2,1,2]` | token `t2,t3`，形状 `[2,1,2]` |
| all-gather 后 | `t0,t1,t2,t3`，形状 `[4,1,2]` | `t0,t1,t2,t3`，形状 `[4,1,2]` |
| Column 输出（假设中间维 `4H=8`） | 全 4 个 token 的中间通道 `0..3` | 全 4 个 token 的中间通道 `4..7` |
| Row partial 输出 | `t0..t3` 各自的一份 `[4,1,2]` partial | `t0..t3` 各自的另一份 `[4,1,2]` partial |
| reduce-scatter 后 | 两份 partial 相加后的 `t0,t1` | 两份 partial 相加后的 `t2,t3` |

特别看 `t0`：虽然最后 rank 0 负责保存 `t0`，计算 RowParallelLinear 时，rank 1 也必须为 `t0` 算出自己那份 hidden 通道贡献。于是进入 TP 矩阵乘区域时，两卡都需要完整 token 序列；等相同 token 的 partial 求和完，才能把 `t0,t1` 的完整结果留在 rank 0、把 `t2,t3` 留在 rank 1。

## 把同一套逻辑套到 Attention：QKV → Attention → 输出投影

“Column → Row 配对”在注意力里通常指：QKV 的列并行投影 → 每个 rank 负责的局部 attention heads → Attention 输出的行并行投影。中间有 attention 运算，不是两个线性层直接相连。这里假设普通 TP、`CP=1`，并且 attention heads 能按 TP rank 均匀切分。

### 不开 SP

```text
每 rank: 完整 X [S,B,H]（复制）
  → QKV ColumnParallel：本 rank 算自己的 Q/K/V heads，序列仍是完整 S
  → 本地 heads 对完整序列做 attention
  → 输出 RowParallel：本 rank 产出完整 [S,B,H] 的 partial
  → all-reduce 求和，得到完整 attention 输出 [S,B,H]（每 rank 都有）
  → dropout / residual 等，仍保留完整序列
```

因为各 rank 管不同 heads，它们的 attention 结果对应不同 hidden 通道；输出投影把输入 hidden 通道切开后，各 rank 计算同一批 token 的 partial 输出，因此要 all-reduce 求和。

### 开 SP

```text
rank r: 本地 X shard [S/P,B,H]
  → 本地 LayerNorm
  → all-gather sequence：各 rank 都得到完整 X [S,B,H]
  → QKV ColumnParallel：本 rank 计算自己的 heads，覆盖完整 S
  → 本地 heads 对完整 K/V 序列做 attention
  → 输出 RowParallel：得到完整 S 上的 partial [S,B,H]
  → reduce-scatter：partial 求和，并按 sequence 切回 [S/P,B,H]
  → 本地 dropout / residual，结果仍为 [S/P,B,H]
```

Attention 这里还有一个直觉：标准 SP 路径下，Q、K、V 都先覆盖完整序列，所以每个 rank 上的局部 heads 可以让每个 query 访问整段 K/V。若直接只拿本地序列片段做 QKV，每张卡就缺少其他 token 的 K/V，无法得到标准全序列 self-attention；当然也可以设计“本地 Q、全局 K/V”等其他通信算法，但那不是这里讨论的标准 TP+SP 布局。

### 一眼对照

| 位置 | 不开 SP 的前向 | 开 SP 的前向 | rank 间激活布局变化 |
|---|---|---|---|
| QKV ColumnParallel 前 | 完整序列已复制，无需 gather | 先 all-gather sequence | SP shard → 完整序列 |
| Attention（本地 heads） | 每 rank 的 heads 看完整序列 | 每 rank 的 heads 也看完整序列 | TP attention 区域里均为完整 S |
| 输出 RowParallel 后 | all-reduce，完整输出复制到各 rank | reduce-scatter，归约后分发序列 shard | 完整 partial → SP shard |
| LayerNorm / 残差等边界 | 每卡完整 `[S,B,H]` | 每卡本地 `[S/P,B,H]` | SP 激活显存约为 `1/P` |

反向通信与前向布局转换互为对应（省略计算本身）：不开 SP 时，Column 输入映射的反向需要 all-reduce 输入梯度、Row 输出 all-reduce 的反向是 identity；开 SP 后，前向 all-gather 的反向是 reduce-scatter，前向 reduce-scatter 的反向是 all-gather。

## 为什么是 all-gather + reduce-scatter

不开 SP 时，典型 TP MLP 的前向边界可以抽象为：

```text
ColumnParallelLinear 前：各 rank 有相同完整输入（前向无需 gather）
RowParallelLinear 后：各 rank 的 partial 输出 all-reduce，得到相同完整输出
```

开 SP 后，输入和输出边界改为序列分片：

```text
ColumnParallelLinear 前：先 all-gather，SP 分片 → 完整序列
RowParallelLinear 后：reduce-scatter，partial 求和 + 完整序列 → SP 分片
```

all-reduce、all-gather、reduce-scatter 都涉及跨 rank 通信；SP 并不是“没有通信”。在常见实现和张量大小假设下，all-gather 与 reduce-scatter 的通信量级和原本的 all-reduce 相近，SP 的主要收益是让非矩阵乘区域的激活每卡从约 `S×B×H` 降到约 `(S/P)×B×H`。代价是边界处进行布局转换；通信能否被融合或与计算重叠取决于实现与配置。

## 常见误区

1. **“开 SP 后每一步所有张量都只有 `S/P` 长。”** 不对。SP 区域是序列分片；进入这里描述的 TP 矩阵乘区域后，序列通常会 all-gather 回完整长度。
2. **“all-gather 是复制了权重。”** 不对。这里 gather 的是激活的 sequence 维；权重仍按 TP 规则切分。
3. **“reduce-scatter 不再做求和。”** 不对。它先归约各 rank 对相同 token 的 partial，再把完整结果按 sequence 分发。
4. **“为什么不让每卡只对自己的 token 直接做完全部 TP？”** 这种数据流需要重新设计跨 rank 对齐/归约；这里讲的是 Megatron 常见的布局转换：先让各 rank 的 token 坐标对齐，完成标准列并行/行并行，再切回 SP 布局。
5. **“SP 把总激活显存全部缩小 P 倍。”** 不一定。缩小的是适用的 SP 激活区域；矩阵乘中间、通信缓冲区、保留用于反向的张量以及 checkpointing 都会影响实际峰值。

## 自测

1. 开 SP 后，LayerNorm 为什么可以在 all-gather 前运行？——它按 token 的 hidden 维独立计算，本地序列片段已经包含完整 hidden 向量。
2. ColumnParallelLinear 后，各 rank 的输出有什么不同？——token 坐标相同、序列长度完整，但中间 hidden 输出通道不同。
3. RowParallelLinear 的 partial 为什么要相加？——各 rank 分别使用不同输入 hidden 分片与对应权重，贡献的是同一 token 完整 hidden 输出的一部分。
4. reduce-scatter 的结果为什么能直接接本地残差？——求和恢复了完整 hidden 向量，scatter 又让每卡拿到与本地残差相同的 token 段。
5. SP 的显存收益与通信代价分别是什么？——适用非矩阵乘区域的序列激活每卡约降为 `1/P`；需要 all-gather / reduce-scatter 在两种布局间转换。

## 已经覆盖 / 下一知识点

- [x] TP 矩阵乘区域前后的 SP 布局转换及前向数据流。
- 下一知识点：沿 Megatron `ColumnParallelLinear.forward` 与 `RowParallelLinear.forward` 源码核对 all-gather / reduce-scatter 的具体调用路径，并留意版本差异、异步和融合实现。
