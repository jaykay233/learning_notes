# 集合通信与拓扑：AllReduce 省的是步数，All-to-All 破的是对齐

```text
本次讲解位置
章节：communication / RDMA
小节：集合通信方向（Q26、Q27 的定量版）
知识点：AllReduce 的"量"与"步"两个轴、SHARP 到底省什么、rail 对齐假设、EP 大小的两难
上次：注册成本账本（39）——为什么存储路径被逐页成本卡住
下次：41 · K8s 多租户（同一套对象，换个运行环境就变成"边界"问题）
```

## 为什么现在讲这个

[32 章 Q26/Q27](./32-rdma-interview-questions.md) 的两道题是分开的：
一道问集合通信算法，一道问 rail-optimized 拓扑。面试里也常被分开问，于是背成两个知识点。

但它们其实是**同一件事的两面**：

```text
集合通信算法 决定"发多少次、每次发多少"
拓扑         决定"每次发要跳几跳、跨不跨 spine"
性能 = f(次数 × 单次固定成本, 总量 / 带宽, 跳数 × 单跳成本)
```

更关键的是，这两者之间有一个**隐含的匹配假设**：

> rail-optimized 拓扑只有在"通信伙伴按 rail 对齐"时才划算。

AllReduce 天然满足这个假设（所有 rank 地位对称，可以按 rail 分组）。
All-to-All 天然破坏它（MoE 的 token 路由是数据相关的，目标 rank 近似随机）。
这就是为什么 [24 章](./24-deepep-v2-hybrid-scaleout-scaleup-dataflow.md) 的 DeepEP 要做分层传输，
而不是简单换一个更好的集合通信库。

这一章把这两件事放到一个账本里算清楚，并且纠正一个流传很广的误解：
**SHARP 不省带宽。**

---

## 心智模型一：AllReduce 有两个轴，别混着比

任何 AllReduce 实现都同时受两个量约束：

```text
时间 ≈ 步数 × 单次固定成本   +   每 rank 数据量 / 带宽
       └──── 步的轴 ────┘      └──── 量的轴 ────┘
```

小消息时第一项主导，大消息时第二项主导。三种算法的差别**只落在其中一个轴上**：

| 算法 | 每 rank 量 | 步数 | 省的是哪个轴 |
| --- | --- | --- | --- |
| Ring | $2\frac{N-1}{N}S$ | $2(N-1)$ | 都不省（基准） |
| Tree（递归 halving+doubling） | $2\frac{N-1}{N}S$ | $2\log_2 N$ | **步数** |
| SHARP（交换机归约） | $2S$ | $2$ | **步数**（量反而略增） |

注意 SHARP 那一格的 `2S`：**它比 ring 的 `2(N-1)/N·S` 还大**。
ring 在 N 很大时量趋近 `2S`，但永远小于 `2S`。

所以 SHARP 的真实卖点是两件事：

1. **砍步数**：从 $2(N-1)$ 到 2。这是数量级差别。
2. **卸载计算**：归约从 GPU 挪到交换机 ASIC，省 GPU 的 SM 和显存读写。

**不是省带宽。** 这一点下面用数字证。

## 账本一：量与步数

脚本 [`40_collective_topology_model.py`](./40_collective_topology_model.py)：

```bash
cd ~/Downloads/learning_notes
python3 communication/40_collective_topology_model.py
```

第一张表（256 MiB 消息，400 Gbps）：

```text
     N      ring 量   ring 步      tree 量   tree 步     SHARP 量   SHARP 步
     8     448.0MB       14     448.0MB        6     512.0MB         2
    16     480.0MB       30     480.0MB        8     512.0MB         2
    64     504.0MB      126     504.0MB       12     512.0MB         2
   256     510.0MB      510     510.0MB       16     512.0MB         2
  1024     511.5MB     2046     511.5MB       20     512.0MB         2
```

两条从表的边缘就能读出来的事实：

- **ring 与 tree 的量完全相同**（448.0 → 511.5 MB 一路上升，两者逐行相等）。
  所以"tree 比 ring 快"从来不是因为发得少，是因为步数少。
- **SHARP 的量恒定 512.0 MB，且大于 ring**。它在 N=8 时是 512 vs 448，多 14%。

## 账本二：换算成时间，差距在哪

```text
   --- 1 MiB 小消息 ---
     N    ring ms    tree ms   SHARP ms    SHARP 加速
     8      0.058      0.046      0.045        1.3x
    64      0.230      0.059      0.045        5.1x
   256      0.807      0.066      0.045       18.0x
  1024      3.111      0.072      0.045       69.2x

   --- 256 MiB 大消息 ---
     N    ring ms    tree ms   SHARP ms    SHARP 加速
     8      9.416      9.404     10.740        0.9x
    64     10.759     10.588     10.740        1.0x
   256     11.460     10.719     10.740        1.1x
  1024     13.796     10.757     10.740        1.3x
```

同样的算法、同样的拓扑，只换消息大小，结论完全反转：

- **1 MiB**：N=1024 时 ring 3.111 ms，SHARP 0.045 ms，**69 倍**。
- **256 MiB**：N=1024 时 ring 13.796 ms，SHARP 10.740 ms，**1.3 倍**；
  而且 **N=8 时 SHARP 比 ring 更慢（0.9x）**，因为它的量更大。

**这就是为什么"SHARP 能加速集合通信"这句话必须带条件。**
正确说法是："SHARP 在**小消息、大 N** 下把步数从 2046 砍到 2"。
在训练的大梯度同步里，消息通常是 hundreds of MiB 级别，SHARP 的收益远没有演示里那么夸张。

顺带一个推论：如果你的 AllReduce 慢，先算一下自己在哪个轴。

```text
消息小（< 1 MiB）   -> 步数轴  -> 换算法（tree/SHARP）、减 N、用分桶并行多个 group
消息大（> 64 MiB）  -> 量轴    -> 换拓扑（更大带宽、rail 对齐）、减 N、用 bf16/压缩
```

选错轴去优化，就是把一个 1.3x 的空间当成 69x 来花预算。

## 心智模型二：rail 拓扑是"把一维全连接压成 8 条独立通道"

```text
        spine（rail 之间互联，只走「部分和」）
          |     |     |     |     |     |     |     |
        R0    R1    R2    R3    R4    R5    R6    R7   ← rail 交换机
         |     |     |     |     |     |     |     |
       GPU0  GPU1  GPU2  GPU3  GPU4  GPU5  GPU6  GPU7  ← 每节点一个 GPU 接一条 rail
```

在 fat-tree 里，任意两个 GPU 之间最多 2 跳，但每一跳都要过 spine 级别的交换机。
rail 优化的做法是**给每个 GPU 一条专属通道**，让"同号 GPU 之间"的通信只走 1 跳。

它的收益只在一种流量形态下成立：

```text
AllReduce（对称集合操作）
  step 1: 节点内 NVLink 归约        -> 不占网
  step 2: 同 rail 的节点间交换      -> 1 跳，spine 零流量
  step 3: 跨 rail 只传「部分和」    -> spine 只承担 1/8 的量
```

**关键词是"可以按 rail 分组"。** AllReduce 里每个 rank 贡献一份等权数据，
调度器完全可以把"同 rail 的那组"先归约掉。数据怎么切由算法决定，不由数据内容决定。

## 账本三：All-to-All 为什么破掉这个假设

MoE 的 All-to-All 不一样：**一个 token 去哪个 rank，由它的 expert 路由决定**。
路由是数据相关的，所以目标 rank 在 `1..EP` 上近似均匀——**和 rail 编号无关**。

```text
    EP     同 rail 概率     跨 spine 比例       跳数(平均)
     8       100.0%          0.0%         1.00
    16        50.0%         50.0%         1.50
    64        12.5%         87.5%         1.88
   256         3.1%         96.9%         1.97
```

同 rail 概率 = `8 / EP`（8 个 rail，目标均匀分布在 EP 个 rank 上）：

- **EP=8**：100% 同 rail。这不是巧合——8 个 rank 正好每条 rail 一个。
  这也是"EP 不要超过单 NVLink 域大小"这条工程规矩的来源。
- **EP=64**：只有 12.5% 同 rail，**87.5% 的流量要跨 spine**，平均跳数 1.88。
  对比 AllReduce 的"spine 只承担 1/8"，这里是"spine 承担 7/8"——**差 7 倍**。
- **EP=256**：96.9% 跨 spine，rail 拓扑退化成普通 fat-tree，花的钱白花了。

## 账本四：EP 大小的两难

```text
    EP    dispatch     combine          合计
     8     58.72MB     58.72MB    117.44MB
    16     29.36MB     29.36MB     58.72MB
    64      7.34MB      7.34MB     14.68MB
   256      1.84MB      1.84MB      3.67MB
```

（每 rank 4096 token，top-8，hidden 7168，bf16）

这张表和上一张表的方向**相反**：

```text
EP 变大  ->  每 rank 的量变小（好：摊薄）      -> 跨 spine 比例变高（坏：跳数翻倍）
```

所以 ep_size 不是"越大越好"或"越小越好"，它有一个由拓扑决定的最优区间：

```text
EP <= 8    量最大，但 100% rail 对齐   -> 网内最优，靠 NVLink 吸收
EP = 16~64 量摊薄到 1/2~1/8，但 50%~87.5% 跨 spine  -> 需要分层/分桶
EP >= 256  量很小，但拓扑优势全失      -> 只能靠拥塞控制兜底（35 章）
```

DeepEP 的选择是**不改拓扑，改数据流**：把 EP 限制在 NVLink 域内，
跨节点部分用分层（scaleout/scaleup）传输，见
[24 章 hybrid 数据流](./24-deepep-v2-hybrid-scaleout-scaleup-dataflow.md)。
本质是"用两次小的、拓扑友好的通信，替代一次大的、拓扑不友好的通信"。

## 常见错法，以及症状

| 错法 | 症状 |
| --- | --- |
| 把 SHARP 当"省带宽"用 | 大消息上收益 1.0x 甚至 0.9x，白买交换机 |
| 小消息用 ring 扩到 1024 卡 | AllReduce 时间随 N 线性涨（步数 × 固定成本） |
| 以为 tree 比 ring 发得少 | 量其实完全一样，误判瓶颈在带宽 |
| EP 开到 64+ 还指望 rail 优化 | 87.5% 流量跨 spine，spine 成为热点 |
| 按 EP=8 的带宽规划 EP=64 的集群 | 每 rank 量小了 8 倍，但跳数涨了 1.88 倍，实测低于预期 |
| 只看平均带宽不看 rail 分布 | 平均值掩盖 spine 拥塞（见 35 章"goodput 不敏感"） |
| All-to-All 不分桶直接发 | 小包洪泛，固定成本主导，带宽用不满 |

第 2 条和第 7 条是同一个原因：**小消息时固定成本主导**（[33 章](./33-rdma-six-layer-spine.md)）。
All-to-All 如果不分桶，每个 token 一个小包，1.5 µs 的固定成本会把带宽彻底吃掉。

## 模型边界

- 步数按"每步一个消息"计，真实实现会流水/重叠相邻步，实际时间低于模型
- ring 每步走 `S/N`，这里按单个消息的固定成本算，没有建模 pipelining
- SHARP 的交换机内部延迟按 0 算，真实有 200~500 ns 量级
- rail 对齐假设"目标均匀分布"，真实路由有倾斜（热点 expert），实际更差
- 没有建模 NVLink 域内的带宽（比网络高一个量级），所以节点内那一步被算成"免费"
- EP 数据量按"每 token 复制 top-k 份"算，没有建模 DeepEP 的量化/压缩

---

## 自测

1. Ring 和 Tree 的每 rank 数据量一样吗？那 Tree 为什么快？
2. SHARP 省的是带宽吗？为什么 N=8 的大消息上它反而更慢？
3. 一个 AllReduce 慢，你怎么判断该优化"步"还是优化"量"？
4. rail-optimized 拓扑的隐含假设是什么？AllReduce 为什么满足、All-to-All 为什么不满足？
5. EP=64 时跨 spine 的流量比例是多少？和 AllReduce 的 1/8 差几倍？
6. 为什么"EP 越大每 rank 量越小"却不一定更快？
7. DeepEP 为什么不直接换一个更好的集合通信库来解决 rail 不对齐？

<details>
<summary>答案</summary>

1. **一样**，都是 $2\frac{N-1}{N}S$（脚本里逐行相等：448.0/480.0/504.0/510.0/511.5 MB）。
   Tree 快是因为步数从 $2(N-1)$ 降到 $2\log_2 N$，省的是串行延迟，不是数据量。
2. **不是。** SHARP 的量是固定 $2S$，比 ring 的 $2\frac{N-1}{N}S$ 还大。
   它省的是步数（→2）并把归约卸载到交换机。
   N=8 时 ring 量 448 MB < SHARP 512 MB，而两者步数差距只有 14 vs 2，
   省下的步数抵不过多发的 64 MB，所以 SHARP 反而慢（0.9x）。
3. 算两个数：`步数 × 1.5µs` 和 `量 / 带宽`。
   前者大 → 优化步（换算法、减 N、分桶）；后者大 → 优化量（换拓扑、减精度、压缩）。
   分界大约在 1 MiB 量级（脚本中 N=1024、1 MiB 时 ring 3.111 ms，
   其中 2046 × 1.5 µs = 3.07 ms 几乎全是步数）。
4. 假设是"通信伙伴可以按 rail 分组"。
   AllReduce 的数据切分由算法决定、所有 rank 对称，所以能分组；
   All-to-All 的目标 rank 由 token 的 expert 路由决定（数据相关、近似随机），
   无法按 rail 分组，所以 87.5% 的流量被迫跨 spine。
5. **87.5%**（同 rail 只有 8/64 = 12.5%）。AllReduce 只需 spine 承担 1/8 = 12.5%，
   所以这里 spine 压力是 AllReduce 的 **7 倍**（0.875 / 0.125）。
6. 因为量只是两个轴之一。EP 变大时，跨 spine 比例从 12.5% 涨到 87.5%，
   跳数和 spine 拥塞在另一个轴上恶化，而且固定成本（每个小包 1.5 µs）不随量缩小。
   EP=256 时每 rank 只有 1.84 MB，但几乎全部跨 spine。
7. 因为问题不在库的算法，在**流量形态**：目标 rank 由数据决定，换库也改不了
   "87.5% 要跨 spine"这个事实。DeepEP 的解法是改数据流——把 EP 限制在 NVLink 域内，
   跨节点用分层传输，用两次拓扑友好的通信替代一次不友好的。

</details>

---

## 一页速记

```text
AllReduce 两个轴（别混着比）:
  时间 ≈ 步数 x 1.5us + 每 rank 量 / 带宽
  ring  量 2(N-1)/N*S   步 2(N-1)
  tree  量 2(N-1)/N*S   步 2log2(N)      <- 省步，不省量
  SHARP 量 2S           步 2             <- 省步 + 卸载归约，不省量（量反而更大）

数字锚点（400Gbps，1MiB / 256MiB）:
  N=1024 1MiB   ring 3.111ms  SHARP 0.045ms  -> 69x
  N=1024 256MiB ring 13.80ms  SHARP 10.74ms  -> 1.3x
  N=8    256MiB ring  9.42ms  SHARP 10.74ms  -> 0.9x（SHARP 更慢）

诊断口诀:
  消息 < 1MiB  -> 步的轴：换算法 / 减 N / 分桶并行
  消息 > 64MiB -> 量的轴：换拓扑 / rail 对齐 / 降精度

rail 拓扑的前提:
  通信伙伴能按 rail 分组（AllReduce 能，All-to-All 不能）
  AllReduce: spine 只承担 1/8
  All-to-All: 同 rail 概率 = 8/EP
    EP=8   0% 跨 spine      <- 最优区
    EP=64  87.5% 跨 spine   <- spine 压力是 AllReduce 的 7 倍
    EP=256 96.9% 跨 spine   <- rail 拓扑失效

EP 的两难:
  EP 大 -> 每 rank 量小（好）但跨 spine 多（坏）
  解法不是改拓扑，是分层数据流（24 章）
```
