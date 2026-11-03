# Adaptive DHiSparse：draft KV ring 与 verification staging

## 本次讲解位置

```text
章节：speculative-decoding
小节：Adaptive DHiSparse / Speculative Decoding over a Load-Adaptive KV Hierarchy
知识点：draft KV ring、hot cache、target tail 与 verification staging 的职责和生命周期
上次：DFlash 2 的候选集与路径选择
下次：DSpark 的 confidence scheduler 如何选择本轮验证长度 Kr
PTX：不适用
```

## 为什么现在讲这个

上一节只说明了 DFlash 2 如何从候选集中选出一条草稿路径，还没有回答这条路径在系统中需要哪些 KV 状态。这里最容易出现两个误解：把 drafter 的 KV 当成目标模型历史 KV 的一部分，或者把所有 hot cache miss 都归到一个叫 staging 的区域。两种理解都会导致错误的容量估算和缓存生命周期设计。Adaptive DHiSparse 把这些状态拆开管理，同时保留投机解码原有的提议和接受规则。

## 1. 心智模型：四类 KV 状态

投机解码一轮包含两个阶段：

```text
drafter 生成候选
    -> target 一次验证多个候选 query
    -> 接受一个前缀，丢弃其余候选
    -> 提交接受 token 的 target KV
```

在这个过程中，至少存在四类不同的 KV 状态：

| 状态 | 内容 | 是否长期保留 |
|---|---|---|
| target 历史 KV | 已经提交的上下文历史 | 是，可按 hot cache、optional history、host backing 分层 |
| target tail | 本轮刚计算、尚未归档的 anchor/draft KV | 否，接受后提交，拒绝后丢弃 |
| draft KV ring | drafter 可见窗口和当前候选块所需的 KV | 在固定窗口内持续存在，按环覆盖 |
| verification staging | 验证多个 query 时，临时放大的目标历史 KV 工作集 | 否，读者完成后复用 |

四者的语义、所有者和生命周期不同，不能当作同一块 cache。

## 2. HiSparse 与投机解码是正交的，但具体系统需要集成

稀疏 attention 和投机解码解决的是不同问题：

- HiSparse 决定目标模型在一次 forward 中访问哪些历史 KV，以及这些 KV 放在 host、hot cache 还是 optional history。
- 投机解码决定 drafter 提议多少 token，以及 target 如何一次验证这些 token。

因此，两者在算法层面可以组合。论文说原始 HiSparse 没有投机解码路径，描述的是具体系统的实现边界：

1. 原实现只处理普通 decode 的单个 query。
2. 它没有 drafter，因此不分配 draft KV。
3. 它没有针对多 query 验证的工作集管理，因此没有 verification staging。

Adaptive DHiSparse 增加的是这段系统级集成，而不是改变稀疏 attention 的数学定义。draft model 仍按自己的规则提议，target 仍按 rejection sampling 或模型原有规则接受前缀。

## 3. hot cache、target tail 和 optional history

HiSparse 把完整历史留在 host，并在 GPU 上保留紧凑索引和有限的目标 KV 工作集。

- **host backing**：已经提交的完整历史 KV，容量最大，访问延迟最高。
- **optional history**：负载较低时利用空闲 HBM 保留的额外历史，压力上来后可以回收。
- **hot cache**：每个 request 和 KV source 固定 \(B\) 行的近期工作集，使用 LRU 管理，服务普通 decode 的时间局部性。
- **target tail**：本轮新计算但还不能立即归档的 KV。投机候选是否被接受，决定这些行最终提交还是丢弃。

普通 decode 只有一个 query。它选出的 top-k 历史行如果不在 GPU 上，可以按 hot-cache 规则从 host 取入，并替换近期不用的行。这里的 hot cache 是长期存在、跨 decode step 保留的缓存。

## 4. draft KV 放在哪里

不能从通用投机解码直接推出“draft KV 一定放在目标模型的 paged KV block 里”。具体框架可以用 paged allocator、连续 allocation，也可以为 draft model 单独预留空间。paged 只描述物理分配方式，不决定 KV 的语义归属。

在 Adaptive DHiSparse 中，drafter 的 KV 是独立状态。它只需要覆盖 drafter 可见的历史窗口 \(W\) 和当前候选块，因此可以使用固定长度的环形区域。设 page 大小或对齐粒度为 \(p\)，单个候选块最多包含 \(K_{\max}\) 个位置，则每个 request 的 draft ring 行数为：

\[
R_d
=
p\left\lceil
\frac{W+p-1+K_{\max}}{p}
\right\rceil
\]

逻辑位置 \(t\) 的物理地址按环覆盖：

\[
\phi(i,t)=iR_d+(t\bmod R_d)
\]

其中 \(i\) 是 request slot。不同 slot 的地址范围互不重叠；同一 request 的两个同时存活位置不会落在同一个 ring 位置。

以论文的 GLM-5.3 + DFlash2 配置为例：

| 参数 | 数值 |
|---|---:|
| drafter 可见窗口 \(W\) | 2,048 |
| 对齐粒度 \(p\) | 64 |
| 最大候选宽度 \(K_{\max}\) | 8 |
| draft ring 行数 \(R_d\) | 2,176 |

计算过程为：

\[
R_d
=
64\left\lceil
\frac{2048+64-1+8}{64}
\right\rceil
=
64\times 34
=
2176
\]

这个容量只随 drafter 的固定窗口增长，不随 target 上下文是 32K 还是 100K 增长。论文给出的该布局中，draft ring 每个 session 增加约 `0.025 GiB`。

## 5. verification staging 到底解决什么问题

目标模型做投机验证时，会同时计算 \(K_r\) 个 query，其中 \(j=0\) 是 anchor，其余是候选 token。每个 query 都有自己的因果可见范围，并各自选择 top-k 历史行。记第 \(j\) 个 query 的选择为 \(S_{r,j}\)，则验证阶段的历史行需求是它们的并集：

\[
U_r=\bigcup_{j=0}^{K_r-1}S_{r,j},
\qquad |U_r|\le K_r k
\]

单个 query 选出的 \(k\) 行可能很容易放进 hot cache，但多个 query 的并集可能超过 \(B\) 行。比如 4 个 query 各自选择 3 行，12 次选择最多涉及 12 个不同位置。如果 hot cache 只有 8 行，它无法在同一轮验证中同时保留所有位置。

这里不能简单扩大每个 request 的 hot cache：hot cache 是长期预留，扩大会直接降低可并发 request 数。也不能在验证过程中随意驱逐仍在被其他 query 或后续 layer 使用的行。staging 提供的是第三种选择：保留固定大小的 hot cache，再为验证的即时 overflow 工作集准备临时 arena。

staging 的容量预留在论文中写成：

\[
M_{\mathrm{stage}}
=
\chi N K_{\max} k\,d_{\mathrm{KV}}
\]

其中：

- \(\chi\)：同时活跃的 payload source 或 layer 数；
- \(N\)：每张 GPU 的 request slot 数；
- \(K_{\max}\)：最大验证宽度；
- \(k\)：每个 query 选择的历史行数；
- \(d_{\mathrm{KV}}\)：一行 KV 的字节数。

这笔空间按最大验证工作集预留，但单个 round 只使用当前 query 实际选中的位置。重叠选择会减少需要搬运和复制的不同行数量。

## 6. staging 不是通用 L2 cache

staging 和 hot cache 的分工可以概括为：

```text
验证 query 的历史选择
        |
        v
tail / optional history / hot cache 已就绪?
        |                         |
        是                        否
        |                         |
直接使用已有地址          取入 GPU；若属于验证 overflow
                                  -> staging arena
        |                         |
        +-----------+-------------+
                    |
              attention readers
                    |
       需要进入 hot cache 的行先 copy back
                    |
               arena 可复用
```

因此，“staging 是 hot cache miss 的存放处”只说对了一半：

- 它的内容确实可能来自 hot cache miss；
- 它不是所有 hot cache miss 的通用归宿；
- 它专门承接验证阶段的即时 overflow；
- 如果某行只缺 hot cache、但仍能在 tail 或 optional history 中直接找到，它不需要进入 staging；
- staging 不在下一轮继续提供时间局部性，读者结束后就会复用。

更精确的定义是：**staging 是目标模型历史 KV 在投机验证期间使用的临时 overflow arena，不是 draft KV，也不是长期 cache。**

## 7. 具体 trace：12 次选择只对应 9 行

假设一轮验证有 4 个 query，每个 query 选择 3 行，hot cache 预算 \(B=8\)。为了只观察选择并集和临时 overflow，先忽略 tail 和 optional history 的命中差异：

| query | 有序选择 |
|---|---|
| \(Q_0\) | `[1, 2, 3]` |
| \(Q_1\) | `[1, 4, 5]` |
| \(Q_2\) | `[2, 6, 7]` |
| \(Q_3\) | `[4, 8, 9]` |

四次选择一共有 12 个 query-row 引用，但去重后的物理行是：

```text
U = {1, 2, 3, 4, 5, 6, 7, 8, 9}
```

共 9 行。若其中 8 行能由 hot cache 的预算承接，就仍有 1 行需要落在 staging。所有 query 可以共享同一物理行的 KV 字节，但不能合并它们各自的有序位置列表和 causal mask：

| 内容 | 是否共享 |
|---|---|
| 同一历史位置 `row=1` 的 K/V payload | 是 |
| 每个 query 的 top-k 顺序 | 否 |
| 每个 query 的 causal mask | 否 |
| 每个 query 的 attention 输出 | 否 |

这就是验证路径必须同时处理“去重搬运”和“多 query 独立 attention”的原因。

## 8. staging 的生命周期

staging 不是一次性复制后立刻释放。它要一直有效到所有目标 attention reader 和必要的 copy-back 完成。

一个可用的生命周期是：

1. resolver 合并同一 request、同一 payload source 下多个 query 的选择；
2. 已驻留的行直接返回 GPU 地址；
3. 验证 overflow 行进入 staging arena；
4. 相关 layer group 的 attention 读取暂存行；
5. 需要晋升到 hot cache 的行先复制回稳定地址；
6. 所有读者结束后，arena 才能被下一组 layer 或下一轮使用。

如果提前覆盖 arena，后续 layer 可能读到属于另一个请求或另一个位置的 KV。提前释放并复用也会让读者看到错误数据，表现为 attention 结果错误或随机波动，而不只是延迟增加。

论文中的 GLM-5.3 有 78 个 payload layer，但同时活跃的 layer group 最多只有 4 个，所以可以复用 4 组 arena，而不必给每层永久预留一块。这个复用把给定配置下的 transient staging 从 `24.0 GiB/GPU` 降到了 `1.23 GiB/GPU`。论文报告在其评估布局中，draft ring 与可复用 staging 合计为每个 session 增加约 `0.060 GiB`。

## 9. 常见误解

| 误解 | 正确理解 | 可观察后果 |
|---|---|---|
| draft KV 一定就是目标模型 paged KV 的一部分 | draft model 可以有独立 KV；是否 paging 只是 allocator 细节 | 用错层数、维度或生命周期，计算出的容量和地址范围都不对 |
| HiSparse 与投机解码正交，所以不需要额外实现 | 算法可组合，但 draft/verify 的调度、回滚和临时工作集仍需集成 | 误以为只打开 speculative 开关就能保留分层缓存收益 |
| verification staging 是所有 hot cache miss 的 L2 缓存 | staging 只服务验证阶段的 overflow；普通 miss 仍走原有层级和 hot-cache 规则 | 把 staging 当成长期缓存，容量估算和命中率都会失真 |
| staging 存的是候选 token 的 draft KV | staging 存的是验证所需的目标历史 KV | 把 drafter 和目标模型的状态混成一个 allocation |
| 多个验证 query 可以共享一次 attention | 它们只共享相同历史行的物理 payload，仍各自执行有序选择和 causal attention | attention 结果错误，尤其是不同候选前缀的可见范围不同 |
| arena 在 copy-back 前就可以立即复用 | 必须等所有 reader 和必要 copy-back 完成 | 读到被覆盖的数据，出现偶发但不一定可复现的错误 |

## 10. 自测

### 问题 1：Adaptive DHiSparse 里，draft KV 和目标历史 KV 是同一块存储吗？

答：不是。draft KV ring 属于 drafter，只覆盖固定窗口和当前候选块；目标历史 KV 由 host backing、optional history、hot cache 和 tail 管理。draft ring 即使按 page 对齐，也不等于目标模型的 paged KV。

### 问题 2：verification staging 主要存放什么？

答：投机验证期间，多个目标 query 的目标历史 KV 选择可能形成超过 hot cache 预算的并集。放不进 hot cache、又必须在当前验证中保持可读的 overflow 行临时进入 staging。

### 问题 3：为什么不能把所有 hot cache miss 都塞进 staging？

答：staging 是按最大验证工作集预留的瞬态空间，不保留跨 step 的时间局部性，过大会挤压并发容量。普通 miss 应继续由 tail、optional history 和 hot-cache/LRU 路径处理。

### 问题 4：多个 query 选中同一个历史位置时，什么可以复用，什么不能复用？

答：可以复用该位置的 K/V payload 和设备地址；不能复用各 query 的 top-k 顺序、causal mask 以及最终 attention 输出。

### 问题 5：为什么 DHiSparse 的 draft ring 大小与 target 上下文长度无关？

答：drafter 只读取固定窗口 \(W\) 和当前最多 \(K_{\max}\) 个候选位置。ring 只需覆盖这段可见区间，逻辑位置通过取模映射到固定物理位置；更早的目标历史由目标模型的 KV 层级负责。

## 参考资料

- [Adaptive DHiSparse: Speculative Decoding over a Load-Adaptive KV Hierarchy](https://arxiv.org/abs/2610.13708)
- [DFlash 2 的候选集与路径选择](03-dflash2-candidate-path-selection.md)
- [DFlash 的 SWA cache 窗口与 draft block 因果可见性](02-dflash-cache-window-and-block-visibility.md)
