# DFlash 2 的候选集与路径选择

## 本次讲解位置

```text
章节：speculative-decoding
小节：DFlash 2: Keep Drafting Parallel
知识点：每个位置保留候选集，按相邻位置分数选出一条草稿路径
上次：DFlash 的 SWA cache 窗口与 draft block 因果可见性
下次：DFlash 2 的 two-tap dynamic convolution 如何缓解 suffix decay
PTX：不适用
```

## 为什么现在讲这个

刚才的伪代码里，DFlash 2 把每个位置从“只挑一个 token”改成“保留一组 token，再选择一条路径”。如果把它直接叫作 beam search，容易误以为系统会同时维护和扩展多条完整前缀；那会高估它的计算和状态开销，也会误解它为什么仍然保持并行草稿。这里把候选打分、路径选择和 target 验证分开。

## 1. 先建立心智模型

DFlash 的 drafter 一次并行地产生整个 block 各位置的 logits。DFlash 2 在此基础上，每个位置保留 Top-K 候选，并考虑相邻位置的衔接关系：

```text
已验证 token x0
       │
       ▼
位置 0 候选 {a1, a2, ..., aK}
       │  每条相邻候选边的分数可并行计算
       ▼
位置 1 候选 {b1, b2, ..., bK}
       │
       ▼
位置 2 候选 {c1, c2, ..., cK}

路径选择：x0 → a* → b* → c*
```

草稿主体没有变成逐 token 自回归。选择路径时会从左向右走，因为当前位置选出的 token 是下一位置的前驱；但这只是沿长度为 `L` 的短 block 做轻量选择，不是每步重新运行 drafter 主干。

## 2. DFlash 与 DFlash 2 的差别

| 部分 | DFlash | DFlash 2 |
|---|---|---|
| 草稿主干 | 并行预测 block 中各位置 | 保持并行预测 |
| 候选 | 通常直接采用各位置的单个预测 | 每个位置保留 Top-K 候选；公开示例模型使用 K=16，K 不是普遍固定常数 |
| 位置间连贯性 | 各位置独立选出的 token 可能彼此不连贯 | 给相邻位置候选组合打分，再挑出连贯路径 |
| block 后段预测 | 可能出现 suffix decay | 还加入 two-tap dynamic convolution 缓解局部依赖不足；这是 DFlash 2 的另一项改动 |
| 最终正确性 | Target 验证草稿 | Target 仍然验证草稿；DFlash 2 的路径选择不是接受判决 |

因此 DFlash 2 可以概括为：**保持 DFlash 的并行草稿主干，提升候选利用率和 block 内局部信息传递。**它不是把整个 speculative decoding 流程替换掉。

## 3. 相邻候选如何打分

对位置 `t` 的候选 token `b`，原始 DFlash 分数记为 `U_t(b)`。给定前一个位置候选 `a`，DFlash 2 的公开说明使用如下形式的邻接打分：

\[
S_t(a,b) = U_t(b) + \langle A(a) \odot H(h_t), B(b) \rangle
\]

- `U_t(b)`：DFlash 原本对当前位置 token `b` 的打分。
- `A(a)`、`B(b)`：前驱和当前候选的紧凑表示。
- `H(h_t)`：由当前位置 drafter hidden state 生成的门控，调节哪些维度参与匹配。
- 第二项：估计 `b` 接在 `a` 后面是否合适。

公开模型说明中候选嵌入维度为 256。实际模型的候选数、维度和权重均由具体 checkpoint 决定。

## 4. 为什么它不是标准 beam search

**标准 beam search** 通常同时保留多个部分路径；每一层把这些路径继续扩展，再按累计分数剪枝，直到完成整个序列。

**DFlash 2 的公开路径选择**则先并行算好相邻候选对的分数，然后从最后一个已验证 token 出发，逐位置挑当前前驱最合适的后继。贪心解码可写成：

\[
y_t = \arg\max_{b \in C_t} S_t(y_{t-1}, b), \qquad y_{-1}=x_0
\]

这里不会保留多条 beam 前缀。它选的是一条局部贪心路径，也不保证等价于“最大化整条路径总分”的全局最优解。最终这条草稿路径还要交给 target 验证。

## 5. 选择阶段伪代码

下面展示数据流。`parallel_range` / `parallel_over` 是并行执行的说明性标记，不是 Python 标准库 API；后面的 CPU 代码给出一个可运行的路径选择小例子。

```python
def dflash2_draft(context_features, last_verified_token, block_len, top_k):
    # 整个 block 的 drafter 主干一次并行计算。
    hidden, logits = dflash_backbone_parallel(
        context_features=context_features,
        anchor_token=last_verified_token,
        block_len=block_len,
    )

    # 每个位置独立取 Top-K，保留 token 候选及其原始 DFlash 分数。
    candidates = []
    unary_scores = []
    for t in range(block_len):
        ids = topk_token_ids(logits[t], top_k)
        candidates.append(ids)
        unary_scores.append({token: logits[t, token] for token in ids})

    # 所有相邻位置的候选边分数并行计算。
    # 位置 0 的前驱是最后一个已验证 token。
    previous_candidates = [{last_verified_token}] + candidates[:-1]
    edge_scores = {}
    for t in parallel_range(block_len):
        for previous in parallel_over(previous_candidates[t]):
            for token in parallel_over(candidates[t]):
                edge_scores[t, previous, token] = (
                    unary_scores[t][token]
                    + pairwise_score(previous, token, hidden[t])
                )

    # 从 anchor 开始选一条路径。该短循环不是 beam search，也不重跑 drafter 主干。
    path = []
    previous = last_verified_token
    for t in range(block_len):
        token = max(
            candidates[t],
            key=lambda candidate: edge_scores[t, previous, candidate],
        )
        path.append(token)
        previous = token

    # 返回的是 draft；target verifier 再决定接受前缀及修正 token。
    return path
```

### 一条具体路径

假设每层只有两个候选。预先并行算好的分数如下：

| 位置 | 前驱 | 候选分数 | 选择 |
|---:|---|---|---|
| 0 | 已验证 token `x0` | `A: 0.80`, `B: 0.91` | `B` |
| 1 | `B` | `C: 0.70`, `D: 1.10` | `D` |
| 2 | `D` | `E: 0.60`, `F: 0.95` | `F` |

最终草稿是 `B → D → F`。位置 2 只需比较“前驱为 D 时”的边分数；路径选择器不再保留 `A`、`C` 等其它部分路径。

## 6. 可运行的 CPU 小例子

以下代码只复现“已有相邻边分数后，贪心地从左向右选择一条路径”，不模拟神经网络、Top-K kernel 或 target verifier。

```python
from collections.abc import Mapping, Sequence


def choose_path(
    candidates: Sequence[Sequence[str]],
    edge_scores: Sequence[Mapping[str, Mapping[str, float]]],
    anchor: str,
) -> list[str]:
    if len(candidates) != len(edge_scores):
        raise ValueError("one edge-score table is required per position")

    path: list[str] = []
    previous = anchor
    for position, candidate_ids in enumerate(candidates):
        if not candidate_ids:
            raise ValueError(f"position {position} has no candidates")
        scores_from_previous = edge_scores[position].get(previous)
        if scores_from_previous is None:
            raise ValueError(
                f"missing edge scores at position {position} for {previous!r}"
            )
        missing = [token for token in candidate_ids if token not in scores_from_previous]
        if missing:
            raise ValueError(f"missing candidate scores at position {position}: {missing}")

        selected = max(candidate_ids, key=scores_from_previous.__getitem__)
        path.append(selected)
        previous = selected

    return path


def main() -> None:
    candidates = [["A", "B"], ["C", "D"], ["E", "F"]]
    edge_scores = [
        {"x0": {"A": 0.80, "B": 0.91}},
        {
            "A": {"C": 0.90, "D": 0.80},
            "B": {"C": 0.70, "D": 1.10},
        },
        {
            "C": {"E": 0.90, "F": 0.50},
            "D": {"E": 0.60, "F": 0.95},
        },
    ]

    path = choose_path(candidates, edge_scores, anchor="x0")
    print("selected path:", " -> ".join(path))


if __name__ == "__main__":
    main()
```

预期输出：

```text
selected path: B -> D -> F
```

执行轨迹：`x0` 到第 0 层选 `B`；到第 1 层只查看 `B` 的后继分数，选 `D`；到第 2 层只查看 `D` 的后继分数，选 `F`。每一步只前进一个位置，所以路径遍历是 `O(LK)`；相邻层所有候选边的打分量级约为 `O(LK^2)`，但这些边分数可并行计算。实际运行耗时还取决于模型的 top-k、矩阵运算和 kernel 实现。

## 7. 与 DeepInfer DFlash 的边界

本节的 DFlash 2 指 Inco AI 公开介绍的 **DFlash 2: Keep Drafting Parallel**。前面 DeepInfer 笔记则固定到 DeepInfer commit `774ce244`，讲的是 DeepSeek-V4 路径中的 DFlash/DSpark 和该仓库的 cache、采样、调度实现。两者名称相关，但不是同一份代码说明。

候选路径选择本身只处理 drafter 的候选 token 与分数；不能据此推断 target KV 与 drafter cache 共用，也不能推断 DeepInfer 的该 commit 已实现 Top-K 路径选择。具体 cache 布局和功能支持必须看相应 checkpoint 与 serving engine 的实现。

## 8. 常见误解与症状

| 误解 | 正确理解 | 可能的错误结论 |
|---|---|---|
| “候选集路径选择就是 beam search” | 它不维护多条部分路径；公开描述是预计算邻边分数后逐位置贪心走一条路径 | 误估需要的 beam 状态和扩展开销 |
| “每个位置的候选组合都必须串行算” | 邻接候选边分数先并行计算；只有路径决策沿 block 前进 | 误以为整个 drafter 又变成自回归 |
| “选中的路径就是最终输出” | 它仍是 draft，target verifier 决定接受前缀及拒绝修正 | 把候选选择当成正确性判定 |
| “DFlash 2 代表所有 DFlash 实现都有相同 cache” | 算法名不能保证 checkpoint、KV layout 或 engine 代码相同 | 把公开模型机制错误套到 DeepInfer 源码 |

## 自测

1. DFlash 2 的路径选择需要同时保存多条前缀并逐层剪枝吗？
   答：不需要；公开描述是先计算邻接候选分数，再从 anchor 开始选一条路径。
2. 哪部分并行、哪部分从左到右？
   答：所有位置的 drafter 前向和邻接候选边打分可并行；路径选择按短 block 从左向右进行。
3. 路径选择器选出 `B → D → F` 后，这些 token 会自动成为最终输出吗？
   答：不会；它们是草稿，仍需 target verifier 验证。
4. Top-K 是否固定必须为 16？
   答：不是；公开模型示例使用 16，实际 K 由模型与实现决定。

## 参考资料

- [DFlash 2: Keep Drafting Parallel, Inco AI](https://inco.ai/blog/dflash2/)
- [DFlash: Block Diffusion for Flash Speculative Decoding](https://arxiv.org/abs/2602.06036)
- [DeepInfer DFlash speculative decoding notes, pinned to commit 774ce244](../models/deepinfer/speculative-decoding.md)
