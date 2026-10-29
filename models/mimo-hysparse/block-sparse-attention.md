# HySparse：Block Sparse Attention 如何消费 Top-K Block Indices

> 这份笔记只解释用户提供的 HySparse 架构图所表达的计算语义。图中的 Block Sparse Attention 不是 SWA 的下游，二者和 Full Attention 之间是跨层 K/V 共享关系。公开 PyTorch 模型实现主要暴露 SWA/GA 路径，因此本文不把下述 gather 步骤声称为某个具体 CUDA kernel 的源码行为。

## 本次讲解位置

```text
章节：models / MiMo HySparse
小节：Block Sparse Attention 的候选 KV gather
知识点：Q' 如何消费 Full Attention 产生的 Top-K block indices
上次：MiMo-V2.6-Flash 的 SWA / GA 可见范围与 GQA
下次：logical block id 到 paged KV、GQA head 和分布式 rank 的 metadata 映射
PTX：不适用；本节讨论模型 attention 的数据流
```

## 为什么现在讲这个

前面的 SWA/GA 笔记只说明了一个 query 能看多远，没有解释 HySparse 为什么同时存在 SWA 和 Block Sparse Attention。把这张图误读成“SWA 先产生 `Q'`，然后继续做 sparse attention”，会得到错误的执行顺序，也会误以为 `Top-K indices` 是 query 的编号。

实际设计解决的是另一个问题：稀疏层既要保留局部细节，又要访问 Full Attention 层判断出的远距离重要 block。理解这个分工，才能判断 causal mask、SWA 与 sparse KV 重叠以及 gate 融合分别应该在哪一步发生。

## 心智模型：Full Attention 找位置，Sparse Attention 读内容

按图，一个 Full Attention 层向多个 sparse 层提供三类信息：

```text
Full Attention 层
  ├─ K_full
  ├─ V_full
  └─ Block indices = TopK(block score derived from S)

后续每个 sparse 层
  ├─ SWA 分支：自己的 Q, K_local, V_local
  └─ Block Sparse 分支：自己的 Q' + Full 层的 K_full, V_full, block indices
```

Full Attention 在这里像“侦察层”：它负责决定远处的哪些 KV block 值得读取。Sparse 层不需要再次对完整历史做全局打分，而是让本层 `Q'` 只在被选中的 block 上做 token-level attention。

`Q'` 不是 SWA 的输出。图中它是 sparse 层输入经过另一个线性投影得到的 query：

```text
Q' = X W_q'
```

Full Attention 输出的也不是 query，而是供后续层共享的 `K_full`、`V_full` 和 block indices。因此不存在“`Q'` 命中某个 top-k index”的匹配动作。Indices 描述 K/V 的位置，`Q'` 是读取这些位置的 query。

## 张量与索引契约

忽略 batch 维，一个常见形状如下：

```text
X        : [S, D]                        sparse 层输入
Q'       : [S, Hq, Dqk]                  Block Sparse 的 query
K_full   : [T, Hkv, Dqk]                 从 Full Attention 层共享
V_full   : [T, Hkv, Dv]                  从 Full Attention 层共享
I        : [S, Hq, k] 或压缩后的 [k]      每个 query/head 的 block id
block_sz : 64
```

不同实现可以把 indices 放在 layer、request、query 或 head 维度上，也可能只保存一个压缩的共享集合。数学上的契约只有两条：

1. `I` 中的 id 必须与 `K_full`、`V_full` 使用同一套 block 编号。
2. 每个 query 只对 `I` 指向的 K/V 做 attention，不能用其他层的 cache 或错误的 head 解释这组 id。

逻辑上的 gather 写作：

\[
K_{\mathrm{sel}} = \operatorname{Gather}(K_{\mathrm{full}}, I),\qquad
V_{\mathrm{sel}} = \operatorname{Gather}(V_{\mathrm{full}}, I)
\]

若 block size 为 \(B_s\)，一个 block id 为 \(b\)，其逻辑 token 范围是：

\[
[bB_s,\ (b+1)B_s-1]
\]

真实 kernel 不一定先把 `K_sel` 复制成连续张量。它可以根据 index 和 page table 直接发起 paged load；这里的 `Gather` 表示 K/V 的候选集合，不要求物理搬运。

## 一次 Block Sparse Attention 的计算

对位置 \(i\) 的 query 和头 \(h\)，先得到候选集合 \(\mathcal{C}(i,h)\)：

\[
\mathcal{C}(i,h)=\{j\mid \lfloor j/B_s\rfloor\in I_{i,h}\}
\]

然后在候选集合内部执行 dense QK/PV：

\[
s_{i,j}^{(h)}
=
\frac{(q'_i^{(h)})^\top k_j^{(g(h))}}{\sqrt{D_{qk}}}
+M_{i,j}^{\mathrm{causal}}
\]

\[
O_{\mathrm{bsa},i}^{(h)}
=
\sum_{j\in\mathcal{C}(i,h)}
\operatorname{softmax}_{j}(s_{i,j}^{(h)})v_j^{(g(h))}
\]

`dense` 只表示在选中 block 内部逐 token 计算，不表示读取完整上下文。若选 \(k=4\) 个 block、每块 64 个 token，候选上限是 256 个 token，而不是整段序列。

### 同时发生 SWA

SWA 分支使用另一组本层 K/V：

\[
O_{\mathrm{swa},i}^{(h)}
=
\operatorname{Attention}_{\mathrm{window}}
\left(Q_i^{(h)},K_{\mathrm{local}},V_{\mathrm{local}}\right)
\]

图中的两个分支独立 softmax，再通过 sigmoid gate 融合：

\[
O_i
=
\sigma(g_{\mathrm{swa}})\odot O_{\mathrm{swa},i}
+
\sigma(g_{\mathrm{bsa}})\odot O_{\mathrm{bsa},i}
\]

因此，SWA 和 sparse 候选有交集时，交集中的 token 会在两个分支各计算一次。这不是重复 bug，也不是对两路 key 拼起来做一次 softmax。

## 具体数值轨迹

令：

```text
序列长度 T       = 12
block size Bs   = 4
query 位置 i    = 10
SWA 窗口 W      = 6
Top-K block ids = [0, 2]
```

Block 划分是：

| block id | token 范围 |
|---:|---:|
| 0 | 0–3 |
| 1 | 4–7 |
| 2 | 8–11 |

Block Sparse 的步骤：

| 步骤 | 结果 |
|---|---|
| gather block 0 和 2 | `[0,1,2,3, 8,9,10,11]` |
| 去掉 query 之后的位置 11 | `[0,1,2,3, 8,9,10]` |
| SWA，位置 10 的前 6 个位置 | `[5,6,7,8,9,10]` |
| 两支交集 | `[8,9,10]` |
| 只属于 Block Sparse | `[0,1,2,3]` |
| 只属于 SWA | `[5,6,7]` |

位置 10 的 query 不能看位置 11。即便 block 2 被 Full Attention 选中，也必须先应用 causal mask。`[8,9,10]` 同时出现在两路结果中，最后由两个 gate 分支融合。

这组 trace 同时说明：Top-K 是 **KV block 的选择**。query 位置 10 是否落在 block 2 中，并不会触发额外分支；它只决定该 query 自身的 mask 和它能读取哪些历史位置。

## 完整可运行示例

下面只验证索引语义，不实现矩阵乘和 softmax，因此不依赖 PyTorch。

文件：`models/mimo-hysparse/block_gather_trace.py`

```python
from typing import Iterable, List, Sequence

BLOCK_SIZE = 4
SEQUENCE_LENGTH = 12
QUERY_POSITION = 10
WINDOW_SIZE = 6
TOP_K_BLOCKS = (0, 2)


def gather_blocks(
    block_ids: Iterable[int],
    block_size: int,
    sequence_length: int,
) -> List[int]:
    """将逻辑 block id 展开成 token 位置，并裁掉最后一个短 block。"""
    result: List[int] = []
    for block_id in block_ids:
        start = block_id * block_size
        end = min(start + block_size, sequence_length)
        result.extend(range(start, end))
    return result


def sliding_window(
    query_position: int,
    window_size: int,
) -> List[int]:
    """返回包含当前 token 的因果 SWA 窗口。"""
    start = max(0, query_position - window_size + 1)
    return list(range(start, query_position + 1))


def apply_causal_mask(
    positions: Sequence[int],
    query_position: int,
) -> List[int]:
    return [position for position in positions if position <= query_position]


def main() -> None:
    gathered = gather_blocks(
        TOP_K_BLOCKS, BLOCK_SIZE, SEQUENCE_LENGTH
    )
    sparse_visible = apply_causal_mask(gathered, QUERY_POSITION)
    swa_visible = sliding_window(QUERY_POSITION, WINDOW_SIZE)

    sparse_set = set(sparse_visible)
    swa_set = set(swa_visible)

    print(f"gathered KV       = {gathered}")
    print(f"sparse after mask = {sparse_visible}")
    print(f"SWA visible       = {swa_visible}")
    print(f"overlap           = {sorted(sparse_set & swa_set)}")
    print(f"sparse only       = {sorted(sparse_set - swa_set)}")
    print(f"SWA only          = {sorted(swa_set - sparse_set)}")

    assert sparse_visible == [0, 1, 2, 3, 8, 9, 10]
    assert swa_visible == [5, 6, 7, 8, 9, 10]
    assert sorted(sparse_set & swa_set) == [8, 9, 10]


if __name__ == "__main__":
    main()
```

运行：

```bash
python3 models/mimo-hysparse/block_gather_trace.py
```

预期输出：

```text
gathered KV       = [0, 1, 2, 3, 8, 9, 10, 11]
sparse after mask = [0, 1, 2, 3, 8, 9, 10]
SWA visible       = [5, 6, 7, 8, 9, 10]
overlap           = [8, 9, 10]
sparse only       = [0, 1, 2, 3]
SWA only          = [5, 6, 7]
```

这个脚本在 macOS、Linux 和 Windows 的普通 Python 3 环境都可运行。它验证的是索引和 mask 语义，不验证 GPU kernel、paged cache 地址或 GQA 映射。

## 常见误读与可观察后果

| 误读 | 实际语义 | 常见症状 |
|---|---|---|
| `Q'` 是 SWA 输出 | `Q'` 是 sparse 层输入的独立投影 | 误把两路写成串行，gate 和 attention 的输入全错 |
| Top-K index 是 query 编号 | Index 选择的是 K/V block | 代码会去错误的 cache 位置取 K/V，输出随机偏离 reference |
| 选中 block 后不需要 causal mask | 只筛 block，block 内仍需 token mask | 当前 query 会读到未来 token，prefill 与训练不一致 |
| SWA 和 sparse 重叠 token 要去重 | 两路独立 attention，后再 gate 融合 | 若强行拼接 key 做一次 softmax，归一化范围和模型定义不一致 |
| gather 一定意味着复制连续 tensor | kernel 可直接按 page/block id 加载 | 多一次无意义的 gather buffer 和拷贝，增加带宽与延迟 |
| Top-K 保证 block 有完整 64 个 token | 尾块可能不满，且 slot 可能无效 | tail 或 padding token 参与 softmax，短序列结果错误 |

实现时还要确认 indices 是逻辑 block id 还是物理 page id。如果前者没有经过 page table 转换，后续 rank、cache pool 或 prefix sharing 都可能读错位置。

## 自我检查

1. `Q'` 是 SWA 的输出吗？
   - 不是。按图，它由 sparse 层输入单独投影，只作为 Block Sparse Attention 的 query。
2. Top-K indices 指向 query 还是 K/V？
   - 指向 K/V block。每个 query 使用这组 block id 找到候选 K/V。
3. 候选 block 被选中后为什么还要 causal mask？
   - Block 是粗粒度选择；block 内仍可能出现 query 之后的 token。
4. SWA 与 Block Sparse 的交集如何处理？
   - 两路分别计算 attention 和 softmax，再经过 sigmoid gate 加权相加，不做去重。
5. 这种 Block Sparse Attention 是否等于在完整序列上做 full attention？
   - 不等于。它是“先按 index 选候选 KV，再在候选集合内部做 dense attention”。

## 进度

已经覆盖：

- [x] Full Attention 提供共享 K/V 与 Top-K block indices。
- [x] `Q'` 经过独立投影，在候选 KV block 上执行 Block Sparse Attention。
- [x] SWA 与 sparse 分支独立 softmax，重叠 token 在 gate 前保留两份贡献。
- [x] causal mask、尾块和逻辑 index 到物理 page 的边界。

下一知识点：

- [ ] 把 logical block id 落到 paged KV：block table、page size、GQA head 映射、DP/EP rank 与分布式 cache 地址如何组成最终读取坐标。

## 来源与边界

- 用户提供的 HySparse 架构图（对话附件，2026-10-29）。本文的双分支、Q'、共享 K/V、Top-K block indices 和 gate 融合均按该图解释。
- [XiaomiMiMo/MiMo-V2.6-Flash-RL 模型卡](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Flash-RL) 与 [Technical Report](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Flash-RL/blob/main/MiMo_V2_6_technical_report.pdf) 用于 SWA/GA 背景；公开 modeling snapshot 未提供图中 Block Sparse branch 的完整 kernel 实现，因此本文不声称具体的 page layout、split 策略或 CUDA kernel 名称。
