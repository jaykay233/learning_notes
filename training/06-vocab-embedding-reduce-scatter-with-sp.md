# 开启 SP 后，Vocab Embedding 为什么用 Reduce-Scatter

## 本次讲解位置

- 章节：`training` / 训练中的并行策略
- 小节：Megatron `VocabParallelEmbedding` 的输出布局
- 知识点：SP 下用 reduce-scatter 代替 all-reduce 完整输出
- 上次：Sequence Parallelism 的 scatter / gather 布局转换
- 下次：沿 Transformer block 继续追踪 embedding 输出与后续 TP 层的布局

## 为什么现在讲这个

看到“开 SP 后 embedding 不用 all-reduce”很容易误会成“embedding 查表只在某一张卡上算，所以别的卡什么都不用做”。实际不是这样：词表被 TP 切开后，每张卡只拥有一部分词向量；各卡算出的是**部分结果**。变化的是输出需要的布局：不开 SP 时每张卡都需要完整序列输出；开 SP 后每张卡只需要自己负责的序列片段。于是 collective 从 all-reduce 改为 reduce-scatter。

## 先给结论

**不是不做归约，而是不再把归约后的完整结果复制给每一张 TP 卡。**

- 不开 SP：`reduce_from_tensor_model_parallel_region` 做 all-reduce，得到完整 `[B,S,H]` 输出，并在 TP ranks 上复制。
- 开 SP：`reduce_scatter_to_sequence_parallel_region` 把“跨词表分片的部分结果求和”与“沿序列维切分”合到一起；每 rank 只拿 `[B,S/P,H]` 的序列片段。

所以“SP 下不用 all-reduce”更准确的说法是：**不再 all-reduce 成完整副本；改用 reduce-scatter。reduce 的求和仍然存在。**

## Vocab Parallel Embedding 为何产生部分结果

设词表大小 `V=8`，TP=2，按词表维切分：

```text
rank 0 持有 token ID [0, 4) 的 embedding 行
rank 1 持有 token ID [4, 8) 的 embedding 行
```

同一个 token ID 会被送到 TP 组内各 rank。每张卡只在自己的词表区间内查表：

- 若 token 属于本卡词表分片，本卡产生对应 embedding；
- 若不属于，本卡把该 token 位置的输出置零。

例如输入 token IDs `[1, 5, 2, 7]`：

```text
rank 0 partial: [emb1, 0,    emb2, 0   ]
rank 1 partial: [0,    emb5, 0,    emb7]
```

对两个 rank 的 partial output 按元素求和，才得到正确完整序列：

```text
[emb1, emb5, emb2, emb7]
```

虽然每个 token 的 embedding 行只在一个词表 shard 上非零，但那个 shard 的 rank 未必是 SP 中负责该 token 序列位置的 rank。还需要 collective 把正确向量送到序列 owner。

## 对比两种输出布局

### 不开 SP：all-reduce 得完整副本

对 partial outputs 求和，并把求和后的完整结果交给所有 TP rank：

```text
rank 0: [emb1, emb5, emb2, emb7]
rank 1: [emb1, emb5, emb2, emb7]
每卡形状: [B, S, H]
```

之后的代码/层可以按 TP 路径消费完整序列。若后面再切为 SP，等于多经历一次完整结果的复制与布局转换。

### 开 SP：reduce-scatter 直接得到序列分片

reduce-scatter 仍先对词表分片的 partial outputs 求和，但不把完整结果复制给每张卡，而是直接按序列维分发：

```text
rank 0: [emb1, emb5]
rank 1: [emb2, emb7]
每卡形状: [B, S/TP, H]
```

这里 rank 0/1 接收的是**序列位置分片**，不一定是自己词表分片里产生的 embedding。词表轴与序列轴是两种不同的 rank 布局。

| 路径 | Collective | 归约求和 | 每卡 forward 输出 |
|---|---|---|---|
| 不开 SP | all-reduce | 有 | 完整序列，TP ranks 上复制 |
| 开 SP | reduce-scatter | 有 | 本 rank 的序列片段 |

All-reduce 可以理解为 `reduce + all-gather`：先求和，再把完整求和结果复制给所有 rank。SP 不需要后面的 all-gather 复制，因此直接 reduce-scatter 到需要的序列分片。

## 与 Megatron 源码对照

在 Megatron-LM 的 `megatron/core/tensor_parallel/layers.py` 中，`VocabParallelEmbedding.forward` 的关键分支可概括为：

```python
if self.reduce_scatter_embeddings:
    # output_parallel 原本是 [B, S, H]，换成 [S, B, H]
    output_parallel = output_parallel.transpose(0, 1).contiguous()
    output = reduce_scatter_to_sequence_parallel_region(
        output_parallel, group=self.tp_group
    )
elif self.tp_group.size() > 1:
    output = reduce_from_tensor_model_parallel_region(
        output_parallel, group=self.tp_group
    )
else:
    output = output_parallel
```

`reduce_scatter_embeddings` 决定采用哪条路径；具体版本里这个标志如何从模型配置传入，要看构造 `VocabParallelEmbedding` 的调用处。`transpose(0, 1)` 是为了让 sequence 成为首维，使按首维进行的 reduce-scatter 正好切序列。代码注释里提到改变数据格式以避免显式 transpose 的写法/优化会随版本不同；读代码时以当前 checkout 为准。

## 完整可运行的两 rank 语义模拟

以下程序在 CPU 上构造两个词表 shard，不启动分布式进程。它显式展示“词表分片产生 partial → 求和 → all-reduce 副本或 reduce-scatter 序列分片”的差别。保存为 `embedding_sp_demo.py`：

```python
import torch


def vocab_parallel_partial(token_ids, full_weight, world_size):
    """模拟每个 TP rank 只查自己的 vocab 行，不属于本 shard 的位置填零。"""
    vocab_size, hidden_size = full_weight.shape
    assert vocab_size % world_size == 0
    vocab_per_rank = vocab_size // world_size
    partials = []
    for rank in range(world_size):
        start = rank * vocab_per_rank
        end = start + vocab_per_rank
        local = torch.zeros(*token_ids.shape, hidden_size, dtype=full_weight.dtype)
        mask = (token_ids >= start) & (token_ids < end)
        if mask.any():
            local[mask] = full_weight[token_ids[mask]]
        partials.append(local)
    return torch.stack(partials, dim=0)


def main():
    world_size = 2
    # embedding[token_id] = [token_id, token_id + 0.5]
    full_weight = torch.stack(
        [torch.arange(8, dtype=torch.float32),
         torch.arange(8, dtype=torch.float32) + 0.5],
        dim=1,
    )
    token_ids = torch.tensor([[1, 5, 2, 7]])  # [B=1, S=4]
    partial = vocab_parallel_partial(token_ids, full_weight, world_size)
    print("rank 0 partial:", partial[0, 0].tolist())
    print("rank 1 partial:", partial[1, 0].tolist())

    # 两种 collective 的 reduce 部分相同：对 vocab shards 的 partial 求和。
    reduced = partial.sum(dim=0)  # [B, S, H]

    # all-reduce = reduce + 将完整结果复制给每个 TP rank。
    all_reduce_output = reduced.unsqueeze(0).expand(world_size, -1, -1, -1)
    print("all-reduce rank 0:", all_reduce_output[0, 0].tolist())
    print("all-reduce rank 1:", all_reduce_output[1, 0].tolist())

    # reduce-scatter = reduce + 沿 sequence 维切分，每卡只留自己的片段。
    assert reduced.shape[1] % world_size == 0
    sequence_chunks = reduced.chunk(world_size, dim=1)
    reduce_scatter_output = torch.stack(
        [sequence_chunks[rank].squeeze(1) for rank in range(world_size)], dim=0
    )
    print("reduce-scatter rank 0:", reduce_scatter_output[0, 0].tolist())
    print("reduce-scatter rank 1:", reduce_scatter_output[1, 0].tolist())


if __name__ == "__main__":
    main()
```

运行：`python embedding_sp_demo.py`（需要 PyTorch）。预期输出：

```text
rank 0 partial: [[1.0, 1.5], [0.0, 0.0], [2.0, 2.5], [0.0, 0.0]]
rank 1 partial: [[0.0, 0.0], [5.0, 5.5], [0.0, 0.0], [7.0, 7.5]]
all-reduce rank 0: [[1.0, 1.5], [5.0, 5.5], [2.0, 2.5], [7.0, 7.5]]
all-reduce rank 1: [[1.0, 1.5], [5.0, 5.5], [2.0, 2.5], [7.0, 7.5]]
reduce-scatter rank 0: [[1.0, 1.5], [5.0, 5.5]]
reduce-scatter rank 1: [[2.0, 2.5], [7.0, 7.5]]
```

这里 `partial.sum(dim=0)` 是 collective 的求和语义；程序用本地 PyTorch 操作模拟其结果，不测量通信成本。

## 常见误区

1. **说“SP 让 embedding 不用通信”**：错。sum + 分发仍需要跨 rank 通信，只是从 all-reduce 换为 reduce-scatter。
2. **以为每个 token 只在一个词表 shard 非零，所以不需 reduce**：那只说明数值上只有一个 shard 提供有效值；有效值仍须到达负责该序列片段的 rank。
3. **把 all-reduce 与 reduce-scatter 当成相同输出**：两者的求和部分相同，但 all-reduce 让每卡拿完整序列；reduce-scatter 让每卡拿不同序列片段。
4. **把 vocab shard owner 与 sequence shard owner 当成同一 rank**：词表分片轴和序列切分轴含义不同，rank 归属不必相同。
5. **把 `reduce_scatter_embeddings=True` 直接等同所有 SP 版本/所有 embedding 实现**：这是具体实现开关，需检查模型构造和 Megatron 版本。

## 自测

1. 开 SP 后归约求和还在吗？——在，通常包含在 reduce-scatter 中。
2. all-reduce 与 reduce-scatter 的核心输出区别是什么？——前者每 rank 得完整求和张量；后者归约后按维度分片，每 rank 只得一片。
3. 为什么 vocab embedding 的有效结果要跨 rank 传？——拥有 token 词向量的 vocab shard rank 不一定是负责该 token 序列位置的 SP rank。
4. 为什么源码先把 `[B,S,H]` 换成 `[S,B,H]`？——让 sequence 成为首维，以便 reduce-scatter 按序列维切分。
