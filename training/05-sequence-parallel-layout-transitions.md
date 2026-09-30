# Sequence Parallelism 的输入/输出布局转换

## 本次讲解位置

- 章节：`training` / 训练中的并行策略
- 小节：Megatron `mappings.py` 的 SP autograd mappings
- 知识点：`_ScatterToSequenceParallelRegion` 与 `_GatherFromSequenceParallelRegion`
- 上次：SP 是在 TP 组中按序列分片、降低某些激活的重复保存
- 下次：沿着 Transformer block 追踪 SP、LayerNorm、Column/Row Parallel Linear 的完整布局

## 为什么现在讲这个

只知道“SP 会切序列”还解释不了代码里的 all-gather / reduce-scatter 为什么出现在这些位置。若把它们当作随意插入的通信，很容易误以为 backward 只是把 forward 操作原样重复。实际上这两个 `autograd.Function` 是**布局转换的边界**：forward 把张量从一种 rank 布局变成另一种，backward 则执行相应映射的梯度变换；归约要不要发生，取决于后续计算是 TP 分片计算还是每卡重复计算。

## 先解读截图里的 “split first / all-gather first”

截图表格中的 “first” 指 **first dimension（张量第 0 维 / 首维）**，不是“先做 split，再做 all-gather”的执行顺序。Megatron 源码对应 `_split_along_first_dim` 和 `_gather_along_first_dim`。典型 transformer hidden state 可能排成 `[sequence, batch, hidden]`，此时首维正是 sequence；但如果调用者的张量维度顺序不同，还是要以该张量实际布局为准。

截图两行可按 forward / backward 读：

| Autograd mapping | Forward | Backward（默认常见路径） | 布局含义 |
|---|---|---|---|
| `_ScatterToSequenceParallelRegion` | 沿首维切片，每个 TP rank 留自己的序列 chunk | 沿首维 all-gather 各 rank 的梯度 chunk | 将复制/完整序列输入切成 SP 布局 |
| `_GatherFromSequenceParallelRegion` | 沿首维 all-gather 序列 chunk，得到完整序列 | 沿首维 reduce-scatter：先把各 rank 对完整张量的梯度相加，再切回本 rank 序列 chunk | SP → 需要完整序列的 TP 布局 |

## 心智模型：把 TP ranks 想成 4 个序列抽屉

令 TP group 大小 `P=4`，序列长度 `S=8`。序列维切分后，每 rank 有 `S/P=8/4=2` 个 token 位置：

```text
完整序列: [t0 t1 t2 t3 t4 t5 t6 t7]
rank 0:   [t0 t1]
rank 1:   [t2 t3]
rank 2:   [t4 t5]
rank 3:   [t6 t7]
```

- **scatter to SP**：每张卡原先都有完整输入时，各自只留下对应抽屉里的 2 个 token。
- **gather from SP**：把 4 个抽屉沿 sequence 维拼起来，每张卡都得到长度 8 的完整序列，供需要该布局的 TP 运算使用。

## 为什么 backward 不是简单反向复制

### 1. Scatter 的 backward 是 all-gather

Scatter forward 把互不重叠的序列区间分给不同 rank。反向时，每个 rank 算出自己那一小段输入的梯度；要得到原先完整输入对应的梯度，就把各 rank 不同区间的梯度**拼接**起来。这里是 all-gather，不是求和，因为各 rank 对应的是不同 token 坐标。

### 2. Gather 的 backward 默认是 reduce-scatter

Gather forward 把不同 rank 持有的序列 chunk 拼成完整序列，并让后续 TP 运算能在各 rank 上处理完整序列布局。反向时，每个 TP rank 可能只算了输出梯度的一部分贡献，所以要先对同一全局张量的梯度**求和**，再按 sequence 切回各 rank 的输入 shard：这就是 reduce-scatter。

Megatron 的 `_GatherFromSequenceParallelRegion` 有 `tensor_parallel_output_grad` 参数：

- 为 `True`（常见默认）时 backward 使用 reduce-scatter，适用于后续 TP 计算各 rank 提供不同梯度贡献的情况。
- 为 `False` 时 backward 只 split，不做跨 rank 求和，适用于后续计算是重复/复制的情况。

因此截图写的 “reduce-scatter first” 是默认 TP-gradient 路径的简写，不是所有调用配置都无条件如此。

## 完整可运行的单进程语义模拟

保存为 `sp_layout_demo.py`。它把 4 个 rank 的局部张量放在一个 Python 进程里，模拟 collective 的数据与 autograd 语义；它**不启动分布式进程，也不测量真实网络通信性能**。

```python
import torch
from torch.autograd import Function


class ScatterToSequenceParallelRegion(Function):
    """输入是每 rank 一份完整序列，输出是每 rank 的本地 sequence chunk。"""

    @staticmethod
    def forward(ctx, replicated_full, world_size):
        # replicated_full: [world_size, sequence]
        ctx.world_size = int(world_size)
        assert replicated_full.ndim == 2
        assert replicated_full.shape[0] == ctx.world_size
        sequence = replicated_full.shape[1]
        assert sequence % ctx.world_size == 0
        chunks_per_rank = replicated_full.chunk(ctx.world_size, dim=1)
        # 对应每个 rank 取自己负责的那一块。
        return torch.stack(
            [chunks_per_rank[rank][rank] for rank in range(ctx.world_size)], dim=0
        )

    @staticmethod
    def backward(ctx, grad_local):
        # 每行是一个 rank 对自己 sequence chunk 的梯度；拼回完整 sequence。
        full_grad = torch.cat(tuple(grad_local.unbind(dim=0)), dim=0)
        # 原输入在每个 rank 都是完整序列，所以每个 rank 得到相同的完整梯度。
        replicated_grad = full_grad.unsqueeze(0).expand(ctx.world_size, -1).contiguous()
        return replicated_grad, None


class GatherFromSequenceParallelRegion(Function):
    """输入是每 rank 的 sequence shard，输出是每 rank 一份完整序列。"""

    @staticmethod
    def forward(ctx, local_shards, world_size, tensor_parallel_output_grad=True):
        # local_shards: [world_size, local_sequence]
        ctx.world_size = int(world_size)
        ctx.tensor_parallel_output_grad = bool(tensor_parallel_output_grad)
        assert local_shards.ndim == 2
        assert local_shards.shape[0] == ctx.world_size
        full_sequence = torch.cat(tuple(local_shards.unbind(dim=0)), dim=0)
        # all-gather 后，每个 rank 都看到相同的完整序列。
        return full_sequence.unsqueeze(0).expand(ctx.world_size, -1).contiguous()

    @staticmethod
    def backward(ctx, grad_replicated_full):
        # grad_replicated_full: [world_size, full_sequence]
        chunks = grad_replicated_full.chunk(ctx.world_size, dim=1)
        if ctx.tensor_parallel_output_grad:
            # TP ranks 给同一全局输出的梯度贡献先求和，再按 sequence scatter。
            summed_grad = grad_replicated_full.sum(dim=0)
            local_grad = torch.stack(
                tuple(summed_grad.chunk(ctx.world_size, dim=0)), dim=0
            )
        else:
            # 后续是重复计算：不把相同语义的梯度重复相加，只取各 rank 对应片段。
            local_grad = torch.stack(
                [chunks[rank][rank] for rank in range(ctx.world_size)], dim=0
            )
        return local_grad, None, None


def main():
    world_size = 4
    sequence = 8

    # 每个 rank 都持有完整输入 [0, 1, ..., 7]。
    full = torch.arange(sequence, dtype=torch.float32)
    replicated_full = full.repeat(world_size, 1).requires_grad_()
    sp_local = ScatterToSequenceParallelRegion.apply(replicated_full, world_size)
    print("scatter output:", sp_local.tolist())
    rank_weights = torch.arange(1, world_size + 1, dtype=torch.float32).unsqueeze(1)
    (sp_local * rank_weights).sum().backward()
    print("scatter input grad on rank 0:", replicated_full.grad[0].tolist())

    # SP shards: rank 0 持有 [0,1]，rank 1 持有 [2,3]，依此类推。
    local_shards = torch.arange(sequence, dtype=torch.float32).reshape(world_size, -1)
    local_shards.requires_grad_()
    gathered = GatherFromSequenceParallelRegion.apply(
        local_shards, world_size, True
    )
    print("gather output on rank 0:", gathered[0].tolist())
    (gathered * rank_weights).sum().backward()
    print("gather input grads:", local_shards.grad.tolist())

    # 如果后续计算是复制路径，关闭 TP 梯度归约，backward 只 split。
    local_shards_copy = torch.arange(sequence, dtype=torch.float32).reshape(
        world_size, -1
    ).requires_grad_()
    gathered_copy = GatherFromSequenceParallelRegion.apply(
        local_shards_copy, world_size, False
    )
    (gathered_copy * rank_weights).sum().backward()
    print("gather grads without TP reduction:", local_shards_copy.grad.tolist())


if __name__ == "__main__":
    main()
```

运行：`python sp_layout_demo.py`（只需 PyTorch，不需 GPU）。预期输出：

```text
scatter output: [[0.0, 1.0], [2.0, 3.0], [4.0, 5.0], [6.0, 7.0]]
scatter input grad on rank 0: [1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0]
gather output on rank 0: [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
gather input grads: [[10.0, 10.0], [10.0, 10.0], [10.0, 10.0], [10.0, 10.0]]
gather grads without TP reduction: [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]]
```

### 追踪数值从哪里来

令 4 个 rank 对本地输出乘权重 `1,2,3,4` 后求和：

- Scatter backward：rank 0 的本地梯度 `[1,1]`、rank 1 的 `[2,2]`、rank 2 的 `[3,3]`、rank 3 的 `[4,4]`，按序列坐标拼成 `[1,1,2,2,3,3,4,4]`。由于原始完整输入在每个 rank 都有副本，每个副本收到同一完整梯度。
- Gather backward（TP 梯度）：完整张量上每个位置收到 `1+2+3+4=10`，然后沿序列均匀切成 4 段，所以每个本地 shard 的梯度都是 `[10,10]`。
- Gather backward（复制路径）：不做跨 rank 求和，各 rank 只返回自己的局部梯度 `[rank_weight, rank_weight]`。

真实分布式实现对应 Megatron `megatron/core/tensor_parallel/mappings.py` 中 `_ScatterToSequenceParallelRegion` 与 `_GatherFromSequenceParallelRegion`，会调用真正的 split / all-gather / reduce-scatter collective，并保存 group、split sizes 等上下文。不同版本可能增加可选参数或更改 group 注入方式。

## 常见误区与症状

1. **把 first 理解为“首先”**：误以为这两种通信要按表格顺序连续执行。这里的 first 是沿第 0 维操作；读函数名 `_along_first_dim` 可确认。
2. **以为 scatter backward 必须 reduce**：如果各 rank 拿的是不同 token 的梯度，应拼接；若此处求和，可能把不同 token 的梯度错误混在一起。
3. **以为 gather backward 永远 reduce-scatter**：当后续是复制计算，重复梯度若再相加会被多算 world size 倍；应按调用语义使用 split-only 分支。
4. **把 SP 与 CP 混为一谈**：这两种 mapping 说的是 TP group 内的序列布局转换，不代表 attention 的上下文并行切分策略。
5. **将单进程模拟当作性能基准**：示例只验证数据与梯度语义，没有 NCCL、跨 GPU 拓扑或真实通信时延。

## 自测

1. `_ScatterToSequenceParallelRegion` 的 forward 为什么不求和？——它在首维选取不重叠的序列区间。
2. 为什么 scatter 的 backward 是 all-gather？——将不同 rank 的局部 token 梯度按坐标拼回完整输入梯度。
3. 默认 TP 路径下 gather backward 为什么需要 reduce-scatter？——先合并各 rank 对完整张量的梯度贡献，再把梯度按 sequence 分给输入 shards。
4. 这里的 “first” 指什么？——首维/第 0 维，不是执行顺序。
