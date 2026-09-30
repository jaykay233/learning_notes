# Reduce-Scatter 的运算逻辑

## 本次讲解位置

- 章节：`training` / 分布式 collective
- 小节：Reduce-Scatter 的数学语义与输出布局
- 知识点：逻辑上先跨 rank reduce，再把结果 scatter 成各 rank 的 shard
- 上次：SP 下 Vocab Embedding 为什么把 all-reduce 改成 reduce-scatter
- 下次：把 reduce-scatter / all-gather 对应到 SP 前向和反向的布局转换

## 为什么现在讲这个

前一节中，Vocab Embedding 的 SP 路径用 `reduce_scatter_to_sequence_parallel_region` 替代 all-reduce。如果不拆开理解这个 collective，很容易误以为 SP 彻底省掉了跨卡求和。需要明确：**reduce-scatter 仍然对所有 rank 的输入做归约，只是归约结果不再完整复制到所有 rank，而是分片交给各 rank。**

## 心智模型

假设有 `P` 个 rank，每个 rank 都贡献一个同形状张量：

1. 对应元素跨 rank 归约（常见是 `sum`）；
2. 将归约后的张量切成 `P` 份；
3. 第 `r` 个 rank 收到第 `r` 份。

所以数学上可看成 **reduce + scatter**。但这是语义分解：真实 collective 通常直接用 Reduce-Scatter 算法完成，不必先在每张卡上物化一份完整的 reduced tensor，再单独启动 scatter。

## 两 rank 数值例子

```text
rank 0 输入: [1,  2,  3,  4]
rank 1 输入: [10, 20, 30, 40]
```

先对相同位置求和：

```text
reduced:    [11, 22, 33, 44]
```

再按两 rank 切成等长两份：

```text
rank 0 输出: [11, 22]   # reduced 的前半段
rank 1 输出: [33, 44]   # reduced 的后半段
```

注意：不是 rank 0 只用自己的输入、rank 1 只用自己的输入；每个输出位置都已包含所有 rank 对应位置的贡献。

## 一个完整的语义模拟程序

保存为 `reduce_scatter_demo.py`：

```python
def reduce_scatter_sum(rank_inputs: list[list[float]]) -> list[list[float]]:
    """模拟 sum Reduce-Scatter；rank_inputs[r] 是 rank r 的完整输入张量展平值。"""
    if not rank_inputs:
        raise ValueError("need at least one rank")

    world_size = len(rank_inputs)
    tensor_size = len(rank_inputs[0])
    if any(len(values) != tensor_size for values in rank_inputs):
        raise ValueError("all ranks must provide tensors with the same shape")
    if tensor_size % world_size != 0:
        raise ValueError("tensor size must be divisible by world size in this demo")

    # reduce：逐元素跨 rank 求和，形成逻辑上的完整归约结果。
    reduced = [sum(rank_values[i] for rank_values in rank_inputs)
               for i in range(tensor_size)]

    # scatter：沿展平后的第 0 维切成 world_size 份，每个 rank 拿一份。
    chunk_size = tensor_size // world_size
    return [
        reduced[rank * chunk_size : (rank + 1) * chunk_size]
        for rank in range(world_size)
    ]


if __name__ == "__main__":
    inputs = [
        [1, 2, 3, 4],
        [10, 20, 30, 40],
    ]
    outputs = reduce_scatter_sum(inputs)
    print("outputs:", outputs)

    # 对照 all-reduce：每 rank 都会拿到完整 reduced tensor。
    reduced = [sum(rank_values[i] for rank_values in inputs)
               for i in range(len(inputs[0]))]
    all_reduce_outputs = [reduced.copy() for _ in inputs]
    print("all-reduce outputs:", all_reduce_outputs)
```

运行：`python reduce_scatter_demo.py`（只需 Python 3）。预期输出：

```text
outputs: [[11, 22], [33, 44]]
all-reduce outputs: [[11, 22, 33, 44], [11, 22, 33, 44]]
```

程序里的 `reduced` 是为了讲解而显式创建的中间结果；真实 Reduce-Scatter 实现可能边归约边传输对应 chunk，避免将完整结果复制/物化到每个 rank。

## 和 All-Reduce 的关系

| Collective | 每个 rank 输入 | 做什么 | 每个 rank 最终拿到 |
|---|---|---|---|
| Reduce | 同形状张量 | 跨 rank 归约 | 结果通常只交给指定 root |
| All-Reduce | 同形状张量 | reduce + 把完整结果分发/复制给所有 rank | 完整 reduced tensor |
| Reduce-Scatter | 同形状张量 | reduce + 按指定布局分片分发 | reduced tensor 的一个 shard |
| All-Gather | 每 rank 一个 shard | 按 rank 顺序拼接 shards | 所有 rank 都拿完整拼接结果 |

对于 sum 归约和均匀分片，概念上可以写作：

```text
all_reduce(x) = all_gather(reduce_scatter(x))
```

SP 场景中，如果后续每张卡只需要自己的 sequence shard，就可以直接停在 reduce-scatter，不必再 all-gather 成完整副本。比如 TP group 大小 `P=4`、序列长度 `S=1024`，形状 `[S,B,H]` 的 reduced 输出按首维切分，每 rank 得到 `[256,B,H]`。

## 源码与执行边界

Megatron 的 `reduce_scatter_to_sequence_parallel_region` 在 `megatron/core/tensor_parallel/mappings.py` 中封装 Reduce-Scatter，并按序列首维切分。Vocab Embedding 的调用位于 `megatron/core/tensor_parallel/layers.py`：先把输出布局调整为 sequence 在首维，再调用该 mapping。

此处讲的是 collective 的逻辑语义。实际网络算法（ring、树或硬件/通信库选择的实现）、是否分块流水、具体 rank 到 chunk 的映射，取决于通信后端和配置。

## 常见误区

1. **以为 Reduce-Scatter 只有 scatter，没有 reduce**：这样每个输出 shard 会缺少其他 rank 的贡献，数值不完整。
2. **把“先 reduce 再 scatter”理解成必须启动两个独立 kernel/collective**：这是数学语义的理解方式，优化实现可以融合两步。
3. **以为每个 rank 都会拿到完整 reduced tensor**：那是 all-reduce；Reduce-Scatter 每 rank 只拿分片。
4. **把 scatter 理解成将各 rank 原输入直接分块给各 rank**：正确顺序是归约相同元素，再把归约结果分片。
5. **忽略张量切分维度**：真实模型常沿 sequence 维分片，并不一定是代码里展平后的首段/末段；布局与 rank 顺序决定每个 rank 收到哪些元素。

## 自测

1. Reduce-Scatter 是否做跨 rank 求和？——常见 sum Reduce-Scatter 会先对相同位置的元素求和。
2. 两个 rank 各有长度 4 的输入时，Reduce-Scatter 后每 rank 通常拿多长？——均匀切分时长度为 `4/2=2`。
3. “逻辑上先 reduce 再 scatter”表示一定执行两个独立通信操作吗？——不表示；实现可以融合为一个 collective。
4. 与 all-reduce 最大的输出区别是什么？——all-reduce 每 rank 得完整结果，reduce-scatter 每 rank 得完整归约结果的一片。
