# 训练并行维度：DP、SP、EP、ETP 与 rank group

## 本次讲解位置

- 章节：`training` / 训练中的并行策略
- 小节：并行维度的含义与 Megatron rank 组生成
- 知识点：DP / expert-DP / ETP、TP+SP、`decompose`、mask、`RankGenerator`、进程组注入
- 上次：PyTorch 自动微分与梯度规则
- 下次：训练数据随机性、FIM 与 SLURM 作业调度

## 为什么现在讲这个

配置里看到 `dp=8`、`expert_dp=4` 或 rank group `[g0,g4,...]` 时，数字看起来像是在数输入副本，很容易把“数据怎么切”和“模型哪些参数由哪些卡持有”混为一谈。真正决定并行通信和参数布局的是：**每个 rank 坐标表示什么，以及一组 collective 究竟在哪些 rank 之间发生**。

## 先用一句话建立模型

把整个训练 world 想成按多个坐标轴排成的网格：某一并行维度的 group，就是固定其他坐标、只沿目标坐标轴变化的一条线。

| 维度 | 主要切分对象 | 常见通信/作用 |
|---|---|---|
| DP | 数据样本；模型参数在副本间复制 | 同步梯度或分片优化器状态 |
| TP | 单层权重/隐藏维 | 层内 all-reduce / all-gather / reduce-scatter 等 |
| SP | TP 区域之间的激活序列布局 | TP 组内切序列，降低激活冗余 |
| PP | 模型层 | 相邻 stage 传递 activation / gradient |
| CP | 长序列上下文 | 按序列上下文划分并配套注意力通信 |
| EP | MoE 专家集合 | token dispatch / combine，专家参数分布到不同 rank |
| ETP | 每个专家内部的张量维 | 将单个专家的张量计算再切到多个 rank |
| `expert_dp` | 专家并行网格之外的专家副本维 | 同一专家 shard 的数据并行副本 |

## DP 本质上是什么

DP（data parallelism）是**模型副本沿数据并行维复制，并通过同步更新保持参数一致**。典型训练把不同 mini-batch 分片送到副本，副本各自算梯度，再做归约并应用同样的优化器更新。

“输入不一样”是常见做法，却不是定义本身：相同输入也可以送到两个副本，但若其参数布局与梯度同步仍构成副本式训练，仍是在 DP 结构中；反过来，只因两张卡输入数据不同，并不能说它们就在做 DP。关键是模型副本与同步更新关系。

**Dense `dp`** 是 dense 参数所在 rank generator 的数据并行维大小。它不单独回答“每个 rank 当前拿到了什么 batch”；实际 batch 还受 micro-batch、gradient accumulation、sampler 和 DP/CP 设置影响。

## SP：TP 组内把序列激活切开

TP 把矩阵权重/隐藏维分片，但 LayerNorm、Dropout 等部分并非都自然沿 TP 维切开；若 TP 组每卡都持有完整序列激活，activation memory 可能成为瓶颈。Megatron 风格 Sequence Parallelism 在这些区域把序列维也分给 TP 组各 rank。

简化的数据流：

1. 某些 TP 运算需要完整或特定分布的输入时，使用 **all-gather** 在 TP rank 间拼回所需布局。
2. 对可按序列切分的区域，每 rank 只处理本地 token 片段。
3. 运算结果需要跨 TP rank 合并时，使用 **reduce-scatter**，一边归约，一边重新按序列切分。

TP=4、序列长度 `S=1024` 时，序列切分区域每卡大约持有 `1024/4=256` 个 token 位置的激活（忽略 padding、batch/head 等维度）。SP 不代表整层 attention 永远只看本卡 token，也不是 CP 的别名。

“几乎零额外通信”是对某些 Megatron TP 路径而言：原本需要的 collective 可改造成 all-gather + reduce-scatter 的布局转换，常把通信量控制在与既有 all-reduce 相近的量级；不是说 SP 完全没有通信或所有模型都零成本。

## DP 与 `expert_dp` 的关系：一个 16-rank 例子

忽略版本中额外的 remat 等并行轴后，常见配置下：

```text
Dense 网格大小：W = TP × PP × CP × DP
Expert 网格大小：W = ETP × EP × PP × expert_dp
```

同一个 world size `W` 必须能被两种网格各自覆盖，所以：

```text
TP × PP × CP × DP = ETP × EP × PP × expert_dp
约掉共同的 PP：TP × CP × DP = ETP × EP × expert_dp
```

例如 `W=16, TP=2, PP=1, CP=1, EP=2, ETP=TP=2`：

```text
Dense DP = 16 / (2×1×1) = 8
expert_dp = 16 / (2×2×1) = 4
```

在一个常见 rank 顺序下，dense DP group（固定 `tp=0`）为：

```text
[g0, g2, g4, g6, g8, g10, g12, g14]   # 8 ranks
```

专家 rank generator 加入 EP 轴后，可将同一 dense DP 方向按 EP 坐标拆为两组 expert-DP：

```text
EP 坐标 0: [g0, g4, g8, g12]          # 4 ranks
EP 坐标 1: [g2, g6, g10, g14]         # 4 ranks
```

因此在该配置和映射顺序下，可以直觉地说 EP 把 dense DP 方向细分为 EP 组和 `expert_dp` 组；但不要把它当成所有配置都成立的简化物理映射。若 `ETP≠TP` 或 `CP>1`，两边各轴定义不同，优先使用上面的 world-size 公式和实际 `RankGenerator` 生成的 group。尤其当 EP/CP 同时非 1 时，Megatron 通常使用两套生成器分别描述 dense 与 expert 布局，而不是把两轴塞进同一个 generator。

## ETP 是什么

ETP（Expert Tensor Parallelism）是**单个 expert 内部**沿 tensor 维的并行大小，不是专家个数。EP=4 表示专家集合跨 4 个 EP 坐标分布；ETP=2 表示每个 expert 的张量计算/参数可能再由 2 个 tensor-parallel rank 分担。很多默认配置会设 `ETP=TP`，但它们语义不同，可以配置为不同值。

## `decompose`：把一维编号还原成坐标

Megatron 的 rank 组 helper 会先把坐标轴大小放进 `parallel_size`，再用 stride 把多维坐标编码为全局编号。若 shape 为 `[2, 3]`，第 0 维最快变化，则 stride 为 `[1, 2]`，编号公式是：

```text
rank = i0×1 + i1×2
```

编号 4 的坐标：

```text
i0 = (4 // 1) % 2 = 0
i1 = (4 // 2) % 3 = 2
```

所以 index 4 对应 `(i0,i1)=(0,2)`。`decompose` 本身只把扁平编号转换为多维坐标；之后再用相应 stride 把“组内被 mask 的坐标”和“组索引对应的未 mask 坐标”合起来，得到 global rank。

Mask 的 `True` 表示这个维度属于正在生成的 group，`False` 表示该维度用于区分不同 group。例如 `shape=[2,2,2]`、`mask=[True,False,True]`：每组固定中间坐标，沿首尾维变化，组大小是 `2×2=4`，结果是：

```text
[0, 1, 4, 5]
[2, 3, 6, 7]
```

### 可运行的坐标与 rank-group 模拟

保存为 `rank_groups_demo.py`：

```python
from math import prod


def decompose(index: int, shape: list[int]) -> list[int]:
    """维度 0 最快变化；把扁平 index 还原为坐标。"""
    coords = []
    for size in shape:
        coords.append(index % size)
        index //= size
    return coords


def masked_rank_groups(parallel_size: list[int], mask: list[bool]) -> list[list[int]]:
    """生成正交 rank group；True 维度在组内变化。"""
    assert len(parallel_size) == len(mask)
    world_size = prod(parallel_size)
    masked_dims = [s for s, selected in zip(parallel_size, mask) if selected]
    unmasked_dims = [s for s, selected in zip(parallel_size, mask) if not selected]
    masked_strides = []
    stride = 1
    for size, selected in zip(parallel_size, mask):
        if selected:
            masked_strides.append(stride)
        stride *= size
    unmasked_strides = []
    stride = 1
    for size, selected in zip(parallel_size, mask):
        if not selected:
            unmasked_strides.append(stride)
        stride *= size

    group_size = prod(masked_dims)
    group_count = world_size // group_size
    groups = []
    for group_index in range(group_count):
        group_coords = decompose(group_index, unmasked_dims)
        ranks = []
        for local_index in range(group_size):
            local_coords = decompose(local_index, masked_dims)
            rank = sum(c * s for c, s in zip(group_coords, unmasked_strides))
            rank += sum(c * s for c, s in zip(local_coords, masked_strides))
            ranks.append(rank)
        groups.append(ranks)
    return groups


if __name__ == "__main__":
    print("decompose(4, [2, 3]) =", decompose(4, [2, 3]))
    print("groups =", masked_rank_groups([2, 2, 2], [True, False, True]))
```

运行：`python rank_groups_demo.py`。预期输出：

```text
decompose(4, [2, 3]) = [0, 2]
groups = [[0, 1, 4, 5], [2, 3, 6, 7]]
```

这个示例刻意只实现坐标映射的核心，不创建 torch distributed communicator；真实 Megatron 还要处理 rank order、rank offset、backend 和进程组初始化。

## EP / CP 的两套 `RankGenerator`

在 Megatron 的常见设计中：

- dense / non-expert generator 含有 CP 轴（`ep=1`）；
- expert generator 含有 EP 轴（`cp=1`）；
- 单个 generator 的断言限制 EP、CP 不能同时大于 1，因为各自属于不同的 rank-grid 视角；
- 两边生成的 PP groups 必须一致，因为同一个模型 pipeline 的 stage 排列不能因是否经过 MoE 专家而改变。

若 rank order 不是 PP-last，某些版本还会断言 dense 与 expert 路径的 DP 尺寸一致。配置更复杂时，以实际 Megatron 版本的源码、断言和 group 列表为准；不同提交可能增添 remat 等新轴。

## 为什么生产代码传 `ProcessGroupCollection`

Megatron `megatron/core` 的模块如果直接调用 `parallel_state.get_*_group()`，就隐式依赖全局并行状态：初始化顺序、当前 rank、全局变量都成为隐藏前提。通过构造函数传入 `ProcessGroupCollection`（或模块需要的具体 process group），则模块显式声明自己会用哪些 group：

```python
# 示意代码：collection 的字段名以所用 Megatron 版本为准。
class SomeParallelLayer:
    def __init__(self, pg_collection):
        self.tp_group = pg_collection.tp
```

这样更容易单测、重用模块、替换并行拓扑，也更容易发现漏传的依赖。全局 getter 在启动/兼容层仍可能存在；规范的重点是核心生产模块尽量不要偷偷读取全局状态。collection 只是 group 的容器/依赖入口，并不会自动创建 group，也不会让不同类型的 group 变得可互换。

## 常见误区

- 把 DP 定义成“输入不一样”。核心是模型副本与同步更新，不是 batch 是否碰巧重复。
- 把 `expert_dp` 直接当 dense `dp` 的另一个名字。它属于专家网格，公式受 ETP、EP 影响。
- 把 EP 和 ETP 混为一谈。前者切专家集合，后者切单个 expert 内部张量。
- 看到某一个 rank group 就推断所有 rank 的固定编号规则。`order` 改变，物理 group 的 rank 列表也会改变。
- 把 SP 说成 CP。SP 通常是 TP 路径中 LayerNorm/Dropout 等区域的序列分片；CP 是另一并行轴，目标和通信路径不同。
- 认为 `ProcessGroupCollection` 负责初始化通信。它是依赖传递方式，process group 必须先被正确创建。

## 自测

1. DP 的核心条件是什么？——模型副本按数据并行方式训练，并保持参数同步更新。
2. EP=2 与 ETP=2 分别切什么？——EP 切专家分布，ETP 切每个专家内部的 tensor 计算。
3. `mask[i]=True` 表示什么？——对应轴属于 group，沿该维坐标枚举 group 成员。
4. 为什么同一个 Megatron rank generator 不同时包含 EP 与 CP？——常见实现为 dense 和 expert 使用不同网格视角，分别由不同 generator 生成。
5. `ProcessGroupCollection` 解决什么问题？——显式传递通信依赖，降低核心模块对全局并行状态的耦合。
