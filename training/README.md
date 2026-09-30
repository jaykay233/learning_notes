# 模型训练知识点

集中记录深度学习 / PyTorch 训练中的概念、公式、代码与常见误区。

## 目录

| 文档 | 内容 |
|---|---|
| [01 · 自动微分与 `derivatives.yaml`](01-autograd-and-derivatives-yaml.md) | `requires_grad`、`retain_grad()`、`create_graph` / `retain_graph`、VJP / JVP 与 YAML key/value |
| [02 · 并行维度与 rank groups](02-parallelism-and-rank-groups.md) | TP / SP / DP / EP / ETP / expert-DP、rank 公式、`decompose`、`RankGenerator`、`ProcessGroupCollection` |
| [03 · 数据随机性、FIM 与 SLURM](03-data-randomness-fim-and-slurm.md) | sampler / dataset seed、Fill-in-the-Middle、可复现性边界与 SLURM 常用命令 |
| [04 · `fan_in` / `fan_out`](04-initialization-fan-in-fan-out.md) | 全连接层与卷积层连接数、Kaiming 初始化尺度 |

## 主题路线

1. 先理解 autograd：哪些计算被追踪、哪些梯度会存进 `.grad`、如何求更高阶导数。
2. 再看 fan-in / fan-out：从权重形状计算输入、输出连接数，理解初始化尺度。
3. 再看分布式并行：把 rank 当作多维网格坐标，弄清 group 如何由某些坐标轴生成。
4. 最后串数据与集群：模型 RNG、sampler RNG、样本增强 RNG 分开管理，SLURM 负责资源调度与启动。

## 阅读提示

- Megatron-LM 与 PyTorch 源码会随版本演进；rank 顺序、进程组接口与 YAML 规则应以正在阅读的 commit 为准。
- 公式说明逻辑关系，实际物理 rank 列表还依赖 parallel order、rank offset 和集群 rank 映射。
