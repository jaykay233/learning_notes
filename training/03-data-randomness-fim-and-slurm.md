# 训练数据随机性、FIM 与 SLURM 入门

## 本次讲解位置

- 章节：`training` / 数据管线与集群作业
- 小节：可复现性边界、Fill-in-the-Middle 增强、作业调度基础
- 知识点：sampler seed、`RandomSeedDataset`、FIM、SLURM 常见命令
- 上次：训练并行维度与 Megatron rank groups
- 下次：按实际训练脚本把数据并行、sampler 与作业启动串起来

## 为什么现在讲这个

模型参数初始化和 dropout seed 都一致，不代表训练可复现。如果 sampler 每轮打乱方式不同，或同一条样本的随机增强变了，优化器看到的 batch 序列就会不同；多机任务是否拿到预期 GPU 和资源，则由集群调度系统决定。拆清模型随机性、数据随机性和作业调度，才能定位“同 seed 但 loss 曲线不一样”或“作业启动方式不对”的原因。

## 一次训练里的随机性有好几层

| 随机源 | 典型控制者 | 控制的内容 |
|---|---|---|
| 模型 / CUDA RNG | framework seed、并行 RNG tracker | 参数初始化、Dropout、并行层随机操作 |
| sampler RNG | 独立 `torch.Generator`，常按 epoch 派生 seed | 样本访问顺序、shuffle |
| sample transform RNG | 按样本 index 派生 seed，覆盖 `torch` / Python `random` / NumPy | 单样本随机裁剪、掩码或增强 |
| dataset-specific RNG | 例如 FIM dataset 持有的 `np.random.RandomState` | 数据集内部的随机构造逻辑 |
| 分布式执行因素 | worker 数、rank、数据分片、版本/硬件 | 样本归属、执行顺序及非确定性算子 |

### Sampler 为什么有自己的 seed

Megatron 的 `MegatronPretrainingRandomSampler` 通过专用 `torch.Generator()` 控制 shuffle，通常会将 epoch 纳入 seed 派生。这样可以让一个 epoch 内的抽样顺序由 seed 决定，并让不同 epoch 得到不同顺序；不必把 sampler 随机状态和模型 dropout 的 CUDA RNG 混在一起。

### `RandomSeedDataset` 的 index seed

这类 wrapper 在 `__getitem__(idx)` 中根据 `idx + curr_seed` 设置各随机库的 seed，再调用底层 dataset。于是同一个样本 index 在同一个 `curr_seed` 下，即使被重复读取，也会得到可复现的随机 transform；换 index 或换 epoch seed，就会变化。

简化逻辑：

```python
seed = idx + curr_seed
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
return dataset[idx]
```

真实实现还要结合 worker/rank 的行为与所用 RNG API 阅读。重置全局 RNG 会影响同一进程随后发生的随机调用，因此数据 wrapper 要在实际 data-loading 上下文里设计与测试。

### FIM（Fill-in-the-Middle）是什么

FIM 是“填中间”式语言模型训练：从一段文本里选出中间片段，把前缀和后缀提供给模型，让模型生成被拿走的中间内容。它不是简单把 token 替换成 mask 的 BERT 式目标，而是重新安排输入片段和特殊标记。

例如原文：

```text
The cat sat on the warm mat.
```

将 `sat on the` 作为 middle，可形成概念结构：

```text
<FIM_PREFIX> The cat <FIM_SUFFIX> the warm mat <FIM_MIDDLE> sat on the
```

实际 token 拼接、loss mask、文档边界和比例由训练实现决定；上例只展示“给 prefix + suffix，预测 middle”的直觉。

Megatron 的 `FIMDataset` 可通过 `np.random.RandomState(seed=...)` 让增强逻辑的随机行为可复现。它与 sampler shuffle seed 是不同随机源。

## 最小可运行的随机性示例

```python
import random
import numpy as np
import torch
from torch.utils.data import Sampler


def sample_transform(idx: int, curr_seed: int) -> tuple[float, float, int]:
    seed = idx + curr_seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return (
        random.random(),
        float(np.random.random()),
        int(torch.randint(0, 1000, ()).item()),
    )


def shuffled_indices(epoch: int, seed: int, n: int) -> list[int]:
    generator = torch.Generator()
    generator.manual_seed(seed + epoch)
    return torch.randperm(n, generator=generator).tolist()


if __name__ == "__main__":
    print("epoch0:", shuffled_indices(epoch=0, seed=100, n=5))
    print("epoch1:", shuffled_indices(epoch=1, seed=100, n=5))
    first = sample_transform(idx=3, curr_seed=1000)
    second = sample_transform(idx=3, curr_seed=1000)
    other = sample_transform(idx=4, curr_seed=1000)
    print("same index reproducible:", first == second)
    print("different index differs:", first != other)
```

运行：`python data_randomness_demo.py`（需要 PyTorch 与 NumPy）。预期保证是：`same index reproducible: True`，`different index differs: True`；具体 shuffle 与随机数值可能随 PyTorch / NumPy 版本而异，所以不把某一版本的数字当跨版本契约。

### 可复现性不等于“设置一个 seed”

要复现训练，还要检查：

- 每个并行 rank 的 seed 派生规则是否一致且符合预期；
- sampler epoch 是否正确更新、是否恢复了 sampler 状态；
- dataset transform 是否使用了独立 RNG，worker seed 是否正确；
- 数据版本、排序、分片、worker 数、batch size、gradient accumulation 是否相同；
- 使用的 GPU kernel 是否确定性，以及 PyTorch/CUDA/NCCL/驱动版本是否一致。

所以“模型 RNG 完全一致”只是可复现性的一部分，并不保证每一步看到同一批数据。

## SLURM 是什么

SLURM 是集群上的作业调度与资源管理系统。可以把它理解为：用户描述需要的节点、GPU、CPU、内存和运行时长；调度器排队、分配资源，再在获批的节点上启动程序。

| 命令 | 用途 |
|---|---|
| `sbatch job.sh` | 提交一个 batch 作业，返回 job ID |
| `squeue -u "$USER"` | 查看自己的排队/运行作业 |
| `scancel JOB_ID` | 取消作业 |
| `srun ...` | 在作业 allocation 中启动命令/任务；也可按集群策略申请交互式资源 |
| `salloc ...` | 请求交互式 allocation，具体用法依集群设置 |

### 一个最小 `sbatch` 模板

```bash
#!/usr/bin/env bash
#SBATCH --job-name=pretrain-demo
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out

set -euo pipefail
# 模块名、环境激活方式、GPU 参数和分布式启动命令都需要按集群调整。
source ~/.bashrc
conda activate train-env

srun python train.py --config config.yaml
```

提交和观察：

```bash
sbatch job.sh
squeue -u "$USER"
scancel <job_id>
```

`#SBATCH` 资源参数不是通用硬件事实：有些集群用 `--gpus=4`、`--gres=gpu:a100:4` 或分区/账户参数；有些环境由 `srun` 负责启动多进程。必须以本集群文档和管理员配置为准。作业 pending 通常表示正在等资源，不代表训练脚本报错；运行失败则查看 `--output` 指向的日志和 job 状态。

## 自测

1. 同一个 epoch sampler seed 一样、但 sample transform 的 seed 规则不一样，训练 batch 一定一致吗？——不一定；样本访问顺序和样本内部增强是不同随机源。
2. FIM 是把随机 token 替换成 mask 吗？——不是；它把文本拆成 prefix / middle / suffix，让模型根据 prefix 与 suffix 补 middle。
3. `sbatch` 与 `srun` 的分工是什么？——`sbatch` 提交 batch 作业；`srun` 在 allocation/集群策略下启动任务步骤。
4. `squeue` 里还看不到训练日志，可能只是代码失败吗？——不一定；作业可能仍在排队等待资源，先检查 job state。
