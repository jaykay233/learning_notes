# Modern GPU Programming for MLSys 学习笔记

课程：

- <https://github.com/mlc-ai/modern-gpu-programming-for-mlsys>

本目录整理对话中围绕 GPU 数据布局、命名轴、Replication 和 Offset
展开的内容，重点关注这些概念如何影响推理系统中的 GEMM、Attention
以及 KV Cache 数据路径。

## 文档索引

| 文档 | 内容 |
|---|---|
| [01-data-layout-and-named-axes.md](01-data-layout-and-named-axes.md) | Shape-Stride、Tile Layout、命名轴、Register Fragment |
| [02-replication-and-offset.md](02-replication-and-offset.md) | `R[...]` 副本、`O[...]` 偏移、TMEM 广播、GPU Mesh |
| [03-practice-and-corrections.md](03-practice-and-corrections.md) | 自测题、原始答案、逐题订正 |
| [04-swizzle-layout.md](04-swizzle-layout.md) | shared memory bank conflict、XOR swizzle、完整地址推导 |

## 当前进度

已完成：

```text
Shape-Stride 模型
Tile Layout
命名轴 @m / @TLane / @TCol / @laneid / @reg
Register Fragment 映射
Replication R[...]
Offset O[...]
Shared memory bank conflict
XOR swizzle
```

## 核心主线

GPU layout 不只回答“数据在哪个地址”，还要回答：

```text
数据属于哪个 thread / lane
数据位于哪个 register fragment slot
数据位于哪个 TMEM lane 和 column
同一逻辑数据是否需要复制到多个物理位置
是否需要给整体布局增加固定偏移
```

Tensor Core kernel 出错时，计算表达式本身往往没有问题，问题更常出现在：

```text
元素放错 lane
元素占用错误的 register slot
TMEM lane / column 不匹配
共享内存 swizzle 与 descriptor 不一致
生产者写出的 layout 与消费者读取的 layout 不一致
```

## 环境说明

- 当前机器：Apple M5 Pro
- 本机没有 CUDA / Blackwell 硬件
- Blackwell 的 TMEM、`tcgen05`、WGMMA 和 TMA 示例主要用于理解 layout 和代码路径
- 概念学习与静态代码分析可以在 macOS 上完成，实际性能测量需要 NVIDIA GPU

## 后续可继续整理

```text
Tensor Core layout 的 Ampere / Hopper / Blackwell 演进
ldmatrix 与 register fragment
WGMMA matrix descriptor
TMEM accumulator 与 tcgen05.ld
Block-scaled MMA 的 scale factor 数据路径
```
