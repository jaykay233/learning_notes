# 15 · Attention 算子部署与优化

> 原课：Transformer、MHA、Softmax/Matmul 在 NPU 上的优化。

---

## 1. 为什么 Attention 是 LLM 部署核心

CV 主干大量 Conv；LLM 前向大量时间花在：

```text
Q,K,V = Linear(x)
S = Softmax(Q K^T / √d)
O = S V
```

再叠加 **逐 token decode**（见 [16](./16-kv-cache.md)），带宽与算力压力都集中在这里。

---

## 2. 各子算子（怎样 + 优化点）

| 子步骤 | 为什么贵 | 优化方向 |
|---|---|---|
| Linear / MatMul | 大 GEMM | 走 Cube、好 layout、融合 QKV |
| \(QK^T\) | 序列变长时算力升 | 稀疏/压缩结构；引擎融合 kernel |
| Softmax | 要读大矩阵，带宽敏感 | 与前后 fused，避免多次写回 |
| \(SV\) | 又一次大乘 | 同 MatMul |

**GQA/MQA（结构级）：** 减少 KV 头数 → 减计算与 **KV 体积**（连到 16）。

---

## 3. Prefill vs Decode（为什么要分）

| 阶段 | 特征 | 瓶颈直觉 |
|---|---|---|
| Prefill | 一次吃长 prompt | 算力与显存峰值 |
| Decode | 每步一个（或少数）token | **反复读 KV**，常带宽墙 |

部署时两者可能走不同 kernel/策略；不能只看「整网平均 FLOPs」。

---

## 4. 和编译器的关系

- 图上能否融 RoPE/Norm/Linear → Frontend/Backend 能力  
- 选高性能 Attention 模板 → kernel 库 / 推理引擎  
- 量化 W8A8 等 → [17](./17-qwen-quant-deploy.md)  

架构侧减负（MLA、QSA…）见 `models/` 笔记。

---

## 5. 练习

1. 为何 Softmax 常成带宽瓶颈？  
2. GQA 如何同时影响 MatMul 与 KV？  
3. 只优化 prefll 不管 decode，线上会怎样？

下一讲：[16](./16-kv-cache.md)
