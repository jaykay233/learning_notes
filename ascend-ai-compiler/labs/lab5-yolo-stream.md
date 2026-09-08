# Lab5 · YOLOv8 Stream 实践

> 原课 Lab5：生产者-消费者并行。前置：[14](../14-yolov8-deploy.md)、[13](../13-profiling.md)。

---

## 1. 为什么做这个 Lab

单帧串行：

```text
letterbox → 推理 → NMS → 下一帧 letterbox …
```

NPU 在等 CPU，CPU 在等 NPU → **吞吐上不去**。  
流水把阶段重叠，吃的是 [05](../05-ascend-npu-hardware.md) 里「延迟隐藏/并行」同一思想。

---

## 2. 目标效果

至少两阶段流水；报告 **串行 FPS vs 流水 FPS**；说明队列深度与延迟的权衡。

---

## 3. 怎样做

```text
[读图+letterbox] →队列→ [NPU 推理] →队列→ [NMS+画框]
```

| 设计点 | 为什么 |
|---|---|
| 队列 | 解耦快慢阶段 |
| 深度 2～N | 太深增延迟与内存；太浅叠不起 |
| 多线程/stream | 真重叠 |

---

## 4. 验收

- [ ] 可运行的多阶段 pipeline  
- [ ] FPS 对比表  
- [ ] 文字解释：主要缓解了 profiling 里哪类等待  

---

## 5. 连接

聊天服务里「tokenize与生成重叠」是同类问题的另一形态。
