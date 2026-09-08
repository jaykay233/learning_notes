# Lab6 · 部署 NPU 加速聊天机器人

> 原课 Lab6。前置：[15](../15-attention-on-npu.md)–[17](../17-qwen-quant-deploy.md)。

---

## 1. 为什么做这个 Lab

把整门课收束成 **可演示产品**：

- Transform：量化模型上昇腾  
- Deploy：多轮对话  
- 关键路径：KV 必须生效（[16](../16-kv-cache.md)）  
- 性能数字：TTFT、tok/s  

这是求职作品集里最值钱的一类项目（[18](../18-interview-career.md)）。

---

## 2. 目标效果

可交互多轮；第二轮起不明显「整段历史当新 prefill」；记录芯片、量化方案、TTFT、tok/s。

---

## 3. 怎样做（步骤 + 为什么）

| 步骤 | 做什么 | 为什么 |
|---|---|---|
| 1 | 选定小/中 Qwen 与量化方案 | 先跑通再追 SOTA |
| 2 | 转可执行 + 接推理引擎 | Deploy |
| 3 | 实现 prefill/decode 循环 | Attention 两阶段 |
| 4 | 多轮复用 KV | 验证 cache |
| 5 | 测延迟指标 | 可写进简历 |
| 6 | （加分）CLI/HTTP | 可演示 |

---

## 4. 验收

- [ ] 多轮对话 demo  
- [ ] 一页 prefill/decode 数据流图  
- [ ] 性能表 + 量化方案  
- [ ] 简述：若无 KV，内存/延迟会怎样  

---

## 5. 连接

对照 `models/qwen-*` 讲清「结构上为何这样量、这样融」。
