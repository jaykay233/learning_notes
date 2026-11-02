# 异构推理学习笔记

本目录记录 GPU 资源池、SRAM ASIC 与其他异构硬件之间的推理切分、调度和性能边界。

| 文档 | 知识点 |
|---|---|
| [01-sqd-subquadratic-disaggregation.md](01-sqd-subquadratic-disaggregation.md) | SQD 按状态是否随上下文增长切分 quadratic/subquadratic 工作；Prefill 共置、top-k KV 预取、roofline 与容量边界 |
| [02-aflex-disaggregated-frequency-scaling.md](02-aflex-disaggregated-frequency-scaling.md) | AFlex 把 PA/PF/DA/DF 做成独立频率域；ILP 资源配比、分级 DVFS、动态 micro-batch 与能耗边界 |

## 当前进度

- [x] SQD 的切分判据与 GPU/SRAM ASIC 放置规则
- [x] Prefill 与 quadratic decode attention 共置的资源互补
- [x] 稀疏 Attention 的 indexer、top-k attention 与 KV 预取边界
- [x] 论文实测能效与 Rubin + LPX 解析模型结果的证据分级
- [x] AFD 的 Attention/FFN 资源解耦与独立 GPU 频率域
- [x] AFlex 的两级控制、A/F 配比、动态 micro-batch 与能耗证据分级
- [ ] top-k 缓存淘汰策略、IndexShare 偏移预算与命中率反解
- [ ] fabric 带宽/延迟扫描与跨芯片 SLO 建模
- [ ] AFlex ILP 目标函数、DVFS 窗口与切换成本预算的数值推导

下一知识点：拆 AFlex 的 ILP 和本地 DVFS 窗口，先写清资源、TP、频率、TTFT/TPOT 约束，再用具体参数演示切换成本何时值得摊销。
