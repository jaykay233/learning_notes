# 异构推理学习笔记

本目录记录 GPU、SRAM ASIC 与其他异构硬件之间的 decode 切分、数据流和性能边界。

| 文档 | 知识点 |
|---|---|
| [01-sqd-subquadratic-disaggregation.md](01-sqd-subquadratic-disaggregation.md) | SQD 按状态是否随上下文增长切分 quadratic/subquadratic 工作；Prefill 共置、top-k KV 预取、roofline 与容量边界 |

## 当前进度

- [x] SQD 的切分判据与 GPU/SRAM ASIC 放置规则
- [x] Prefill 与 quadratic decode attention 共置的资源互补
- [x] 稀疏 Attention 的 indexer、top-k attention 与 KV 预取边界
- [x] 论文实测能效与 Rubin + LPX 解析模型结果的证据分级
- [ ] top-k 缓存淘汰策略、IndexShare 偏移预算与命中率反解
- [ ] fabric 带宽/延迟扫描与跨芯片 SLO 建模

下一知识点：继续拆 SQD 的 top-k KV 缓存，先定义组头、复用层、预取窗口和容量预算，再解释每种淘汰策略如何影响命中率与暴露延迟。
