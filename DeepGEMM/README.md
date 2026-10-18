# DeepGEMM 学习笔记

本目录整理对话中逐步梳理的 DeepGEMM / MegaMoE 知识点：先看项目定位，再跟 dispatch/pull、量化 scale、epilogue 与 grouped GEMM 数据流。笔记用概念图、坐标例子与同步边界解释源码行为，不复制整份仓库源码。

| 文档 | 知识点 |
|---|---|
| [01-deepgemm-project-and-kernel-model.md](01-deepgemm-project-and-kernel-model.md) | DeepGEMM 与 CUTLASS 的范围差异、MegaMoE 在仓库中的位置、`tcgen05.mma` 与 UMMA、Gluon 对照目标 |
| [02-megamoe-dispatch-pull.md](02-megamoe-dispatch-pull.md) | workspace、expert send count、atomic slot offset、metadata、同步发布、远端 pull 与 TMA chunk |
| [03-megamoe-x-sf-utccp-and-warp-roles.md](03-megamoe-x-sf-utccp-and-warp-roles.md) | `x_sf`、SFA/SFB、A/SFA 与 B/SFB loader warp、UTCCP 4×32 转换的索引轨迹和原因 |
| [04-megamoe-l1-l2-epilogue.md](04-megamoe-l1-l2-epilogue.md) | Linear1/Linear2 epilogue、FP8/SF 中间激活、BF16 combine 输出、同步与数值模型 |
| [05-k-grouped-gemm-dataflow.md](05-k-grouped-gemm-dataflow.md) | K-grouped contiguous：K 分段、padding/边界元数据、独立归约与 MoE 权重反向 |
| [06-k-grouped-scheduler-cta-mapping.md](06-k-grouped-scheduler-cta-mapping.md) | K-grouped persistent scheduler：group/MN task 映射、组内 K loop、CTA 分配轨迹 |

## 当前进度

- [x] DeepGEMM / CUTLASS / MegaMoE 项目边界与核心路径
- [x] dispatch workspace、expert slot、metadata 与 remote pull
- [x] `x_sf`、scale 布局、UTCCP 4×32 与 loader/MMA warp 角色
- [x] Linear1 / Linear2 epilogue
- [x] K-grouped GEMM 的 K 维连续布局、分组归约与 MoE 权重反向数据流
- [x] K-grouped scheduler 的 group/MN tile 任务映射与组内 K loop
- [ ] M-grouped / MegaMoE scheduler 的动态 group tile 边界与负载分配
- [ ] L2 epilogue 之后的 top-k combine 完整数据流

下一知识点：对照 M-grouped scheduler，理解各组 M tile 数不同时的 global task 边界与负载分配；之后继续 L2 epilogue 后的 top-k combine 数据流。
