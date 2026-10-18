# DeepGEMM 学习笔记

本目录整理对话中逐步梳理的 DeepGEMM / MegaMoE 知识点：先看项目定位，再跟 dispatch/pull、量化 scale、epilogue 与 grouped GEMM 数据流。笔记用概念图、坐标例子与同步边界解释源码行为，不复制整份仓库源码。

| 文档 | 知识点 |
|---|---|
| [01-deepgemm-project-and-kernel-model.md](01-deepgemm-project-and-kernel-model.md) | DeepGEMM 与 CUTLASS 的范围差异、MegaMoE 在仓库中的位置、`tcgen05.mma` 与 UMMA、Gluon 对照目标 |
| [02-megamoe-dispatch-pull.md](02-megamoe-dispatch-pull.md) | workspace、expert send count、atomic slot offset、metadata、同步发布、远端 pull 与 TMA chunk |
| [03-megamoe-x-sf-utccp-and-warp-roles.md](03-megamoe-x-sf-utccp-and-warp-roles.md) | `x_sf`、SFA/SFB、A/SFA 与 B/SFB loader warp、UTCCP 4×32 转换的索引轨迹和原因 |
| [04-megamoe-l1-l2-epilogue.md](04-megamoe-l1-l2-epilogue.md) | Linear1/Linear2 epilogue、FP8/SF 中间激活、BF16 combine 输出、同步与数值模型 |
| [05-k-grouped-gemm-dataflow.md](05-k-grouped-gemm-dataflow.md) | K-grouped contiguous：K 分段、padding/边界元数据、独立归约与 MoE 权重反向 |

## 当前进度

- [x] DeepGEMM / CUTLASS / MegaMoE 项目边界与核心路径
- [x] dispatch workspace、expert slot、metadata 与 remote pull
- [x] `x_sf`、scale 布局、UTCCP 4×32 与 loader/MMA warp 角色
- [x] Linear1 / Linear2 epilogue
- [x] K-grouped GEMM 的 K 维连续布局、分组归约与 MoE 权重反向数据流
- [ ] L2 epilogue 之后的 top-k combine 完整数据流
- [ ] DeepGEMM MegaMoE / grouped scheduler 如何将 group 映射到 CTA tile

下一知识点：沿 K-grouped scheduler 的 `grouped_layout`，追踪每个 group 的 K 范围如何映射到 CTA 的 M/N tile；之后继续 L2 epilogue 后的 top-k combine 数据流。
