# DeepGEMM 学习笔记

本目录整理对话中逐步梳理的 DeepGEMM / MegaMoE 知识点：先看项目定位，再跟 dispatch/pull 和量化 scale 数据流，最后看 epilogue。笔记用概念图、坐标例子与同步边界解释源码行为，不复制整份仓库源码。

| 文档 | 知识点 |
|---|---|
| [01-deepgemm-project-and-kernel-model.md](01-deepgemm-project-and-kernel-model.md) | DeepGEMM 与 CUTLASS 的范围差异、MegaMoE 在仓库中的位置、`tcgen05.mma` 与 UMMA、Gluon 对照目标 |
| [02-megamoe-dispatch-pull.md](02-megamoe-dispatch-pull.md) | workspace、expert send count、atomic slot offset、metadata、同步发布、远端 pull 与 TMA chunk |
| [03-megamoe-x-sf-utccp-and-warp-roles.md](03-megamoe-x-sf-utccp-and-warp-roles.md) | `x_sf`、SFA/SFB、A/SFA 与 B/SFB loader warp、UTCCP 4×32 转换的索引轨迹和原因 |
| [04-megamoe-l1-l2-epilogue.md](04-megamoe-l1-l2-epilogue.md) | Linear1/Linear2 epilogue、FP8/SF 中间激活、BF16 combine 输出、同步与数值模型 |

## 当前进度

- [x] DeepGEMM / CUTLASS / MegaMoE 项目边界与核心路径
- [x] dispatch workspace、expert slot、metadata 与 remote pull
- [x] `x_sf`、scale 布局、UTCCP 4×32 与 loader/MMA warp 角色
- [x] Linear1 / Linear2 epilogue
- [ ] L2 epilogue 之后的 top-k combine 完整数据流
- [ ] DeepGEMM MegaMoE 调度策略与 grouped/MoE scheduler 细节

下一知识点：沿 combine buffer 和 source metadata，逐 token 追踪多个 top-k expert 结果如何聚合回输出。
