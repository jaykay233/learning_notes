# DeepGEMM：项目定位、GEMM 主线与 CUTLASS 的区别

## 本次讲解位置

```text
章节：DeepGEMM 源码学习
小节：项目定位与 kernel 执行模型
知识点：DeepGEMM 是什么、MegaMoE 是否在仓库中、和普通 CUTLASS GEMM 的区别
上次：Blackwell tcgen05 / TMEM 与 MegaMoE UTCCP SF 布局基础
下次：MegaMoE dispatch 如何统计 expert token 并分配 slot
PTX：tcgen05.mma 是具体 PTX 指令；UMMA 是 Blackwell 矩阵乘路径/抽象称呼，不是第二个独立 Tensor Core
```

### 为什么现在讲这个

如果把 DeepGEMM 看成“一个 GEMM kernel”，会漏掉它为了真实推理负载所实现的调度、量化、路由、跨 GPU 数据搬运与 combine。如果把 `tcgen05.mma` 和 UMMA 当作两套硬件，也会把指令层与库/编程模型层混淆。本节先划清项目、kernel、硬件指令三个边界，后面的 MegaMoE 数据流才有上下文。

## 1. 一句话心智模型

**CUTLASS** 提供可组合的 GPU kernel 构件和 GEMM 模板/抽象；**DeepGEMM** 是面向高性能 GEMM 与 MoE 场景的一组具体实现，会把 Blackwell 计算、数据搬运、调度、量化布局、同步和通信组织成完整路径。

```text
普通 GEMM 的概念流程：
  读取 A/B tile -> Tensor Core 累加 -> epilogue -> 写 C

DeepGEMM MegaMoE 的系统级流程：
  routed token 分发/计数
    -> 远端 pull token 与 scale
    -> 按 expert 调度 GEMM tile
    -> Linear1 + L1 epilogue
    -> Linear2 + L2 epilogue
    -> 跨 rank combine
```

并不是所有 DeepGEMM kernel 都走 MegaMoE 这条路径；dense GEMM、grouped GEMM 和 MoE dispatch/combine 是不同 workload 与实现分支。

## 2. DeepGEMM 和 CUTLASS：不是二选一

| 维度 | CUTLASS（常见理解） | DeepGEMM |
|---|---|---|
| 定位 | GPU kernel 构件、模板和高性能 GEMM 实现框架 | 面向特定 GEMM/MoE workload 的 kernel 与完整执行路径 |
| 核心关心点 | operand/tile/layout、mainloop、epilogue、调度策略的组合 | 除 GEMM 外，也关心 token/expert 调度、FP8/FP4 scale、workspace、跨 rank 数据流 |
| 运行范围 | 一个 GEMM kernel 可以只负责矩阵 tile 计算 | MegaMoE 一个实现可能承担 dispatch/pull、两层 GEMM、epilogue、combine 协调 |
| 关系 | 可直接使用 CUTLASS/CuTe 的类型、原语与模板 | 能使用 CUTLASS/CuTe；专用 scheduler 和 MoE 流程仍是 DeepGEMM 自己组织的逻辑 |

所以“普通 CUTLASS GEMM”并非低级或慢，而是讨论范围通常更窄：给定 A、B、C，解决这个矩阵乘的 tile、流水和写回。MegaMoE 还需要回答“每个 token 属于哪个 expert、数据在哪个 rank、结果写回到谁的 top-k 槽”。

## 3. MegaMoE 在不在 DeepGEMM 仓库里

在。仓库有 MegaMoE 专用实现文件，不是另一个独立仓库里的概念。例如当前 checkout 的文件：

`/Users/saboxu/Downloads/DeepGEMM/deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh`

该路径名称也提示了实现的适用范围：SM100/Blackwell 路径、FP8/FP4 与 MegaMoE 组合。读源码时要从 dispatcher/launch 路径找到 kernel 配置，再跟到具体实现；不能由这个文件推断所有 GPU 架构、所有 MoE 配置都共用同一个 kernel。

## 4. 一次 MegaMoE GEMM 的高层核心流程

```text
A. 准备与分发
   routed token -> expert 计数 -> metadata/slot -> global 同步

B. 输入拉取
   从源 rank 的 symmetric buffer 拉 FP8 token 和 scale
   -> 本地按 expert 排列的输入池

C. Expert GEMM
   scheduler 取 expert / M / N 工作块
   -> TMA 搬 A、B（及 scale）到 SMEM
   -> UMMA / tcgen05.mma 在 Tensor Core 上累加到 TMEM

D. 两段输出处理
   Linear1 epilogue：SwiGLU、路由权重、量化，为 Linear2 准备输入
   Linear2 epilogue：BF16 输出写入目标 combine 槽

E. Combine
   把相同原始 token 的 top-k expert 输出做最终聚合
```

每一步解决不同问题：workspace/meta 解决身份和位置，TMA 解决大块搬运，TMEM/UMMA 解决 Tensor Core 累加，epilogue 解决数据格式和去向，combine 解决 top-k 归并。

## 5. `tcgen05.mma` 与 UMMA 的层次

| 名称 | 所在层次 | 怎么理解 |
|---|---|---|
| UMMA | Blackwell 矩阵乘执行路径/编程抽象 | 指 Tensor Core 的异步矩阵乘方式，和 shared memory 输入、TMEM accumulator 等协同 |
| `tcgen05.mma` | PTX 指令层 | 发起相应 tcgen05 矩阵乘操作的具体指令 |
| `tcgen05` | 指令族前缀 | 不只有 mma，还包括与 Tensor Memory、copy、wait 等有关的指令 |

可以把 `tcgen05.mma` 理解为“通过 tcgen05 PTX 接口发起 Blackwell UMMA 路径的一条具体指令”。但不要反过来把 `tcgen05` 整个指令族都叫 UMMA，也不要认为 CuTe/CUTLASS 的 UMMA 名称意味着另有一个与 tcgen05 无关的矩阵乘单元。

## 6. Triton/Gluon 对照实现该如何定位

用 Triton/Gluon 学习 DeepGEMM，最佳目标通常是**复现并解释执行模型**，不保证逐项达到生产 kernel 的硬件特性或性能：

- 先实现数学语义与 tile 分解；
- 再对照 TMA/async copy、warp specialization 与 pipeline；
- 然后研究 TMEM/Blackwell MMA、block-scaled scale 布局；
- 最后才是 CTA cluster、multicast、UTCCP、NVLink symmetric buffer 和复杂 grouped/MoE scheduler。

此前讨论拆出的高难度对照项包括：

| 特性 | 需要验证的核心问题 |
|---|---|
| 多角色 warp specialization | producer/consumer 如何并行、SMEM stage 如何交接、barrier 生命周期如何闭合 |
| 生产级多 stage 调度 | stage 数、prefetch 距离、stall/复用条件是否与工作量匹配 |
| CTA cluster multicast / 2-SM UMMA-TMEM | 多 CTA 的共享范围、TMEM ownership、cluster 同步是否被编译器和目标硬件支持 |
| FP8/FP4 block-scaled MMA | 数据编码、SFA/SFB 形状和 TMEM 放置是否匹配硬件契约 |
| UTCCP 与 SF 数据路径 | SMEM → TMEM 的专用 copy 是否可表达、scale layout 是否逐坐标正确 |
| grouped/MoE scheduler | expert token 数动态变化时，tile 如何分配且不丢不重 |
| 完整 epilogue 组合 | 激活、路由权重、量化、SF 写回和输出路由的数值/同步顺序是否一致 |

这些能力是否能直接由 Triton/Gluon 表达，取决于 Triton 版本、编译器 lowering、GPU 架构以及接口暴露情况。有些需要底层 PTX/C++ 扩展，有些属于跨 CTA/rank 的系统调度，不能仅凭一个普通 Triton kernel 宣称已经实现。学习实现应明确标注“数学正确”“布局对照”“异步/并发协议正确”“目标硬件路径已验证”“性能已测量”等不同完成度。

## 7. 常见误解和可见后果

| 误解 | 修正 | 可能后果 |
|---|---|---|
| DeepGEMM 就是一条 GEMM 指令 | 它是多种专用 GEMM 与 MoE 路径的集合 | 只看 mainloop，漏掉 dispatch/通信/数据格式约束 |
| MegaMoE 在仓库外 | 仓库内有专用实现 | 找错代码入口或以为讨论的是伪代码 |
| CUTLASS 和 DeepGEMM 互斥 | DeepGEMM 可复用 CUTLASS/CuTe 构件 | 错误地把抽象层和产品化实现当成竞品硬件 |
| `tcgen05.mma` 与 UMMA 是两套硬件 | 一个是具体 PTX 指令，一个是相应执行路径/抽象命名 | 混淆 ISA 与编程模型 |
| Gluon 版只要数学对就等价于 DeepGEMM | 还需验证布局、同步、异步完成、scheduler 与多 GPU 数据路径 | 数值可能对但性能、死锁或并发正确性不等价 |

## 自测（含答案）

1. **MegaMoE 是不是独立于 DeepGEMM 仓库？**  答：不是，仓库内有专用实现路径。
2. **CUTLASS GEMM 和 DeepGEMM MegaMoE 最大的范围差别是什么？**  答：前者常聚焦单次矩阵乘；后者还要处理 token/expert 调度、跨 rank 数据流和 combine。
3. **`tcgen05.mma` 与 UMMA 如何对应？**  答：前者是 PTX 指令，后者是 Blackwell 矩阵乘执行路径/抽象称呼；不是两套独立 Tensor Core。
4. **Gluon 数值结果和参考一致，是否证明复现了 DeepGEMM？**  答：不够，还要验证 layout、同步、硬件 lowering、scheduler 和真实 GPU 路径。
