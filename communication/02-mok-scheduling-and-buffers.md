# MoK：设备端调度、pull/push 与环形 Token Buffer

> 承接 [01-proxy-gdaki-gpi.md](./01-proxy-gdaki-gpi.md)。本文整理 Cursor Mixture-of-Kittens（MoK）在 MoE 训练中的通信与调度设计，并把 DeepEP 风格的 push 路径和 buffer 使用方式放在一起对照。
>
> 主要参考：
> - [Mixture-of-Kittens: A Megakernel for MoE Training](https://cursor.com/blog/mixture-of-kittens)
> - [MoK functional API](https://github.com/cursor/mixture-of-kittens/blob/22fc95ae6e331a738c4a58a227a8b03cac586e12/mok/functional.py)
> - [MoK device-side scheduler](https://github.com/cursor/mixture-of-kittens/blob/22fc95ae6e331a738c4a58a227a8b03cac586e12/csrc/scheduler.cuh)
> - [DeepEP repository](https://github.com/deepseek-ai/DeepEP)

---

## 0. 一句话理解

MoK 不是单纯把 MoE 的 grouped GEMM 换成更快的 kernel，而是把整层 MoE 做成一个运行在 NVL72 scale-up 域内的设备端状态机：

```text
设备端 schedule
+ 前向 dispatch 使用 pull
+ 前向 combine 使用 push
+ minibatch 重叠通信与专家计算
+ macrobatch 环形 token buffer
+ 常驻 megakernel
```

它优化的核心对象从单个 GEMM kernel 扩大到了：

```text
谁拥有 schedule
谁发出就绪信号
通信由源端 push 还是目的端 pull
buffer 何时可以被下一批覆盖
哪些 SM 负责通信，哪些 SM 负责计算
```

---

## 1. 传统 MoE 路径的边界成本

常见训练实现会把一次 MoE 路径拆成：

```text
router
-> collective / all-to-all dispatch
-> grouped GEMM
-> elementwise kernel
-> collective / all-to-all combine
```

每个阶段之间都要形成 kernel 边界：

```text
上一个 kernel 完成
-> 下一个 kernel 才能确认哪些 token 已经到达
-> CPU 可能参与计数、buffer 分配或 launch
-> 继续下一阶段
```

在稳态训练中，这种边界会造成：

- dispatch、专家计算和 combine 难以充分重叠
- CPU 可能被拉进 token count 和 buffer 分配的等待链
- 动态 token 数容易触发 D2H 同步
- 为了避免动态分配，部分实现会设置容量上限并丢弃溢出 token
- 通信 SM、计算 SM 和跨机通信之间缺少统一的资源仲裁

MoK 对应的四个主要优化是：

1. 用目的端 schedule 统一描述路由结果。
2. 为前向和后向的四个阶段分别选择 pull 或 push。
3. 用 minibatch 和 SM 软件分区隐藏通信。
4. 用 macrobatch 环形 buffer 消除热路径里的 CPU 分配和 token dropping。

---

## 2. 对称内存 workspace

MoK 从一个对称内存 workspace 开始。

每个 expert-parallel rank 分配相同布局：

```text
input buffer
output buffer
gradient buffer
routing / token buffer
schedule metadata
barrier / counter
```

通过 PyTorch symmetric memory 建立 peer buffer 指针后，GPU 可以用单边访问直接读取或写入远端 rank 的数据。

```text
local address
+ peer offset
= remote buffer address
```

热路径因此不需要让 CPU 为每个 minibatch 重新准备远端通信描述。

数据路径可以是：

```text
GPU issue / TMA / NVLink load-store
-> remote GPU memory
```

而不是：

```text
GPU 通知 CPU
-> CPU 准备描述
-> CPU 提交
-> NIC 访问 GPU 内存
```

这为后面的目的端 schedule 和设备端 pull 提供了固定地址基础。

---

## 3. 设备端 scheduler 与目的端 schedule

### 3.1 Router 输出的到底是什么

router 对每个 token 产生 expert 分数：

```text
logits = hidden_states @ W_router.T
scores = softmax(logits)
top_experts = topk(scores, k)
```

CPU 通常只知道静态形状：

```text
token 数
hidden dimension
expert 数
top-k
```

它不知道：

```text
每个 token 选中了哪些 expert
每个 expert 接收到多少 token
每个 peer 向本地专家发送多少 token
每个 expert 的 prefix-sum offset
```

这些结果由 GPU 上的 router 产生。

### 3.2 一个动态数量的例子

假设有 4 个 token，4 个 expert，top-2。

第一批数据可能得到：

| token | top-2 |
|---|---|
| t0 | E0, E2 |
| t1 | E0, E1 |
| t2 | E1, E3 |
| t3 | E0, E1 |

统计结果：

```text
E0: 3
E1: 3
E2: 1
E3: 1
```

换一批同样形状的输入：

| token | top-2 |
|---|---|
| t0 | E1, E2 |
| t1 | E2, E3 |
| t2 | E0, E2 |
| t3 | E1, E3 |

统计结果变成：

```text
E0: 1
E1: 2
E2: 3
E3: 2
```

如果 top-k 固定，routed token 的副本总数固定：

```text
总 routed token = token 总数 * top_k
```

但每个 expert、每个 rank 的接收数量是动态的。buffer 容量、prefix sum、写地址和通信计划都依赖这个动态分布。

CPU 如果一定要知道这些数量，就需要把计数拷回 host：

```python
counts_cpu = expert_counts.cpu()
```

这通常引入 device-to-host copy 和同步，打断流水线。

### 3.3 MoK 的 scheduler 做什么

router 的 `top_experts` 进入设备端 scheduler。scheduler 统计：

```text
每个本地 expert 从每个 peer 接收多少 token
```

然后把每个 expert 的 token 数按 256 对齐，并构造两个核心数组：

```text
peer_rank[i]
peer_token_idx[i]
```

一行 schedule 的语义是：

```text
本地专家输入位置 i
应该从 peer_rank[i] 这个 rank
读取 peer_token_idx[i] 这个 token
```

这也叫目的端 schedule，因为持有它的 expert rank 知道“我应该从哪里拉什么”。

schedule 在最坏情况下通常只有数 MB，构建时间低于整个 MoE runtime 的 3%。

---

## 4. 同一张 schedule 驱动四次通信

MoK 没有机械地对所有阶段使用相同方向，而是按阶段选择：

| 阶段 | 方向 | 谁发起 | 数据语义 |
|---|---|---|---|
| 前向 dispatch | pull | 专家 / 目的端 | 从 token 源 rank 拉输入 |
| 前向 combine | push | 专家 / 源端 | 把专家结果写回 token 原 rank |
| 反向 reverse-combine | pull | 目的端 | 拉反向阶段所需数据 |
| 反向 reverse-dispatch | push | 源端 | 把梯度写回对应位置 |

如果从阶段相对关系看：

```text
dispatch:
source = token owner
destination = expert owner

combine:
source = expert owner
destination = token owner
```

所以“源端”和“目的端”不能脱离具体阶段使用。

### 4.1 DeepEP 风格与 MoK 的方向差异

DeepEP 风格的 dispatch 和 combine 更接近发送方主动发送：

```text
DeepEP dispatch:
token source push -> expert destination

DeepEP combine:
expert source push -> token destination
```

MoK：

```text
forward dispatch:        expert destination pull <- token source
forward combine:         expert source push -> token destination
reverse-combine:         pull
reverse-dispatch:        push
```

因此不能简单说“DeepEP 全是 push，MoK 全是 pull”。准确说法是：

> MoK 在 forward dispatch 和反向的对应阶段改成目的端 pull，但在 combine 和另一段反向通信中仍然是 push。

### 4.2 为什么 dispatch 适合 pull

专家端已经持有目的端 schedule：

```text
peer_rank[i]
peer_token_idx[i]
```

它不需要源端知道远端写入地址，也不需要每个源端在写完后向大量 peer 发 ready 信号。

如果改用 push，源端通常还需要：

- 知道远端 expert 的 buffer 布局
- 计算远端写入 offset
- 处理目标端的容量和 backpressure
- 写入后做 memory fence
- 向目标端发送就绪信号

在极端情况下，一个 rank 可能要把状态通知到最多 71 个 peer。

### 4.3 Pull 不一定搬运得更少

文章给出的 256×256 BF16 tile 对照：

```text
push: 约 159.6 KB
pull: 约 172.0 KB
```

单看字节数，pull 并不占优。

但 NVL72 的单个 GPU 有 1.8 TB/s NVLink 带宽，需要同时利用两个方向。专家负载不均衡时，目的端各自拉取自己需要的数据，可以获得更好的链路利用。

报告中的结果包括：

```text
带宽利用率最高提升 29%
push dispatch 信号阶段 103 μs
pull dispatch 信号阶段 18 μs
缩短约 5.8 倍
```

结论是通信方向不能只看 payload 字节，还要看：

- 远端状态的所有权
- 就绪信号扇出
- 双向链路利用率
- 负载不均衡时的调度自由度

---

## 5. Schedule 到底在源端还是专家端

MoK 的 dispatch schedule 主要在专家端，也就是当前阶段的接收方构建和使用。

但每张 GPU 都同时扮演两个角色：

```text
对于本地 token：它是 source rank
对于本地 expert：它是 expert / destination rank
```

所以更准确的表述是：

```text
source GPU:
运行 router
提供本地 token
让路由事实对 expert 端可见

expert GPU:
统计本地专家的入站 token
生成本地 schedule
根据 schedule 主动 pull
```

举例：

```text
rank0 持有 t0, t1
rank1 持有 t2, t3
rank2 持有 expert E0
```

router 结果是：

```text
t0 -> E0
t1 -> E0
t2 -> E0
t3 -> 其他 expert
```

rank2 得到本地 schedule：

```text
peer_rank      = [0, 0, 1]
peer_token_idx = [0, 1, 2]
```

随后 rank2 执行：

```text
pull rank0[t0]
pull rank0[t1]
pull rank1[t2]
```

源端不需要知道 E0 在 rank2 本地 buffer 的哪个位置，也不需要逐个发送 ready 信号。

这就是 expert-side schedule 和 pull 的直接联系：

```text
schedule 决定“从哪里拉什么”
pull 由持有 schedule 的专家端执行
```

---

## 6. Minibatch 与 megakernel

### 6.1 Minibatch 解决层内重叠

minibatch 不是单纯把 batch 切小，而是制造独立的通信和计算工作单元。

```text
通信 SM:
dispatch microbatch t

计算 SM:
expert GEMM microbatch t-1

通信 SM:
combine microbatch t-2
```

这样可以形成层内 pipeline。

它和 pipeline parallelism 的 microbatch 思路相似，但作用域不同：

```text
PP microbatch:
不同 pipeline stage 计算不同 microbatch

MoE minibatch:
同一 MoE 层里重叠 dispatch、专家计算和 combine
```

### 6.2 Minibatch 大小怎么选

粒度太细：

- 专家 GEMM 矩阵太小
- Tensor Core 吃不满
- kernel 启动和同步占比上升

粒度太粗：

- dispatch、计算和 combine 之间出现启动空洞
- 最后一部分计算形成尾部空洞

MoK 用“至少覆盖两个完整计算 wave”作为起点，再通过 benchmark 调整。

Kimi 2.5 形状的公开数据：

```text
hidden = 7168
expert intermediate = 2048
计算 CTA = 148

模型预测 minibatch 至少需要 2368 token
实测 2560 token 约 3.425 ms
实测 512 token 约 5.981 ms
公开代码默认 4096
```

minibatch、前后向通信 SM 数和 macrobatch 大小都保留为显式调参项。

### 6.3 SM 软件分区

Megakernel 内部将 SM 软件分区：

```text
通信 SM:
TMA 搬运
dispatch
combine
barrier / signaling

计算 SM:
up projection
gate
SwiGLU
down projection
```

本地计数器承担通信与计算之间的就绪信号。

TMA 的特点是只需要不到三分之一的通信 SM 就可能压满 NVLink，因此大部分 SM 可以留给专家计算。

共享专家也会被塞进第一段 dispatch 的空隙，以减少单独执行造成的暴露时间。

### 6.4 常驻 kernel 与 CLC

常驻 megakernel 能精确控制通信 SM 数量，并把细粒度状态留在 GPU 内。

但常驻 kernel 也可能长期霸占 SM。Blackwell 的 Cluster Launch Control（CLC）允许更高优先级工作取消尚未启动的 cluster，使 RDMA、FSDP all-gather 等跨机通信仍能及时获得 SM。

---

## 7. Macrobatch 环形 token buffer

### 7.1 动态 token 数为什么容易把 CPU 拉回热路径

MoE 路由后的总量和分布会随数据变化。常见实现有两类路径。

动态分配：

```text
GPU router
-> D2H token count
-> CPU 同步等待
-> CPU 分配 / 调整 buffer
-> 再 launch dispatch
```

它不丢 token，但会阻塞流水线。

固定容量并丢弃：

```text
CPU 预先固定容量
-> token 超过上限时丢弃
```

它避免动态分配，但改变训练语义。

MoK 希望同时做到：

```text
不等待 CPU
不重新分配
不丢 token
允许数量波动
```

### 7.2 物理分配和运行时分配要分开看

启动阶段，CPU 仍然负责：

- 分配 symmetric memory workspace
- 注册 buffer
- 交换 peer 指针
- 建立通信资源
- 配置 barrier 和常驻 kernel

训练热路径，GPU 负责：

```text
router
-> top-k
-> 每 expert / 每 peer 的 token count
-> prefix sum
-> 256 对齐
-> schedule
-> ring slot 选择
-> address offset 计算
-> dispatch / compute / combine
```

这里的“分配”不是运行时 `cudaMalloc`，而是在预分配空间里做逻辑分配：

```text
哪个 slot
哪个 expert 区域
哪个 token offset
从哪个 peer 拉
写到哪个位置
```

可以类比成：

```text
CPU 先建好停车场
GPU 根据实时 token 数决定每批 token 停到哪个车位
```

不是每来一批 token 就重新建一个停车场。

### 7.3 环形 slot 的生命周期

一个 macrobatch 的数据要经历：

```text
dispatch -> expert compute -> combine
```

如果只有一个固定区域，下一轮 dispatch 必须等上一轮 combine 完全结束。

环形 buffer 将空间分成多个 slot：

```text
slot 0
slot 1
...
slot N-1
slot 0 再次复用
```

流水线可以变成：

```text
时间 ->
macrobatch t:     dispatch -> compute -> combine
macrobatch t+1:              dispatch -> compute -> combine
macrobatch t+2:                       dispatch -> compute -> combine
```

不同资源可以同时处理不同阶段：

```text
通信 SM 处理某个 minibatch 的 dispatch / combine
计算 SM 处理另一个 minibatch 的 expert GEMM
下一 macrobatch 开始向其他 slot 写数据
```

### 7.4 细粒度 barrier 解决覆盖安全

一个 slot 不应被当成一整块只能加一把锁的内存。更合理的方式是把生命周期拆细：

```text
dispatch 是否写完？
compute 是否可以读？
combine 是否已经读完？
这个区域是否可以覆盖？
activation 是否还要留给 backward？
```

只要访问区域不冲突，或者在 barrier 保护下满足先后关系，上一 macrobatch 的 combine 就可以和下一 macrobatch 的 dispatch 交错。

### 7.5 “无需 token dropping”的边界

它不代表容量可以无限增长。MoK 仍然需要预留一个上界：

```text
固定容量 >= 设计范围内 macrobatch 的峰值 token 数
```

运行时变化通过 device schedule 和有效 offset 吸收：

```text
token 数变化 -> 改变有效 schedule / count
容量不变化 -> 不重新分配，也不丢 token
```

代价是必须按峰值预留显存。文章说环形 buffer 通常只有数百 MB，相对于权重、activation 和 optimizer state 属于可接受成本。

---

## 8. 反向执行、排序与确定性

### 8.1 反向遍历 macrobatch

MoK 在前向中反向遍历 macrobatch。进入 backward 后，最近写入的 activation 会最先消费，有利于减少 activation replay 和额外存储。

### 8.2 dgrad 与 wgrad

反向依赖顺序是：

```text
先完成所有 minibatch 的 dgrad
再累计整个 token 集合上的 wgrad
```

reverse-dispatch 不依赖 wgrad，因此可以先推进关键路径。

wgrad 推迟到全部 dgrad 完成后统一归约，既减少关键路径阻塞，也能在完整 token 集合上获得更稳定的数值。

### 8.3 融合与确定性

MoK 还把：

- router weight gradient 融入 SwiGLU backward
- dispatch 中融合 activation 的 MXFP8 量化

共享专家保持 BF16，路由专家支持 BF16 和 MXFP8。

固定 schedule 和固定浮点运算顺序使前后向具备 bitwise determinism。对于内部消融实验和 on-policy RL，这能降低执行顺序漂移带来的噪声。

---

## 9. Microbatch 与 DeepEP Buffer 的关系

### 9.1 Microbatch 不等于“只有一个 buffer”

microbatch 的目标是制造可以流水化的独立工作单元。

但能否重叠，还取决于每批数据是否拥有独立的 buffer 生命周期。

有两种设计：

```text
每批使用不同 buffer / ring slot
-> 多批通信和计算可以同时存在

所有批共用同一块通信工作区
-> 需要 event / barrier 保证使用不冲突，必要时会串行
```

所以 microbatch 本身只提供流水化机会，不自动保证多批 buffer 可以并发复用。

### 9.2 DeepEP 不是只有一块物理 buffer

旧版 DeepEP 的 `Buffer` 明确有：

```text
num_nvl_bytes
num_rdma_bytes
```

分别对应 NVLink 和 RDMA 通信区域。

当前主线的 `ElasticBuffer` 统一了接口，但 `num_bytes` 描述的是总 buffer 大小，并明确不一定包含 workspace。此外还有：

- metadata / handle
- top-k 和 prefix-sum
- source token metadata
- buffer slot metadata
- channel 和 QP 等传输资源

可以概括为：

```text
ElasticBuffer
├── NVL / RDMA 数据区域
├── workspace
├── metadata / handle
├── communication channel
└── QP / transport resources
```

### 9.3 为什么会产生“一个 buffer”的印象

常见用法是：

```python
buffer = ElasticBuffer(...)

buffer.dispatch(...)
buffer.combine(...)
buffer.dispatch(...)
buffer.combine(...)
```

一个 EP group 或一个模型层通常复用一个 `Buffer` 对象，而不是每个 token 或每个 minibatch 创建一个 buffer。

但这表示 API 层复用一个对象，不表示内部只有一块存储。实际部署也可以按层、model chunk、pipeline stage 或 EP group 创建多个 buffer。

### 9.4 MoK ring 与 DeepEP workspace 的对照

| 维度 | DeepEP 常见模式 | MoK |
|---|---|---|
| 顶层对象 | 一个或多个 `Buffer` / `ElasticBuffer` | 预分配 communication workspace |
| 数据区域 | NVL、RDMA、workspace 等 | 以 macrobatch 为单位的 ring slots |
| 容量策略 | 按 `num_max_tokens_per_rank` 预留 | 覆盖 macrobatch 峰值 |
| layout | dispatch layout / handle / prefix sum | device scheduler 生成目的端 schedule |
| token count | 某些路径可能 CPU sync，也支持 cached handle / 关闭部分同步 | 热路径尽量不回到 CPU |
| buffer 生命周期 | 复用通信 workspace | 多个 slot 跨 macrobatch 轮转 |
| 流水重叠 | 依赖 async、event 和用户使用方式 | 明确用 slot + barrier 支持相邻 macrobatch 交错 |

更准确的 DeepEP 结论是：

> DeepEP 通常复用一个通信 Buffer 对象，但对象内部包含多种数据区域和资源，不是只有一个物理 buffer。它也没有对上层暴露一个像 MoK 一样、专门跨 macrobatch 轮转复用的 ring slot 抽象。

### 9.5 Dispatch 方向的对照

DeepEP：

```text
token source push -> expert destination
expert source push -> token destination
```

MoK：

```text
expert destination pull <- token source
expert source push -> token destination
```

核心区别不只是数据操作方向，而是：

```text
谁持有 schedule
谁负责远端写入地址
谁发出 ready 信号
谁决定本地 buffer 布局
谁承担负载均衡
```

---

## 10. 收益数字应该怎样读

Cursor 在一台 GB300 NVL72 上测试了四组接近主流大模型的 MoE 形状，包括：

```text
256 - 512 个专家
top-6 到 top-10 路由
BF16 / MXFP8
```

对比基线包括：

- NCCL + PyTorch
- DeepEP + PyTorch
- DeepEP + Transformer Engine
- HybridEP + Megatron

报告峰值加速：

| 模式 | 峰值加速 |
|---|---:|
| MXFP8 forward | 2.37x |
| MXFP8 backward | 1.78x |
| BF16 forward | 1.92x |
| BF16 backward | 1.58x |

更接近真实训练的 512 GPU 结果：

```text
DeepEP + 自定义 MXFP8 kernel: 760.9 tokens/s/GPU
MoK:                          1070.2 tokens/s/GPU
端到端提升:                   1.41x
吞吐增加:                     41%
```

这些数字来自项目团队自己的硬件和训练栈。仓库提供单层 benchmark，但 512 GPU 的生产训练结果依赖 Cursor 内部 Composer 栈，外部不能完整复现。

因此：

```text
2.37x 不适合直接外推为整机训练收益
1.41x 更能体现端到端收益
```

---

## 11. 适用场景与边界

MoK 的甜点区比较明确：

```text
DeepSeek-V3 风格的训练 MoE
expert parallel group 位于单一 GB200 / GB300 NVL72 NVLink domain
专家负载存在波动
通信与同步占可观比例
团队愿意按模型形状调 minibatch、macrobatch 和通信 SM 数
```

当前公开实现要求：

```text
Blackwell SM100 / SM103
CUDA 13
PyTorch 2.10
有限的 expert-parallel size
```

不提供：

- Hopper 的完整可移植实现
- AMD 实现
- 普通 PCIe 集群实现
- 直接用于在线推理的 runtime

MoK 主要优化 scale-up 域内的一层 MoE。跨 rack 仍依赖 FSDP、RDMA 等外层并行。

小规模 MoE、专家计算远大于通信、或者输入形状频繁变化的任务，未必能摊平 schedule 和 megakernel 的调优成本。

---

## 12. 一页速记

```text
问题:
collectives + grouped GEMM + elementwise kernel 形成边界
动态 token 数容易导致 CPU sync、动态分配或 token dropping

MoK:
1. symmetric memory 提供固定 peer 地址
2. device scheduler 生成目的端 schedule
3. forward dispatch pull
4. forward combine push
5. backward 对应阶段使用 pull / push 组合
6. minibatch 让通信与专家计算重叠
7. SM 软件分区 + TMA + device counter
8. macrobatch ring buffer 让 slot 跨代复用
9. 细粒度 barrier 保证覆盖安全

关键判断:
通信成本不只是字节数
还要看远端状态所有权、信号扇出和双向链路利用率

DeepEP:
dispatch / combine 更接近 source push
一个 Buffer 对象不等于一块物理 buffer
内部有 NVL / RDMA / workspace / metadata / QP 等资源
通常复用通信 workspace，但没有 MoK 式 macrobatch ring 抽象

最终理解:
MoK 把 MoE 优化的单位从单个 GEMM kernel
提升成了整层的设备端状态机和流水线
```

---

## 13. 源码阅读顺序

建议按下面顺序核对：

1. `mok/functional.py`
   - `build_schedule`
   - `forward`
   - `backward`
   - schedule 和 handle 的生命周期

2. `csrc/scheduler.cuh`
   - token count
   - peer count
   - 256 对齐
   - `peer_rank`
   - `peer_token_idx`

3. megakernel 主体
   - 通信 SM 与计算 SM 分区
   - TMA pipeline
   - minibatch overlap
   - barrier / counter

4. DeepEP 对照
   - `legacy.py` 中的 `num_nvl_bytes` 和 `num_rdma_bytes`
   - `elastic.py` 中的 `ElasticBuffer` 与 `EPHandle`
   - `get_buffer_size_hint` / `num_max_tokens_per_rank`
   - dispatch layout、cached handle 和 buffer slot metadata
