# Proxy、GDAKI 与 GPI：通信队列、QP 与提交路径

> 通信专题笔记。本文讨论 GPU 发起网络通信时，CPU Proxy、GDAKI/GPUNetIO 和 GPI 三种路径分别由谁提交请求、谁选择 QP，以及为什么会出现按 peer 分队列或按 QP 拆分的实现。
>
> 参考资料：
> - [NVIDIA DOCA GPUNetIO Architecture and Design](https://networking-docs.nvidia.com/doca/archive/3-5-0/gpunetio-architecture-and-design)
> - [NVIDIA DOCA GPUNetIO API Reference](https://networking-docs.nvidia.com/doca/archive/3-5-0/gpunetio-api-reference)
> - 讨论中提供的 Proxy / GDAKI / GPI 伪代码与源码精简

---

## 0. 先记结论

传统 CPU Proxy 并不是新东西。传统网络传输一直有 CPU 侧进度线程：

```text
GPU kernel 把数据写进通信 buffer，并更新 FIFO / producer index
CPU progress thread 看见新工作
CPU 调网络插件，构造 WQE
CPU 把 WQE post 到 QP，并敲 doorbell
NIC 从 GPU 内存直接取数据发送
```

真正变化的是：**通信计划由谁决定，以及 GPU 和 CPU 各自负责到哪一步。**

三种路径可以先用一句话区分：

| 路径 | GPU 做什么 | CPU 做什么 | 谁选择目标 QP |
|---|---|---|---|
| CPU Proxy | 把逻辑工作描述符写进按 peer 分的队列 | 轮询队列，拼 WQE，提交到 QP | CPU |
| GDAKI / GPUNetIO | 直接操作或提交到每个 peer 的 QP | 常规路径不参与 GPU 数据提交 | GPU |
| GPI | 把 GFD 写进 context 级 NIC queue | 常规路径不参与提交 | NIC |

最核心的变化是：

```text
CPU Proxy: GPU -> 按 peer 分队列 -> CPU -> qp[peer] -> NIC
GDAKI:     GPU -> qp[peer] 的 SQ / doorbell -> NIC
GPI:       GPU -> context NIC queue -> NIC 内部选 QP -> 网络
```

---

## 1. 先把名词分开

这些词经常混在一起，但它们的层级不同。

| 名词 | 含义 |
|---|---|
| `rank` | 一个 communicator 内的逻辑身份，通常是 0 到 `N-1` |
| `peer` | 当前 rank 的直接通信对端，通常用 communicator-relative rank 表示 |
| `QPN` | RDMA QP 的编号 |
| `QP` | RDMA Queue Pair，包含发送队列和接收队列 |
| `SQ` | Send Queue，WQE 排队的发送侧 |
| `WQE` | Work Queue Element，网卡原生的发送命令 |
| `DBREC` | Doorbell Record，通常是内存中的 producer index 记录 |
| `doorbell` | NIC 的 MMIO 寄存器。写它表示“发送队列有新工作” |
| `BlueFlame / UAR` | 允许 CPU 或 GPU 把小型 WQE 直接写进 NIC 寄存器的提交区域 |
| `CQ / CQE` | Completion Queue 以及其中的 completion entry |
| `GFD` | GPU-side work descriptor，逻辑通信描述符，不等同于网卡 WQE |
| `context` | 一组通信资源或提交上下文的标识，不等同于一个 peer |

### 1.1 常规语境里的 `peer` 是什么

在 NCCL、MPI 或 RDMA 通信中，`peer` 通常表示：

```text
我这次要直接通信的另一个 rank / endpoint
```

例如 8 卡训练时，一个 rank 的 peer 可能是另外 7 个 rank。

它一般不是：

- 一个 MoE expert
- 一个 token
- 一个 GFD
- 一个 batch
- 一个请求

MoE dispatch 时可能有大量 expert 目的地，但底层通信资源仍然可以按物理 peer 建立。expert id、目标 rank、源 rank 和 token 范围等，可以放进 GFD 或 WQE 里的字段。

所以要先区分三个层级：

```text
业务目标: expert / token / request
通信描述: peer / remote rank / memory handle / size
传输资源: QP / SQ / WQE / memory region
```

不能因为 expert 很多，就自然地推导出必须给每个 expert 建一个 QP。

---

## 2. 传统 CPU Proxy 的模型

### 2.1 AllReduce 的计划为什么适合提前排好

传统 AllReduce 的通信计划通常比较固定：

- 在哪个 ring 或 tree 阶段发送
- 下一个 peer 是谁
- 每次发送多少字节
- 使用哪个 QP

这些信息在 CPU 入队时就可以基本确定。GPU 负责把数据填进 step buffer，并更新 FIFO 的大小或 producer index。CPU progress thread 看到后，通过网络插件把对应的 WQE 提交给 NIC。

一个概念化流程：

```text
GPU:
  写数据
  更新 FIFO / producer index

CPU progress:
  轮询 FIFO
  取出发送意图
  构造 WQE
  ibv_post_send(qp, &wr, &bad_wr)

NIC:
  DMA 读取 GPU 内存并通过网络发送
```

数据路径仍可以是 GPUDirect RDMA，即 NIC 直接读写 GPU 显存。CPU 负责的是控制路径和提交路径，不是必须亲自搬运 payload。

### 2.2 MoE dispatch 为什么打破旧假设

MoE dispatch 的目标在路由结果计算出来之前并不知道：

```text
token -> 哪个 expert -> expert 在哪个 rank
```

这意味着：

- 目标 peer 数量可能动态变化
- 每个 peer 的发送大小可能动态变化
- 不同 token 可能去不同 expert
- 某些 peer 某一步可能没有数据

如果 CPU 必须提前把完整发送对象和长度排好，就会遇到困难。因此通信计划的所有权开始从 CPU 转向 GPU 或 NIC。

---

## 3. Proxy 伪代码逐行解释

讨论中的 Proxy 简化代码：

```python
idx = pis[peer].fetch_add(1)
queues[peer][idx & mask] = GFD(op, src, dst, size)

# CPU（每个 communicator 一个 pinned 线程，死循环）
for peer in range(nRanks):
    gfd = poll_and_clear(queues[peer])
    ibv_post(qp[peer], gfd)     # CPU 拼 WQE、敲 doorbell
```

这是概念性伪代码，不是可直接运行的完整实现。

### 3.1 `pis[peer].fetch_add(1)`

```python
idx = pis[peer].fetch_add(1)
```

含义：

- `pis[peer]` 是某个 peer 的 producer index
- `fetch_add(1)` 原子地返回旧值，再把值加一
- 每个并发的 GPU thread 拿到不同的 `idx`
- `pis` 是 producer indices 的缩写

多个 GPU thread 同时写队列时，不能简单用 `idx += 1`，否则会出现竞态。

### 3.2 `queues[peer][idx & mask]`

```python
queues[peer][idx & mask] = GFD(...)
```

这不是一个语言层面的循环。

它只是执行一次“计算槽位并赋值”：

```text
1. 取出 peer 对应的队列
2. 用 idx & mask 算出环形缓冲区槽位
3. 把 GFD 写入该槽位
```

如果队列容量是 2 的幂：

```text
idx & mask
```

等价于：

```text
idx % capacity
```

例如 `mask = capacity - 1`：

```text
capacity = 8, mask = 7
idx:  0 1 2 3 4 5 6 7 8 9 ...
slot: 0 1 2 3 4 5 6 7 0 1 ...
```

`idx` 一直单调增加，真正的槽位循环复用。

### 3.3 `queues[peer]` 的语义

`queues[peer]` 不是一个普通 Python list，更接近：

```text
GPU 可写、CPU 可读的 pinned / shared memory mailbox
```

它按 peer 分开，主要用于：

- 保留 peer 内顺序
- 避免不同 peer 互相阻塞
- 给不同 peer 独立 backpressure
- 让 CPU 可以按 peer 分片轮询

CPU 端：

```python
gfd = poll_and_clear(queues[peer])
ibv_post(qp[peer], gfd)
```

表示：

1. 查看该 peer 是否有新 GFD
2. 取走工作并清除 ready 标记
3. 把逻辑描述转换成 NIC WQE
4. 提交到该 peer 的 QP
5. 更新 DBREC 或敲 doorbell

真实实现通常还需要：

- memory fence / acquire-release
- sequence number
- ready / valid bit
- 环形队列满时 backpressure
- 批量 poll
- error handling
- CPU 与 GPU 的缓存一致性处理

---

## 4. GFD、WQE 和 QP 不是一回事

这是理解 QP 是否爆炸的关键。

```text
GFD: 逻辑工作，例如 WRITE 到 peer P 的某个 memory handle
WQE: NIC 真正读取的发送命令
QP:  承载 WQE 的发送 / 接收队列和连接状态
```

一个 QP 可以承载很多 WQE，一个 WQE 可以描述很多字节，一个 GFD 也不要求独占 QP。

因此：

```text
大量 GFD != 大量 QP
大量 token != 大量 QP
大量 expert != 大量 QP
```

是否要建新 QP，取决于传输语义和物理连接，而不是取决于上层业务对象数量。

---

## 5. QP 爆炸问题

### 5.1 RC QP 的粒度

RDMA RC QP 通常是有连接语义的：

```text
一个本地 RC QP 对应对端的一个 QPN
```

如果每个 rank 都要直接和所有其他 rank 通信，那么单个 rank 的 QP 数量大致是：

```text
N - 1
```

如果把整个系统所有方向的连接都展开，则大约是：

```text
N * (N - 1) / 2
```

也就是：

```text
O(N^2)
```

8 个 rank 时还不明显，但几百到几千个 rank 时，QP 数量、状态、内存和建链时间都会成为问题。

### 5.2 哪些设计容易导致爆炸

高风险映射：

| 对象粒度 | 风险 |
|---|---|
| 每个 expert 一个 QP | expert 数量可能远大于 rank |
| 每个 token 一个 QP | 每步创建和销毁几乎不可行 |
| 每个 source-expert / destination-expert 对一个 QP | 组合数爆炸 |
| 每个 batch 或 request 一套 QP | 生命周期短，资源复用差 |
| 每个 communicator 再乘完整 peer 矩阵 | 多 communicator 相乘 |
| TP / EP / PP / DP 各自独立建满 | 通信组数量相乘 |

如果系统还按 layer、channel、rail 或并行流继续拆，QP 数量还会继续放大。

### 5.3 更稳妥的映射

优先让 QP 粒度贴近物理传输 peer：

```text
QP 连接: local rank <-> remote rank / remote QPN
GFD/WQE: 描述 expert、token range、memory handle、size 和 op
```

也就是说：

```text
QP 负责“和谁连”
descriptor 负责“这次传什么”
```

这样可以避免把上层动态路由直接映射成底层连接数量。

如果业务想隐藏大量 QP，可以考虑：

- QP 池化
- 多消息复用一个 QP
- 使用支持更细粒度目标选择的传输语义
- 由 NIC 根据描述符自动选择内部 QP

最后一条正是 GPI 与 GDAKI、Proxy 最大的结构差异之一。

---

## 6. GPUNetIO 的提交方式

GDAKI / GPUNetIO 讨论的几种模式，核心都在回答：

```text
WQE 放在哪里？
要不要更新 DBREC？
谁敲 doorbell？
NIC 还要不要回 GPU 取 WQE？
```

对照表：

| 模式 | WQE 位置 | DBREC | Doorbell 所有者 | NIC 是否回读 WQE |
|---|---|---|---|---|
| `submit_db` | SQ 内存 | 更新 | GPU | 是 |
| `no_dbr` | SQ 内存 | 跳过 | GPU | 是 |
| `blueflame` | BlueFlame / UAR 区域 | 通常合并处理 | GPU 写 BF 区域 | 小型 WQE 可不回读 |
| `cpu_proxy` | SQ 或共享内存 | CPU 侧处理 | CPU | 通常需要 |

### 6.1 `submit_db`

这是通用、可靠的常规路径：

```text
GPU 或 CPU 准备 WQE
更新 QP 的 doorbell record
敲 NIC doorbell
NIC 看到 producer index 变化
NIC 从内存读取 WQE
```

DBREC 的作用是告诉 NIC：

```text
发送队列的生产者位置已经到哪里了
```

讨论里提到“敲 doorbell -> 写 DBR -> 再敲一次”时，更适合把它理解为某种实现或源码路径的精简描述。公开文档更常见的抽象是：

```text
更新 DBREC
敲 doorbell
```

不同硬件代际、驱动路径和优化模式可能有额外细节，但核心目的相同：让 NIC 发现新 WQE。

### 6.2 `no_dbr`

GPU 不更新 QP 的 DBREC，只提交发送请求并敲 doorbell。

前提通常是硬件支持新的可靠 doorbell 机制。例如 DOCA GPUNetIO 中可以看到：

```text
DOCA_GPUNETIO_VERBS_NIC_HANDLER_GPU_SM_NO_DBR
```

它减少一次内存更新和相关的可见性处理，但依赖具体网卡能力和驱动支持，不能把它当作所有平台通用的默认行为。

### 6.3 `blueflame`

BlueFlame 的核心不是“另一种 RDMA op”，而是**另一种把 WQE 交给 NIC 的方式**。

常规路径：

```text
WQE 在 host / GPU 内存
敲 doorbell
NIC 通过 PCIe 读取 WQE
```

BlueFlame：

```text
把最多 64 字节的 WQE 直接写进 NIC 的 BlueFlame / UAR 区域
NIC 不需要再通过 PCIe 回读这个小型 WQE
```

因此它特别适合：

- 小消息
- 小 WQE
- 对提交延迟敏感的路径

限制也很明确：

- 一般最多 64 字节
- 需要硬件和驱动支持
- 不同代 NIC 的行为不同
- 不一定适合大 WQE 或所有操作类型

在 DOCA 里可以看到：

```text
DOCA_GPUNETIO_VERBS_NIC_HANDLER_GPU_SM_BF
doca_gpu_dev_verbs_submit_bf
```

### 6.4 `cpu_proxy`

GPU 先准备好 WQE 或逻辑描述，但不亲自敲 doorbell。CPU 运行代理进度函数，发现工作后代为提交：

```text
GPU: 准备 WQE / descriptor
CPU: progress / poll
CPU: 补最后的提交步骤并敲 doorbell
```

公开 API 中可以看到：

```text
DOCA_GPUNETIO_VERBS_NIC_HANDLER_CPU_PROXY
doca_gpu_verbs_cpu_proxy_progress
```

它适合：

- 兼容旧硬件或旧路径
- 需要 CPU 控制面参与
- 不希望 GPU 直接管理某些 NIC 状态
- 调试和对照

代价是重新引入 GPU 到 CPU 的通信延迟和 CPU 轮询开销。

### 6.5 `AUTO`

GPUNetIO 可以自动选择当前硬件和配置下最快的合法提交路径：

```text
DOCA_GPUNETIO_VERBS_NIC_HANDLER_AUTO
```

具体选择取决于网卡型号、驱动、QP 类型、WQE 大小和可用 API。

---

## 7. BlueFlame 与 GPI 为什么看起来像，但层次不同

GPI 讨论中的简化伪代码：

```python
ch = channel[contextId]          # 每个 context 一条队列，不再按 peer 分
gfd = GFD64(
    op=WRITE,
    dst_pe=peer,
    src_handle=src_handle,
    dst_handle=dst_handle,
    size=size,
)

slot = ch.nic_queue[ch.pi.fetch_add(1) & mask]
mmio_store(slot, gfd)            # 写进网卡即提交

# 完成：网卡回写计数器，GPU 读计数器
```

### 7.1 表面上的相似点

从 GPU 侧尾部动作看，两者都可能表现为：

```text
一次 64 字节 MMIO write
```

所以看起来“都是写一下门铃或寄存器”，开销也在同一量级。

### 7.2 本质区别

| 维度 | BlueFlame | GPI |
|---|---|---|
| 写入对象 | 某个 QP 的 WQE | 逻辑 GFD |
| 写入位置 | QP 相关 UAR / BlueFlame 区域 | context 级 NIC queue |
| 目标选择 | 已经绑定到该 QP 或提交上下文 | `dst_pe` 放在 GFD 内 |
| QP 状态 | GPU 侧需要知道 QP 和 WQE 格式 | GPU 不需要维护目标 QP |
| QP 选择者 | GPU / 调用者 | NIC |
| 完成方式 | CQ / CQE | 计数器等轻量同步 |
| 抽象层级 | 提交机制优化 | 队列与传输资源模型 |

可以把 BlueFlame 看成：

```text
“我仍然按 QP 提交，只是不走 NIC 回读 WQE 的慢路径”
```

把 GPI 看成：

```text
“我根本不把一个可操作的 QP 暴露给 GPU；
 GPU 只投递逻辑描述，NIC 自己完成目标选择和 QP 管理”
```

更准确地说：

```text
BlueFlame 是 GDAKI 内部的一种提交方式。
GPI 是另一种队列粒度和资源所有权设计。
```

两者可以在某一段路径上都表现为一次 64 字节 MMIO，但不代表它们解决的是同一个问题。

---

## 8. 队列粒度为什么不同

讨论中的关键句：

```text
Proxy 按 peer 分队列
GDAKI 按 peer 分 QP
GPI 按 context 只有一条队列
```

这不是三个随意的实现选择，而是资源所有权不同导致的。

### 8.1 Proxy：按 peer 分队列

在 Proxy 中，`queues[peer]` 是：

```text
GPU -> CPU 的生产者 / 消费者邮箱
```

它按 peer 分开，主要为了：

- 保留每个 peer 的提交顺序
- 避免一个慢 peer 阻塞所有 peer
- 每个 peer 独立 backpressure
- CPU 可以按 peer 分片轮询
- 更容易定位每个 peer 的进度

CPU 拿到 GFD 后，再决定 `qp[peer]`。

### 8.2 GDAKI：按 peer 分 QP

GDAKI 把 QP 状态进一步暴露给 GPU。对于标准 RC 语义：

```text
QP -> remote peer / QPN
```

所以如果 GPU 要直接 post 到发送队列：

```text
每个目标 peer 都需要对应的 QP
```

这样就省掉了 GPU 到 CPU 的 mailbox，也省掉了 CPU 拼 WQE 的路径。代价是 GPU 侧要理解更多传输状态，并承担 QP 数量和资源管理压力。

### 8.3 GPI：按 context 分队列

GPI 把选择目标 QP 的工作交给 NIC：

```text
GPU 只知道 context
GFD 里面写 dst_pe
NIC 根据 dst_pe 选择内部 QP / transport resource
```

因此 GPU 侧只需要一条 context queue，不再维护：

- 每个 peer 的 SQ
- 每个 peer 的 QP 状态
- 每个 peer 的 WQE 格式

它把复杂度从 GPU 转移到 NIC。

### 8.4 单 context queue 会不会成为瓶颈

会有风险，但“一个 context 一条队列”并不自动等于“全 GPU 只有一条全局队列”。

可能的设计包括：

- 多个 context
- 多个 channel
- NIC 内部并行队列
- descriptor 级路由和批处理

关键仍然是：

```text
队列需要提供足够并行度、顺序控制和 backpressure，
同时不能要求 GPU 为每个业务目的地维护独立连接状态。
```

---

## 9. 三种路径的总对照

### 9.1 控制流

```text
CPU Proxy:
GPU producer
  -> per-peer mailbox
  -> CPU progress
  -> qp[peer]
  -> NIC

GDAKI:
GPU thread
  -> qp[peer] send queue
  -> DBREC / doorbell
  -> NIC

GPI:
GPU thread
  -> context nic_queue
  -> GFD(dst_pe=...)
  -> NIC 内部选择 transport
  -> network
```

### 9.2 优缺点

| 路径 | 优点 | 代价 |
|---|---|---|
| CPU Proxy | GPU 状态简单，控制灵活，兼容性好 | 多一次 GPU 到 CPU 同步，CPU 轮询和 WQE 构造开销 |
| GDAKI | 去掉 CPU 提交跳数，延迟低，GPU 自主性强 | GPU 要管理 QP 状态和 WQE；可能 QP 爆炸 |
| BlueFlame | 小 WQE 延迟低，NIC 不回读 WQE | 64 字节限制，依赖硬件和驱动 |
| GPI | GPU 不维护目标 QP，NIC 负责目标解复用和资源管理 | 实现依赖 NIC 能力，完成和错误信息可能更粗 |

### 9.3 选择的判断问题

设计或阅读实现时，依次问：

```text
1. 谁生成通信计划？
2. 谁构造 WQE？
3. WQE 放在哪里？
4. 谁更新 DBREC？
5. 谁敲 doorbell？
6. NIC 是否还需要通过 PCIe 回读 WQE？
7. GPU 是否知道目标 QP？
8. 完成通知是 CQ 还是计数器？
9. 队列粒度是 peer、QP 还是 context？
10. 上层动态目的地是否被错误地放大成底层连接数量？
```

---

## 10. 最容易混淆的几件事

### 10.1 “GPU 发起”不等于“完全没有 CPU”

即使走 GDAKI 或 BlueFlame，CPU 仍可能负责：

- 建链
- QP 创建和销毁
- 内存注册
- 错误处理
- 控制面和配置
- 某些 fallback 路径

优化的是热路径提交，不是取消所有 CPU 工作。

### 10.2 “GFD 多”不等于“QP 多”

GFD 是逻辑工作。多个 GFD 可以复用一个 QP。

真正危险的是：

```text
把 expert、token、request 等动态对象直接映射成 QP
```

### 10.3 `peer` 不等于 MoE expert

`peer` 通常是对端 rank 或传输端点。一个 peer 上可以有多个 expert，也可以在一段时间内接收很多 token。

### 10.4 BlueFlame 不等于 GPI

两者都可能表现为一次 64 字节 MMIO，但：

```text
BlueFlame 改的是 WQE 的提交方式。
GPI 改的是队列粒度、QP 所有权和目标选择位置。
```

### 10.5 `AUTO` 不是一种独立硬件机制

`AUTO` 只是选择策略，最终仍会落到 `submit_db`、`no_dbr`、`blueflame` 或 `cpu_proxy` 等具体路径。

---

## 11. 一页速记

```text
传统通信:
  GPU 填数据并推进 producer
  CPU progress 轮询
  CPU post WQE + doorbell
  NIC GPUDirect RDMA

Proxy:
  GPU 写 per-peer mailbox
  CPU 消费并提交 qp[peer]
  优势: GPU 不用管 QP / WQE
  代价: CPU 跳数和轮询

GDAKI:
  GPU 直接操作 qp[peer] 的提交路径
  submit_db / no_dbr / blueflame / cpu_proxy 是提交变体
  优势: 热路径短
  代价: QP / WQE / 状态管理更重

GPI:
  GPU 写 context NIC queue
  GFD 内携带 dst_pe
  NIC 选内部 QP
  优势: GPU 不再按 peer 管 QP
  代价: 依赖 NIC 能力，完成信息可能更轻量

QP:
  按 peer 建通常可接受
  按 expert / token / request 建通常是危险信号
  上层动态路由应尽量进入 descriptor，而不是进入连接数量
```

---

## 12. 公开行为与概念实现的边界

本文中以下内容属于公开文档较容易核对的机制：

- GPUNetIO 的 `AUTO`、`CPU_PROXY`、`GPU_SM_DB`、`GPU_SM_BF`、`GPU_SM_NO_DBR` 等 handler 概念
- `doca_gpu_dev_verbs_submit_bf`
- `doca_gpu_verbs_cpu_proxy_progress`
- BlueFlame 使用 64 字节 UAR / BlueFlame 区域的提交思路
- DBREC、doorbell、WQE、CQ 的基本职责

以下内容更偏源码或讨论中的概念化解释：

- `queues[peer]` 和 `pis[peer]` 的具体命名与布局
- GFD64 的字段定义
- “GPI 一条 context queue、由 NIC 选 QP”的具体内部实现
- “一次 MMIO 写后由 NIC 回写计数器完成同步”的生产级语义

阅读真实实现时，需要再结合具体版本的源码、网卡型号、驱动和 DOCA/NCCL 版本确认。
