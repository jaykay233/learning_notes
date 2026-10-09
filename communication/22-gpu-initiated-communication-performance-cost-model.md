# GPU-initiated communication 性能解剖：issue、proxy、message rate 与资源代价

> 承接 [21-gpu-initiated-rdma-submission-dc-qp-sharing.md](./21-gpu-initiated-rdma-submission-dc-qp-sharing.md)。
> 21 已经把 SM payload、WQE、dbrec、UAR doorbell、WQE fetch、CQE、
> RC / DC / DCI / DCT 和 QP sharing 的机制讲清；本文继续回答第 4 章的问题：
> 这些机制分别花多少钱，为什么生产库会慢，CPU proxy 什么时候能追上 GPU，
> message rate 和 kernel / NIC 资源又付出了什么代价。
>
> 主要参考：
> - *GPU-Initiated Communication: Dissecting Down to the Bone*，arXiv:2610.01380v1
> - 本地论文：`/Users/saboxu/Downloads/AF分离/2610.01380v1.pdf`

```text
本次讲解位置
章节：communication / GPU-initiated communication
小节：4.1 One-operation cost / 4.2 Proxy design space / 4.3 Message rate and resource costs
知识点：issue / put+completion / RTT、ordering 与 DBR、CPU proxy、R/T/B/handoff、QP sharing、kernel occupancy、NIC active connections
上次：payload、doorbell、WQE fetch、CQE、DCI / DCT / QP sharing 的硬件路径（21）
下次：把平台差异和 DeepEP / NVSHMEM / NCCL 的通信选型映射到真实 MoE workload
原语：release / fence、WQE、DBR、UAR doorbell、CQE、QP、context、ring、batch
```

## 为什么现在讲这个

21 回答了“谁提交、数据怎样到 NIC、完成怎样回来”，但只知道机制还不够。机制相同的两条路径可能有数倍延迟差：

```text
硬件操作相同
+ 不同的 queue management
+ 不同的 ordering scope
+ 不同的 QP lookup
+ 不同的 completion scope
= 完全不同的 issue / RTT / message rate
```

这一篇要防止四种常见误判：

1. 把 `issue` 当成完整 RTT。
2. 把 CPU proxy 的 GPU enqueue 时间当成完整 CPU 提交成本。
3. 把共享 QP 或 batching 的吞吐收益误认为一定降低单请求延迟。
4. 只统计 QP 数量，却忽略 NIC 是 send-only、receive-only，还是同时收发。

---

## 1. 先建立统一成本模型

### 1.1 一次操作的三个计时口径

| 指标 | 含义 | 包含 | 不包含 |
|---|---|---|---|
| Issue | GPU/CPU 线程把一笔工作提交出去 | WQE、queue management、ordering、DBR、doorbell | NIC 执行和完成等待 |
| Put+completion | 从发起直到本地 completion 路径返回 | issue、NIC/网络、远端处理、CQ polling | 远端应用真正消费数据 |
| RTT | 请求发出到观察到远端 reply | put+completion 加 reply 通知 | 大规模持续负载能力 |

可以近似写成：

```text
T_issue =
    WQE construction
  + queue management
  + QP lookup
  + ordering / fence
  + DBR update
  + doorbell

T_put_completion =
    T_issue
  + NIC fetch / network / remote DMA
  + completion routine

T_rtt =
    T_put_completion
  + remote-side reply path
```

### 1.2 吞吐模型

```text
goodput ~= message_rate * payload_size

total_message_rate ~=
    parallel_submitters
  * per_submitter_rate

per_submitter_rate ~=
    batch_size / (fixed_submit_cost + batch_size * per_wqe_cost)
```

这组关系解释了论文中最重要的几个现象：

- 小消息时，`message_rate` 是主瓶颈。
- 大 payload 时，固定提交成本被摊薄，逐渐转向链路带宽瓶颈。
- 多 QP、多 worker、多 SM 提供 parallel submitters。
- batching 和 cooperative publication 降低 per-submitter 的固定成本。

### 1.3 除了时间，还要统计资源

| 资源面 | 代价 |
|---|---|
| GPU | register、occupancy、spill、SM issue cycles |
| CPU | 专用 core、轮询、NUMA、cache/TLB warm 状态 |
| NIC | QP context、completion、memory translation、ICM/cache |
| 软件 | QP lookup、queue management、completion scope、锁 |

---

## 2. order、ordering 和 DBR 到底在保证什么

### 2.1 ordering 是可见顺序，不是代码顺序

正常提交会写：

```cuda
wqe->opcode = WRITE;
wqe->remote_addr = dst;
wqe->length = size;
wqe->source_addr = src;

dbr->producer = pi + 1;

*nic_doorbell = pi + 1;
```

程序顺序是：

```text
WQE -> DBR -> doorbell
```

但 GPU 的 relaxed memory model 不保证 NIC 也按这个顺序观察。NIC 可能先看到 doorbell，再读到旧 WQE 或部分 WQE。

安全契约是：

```text
WQE 可见
DBR 可见
payload 可见（non-inline）
        before
doorbell 可见
```

### 2.2 DBR 和 UAR doorbell 不是一回事

| 名称 | 位置 | 作用 |
|---|---|---|
| Doorbell Record，DBR | host / GPU memory 中的普通状态 | 记录 producer 进度或队列状态 |
| UAR doorbell | NIC PCIe BAR 映射出的 MMIO 区域 | 通知 NIC：“有新工作，去处理” |

典型路径：

```text
GPU/CPU 写 WQE
-> 更新 DBR
-> release/fence
-> 写 UAR doorbell
-> NIC fetch WQE
```

只写 DBR 不会可靠地叫醒 NIC；只写 UAR doorbell 而 WQE/DBR 尚未可见，则可能让 NIC 读到陈旧状态。

### 2.3 表 4 的 ordering 成本

| Ordering | P-IB issue | P-IB put+completion | 解释 |
|---|---:|---:|---|
| None，unsafe | 0.19 us | 3.46 us | 最快，但不保证安全 |
| GPU-scope fence | 0.70 us | 4.03 us | 常用安全基线 |
| GPU-scope release store | 0.70 us | 4.00 us | DeepEP 类 release 发布 |
| `__threadfence()` | 1.22 us | 4.48 us | NVSHMEM 使用的更强设备 fence |
| system-scope release store | 2.37 us | 5.66 us | scope 扩到 system |
| system-scope fence | 2.62 us | 5.92 us | issue 约为 GPU-scope fence 的 3.7 倍 |

论文实际没有观察到去掉 fence 后的 corruption，但这只证明当前实验配置没有复现问题，不能证明删除 ordering 普遍安全。

### 2.4 远端 completion 也需要 ordering

远端路径同样不是“看到 CQE 就一定能看到 payload”：

```text
远端 NIC DMA 写 payload
远端 NIC 写 CQE / flag
发起端 GPU 观察 CQE
```

GPU 是 relaxed memory model，等待 CQE 的代码仍需要 acquire/fence 类语义，才能建立 payload 与 flag 之间的可见性。

---

## 3. 4.1：一次操作的成本怎样拆开

### 3.1 最小 GPU 路径

mini-gda 对 8 B inline put 的结果：

```text
issue:           0.70 us
put+completion:  4.03 us
RTT:             6.85 us
```

doorbell 写完之后，有效 CQE 约 `3.30 us` 出现。于是：

```text
0.70 us SM 软件提交
3.30 us doorbell -> NIC/网络/远端 -> 本地观察到 CQE
其余时间 completion routine、queue update、RTT reply
```

### 3.2 生产库为什么慢

| 路径 | Issue | Put+completion | RTT |
|---|---:|---:|---:|
| mini-gda inline | 0.70 | 4.03 | 6.85 |
| GDAKI inline | 1.60 | 6.02 | 10.50 |
| NVSHMEM internal | 4.26 | 9.18 | - |
| NVSHMEM public | 5.31 | 11.10 | 21.50 |
| NVSHMEM IBRC | 0.99 | 6.40 | 11.18 |
| mini-proxy baseline | 2.27 | 7.07 | 14.59 |
| mini-proxy tuned | 0.13 | 4.10 | 5.89 |

额外成本主要来自：

```text
QP lookup
slot reservation
ready-head 更新
doorbell lock
更强的 ordering
更宽的 completion scope
```

`mini-proxy tuned` 的 0.13 us 只表示 GPU enqueue，不包含 CPU worker 构造 WQE、调用 `ibv_post_send` 和写 doorbell 的时间。

### 3.3 CPU proxy 什么时候能追上甚至超过

需要同时满足：

```text
8 B 小消息
最多一笔 outstanding
GPU 只写 16 B descriptor
一个 producer 对应一个 ring
CPU worker 固定在 NIC NUMA node
专用 core，持续 busy polling
cache / TLB / ring page 已 warm
probe 与 bulk traffic 隔离
```

此时 tuned proxy 在 P-IB 的 RTT 为 `5.89 us`，低于 mini-gda 的 `6.85 us`；在 P-GB200 上也更低。但 P-IB 的 `put+completion` 仍略高于 mini-gda，因此不能理解成所有指标全面超过。

proxy 仍然没有消除 GPU enqueue：

```text
GPU 读参数
写 descriptor
发布 producer index
做 release / ordering
```

所以 SM 时钟下降时，这部分仍然变慢。

### 3.4 Inline 的收益边界

8 B inline 把 payload 直接放进 WQE，NIC 不必读取 source buffer：

```text
节省约 0.61 us completion 时间
issue 基本不变
```

但 payload 增大后：

```text
每增加约 16 B，issue 增加约 0.1 us
到约 92 B 时，节省的 completion 成本基本消失
更大 WQE 还会降低 doorbell batching 的消息率
```

这就是 NVSHMEM、GDAKI 和 DeepEP 只 inline 8 B 的原因。

---

## 4. QP lookup、queue management、completion scope

这四个词可以放在同一条路径里：

```text
QP lookup
-> queue management
-> ordering
-> NIC transfer
-> completion scope
```

| 概念 | 回答的问题 | 典型内容 |
|---|---|---|
| QP lookup | 该用哪个 QP？ | peer/context/expert 到 QP、SQ、CQ 的映射 |
| Queue management | 怎样占用和推进队列？ | slot reservation、producer index、锁、DBR、回收 |
| Ordering | 什么时候允许 NIC 看到？ | WQE/DBR/payload 与 doorbell 的可见顺序 |
| Completion scope | 完成后等哪些队列？ | 单 WQE、单 QP、peer、context、全部 QP |

### 4.1 QP lookup

```cuda
peer = dst_rank;
qp = qp_table[peer];
sq = &qp->sq;
wqe = &sq->wqe[sq->pi & sq->mask];
```

GPU-initiated 路径中的 lookup 可能发生 global memory 读取、边界检查、指针追踪和 cache miss。MoE 按 peer/expert 选择 QP 时，warp 内不同 token 还可能产生 divergence。

### 4.2 Queue management

```cuda
slot = reserve_slot(qp);
build_wqe(slot, desc);
update_ready_head(qp);
publish_in_order(qp);
update_dbr(qp);
doorbell(qp);
```

它不只是“写 WQE”，还包括：

- producer/consumer index
- ring wrap 和 mask
- 防止覆盖尚未完成的 slot
- QP lock 或原子预留
- ready-head 更新
- DBR 和 doorbell batching
- CQE 回来后回收 slot

### 4.3 Completion scope

```text
all-QP quiet:
    poll 所有配置过的 QP，即使有些从未使用

used-QP quiet:
    只 poll 当前操作相关的 QP

single-QP quiet:
    只等待该 QP 上之前的 WQE
```

论文图 4 中，NVSHMEM all-QP quiet 在 1 到 16 QP 时 put+completion 基本翻倍；DeepEP 的 used-QP quiet 保持平稳。completion scope 是 QP 数量最直接的延迟成本之一。

---

## 5. 4.2：CPU proxy 的设计空间

### 5.1 四个设计变量

| 变量 | 含义 | 主要影响 |
|---|---|---|
| `R` | descriptor ring 数量 | probe 与 bulk 是否互相阻塞 |
| `T` | CPU worker 数量，每个 worker 通常有自己的 QP | parallel posting capacity |
| `B` | 每次 `ibv_post_send` 链多少 WR | 固定成本摊销 |
| handoff | GPU reserve、publish descriptor、观察 worker 进度 | enqueue latency 和 ordering |

### 5.2 Idle latency 与 loaded latency 是两回事

Idle 时：

```text
NVSHMEM IBRC RTT        11.1 us
NVSHMEM public GPU      21.5 us
mini-gda / GDAKI        比 IBRC 更快
```

IBRC 在某组公开测试中更快，但不能据此说 proxy 有内在优势。fabric-lib、UCCL-EP、MSCCL++ 的 RTT 也主要由 notification protocol 决定。

Loaded 时，共享 FIFO 的代价远大于单次软件成本：

```text
IBRC single FIFO:
    一个 background CTA 就可把 RTT 拖慢约 670 倍

mini-proxy one ring:
    共享 ring 时被 bulk traffic 挡住

GIN Proxy shared context:
    worker 增多仍受共享 context 限制

GDAKI shared context:
    64 CTA 时约 362 us，只承载 2.6 M msg/s

GDAKI private context:
    约 12 到 16 us，承载 73 到 79 M msg/s
```

核心原因是 queue-level head-of-line blocking：

```text
probe WQE 排在 bulk WQE 后面
-> 必须等 bulk 先完成
-> latency-sensitive 请求失去优先级
```

### 5.3 Queue isolation 比多一个 worker 更重要

保留独立 ring/context 可以把 loaded RTT 降低一到两个数量级。单独“多保留一个 CPU worker”收益有限，而且会从 bulk traffic 拿走 capacity。

比较配置时必须同时看：

```text
probe latency
background 实际承载 M msg/s
```

一个配置 RTT 更低，可能只是因为它少发了流量。

### 5.4 Batching 和 workers 提高 capacity

```text
T=1, R=32, B 从 1 增到 16:
    message rate 提升约 5.5 倍
```

原因是一次 `ibv_post_send` 摊销多个 WR 的固定成本。

多个 worker 提供 parallel posting，但会遇到：

- cache 和锁竞争
- shared context
- refcount / service state
- NIC QP 和 posting capacity

P-GB200 上 mini-proxy 可扩到 8 workers、B=16，达到约 `140 M msg/s`，接近同平台 IBGDA 的 `156 M msg/s`。P-IB 上最强 GPU submission 仍可到约 `250 M msg/s`。

### 5.5 大 payload 隐藏 posting rate

达到参考带宽 90% 所需 payload：

| 路径 | payload |
|---|---:|
| NVSHMEM IBGDA / mini-gda | 512 B |
| GDAKI / mini-proxy / UCCL-EP | 2 KiB |
| MSCCL++ / GIN Proxy / fabric-lib | 7 KiB |

DeepSeek-V3 的 7,168 B FP8 hidden-state vector 在此尺寸下，大多数路径都能超过参考带宽 90%。冷 IBRC 要到 32 KiB，warm IBRC 到 7 KiB。

large payload 能让 goodput 收敛，但不代表 completion latency 收敛。

---

## 6. 4.3.1：message rate 从哪里来

### 6.1 单个提交者无法打满 NIC

```text
1 thread, 1 QP, 1 WQE/doorbell: 1.79 M msg/s
```

只增加 QP 或只增加 thread，若没有形成独立提交路径，也不会自动线性增长。

### 6.2 Cooperative QP sharing

高效做法是一个 warp 协同发布：

```text
warp 一次性预留 32 个 slot
32 个 lane 并行构造 WQE
一个 lane 按顺序发布整组
更新一次 DBR，敲一次 doorbell
```

结果：

```text
1 warp:      21.6 M msg/s
2+ warps:    30.7 M msg/s
```

收益来自把 reservation、lock、ordered publication 和 doorbell 从 per-WQE 变成 per-warp/per-batch。

### 6.3 Batching 与多 QP 是乘数

```text
1 thread 从 1 增到 16 WQE/doorbell:
    1.79 -> 4.70 M msg/s

16 QP + 16 SM + B16:
    74.4 M msg/s

64 QP + 64 SM + B32:
    260 M msg/s
```

共享 QP 提高单 QP ceiling，多 QP 提供更多独立 SQ。两者解决不同瓶颈。

### 6.4 GPU 不一定是 doorbell 提交者

GPU 构造 WQE，host thread 只负责 ring doorbell，也能达到约 `257-258 M msg/s`。代价是 `put+completion` 增加约 2.6 到 5.8 us。

这再次说明：

```text
峰值 message rate
不等于
最低 completion latency
```

### 6.5 Publication order

GDAKI 按 reservation order 发布。Lane 进度漂移会让快的 lane 等待慢的 lane。论文在 32 threads/context 下，每次 put 后加 `__syncwarp`，GDAKI rate 约提升 3 倍。

共享 QP 的瓶颈往往不是“写 WQE”本身，而是：

```text
slot reservation
ready-head
ordered publication
doorbell lock
```

---

## 7. 4.3.2：通信代码给 kernel 带来什么成本

### 7.1 四种 kernel variant

| Variant | 行为 |
|---|---|
| no comm | 没有通信代码 |
| code only | 代码编译进去，但分支不执行 |
| send then compute | 提交后继续计算，等待后台完成 |
| send wait then compute | 提交后等待 completion，再继续计算 |

通信代码会影响：

```text
register allocation
block residency
spill
occupancy
instruction cache
```

### 7.2 dormant code 也会变慢

compute caller：

```text
mini-gda / mini-proxy / separate NVSHMEM: 基本不损失
inlined NVSHMEM / GDAKI: 损失 11% 到 15%
```

streaming caller：

```text
NVSHMEM / GDAKI: 还没有发送就损失约 37%
8 blocks/SM 下，mini-gda dormant path 也可损失 27%
```

根因是更高 register demand 降低 block residency，latency hiding 变差。

### 7.3 等待 completion 比间歇提交更贵

```text
发送不等待:
    kernel 平均只产生约 4 M msg/s，损失较小

每次 8 B 都等待:
    mini-gda / GDAKI 额外损失 2% 到 5%
    NVSHMEM / mini-proxy 额外损失 19% 到 27%
```

QP 数量放大 NVSHMEM 的等待损失：

```text
16 QP:  7%
64 QP:  20%
528 QP: 64%
```

DeepEP V1 low-latency dispatch 中，issue 约 `25 us` 就结束，整个操作约 `0.49 ms`。剩余时间主要在 receive-side waiting 和 copying。因此只优化 issue latency 只解决一小部分问题。

---

## 8. 4.3.3：NIC 为连接数付什么成本

RC QP 的 per-connection state 位于 NIC 的 ICM 中，host memory 做 backing，NIC cache 缓存热状态，同时还要保存 completion 和 memory translation 状态。

### 8.1 流量角色决定连接成本

| 流量类型 | 结果 |
|---|---|
| send-only | 到 32,768 active QP 仍接近 242 M msg/s |
| simultaneous send/receive | 从 152 降到 74 M msg/s，损失约 51% |
| receive-only | 介于两者之间，连接数约十倍后才明显下降 |
| 长连接复用 | 1,024 到 4,096 QP 可维持 139 到 143 M msg/s |

只统计 QP 数量不足以预算 NIC capacity。同样 3,000 connections：

```text
send-only 可能承受
simultaneous send/receive 已经明显退化
```

### 8.2 Connection reuse 与 payload

每个 connection visit 从 32 writes 提高到 8,192 writes，可显著 flatten simultaneous-traffic curve。原因是 NIC 不必频繁切换连接上下文。

4,096 QP、同时收发：

```text
128 B writes:
    只有 send-only goodput 的 55%

>=256 B:
    保留 98%
```

更大 payload 摊薄 per-connection 的固定成本。

### 8.3 All-to-all 与 DC

```text
3,000 active connections:
    约损失 59%

128-PE dense sweep:
    onset 约 1,350 active connections/NIC
```

DC 不能彻底消除：

```text
256 writes/destination:
    1,984 DCI-peer pairs 时保留约 95%
    2,976 时只剩约 23%

相同 burst 下 RC:
    2,976 时仍保留约 56%
```

DC 单笔操作只比 RC 贵约 `0.45 us`，但每个 WQE 都切换 destination 时，NVSHMEM DC 路径 message rate 可能下降约 60 倍。短 per-destination burst 能恢复大部分性能。

---

## 9. 共享 QP 为什么未必降低 latency

共享 QP 可以提高 message rate，因为 cooperative publication 摊销了固定成本；但 latency 可能上升，因为：

```text
probe 和 bulk WQE 共享 SQ
-> head-of-line blocking

cooperative batch
-> 单个 WQE 等整组发布

公共 lock / ready-head / ordered doorbell
-> p50 和 p99 抖动

共享 completion state
-> polling 和 quiet 范围更大
```

因此必须区分：

```text
message rate:
    队列整体每秒能做多少工作

single-op latency:
    一笔请求自己等多久

p99 latency:
    最慢请求是否会因为别人的 bulk WQE 被卡住
```

生产设计通常把 bulk 和 latency-sensitive probe/control 分开：

```text
data plane:
    大 QP、共享 QP、batching、追 goodput

control plane:
    独立 QP/context/ring、低 completion scope、追 p99
```

---

## 10. 最容易混淆的概念

| 容易混淆 | 正确区分 |
|---|---|
| issue vs RTT | issue 只管提交；RTT 还包含网络、远端和 reply |
| GPU-initiated vs GPU-submitted | 前者是语义发起，后者是真正完成提交 |
| proxy enqueue vs CPU submit | GPU 写 descriptor 不等于 CPU 构造完 WQE 并敲 doorbell |
| DBR vs UAR doorbell | DBR 是内存进度记录；UAR 是通知 NIC 的 MMIO |
| QP lookup vs queue management | 一个找 QP；一个管理该 QP 的 WQE ring 和状态 |
| completion scope vs message rate | 一个决定等待范围；一个决定提交吞吐 |
| 共享 QP vs 多 QP | 前者提高单 QP 利用率和 capacity；后者提供独立 SQ 并行度 |
| message rate vs goodput | 前者是 msg/s；后者还乘 payload，受带宽影响 |
| QP 数量 vs NIC 成本 | 还要看 send/receive role、reuse、payload、active set |

---

## 11. 自检问题

### Q1：mini-proxy 的 issue 是 0.13 us，为什么不能直接说它比 mini-gda 快 5 倍？

因为这 0.13 us 只测 GPU enqueue。CPU worker 还要 poll ring、构造 WQE、调用 `ibv_post_send`、更新 DBR 和写 doorbell，而且可能需要专用 CPU core。完整比较要看 put+completion 和 RTT。

### Q2：为什么 WQE 已经写好，还要 release/fence 后才写 doorbell？

代码顺序不代表 NIC 观察顺序。release/fence 保证 WQE、DBR、payload 对 NIC 或目标 scope 可见之后，doorbell 才可见。否则 NIC 可能读到旧 WQE 或旧 payload。

### Q3：单个 QP 为什么难以继续提高 message rate？

一个 QP 只有一条有序 SQ、一个 producer index、一个 DBR、一个 doorbell publication 序列。共享它需要锁和按序发布；协同发布能提高单 QP ceiling，但不能突破单 QP 的硬件队列上限。

### Q4：共享 QP 让吞吐更高，为什么 latency 仍然可能变差？

batching、ordered publication 和共享 SQ 会让单个请求等待其他请求；probe 与 bulk 共用 QP 时还会发生 head-of-line blocking。吞吐和单请求延迟不是同一个指标。

### Q5：为什么 NIC 只发送可以扩展到 32,768 QP，同时收发却迅速下降？

simultaneous send/receive 同时要维护更多 QP state、completion、translation 和 packet-processing 状态，并产生双向 backpressure。send-only 没有同等的 receive-side 状态和资源竞争。

---

## 12. 一页速记

```text
计时:
  issue             = WQE + queue + ordering + doorbell
  put+completion    = issue + network + completion
  RTT               = put+completion + reply

ordering:
  WQE / DBR / payload 可见
    before
  UAR doorbell 可见

  remote payload 可见
    before
  remote flag / CQE 可见

proxy:
  R = rings
  T = workers/QPs
  B = wr chaining
  handoff = GPU descriptor publication + CPU progress visibility

message rate:
  cooperative QP sharing 摊销 lock / reservation / doorbell
  batching 摊销 per-WQE 固定成本
  多 QP / 多 SM / 多 worker 提供并行提交者

latency:
  shared FIFO -> head-of-line blocking
  probe 与 bulk 必须隔离
  大 batch 有等待成本，负载高时 backlog 会掩盖它

resource cost:
  kernel: register / occupancy / spill / waiting
  CPU: dedicated core / polling / NUMA / warm state
  NIC: QP context / translation / CQ / active connections

平台结论:
  P-IB: GPU submission 峰值更高，proxy 仍可在特定 low-load 小消息 RTT 上接近或超过
  P-GB200: proxy 可接近 IBGDA，但 CPU、NUMA 和 completion scope 更关键
```

---

## 参考论文段落的阅读顺序

```text
1. 4.1 的三个计时口径：issue / put+completion / RTT
2. Table 3 与 Table 4：生产库开销和 ordering scope
3. Figure 3：inline payload 的延迟与 message-rate tradeoff
4. Figure 4：completion scope 随 QP 数量增长
5. Figure 5：SM 时钟对 issue 的影响
6. 4.2：R/T/B/handoff 与 Figure 6 / 7 / 8
7. 4.3.1：cooperative QP sharing 与 Table 5
8. 4.3.2：Figure 9 的 kernel occupancy / waiting cost
9. 4.3.3：Figure 10 的 active connections / role / reuse
```
