# RDMA 常见面试题与答题要点

> 用途：面试前自测。每题先只看问题，口述 30 秒，再对照要点补漏。
> 题目来自公开面经与常见追问（百度智能云 / 系统部、阿里云、腾讯云的 RDMA
> 相关轮次，以及 [RDMA-tutorial](https://github.com/StarryVae/RDMA-tutorial)
> 里的面经汇总），答案要点按“面试官想听的那一句”整理，不追求教科书式完整。
>
> 关联笔记：[21 · GPU-initiated RDMA 提交路径](./21-gpu-initiated-rdma-submission-dc-qp-sharing.md)、
> [22 · GPU-initiated 通信成本模型](./22-gpu-initiated-communication-performance-cost-model.md)、
> [03 · 低延迟 collective](./03-low-latency-collectives-and-synchronization.md)。

```text
本次讲解位置
章节：communication / RDMA 面试
小节：面试题库与答题骨架
知识点：kernel bypass、MR、QP 状态机、RC/UD/DC、PSN 重传、PFC/ECN/DCQCN、verbs、排障、部署
上次：MoK dispatch·combine 方向（31）
下次：按岗位（存储 / AI Infra / 网卡与驱动 / 分布式）挑专题深挖
```

## 先记住这五条

被追问到没准备的地方时，这五句能兜住大部分方向：

1. **快在三点**：绕过内核协议栈、零拷贝、把分段/重组/可靠传输卸到 RNIC，收完成靠轮询而不是中断。
2. **RC 的可靠**：PSN + ACK/NAK + 超时重传；一旦判定不可恢复，QP 进 ERROR，挂起的 WR 全部 flush。
3. **RoCE 的稳**：PFC 保无损，ECN + DCQCN 管速率；只开 PFC 会死锁和 HOL，只开 ECN 会丢包重传。
4. **MR 为什么贵**：注册要 pin 页并建立地址转换，所以生产代码池化 MR、复用 PD/QP，而不是每条消息注册一次。
5. **单边与双边**：Read/Write/Atomic 不消耗对端 recv WR；Send 消耗。`WITH_IMM` 是给单边操作补一个 4 字节通知的手段。

---

## 一、基础概念

### Q1. 什么是 RDMA？它到底解决了什么问题？

要点：

- Remote Direct Memory Access：一方直接读写另一方注册过的虚拟内存，对端 CPU 不参与。
- 解决的三个开销：内核协议栈的上下文切换与逐层处理、用户态与内核态之间的多次拷贝、CPU 参与收发与校验。
- 代价：内存必须提前注册、语义比 socket 窄（固定 MR、无流式接口）、网络要求无损或低丢包。

### Q2. RDMA 有哪几种实现？IB、RoCE v1/v2、iWARP 怎么区分？

| 实现 | 链路/封装 | 关键点 |
| --- | --- | --- |
| InfiniBand | 私有二层，自带 credit 流控 | 端到端无损，交换机与网卡同生态 |
| RoCE v1 | 以太网二层 EtherType | 不能路由，跨不了三层 |
| RoCE v2 | UDP/IP 封装（默认 4791） | 可路由，生产主流；依赖 PFC/ECN 造无损 |
| iWARP | 直接跑在 TCP 上 | 天生可靠、可跨广域，但协议处理重、生态弱 |

答题落点：RoCE v2 把「无损」这件事从链路层推给了运维配置，这是它所有部署问题的根源。

### Q3. 为什么 RDMA 的延迟能到 1~2 us 量级？

要点：

- 提交路径：用户态写 WQE 到 SQ，写 doorbell 寄存器，没有系统调用。
- 数据路径：NIC 直接 DMA 读写已注册内存，不经过内核缓冲区。
- 完成路径：轮询 CQ，避免中断与唤醒。
- 剩下的固定成本：doorbell 的 MMIO 写、PCIe 往返、网卡流水线、对端地址转换与 cache miss。
- 数字要带条件：小消息、同机架、关闭 CQ moderation、独占核，才有这个量级。

### Q4. 为什么内存必须注册？pin 住的是什么？

要点：

- 网卡拿不到页表，需要一张虚拟地址 → 物理地址/IOVA 的转换表，注册时建立。
- 页被换出或迁移会让网卡 DMA 到错误位置，所以注册期间页面被 pin。
- 注册还生成 lkey/rkey（保护域内的访问凭证），并做访问权限检查。
- 反过来说：`ibv_reg_mr` 是昂贵操作，按连接/按消息注册是典型性能错误。

### Q5. lkey、rkey、PD、AH 分别是什么？

- **PD（Protection Domain）**：资源隔离容器，QP、MR 必须同 PD 才能互相访问。
- **lkey**：本地访问用，post_send 时告诉网卡这段内存的凭证。
- **rkey**：远端访问用，必须通过带外或握手交给对端，配合远端虚拟地址使用。
- **AH（Address Handle）**：UD 发送时的目的地址描述（GID、QPN、P_Key 等）。

### Q6. QP 有哪几种类型？RC、UC、UD、XRC、DC 的取舍？

| 类型 | 连接性 | 可靠性 | 典型用途 |
| --- | --- | --- | --- |
| RC | 一对一 | 可靠、保序 | 主流，性能最好，连接数 O(N²) |
| UC | 一对一 | 不可靠 | 少用，适合可容忍丢失的场景 |
| UD | 一对多 | 不可靠、乱序 | 多播、建连前的发现、小控制消息 |
| XRC | 一对多共享接收队列 | 可靠 | 大规模 RC 连接爆炸的缓解 |
| DC (DCT/DCI) | 动态连接 | 可靠 | 连接按需建立，省 NIC 连接 cache |

追问点：RC 的问题不是带宽而是**连接状态**——每连接一份 QP context，网卡 cache 装不下就频繁 miss。

### Q7. 讲一下 QP 的状态机。

```text
RESET → INIT → RTR → RTS
                  ↘ ERROR（任何不可恢复错误）
```

- **INIT**：设置 port、P_Key、访问权限，QP 还不能收发。
- **RTR（Ready to Receive）**：填对端 QPN、PSN、path MTU、GID/AV，此时能收不能发。
- **RTS（Ready to Send）**：设置重传次数、超时、responder resources，开始能发。
- **ERROR**：硬件错误、重传耗尽、访问违规；挂起 WR 以 flush error 完成。

### Q8. Send/Recv 和 Read/Write 的本质区别？

- **双边**：Send 需要接收端提前 post recv WR；接收端 CPU 必须参与准备缓冲。
- **单边**：Write/Read 只要求对端 MR 存在且给了 rkey，对端 CPU 完全不参与。
- 因此 Write 适合「推数据 + 事件通知」，Read 适合「拉数据」，Send/Recv 适合控制面与流式语义。

### Q9. 什么是 immediate data？为什么需要它？

要点：

- `IBV_WR_SEND_WITH_IMM` / `WRITE_WITH_IMM` 在完成消息里带 4 字节立即数。
- 单边写本身不给对端任何完成事件，对端只能轮询内存或等超时；`WITH_IMM` 用一次 CQE 把「写完了」告诉对端。
- 常见用法：用 imm 编码消息类型、长度或 slot id。

### Q10. inline data 与 SGE 分别省了什么？

- **inline**：小消息（通常 ≤ 几十到几百字节，受 `max_inline_data` 限制）直接放进 WQE，省一次 payload DMA 和一次读内存。
- **SGE（scatter-gather）**：一个 WR 可以挂多段不连续内存，网卡按描述符拼接，避免为了连续而 memcpy。

---

## 二、传输层与可靠性

### Q11. PSN 是什么？重传怎么发生？

要点：

- Packet Sequence Number 按 QP 单调递增，接收端据此检测丢包、乱序与重复。
- 发送端收到 ACK 推进窗口；超时（`local_ack_timeout`）或收到 NAK 触发重传。
- 基本行为是 go-back-N：丢了第 N 个包，后面已发的也要重发。
- 重试次数由 `retry_cnt` / `rnr_retry` 控制，耗尽后 QP 进 ERROR。

### Q12. responder resources 和 rnr_retry 是什么？

- **responder resources**：接收端允许多少个乱序/未完成的请求同时在途，影响对端能打多深。
- **rnr_retry**：接收端没有可用 recv WR 时（Receiver Not Ready），发送端重试的次数；设 0 表示不重试直接报错。
- 经典故障：接收端没预投递 recv，发送端一直 RNR 重试，最后 `IBV_WC_RNR_RETRY_EXC_ERR`。

### Q13. 什么情况下 QP 会被 flush？应用看到什么？

要点：

- 访问违规（rkey 错、权限不足、越界）、重传耗尽、路径 MTU 不匹配、对端 QP 消失等。
- QP 进 ERROR 后，所有未完成的 WR 以 `IBV_WC_WR_FLUSH_ERR` 完成，应用必须把它们当作失败而不是完成。
- 恢复方式通常不是 reset 单个 QP，而是重建连接（尤其 RC）。

### Q14. 内存序：什么时候需要 fence？

要点：

- 同一 QP 内，RDMA 操作本身保序；跨 QP、或「先 Write 数据再 Send 标志」这种依赖需要 `IBV_SEND_FENCE`。
- fence 保证之前的 WR 在之后 WR 开始前完成读侧可见性，但不能替代应用层的 acquire/release。
- 对端看到 CQE 只说明本地完成，不自动等于对端内存对应用可见（尤其配合 CPU store 时）。

### Q15. Atomic 操作有哪些限制？

要点：

- `CMP_AND_SWP`、`FETCH_AND_ADD`，要求目标 MR 以 `IBV_ACCESS_REMOTE_ATOMIC` 注册。
- 只对 8 字节对齐的 64 位字有效（部分实现支持 32 位扩展）。
- 延迟比 Write 高，通常用来做计数器、锁、门闩，不做批量数据路径。

### Q16. MTU 与 path MTU 为什么常选 4K？

要点：

- IB/RoCE 的 MTU 是 256 / 512 / 1024 / 2048 / 4096。
- 越大，每字节的头部与处理开销越低；但单包时延、交换机 buffer 占用、PFC 反压粒度也变大。
- 双方 path MTU 必须一致，否则 RTR 配置就会失败或运行时报错。

### Q17. UD 为什么不可靠却仍然重要？

要点：

- 无连接、一个 QP 可对多个目的发，省连接状态，适合发现、控制、多播。
- 每个包带 GRH（Global Routing Header），RoCEv2 下承载 GID 等路由信息。
- 不保序、不重传，消息上限受 MTU 限制（无分段重组）。

### Q18. RC 的 ACK 是每个包都回吗？

要点：

- 不是。接收端按策略聚合 ACK（例如每 N 个包或超时），否则 ACK 本身会占满小消息路径。
- 这也是为什么「可靠」不等于「同步确认」：发送端靠窗口与超时兜底。

### Q19. 什么是 Memory Window 和 Fast Registration？

要点：

- FRWR：用 `ibv_reg_mr` 的轻量替代，把 MR 注册拆成「注册钥匙 + 绑定地址」，适合频繁换缓冲（如存储 IO）。
- MW：在不重新注册大块内存的前提下，临时授予远端一段子区间访问权。
- 动机都是让「地址变化」变便宜，避免每次 IO 都 pin 一遍。

---

## 三、RoCE、无损网络与拥塞控制

### Q20. RoCEv2 为什么需要 PFC？

要点：

- 以太网默认「拥塞就丢」，而 RC 的丢包代价是 go-back-N 重传，微秒级网络里一次丢包可能放大成毫秒级抖动。
- PFC 按 802.1Qbb 的优先级发 pause 帧，让上游在 buffer 耗尽前停发，从而做到单优先级无损。
- 代价：反压会沿链路传播，配置不当会死锁、形成 HOL、甚至 PFC storm。

### Q21. PFC 死锁是怎么发生的？怎么防？

要点：

- 环路 + 逐跳反压：A 等 B 的 buffer，B 等 A 的 buffer，双方都停。
- 防护：STP/路由保证无环、PFC watchdog（检测长时间 pause 就丢弃并告警）、限制反压域、按优先级隔离。
- 现场手段：看 `pfc_watchdog` / pause 帧计数，逐跳定位是谁在反压。

### Q22. DCQCN 的闭环是怎样的？

要点：

1. 交换机按队列长度对包打 ECN（CE 标记）。
2. 接收端看到 CE，向发送端发 CNP（Congestion Notification Packet）。
3. 发送端收到 CNP 后按比例降速（rate decrease），之后按定时器逐步恢复（rate increase）。
- 三个可调面：交换机 ECN 门限、接收端 CNP 生成间隔、发送端加减速参数。
- 它解决的是「平均速率」，PFC 解决的是「瞬时 buffer 不溢出」，两者互补。

### Q23. IB 的 credit 流控和 PFC 有什么本质区别？

- IB credit 是**链路级、逐跳、基于接收方剩余 buffer 的授权**，发送方只在有 credit 时发，天然不丢。
- PFC 是**以太网的补丁**：先发后反压，pause 帧有传播延迟，所以仍需 buffer 余量。
- 因此 IB 的无损是协议设计出来的，RoCE 的无损是配置和调参维持的。

### Q24. ECN、DSCP、P_Key、priority 怎么配合？

要点：

- DSCP/ToS 标记流量类别 → 交换机映射到队列 → 该队列开 PFC 与 ECN。
- 主机侧要保证 RDMA 流量走对的 priority（`prio` / `traffic_class`），否则无损队列形同虚设。
- P_Key 是 IB 的分区隔离，不是 QoS；RoCE 下 VLAN 与 P_Key 有映射关系。

### Q25. 什么是自适应路由？和 ECMP 的差别？

要点：

- ECMP 按五元组哈希选路，大象流容易撞同一条链路，造成局部拥塞。
- 自适应路由（IB 的 AR、部分以太网方案的逐包/逐流切换）根据链路负载动态换路。
- 代价是可能乱序，需要网卡/协议容忍重排（IB 的 RC 能处理，但会退化为重传或增加 buffer）。

### Q26. SHARP 是什么？

要点：

- 把 AllReduce 等归约下推到交换机里做树内聚合，减少端点间的数据往返与带宽放大。
- 适合大模型训练里的梯度归约；对 all-to-all 这类模式帮助有限。

### Q27. 什么是 rail-optimized 拓扑？为什么 AI 集群这么连？

要点：

- 同一 rail 上的 GPU 连到同一组 leaf，跨 rail 只走少量上行。
- 让集合通信尽量在同 rail 内完成，减少跨 spine 跳数与拥塞；故障域也更清晰。
- 代价是布线复杂，且 all-to-all（MoE）会打穿 rail 假设，需要额外设计。

### Q28. RoCE 丢包之后到底会发生什么？为什么生产坚持 lossless？

要点：

- 丢包 → 接收端检测 PSN 空洞 → NAK 或发送端超时 → go-back-N 重传。
- 一次重传不仅补一个包，还会让窗口回退、后续包重复发送，尾部延迟被打散。
- 大规模集群里丢包率虽低，但「一次丢包影响一条大流」会被放大，所以默认按无损部署；lossy RoCE、IRN 是另一条路线。

---

## 四、编程模型

### Q29. 手写一遍 verbs 的初始化流程。

```text
ibv_get_device_list → ibv_open_device
→ ibv_alloc_pd
→ ibv_reg_mr
→ ibv_create_cq
→ ibv_create_qp
→ ibv_modify_qp(RESET→INIT→RTR→RTS)
→ ibv_post_send / ibv_post_recv
→ ibv_poll_cq
```

能被追问的细节：每个 `modify_qp` 的 attr 填什么、`qp_state` 与 `cur_qp_state` 的关系、`sq_sig_all` 对完成事件的影响。

### Q30. RDMA_CM 和手动建连怎么选？

- **RDMA_CM**：`rdma_create_id` → `rdma_resolve_addr` → `rdma_resolve_route` → `rdma_connect/accept`，内部帮你做地址/路由/事件管理。
- **手动**：自己用 TCP 交换 QPN、GID、rkey、虚拟地址，再三次 modify_qp；可控性最强，也最容易写错。
- 生产库常见做法：CM 建连 + 私有数据携带业务元信息，减少一次握手。

### Q31. 连接握手要交换哪些东西？

要点：

- QPN、LID/GID、GID index（RoCEv2 必须选对，常见坑）、P_Key、MTU、PSN 起点。
- 业务侧：MR 的虚拟地址与 rkey、缓冲区大小、slot/队列布局、版本号。
- 顺序问题：先交换再改 QP；RTR 需要对方的 QPN/PSN，RTS 之前双方都要到 RTR。

### Q32. Send 和 Write 怎么选？

| 维度 | Send/Recv | Write |
| --- | --- | --- |
| 对端 CPU | 必须 post recv | 不参与 |
| 对端内存地址 | 不需要知道 | 必须知道地址 + rkey |
| 通知机制 | 天然有 CQE | 需要 `WITH_IMM` 或对端轮询 |
| 流控 | recv WR 数量即信用 | 应用自己控制 |

选择依据：控制消息、长度不定、需要天然背压 → Send/Recv；单边推数据、接收端被动 → Write。

### Q33. `wc.status` 常见值代表什么？

- `IBV_WC_SUCCESS`
- `IBV_WC_LOC_LEN_ERR` / `LOC_PROT_ERR` / `LOC_ACCESS_ERR`：本地参数或权限问题
- `IBV_WC_REM_ACCESS_ERR`：对端 rkey/权限/越界
- `IBV_WC_REM_INV_REQ_ERR`：对端 QP 状态或参数不匹配
- `IBV_WC_RETRY_EXC_ERR`：重传耗尽（常见于对端 QP 没了或链路断）
- `IBV_WC_RNR_RETRY_EXC_ERR`：对端没有 recv WR
- `IBV_WC_WR_FLUSH_ERR`：QP 已 ERROR，挂起 WR 被清空

### Q34. 多线程怎么用 QP 和 CQ？

要点：

- QP 的 post 操作本身不保证线程安全（取决于 provider），通用做法是每线程独立 QP，或用锁/doorbell batching 合并提交。
- CQ 可被多线程 poll，但一个 CQE 只能被消费一次；多 poller 需要分片或加锁。
- 共享 QP 的公平性靠应用自己调度（这正好接面经里「无锁 QP share」的追问）。

### Q35. 怎么发现对端异常退出？

要点：

- RDMA_CM 事件：`RDMA_CM_EVENT_DISCONNECTED`、`DEVICE_REMOVAL`、`TIMEWAIT_EXIT`。
- 应用层心跳/租约：单边操作不会因为对端进程死了而自动失败，必须有超时检测。
- 注意：对端进程崩溃但 MR 仍在（内核/驱动未回收）时，Write 可能「成功」，这是单边语义的陷阱。

---

## 五、性能与调优

### Q36. 100G 网卡为什么经常跑不满？

要点：

- 线速 100 Gbps ≈ 12.5 GB/s；PCIe Gen3 x16 理论 15.75 GB/s，实际有效约 12~13 GB/s，读写混流就成瓶颈。
- 内存带宽与 NUMA：跨 socket 访问让 DMA 读变慢。
- 消息太小：每消息固定开销（doorbell、CQE、PCIe）占主导，带宽利用率低。
- 软件侧：锁、内存分配、CQ moderation、非独占核。

### Q37. 给定消息大小，怎么估算 message rate 和所需在途请求数？

要点：

- 消息率 ≈ 带宽 / 消息大小（含头部开销），小消息时受 IOPS/延迟上限约束而不是带宽。
- Little's law：在途请求数 = 消息率 × RTT。要达到 10 M msg/s、RTT 2 us，需要约 20 个在途。
- 队列深度、`max_send_wr`、responder resources 都要覆盖这个数。

### Q38. 怎么把延迟压到极限？

要点：

- 独占核 + 关 C-state / 频率锁频；中断与进程绑 NUMA。
- 大页 + 预注册 MR；小消息用 inline。
- 关闭 CQ moderation；用 busy polling；doorbell batching 换吞吐。
- 减少 QP 数量、避免跨 NUMA 的 CQ 访问。

### Q39. 队列深度和 outstanding WR 怎么定？

要点：

- 先算 BDP = 带宽 × RTT，再除以消息大小得到在途包数。
- 乘以安全系数（2~4）应对抖动；受网卡 `max_qp_wr`、`max_cqe` 限制。
- 太浅 → 打不满带宽；太深 → 尾延迟变大、内存与 cache 压力上升。

### Q40. GPUDirect RDMA 的原理？为什么能省一次拷贝？

要点：

- 把 GPU 显存的一段映射给网卡，让 NIC 通过 PCIe P2P 直接 DMA 到显存，绕过主机内存中转。
- 需要：支持 P2P 的拓扑（同 PCIe switch / NVLink 到网卡）、IOMMU 配置、GPU 内存注册。
- 与 NCCL 的关系：NCCL 在支持的平台上走 GDR，否则退回 host bounce buffer。

### Q41. 哪些因素会让 GPUDirect 退化？

要点：

- PCIe 拓扑跨 switch 且 ACS 开启，P2P 被禁止。
- IOMMU 未开 passthrough / 映射不正确。
- 显存未注册或注册类型不对；驱动/内核版本不匹配。
- 退化后仍能跑，但多一次 host 拷贝，带宽和延迟都变差。

---

## 六、排障场景

### Q42. `ib_write_bw` 两端连不上，怎么查？

要点：

1. `ibv_devices` / `ibv_devinfo` / `ibstat` 看设备与端口状态（ACTIVE）。
2. RoCE 下用 `show_gids` 确认 GID index 与 IP 对应，命令里显式指定 `-x <gid_index>`。
3. 检查 MTU 一致性、P_Key、防火墙（RoCEv2 是 UDP 4791）。
4. 看 `dmesg` 与网卡计数器，确认不是链路/驱动问题。

### Q43. `modify_qp` 到 RTR 返回 EINVAL，常见原因？

要点：

- GID index 填错（RoCEv2 常见，IPv4/IPv6/v1/v2 各占一个索引）。
- path MTU 超过设备或对端能力；QPN 填错；P_Key 不匹配。
- 端口未 ACTIVE；QP 类型与 attr 不匹配（例如 UD 填了 RC 才有的字段）。

### Q44. 收不到 CQE 怎么办？

要点：

- 是否 `sq_sig_all=0` 且该 WR 没设 `IBV_SEND_SIGNALED`。
- 是否 CQ 被其他线程 poll 走，或 poll 的 CQ 不是这个 QP 的。
- QP 是否已 ERROR（此时应有 flush CQE，若也没有，说明 CQ 参数/事件路径有问题）。
- `ibv_req_notify_cq` 的时序：必须在 poll 不到之后重新 arm，否则会丢事件。

### Q45. 延迟抖动大，从哪里下手？

要点：

- 先分「网络侧」和「主机侧」：交换机 PFC/ECN 计数、端口错误计数；主机侧看中断、CQ moderation、跨 NUMA。
- 常见元凶：PFC 反压、ECN 误标、buffer 不足、CPU 被其他任务抢占、内存页迁移。
- 对比实验：同机 loopback、同交换机、跨交换机，三段定位。

### Q46. IB 端口计数器怎么看？

要点：

- `symbol_error` / `port_rcv_errors`：物理层与链路层问题。
- `packet_seq_err` / `out_of_sequence`：丢包与乱序。
- `local_ack_timeout_err`：ACK 没回来，通常是链路或对端问题。
- `port_xmit_discards`：拥塞或策略丢弃。

### Q47. PFC storm 现场怎么定位？

要点：

- 看交换机 `pfc_watchdog` 触发与 pause 帧速率，确认是哪个 priority、哪个端口。
- 逐跳追：反压源往往不在告警点，而是下游某个慢端口或环路。
- 临时缓解：关掉该 prio 的 PFC 并限速，先恢复业务，再查根因。

### Q48. 程序 hang 在 poll 或等事件，怎么排查？

要点：

- 谁没 post recv（RC 的 Send 必须有对端 recv WR）。
- 谁没 poll CQ；是否两个线程抢同一个 CQ。
- 是否在等一个永远不会来的 `WRITE_WITH_IMM`（对端地址/rkey 错，写失败但被忽略）。
- 用 `strace`/`gdb` 看线程栈，用网卡计数器确认包是否真的发出。

---

## 七、架构与开放题

### Q49. 设计一个基于 RDMA 的 RPC 或 KV，你会怎么搭？

骨架：

- 连接层：按对端复用 QP，控制面用 CM，数据面用 Write + `WITH_IMM` 通知。
- 内存层：启动时注册大 arena，按 slot 分配；MR 与 rkey 只交换一次。
- 提交层：批量 post、doorbell batching、独立 CQ 线程或 busy poll。
- 故障层：心跳、超时、重建连接、幂等重试。

### Q50. 一万个节点全连接，RC 撑得住吗？

要点：

- 撑不住：连接状态 O(N²)，NIC 的 QP context cache 装不下，查找 miss 直接吃延迟。
- 方案：按需建连（DC）、共享接收队列（XRC/SRQ）、控制面走 UD、分层聚合（代理/网关）。
- 还要考虑建连风暴与 rkey 分发的管理成本。

### Q51. RDMA 和 DPDK 怎么选？

- DPDK：仍由 CPU 跑协议栈，灵活、可自定义，但吃核、吃内存带宽。
- RDMA：协议卸载到网卡，延迟与 CPU 占用最低，但语义固定、内存需注册。
- 现实里两者常共存：数据面 RDMA，控制面/异常路径走内核或 DPDK。

### Q52. Kubernetes 里怎么给 Pod 用 RDMA？

要点：

- SR-IOV VF 直通 + 设备插件上报 `rdma/xxx` 资源；或 macvlan/host network + RDMA CNI。
- 需要把 GID/设备节点、ulimit（memlock）、大页一起配好。
- 多网卡/多 rail 场景要做拓扑感知调度，否则跨 rail 通信退化。

### Q53. 流控与背压怎么做？

要点：

- Send/Recv 用 recv WR 数量天然限流；Write 需要应用层信用/窗口。
- 接收端按 slot 发 credit，发送端维护在途计数；超发就丢或等。
- 注意区分「网络拥塞」（PFC/ECN 管）和「应用消费慢」（信用管）。

### Q54. RDMA 的安全边界在哪？

要点：

- rkey + 虚拟地址就是能力凭证，泄露即可读写对端内存；要靠权限位、最小暴露、定期轮换。
- MR 越界、错误的长度字段是最常见的自伤。
- 容器/多租户下共享设备要隔离 PD/P_Key，避免一个 Pod 打到另一个 Pod 的 QP。

---

## 八、容易被追问的坑

1. **「RDMA 没有拷贝」不完全对**：注册时可能拷、跨设备时可能 bounce；准确说法是数据路径上没有内核缓冲区的拷贝。
2. **「单边操作不需要对端参与」只在数据面成立**：对端仍要提前注册 MR、给 rkey、保证内存不被释放。
3. **「RC 可靠所以不会丢」错**：可靠是端到端重传保证，不是链路不丢；丢包的代价体现在延迟和带宽放大。
4. **「PFC 开了就无损」错**：还要 buffer 预留、优先级规划、死锁防护，否则只是把丢包换成停顿。
5. **「延迟 1 us」要带条件**：消息大小、是否 inline、CQ moderation、CPU 频率、拓扑都会把它拉开数倍。
6. **「QP 越多吞吐越高」不一定**：连接状态与 cache miss 会反噬，DC/XRC 就是为此存在。

---

## 附：面经原题对照

| 面经里的问法 | 对应题号 |
| --- | --- |
| RDMA 传输与网卡交互的整个过程 | Q3、Q6、Q29 |
| 为什么需要 RDMA / RDMA 和 TCP 的本质差别 | Q1、Q3、Q20、Q22 |
| RDMA 建立连接的状态机 | Q7、Q31 |
| RDMA 哪些参数对性能有影响 | Q11、Q12、Q16、Q36–Q39 |
| RDMA 流量控制 / 调过哪些参数 | Q20–Q24 |
| RDMA 内存管理 | Q4、Q5、Q19 |
| RDMA 网卡里存了哪些信息 | Q5、Q6、Q7 |
| QP share 怎么做到无锁又公平 | Q34、Q49 |
| poll CQE 之后干什么 | Q13、Q33、Q44 |
| busy polling 怎么提高 CPU 利用率 | Q38 |
| RDMA 和虚拟化 / 容器结合 | Q52、Q54 |
| RoCE 组网会遇到的问题 / 部署难点 | Q20–Q28、Q47 |
| 拥塞控制：TCP 在哪感知，RoCE 在哪感知 | Q22 |
| 可扩展性是不是网卡 cache miss | Q6、Q50 |
