# RDMA 六层主线：把 54 道题挂到同一张图上

```text
本次讲解位置
章节：communication / RDMA
小节：六层问题图（L0 物理路径 → L1 对象 → L2 语义 → L3 可靠性 → L4 网络 → L5 工程）
知识点：一条消息的路径、路径上谁能碰谁、能做什么、丢了怎么办、路本身怎么不丢、怎么用好和怎么查坏
上次：32 · RDMA 面试题库骨架（Q1–Q54 逐题要点）
下次：34 · 连接状态账本（Q6 / Q50 的定量版）→ 35 · 拥塞控制参数闭环
```

## 为什么现在讲这个

[32 章](./32-rdma-interview-questions.md) 是题库：每道题给一句"面试官想听的话"。
问题是这 54 句话之间没有依赖关系，背到 Q36 就变成硬记，一追问"为什么"就断。

真正的关系是：**54 道题全是同一条路径上的六个切面**。先有路径，题目是被路径生成的，
不是被记住的。这一章做的事只有一件——把路径立起来，然后指出每道题挂在哪个节点上。

判断自己有没有学会，标准很简单：给你一个没背过的现象（比如"64 KB 消息比 1 KB 消息的
有效带宽高一个数量级"），你能从图上推出来，而不是从题库里翻出来。

## 心智模型：一条路径，六个切面

整个系列只依赖一个模型，后面所有章节都是它的展开：

```text
一条消息 = 6 段固定的物理流程
   填 WQE -> 敲 doorbell -> 网卡取 WQE -> 上线 -> 对端写内存 -> 产 CQE

这条流程有 3 个不变量，54 题全是它们的分身：
   1) 每段都有固定成本        -> 小消息被固定成本支配（33 章账本）
   2) 资源是有限的、可记账的  -> 连接/缓冲/表项都有字节数（34、39 章）
   3) 每个"更强"的保证都要    -> 可靠性、无损、顺序、隔离
      换一个资源付费           都是拿带宽/内存/CPU/延迟换的（42-45 章）
```

用这三条去套任何一道题：

```text
问"为什么慢"   -> 找哪一段的固定成本在放大（第 1 条）
问"为什么贵"   -> 找哪个资源在按 N 或按字节增长（第 2 条）
问"A 和 B 怎么选" -> 问"各自换的是哪个资源"（第 3 条）
```

这就是为什么这一章不需要背 54 个答案：**答案由不变量生成，不由记忆提供。**

---

## 全图

```text
L0  物理路径    一条消息从用户态到对端内存，六段
     └─ Q1 Q2 Q3 Q10 Q36

L1  对象        这条路径上谁能碰谁：PD / MR / QP / CQ / AH
     └─ Q4 Q5 Q6 Q7 Q19 Q29 Q30 Q31 Q43 Q50

L2  语义        走这条路能做什么：双边 / 单边 / 原子 / 通知 / 背压
     └─ Q8 Q9 Q14 Q15 Q16 Q17 Q32 Q53

L3  可靠性      路上丢了怎么办：PSN / ACK / 重传 / flush
     └─ Q11 Q12 Q13 Q18 Q28 Q33 Q35

L4  网络        路本身怎么不丢：credit / PFC / ECN / DCQCN / 拓扑
     └─ Q20 Q21 Q22 Q23 Q24 Q25 Q26 Q27 Q47

L5  工程        怎么用好、怎么查坏、怎么设计
     └─ Q34 Q37 Q38 Q39 Q40 Q41 Q42 Q44 Q45 Q46 Q48 Q49 Q51 Q52 Q54
```

上层不能绕过下层的约束：L2 的"单边不用对端 CPU"之所以成立，是因为 L0 第 5 段
NIC 能自己完成地址翻译；L3 的"go-back-N"之所以昂贵，是因为 L4 的以太网默认会丢。
**任何一道题的完整答案，都要说清它在哪一层、受哪一层约束。**

---

# L0 物理路径：一条消息的六段

## 先看 TCP 在 CPU 上做了什么

```text
用户态 send()
  → 系统调用进内核（保存/切换上下文）
  → 协议栈逐层处理（skb 分配、TCP 状态机、校验和）
  → 用户缓冲区 memcpy 到内核 skb
  → 驱动把 skb 交给网卡（doorbell）
  → 中断回来，软中断收包，再 memcpy 到用户缓冲区
```

RDMA 把三件事挪走了：

| 被挪走的开销 | 挪到哪 | 你为此付的新代价 |
| --- | --- | --- |
| 上下文切换 + 系统调用 | 用户态写 WQE，MMIO 写 doorbell | 用户态直接操作队列，写错了网卡照发 |
| 用户/内核缓冲区之间拷贝 | NIC 直接 DMA 已注册内存 | 内存必须提前 pin + 建翻译表 |
| CPU 跑可靠传输 | RNIC 硬件做 PSN/ACK/重传 | 网络必须无损或极低丢包 |

所以"RDMA 快"的准确表述是：**把每消息的 CPU 成本换成每字节的 DMA 成本，
把协议成本换成内存注册的前置成本和网络配置成本。**

## 六段路径

一条 1 KB 的 `SEND`，同机架、100 Gbps、关闭 CQ moderation：

```text
[1] 用户态组 WQE → 写进 SQ（host memory）
        ↓
[2] MMIO 写 doorbell（PCIe BAR）
        ↓
[3] NIC 读 WQE（PCIe read），按 SGE 读 payload（PCIe read）
        ↓
[4] 封装 BTH/GRH，上线
        ↓
[5] 对端 NIC 收包 → 地址翻译 → DMA write 到对端内存 → 回 ACK
        ↓
[6] 本地 NIC 写 CQE 到 CQ，应用 poll 到
```

| 段 | 动作 | 谁在做 | 典型量级 | 性质 |
| --- | --- | --- | --- | --- |
| 1 | 组 WQE + 写 SQ | CPU | 几十 ns | 纯内存写 |
| 2 | doorbell MMIO | CPU → PCIe | 几十 ~ 一两百 ns | 每消息固定 |
| 3 | 读 WQE、读 payload | NIC → PCIe | 两次数百 ns 的**串行**往返 | 每消息固定 |
| 4 | 线上传输 1 KB | 链路 | 81.9 ns | 与字节数成正比 |
| 5 | 对端翻译 + DMA + ACK | 对端 NIC | 数百 ns | 每消息固定 |
| 6 | CQE + poll | NIC → CPU | 几十 ns | 每消息固定 |

## 用真实数字算一遍

[`33_rdma_cost_model.py`](./33_rdma_cost_model.py) 把这一节的四张表都算出来，纯算术，不需要 RDMA 硬件：

```bash
cd ~/Downloads/learning_notes
python3 communication/33_rdma_cost_model.py
```

输出（节选）：

```text
== 1. 线上时间：消息大小 / 链路带宽 ==
      消息          100G          200G          400G
    1 KB        81.9 ns        41.0 ns        20.5 ns
    4 KB       327.7 ns       163.8 ns        81.9 ns
   64 KB      5242.9 ns      2621.4 ns      1310.7 ns
    1 MB     83886.1 ns     41943.0 ns     20971.5 ns

== 2. 线上时间占端到端延迟的比例（100G，固定成本 1500 ns）==
      消息       线上 ns      端到端 ns        线上占比     有效 Gbps
    1 KB        81.9      1581.9       5.2%         5.2
    4 KB       327.7      1827.7      17.9%        17.9
   64 KB      5242.9      6742.9      77.8%        77.8
  256 KB     20971.5     22471.5      93.3%        93.3
    1 MB     83886.1     85386.1      98.2%        98.2

== 3. 单 QP 消息率上限 = 1 / (固定成本 + 线上时间) ==
      消息                乐观                典型                保守
    1 KB         0.924 M/s         0.632 M/s         0.387 M/s
    4 KB         0.753 M/s         0.547 M/s         0.354 M/s
   64 KB         0.160 M/s         0.148 M/s         0.129 M/s
```

第一段 81.9 ns 的算法：

```text
1 KB = 1024 B = 8192 bit
100 Gbps = 100 × 10^9 bit/s
8192 / 100e9 = 81.9 × 10^-9 s = 81.9 ns
```

**这张表就是 Q36 和 Q37 的全部答案来源。** 1 KB 消息的端到端延迟里，
只有 5.2% 花在线上；也就是说 100 G 网卡测 1 KB 消息只能跑出约 5 Gbps，
不是网卡坏了，是每个消息都要重新付一遍 1.5 us 的固定成本。

拐点在哪？看第二张表：64 KB 时线上占比 77.8%，256 KB 时 93.3%。
所以"小消息"的经验分界大约在 32 KB～64 KB ——低于它，优化方向是**减少每消息开销**
（inline、doorbell batching、少 QP、关 moderation）；高于它，优化方向才是**提升带宽**
（大 MTU、多 QP、PCIe 代际）。

## 四个直接推论

**推论 1：inline 省掉的是一次串行的 PCIe 往返，不是"一点带宽"。**
第 3 段里，NIC 必须先读到 WQE 才知道 payload 在哪，这是依赖关系。
把 payload 直接塞进 WQE，就砍掉了一次数百 ns 的依赖读。
小消息上这个收益是延迟级（几十 %），不是带宽级。这是 Q10 真正的卖点。

**推论 2：`ibv_reg_mr` 贵在"建立翻译表"，所以它只能发生在连接建立期。**
第 5 段对端 NIC 做地址翻译时要查 MTT，表项在注册时就建好了。
注册一次 = pin 页 + 建表 + 生成 lkey/rkey。按消息注册，等于把每消息固定成本从
1.5 us 抬到几十 us —— 吞吐随消息率线性崩塌。这是 Q4 的完整因果。

**推论 3：单边写对端拿不到任何完成事件，因为第 5 段根本没有"通知"这个动作。**
对端 NIC 把数据 DMA 进内存就结束了，对端 CPU 不知道。
`WITH_IMM` 是在第 5 段末尾额外生成一个 CQE 投到对端 CQ。这是 Q9 的机制来源。

**推论 4：200G 网卡在 PCIe Gen3 x16 上跑不满。**
Gen3 x16 理论 15.75 GB/s，实际有效 12～13 GB/s；200 Gbps 需要 25 GB/s。
所以 Q36 的答案不只是"消息太小"，还有**总线代际与网卡带宽的匹配**。

---

# L1 对象：路径上谁能碰谁

## 五个对象，一个保护域

| 对象 | 是什么 | 生命周期 | 它决定路径的哪一段 |
| --- | --- | --- | --- |
| PD (Protection Domain) | 隔离容器 | 近似进程级 | 谁能访问谁：QP 与 MR 必须同 PD |
| MR (Memory Region) | 已注册内存 | 长（连接期） | 第 5 段能否翻译地址 |
| lkey | 本地访问凭证 | 随 MR | 第 3 段网卡读 payload 的凭据 |
| rkey | 远端访问凭证 | 随 MR | 第 5 段对端写你内存的凭据 |
| QP (Queue Pair) | SQ + RQ | 随连接 | 第 1/2/4 段的队列表 |
| CQ (Completion Queue) | 完成队列 | 随连接 | 第 6 段 |
| AH (Address Handle) | 目的地址描述 | UD 发送时用 | 第 4 段的目的地 |

一道很容易答错的追问：**PD 不是权限位，是命名空间。** 它只回答"这个 QP 能不能引用
这个 MR"，不回答"这个 MR 能不能被远端写"。后者是 `qp_access_flags` 与
`IBV_ACCESS_REMOTE_WRITE` 的组合。面试官问"怎么限制某个 QP 只能读不能写"，
答案是改 QP 的 access flags 或换 rkey 的权限位，不是换 PD。

## verbs 初始化：先建对象，再改状态（Q29）

```text
ibv_get_device_list
→ ibv_open_device
→ ibv_alloc_pd
→ ibv_reg_mr                 ← 最贵的一步，只做一次
→ ibv_create_cq              ← cqe = max_send_wr + max_recv_wr
→ ibv_create_qp              ← 填 cap：wr/sge/inline
→ ibv_modify_qp × 3          ← RESET→INIT→RTR→RTS
→ ibv_post_send / ibv_post_recv
→ ibv_poll_cq
```

`ibv_create_qp` 出来就是 `RESET`。**三个 modify 的顺序不能换**，
因为它们是"逐层打开开关"：

| 状态 | 能收 | 能发 | 必须填的字段 |
| --- | --- | --- | --- |
| RESET | 否 | 否 | 无（只设 `qp_state`） |
| INIT | 否 | 否 | `port_num`、`pkey_index`、`qp_access_flags` |
| RTR | **是** | 否 | `dest_qp_num`、`path_mtu`、`rq_psn`、`max_dest_rd_atomic`、`min_rnr_timer`、`ah_attr` |
| RTS | 是 | **是** | `sq_psn`、`timeout`、`retry_cnt`、`rnr_retry`、`max_rd_atomic` |
| ERROR | 否 | 否 | 只读；挂起 WR 全部 flush |

两个容易被追问的细节：

1. **`qp_access_flags` 在 INIT 填，不在 RTR 填。** 填晚了一步就 EINVAL。
2. **`cur_qp_state` 只在从 ERROR 回 RESET 时用。** 正常正向迁移只设 `qp_state`，
   内核自己维护当前状态；要复位才需要显式声明"我知道它现在是 ERROR"。

这条正好解释 Q43：`modify_qp` 到 RTR 返回 `EINVAL`，99% 是这四类——
GID index 填错、`path_mtu` 超过设备或对端能力、`dest_qp_num`/P_Key 不对、
端口没到 ACTIVE。全部是"字段与状态不匹配"，不是"网卡坏了"。

## QP 类型：连接状态才是稀缺资源（Q6、Q50）

| 类型 | 连接性 | 接收侧状态 | 保序 | 典型用途 |
| --- | --- | --- | --- | --- |
| RC | 一对一 | 每对一份 | 是 | 主流，延迟最低 |
| UC | 一对一 | 每对一份 | 不保证 | 少用 |
| UD | 一对多 | 每 QP 一份 | 否 | 发现、控制、多播 |
| XRC | 发送端每对、接收端共享 SRQ | 接收侧大幅减少 | 是 | 缓解接收侧连接爆炸 |
| DC (DCI/DCT) | 动态 | NIC 只缓存活跃连接 | 是 | 大规模 all-to-all |

Q50 问"一万节点全连接撑不撑得住"，答案不在带宽，在**连接上下文**：
RC 是 O(N²) 对 QP，每对一份 context，NIC 的 QP context cache 装不下就每次查找都 miss，
miss 一次就是一次 DDR 访问，几十到上百 ns，直接把延迟打穿。
所以 XRC 削的是**接收侧**状态，DC 削的是**非活跃连接**的状态，都不是为了带宽。

这也是 Q6 的答题落点：**"QP 越多吞吐越高"是错的**，连接状态和 cache miss 会反噬。

---

# L2 语义：走这条路能做什么

## 四象限：谁准备缓冲，谁收到通知

| 操作 | 谁准备接收缓冲 | 对端 CPU 参与 | 对端有 CQE | 典型语义 |
| --- | --- | --- | --- | --- |
| SEND | 接收端 post recv WR | 是 | 是 | 控制消息、流式 |
| SEND_WITH_IMM | 同上 | 是 | 是（带 4 字节） | 带标签的消息 |
| WRITE | 只需 MR + rkey | 否 | **否** | 推数据 |
| WRITE_WITH_IMM | 同上 | 否 | 是（带 4 字节） | 推数据 + 事件 |
| READ | 本地缓冲 | 否 | 是（本地） | 拉数据 |
| ATOMIC | 只需 MR + rkey | 否 | 是（本地） | 计数器、锁 |

这张表一行就能回答 Q8、Q9、Q32：**区别全部收敛到"对端要不要准备缓冲"和"对端要不要
被通知"两件事上。** 单边操作不需要对端准备，所以也不需要（也没有）对端 CQE；
`WITH_IMM` 是唯一能在这条路径上补一个通知的手段。

## 保序的边界（Q14）

```text
同一个 QP 内：RDMA 操作按 post 顺序执行，不需要 fence
跨 QP：       没有顺序保证，需要 IBV_SEND_FENCE
CPU 与 NIC：  CPU store 之后要 post_send，中间需要写屏障
```

最经典的错误是"先 WRITE 数据，再 SEND 一个通知"，且两者在不同 QP 上。
此时接收端可能先收到通知、后看到数据。`IBV_SEND_FENCE` 保证的是**同一个 QP 内**
前序 WR 在读侧先于后序 WR 可见；它不保证跨 QP，也不替代应用层的 acquire/release。

第二个坑在 Q14 的后半句：**本地收到 CQE 只说明本地完成，不说明对端内存对应用可见。**
对端应用要看到数据，需要它自己在消费侧做屏障，或者依赖 `WITH_IMM` 的 CQE 语义。

## 原子操作（Q15）

- `IBV_WR_ATOMIC_CMP_AND_SWP` / `IBV_WR_ATOMIC_FETCH_AND_ADD`
- 目标 MR 必须以 `IBV_ACCESS_REMOTE_ATOMIC` 注册
- 只对 8 字节对齐的 64 位字有效
- 延迟比 WRITE 高（要读-改-写一个往返），**不要放在批量数据路径上**

## 背压：Send 有天然的，Write 没有（Q53）

这是 Q53 的核心，也是设计 RDMA 程序最容易漏的一层：

```text
SEND  : recv WR 的数量就是信用额度。对端不 post，你就 RNR，天然限流。
WRITE : 没有任何限流。对端消费慢，你的数据照样写满它的缓冲区。
```

所以用 WRITE 推数据，必须自己造一套 credit：接收端按 slot 发放信用，
发送端维护在途计数，超发就等。**发送端还得分清"网络拥塞"和"应用消费慢"**——
前者归 PFC/ECN 管，后者只能归应用信用管，用错工具会互相掩盖。

## MTU 与 UD（Q16、Q17）

MTU 取 256/512/1024/2048/4096，**双方必须一致**，否则 RTR 就配不上。
4K 是常见默认：头部与处理开销摊得最薄；代价是单包时延变大、交换机 buffer
被一个包占得更久、PFC 反压粒度变粗。

UD 每个包带 40 字节 GRH，无分段重组，所以**消息不能超过 MTU**。
它的价值不是性能，是省连接状态：一个 QP 能对任意目的发，适合建连前的发现和
小控制消息。

---

# L3 可靠性：路上丢了怎么办

## 三个机制，一条因果链

```text
PSN（序号）  →  接收端据此发现丢包 / 乱序 / 重复
ACK / NAK    →  告诉发送端"收到哪了"或"从哪开始重发"
超时重传      →  ACK 没回来时兜底
```

PSN 是 24 位，在 QP 内单调递增，回绕周期是 \(2^{24}\) 个包。
发送端维护窗口，收到 ACK 推进窗口；收到 NAK 或 `local_ack_timeout` 超时则重传。

`local_ack_timeout` 的编码是 4.096 us × \(2^{\text{value}}\)，value 取 0–31。
所以最小 4.096 us，最大约 2.2 秒。**这个参数不能小于链路 RTT**，
否则你会把"正常在途"误判成"丢了"。

## go-back-N 为什么这么贵（Q11、Q28）

RC 的重传不是"只补丢的那一个包"，是**从丢失点开始重发窗口内所有已发未确认的包**。

`33_rdma_cost_model.py` 第六段：

```text
窗口 W = 64，第 k 个包丢了 → 重发 W - k + 1 个包

   丢失位置 k    重发包数    放大倍数
         1          64         64x
        16          49         49x
        32          33         33x
        48          17         17x
        64           1          1x
```

**丢得越早，放大越狠。** 这就是 Q20 说"以太网丢包的代价是 go-back-N"的定量版本，
也是 Q28 说"一次丢包会把尾延迟打散"的原因——不只多传 N 个包，
还让窗口回退、后续包重复发送、接收端乱序缓冲被占满。

## 三个"重试"参数别混（Q12）

| 参数 | 谁设 | 管什么 | 耗尽后的错误 |
| --- | --- | --- | --- |
| `retry_cnt` | 发送端 | 超时/NAK 后的重传次数，7 = 无限 | `IBV_WC_RETRY_EXC_ERR` |
| `rnr_retry` | 发送端 | 对端 RNR（没 recv WR）后的重试次数，7 = 无限 | `IBV_WC_RNR_RETRY_EXC_ERR` |
| `min_rnr_timer` | 接收端 | RNR 时告诉发送端"等我多久再来" | — |
| `max_dest_rd_atomic` | 接收端能力 | 对端能同时挂多少个 RDMA READ/Atomic | `IBV_WC_REM_INV_REQ_ERR` 类 |

**这里要修正 32 章 Q12 的一处不精确。** 题库里把 "responder resources" 说成
"接收端允许多少个乱序或未完成的请求同时在途"，太宽了。准确地说，
`max_dest_rd_atomic`（responder resources）**只约束 RDMA READ 和 Atomic**
这类"需要响应方回数据"的请求，**不约束 SEND**。SEND 的背压来自 recv WR 数量。
这个区别在面试里是加分项，在调优里是决定要不要调这个参数的分水岭：
只跑 SEND/WRITE 的业务，调它没有意义。

## flush vs retry exhausted：两种完全不同的失败（Q13、Q33）

```text
RETRY_EXC_ERR      ：包发得出去，对端不应答（QP 没了、链路断、对端进程死）
RNR_RETRY_EXC_ERR  ：对端活着，但没 post recv WR
WR_FLUSH_ERR       ：QP 已经进 ERROR，队列里剩下的 WR 被"连带"清空
```

区别在于**因果**：前两个是这个 WR 本身失败的根因；`WR_FLUSH` 是这个 WR 无辜受牵连。
排障时应当**先找第一个非 flush 的错误**，它在 CQE 序列里通常排在这批 flush 之前。
只盯 `WR_FLUSH` 去查这个 WR 的参数，会永远查不出来。

完整的 `wc.status` 分区：

| 分区 | 代表值 | 说明 |
| --- | --- | --- |
| 本地参数/权限 | `LOC_LEN_ERR`、`LOC_PROT_ERR`、`LOC_ACCESS_ERR` | 你给网卡的东西不合法 |
| 远端拒绝 | `REM_ACCESS_ERR`、`REM_INV_REQ_ERR` | 对端 rkey/权限/QP 状态不对 |
| 传输失败 | `RETRY_EXC_ERR`、`RNR_RETRY_EXC_ERR` | 对端没应答或没缓冲 |
| 连带清空 | `WR_FLUSH_ERR` | QP 已 ERROR |

## 单边语义的"假成功"（Q35）

对端进程崩了，但 MR 还在内核/驱动里没回收，你的 `WRITE` 会返回 `IBV_WC_SUCCESS`。
数据真的写进物理内存了，只是没有人消费。

**这不是 bug，是单边语义的定义。** 所以任何用 WRITE 的系统都必须自己带心跳或租约，
不能把 "write 成功" 当成 "对端应用处理了"。Q35 的答案落点就在这里。

## 让注册变便宜：FRWR 与 MW（Q19）

`ibv_reg_mr` 每次都要 pin 页 + 建表，存储 IO 场景里缓冲频繁更换，受不了这个成本。
两条退路：

| 机制 | 做法 | 适合 |
| --- | --- | --- |
| FRWR (Fast Registration WR) | 预先建一个"钥匙" MR，再用 `IBV_WR_FAST_REG_MR` 把它绑到新地址，用完 `IBV_WR_LOCAL_INV` 失效 | 内核态存储 IO（NFS/RDMA、NVMe-oF） |
| MW (Memory Window) | 对一块已注册的大 MR，用 `IBV_WR_BIND_MW` 临时开放一个子区间 | 用户态，需要动态授权远端访问 |

两个机制的动机完全一样：**把"地址变化"的成本从 pin 页降到写一个 WR**。

---

# L4 网络：路本身怎么不丢

## 无损的两种造法

| | IB credit | RoCE PFC |
| --- | --- | --- |
| 机制 | 接收方逐跳广告剩余 buffer 额度，发送方只有拿到 credit 才发 | 先发，buffer 快满时发 pause 帧让上游停 |
| 时机 | **事前授权** | **事后反压** |
| 谁保证 | 协议设计 | 配置 + 调参 |
| 失败模式 | 配置错就发不动（不会丢） | 配错就死锁 / HOL / storm |

一句话记住 Q23：**IB 的无损是设计出来的，RoCE 的无损是运维维持出来的。**
这句话能延伸到 Q2、Q20、Q24、Q47 全部题目。

PFC 的 pause 帧有一个常被忽略的单位：**quanta = 512 bit times**。
100 Gbps 下 1 quanta = 5.12 ns，最大值 65535 quanta ≈ 336 us。
所以一条 "pause 65535" 能把上游停 300 多微秒——这就是 PFC storm 能打死整个集群的原因。

## PFC 死锁：环路加逐跳反压（Q21）

```text
A 的 buffer 被 B 的流量占满 → A 给上游发 pause
B 的 buffer 被 A 的流量占满 → B 给上游发 pause
如果 A、B 互为上游 → 双方都等对方腾 buffer → 永久停住
```

四层防护，缺一不可：

1. **无环**：STP / 路由 / 拓扑设计，这是前提，不是优化
2. **watchdog**：检测某个端口持续收到 pause 超过阈值，就忽略/丢弃并告警
3. **限制反压域**：别让一个慢端口把整片网络按停
4. **优先级隔离**：只给 RDMA 那一个 priority 开 PFC，别开全队列

现场定位的第一步永远是**看 watchdog 和 pause 帧计数**，第二步是**逐跳往上追**——
反压源通常不在告警的那个端口，而在下游某个慢端口或环路上。

## ECN + DCQCN：管平均速率（Q22）

```text
交换机：队列长度超过 Kmin → 按概率打 CE 标记（到 Kmax 概率到 Pmax）
        ↓
接收端：看到 CE → 生成 CNP，按流回给发送端（有速率限制，不会每个包都发）
        ↓
发送端：收到 CNP → 按 alpha 降速；没有 CNP → 定时器逐步升速
```

三个可调面，正好对应三段：

| 位置 | 参数 | 调错的表现 |
| --- | --- | --- |
| 交换机 | `Kmin` / `Kmax` / `Pmax` | 门限太低 → 频繁误标，吞吐掉；太高 → 不触发，靠丢包 |
| 接收端 | CNP 生成间隔 | 太密 → 发送端被过度降速；太疏 → 反应不过来 |
| 发送端 | `g`、加性增、乘性减系数 | 恢复太慢 → 带宽利用率低；太快 → 振荡 |

**PFC 和 DCQCN 不是二选一，是不同时间尺度**：PFC 管"这一瞬间 buffer 别溢出"（微秒级），
DCQCN 管"这条流的平均速率别超过公平份额"（毫秒级）。只开 PFC，你会用停顿时不时替代丢包；
只开 ECN，网络一抖就丢包重传。

## 标记怎么一路传下去（Q24）

```text
应用设 DSCP/ToS
  → 交换机按 DSCP 映射到队列（priority）
  → 该队列开 PFC + ECN
  → 主机的 QP 必须走对 priority（ibv_modify_qp 的 traffic_class / prio）
```

最后一环最容易漏：**主机侧 QP 的 priority 配错，前面所有配置都是白做的**，
因为流量根本没进那个无损队列。P_Key 是 IB 的分区隔离，不是 QoS，
两者不要混答。

## 路由、聚合与拓扑（Q25、Q26、Q27）

| 机制 | 解决什么 | 代价 |
| --- | --- | --- |
| ECMP | 等价多路径分流 | 五元组哈希，大象流会撞同一条链路 |
| 自适应路由 | 按链路负载动态换路 | 可能乱序，触发重传，需要更大 buffer |
| SHARP | 把 AllReduce 的归约下推到交换机 | 只适合归约类，all-to-all 用不上 |
| rail-optimized | 让同号 GPU 走同一组 leaf | 布线复杂，MoE 的 all-to-all 会打穿假设 |

Q27 的深层原因值得单独说：rail-optimized 是把"集合通信的通信模式"（同号卡互换）
编码进了物理拓扑。一旦模型变成 all-to-all（MoE 的 dispatch/combine），
通信不再是同号卡之间，rail 假设就失效了——这正是 DeepEP 那批工作要解决的问题之一。

---

# L5 工程：怎么用好、怎么查坏

## 性能模型：三个公式覆盖 Q36–Q39

```text
① 端到端延迟 = 每消息固定成本 + 线上时间
              ≈ 1.5 us + size × 8 / 带宽

② 单 QP 消息率上限 = 1 / (固定成本 + 线上时间)
                  1 KB@100G ≈ 0.63 M msg/s

③ Little's law：在途请求数 N = 消息率 × RTT
                10 M msg/s × 2 us = 20 个
```

第 ② 条解释了一个常见困惑：为什么单 QP 打不到 10 M msg/s。
0.63 M/s 就是它的物理天花板，想上 10 M/s 必须**多 QP 并行**，
或者把固定成本降下去（inline、doorbell batching）。这是 Q39 的全部依据。

第 ③ 条是 Q37 的标准答法，也是队列深度的起点：

```text
队列深度 = BDP / 消息大小 × 安全系数
BDP(100G, RTT 2us) = 12.5 GB/s × 2 us = 25 KB
1 KB 消息 → 25 个在途 → 取 2~4 倍 → max_send_wr = 64~96
```

脚本第五段的输出可以直接抄：

```text
    链路   RTT us    BDP KB      1KB 消息在途     x2 安全
  100G      2.0      24.4          24.4        48
  200G      2.0      48.8          48.8        97
  400G      5.0     244.1         244.1       488
```

注意 400G / RTT 5us 需要 488 个在途——这已经超过不少网卡的 `max_qp_wr` 默认值，
也解释了为什么大带宽下必须用 DC 或者多 QP。

## 延迟极限清单（Q38）

按收益从大到小，全部能挂回 L0 的六段：

| 手段 | 打的是哪一段 | 注意 |
| --- | --- | --- |
| 独占核 + 关 C-state + 锁频 | 1、6 | 被抢占会引入几十 us 抖动 |
| 进程/中断绑同一 NUMA | 3、5 | 跨 socket 的 DMA 读明显更慢 |
| 大页 + 预注册 MR | 1、5 | 4 KB 页 1 GB 就是 26 万表项 |
| 小消息 inline | 3 | 砍一次依赖 PCIe 读 |
| 关 CQ moderation | 6 | 换延迟不换吞吐 |
| busy polling | 6 | 吃满一个核，换掉中断延迟 |
| doorbell batching | 2 | 换吞吐，牺牲单消息延迟 |
| 减少 QP 数量 | 3 | 降低 NIC context 与 CQ 的 cache 压力 |

## GPUDirect：把第 5 段的目的地从 host 内存换成显存（Q40、Q41）

```text
普通路径：NIC → host 内存（bounce buffer）→ cudaMemcpy → 显存
GDR 路径：NIC → PCIe P2P → 显存
```

省掉的是中间那次 host 内存中转和一次拷贝。成立的四个条件，缺一个就退化：

1. 拓扑支持 P2P（同一 PCIe switch，或 NVLink 直连网卡）
2. ACS 关闭（或做了 P2P 定向），否则 IOMMU 会把 P2P 挡掉
3. 显存已注册给网卡（`nvidia-peermem` / `ib_register_peer_memory_client`）
4. 驱动与内核版本匹配

**退化的表现不是报错，是"能跑但慢"**——多一次 host 拷贝，带宽掉、延迟涨。
所以 Q41 的答案要落在"怎么确认它在不在 GDR 路径上"，而不是"支持不支持"。

## 排障：先分层，再看计数器（Q42–Q48）

把六层倒过来用，就是一棵排查树：

```text
现象
├─ 根本连不上 ─────────→ L1/L4
│   ├─ ibv_devices / ibstat：设备在不在、端口 ACTIVE 吗
│   ├─ show_gids：GID index 和 IP 对得上吗（RoCEv2 第一大坑）
│   ├─ MTU 一致吗、P_Key 对吗、UDP 4791 通吗
│   └─ dmesg / 网卡计数器：链路层还是驱动
│
├─ modify_qp 失败 ─────→ L1
│   └─ EINVAL 四类：GID index / path_mtu / qpn·pkey / 端口状态
│
├─ 收不到 CQE ────────→ L3/L5
│   ├─ sq_sig_all=0 且没设 SIGNALED？
│   ├─ CQ 被别的线程 poll 走了？poll 的是这个 QP 的 CQ 吗？
│   ├─ QP 已 ERROR？那应该有 flush CQE，没有就是 CQ 参数问题
│   └─ ibv_req_notify_cq 的 arm 时序：必须在 poll 不到之后重新 arm
│
├─ hang 在 poll ──────→ L2/L3
│   ├─ 对端没 post recv（SEND 必须有）
│   ├─ 两个线程抢同一个 CQ
│   └─ 在等一个永远不会来的 WITH_IMM（rkey/地址错，写失败被忽略）
│
├─ 延迟抖动 ──────────→ L4/L5
│   ├─ 三分段：同机 loopback → 同交换机 → 跨交换机
│   ├─ 交换机：PFC pause 计数、ECN 标记率、pfc_watchdog
│   └─ 主机：中断、CQ moderation、跨 NUMA、CPU 抢占
│
└─ 带宽上不去 ────────→ L0
    ├─ 消息多大？小于 64 KB 就是固定成本主导
    ├─ PCIe 代际够吗？（200G 需要 Gen4 x16 以上）
    └─ 多少个 QP？单 QP 的消息率有天花板
```

四个最常用的计数器，对应四种不同的病因：

| 计数器 | 说明什么 |
| --- | --- |
| `symbol_error` / `port_rcv_errors` | 物理层与链路层（线、光模块、误码） |
| `packet_seq_err` / `out_of_sequence` | 丢包与乱序（PFC 没兜住、ECN 误标） |
| `local_ack_timeout_err` | ACK 没回来（对端问题或链路断） |
| `port_xmit_discards` | 拥塞或策略丢弃（buffer 不够、ACL） |

**排障的通用顺序是：先确认包发出去了没有，再确认对端收到没有，最后才看应用。**
网卡计数器能回答前两个问题，`strace`/`gdb` 只能回答第三个。顺序反了就会在应用层空转。

## 架构选型（Q49–Q54）

**RDMA RPC / KV 的四层骨架（Q49）**

| 层 | 决策 | 理由 |
| --- | --- | --- |
| 连接层 | 按对端复用 QP；控制面走 CM | 建连昂贵，复用摊薄 |
| 内存层 | 启动时注册大 arena，按 slot 分 | 注册只做一次，rkey 只换一次 |
| 提交层 | 批量 post + doorbell batching + 独立 CQ 线程 | 摊薄每消息固定成本 |
| 故障层 | 心跳、超时、重建连接、幂等重试 | 单边语义不会自动报错 |

**一万节点（Q50）** 已经在上文 L1 讲过：问题不是带宽，是 O(N²) 的连接状态与
NIC 的 context cache。四条出路：DC、XRC/SRQ、控制面走 UD、分层代理。

**RDMA vs DPDK（Q51）**：DPDK 是"用 CPU 更高效地跑协议栈"，RDMA 是"把协议栈
卸给网卡"。前者可编程、可自定义协议，吃核吃内存带宽；后者延迟和 CPU 占用最低，
但语义固定、内存要注册。生产里常共存：数据面 RDMA，控制面和异常路径走内核或 DPDK。

**Kubernetes（Q52）**：SR-IOV VF 直通 + 设备插件上报 `rdma/xxx` 资源，
或者 macvlan/host network + RDMA CNI。三个最容易漏的配置：
`memlock` 的 ulimit、大页、GID index 与 Multus 网络的对应关系。
多 rail 场景还要做拓扑感知调度，否则跨 rail 通信会退化。

**安全（Q54）**：`rkey + 虚拟地址` 本身就是能力凭证，拿到就能读写，
没有额外的身份认证。所以边界靠三件事：权限位最小化（只给 `REMOTE_READ` 就别给
`REMOTE_WRITE`）、rkey 轮换、多租户下 PD/P_Key 隔离。

---

# 54 题定位表

每道题的"一句话锚点"和它挂的层。答题时先定位层，再展开。

| 题 | 层 | 一句话锚点 |
| --- | --- | --- |
| Q1 | L0 | 三次搬家：上下文、拷贝、协议 |
| Q2 | L0/L4 | 封装方式决定"无损由谁保证" |
| Q3 | L0 | 固定成本约 1.5 us，1 KB 线上只占 5% |
| Q4 | L1 | 贵在 pin 页 + 建 MTT，不在拷贝 |
| Q5 | L1 | PD 是命名空间，lkey/rkey 是凭证 |
| Q6 | L1 | 连接状态是稀缺资源，不是带宽 |
| Q7 | L1 | 四个状态是四组字段的开关 |
| Q8 | L2 | 谁准备缓冲，谁收到通知 |
| Q9 | L2 | 第 5 段末尾补一个 CQE |
| Q10 | L0 | inline 砍依赖 PCIe 读，SGE 砍拼接拷贝 |
| Q11 | L3 | 序号 + 窗口 + go-back-N |
| Q12 | L3 | `max_dest_rd_atomic` 只管 READ/Atomic，不管 SEND |
| Q13 | L3 | QP 进 ERROR，挂起 WR 连带清空 |
| Q14 | L2 | 同 QP 保序，跨 QP 不保序 |
| Q15 | L2 | 8 字节对齐 + `REMOTE_ATOMIC` |
| Q16 | L2/L3 | 越大开销越低，但反压粒度变粗 |
| Q17 | L2 | 省连接状态，代价是无序无重传 |
| Q18 | L3 | A 位驱动 + ACK 聚合，不是每包必回 |
| Q19 | L1 | 把注册拆成"钥匙 + 绑定" |
| Q20 | L4 | 丢包的代价是 go-back-N 放大 |
| Q21 | L4 | 环路 + 逐跳反压 = 死锁 |
| Q22 | L4 | 交换机标记 / 接收端 CNP / 发送端调速率 |
| Q23 | L4 | IB 事前授权，RoCE 事后反压 |
| Q24 | L4 | DSCP → 队列 → PFC/ECN，主机 prio 是最后一环 |
| Q25 | L4 | 哈希撞流 vs 动态换路，代价是乱序 |
| Q26 | L4 | 归约下推到交换机，只对 reduce 类有效 |
| Q27 | L4 | 把 collective 的通信模式编码进拓扑 |
| Q28 | L3/L4 | 窗口回退，尾延迟被打散 |
| Q29 | L1 | 先建对象，再逐层打开状态开关 |
| Q30 | L1 | CM 管地址/路由/事件，手动最可控也最易错 |
| Q31 | L1 | 协议侧六项 + 业务侧 rkey/vaddr |
| Q32 | L2 | 要背压用 SEND，要单边用 WRITE |
| Q33 | L3 | 本地 / 远端 / 重试 / flush 四分区 |
| Q34 | L5 | post 未必线程安全，CQE 只能消费一次 |
| Q35 | L3 | 单边写成功 ≠ 对端应用收到了 |
| Q36 | L5 | 固定成本 + PCIe 代际，不是网卡坏了 |
| Q37 | L5 | Little's law：N = 消息率 × RTT |
| Q38 | L5 | 逐段打：核、NUMA、inline、moderation |
| Q39 | L5 | BDP / 消息大小 × 2~4 |
| Q40 | L5 | PCIe P2P 直达显存，省 host 中转 |
| Q41 | L5 | ACS / IOMMU / 注册 / 驱动，退化表现为变慢 |
| Q42 | L5 | 设备 → GID → MTU/P_Key → 防火墙 |
| Q43 | L1 | 字段与状态不匹配，四类原因 |
| Q44 | L5 | signaled / CQ 归属 / ERROR / arm 时序 |
| Q45 | L5 | 三分段定位：loopback → 同交换机 → 跨交换机 |
| Q46 | L5 | 物理 / 丢包 / 超时 / 丢弃，四类计数器 |
| Q47 | L4/L5 | watchdog + 逐跳找反压源 |
| Q48 | L5 | recv WR / CQ 竞争 / 永不来的 IMM |
| Q49 | L5 | 连接、内存、提交、故障四层 |
| Q50 | L1/L5 | O(N²) 与 context cache miss |
| Q51 | L5 | 卸载 vs 可编程，常态是共存 |
| Q52 | L5 | SR-IOV + 设备插件 + memlock/大页 |
| Q53 | L2 | recv WR 即信用；WRITE 要自建窗口 |
| Q54 | L5 | rkey 即能力凭证 |

---

# 答题模板

被问到任意一题时，按这四步组织，比背要点稳：

```text
① 定位层      "这属于传输层的可靠性问题"
② 说机制      "靠 PSN + ACK/NAK + 超时重传，重传是 go-back-N"
③ 给代价/边界 "所以一次丢包的代价是窗口内重发，尾延迟被打散"
④ 落到工程    "生产里因此坚持无损部署，靠 PFC 兜瞬时、DCQCN 管平均"
```

四步里第 ③ 步最能区分"背过"和"懂了"。**每道题都有一个代价或边界**，
说不出代价，就说明还停在定义层。

---

# 三处需要修正题库的地方

写这一章时对 32 章做了交叉检查，有三处表述不够精确，改掉之后面试更稳：

| 位置 | 原表述 | 修正 |
| --- | --- | --- |
| Q12 | "responder resources 是接收端允许多少个乱序或未完成的请求同时在途" | 它是 `max_dest_rd_atomic`，**只约束 RDMA READ / Atomic**，不约束 SEND；SEND 的背压来自 recv WR 数量 |
| Q18 | "接收端按策略聚合 ACK，比如每 N 个包回一次" | 实际是 **A 位（Ack Request）驱动**：请求包带 A=1 时接收端必须回 ACK，否则可合并/延后 |
| Q40 | "让网卡 DMA 到显存，绕过主机内存中转" | 准确说是**绕过 host bounce buffer**；第一次 P2P 仍要建立 BAR/地址映射，不是零配置 |

---

# 自测

先合上笔记口述，再看答案。

1. 1 KB 消息在 100G 上，端到端延迟里线上时间占多少？这解释了哪个常见误判？
2. 为什么 `ibv_reg_mr` 不能放进消息路径？说出它具体做了哪三件事。
3. `IBV_WC_RETRY_EXC_ERR` 和 `IBV_WC_WR_FLUSH_ERR` 的因果区别？排障时先看哪个？
4. 只跑 SEND/WRITE 的业务，调 `max_dest_rd_atomic` 有用吗？为什么？
5. 单 QP 打不到 10 M msg/s，是配置问题还是物理上限？给出算式。
6. 400G、RTT 5 us、1 KB 消息，队列深度该设多少？会遇到什么限制？
7. PFC 和 DCQCN 分别管哪个时间尺度？只开一个会发生什么？
8. 一个 Write 返回 `IBV_WC_SUCCESS`，能说明对端应用处理了吗？为什么？

<details>
<summary>答案</summary>

1. 81.9 ns / 约 1582 ns ≈ **5.2%**。解释的是"100G 网卡测小消息只能跑出几 Gbps"
   这个误判——不是硬件坏，是每消息固定成本主导（Q36）。
2. 它做三件事：pin 页（防换出/迁移）、建立虚拟地址到物理地址/IOVA 的翻译表（MTT）、
   生成 lkey/rkey 并做权限检查。单次成本是 O(页数)，1 GB / 4 KB 页 = 26 万表项。
   放进消息路径等于把每消息成本从 1.5 us 抬到几十 us（Q4）。
3. `RETRY_EXC` 是这个 WR 自己失败的根因（发得出去、对端不答）；`WR_FLUSH` 是 QP 已 ERROR
   后的连带清空，WR 本身无辜。**先找 CQE 序列里第一个非 flush 的错误**（Q13/Q33）。
4. 没用。`max_dest_rd_atomic` 只约束 RDMA READ / Atomic 的并发在途数，
   SEND/WRITE 的流控来自 recv WR 数量和 PFC/ECN（Q12）。
5. **物理上限**。单 QP 消息率 = 1/(固定成本 + 线上时间) = 1/(1500 ns + 82 ns) ≈ 0.63 M/s。
   要 10 M/s 必须多 QP，或用 doorbell batching / inline 压低固定成本（Q37/Q39）。
6. BDP = 50 GB/s × 5 us = 250 KB；1 KB 消息 → 250 个在途 → ×2 ≈ **500**。
   会遇到 `max_qp_wr` 默认值不够，需要调大或改多 QP / DC（Q39）。
7. PFC 管**瞬时**（微秒级，buffer 不溢出）；DCQCN 管**平均速率**（毫秒级，公平份额）。
   只开 PFC：把丢包换成停顿，容易死锁和 HOL。只开 ECN：一抖就丢包，go-back-N 放大（Q20/Q22）。
8. 不能。单边语义下"写成功"只说明数据进了对端物理内存。对端进程崩了但 MR 没回收时，
   Write 依然返回成功。所以必须靠应用层心跳/租约（Q35）。

</details>

---

## 题目 → 章节对照

上面的定位表解决"挂在哪一层"，这张表解决"去读哪一章"：

| 题号 | 章节 | 主题 |
| --- | --- | --- |
| Q1–Q3 | [33](./33-rdma-six-layer-spine.md) | 六段路径与固定成本主线 |
| Q4、Q19 | [39](./39-rdma-memory-registration-cost.md) | 内存注册成本、FRWR / MW |
| Q5、Q6、Q7、Q29–Q31、Q34 | [34](./34-rdma-connection-state-ledger.md)、[38](./38-verbs-program-walkthrough.md) | 对象树、状态机、verbs 实操 |
| Q8–Q10、Q14–Q17、Q32 | [43](./43-rdma-l2-remaining-semantics.md) | 双边/单边、inline、imm、atomic、UD、fence |
| Q11–Q13、Q18、Q28、Q33、Q35 | [42](./42-rdma-reliability-psn-window.md) | PSN、ACK、go-back-N、flush、假成功 |
| Q20–Q25 | [35](./35-rdma-congestion-control-loop.md)、[45](./45-rdma-vs-dpdk-and-routing.md) | PFC、ECN/DCQCN、标记、路由 |
| Q26、Q27 | [40](./40-rdma-collective-topology.md) | 集合通信与拓扑映射 |
| Q36–Q39 | [46](./46-rdma-performance-capacity-planning.md) | 有效带宽、消息率、延迟、深度 |
| Q40、Q41 | [37](./37-rdma-gpudirect-degradation.md) | GPUDirect 路径与退化 |
| Q42–Q48 | [36](./36-rdma-troubleshooting-decision-tree.md) | 分层排障与判定树 |
| Q49、Q50、Q53 | [47](./47-rdma-rpc-kv-design.md)、[44](./44-rdma-flow-control-backpressure.md) | RPC/KV 设计、背压与 credit |
| Q51、Q25 | [45](./45-rdma-vs-dpdk-and-routing.md) | RDMA vs DPDK、自适应路由 |
| Q52、Q54 | [41](./41-rdma-kubernetes-multitenancy.md) | K8s、多租户隔离边界 |

反查方法：`python3 communication/verify_rdma_series.py` 会检查每一章的小节、
脚本可执行性与链接有效性 —— 这张表里任何一个链接失效都会报错。

---

## 常见错法：六层之间最容易串的十件事

这一章的价值一半在"把 54 题挂上路径"，另一半在"把容易混淆的跨层概念分开"：

| 错法 | 它错在哪一层 | 症状 |
| --- | --- | --- |
| 把 PD 当隔离边界 | L1 vs 部署 | 同 VF 内拿到 rkey 就能互读内存（41 章） |
| 以为 `max_dest_rd_atomic` 限制所有操作 | L1 | SEND/WRITE 调它完全无效（42 章） |
| 把 CQ 深度按单边设 | L1 | 压测时静默丢 CQE，低负载正常 |
| 以为 WRITE 有背压 | L2 | 数据被静默覆盖，`wc.status` 全是 SUCCESS（44 章） |
| 把 inline 当成"省一次拷贝" | L2 | 实际省的是 NIC 回读 payload 的 PCIe 访问（43 章） |
| 把 `RETRY_EXC` 和 `WR_FLUSH` 当成两类故障 | L3 | 追着 flush 查，根因在第一个非 flush 错误上 |
| 用 `p=0.1%` 论证"不需要无损" | L3 vs L4 | 忽略窗口：W=256 时这是 19% 的税（42 章） |
| 以为 PFC 能保护应用缓冲 | L4 vs L2 | 网络零丢包，应用仍读到半新半旧（44 章） |
| 以为 SHARP 省带宽 | L4 | 大消息上收益 1.0x，量反而更大（40 章） |
| 以为"更好的负载均衡"总更好 | L4 | 小包 + 按包喷洒 = 乱序风暴（45 章） |

十条里有七条是**跨层混淆**：把 L1 的机制当 L2 的保证，把 L4 的手段当 L2 的效果。
这正是为什么需要"六层"这个模型——它让你在听到一个论断时，先问"这是哪一层在起作用"。

---

## 一页速记

```text
一条消息 = 6 段（L0）
  填 WQE -> doorbell -> 取 WQE -> 上线 -> 对端写内存 -> 产 CQE
  1 KB @ 100G: 线上 81.9 ns / 总 1582 ns = 5.2%  -> 小消息被固定成本支配

三个不变量（54 题的生成器）
  1) 每段都有固定成本          -> 问"为什么慢"
  2) 资源可按 N 或按字节记账   -> 问"为什么贵"
  3) 每个更强的保证都要付费     -> 问"A/B 怎么选"

六层与各自的锚点
  L1 对象   PD/MR/QP/CQ/AH     -> 34 章 连接状态账本（ICM vs 片上 cache）
  L2 语义   SEND/WRITE/READ/ATOMIC -> 43 章（MTU/inline/imm/UD）+ 44 章（背压）
  L3 可靠   PSN/ACK/go-back-N  -> 42 章（放大系数 = 1 + p(W+1)/2）
  L4 网络   credit/PFC/DCQCN   -> 35 章（headroom、ECN 阈值）+ 40 章（拓扑）
  L5 工程   verbs/排障/GPU/存储 -> 36-39、41、45 章

四条实践判据
  CQE 序列里先找第一个非 flush 错误
  WRITE 必须自己做 credit（除非你能接受静默覆盖）
  丢包/乱序的代价都按窗口放大 -> 生产坚持无损
  每个"更强"都是换资源 -> 问换的是哪个（带宽/内存/CPU/延迟）
```

---

# 怎么用这份笔记

```text
第一遍：只看"全图"和 L0，把六段路径背下来。这是其余五层的坐标系。
第二遍：按 L1→L5 顺序读，每读完一层，合上笔记口述该层的 3 个核心机制。
第三遍：只看"54 题定位表"，遮住"锚点"列，自己说。
第四遍：做自测。答不出的题，回到对应层的正文，而不是回 32 章题库。
```

[`33_rdma_cost_model.py`](./33_rdma_cost_model.py) 可以改参数再跑：
把 `FIXED_NS` 调到 800 看乐观情况，把 `LINKS` 加上 `"25G"` 看低带宽下的拐点，
都能帮你在面试里现场估算，而不是背结论。
