# 排障判定树：把七道题压成一张可执行的表

```text
本次讲解位置
章节：communication / RDMA
小节：L5 排障（33 章只给了树骨架，Q42–Q48 还是七个独立现象）
知识点：三层证据、计数器增量、pause 占比计算、wc.status 的根因与连带、假设排序
上次：拥塞控制参数闭环（35）——pause 占比的分子分母就来自那一章
下次：37 · GPUDirect 退化判定（"稳态变慢"与"抖动"是两种监控形态）```

## 为什么现在讲这个

[33 章](./33-rdma-six-layer-spine.md) 的 L5 给了排障树的骨架，但它停在了"查这个、查那个"。
[32 章](./32-rdma-interview-questions.md) 的 Q42–Q48 是七道彼此独立的现象题：
连不上、RTR 失败、收不到 CQE、延迟抖、PFC storm、hang 在 poll、计数器怎么看。

**它们是同一个决策过程的七个入口，不是七个知识点。** 现场出问题时你不会先想
"这属于 Q43 还是 Q44"，你只有一堆计数器和一个现象。所以真正要练的能力是：

> 给定现象和几个计数器，**排出假设的先后**，并知道下一步该测什么。

这一章把这件事做成可执行的：一棵三层判定树、一张计数器→病因映射、
一个能跑出排序假设的小程序。练完的标准是——给你一组陌生的计数器，
你能说出"先查什么、为什么不是另一个"。

---

## 心智模型：只有三层证据，一层比一层难拿

```text
第 1 层：包发出去了吗？        ← 网卡发送计数器
第 2 层：对端收到了吗？        ← 对端接收计数器 / 对端 QP 状态
第 3 层：应用消费了吗？        ← 只能靠应用栈、日志、心跳
```

**铁律：严格按 1 → 2 → 3 的顺序查，不许跳。**

原因很实在：第 1、2 层有硬件计数器，几秒钟就能测；第 3 层要复现、要加日志、
要读别人的代码。跳过前两层直接查应用，是本领域最常见的浪费——
你会花两小时读一个其实根本没收到包的接收端代码。

对应到题目上：

| 层 | 计数器/证据 | 能回答的题 |
| --- | --- | --- |
| 1 | `port_xmit_*`、`port_xmit_discards`、`symbol_error` | Q42、Q46、Q47 |
| 2 | `port_rcv_*`、`packet_seq_err`、`local_ack_timeout_err` | Q46、Q45 |
| 3 | `wc.status`、应用栈、心跳 | Q44、Q48、Q35 |

注意 `wc.status` 属于**第 3 层的入口但常常反映第 2 层的事实**：
`RETRY_EXC_ERR` 是应用看到的，但它描述的是"对端一直没应答"。这正是容易误判的地方。

## 一个必须先纠正的习惯：计数器要取增量

网卡计数器是**累计值，设备复位才清零**。看到 `port_rcv_errors = 2,000,000`
不能说明任何事——可能是机器开机以来累计的，其中 99.99% 发生在半年前的一次换线。

```bash
# 正确做法：间隔取两次，做差，再除以时间
perfquery -x <lid> > /tmp/c1.txt; sleep 10; perfquery -x <lid> > /tmp/c2.txt
# 逐项比较，只看增长的项
```

同理，`ibv_devinfo` 里的端口计数、交换机上的 `show interface counters`，
**全部要按增长速率看**。

---

## 判定树

```text
现象
│
├─ A. 根本连不上 / ib_write_bw 起不来
│   ├─ A1 ibv_devices 看不到设备 ──────────→ 驱动没加载、容器没透传 /dev/infiniband
│   ├─ A2 端口不是 ACTIVE ───────────────→ 线缆、光模块、交换机端口 disable
│   ├─ A3 port ACTIVE 但连不上 ──────────→ ① GID index 选错（RoCEv2 第一大坑）
│   │                                      ② MTU 不一致
│   │                                      ③ P_Key 不匹配
│   │                                      ④ 防火墙挡 UDP 4791
│   └─ A4 modify_qp 到 RTR 返 EINVAL ───→ 见下面「RTR 四类原因」
│
├─ B. 能收能发，但收不到完成
│   ├─ B1 完全没 CQE ───────────────────→ ① sq_sig_all=0 且没设 SIGNALED
│   │                                      ② poll 的不是这个 QP 的 CQ
│   │                                      ③ CQ 被别的线程消费了
│   │                                      ④ req_notify_cq 没在 poll 不到后重新 arm
│   └─ B2 有 CQE 但 status 是错误 ──────→ 见「wc.status 因果链」
│
├─ C. hang 在 poll
│   ├─ C1 对端没 post recv ─────────────→ SEND 必须有 recv WR；改 WRITE+WITH_IMM 可去掉依赖
│   ├─ C2 在等一个永不来的 WITH_IMM ────→ 对端地址/rkey 错，写失败但没人看 CQE
│   └─ C3 两个线程抢同一个 CQ ──────────→ 分片或加锁
│
├─ D. 带宽上不去
│   ├─ D1 消息小 ───────────────────────→ 固定成本主导（33 章：1 KB 只有 5% 线上时间）
│   ├─ D2 单 QP ────────────────────────→ 消息率天花板 1/(固定成本+线上时间)
│   ├─ D3 PCIe 代际不够 ────────────────→ 200G 需要 25 GB/s，Gen3 x16 只有 12~13
│   └─ D4 多个 QP 但仍不满 ─────────────→ 看 completion scope、跨 NUMA、CQ moderation
│
├─ E. 延迟抖动
│   ├─ E1 三分段定位 ───────────────────→ 同机 loopback → 同交换机 → 跨交换机
│   ├─ E2 pause 占比高 ─────────────────→ 见「pause 占比」算法
│   ├─ E3 ECN 误标 ─────────────────────→ 门限过低；队列均值被压得过低
│   └─ E4 主机侧 ───────────────────────→ 中断、CQ moderation、跨 NUMA、CPU 抢占
│
└─ F. 间歇性重传
    ├─ F1 symbol_error 增长 ─────────────→ 物理层：线缆/光模块/连接器
    ├─ F2 packet_seq_err 增长 ──────────→ 真丢包：PFC 漏配、ECN 误标、buffer 不足
    └─ F3 只有 ack_timeout ─────────────→ 反向路径问题，或对端 QP 已经没了
```

### RTR 四类原因（Q43 的完整版）

`modify_qp` 到 RTR 返回 `EINVAL`，几乎全部是"字段与状态不匹配"，不是硬件坏：

| 原因 | 怎么确认 |
| --- | --- |
| GID index 错 | `show_gids`，看 IPv4/IPv6/v1/v2 各占哪个索引 |
| `path_mtu` 超过设备或对端能力 | 对比 `ibv_devinfo` 的 `max_mtu` 与对端 |
| `dest_qp_num` / P_Key 不对 | 打印双方握手值 |
| 端口未 ACTIVE | `ibstat` |

顺带一条 33 章讲过的：`qp_access_flags` 要在 **INIT** 填，填到 RTR 就 EINVAL。
这是"字段与状态不匹配"里最隐蔽的一种，因为它在文档里只出现一次。

---

## wc.status 因果链：区分根因与连带

这是本章第二个核心。`wc.status` 有十几个值，但**只有四类因果**：

```text
① 本地参数错        LOC_LEN_ERR / LOC_PROT_ERR / LOC_ACCESS_ERR
                    你交给网卡的东西不合法（越界、跨 PD、权限不够）

② 远端拒绝          REM_ACCESS_ERR / REM_INV_REQ_ERR
                    对端认为请求非法（rkey 错、DC key 错、QP 状态不对）

③ 传输失败          RETRY_EXC_ERR / RNR_RETRY_EXC_ERR
                    对端不应答（QP 没了/链路断）或没缓冲（recv WR 用完）

④ 连带清空          WR_FLUSH_ERR
                    QP 已 ERROR，剩下的 WR 无辜被清
```

**排障铁律：④ 永远不是根因。** 看到一批 `WR_FLUSH_ERR`，
要在 CQE 序列里**往前找第一个非 flush 的错误**。只盯 flush 去查那个 WR 的参数，
会永远查不出来——那个 WR 本身可能完全合法。

```text
CQE 时间顺序：
  ...  RETRY_EXC_ERR  ← 根因在这里（QP 因此进 ERROR）
       WR_FLUSH_ERR   ← 连带，忽略
       WR_FLUSH_ERR   ← 连带，忽略
```

还有一个 DC 特有的坑（[34 章](./34-rdma-connection-state-ledger.md) 提过）：
DC 有两套凭证，`DC key` 管"能不能进这个 DCT"，`rkey` 管"能不能写这段内存"。
两者报错长得几乎一样，都是 `REM_ACCESS_ERR` 一类的访问错误。
**连接建好了但每笔都报访问错**，先查 DC key，别去查 MTU。

---

## pause 占比：把 pause 帧计数变成可判断的数字

只看到 `pause frames: 300/s` 没法判断严重程度，因为 pause 有时长。
用 [35 章](./35-rdma-congestion-control-loop.md) 的 quanta 换算：

```text
单条 pause 时长 = quanta × 512 bit / 线速率
pause 占用比例  = 每秒条数 × 单条时长
```

代入本仓库的示例场景（100 Gbps，300 条/s，每条 60000 quanta）：

```text
单条 = 60000 × 512 / 100e9 s = 307.2 us
占比 = 300 × 307.2 us / 1 s   = 92.16 ms/s = 9.2%
```

**9.2% 的时间这条链路是被按停的。** 换算成尾延迟：任何跨这条链路的消息，
有约 1/11 的概率撞上一次最多 307 us 的停顿。这就是"带宽正常、p99 毫秒级抖动"的来源。

经验线（写在判定器里）：

| 占比 | 判断 |
| --- | --- |
| < 1% | 正常波动，不用管 |
| 1% ~ 5% | 值得看一眼，找是否有慢端口 |
| > 5% | 这条链路基本靠 pause 活着，必须逐跳查 |

**注意"带宽正常"完全不能作为 PFC 健康的证据**，因为 pause 是短时按停，
在 1 秒尺度上对 goodput 的影响很小——和 [35 章](./35-rdma-congestion-control-loop.md)
"goodput 区分不出配置好坏"是同一件事。

---

## 判定器：把上面所有规则写成代码

[`36_diagnose.py`](./36_diagnose.py) 是一个约 200 行的规则引擎。输入是一次观察，输出是
**排序后的假设**，每条都带证据和下一步命令。

核心结构：

```python
@dataclass
class Obs:
    symptom: str = ""
    port_state: str = "ACTIVE"
    symbol_err: int = 0            # 增量，不是累计值
    pkt_seq_err: int = 0
    ack_timeout: int = 0
    xmit_discards: int = 0
    pause_frames_per_s: float = 0.0
    pause_max_quanta: int = 0
    ecn_marked_pct: float = 0.0
    wc_status: str = ""
    pending_recv: int = -1         # -1 = 未知
    threads_polling_cq: int = 1
    msg_size_b: int = 0
    link_gbps: float = 100.0
    goodput_gbps: float = 0.0


def _pause_time_pct(o):
    """pause 占用比例 = 每秒条数 × 单条 quanta × 512 bit / 线速率。"""
    ns_per_quanta = 512.0 / (o.link_gbps * 1e9) * 1e9
    return (o.pause_frames_per_s * o.pause_max_quanta * ns_per_quanta / 1e9) * 100


def diagnose(o):
    h = []
    # 物理层
    if o.symbol_err > 0 or o.rcv_err > 0:
        h.append(Hypothesis(0, "L0 物理", "物理层误码（线缆/光模块/连接器）",
                            f"symbol_error={o.symbol_err}", "perfquery -x；换线/换模块", 95))
    # 拥塞：pause 占比
    pct = _pause_time_pct(o)
    if pct > PAUSE_ALERT_PCT:
        h.append(Hypothesis(0, "L4 网络", "PFC 反压占比过高",
                            f"300 条/s × 60000 quanta ≈ 停顿 {pct:.1f}% 时间",
                            "逐跳查 pause 源", 88))
    # ... 其余规则见文件
    h.sort(key=lambda x: -x.score)
    return h
```

它的三个设计决定值得注意：

1. **每个假设带 score，输出排序而不是列表。** 现场要的是"先查哪个"，不是"可能是这些"。
2. **每个假设带 next_cmd。** 规则命中不等于确诊，只代表下一步测什么。
3. **未知字段用 `-1` 而不是 `0`。** `pending_recv=0`（确实没投）和
   `pending_recv=-1`（没测）是完全不同的结论。

跑起来：

```bash
cd ~/Downloads/learning_notes
python3 communication/36_diagnose.py             # 四个场景
python3 communication/36_diagnose.py --demo pfc  # 单个场景
```

四个场景的实际输出：

```text
场景：SEND 卡住，poll 拿到错误
  [1] (L2 语义) 接收端 recv WR 已耗尽
      证据：pending_recv=0，SEND 必然 RNR
      下一步：补投 recv WR；或改用 WRITE + WITH_IMM 去掉 recv 依赖
  [2] (L3 可靠性) 对端没有可用 recv WR（Receiver Not Ready）
      证据：wc.status=IBV_WC_RNR_RETRY_EXC_ERR

场景：延迟抖动，带宽正常
  [1] (L4 网络) PFC 反压占比过高
      证据：300 条/s × 60000 quanta ≈ 停顿 9.2% 时间
      下一步：逐跳查 pause 源：先查本端口 pause 计数，再往上游/下游找慢端口

场景：带宽只有 5 Gbps
  [1] (L5 工程) 小消息固定成本主导，不是硬件故障
      证据：1024 B 消息，有效带宽 5%
      下一步：算 1/(固定成本+线上时间)；用 inline / doorbell batching / 多 QP

场景：间歇性重传，无拥塞
  [1] (L0 物理) 物理层误码（线缆/光模块/连接器）
      证据：symbol_error=42, port_rcv_errors=17
  [2] (L3 可靠性) 丢包/乱序，触发 go-back-N 重传
      证据：packet_seq_err=31
```

注意第三个场景：`ib_write_bw` 只跑出 5.2 Gbps 时，判定器给出的第一假设是
**"这不是故障"**。这正是 33 章 L0 的推论——1 KB 消息在 100G 上有效带宽就是 5% 左右。
把它当成网卡故障去换硬件，是新人最常见的浪费。

---

## 常见误判，以及它为什么错

| 误判 | 为什么错 | 正确做法 |
| --- | --- | --- |
| 用累计计数器判断 | 设备复位才清零，包含几个月前的历史 | 间隔取两次做差 |
| 看到 `WR_FLUSH_ERR` 就查那个 WR | flush 是连带，根因在前面的 CQE | 找第一个非 flush 错误 |
| "带宽满 = 网络健康" | pause 在 1 s 尺度上不影响 goodput | 看 pause 占比与队列均值 |
| "开了 PFC = 无损" | 还要 headroom、优先级、死锁防护（35 章） | 算 headroom，查 watchdog |
| 小消息跑不满就换网卡 | 固定成本主导，不是硬件 | 先算 1/(固定+线上时间) |
| 跳过计数器直接读应用代码 | 前两层几秒能测，第三层要几小时 | 严格 1→2→3 |
| GID index 随便填 | RoCEv2 下 IPv4/IPv6/v1/v2 各占一个索引 | `show_gids` 后显式指定 |
| 只看 CQE 数量不看 status | 错误完成也是完成 | 每个 CQE 都检查 `wc.status` |

最后一条特别值得记：**`ibv_poll_cq` 返回 1 不代表成功。** 很多示例代码只判断
`polled > 0` 就当完成，把 `wc.status` 丢掉，于是错误被静默吞掉——
这正是 Q48「hang 在等一个永不来的 WITH_IMM」的成因。

---

## 模型边界

判定器是规则而不是学习模型，所以：

- **只在单点网络里可靠**：多跳 PFC 反压需要逐跳采集，本工具只看单端口
- **不覆盖软件 bug**：内存生命周期、线程竞争、错误处理缺失，规则看不出来
- **score 是经验权重**，不是概率，只用来排序
- **场景是构造的**：真实排障要先补齐 `Obs` 里那些 `-1` 字段

真实用法：把 `Obs` 填成现场实测值，跑一遍拿到排序，按 `next_cmd` 采下一轮数据，
再填回去。**迭代三轮通常就能收敛**——这也是为什么每条假设都要带命令。

---

## 自测

1. 三层证据是什么？为什么必须按顺序？哪一层没有计数器？
2. `port_rcv_errors` 显示 200 万，能说明什么问题？应该怎么测？
3. 一批 CQE 里 1 条 `RETRY_EXC_ERR` 加 20 条 `WR_FLUSH_ERR`，根因是哪个？
4. 100G 链路上每秒 300 条 pause、每条 60000 quanta，停顿占比多少？意味着什么？
5. `ib_write_bw -s 1024` 只跑出 5 Gbps，第一假设应该是什么？
6. DC 场景下"连接建好了但每笔都 REM_ACCESS_ERR"，先查什么？

<details>
<summary>答案</summary>

1. ① 包发出去了吗（发送计数器）② 对端收到了吗（接收计数器/对端 QP 状态）
   ③ 应用消费了吗（应用栈/日志）。必须按序因为前两层几秒可测、第三层要数小时。
   **第三层没有硬件计数器。**
2. **说明不了任何事**——累计值可能包含半年前的换线。要间隔取两次做差，
   除以时间得到增长速率，再看是否随流量增长。
3. 根因是第 1 条 `RETRY_EXC_ERR`（对端不应答，QP 因此进 ERROR）。
   后面 20 条 flush 是连带，那个 WR 本身无辜。
4. `60000 × 512 / 100e9 = 307.2 us`，`300 × 307.2 us = 92.16 ms/s = 9.2%`。
   意味着这条链路约 1/11 的时间被按停，任何跨它的消息都可能撞上最多 307 us 的停顿——
   这是"带宽正常但 p99 毫秒级抖动"的典型成因。
5. **"这不是故障"**。1 KB 消息在 100G 上线上时间只占约 5%（33 章），
   有效带宽 5 Gbps 属于正常。应先算固定成本，再考虑 inline / batching / 多 QP。
6. **DC key**（AV 里的 64 位凭证），不是 MTU、不是链路。DC 有两套凭证，
   连接通但每笔访问错误，说明 DCT 能到、但 DC key 不被接受。

</details>

---

## 一页速记

```text
三层证据，严格按序:
  1 发出去了吗   发送计数器        秒级
  2 对端收到了吗 接收计数器/QP 状态 秒级
  3 应用消费了吗 栈/日志/心跳       小时级

计数器:
  永远取增量，累计值毫无意义
  perfquery -x 取两次做差

四类 wc.status 因果:
  本地参数错 / 远端拒绝 / 传输失败 / 连带清空
  WR_FLUSH_ERR 永远不是根因 -> 往前找第一个非 flush 错误

pause 占比:
  单条 = quanta x 512 bit / 线速率
  占比 = 条数/s x 单条时长
  100G, 300/s, 60000q -> 9.2%  <- 带宽仍然"正常"

三个最容易误判:
  带宽满 != 网络健康（看 pause 占比与队列均值）
  poll 到 CQE != 成功（必须查 wc.status）
  跑不满 != 硬件坏（先算固定成本）
```
