#!/usr/bin/env python3
"""流控与背压账本：为什么 WRITE 会静默覆盖数据，而 SEND 不会。

    python3 44_flow_control_backpressure.py

四个问题：
  1. 接收端停顿一下，无流控的 WRITE 会覆盖多少数据？
  2. 要吸收一次停顿，缓冲得挖多大？（答案会让人不舒服）
  3. 用 credit 替代缓冲，代价是什么？
  4. 链路 credit / PFC / DCQCN / 应用 credit 各自保护谁？

常量标记：
  [规范]  IBTA 语义
  [假设]  速率与容量（会改绝对值，不改结论）
"""

# ---------------------------------------------------------------- [假设]
SLOT_B = 4096                      # 每个 slot 4 KB（典型 NVMe-oF / RPC 缓冲）
LINK_GBPS = 400.0
BYTES_PER_NS = LINK_GBPS * 1e9 / 8 / 1e9        # 50 B/ns
WIRE_NS_PER_MSG = SLOT_B / BYTES_PER_NS         # 81.9 ns
LINE_RATE_MPS = 1e3 / WIRE_NS_PER_MSG           # 每微秒的消息数 ≈ 12.2
RTT_NS = 3000.0
DRAIN_MULT = 2.5        # 停顿结束后，消费者以 2.5x 线速排空积压


# ---------------------------------------------------------------- 环
class Ring:
    """固定接收环。写指针无界推进，读指针按消费推进。

    语义：消息编号 0,1,2,...；consumed = 已消费条数，
    所以 seq >= consumed 的消息都还没被消费。
    """

    def __init__(self, slots):
        self.slots = slots
        self.written = 0
        self.consumed = 0
        self.overwrites = 0
        self.slot_seq = [None] * slots

    def write(self, seq):
        idx = self.written % self.slots
        cur = self.slot_seq[idx]
        if cur is not None and cur >= self.consumed:
            self.overwrites += 1      # 覆盖了没被消费的数据
        self.slot_seq[idx] = seq
        self.written += 1

    def consume(self, n):
        took = 0
        for _ in range(n):
            idx = self.consumed % self.slots
            cur = self.slot_seq[idx]
            if cur is not None and cur >= self.consumed:
                # 正常前进；若发生过覆盖（cur > consumed）则跳到最新，重新同步
                self.consumed = cur + 1
                took += 1
            else:
                break
        return took

    def backlog(self):
        return self.written - self.consumed


def simulate_no_credit(slots, stall_us, dur_us=None, stall_at=100):
    """生产恒为线速（WRITE 不知道对端状态）；消费在停顿窗口暂停，
    其余时间以 DRAIN_MULT x 线速排空积压（停顿结束后能追赶）。"""
    if dur_us is None:
        dur_us = stall_us + 4000        # 保证停顿结束后有足够时间排空
    ring = Ring(slots)
    p_acc = c_acc = 0.0
    seq = 0
    for t in range(dur_us):
        p_acc += LINE_RATE_MPS
        n = int(p_acc)
        p_acc -= n
        for _ in range(n):
            ring.write(seq)
            seq += 1
        stalling = stall_at <= t < stall_at + stall_us
        if not stalling:
            c_acc += DRAIN_MULT * LINE_RATE_MPS
            n = int(c_acc)
            c_acc -= n
            took = ring.consume(n)
            if took < n:
                c_acc = 0.0      # 排空了就别攒额度
    return ring, seq


def table_no_credit():
    print("== 1. 接收端停顿一下，无流控的 WRITE 会覆盖多少 ==")
    print(f"   slot {SLOT_B} B；链路 {LINK_GBPS:.0f} Gbps -> 线速 "
          f"{LINE_RATE_MPS:.1f} msg/us")
    print("   生产恒为线速（WRITE 无法感知对端），消费在 100us 处停顿 N us\n")
    print(f"{'停顿 us':>9}{'停顿期间写入':>13}" + "".join(
        f"{f'环深{s}':>13}" for s in (256, 1024, 4096, 16384)))
    for stall in (10, 100, 1000, 10000):
        burst = int(LINE_RATE_MPS * stall)
        row = f"{stall:>9}{burst:>13}"
        for slots in (256, 1024, 4096, 16384):
            ring, total = simulate_no_credit(slots, stall)
            row += f"{ring.overwrites:>13}"
        print(row)
    print("\n  **覆盖数随停顿增长、随环深下降；只有环深 ≥ 停顿期间写入量时才归零。**")
    print("  被覆盖的消息，发送端拿到的是 IBV_WC_SUCCESS：网卡确实写成功了，")
    print("  只是写到了别人还没读的格子上。应用看到「偶发数据错乱」。")
    print("  这就是 WRITE「零背压」的真实代价：不是慢，是静默错。")


def table_ring_sizing():
    print("\n== 2. 要吸收一次停顿，缓冲得挖多大 ==")
    print("   所需条数 = 线速(msg/us) x 停顿时长(us)；容量 = 条数 x slot")
    print(f"\n{'停顿':>10}{'所需条数':>12}{'所需容量':>13}{'等价于':>18}")
    for stall, label in ((10, "10 us"), (100, "100 us"), (1000, "1 ms"),
                         (10000, "10 ms")):
        n = int(LINE_RATE_MPS * stall)
        mib = n * SLOT_B / (1 << 20)
        equiv = f"{LINK_GBPS:.0f}G x {stall / 1000:.2f} ms"
        print(f"{label:>10}{n:>12}{mib:>12.1f}M{equiv:>18}")
    n1 = int(LINE_RATE_MPS * 1000)
    mib1 = n1 * SLOT_B / (1 << 20)
    mib10 = int(LINE_RATE_MPS * 10000) * SLOT_B / (1 << 20)
    print(f"\n  1 ms 停顿 @ 400G 需要 {mib1:.1f} MiB 缓冲；10 ms 需要 {mib10:.1f} MiB。")
    print(f"  而且这是**每个连接**的量 —— 1000 条连接就是 "
          f"{mib1 * 1000 / 1024:.0f} GiB。")
    print("  所以「靠加大缓冲吸收抖动」在高速网络上根本不成立：")
    print("  要么限速（credit），要么让发送端自己慢下来（DCQCN），二选一。")


def simulate_credit(slots, stall_us, dur_us=None, stall_at=100, rtt_us=None):
    """credit 归还有 RTT 延迟；发送端没有 credit 就停。"""
    if rtt_us is None:
        rtt_us = RTT_NS / 1000
    if dur_us is None:
        dur_us = stall_us + 4000
    ring = Ring(slots)
    credit = slots
    returns = []
    p_acc = c_acc = 0.0
    seq = 0
    blocked_us = 0
    for t in range(dur_us):
        while returns and returns[0][0] <= t:
            _, n = returns.pop(0)
            credit = min(credit + n, slots)
        p_acc += LINE_RATE_MPS
        want = int(p_acc)
        p_acc -= want
        sent = 0
        for _ in range(want):
            if credit <= 0:
                blocked_us += 1
                break
            credit -= 1
            ring.write(seq)
            seq += 1
            sent += 1
        # 被阻塞的量不累积欠账：发送端是速率受限的，错过的就是错过了
        stalling = stall_at <= t < stall_at + stall_us
        if not stalling:
            c_acc += DRAIN_MULT * LINE_RATE_MPS
            n = int(c_acc)
            c_acc -= n
            took = ring.consume(n)
            if took < n:
                c_acc = 0.0
            if took:
                returns.append((t + rtt_us, took))
    return ring, seq, blocked_us


def table_credit():
    print("\n== 3. 用 credit 替代缓冲：覆盖归零，代价是被限速 ==")
    print(f"   credit 归还延迟 = RTT {RTT_NS / 1000:.0f} us；停顿 1000 us\n")
    print(f"{'环深':>6}{'写入条数':>10}{'被覆盖':>9}{'阻塞 us':>10}"
          f"{'有效 msg/us':>13}{'缓冲容量':>11}")
    for slots in (64, 256, 1024, 4096):
        ring, total, blocked = simulate_credit(slots, 1000)
        eff = total / 4000
        mib = slots * SLOT_B / (1 << 20)
        print(f"{slots:>6}{total:>10}{ring.overwrites:>9}{blocked:>10}"
              f"{eff:>13.1f}{mib:>10.1f}M")
    print("\n  覆盖全部为 0 —— credit 解决的是**正确性**，不是吞吐。")
    print("  环深决定「停顿期间能吸收多少突发」：")
    print("    64 条（0.2 MiB）  -> 阻塞 ~998 us（几乎整个停顿期都在等）")
    print("    4096 条（16 MiB） -> 阻塞 ~668 us（先吃掉 16 MiB 突发再等）")
    print("  但欠账最终都要还：停顿 1000 us 造成的生产缺口，只能靠之后的追赶补回。")
    print("  **credit 不能让你超过消费者的处理能力。**")
    print("  它把「覆盖数据」换成了「发送端等待」—— 一个诚实的失败换一个静默的错误。")


def table_layers():
    print("\n== 4. 四层流控各保护谁 ==")
    print("""
  ┌──────────────────────────────────────────────────────────────┐
  │ 第 4 层  应用 credit（semantic）                             │
  │   保护：接收端应用缓冲不溢出                                  │
  │   机制：recv WR 计数 / 显式 credit 消息                        │
  │   唯一能理解「消费者忙」的层 —— 前三层都不懂业务语义           │
  ├──────────────────────────────────────────────────────────────┤
  │ 第 3 层  DCQCN / ECN（端到端速率）                            │
  │   保护：网络平均队列，避免持续拥塞                             │
  │   机制：交换机打 ECN -> CNP -> 发送端降速、加性恢复            │
  │   时间尺度：毫秒；不保证零丢包（35 章）                        │
  ├──────────────────────────────────────────────────────────────┤
  │ 第 2 层  PFC（逐跳暂停）                                     │
  │   保护：交换机/网卡入口缓冲不溢出                              │
  │   机制：802.1Qbb pause frame，按优先级                         │
  │   时间尺度：微秒；能保证零丢包，但会反压（35 章）              │
  ├──────────────────────────────────────────────────────────────┤
  │ 第 1 层  链路 credit（硬件）                                  │
  │   保护：下一跳的接收缓冲（IB 原生）                            │
  │   机制：每 VL 一个信用计数器，收一个包还一个                    │
  │   时间尺度：纳秒；逐跳，端到端不感知                           │
  └──────────────────────────────────────────────────────────────┘

  关键分界：1、2 是逐跳的（保网络），3、4 是端到端的（保端点）。
            1、3 靠「计数」，2 靠「信号」，4 靠「业务语义」。

  Q23 的答案：IB credit 与 PFC 的本质区别
    IB credit  收下才还信用 -> 发送端永不超过接收端容量 -> 天生无损、无需 headroom
    PFC        快满了才喊停 -> 有传播延迟     -> 必须按延迟算 headroom（35 章）
    两者都逐跳，但一个精确计数、一个有信号延迟。
""")


def table_send_vs_write():
    print("== 5. SEND 与 WRITE 的背压差异（Q53 的核心）==")
    print(f"{'':>10}{'SEND':>34}{'WRITE':>34}")
    rows = [
        ("接收端参与", "必须预投 recv WR", "完全不参与"),
        ("天然背压", "有：WR 用光就 RNR", "无：写到哪算哪"),
        ("环满时", "发送端阻塞或 RNR", "静默覆盖"),
        ("应用看到", "RNR_RETRY_EXC_ERR", "IBV_WC_SUCCESS"),
        ("谁保护缓冲", "网卡 + 应用", "只有应用自己"),
        ("适合", "请求/响应、需要边界", "大块单向写、已知地址"),
    ]
    for a, b, c in rows:
        print(f"{a:>10}{b:>34}{c:>34}")
    print("\n  **同一个接收环，SEND 安全，WRITE 不安全。**")
    print("  因为 SEND 的「写到哪里」由接收端投 WR 时决定 —— 它天然知道缓冲边界；")
    print("  WRITE 的「写到哪里」由发送端 rkey 决定 —— 接收端从未表态。")
    print("  所以用 WRITE 就必须自己做应用层 credit，这一步无法绕过。")


def table_design():
    print("\n== 6. 设计 RPC / KV 时的背压方案（Q49）==")
    print("""
  请求路径（小消息，要边界）
    客户端 -> 服务端：SEND（可配 inline，38 章约 14% 延迟收益）
    服务端预投 recv WR 池 = 天然并发上限
      WR 用光 -> 客户端收到 RNR 或自己限流
    最省事的一层背压：**队列深度本身就是限流器**

  数据路径（大块，要带宽）
    RDMA WRITE 到预注册的 slot ring
    必须配 credit：服务端消费完一个 slot 归还一个
    环深按「最坏停顿」算（第 2 张表），不是按平均
    发送端按 credit 发，不要按「我觉得对端该读完了」发

  失败与超时
    对端异常退出（Q35）：连接不会立刻断，靠心跳
    心跳用 SEND（有背压、有 CQE）；不要用 WRITE（无 CQE 可观测）
    RNR 是信号不是错误：用它区分「对端忙」和「对端死」

  经验值
    recv WR 池：16~64 条/连接（覆盖 RTT 内请求数，42 章 Little's law）
    data credit：按 2~10 ms 的最坏停顿折算（高速网络上会很大，见第 2 表）
    timeout：跨机架 14（67 ms）；同机架 8（1.05 ms）（42 章）
""")


if __name__ == "__main__":
    table_no_credit()
    table_ring_sizing()
    table_credit()
    table_layers()
    table_send_vs_write()
    table_design()
