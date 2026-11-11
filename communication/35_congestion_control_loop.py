#!/usr/bin/env python3
"""DCQCN + PFC 参数闭环模拟：把 35 章的四张表算出来。

    python3 35_congestion_control_loop.py

重要边界：这是**合成演示**，不是实测。它复现 DCQCN 的控制回路结构，
用来回答"参数调错会以什么形状出问题"，不回答"某型号交换机该填多少"。
真实参数必须在现场用 ECN/pause 计数标定。

常量标记：
  [规范]  来自 802.1Qbb / IBTA / DCQCN 论文
  [假设]  取量级；改绝对值不改结论
  [锚点]  来自 33/34 章的既有结论
"""

# ---------------------------------------------------------------- [规范]
QUANTA_BITS = 512.0          # 1 pause quanta = 512 bit times
MAX_QUANTA = 65535           # pause 帧长度字段上限
MTU_B = 4096

# ---------------------------------------------------------------- [假设]
LINK_GBPS = 100.0
BYTES_PER_US = LINK_GBPS * 1e9 / 8 / 1e6     # 12500 B/us
RTT_US = 5.0                                  # BDP = 62.5 KB
BUFFER_B = 2 << 20                            # 每端口 2 MB 共享缓冲
CNP_INTERVAL_US = 50.0                        # 接收端 CNP 限速（每流）
RATE_INCREASE_US = 55.0                       # 发送端加性增定时器
RAI_BYTES_US = 250.0                          # 每次加性增幅度
G = 1.0 / 256                                 # alpha 平滑系数
ALPHA_MIN = 1.0 / 256
R_MIN = 200.0                                 # 约 1.6 Gbps
N_SENDERS = 2                                 # 都往同一个出口打，起始各 1/2 线速


class Config:
    def __init__(self, name, kmin_kb, kmax_kb, pmax=0.10, senders=N_SENDERS, xoff_kb=1536):
        self.name = name
        self.kmin = kmin_kb * 1024
        self.kmax = kmax_kb * 1024
        self.pmax = pmax
        self.senders = senders
        self.xoff = xoff_kb * 1024      # 高于它就发 pause
        self.xon = self.xoff // 4       # 低于它才放开（迟滞）

    def ecn_prob(self, q):
        """[规范] DCQCN 标记函数：Kmin~Kmax 线性升到 Pmax。"""
        if q <= self.kmin:
            return 0.0
        if q >= self.kmax:
            return self.pmax
        return self.pmax * (q - self.kmin) / (self.kmax - self.kmin)


def simulate(cfg, duration_us=4000.0, dt_us=1.0, seed=7):
    import random

    rng = random.Random(seed)
    start_rate = BYTES_PER_US / cfg.senders
    q = 0.0
    rate = [start_rate] * cfg.senders
    alpha = [ALPHA_MIN] * cfg.senders
    last_cnp = [-1e9] * cfg.senders
    last_inc = [0.0] * cfg.senders
    latched = [False] * cfg.senders

    serviced = offered = dropped = marked = 0.0
    paused_us = 0.0
    qmax = 0.0
    qsum = 0.0
    trace = []

    for step in range(int(duration_us / dt_us)):
        t = step * dt_us

        inject = sum(0.0 if latched[i] else rate[i] for i in range(cfg.senders)) * dt_us
        offered += inject

        q += inject
        out = min(q, BYTES_PER_US * dt_us)
        q -= out
        serviced += out

        # 缓冲溢出才算丢；xoff 只是发 pause 的门限
        if q > BUFFER_B:
            drop = q - BUFFER_B
            q = BUFFER_B
            dropped += drop

        # ECN：按当前队列长度算标记概率
        p = cfg.ecn_prob(q)
        ce_bytes = p * out
        marked += ce_bytes
        qmax = max(qmax, q)
        qsum += q

        # 接收端回 CNP（限速）-> 发送端按 alpha 降速
        for i in range(cfg.senders):
            got_cnp = ce_bytes > 0 and rng.random() < p
            if got_cnp and (t - last_cnp[i]) >= CNP_INTERVAL_US:
                last_cnp[i] = t
                alpha[i] = alpha[i] * (1 - G) + G
                rate[i] = max(R_MIN, rate[i] * (1 - alpha[i] / 2))
            elif (t - last_inc[i]) >= RATE_INCREASE_US and (t - last_cnp[i]) > RATE_INCREASE_US:
                last_inc[i] = t
                rate[i] = min(BYTES_PER_US, rate[i] + RAI_BYTES_US * 8 * G)
                alpha[i] = max(ALPHA_MIN, alpha[i] * (1 - G))

        # PFC 闸门
        for i in range(cfg.senders):
            if q > cfg.xoff:
                latched[i] = True
            elif q < cfg.xon:
                latched[i] = False
        if any(latched):
            paused_us += dt_us

        if step % 200 == 0:
            trace.append((t, q, sum(rate) * 8 / 1e3))

    dur_s = duration_us / 1e6
    return {
        "name": cfg.name,
        "goodput_gbps": serviced * 8 / 1e9 / dur_s,
        "qmax_kb": qmax / 1024,
        "qavg_kb": qsum / (duration_us / dt_us) / 1024,
        "dropped_kb": dropped / 1024,
        "marked_mb": marked / 1e6,
        "paused_pct": paused_us / duration_us,
        "trace": trace,
    }


CONFIGS = [
    Config("极限低 8/16K P.6", 8, 16, pmax=0.6),
    Config("过低 16/64K", 16, 64),
    Config("适中 128/768K", 128, 768),
    Config("过高 1536/4096K", 1536, 4096),
]


def table_configs():
    print("== 1. 只改 ECN 门限，四种配置的闭环结果 ==")
    print(
        f"   链路 {LINK_GBPS:.0f} Gbps，{N_SENDERS} 个 DCQCN 发送端，"
        f"缓冲 {BUFFER_B/1e6:.0f} MB，xoff 1.5 MB"
    )
    print(f"\n{'配置':>18}{'goodput':>10}{'队列峰值':>10}{'队列均值':>10}{'丢弃':>8}{'标记':>8}{'PFC暂停':>9}")
    rows = [simulate(c) for c in CONFIGS]
    for r in rows:
        print(
            f"{r['name']:>18}{r['goodput_gbps']:>8.1f}G"
            f"{r['qmax_kb']:>8.0f}K{r['qavg_kb']:>8.0f}K"
            f"{r['dropped_kb']:>7.0f}K{r['marked_mb']:>7.1f}M{r['paused_pct']:>9.1%}"
        )
    print("\n  极限低：标记最密、队列几乎为 0 -> 尾延迟最好，但每流速率被反复压")
    print("  适中  ：队列压在 Kmin~Kmax 之间，标记量刚好，几乎不靠 PFC")
    print("  高门限：不标记 -> 队列涨到 xoff -> 靠 PFC 兜，队列均值最高")
    print("\n  关键：goodput 这一列根本区分不出配置好坏，全都在 100G 附近。")
    print("  因为发送端是 2x 过载的，压到一半仍然能打满线速。")
    print("  真正的判据是三列：队列均值(尾延迟)、PFC 暂停率(抗压)、标记量(参数是否生效)。")
    print("\n  模型边界：本模型没有 short flow / incast，所以看不到 over-marking 的")
    print("  吞吐代价（短流被压速后没时间恢复）。真实集群里那一项才是 Pmax 的主要约束。")
    return rows


def table_trace(rows):
    print("\n== 2. 四种配置的队列轨迹（每 200 us 采样，单位 KB）==")
    print(f"{'t us':>8}" + "".join(f"{r['name']:>19}" for r in rows))
    for k in range(min(10, len(rows[0]["trace"]))):
        t = rows[0]["trace"][k][0]
        line = f"{t:>8.0f}"
        for r in rows:
            line += f"{r['trace'][k][1] / 1024:>19.0f}"
        print(line)
    print("  适中配置贴着 128~768K 振荡；过高配置一路爬到 1536K 才被 PFC 按停；")
    print("  极限低配置把队列压在 16K 以下——队列很短，代价全在发送端速率上。")


def table_pfc_quanta():
    print("\n== 3. PFC pause 的量级（1 quanta = 512 bit times）==")
    ns_per_q = QUANTA_BITS / (LINK_GBPS * 1e9) * 1e9
    print(f"   链路 {LINK_GBPS:.0f} Gbps：")
    print(f"   1 quanta    = {ns_per_q:.3f} ns")
    print(f"   最大 65535  = {MAX_QUANTA * ns_per_q / 1000:.1f} us")
    print("   quanta 是 bit 单位，与 MTU 无关；换链路速率时绝对时长跟着变。")
    print("   一条满值 pause 停住上游三百多微秒 —— PFC storm 的杀伤力来源。")


def table_headroom():
    print("\n== 4. PFC headroom：从发出 pause 到上游真停住，还在飞多少字节 ==")
    print("   headroom ≈ 2 × (单向延迟 × 线速率) + 2 × MTU")
    print(f"\n{'链路':>7}{'单向延迟 us':>14}{'headroom KB':>15}")
    for gbps in (100, 400):
        rate = gbps * 1e9 / 8 / 1e6
        for delay_us in (1.0, 2.0, 5.0):
            hr = 2 * delay_us * rate + 2 * MTU_B
            print(f"{gbps:>5}G{delay_us:>14.1f}{hr / 1024:>15.0f}")
    print("\n   400G + 2 us：单端口单优先级约 200 KB 专用 headroom。")
    print("   32 端口 × 8 优先级 = 51 MB 只用来防反压，还没算数据缓冲。")
    print("   所以高速率必须用共享/动态 headroom，按端口静态预留会吃光缓冲。")


def table_misconfig():
    print("\n== 5. 参数调错的症状对照 ==")
    rows = [
        ("Kmin 太低", "标记率长期偏高", "goodput 掉，速率被压平", "抬高 Kmin"),
        ("Kmax 太低", "概率瞬间到 Pmax", "速率剧烈振荡，p99 抖", "拉开 Kmax-Kmin"),
        ("两个都过高", "几乎不标记", "队列涨到 xoff，PFC 暂停率高", "下调门限/减小 xoff"),
        ("Pmax 太高", "过度降速", "带宽利用率低", "降到 0.05~0.1"),
        ("Pmax = 0", "ECN 完全失效", "只剩 PFC，等价于只有保险丝", "打开 ECN"),
        ("CNP 间隔太密", "连续打击发送端", "速率长期贴地", "拉开 CNP 间隔"),
        ("加性增太快", "恢复过猛", "周期性锯齿，误标增加", "减小 RAI/延长定时器"),
        ("忽略 RTT 差异", "长短流同参数", "长流被过度惩罚", "按 RTT 分档或加权"),
    ]
    print(f"{'错误':>14}{'第一现象':>16}   {'二阶症状':<26}修正")
    for a, b, c, d in rows:
        print(f"{a:>14}{b:>16}   {c:<26}{d}")


if __name__ == "__main__":
    rows = table_configs()
    table_trace(rows)
    table_pfc_quanta()
    table_headroom()
    table_misconfig()
