#!/usr/bin/env python3
"""性能建模与容量规划（Q36-Q39 的实操版）。

    python3 46_perf_capacity_planner.py

四个问题：
  Q36 有效带宽为什么随消息变大而上升？给出曲线。
  Q37 想提高消息率，inline / batching / 多 QP 各能提多少？
  Q38 延迟下限是多少？瓶颈在哪一段？
  Q39 队列深度设多少？设小了会怎样（不会报错，只会慢）？

核心反推公式（Little's law 的逆用）：
    可达带宽 = 在途条数 x 消息大小 / RTT

常量标记：
  [规范]  协议定义
  [假设]  各段成本（量级假设，改绝对值不改排序）
"""

import math

# ---------------------------------------------------------------- [假设]
CPU_GHZ = 3.0
CYC_PER_NS = CPU_GHZ                    # 3 cycles/ns

# 一条消息的固定成本分解（单位 ns），合计 1500 ns = 1.5 us
SEGMENTS = [
    ("填 WQE（用户态）",       "CPU",  350),
    ("doorbell（MMIO 写）",    "PCIe", 150),
    ("网卡取 WQE（PCIe 读）",  "PCIe", 250),
    ("对端写内存（PCIe 写）",  "PCIe", 300),
    ("产 CQE + 用户态 poll",   "CPU",  450),
]
FIXED_NS = sum(s[2] for s in SEGMENTS)   # 1500

NIC_MSG_RATE_CAP = 150e6        # [假设] 单卡消息率上限
DEFAULT_MAX_SEND_WR = 128       # [规范] 常见默认值（不同厂商 128~256）

LINKS = {"100G": 100.0, "200G": 200.0, "400G": 400.0}


def wire_ns(size_b, gbps):
    return size_b * 8 / gbps


def msg_rate(fixed_ns, size_b, gbps):
    """单 QP 消息率（msg/s）。"""
    return 1e9 / (fixed_ns + wire_ns(size_b, gbps))


def efficiency(size_b, gbps):
    w = wire_ns(size_b, gbps)
    return w / (FIXED_NS + w)


# ---------------------------------------------------------------- Q36
def table_q36():
    print("== Q36. 有效带宽为什么会随消息变大而上升 ==")
    print(f"   每条消息固定成本 {FIXED_NS} ns（下面分解），线上时间 = size*8/带宽")
    print(f"   有效带宽占比 = 线上时间 / (固定成本 + 线上时间)\n")
    print(f"{'消息大小':>10}" + "".join(f"{k + ' 效率':>12}" for k in LINKS)
          + f"{'400G 有效':>12}{'消息率':>11}")
    for size, label in ((64, "64 B"), (256, "256 B"), (1024, "1 KB"),
                        (4096, "4 KB"), (16384, "16 KB"), (65536, "64 KB"),
                        (1048576, "1 MB")):
        row = f"{label:>10}"
        for gbps in LINKS.values():
            row += f"{efficiency(size, gbps):>11.1%}"
        eff400 = efficiency(size, 400.0)
        row += f"{400.0 * eff400:>11.1f}G"
        row += f"{msg_rate(FIXED_NS, size, 400.0) / 1e6:>10.2f}M"
        print(row)
    print("\n  两个直接结论：")
    print("    1) 1 KB @ 100G 效率只有 5.2% —— 「100G 网卡跑小消息只有几 Gbps」不是硬件坏，")
    print("       是固定成本主导。这个数字和 33 章一致。")
    eff1k_100 = efficiency(1024, 100.0)
    eff1k_400 = efficiency(1024, 400.0)
    print(f"    2) 同样的 1 KB，400G 的效率反而更低（{eff1k_400:.1%} vs {eff1k_100:.1%}）：")
    print("       带宽涨了 4 倍，固定成本一分没少 —— **链路越快，小消息越浪费**。")
    print("       所以高速链路必须靠「更大的消息」或「更低的固定成本」才不白买。")


def table_fixed_breakdown():
    print("\n== Q36 附. 固定成本花在哪 5 段 ==")
    print(f"{'段':>24}{'介质':>8}{'ns':>8}{'占比':>8}")
    for name, via, ns in SEGMENTS:
        print(f"{name:>24}{via:>8}{ns:>8}{ns / FIXED_NS:>7.1%}")
    print(f"{'合计':>24}{'':>8}{FIXED_NS:>8}{1.0:>7.1%}")
    print(f"\n  三段在 PCIe 上（{sum(n for _, v, n in SEGMENTS if v == 'PCIe')} ns"
          f" = {sum(n for _, v, n in SEGMENTS if v == 'PCIe') / FIXED_NS:.0%}）：")
    print("  这解释了为什么 GPUDirect 之外的所有优化都绕不开 PCIe，")
    print("  也解释了 inline 为什么有效 —— 它砍掉的是「网卡取 WQE 后再读 payload」这一次 PCIe 访问。")


# ---------------------------------------------------------------- Q37
def table_q37():
    print("\n== Q37. 提高消息率：三种手段各能提多少 ==")
    size, gbps = 1024, 100.0
    base = msg_rate(FIXED_NS, size, gbps)
    print(f"   基准：1 KB @ 100G，固定成本 {FIXED_NS} ns -> {base / 1e6:.3f} M msg/s")
    print(f"   目标：打满 100G 需要 {100e9 / 8 / size / 1e6:.2f} M msg/s\n")
    print(f"{'手段':>26}{'固定成本 ns':>13}{'msg/s':>11}{'相对提升':>10}"
          f"{'需要多少 QP':>13}")

    def report(name, fixed):
        r = 1e9 / (fixed + wire_ns(size, gbps))
        need = (100e9 / 8 / size) / r
        print(f"{name:>26}{fixed:>13.1f}{r / 1e6:>10.3f}M"
              f"{r / base:>9.2f}x{need:>13.1f}")

    report("基线（每消息一个 WR）", FIXED_NS)
    report("inline（省 WQE 回读 payload）", FIXED_NS - 250)
    report("doorbell batching x16（摊销 150）", FIXED_NS - 150 + 150 / 16)
    report("CQ moderation（摊销 450）", FIXED_NS - 450 + 450 / 16)
    combined = FIXED_NS - 250 - 150 - 450 + (150 + 450) / 16
    report("inline + 双 batching", combined)

    print("\n  读法：")
    print("    inline 砍 250 ns -> 单 QP 提 19%，但它依赖 payload，大消息无效")
    print("    只摊销 doorbell 只提 10% —— 150 ns 本来就不是大头")
    print("    摊销 CQE（CQ moderation）提 36%，因为它占固定成本的 30%")
    print("    三者叠加提 106%（2.06x），代价是**延迟变差**（等批满、等中断合并）")
    print("  最后一行是关键：**这些手段都不改变「单 QP 有上限」这件事**，只抬高上限。")
    need_qp = (100e9 / 8 / size) / (1e9 / (combined + wire_ns(size, gbps)))
    print(f"  即使叠满优化，打满 100G 仍需约 {math.ceil(need_qp)} 个 QP（1 KB 消息）。")


# ---------------------------------------------------------------- Q38
def table_q38():
    print("\n== Q38. 延迟下限：把 1 KB 消息拆到六段 ==")
    size, gbps, rtt_ns = 1024, 100.0, 2000.0
    w = wire_ns(size, gbps)
    print(f"   1 KB @ 100G，RTT {rtt_ns / 1000:.0f} us"
          f"（传播按 RTT/2 计）\n")
    items = [
        ("填 WQE（用户态）", 350),
        ("doorbell（MMIO 写）", 150),
        ("网卡取 WQE（PCIe 读）", 250),
        ("对端写内存（PCIe 写）", 300),
        ("产 CQE + 用户态 poll", 450),
        ("线上传输（1 KB）", w),
        ("传播 + 交换机（RTT/2）", rtt_ns / 2),
    ]
    total = sum(ns for _, ns in items)
    print(f"{'项':>26}{'ns':>10}{'占端到端':>10}")
    for name, ns in items:
        print(f"{name:>26}{ns:>10.1f}{ns / total:>9.1%}")
    print(f"{'合计':>26}{total:>10.1f}{1.0:>9.1%}")

    fixed = FIXED_NS
    print(f"\n  换算成 us：{total / 1000:.2f} us")
    print("  三条可操作的结论：")
    print(f"    1) 固定成本 {fixed} ns 占 {fixed / total:.0%}，比线上 {w:.0f} ns"
          f"（{w / total:.1%}）大一个数量级 ——")
    print("       所以降延迟的第一优先级是压固定成本，不是换更快的网卡。")
    print(f"    2) 传播占 {(rtt_ns / 2) / total:.0%}，是第二大项；")
    print("       跨机架多 2 us RTT 会再吃掉 40% 左右 —— 这就是「同机架优先」的量化理由。")
    print(f"    3) 接收侧 {450} ns + doorbell {150} ns 可用 busy polling 与"
          " batching 压掉一部分，代价是一个核与吞吐。")


# ---------------------------------------------------------------- Q39
def table_q39():
    print("\n== Q39. 队列深度：设小了不报错，只是永远达不到线速 ==")
    print("   反推公式：可达带宽 = 在途条数 x 消息大小 / RTT")
    print("   取 RTT = 5 us（跨机架）\n")
    rtt_ns = 5000.0
    print(f"{'深度':>6}{'1 KB':>10}{'4 KB':>10}{'64 KB':>10}"
          f"{'打满 400G 所需深度':>20}")
    for depth in (16, 64, 128, 256, 512, 1024):
        row = f"{depth:>6}"
        for size in (1024, 4096, 65536):
            bps = depth * size * 8 / (rtt_ns * 1e-9)
            row += f"{min(bps / 1e9, 400.0):>9.1f}G"
        need_1k = math.ceil(400e9 * rtt_ns * 1e-9 / 8 / 1024)
        row += f"{'—' if depth >= need_1k else f'1KB 需 {need_1k}':>20}"
        print(row)
    print("\n  读法：")
    print(f"    1 KB 打满 400G 需要 {math.ceil(400e9 * rtt_ns * 1e-9 / 8 / 1024)} 个在途；"
          "深度 128 只能跑到约 210G。")
    print("    而且**没有任何错误**：CQE 全 SUCCESS，计数器全干净，只是速率上不去。")
    print("    这就是为什么容量规划必须回到 Little's law，而不是「把深度调到最大」。")

    print(f"\n  默认坑：max_send_wr 常见默认 {DEFAULT_MAX_SEND_WR}")
    for size, label in ((1024, "1 KB"), (4096, "4 KB")):
        bps = DEFAULT_MAX_SEND_WR * size * 8 / (rtt_ns * 1e-9)
        capped = min(bps, 400e9)
        print(f"    {label} 消息 + 深度 {DEFAULT_MAX_SEND_WR} -> "
              f"{capped / 1e9:.1f} Gbps（400G 链路的 {capped / 400e9:.0%}）")
    print("\n  所以 400G 上必须：调大 max_send_wr，或用 DC（动态连接），或用多 QP 并行。")
    print("  这三者的共同点：都是**增加在途条数**。换的是内存/表项，买的是带宽。")


def table_depth_need():
    print("\n== Q39 附. 需要的深度怎么算（含安全系数）==")
    print("   深度 = BDP / 消息大小 x 安全系数；BDP = 带宽 x RTT")
    print(f"\n{'链路':>7}{'RTT us':>9}{'BDP KB':>10}{'1KB 所需':>11}"
          f"{'x2':>8}{'x4':>8}{'超过默认 128?':>15}")
    for link, gbps in LINKS.items():
        for rtt_us in (2.0, 5.0, 20.0):
            bdp_b = gbps * 1e9 / 8 * rtt_us * 1e-6
            need = bdp_b / 1024
            flag = "是" if need * 2 > DEFAULT_MAX_SEND_WR else "否"
            print(f"{link:>7}{rtt_us:>9.1f}{bdp_b / 1024:>10.1f}"
                  f"{need:>11.1f}{need * 2:>8.0f}{need * 4:>8.0f}{flag:>15}")
    print("\n  注意 400G / RTT 20 us（跨机房）需要 2000 个在途 ——")
    print("  这不是「调一个参数」能解决的，必须改架构（DC / 多 QP / 流水线）。")


# ---------------------------------------------------------------- 综合
def table_bottleneck():
    print("\n== 综合：给定配置，判断谁是瓶颈 ==")
    print("   四个上限取最小：线速 / 固定成本 / 在途深度 / 网卡消息率\n")

    def assess(size, gbps, rtt_ns, depth, qps):
        wire_bps = size * 8 / (FIXED_NS + wire_ns(size, gbps)) * 1e9 * qps
        depth_bps = depth * qps * size * 8 / (rtt_ns * 1e-9)
        # 150 M msg/s 是整卡上限，不随 QP 数放大
        nic_bps = NIC_MSG_RATE_CAP * size * 8
        link_bps = gbps * 1e9
        caps = {
            "线速": link_bps,
            "固定成本": wire_bps,
            "在途深度": depth_bps,
            "网卡消息率": nic_bps,
        }
        got = min(caps.values())
        # 固定成本上限永远略低于线速（固定成本不可能为零），
        # 所以差 1% 以内就认为「已经到线速，瓶颈在链路」
        bind = "线速" if got >= 0.99 * link_bps else min(caps, key=caps.get)
        return got, bind, caps

    print(f"{'场景':>34}{'可达':>10}{'占比':>8}   瓶颈")
    cases = [
        ("1 KB, depth128, 1QP, RTT 5us", 1024, 400.0, 5000.0, 128, 1),
        ("4 KB, depth8, 1QP, RTT 20us", 4096, 400.0, 20000.0, 8, 1),
        ("64 B, depth512, 512QP", 64, 400.0, 5000.0, 512, 512),
        ("64 KB, depth256, 2QP, RTT 5us", 65536, 400.0, 5000.0, 256, 2),
        ("16 MB, depth64, 1QP, RTT 5us", 16777216, 400.0, 5000.0, 64, 1),
    ]
    for label, size, gbps, rtt, depth, qps in cases:
        got, bind, caps = assess(size, gbps, rtt, depth, qps)
        print(f"{label:>34}{got / 1e9:>9.1f}G"
              f"{got / (gbps * 1e9):>7.0%}   {bind}")
    print("\n  这张表就是排障时的「先算再查」：")
    print("    「1 KB 单 QP 只有 5.4G」        -> 固定成本，这就是物理天花板")
    print("    「4 KB / depth 8 / RTT 20us 只有 13.1G」-> 在途深度不足")
    print("    「64 B 加到 512 QP 也只有 76.8G」-> 网卡消息率上限（150 M msg/s）")
    print("    「16 MB 才摸到线速」             -> 只有大消息才轮到链路当瓶颈")
    print("\n  注意顺序：**绝大多数「带宽上不去」都还没走到查链路那一步。**")


if __name__ == "__main__":
    table_q36()
    table_fixed_breakdown()
    table_q37()
    table_q38()
    table_q39()
    table_depth_need()
    table_bottleneck()
