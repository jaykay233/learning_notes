#!/usr/bin/env python3
"""RDMA 小消息成本模型：把 33 章里的四张表算出来。

不依赖 RDMA 硬件，纯算术。用法：

    python3 33_rdma_cost_model.py

模型边界（先说清楚，避免把估计当实测）：

    端到端延迟 ~= 每消息固定成本 + 线上时间
    线上时间   = 消息字节数 x 8 / 链路带宽

固定成本是"量级估计"，包含 doorbell MMIO、PCIe 读 WQE、NIC 流水线、
对端地址翻译与 DMA 回写。真实值必须用 ib_write_lat / ib_write_bw 量，
本脚本只负责回答"量级关系和拐点在哪"。
"""

# 链路带宽，单位 Gbps
LINKS = {"100G": 100, "200G": 200, "400G": 400}

# 每消息固定成本（纳秒）。同机架、关闭 CQ moderation、小消息。
FIXED_NS = {"乐观": 1000, "典型": 1500, "保守": 2500}

SIZES = [
    ("1 KB", 1 << 10),
    ("4 KB", 4 << 10),
    ("64 KB", 64 << 10),
    ("256 KB", 256 << 10),
    ("1 MB", 1 << 20),
    ("8 MB", 8 << 20),
]


def wire_ns(nbytes: int, gbps: float) -> float:
    """线上时间：bits / bit-per-second -> 秒 -> 纳秒。"""
    return nbytes * 8 / (gbps * 1e9) * 1e9


def table_wire() -> None:
    print("== 1. 线上时间：消息大小 / 链路带宽 ==")
    print(f"{'消息':>8}" + "".join(f"{k:>14}" for k in LINKS))
    for name, n in SIZES:
        row = f"{name:>8}"
        for gbps in LINKS.values():
            row += f"{wire_ns(n, gbps):>12.1f} ns"
        print(row)


def table_share() -> None:
    print("\n== 2. 线上时间占端到端延迟的比例（100G，固定成本 1500 ns）==")
    fixed = FIXED_NS["典型"]
    print(f"{'消息':>8}{'线上 ns':>12}{'端到端 ns':>12}{'线上占比':>12}{'有效 Gbps':>12}")
    for name, n in SIZES:
        w = wire_ns(n, 100)
        total = fixed + w
        eff = n * 8 / (total * 1e-9) / 1e9
        print(f"{name:>8}{w:>12.1f}{total:>12.1f}{w / total:>11.1%}{eff:>12.1f}")


def table_rate() -> None:
    print("\n== 3. 单 QP 消息率上限 = 1 / (固定成本 + 线上时间) ==")
    print("   含义：一个 QP、一条队列、同步等完成时的天花板。")
    print(f"{'消息':>8}" + "".join(f"{k:>18}" for k in FIXED_NS))
    for name, n in SIZES:
        row = f"{name:>8}"
        for fixed in FIXED_NS.values():
            rate = 1e9 / (fixed + wire_ns(n, 100))
            row += f"{rate / 1e6:>14.3f} M/s"
        print(row)


def little() -> None:
    print("\n== 4. Little's law：达到目标消息率需要的在途请求数 ==")
    print("   N = 消息率 x RTT（消息率 msg/s，RTT 秒）")
    rows = [
        (0.5e6, 2.0),
        (2.0e6, 2.0),
        (10.0e6, 2.0),
        (10.0e6, 5.0),
        (50.0e6, 2.0),
    ]
    print(f"{'目标消息率':>14}{'RTT us':>10}{'在途 N':>10}")
    for rate, rtt_us in rows:
        n = rate * rtt_us * 1e-6
        print(f"{rate / 1e6:>12.1f} M{rtt_us:>10.1f}{n:>10.1f}")


def bdp() -> None:
    print("\n== 5. 队列深度：BDP / 消息大小 x 安全系数 ==")
    print("   BDP(bytes) = 带宽(GB/s) x RTT")
    print(f"{'链路':>6}{'RTT us':>9}{'BDP KB':>10}{'1KB 消息在途':>14}{'x2 安全':>10}")
    for name, gbps in LINKS.items():
        for rtt_us in (2.0, 5.0):
            gbps_bytes = gbps * 1e9 / 8 / 1e9  # GB/s
            bdp_bytes = gbps_bytes * 1e9 * rtt_us * 1e-6
            n1k = bdp_bytes / 1024
            print(
                f"{name:>6}{rtt_us:>9.1f}{bdp_bytes / 1024:>10.1f}"
                f"{n1k:>14.1f}{int(n1k * 2):>10}"
            )


def go_back_n() -> None:
    print("\n== 6. go-back-N 放大：窗口 W 内第 k 个包丢了 ==")
    print("   重发数量 = W - k + 1（丢失点之后的全部已发但未被确认的包）")
    window = 64
    print(f"{'丢失位置 k':>12}{'重发包数':>10}{'放大倍数':>10}")
    for k in (1, 16, 32, 48, 64):
        resent = window - k + 1
        print(f"{k:>12}{resent:>10}{resent:>10}x")


if __name__ == "__main__":
    table_wire()
    table_share()
    table_rate()
    little()
    bdp()
    go_back_n()
