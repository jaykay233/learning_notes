#!/usr/bin/env python3
"""L2 剩余语义的代价账本：MTU / inline / immediate / atomic / UD。

    python3 43_l2_semantics_cost.py

五个问题，每个都对应一个容易被当成"实现细节"的设计选择：
  1. MTU 从 256 改到 4K，凭什么能差一倍带宽？
  2. inline 到底省了哪一次访问？边界在哪？
  3. immediate 省了一次内存读，但付了什么？
  4. atomic 什么时候比 READ+本地算+WRITE 划算？
  5. UD 不可靠，为什么在"小消息 + 海量对端"场景仍然最优？

常量标记：
  [规范]  IBTA 的字段与限制
  [假设]  成本量级（会改绝对值，不改结论）
"""

from math import ceil

# ---------------------------------------------------------------- [假设]
LINK_GBPS = 400.0
BYTES_PER_NS = LINK_GBPS * 1e9 / 8 / 1e9        # 50 B/ns
RTT_NS = 3000.0                                 # 同集群单跳 RTT 量级

# 每包协议开销：Ethernet(14) + FCS(4) + preamble(8) + IFG(12)
#                + IPv4(20) + UDP(8) + BTH(12) + ICRC(4)
PKT_OVERHEAD_B = 82
# NIC 的每包处理能力上限：12.5 Mpps（[假设]，按设备数据手册查）
PKT_PROCESS_NS = 80.0
# 非 inline 时，NIC 要额外读一次 payload（PCIe 往返 + DMA 建立）
PAYLOAD_FETCH_NS = 250.0
# 33 章的每消息固定成本，用来做延迟百分比的分母
FIXED_MSG_NS = 1500.0
# inline 的容量上限（设备相关，这里取常见值）
MAX_INLINE_B = 256
# atomic 的 NIC 内部处理
ATOMIC_EXEC_NS = 120.0


# ---------------------------------------------------------------- 1. MTU
def mtu_table():
    print("== 1. MTU：字节吞吐与包处理是两个独立资源，取 max ==")
    print(f"   链路 {LINK_GBPS:.0f} Gbps = {BYTES_PER_NS:.0f} B/ns")
    print(f"   每包协议开销 {PKT_OVERHEAD_B} B；NIC 每包处理 {PKT_PROCESS_NS:.0f} ns"
          f"（= {1e3 / PKT_PROCESS_NS:.1f} Mpps 上限）")
    size = 1 << 20      # 1 MiB
    print(f"   消息 {size >> 10} KiB")
    print("   总时间 = max(线上字节/带宽, 包数 x 每包处理)   <- 不是相加")
    print(f"\n{'MTU':>8}{'包数':>8}{'头开销':>9}{'字节 ns':>10}"
          f"{'包处理 ns':>11}{'瓶颈':>8}{'有效 Gbps':>11}{'效率':>8}")
    for mtu in (256, 512, 1024, 2048, 4096):
        pkts = ceil(size / mtu)
        on_wire = pkts * (mtu + PKT_OVERHEAD_B)
        wire = on_wire / BYTES_PER_NS
        proc = pkts * PKT_PROCESS_NS
        total = max(wire, proc)
        eff = size * 8 / total       # bit/ns = Gbps
        bound = "包处理" if proc > wire else "带宽"
        print(f"{mtu:>8}{pkts:>8}{1 - size / on_wire:>8.1%}{wire:>10.0f}"
              f"{proc:>11.0f}{bound:>8}{eff:>11.1f}{eff / LINK_GBPS:>8.1%}")
    print("\n  4K 时带宽是瓶颈（98%）；256B 时每包处理是瓶颈（只剩 6.4%）。")
    print("  这就是「小 MTU 打不满高带宽口」的机理：不是头部浪费几字节，")
    print("  是**瓶颈从 B/ns 换成了 pps**，而 pps 是网卡的硬上限。")
    print("  也解释了 path_mtu 必须两端一致：协商出的 MTU 决定你在哪条曲线上。")


# ---------------------------------------------------------------- 2. inline
def inline_table():
    print("\n== 2. inline：省掉的是 NIC 回读 payload 的那一次访问 ==")
    print("   非 inline：BTH 里放 SGE 地址 -> NIC 通过 PCIe 回读 payload -> 再上链")
    print("   inline  ：payload 直接跟在 WQE 后面，NIC 读 WQE 时一起拿到")
    print(f"\n   [假设] 回读一次 {PAYLOAD_FETCH_NS:.0f} ns，inline 上限 {MAX_INLINE_B} B")
    print(f"   延迟基数 = 固定成本 {FIXED_MSG_NS:.0f} ns + 回读 + 线上时间")
    print(f"\n{'消息':>8}{'inline':>10}{'非 inline ns':>14}{'inline ns':>12}"
          f"{'省下(占端到端)':>16}")
    for size in (32, 64, 128, 256, 512, 1024):
        base = FIXED_MSG_NS + PAYLOAD_FETCH_NS + size / BYTES_PER_NS
        inl = FIXED_MSG_NS + size / BYTES_PER_NS
        ok = "是" if size <= MAX_INLINE_B else "否"
        if size <= MAX_INLINE_B:
            print(f"{size:>8}{ok:>10}{base:>14.1f}{inl:>12.1f}"
                  f"{1 - inl / base:>15.1%}")
        else:
            print(f"{size:>8}{ok:>10}{base:>14.1f}{'—':>12}{'—':>16}")
    print(f"\n  256 B 时省 {PAYLOAD_FETCH_NS:.0f} ns，占端到端约 "
          f"{PAYLOAD_FETCH_NS / (FIXED_MSG_NS + PAYLOAD_FETCH_NS + 256 / BYTES_PER_NS):.1%}。")
    print("  绝对值不大，但对延迟敏感的 RPC 是可观的；对吞吐几乎无影响。")
    print("  超过 inline 上限必须走 SGE —— 所以 max_inline_data 要和消息分布对齐。")
    print("  代价：inline 占 WQE 空间（深度变浅），且 inline 的数据不能被远端 READ。")


# ---------------------------------------------------------------- 3. immediate
def imm_table():
    print("\n== 3. immediate：32 bit 带外信令，省一次内存读 ==")
    print("""
  不用 imm 的做法（WRITE + 轮询标志位）:
    发送端 WRITE 数据 -> 接收端在数据里约定一个 flag 字段
    接收端 -> 本地 spin 读 flag（每次读都是一次内存访问，且要跨 cacheline）

  用 imm 的做法（WRITE_WITH_IMM）:
    发送端 WRITE 数据 + 32 bit imm -> 接收端收到一个 RECV CQE，imm 在 wc.imm_data
    接收端 -> poll CQ 拿到 imm，不碰数据区
""")
    spin_ns = 80.0
    print(f"   [假设] 一次本地 flag 轮询 {spin_ns:.0f} ns，平均轮询 3 次才命中")
    print(f"\n{'方案':>22}{'接收端额外成本':>16}   说明")
    print(f"{'WRITE + 轮询 flag':>22}{f'{spin_ns * 3:.0f} ns':>16}"
          f"   依赖数据可见性，易读到半新半旧")
    print(f"{'WRITE_WITH_IMM':>22}{'0（复用 poll）':>14}"
          f"   需要一个 recv WR 承接")
    print("\n  三条约束，缺一个就会踩坑：")
    print("    1) 接收端必须预投 recv WR —— 否则 RNR（42 章）")
    print("    2) imm 只有 32 bit，装不下指针；常见用法是「消息序号 / slot id」")
    print("    3) imm 的 CQE 顺序 = 同一个 QP 内有序；跨 QP 无序（需要 fence）")


# ---------------------------------------------------------------- 4. atomic
def atomic_table():
    print("\n== 4. atomic：一次 RTT 完成「读-改-写」 ==")
    print(f"   [假设] RTT {RTT_NS:.0f} ns，atomic 内部执行 {ATOMIC_EXEC_NS:.0f} ns")
    rdma_rtt = RTT_NS + 8 / BYTES_PER_NS
    local_rtt = RTT_NS + 8 / BYTES_PER_NS
    atomic_cost = RTT_NS + ATOMIC_EXEC_NS
    print(f"\n{'方案':>26}{'往返次数':>10}{'成本 ns':>12}   适用")
    print(f"{'READ + 本地算 + WRITE':>26}{2:>10}{rdma_rtt + local_rtt:>12.0f}"
          f"   需要 CAS 语义、条件复杂")
    print(f"{'ATOMIC (fetch_and_add)':>26}{1:>10}{atomic_cost:>12.0f}"
          f"   计数器、配额、序号分配")
    print(f"\n  省下的 {(rdma_rtt + local_rtt) / atomic_cost:.1f}x 来自「少一次往返」，不是「算得快」。")
    print("\n  但 atomic 有硬限制（Q15）：")
    print("    · 只支持 8 字节（32/64 bit），没有 128 bit 的通用 atomic")
    print("    · 必须在 REMOTE_ATOMIC 权限的 MR 上，且 8 字节对齐")
    print("    · 同一个地址上的 atomic 在 NIC 内**串行化** -> 热点地址会被打成瓶颈")
    print("    · 不支持「读一批再原子写回」的事务语义，跨对象一致性要自己做")
    print("\n  最后一条最重要：atomic 给你的是单地址原子性，不是事务。")


# ---------------------------------------------------------------- 5. UD
def ud_table():
    print("\n== 5. UD：用「不可靠」换「无连接状态」 ==")
    print("\n   RC 每条连接的状态成本见 34 章；UD 只有 QP 本身，没有对端状态。\n")
    print(f"{'场景':>22}{'RC':>16}{'UD':>18}   原因")
    rows = [
        ("对端数量", "O(N) 状态", "O(1) 状态", "UD 不需要建连"),
        ("单消息上限", "几乎无限", "<= path MTU", "UD 一条消息一个包"),
        ("消息大小 64 B", "1.5 us 级", "1.5 us 级", "都是单包，固定成本主导"),
        ("消息大小 1 MB", "一条消息", "做不到", "UD 超过 MTU 无法发送"),
        ("可靠性", "硬件重传", "无", "UD 丢了就丢了"),
        ("顺序", "同 QP 有序", "不保证", "多路径/多包无序号"),
        ("典型用途", "存储、训练梯度", "服务发现、元数据、广播", ""),
    ]
    for a, b, c, d in rows:
        print(f"{a:>22}{b:>16}{c:>18}   {d}")
    print("\n  UD 的关键取舍：**小消息 + 海量对端**。")
    print("  N=10000 个对端时，RC 要 10000 条连接（34 章的账本撑不住），")
    print("  UD 只要几条 QP —— 代价是每个包必须自带 GRH（目标地址），且不保证送达。")
    print("  这也解释了 Q17：UD 不可靠却重要，因为它是唯一能规模化的「无状态」模式。")


# ---------------------------------------------------------------- 6. 选型
def chooser():
    print("\n== 6. 语义选型速查 ==")
    rows = [
        ("小请求 + 小响应（RPC）", "SEND/RECV + inline", "两端都要 WR，但延迟最低"),
        ("大块单向传输", "RDMA WRITE", "零接收端参与，但要提前知道地址"),
        ("写完通知（无数据）", "WRITE_WITH_IMM", "省一次 flag 轮询，需预投 recv WR"),
        ("远端计数器/配额", "ATOMIC fetch_and_add", "一次 RTT，别用它做事务"),
        ("条件更新（CAS）", "ATOMIC cmp_and_swap", "热点地址会串行化"),
        ("读远端的大块数据", "RDMA READ", "要 fence 才能保证后续写可见"),
        ("服务发现 / 广播", "UD", "无连接状态，消息 <= MTU"),
        ("跨 QP 的顺序保证", "IBV_SEND_FENCE", "只在同 QP 内有序，跨 QP 必须显式 fence"),
    ]
    print(f"{'需求':>24}   {'选择':>26}   理由")
    for a, b, c in rows:
        print(f"{a:>24}   {b:>26}   {c}")


if __name__ == "__main__":
    mtu_table()
    inline_table()
    imm_table()
    atomic_table()
    ud_table()
    chooser()
