#!/usr/bin/env python3
"""两件事的定量对比：
  A. RDMA vs DPDK vs 内核 TCP —— CPU 到底花在哪（Q51）
  B. 自适应路由 vs ECMP —— 负载均衡与乱序的交换（Q25）

    python3 45_rdma_vs_dpdk_and_routing.py

常量标记：
  [规范]  协议定义
  [假设]  cycles / 延迟（会改绝对值，不改结论）
"""

import random

CPU_GHZ = 3.0
CYC_PER_US = CPU_GHZ * 1000          # 3000 cycles/us
LINK_GBPS = 400.0
BYTES_PER_NS = LINK_GBPS * 1e9 / 8 / 1e9


# ================================================================ A. CPU
# [假设] 处理一条消息所需 CPU 周期（含 syscall / 协议 / 拷贝的各段）
COST = {
    "内核 TCP": 20000,     # 6.7 us：syscall + 协议栈 + 两次拷贝
    "DPDK":     600,       # 0.20 us：poll + 解析 + 拷贝 + 用户态协议
    "RDMA":     300,       # 0.10 us：doorbell + poll CQE，协议在网卡
}

# [假设] 每包 1500 B 时 DPDK 的纯转发能力（业内常见量级 10~20 Mpps/core）
DPDK_MPPS_PER_CORE = 12.0
# [假设] 单张 400G 双口网卡的消息率上限（PCIe + CQ 处理能力限制）
NIC_MSG_RATE_CAP = 150e6


def table_cpu():
    print("== A1. 一条消息花多少 CPU（3 GHz）==")
    print(f"{'方案':>10}{'cycles/msg':>12}{'us/msg':>9}"
          f"{'10M msg/s 需要核数':>20}")
    for name, cyc in COST.items():
        us = cyc / CYC_PER_US
        cores = 10e6 * cyc / (CPU_GHZ * 1e9)
        print(f"{name:>10}{cyc:>12}{us:>9.2f}{cores:>20.1f}")
    print("\n  DPDK 与 RDMA 都绕过了内核，差距来自**协议由谁执行**：")
    print("    DPDK：ACK/重传/拥塞全在用户态代码里 -> CPU 随功能线性涨")
    print("    RDMA：这些在网卡里              -> CPU 基本恒定")
    print("  所以两者不是「快/慢」的关系，是「谁的 CPU 在干活」的关系。")


def table_scaling():
    print("\n== A2. 扩展性：加核能不能提消息率 ==")
    print(f"   网卡消息率上限按 {NIC_MSG_RATE_CAP / 1e6:.0f} M msg/s 计"
          f"（[假设] 单张 400G 双口）")
    print(f"{'方案':>10}{'1 核':>11}{'8 核':>11}{'64 核':>12}{'瓶颈':>10}")
    for name, cyc in COST.items():
        one = (CPU_GHZ * 1e9) / cyc
        vals = []
        for cores in (1, 8, 64):
            raw = one * cores
            vals.append(min(raw, NIC_MSG_RATE_CAP))
        bottleneck = "网卡" if name == "RDMA" else "CPU"
        print(f"{name:>10}{vals[0] / 1e6:>10.1f}M{vals[1] / 1e6:>10.1f}M"
              f"{vals[2] / 1e6:>11.1f}M{bottleneck:>10}")
    print("\n  RDMA 加核很快就撞到网卡上限：**单核就够，加核没用**。")
    print("  DPDK 线性扩展，但代价是把核绑在轮询上（无法超卖，空转也占满）。")
    print("  这就是选型的核心权衡：RDMA 花网卡的钱，DPDK 花 CPU 的钱。")


def table_when():
    print("\n== A3. 什么时候选哪个 ==")
    rows = [
        ("要极低延迟 + 低 CPU", "RDMA", "协议在网卡，单核可跑 10M msg/s"),
        ("要对端完全无感知（大块写）", "RDMA WRITE", "零接收端参与，但要 credit（44 章）"),
        ("只有以太网、没有 RNIC", "DPDK", "纯软件，功能自己写"),
        ("要自己改协议（自定义头）", "DPDK", "RDMA 的传输层不可改"),
        ("要跑在虚拟化/容器里且不想管 VF", "DPDK", "VF 与 PFC 配置复杂（41 章）"),
        ("消息极小但要求高吞吐", "RDMA", "固定成本主导（33 章），网卡卸载更划算"),
        ("需要硬件级多租户隔离", "RDMA + SR-IOV", "PD 不是墙，VF 才是（41 章）"),
    ]
    print(f"{'需求':>30}   {'选择':>14}   理由")
    for a, b, c in rows:
        print(f"{a:>30}   {b:>14}   {c}")
    print("\n  一句话：**RDMA 买「不用 CPU」，DPDK 买「可以改协议」。**")
    print("  两者不冲突：很多系统用 DPDK 做控制面/自定义封装，用 RDMA 做数据面。")


# ================================================================ B. 路由
def ecmp_table(flows=16, paths=8, trials=2000):
    print("\n== B1. ECMP：按流哈希，冲突不可避免 ==")
    print(f"   {flows} 条流 / {paths} 条等价路径，随机哈希 {trials} 次")
    rng = random.Random(42)
    maxloads = []
    overloaded = 0
    for _ in range(trials):
        buckets = [0] * paths
        for _ in range(flows):
            buckets[rng.randrange(paths)] += 1
        mx = max(buckets)
        maxloads.append(mx)
        if mx > flows / paths * 1.5:
            overloaded += 1
    avg = flows / paths
    maxloads.sort()
    p50 = maxloads[len(maxloads) // 2]
    p99 = maxloads[int(len(maxloads) * 0.99)]
    print(f"\n   平均每条路径 {avg:.2f} 条流")
    print(f"   最热路径负载  p50={p50}  p99={p99}  max={maxloads[-1]}")
    print(f"   最热路径 > 1.5x 平均 的概率：{overloaded / trials:.1%}")
    print("\n  这就是 ECMP 的固有问题：**流的数量是离散的，哈希无法均分。**")
    print(f"  8 条路径 16 条流时，最热路径 p99 达到 {p99} 条流 —— 是平均的 "
          f"{p99 / avg:.2f} 倍。")
    print("  而且「大象流」（训练梯度、存储大 IO）让这个问题更严重：")
    print("  一条流占满一条路径，其余流只能挤剩下的。")


def ecmp_bigflow():
    print("\n== B2. 大象流：一条流压满一条路径 ==")
    print("   8 条等价路径；每条大象流独占一条（无法再分给其他流）")
    print(f"\n{'大象流数':>9}{'被占路径':>10}{'老鼠流可用路径':>16}"
          f"{'老鼠流可用带宽':>16}")
    for elephants in (0, 1, 2, 4, 6):
        free = 8 - elephants
        print(f"{elephants:>9}{elephants:>10}{free:>16}{free / 8:>15.0%}")
    print("\n  关键：**逐流哈希无法把一条流拆到多条路径上。**")
    print("  一条 400G 的大象流就能吃掉 1/8 的等价路径容量，")
    print("  其余所有老鼠流只能在剩下的 7/8 里抢 —— 这就是「局部热点」。")
    print("  而且这和流数无关：加再多老鼠流，也补不回被大象占掉的那条路径。")


def adaptive_reorder(paths=4, spread_us=0.5, pkt_gap_us=0.08, n=20000, seed=7):
    """每条包独立随机选路；路径延迟按 spread 均匀铺开（确定性），
    而不是每次运行随机采样 —— 真实的路径延迟是固定的。"""
    rng = random.Random(seed)
    if paths == 1:
        delays = [0.0]
    else:
        delays = [i * spread_us / (paths - 1) for i in range(paths)]
    arrivals = []
    for i in range(n):
        send_t = i * pkt_gap_us
        d = delays[rng.randrange(paths)]
        arrivals.append((send_t + d, i))
    arrivals.sort()
    reordered = 0
    max_seen = -1
    for _, seq in arrivals:
        if seq < max_seen:
            reordered += 1
        max_seen = max(max_seen, seq)
    return reordered, n, delays


def table_adaptive():
    print("\n== B3. 自适应路由：按包喷洒，用乱序换均衡 ==")
    print("   4 条路径，延迟在 [0, spread] 上均匀铺开；每包随机选路")
    print("   判据：只要「路径延迟差 > 包间隔」，相邻包就会互换顺序")
    print(f"\n{'延迟差 us':>11}{'包间隔 us':>11}{'包大小 B':>10}"
          f"{'乱序包数':>10}{'乱序率':>9}")
    for spread in (0.1, 0.5, 1.0):
        for gap in (0.08, 0.5, 2.0):
            r, n, _ = adaptive_reorder(paths=4, spread_us=spread, pkt_gap_us=gap)
            size = int(gap * BYTES_PER_NS * 1000)   # us -> ns -> B
            print(f"{spread:>11.1f}{gap:>11.2f}{size:>10}"
                  f"{r:>10}{r / n:>8.2%}")
    print("\n  **阈值行为**：包间隔 > 延迟差时乱序率为 0，否则高得离谱。")
    print("  包间隔 = 包大小 / 线速：400G 下 64 B 是 0.001 us，4 KB 是 0.08 us。")
    print("  所以小包 + 按包喷洒 = 灾难，大包 + 按包喷洒 = 几乎无害。")


def table_reorder_cost():
    print("\n== B4. 乱序的代价 = 丢包的代价（接 42 章）==")
    print("   go-back-N 只认累积 ACK：乱序到达的包被当成丢失，直接丢弃并重传")
    print("   放大系数按 1 + p*(W+1)/2 估算（42 章的一阶近似）")
    print(f"\n{'乱序率':>9}{'W=32 放大':>12}{'W=32 有效':>12}"
          f"{'W=256 放大':>13}{'W=256 有效':>12}")
    for r in (0.001, 0.01, 0.05, 0.1, 0.3):
        a32 = 1 + r * (32 + 1) / 2
        a256 = 1 + r * (256 + 1) / 2
        print(f"{r:>8.1%}{a32:>12.3f}{LINK_GBPS / a32:>11.1f}G"
              f"{a256:>13.3f}{LINK_GBPS / a256:>11.1f}G")
    print("\n  读法：乱序率 10% + W=256 -> 放大 13.85 倍 -> 400G 只剩 28.9G。")
    print("  所以「自适应路由提升利用率」必须加条件：**小窗口或大消息才成立。**")
    print("  小包 + 大窗口 + 多路径 = 乱序风暴，比 ECMP 的局部热点更糟。")
    print("  这也是 42 章那句「乱序 = 假丢包」的完整后果链：")
    print("    按包喷洒 -> 乱序 -> 接收端丢弃 -> go-back-N 重传 -> 带宽被自己吃掉")


def table_routing_choice():
    print("\n== B5. 路由策略选择 ==")
    rows = [
        ("大量小包（RPC/All-to-All）", "ECMP 或按流", "按包喷洒会乱序，代价远大于均衡收益"),
        ("少量大流（训练梯度）", "自适应 / 按包喷洒", "大流下乱序影响小，均衡收益大"),
        ("大象流 + 老鼠流混跑", "按流 + 拥塞控制", "逐流哈希必然热点，靠 DCQCN 兜底（35 章）"),
        ("对尾延迟极敏感", "ECMP + 多路径备份", "宁可有热点也不要乱序"),
        ("存储大 IO", "自适应 + 大 MTU", "大包间隔大，乱序率低"),
    ]
    print(f"{'场景':>28}   {'选择':>20}   理由")
    for a, b, c in rows:
        print(f"{a:>28}   {b:>20}   {c}")


if __name__ == "__main__":
    table_cpu()
    table_scaling()
    table_when()
    ecmp_table()
    ecmp_bigflow()
    table_adaptive()
    table_reorder_cost()
    table_routing_choice()
