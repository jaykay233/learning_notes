#!/usr/bin/env python3
"""集合通信与 rail 拓扑账本：AllReduce 三种实现，以及 All-to-All 为什么会破 rail 对齐。

    python3 40_collective_topology_model.py

三个问题：
  1. Ring / Tree / SHARP 的差别在"量"还是"步"？
  2. rail-optimized 拓扑靠什么省钱，前提是什么？
  3. MoE 的 All-to-All 为什么破坏这个前提，代价怎么算？

常量标记：
  [规范]  IBTA / 拓扑定义
  [假设]  带宽与步数（会改绝对值，不改结论）
  [锚点]  与 33 章固定成本、35 章拥塞口径对齐
"""

from math import log2, ceil

# ---------------------------------------------------------------- [假设]
LINK_GBPS = 400.0
BYTES_PER_NS = LINK_GBPS * 1e9 / 8 / 1e9        # 50 B/ns
FIXED_NS = 1500.0                               # 33 章每消息固定成本
RAILS = 8                                       # rail-optimized 里的 rail 数


def wire_ns(nbytes):
    return nbytes / BYTES_PER_NS


# ---------------------------------------------------------------- 1. AllReduce
def ring_volume(size_b, n):
    """每个 rank 的收发量：2*(N-1)/N * S（reduce-scatter + allgather）。"""
    return 2 * (n - 1) / n * size_b


def ring_steps(n):
    """ring 的串行步数：2*(N-1)，每步走 S/N。"""
    return 2 * (n - 1)


def tree_volume(size_b, n):
    """递归 halving + doubling：量与 ring 相同，但步数是 2*log2(N)。"""
    return 2 * (1 - 1 / n) * size_b


def tree_steps(n):
    return 2 * ceil(log2(n))


def sharp_volume(size_b, n):
    """SHARP：交换机做归约，每 rank 上送一份、下收一份，无 rank 间转发。
    量是 2*S（上 + 下），步数 2（up + down）。"""
    return 2 * size_b


def sharp_steps():
    return 2


def table_allreduce():
    print("== 1. AllReduce 三种实现：量几乎一样，步数差一个数量级 ==")
    size = 256 << 20      # 256 MiB
    print(f"   消息 {size >> 20} MiB，链路 {LINK_GBPS:.0f} Gbps = {BYTES_PER_NS:.0f} B/ns")
    print(f"\n{'N':>6}{'ring 量':>12}{'ring 步':>9}{'tree 量':>12}{'tree 步':>9}"
          f"{'SHARP 量':>12}{'SHARP 步':>10}")
    for n in (8, 16, 64, 256, 1024):
        rv = ring_volume(size, n) / (1 << 20)
        tv = tree_volume(size, n) / (1 << 20)
        sv = sharp_volume(size, n) / (1 << 20)
        print(f"{n:>6}{rv:>10.1f}MB{ring_steps(n):>9}{tv:>10.1f}MB"
              f"{tree_steps(n):>9}{sv:>10.1f}MB{sharp_steps():>10}")
    print("\n  读法：ring 与 tree 的量几乎一样（都是 2S 量级），")
    print("        差别全在步数：ring 是 2(N-1)，tree 是 2*log2(N)。")
    print("        SHARP 的量是固定的 2S —— 注意它并不比 ring 少，甚至略多。")
    print("        SHARP 省的是步数与聚合次数，不是链路带宽。")



def table_allreduce_time():
    print("\n== 2. 换算成时间：什么时候步数才是瓶颈 ==")
    print("   时间 ≈ 步数 x 固定成本 + 每 rank 量 / 带宽     （每步一个消息）")
    print(f"   固定成本 {FIXED_NS:.0f} ns/步，带宽 {BYTES_PER_NS:.0f} B/ns")
    for size, label in ((1 << 20, "1 MiB 小消息"), (256 << 20, "256 MiB 大消息")):
        print(f"\n   --- {label} ---")
        print(f"{'N':>6}{'ring ms':>11}{'tree ms':>11}{'SHARP ms':>11}{'SHARP 加速':>12}")
        for n in (8, 64, 256, 1024):
            r = ring_steps(n) * FIXED_NS + wire_ns(ring_volume(size, n))
            t = tree_steps(n) * FIXED_NS + wire_ns(tree_volume(size, n))
            s = sharp_steps() * FIXED_NS + wire_ns(sharp_volume(size, n))
            print(f"{n:>6}{r / 1e6:>11.3f}{t / 1e6:>11.3f}{s / 1e6:>11.3f}"
                  f"{r / s:>11.1f}x")
    print("\n  1 MiB：N=1024 时 ring 要 2046 步（3.11 ms），SHARP 只要 2 步（0.045 ms）。")
    print("  256 MiB：N=1024 时 ring 13.80 ms、SHARP 10.74 ms，加速只剩 1.3x；")
    print("            N=8 时 SHARP 甚至更慢（10.74 vs 9.42 ms，0.9x）——")
    print("            因为 SHARP 的量是 2S，而 ring 是 2(N-1)/N*S < 2S。")
    print("  -> 小消息的瓶颈是步数（固定成本）；大消息的瓶颈是量，SHARP 帮不上忙。")
    print("  -> 所以 SHARP 的正确卖点是「卸载 GPU 的归约计算 + 砍步数」，不是省带宽。")


# ---------------------------------------------------------------- 3. rail
def table_rail():
    print("\n== 3. rail-optimized 靠什么省钱 ==")
    print(f"   拓扑：{RAILS} 个 rail，GPU r 接到 rail 交换机 r；rail 之间经 spine 互联")
    print("""
  一次 AllReduce 的层次:
    1) 节点内    NVLink 域内归约（不占网）
    2) rail 内   同 rail 的 8 个节点经 rail 交换机交换
    3) rail 间   只有「部分和」跨 spine，数据量 / 8

  关键假设: 通信伙伴的分布是按 rail 对齐的。
  只要流量不出 rail，spine 只承担 1/8 的聚合量；一旦出 rail，跳数翻倍。
""")


def table_alignment():
    print("== 4. All-to-All 破坏 rail 对齐的定量后果 ==")
    print("   MoE All-to-All：每个 rank 把 token 发给持有目标 expert 的 rank")
    print(f"   rail = rank mod {RAILS}，目标 rank 在 1..EP 上近似均匀")
    print(f"\n{'EP':>6}{'同 rail 概率':>14}{'跨 spine 比例':>15}{'跳数(平均)':>13}")
    for ep in (8, 16, 64, 256):
        p_same = min(RAILS / ep, 1.0) if ep >= RAILS else 1.0
        hops = p_same * 1 + (1 - p_same) * 2
        print(f"{ep:>6}{p_same:>13.1%}{1 - p_same:>14.1%}{hops:>13.2f}")
    print("\n  EP=8 时全部同 rail -> 1 跳、spine 零流量（这是 rail 拓扑的最优区）")
    print("  EP=64 时 87.5% 流量跨 spine -> 跳数 1.88、spine 承压 8 倍")
    print("  EP=256 时 96.9% 跨 spine -> rail 拓扑基本失效，退化成普通 fat-tree")


def table_moe_volume():
    print("\n== 5. MoE 的 All-to-All 数据量（每 rank 每步）==")
    hidden = 7168
    dtype_b = 2                      # bf16
    tokens = 4096
    topk = 8
    print(f"   假设：每 rank {tokens} token，top-{topk}，hidden {hidden}，bf16")
    print(f"\n{'EP':>6}{'dispatch':>12}{'combine':>12}{'合计':>12}")
    for ep in (8, 16, 64, 256):
        # 每个 token 复制到 topk 个 expert，均匀落在 ep 个 rank 上
        bytes_out = tokens * topk * hidden * dtype_b / ep
        print(f"{ep:>6}{bytes_out / 1e6:>10.2f}MB{bytes_out / 1e6:>10.2f}MB"
              f"{2 * bytes_out / 1e6:>10.2f}MB")
    print("\n  注意：EP 越大，每 rank 的量越小（摊薄），但跨 spine 比例越高。")
    print("  这就是 ep_size 的两难：容量要 EP 大，拓扑要 EP 小。")
    print("  DeepEP 的解法不是改拓扑，而是把 EP 限制在 NVLink 域内 + 分层传输")
    print("  （见 24 章 hybrid scaleout/scaleup）。")


def table_choice():
    print("== 6. 选型速查 ==")
    rows = [
        ("小消息 AllReduce（<1MiB）", "SHARP 或 tree", "步数主导，ring 的 2(N-1) 步不可接受"),
        ("大消息 AllReduce（>64MiB）", "ring 或 tree 都行", "量主导，三者差距收敛到 2x 内"),
        ("EP <= 8（单 NVLink 域）", "rail 对齐 / 层级", "全部同 rail，spine 零流量"),
        ("EP = 64 跨节点", "分层 + 分桶", "87.5% 跨 spine，需靠分层减少跨 rail 次数"),
        ("EP = 256+", "别指望 rail 优化", "拓扑退化为 fat-tree，只能靠拥塞控制"),
        ("推理 decode（小 batch）", "低延迟优先", "固定成本主导，看 22 章的成本模型"),
    ]
    print(f"{'场景':>26}   {'选择':>18}   理由")
    for a, b, c in rows:
        print(f"{a:>26}   {b:>18}   {c}")


if __name__ == "__main__":
    table_allreduce()
    table_allreduce_time()
    table_rail()
    table_alignment()
    table_moe_volume()
    table_choice()
