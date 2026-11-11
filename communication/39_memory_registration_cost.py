#!/usr/bin/env python3
"""存储 IO 的注册成本账本：reg_mr / FRWR / MW 分别在为什么付钱。

    python3 39_memory_registration_cost.py

回答三个问题：
  1. 为什么"每个 IO 注册一次 MR"是致命的？它按什么收费？
  2. FRWR 省了什么、付出了什么？
  3. 什么时候 FRWR 反而不划算？

常量标记：
  [规范]  IBTA 的 opcode / 语义约束
  [假设]  成本量级（会改绝对值，不改结论）
  [锚点]  与 33 章固定成本模型对齐
"""

# ---------------------------------------------------------------- [规范]
PAGE_B = 4096
# mtt 表项：每页一个翻译项
MTT_PER_PAGE = 1

# ---------------------------------------------------------------- [假设]
# 单页 pin + 建表项的成本。get_user_pages 是大头。
COST_PER_PAGE_NS = 100.0
# reg_mr 的固定部分：分配 context + 权限检查（很小，大头是逐页成本）
REG_FIXED_NS = 200.0
# FRWR 一笔的成本：一次 fast_reg WR + 一次 local_inv
FRWR_NS = 500.0
# MW bind 一笔的成本
MW_BIND_NS = 300.0

LINK_GBPS = 100.0
BYTES_PER_NS = LINK_GBPS * 1e9 / 8 / 1e9      # 12.5 B/ns
RTT_NS = 5000.0
FIXED_1K_NS = 1500.0                          # 33 章的每消息固定成本


def link_ns(size_b):
    return size_b / BYTES_PER_NS


def reg_cost(size_b):
    pages = max(1, -(-size_b // PAGE_B))       # 向上取整
    return REG_FIXED_NS + pages * MTT_PER_PAGE * COST_PER_PAGE_NS


def table_core():
    print("== 1. 三种做法的单次 IO 额外成本 ==")
    print(f"   链路 {LINK_GBPS:.0f} Gbps = {BYTES_PER_NS:.1f} B/ns；"
          f"每页 {COST_PER_PAGE_NS:.0f} ns；reg_mr 固定 {REG_FIXED_NS:.0f} ns")
    print(f"\n{'IO 大小':>10}{'页数':>7}{'线上 ns':>11}{'reg_mr ns':>12}"
          f"{'带宽税':>8}{'FRWR ns':>10}{'带宽税':>8}{'MW ns':>9}")
    sizes = [4 << 10, 64 << 10, 256 << 10, 512 << 10, 1 << 20, 4 << 20]
    for s in sizes:
        pages = max(1, -(-s // PAGE_B))
        l = link_ns(s)
        r = reg_cost(s)
        label = f"{s // 1024} KB"
        print(f"{label:>10}{pages:>7}{l:>11.1f}{r:>12.1f}{r / l:>7.0%}"
              f"{FRWR_NS:>10.1f}{FRWR_NS / l:>7.0%}{MW_BIND_NS:>9.1f}")
    print("\n  带宽税 = 额外成本 / 线上时间。reg_mr 从 64 KB 往上稳定在 ~30%，")
    print("  原因：每页成本 100 ns，而每页的线上时间是 4096 / 12.5 = 327.7 ns，")
    print("        两者都随大小线性涨，比值固定 = 100 / 327.7 ≈ 30.5%。")
    print("  这就是「存储节点永远跑不满线速」的结构性原因：不是网卡不行，")
    print("  是每笔 IO 都在重新建翻译表。FRWR 是 O(1)，所以大 IO 上几乎免费。")
    print("\n  注意 4 KB 那一行：reg_mr 只占线上时间的 92%，但线上时间本身只有 328 ns。")
    print("  小 IO 的正确口径是延迟账，不是带宽账 —— 见下一张表。")


def table_latency():
    print("\n== 2. 小 IO 用延迟口径看 ==")
    print(f"   端到端 = 固定成本 {FIXED_1K_NS:.0f} ns + 线上 + 注册成本")
    print(f"{'IO 大小':>10}{'基线 ns':>11}{'reg_mr':>12}{'增加':>8}{'FRWR':>12}{'增加':>8}")
    for s in (4 << 10, 16 << 10, 64 << 10):
        base = FIXED_1K_NS + link_ns(s)
        r = reg_cost(s)
        print(f"{s // 1024:>8} KB{base:>11.0f}{base + r:>12.0f}{r / base:>7.0%}"
              f"{base + FRWR_NS:>12.0f}{FRWR_NS / base:>7.0%}")
    print("\n  4 KB 时 reg_mr 让延迟涨 ~15%，FRWR 涨 ~25% —— FRWR 反而更差。")
    print("  因为 4 KB 的注册成本只有 300 ns，比一次 fast_reg WR(500 ns) 还便宜。")


def table_crossover():
    print("\n== 3. 交叉点：FRWR 什么时候还不如 reg_mr ==")
    print("   解  REG_FIXED + pages x PAGE_COST  =  FRWR_NS")
    per_page = COST_PER_PAGE_NS
    pages = (FRWR_NS - REG_FIXED_NS) / per_page
    print(f"   pages = ({FRWR_NS:.0f} - {REG_FIXED_NS:.0f}) / {per_page:.0f} = {pages:.1f}")
    if pages > 0:
        print(f"   -> IO 小于 {pages * PAGE_B / 1024:.0f} KB 时，"
              f"reg_mr 的单次成本低于 FRWR")
        print("   原因：reg_mr 固定部分很小，页少时比一次 fast_reg WR 还便宜。")
    else:
        print("   -> 本参数下 reg_mr 在任意大小都更贵，说明固定成本假设偏大")
    print("   但真实实现里 reg_mr 还要抢内核锁、pin 全局页表，这些没进模型；")
    print("   实际交叉点会往大 IO 方向移动，而且高并发下 reg_mr 的争用才是真问题。")


def table_bandwidth_tax():
    print("\n== 4. 带宽税：每 IO 都 reg_mr 时，有效带宽上限 ==")
    print(f"{'IO 大小':>10}{'理想 Gbps':>12}{'实际 Gbps':>12}{'损失':>8}")
    for s in (4 << 10, 64 << 10, 512 << 10, 1 << 20, 4 << 20):
        l = link_ns(s)
        r = reg_cost(s)
        total = l + r
        ideal = LINK_GBPS
        actual = LINK_GBPS * l / total
        print(f"{s // 1024:>8} KB{ideal:>12.1f}{actual:>12.1f}{1 - actual / ideal:>7.0%}")
    print("\n  这就是「存储节点永远跑不满线速」的一个结构性原因：")
    print("  不是网卡不行，是每笔 IO 都在重新建翻译表。")


def table_key_depth():
    print("\n== 5. FRWR 的真实约束：key 数量决定在途 IO 数 ==")
    print("   一笔在途 IO 占一个 key（用了要 local_inv 才能复用）")
    print(f"{'IO 大小':>10}{'线上 ns':>11}{'打满需在途':>12}{'需要 key 数':>13}")
    for s in (4 << 10, 64 << 10, 512 << 10):
        l = link_ns(s)
        # 在途数 = (线上时间 + RTT) / 单 IO 占用时间，近似用 (link+RTT)/link
        need = (l + RTT_NS) / l
        print(f"{s // 1024:>8} KB{l:>11.1f}{need:>12.1f}{int(-(-need // 1)):>13}")
    print("\n  读法：小 IO 要几十个 key 才能打满带宽，大 IO 只要 1~2 个。")
    print("        所以 FRWR key 池深度不是「够用就行」，它直接决定能压出多少吞吐。")
    print("        典型实现取 max_fast_reg_page_list_len 与 QP 深度里较小的那个。")


def table_path():
    print("\n== 6. 一次 NVMe-oF 读 IO 的两端各付什么 ==")
    print("""
  主机（initiator）
    1. 选一个空闲 key（FRWR MR）
    2. post_send: IBV_WR_RDMA_READ，目标 = 对端 buffer + rkey
    3. 数据直接落进主机用户缓冲（不经内核拷贝）
    4. post_send: IBV_WR_LOCAL_INV，把 key 失效以便复用
    5. CQE 到达 -> 这笔 IO 完成

  目标（target）
    1. 收到 READ 请求
    2. 从存储设备（或 page cache）把数据取到 buffer
    3. 该 buffer 必须已注册，或临时 fast_reg
    4. NIC 从 buffer DMA 出去

  两端各自的关键约束:
    主机侧  key 池深度、LOCAL_INV 的时机（早于数据可见会读到脏页）
    目标侧  注册/失效的顺序：必须先 fast_reg 再让对端 READ，
            读完必须 local_inv，否则对端可以继续读一段已释放的内存
""")


def table_failure():
    print("== 7. 顺序错了会长什么样 ==")
    rows = [
        ("先 fast_reg 后写数据", "对端读到旧数据", "数据与注册顺序反了"),
        ("忘了 local_inv", "key 池耗尽 -> 新 IO 全部 ENOMEM", "吞吐随时间线性下降"),
        ("数据可见前就 local_inv", "偶尔读到脏页 / 半新半旧", "压力越大越容易复现"),
        ("reg_mr 放在 IO 路径", "带宽恒定损失 ~30%", "换网卡无效"),
        ("key 池只有 4 个却跑 4 KB IO", "带宽只有理想值的一小部分", "队列深度看着够，但 key 不够"),
        ("MW bind 不给远端写权限", "对端 REM_ACCESS_ERR", "连接是通的"),
    ]
    print(f"{'错误':>26}{'现象':>34}  说明")
    for a, b, c in rows:
        print(f"{a:>26}{b:>34}  {c}")


if __name__ == "__main__":
    table_core()
    table_latency()
    table_crossover()
    table_bandwidth_tax()
    table_key_depth()
    table_path()
    table_failure()
