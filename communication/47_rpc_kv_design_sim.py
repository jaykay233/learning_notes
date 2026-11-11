#!/usr/bin/env python3
"""端到端设计：RDMA RPC 与 KV 存储（Q49、Q50 的可算版本）。

    python3 47_rpc_kv_design_sim.py

三个问题：
  1. RPC 的并发上限设多少？设小了 p99 爆炸，设大了内存与 CQ 吃掉。
  2. KV 的 GET 用单边 READ 还是走 RPC？省下的到底是什么？
  3. 单边 READ 绕过了服务端 —— 它带来的正确性代价是多少纳秒？

所有成本沿用 46 章的固定成本模型：
    每条消息固定成本 1500 ns
常量标记：
  [规范]  IBTA 语义
  [假设]  服务时间与速率
"""

import heapq
import random

FIXED_NS = 1500.0               # 46 章：一条消息的固定成本
RTT_NS = 3000.0                 # [假设] 同机架 RTT
BYTES_PER_NS = 50.0             # 400 Gbps


# ---------------------------------------------------------------- 1. RPC 排队
def simulate_rpc(lam_per_us, service_us, workers, n=100_000, seed=1):
    """M/M/W 排队：W 个服务槽（= recv WR 池深度），指数服务时间。"""
    rng = random.Random(seed)
    t = 0.0
    busy_until = [0.0] * workers
    heap = []
    for w in range(workers):
        heapq.heappush(heap, (0.0, w))     # (空闲时刻, 槽编号)
    lat = []
    for _ in range(n):
        t += rng.expovariate(lam_per_us)
        # 等待最早空闲的槽
        free_at, w = heapq.heappop(heap)
        start = max(t, free_at)
        svc = rng.expovariate(1.0 / service_us)
        finish = start + svc
        heapq.heappush(heap, (finish, w))
        lat.append(finish - t)
    lat.sort()
    # 稳态：丢掉前 10% 预热
    steady = lat[n // 10:]
    p50 = steady[len(steady) // 2]
    p99 = steady[int(len(steady) * 0.99)]
    # 吞吐：总完成数 / 总耗时（按最后一个完成时间 - 第一个送出时间近似）
    return 1.0 / lam_per_us, p50, p99


def table_concurrency():
    print("== 1. RPC 并发上限（= recv WR 池深度）：排队只影响尾延迟 ==")
    print("   服务时间 2.0 us/请求（[假设] 解析+查表+回复）")
    print("   吞吐上限 = W / 服务时间；接近它时排队时间发散\n")
    print(f"{'W':>4}{'到达 M/s':>10}{'利用率':>8}"
          f"{'p50 us':>9}{'p99 us':>9}{'状态':>10}")
    for w in (4, 16, 64, 256):
        ceiling = w / 2.0            # 每 us
        for frac in (0.5, 0.9, 0.99):
            lam = ceiling * frac
            _, p50, p99 = simulate_rpc(lam, 2.0, w, n=40_000, seed=7)
            state = ("空闲" if frac < 0.6 else
                     "有排队" if frac < 0.95 else "接近饱和")
            print(f"{w:>4}{lam:>10.2f}{frac:>8.0%}"
                  f"{p50:>9.1f}{p99:>9.1f}{state:>10}")
    print("\n  读法：")
    print("    p99/p50 的比值在低负载时本来就大（p50 只有 1.6 us），所以看**绝对值**：")
    print("    W=4 时从半载到 99% 负载，p99 从 9.6 us 涨到 166 us（17 倍），")
    print("    而 p50 只从 1.8 涨到 45.7 us。这就是排队论的形状，不是实现问题。")
    print("    加大 W 能把 p99 拉回 10 us 量级，代价是内存与 CQ 深度：")
    print("    每条 recv WR 都占内存，CQ 也要 >= max_send_wr + max_recv_wr（34 章），")
    print("    否则 W 大了 CQ 不够，会静默丢 CQE。")


# ---------------------------------------------------------------- 2. GET 两条路
def table_get_paths(size_b=4096):
    print("\n== 2. KV 的 GET：单边 READ 与走 RPC 的对比 ==")
    print(f"   取回 {size_b} B；RTT {RTT_NS / 1000:.1f} us；固定成本 {FIXED_NS} ns\n")
    wire = size_b / BYTES_PER_NS
    rows = [
        ("单边 RDMA READ", FIXED_NS + RTT_NS + wire, 0.0, "客户端发 WQE，服务端不参与"),
        ("RPC（SEND 请求 + SEND 回复）", FIXED_NS + RTT_NS + 2000 + FIXED_NS + wire,
         2000.0, "服务端要解析、查表、组回复"),
        ("RPC + 服务端缓存命中", FIXED_NS + RTT_NS + 500 + FIXED_NS + wire,
         500.0, "只省服务时间，协议开销不变"),
    ]
    print(f"{'路径':>30}{'延迟 us':>11}{'服务端 CPU ns':>14}{'单核 ops/s':>13}")
    for name, lat, cpu, _ in rows:
        rate = "∞（不占 CPU）" if cpu == 0 else f"{1e9 / cpu / 1e3:.0f}K"
        print(f"{name:>30}{lat / 1000:>11.2f}{cpu:>14.0f}{rate:>13}")
    print()
    for name, lat, cpu, note in rows:
        print(f"  {name}：{note}")
    print("\n  关键差别不在延迟（两条路都受 RTT 支配，差 3.5 us 是服务时间），")
    print("  而在**服务端 CPU**：单边 READ 把「一次 RPC 的服务成本」变成 0，")
    print("  服务端连中断都不用进。这就是 KV 用单边语义的全部理由：不是快，是省机器。")
    print(f"  按每条 RPC 消耗 2 us 服务端 CPU、500 K ops/s/核 算：")
    print(f"    1 M ops/s 需要 {1e6 / 5e5:.0f} 个核；换成单边 READ 后需要 0 个核。")


# ---------------------------------------------------------------- 3. rkey 代价
def table_rkey_cost(size_b=4096):
    print("\n== 3. 单边 READ 的正确性代价：rkey 有效期 ==")
    print("   客户端拿到 rkey 后直接读服务端内存；服务端不知道谁在读、读到哪一刻。")
    print("   所以「释放/复用这块内存」必须等够长：\n")
    wire = size_b / BYTES_PER_NS
    read_ns = FIXED_NS + RTT_NS + wire
    print(f"   一次 READ 的在途时间 = 固定 {FIXED_NS:.0f} + RTT {RTT_NS:.0f}"
          f" + 线上 {wire:.0f} = {read_ns:.0f} ns")
    print("\n   「释放后仍可能被读到」的那段时间里，有几块 buffer 处于这个状态：")
    print("   slot 数 = 目标速率 x 在途时间\n")
    print(f"{'目标 ops/s':>11}{'RTT 3us slot':>14}{'缓冲 KiB':>10}"
          f"{'RTT 500us slot':>16}{'缓冲 MiB':>10}")
    for r in (100_000, 1_000_000, 10_000_000):
        for_3 = r * read_ns * 1e-9
        for_500 = r * (FIXED_NS + 500_000 + wire) * 1e-9
        print(f"{r:>11}{for_3:>14.1f}{for_3 * size_b / 1024:>10.1f}"
              f"{for_500:>16.0f}{for_500 * size_b / (1 << 20):>10.1f}")
    print("\n  两个结论：")
    print("    1) **内存不是主要代价**：正常 RTT 下只有几十块 buffer（KiB 量级）。")
    print("       即使 RTT 抖到 500 us，10 M ops/s 也只有约 5000 块 = 20 MiB —— 疼但可算。")
    print("       真正的代价是**正确性窗口**，不是内存。")
    print("    2) 如果服务端「释放即复用」，那么在 t 时刻释放的 buffer，")
    print("       到 t+在途时间 之前仍可能被读 —— 读到的却是新写入的内容。")
    print("       **没有错误**：合法内存、合法长度、错误内容。")
    print("       所以生命周期必须绑在 **key 失效** 上，不能绑在定时器上：")
    print("       用 FRWR + LOCAL_INV（39 章），读完成才作废 key。")
    print("\n  这就是 Q50 的答案核心：单边语义省 CPU，但把「生命周期管理」")
    print("  从「服务端串行处理」搬到了「客户端的并发假设」上。")
    print("  账：省 2 us CPU/op，付出一个 RTT 的正确性窗口 + 可撤销 key 的复杂度。")


# ---------------------------------------------------------------- 4. 写入
def table_write_path():
    print("\n== 4. PUT 路径：WRITE 必须配 credit（接 44 章）==")
    print(f"   服务端 slot ring；消费停顿时长决定需要的环深\n")
    print(f"{'停顿 ms':>9}{'需要条数':>12}{'容量 MiB':>11}{'无 credit 的后果':>22}")
    for stall_ms in (0.1, 1.0, 10.0):
        # 50 B/ns x stall_ms x 1e6 ns/ms / 4096
        n = int(BYTES_PER_NS * stall_ms * 1e6 / 4096)
        mib = n * 4096 / (1 << 20)
        print(f"{stall_ms:>9.1f}{n:>12}{mib:>10.1f}"
              f"{'静默覆盖未消费数据':>22}")
    print("\n  读法：PUT 走 WRITE 时，服务端「消费停顿」这件事发送端完全不知道，")
    print("  所以必须由服务端显式归还 credit（44 章）。没有 credit 的 WRITE 会在停顿期间覆盖。")
    print("  而 credit 环深按「最坏停顿」算 —— 1 ms 停顿在 400G 上就是 47.7 MiB/连接。")


# ---------------------------------------------------------------- 5. 全图
def table_assembly():
    print("\n== 5. 把 33-46 的账本合到一张图上 ==")
    print("""
  一个 RDMA KV 节点的完整路径与对应章节：

    客户端                                          服务端
    ─────────────────────────────────────────────────────────────
    [1] 建连/换状态        RESET→INIT→RTR→RTS          34 章
    [2] 拿到 rkey（GET）   带版本与租约                  39/47 章
    [3] 填 WQE（inline）   350+150 ns                   43/46 章
    [4] 上线 + 交换机      传播 + 队列                  35 章
    [5] 单边 READ          ── 服务端不参与 ──            47 章
    [6] 产 CQE + poll      450 ns                       46 章
    ─────────────────────────────────────────────────────────────
    PUT 路径额外需要：
    [P1] credit 归还        环深 = 最坏停顿 x 线速        44 章
    [P2] 内存注册           FRWR/MW 而非每次 reg_mr      39 章
    [P3] 失败处理           RNR / flush / 心跳用 SEND     42/44 章

  选型结论（对应 Q49-Q53）：
    请求/响应  -> SEND（有背压，天然并发上限）
    大块读     -> 单边 READ（省服务端 CPU，代价是 rkey 生命周期）
    大块写     -> WRITE + credit（省 CPU 但要自己做流控）
    海量小消息 -> SEND + inline + batching + 多 QP（46 章）
    CPU 是瓶颈 -> RDMA；协议要可改 -> DPDK（45 章）
""")


if __name__ == "__main__":
    table_concurrency()
    table_get_paths()
    table_rkey_cost()
    table_write_path()
    table_assembly()
