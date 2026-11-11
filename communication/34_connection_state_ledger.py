#!/usr/bin/env python3
"""连接状态账本：RC / SRQ / XRC / DC 各自为 O(N^2) 付哪一段。

配合 34-rdma-connection-state-ledger.md 使用，纯算术，不需要 RDMA 硬件。

    python3 34_connection_state_ledger.py

区分三类常量：
  [规范/PRM]  来自 IBTA 规范或 mlx5 PRM，可查证
  [假设]      公开资料不给确切值，取量级；会改绝对值，不改结论
  [锚点]      来自 21/22 章引用的实测数字，用来校准模型
"""

# ---------------------------------------------------------------- [规范/PRM]
QP_CTX_B = 512          # mlx5 QP context 对齐单位
DCT_CTX_B = 512         # DCT context 量级
RECV_WQE_B = 16         # recv WQE segment
RECV_DEPTH = 64         # 默认 recv WR 深度
RC_WQE_B = 48           # RC pointer WQE
DC_WQE_B = 64           # DC pointer WQE（多 16 B AV）

# ---------------------------------------------------------------- [假设]
HIT_NS = 20
MISS_NS = 80

# ---------------------------------------------------------------- [锚点]
DC_SWITCH_NS = 450      # 22 章：DC 单笔比 RC 贵 0.45 us
ONSET_CONN = 1350       # 22 章：128-PE dense sweep 的退化起点
FIXED_1K_NS = 1500      # 33 章：1 KB 消息固定成本


# ================================================================ 1. 四段账本
def rc_bytes(n_peers: int) -> dict:
    ctx = n_peers * QP_CTX_B
    ring = n_peers * RECV_DEPTH * RECV_WQE_B
    return {"context": ctx, "recv_ring": ring, "total": ctx + ring}


def srq_bytes(n_peers: int) -> dict:
    ctx = n_peers * QP_CTX_B
    ring = RECV_DEPTH * RECV_WQE_B
    return {"context": ctx, "recv_ring": ring, "total": ctx + ring}


def xrc_bytes(n_peers: int) -> dict:
    send_ctx = n_peers * QP_CTX_B
    recv_ctx = QP_CTX_B
    recv_ring = RECV_DEPTH * RECV_WQE_B
    return {
        "send_context": send_ctx,
        "recv_context": recv_ctx,
        "recv_ring": recv_ring,
        "total": send_ctx + recv_ctx + recv_ring,
    }


def dc_bytes(n_peers: int, n_dci: int = 8) -> dict:
    dci_ctx = n_dci * QP_CTX_B
    dct_ctx = n_peers * DCT_CTX_B
    return {
        "dci_context": dci_ctx,
        "av_per_wqe": DC_WQE_B - RC_WQE_B,
        "dct_context": dct_ctx,
        "total": dci_ctx + dct_ctx,
    }


def table_footprint(nodes=(8, 64, 128, 1024, 3000)) -> None:
    print("== 1. 每节点连接状态（全连接，n_peers = N-1），单位 KiB ==")
    print(f"{'N':>6}{'peer':>7}{'RC':>10}{'SRQ':>10}{'XRC(发+收)':>13}{'DC':>10}")
    for n in nodes:
        p = n - 1
        print(
            f"{n:>6}{p:>7}"
            f"{rc_bytes(p)['total'] / 1024:>10.0f}"
            f"{srq_bytes(p)['total'] / 1024:>10.0f}"
            f"{xrc_bytes(p)['total'] / 1024:>13.0f}"
            f"{dc_bytes(p)['total'] / 1024:>10.0f}"
        )
    print("  SRQ 只把 recv ring 从 O(N) 降到 O(1)，context 仍是 O(N)。")
    print("  XRC 把接收侧 context 降到 O(1)；DC 把发起侧持久 context 降到 O(n_dci)。")


# ================================================================ 2. 容量 vs 驻留
def table_two_tier(cache_b: int = 512 << 10) -> None:
    print("\n== 2. 两级存储：ICM 容量 vs 片上 context cache ==")
    print("   ICM 是 DRAM backing，GB 级，装得下；")
    print("   context cache 是片上 SRAM，几百 KiB 级，装不下就每次 miss。")
    cap = cache_b // QP_CTX_B
    print(f"\n   假设片上 cache = {cache_b / 1024:.0f} KiB -> 可驻留 {cap} 个 512 B context")
    print(f"{'N':>6}{'peer':>7}{'ICM KiB':>12}{'工作集 KiB':>14}{'驻留比':>10}{'判断':>8}")
    for n in (128, 512, 1024, 1350, 3000):
        p = n - 1
        ws = rc_bytes(p)["context"] / 1024
        ratio = min(1.0, cap * QP_CTX_B / 1024 / ws) if ws else 1.0
        verdict = "命中" if ws <= cap * QP_CTX_B / 1024 else "抖动"
        print(f"{n:>6}{p:>7}{ws:>12.0f}{ws:>14.0f}{ratio:>9.0%}{verdict:>8}")
    print("   驻留比 < 100% 就进入「每次重新加载」区间，这是 22 章 59% 损失的来源。")
    print("   注意：ICM 那一列从头到尾都装得下，所以按 ICM 容量论证是错的。")


def table_crossover() -> None:
    print("\n== 3. 交叉点：多少连接才开始抖动 ==")
    print("   驻留上限 N* = cache / context_size")
    caches = [(128, "128 KiB"), (256, "256 KiB"), (512, "512 KiB"), (1024, "1 MiB")]
    ctxs = [(512, "512 B"), (1024, "1 KiB"), (2048, "2 KiB")]
    print(f"{'cache':>10}" + "".join(f"{c[1]:>12}" for c in ctxs))
    for kb, label in caches:
        row = f"{label:>10}"
        for b, _ in ctxs:
            row += f"{kb * 1024 // b:>12}"
        print(row)
    print('   校准：取 cache=512 KiB、ctx=512 B -> N*=1024；')
    print(f"   实测 onset 约 {ONSET_CONN}，同量级，说明模型方向对、量级对。")
    print("   若把 ctx 当 4 KiB（带更多 inline state），N* 立刻掉到 128，")
    print("   这解释了为什么换个网卡/驱动，拐点位置就变了。")


# ================================================================ 4. 摊销
def table_reuse(visits=(1, 32, 256, 1024, 2048, 8192)) -> None:
    print("\n== 4. 每次连接访问写多少笔，才能把 context 加载成本摊掉 ==")
    wire_1k = 1024 * 8 / 100e9 * 1e9  # 81.9 ns
    print(f"{'writes/visit':>14}{'每笔均摊 ns':>14}{'占 1 KB@100G 线上时间':>24}")
    for v in visits:
        per = MISS_NS / v
        print(f"{v:>14}{per:>14.2f}{per / wire_1k:>22.0%}")
    print("  22 章「32 -> 8192 writes/visit 曲线压平」的机制就在这里：")
    print("  访问 1024 笔时，context 加载成本已低于 1 KB 的线上时间。")


def table_dc_switch(bursts=(1, 4, 16, 64, 256)) -> None:
    print("\n== 5. DC 的 destination 切换分摊（锚点：每 WQE 换 = 450 ns）==")
    print(f"{'per-destination burst':>22}{'每笔均摊 ns':>14}{'占 1 KB 固定成本':>18}")
    for b in bursts:
        per = DC_SWITCH_NS / b
        print(f"{b:>22}{per:>14.1f}{per / FIXED_1K_NS:>17.0%}")
    print("  burst=1 时 DC 单笔贵 450 ns（30%）；burst>=64 时基本归还。")
    print("  22 章「每 WQE 换目标 -> message rate 掉 60 倍」就是 burst 退化成 1。")


def table_verdict() -> None:
    print("\n== 6. 谁砍掉了哪一段 ==")
    rows = [
        ("RC", "O(N)", "O(N)", "O(N)", "基准"),
        ("SRQ", "O(N)", "O(1)", "O(N)", "只省 ring 内存，不省 context"),
        ("XRC", "O(N)", "O(1)", "O(1)", "省接收侧 context，发起侧不变"),
        ("DC", "O(n_dci)", "O(1)", "O(N) DCT", "省发起侧 context，换 per-WQE AV"),
    ]
    print(f"{'方案':>5}{'发起 ctx':>11}{'接收 ctx':>11}{'recv ring':>12}  备注")
    for a, b, c, d, e in rows:
        print(f"{a:>5}{b:>11}{c:>11}{d:>12}  {e}")


if __name__ == "__main__":
    table_footprint()
    table_two_tier()
    table_crossover()
    table_reuse()
    table_dc_switch()
    table_verdict()
