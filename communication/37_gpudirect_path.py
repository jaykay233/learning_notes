#!/usr/bin/env python3
"""GPUDirect 路径判定与带宽账本。

    python3 37_gpudirect_path.py

回答三个问题：
  1. 这条拓扑上 P2P 到底通不通？
  2. 不通时退化成什么，多付多少？
  3. 一次 AllReduce 里这个退化占多少比例？

常量标记：
  [规范]  PCIe 代际的理论线速率
  [假设]  实际可达比例、延迟量级（会改绝对值，不改结论）
  [锚点]  与 33 章的固定成本模型对齐
"""

# ---------------------------------------------------------------- [规范]
PCIE = {
    # 单向 GB/s（理论）
    "gen3 x16": 15.75,
    "gen4 x16": 31.5,
    "gen5 x16": 63.0,
}
# 实际可达比例（编码开销、TLP、流控）
EFFICIENCY = 0.80

# ---------------------------------------------------------------- [假设]
# bounce 路径：NIC -> host 内存 -> GPU，两段 PCIe，且要 CPU 参与一次拷贝
BOUNCE_EXTRA_COPY = True
# 单次 bounce 的额外延迟（DMA 启动 + memcpy 调度），量级
BOUNCE_LATENCY_US = 3.0
# GDR 路径相对 bounce 的额外延迟（建立映射后的稳态）
GDR_LATENCY_US = 0.0


class Topology:
    def __init__(self, name, gpu_nic_same_switch, acs_enabled, nvlink_to_nic,
                 pcie="gen4 x16", nic_gbps=200, p2p_supported=True):
        self.name = name
        self.same_switch = gpu_nic_same_switch
        self.acs = acs_enabled
        self.nvlink = nvlink_to_nic
        self.pcie = pcie
        self.nic_gbps = nic_gbps
        self.p2p_supported = p2p_supported

    def p2p_ok(self):
        """P2P 是否可用。"""
        if not self.p2p_supported:
            return False, "平台/驱动不支持 P2P"
        if self.nvlink:
            return True, "NVLink 直连网卡，绕开 PCIe ACS"
        if self.acs:
            return False, "ACS 开启，P2P 被 IOMMU 拦在 switch 边界"
        if not self.same_switch:
            return False, "跨 PCIe switch 且未做 P2P 定向"
        return True, "同一 PCIe switch，ACS 关闭"


def bandwidth_gbps(topo):
    """返回 (有效单向带宽 Gbps, 路径名, 延迟 us)。"""
    nic_bw = topo.nic_gbps
    if topo.nvlink:
        # NVLink 到网卡：瓶颈通常在网卡
        return nic_bw, "NVLink→NIC", GDR_LATENCY_US

    ok, _ = topo.p2p_ok()
    link = PCIE[topo.pcie] * EFFICIENCY  # GB/s
    link_gbps = link * 8                # Gbps
    if ok:
        return min(nic_bw, link_gbps), "GDR (P2P)", GDR_LATENCY_US
    # bounce：数据要过两段 PCIe（NIC→host，host→GPU），吞吐按一半算
    return min(nic_bw, link_gbps) / 2, "bounce buffer", BOUNCE_LATENCY_US


def table_paths(topos):
    print("== 1. 常见拓扑的 P2P 判定 ==")
    print(f"{'拓扑':>26}{'P2P':>6}{'路径':>16}{'有效 Gbps':>12}{'延迟 us':>9}")
    out = []
    for t in topos:
        ok, _ = t.p2p_ok()
        bw, path, lat = bandwidth_gbps(t)
        out.append((t, ok, path, bw, lat))
        print(f"{t.name:>26}{'通' if ok else '不通':>6}{path:>16}{bw:>11.0f}{lat:>9.1f}")
    return out


def table_reasons(topos):
    print("\n== 2. 每个拓扑的判定依据 ==")
    for t in topos:
        ok, why = t.p2p_ok()
        print(f"  {t.name:<26} {'通' if ok else '不通'}  <- {why}")


def table_cost(topos, total_mb=1024.0, iterations=1):
    print(f"\n== 3. 传 {total_mb:.0f} MB 的时间差（{iterations} 次）==")
    print(f"{'拓扑':>26}{'GDR 时间 ms':>14}{'实际时间 ms':>14}{'倍数':>8}")
    base_bw, _, _ = bandwidth_gbps([t for t in topos if t.nvlink][0])
    for t in topos:
        bw, path, _ = bandwidth_gbps(t)
        # MB × 8 -> Mbit；Mbit / Gbps -> 毫秒（1e6 bit / 1e9 bit/s = 1e-3 s）
        gdr_ms = total_mb * 8 / base_bw
        act_ms = total_mb * 8 / bw
        print(f"{t.name:>26}{gdr_ms:>14.1f}{act_ms:>14.1f}{act_ms / gdr_ms:>7.1f}x")
    print("  'GDR 时间' 是所有拓扑都理想的参照线（NVLink→NIC 那条）。")


def table_allreduce(total_mb=1024.0, ranks=8):
    print(f"\n== 4. 放进一次 AllReduce：{total_mb:.0f} MB × {ranks} rank ==")
    # ring AllReduce 每 rank 发送 2*(N-1)/N * size
    factor = 2 * (ranks - 1) / ranks
    print(f"   ring 每 rank 发送量 = 2×(N-1)/N × size = {factor:.2f} × {total_mb:.0f} MB"
          f" = {factor * total_mb:.1f} MB")
    good, bad = 400.0, 200.0   # Gbps
    t_good = factor * total_mb * 8 / good / 1000 * 1000
    t_bad = factor * total_mb * 8 / bad / 1000 * 1000
    print(f"   GDR 400 Gbps  : {t_good:.1f} ms")
    print(f"   bounce 200 Gbps:{t_bad:.1f} ms")
    print(f"   退化损失       : {t_bad - t_good:.1f} ms（{t_bad / t_good:.1f}x）")
    print("   注意这是稳态带宽账；短消息场景里退化主要体现为延迟，不是带宽。")


if __name__ == "__main__":
    topos = [
        Topology("GPU/NIC 同 switch, ACS off", True, False, False,
                 "gen4 x16", 200),
        Topology("GPU/NIC 同 switch, ACS on", True, True, False,
                 "gen4 x16", 200),
        Topology("GPU/NIC 跨 switch, ACS off", False, False, False,
                 "gen4 x16", 200),
        Topology("NVLink 直连网卡", False, True, True,
                 "gen5 x16", 400),
        Topology("驱动不支持 P2P", True, False, False,
                 "gen4 x16", 200, p2p_supported=False),
    ]
    table_paths(topos)
    table_reasons(topos)
    table_cost(topos)
    table_allreduce()
