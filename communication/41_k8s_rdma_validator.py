#!/usr/bin/env python3
"""Kubernetes 里跑 RDMA 的配置校验器：每一项检查对应一个真实故障症状。

    python3 41_k8s_rdma_validator.py

它不做任何集群操作，只对"一份配置"做规则判定。目的是把
「这样配会死在哪里」变成可执行的检查表，而不是靠记忆。

用法（改 main 里的场景即可）：
    python3 41_k8s_rdma_validator.py
"""

from dataclasses import dataclass, field
from enum import Enum


class Level(Enum):
    OK = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"


@dataclass
class Cfg:
    name: str
    # --- 资源形态 ---
    rdma_mode: str = "sriov"          # sriov | shared | hostnet
    vf_count: int = 8                 # 节点上暴露的 VF 数
    pods: int = 1                     # 需要 RDMA 的 pod 数
    # --- 容器权限 ---
    memlock: str = "unlimited"        # unlimited | 64KiB | ...
    ipc_lock: bool = True             # securityContext.capabilities 是否含 IPC_LOCK
    devices_mounted: bool = True      # /dev/infiniband 是否挂进容器
    # --- 网络 ---
    cni: str = "multus"               # multus | none
    netns_shared_with_host: bool = False
    # --- GID ---
    gid_index: int = 3
    gid_type: str = "RoCE v2"         # RoCE v2 | RoCE v1 | IB/RoCE v1
    # --- 多租户 ---
    tenants: int = 1
    separate_vf_per_tenant: bool = True
    pkey: str = "0xffff"


@dataclass
class Result:
    level: Level
    check: str
    symptom: str


def check_memlock(c: Cfg) -> Result:
    if c.memlock == "unlimited":
        return Result(Level.OK, "memlock=unlimited", "")
    return Result(
        Level.FAIL,
        f"memlock={c.memlock}",
        "ibv_reg_mr 返回 ENOMEM；小 MR 可能侥幸成功，一大页就失败",
    )


def check_ipc_lock(c: Cfg) -> Result:
    if c.ipc_lock:
        return Result(Level.OK, "capabilities: IPC_LOCK", "")
    return Result(
        Level.FAIL,
        "缺 IPC_LOCK",
        "mlock/大页分配失败；表现为进程启动时报 EPERM，而不是网络错误",
    )


def check_devices(c: Cfg) -> Result:
    if c.devices_mounted:
        return Result(Level.OK, "/dev/infiniband 已挂载", "")
    return Result(
        Level.FAIL,
        "缺 /dev/infiniband",
        "ibv_get_device_list 返回 0 个设备；应用报 'no RDMA devices found'",
    )


def check_cni(c: Cfg) -> Result:
    if c.netns_shared_with_host:
        return Result(
            Level.WARN,
            "hostNetwork=true",
            "能跑通，但 pod 与宿主共享 netns：GID/路由互相污染，且失去网络隔离",
        )
    if c.cni == "multus":
        return Result(Level.OK, "Multus + RDMA 设备网络", "")
    return Result(
        Level.FAIL,
        "没有给 pod 挂 RDMA 网络",
        "pod 默认网卡是 veth，走内核协议栈，RDMA 完全不可用",
    )


def check_gid(c: Cfg) -> Result:
    if c.gid_type != "RoCE v2":
        return Result(
            Level.FAIL,
            f"gid_index={c.gid_index} 是 {c.gid_type}",
            "对端是 RoCEv2 时握手失败或静默回退到 TCP；ibv_modify_qp 可能返回 EINVAL",
        )
    if c.rdma_mode == "shared" and c.gid_index == 0:
        return Result(
            Level.WARN,
            "gid_index=0",
            "index 0 常是 link-local 地址；多 rail 机器上可能选到错误的那条",
        )
    return Result(Level.OK, f"gid_index={c.gid_index} ({c.gid_type})", "")


def check_capacity(c: Cfg) -> Result:
    if c.rdma_mode == "hostnet":
        return Result(
            Level.WARN,
            "hostNetwork 模式",
            f"{c.pods} 个 pod 抢同一块物理口：没有 VF 隔离，QoS/限速只能靠交换机",
        )
    if c.rdma_mode == "shared":
        return Result(
            Level.OK,
            "共享设备插件",
            f"{c.pods} 个 pod 共享同一批设备，靠 time-slicing；适合推理不适合训练",
        )
    if c.vf_count >= c.pods:
        return Result(Level.OK, f"{c.vf_count} VF >= {c.pods} pod", "")
    return Result(
        Level.FAIL,
        f"{c.vf_count} VF < {c.pods} pod",
        f"{c.pods - c.vf_count} 个 pod 卡在 Pending（Insufficient rdma/...）",
    )


def check_isolation(c: Cfg) -> Result:
    if c.tenants <= 1:
        return Result(Level.OK, "单租户", "")
    if c.separate_vf_per_tenant:
        return Result(
            Level.OK,
            f"{c.tenants} 租户各占独立 VF",
            "VF 是最强的隔离边界：独立的 QP 空间、独立的 GID",
        )
    return Result(
        Level.FAIL,
        f"{c.tenants} 租户共用一个 VF",
        "PD 只在进程内做命名空间隔离，不是硬件边界："
        "拿到 rkey 就能读写同 VF 内其他进程的内存",
    )


def check_pkey(c: Cfg) -> Result:
    if c.pkey == "0xffff":
        return Result(
            Level.OK,
            "pkey=0xffff（full membership）",
            "默认成员，能和不设 pkey 的对端通",
        )
    return Result(
        Level.WARN,
        f"pkey={c.pkey}",
        "需要 SM/交换机侧配置一致；RoCE 下多数部署不用 pkey 做隔离",
    )


CHECKS = [
    check_memlock,
    check_ipc_lock,
    check_devices,
    check_cni,
    check_gid,
    check_capacity,
    check_isolation,
    check_pkey,
]


def evaluate(c: Cfg) -> list[Result]:
    return [fn(c) for fn in CHECKS]


def report(c: Cfg) -> bool:
    print("\n" + "=" * 74)
    print(f"场景：{c.name}")
    print("=" * 74)
    results = evaluate(c)
    for r in results:
        mark = {"PASS": "  ok ", "WARN": " warn", "FAIL": " FAIL"}[r.level.value]
        print(f"{mark}  {r.check}")
        if r.symptom:
            print(f"        -> {r.symptom}")
    fails = [r for r in results if r.level is Level.FAIL]
    warns = [r for r in results if r.level is Level.WARN]
    print(f"\n结论：{len(fails)} FAIL / {len(warns)} WARN / "
          f"{len(results) - len(fails) - len(warns)} PASS", end="")
    if fails:
        print("  => 这个配置跑不起来，先修 FAIL")
    elif warns:
        print("  => 能跑，但有隐患")
    else:
        print("  => 配置自洽")
    return not fails


def capacity_table():
    print("\n== 附：每节点能放多少个 RDMA pod ==")
    print(f"{'模式':>12}{'VF/口':>8}{'口数':>6}{'可用 pod':>10}   说明")
    rows = [
        ("SR-IOV", 8, 1, 8, "VF 数 = 上限；每 pod 独占一个 VF"),
        ("SR-IOV", 32, 2, 64, "双口每口 32 VF"),
        ("共享插件", 0, 1, 32, "时间片共享，不保证带宽"),
        ("hostNetwork", 0, 1, 1, "整个节点只能有一个 pod 用这块口"),
    ]
    for mode, vf, ports, pods, note in rows:
        print(f"{mode:>12}{vf:>8}{ports:>6}{pods:>10}   {note}")
    print("\n  注意 VF 数是硬上限：超过就 Pending，不会降级到共享。")


def main():
    scenarios = [
        Cfg(name="单租户训练 pod（配对了）"),
        Cfg(name="忘了 memlock（最常见的第一个坑）", memlock="64KiB"),
        Cfg(name="用 hostNetwork 图省事", rdma_mode="hostnet",
            netns_shared_with_host=True),
        Cfg(name="GID index 选错（RoCEv1 的地址）", gid_type="RoCE v1",
            gid_index=0),
        Cfg(name="VF 不够，超卖", vf_count=4, pods=8),
        Cfg(name="多租户共用同一 VF", tenants=4, pods=4,
            separate_vf_per_tenant=False),
        Cfg(name="共享设备插件跑推理", rdma_mode="shared", gid_index=0, pods=16),
    ]
    ok = 0
    for s in scenarios:
        if report(s):
            ok += 1
    print(f"\n{len(scenarios)} 个场景里 {ok} 个配置自洽。")
    capacity_table()


if __name__ == "__main__":
    main()
