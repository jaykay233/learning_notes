#!/usr/bin/env python3
"""RDMA 排障判定器：把计数器变成排序后的假设列表。

    python3 36_diagnose.py            # 跑三个内置场景
    python3 36_diagnose.py --demo pfc # 只跑一个

设计原则（36 章的核心）：
  1. 先分层：包发出去没有 -> 对端收到没有 -> 应用消费没有
  2. 计数器只回答前两层，应用问题必须靠栈和日志
  3. 每个假设都带"下一步命令"，避免猜

这不是专家系统，是把 54 题里可判定的那部分写成规则。
规则命中不代表确诊，只代表"下一步该测什么"。
"""

from dataclasses import dataclass, field


# pause 停顿占比的经验告警线。
# 1% 就值得看一眼，5% 说明这条链路基本靠 pause 活着。
PAUSE_ALERT_PCT = 5.0


@dataclass
class Obs:
    """一次现场观察。字段为 None 表示没测。"""

    # 现象
    symptom: str = ""            # 用户描述
    # 端口与链路
    port_state: str = "ACTIVE"   # ACTIVE / DOWN / INIT
    symbol_err: int = 0          # symbol_error 增量
    rcv_err: int = 0             # port_rcv_errors
    # 传输层
    pkt_seq_err: int = 0
    out_of_seq: int = 0
    ack_timeout: int = 0
    xmit_discards: int = 0
    # 拥塞
    pause_frames_per_s: float = 0.0
    pause_max_quanta: int = 0
    ecn_marked_pct: float = 0.0
    # QP / 完成
    wc_status: str = ""
    qp_state: str = ""           # RTS / ERROR / ...
    cqe_seen: bool = True
    # 应用
    pending_recv: int = -1       # -1 = 未知
    threads_polling_cq: int = 1
    # 性能
    msg_size_b: int = 0
    link_gbps: float = 100.0
    goodput_gbps: float = 0.0


@dataclass
class Hypothesis:
    rank: int
    layer: str
    cause: str
    evidence: str
    next_cmd: str
    score: float = 0.0


def _pause_time_pct(o: Obs) -> float:
    """pause 占用比例 = 每秒 pause 条数 × 每条时长 / 1 s。"""
    ns_per_quanta = 512.0 / (o.link_gbps * 1e9) * 1e9
    return (o.pause_frames_per_s * o.pause_max_quanta * ns_per_quanta / 1e9) * 100


def diagnose(o: Obs) -> list:
    h = []

    # ---------------- L0/L4：链路层
    if o.port_state != "ACTIVE":
        h.append(Hypothesis(
            0, "L1 对象", f"端口未 ACTIVE（当前 {o.port_state}）",
            "modify_qp 到 RTR 会失败，或 QP 直接 ERROR",
            "ibstat；检查线缆/光模块/交换机端口 disable", 100))

    if o.symbol_err > 0 or o.rcv_err > 0:
        h.append(Hypothesis(
            0, "L0 物理", "物理层误码（线缆/光模块/连接器）",
            f"symbol_error={o.symbol_err}, port_rcv_errors={o.rcv_err}",
            "perfquery -x；换线/换模块；看误码是否随流量增长", 95))

    # ---------------- L4：拥塞
    pct = _pause_time_pct(o)
    if pct > PAUSE_ALERT_PCT:
        h.append(Hypothesis(
            0, "L4 网络", "PFC 反压占比过高",
            f"{o.pause_frames_per_s:.0f} 条/s × {o.pause_max_quanta} quanta "
            f"≈ 停顿 {pct:.1f}% 时间",
            "逐跳查 pause 源：先查本端口 pause 计数，再往上游/下游找慢端口", 88))

    if o.xmit_discards > 0 and o.ecn_marked_pct < 1.0:
        h.append(Hypothesis(
            0, "L4 网络", "队列溢出但 ECN 几乎没标记：门限过高或 ECN 没开",
            f"xmit_discards={o.xmit_discards}, ecn_marked={o.ecn_marked_pct}%",
            "查交换机 Kmin/Kmax/Pmax；确认 DSCP 映射到了正确队列", 86))

    if o.ecn_marked_pct > 20:
        h.append(Hypothesis(
            0, "L4 网络", "ECN 过度标记：门限过低或 Pmax 过高",
            f"标记率 {o.ecn_marked_pct}%",
            "抬高 Kmin 或降低 Pmax；对比队列均值是否被压得过低", 70))

    # ---------------- L3：可靠性
    if o.pkt_seq_err > 0 or o.out_of_seq > 0:
        h.append(Hypothesis(
            0, "L3 可靠性", "丢包/乱序，触发 go-back-N 重传",
            f"packet_seq_err={o.pkt_seq_err}, out_of_seq={o.out_of_seq}",
            "查 PFC 是否漏配、ECN 是否误标；对比重传放大倍数", 90))

    if o.ack_timeout > 0:
        h.append(Hypothesis(
            0, "L3 可靠性", "ACK 超时：对端不应答或链路单向不通",
            f"local_ack_timeout_err={o.ack_timeout}",
            "ping 对端 GID；查对端 QP 是否 ERROR；查反向路径", 84))

    status_rules = {
        "IBV_WC_RETRY_EXC_ERR": (
            "重传耗尽：包发得出去，对端不应答",
            "查对端 QP 状态、链路、retry_cnt；确认对端进程还活着",
            92),
        "IBV_WC_RNR_RETRY_EXC_ERR": (
            "对端没有可用 recv WR（Receiver Not Ready）",
            "查接收端投递逻辑：recv WR 数量 vs 在途 SEND 数",
            96),
        "IBV_WC_WR_FLUSH_ERR": (
            "QP 已 ERROR，挂起 WR 被连带清空",
            "往前找第一个非 flush 的错误——它才是根因",
            80),
        "IBV_WC_REM_ACCESS_ERR": (
            "对端拒绝：rkey / 权限 / 越界（DC 下也可能是 DC key 错）",
            "核对 rkey 与 MR 权限位；DC 场景先查 AV 里的 DC key",
            94),
        "IBV_WC_REM_INV_REQ_ERR": (
            "对端认为请求非法：QP 状态或参数不匹配",
            "核对 path MTU、max_dest_rd_atomic、QP 类型",
            88),
        "IBV_WC_LOC_PROT_ERR": (
            "本地保护错误：lkey 与 MR 不匹配或跨 PD",
            "确认 lkey 属于同一 PD 的 MR；检查 SGE 越界",
            90),
        "IBV_WC_LOC_LEN_ERR": (
            "本地长度错误：SGE 长度与 WR 不一致",
            "累加 SGE 长度是否等于 wr.length；检查 MR 边界",
            88),
        "IBV_WC_LOC_ACCESS_ERR": (
            "本地访问权限不足",
            "检查 MR 注册时的 access flags（本地写需要 LOCAL_WRITE）",
            86),
    }
    if o.wc_status in status_rules:
        cause, nxt, sc = status_rules[o.wc_status]
        h.append(Hypothesis(0, "L3 可靠性", cause, f"wc.status={o.wc_status}", nxt, sc))

    if not o.cqe_seen:
        h.append(Hypothesis(
            0, "L5 工程", "收不到 CQE：信号或 CQ 归属问题",
            "无 completion 返回",
            "查 sq_sig_all / IBV_SEND_SIGNALED；查 req_notify_cq 是否在 poll 不到后重新 arm",
            85))

    if o.threads_polling_cq > 1:
        h.append(Hypothesis(
            0, "L5 工程", "多线程竞争同一个 CQ，CQE 被别的线程消费",
            f"{o.threads_polling_cq} 个 poller",
            "按 CQE 分片或加锁；确认 poll 的是本 QP 的 CQ", 75))

    if o.pending_recv == 0:
        h.append(Hypothesis(
            0, "L2 语义", "接收端 recv WR 已耗尽",
            "pending_recv=0，SEND 必然 RNR",
            "补投 recv WR；或改用 WRITE + WITH_IMM 去掉 recv 依赖", 97))

    # ---------------- L5：带宽
    if o.msg_size_b and o.goodput_gbps:
        eff = o.goodput_gbps / o.link_gbps
        if o.msg_size_b < 65536 and eff < 0.5:
            h.append(Hypothesis(
                0, "L5 工程", "小消息固定成本主导，不是硬件故障",
                f"{o.msg_size_b} B 消息，有效带宽 {eff:.0%}",
                "算 1/(固定成本+线上时间)；用 inline / doorbell batching / 多 QP", 72))

    if not h:
        h.append(Hypothesis(0, "无法判定", "现有观测不足以定位",
                            "所有已测信号正常", "补测：网卡计数器、对端 QP 状态、应用侧栈", 10))

    h.sort(key=lambda x: -x.score)
    for i, x in enumerate(h, 1):
        x.rank = i
    return h


def show(name: str, o: Obs) -> None:
    print(f"\n{'=' * 74}")
    print(f"场景：{name}")
    print(f"{'=' * 74}")
    for hyp in diagnose(o):
        print(f"  [{hyp.rank}] ({hyp.layer}) {hyp.cause}")
        print(f"      证据：{hyp.evidence}")
        print(f"      下一步：{hyp.next_cmd}")


SCENARIOS = {
    "rnr": ("SEND 卡住，poll 拿到错误",
            Obs(symptom="hang in poll", wc_status="IBV_WC_RNR_RETRY_EXC_ERR",
                pending_recv=0)),
    "pfc": ("延迟抖动，带宽正常",
            Obs(symptom="p99 抖动毫秒级", pause_frames_per_s=300,
                pause_max_quanta=60000, link_gbps=100,
                ecn_marked_pct=0.0)),
    "small": ("带宽只有 5 Gbps",
              Obs(symptom="ib_write_bw 跑不满", msg_size_b=1024,
                  goodput_gbps=5.2, link_gbps=100)),
    "phy": ("间歇性重传，无拥塞",
            Obs(symptom="偶发 RETRY_EXC", symbol_err=42, rcv_err=17,
                pkt_seq_err=31, link_gbps=100)),
}


if __name__ == "__main__":
    import sys

    only = sys.argv[2] if len(sys.argv) > 2 and sys.argv[1] == "--demo" else None
    for key, (name, obs) in SCENARIOS.items():
        if only and key != only:
            continue
        show(name, obs)
    print(f"\n{'=' * 74}")
    print("提醒：规则命中只代表下一步该测什么，不代表确诊。")
    print("      每个假设都给了命令，跑完命令再回来更新观测。")
