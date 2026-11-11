#!/usr/bin/env python3
"""一个 RC QP 的状态机 + 队列模拟器，用来在不插网卡的机器上跑出真实 trace。

    python3 38_verbs_emulator.py

它模拟的对象（对应 38 章的正文）：
  ibv_modify_qp   四个状态的字段校验（填错就是 EINVAL）
  ibv_post_send   往 SQ 追加 WR
  ibv_post_recv   往 RQ 追加 recv WR
  ibv_poll_cq     消费 CQE
  RNR 重试        对端没有 recv WR 时的行为
  flush           QP 进 ERROR 后剩余 WR 的连带完成

边界：这是**语义模拟**，不建模 PCIe、网络延迟、真实网卡流水线。
它保证"状态与字段的因果关系"和真实 verbs 一致，不保证任何时间数字。
"""

from dataclasses import dataclass, field
from enum import Enum


class QPState(Enum):
    RESET = "RESET"
    INIT = "INIT"
    RTR = "RTR"
    RTS = "RTS"
    ERROR = "ERROR"


class Status(Enum):
    SUCCESS = "IBV_WC_SUCCESS"
    RNR_RETRY_EXC = "IBV_WC_RNR_RETRY_EXC_ERR"
    REM_ACCESS = "IBV_WC_REM_ACCESS_ERR"
    WR_FLUSH = "IBV_WC_WR_FLUSH_ERR"


@dataclass
class WQEEntry:
    wr_id: int
    opcode: str = "SEND"
    length: int = 0
    signaled: bool = True
    # 模拟非法参数：1 = rkey 错，2 = 越界
    bad: int = 0


@dataclass
class CQE:
    wr_id: int
    status: Status
    opcode: str


class QP:
    """最小 RC QP 模型。字段校验规则按 IBTA 的 modify_qp 要求。"""

    REQUIRED = {
        QPState.INIT: ("port_num", "pkey_index", "qp_access_flags"),
        QPState.RTR: ("dest_qp_num", "path_mtu", "rq_psn",
                      "max_dest_rd_atomic", "min_rnr_timer", "ah_attr"),
        QPState.RTS: ("sq_psn", "timeout", "retry_cnt", "rnr_retry",
                      "max_rd_atomic"),
        QPState.ERROR: (),
        QPState.RESET: (),
    }

    def __init__(self, name, retry_cnt=7, rnr_retry=6):
        self.name = name
        self.state = QPState.RESET
        self.attrs = {}
        self.sq = []
        self.rq = []
        self.cq = []
        self.retry_cnt = retry_cnt
        self.rnr_retry = rnr_retry
        self.sq_sig_all = True
        self.access = set()
        self.log = []
        self.trace = []

    # ---------------------------------------------------------- modify_qp
    def modify_qp(self, target, **attrs):
        err = transition_error(target, self.state)
        if err:
            raise ValueError(f"{target.value}: {err}")
        if target is QPState.RESET:
            self.state = QPState.RESET
            self.trace.append(f"modify_qp {self.name} -> RESET")
            return 0
        missing = [a for a in self.REQUIRED[target] if a not in attrs]
        if missing:
            raise ValueError(
                f"{target.value}: EINVAL，缺字段 {missing}"
            )
        if target == QPState.RTR and attrs.get("path_mtu", 0) > 4096:
            raise ValueError("RTR: EINVAL，path_mtu 超过设备能力")
        self.attrs.update(attrs)
        self.state = target
        self.trace.append(f"modify_qp {self.name} -> {target.value}")
        return 0

    # ---------------------------------------------------------- post
    def post_send(self, entry: WQEEntry):
        if self.state not in (QPState.RTS, QPState.ERROR):
            raise ValueError(f"post_send: QP 在 {self.state.value}，需要 RTS")
        if self.state == QPState.ERROR:
            # ERROR 下 post 的 WR 立刻以 flush 完成
            if entry.signaled:
                self.cq.append(CQE(entry.wr_id, Status.WR_FLUSH, entry.opcode))
            return
        self.sq.append(entry)

    def post_recv(self, entry: WQEEntry):
        if self.state == QPState.ERROR:
            if entry.signaled:
                self.cq.append(CQE(entry.wr_id, Status.WR_FLUSH, "RECV"))
            return
        self.rq.append(entry)

    # ---------------------------------------------------------- poll
    def poll_cq(self):
        if not self.cq:
            return None
        return self.cq.pop(0)

    def enter_error(self, reason):
        if self.state == QPState.ERROR:
            return
        self.state = QPState.ERROR
        self.trace.append(f"QP {self.name} -> ERROR（{reason}）")

    def flush_all(self):
        """QP 进 ERROR 后，SQ/RQ 里剩余的 WR 全部以 WR_FLUSH_ERR 完成。"""
        n = 0
        for e in self.sq:
            if e.signaled:
                self.cq.append(CQE(e.wr_id, Status.WR_FLUSH, e.opcode))
            n += 1
        for e in self.rq:
            if e.signaled:
                self.cq.append(CQE(e.wr_id, Status.WR_FLUSH, "RECV"))
            n += 1
        self.sq.clear()
        self.rq.clear()
        return n


def transition_error(target, cur):
    """四个状态必须逐级前进；RESET 是唯一的复位入口。"""
    order = [QPState.RESET, QPState.INIT, QPState.RTR, QPState.RTS]
    if target is QPState.ERROR:
        return None
    if target is QPState.RESET:
        return None                      # 任何状态都可以回 RESET
    if cur is QPState.ERROR:
        return f"ERROR 必须先回 RESET 才能到 {target.value}"
    if order.index(target) != order.index(cur) + 1:
        return (f"不能从 {cur.value} 直接到 {target.value}，"
                f"必须逐级（RESET→INIT→RTR→RTS）")
    return None


class Endpoint:
    """一对 RC QP 的模拟：发送端 post，接收端消费 recv WR。"""

    def __init__(self, name):
        self.name = name
        self.qp = QP(name)

    def setup(self, peer_qpn=42):
        self.qp.modify_qp(QPState.INIT, port_num=1, pkey_index=0,
                          qp_access_flags=0x0F)
        self.qp.modify_qp(QPState.RTR, dest_qp_num=peer_qpn, path_mtu=4096,
                          rq_psn=0, max_dest_rd_atomic=8,
                          min_rnr_timer=12, ah_attr="grh")
        self.qp.modify_qp(QPState.RTS, sq_psn=0, timeout=14,
                          retry_cnt=7, rnr_retry=6, max_rd_atomic=8)


def deliver(sender: QP, receiver: QP, entry: WQEEntry) -> str:
    """把一笔 SEND 交给对端，返回结果描述。"""
    if entry.bad == 1:
        # 对端认为 rkey/权限不对：这笔自己失败，并把 QP 打进 ERROR
        sender.cq.append(CQE(entry.wr_id, Status.REM_ACCESS, entry.opcode))
        return "REM_ACCESS"
    if entry.bad == 2:
        sender.enter_error("本地越界")
        return "LOC_LEN"

    if not receiver.rq:
        # RNR：按 rnr_retry 重试，耗尽才报错
        if sender.rnr_retry == 7:
            return "RNR(无限重试，卡住)"
        sender.cq.append(CQE(entry.wr_id, Status.RNR_RETRY_EXC, entry.opcode))
        return "RNR_EXC"

    r = receiver.rq.pop(0)
    if entry.signaled:
        sender.cq.append(CQE(entry.wr_id, Status.SUCCESS, entry.opcode))
    if r.signaled:
        receiver.cq.append(CQE(r.wr_id, Status.SUCCESS, "RECV"))
    return "OK"


def scenario_happy():
    print("\n" + "=" * 72)
    print("场景 A：正常路径（recv WR 提前投好）")
    print("=" * 72)
    a, b = Endpoint("A"), Endpoint("B")
    a.setup()
    b.setup(peer_qpn=7)
    print("初始状态：", a.qp.state.value, "/", b.qp.state.value)

    for i in range(3):
        b.qp.post_recv(WQEEntry(wr_id=100 + i, signaled=True))
    print(f"B 预投 recv WR：{len(b.qp.rq)} 个")

    for i in range(3):
        e = WQEEntry(wr_id=i, length=1024, signaled=True)
        a.qp.post_send(e)
        res = deliver(a.qp, b.qp, e)
        print(f"  send wr_id={i} -> {res}，A 的 CQ 有 {len(a.qp.cq)} 条")

    while (c := a.qp.poll_cq()) is not None:
        print(f"  A poll 到 wr_id={c.wr_id} status={c.status.value}")
    while (c := b.qp.poll_cq()) is not None:
        print(f"  B poll 到 wr_id={c.wr_id} status={c.status.value}")


def scenario_einval():
    print("\n" + "=" * 72)
    print("场景 B：modify_qp 字段填错（对应 Q43）")
    print("=" * 72)
    a = Endpoint("A")
    try:
        a.qp.modify_qp(QPState.INIT, port_num=1, pkey_index=0)
    except ValueError as e:
        print("  ✗", e)
    print("  -> qp_access_flags 是 INIT 的必填项，填在 RTR 就会是这个错")

    a.setup(peer_qpn=42)
    print(f"  正确顺序后状态：{a.qp.state.value}")

    print("\n  跳过 RTR 直接 RTS：")
    b = Endpoint("B")
    try:
        b.qp.modify_qp(QPState.RTS, sq_psn=0, timeout=14, retry_cnt=7,
                       rnr_retry=6, max_rd_atomic=8)
    except ValueError as e:
        print("  ✗", e)
    print("  -> 四个状态是逐层打开开关，不能跳级")


def scenario_rnr():
    print("\n" + "=" * 72)
    print("场景 C：对端没投 recv WR（对应 Q12 / Q48）")
    print("=" * 72)
    a, b = Endpoint("A"), Endpoint("B")
    a.setup()
    b.setup(peer_qpn=7)
    print(f"  B 的 RQ 长度：{len(b.qp.rq)}")

    e = WQEEntry(wr_id=0, length=64)
    a.qp.post_send(e)
    res = deliver(a.qp, b.qp, e)
    c = a.qp.poll_cq()
    print(f"  send -> {res}，A 拿到 {c.status.value if c else 'None'}")
    print("  -> 应用 hang 在等完成，但拿到的是 RNR 而不是 SUCCESS")

    print("\n  修法一：补投 recv WR")
    b.qp.post_recv(WQEEntry(wr_id=100))
    e2 = WQEEntry(wr_id=1, length=64)
    a.qp.post_send(e2)
    print("  send ->", deliver(a.qp, b.qp, e2))

    print("  修法二：改成 WRITE + WITH_IMM，彻底去掉对 recv WR 的依赖")
    print("        （语义上换成单边写，对端不再需要预投缓冲）")


def scenario_flush():
    print("\n" + "=" * 72)
    print("场景 D：一个错误把 QP 打进 ERROR，后面的 WR 全部连带 flush")
    print("=" * 72)
    a, b = Endpoint("A"), Endpoint("B")
    a.setup()
    b.setup(peer_qpn=7)

    # 先投一堆 send，其中第 2 笔是非法的
    entries = [
        WQEEntry(wr_id=0, length=1024),
        WQEEntry(wr_id=1, length=1024, bad=1),   # rkey/权限错
        WQEEntry(wr_id=2, length=1024),
        WQEEntry(wr_id=3, length=1024),
        WQEEntry(wr_id=4, length=1024),
    ]
    for e in entries:
        a.qp.post_send(e)
    print(f"  A 的 SQ 里有 {len(a.qp.sq)} 笔")

    # B 预投足够 recv
    for i in range(8):
        b.qp.post_recv(WQEEntry(wr_id=200 + i))

    # 按 SQ 顺序逐笔处理，出错即停
    while a.qp.sq and a.qp.state is not QPState.ERROR:
        e = a.qp.sq.pop(0)
        res = deliver(a.qp, b.qp, e)
        if res != "OK":
            print(f"  wr_id={e.wr_id} -> {res}，QP 进入 ERROR")
            a.qp.enter_error(res)
            n = a.qp.flush_all()
            print(f"  连带 flush {n} 笔未完成的 WR")

    print("\n  CQE 序列（注意根因排在最前）：")
    seq = []
    while (c := a.qp.poll_cq()) is not None:
        seq.append(c)
    for c in seq:
        print(f"    wr_id={c.wr_id}  {c.status.value}")

    first_bad = next((c for c in seq
                      if c.status is not Status.WR_FLUSH
                      and c.status is not Status.SUCCESS), None)
    print(f"\n  第一个非 flush 的错误：wr_id={first_bad.wr_id} "
          f"{first_bad.status.value}")
    print("  -> 排障要找的是它，不是后面那批 WR_FLUSH_ERR（对应 36 章的规矩）")


def scenario_sigall():
    print("\n" + "=" * 72)
    print("场景 E：sq_sig_all=0 且没设 SIGNALED，收不到 CQE（对应 Q44）")
    print("=" * 72)
    a, b = Endpoint("A"), Endpoint("B")
    a.setup()
    b.setup(peer_qpn=7)
    a.qp.sq_sig_all = False
    for i in range(4):
        b.qp.post_recv(WQEEntry(wr_id=300 + i))
    entries = [WQEEntry(wr_id=i, length=512, signaled=False) for i in range(4)]
    for e in entries:
        a.qp.post_send(e)
        deliver(a.qp, b.qp, e)
    print(f"  发了 4 笔、全部未 signal，A 的 CQ 长度 = {len(a.qp.cq)}")
    print("  -> 应用会 hang 在 poll 里，但网卡和连接都是好的")

    print("\n  把最后一笔设成 signaled：")
    e = WQEEntry(wr_id=4, length=512, signaled=True)
    b.qp.post_recv(WQEEntry(wr_id=400))
    a.qp.post_send(e)
    deliver(a.qp, b.qp, e)
    c = a.qp.poll_cq()
    print(f"  poll 到 wr_id={c.wr_id} status={c.status.value}")
    print("  -> 它同时也证明前 4 笔已经完成（batching 的常用手段）")


if __name__ == "__main__":
    scenario_happy()
    scenario_einval()
    scenario_rnr()
    scenario_flush()
    scenario_sigall()
    print("\n" + "=" * 72)
    print("以上全是语义模拟：状态/字段/连带的因果关系与真实 verbs 一致，")
    print("但不含任何时间数字。时间成本见 33 章（固定成本）与 35 章（拥塞）。")
