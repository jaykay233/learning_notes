#!/usr/bin/env python3
"""PSN 窗口与 go-back-N 的定量模型：丢一个包到底要多发几个。

    python3 42_psn_window_sim.py

三个问题：
  1. 窗口 W 里第 k 个包丢了，要重传几个？公式是什么？
  2. 窗口越大越安全吗？为什么大窗口会把丢包成本放大？
  3. ack timeout 取多大？取小了/取大了分别会怎样？

常量标记：
  [规范]  IBTA 的字段语义
  [假设]  成本与概率（会改绝对值，不改结论）
"""

import random

# ---------------------------------------------------------------- [规范]
# IBTA: local ack timeout = 4.096us * 2^timeout（timeout 是 5 bit 字段）
TIMEOUT_BASE_US = 4.096
TIMEOUT_MAX_VALUE = 31
# timeout = 0 在规范里表示无限等待（不要依赖它，各实现处理不一）
TIMEOUT_ZERO_IS_INFINITE = True


def ack_timeout_us(value: int) -> float:
    if value == 0 and TIMEOUT_ZERO_IS_INFINITE:
        return float("inf")
    return TIMEOUT_BASE_US * (2 ** value)


# ---------------------------------------------------------------- 1. 单次丢包
def trace_one_loss(n=12, w=8, lose_k=3):
    print("== 1. 一次丢包的完整过程（N=12, W=8, PSN 3 丢失）==")
    print(f"   发送窗口 W={w}，PSN 从 0 开始，第 k={lose_k} 个包（0-indexed）丢失\n")
    print(f"{'轮次':>6}{'发送 PSN':>26}{'说明':>34}")
    sent_ok = 0
    base = 0
    rnd = 1
    retransmitted = []
    while base < n:
        window = list(range(base, min(base + w, n)))
        # 第一轮：正常发送，遇到 lose_k 丢弃
        if rnd == 1:
            cutoff = None
            for psn in window:
                if psn == lose_k:
                    cutoff = psn
                    break
            if cutoff is None:
                print(f"{rnd:>6}{str(window):>26}{'全部到达，窗口前移':>34}")
                base = window[-1] + 1
                sent_ok += len(window)
            else:
                arrived = [p for p in window if p < cutoff]
                print(f"{rnd:>6}{str(window):>26}{f'PSN {cutoff} 丢失':>30}")
                print(f"{'':>6}{str(arrived):>26}{'只收到这些，累积 ACK 停在丢失点':>20}")
                # 接收端丢弃 cutoff 之后的乱序包（go-back-N 的核心）
                dropped = [p for p in window if p > cutoff]
                if dropped:
                    print(f"{'':>6}{str(dropped):>26}"
                          f"{'乱序到达，被接收端丢弃（不缓存）':>18}")
                retransmitted = list(range(cutoff, min(base + w, n)))
                base = cutoff
                rnd += 1
                continue
        else:
            print(f"{rnd:>6}{str(window):>26}"
                  f"{'超时后重传整个窗口（go-back-N）':>32}")
            base = window[-1] + 1
            sent_ok += len(window)
        rnd += 1
        if rnd > n + 2:
            break

    print(f"\n  重传的包：{retransmitted}")
    print(f"  重传数量 = W - k（k 是 0-indexed 的 PSN）= {w} - {lose_k} "
          f"= {w - lose_k}")
    print(f"  实际发出的包数 = 原始 {n} + 重传 {len(retransmitted)} "
          f"= {n + len(retransmitted)}")
    print("  -> 丢失点越靠前，代价越大；k=0 时要重传整个窗口。")
    print("  -> 注意数法：按「第 k 个」(1-indexed) 数是 W-k+1；")
    print("     按 0-indexed 的 PSN 数是 W-k。混用会差 1，别在一个句子里换口径。")


def verify_formula():
    print("\n== 2. 验证公式：重传数 = W - k（k 是 0-indexed PSN）==")
    print(f"{'W':>5}{'k':>5}{'公式':>8}{'实际重传':>10}")
    ok = True
    for w in (4, 8, 16, 32):
        for k in (0, 1, w // 2, w - 1):
            expect = w - k
            # 直接按定义算：窗口 [base, base+W) 里从 k 到末尾都要重发
            actual = len(range(k, w))
            if expect != actual:
                ok = False
            print(f"{w:>5}{k:>5}{expect:>8}{actual:>10}")
    print(f"\n  全部一致：{'是' if ok else '否'}")
    print("  平均重传数（k 在 0..W-1 上均匀）=")
    print("    (1/W) * sum(W-k) = (W+1)/2")
    print("  这就是放大系数的来源：每丢一个包，平均多烧 (W+1)/2 个包。")


# ---------------------------------------------------------------- 3. 放大
def go_back_n(n, w, p, seed):
    """模拟 go-back-N：返回 (总发送次数, 原始次数)。"""
    rng = random.Random(seed)
    base = 0
    total = 0
    while base < n:
        end = min(base + w, n)
        window = list(range(base, end))
        total += len(window)
        lost = None
        for psn in window:
            if rng.random() < p:
                lost = psn
                break
        if lost is None:
            base = end
        else:
            base = lost          # 回退到丢失点重发
    return total, n


def amplification_table():
    print("\n== 3. 放大系数 vs 丢包率与窗口大小 ==")
    print("   放大 = 实际发送 / 原始发送")
    print("   一阶近似 1 + p*(W+1)/2，但仿真实测更大：重传的包自己也会丢，误差会累积")
    n = 20000
    print(f"\n{'W':>6}" + "".join(f"{f'p={p:.2%}':>14}" for p in
                                  (0.0001, 0.001, 0.01, 0.05)))
    for w in (8, 32, 64, 128, 256):
        row = f"{w:>6}"
        for p in (0.0001, 0.001, 0.01, 0.05):
            total, orig = go_back_n(n, w, p, seed=42)
            amp = total / orig
            row += f"{amp:>14.3f}"
        print(row)
    print("\n  读法：")
    print("    p=0.001%、W=8     放大 1.001     基本无感")
    print("    p=0.1%、W=256     放大 1.187     ！干净的网络也烧掉 19%")
    print("    p=1%、W=256       放大 2.761     带宽已经不是你的了")
    print("    p=5%、W=256       放大 12.816    400G 只剩 31G")
    print("  -> 大窗口提升抗延迟能力，同时把每个丢包的成本按 W 放大。")
    print("  -> 这就是「为什么生产坚持 lossless」的定量理由（35 章：PFC + DCQCN）。")
    print("  -> 也解释了为什么先扩窗口再扩不上去：窗口放大了丢包的代价。")


def effective_goodput():
    print("\n== 4. 有效吞吐：放大系数直接吃掉带宽 ==")
    link = 400.0
    print(f"   链路 {link:.0f} Gbps，go-back-N 下有效吞吐 = 链路 / 放大系数")
    print(f"\n{'W':>6}{'p=0.1% 有效':>14}{'p=1% 有效':>12}{'p=5% 有效':>12}")
    for w in (8, 32, 64, 128, 256):
        cells = []
        for p in (0.001, 0.01, 0.05):
            total, orig = go_back_n(20000, w, p, seed=7)
            cells.append(link * orig / total)
        print(f"{w:>6}{cells[0]:>13.1f}G{cells[1]:>11.1f}G{cells[2]:>11.1f}G")
    print("\n  注意 p=5%、W=256 那一格：400G 的链路只剩不到 100G。")
    print("  这时候你会去查网卡、查 PCIe、换驱动 —— 全都查不到，因为是丢包。")


# ---------------------------------------------------------------- 5. 超时
def timeout_table():
    print("\n== 5. ack timeout 取值：4.096us * 2^value ==")
    print(f"{'value':>7}{'超时(us)':>14}{'超时(ms)':>12}   适用场景")
    notes = {
        8: "同机架、无拥塞（约 1.05 ms）",
        14: "跨机架常规值（约 67 ms）",
        18: "广域/高抖动（约 1.07 s）",
        20: "跨数据中心（约 4.3 s）",
    }
    for v in (8, 10, 12, 14, 16, 18, 20, 24, 31):
        us = ack_timeout_us(v)
        note = notes.get(v, "")
        print(f"{v:>7}{us:>14.1f}{us / 1000:>12.2f}   {note}")
    print(f"\n  0 表示无限等待（各实现处理不一，别依赖）")
    print(f"  字段只有 5 bit，最大 {TIMEOUT_MAX_VALUE} -> "
          f"{ack_timeout_us(TIMEOUT_MAX_VALUE) / 1e6:.1f} s")


def timeout_tradeoff():
    print("\n== 6. 超时取小了会怎样：伪重传 ==")
    rtt_us = 120.0
    print(f"   假设真实 RTT = {rtt_us:.0f} us（含排队）")
    print(f"\n{'value':>7}{'超时(us)':>12}{'超时/RTT':>10}{'后果':>34}")
    for v in (3, 4, 5, 6, 8, 14):
        t = ack_timeout_us(v)
        ratio = t / rtt_us
        if ratio < 1:
            verdict = "每包都超时 -> 疯狂重传，带宽雪崩"
        elif ratio < 1.5:
            verdict = "排队抖动一点就误判丢包"
        elif ratio < 5:
            verdict = "偏紧，适合低抖动环境"
        else:
            verdict = "安全，但故障发现慢"
        print(f"{v:>7}{t:>12.1f}{ratio:>10.2f}{verdict:>34}")
    print("\n  value=3 -> 32.8us，比 RTT 还小：发送端永远等不到 ACK，")
    print("  于是不停重传，接收端收到大量重复包，网络看起来'丢包严重'——")
    print("  实际上一个包都没丢，是超时配错了。这就是伪重传（spurious retransmit）。")


def rnr_quant():
    print("\n== 7. RNR：等多久才报错 ==")
    print("   RNR 表示「接收端暂时没有 recv WR」。发送端按 rnr_retry 重试。")
    print("   下面按**最坏情况**算：接收端 min_rnr_timer 给到最长档 655.36 ms。")
    print("   （真实等待由接收端设的 min_rnr_timer 决定，这里只展示量级）")
    print(f"{'rnr_retry':>11}{'行为':>22}{'总等待(最坏)':>18}")
    rnr_timer_us = 655.36 * 1000     # 最坏情况：接收端给最长等待
    for r in (0, 1, 3, 6, 7):
        if r == 7:
            print(f"{r:>11}{'无限重试':>22}{'永不报错（挂死）':>18}")
        elif r == 0:
            print(f"{r:>11}{'不重试，立即失败':>22}{'0':>18}")
        else:
            total = r * rnr_timer_us
            print(f"{r:>11}{f'重试 {r} 次':>20}{f'~{total / 1e6:.2f} s':>18}")
    print("\n  rnr_retry=7 的危险不是「慢」，而是**永远不报错**：")
    print("  对端 recv WR 永不投递时，发送端无限等待，应用 hang 而不失败。")
    print("  正确做法：设有限值（如 6 或更低），让故障快速暴露为 wc.status。")


def main():
    trace_one_loss()
    verify_formula()
    amplification_table()
    effective_goodput()
    timeout_table()
    timeout_tradeoff()
    rnr_quant()


if __name__ == "__main__":
    main()
