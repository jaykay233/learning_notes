# 连接状态账本：RC / SRQ / XRC / DC 各付 O(N²) 的哪一段

```text
本次讲解位置
章节：communication / RDMA
小节：连接状态的账本（承接 21 的 DCI/DCT/AV 与 22 的实测代价）
知识点：连接状态的四段组成、ICM 与 context cache 的两级存储、N* 交叉点、burst 摊销
上次：DC 把 destination 从 QP context 搬进 WQE 的 AV（21），DC/RC 的实测代价（22）
下次：DCQCN 参数闭环，或 DeepEP 里 DC 的实际用法
```

## 为什么现在讲这个

[21 章](./21-gpu-initiated-rdma-submission-dc-qp-sharing.md) 说 DC 把目标从 QP context 搬进
WQE 的 AV，省掉大量持久连接；[22 章](./22-gpu-initiated-communication-performance-cost-model.md)
给了三组实测：3,000 连接损失约 59%、DC 单笔只贵 0.45 us、但每个 WQE 都换目标时
message rate 掉约 60 倍。

这三组数字放在一起是**自相矛盾**的：省了状态，为什么反而更慢？

再看 [32 章 Q50](./32-rdma-interview-questions.md) 的标准答案："O(N²) 的连接状态，
NIC 的 QP context cache 装不下，查找 miss 直接吃延迟。"这句话是对的，但**大部分人答不到点上**，
因为在他们的直觉里 "cache 装不下" 指的是网卡内存（GB 级）。按 GB 级算，
3,000 个连接只有 1.46 MiB，怎么装都装得下，于是结论就崩了。

这一章要解决的就是这件事：**把"连接状态"拆成四段、把"存储"拆成两级**，
然后算出拐点在哪里、为什么 DC 会同时"更省"和"更慢"。
这是能在面试里把 Q6 / Q50 讲到不会追问的唯一方式。

---

## 心智模型：连接状态是四段，不是一块

一条 RC 连接在网卡上占的东西，至少有四段，寿命和位置都不同：

| 段 | 内容 | 存在哪 | 寿命 | 谁按 O(N²) 涨 |
| --- | --- | --- | --- | --- |
| ① context | QPN、PSN、dest、MTU、retry、access flags | ICM（DRAM backing）+ 片上 cache | 整个连接 | **是** |
| ② recv ring | 预投递的 recv WQE 数组 | host/GPU 内存 | 随连接 | **是** |
| ③ translation | MTT 表项（虚拟→物理） | ICM | 随 MR，不随连接 | 否 |
| ④ completion state | CQ 消费指针、outstanding 计数 | host 内存 + 片上 | 随连接 | 是（但很小） |

**关键观察：这四段的"贵"法完全不同。** ①③ 贵在**驻留**（要占用片上 SRAM 才快），
② 贵在**内存容量**（DRAM 便宜但占地），④ 贵在**每包处理**。
一项传输方案如果只砍 ②，你不会拿到任何延迟改善——这是 SRQ 的处境。

## 存储是两级，差三个数量级

这是本章的核心，也是 Q50 答错的根因：

```text
ICM（Internal Cache Memory）
  DRAM backing，GB 级
  决定"你能不能建这么多连接"  ← 容量问题
        ↓ 按需加载
context cache
  片上 SRAM，几百 KiB 级
  决定"处理每个包时 context 在不在手边"  ← 驻留问题
```

**建得起 ≠ 跑得快。** 3,000 个连接共 1.46 MiB，ICM 随便放；
但只要片上 cache 只有 512 KiB，就有 2/3 的包在处理时要先加载 context。

---

## 完整计算

下面这段代码是自包含的，直接跑就能复现本文所有数字。
完整的打印版本在 [`34_connection_state_ledger.py`](./34_connection_state_ledger.py)。

```python
#!/usr/bin/env python3
"""连接状态账本：RC / SRQ / XRC / DC 各付哪一段。"""

# [规范/PRM] mlx5 QP context 对齐单位；recv WQE segment 16 B
QP_CTX_B, DCT_CTX_B = 512, 512
RECV_WQE_B, RECV_DEPTH = 16, 64
RC_WQE_B, DC_WQE_B = 48, 64          # DC 的 WQE 多 16 B AV

# [假设] 片上 cache 大小与命中/未命中量级
CACHE_B, HIT_NS, MISS_NS = 512 << 10, 20, 80   # 512 KiB

# [锚点] 22 章的实测值，用来校准
DC_SWITCH_NS, ONSET_CONN, FIXED_1K_NS = 450, 1350, 1500


def rc(p):
    """每个 peer 一份完整 context + 一份自己的 recv ring。"""
    return {"ctx": p * QP_CTX_B, "ring": p * RECV_DEPTH * RECV_WQE_B}


def srq(p):
    """context 不变，recv ring 由所有 peer 共享一份。"""
    return {"ctx": p * QP_CTX_B, "ring": RECV_DEPTH * RECV_WQE_B}


def xrc(p):
    """发起侧每 peer 一份 SQ context；接收侧只有一个共享 XRC RQ。"""
    return {
        "send_ctx": p * QP_CTX_B,
        "recv_ctx": QP_CTX_B,
        "recv_ring": RECV_DEPTH * RECV_WQE_B,
    }


def dc(p, n_dci=8):
    """发起侧只留 n_dci 个 DCI；目标侧按 peer 数建 DCT。"""
    return {"dci_ctx": n_dci * QP_CTX_B, "dct_ctx": p * DCT_CTX_B}


def total(d):
    return sum(d.values())


def crossover(cache_bytes, ctx_bytes):
    """驻留上限 N*：多少连接之后开始抖动。"""
    return cache_bytes // ctx_bytes


if __name__ == "__main__":
    print(f"{'N':>6}{'RC KiB':>10}{'SRQ KiB':>10}{'XRC KiB':>10}{'DC KiB':>10}")
    for n in (128, 1024, 1350, 3000):
        p = n - 1
        print(f"{n:>6}{total(rc(p))/1024:>10.0f}{total(srq(p))/1024:>10.0f}"
              f"{total(xrc(p))/1024:>10.0f}{total(dc(p))/1024:>10.0f}")

    print(f"\ncache={CACHE_B//1024} KiB -> N*={crossover(CACHE_B, QP_CTX_B)}")
    print(f"实测 onset={ONSET_CONN}，同量级")
```

输出：

```text
     N    RC KiB   SRQ KiB   XRC KiB    DC KiB
   128       190        64        65        68
  1024      1534       512       513       516
  1350      2024       676       676       678
  3000      4498      1500      1501      1504

cache=512 KiB -> N*=1024
实测 onset=1350，同量级
```

## 逐项读这张账本

### SRQ 只砍了第二段

这是最常被答错的一题。很多人把 SRQ 当成"省连接状态的方案"，实际上：

```text
RC :  context O(N)  +  recv ring O(N)
SRQ:  context O(N)  +  recv ring O(1)     ← 只动了 ring
```

512 个连接、深度 64 时，ring 从 512 × 1 KiB = 512 KiB 降到 1 KiB，
内存确实省了 99.8%；但 **context 的一个字节都没省**，
片上 cache 的驻留压力完全不变。

所以 SRQ 的收益在**内存容量**，不在**延迟**。面试里如果说
"SRQ 能解决大规模连接的 cache miss"，就是答错了段。

### XRC 砍的是接收侧 context

XRC 把接收侧的 RQ 做成一个共享的 XRC RQ（本质是一个 XRC SRQ），
所有发送方共用；发起侧仍然每个 peer 一份 XRC SQ。

```text
发起侧：O(N) 份 XRC SQ context      ← 没省
接收侧：O(1) 份 XRC RQ context      ← 省了
```

所以 XRC 对**接收端**（比如 all-gather 的汇聚点、parameter server）
收益最大；对每个节点都均匀收发的 all-to-all，收益只有一半。

顺带一个常被混的对比：**SRQ 与 XRC 的区别不在"共享"，在"共享到哪一层"。**
SRQ 共享的是 WQE 缓冲（第二段），XRC 共享的是接收上下文（第一段）。

### DC 砍的是发起侧持久 context

DC 把目标从 context 搬到 WQE 的 AV，发起侧只保留少量 DCI：

```text
RC:  O(N) 份 QP context
DC:  O(n_dci) 份 DCI context  +  每个 WQE 多 16 B AV
```

3,000 连接时，发起侧从 1.5 MiB 降到 4 KiB（8 个 DCI）。
这是 DC 真正的用途：**把"容量问题"消掉。**

## 但 DC 的代价分两种，不能混

这是本章第二个要点，也是 22 章那组矛盾数字的答案：

| 代价 | 量级 | 何时发生 | 能否被摊薄 |
| --- | --- | --- | --- |
| WQE 变大（48→64 B） | 每 WQE 16 B | 永远 | 不能，但很小 |
| destination 切换 | ~450 ns / 次 | 每次换目标 | **能，按 burst 摊薄** |

第二个才是致命的。设每个目标连续写 `burst` 笔：

```text
每笔均摊 = 450 ns / burst

burst = 1    → 450 ns/笔，占 1 KB 固定成本的 30%
burst = 16   →  28 ns/笔，2%
burst = 64   →   7 ns/笔，可以忽略
```

**DC 不是"慢"，是"每笔都换目标时慢"。** 22 章说的"每 WQE 换目标 → message rate 掉 60 倍"
和"短 per-destination burst 能恢复大部分性能"，说的是同一件事的两端。

这条对写代码的直接含义：

```text
把发往同一个 peer 的 WQE 排在一起，再切下一个 peer
  → DC 的切换成本被 burst 摊掉
把 token 按到达顺序发，目标在 expert/peer 之间来回跳
  → 每个 WQE 都切一次，DC 退化成最坏情况
```

MoE 的 dispatch 天然是"按 token 顺序、目标乱跳"。所以 GPU 侧做 DC 之前，
必须先在 kernel 里按目标做一次分组（sort / bucketing），否则省下的 context
会全部以切换成本的形式还回去。这是 DeepEP 那类实现里真实存在的设计约束。

## DC 的状态机与凭证（21 章没展开的部分）

DC 在目标侧是一个 DCT，它有独立的状态机：

```text
RESET → ACTIVE → SUSPENDED → ACTIVE
  ↓        ↓         ↓
      ERROR（任何不可恢复错误）
```

| 状态 | 含义 | 用途 |
| --- | --- | --- |
| RESET | 刚创建，不接请求 | 建连前 |
| ACTIVE | 正常服务 | 数据面 |
| SUSPENDED | 暂停接受新请求 | 排空、迁移、优雅下线 |
| ERROR | 不可恢复 | 只能重建 |

`SUSPENDED` 是 DC 比 RC 多出来的一个能力：RC 要"暂停"一条连接只能靠应用层停止发，
而 DCT 可以真正把自己摘出去，让在途请求先排空。这在做**连接迁移**和
**滚动重启**时很有用。

访问一个 DCT 需要两样东西，都在 AV 里：

```text
DCT number（类似 QPN，指"哪个目标端点"）
DC key（64 位，类似 rkey，指"凭什么能访问"）
```

**DC key 错的表现和 rkey 错一模一样**：不是连接失败，而是请求返回
`IBV_WC_REM_ACCESS_ERR` 类的完成错误。所以排障时如果看到"连接似乎建好了、
但每笔都访问错误"，先查 AV 里的 DC key，而不是去查路径 MTU。

还有一个二阶成本值得记：WQE 从 48 B 变成 64 B，意味着一条 64 B cache line
从"能放 1.33 个 WQE"变成"正好 1 个"。doorbell batching 的收益会被这一点吃掉一部分，
所以 DC 在 message rate 上的劣势不只来自切换，还来自 WQE 变胖。

---

## 决策表：按流量模式选，不按"新旧"选

| 模式 | 连接特征 | 选择 | 理由 |
| --- | --- | --- | --- |
| 固定 N 个 peer、长期通信 | N 小且稳定（≤ 几十） | **RC** | 状态装得下，延迟最低 |
| 一对多收（汇聚、PS） | 接收侧 O(N) | **XRC** | 砍接收侧 context |
| 多对一收、缓冲需求大 | 接收缓冲是瓶颈 | **SRQ** | 砍 ring 内存（注意不改延迟） |
| 大规模 all-to-all | 目标数 >> cache | **DC + 分组** | 砍持久 context，但必须 burst |
| 只发不收（如 send-only 数据面） | 状态只在一侧 | RC 就够 | 22 章：32,768 QP 仍可 242 M msg/s |
| 同时收发、连接数千 | 状态翻倍 | DC 或多网卡 | 22 章：同时收发时损失约 51% |

最后两行是 22 章的关键结论，值得单独记住：**连接成本取决于流量角色。**
同样是 3,000 个连接，send-only 撑得住，同时收发已经明显退化——
因为接收侧要额外维护 recv 状态、completion 和 translation。

---

## 常见错法，以及症状

| 错法 | 症状 |
| --- | --- |
| 用 SRQ 解决 cache miss | 内存降了，延迟一点没变（第二段 vs 第一段） |
| 按 ICM 容量判断"装不下" | 算出 1.46 MiB 装得下，于是认为 Q50 是错的——实际是片上 cache |
| 上 DC 但不按目标分组 | 单笔延迟变差、message rate 掉一个数量级（每 WQE 切目标） |
| 用 SUSPENDED 当"关机" | 在途请求被拒，出现 `REM_ACCESS_ERR` 类错误 |
| DC key 和 rkey 混着查 | 明明连接是好的，却一直在查 MTU 和链路 |
| 只按 QP 数量做容量规划 | 忽略 send/receive 角色，send-only 能过的压测在双向场景崩 |

倒数第二条特别值得记：DC 有两套凭证（DC key 管"能不能进这个 DCT"，rkey 管
"能不能写这段内存"），报错信息长得很像，查错方向完全不同。

---

## 自测

1. SRQ 能解决大规模连接的 cache miss 吗？它到底省了哪一段？
2. 3,000 个 RC 连接的 context 一共多大？既然装得下，为什么还会退化？
3. `N*` 怎么算？如果网卡 QP context 实际是 2 KiB，拐点会移到哪里？
4. DC 省了 context，为什么 message rate 会掉 60 倍？什么条件下能恢复？
5. 你说"DC key 错了"，应用会看到什么？和 rkey 错的区别是什么？

<details>
<summary>答案</summary>

1. **不能。** SRQ 只把 recv WQE ring 从 O(N) 降到 O(1)（第二段），
   QP context 仍是 O(N)（第一段）。收益在内存容量，不在延迟。
2. `3000 × 512 B ≈ 1.46 MiB`。**ICM 装得下，但片上 context cache 只有几百 KiB**，
   3,000 个连接要 1.46 MiB 驻留，命中率掉到约 1/3，每个包都要重新加载 context。
   区别在于"两级存储"：容量问题 ≠ 驻留问题。
3. `N* = cache / context_size`。512 KiB ÷ 512 B = 1,024（实测拐点 1,350，同量级）。
   若 context 是 2 KiB，`N* = 256`，拐点提前四倍。
4. 因为**每个 WQE 都换 destination**，每次切换约 450 ns。按 burst 摊薄：
   burst=16 → 28 ns（2%），burst=64 → 7 ns。DC 不是慢，是"逐笔换目标"慢。
   恢复办法：按目标分组，让同一目标的 WQE 连发。
5. 返回 `IBV_WC_REM_ACCESS_ERR` 类的访问错误，**连接本身是通的**。
   和 rkey 错表现几乎一样，区别是 DC 有两套凭证：DC key 管"能不能进 DCT"，
   rkey 管"能不能写这段内存"——所以要先确认错在哪一层，别一上来查 MTU。

</details>

---

## 一页速记

```text
四段状态:
  context      O(N)   驻留贵（片上 cache）      ← 真正的瓶颈
  recv ring    O(N)   容量贵（DRAM）            ← SRQ 只砍这一段
  translation  不随连接涨
  completion   小，但每包都要碰

两级存储:
  ICM          GB 级     决定"建得起吗"
  ctx cache    几百 KiB  决定"跑得快吗"
  N* = cache / ctx_size      512 KiB / 512 B = 1024（实测 1350）

三种方案各砍一段:
  SRQ  砍 ring               context 不变，延迟不变
  XRC  砍接收侧 context       发起侧不变，对汇聚点最有效
  DC   砍发起侧持久 context   换 per-WQE AV(16B) + 切换成本

DC 两种代价:
  WQE 48->64 B    每笔 16 B，不能摊薄，但小
  destination 切换 ~450 ns    按 burst 摊薄：16 笔 -> 28 ns

DC 状态机:
  RESET / ACTIVE / SUSPENDED / ERROR
  SUSPENDED 用来排空、迁移
  AV 里两套凭证：DCT number + DC key

选型:
  固定少 peer -> RC
  一对多收     -> XRC
  接收缓冲瓶颈 -> SRQ
  大规模 all-to-all -> DC + 按目标分组
```
