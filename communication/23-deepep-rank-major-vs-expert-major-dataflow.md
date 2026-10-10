# DeepEP：Rank-major vs Expert-major 数据流对照卡

> 承接 [20](./20-deepep-v2-buffer-coordinates-and-handle-arrays.md) 的寄信模型（Route / Layout / ForwardLoc / ReturnInfo / LocalMap）。本文把两条主干数据流钉成可背诵的对照卡：**V1 机内 rank-major**（`intranode.cu`）与 **V1 LL expert-major**（`internode_ll.cu`）。细节实现见 [10](./10-deepep-v1-intranode-dispatch-kernel.md)–[14](./14-deepep-v1-low-latency-combine-and-pack.md)、[16](./16-deepep-v1-internode-dispatch-combine.md)。

```text
本次讲解位置
章节：communication / DeepEP 布局对照
小节：两条数据流对照卡
知识点：rank-major（V1 normal 机内）vs expert-major（V1 LL）端到端数据流
上次：V2 buffer 坐标与通用 handle 数组（20）
下次：hybrid 两级中转时两条流如何叠加
原语：环形队列 head/tail、atomic 抢 slot、layout_range (count,begin)
```

## 为什么现在讲这个

笔记 20 把五类数组抽象出来了，但读代码时仍容易把两条流混在一起：rank-major 有 channel / `send_head` / `recv_src_idx[r]`；expert-major 没有 channel，回程直接 `[expert][t]`。把端到端数据流压成两张对照卡，以后看到任意一个数组都能立刻问：「我在哪条流上？这一步是 Layout 还是 ReturnInfo？」

---

## 1. 一行语义（先钉死）

| | Rank-major（V1 normal 机内） | Expert-major（V1 LL） |
|---|---|---|
| 一行是什么 | `(token, 目的 rank)` 去重一份 | `(token, 某一个 expert)` 一份 |
| 用户张量 | `recv_x[r]` + 显式 `recv_topk_idx` | `recv_x[local_expert][j]`，expert 在第一维 |
| 谁切发送责任 | **channel** 切源 token 区间 | 无 channel；按 topk→expert |
| 回程怎么找格 | `send_head[t][R] → 环形 slot` | 直接 `[global_expert][原 token_idx]` |
| combine 权重 | 不加权重求和 | 源卡按 `topk_weights` 加权 |

---

## 2. Rank-major 数据流（V1 机内）

```text
源卡 token 0..N-1
  │ channel 切区间 → 决定「谁发」
  │ is_token_in_rank → 决定「发给哪个 dst」
  ▼
notify 统计并做前缀和：
  rank_prefix[src][dst]     → dst 上 recv_x 按源 rank 分段
  channel_prefix[dst][ch]   → 每个源→dst 流里再按 channel 分段
  ▼
dispatch：
  源：发 x + 改写后的 topk_idx（本卡 local id / -1）
      记 send_head[t][R] = 环形尾下标（不发则为 -1）
  dst：r = rank_offset(src) + channel_offset(ch) + 段内序号
       recv_x[r]
       recv_src_idx[r] = t          ← ReturnInfo
       recv_topk_idx[r][k]          ← LocalMap
  （环形缓冲在 dst 上：[channel][src_rank][slot]）
  ▼
用户：按 recv_topk_idx 自己 permute / GEMM → 同行序 y[r]
  ▼
combine：
  expert 卡：按同一把 rank×channel 尺子扫 y[r]，写入源卡
             源卡缓冲 [channel][expert_rank][slot]
  源卡：for t:
          slot = send_head[t][R] % buf_len   （R = expert rank）
          从各 R 的槽读回 → reduce → combined_x[t]
```

### 关键纠正

| 易混写法 | 正确含义 |
|---|---|
| `recv_x[r] = t` | 错。应是 `recv_src_idx[r] = t`；`recv_x[r]` 是 hidden |
| `send_head` 在 combine 才写 | 错。**dispatch 发送时**写在源卡 |
| 环形中间维是 dst | 错。缓冲在接收方；dispatch 时中间维是 **src rank**；combine 回程时中间维是 **expert rank** |

### 源码落点（机内）

| 步骤 | 位置 |
|---|---|
| channel 切 token | `utils.cuh` `get_channel_task_range`；`intranode.cu` L329–L330 |
| `is_token_in_rank` | `layout.cu` L82–L95 |
| rank / channel 前缀和 | `intranode.cu` L55–L125 |
| 写 `send_head`、改写 topk | `intranode.cu` L358–L392 |
| 收成 `recv_x` / `recv_src_idx` | `intranode.cu` L424–L511 |
| combine 用 `send_head` 取槽 | `intranode.cu` L918–L971 |
| handle | `legacy.py` L401：`(rank_prefix, channel_prefix, recv_channel_prefix, recv_src_idx, is_token_in_rank, send_head)` |

---

## 3. Expert-major 数据流（V1 LL）

```text
源卡 token 0..N-1，topk_idx[t][k] = 全局 expert
  │ 每个 (t,k) 命中一个 expert → 发一份副本（-1 跳过）
  │ 发送方对每个全局 expert 用 atomic 抢 slot
  ▼
对端 staging（在 dst 卡上）：
  [local_expert][src_rank][slot]
  （消息里带源 token_idx）
  ▼
dst pack 成用户张量（按 expert 分桶）：
  packed_recv_x[local_expert][0 .. ranks*MaxTok)
  layout_range[e][src] = (count, begin)   ← Layout
  src_info[e][j] = 源 token_idx t         ← ReturnInfo
  （LocalMap 隐含在第一维 e，无 recv_topk_idx）
  ▼
用户：直接按 expert 做 grouped GEMM → y[e][j]
  ▼
combine：
  expert 卡按 layout_range 扫，读 src_info[e][j]=t
  写回 源卡 [global_expert][t]            ← 无 send_head / 无环形查表
  源卡：用自己的 topk_idx[t][*] 知道收哪几个 expert
        按 topk_weights 加权 reduce → combined_x[t]
```

### 源码落点（LL）

| 步骤 | 位置 |
|---|---|
| atomic 抢 slot + staging 地址 | `internode_ll.cu` L254–L262 |
| `layout_range` / `src_info` | `internode_ll.cu` L407–L429 |
| 用户张量形状 | `legacy.py` L589–L602 |
| combine 写 `[global_expert][t]` | `internode_ll.cu` L842–L845 |
| handle | `legacy.py` L617：`(src_info, layout_range, MaxTok, hidden, num_experts)` |

---

## 4. 并排对照

```text
Rank-major (V1 normal)              Expert-major (V1 LL)
─────────────────────              ────────────────────
channel 切谁发                      无 channel；按 topk→expert 发
is_token_in_rank → dst              dst = expert / E_local
rank×channel 前缀和排 recv_x        layout_range[e][src]=(count,begin)
recv_x[r] 一行=到本卡的 token       recv_x[e][j] 一行=(token,expert)
recv_topk_idx = LocalMap            LocalMap 隐含在 e
recv_src_idx[r]=t                   src_info[e][j]=t
send_head[t][R]→环形 slot           无 ForwardLoc；回程 [expert][t]
回程缓冲 [ch][expert_rank][slot]    回程缓冲 [global_expert][t]
combine 不加权重                    combine 源卡加权
```

映射回笔记 20 的五类：

| 抽象 | Rank-major | Expert-major |
|---|---|---|
| Route | `topk_idx`（layout 用；combine 可不传） | `topk_idx` + `topk_weights`（combine 必传） |
| Layout | `rank_prefix` / `channel_prefix` | `layout_range[e][src]` |
| ForwardLoc | `send_head[t][R]` | 无（约定 `[expert][t]`） |
| ReturnInfo | `recv_src_idx[r]` | `src_info[e][j]` |
| LocalMap | `recv_topk_idx[r][k]` | 隐含在 `e` |

---

## 5. 具体数字走读（同一 token）

设定：4 rank，每 rank 2 expert，topk=2。源 rank0 的 token `t=5`，`topk=[1, 4]`
→ expert1 在 rank0，expert4 在 rank2。

### Rank-major

```text
is_token_in_rank[5] = [True@rank0, False, True@rank2, False]
发给 rank0、rank2 各一份 x（去重到 rank，不是按 expert 拆两行给同一卡）

dst=rank2 上某行 r：
  recv_x[r] = token5 向量
  recv_src_idx[r] = 5
  recv_topk_idx[r] = [-1, 0, ...]   # 本地 expert0(=全局4)；其它 lane -1

源卡 send_head[5][2] = 去程写入的环形尾下标
combine：源卡读 [ch][expert_rank=2][slot] 与本卡贡献，reduce 到 combined_x[5]
```

### Expert-major

```text
(t=5,k→expert1) → 发到 rank0 的 local_expert1，抢 slot
(t=5,k→expert4) → 发到 rank2 的 local_expert0，抢 slot
两份副本，不是「到 rank 去重一份」

rank2 上：
  packed_recv_x[e=0][j] = token5 副本
  src_info[0][j] = 5
  layout_range[0][src=0] 含这段 (count, begin)

combine：写源卡 [global_expert=4][t=5]
源卡按 topk 收 expert1 与 expert4 两格，加权求和 → combined_x[5]
```

---

## 6. 可运行小验证：两套索引怎么排行

文件：`communication/23_layout_dataflow_sim.py`。不模拟通信，只演示「同一批到达」在两种布局下的行号 / ReturnInfo。

```python
#!/usr/bin/env python3
"""Show how the same arrivals become rank-major rows vs expert-major rows."""

from __future__ import annotations


def main() -> None:
    # Arrivals at dst rank2 (local experts 0,1 = global 4,5)
    # Each record: (src_rank, src_token, local_experts_hit)
    arrivals = [
        (0, 5, [0]),       # token5 hits local expert0 only
        (0, 17, [0, 1]),   # token17 hits both local experts
        (1, 3, [1]),
    ]
    num_channels = 2

    print("=== Rank-major (dedupe per src token; LocalMap explicit) ===")
    # Fake: channel = src_token % num_channels; order by src, then channel, then t
    rows = sorted(arrivals, key=lambda a: (a[0], a[1] % num_channels, a[1]))
    recv_src_idx = []
    recv_topk = []
    for r, (src, t, locals_) in enumerate(rows):
        local_map = [e if e in locals_ else -1 for e in range(2)]
        # pad to topk=2 style: put hits in order
        local_map = (locals_ + [-1, -1])[:2]
        recv_src_idx.append(t)
        recv_topk.append(local_map)
        print(f"  r={r}: src={src} recv_src_idx={t} recv_topk_idx={local_map}")

    print("=== Expert-major (one row per (token, expert); LocalMap = axis0) ===")
    # Pack by local expert, then by src order
    for e in range(2):
        j = 0
        for src, t, locals_ in arrivals:
            if e not in locals_:
                continue
            print(f"  e={e} j={j}: src_info={t}  (from src_rank={src})")
            j += 1
        print(f"  layout count for e={e}: {j}")


if __name__ == "__main__":
    main()
```

运行：

```bash
python3 communication/23_layout_dataflow_sim.py
```

期望输出：

```text
=== Rank-major (dedupe per src token; LocalMap explicit) ===
  r=0: src=0 recv_src_idx=5 recv_topk_idx=[0, -1]
  r=1: src=0 recv_src_idx=17 recv_topk_idx=[0, 1]
  r=2: src=1 recv_src_idx=3 recv_topk_idx=[1, -1]
=== Expert-major (one row per (token, expert); LocalMap = axis0) ===
  e=0 j=0: src_info=5  (from src_rank=0)
  e=0 j=1: src_info=17  (from src_rank=0)
  layout count for e=0: 2
  e=1 j=0: src_info=17  (from src_rank=0)
  e=1 j=1: src_info=3  (from src_rank=1)
  layout count for e=1: 2
```

注意 token17：rank-major 只有 **1 行**（LocalMap 里两个 expert）；expert-major 拆成 **e=0 与 e=1 各一行**。

---

## 7. 常见误判

| 误判 | 实际 |
|---|---|
| 两条流的 `recv_x` 行语义一样 | 不同：去重到 rank vs 按 expert 展开 |
| expert-major 也有 `send_head` | LL 没有；回程直接 `[expert][t]` |
| rank-major 的 `r` 就是源 `t` | 否；`r` 是接收行，`t = recv_src_idx[r]` |
| channel 决定发给哪个 expert | 否；channel 只切发送责任；expert 由 `topk_idx` → `is_token_in_rank` / local 改写 |
| V2 expand = V1 LL | 都是 expert-major 语义，但 V2 staging 仍 rank-major，epilogue 才展开；回程是 `[贡献者][t]` 不是 `[expert][t]` |

---

## 8. 自测题

1. **Rank-major 里 `recv_x[r]` 和 `recv_src_idx[r]` 各是什么？**
   答：前者是第 r 个接收 token 的 hidden；后者是它对应的源卡原 token 号 `t`。

2. **Expert-major 为什么不需要 `recv_topk_idx`？**
   答：行落在 `packed_recv_x[e][*]`，第一维 `e` 就是本地 expert。

3. **源卡 combine 时，rank-major 用什么找到回程槽？expert-major 呢？**
   答：前者 `send_head[t][R] % buf_len`；后者直接读 `[global_expert][t]`，expert 列表来自源卡 `topk_idx[t]`。

4. **同一 token 命中某卡两个本地 expert：两条流各产生几行？**
   答：rank-major 1 行 + LocalMap 两个 id；expert-major 2 行（每 expert 一行）。

5. **V2 expand 更像哪条流？和 LL 最大差别？**
   答：用户张量像 expert-major；staging 仍 rank-major，回程是 `[贡献者][t]` 而非 `[expert][t]`（见笔记 20）。

---

## 9. 学习进度

- [x] V1/V2 坐标与通用五类数组（20）
- [x] Rank-major / Expert-major 端到端数据流对照卡
- [ ] Hybrid：scaleup + scaleout 时两条流如何叠加

### 下一知识点

`hybrid_dispatch` / `hybrid_combine`：两级 staging 下，rank-major 段与 expand 段如何衔接，ReturnInfo 多一层 scaleout 元数据。
