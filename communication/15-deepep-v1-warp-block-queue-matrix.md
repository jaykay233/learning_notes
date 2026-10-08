# DeepEP V1：Normal / Low-latency × Dispatch / Combine × Intranode / Internode —— Warp·Block·队列对照

> 承接 [10](./10-deepep-v1-intranode-dispatch-kernel.md)–[14](./14-deepep-v1-low-latency-combine-and-pack.md)。本文把讨论里反复对照、但尚未集中落盘的三件事写成一张可查表：
>
> 1. **发送**侧 block / warp 分配：谁负责什么
> 2. **中转**队列模型：环形 head/tail 还是定容 staging
> 3. **接收**侧 block / warp 分配：谁负责什么
>
> 代码：`csrc/kernels/legacy/intranode.cu`、`internode.cu`、`internode_ll.cu`；常量 `LEGACY_NUM_MAX_NVL_PEERS=8`（`compiled.cuh`）。

```text
本次讲解位置
章节：communication / DeepEP V1
小节：SM / warp / 队列三维对照矩阵
知识点：normal 机内偶数发奇数收；normal 跨机 RDMA+NVL 双环；LL 无 channel、warp-group×expert、定容 staging
上次：LL combine 打包回程（14）
下次：normal internode 精读，或 V2 elastic
PTX / 原语：st.release.sys / ld.acquire.sys、bar.sync、atomicAdd 抢槽、ibgda put、grid.sync
```

## 为什么现在讲这个

读完 intranode 与 LL 后，最容易混的不是「某行代码在干什么」，而是**换一条路径后角色还在不在同一套坐标系里**：

| 误读 | 正确 |
|---|---|
| LL 也有 channel 偶数发奇数收 | LL **没有** channel；同一 grid 先 SEND 再 RECV（可分 phase） |
| head/tail = 链表指针 | 是**单调计数器 + 环形槽** `slot = idx % N`，不是节点链表 |
| internode 与 intranode 同构 | 跨机多一层 **RDMA 环 + NVL 环 + forwarder warp** |
| combine 与 dispatch 同一种 warp 切法 | 机内 combine 发送轮转、接收 warp0 专推 head；LL combine 按 expert 段回写 |

不把三维矩阵钉死，后面精读任意一条路径都会把「负责谁」读串。

## 0. 先画总图：四条主路径

```text
                    ┌──────────── normal ────────────┐     ┌──── low-latency ────┐
                    │  intranode.cu   internode.cu   │     │  internode_ll.cu    │
dispatch            │  机内 NVLink     RDMA+NVL 转发  │     │  直连 RDMA（定容） │
combine             │  机内 NVLink     RDMA+NVL 回程  │     │  直连 RDMA（定容） │
中转                │  每 (ch,rank) 一环  双层环     │     │  staging 数组+门铃 │
最终布局            │  rank-major                    │     │  expert-major      │
```

**重要边界：**

- V1 **LL 只有 `internode_ll.cu` 这一条**，没有单独的「LL intranode」kernel；机内也走 RDMA buffer / P2P ptr。
- Normal **单机**走 `intranode.cu`；**跨机**走 `internode.cu`（机内 NVL 段 + 机间 RDMA 段）。
- Normal 的「channel」≈ `num_sms/2`；LL 的「条带」是 `token_idx = sm_id; += num_sms`，**不是** channel 队列。

默认数量级（便于脑内代入，以配置为准）：

| 量 | Normal 机内典型 | Normal 跨机典型 | LL 典型 |
|---|---|---|---|
| `num_sms` / grid | 20（偶数） | `num_channels*2` | 用户/配置 `num_sms` |
| 每 block 线程 | 768（24 warp） | dispatch ≈ (7+1+8)×32=512；combine ≈ (24+1)×32 | `num_warp_groups * num_warps_per_group * 32` |
| channel 数 | 10 | 同左 | **无** |
| 中转容量 | `num_recv_buffer_tokens`（如 256）环 | RDMA/NVL 各自 chunk 环 | `C = num_max_dispatch_tokens_per_rank` 定容 |

---

## 1. Normal · Intranode · Dispatch

源码：`intranode.cu` `dispatch`（约 L211–L530）。

### 1.1 发送：block / warp

```text
is_sender = (sm_id % 2 == 0)          // 偶数 block 发
responsible_channel = sm_id / 2       // 本 block 的 channel
responsible_rank = thread_id / (768/P) // 本组目标 rank（P=8 → 每组 96 线程 = 3 warp）
```

| 单位 | 负责 |
|---|---|
| **偶数 block `2c`** | channel `c` 的**全部发送** |
| **线程组 `[96R, 96R+96)`** | 往目标 rank `R` 的队列写（本组 3 warp） |
| **组内 warp `w=0..2`** | 交错拷贝：`tail % 3 == w` 的槽由该 warp 写 x/topk/src |
| **组内 warp0 + elect** | 写 `start/end offset`（编码 `-v-1`）；批末 `st.release.sys` 推进 **tail** |
| **组内 named barrier** | `bar.sync R, 96`：只同步「对端 R」这 3 个 warp |

token 范围：`get_channel_task_range(N, num_channels, c, …)` —— channel 切 token 条带。

### 1.2 中转队列

**不是链表。** 每个 `(channel c, 发送 rank S → 接收 rank R)` 一条**环形队列**，数据与元数据都在 **接收卡 R** 的 IPC buffer：

```text
元数据（int）：start_offset, end_offset, head_idx, tail_idx
数据环：N = num_recv_buffer_tokens 槽
  slot = 序号 % N
  每槽：x / src_idx / topk_idx / topk_weights / scales

流控：
  发送端写数据 → release 推 tail（累计序号，只增）
  接收端 acquire 读 tail → 搬完后写 head（释放槽）
  占用 = tail - head；满则发送端自旋等 head
```

另：`send_head[token][dst_rank] = 本 channel 内序号或 -1`，给 combine 回家用。

### 1.3 接收：block / warp

```text
奇数 block 2c+1：channel c 的接收端
responsible_rank = 来源 rank S（同一套 thread_id/96 分组）
```

| 单位 | 负责 |
|---|---|
| **奇数 block** | 从各来源 S 的环读出 → 写入本卡 `recv_x`（rank-major） |
| **每来源 3 warp** | 交错搬 `[head, tail)` 内 chunk |
| **线程0 / warp0** | 等 offset、读 acquire **tail**，写入 smem 广播 |
| **组内最后 warp** | 批末写 **head** 释放槽 |
| **落点** | `rank_prefix + channel_start + 序号`（确定行，无原子抢行） |

---

## 2. Normal · Intranode · Combine

源码：`intranode.cu` `combine`（约 L705–L1113）；准备阶段 `cached_notify_combine`。

### 2.1 发送：block / warp（与 dispatch **不同**）

仍是：偶数 block 发、奇数收、`responsible_channel = sm_id/2`。

但发送端对端划分改为**轮转**：

```text
send_warp_id         = thread_id / 32          // 0..23
send_rank_id         = (channel + send_warp_id) % P
send_warp_id_in_rank = send_warp_id / P        // 0..2
```

| 单位 | 负责 |
|---|---|
| **偶数 block `2c`** | 把本卡上「属于 channel c、应回各 src rank」的 expert 输出写入对方环 |
| **warp → `send_rank_id`** | 轮转目标，避免 24 warp 全挤一个 dst |
| **组内 3 warp** | 交错拷贝 chunk；warp0 release **tail** |

token/段范围来自 `channel_prefix_matrix` + 该 rank 段长（dispatch 留下的布局）。

### 2.2 中转队列

与 dispatch **同一类环形队列**（仍在「回家目标卡」显存），head/tail 流控相同。
`cached_notify_combine` 会把 `send_head` 里的空洞编成 `−last−1`，避免 combine 卡在「没发给该 rank」的 token 上。

### 2.3 接收：block / warp（与 dispatch **不同**）

| 单位 | 负责 |
|---|---|
| **奇数 block `2c+1`** | 本卡原始 token 的 channel 条带上做 **reduce** |
| **warp 0** | 专职 **head updater**：lane `i<P` 轮询各 rank 的 tail；取各 worker 的 `min(warp_channel_head_idx)` 推进全局 head |
| **warp 1..23** | worker：按 token 交错；lane `i` 读 `send_head[token][i]`；`__any_sync` 等齐 → 从各 rank 槽 Σ（+bias）→ `combined_x` |

一句话：**dispatch 接收按来源拷贝；combine 接收按 token 归约，warp0 只管推进 head。**

---

## 3. Normal · Internode · Dispatch

源码：`internode.cu` `dispatch`（约 L452–L1250）。`LEGACY_NUM_MAX_NVL_PEERS=8`，`kNumDispatchRDMASenderWarps=7`。

### 3.1 Block 极性（与机内「偶数发」语义不同）

```text
num_channels = num_sms / 2
channel_id   = sm_id / 2
is_forwarder = (sm_id % 2 == 0)   // 偶数 = forwarder（RDMA→NVL）
                                 // 奇数 = RDMA sender + NVL receivers
```

每 block warp 角色（`WarpRole`，约 L487–L516）：

| Block | Warp 范围 | 角色 | 负责 |
|---|---|---|---|
| **偶数（forwarder）** | `warp_id < 8` | `kRDMAAndNVLForwarder` | 从某 RDMA 源环读 → 写入目标 **NVL peer** 的 NVL 环；`target = (warp_id+channel)%8` |
| 偶数 | 其余 | `kForwarderCoordinator` | forwarder 侧协调 / 推进 RDMA head 等 |
| **奇数（非 forwarder）** | `warp_id < 7` | `kRDMASender` | 本 channel token 条带 → 打包装进 **RDMA 对称环**，ibgda put；lane 扫各 `dst_rdma_rank` |
| 奇数 | `warp_id == 7` | `kRDMASenderCoordinator` | 协调发送、推 RDMA **tail**（AMO）等 |
| 奇数 | 其后 8 warp | `kNVLReceivers` | 从各 NVL peer 环读 → 落本卡 `recv_x`；`target` 轮转 |

### 3.2 中转队列（双层环）

```text
RDMA 层（SymBuffer，每 channel × 每 rdma_rank）：
  rdma_channel_data[槽] + rdma_channel_head/tail (+ meta)
  机间：sender put → 对端 forwarder 消费 → 推 head

NVL 层（AsymBuffer，机内 peer）：
  nvl_channel_x[槽] + nvl_channel_head/tail + prefix start/end
  forwarder 写入目标 nvl peer；本卡 NVLReceivers 读出

仍是环形 head/tail，不是链表；只是多了一跳「RDMA 环 → NVL 环」。
```

回程元数据：`send_rdma_head` / `send_nvl_head`（类似机内 `send_head`）。

### 3.3 接收侧落点

最终仍是 **rank-major `recv_x`**；NVLReceivers 按 prefix / channel 偏移落行。Forwarder **不是**最终消费者，只是跨机中转。

---

## 4. Normal · Internode · Combine

源码：`internode.cu` `combine`（约 L1721–L2280）。`kNumCombineForwarderWarps=24`。

### 4.1 Block 极性（相对 dispatch **翻转**）

```text
is_forwarder_sm = (sm_id % 2 == 1)   // 奇数 = NVL→RDMA forwarder
                                     // 偶数 = NVL sender + RDMA receiver
```

| Block | 角色 | 负责 |
|---|---|---|
| **偶数** | `kNVLSender`（前 8 warp，轮转 dst nvl） | 把本卡 expert 输出写入目标 NVL peer 的环 |
| 偶数 | `kRDMAReceiver` | 从 RDMA 环取回 → reduce 进 `combined_x` |
| 偶数 | `kCoordinator` | 协调 |
| **奇数** | `kNVLAndRDMAForwarder` | 读 NVL 环 → 打包装 RDMA 环发回源 RDMA rank |
| 奇数 | `kCoordinator` | 协调 |

### 4.2 中转队列

仍是 **NVL 环 + RDMA 环** 的 head/tail；combine 方向与 dispatch 相反（expert 主机 → 原 token 卡）。
NVL 侧注释写明：为避免死锁，**按不同 RDMA 源拆开 NVL buffer**（约 L1794）。

### 4.3 接收

`kRDMAReceiver` 在原卡上按 `combined_rdma_head` / `combined_nvl_head` 与 token 条带做归约（细节同「用 dispatch 留下的 head 元数据回家」这一家族）。

---

## 5. Low-latency · Dispatch（仅 `internode_ll.cu`）

源码：约 L128–L461；host 启动约 L467–L551。

### 5.1 发送：block / warp（无偶数发奇数收）

```text
num_warps = num_warp_groups * num_warps_per_group   // 通常凑满 32 warp
warp_group_id = warp_id / num_warps_per_group
sub_warp_id   = warp_id % num_warps_per_group
responsible_expert_idx = sm_id * num_warp_groups + warp_group_id
```

| 单位 | SEND 职责 |
|---|---|
| **每个 SM** | token 条带：`for token = sm_id; token < N; token += num_sms` |
| **warp `0 .. num_warps-2`** | 数据 warp：`warp_id < K` 时读 `topk_idx[token][warp_id]`；抢槽 `atomicAdd`；put 到 dst 的 staging；累加 finish |
| **末 warp `num_warps-1`** | 统计：读 topk 给 `shared_num_tokens_sent_per_expert`；SM0 做 `+TAG` 初始化 finish |
| **发 count** | `responsible_expert_idx < E && sub_warp_id==0 && lane_id==0`：等 finish=`2×TAG` → 写 `rdma_recv_count = -n-1` |

同一 warp 在 SEND 后半还可参与「按 expert 发 count」；**不是**「一个 warp 绑死一个 expert 发数据」。

### 5.2 中转队列（定容 staging，非环）

```text
staging: rdma_recv_x[local_expert][src_rank][slot]   // slot ∈ [0, C)
门铃:    rdma_recv_count[local][src] = -n-1         // 到齐人数
完成:    atomic_finish_counter_per_expert            // 发端本地 2×TAG，不是数据 dst

特点：
  - 容量 C 固定，一次 dispatch 内不回收复用（与 normal 环不同）
  - slot 由 atomicAdd 抢，≠ 源 token_idx
  - 无 head/tail 流控；靠 count 门铃 + finish 协议
```

RECV 再压成 expert-major：`begin = atomicAdd(packed_recv_count[local], n)`（先来后到），写 `layout_range` / `src_info`。

### 5.3 接收：block / warp

| 单位 | RECV 职责 |
|---|---|
| **warp group → `responsible_expert_idx`** | 拆成 `(src_rank, local_expert)`：**等该对的 count** |
| **sub_warp 1 + lane0** | 轮询 `rdma_recv_count`；解码 `n=-count-1`；`atomicAdd` 抢 packed `begin`；写 layout |
| **组内所有 sub_warp** | 交错拷贝 staging → `packed_recv_x[local][begin+i]`，消息头 → `src_info` |
| **group barrier** | `bar.sync(warp_group_id+2, …)` 同步 n/begin |

若 SEND+RECV 同 launch：中间 `cg::this_grid().sync()`。

---

## 6. Low-latency · Combine

源码：约 L715–L1138。

### 6.1 发送：block / warp

仍用 **warp group → `responsible_expert_idx`**，但语义变为「回家方向」：

```text
dst_rank          = responsible_expert_idx / L     // 当初的 src
local_expert      = responsible_expert_idx % L
global_expert     = rank * L + local
layout_range[local][dst] → (n, offset)
src_info[local][row] → 源 token_idx
put → 源卡 rdma_recv_x[global_expert * C + src_idx]
门铃 → rdma_recv_flag[global_expert] = 1（整段发完，不是 -n-1）
```

| 单位 | 负责 |
|---|---|
| **warp group** | 一个 `(dst_rank, local_expert)` 段的 SEND |
| **组内 sub_warps** | 交错发 packed 行；TMA 装 hidden；ibgda put |
| **sub_warp 1 + lane0** | 组 barrier 后写 **flag=1** |

### 6.2 中转队列

```text
源卡定容槽：rdma_recv_x[global_expert][src_token_idx]   // 按「专家 × 源 token」寻址
门铃：rdma_recv_flag[expert] = 1                      // 该专家整段到齐
无 head/tail；无 channel
```

### 6.3 接收：block / warp

等 flag 后 `grid.sync`，再**重绑 warp 组**做 decode/reduce（约 L978+）：

| 单位 | 负责 |
|---|---|
| **先：原 warp group** | `sub_warp0` 等 `rdma_recv_flag[responsible_expert_idx]` |
| **后：decode group** | token 条带 `token = sm_id + num_sms*group; += num_sms*num_groups` |
| **decode 末 warp** | TMA load：按 `topk_idx[token][k]` 取 `rdma_recv_x[expert*C+token]` |
| **其余 decode warps** | 解码 / × `topk_weights` 累加 → `combined_x[token]` |

---

## 7. 总对照表（查表用）

### 7.1 发送侧

| 路径 | Block 怎么分 | Warp 怎么分 | 每人干什么 |
|---|---|---|---|
| N-intra dispatch | 偶发奇收；channel=`sm/2` | 每对端 3 warp（`tid/(768/P)`） | 写环、推 tail、记 send_head |
| N-intra combine | 同上极性 | `(ch+warp)%P` 轮转 dst | 写回家环、推 tail |
| N-inter dispatch | 偶=forwarder，奇=RDMA发+NVL收 | 7 RDMA sender +1 coord +8 NVL / 8 forwarder+coord | RDMA put 或 RDMA→NVL 转发 |
| N-inter combine | **奇**=forwarder，**偶**=NVL发+RDMA收 | 8 NVL sender + RDMA recv / forwarder 组 | 反向双环 |
| LL dispatch | 全体 SM 同构；phase SEND | 前 warps=数据条带；末 warp=计数；group 发 count | 抢 staging 槽 + finish/count |
| LL combine | 全体 SM；phase SEND | warp group=一段 `(dst,local)` | layout+src_info put + flag |

### 7.2 中转模型

| 路径 | 结构 | 流控 / 门铃 | 复用 |
|---|---|---|---|
| N-intra | 每 `(channel, src→dst)` **一环** | head/tail 单调计数；`slot=idx%N` | 环内复用 |
| N-inter | **RDMA 环 + NVL 环**（双层） | 各层 head/tail；meta/prefix | 各层 chunk 复用 |
| LL dispatch | `staging[local][src][slot]` 数组 | `count=-n-1`；发端 `finish=2×TAG` | 同一次内不定容回收 |
| LL combine | `buf[expert][src_token]` 数组 | `flag=1` | 定容；按 token 槽 |

### 7.3 接收侧

| 路径 | Block / 角色 | Warp | 最终写入 |
|---|---|---|---|
| N-intra dispatch | 奇数 block / 来源分组 | 3 warp 搬环；末 warp 推 head | `recv_x` rank-major |
| N-intra combine | 奇数 block | warp0 推 head；1..23 reduce | `combined_x` |
| N-inter dispatch | 奇数 block 的 NVLReceivers | 每 NVL peer 一轮转 warp | `recv_x` rank-major |
| N-inter combine | 偶数 block 的 RDMAReceiver | 按 head 元数据归约 | `combined_x` |
| LL dispatch | 同 SM 的 warp group=`(src,local)` | sub1 等 count；全组打包 | `packed_recv_x` expert-major |
| LL combine | grid sync 后 decode 重组 | TMA warp 拉 topk 槽；worker ×weight | `combined_x` |

### 7.4 一句话速记

```text
Normal 机内：偶数发奇数收 × channel ×（每对端 3 warp）× 单环 head/tail
Normal 跨机：偶/奇拆 forwarder vs sender/receiver × 双环 RDMA+NVL（combine 极性对调）
LL：无 channel；token 条带 + warp-group×expert；定容数组 + count/flag；打包 expert-major
```

---

## 8. 完整可运行核对（队列类型判别）

```python
#!/usr/bin/env python3
"""Classify DeepEP path by queue model and SM polarity."""

from __future__ import annotations


def classify(mode: str, scope: str, op: str) -> dict[str, str]:
    key = (mode, scope, op)
    table = {
        ("normal", "intra", "dispatch"): {
            "sm": "even=send, odd=recv; channel=sm//2",
            "queue": "ring head/tail per (channel,src->dst)",
            "recv_warp": "3 warps per source rank; last advances head",
        },
        ("normal", "intra", "combine"): {
            "sm": "even=send, odd=recv; channel=sm//2",
            "queue": "same ring family; send_head holes encoded",
            "recv_warp": "warp0 head updater; others reduce tokens",
        },
        ("normal", "inter", "dispatch"): {
            "sm": "even=forwarder RDMA->NVL; odd=RDMA send+NVL recv",
            "queue": "RDMA ring + NVL ring",
            "recv_warp": "NVLReceivers -> rank-major recv_x",
        },
        ("normal", "inter", "combine"): {
            "sm": "odd=forwarder NVL->RDMA; even=NVL send+RDMA recv",
            "queue": "NVL ring + RDMA ring (reverse)",
            "recv_warp": "RDMAReceiver reduce to combined_x",
        },
        ("ll", "inter", "dispatch"): {
            "sm": "all SMs; SEND then RECV phases; no channel",
            "queue": "staging[local][src][slot] + count=-n-1",
            "recv_warp": "warp-group per (src,local); pack expert-major",
        },
        ("ll", "inter", "combine"): {
            "sm": "all SMs; SEND then RECV; warp-group per return segment",
            "queue": "buf[expert][src_token] + flag=1",
            "recv_warp": "wait flag; decode groups weighted reduce",
        },
    }
    if key not in table:
        raise KeyError(f"no LL-intranode path; got {key}")
    return table[key]


def main() -> None:
    assert "ring" in classify("normal", "intra", "dispatch")["queue"]
    assert "staging" in classify("ll", "inter", "dispatch")["queue"]
    assert "forwarder" in classify("normal", "inter", "dispatch")["sm"]
    # polarity flip
    d = classify("normal", "inter", "dispatch")["sm"]
    c = classify("normal", "inter", "combine")["sm"]
    assert "even=forwarder" in d and "odd=forwarder" in c
    try:
        classify("ll", "intra", "dispatch")
        raise AssertionError("should not exist")
    except KeyError:
        pass
    print(classify("ll", "inter", "combine"))
    print("PASS")


if __name__ == "__main__":
    main()
```

期望输出末行 `PASS`。

## 9. 常见错误与症状

| 误读 | 正确 | 症状 |
|---|---|---|
| head/tail 是链表 next 指针 | 单调序号 + `% N` 环 | 去找「节点」字段找不到 |
| LL 也有 send/recv 成对 SM | 同 SM 两 phase；expert 维用 warp group | 找不到 `is_sender` |
| internode combine 与 dispatch 同极性 | combine 奇偶对调 | 读错谁是 forwarder |
| staging slot = 源 token | slot 仅中转坑；身份在消息头/`src_info` | combine 写错回家槽 |
| 机内 combine 仍「3 warp 按来源搬」 | warp0 推 head，其余按 token reduce | 对不上 `__any_sync` |

## 10. 速记卡

```text
Normal 机内：偶发奇收 · 每对端 3 warp · 单环 head/tail · combine 收端 warp0 专推 head
Normal 跨机：双环 · dispatch 偶 forwarder / combine 奇 forwarder · 角色枚举 WarpRole
LL：无 channel · token×SM 条带 · warp-group×expert · staging/flag 定容 · expert-major
slot≠src_idx≠row；环≠链表
```

## 11. 自测题与答案

1. **Normal 机内 dispatch：block 4、P=8、tid=200，负责什么？**
   答：偶数→发送；channel=`4/2=2`；`responsible_rank=200/96=2` → channel 2 往 rank 2 写环。

2. **LL 有没有「偶数 SM 只发、奇数 SM 只收」？**
   答：没有。同一 SM 可跑 SEND+RECV；分工是 token 条带 + warp-group 专家维。

3. **Internode dispatch 与 combine 的 forwarder 各在奇偶哪侧？**
   答：dispatch 偶数 forwarder；combine 奇数 forwarder。

4. **为何说中转不是链表？**
   答：只有 head/tail 两个计数器与固定长度槽数组；`slot = idx % N`，没有 per-token next 指针。

5. **LL dispatch 接收端「等齐」看的是 head 还是 count？**
   答：`rdma_recv_count`（`-n-1`）；没有 channel head。

## 12. 学习进度

- [x] 机内 / LL 各路径精读（10–14）
- [x] Normal×LL × dispatch/combine × intra/inter 的 warp·block·队列矩阵
- [ ] Normal internode 按 WarpRole 逐段精读（可选下一课）

### 下一知识点

`internode.cu` dispatch：按 `kRDMASender` → `kRDMAAndNVLForwarder` → `kNVLReceivers` 走读一跳 token。
