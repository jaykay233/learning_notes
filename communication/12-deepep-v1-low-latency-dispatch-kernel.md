# DeepEP V1：Low-latency `dispatch` kernel —— 定容槽位、原子完成计数与 expert-major 打包

> 承接数据流笔记 [05](./05-deepep-v1-low-latency-dataflow.md)、[06](./06-deepep-v1-normal-and-low-latency-dataflow.md)，以及机内 normal 实现 [10](./10-deepep-v1-intranode-dispatch-kernel.md)/[11](./11-deepep-v1-intranode-combine-and-warp-roles.md)。本文进入 **`internode_ll.cu` 的 `dispatch` kernel 读码**：没有环形队列与 `notify_dispatch`，改成「按 expert 定容槽 + 原子抢号 + RDMA/P2P 写出 + 收端压成 expert-major」。
>
> 代码：`/Users/saboxu/Downloads/communication/DeepEP/csrc/kernels/legacy/internode_ll.cu`（kernel ≈ L129–L462，host ≈ L464–L553）；常量 `compiled.cuh`；Python `deep_ep/buffers/legacy.py` 的 `low_latency_dispatch`。

```text
本次讲解位置
章节：communication / DeepEP V1
小节：low-latency dispatch kernel 读码
知识点：SEND/RECV phase；warp 分工（发数据 / 计专家人数）；atomic 槽位；finish 计数凑齐 2×TAG；count 编码 −n−1；收端按 (local_expert, src_rank) 打包进 [L,C·P,H]
上次：机内 combine / warp / handle·cached notify
下次：low-latency combine（layout_range + topk_weights 加权收回）
PTX / 原语：atomicAdd、ld.acquire / st.release.sys、ibgda put、nvshmem P2P、cg::this_grid().sync、bar.sync
```

## 为什么现在讲这个

读完 normal 机内路径后，容易带着「channel / head-tail / send_head」去扫 LL，结果对不上号。LL dispatch 要解决的是另一组约束：

1. **Decode 要低延迟**：不能先跑 `get_dispatch_layout` + `notify_dispatch` 再搬数据；路由在同一个 kernel 里完成。
2. **容量事先固定**：调用方给出 `C = num_max_dispatch_tokens_per_rank`，且 `N < C`。槽位地址用公式算，不再维护环形 head/tail。
3. **接收要直接给 GEMM**：产出 `packed_recv_x[L, C·P, H]`（expert-major），而不是 rank-major + 旁路 `recv_topk_idx`。
4. **发送完成与「该发几个」必须对齐**：用 `atomic_finish_counter` 拼出 `2 × LEGACY_FINISHED_SUM_TAG`，再把人数用 `−n−1` 通知对端——这是读懂收端自旋的关键。

搞不清这四点，会把 RDMA staging 区当成最终 `recv_x`，或把 `C·P` 容量当成有效 token 数。

## 0. 心智模型

```text
每个 rank 启动 num_sms 个 block（约 ceil(E / num_warp_groups)）
block 内：num_warp_groups × num_warps_per_group 个 warp（线程数 = 该积 × 32）

SEND phase（可单独跑，配合 recv hook）:
  多数 warp：按 token 跨 SM 条带
    读 x →（可选 FP8 cast）写入本卡 rdma_x 打包消息
    对本 token 的每个 topk expert：atomicAdd 抢 slot → RDMA/P2P 写到
      对端 rdma_recv_x[local_expert][src_rank][slot]
  最后一个 warp：统计本 SM 负责的全局 expert 各该发多少
    与发送 finish 计数拼成 2×TAG 后，把 −n−1 写入对端 rdma_recv_count

RECV phase:
  每个 warp group 负责一对 (src_rank, local_expert)（下标 = src·L + local）
  等 rdma_recv_count ≠ 0 → 解码 n → atomicAdd 得到本 expert 内打包起点
  把 staging[local][src][0..n) 拷进 packed_recv_x[local][begin ..)
  写下 layout_range[local][src] = pack(n, begin)；累加 packed_recv_count[local]
```

对比 normal 一句话：**normal 是「按目的 rank 排队进环」；LL 是「按目的 expert 抢定容槽，再压成 expert 连续区」。**

## 1. 术语与启动参数

| 名字 | 典型来源 | 含义 |
|---|---|---|
| `E` / `P` / `L` | `num_experts` / `num_ranks` / `E/P` | 全局 expert、EP 规模、每卡本地 expert |
| `N` | `x.size(0)` | 本卡本步 token 数，必须 `< C` |
| `C` | `num_max_dispatch_tokens_per_rank` | 每卡每 expert 方向的槽位上限 |
| `K` | `num_topk` | top-k，代码里 `kNumMaxTopK = 11` |
| `num_warp_groups` | `ceil(E / num_device_sms)` | 每 SM 管几组 expert |
| `num_warps_per_group` | `32 / num_warp_groups` | 每组几个 warp；要求 `> 1`（收端 sub_warp 0/1 分工） |
| `num_sms` | `ceil(E / num_warp_groups)` | grid 大小 |
| `responsible_expert_idx` | `sm_id * num_warp_groups + warp_group_id` | 本 warp group 的逻辑下标 |
| `LEGACY_FINISHED_SUM_TAG` | `1024` | finish 计数协议常量 |
| `phases` | `SEND=1`, `RECV=2` | 可拆开发送与接收（`return_recv_hook`） |

消息布局（每槽一次 put）：

```text
[ int4 头：src_token_idx + 3 个保留 int ]
[ hidden 载荷：FP8 或 BF16 ]
[ 可选 FP8 scales ]
```

Staging（对端可见的 RDMA 接收暂存）：

```text
rdma_recv_x[local_expert][src_rank][slot]   // slot ∈ [0, C)
rdma_recv_count[local_expert][src_rank]     // 编码后的人数，0 = 未到
```

最终打包（返回给 Python 的 `recv_x`）：

```text
packed_recv_x[local_expert][0 .. C·P)   // 前 recv_count[local] 个有效
packed_recv_src_info[...]               // 每个有效槽的源 token 下标
packed_recv_layout_range[local][src]    // pack(num, begin)
```

## 2. Warp 分工（SEND）

```text
warp_id ∈ [0, num_warps - 2]  → 数据发送 warp
warp_id == num_warps - 1      → 计数 / 清理 / 凑 finish 的 warp
```

数据 warp 主循环：

```cpp
for (token_idx = sm_id; token_idx < num_tokens; token_idx += num_sms) {
    // 全 block 协作把 x[token]（+FP8）写入 rdma_x[token] 消息
    bar.sync 1, num_threads;   // 不含最后一个计数 warp

    dst_expert_idx = (warp_id < num_topk) ? topk_idx[token][warp_id] : -1;
    if (dst_expert_idx >= 0) {
        slot = atomicAdd(atomic_counter_per_expert + dst_expert_idx, 1);  // lane0
        slot = __shfl_sync(..., 0);
        // put 到对端 staging[local][我的 rank][slot]
        atomic_add_release(finish[dst_expert_idx], 1);  // 本条发送已 issue
    }
}
```

要点：

- **同一 token 的不同 topk** 由不同 warp 发（`warp_id < K`），同一 rank 上多个 local expert 会各占一槽——这就是 expert-major 下「一份输入可复制进多个 expert 区」的来源。
- **槽位顺序不确定**：`atomicAdd` 决定 slot；只保证不越界（依赖 `N < C` 且路由合理）。
- 同节点优先 `nvshmemi_get_p2p_ptr` 直接拷；否则 `nvshmemi_ibgda_put_nbi_warp`。

计数 warp（每 SM 一个）：

1. **仅 SM0**：清 next buffer；给每个 expert 的 `finish` 先加 `+TAG`。
2. 统计本 SM 负责的全局 expert 区间 `[sm_id * G, …)` 上，`topk_idx` 里出现次数 `sum`。
3. 对每个 expert：`finish += TAG - sum`。

拼在一起：

```text
finish 目标 = TAG          (SM0 垫高)
            + (TAG - sum)  (计数 warp)
            + sum × (+1)   (每个实际 put 完成)
            = 2 × TAG
```

`sum = 0` 时无需任何 put，计数 warp 加完就已是 `2×TAG`，仍会发出 `−0−1 = −1` 的 count，让收端知道「0 个 token」。

## 3. 人数通知与地址公式

凑齐 `2×TAG` 后（`sub_warp_id==0 && lane_id==0`）：

```cpp
st/amo: rdma_recv_count[dst_local_expert][我的 rank] = -num_tokens_sent - 1;
```

与 normal 的 offset 编码同一套路：`0` 表示未到，解码 `n = -v - 1`。

发送数据地址（源 rank = 我，目的 expert 全局 id = `e`）：

```text
dst_rank = e // L
local    = e %  L
byte_offset =
    local * (P * C) * msg_bytes
  + my_rank * C * msg_bytes
  + slot * msg_bytes
```

数值例子：`P=8, L=4, C=16, my_rank=3, e=9 → dst_rank=2, local=1, slot=5`

```text
offset = 1*(8*16)*msg + 3*16*msg + 5*msg
       = (128 + 48 + 5) * msg
       = 181 * msg
```

落在 **rank 2** 的 staging 上：`local_expert=1`、来自 rank 3、第 5 号槽。

## 4. RECV phase：等到人，再压紧

`phases` 只有 SEND 时直接 `return`（hook 稍后再跑 RECV）。若 SEND|RECV 同核，先 `cg::this_grid().sync()`，保证本卡 `packed_recv_count` 清零可见。

每个 warp group 负责下标 `idx = responsible_expert_idx`：

```text
src_rank         = idx // L
local_expert_idx = idx %  L
```

即覆盖所有 `(src, local)` 对，共 `P·L = E` 个，与全局 expert 数相同，但语义是「收来自 src、进入我的 local expert」。

`sub_warp_id == 1` 的 lane0：

```cpp
while (ld_acquire_sys(rdma_recv_count[local][src]) == 0) ;  // 超时可 mask/trap
n = -count - 1;
begin = atomicAdd(packed_recv_count + local, n);
layout_range[src] = pack(n, begin);
```

然后组内所有 sub_warp 按 `i = sub_warp_id; i < n; i += num_warps_per_group` 把 staging 拷进：

```text
packed_recv_x[local][begin + i]
packed_recv_src_info[local][begin + i] = 消息头里的源 token 下标
```

注意：`begin` 来自各 src 到达时的 `atomicAdd`，**不同 src 写入 packed 区的先后不确定**；`layout_range` 记录每段，combine 按 src 找回，不依赖「按 rank 顺序紧排」。

## 5. 与 normal dispatch 对照

| 维度 | Normal 机内 | Low-latency |
|---|---|---|
| 前置 kernel | layout + notify（可 cache） | 无 |
| 缓冲 | 每 (channel, src) 环形队列 | 每 (local_expert, src) 定容 `C` 槽 |
| 流控 | head/tail + 分批 | 靠 `N < C`；抢完 slot 即写 |
| 完成同步 | release tail / acquire | finish 凑 `2×TAG` + count `−n−1` |
| 产出布局 | rank-major + `recv_topk_idx` | expert-major packed + `recv_count` |
| 同 rank 多 expert | 通常一份 token | 每 expert 各占槽（可复制） |
| 通信后端 | NVLink IPC | IBGDA / NVSHMEM P2P（要求全员 RDMA 可见） |

## 6. 完整可运行验证（地址 + finish + 打包）

不依赖 GPU：模拟「抢槽、凑 finish、编码 count、收端 decode + 紧排」。

```python
#!/usr/bin/env python3
"""Simulate DeepEP V1 low-latency dispatch slot + finish + pack."""

from __future__ import annotations

TAG = 1024


def encode_count(n: int) -> int:
    return -n - 1


def decode_count(v: int) -> int:
    assert v != 0
    return -v - 1


def staging_offset(local: int, src_rank: int, slot: int, P: int, C: int) -> int:
    return (local * P * C + src_rank * C + slot)


def finish_reaches_two_tag(sum_tokens: int, sends_done: int, sm0_tag: bool, counted: bool) -> bool:
    total = 0
    if sm0_tag:
        total += TAG
    if counted:
        total += TAG - sum_tokens
    total += sends_done
    return total == 2 * TAG


def pack_from_sources(arrivals: list[tuple[int, list[int]]]) -> tuple[list[int], dict[int, tuple[int, int]]]:
    """arrivals: list of (src_rank, token_ids in staging order)."""
    packed: list[int] = []
    layout: dict[int, tuple[int, int]] = {}
    for src, tokens in arrivals:
        begin = len(packed)
        packed.extend(tokens)
        layout[src] = (len(tokens), begin)
    return packed, layout


def main() -> None:
    P, L, C = 4, 2, 8
    my_rank = 1
    # token 0 -> experts 2,3  (dst_rank=1, local 0/1); token 1 -> expert 2
    topk = [[2, 3], [2, -1]]

    counters = [0] * (P * L)
    staging: dict[tuple[int, int, int], int] = {}
    # (dst_rank, local, slot) -> src_token；此处只模拟发往 rank1

    for t, experts in enumerate(topk):
        for e in experts:
            if e < 0:
                continue
            dst_rank, local = divmod(e, L)
            if dst_rank != 1:
                continue
            slot = counters[e]
            counters[e] += 1
            assert slot < C
            off = staging_offset(local, my_rank, slot, P, C)
            staging[(local, my_rank, slot)] = t
            assert off == local * P * C + my_rank * C + slot

    assert counters[2] == 2 and counters[3] == 1

    # finish protocol for expert 2 (sum=2)
    assert finish_reaches_two_tag(2, 0, True, True) is False
    assert finish_reaches_two_tag(2, 2, True, True) is True
    assert finish_reaches_two_tag(0, 0, True, True) is True

    enc = encode_count(2)
    assert enc == -3 and decode_count(enc) == 2

    # Rank1 local0 receives from src1: tokens [0,1] in slot order
    arrivals = [(1, [0, 1]), (0, [7])]  # another src arrived first or later
    packed, layout = pack_from_sources(arrivals)
    assert packed == [0, 1, 7]
    assert layout[1] == (2, 0)
    assert layout[0] == (1, 2)

    print("counters", counters)
    print("encode/decode", enc, decode_count(enc))
    print("packed", packed, "layout", layout)
    print("PASS")


if __name__ == "__main__":
    main()
```

期望输出：

```text
counters [0, 0, 2, 1, 0, 0, 0, 0]
encode/decode -3 2
packed [0, 1, 7] layout {1: (2, 0), 0: (1, 2)}
PASS
```

## 7. 常见错误与症状

| 错误理解 | 正确理解 | 症状 |
|---|---|---|
| `recv_x` 的 `C·P` 行全有效 | 只有前 `recv_count[l]` 有效 | GEMM 吃到垃圾 / 算多 |
| staging 顺序 = packed 顺序 | packed 按到达 atomic 紧排；靠 `layout_range` 还原 | combine 找错 src 段 |
| 不需要等 finish 就能发 count | 必须 `finish == 2×TAG` | count 偏小，收端少拷 |
| `responsible_expert_idx` 收发语义相同 | 发侧当全局 expert；收侧当 `(src, local)` | 地址算到别人的区 |
| `N ≥ C` 仍能跑 | 接口要求严格小于 | 槽位越界，静默踩内存或断言 |
| 只有 SEND phase 就当数据已进 `packed_recv_x` | hook 模式下必须再调 RECV | 读到未打包的空/旧数据 |

## 8. 速记卡

```text
启动：G=ceil(E/SMs), Wpg=32/G, grid=ceil(E/G), threads=G*Wpg*32
SEND：多数 warp 按 token 条带发；末 warp 计数；finish = TAG + (TAG-sum) + sum·1
槽：atomicAdd(per expert)；dst = staging[e%L][my_rank][slot] @ owner(e)
通知：count[local][src] = -n-1
RECV：idx→(src,local)；等 count；n=-v-1；begin=atomicAdd；拷到 packed[local][begin+)
handle：(src_info, layout_range, C, H, E)
phases：可只 SEND，hook 再 RECV；同核则 grid.sync
```

## 9. 自测题与答案

1. **为何 `finish` 目标是 `2×TAG` 而不是 `sum`？**
   答：需要同时确认「计数 warp 已给出 sum」和「sum 次 put 都已 issue」。`TAG+(TAG-sum)+sum = 2×TAG`，`sum=0` 也能成立。

2. **`P=8,L=4`，warp group 的 `responsible_expert_idx=13` 在 RECV 负责谁？**
   答：`src_rank=13//4=3`，`local_expert=13%4=1` → 收 rank3 发给本卡 local expert1 的 staging。

3. **同一 token topk=`[4,5]` 且 4、5 同属一卡，normal 与 LL 接收有何差异？**
   答：normal 通常该卡只收一行，`recv_topk_idx` 标两个 local；LL 在两个 local expert 区各占一槽（数据复制）。

4. **`return_recv_hook=True` 时第一次 kernel 做了什么、没做什么？**
   答：只跑 SEND（issue RDMA/P2P + 写 count）；不打包进 `packed_recv_x`。必须调用 hook 跑 RECV。

5. **`layout_range[local][src] = pack(3, 10)` 表示什么？**
   答：该 src 发给该 local expert 的 3 个 token，落在 packed 行 `[10, 13)`；`src_info` 同下标可查源 token。

## 10. 学习进度

- [x] DeepEP V1 normal 机内：layout → notify → dispatch → combine / warp / handle
- [x] DeepEP V1 low-latency 数据流与布局概念（05/06）
- [x] Low-latency `dispatch` kernel：phase、槽位原子、finish/`−n−1`、expert-major 打包

### 下一知识点

low-latency `combine`：用 `layout_range` / `src_info` 把 expert 输出按 `topk_weights` 加权写回源 rank，以及 SEND/RECV phase 与 `zero_copy`。
