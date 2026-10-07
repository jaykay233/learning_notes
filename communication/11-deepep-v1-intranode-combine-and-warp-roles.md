# DeepEP V1：机内 `combine`、warp 分工与 handle / cached notify

> 承接 [10-deepep-v1-intranode-dispatch-kernel.md](./10-deepep-v1-intranode-dispatch-kernel.md)。10 讲清楚了 dispatch 如何把 token 经环形队列送到目标卡、如何写 `send_head`。本文把 **combine 回程**、**dispatch / combine 的 warp 分配对照**、**handle 与 cached notify**、以及读代码时碰到的 `__any_sync` / `tma_store_wait` / `send_head` 负数编码 / rank-major 与 expert 对应关系一次收束。
>
> 代码对照：本地 DeepEP checkout `/Users/saboxu/Downloads/communication/DeepEP`。`cached_notify_dispatch` L175–L209、`cached_notify_combine` L626–L703、`combine` L705–L1113（均在 `csrc/kernels/legacy/intranode.cu`）；Python 入口 `deep_ep/buffers/legacy.py` 的 `dispatch` / `combine`；host 在 `csrc/legacy/buffer.hpp`。

```text
本次讲解位置
章节：communication / DeepEP V1
小节：intranode combine 与 warp / handle 收束
知识点：dispatch/combine 的 SM·warp 分工；cached_notify_dispatch / cached_notify_combine；handle 决定是否走 cache；combine 用 send_head 等待并 reduce；__any_sync / tma_store_wait<N>；send_head 空洞编码 −last−1；rank-major 靠 recv_topk_idx 标本地 expert
上次：intranode dispatch kernel（环形队列、head/tail、send_head、TMA 两半）
下次：internode（RDMA + NVLink）转发路径，或 elastic V2 布局
PTX：cp.async.bulk.wait_group N、elect.sync、__any_sync / __shfl_sync、st.release.sys / ld.acquire.sys、bar.sync
```

## 为什么现在讲这个

读完 dispatch 后，还有四个容易混在一起的问题：

1. **角色对调了，但 warp 切法并不对称。** combine 仍是「偶数 block 发、奇数 block 收」，但发送端按 `(channel + warp_id) % P` 轮转目标 rank，接收端则是 **warp0 专职推 head、其余 warp 做 reduce**。若仍按 dispatch 的「每对端 3 warp」去读，会把 lane 角色全读错。
2. **`cached_notify_*` 不是「第二次才走的 combine」。** dispatch 的 notify 有完整版 / cache 版；combine 的准备阶段 **每次都叫** `cached_notify_combine`。名字里的 cached 指复用 / 加工 dispatch 留下的元数据，不是「第二次才能 combine」。
3. **`send_head` 里的 −1 若原样拿去等，环形队列 head 会卡在空洞上。** `cached_notify_combine` 从后往前扫，把空洞编成 `−last_head−1`，combine 更新 head 时才能越过「没发给该 rank」的 token。
4. **rank-major 的 `recv_x` 行号不编码 expert。** expert 身份在并行张量 `recv_topk_idx`（本地 id）里；不知道这一点，会误以为通信库已经排成 expert-major。

不把这四点写清，就没法把 forward dispatch → expert GEMM → combine → backward 再 dispatch（带 handle）整条链路串起来。

## 0. 心智模型：两条数据传输路径

### 0.1 Dispatch（token → 目的卡，rank-major）

```text
源卡 tokens + topk_idx
        │  is_token_in_rank[t][R]
        ▼
偶数 block（channel c）发送组 responsible_rank = R
        │  写环形队列（在 R 的 HBM，经 IPC）
        │  slot = 序号 % N；send_head[t][R] = 序号 或 −1
        │  topk → 本地 expert id（不属于 R 的槽写 −1）
        ▼
奇数 block（channel c）接收组 responsible_rank = S（来源）
        │  行号 = rank_offset(S) + channel_start + 序号
        ▼
recv_x[rank-major] + recv_topk_idx + recv_src_idx + send_head(handle)
```

### 0.2 Combine（expert 输出 → 原卡，按 token 求和）

```text
各卡 expert 输出（仍是 dispatch 留下的 rank-major 布局）
        │  偶数 block：按 src_idx / prefix 把结果写回「原发送卡」的环形队列
        ▼
原发送卡奇数 block：
        │  lane i 读 send_head[token][i]
        │  __any_sync：任一相关 rank 的 tail 还没到 → 整 warp 等
        │  从各 rank 的槽位取出结果 Σ（+ bias）
        │  更新 warp_channel_head_idx（含空洞解码）
        ▼
combined_x[原始 token 顺序]
```

一句话：**dispatch 按目的 rank 摊开；combine 按 `send_head` 把摊开的副本按 token 收回并相加。**

## 1. Warp / SM 分配对照（重点）

默认：`num_sms = 20` → `num_channels = 10`；`kNumThreads = 768` → 24 warp；`P = 8` 卡。

### 1.1 共同骨架

| 量 | 公式 | 含义 |
|---|---|---|
| `is_sender` | `sm_id % 2 == 0` | 偶数 block 发，奇数 block 收 |
| `responsible_channel` | `sm_id / 2` | 本 block 负责的 channel |
| 每 block 线程 | 768 | 24 warp |

### 1.2 Dispatch 的 warp 切法

```text
num_threads_per_rank = 768 / P = 96     → 每对端 3 warp
responsible_rank     = thread_id / 96

发送 block（偶数）：
  线程 [96R, 96R+96) → 往目标 rank R 的队列写
  组内 send_warp_id_in_rank = 0..2
  拷贝条件：cached_channel_tail_idx % 3 == send_warp_id_in_rank
  发布 tail：warp0 + elect_one_sync；命名 barrier bar.sync R, 96

接收 block（奇数）：
  同样 3 warp / 来源 rank S
  线程0 读 acquire tail → smem → bar.sync 广播
  最后一个 warp 写 head 释放槽位
```

数值例子（本卡 rank = 1，block 4，线程 200）：

| 字段 | 值 |
|---|---|
| `is_sender` | true（偶数） |
| `responsible_channel` | 2 |
| `responsible_rank` | `200/96 = 2` |
| 角色 | channel 2 上，往 **rank 2** 发 token |

### 1.3 Combine 的 warp 切法（与 dispatch 不同）

**发送端（偶数 block，expert 主机把结果送回家）：**

```text
num_send_warps_per_rank = (768/32) / P = 3
send_warp_id            = thread_id / 32          // 0..23
send_rank_id            = (responsible_channel + send_warp_id) % P
send_warp_id_in_rank    = send_warp_id / P        // 0..2

用意：同一 channel 内，24 个 warp 轮转覆盖所有回家方向，
      避免所有 warp 挤向同一个 dst。
拷贝：for i = send_warp_id_in_rank; i < chunk; i += 3
发布：bar.sync send_rank_id, 96；warp0 release tail
```

例子：channel = 3，`send_warp_id = 5` → `send_rank_id = (3+5)%8 = 0`，`send_warp_id_in_rank = 0`。
warp 5 负责把「本卡上属于 channel 3、应回 rank 0」的那段结果写入 rank 0 的队列。

**接收端（奇数 block，原发送卡做 reduce）：**

```text
num_recv_warps = 24
recv_warp_id   = thread_id / 32

warp 0（thread_id < 32）：
  专职 Queue head updater
  lane i < P：轮询 channel_tail_idx[i]（acquire）
  取各 worker warp 的 min(warp_channel_head_idx[*][i])，推进全局 head

warp 1..23：
  worker：按 token 交错
    token_idx = start + (recv_warp_id - 1); stride = num_recv_warps - 1
  lane i < P：读 send_head[token][i] 为 expected_head
  __any_sync 等待；再对 topk 相关 rank 做 Σ；更新本 warp 的 head 进度
```

| 对比项 | Dispatch | Combine |
|---|---|---|
| 发送端对端划分 | `thread_id / 96` 固定对端 | `(channel + warp_id) % P` 轮转对端 |
| 接收端 | 仍按来源 rank 分组拷贝 | warp0 推 head；其余 warp 按 token 交错 reduce |
| 每对端 warp 数 | 发送/接收都是 3 | 发送仍约 3；接收是 1 个 updater + 23 个 reducer |
| TMA smem / warp | 8192 B（dispatch） | 4096 B（combine，`kNumTMABytesPerWarp`） |

## 2. Handle 与「怎么知道已经算过」

**库不会自动探测布局是否变了。** 完全看调用方有没有把上次 `dispatch` 返回的 `handle` 再传进来。

### 2.1 Python 判定

```python
# deep_ep/buffers/legacy.py
if handle is not None:
    rank_prefix_matrix, channel_prefix_matrix, ..., send_head = handle
    # 走 cached：传入 cached_num_recv_tokens + 两张 prefix
else:
    # 首次：get_dispatch_layout + notify_dispatch，再打包 handle 返回
    handle = (rank_prefix_matrix, channel_prefix_matrix, ..., send_head)
```

C++：

```cpp
bool cached_mode = cached_rank_prefix_matrix.has_value();
```

### 2.2 典型调用序列

```text
forward dispatch(handle=None)  → notify_dispatch（完整）+ dispatch kernel
expert GEMM（框架侧按 recv_topk_idx 分组）
combine(handle=必须)           → cached_notify_combine + combine kernel

backward 再 dispatch（同 topk）:
  dispatch(handle=同一个)      → cached_notify_dispatch + dispatch kernel
```

| 阶段 | 准备 kernel | 数据 kernel |
|---|---|---|
| 首次 dispatch | `notify_dispatch` | `dispatch` |
| 带 handle 的 dispatch | `cached_notify_dispatch` | 同一个 `dispatch` |
| 每次 combine | **总是** `cached_notify_combine` | `combine` |

注意：**不是「第一次完整 dispatch+combine，以后全是 cache 版」。**
cached 的是 dispatch 的 **notify**；combine 的准备函数名字里带 cached，但没有非 cache 对照版。

### 2.3 `cached_notify_dispatch` 做什么

完整代码：

```cpp
template <int kNumRanks>
__global__ void cached_notify_dispatch(
    const int* rank_prefix_matrix, int num_memset_int,
    void** buffer_ptrs, int** barrier_signal_ptrs, int rank) {
    barrier_block<kNumRanks, true>(barrier_signal_ptrs, rank);

    auto thread_id = static_cast<int>(threadIdx.x), num_threads = static_cast<int>(blockDim.x);
    auto ptr = static_cast<int*>(buffer_ptrs[rank]);
    for (int i = thread_id; i < kNumRanks * kNumRanks; i += num_threads)
        ptr[i] = rank_prefix_matrix[i];
    for (int i = thread_id; i < num_memset_int; i += num_threads)
        ptr[kNumRanks * kNumRanks + i] = 0;

    barrier_block<kNumRanks>(barrier_signal_ptrs, rank);
}
```

| 步骤 | 含义 |
|---|---|
| barrier | 等上一阶段结束 |
| copy prefix | 把缓存的 `rank_prefix_matrix` 写回本卡 IPC buffer |
| memset | 清 channel offset / head / tail（`num_memset_int = C·P·4`） |
| barrier | 清完再开 dispatch |

**不重算** channel 人数——那是首次 `notify_dispatch` 的事。

### 2.4 `cached_notify_combine` 做什么

两个 SM 角色：

**SM 0：** barrier → 清 head/tail（`C·P·2` 个 int）→ barrier。

**其它 SM（每 channel 一个）：** 从后往前扫本 channel 的 `send_head[·][rank_id]`：

```cpp
int last_head = 1 << 25;
for (token from end downto start) {
    current_head = send_head[token][rank_id];  // dispatch 留下的：槽位 or −1
    // warp 内广播，遇到 ≥0 则 last_head = head
    // 遇到 <0：expected_head = -last_head - 1
    if (current_head < 0)
        send_head[token][rank_id] = expected_head;
}
```

含义：空洞不再是无信息的 −1，而是 **编码「向后看到的下一个真实槽位」**，好让 combine 推进 head 时跳过空洞。

## 3. Combine 接收端关键路径

### 3.1 等待：`__any_sync`

```cpp
int expected_head = -1;
if (lane_id < kNumRanks)
    expected_head = ld_nc_global(send_head + token_idx * kNumRanks + lane_id);

while (__any_sync(0xffffffff,
                  channel_tail_idx[lane_id] <= expected_head and expected_head >= 0)) {
    // timeout → trap
}
```

| API | 语义 |
|---|---|
| `__any_sync(mask, pred)` | mask 内 **任一** lane 的 pred 为真 → 全体得 true |
| `__all_sync(mask, pred)` | 全部为真才 true |

例子：`send_head[14] = [−1, 2, −1, 5, …]`，lane1 的 tail 还是 1：

- lane1 条件真（1 ≤ 2）→ `__any_sync` 真 → 继续等
- 等 tail 变成 3 后所有相关条件假 → 退出

### 3.2 取出并 reduce

```cpp
for (i = 0; i < kNumRanks; ++i) {
    expected_head_i = __shfl_sync(..., expected_head, i);
    if (expected_head_i >= 0) {
        slot_indices[num_topk_ranks] = expected_head_i % N;
        topk_ranks[num_topk_ranks++] = i;
    }
}
// 对 topk_ranks 各读一槽，float 累加（+ bias），写回 combined_x[token]
```

只有 `expected_head ≥ 0` 的 rank 才参与求和；编码后的负数空洞不参与 reduce。

### 3.3 `tma_store_wait<N>()`

```cpp
template <int N>
__device__ void tma_store_wait() {
    asm volatile("cp.async.bulk.wait_group %0;" :: "n"(N) : "memory");
}
```

语义：等到 **已 commit、未完成** 的 bulk store group 个数 ≤ N。

| 调用 | 含义 |
|---|---|
| `tma_store_wait<0>()` | 等全部写完（复用 smem 前） |
| `tma_store_wait<kNumStages-1>()`（stages=8 → 7） | 流水线：最多 7 个在飞再开下一拍 |

combine 每个 token 开始 reduce 前 `wait<0>`；流水写回时 `wait<7>`。

### 3.4 更新 head（为何还有负数分支）

```cpp
warp_channel_head_idx[recv_warp_id][lane_id] =
    (expected_head < 0) ? -expected_head - 1 : expected_head + 1;
```

| `expected_head` | 含义 | 写入 |
|---|---|---|
| `≥ 0` | 真发给过该 rank，槽位 = 该值 | `expected_head + 1`（读完，释放到下一格） |
| `< 0` | 空洞；值为 `−last−1` | `−expected_head − 1`（解码出可推进到的 head） |

**写入 `warp_channel_head_idx` 的结果始终非负。** 负数只存在于 `send_head` 的编码里。

向前处理例子（cached_notify 已从后往前编码）：

| token | 是否发给 rank1 | notify 后 send_head | combine 更新 head |
|---|---|---|---|
| A | 槽 5 | `5` | `6` |
| B | 不发 | `−7`（向后真实槽是 6） | `6` |
| C | 槽 6 | `6` | `7` |

warp0 取各 worker 的 `min_head`，单调推全局 `channel_head`，发送端靠 `tail − head` 判断空位。

## 4. Rank-major 怎么对应 expert

`recv_x` 布局（本卡视角）：

```text
[来自 rank0 …][来自 rank1 …][…][来自 rank_{P-1} …]
```

行号 **不** 表示 expert。发送时已把全局 expert 映射成本地 id：

```cpp
recv_expert_begin = responsible_rank * num_experts_per_rank;
idx_value = (idx in [begin, end)) ? idx - begin : -1;
```

接收后：

| 张量 | 作用 |
|---|---|
| `recv_topk_idx[row, k]` | 本地 expert id，或 −1 |
| `num_recv_tokens_per_expert_list` | 每个本地 expert 收了多少（开 GEMM buffer） |
| `recv_src_idx[row]` | 源卡上的原 token 下标（combine 用） |

DeepEP **不**在通信里排成 expert-major；框架读 `recv_topk_idx` 再 permute / 分组做 GEMM。

## 5. `elect_one_sync` 与 `__shfl_sync(..., 0)` 的风险

```cpp
#ifndef DISABLE_SM90_FEATURES
    // elect.sync：规范只保证确定性，不保证选中 lane 0
#else
    return get_lane_id() == 0;
#endif
```

若某处先 `if (elect_one_sync()) x = ...;` 再 `__shfl_sync(mask, x, 0)`，而选中的不是 lane 0，会广播未初始化值。
非 SM90 路径显式 lane0 安全；SM90 上实测常选 0，但 **PTX 不保证**。读到「elect + shfl 0」配对时要警惕。

## 6. 完整可运行验证：`send_head` 编码与 head 更新

把 `cached_notify_combine` 的从后往前编码，以及 combine 的 head 更新，缩成纯 Python，不依赖 GPU。

```python
#!/usr/bin/env python3
"""Verify send_head hole encoding and combine head updates."""

from __future__ import annotations


def cached_notify_patch(send_head: list[int]) -> list[int]:
    """Backward scan for one (channel, rank) column. Input: per-token heads."""
    out = list(send_head)
    last_head = 1 << 25
    for i in range(len(out) - 1, -1, -1):
        head = out[i]
        if head < 0:
            out[i] = -last_head - 1
        else:
            last_head = head
    return out


def combine_head_updates(patched: list[int]) -> list[int]:
    """Forward scan: what warp_channel_head_idx becomes after each token."""
    heads = []
    for expected in patched:
        if expected < 0:
            heads.append(-expected - 1)
        else:
            heads.append(expected + 1)
    return heads


def main() -> None:
    # Dispatch left: slot or -1
    raw = [5, -1, -1, 6, -1, 8]
    patched = cached_notify_patch(raw)
    updates = combine_head_updates(patched)

    assert patched == [5, -7, -7, 6, -9, 8], patched
    assert updates == [6, 6, 6, 7, 8, 9], updates

    # Any-sync style wait predicate
    expected = [ -1, 2, -1, 5]
    tails = [0, 1, 0, 6]
    any_wait = any(tails[i] <= expected[i] and expected[i] >= 0 for i in range(4))
    assert any_wait is True
    tails[1] = 3
    any_wait = any(tails[i] <= expected[i] and expected[i] >= 0 for i in range(4))
    assert any_wait is False

    # Local expert transform
    begin, end = 8, 12
    topk = [3, 9, 11, 0]
    local = [v - begin if begin <= v < end else -1 for v in topk]
    assert local == [-1, 1, 3, -1], local

    print("send_head raw     ", raw)
    print("after notify patch", patched)
    print("combine head seq  ", updates)
    print("local topk        ", local)
    print("PASS")


if __name__ == "__main__":
    main()
```

### 运行与期望输出

```bash
python3 /tmp/deepep_combine_head_sim.py
```

（把上面脚本存成该路径，或任意路径。）

```text
send_head raw      [5, -1, -1, 6, -1, 8]
after notify patch [5, -7, -7, 6, -9, 8]
combine head seq   [6, 6, 6, 7, 8, 9]
local topk         [-1, 1, 3, -1]
PASS
```

中间量核对（token B = 下标 1）：

| 步骤 | 计算 |
|---|---|
| 后向扫到槽 6 | `last_head = 6` |
| 空洞编码 | `−6 − 1 = −7` |
| combine 解码 | `−(−7) − 1 = 6` |
| 与 A 读完后 head | 同为 6，队列不卡住 |

## 7. 常见错误与症状

| 错误理解 | 正确理解 | 症状 |
|---|---|---|
| combine 接收端也是「每对端 3 warp 拷贝」 | warp0 推 head，其余按 token 交错 reduce | 读错 `recv_warp_id - 1` 步长，漏 token / 重复 reduce |
| 第一次完整 combine，以后才 cached_notify_combine | combine **每次**都走 `cached_notify_combine` | 误以为首次可以不带 handle |
| handle 是框架自动缓存的 | 用户传入上次返回的 tuple | 传 `None` 又重算 layout；传错 handle 则人数/行号错乱 |
| `expected_head < 0` 时写入负 head | 写入的是解码后的非负 head | 自己实现时把负数写进队列 head，发送端空位计算炸掉 |
| rank-major 行号 = expert id | expert 在 `recv_topk_idx` | GEMM 吃错行，expert 输入对不上 |
| `tma_store_wait<0>` 可省略 | 异步 store 未完成就改 smem | 偶发写回数据损坏 |
| `__any_sync` 只看 lane0 | 任一相关 rank 未到齐都继续等 | 过早 reduce，缺副本 |

## 8. 速记卡

```text
Dispatch warp：thread_id/96 → 固定对端；组内 3 warp 轮流拷；bar.sync R,96 后 release tail
Combine 发送：send_rank_id = (channel + warp_id) % P；组内 stride 3 拷回家
Combine 接收：warp0 推 min head；warp1..23 交错 token；lane→rank 读 send_head
等待：__any_sync(tail<=expected && expected>=0)
空洞：notify 后向编码 −last−1；combine 更新 head = 解码 或 expected+1
TMA：wait<0> 清空在飞；wait<stages-1> 流水
Cache：handle is not None → cached_notify_dispatch；combine 总是 cached_notify_combine
Expert：recv_topk_idx 本地 id，不是行号
```

## 9. 自测题与答案

1. **combine 奇数 block、`recv_warp_id = 4`，本 channel token 范围 `[100, 130)`。它处理哪些 token？**
   答：`token_idx = 100 + (4−1) = 103`，步长 `24−1 = 23` → 103, 126（下一个 149 越界）。

2. **为何 `cached_notify_combine` 要从后往前扫？**
   答：向前看时「下一个真实槽位」还不知道；从后往前扫时 `last_head` 已是后方真实槽，空洞才能编成有意义的 `−last−1`，前进 combine 时 head 不会回退。

3. **`tma_store_wait<7>` 和 `wait<0>` 在 combine 里各用在哪？**
   答：token 开始前 `wait<0>` 保证上轮 smem 可复用；hidden 流水里 `wait<7>`（8 stage）只保证最老 stage 完成，允许最多 7 个 group 在飞。

4. **用户第二次 `dispatch(handle=None)` 会怎样？**
   答：即使 topk 碰巧相同，也会再跑完整 `notify_dispatch`（重算 prefix、CPU 同步计数），不会走 `cached_notify_dispatch`。

5. **某行 `recv_topk_idx = [−1, 2, −1, −1]`，本卡 4 个本地 expert。这一行给谁算？**
   答：本地 expert 2；其它 topk 槽不是本卡的。行在 `recv_x` 里的位置只说明来自哪个源 rank 段，不说明 expert。

## 10. 学习进度

- [x] DeepEP V1 normal：layout → rank dispatch → local expert metadata → handle → combine
- [x] 区分 rank-major 通信布局、expert-major 计算布局与 gate 加权
- [x] DeepEP V1 low-latency：定容 dispatch、packed expert 输入、handle 回程元数据、weighted combine 与 hook
- [x] 机内 `get_dispatch_layout` kernel
- [x] 机内 `barrier_block`
- [x] 机内 `notify_dispatch`
- [x] 机内 `dispatch` kernel：环形队列、head/tail、`send_head`、TMA、named barrier
- [x] 机内 `combine` / warp 分工 / handle·cached notify / `__any_sync`·`tma_store_wait` / `send_head` 空洞编码 / rank-major 与 `recv_topk_idx`

### 下一知识点

internode（跨节点）路径：RDMA 段与 NVLink 段如何接力转发，`internode::cached_notify` 与机内版本的差异；或回到 elastic V2 的 GEMM 友好布局。
