# DeepEP V1：Low-latency `dispatch` 精读问答（条带 / staging / finish / count）

> 承接 [12-deepep-v1-low-latency-dispatch-kernel.md](./12-deepep-v1-low-latency-dispatch-kernel.md)。12 给出 SEND/RECV 主线；本文把后续逐行追问收成一张「易混点对照表」：条带与 warp 角色、`responsible_expert_idx` 双重语义、staging vs slot、`finish==2×TAG`、谁发 count、与 normal channel 的类比。
>
> 代码：`/Users/saboxu/Downloads/communication/DeepEP/csrc/kernels/legacy/internode_ll.cu`（`__global__ dispatch` L128–L462）；`TAG` 在 `compiled.cuh`。

```text
本次讲解位置
章节：communication / DeepEP V1
小节：LL dispatch 精读中的易混概念
知识点：grid-stride 条带；非末 warp put / 末 warp 统计；finish 三笔账；count 每 expert 一次门铃；staging≠finish 数组
上次：LL dispatch kernel 主流程（12）
下次：low-latency combine
PTX / 原语：atomicAdd、atomic_add_release_global、__shfl_sync、ld.acquire、st.release.sys
```

## 为什么现在讲这个

主流程读完后，最容易在四个地方翻车：

1. 把「一个 block / 一个 warp」误读成「只负责一个 token / 一个固定 expert」。
2. 把 `atomic_finish_counter` 当成数据目的地，或当成接收方也在写的跨卡变量。
3. 以为「发 count」跟每次 put 绑在一起、或以为只有 SM0 在发 count。
4. 分不清 staging（`rdma_recv_x`）和最终 `packed_recv_x`，以及它和 normal channel 环的异同。

本文用问答把这些钉死，并给出校正后的四步 finish 流程。

## 1. 总览：SEND 侧四类动作（校正版）

```text
1. SM0 末 warp：全体 expert  finish += TAG                         // L294–L295
2. 每个 SM 的末 warp（含 SM0）：本 SM 负责的 expert  finish += (TAG - sum)  // L317
3. 非末 warp：put 到对端 staging 后  finish[expert] += 1           // L277
4. 该 expert 所属 warp group 的 sub_warp0 + lane0（任意 SM）：
      等 finish == 2×TAG → 写对端 rdma_recv_count = -n-1           // L324–L338
```

常见误读：「只有非 SM0 末 warp 做 `TAG-sum`」「只有 SM0 非末 warp 发 count」——都不对。

## 2. 条带：`token_idx = sm_id; += num_sms`

```cpp
// internode_ll.cu L202–L210
for (int token_idx = sm_id; token_idx < num_tokens; token_idx += num_sms) {
    const auto x_int4 = static_cast<const int4*>(x) + token_idx * hidden_bf16_int4;
    const auto rdma_x_src_idx = reinterpret_cast<int*>(
        static_cast<uint8_t*>(rdma_x) + token_idx * num_bytes_per_msg);
    auto dst_expert_idx = warp_id < num_topk
        ? static_cast<int>(__ldg(topk_idx + token_idx * num_topk + warp_id))
        : -1;
    thread_id == 0 ? (*rdma_x_src_idx = token_idx) : 0;
```

| 说法 | 对错 |
|---|---|
| 一个 block 只负责一个 token | ❌ 条带上多个 token |
| 循环内当前处理一个 token | ✅ |
| `x_int4` / `rdma_x_src_idx` | 输入行 / 本卡发送消息头 |
| `dst_expert_idx = topk[token][warp_id]`（`warp_id<K`） | 本 warp 负责该 token 的第 `warp_id` 个 topk 槽 |
| `*rdma_x_src_idx = token_idx`（thread0） | 消息头写入源 token 下标 |

`num_sms=4`、10 个 token 时：block1 → token 1,5,9。

## 3. Warp / block 角色（SEND 数据段）

```cpp
// L195 vs L280
if (warp_id < num_warps - 1) { /* 打包 + put */ }
else if (warp_id == num_warps - 1) { /* 统计 / +TAG */ }
```

| 角色 | 条件 | 工作 |
|---|---|---|
| 数据 warp | `warp_id < num_warps-1` | 协作打包；`warp_id<K` 时 put |
| 末 warp | `warp_id == num_warps-1` | 统计 sum；SM0 清 next + 全体 `+TAG` |
| 发 count | `sub_warp_id==0 && lane_id==0`（syncthreads 后） | 等 finish，写对端 count——**不必是末 warp，也不必是 SM0** |

对**当前** token：多 warp 一起打一份 `rdma_x`；`warp_i`（`i<K`）额外负责 put 到 `topk[token][i]`。
不是「一个 warp 固定承包某个全局 expert 的所有 token」。

`responsible_expert_idx = sm_id * num_warp_groups + warp_group_id` 挂在 **warp group** 上（L164），不是单 warp 一 expert。

## 4. `responsible_expert_idx` 双重语义

同一公式，SEND 发 count vs RECV 变量名不同：

**SEND（全局 expert → 目的 rank/local）** — L324–L326：

```cpp
dst_rank = responsible_expert_idx / num_local_experts;
dst_expert_local_idx = responsible_expert_idx % num_local_experts;
```

**RECV（同一下标 → 来源 rank / 本卡 local）** — L363–L364：

```cpp
src_rank = responsible_expert_idx / num_local_experts;
local_expert_idx = responsible_expert_idx % num_local_experts;
```

算术相同，角色相反（发往谁 vs 从谁收）。下标空间都是 `E = P·L`。

## 5. slot 与 staging

### slot

```cpp
// L254–L262：抢 slot，dst = rdma_recv_x[local][my_rank][slot]
int slot_idx = lane_id == 0 ? atomicAdd(atomic_counter_per_expert + dst_expert_idx, 1) : 0;
slot_idx = __shfl_sync(0xffffffff, slot_idx, 0);
```

- slot = staging 第三维下标 `0 .. C-1`，`C = num_max_dispatch_tokens_per_rank`
- **不是** 源 `token_idx`；由 `atomicAdd` 抢号，同 expert 内顺序不确定
- 源 token 下标在消息头里（L210）

### staging

口头「staging」= 参数 **`rdma_recv_x`**（通信中转），不是最终 `packed_recv_x`。

```text
源 put → rdma_recv_x[local][src][slot]  （staging）
RECV  → packed_recv_x[local][begin+)   （给 GEMM）
```

RECV 定位段首（尚未加 slot）：

```cpp
// L365–L367：staging 段首 [local][src][slot=0]
const auto rdma_recv_x_uint8 = static_cast<uint8_t*>(rdma_recv_x) +
    local_expert_idx * num_ranks * num_max_dispatch_tokens_per_rank * num_bytes_per_msg +
    src_rank * num_max_dispatch_tokens_per_rank * num_bytes_per_msg;
```

与 normal **channel 环形队列**角色类似（都是中间缓存），差别：

| | Normal channel | LL staging |
|---|---|---|
| 流控 | head/tail，可读后回收 | 定容 `C`，一步内不回收 |
| 索引 | channel × 对端 | local_expert × src × slot |

## 6. 数据目的地 ≠ `atomic_finish_counter`

| 名字 | 是否数据目的地 | 谁读写 |
|---|---|---|
| `dst_ptr` → `rdma_recv_x + …` | ✅ token 消息 | 发送方写对端 staging |
| `atomic_counter_per_expert` | ❌ 抢 slot | **仅发送方本卡** |
| `atomic_finish_counter_per_expert` | ❌ 完成本地握手 | **仅发送方本卡** |
| `rdma_recv_count` | ❌ 人数门铃 | 发送方写对端；接收方读 |

put / P2P 走 L259–L271；`finish += 1` 在 L277，是记账不是地址。

接收方等的是 count，不是 finish：

```cpp
// L387：接收方等的是 count，不是 finish
while ((num_recv_tokens = ld_acquire_sys_global(
           rdma_recv_count + local_expert_idx * num_ranks + src_rank)) == 0) ...
```

## 7. `finish == 2×TAG` 与「发 count」

`TAG = 1024`（`LEGACY_FINISHED_SUM_TAG`，`compiled.cuh`）。

```text
finish = TAG + (TAG - sum) + sum×1 = 2×TAG
```

等齐再写门铃（L330–L338）：

```text
对端 rdma_recv_count[local][我的 rank] = -n - 1
  0 表示未到；解码 n = -v - 1
```

**发 count** = 通知对端「该方向数据发齐了，共 n 条」，不是再发 token。

次数：本卡对每个全局 expert **至多 1 次** count（很多 `n=0` 仍写 `-1`），不是每条 put 跟一次。收端按 `[L][P]` 每格各自等，所以要每格一个门铃。

SM0 末 warp 垫 TAG 的精确行：

```cpp
// L294–L295（需 L280 末 warp + L282 sm_id==0）
for (int i = lane_id; i < num_experts; i += 32)
    atomic_add_release_global(atomic_finish_counter_per_expert + i, LEGACY_FINISHED_SUM_TAG);
```

（外层条件：`warp_id == num_warps-1` 且 `sm_id == 0`，L280–L282。）

## 8. 完整可运行核对（finish 公式）

```python
#!/usr/bin/env python3
TAG = 1024

def finish_ok(sum_tokens: int, sends_done: int) -> bool:
    # SM0 +TAG; 末 warp +(TAG-sum); puts +1 each
    return TAG + (TAG - sum_tokens) + sends_done == 2 * TAG

assert finish_ok(3, 3) and finish_ok(0, 0)
assert not finish_ok(3, 2)
assert (-3 - 1) == -4 and (-(-4) - 1) == 3  # encode/decode count
print("PASS")
```

期望：`PASS`。

## 9. 常见错误与症状

| 误读 | 正确 | 症状 |
|---|---|---|
| finish 数组是数据目的地 | 目的地是 `rdma_recv_x` | 读错同步对象 |
| 接收方也写 finish | 仅发送方 workspace | 跨卡找 finish 找不到 |
| 只有 SM0 发 count | 各 expert 所属组的 sub_warp0/lane0 | 漏响门铃 / 误解负载 |
| 每 put 发一次 count | 每 expert 方向一次 | 夸大元数据流量 |
| slot = token_idx | slot 是 atomic 抢的坑号 | combine 对不上源 token（应看消息头） |
| block 固定一个 token | grid-stride 多 token | 漏算并行度 |

## 10. 速记卡

```text
条带：token = sm_id + k*num_sms
非末 warp：打包 + put；末 warp：统计 / +TAG / +(TAG-sum)
finish 仅本卡；2×TAG 后发 count=-n-1 到对端
staging=rdma_recv_x[local][src][slot]；packed 是最终
SEND idx→dst_*；RECV idx→src_*（公式同，语义反）
```

## 11. 自测题与答案

1. **SM0 末 warp 的 `+TAG` 在哪一行？**
   答：L294–L295（且需 L280 末 warp + L282 `sm_id==0`）。

2. **发 count 的线程条件是什么？是否必须是 SM0？**
   答：`responsible_expert_idx < E && sub_warp_id==0 && lane_id==0`；任意拥有该 expert 下标的 SM，不必 SM0。

3. **为何 `sum=0` 仍能到达 `2×TAG`？**
   答：`TAG+(TAG-0)+0 = 2×TAG`，于是仍可发 `count=-1`，收端解码得 0，不必空等。

4. **staging 与 normal channel 最关键的一点相同、一点不同？**
   答：同：都是通信中间缓存。异：LL 定容不回收；normal 用 head/tail 环形复用。

5. **`dst_expert_idx = topk[token][warp_id]` 是否表示一个 warp 固定一个全局 expert？**
   答：否。只表示**当前 token** 的第 `warp_id` 个 topk 槽；下一轮 token 会换。

## 12. 学习进度

- [x] LL dispatch 主流程（12）
- [x] LL dispatch 精读问答：条带、staging/slot、finish/count、双重 idx 语义
- [ ] LL combine

### 下一知识点

low-latency `combine`：`layout_range` / `src_info` + `topk_weights` 加权收回。
