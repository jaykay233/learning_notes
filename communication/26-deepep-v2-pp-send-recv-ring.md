# DeepEP V2 PP：环上相邻 send/recv 与 inflight 槽位

> 承接 [25](./25-deepep-v2-engram-fetch-rdma-get.md)。Engram 是按 index **pull**；ElasticBuffer 上还有一条实验旁路：**PP send/recv**——只允许环上 prev/next，本地 TMA 进 staging，再 `gin.put` 到对端 recv 槽，用 Gin signal 做「槽满 / 槽空」握手。本文钉死：四区 buffer、slot 取模、send 等释放 / recv 等到达、两条 signal 各管什么。
>
> 源码：`deep_ep/include/deep_ep/impls/pp_send_recv.cuh`，`deep_ep/include/deep_ep/common/layout.cuh`，`deep_ep/buffers/elastic.py`，`csrc/elastic/buffer.hpp`，`tests/elastic/test_pp.py`。

```text
本次讲解位置
章节：communication / DeepEP V2 PP
小节：pp_send / pp_recv / get_buffer_offset
知识点：环邻接、四区×inflight 槽、TMA staging、put+SignalInc、释放 signal
上次：Engram RDMA GET（25）
下次：AGRS session（若继续 Elastic 实验 API）
原语：gin.put、ncclGin_SignalInc、gin.signal、check_signal、grid.sync
```

## 为什么现在讲这个

读完 Engram 容易把「凡是 Elastic 实验 API」都想成 GET。PP 正好相反侧重点：

- **只连环上邻居**（`prev` / `next`），不是任意 peer 的 entry 空间。
- 数据面是 **put**（本卡 staging → 对端 recv 槽），不是 get。
- 正确性靠 **有界 inflight 槽 + 两条 signal**（到达 / 释放）；槽满发不出去、槽空收不到——和 EP MoE 的 layout/count 完全两套账。

搞混「send 等的 signal」和「recv 等的 signal」会把 timeout 原因看反。

---

## 1. 它在整图里的位置

```text
ElasticBuffer
  ├─ EP 直连 / hybrid
  ├─ Engram（25）：gin.get by index
  ├─ PP（本篇）：环邻接 put + signal 握手
  └─ AGRS session（下一条实验路径）
```

```200:200:deep_ep/buffers/elastic.py
        - pipeline-parallel send/recv (PP, using NVLink)
```

注释写 NVLink，但 kernel 走的是 `gin.put` / `gin.signal`（World team）；同机多卡时常落在 NVLink 路径上，语义仍是 Gin 提交。

---

## 2. API：config → send / recv（仅邻居）

```606:636:deep_ep/buffers/elastic.py
    def pp_set_config(self, num_max_tensor_bytes: int, num_max_inflight_tensors: int):
        ...
    def pp_send(self, t: torch.Tensor, dst_rank_idx: int, num_sms: int = 0) -> None:
        # dst 必须是 prev 或 next
    def pp_recv(self, t: torch.Tensor, src_rank_idx: int, num_sms: int = 0) -> None:
        # src 必须是 prev 或 next
```

Host 侧把邻居钉死，并检查 buffer 容量：

```327:337:csrc/elastic/buffer.hpp
        this->prev_rank_idx = (rank + num_ranks - 1) % num_ranks;
        this->next_rank_idx = (rank + 1) % num_ranks;
        // assert: tensor_bytes * inflight * 2 * 2 <= num_buffer_bytes
```

Hint 公式（四区 × inflight）：

```454:456:deep_ep/buffers/elastic.py
        # Each buffer (send and recv, * 2) contains prev and next rank (* 2)
        return align(num_max_tensor_bytes * num_max_inflight_tensors * 2 * 2, ...)
```

```text
pp_set_config(max_bytes, inflight)
        │
        ▼
pp_send(t, next|prev)   ←→   peer.pp_recv(out, me)
```

与 Engram 不同：**没有 hook**；每次 send/recv 在当前 stream 上同步等到本侧握手步骤完成（对端仍可能异步）。

---

## 3. 环方向 → 本地「左/右」下标

```11:16:deep_ep/include/deep_ep/impls/pp_send_recv.cuh
template <int kNumRanks>
__device__ __forceinline__ std::pair<int, int> get_buffer_offset(
    const int& src_rank_idx, const int& dst_rank_idx) {
    const auto next_rank_idx = (src_rank_idx + 1) % kNumRanks;
    return dst_rank_idx == next_rank_idx ? std::make_pair(0, 1) : std::make_pair(1, 0);
}
```

| 边 | 返回 `(在 dst 眼里的我, 在我眼里的 dst)` |
|---|---|
| `src → next` | `(0, 1)` |
| `src → prev` | `(1, 0)` |

约定：**下标 0 =「next 方向」一侧，下标 1 =「prev 方向」一侧**（对 send/recv 两侧对称使用）。

---

## 4. 四区 buffer × inflight 槽

```text
buffer 按「区」切开，每区有 inflight 个槽，每槽 max_tensor_bytes：

区 0：本卡 recv（来自「把我当 next」的邻居，即 prev→我）
区 1：本卡 recv（来自「把我当 prev」的邻居，即 next→我）
区 2：本卡 send staging（发往 prev）
区 3：本卡 send staging（发往 next）
```

Send 侧怎么取指针：

```133:139:deep_ep/include/deep_ep/impls/pp_send_recv.cuh
    const auto slot_idx = send_count % num_max_inflight_tensors;
    auto send_buffer_ptr = ... ((dst_idx_in_local + 2) * inflight + slot) * bytes;
    auto recv_buffer_ptr = ... ((local_idx_in_dst + 0) * inflight + slot) * bytes;
```

| 指针 | 含义 |
|---|---|
| `send_buffer_ptr` | **本卡** staging（区 2 或 3） |
| `recv_buffer_ptr` | **对端** recv 区（区 0 或 1）；put 的目的地 |

例：rank 2 → next(3)，`get_buffer_offset → (0,1)`：

```text
本卡 staging = 区 (1+2)=3，slot
put 到 rank3 的区 0，同一 slot
```

Recv 侧：

```186:190:deep_ep/include/deep_ep/impls/pp_send_recv.cuh
    const auto slot_idx = recv_count % num_max_inflight_tensors;
    const auto recv_buffer_ptr = ... ((src_idx_in_local + 0) * inflight + slot) * bytes;
```

rank 3 从 prev(2) 收：`get_buffer_offset(2,3)=(0,1)` → 读本卡区 0，与 put 对齐。

计数存在 workspace：

```153:164:deep_ep/include/deep_ep/common/layout.cuh
    get_pp_send_count_ptr(offset)  // offset ∈ {0,1}，两个方向各一个 int64
    get_pp_recv_count_ptr(offset)
```

---

## 5. Send 流水（L141–L164）

```141:164:deep_ep/include/deep_ep/impls/pp_send_recv.cuh
    // 1) elect_one: 等释放 signal，再 TMA：x → send_buffer
    check_signal(..., kNumRanks + dst_idx_in_local + 2,
                 send_count - num_max_inflight_tensors + 1, ...);
    tma_copy(..., x, send_buffer_ptr, ...);
    grid.sync();
    // 2) SM0 elect_one: put staging → 对端 recv + SignalInc(到达)
    gin.put(..., recv_buffer_ptr, send_buffer_ptr, ..., dst_rank_idx,
            ncclGin_SignalInc(local_idx_in_dst + kNumRanks));
    *send_count_ptr += 1;
```

### 5.1 等槽释放（inflight 信用）

```text
target = send_count - inflight + 1

含义：第 send_count 次发送复用 slot = send_count % inflight，
      必须等「至少再释放到这个序号」——对端 recv 完成后会 SignalInc 释放。
```

| `inflight` | `send_count` | 等待 `signal >=` |
|---|---|---|
| 4 | 0 | −3 → 实际 0 已满足（初始可发） |
| 4 | 4 | 1（第 0 槽必须已被 recv 释放过） |
| 4 | 5 | 2 |

超时文案：`recv buffer is full`——对端还没腾出槽。

### 5.2 TMA staging

`tma_copy`（同文件 L44–L112）：smem 双缓冲，`tma_load_1d` → `tma_store_1d`，多 SM 按块切分，最后 `tma_store_wait`。目的：用户 tensor `x` → 本卡对称窗口里的 send 区，再交给 Gin。

### 5.3 put + 到达 signal

```text
put:  本卡 send 区 → 对端 recv 区（同 slot）
附带 SignalInc( local_idx_in_dst + kNumRanks )
本卡 send_count++
```

---

## 6. Recv 流水（L192–L211）

```192:211:deep_ep/include/deep_ep/impls/pp_send_recv.cuh
    // 1) 等到达 signal >= recv_count + 1，再 TMA：recv_buffer → x
    check_signal(..., src_idx_in_local + kNumRanks, recv_count + 1, ...);
    tma_copy(..., recv_buffer_ptr, x, ...);
    grid.sync();
    // 2) 通知对端：槽可复用
    gin.signal(src_rank_idx, SignalInc(kNumRanks + local_idx_in_src + 2));
    *recv_count_ptr += 1;
```

| 步骤 | 等什么 | 做什么 |
|---|---|---|
| 前半 | `signal[src方向+N] >= recv_count+1` | 数据已 put 进槽 → 拷到用户 `x` |
| 后半 | — | `signal` 对端 `N+方向+2`，释放发送信用 |

超时文案：`recv buffer is empty`——对端还没 put 完。

---

## 7. 两条 signal 对照（勿对调）

设 `N = kNumRanks`，方向下标 `d ∈ {0,1}`：

| Signal 下标 | 谁加 | 谁等 | 语义 |
|---|---|---|---|
| `N + d` | send 的 `put` 附带 Inc | recv 的 `check_signal` | **数据到达** |
| `N + d + 2` | recv 的 `gin.signal` Inc | send 的 `check_signal` | **槽释放 / 信用** |

```text
Send: 等释放(N+d+2) → TMA 进 staging → put+Inc到达(N+d')
Recv: 等到达(N+d)   → TMA 出到 x   → signal Inc释放(N+d'+2)
```

`check_signal`（L18–L36）读 Gin `signals_table`，`ld_acquire_sys`，带 GPU timeout。

---

## 8. 与 Engram / EP 对照

| | PP | Engram | EP hybrid |
|---|---|---|---|
| 拓扑 | 环 prev/next | 任意 entry owner | scaleout×scaleup |
| 原语 | put + signal | get | put + layout/count |
| 缓冲 | 四区×inflight 槽 | storage + fetched | RDMA/NVL buffer |
| 完成模型 | 同步 kernel 内握手 | fetch/wait 拆核 | notify/epilogue |

---

## 9. 可运行小验证：方向、区号、信用阈值

文件：`communication/26_pp_ring_slot_sim.py`。

```python
#!/usr/bin/env python3
"""Simulate PP ring buffer regions and send credit targets."""

from __future__ import annotations


def buffer_offset(src: int, dst: int, num_ranks: int) -> tuple[int, int]:
    nxt = (src + 1) % num_ranks
    return (0, 1) if dst == nxt else (1, 0)


def regions(src: int, dst: int, num_ranks: int, inflight: int, send_count: int):
    local_in_dst, dst_in_local = buffer_offset(src, dst, num_ranks)
    slot = send_count % inflight
    send_region = dst_in_local + 2
    recv_region_on_dst = local_in_dst + 0
    credit_target = send_count - inflight + 1
    arrive_sig = local_in_dst + num_ranks
    release_sig = num_ranks + dst_in_local + 2
    return {
        "pair": buffer_offset(src, dst, num_ranks),
        "slot": slot,
        "send_region": send_region,
        "dst_recv_region": recv_region_on_dst,
        "credit_target": credit_target,
        "arrive_signal": arrive_sig,
        "release_wait_signal": release_sig,
    }


def main() -> None:
    n, inflight = 4, 4
    print("rank2 -> next(3):")
    print(" ", regions(2, 3, n, inflight, send_count=0))
    print(" rank2 -> next, 5th send (count=4):")
    print(" ", regions(2, 3, n, inflight, send_count=4))
    print("rank2 -> prev(1):")
    print(" ", regions(2, 1, n, inflight, send_count=0))


if __name__ == "__main__":
    main()
```

运行：

```bash
python3 communication/26_pp_ring_slot_sim.py
```

期望：

```text
rank2 -> next(3):
  {'pair': (0, 1), 'slot': 0, 'send_region': 3, 'dst_recv_region': 0,
   'credit_target': -3, 'arrive_signal': 4, 'release_wait_signal': 7}
 rank2 -> next, 5th send (count=4):
  {'pair': (0, 1), 'slot': 0, 'send_region': 3, 'dst_recv_region': 0,
   'credit_target': 1, 'arrive_signal': 4, 'release_wait_signal': 7}
rank2 -> prev(1):
  {'pair': (1, 0), 'slot': 0, 'send_region': 2, 'dst_recv_region': 1,
   'credit_target': -3, 'arrive_signal': 5, 'release_wait_signal': 6}
```

（`N=4` 时到达信号为 4/5，释放等待为 6/7。）

真实多卡：`tests/elastic/test_pp.py`（stress + kineto）。

---

## 10. 常见误判

| 误判 | 实际 |
|---|---|
| 可发任意 rank | Host 断言只能 prev/next |
| PP 和 Engram 一样是 get | PP 是 put + signal |
| send 超时 = 本卡 staging 满 | 文案是对端 **recv 槽未释放** |
| recv 超时 = 本卡拷贝慢 | 对端 **还没 put / 到达 signal 不够** |
| `send_count` 与 `recv_count` 共用 | 各方向各一对，workspace 里 2+2 个 int64 |

---

## 11. 自测题

1. **rank 0 发给 next，staging 在几区？对端 recv 几区？**
   答：本卡区 3；对端区 0。

2. **`inflight=4`，第 5 次 send（count=4）要等释放 signal ≥？**
   答：`4-4+1=1`。

3. **到达 signal 与释放 signal 谁发谁等？**
   答：到达=send put Inc，recv 等；释放=recv signal Inc，send 等。

4. **为什么 send 里有 `grid.sync`？**
   答：多 SM 一起 TMA 写完 staging 后，才允许 SM0 发起 put。

5. **PP 有没有 Engram 那种 hook？**
   答：没有；send/recv 各自在 kernel 内完成握手步骤。

---

## 12. 学习进度

- [x] Engram fetch（25）
- [x] PP 环 send/recv + inflight 槽
- [ ] AGRS session（Elastic 下一条实验 API）

### 下一知识点

`create_agrs_session` / all-gather reduce-scatter：session 内多 tensor 共享 buffer 槽与 signal。
