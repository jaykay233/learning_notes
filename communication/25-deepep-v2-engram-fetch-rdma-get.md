# DeepEP V2 Engram：按 index 远程 RDMA GET

> 承接 [24](./24-deepep-v2-hybrid-scaleout-scaleup-dataflow.md)。EP hybrid 主干读完后，ElasticBuffer 上还有一条实验旁路：**Engram**——把 KV-like storage 按全局 entry index 用 `gin.get` 拉到本卡。本文讲清 API、路由、`defer 首包 + Aggregate + flush_async`（不是合并数据）、doorbell 是谁敲给谁，以及「get 在飞」是什么意思。
>
> 源码：`deep_ep/include/deep_ep/impls/{engram_fetch,engram_fetch_wait}.cuh`，`csrc/kernels/elastic/engram.hpp`，`deep_ep/buffers/elastic.py`，`csrc/elastic/buffer.hpp`。

```text
本次讲解位置
章节：communication / DeepEP V2 Engram
小节：engram_write / engram_fetch / engram_fetch_wait
知识点：RDMA GET pull、per-peer defer+aggregate+flush、doorbell、issue/wait 拆核
上次：Hybrid Scaleout+Scaleup 完整数据流（24）
下次：pp_send / pp_recv 流水线环（若继续 Elastic 实验 API）
原语：gin.get、ncclGinOptFlagsAggregateRequests、flush_async、gin.wait
```

## 为什么现在讲这个

Engram 和 EP dispatch/combine **不是同一条数据面**：

- EP：按 topk **push** token 去 expert，再 push 结果回来。
- Engram：storage 已按 rank 切开写好；按 `indices` **pull** 条目到本卡。

读 `engram_fetch.cuh` 时最容易误判的是「defer 首包 + Aggregate」——会以为在合并多条 entry 的结果。其实合并的是 **对本机 NIC 的提交/门铃**，每条 get 仍写 `fetched` 的不同行。本文把这条旁路和提交技巧钉死。

---

## 1. 它在整图里的位置

```text
ElasticBuffer
  ├─ EP 直连 / hybrid     （dispatch ↔ combine）
  ├─ Engram（本篇）       （write storage → fetch by index）
  └─ PP send/recv         （流水线环，另一条实验 API）
```

```199:199:deep_ep/buffers/elastic.py
        - Engram (remote KV cache fetch, using RDMA)
```

---

## 2. API：write → fetch(hook) → wait

**Write**：本卡 shard 写入对称可见的 storage（含前后 barrier）。

```569:582:deep_ep/buffers/elastic.py
    def engram_write(self, storage: torch.Tensor,
                     sf: Optional[torch.Tensor] = None) -> None:
```

- `storage`: `[num_entries, hidden]`，BF16 或 FP8。
- `sf`: FP8 时全局复制的 scale 表；**RDMA 只搬 data，SF 本地 gather**（注释见 `elastic.py` L428–L429）。

**Fetch**：发起 get，立刻返回 **hook**；调用 hook 才 wait 并拿 tensor（便于和计算重叠，类似 LL `return_recv_hook`）。

```584:604:deep_ep/buffers/elastic.py
    def engram_fetch(self, indices: torch.Tensor, ...) -> Callable:
        # indices: [num_tokens, num_entries_per_token]
        # hook() -> (data, sf)
        # data: [num_tokens * num_entries_per_token, hidden]
```

```text
engram_write(storage[, sf])
        │
        ▼
engram_fetch(indices)     # launch engram_fetch_impl，返回 hook
        │  （中间可插别的 kernel）
        ▼
hook():
  engram_fetch_wait_impl  # gin.wait
  return (fetched, fetched_sf)
```

---

## 3. 一条 get 是什么：「发 get」

```46:50:deep_ep/include/deep_ep/impls/engram_fetch.cuh
        gin.get<...>(math::advance_ptr(storage, src_byte_offset),
                        math::advance_ptr(fetched, token_idx * kNumHiddenBytes),
                        kNumHiddenBytes, peer_idx, extra_options);
```

```text
请 peer 一侧：
  从 storage + offset 起拷 hidden 字节
  写到本卡 fetched[这一行]
```

| | GET（Engram） | PUT（EP 常见） |
|---|---|---|
| 谁主动 | 要数据的这边 | 有数据的这边 |
| 方向 | 远端 → 本卡 | 本卡 → 远端 |

**「发 get」= 调用 `gin.get` 把任务交给 Gin/NIC**，≠ 数据已到；安全读要等 `gin.wait`。

---

## 4. 路由：global entry → peer → 字节偏移

```58:68:deep_ep/include/deep_ep/impls/engram_fetch.cuh
            const auto global_idx = __ldg(indices + i);
            const auto owner_rank_idx = global_idx / kNumEntriesPerRank;
            const auto local_entry_idx = global_idx % kNumEntriesPerRank;
            const auto peer_idx = owner_rank_idx / kNumRanksPerRDMAPeer;
            const auto intra_peer_rank_idx = owner_rank_idx % kNumRanksPerRDMAPeer;
            const auto src_byte_offset = intra_peer_rank_idx * kIntraRankStorageStride
                                      + local_entry_idx * kNumHiddenBytes;
```

| `allow_hybrid_mode` | RDMA peer | team |
|---|---|---|
| true | `num_scaleout`（每 peer 含 `num_scaleup` 张卡的 storage） | `Rail` |
| false | 每个全局 rank 一个 peer | `World` |

（`engram.hpp` L44–L51。）

注意：这里的 `global_idx` 是 **Engram entry 空间**，不是 EP 的 `g = src_rank*MaxTok+t`。

---

## 5. 每 peer：defer 首包 + Aggregate + flush_async

### 5.1 代码在干什么

```71:106:deep_ep/include/deep_ep/impls/engram_fetch.cuh
            const auto request_idx = atomicAdd_block(num_requests_per_peer + peer_idx, 1);
            if (request_idx == 0) {
                deferred_token_idx[peer_idx] = i;          // 先记下，还不发
                deferred_src_byte_offset[peer_idx] = src_byte_offset;
            } else {
                issue_rdma_get(..., AggregateRequests 或 偶尔强制提交);
            }
        // 循环结束后：
                issue_rdma_get(deferred_...);
                gin.flush_async(..., request_ptr);   // → last_gin_requests
```

**不是合并结果。** 每条 get 仍写 `fetched` 不同行：

```text
get → fetched[row_a]
get → fetched[row_b]
...
各行独立
```

合并/延迟的是 **对本机 NIC 的提交方式**。

### 5.2 为什么要 defer 首包

循环扫 `indices` 时，**不知道「发往该 peer 的最后一条」是哪条**，没法在循环里对「最后一条」flush。

```text
循环中：第 1 次碰到该 peer → 只 defer
        第 2..N 次 → Aggregate 入队（少敲门铃）
循环后：补发 deferred + flush_async
        → 每 (qp, peer) 只收尾一次，句柄写入 last_gin_requests
```

| 发给该 peer 的条数 | 循环里 | 循环后 |
|---|---|---|
| 1 | 只 defer | 发这一条 + flush |
| N>1 | defer 1 + Aggregate 发 N−1 | 补发 1 + flush |

中途若 `request_idx % kGinQPFlushDepth == depth-1`（`kGinQPFlushDepth=768`），临时去掉 Aggregate，避免 QP 队列撑爆。

### 5.3 能否「全扫完再发」？

可以，那是另一写法（先记完描述符再批量 get）。当前选边扫边发，是为了：

- 少占临时存储（每 peer 只多存 1 条 deferred）
- 扫后面 index 时，前面的 get **可能已在飞**（见 §7）
- 循环内可按 depth 中途冲队列

---

## 6. Doorbell：谁敲给谁

**Doorbell 是本卡 GPU 线程敲给本机 NIC 的**，不是 rank A 给 rank B 敲铃。

```text
本卡 GPU
  写好 WQE（从 peer storage 哪段 → 本卡 fetched 哪段）
  → 写 NIC doorbell（UAR）
  → 本机网卡去取 WQE、发 RDMA、DMA 进 GPU 内存
```

| 角色 | 是谁 |
|---|---|
| 敲铃的 | 跑 `gin.get` / `flush_async` 的 GPU 线程 |
| 被敲的 | **本机** RDMA 网卡 |
| 不是 | 对端 GPU / 对端 kernel |

```text
AggregateRequests ≈ WQE 先入队，暂不（或少）敲门铃
flush_async / 去掉 Aggregate ≈ 真正敲本机 NIC，批量提交
```

---

## 7. 「前面的 get 已在飞」是什么意思

发 get 的循环和网卡搬数是两条线：

```text
GPU：扫 index#1 发 → 扫 #2 发 → 扫 #3 …
NIC：可能同时在搬 #1、#2 的数据进 fetched
```

「在飞」= 请求已交给网卡，DMA 可能正在进行或已完成，但 **还没 `gin.wait`**。
≠ 已经能安全读；安全读在 hook 的 wait 之后。

相对「全扫完再发」：边发可以把 **算路由** 和 **网络传输** 叠起来，减少网卡空闲。

---

## 8. Wait 内核

```19:27:deep_ep/include/deep_ep/impls/engram_fetch_wait.cuh
    for (int i = thread_idx; i < kNumRDMAPeers; i += kNumThreads) {
        // last_gin_requests[qp][peer] 非全 0 → gin.wait
```

```text
fetch：发出去 + 留下 last_gin_requests
wait：等这些句柄 → fetched 可安全读
```

某 peer 从未发过 get → 句柄被写成 0，跳过 wait。

---

## 9. 与 EP 对照

| | Engram | EP dispatch/combine |
|---|---|---|
| 原语 | 单向 `gin.get`（pull） | `gin.put` staging + 回程 reduce |
| 索引 | 全局 entry id | topk → expert/rank |
| 完成同步 | fetch / wait 拆核 | barrier / tail / epilogue |
| hybrid | 只复用 peer=scaleout 路由 | 完整两级数据面 |
| 元数据 | 无 linked list / `g` | `recv_src_metadata` 等 |

---

## 10. 可运行小验证：entry → owner / peer

文件：`communication/25_engram_route_sim.py`。

```python
#!/usr/bin/env python3
"""Route Engram global entry indices to RDMA peers (hybrid vs flat)."""

from __future__ import annotations


def route(global_idx: int, entries_per_rank: int, ranks_per_peer: int):
    owner = global_idx // entries_per_rank
    local = global_idx % entries_per_rank
    peer = owner // ranks_per_peer
    intra = owner % ranks_per_peer
    return owner, local, peer, intra


def main() -> None:
    entries_per_rank = 1000
    # Hybrid: 2 scaleout × 4 scaleup
    print("hybrid (peer=scaleout, ranks_per_peer=4):")
    for idx in (0, 999, 1000, 4500, 7999):
        print(f"  entry {idx:4d} -> owner/local/peer/intra={route(idx, entries_per_rank, 4)}")

    print("flat (peer=rank, ranks_per_peer=1):")
    for idx in (0, 1000, 4500):
        print(f"  entry {idx:4d} -> owner/local/peer/intra={route(idx, entries_per_rank, 1)}")


if __name__ == "__main__":
    main()
```

运行：

```bash
python3 communication/25_engram_route_sim.py
```

期望输出：

```text
hybrid (peer=scaleout, ranks_per_peer=4):
  entry    0 -> owner/local/peer/intra=(0, 0, 0, 0)
  entry  999 -> owner/local/peer/intra=(0, 999, 0, 0)
  entry 1000 -> owner/local/peer/intra=(1, 0, 0, 1)
  entry 4500 -> owner/local/peer/intra=(4, 500, 1, 0)
  entry 7999 -> owner/local/peer/intra=(7, 999, 1, 3)
flat (peer=rank, ranks_per_peer=1):
  entry    0 -> owner/local/peer/intra=(0, 0, 0, 0)
  entry 1000 -> owner/local/peer/intra=(1, 0, 1, 0)
  entry 4500 -> owner/local/peer/intra=(4, 500, 4, 0)
```

---

## 11. 常见误判

| 误判 | 实际 |
|---|---|
| defer/Aggregate 合并多条 entry | 只合并提交；每行 fetched 仍独立 |
| doorbell 是敲给对端 GPU | 敲给 **本机 NIC** |
| 发 get = 数据已到 | 只是下单；要 wait |
| Engram 的 global_idx 就是 EP 的 `g` | 否；一个是 entry 空间，一个是 `(src_rank,t)` |
| fetch 内核里已经 wait | wait 在 hook 的 `engram_fetch_wait` |

---

## 12. 自测题

1. **Engram 用 get 还是 put？和 EP 主流差在哪？**
   答：get（pull）；EP 主流是 put staging + 回程。

2. **为什么把某 peer 的第 0 条 get 挪到循环后发？**
   答：循环内不知最后一条；留一条与 `flush_async` 绑成每 peer 唯一收尾。

3. **Doorbell 谁给谁？**
   答：本卡 GPU → 本机 NIC。

4. **「get 在飞」能否直接读 fetched？**
   答：不能；要等 hook 里 `gin.wait`。

5. **FP8 的 scale 走 RDMA 吗？**
   答：不走；本地从 `sf_table` gather。

---

## 13. 学习进度

- [x] V2 直连 / hybrid EP 主干（20、24）
- [x] Engram fetch：GET + defer/aggregate/flush + doorbell
- [x] PP send/recv 流水线环（见 [26](./26-deepep-v2-pp-send-recv-ring.md)）

### 下一知识点

已写 [26](./26-deepep-v2-pp-send-recv-ring.md)；后续可接 AGRS session。
