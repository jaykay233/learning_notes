# DeepEP V1：Normal 跨机深挖问答（layout / meta / Sender / sync / combine / LL combine）

> 承接 [16](./16-deepep-v1-internode-dispatch-combine.md)。16 按 WarpRole 走完跨机 dispatch/combine；本文把后续追问收成一张对照：跨机是否还有 `get_dispatch_layout`、18 个 meta 是什么、`kRDMASender` 装箱逻辑、`__shfl_sync` vs `__syncwarp`、combine 两级 reduce、LL combine 对照，以及「远端 tail 不用多 warp 同步、head 取 min」。
>
> 代码：`csrc/kernels/legacy/layout.cu`、`internode.cu`、`internode_ll.cu`（行号以 DeepEP 仓库当前文件为准）。

```text
本次讲解位置
章节：communication / DeepEP V1
小节：normal internode 精读追问
知识点：跨机 layout+notify；meta=-v-1 前缀；Sender 扫全量+窗口；shfl 广播值 / syncwarp 对齐时序；combine 两级 reduce；LL staging+flag；tail 只读 / head 取 min
上次：跨机 dispatch/combine WarpRole 走读（16）
下次：SourceMeta 位图细节，或 V2 elastic
PTX / 原语：__shfl_sync、__syncwarp、barrier.sync、ld.acquire.sys、st.release、AMO、TMA
```

## 为什么现在讲这个

16 读完角色表之后，最容易在六处翻车：

1. 以为跨机不再调用 `get_dispatch_layout`，或以为 notify 只做机内交换。
2. 把 Sender 发的 18 个 meta 当成「token 内容」或「总数」，分不清前缀起点/终点。
3. 分不清「7 个 Sender 扫全量计数」和「按 token `%7` 认领装箱」。
4. 把 `__syncwarp` 当成「让寄存器值一致」——其实一致靠相同输入 + 冗余计算，或靠 `__shfl_sync`。
5. combine 和 LL combine 混用：一个是双环两级 reduce，一个是 staging + `flag=1`。
6. 以为多个 consumer warp 也要同步远端 tail——其实只读；要共识的是 head。

---

## 1. 跨机还有 `get_dispatch_layout` 吗？

**有。** 跨机 normal 仍是三步：`get_dispatch_layout` → `notify_dispatch` → `dispatch`。有 handle 则走 cached，跳过前两步。

```text
Python                          C++                              kernel
get_dispatch_layout()  ──►  layout::get_dispatch_layout     ──► layout.cu
dispatch(handle=None)  ──►  internode::notify_dispatch      ──► internode.cu ~L98
                       ──►  internode::dispatch             ──► internode.cu ~L452
dispatch(handle=h)     ──►  internode::cached_notify + dispatch
```

### 1.1 同一个 layout kernel，多一个输出（layout.cu L55–L120）

跨机时 `num_tokens_per_rdma_rank != nullptr`，额外统计每个 token 发往哪些**节点**（同一节点多卡只计一次）：

```cpp
// layout.cu L82–L100（示意）
is_in_rank[rank_idx]++;
is_in_rdma_rank[rank_idx / 8]++;
// ...
num_tokens_per_rdma_rank_per_thread[...] += (is_in_rdma_rank[j] > 0);
```

| 输出 | 机内 | 跨机 |
|---|---|---|
| `num_tokens_per_rank[P]` | ✅ | ✅ |
| `num_tokens_per_rdma_rank[R]` | ❌（nullptr） | ✅ |
| `num_tokens_per_expert[E]` | ✅ | ✅ |
| `is_token_in_rank[N,P]` | ✅ | ✅ |

### 1.2 跨机 `notify_dispatch`：计数也走 RDMA→NVL（internode.cu L98–L347）

| 阶段 | 谁 | 做什么 |
|---|---|---|
| SM0 | 交换计数 | RDMA put 打包（每目标节点 8 卡 + experts + 节点总数）→ 汇总 `recv_rdma_rank_prefix_sum` → NVL 再分发 → 汇总 `recv_gbl_rank_prefix_sum`；顺便清 RDMA/NVL 区 |
| SM 1..R | 算前缀矩阵 | 用本地 `is_token_in_rank` 算 `rdma_channel_prefix_matrix[R][C]`、`gbl_channel_prefix_matrix[P][C]`，再做 channel 方向前缀和 |

这两张矩阵就是 Sender 发 18 个 meta 的来源；`recv_gbl_rank_prefix_sum` 决定 `recv_x` 的 rank-major 段起点。

### 1.3 和机内对照

| | 机内 | 跨机 |
|---|---|---|
| layout | 同 kernel | 同 kernel + `num_tokens_per_rdma_rank` |
| notify 交换 | 两次 `barrier_block` + IPC | RDMA put + NVSHMEM sync + NVL + `barrier_block` |
| channel 前缀 | `channel_prefix_matrix[P][C]` | `rdma_*[R][C]` + `gbl_*[P][C]` |
| cached | `cached_notify_dispatch` | `cached_notify`（只清 buffer，前缀从 handle 取） |

---

## 2. meta 是啥？数量信息吗？

**是数量信息，而且是前缀和形式。** 不含 token 数据。每一对「来源卡 → 目标节点」、每个 channel，各发 18 个 int，编码成 `-v-1`（与清零后的 0 区分）。

### 2.1 18 个 int（Sender L596–L611）

| lane | 内容 | 谁用 |
|---|---|---|
| 0–7 | `gbl_channel_prefix_matrix[(dst节点,卡t)][c-1]` = 本 channel **起点** | Forwarder 转给 NVLReceiver |
| 8–15 | 同上 `[c]` = 本 channel **终点** | 同上 → `end-start` = 本 channel 发给该卡多少 |
| 16 | `rdma_channel_prefix_matrix[dst][c-1]` | Forwarder：节点级累计起点 |
| 17 | `rdma_channel_prefix_matrix[dst][c]` | Forwarder：`17-16` = 本 channel RDMA 环会来多少 token |

例：发往 (节点1, 卡5) 各 channel 数量 `[4,6,3]` → 前缀 `[4,10,13]`；channel 2 的 meta：lane5=`10`，lane13=`13` → 本 channel 3 个 token。节点级去重后可能更少（一个 token 去节点 1 多张卡，RDMA 只传一份）。

### 2.2 接收方怎么用

**Forwarder（L857–L878）**：lane j = 来源节点 j；取属于本 `dst_nvl_rank` 的 start/end 转发到 NVL；用 lane16/17 得 `num_tokens_to_recv_from_rdma`（环上没有结束标记，靠这个数收完）。

**NVLReceiver（L1068–L1102）**：

```text
写入位置 = recv_gbl_rank_prefix_sum[来源rank-1]  +  start_offset（meta）
```

即 rank-major 段内再按 channel 偏移。

### 2.3 为何 notify 已经交换过数量还要 meta？

| | notify | dispatch meta |
|---|---|---|
| 粒度 | 每来源 rank **总数** | 每来源 × **每 channel** 起点/终点 |
| 时机 | dispatch 前（经 CPU 同步分配 buffer） | dispatch 开头，每个 channel 各发一次 |

按 channel 拆的前缀只在发送方本地算过，接收方不知道，所以还要再发一遍。

---

## 3. `kRDMASender` 逻辑梳理（L587–L757）

角色：奇数 SM 上 warp 0–6；**打包工**。真正的大数据 RDMA put + 推远端 tail 交给 warp 7 `kRDMASenderCoordinator`。

```text
奇数 SM（channel c）
  warp0..6  Sender   ──写──► send_buffer[dst][slot]
                 │ smem window/tail
                 ▼
  warp7     Coord    ──put+AMO──► 远端 recv + rdma_channel_tail
```

### 3.1 发 meta（L592–L624）

按 `dst_rdma_rank = warp_id, warp_id+7, ...` 分摊目标节点；本节点写 `recv_buffer`，跨节点 `put_nbi_warp` 18 个 int。

### 3.2 与 Coord 会合（L625 / L568）

`barrier.sync 0, 8*32`：7+1 个 warp。Coord 先清 smem 的 `lock/tail/window`，Sender 过 barrier 后才能用。

### 3.3 token 主循环要点

| 步骤 | 行号 | 做什么 |
|---|---|---|
| 全员扫 token | L633–L640 | 每个 warp 都扫本 channel 全部 token；lane j 读节点 j 的 8 bool（`uint64`），累加 `global_rdma_tail_idx`（槽序号） |
| 认领 | L643–L645 | `(token - start) % 7 == warp_id` 才装箱 |
| 等空位 | L647–L650 | `rdma_tail_idx - head >= N` 时读远端推回的 head |
| 记 combine 线索 | L667–L668 | `send_rdma_head[token][dst] = rdma_tail_idx`（或 -1） |
| 收集目标 | L675–L686 | shuffle 把「lane=节点」收成紧凑 `dst_send_buffers[]`；最多 `min(R,8)` 个节点 |
| 一读多写 | L688–L727 | hidden/scales/SourceMeta/topk 写到所有目标槽 |
| 滑动窗口 | L730–L756 | 乱序完成 → 32-bit window + lock → 只按序推进 smem `rdma_send_channel_tail` |

**为何 7 个 warp 都扫全量？** 槽序号必须连续且一致；各自从头数最简单，无需跨 warp 通信。

### 3.4 Coord 读 smem tail → 批量 put → AMO（L804–L844）

攒够 chunk 或凑齐剩余后 `put_nbi_warp`，再 `amo_nonfetch_add` 远端 `rdma_channel_tail`。本节点只 `memory_fence`。

---

## 4. `__shfl_sync` vs `__syncwarp`：值一致靠什么？

### 4.1 Forwarder 里为何要 shfl head/tail（L962–L963）

Forwarder 把 lane 当「长度为 R 的数组」：lane j 只存来源节点 j 的 head/tail。读完后只有 lane `src_rdma_rank` 手里是这条环的真值，必须广播：

```cpp
// internode.cu L962–L963
auto src_rdma_head = __shfl_sync(0xffffffff, cached_rdma_channel_head, src_rdma_rank);
auto src_rdma_tail = __shfl_sync(0xffffffff, cached_rdma_channel_tail, src_rdma_rank);
```

### 4.2 为何 `src_rdma_tail = i + 1` 不用再同步（L997–L998）？

广播之后，循环里 `i`、`num_tokens_sent`、`is_in_dst_nvl_rank` 在 32 个 lane 上输入相同 → **冗余计算**，各算一遍结果相同，无需 shfl。

**判断标准：** 值只在某一个 lane 手里 → `__shfl_sync`；本来全 warp 一样 → 不用。

### 4.3 `__syncwarp` 不是让值变一致

| 原语 | 解决什么 |
|---|---|
| `__shfl_sync` | **值**：把某 lane 寄存器发给别人 |
| 冗余计算 | **值**：每人各算一遍 |
| `__syncwarp` | **时序**：汇合 + 保证此前操作对 warp 内可见 |

Forwarder 循环里的 `__syncwarp`（L990/L994/L1002）配合 `elect_one_sync`：一个 lane 发 TMA，其他人要等 barrier 登记完、store 完成，避免抢同一块 `tma_buffer`。去掉 L963 的 shfl，加再多 syncwarp 也救不了——它不会改寄存器。

---

## 5. Normal internode combine 再串一遍

相对 dispatch：**奇数 SM = forwarder**（极性对调）。路径 = NVL → 机内 reduce → RDMA → 跨节点 reduce。

```text
x（专家输出，仍按 dispatch 布局）
  → 偶 SM NVLSender：按 RDMA 源拆子环写入 NVL
  → 奇 SM Forwarder：用 combined_nvl_head 对齐槽，combine_token（跨 8 卡求和）→ RDMA put + AMO tail
  → 偶 SM RDMAReceiver：用 combined_rdma_head 对齐槽，combine_token（跨节点求和）→ combined_x
  → 两侧 Coordinator：取 min head 释放（NVL head / RDMA head）
```

### 5.1 角色（L1748–L1784，host L2312–L2375）

| SM | warps | 角色 |
|---|---|---|
| 偶 | 0–7 | `kNVLSender` → 各管一个 `dst_nvl_rank` |
| 偶 | 8.. | `kRDMAReceiver` |
| 偶 | 末 | Coord 推 **RDMA head** |
| 奇 | 0–23（典型） | `kNVLAndRDMAForwarder`，按 `dst_rdma` 组成 large warp |
| 奇 | 末 | Coord 推 **NVL head** |

`kNumCombineForwarderWarps=24`，`kNumWarpsPerForwarder=24/R`。

### 5.2 NVLSender：按 RDMA 源拆环（L1789–L1927）

`nvl_channel_head/tail` 形状含 `kNumRDMARanks`：lane j 只管「来源节点 j」那段，避免死锁。任务区间来自 `gbl_channel_prefix_matrix`。TMA 写入 `x + SourceMeta + topk_weights`，再 `st_release` 推对应子环 tail。

### 5.3 Forwarder：第一次 reduce（L1960–L2149）

1. 等 RDMA 环空位（chunk）。
2. 读 `combined_nvl_head[token][src_nvl]`（dispatch 的 `send_nvl_head`）；`<0` = 该源无贡献。
3. 等该源 NVL tail 越过槽 → `combine_token` 跨 NVL 源求和，写入 RDMA send 槽。
4. large warp 末 sub_warp：put + AMO 推远端 RDMA tail。

### 5.4 RDMAReceiver：第二次 reduce（L2150–L2227）

处理**原序列** token；`combined_rdma_head[token][src_rdma]` 对齐各节点槽；`combine_token` 跨节点求和（可加 bias）写 `combined_x`。

### 5.5 和 LL combine 对照

| | Normal internode combine | LL combine |
|---|---|---|
| 队列 | NVL 环 + RDMA 环 | staging `[expert][token]` |
| 线索 | `combined_nvl_head` / `combined_rdma_head` | `layout_range` + `src_info` |
| Reduce | 两次（机内 8 卡 + 跨节点） | 一次 topk 加权 |
| 信号 | head/tail + AMO | `rdma_recv_flag = 1` |
| Forwarder | 有（奇 SM） | 无 |

---

## 6. Internode LL combine 长什么样？（internode_ll.cu L715–L1137）

**没有**双环中转。对称于 LL dispatch：

```text
SEND: x[local_expert][dst 段] --(layout_range, src_info)--> rdma_recv_x[global_expert][src_idx]
      整段发完写 flag=1
RECV: 等相关 expert 的 flag → grid.sync → 按 token×topk 加权 reduce → combined_x
```

| 阶段 | 要点 |
|---|---|
| 依赖 | dispatch 留下的 `layout_range=(num,offset)`、`src_info=原 token 下标`（dispatch L411/L429） |
| SEND 分工 | warp group 认领全局 expert；`sub_warp` 交错 token；P2P 直写或 IBGDA put（L835–L912） |
| 门铃 | 组内 `bar.sync` 后 `sub_warp_id==1` 写 `rdma_recv_flag[expert]=1`（L915–L928）——不是 count `-n-1` |
| RECV | 等 flag（可 timeout mask）→ `cg::this_grid().sync`（L976） |
| Reduce | 重组为 TMA load warp + decode warps；对 topk 做 `Σ w_i x_i`（L1033–L1135） |

---

## 7. 远端 tail 几个 warp 不用同步？只需同步共同的 head？

**消费侧：对。**

- 多个 consumer warp **各自读**同一条环的 tail（`ld_acquire`），互不修改，无需互相同步。
- 进度写在各自 smem（如 `rdma_receiver_rdma_head[warp][src]`）。
- Coordinator 取**未退休 warp 的 min**，再推远端/本端 **head** 释放槽。

例（combine RDMAReceiver + Coord，L2174–L2260）：

```cpp
// 各 warp 独立读 tail
cached_channel_tail_idx = ld_acquire_sys_global(rdma_channel_tail.buffer(lane_id));
// Coord 取 min 后 AMO 推 head
nvshmemi_ibgda_amo_nonfetch_add(rdma_channel_head.buffer(...), min_head - last_rdma_head, ...);
```

| 指针 | 谁写 | 多 warp |
|---|---|---|
| **tail** | 对端（或本端）生产者单调加 | 消费者只读，无需互相同步 |
| **head** | 本端消费者集体释放 | 必须所有仍在干活的 warp 都过了该槽 → **min** |

**生产侧补一句：** 推 remote tail 通常已收成「一个人发」——dispatch 是 SenderCoord；combine Forwarder 是 large warp 末 sub_warp（发前 `sync_large_warp` 只保证数据可见，不是多 warp 协商同一个 tail 值）。

---

## 8. 速记卡

```text
跨机三步：layout(+rdma计数) → notify(RDMA+NVL换计数+前缀矩阵) → dispatch
meta = 18 个 -v-1 前缀（每卡 start/end + 节点级 start/end），不是 token 内容
Sender：全员扫序号 + %7 认领装箱 + 窗口按序提交；Coord 批量 put+AMO
值一致：shfl 或冗余计算；syncwarp 只管时序
Combine：NVL 拆环 → 机内 reduce → RDMA → 跨节点 reduce；奇 SM=forwarder
LL combine：staging[expert][src_idx] + flag=1 + topk 加权；无 forwarder
Consumer：tail 只读不互相同步；head 取 min 再推
```

## 9. 自测题与答案

1. **跨机 layout 比机内多产出什么？**
   答：`num_tokens_per_rdma_rank[R]`（节点级去重计数）。

2. **meta 的 lane16/17 差值是什么？**
   答：本 channel 发往该目标节点、在 RDMA 环上的 token 数（节点级去重）。

3. **为何 Sender 每个 warp 都要扫全部 token？**
   答：得到连续且一致的 `global_rdma_tail_idx`，无需跨 warp 通信。

4. **Forwarder 里 `__syncwarp` 能否替代 `__shfl_sync` 广播 tail？**
   答：不能。syncwarp 不对齐寄存器值。

5. **combine 消费侧多个 warp 要同步远端 tail 吗？**
   答：不要；各自读即可。要跨 warp 共识的是 head（取 min）。

6. **LL combine 完成信号是 count 还是 flag？**
   答：`rdma_recv_flag[expert]=1`（每 expert 一段发完敲一次）。

## 10. 学习进度

- [x] Normal 跨机 WarpRole 走读（16）
- [x] 跨机 layout / notify / meta / Sender / sync / combine / LL combine 追问
- [ ] SourceMeta 位图与 `is_token_in_nvl_rank` 边界情况（可选）

### 下一知识点

`SourceMeta` 打包细节（`src_rdma_rank` + 8-bit NVL 位图）与 combine 侧如何消费；或转入 V2 elastic 路径。
