# DeepEP V2 直连：buffer 复用、去程/回程坐标，与通用 handle 数组结构

> 承接 [16](./16-deepep-v1-internode-dispatch-combine.md)、[14](./14-deepep-v1-low-latency-combine-and-pack.md)、[11](./11-deepep-v1-intranode-combine-and-warp-roles.md)。前面把 V1 normal / LL 的 dispatch、combine 各自读完了；本文把 V2 直连路径（`dispatch.cuh` → `dispatch_copy_epilogue.cuh` → `combine.cuh` → `combine_reduce_epilogue.cuh`）的坐标系钉死，再抽出一套跨 V1/V2 通用的「编号 / 布局 / 去程定位 / 回程信息 / 本地 expert 映射」数组模型。
>
> 代码：`deep_ep/include/deep_ep/impls/{dispatch,dispatch_copy_epilogue,combine,combine_utils,combine_reduce_epilogue}.cuh`、`deep_ep/include/deep_ep/common/layout.cuh`、`csrc/elastic/buffer.hpp`、`csrc/kernels/elastic/combine.hpp`；V1 对照：`csrc/kernels/legacy/{intranode,internode,internode_ll}.cu`、`csrc/legacy/{config,buffer}.hpp`、`deep_ep/buffers/{legacy,elastic}.py`。

```text
本次讲解位置
章节：communication / DeepEP V2
小节：直连 dispatch/combine 的 buffer 坐标与 handle
知识点：一块 buffer 两套视图；去程 [src][slot]；回程 [贡献者][原 token_idx]；rank-major vs expert-major 的通用数组
上次：elect_one_sync + shfl lane0 陷阱（19）
下次：hybrid_dispatch / hybrid_combine 的 scaleout 两级中转
原语：Gin put / get_sym_ptr、TMA bulk、atomicAdd 抢槽
```

## 为什么现在讲这个

读完 V2 的 `dispatch.cuh` 和 `combine.cuh` 后，最容易卡住的是三个问题：

1. dispatch 把 token 写到对端 `buffer[src][slot]`，combine 却写回源卡 `buffer[strip][token_idx]`——为什么回程不沿用 `[rank][slot]`？
2. `recv_x` 到底是 rank-major 还是 expert-major？底层通信布局和用户张量布局是不是一回事？
3. V1 normal、V1 LL、V2 各自的 handle 里有一堆 `send_head`、`recv_src_idx`、`src_info`、`recv_src_metadata`……它们本质上是不是同几类东西？

不弄清坐标系，读 combine 时会把「放在哪」（slot）和「是谁」（token_idx）混在一起，也看不出 V2 相对 V1 normal 省掉了什么（环形队列 + `send_head` 查表）。本文先把 V2 的坐标讲透，再给一张通用表，以后读任何 EP 通信库都可以按这张表对号入座。

---

## 1. 心智模型：先分清几个编号空间

一次 dispatch + combine 里，同一个 token 会在不同卡上以不同编号出现：

| 编号 | 定义 | 在哪有意义 | V2 代码 |
|---|---|---|---|
| `t` 源 token 号 | 源卡输入 `x` 的行号 | 源卡 | `token_idx` |
| `g` 源全局号 | `src_rank * MaxTok + t` | 全局唯一 | `get_src_token_global_idx_ptr()`（`dispatch.cuh` L331–L332） |
| `(src, slot)` 去程槽 | 对端 staging 里「源 src 发来的第 slot 个去重 token」 | 目的卡 | `stored_dst_slot_idx`（L336–L350） |
| `r` 接收行（rank-major） | 按源 rank 段顺序扫 staging 得到的行 | 目的卡 | epilogue 的 `i` |
| 展开行（expert-major） | 按本地 expert 分区（带对齐）的行 | 目的卡 | epilogue 的 `dst_tensor_idx` |
| `k` topk lane | `0..K-1` | 源卡、metadata | `master_src_topk_idx` |

核心原则：**每次通信的 buffer 坐标，都按接收方最方便读的方式定。** dispatch 的接收方是 expert 卡，要紧凑；combine 的接收方是源卡，要按自己的 `t` 求和。

---

## 2. 一块 buffer，两套视图

物理上只有一块对称窗口（workspace 后面那段）：

```cpp
// csrc/elastic/buffer.hpp L129
buffer = static_cast<uint8_t*>(workspace) + num_workspace_bytes;
```

dispatch、dispatch epilogue、combine 都传这个指针（`buffer.hpp` L994、L1127、L1293）；大小取两种布局的最大值：

```cpp
// csrc/elastic/buffer.hpp L684–L685
return math::align(std::max(num_dispatch_bytes, num_combine_bytes), symmetric::kNumAlignmentBytes);
```

两套视图都从 `buffer` 开头起算，各自切成「recv 区 + send 区」：

```text
dispatch 视图（dispatch.cuh L266–L267）
  [ recv：kNumRanks 条 × MaxTok 格 | send：1 条 × MaxTok 格 ]
    坐标 [源 rank][slot]               RDMA 暂存（NVLink 不用）
    每格：hidden(可 FP8) + SF + topk idx + topk weights + src_token_global_idx

combine 视图（combine.cuh L53–L58）
  [ recv：kNumTokensInLayout 条 × MaxTok 格 | send：kNumRanks 条 × MaxTok(×K) 格 ]
    坐标 [贡献者][原 token_idx]                  RDMA 暂存（NVLink 不用）
    每格：BF16 hidden + topk weights（with_metadata=false，无 SF）
```

为什么能复用：dispatch epilogue 先把 staging 拷到 `recv_x`；用户跑完 expert；combine 开头做一次全局 barrier（`combine.cuh` L76–L80）确认对端可写。直连模式下 `launch_combine` 返回的「待 reduce buffer」就是 `buffer` 本身（`combine.hpp` L181–L182）。

send 区只在 RDMA 时用：Gin `put` 的源要在对称窗口里，所以先 TMA 到本地 send 区再 `put`；NVLink 直接 `get_sym_ptr` + TMA 写对端（`buffer.hpp` L596–L597、L626–L627 里 NVLink 时 send 条数为 0）。

---

## 3. 去程：dispatch 写 `buffer[src][slot]`

### 3.1 一格是一整条 token 记录

`TokenLayout`（`layout.cuh` L179–L243）按 32B 对齐依次排：hidden → SF → topk idx → topk weights → `src_token_global_idx` →（hybrid）linked list idx。所以 `buffer[src][slot]` **不是一个整数**，`slot ≠ token_idx`；token 的身份写在格子里的 `src_token_global_idx` 字段。

### 3.2 slot 怎么来

```cpp
// dispatch.cuh L344–L349
if (ptx::deduplicate(stored_dst_rank_idx, lane_idx) and stored_dst_rank_idx >= 0)
    stored_dst_slot_idx = atomicAdd(workspace_layout.get_scaleup_atomic_sender_counter() + stored_dst_rank_idx, 1);
if (lane_idx < kNumTopk) {
    const auto value = stored_dst_slot_idx >= 0 ? rank_idx * kNumMaxTokensPerRank + stored_dst_slot_idx : -1;
    dst_buffer_slot_idx[token_idx * kNumTopk + lane_idx] = value;
}
```

- 计数器在**发送方**本地 workspace，每个目的 rank 一个。
- 同一 token 的多个 topk 落在同一 rank，只有 master lane（`deduplicate` 取最高 lane，`ptx.cuh` L412–L421）抢一个 slot。
- 结果记进源卡的 `dst_buffer_slot_idx[t][k]`，供 cached dispatch 复用（L338–L342），combine 不用它。

### 3.3 为什么去程不用 `t` 当列

接收方事先不知道会来哪些 token。若用 `t` 当列，条带稀疏（源发了 3、17、900，就得扫完 MaxTok 格判有效）。紧凑 slot 让接收方只需知道每条长度就能连续读 `[0, count)`——这正是 notify 阶段要交换计数、算 `psum_num_recv_tokens_per_scaleup_rank` 的原因（L232–L257）。

---

## 4. 从 rank-major staging 到 `recv_x`：`do_expand` 开关

底层 staging 永远按源 rank 分条。用户拿到的 `recv_x` 由 epilogue 决定：

```cpp
// dispatch_copy_epilogue.cuh L115–L121
if (not kDoExpand and ptx::elect_one_sync()) {
    dst_tensor_idx = i;                                   // rank-major：收包序
} else if (kDoExpand and kCachedMode and lane_idx < kNumTopk) {
    dst_tensor_idx = recv_src_metadata[i * kMetadataStride + 2 + lane_idx];
} else if (kDoExpand and not kCachedMode and dst_expert_idx >= 0) {
    dst_tensor_idx = atomicAdd(psum_num_recv_tokens_per_expert + dst_expert_idx, 1);   // expert-major
}
```

| 模式 | `recv_x` 布局 | 一行代表 |
|---|---|---|
| 非 expand | rank-major（按源 rank 段的收包序） | 一个去重后的 recv token；本地 expert 记在 `recv_topk_idx[r][k]`（L108–L109） |
| expand | local-expert-major，每个 expert 区间按 `kExpertAlignment` 对齐 | 一个 (token, 本地 expert) 副本；可选清零 padding（L231 起） |

expert 区间基址来自 notify 里的 exclusive 前缀和：`expert_count[i] = align(sum, alignment)`（`dispatch.cuh` L215），warp1 `do_psum(..., is_exclusive=1)`（L254–L256）。

### 4.1 epilogue 写下的回程信息

```cpp
// dispatch_copy_epilogue.cuh L194–L205
recv_src_metadata[i * kMetadataStride + 0] = *tma_buffer.get_src_token_global_idx_ptr();   // g
recv_src_metadata[i * kMetadataStride + 1] = current_rank_idx * kNumTopk + master_src_topk_idx;
...
recv_src_metadata[i * kMetadataStride + 2 + lane_idx] = dst_tensor_idx;   // expand：各 topk 副本的展开行
```

注意：**metadata 永远按 rank-major 的接收行 `i` 记**，即使 expand 时数据是 expert-major。combine 也按 `i` 扫，再通过 `[2+k]` 去 gather 展开行。

### 4.2 行序不确定，靠 `g` 排成确定序

slot 来自 atomic，展开行也来自 `atomicAdd(psum_num_recv_tokens_per_expert)`，所以接收行顺序每次可能不同。`EPHandle.deterministic_sort`（`elastic.py` L100–L192）用 `metadata[:, 0]`（源全局号 `g`）当排序键：

- 非 expand：整体按 `g` 置换 `recv_x / recv_sf / recv_topk_* / recv_src_metadata`（L141–L149）。
- expand：键为 `expert * 1e10 - 5e9 + g`（padding 只有 `expert * 1e10`），先按 expert、再按 `g` 排，padding 排在有效行之后；只置换展开张量，再把 `metadata[:, 2:]` 改成新位置（L156–L192）。

所以 `g` 同时有两个用途：回程时给出 `t = g % MaxTok`，以及给接收行排出一个确定的顺序。

---

## 5. 回程：combine 写源卡 `buffer[贡献者][原 token_idx]`

### 5.1 解 metadata

```cpp
// combine.cuh L88–L92
const int src_token_idx = __ldg(src_metadata + i * kMetadataStride) % kNumMaxTokensPerRank;
const int src_rank_topk_idx = __ldg(src_metadata + i * kMetadataStride + 1);
const int src_rank_idx = src_rank_topk_idx / kNumTopk;
const int src_topk_idx = src_rank_topk_idx % kNumTopk;
```

### 5.2 写入地址

```cpp
// combine.cuh L99–L105
auto token_buffer = recv_buffer.get_rank_buffer(kUseRankLayout ? rank_idx : src_topk_idx)
                               .get_token_buffer(src_token_idx);
token_buffer.set_base_ptr(gin.get_sym_ptr<team_t>(token_buffer.get_base_ptr(), src_rank_idx));
// RDMA 时改走 send_buffer[src_rank][src_token_idx]，循环末尾再 gin.put（L228–L236）
```

### 5.3 「贡献者」是什么

源卡 token `t` 的结果 = 若干份部分结果之和；每份的来源叫一个贡献者。它们都写到同一列 `t`，第一维用来区分：

```cpp
// combine_utils.cuh L8–L18
use_rank_layout = allow_multiple_reduction and (num_ranks <= num_topk);
num_tokens_in_layout = use_rank_layout ? num_ranks : num_topk;
```

| 模式 | 一个贡献者 | 第一维下标 |
|---|---|---|
| 允许多次 reduce，rank 数 ≤ topk | 一个 expert rank（本卡多 expert 已合并） | 该 rank 的 `rank_idx` |
| 其他非 expanded-send 情况 | 一个 expert rank | 该 rank 的 master topk lane |
| expanded send（expand 且禁止多次 reduce） | 一个 topk expert | topk 下标 `k`（L184） |

下标**不是 expert id**：第一维最多 `min(rank 数, topk)` 或 topk 条，放不下全局 expert id。

### 5.4 源卡怎么读

```cpp
// combine_reduce_epilogue.cuh L66–L69：用源卡自己的 topk_idx 推出贡献的 rank
stored_dst_expert_idx = static_cast<int>(combined_topk_idx[token_idx * kNumTopk + lane_idx]);
stored_dst_rank_idx = stored_dst_expert_idx >= 0 ?
    stored_dst_expert_idx / (kNumScaleoutRanks == 1 ? kNumExpertsPerRank : kNumExpertsPerScaleout) : -1;
// L93：下标 = rank 号（rank layout）或 lane 号
return kUseRankLayout ? ptx::exchange(stored_dst_rank_idx, idx) : idx;
// L103–L106：所有贡献都在同一列 token_idx
comm_buffer.get_rank_buffer(slot_idx).get_token_buffer(token_idx).get_base_ptr();
// L121–L122：写到输出同一行 token_idx
```

dispatch epilogue 记 master lane 用 `get_master_lane_idx`（取最高 lane），源卡 `deduplicate` 也保留最高 lane，两边规则一致，所以不用传额外映射。

### 5.5 为什么回程用原 `t` 而不是 `[rank][slot]`

源卡其实知道 slot（`dst_buffer_slot_idx`），`[rank][slot]` 也能做，但：

```text
[rank][slot]：token 5 在 rank2 拿 slot0、在 rank3 拿 slot7
  → 求和前要先查表，去第 0 列、第 7 列取
[贡献者][t]：都在第 5 列，直接读
```

| 好处 | 原因 |
|---|---|
| 写端不协调 | 地址由 metadata 直接算出，无 atomic |
| 不冲突 | 每个 (贡献者, t) 只有一个写者 |
| 读端不查表 | 源卡用自己的 `topk_idx` 推出要读哪些条 |
| 与输出对齐 | `combined_x` 按原 token 顺序，列号即输出行号 |
| 条数少 | 第一维 ≤ topk；`[rank][slot]` 必须开满 `kNumRanks` 条 |
| 容量刚好 | `t < num_tokens ≤ MaxTok` |

---

## 6. combine 三条路径与 expand 的关系

```cpp
// combine.cuh L125–L126
auto reduce_valid_mask = ptx::gather(stored_topk_slot_idx >= 0);
auto no_local_reduce = not kUseExpandedLayout or (kAllowMultipleReduction and __popc(reduce_valid_mask) == 1);
```

| 条件 | 路径 | 行号 |
|---|---|---|
| 非 expand，或 expand 且本卡只有 1 个副本 | Case A：TMA 拷 `x[i]` 或 `x[唯一展开行]` 回源 | L127–L143 |
| expand + 允许多次 reduce + 本卡多副本 | Case B：`combine_reduce` 先在本卡 BF16 求和 | L144–L176 |
| expand + 禁止多次 reduce（`kDoExpandedSend`，L42） | Case C：每个 topk 副本单独回传，求和全留给源卡 | L177–L212 |

`combine_reduce`（`combine_utils.cuh` L55–L169）：≤2 路且无 bias 走 `nv_bfloat162` 直加；否则 float2 累加再 cast。V2 kernel 内是**不带权求和**，权重随数据回传（L215–L226）供用户 / backward 使用。

---

## 7. V1 对照：复用都有，回程坐标不同

| 路径 | 内存复用 | 复用前怎么保证安全 | combine 回程坐标 |
|---|---|---|---|
| V1 机内 normal | `buffer_ptrs`（dispatch L256–L257 带 `rank_prefix_matrix` 偏移；combine L759–L776 从头起算） | `cached_notify_combine` barrier + 清零（L627–L641） | `[channel][rank][环形 slot]`，源卡查 `send_head[t][rank]`（L923–L944） |
| V1 跨机 normal | `rdma_buffer_ptr` + NVL 两块，dispatch（L526–L560）与 combine（L1796–L1805、L1931–L1949）同形状、方向对调 | `cached_notify`（L1316 起） | RDMA/NVL 环形 slot，靠 `send_rdma_head` / `send_nvl_head` |
| V1 LL | 同一半里 dispatch/combine 指针相同（`config.hpp` L171–L180），另有奇偶双缓冲，每次调用翻转并清下一半（`buffer.hpp` L1504–L1505） | 清下一半信号区 | `[全局 expert][原 token_idx]`（`internode_ll.cu` L844–L845） |
| V2 直连 | `buffer` 两套视图 | combine 开头全局 barrier | `[贡献者][原 token_idx]` |

V2 去掉了 V1 normal 的环形队列和 `send_head` 查表，借鉴 LL「按原 token 下标直接寻址」，并把第一维从 `num_experts` 压到 ≤ topk。

---

## 8. 通用整理：rank-major / expert-major 下的数组结构

先忘掉 DeepEP 里那些具体名字，只问一件事：

**dispatch 之后，目的卡上的 `recv_x` 每一行代表什么？**

答案只有两种。这两种决定了后面所有数组该长什么样。

### 8.1 用一个具体例子钉死两种布局

假设：4 张卡，每张 2 个 expert（全局 expert `0..7`），topk = 2。源卡 rank0 的 token `t=5` 选中 expert `1` 和 `4`：

- expert1 → 落在 rank0
- expert4 → 落在 rank2

所以这个 token 要发给 **rank0** 和 **rank2** 各一份。

**Rank-major：一行 =「这个 token 到了这张卡」**

目的卡 rank2 收到后只占一行：

```text
recv_x 第 r 行 = token5 整份向量
旁边必须另有一张表告诉你：本卡该喂给哪个 expert
→ LocalMap[r] = [本地 expert0(=全局4),  -1, ...]
```

本卡如果有两个 expert 都命中同一个 token，**仍然只有一行**，LocalMap 里写两个本地 id。用户自己按 LocalMap 拆开做 GEMM，再自己把结果合成一份交给 combine。

**Expert-major：一行 =「这个 token × 某一个 expert」**

同样的 token，在 rank2 上可能占多行（每个命中的本地 expert 一行）：

```text
recv_x 第 e0 区间里某一行 = token5 给 expert4 的副本
recv_x 第 e1 区间里某一行 = token5 给 expert5 的副本（若也命中）
```

行落在哪个区间，就等于告诉你 expert 是谁，所以 **不需要** 再存一张 `recv_topk_idx`。用户可以直接 grouped GEMM；求和交给 combine。

| | Rank-major | Expert-major |
|---|---|---|
| 一行是什么 | 到这张卡的一个 token | 到这张卡某个 expert 的一个副本 |
| 行数 | 少（按 token 去重） | 多（按 topk 命中展开） |
| 「属于哪个 expert」写在哪 | 旁边一张表（LocalMap） | 行本身所在的区间 |

DeepEP 里开关就是 `kDoExpand`：关 = rank-major，开 = expert-major（`dispatch_copy_epilogue.cuh` L108–L121）。

### 8.2 不管哪种布局，通信都要答 4 个问题

把一次 EP 想成寄信再回信：

```text
去程（dispatch）
  源卡：token5 要寄到 rank0、rank2
  目的卡：信放到自己邮箱的哪一格？邮箱里有几封？

回程（combine）
  目的卡：算完了，回信寄回 rank0 的 token5
  源卡：token5 要收几封回信？从哪几格读？
```

对应到数据结构，就是 **5 块**（其中 1 块可选）：

```text
源卡手里
  ① Route（路由）        「token5 去哪些 expert」→ 也推出「该收哪几份回信」
  ② ForwardLoc（去程定位）「token5 在对端邮箱的第几格」（有时可以不要）

目的卡手里
  ③ Layout（布局/计数）   「邮箱怎么分段、每段多长」
  ④ ReturnInfo（回程信息）「我这一行的结果，该寄回哪张卡的哪个 token」
  ⑤ LocalMap（本地映射） 「我这一行对应哪些本地 expert」
                           （rank-major 必有；expert-major 常隐含在区间里）
```

| 抽象名 | 回答的问题 | 记在哪张卡 | 谁写 |
|---|---|---|---|
| Route | 每个 token 去哪些 expert、权重多少 | 源卡 | 用户（gate） |
| Layout | 每段多长、从哪开始 | 两边 | dispatch（notify） |
| ForwardLoc | 我的 token `t` 在对端放到了哪格 | 源卡 | dispatch 发送方 |
| ReturnInfo | 我这一行来自哪张卡的哪个 token | 目的卡 | dispatch 接收方 |
| LocalMap | 这一行要给哪些本地 expert / 对应哪些展开行 | 目的卡 | dispatch 接收方 |

### 8.3 Rank-major：必有什么、可选什么

目的卡 rank2 的 `recv_x` 按源卡分段排：

```text
recv_x:
  [来自 rank0 ...] [来自 rank1 ...] [来自 rank3 ...]
       ↑
     其中某一行 r = 源 rank0 的 token5
```

**必有三张表**

1. **Layout**：第几行到第几行是 rank0 发来的。没有它，不知道该扫哪一段。
2. **ReturnInfo**：`ReturnInfo[r] = (源卡, 源 token 号)`，例如 `(rank0, t=5)`。combine 靠它寄回去。
3. **LocalMap**：因为一行只代表「到了这张卡」，不代表「某个 expert」：

```text
LocalMap[r] = [0, -1, ...]   # 本卡本地 expert0（=全局4）要命中
```

**可选一张表**

4. **ForwardLoc**：源卡记 `token5 在 rank2 邮箱的第 slot 格`。V1 环形队列要靠它找回程格；V2 回程直接写源卡 `[贡献者][t=5]`，这张表通常不要（cached dispatch 才留着复用）。

**数据流**

```text
dispatch 后，目的卡有：
  recv_x[r]          一行一个去重 token
  LocalMap[r][k]     本地 expert
  ReturnInfo[r]      回哪张卡、哪个 t
  Layout             各源 rank 段从哪到哪

用户：按 LocalMap 自己拆开做 expert，再合成一行结果 y[r]
combine：读 ReturnInfo[r] → 写回源卡「token t 的某一格」
源卡：看自己的 Route，知道 token5 该从哪些格把结果加起来
```

适用：V1 normal（机内 / 跨机）、V2 非 expand。产出清单：

```text
dispatch 产出（目的卡）
  recv_x            [R, H]       R = Σ_src 去重后发来的 token 数，按源 rank 段排
  recv_topk_idx     [R, K]       = LocalMap；本地 expert id 或 −1
  recv_topk_weights [R, K]
  每 expert 计数     [E_local]    给用户开 GEMM buffer
  rank 段前缀和                  = Layout
  回程信息           [R] 或 [R, 2+K]  = ReturnInfo
dispatch 产出（源卡）
  去程定位（V1：send_head；V2：dst_buffer_slot_idx，可选）= ForwardLoc

combine 需要
  x [R, H]（与 recv_x 同行序）+ handle（Layout + ReturnInfo + V1 的 ForwardLoc）
```

### 8.4 Expert-major：必有什么

同样的 token5，在目的卡上落在某个 expert 区间里：

```text
recv_x:
  expert0 区间: [tokenA, token5, tokenC, pad, pad, ...]
  expert1 区间: [tokenD, ...]
```

**必有两张表（LocalMap 被吃掉了）**

1. **Layout（按 expert 分段）**：expert0 从第 0 行到第 7 行，expert1 从第 8 行起……行落在哪个区间，就等于 LocalMap。所以不再需要 `recv_topk_idx[r][k]`。
2. **ReturnInfo**：仍然要回答「这一行结果寄回谁」。有两种记法，本质一样：

| 记法 | 怎么记 | DeepEP 哪里用 |
|---|---|---|
| A. 按展开行记 | 每个 GEMM 输入行一条 `(src, t)` | V1 LL：`src_info[e][j]` |
| B. 按去重行记 + 指针 | 去重行记 `(src, t)`，再另存「第 k 个 topk 对应展开行几」 | V2 expand：`metadata[0..1]` + `metadata[2+k]` |

V2 用记法 B（`combine.cuh` L89–L92 解 `(src, t)`；L114–L117 读展开行指针）：

```text
metadata[i] = [全局 token 号 g,  src*K+master,  展开行0, 展开行1, ...]
                 ↑ 回哪个 t        ↑ 回哪张卡      ↑ LocalMap 被换成「展开行指针」
```

**数据流**

```text
dispatch 后，目的卡有：
  recv_x[展开行]     一行一个 (token, expert) 副本
  Layout             每个本地 expert 的起止
  ReturnInfo         怎么寄回源卡（记法 A 或 B）

用户：直接 grouped GEMM，每行出一个结果
combine：用 ReturnInfo（+展开行指针）找到源卡的 t，把各 expert 结果寄回去
源卡：仍用 Route 决定 token5 收哪几份，再求和
```

适用：V1 LL、V2 expand。产出清单：

```text
dispatch 产出（目的卡）
  recv_x
    V1 LL：[E_local, ranks × MaxTok, H]，每 expert 固定容量，内部按 src rank 分段
    V2 expand：[Σ_e align(count_e), H]，expert 间对齐 padding
  每 expert 计数 / 前缀和          = Layout
  回程信息                         = ReturnInfo
    V1 LL：每个展开行一条 src_info（源 token 号）+ 每 (expert, src) 的 (count, begin)
    V2 expand：仍按 rank-major 接收行记 metadata，[2+k] 指向展开行

combine 需要
  x 与 recv_x 同布局 + ReturnInfo
  源卡还需要自己的 Route（topk_idx）；V1 LL 另需 topk_weights（源卡加权）
```

### 8.5 装箱单：两种布局最少要带什么

```text
                    Rank-major              Expert-major
────────────────────────────────────────────────────────────
recv_x 一行是啥      (token, 目的卡)          (token, 本地 expert)
行数                 少（去重）               多（展开）

Layout               按「源 rank」分段         按「本地 expert」分段
LocalMap             必有：recv_topk_idx      没有：区间本身就是
ReturnInfo           每去重行一条 (src,t)     每展开行一条，或
                                              去重行+(展开行指针)
ForwardLoc           环形/cached 才要         通常不要
多 expert 谁求和     用户在 expert 侧          combine（本卡或源卡）
```

一句话：**布局只决定「一行代表 token 还是 (token,expert)」；其余数组都是在为「怎么分段读、怎么寄回去」服务。** Rank-major 缺不了 LocalMap；Expert-major 把 LocalMap 融进 Layout，但 ReturnInfo 一点都不能少。

### 8.6 回程四问 + DeepEP 对号入座

每一份部分结果要回到源卡，必须能回答：

1. 回哪张卡（src rank）
2. 回到哪个 token（src token 号）
3. 放在源卡哪一格、怎么和同一 token 的其他份区分
4. 源卡上 token `t` 要收哪几份

| 问题 | Rank-major 怎么答 | Expert-major 怎么答 |
|---|---|---|
| 去程放到哪 | Layout 的 rank 段 + 紧凑 slot | Layout 的 expert 段 + 展开行 |
| 这一行属于谁算 | LocalMap 显式写本地 expert | 看行落在哪个 expert 区间 |
| 结果寄回谁 | ReturnInfo[r]=(rank0,5) | ReturnInfo 同左（或带展开行指针） |
| 源卡收几份 | 看 Route：命中几个 rank | 看 Route：命中几个 expert/rank |

DeepEP 各实现的具体名字：

| 抽象 | V1 normal 机内 | V1 normal 跨机 | V1 LL | V2 直连 |
|---|---|---|---|---|
| Route | `topk_idx`、`topk_weights`（combine 时不需要 topk_idx） | 同左 | `topk_idx`、`topk_weights`（combine 必须再传，用于加权） | `topk_idx`（handle 里存一份，combine epilogue 用） |
| Layout | `rank_prefix_matrix`、`channel_prefix_matrix`、`recv_channel_prefix_matrix`、`num_recv_tokens_per_expert_list` | `rdma/gbl_channel_prefix_matrix`、`recv_rdma_rank_prefix_sum`、`recv_gbl_rank_prefix_sum` 等 | `packed_recv_count`、`packed_recv_layout_range`（每 (expert, src rank) 的 `(count, begin)`，`internode_ll.cu` L411） | `psum_num_recv_tokens_per_scaleup_rank`、`psum_num_recv_tokens_per_expert`、`num_unaligned_recv_tokens_per_expert` |
| ForwardLoc | `send_head[t][rank]`（环形尾位置，−1 不发；`intranode.cu` L359–L360） | `send_rdma_head`、`send_nvl_head` | 无（回程直接按 `[expert][t]`） | `dst_buffer_slot_idx[t][k]`（仅 cached dispatch 用） |
| ReturnInfo | `recv_src_idx[r]`（`intranode.cu` L500–L501） | `recv_src_meta[r]` = `SourceMeta{src_rdma_rank, nvl 位图}`（`internode.cu` L22–L23） | `packed_recv_src_info[e][j]` = 源 token 号（L429） | `recv_src_metadata[r] = [g, src*K+master, 展开行×K]` |
| LocalMap | `recv_topk_idx[r][k]`（本地 id 或 −1） | 同左 | 隐含在 `[e][j]` 布局里 | 非 expand：`recv_topk_idx[r][k]`；expand：`metadata[r][2+k]` |

回程四问在各实现里的落点：

| 问题 | V1 normal 机内 | V1 LL | V2 直连 |
|---|---|---|---|
| 回哪张卡 | 行所在的 rank 段 | `layout_range` 的 src rank 下标 | `metadata[1] / K` |
| 哪个 token | `recv_src_idx[r]` | `src_info[e][j]` | `metadata[0] % MaxTok` |
| 放哪格 | 环形 slot（head/tail 流控） | `[全局 expert][t]` | `[贡献者][t]` |
| `t` 收哪几份 | `send_head[t][rank] ≥ 0` | 源卡 `topk_idx[t]` | 源卡 `topk_idx[t]`（去重到 rank 或 lane） |

handle 组成（Python 层）：V1 机内 `(rank_prefix_matrix, channel_prefix_matrix, recv_channel_prefix_matrix, recv_src_idx, is_token_in_rank, send_head)`（`legacy.py` L401）；V1 跨机 10 元组含 `recv_src_meta, send_rdma_head, send_nvl_head`（L500–L502）；V1 LL `(packed_recv_src_info, packed_recv_layout_range, MaxTok, hidden, num_experts)`（L617）；V2 是 `EPHandle` 对象（`elastic.py` L25 起）。

V1 跨机 normal 结构同机内，只是两级：RDMA 段 + NVL 段，靠 `SourceMeta` 位图和 `send_rdma_head` / `send_nvl_head`。

---

## 9. 完整可运行验证：纯 Python 模拟 V2 直连坐标

文件：`communication/20_v2_dispatch_combine_sim.py`。模拟发送方抢 slot、`buffer[dst][src][slot]` staging、epilogue（rank-major / expand）、metadata、combine 写回 `[贡献者][t]`、源卡 reduce，并与参考结果逐元素比较。所有写入前都断言该格为空，用来验证「每格只有一个写者」。

```python
#!/usr/bin/env python3
"""Simulate DeepEP V2 direct-mode dispatch/combine index bookkeeping in pure Python.

Models: sender-side slot counters, buffer[src][slot] staging, rank-major vs
expert-major (expand) recv tensors, recv_src_metadata, and combine write-back
into buffer[contributor][original token_idx] followed by source-side reduce.
"""

from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    num_ranks: int
    num_experts: int
    num_topk: int
    max_tokens: int
    hidden: int = 4
    alignment: int = 4

    @property
    def experts_per_rank(self) -> int:
        return self.num_experts // self.num_ranks


def expert_fn(expert: int, vec: list[int]) -> list[int]:
    return [v * (expert + 1) + expert for v in vec]


def vec_add(a: list[int], b: list[int]) -> list[int]:
    return [x + y for x, y in zip(a, b)]


def align(n: int, a: int) -> int:
    return (n + a - 1) // a * a


def master_lane(values: list[int], k: int) -> bool:
    """ptx::deduplicate: keep the highest lane among lanes with the same value."""
    return max(j for j, v in enumerate(values) if v == values[k]) == k


def make_inputs(cfg: Config, seed: int):
    rng = random.Random(seed)
    xs, topks = [], []
    for _ in range(cfg.num_ranks):
        n = rng.randint(1, cfg.max_tokens)
        xs.append([[rng.randint(-3, 3) for _ in range(cfg.hidden)] for _ in range(n)])
        rows = []
        for _ in range(n):
            row = rng.sample(range(cfg.num_experts), cfg.num_topk)
            rows.append([e if rng.random() > 0.2 else -1 for e in row])
        topks.append(rows)
    return xs, topks


def dispatch(cfg: Config, xs, topks):
    """Return buffer[dst][src][slot] records, sender counters, dst_buffer_slot_idx."""
    R, K, M = cfg.num_ranks, cfg.num_topk, cfg.max_tokens
    buffer = [[[None] * M for _ in range(R)] for _ in range(R)]
    counter = [[0] * R for _ in range(R)]
    dst_buffer_slot_idx = []
    for src in range(R):
        per_src = []
        for t, (x, tk) in enumerate(zip(xs[src], topks[src])):
            dst_ranks = [e // cfg.experts_per_rank if e >= 0 else -1 for e in tk]
            row = [-1] * K
            for k in range(K):
                d = dst_ranks[k]
                if d < 0 or not master_lane(dst_ranks, k):
                    continue
                slot = counter[src][d]
                counter[src][d] += 1
                assert buffer[d][src][slot] is None
                buffer[d][src][slot] = {"x": x, "topk": tk, "src_global": src * M + t}
                row[k] = src * M + slot
            per_src.append(row)
        dst_buffer_slot_idx.append(per_src)
    return buffer, counter, dst_buffer_slot_idx


def copy_epilogue(cfg: Config, buffer, counter, dst: int, expand: bool):
    """Scan buffer[dst][src][0:count] in src order; build recv tensors + metadata."""
    R, K, EPR = cfg.num_ranks, cfg.num_topk, cfg.experts_per_rank
    lo, hi = dst * EPR, (dst + 1) * EPR
    records = [(src, buffer[dst][src][s]) for src in range(R) for s in range(counter[src][dst])]

    expert_count = [0] * EPR
    for _, rec in records:
        for e in rec["topk"]:
            if lo <= e < hi:
                expert_count[e - lo] += 1
    base = [0]
    for c in expert_count:
        base.append(base[-1] + align(c, cfg.alignment))
    cursor = base[:-1]

    recv_x, recv_topk_idx, metadata = [], [], []
    if expand:
        recv_x = [None] * base[-1]
    for src, rec in records:
        local = [e - lo if lo <= e < hi else -1 for e in rec["topk"]]
        master = max(k for k in range(K) if local[k] >= 0)
        meta = [rec["src_global"], src * K + master] + [-1] * K
        if expand:
            for k in range(K):
                if local[k] >= 0:
                    row = cursor[local[k]]
                    cursor[local[k]] += 1
                    recv_x[row] = (rec["x"], local[k])
                    meta[2 + k] = row
        else:
            recv_x.append(rec["x"])
            recv_topk_idx.append(local)
        metadata.append(meta)
    return recv_x, recv_topk_idx, metadata, base


def expert_compute(cfg: Config, dst: int, recv_x, recv_topk_idx, expand: bool):
    lo = dst * cfg.experts_per_rank
    if expand:
        return [None if item is None else expert_fn(lo + item[1], item[0]) for item in recv_x]
    out = []
    for x, local in zip(recv_x, recv_topk_idx):
        acc = [0] * cfg.hidden
        for e in local:
            if e >= 0:
                acc = vec_add(acc, expert_fn(lo + e, x))
        out.append(acc)
    return out


def combine(cfg: Config, y, metadata, me: int, expand: bool, multi_reduce: bool, comb):
    """Write partial results into comb[src][contributor][token_idx]."""
    K, M = cfg.num_topk, cfg.max_tokens
    rank_layout = multi_reduce and cfg.num_ranks <= K
    expanded_send = expand and not multi_reduce
    for i, meta in enumerate(metadata):
        t = meta[0] % M
        src, master = meta[1] // K, meta[1] % K
        slots = meta[2:]
        if expanded_send:
            for k in range(K):
                if slots[k] >= 0:
                    assert comb[src][k][t] is None
                    comb[src][k][t] = y[slots[k]]
            continue
        if expand:
            vec = [0] * cfg.hidden
            for s in slots:
                if s >= 0:
                    vec = vec_add(vec, y[s])
        else:
            vec = y[i]
        strip = me if rank_layout else master
        assert comb[src][strip][t] is None
        comb[src][strip][t] = vec


def source_reduce(cfg: Config, comb, src: int, topks, expand: bool, multi_reduce: bool):
    K = cfg.num_topk
    rank_layout = multi_reduce and cfg.num_ranks <= K
    out = []
    for t, tk in enumerate(topks[src]):
        ranks = [e // cfg.experts_per_rank if e >= 0 else -1 for e in tk]
        if expand and not multi_reduce:
            strips = [k for k in range(K) if ranks[k] >= 0]
        else:
            lanes = [k for k in range(K) if ranks[k] >= 0 and master_lane(ranks, k)]
            strips = [ranks[k] if rank_layout else k for k in lanes]
        acc = [0] * cfg.hidden
        for s in strips:
            acc = vec_add(acc, comb[src][s][t])
        out.append(acc)
    return out


def reference(cfg: Config, xs, topks, src: int):
    out = []
    for x, tk in zip(xs[src], topks[src]):
        acc = [0] * cfg.hidden
        for e in tk:
            if e >= 0:
                acc = vec_add(acc, expert_fn(e, x))
        out.append(acc)
    return out


def run(cfg: Config, seed: int, expand: bool, multi_reduce: bool) -> int:
    xs, topks = make_inputs(cfg, seed)
    buffer, counter, _ = dispatch(cfg, xs, topks)
    rank_layout = multi_reduce and cfg.num_ranks <= cfg.num_topk
    num_strips = cfg.num_ranks if rank_layout else cfg.num_topk
    comb = [[[None] * cfg.max_tokens for _ in range(num_strips)] for _ in range(cfg.num_ranks)]
    for dst in range(cfg.num_ranks):
        recv_x, recv_topk_idx, metadata, _ = copy_epilogue(cfg, buffer, counter, dst, expand)
        y = expert_compute(cfg, dst, recv_x, recv_topk_idx, expand)
        combine(cfg, y, metadata, dst, expand, multi_reduce, comb)
    for src in range(cfg.num_ranks):
        assert source_reduce(cfg, comb, src, topks, expand, multi_reduce) == reference(cfg, xs, topks, src)
    return num_strips


def trace(cfg: Config, seed: int) -> None:
    xs, topks = make_inputs(cfg, seed)
    buffer, counter, slot_idx = dispatch(cfg, xs, topks)
    tk = topks[0][0]
    print(f"src=0 token=0 topk={tk} dst_ranks={[e // cfg.experts_per_rank if e >= 0 else -1 for e in tk]}")
    print(f"  dst_buffer_slot_idx={slot_idx[0][0]}  (value = src*MaxTok + slot, -1 = non-master lane)")
    for dst in range(cfg.num_ranks):
        _, _, metadata, _ = copy_epilogue(cfg, buffer, counter, dst, expand=True)
        for row, meta in enumerate(metadata):
            if meta[0] == 0:
                print(f"  rank{dst}: recv row {row} metadata={meta}")


def main() -> None:
    configs = [
        Config(num_ranks=4, num_experts=8, num_topk=4, max_tokens=8),
        Config(num_ranks=8, num_experts=16, num_topk=2, max_tokens=8),
    ]
    trace(configs[0], seed=0)
    for cfg in configs:
        for expand in (False, True):
            for multi in (True, False):
                strips = 0
                for seed in range(50):
                    strips = run(cfg, seed, expand, multi)
                print(f"R={cfg.num_ranks} K={cfg.num_topk} expand={expand!s:5} multi_reduce={multi!s:5} "
                      f"combine strips={strips} OK")
    print("all checks passed")


if __name__ == "__main__":
    main()
```

运行：

```bash
python3 communication/20_v2_dispatch_combine_sim.py
```

期望输出：

```text
src=0 token=0 topk=[4, 0, 5, -1] dst_ranks=[2, 0, 2, -1]
  dst_buffer_slot_idx=[-1, 0, 0, -1]  (value = src*MaxTok + slot, -1 = non-master lane)
  rank0: recv row 0 metadata=[0, 1, -1, 0, -1, -1]
  rank2: recv row 0 metadata=[0, 2, 0, -1, 8, -1]
R=4 K=4 expand=False multi_reduce=True  combine strips=4 OK
R=4 K=4 expand=False multi_reduce=False combine strips=4 OK
R=4 K=4 expand=True  multi_reduce=True  combine strips=4 OK
R=4 K=4 expand=True  multi_reduce=False combine strips=4 OK
R=8 K=2 expand=False multi_reduce=True  combine strips=2 OK
R=8 K=2 expand=False multi_reduce=False combine strips=2 OK
R=8 K=2 expand=True  multi_reduce=True  combine strips=2 OK
R=8 K=2 expand=True  multi_reduce=False combine strips=2 OK
all checks passed
```

（`R=8 K=2` 时 rank 数 > topk，第一维只开 2 条，用 master lane 区分贡献者。）

### 9.1 走读 trace（R=4, E=8, K=4, 每 rank 2 个 expert, MaxTok=8, alignment=4）

源卡 rank0 的 token0：`topk=[4, 0, 5, -1]`。

| lane | expert | 目的 rank `= e / 2` | 是否 master | 去程 |
|---|---|---|---|---|
| 0 | 4 | 2 | 否（rank2 的最高 lane 是 2） | 不单独发 |
| 1 | 0 | 0 | 是 | rank0 计数器 → slot 0，`dst_buffer_slot_idx = 0*8+0 = 0` |
| 2 | 5 | 2 | 是 | rank2 计数器 → slot 0，值 0 |
| 3 | −1 | — | — | — |

epilogue（expand）后的 metadata：

| 目的卡 | metadata | 含义 |
|---|---|---|
| rank0 | `[0, 1, -1, 0, -1, -1]` | `g=0`→源 token 0；`1 = src0*4 + master lane 1`；lane1 的本地 expert0 落在展开行 0 |
| rank2 | `[0, 2, 0, -1, 8, -1]` | master lane 2；lane0（expert4→本地0）落在行 0；lane2（expert5→本地1）落在行 8（本地 expert0 区间对齐后长 8，所以 expert1 从 8 开始） |

combine 回程（R=4 ≤ K=4，允许多次 reduce → rank layout）：

| 写方 | 动作 | 写到源卡 rank0 |
|---|---|---|
| rank0 | 只有 1 个副本，Case A | `comb[strip=0][t=0]` |
| rank2 | 行 0 + 行 8 本卡求和，Case B | `comb[strip=2][t=0]` |

源卡 reduce：topk 推出 rank `[2, 0, 2, -1]`，去重保留 lane 1、2 → rank {0, 2} → 读 `comb[0][0] + comb[2][0]`。

若改成 expanded send（禁止多次 reduce）：rank0 写 `comb[1][0]`，rank2 写 `comb[0][0]` 和 `comb[2][0]`；源卡按 lane {0,1,2} 读三份再求和。

---

## 10. 常见误判与症状

| 误判 | 实际 | 后果 / 症状 |
|---|---|---|
| `buffer[src][slot]` 存的是 token_idx | 存整条 token 记录，身份在 `src_token_global_idx` 字段 | 把 slot 当 token 号回写，combine 结果错位 |
| combine 写回 dispatch 时的同一格 | 写的是源卡上另一套坐标 `[贡献者][t]` | 去 expert 卡找回程数据，找不到 |
| dispatch / combine 是两块 buffer | 同一块内存，不同时间两套视图 | 误以为 combine 期间 dispatch 数据还在 staging 里 |
| V2 是 expert-major | staging 永远 rank-major；`recv_x` 只有 `do_expand` 才 expert-major | 非 expand 时直接当 grouped GEMM 输入，expert 混在一起 |
| 贡献者下标 = expert id | 是 rank 号或 topk lane | 第一维越界（只有 ≤ topk 条） |
| expand 时 metadata 按展开行记 | 仍按 rank-major 接收行 `i` 记，`[2+k]` 指向展开行 | combine 扫行数用错（应扫 `num_recv_tokens`） |
| V2 接收行顺序是固定的 | slot 和展开行都由 atomic 分配，需 `deterministic_sort` 按 `g` 排序 | 两次运行 `recv_x` 行序不同，结果比对 / bitwise 复现失败 |
| 两边 master lane 规则可以不同 | 必须同为最高 lane | 源卡读错 strip，某些贡献丢失或读到旧数据 |

---

## 11. 自测题与答案

1. **dispatch 和 combine 用的是两块 buffer 吗？**
   答：不是。同一块对称窗口，dispatch 视图 `[kNumRanks][MaxTok]`，combine 视图 `[kNumTokensInLayout][MaxTok]`，都从开头起算；大小取两者最大值。

2. **为什么去程用 slot、回程用原 token_idx？**
   答：去程接收方不知道会来哪些 token，需要紧凑 slot + 计数前缀和连续读；回程接收方是 token 主人，用自己的 `t` 直接寻址，免查表、与输出行对齐、第一维 ≤ topk。

3. **R=8、K=2、允许多次 reduce 时，combine 第一维几条、下标是什么？**
   答：`use_rank_layout` 要求 `R ≤ K`，不满足，所以 2 条，下标是该 rank 的 master topk lane。

4. **V2 expand 模式下 `recv_src_metadata` 有几行？`[2+k]` 是什么？**
   答：行数 = rank-major 接收 token 数；`[2+k]` 是该 token 第 k 个 topk 副本在 expert-major `recv_x` 里的展开行，−1 表示不在本卡。

5. **把五类数组对到 V1 LL：去程定位和回程信息分别是什么？**
   答：去程定位没有（回程直接写 `[全局 expert][t]`）；回程信息是 `packed_recv_src_info[e][j]`（源 token 号）加 `packed_recv_layout_range[e][src]`（count, begin）。

---

## 12. 学习进度

- [x] V1 normal 机内 / 跨机、LL dispatch/combine（10–17）
- [x] LL recv hook（18）、elect+shfl 陷阱（19）
- [x] V2 直连：buffer 两套视图、去程/回程坐标、expand、通用 handle 数组
- [ ] V2 hybrid：scaleout 两级中转与 linked list

### 下一知识点

`hybrid_dispatch.cuh` / `hybrid_combine.cuh`：scaleup 与 scaleout 两级 staging、`token_metadata_at_forward`、`channel_linked_list`，以及两级 reduce 时贡献者坐标怎么叠加。
