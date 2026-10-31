# MoK 的元数据管线：count → pad → schedule 与行语义的三次变化

> 承接 [02-mok-scheduling-and-buffers.md](./02-mok-scheduling-and-buffers.md) 与 [29-mok-moe-task-graph-and-grouped-gemm.md](./29-mok-moe-task-graph-and-grouped-gemm.md)。02 讲了"schedule 为什么存在、pull/push 方向"，29 讲了"这张 schedule 怎么变成 tile 级任务"。夹在中间的那一段一直没拆：**count / pad / schedule 三个 kernel 到底怎么把路由决策编译成一张地址表**。
>
> 代码基准：`mixture-of-kittens` 本地仓库。
>
> - `csrc/scheduler.cuh`：`count_kernel` / `pad_kernel` / `schedule_kernel`
> - `csrc/utils.cuh`：`all_gather_top_experts_kernel` / `barrier_all_kernel`
> - `csrc/megakernel/dispatch_combine.cuh`：表的唯一消费端
> - `mok/functional.py`：`build_schedule` / `forward`
> - 复算脚本：[30_mok_schedule_pipeline_sim.py](./30_mok_schedule_pipeline_sim.py)

---

```text
本次讲解位置
章节：communication
小节：MoK 元数据管线
知识点：二维直方图、256 对齐的动机、轮转交错排位公式、dst 的连续性与双射、行语义三次变化
上次：29 笔记 —— 任务表公式、CLC 抢活、compute_cluster_idx 反算 tile
下次：dispatch_kernel 的 MXFP8 转置量化与 e8m0 scale 布局
```

## 0. 一句话

**count 说"每段多长"，pad 说"对齐后多长"，schedule 说"第几行去哪个 rank 取哪个槽位"。三者都是纯整数、零浮点、零通信（只有中间一次 barrier），产出的表是 dispatch 唯一的地址来源。**

---

## 1. 为什么要把元数据单独拉成一条线

EP=8 时，rank 0 只拥有全局 384 个专家里的 48 个。要算这 48 个专家的 GEMM，它需要**全世界的 token**——而 458,752 个槽位散落在 8 张卡的显存里。

```text
         rank0 的专家段（256 对齐）
         ┌───────────────────────────┐
         │ row 0     ?               │
         │ row 1     ?               │   ← 谁来填？从哪取？
         │ ...                       │
         │ row 57343 ?               │
         └───────────────────────────┘
```

dispatch 用的是 `tma::load_async`（异步 DMA），**地址必须在发射前就确定**。不能"边搬边问"，也不能等对端告诉你——对端根本不知道谁要读它的哪一行。

所以必须在进 megakernel 之前，把"谁去哪"彻底算完。这条线就是三个 kernel：

```text
① all_gather_top_experts    topk_all[8, 7168, 8] int32 = 1.8 MB
② barrier_all               一次跨卡对齐（仅此一次）
③ count    448 blk × 1024   cnt[e][r]        int32[48×8] = 1.5 KiB
④ pad       48 blk × 1      len[e] + total   int32[48] + 1
⑤ schedule 384 blk × 1024   rank[i], tok[i]  2 × int32[229376] = 1.75 MiB
        ─────────────────────────────────────────────────────
⑥ x.copy_ + barrier_all     数据从这里才开始动
⑦ megakernel
```

**①–⑤ 一个 bf16 都不碰。** 这是理解整个设计的第一个分水岭。

---

## 2. count：数的是"槽位"，不是 token

```cpp
// scheduler.cuh:56-57
if (expert_idx >= first_expert && expert_idx < last_expert)
    atomicAdd(&tokens_per_expert_and_peer[(expert_idx - first_expert) * world_size + peer_rank], 1);
```

被 `+1` 的是 `(peer, token, k)` 三元组，也就是"某个 peer 的某个 token 的第 k 个专家选择"。所以语义是：

> 本地专家 e 收到了**多少个槽位**（来自 rank r，topk 已展开）

一个 topk=8 的 token 如果 8 个选择全落在本 rank 上，它贡献 **8 次**。这不是 bug：grouped GEMM 的工作量单位就是"槽位数"，一个 `(token, expert)` 组合算一遍。

跑个最小例子（EP=2、4 个全局专家、每 rank 4 token、topk=2、`PAD` 缩成 4）：

```text
rank0 拥有专家 [0,2)
  e0: cnt=[3, 2]  raw=5  len=8
  e1: cnt=[2, 2]  raw=4  len=4
  num_tokens=12

rank1 拥有专家 [2,4)
  e2: cnt=[1, 2]  raw=3  len=4
  e3: cnt=[2, 2]  raw=4  len=4
  num_tokens=8
```

两个必检的性质：

- **全局合计 = W×T×K = 16**。`if` 过滤保证每个槽位只在**拥有该专家**的那一个 rank 上被数到，全局恰好一次。
- e0 的 5 加 e1 的 4 是 9，**不等于** T×K=8——因为有个 token 的两个选择都落在 rank0。**token ≠ 槽位**。

真实规模（Kimi 形状，`tests/utils.py:13`）：全局 458,752 槽位，均衡时每 rank 数到 57,344 = 224×256。

### 2.1 三级归约：把 45 万次 L2 原子压到 2 万

```cpp
// scheduler.cuh:45-48  ← 注意：这个 extern __shared__ 数组和 G 的字段同名
extern __shared__ int tokens_per_expert_and_peer[];   // (num_local_experts, world_size)
for (int i = threadIdx.x; i < G.tokens_per_expert_and_peer.cols(); i += blockDim.x)
    tokens_per_expert_and_peer[i] = 0;
__syncthreads();
```

> **命名陷阱**：不带 `G.` 前缀是共享内存，带 `G.` 是显存。`:62` 那行 `atomicAdd(&G.tokens_per_expert_and_peer[{i}], tokens_per_expert_and_peer[i])` 两个都在，靠前缀区分。

三级：

| 级别 | 位置 | 次数 |
|---|---|---|
| A. 块内原子 | smem（SM 内硬件原子，个位数周期） | 458,752 |
| B. 一次性 flush | `atomicAdd(&G..., 块内批值)` | 448 × ~48 ≈ 21,500 |
| C. 跳过零项 | `if (tokens_per_expert_and_peer[i] != 0)` | 跳过 87.5% |

| | 全局 L2 原子次数 | 每次的值 |
|---|---|---|
| 不用 smem | **458,752** | 1 |
| 用 smem | ≈ **21,500** | 平均 ~2.7 |

**约 21 倍削减**，且每个 L2 地址的争用从 ~9,500 次降到 ~450 次。这才是这一步的真正收益。

`!= 0` 那个跳过为什么能命中 87.5%？因为 `rank_stride = 57344` 是 blockDim=1024 的整数倍（56 倍），所以**一个 block 的 1024 个元素必然落在同一个 peer 内**。每块 384 个槽里只有 48 个（那一个 peer 的整列）可能非零。

顺带一个设计观察：既然 block ⊂ 单 peer，smem 其实只需开 `[48]`。代码开了 `[48][8]`，用 1.5 KiB 换"block 不必局限于单 peer"这个假设不成立也能跑——很便宜的通用的。

### 2.2 为什么必须二维

一维 `tokens_per_expert[48]` 在 count 阶段完全够用（`pad` 也确实只把它加起来当总数）。二维存在的唯一理由是**给 `schedule` 用**，见 §4。

### 2.3 零初始化的两种要求

| 张量 | 分配 | 为什么 |
|---|---|---|
| `tokens_per_expert_and_peer` | `at::zeros`（`:161`） | 靠 `atomicAdd` **累加**，初值必须 0 |
| smem 副本 | 手动清零（`:46-47`） | smem 是脏的 |
| `tokens_per_expert` | `at::empty`（`:160`） | `pad` 是**赋值** `=`，每块独占一个 index |
| `schedule_peer_rank` | `at::empty` + `fill_(-1)`（`:157,162`） | 赋值的，`-1` 是 padding 标记 |

---

## 3. pad：只对齐，不填充

```cpp
// scheduler.cuh:66-77
static __device__ __forceinline__ void pad_kernel(const globals &G) {
    const int local_expert = blockIdx.x;
    const int world_size = G.topk.depth();
    int num_tokens = 0;
    for (int peer_rank = 0; peer_rank < world_size; ++peer_rank)
        num_tokens += G.tokens_per_expert_and_peer[{local_expert * world_size + peer_rank}];
    const int padded_num_tokens = (num_tokens + 256 - 1) / 256 * 256;
    G.tokens_per_expert[{local_expert}] = padded_num_tokens;
    atomicAdd(&G.num_tokens[{0}], padded_num_tokens);
}
```

**`pad` 不 pad 任何数据。** 它没有任何"写零填充行"的动作；填充行是 `fill_(-1)` 留下的，`pad` 只是决定"哪个区间不需要人写"。

launch 是理解它的关键：

```cpp
// scheduler.cuh:178-179
<<<num_local_experts, 1, 0, stream>>>(G);   // 48 blocks，每块 1 个线程
```

**48 个 block、每块 1 线程、0 smem。** 全 GPU 同时只有 48 个活跃线程。原因是 `blockIdx.x` **直接就是专家号**，一行分支都不用有。

### 3.1 向上取整的各种输入

```text
   raw   padded  waste
     0        0      0    ← 注意！不是 256
     1      256    255
   255      256      1
   256      256      0
   257      512    255
  1000     1024     24
  7168     7168      0
```

`raw = 0 → padded = 0`，空专家占 **0 行**、不浪费任何 tile。所以 batch 很小、大量专家空转时不会凭空膨胀。

### 3.2 为什么是 256 —— 唯一真正的理由

`EXPERT_PADDING = 256` 和 `types.cuh:3` 的 `MLP_Mb = 256` 是同一个数字。收益在 grouped GEMM 里兑现：

```cpp
// grouped_gemm.cuh:100
const int expert_row_blocks = tokens_per_expert[{expert_idx}] / config::MLP_Mb;
```

**一个不做任何 rounding 的整数除法。** 它只有当段长已是 256 的倍数时才正确——pad 就是那个保证。

接着 `grouped_gemm.cuh:101-110` 在任务空间里线性扫：每个专家贡献 `expert_row_blocks × col_blocks` 个任务，逐段扣减 `task_idx` 直到落进去。如果某段不是 256 的倍数，就会有一个 `MLP_Mb` tile **横跨两个专家**——一个 tile 只能有一个权重矩阵，直接算错。

```text
tile 边界：   0    256   512   768  1024
专家段：      [─── e0 ───][── e1 ──][──── e2 ────]
              每 256 行必然整套落在一个专家内
```

另外 `DISPATCH_Mb = 128`（`types.cuh:24`），一个 256 段 = 恰好 2 个 dispatch chunk，dispatch 侧的对齐也自动成立。

### 3.3 pad 故意不做两件事

**① 不写填充行**，只"宣布"区间 `[base, base+len)` 属于某专家。`[base+cnt, base+len)` 保留 `-1`，消费端据此跳过。

**② 不算前缀和**，让 `schedule` 每次自己累加（`scheduler.cuh:96-98`）。代价是 O(E²) 冗余读（384 个 block × 平均 24 次 ≈ 9,200 次全局读，全在 L2 里广播命中），换来 `tokens_per_expert` 保持"**每段多长**"这个单一语义——调试时直接看到 `[256, 512, 7168, ...]` 就知道每段多长，不用做差分。

---

## 4. schedule：把三元组翻译成行号

### 4.1 输出语义

```text
schedule_peer_rank[i]      = 从哪个 rank 取
schedule_peer_token_idx[i] = 取该 rank 的第几个槽位 = token * topk + k
i                          = 本 rank 的 GEMM 行号
```

它是 **gather index**，不是数据。`num_tokens` 之外的位置（`i >= num_tokens`）保持 `-1`。

launch：

```cpp
// scheduler.cuh:179-180
<<<num_local_experts * world_size, 1024, world_size * sizeof(int), stream>>>
// = 384 blocks × 1024 threads, smem = 32 B
```

**grid 大小 = 任务总量，所以里面那个 `for (idx = blockIdx.x; idx < ...; idx += gridDim.x)` 每个 block 只跑一轮。** 每个 block 负责一个 `(本地专家, peer)` 对：

```cpp
// scheduler.cuh:92-94
const int local_expert = idx / world_size;
const int peer_rank    = idx % world_size;
```

为什么这么切？因为**写入目标天然按 (专家, peer) 分开**：一个 block 只往自己那段写，**不需要任何锁、原子或协商**。代价是每块都要把整个 peer slab 扫一遍再丢弃（见 §4.6）。

### 4.2 两趟扫描

**第一趟（`:106-110`）只为数个数**：每个线程数自己负责的 56 个槽位里有多少命中了本专家。

**第二趟（`:131-145`）才真正写**。中间夹着一次两级前缀和求出 `j`。

### 4.3 `j`：压缩后的名次

```cpp
// scheduler.cuh:111-117  warp 内 Hillis-Steele scan
int inclusive = _tokens_from_peer_rank;
for (int offset = 1; offset < WARP_THREADS; offset *= 2) {
    const int n = __shfl_up_sync(0xffffffff, inclusive, offset);
    if (warp::laneid() >= offset) inclusive += n;
}
if (warp::laneid() == WARP_THREADS - 1) cumulative_tokens_from_peer_rank[warpid()] = inclusive;
```

```cpp
// scheduler.cuh:119-127  跨 warp scan
if (warpid() == 0) {
    int warp_total = (warp::laneid() < config::NUM_WARPS) ? cumulative_tokens_from_peer_rank[warp::laneid()] : 0;
    ...
}
```

```cpp
// scheduler.cuh:129
int j = (warpid() == 0 ? 0 : cumulative_tokens_from_peer_rank[warpid() - 1]) + inclusive - _tokens_from_peer_rank;
```

`inclusive - 自己` = exclusive，再加前面 warp 的总和 = **本线程第一个命中拿到的 `j`**。之后在写入循环里 `++j`。

> **隐藏硬约束**：`NUM_THREADS = 1024 → NUM_WARPS = 32`，**恰好等于 warp 的 32 个 lane**。所以 `laneid() < NUM_WARPS` 那个守卫恒为真、是死代码。若有人把 `NUM_THREADS` 改成 2048（64 warps），这段会静默漏掉一半。这是个没有 static_assert 保护的不变量。

**`j` 的含义只有一个：在"从 rank r 发往专家 e"的这一组元素里，我排第几个。** 范围 `[0, cnt[e][r])`。

### 4.4 交错排位公式

```cpp
// scheduler.cuh:134-141
int dst_token_idx = expert_base;
for (int rank = 0; rank < world_size; ++rank) {
    const int num_tokens = tokens_per_peer_rank[rank];          // = cnt[e][rank]
    dst_token_idx += min(num_tokens, j);                        // ② 前面整层的高度
    dst_token_idx += (rank < peer_rank && num_tokens > j) ? 1 : 0;  // ③ 本层内前面的兄弟
}
G.schedule_peer_rank[{dst_token_idx}]      = peer_rank;
G.schedule_peer_token_idx[{dst_token_idx}] = peer_token_idx;
```

翻译成中文：**先按"层"排（层 = `j`），层内按 peer 号排。**

```text
第 0 层：所有 peer 的第 0 个 → (j0,r0) (j0,r1) ... (j0,r7)
第 1 层：所有 peer 的第 1 个 → (j1,r0) (j1,r1) ...
第 2 层：只有 token 数 ≥ 3 的 peer 参与
```

`cnt = [3, 2, 4]` 的实际落位：

```text
        j=0          j=1          j=2          j=3
  j=0   r0->dst0     r1->dst1     r2->dst2
  j=1   r0->dst3     r1->dst4     r2->dst5
  j=2   r0->dst6       absent     r2->dst7
  j=3     absent       absent     r2->dst8
```

行序（= A 矩阵的行序）：

```text
row 0: peer0 的第 0 个      row 5: peer2 的第 1 个
row 1: peer1 的第 0 个      row 6: peer0 的第 2 个
row 2: peer2 的第 0 个      row 7: peer2 的第 2 个   ← peer1 没有第 2 个
row 3: peer0 的第 1 个      row 8: peer2 的第 3 个
row 4: peer1 的第 1 个
```

两个公式项的推导：

- `Σ_rank min(cnt[r], j) = Σ_rank Σ_{j'<j} [cnt[r] > j'] = Σ_{j'<j} #peer with cnt > j'` = **第 0..j-1 层的总高度**。`min` 的妙处：`min(cnt[r], j)` 正好等于"r 在前 j 层里贡献了几个"（每层最多 1 个，且只有前 cnt[r] 层有）。
- `(rank < peer_rank && num_tokens > j)` 在数：**本层中 peer 号比我小的还有几个在场**。`num_tokens > j` 就是"缺席"检查——短的 peer 在某层没元素，不占位。

### 4.5 那个 `+1` 是防撞车的核心

8 个 block（同一专家、不同 peer）在并发跑，都写进 `[expert_base, expert_base+total)` 这个区间。

- **去掉 `+1`**：`dst = base + min(cnt[r], j)` → 所有 peer 的 `j=0` 都算到 `base+0`，**8 个 block 直接对撞**。
- **保留 `+1`**：`j=0` 时 `dst = base + r`，天然错开成 `base+0 … base+7`。

所以 `schedule_kernel` 里**一个 atomic 都没有**，全靠这个公式保证互斥。

### 4.6 三条可验证的性质

用 `count` 那步的计数直接验证（脚本 `30_mok_schedule_pipeline_sim.py`）：

```text
cnt = [3, 2, 4]  →  dst 取值 = [0, 1, 2, 3, 4, 5, 6, 7, 8]
随机 3000 组 cnt（1~6 个 peer，计数 0~5）：全部通过
```

1. **连续**：`dst` 恰好填满 `[base, base + Σcnt)`，一空不空、一重不重。
2. **双射**：`(peer, j) ↔ dst` 一一对应 → 并发写不撞车。证明：`j` 是 `[0,cnt)` 上的置换（前缀和保证），`dst(r,j)` 对 `(r,j)` 单射（层号 + 层内 rank 号唯一确定），复合即双射。
3. **padding 是显式的洞**：`Σcnt` 通常不是 256 倍数，`dst` 只覆盖真实槽位；`[base+Σcnt, base+len)` 没人写，保持 `fill_(-1)`。

> **注意**：`dst` 是**全局行号**，不是物理 buffer 行号。落到 `macrobatch_size` 行的环形 buffer 时要减偏移：`buffer_row = dst - macrobatch_idx * macrobatch_size`（`grouped_gemm.cuh:65,106`）。

### 4.7 `j` ≠ `peer_token_idx`（最常见的混淆）

| | 含义 | 范围 | 谁算的 | 是否写进表 |
|---|---|---|---|---|
| `j` | 压缩名次：该 peer 发往该专家的第几个 | `[0, cnt[e][r])` | 前缀和 | **否**，算完即弃 |
| `peer_token_idx` | 绝对编号 `token×topk + k` | `[0, T×topk)` | 循环变量直接来的 | **是** |

```text
cnt[e0] = [3, 2, 4]，看 rank2 那列：

peer_token_idx   token   k     j    dst
    4800           600   0     0     2
    4812           601   4     1     5
    4820           602   4     2     7
    4828           603   4     3     8
```

`peer_token_idx` 不连续（跳过没选 e0 的槽位），`j` 是连续的 0,1,2,3。

**为什么必须存 `peer_token_idx` 而不是 `j`**：消费端要算源地址

```cpp
// dispatch_combine.cuh:76
&peer_buf[peer_rank][(size_t)(peer_token_idx / row_divisor) * cols + ...]
//                              ↑ row_divisor = topk  (forward.cuh:107)
//                  = 4812 / 8 = 601 → 取 rank2 的第 601 行
```

存 `j=1` 就会去取 rank2 的**第 1 行**，那是完全错的 token。**`j` 只负责错开位置，`peer_token_idx` 才是真正的地址。**

`peer_token_idx` 里含了 `k`（不拆开存，因为 router weight 要用）。dispatch 除以 `topk` 折回 token 行；combine 不除，因为目标 buffer 按 `(token,k)` 排（`dispatch_combine.cuh:237`）。

### 4.8 `j` 的分配顺序：确定，但不是升序

`j` 由线程号前缀和决定，线程 `t` 先拿完自己负责的 56 个（`t, t+1024, t+2048, …`），线程 `t+1` 才开始。所以 `j` 的顺序是 **`(idx % 1024, idx / 1024)` 字典序**。

反例：线程 1 在第 1 轮迭代命中 `idx=1025`，线程 2 在第 0 轮命中 `idx=2` —— **`1025` 会拿到更小的 `j`，排在 `2` 前面**。

这不影响正确性（dispatch 逐行 `tma::load_async`，行间无顺序依赖；combine 用同一张表对称写回），但**不要说"按 peer_token_idx 升序"**。准确表述是：**顺序确定、但非升序**。

### 4.9 全局一致：没有通信，但结果逐位相同

8 个 rank **跑同一份代码、读同一份被 all_gather 全面复制过的路由表、用同一套整数公式**。没有原子序依赖、没有浮点、没有时钟依赖 → 结果**必然逐位相同**。

这是整个设计的支点：megakernel 里 dispatch 靠这张表"不看邻居也知道该往哪发"，combine 靠它反向写回。**一个 barrier 换掉了后面所有元数据同步。**

注意是**同一个函数**，不是"同一张表"——每个 rank 的 `i=0` 指向完全不同的数据（rank0 的 `row 0` 是专家 0 的第一行，rank3 的 `row 0` 是专家 144 的第一行）。

### 4.10 溢出检查与代价

```cpp
// scheduler.cuh:86
if (G.num_tokens[{0}] > G.schedule_peer_rank.cols()) asm volatile("{trap;}");
```

`{trap}` 不是 `assert`，是让整个 kernel **硬件级中止**——溢出后再写会越界污染别的 tensor，静默出错比崩掉更糟。

容量 `capacity = T·topk·max(2, ceil(EP·0.5))`（`functional.py:198,209`）。EP=8 → 4 倍余量（229,376 vs 57,344，利用率 25%），为专家倾斜准备的。

代价是**读放大**：

| | 值 |
|---|---|
| 每个 block 扫描 | 57,344 元素 = 229 KB（一个 peer 的整块） |
| 实际命中 | 平均 57,344/384 ≈ **149**（命中率 0.26%） |
| 总扫描量 | 384 × 229 KB ≈ **88 MB** |
| 路由表本身 | 1.8 MB（**全在 L2**） |
| 全局原子 | **0** |

88 MB 全是 L2 命中流量（表 1.8 MB ≪ L2），带宽上几十微秒。**用读放大换绝对无锁 + 绝对确定性。**

---

## 5. barrier_all：唯一那一次跨卡同步

```cpp
// utils.cuh:253-257
static constexpr int NUM_BLOCKS = 1;
static constexpr int NUM_THREADS = 1;
static constexpr int DYNAMIC_SHARED_MEMORY = 0;
```

**1 个 block、1 个线程。** 全部内容：

```cpp
// utils.cuh:268-279
const uint32_t target = atomicAdd(G.target_ptr, G.ep_size) + G.ep_size;
asm volatile("{multimem.red.release.sys.global.add.u32 [%0], 1;}" :: "l"(G.multicast_ptr) : "memory");
asm volatile("{fence.proxy.alias;}" ::: "memory");
uint32_t value;
do { asm volatile("{ld.relaxed.sys.global.u32 %0, [%1];}" : "=r"(value) : "l"(G.local_ptr) : "memory"); }
while (value < target);
asm volatile("{fence.acquire.sys;}" ::: "memory");
```

逐句：

1. **`target` 是递增的轮次号**。`barrier_target` 是**每 rank 本地**的 `torch.zeros(1, int32)`（`functional.py:247`），每次 barrier `+= ep_size` → 序列 `ep, 2ep, 3ep, …`。单调递增、永不复用 → 天然免疫 ABA，且 barrier buffer 永远不用重置。代价：int32 在 EP=8 下撑 2³¹/8 ≈ 2.68 亿轮。
2. **`multimem.red`** 一条指令完成"通知所有 rank"。multimem 地址映射到 N 份同 offset 副本，`.red` 是无返回值的归约，**硬件 fabric 把 +1 同时施加到全部 8 份上**——不是 8 条消息，是一发多落。`.release.sys` 保证这条之前的写（`all_gather` 写进 symmetric buffer 的路由表）在别人看到计数后可见；`.sys` 的作用域必须跨 GPU。
3. **`fence.proxy.alias`**：写走 multicast 别名、读走 unicast 别名，两个别名可能被硬件当成不同 proxy 缓存，必须显式刷。
4. **轮询**本地单播别名，`ld.relaxed.sys`（顺序由 fence 保证）。一轮后每份副本 = `n × ep_size` = target。
5. **`fence.acquire.sys`**：把"别人写的、我还没看见的"拉进可见域。

**为什么一条 `multimem.red` 就够**：symmetric memory + NVLS multicast object 让"到达"变成一次归约广播，而不是 `ep_size` 次点对点 ack。所以没有 `dist.barrier()`、没有 CPU 参与、没有 host 同步。

**顺序保证的真正来源**：`all_gather` kernel 和 `barrier` kernel 在**同一条 stream** 上，kernel 边界已给出全局完成序。barrier 只负责两件跨 GPU 的事：时间对齐 + acquire 后的数据可见性。

---

## 6. 完整数据流：行语义变化三次

这是最容易绕晕的地方。**同一个 57344 行，在不同 buffer 里指的是不同东西。**

```text
阶段      buffer                    行轴语义                 行数
─────────────────────────────────────────────────────────────────────────
输入      x                       token（发送方视角）        7168
①         x_buffer[peer]          token（对称 buffer）       7168
②         x_fp8_routed            ★ 槽位（本 rank 要算的）   57344
③         gate/up_routed          槽位                       57344
④         hidden_fp8_routed       ★ 槽位（第二个 A）         57344
⑤         y_routed                槽位                       57344
⑥         combine_buffer          ★ (token,k) = token×topk    57344  ← 数值巧合
输出      output                  token                      7168
```

关键跳跃：

- **x_buffer → x_fp8_routed**：`token` → `槽位`。dispatch 按表 gather，一个 token 可能被 DMA 多次（topk 中几个落本 rank）。
- **y_routed → combine_buffer**：`槽位` → `(token,k)`。combine 用 `peer_token_idx` 直接作目标下标（不除 topk），因为接收 buffer 按 `(token,k)` 排，每个 `(token,k)` 只有一个来源 rank，不会撞。
- **combine_buffer → output**：`(token,k)` → `token`。由 `fwd_epilogue` 做加权求和。

### 6.1 一个具体的行，走完全程

```text
schedule 写：
    schedule_peer_rank[7]      = 2
    schedule_peer_token_idx[7] = 4812

dispatch 读（行 7）：
    peer_rank = 2, peer_token_idx = 4812
    → 从 rank2 的 x_buffer 取第 4812/8 = 601 行
    → 量化后写进本 rank 的 x_fp8_routed 第 7 行

gate GEMM 读：
    A 第 7 行 × e0 的 w_routed_gate（因 row 7 ∈ [0,1280) 属于 e0）
    → gate_routed 第 7 行

… SwiGLU、down GEMM 都在同一个行号 7 上做 …

combine 写：
    → 写到 rank2 的 combine_buffer 第 4812 行 = (token 601, k=4)
```

### 6.2 megakernel 内部：五类任务，一条流水

```text
通信 SM (40)                          计算 SM（CLC 抢任务）
──────────────                        ──────────────────────────────
                                       ┌─ shared gate   x × w_gate  ┐
dispatch                               ├─ shared up     x × w_up    ┤ bf16
 按表 TMA gather                        ├─ shared swiglu             │
 128行 × 512列 / 块                     └─ shared down   h × w_down  ┘
   ↓ x_routed_ready (14 个 minibatch)
                                       ┌─ routed gate   x_fp8 × w_gate ┐
                                       ├─ routed up     x_fp8 × w_up   ┤ MXFP8
                                       ├─ routed swiglu + 量化          │
                                       └─ routed down   h_fp8 × w_down ┘
   ↑ y_routed_ready (14 个 minibatch)          │
combine                                       ▼
 按表反向 TMA store                    y_routed
   → combine_buffer
```

**shared 和 routed 两分支完全独立**：shared 不经 dispatch/combine，A 直接用 `x_shared`（就是 `x` 本身），也不需要量化。它给每个 token 一个"必过"的通道。

### 6.3 阶段间没有全局 barrier，只有计数器

| 跳 | 计数器 | 粒度 | 谁 arrive / 谁 wait |
|---|---|---|---|
| dispatch → gate/up | `x_routed_ready` | minibatch (4096 行) | dispatch 每块 +1；GEMM 等 448 次 |
| gate/up → swiglu | `gate_up_tile_ready` | 256 行 tile | GEMM epilogue；swiglu |
| swiglu → down | `hidden_row_block_ready` | 256 行 | swiglu；down GEMM |
| down → combine | `y_routed_ready` | minibatch | down GEMM；combine |
| combine → 下一窗口的 down | `y_routed_done` | 256 行 | combine；防止 dispatch 覆写还在被读的行 |

`required_count` 的算法：

```cpp
// grouped_gemm.cuh:129-131
const int required_count = ((minibatch_rows + DISPATCH_Mb - 1) / DISPATCH_Mb)
                         * ((a_gmem.cols() + DISPATCH_Nb - 1) / DISPATCH_Nb);
```

minibatch=4096、`DISPATCH_Mb=128` → 32 个 chunk；H=7168、`DISPATCH_Nb=512` → 14 列块；**32×14 = 448 次 arrive 才放行**。

**这就是 overlay 的基础**：dispatch 完成前 4096 行，GEMM 就能开工。

### 6.4 收尾：epilogue 为什么必须存在

`combine_buffer` 里是 K 份独立结果，需要加权求和：

```text
output[t] = y_shared[t] + Σ_{k=0..7} weight[t][k] · combine_buffer[t·8 + k]
```

```cpp
// utils.cuh:70-77
if (tid == 0) tma::load_async(token_vecs[...], g.y_shared, {token_idx, col_block_idx}, ...);
else if (tid < num_tokens_per_stage)
    tma::load_async(token_vecs[...], g.combine_buffer, {token_idx * topk + tid - 1, col_block_idx}, ...);
```

`tid==0` 取 shared 那份，`tid=1..topk` 取 routed 的 K 份，然后 `group::mul` + `group::add` 累加（`utils.cuh:85-90`）。**它同时合并了 shared 和 routed**，这是整层 MoE 的最后一次加法。

---

## 7. CLC 与 grid 上界（补 29 笔记的数字）

CLC 的机制见 29 笔记 §CLC。这里只补一个和本文强相关的点：**grid 是按 capacity 算的，不是按任务数。**

```cpp
// types.cuh:137-139
const int num_minibatches = (schedule_peer_rank.cols() + minibatch_size - 1) / minibatch_size;
```

`schedule_peer_rank.cols()` 是 **capacity = 229,376**，而运行时真值 `num_tokens = 57,344`。Kimi 形状（H=7168, I=2048, comm=40）算一下：

```text
shared_tasks    = 2×224 + 150 + 784            = 1382
minibatch_tasks = 2×128 + 86 + 448             = 790

grid（按 capacity）: num_minibatches = 56 → 45,642 clusters
真实任务（按 num_tokens）: num_minibatches = 14 → 12,462 clusters
被 drain 掉的:                                    33,180 clusters（72.7%）
```

```cpp
// forward.cuh:20-27（运行时才知道真值）
const int num_tokens = g.num_tokens[{0}];
const int true_num_clusters = comm_clusters + shared_tasks + true_num_global_minibatches * minibatch_tasks;
if (cluster_idx >= true_num_clusters) return;
```

所以 `drain`（`forward.cuh:271-290`）不是边缘情况，而是**常态**：用一个循环把剩余队列全部 `try_cancel` 掉，让它们一个都不上 SM。`CLC_DRAIN_PIPE_DEPTH = 8`（`types.cuh:37`）是为了最多并行取消 8 个、加快清空。

**grid 是"上界"而不是"任务数"**——这是 29 笔记里那句话的具体数字含义。

---

## 8. 常见误解清单

| 误解 | 纠正 | 证据 |
|---|---|---|
| "count 数的是 token" | 数的是**槽位**（含 topk 展开） | `scheduler.cuh:56` 的 `+1` 打在 `(peer,token,k)` 上 |
| "pad 会写零填充行" | 不写。填充是 `fill_(-1)` 留下的，pad 只宣布区间 | `scheduler.cuh:162` + `:140-141` |
| "专家内部按 peer 分块排列" | **按 peer 轮转交错**（层 = j，层内按 peer 号） | `:139` 那个 `+1` |
| "`+1` 是纠偏" | 是**主动交错**，用于并发写不撞车 | 去掉它 8 个 block 全撞 `base+0` |
| "`j` 就是 peer_token_idx" | `j` 是压缩名次（算完即弃）；`peer_token_idx` 是绝对地址 | `:140-141` 存的是后者 |
| "同一 peer 内按 token 升序" | 按 `(idx%1024, idx/1024)` 字典序，**确定但非升序** | `:129` 的前缀和 + `:132` 的循环 |
| "`dst` 就是 buffer 行号" | 是**全局**行号；buffer 行 = `dst - macrobatch_idx*macrobatch_size` | `grouped_gemm.cuh:65,106` |
| "schedule 需要跨 rank 通信" | 零通信。8 rank 跑同一函数，结果逐位相同 | 纯整数、无原子、无浮点 |
| "schedule 有原子操作" | **零原子**，全靠交错公式互斥 | `scheduler.cuh:131-146` |
| "grid 就是任务数" | grid 按 **capacity** 算，是上界，超出部分 drain | 72.7% 被 drain（§7） |
| "combine_buffer 和 y_routed 行语义一样" | 前者 `(token,k)`，后者槽位；数值都是 57344 只是巧合 | `types.cuh:109` |

---

## 9. 一页速记

```text
输入：top_experts[7168, 8] int64  (每 rank 一份)
      ↓ all_gather（TMA multicast，1.8 MB int32）
      topk_all[8, 7168, 8]        ← 8 个 rank 各持全量副本
      ↓ barrier_all（1 blk × 1 thr，multimem.red + 轮询）
      ↓ count（448 blk × 1024，smem 直方图，2 万次 L2 原子）
      cnt[e][r]  int32[48×8]      ← 每 (本地专家, peer) 多少槽位
      ↓ pad（48 blk × 1 thr）
      len[e]  int32[48]           ← 向上取整到 256
      num_tokens int32[1]         ← 三处消费：macrobatch 数 / drain 上界 / 溢出检查
      ↓ schedule（384 blk × 1024，warp 前缀和，零原子）
      rank[i] / tok[i]            ← i 即 GEMM 行号，tok[i] = token×topk + k

三条不变量：
  ① cnt 全局合计 = 8 × T × topk（每个槽位只被拥有它的 rank 数一次）
  ② dst 在 [base, base+Σcnt) 上连续且双射（3000 组随机验证）
  ③ 8 个 rank 的表逐位相同（同一函数、同一输入、纯整数）

行语义三次变化：
  token(x_buffer) → 槽位(x_fp8 … y_routed) → (token,k)(combine_buffer) → token(output)

消费端只有一行：
  tma::load_async(&peer_buf[ rank[i] ][ tok[i]/topk * cols + ... ])
```

### 源码阅读顺序

```text
1. csrc/scheduler.cuh:34-63     count（先看 45-48 的 smem 清零，注意同名陷阱）
2. csrc/scheduler.cuh:66-77     pad（10 行，看 launch 的 <<<48,1>>>）
3. csrc/scheduler.cuh:96-98     expert_base 前缀和（O(E²) 的取舍）
4. csrc/scheduler.cuh:106-129   两趟扫描的第一趟 + j 的前缀和
5. csrc/scheduler.cuh:134-141   dst 公式与那个 +1（本文核心）
6. csrc/utils.cuh:268-279       barrier_all（5 条指令，逐条读 fence）
7. csrc/megakernel/dispatch_combine.cuh:44-80  表的消费端
8. csrc/megakernel/grouped_gemm.cuh:96-110     段长 → tile_coord
```
