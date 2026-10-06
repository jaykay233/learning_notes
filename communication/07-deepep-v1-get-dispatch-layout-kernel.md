# DeepEP V1 Normal：机内 `get_dispatch_layout` kernel 逐行拆解

> 承接 [04-deepep-v1-normal-dataflow.md](./04-deepep-v1-normal-dataflow.md) 第 3 节。04 只讲了 `get_dispatch_layout` 的输入输出语义，本文进到 CUDA kernel 内部，讲清楚 block 怎么分工、shared memory 计数矩阵怎么用、为什么不需要原子操作，以及单机（≤ 8 卡）时哪些代码实际不会生效。
>
> 代码对照：本地 DeepEP checkout `/Users/saboxu/Downloads/communication/DeepEP`。kernel 在 `csrc/kernels/legacy/layout.cu`，host 端入口在 `csrc/legacy/buffer.hpp` 的 `Buffer::get_dispatch_layout`（约 L337–L404），Python 封装在 `deep_ep/buffers/legacy.py`（约 L293）。

```text
本次讲解位置
章节：communication / DeepEP V1
小节：Normal 模式第一步 get_dispatch_layout 的 kernel 实现（机内）
知识点：block 两类分工、per-thread shared memory 计数 + 按列归约、token-rank 去重、单机下 RDMA 分支不生效
上次：DeepEP V1 normal 与 low-latency 端到端数据流对照
下次：机内 intranode_dispatch 的 notify 阶段如何用 num_tokens_per_rank 算 rank_prefix_matrix
PTX：不适用（kernel 只用到 shared memory、__syncthreads 和普通 global 读写）
```

## 为什么现在讲这个

04 把 `num_tokens_per_rank`、`num_tokens_per_expert`、`is_token_in_rank` 当作黑盒输出。接下来读 `intranode_dispatch` 时，这三个量会直接决定接收缓冲区按多少行分配、每个 token 发给哪些 GPU。不清楚它们怎么算出来的，读 dispatch 时会遇到三个具体问题：

1. **以为 `num_tokens_per_rank[r]` 等于 rank `r` 上各 expert 计数之和。** 实际上它按 token-rank 去重，前者一般更小。用 expert 计数之和去估接收行数，会把缓冲区估大，对不上 `recv_x` 的行数。
2. **把 `sm_id` 当成物理 SM 编号。** 它就是 `blockIdx.x`。grid 大小由 expert 数和 rank 数算出来，和 GPU 的 SM 数无关。
3. **看不懂 `thread_id` 在同一个 kernel 里换了两次含义。** 计数阶段它是行号，归约阶段它变成列号。这是 DeepEP 里反复出现的「先私有计数、再按列归约」写法，读懂一次后面 dispatch 里的计数代码都是同一个套路。

## 0. 心智模型

```text
输入 topk_idx [N, K]：每个 token 去哪些全局 expert（-1 表示该槽位为空）

grid = ceil(E/4) 个 "expert block"  +  ceil(P/8) 个 "rank block"

expert block b：只盯 4 个 expert [4b, 4b+4)
  256 个线程把 N 个 token 分着扫一遍
  -> 每线程在 shared memory 自己那一行计数（无竞争、无原子操作）
  -> __syncthreads
  -> 线程 0..3 各加一列 -> num_tokens_per_expert

rank block b'：只盯 8 个 rank [8b', 8b'+8)
  256 个线程同样分着扫 token
  -> 每个 token 先在寄存器里标记命中了哪些 rank（同 rank 多个 expert 只算一次）
  -> 直接写 is_token_in_rank[i, r]
  -> 每线程在 shared memory 自己那一行累加 token-rank 计数
  -> __syncthreads
  -> 线程 0..7 各加一列 -> num_tokens_per_rank
  -> （仅多机）线程 0 加一列 -> num_tokens_per_rdma_rank
```

一句话：**每个 block 都完整扫一遍 `topk_idx`，但只统计自己负责的那一小段 expert 或 rank。** 用重复读取换掉了原子操作和跨 block 同步。

## 1. 术语与维度

| 符号 / 名字 | 含义 | 本文默认值 |
|---|---|---|
| `N` = `num_tokens` | 当前 rank 本地 token 数 | — |
| `K` = `num_topk` | 每 token 的路由槽位数 | — |
| `E` = `num_experts` | 全局 expert 数 | — |
| `P` = `num_ranks` | EP group 的 rank 数；单机 ≤ 8 | — |
| `kNumThreads` | 每个 block 的线程数 | 256 |
| `kNumExpertsPerSM` | 每个 expert block 负责的 expert 数（名字里的 SM 实指 block） | 4 |
| `kNumRanksPerSM` | 每个 rank block 负责的 rank 数 | 8 |
| `LEGACY_NUM_MAX_NVL_PEERS` | 一个节点内 NVLink 互联的 GPU 数 | 8 |
| `kNumRDMARanksPerSM` | 每个 rank block 负责的节点数 = 8 / 8 | 1 |
| `num_expert_per_rank` | 每个 rank 持有的 expert 数 = `E / P`（整数除法） | — |

不变量：

- 三个模板参数都是 host 端写死的 `constexpr`，用来定 shared memory 数组和寄存器数组的大小，所以必须是编译期常量。
- `kNumExpertsPerSM <= kNumThreads`、`kNumRanksPerSM <= kNumThreads`：保证归约时每一列都有一个线程负责，由 `EP_STATIC_ASSERT` 在编译期检查。
- `kNumRanksPerSM % LEGACY_NUM_MAX_NVL_PEERS == 0`：保证一个 rank block 覆盖整数个节点，节点计数不会被两个 block 拆开。

## 2. 完整代码

以下是 `csrc/kernels/legacy/layout.cu` 的完整内容，未做删减：

```cpp
#include <deep_ep/common/exception.cuh>

#include "compiled.cuh"
#include "launch.cuh"

namespace deep_ep::legacy {

namespace layout {

template <int kNumThreads, int kNumExpertsPerSM, int kNumRanksPerSM>
__global__ void get_dispatch_layout(const topk_idx_t* topk_idx,
                                    int* num_tokens_per_rank,
                                    int* num_tokens_per_rdma_rank,
                                    int* num_tokens_per_expert,
                                    bool* is_token_in_rank,
                                    int num_tokens,
                                    int num_topk,
                                    int num_ranks,
                                    int num_experts) {
    auto sm_id = static_cast<int>(blockIdx.x);
    auto thread_id = static_cast<int>(threadIdx.x);

    // Count expert statistics
    __shared__ int num_tokens_per_expert_per_thread[kNumThreads][kNumExpertsPerSM];
    int expert_begin_idx = sm_id * kNumExpertsPerSM, expert_end_idx = min(expert_begin_idx + kNumExpertsPerSM, num_experts);
    if (expert_begin_idx < expert_end_idx) {
        // Per-thread count
        #pragma unroll
        for (int i = 0; i < kNumExpertsPerSM; ++i)
            num_tokens_per_expert_per_thread[thread_id][i] = 0;
        #pragma unroll
        for (int i = thread_id; i < num_tokens; i += kNumThreads) {
            auto shifted_topk_idx = topk_idx + i * num_topk;
            #pragma unroll
            for (int j = 0, expert_idx; j < num_topk; ++j) {
                expert_idx = static_cast<int>(shifted_topk_idx[j]);
                if (expert_begin_idx <= expert_idx and expert_idx < expert_end_idx)
                    ++num_tokens_per_expert_per_thread[thread_id][expert_idx - expert_begin_idx];
            }
        }
        __syncthreads();

        // Sum up
        EP_STATIC_ASSERT(kNumExpertsPerSM <= kNumThreads, "Too many experts per SM");
        if (expert_begin_idx + thread_id < expert_end_idx) {
            int sum = 0;
            #pragma unroll
            for (int i = 0; i < kNumThreads; ++i)
                sum += num_tokens_per_expert_per_thread[i][thread_id];
            num_tokens_per_expert[expert_begin_idx + thread_id] = sum;
        }
        return;
    }

    if (num_tokens_per_rdma_rank != nullptr)
        EP_DEVICE_ASSERT(num_ranks % LEGACY_NUM_MAX_NVL_PEERS == 0 and num_ranks > LEGACY_NUM_MAX_NVL_PEERS);

    // Count rank statistics
    constexpr int kNumRDMARanksPerSM = kNumRanksPerSM / LEGACY_NUM_MAX_NVL_PEERS;
    __shared__ int num_tokens_per_rank_per_thread[kNumThreads][kNumRanksPerSM];
    __shared__ int num_tokens_per_rdma_rank_per_thread[kNumThreads][kNumRDMARanksPerSM];
    auto sm_begin = (num_experts + kNumExpertsPerSM - 1) / kNumExpertsPerSM;
    int rank_begin_idx = (sm_id - sm_begin) * kNumRanksPerSM, rank_end_idx = min(rank_begin_idx + kNumRanksPerSM, num_ranks);
    int rdma_rank_begin_idx = rank_begin_idx / LEGACY_NUM_MAX_NVL_PEERS, rdma_rank_end_idx = rank_end_idx / LEGACY_NUM_MAX_NVL_PEERS;
    if (rank_begin_idx < rank_end_idx) {
        const auto num_expert_per_rank = num_experts / num_ranks;
        auto expert_begin = rank_begin_idx * num_expert_per_rank;
        auto expert_end = rank_end_idx * num_expert_per_rank;

        // Per-thread count
        #pragma unroll
        for (int i = 0; i < kNumRanksPerSM; ++i)
            num_tokens_per_rank_per_thread[thread_id][i] = 0;
        #pragma unroll
        for (int i = 0; i < kNumRDMARanksPerSM; ++i)
            num_tokens_per_rdma_rank_per_thread[thread_id][i] = 0;
        #pragma unroll
        for (int i = thread_id; i < num_tokens; i += kNumThreads) {
            auto shifted_topk_idx = topk_idx + i * num_topk;
            int is_in_rank[kNumRanksPerSM] = {0}, is_in_rdma_rank[kNumRDMARanksPerSM] = {0};
            #pragma unroll
            for (int j = 0, expert_idx, rank_idx; j < num_topk; ++j) {
                expert_idx = static_cast<int>(shifted_topk_idx[j]);
                if (expert_begin <= expert_idx and expert_idx < expert_end) {
                    // Count single rank
                    rank_idx = expert_idx / num_expert_per_rank - rank_begin_idx;
                    is_in_rank[rank_idx]++, is_in_rdma_rank[rank_idx / LEGACY_NUM_MAX_NVL_PEERS]++;
                }
            }

            auto shifted_is_token_in_rank = is_token_in_rank + i * num_ranks;
            #pragma unroll
            for (int j = 0; j + rank_begin_idx < rank_end_idx; ++j) {
                shifted_is_token_in_rank[j + rank_begin_idx] = (is_in_rank[j] > 0);
                num_tokens_per_rank_per_thread[thread_id][j] += (is_in_rank[j] > 0);
            }

            #pragma unroll
            for (int j = 0; j + rdma_rank_begin_idx < rdma_rank_end_idx; ++j)
                num_tokens_per_rdma_rank_per_thread[thread_id][j] += (is_in_rdma_rank[j] > 0);
        }
        __syncthreads();

        // Sum up
        EP_STATIC_ASSERT(kNumRanksPerSM <= kNumThreads, "Too many ranks per SM");
        if (rank_begin_idx + thread_id < rank_end_idx) {
            int sum = 0;
            #pragma unroll
            for (int i = 0; i < kNumThreads; ++i)
                sum += num_tokens_per_rank_per_thread[i][thread_id];
            num_tokens_per_rank[rank_begin_idx + thread_id] = sum;
        }

        if (num_tokens_per_rdma_rank != nullptr and rdma_rank_begin_idx + thread_id < rdma_rank_end_idx) {
            int sum = 0;
            #pragma unroll
            for (int i = 0; i < kNumThreads; ++i)
                sum += num_tokens_per_rdma_rank_per_thread[i][thread_id];
            num_tokens_per_rdma_rank[rdma_rank_begin_idx + thread_id] = sum;
        }
    }
}

void get_dispatch_layout(const topk_idx_t* topk_idx,
                         int* num_tokens_per_rank,
                         int* num_tokens_per_rdma_rank,
                         int* num_tokens_per_expert,
                         bool* is_token_in_rank,
                         int num_tokens,
                         int num_topk,
                         int num_ranks,
                         int num_experts,
                         cudaStream_t stream) {
    constexpr int kNumThreads = 256, kNumExpertsPerSM = 4, kNumRanksPerSM = 8;
    int num_sms = ((num_experts + kNumExpertsPerSM - 1) / kNumExpertsPerSM) + (num_ranks + kNumRanksPerSM - 1) / kNumRanksPerSM;
    EP_STATIC_ASSERT(kNumRanksPerSM % LEGACY_NUM_MAX_NVL_PEERS == 0, "Invalid number of ranks per SM");

    SETUP_LAUNCH_CONFIG(num_sms, kNumThreads, stream);
    LAUNCH_KERNEL(&cfg,
                  (get_dispatch_layout<kNumThreads, kNumExpertsPerSM, kNumRanksPerSM>),
                  topk_idx,
                  num_tokens_per_rank,
                  num_tokens_per_rdma_rank,
                  num_tokens_per_expert,
                  is_token_in_rank,
                  num_tokens,
                  num_topk,
                  num_ranks,
                  num_experts);
}

}  // namespace layout

}  // namespace deep_ep::legacy
```

`SETUP_LAUNCH_CONFIG` / `LAUNCH_KERNEL` 定义在 `csrc/kernels/legacy/launch.cuh`。SM90 路径下展开为：

```cpp
cudaLaunchConfig_t cfg = {(num_sms), (num_threads), 0, stream, nullptr, 0};
cudaLaunchAttribute attr[2];
attr[0].id = cudaLaunchAttributeCooperative;
attr[0].val.cooperative = 1;
attr[1].id = cudaLaunchAttributeClusterDimension;
attr[1].val.clusterDim.x = (num_sms % 2 == 0 ? 2 : 1);
attr[1].val.clusterDim.y = 1;
attr[1].val.clusterDim.z = 1;
cfg.attrs = attr;
cfg.numAttrs = 2;
CUDA_RUNTIME_CHECK(cudaLaunchKernelEx(&cfg, kernel, args...));
```

cooperative launch 要求所有 block 能同时驻留在 GPU 上；cluster 维度 2 是 DeepEP 通用启动宏的默认设置。这两者都**不会**把 block 固定到某个物理 SM 上，这个 kernel 内部也没有用到 grid 同步或 cluster 特性。

## 3. host 端：谁调用、张量怎么分配

`csrc/legacy/buffer.hpp` 的 `Buffer::get_dispatch_layout` 做了这些事：

1. 检查 `topk_idx` 是二维且内存连续，`num_experts > 0`。**没有检查 `num_experts % num_ranks == 0`**（见第 8 节第 5 条）。
2. 让通信流 `comm_stream` 等待计算流，或等待传入的 `previous_event`。
3. 用 `torch::empty` 分配输出（**不初始化**）：
   - `num_tokens_per_rank`：`[P]` int32
   - `num_tokens_per_expert`：`[E]` int32
   - `is_token_in_rank`：`[N, P]` bool
   - `num_tokens_per_rdma_rank`：只有 `is_internode_available()`（即 `num_ranks > 8`）时才分配 `[P/8]`，否则是空的 `optional`，传给 kernel 的是 `nullptr`。
4. 在 `comm_stream` 上启动 kernel。
5. `async=True` 时返回 event 并对张量 `record_stream`；否则让计算流等通信流。

因为输出是 `torch::empty`，**kernel 必须把每个元素都写到**。第 4、5 节会看到这一点是怎么保证的：每个 expert 恰好属于一个 expert block，每个 rank 恰好属于一个 rank block，`is_token_in_rank` 的每一行由唯一一个线程写满该 block 负责的所有列。

## 4. block 分工：`sm_id` 就是 `blockIdx.x`

```text
num_sms = ceil(E / 4) + ceil(P / 8)

blockIdx.x ∈ [0, ceil(E/4))              -> expert block，负责 expert [4b, min(4b+4, E))
blockIdx.x ∈ [ceil(E/4), num_sms)        -> rank block，第 b' = blockIdx.x - ceil(E/4) 个，
                                             负责 rank [8b', min(8b'+8, P))
```

kernel 里没有显式的 `if (is_expert_block)`。分流靠的是区间是否为空：

- expert block：`expert_begin_idx = 4 * sm_id < E`，区间非空，进入第一段，最后 `return`。
- rank block：`4 * sm_id >= E`，expert 区间为空，跳过第一段；`sm_id - sm_begin` 换算出从 0 开始的 rank block 编号。

代入几组具体数：

| 场景 | E | P | expert block 数 | rank block 数 | `num_sms` |
|---|---:|---:|---:|---:|---:|
| 小例子（第 7 节） | 8 | 4 | ceil(8/4) = 2 | ceil(4/8) = 1 | 3 |
| 单机 8 卡，256 experts | 256 | 8 | 256/4 = 64 | 8/8 = 1 | 65 |
| 2 机 16 卡，256 experts | 256 | 16 | 64 | 16/8 = 2 | 66 |
| E 不是 4 的倍数 | 10 | 2 | ceil(10/4) = 3 | 1 | 4 |

单机（P ≤ 8）时 rank block **永远只有 1 个**。H800 有 132 个 SM，65 个 block 远少于 SM 数；反过来 E 很大时 block 数也可能超过 SM 数，一个 SM 上会同时跑多个 block。所以把 `sm_id` 读成 `block_id`，把 `kNumExpertsPerSM` 读成「每个 block 负责的 expert 数」。

## 5. 任务一：expert 计数（expert block）

### 5.1 shared memory 计数矩阵

```cpp
__shared__ int num_tokens_per_expert_per_thread[kNumThreads][kNumExpertsPerSM];  // [256][4]
```

- 第一维：`thread_id`，0–255。
- 第二维：**本 block 内的局部 expert 下标** 0–3，不是全局 expert 编号。换算关系 `局部下标 = expert_idx - expert_begin_idx`。
- 大小 256 × 4 × 4 B = 4 KB。

### 5.2 第一步：每个线程只写自己那一行

| 问题 | 回答 |
|---|---|
| 做什么 | 清零自己那一行；按步长 256 扫 token（线程 `t` 处理 token `t, t+256, t+512, …`）；对落在 `[expert_begin_idx, expert_end_idx)` 的 expert 给对应列加 1 |
| 谁执行 | block 内全部 256 个线程 |
| 访问什么 | 读 global 的 `topk_idx[i * K + j]`；读写 shared memory 中自己那一行 |
| 为什么不需要原子操作 | 每个线程只写第 `thread_id` 行，任意两个线程写的地址不重叠 |
| `-1` 怎么处理 | `-1` 不在任何非负区间里，`if` 不成立，自然跳过 |

### 5.3 第二步：`__syncthreads()` 后按列归约

```cpp
if (expert_begin_idx + thread_id < expert_end_idx) {
    int sum = 0;
    for (int i = 0; i < kNumThreads; ++i)
        sum += num_tokens_per_expert_per_thread[i][thread_id];
    num_tokens_per_expert[expert_begin_idx + thread_id] = sum;
}
```

这里 `thread_id` 换了含义：**线程 `k` 负责第 `k` 列，也就是局部 expert `k`**。它竖着把 256 行加起来。

```text
              列0          列1          列2          列3
行0  (线程0)    a0           b0           c0           d0
行1  (线程1)    a1           b1           c1           d1
...
行255(线程255)  a255         b255         c255         d255
                 ↓            ↓            ↓            ↓
              线程0求和     线程1求和     线程2求和     线程3求和
```

| 问题 | 回答 |
|---|---|
| 谁执行 | 只有 `thread_id < expert_end_idx - expert_begin_idx` 的线程，通常是线程 0–3；最后一个 block 可能更少 |
| 同步关系 | `__syncthreads()` 保证所有线程的计数都写完，归约线程才开始读别人的行 |
| 线程够不够 | 只需要 4 个，block 有 256 个；`EP_STATIC_ASSERT(kNumExpertsPerSM <= kNumThreads)` 在编译期保证 |
| 尾部处理 | `expert_end_idx = min(…, num_experts)` 截短区间；E 不是 4 的倍数时，多出来的列清零了但没人写也没人读 |

## 6. 任务二：rank 计数（rank block）

套路和任务一相同（私有行计数 → `__syncthreads` → 按列归约），多了两点。

### 6.1 每个 token 对同一个 rank 只计一次

```cpp
int is_in_rank[kNumRanksPerSM] = {0}, is_in_rdma_rank[kNumRDMARanksPerSM] = {0};
for (int j = 0, expert_idx, rank_idx; j < num_topk; ++j) {
    expert_idx = static_cast<int>(shifted_topk_idx[j]);
    if (expert_begin <= expert_idx and expert_idx < expert_end) {
        rank_idx = expert_idx / num_expert_per_rank - rank_begin_idx;
        is_in_rank[rank_idx]++, is_in_rdma_rank[rank_idx / LEGACY_NUM_MAX_NVL_PEERS]++;
    }
}
```

- `is_in_rank` 是**寄存器里的局部数组**，每处理一个 token 就重新清零。
- `expert_idx / num_expert_per_rank` 把全局 expert 换算成全局 rank，再减 `rank_begin_idx` 得到 block 内局部 rank 下标 0–7。
- 计数时只看 `is_in_rank[j] > 0`。一个 token 的 2 个 expert 都在 rank 3 上，`is_in_rank[3] = 2`，但 `num_tokens_per_rank[3]` 只加 1。原因：dispatch 只给 rank 3 发一份 `x`，到了 rank 3 再由 `recv_topk_idx` 指明对应哪两个 local expert。

### 6.2 `is_token_in_rank` 直接写 global，不需要归约

```cpp
auto shifted_is_token_in_rank = is_token_in_rank + i * num_ranks;
for (int j = 0; j + rank_begin_idx < rank_end_idx; ++j) {
    shifted_is_token_in_rank[j + rank_begin_idx] = (is_in_rank[j] > 0);
    num_tokens_per_rank_per_thread[thread_id][j] += (is_in_rank[j] > 0);
}
```

`is_token_in_rank` 是 `[N, P]` 行主序。token `i` 只由一个线程处理，所以这一行属于本 block 的那几列只有这个线程会写，直接写 global memory 即可，不经过 shared memory。

### 6.3 归约

- `num_tokens_per_rank`：线程 0–7 各负责一列（单机 P < 8 时只有前 P 个线程），各自加 256 行。
- `num_tokens_per_rdma_rank`：`kNumRDMARanksPerSM = 1`，只有线程 0 负责，而且只在指针非空时写回。

### 6.4 单机下哪些代码不生效

单机 P ≤ 8，`is_internode_available()` 为假，`num_tokens_per_rdma_rank == nullptr`：

| 代码 | 单机时的行为 |
|---|---|
| `if (num_tokens_per_rdma_rank != nullptr) EP_DEVICE_ASSERT(...)` | 条件为假，assert 不执行 |
| `kNumRDMARanksPerSM`、`is_in_rdma_rank`、`num_tokens_per_rdma_rank_per_thread` | 仍然计算（`rdma_rank_end_idx = P/8 = 0`，累加循环一次都不跑），结果不写回 |
| 最后一段 `num_tokens_per_rdma_rank[...] = sum` | 因指针为 `nullptr` 被跳过 |
| `num_tokens_per_rank`、`is_token_in_rank` | **正常计算，后续 intranode dispatch 依赖它们** |

`kNumRDMARanksPerSM` 的意思是一个 rank block 覆盖几个节点：每节点 8 卡，block 管 8 个 rank，正好 1 个节点。多机时它保证 `num_tokens_per_rdma_rank`（发往每个节点的去重 token 数，跨节点 RDMA 只发一份、节点内再 NVLink 转发）不会被两个 block 各算一半。

## 7. 具体执行追踪

参数：E = 8，P = 4，`num_expert_per_rank = 8 / 4 = 2`，N = 5，K = 2。

```text
topk_idx = [[0,  3],    # token 0
            [2,  3],    # token 1
            [5, -1],    # token 2，第 2 个槽位为空
            [7,  1],    # token 3
            [4,  5]]    # token 4
```

expert 到 rank 的归属：rank 0 = {0, 1}，rank 1 = {2, 3}，rank 2 = {4, 5}，rank 3 = {6, 7}。

### 7.1 block 划分

| blockIdx.x | 类型 | 区间计算 | 负责 |
|---:|---|---|---|
| 0 | expert block | `0*4=0`，`min(0+4, 8)=4` | expert 0–3 |
| 1 | expert block | `1*4=4`，`min(4+4, 8)=8` | expert 4–7 |
| 2 | rank block | `4*2=8 >= 8`，expert 区间空；`sm_begin = ceil(8/4) = 2`；`rank_begin = (2-2)*8 = 0`，`rank_end = min(0+8, 4) = 4`；`expert_begin = 0*2 = 0`，`expert_end = 4*2 = 8` | rank 0–3 |

N = 5 < 256，所以线程 `t`（t = 0–4）恰好处理 token `t`，线程 5–255 的 token 循环一次都不进入，但它们的行也清零了，归约时加进去的都是 0。

### 7.2 block 0 的 shared memory（列 = expert 0, 1, 2, 3）

| 行（线程） | token 的 topk | 落在 [0, 4) 的 expert | 列0 | 列1 | 列2 | 列3 |
|---|---|---|---:|---:|---:|---:|
| 0 | [0, 3] | 0, 3 | 1 | 0 | 0 | 1 |
| 1 | [2, 3] | 2, 3 | 0 | 0 | 1 | 1 |
| 2 | [5, -1] | 无 | 0 | 0 | 0 | 0 |
| 3 | [7, 1] | 1 | 0 | 1 | 0 | 0 |
| 4 | [4, 5] | 无 | 0 | 0 | 0 | 0 |
| 5–255 | — | — | 0 | 0 | 0 | 0 |
| **线程 0–3 按列求和** | | | **1** | **1** | **1** | **2** |

写回 `num_tokens_per_expert[0..3] = [1, 1, 1, 2]`。

### 7.3 block 1 的 shared memory（列 = expert 4, 5, 6, 7）

| 行（线程） | token 的 topk | 落在 [4, 8) 的 expert | 列0 (e4) | 列1 (e5) | 列2 (e6) | 列3 (e7) |
|---|---|---|---:|---:|---:|---:|
| 0 | [0, 3] | 无 | 0 | 0 | 0 | 0 |
| 1 | [2, 3] | 无 | 0 | 0 | 0 | 0 |
| 2 | [5, -1] | 5 | 0 | 1 | 0 | 0 |
| 3 | [7, 1] | 7 | 0 | 0 | 0 | 1 |
| 4 | [4, 5] | 4, 5 | 1 | 1 | 0 | 0 |
| **按列求和** | | | **1** | **2** | **0** | **1** |

写回 `num_tokens_per_expert[4..7] = [1, 2, 0, 1]`。

### 7.4 block 2：rank 统计

每个 token 的 `rank_idx = expert_idx / 2 - 0`：

| token | topk | 换算 rank | `is_in_rank[0..3]` | `> 0` 后 = `is_token_in_rank[i]` |
|---:|---|---|---|---|
| 0 | [0, 3] | 0/2=0, 3/2=1 | [1, 1, 0, 0] | [1, 1, 0, 0] |
| 1 | [2, 3] | 2/2=1, 3/2=1 | [0, **2**, 0, 0] | [0, **1**, 0, 0] |
| 2 | [5, -1] | 5/2=2；-1 跳过 | [0, 0, 1, 0] | [0, 0, 1, 0] |
| 3 | [7, 1] | 7/2=3, 1/2=0 | [1, 0, 0, 1] | [1, 0, 0, 1] |
| 4 | [4, 5] | 4/2=2, 5/2=2 | [0, 0, **2**, 0] | [0, 0, **1**, 0] |
| **线程 0–3 按列求和** | | | | **[2, 2, 2, 1]** |

`num_tokens_per_rank = [2, 2, 2, 1]`，`num_tokens_per_rdma_rank` 为 `None`。

### 7.5 两种计数的差别

```text
sum(num_tokens_per_expert) = 1+1+1+2+1+2+0+1 = 9   # 有效 token-expert 路由数 = 10 个槽位 - 1 个 -1
sum(num_tokens_per_rank)   = 2+2+2+1         = 7   # 去重后的 token-rank 对数
差值 2 = token 1 在 rank 1 上命中 2 个 expert（多 1） + token 4 在 rank 2 上命中 2 个 expert（多 1）
```

物理含义：本 rank 在 dispatch 时一共只要发 7 行 token；目标 rank 上的 expert 计算一共要处理 9 个 token-expert 对。

## 8. 常见错误与症状

| 错误理解 | 正确理解 | 可观察到的症状 |
|---|---|---|
| `sm_id` 是物理 SM 编号，block 数等于 GPU SM 数 | `sm_id = blockIdx.x`，block 数 = ceil(E/4) + ceil(P/8) | 估算并行度或用 profiler 对照时，block 数和预期对不上（例如 256 experts 单机只有 65 个 block） |
| shared 数组第二维是全局 expert 编号 | 是局部下标 `expert_idx - expert_begin_idx`，范围 0–3 | 自己改写时直接用 `expert_idx` 做下标，会越过 `[256][4]` 的边界，计数错乱或踩坏 shared memory |
| `num_tokens_per_rank[r]` = rank r 上各 expert 计数之和 | 按 token-rank 去重，通常更小 | 7.5 节的例子里用 9 估接收行数，实际 `recv_x` 只有 7 行 |
| 归约前少了 `__syncthreads()` 也没关系 | 归约线程要读其它线程的行，必须等所有计数写完 | 结果不稳定，同一输入多次运行计数不同，且通常偏小 |
| `num_experts` 不能被 `num_ranks` 整除也行 | `num_expert_per_rank = E / P` 是整数除法，`expert_end = rank_end * (E/P) < E`，尾部 expert 不归属任何 rank；host 端没有检查这一点 | `num_tokens_per_expert` 正常，但路由到尾部 expert 的 token 在 `is_token_in_rank` 里全为 false，后续 dispatch 不会发送这些 token |
| 单机时 RDMA 统计也会被用到 | 单机 `num_tokens_per_rdma_rank` 是 `nullptr`，Python 端拿到 `None` | 单机代码里对它调用 `.sum()` 等操作会报 `NoneType` 错误 |

## 9. 验证：在 CPU 上逐 block 重放这个 kernel

当前笔记的机器是 macOS，没有 NVIDIA GPU，无法运行 DeepEP 本身。下面的脚本只依赖 PyTorch（CPU 即可），按 kernel 的 block、线程划分逐步重放，再和直接用 PyTorch 算子写的参考实现对比。它能验证上面对算法和分工的理解，**不能**代替在 GPU 上对真实 kernel 的运行验证。

保存为 `dispatch_layout_ref.py`：

```python
import math

import torch

K_NUM_THREADS = 256
K_NUM_EXPERTS_PER_SM = 4
K_NUM_RANKS_PER_SM = 8
NUM_MAX_NVL_PEERS = 8


def simulate_kernel(topk_idx: torch.Tensor, num_ranks: int, num_experts: int):
    """Replay layout.cu::get_dispatch_layout block by block, thread by thread."""
    num_tokens, num_topk = topk_idx.shape
    topk = topk_idx.tolist()
    num_expert_blocks = math.ceil(num_experts / K_NUM_EXPERTS_PER_SM)
    num_rank_blocks = math.ceil(num_ranks / K_NUM_RANKS_PER_SM)
    num_sms = num_expert_blocks + num_rank_blocks
    has_rdma = num_ranks > NUM_MAX_NVL_PEERS

    num_tokens_per_expert = torch.full((num_experts,), -1, dtype=torch.int32)
    num_tokens_per_rank = torch.full((num_ranks,), -1, dtype=torch.int32)
    num_tokens_per_rdma_rank = torch.full((num_ranks // NUM_MAX_NVL_PEERS,), -1, dtype=torch.int32) if has_rdma else None
    is_token_in_rank = torch.zeros((num_tokens, num_ranks), dtype=torch.bool)

    for sm_id in range(num_sms):
        # Task 1: expert statistics
        expert_begin = sm_id * K_NUM_EXPERTS_PER_SM
        expert_end = min(expert_begin + K_NUM_EXPERTS_PER_SM, num_experts)
        if expert_begin < expert_end:
            per_thread = [[0] * K_NUM_EXPERTS_PER_SM for _ in range(K_NUM_THREADS)]
            for thread_id in range(K_NUM_THREADS):
                for i in range(thread_id, num_tokens, K_NUM_THREADS):
                    for expert_idx in topk[i]:
                        if expert_begin <= expert_idx < expert_end:
                            per_thread[thread_id][expert_idx - expert_begin] += 1
            # __syncthreads()
            for thread_id in range(K_NUM_THREADS):
                if expert_begin + thread_id < expert_end:
                    num_tokens_per_expert[expert_begin + thread_id] = sum(per_thread[i][thread_id] for i in range(K_NUM_THREADS))
            continue

        # Task 2: rank statistics
        rdma_per_sm = K_NUM_RANKS_PER_SM // NUM_MAX_NVL_PEERS
        rank_begin = (sm_id - num_expert_blocks) * K_NUM_RANKS_PER_SM
        rank_end = min(rank_begin + K_NUM_RANKS_PER_SM, num_ranks)
        rdma_begin, rdma_end = rank_begin // NUM_MAX_NVL_PEERS, rank_end // NUM_MAX_NVL_PEERS
        if rank_begin >= rank_end:
            continue
        experts_per_rank = num_experts // num_ranks
        expert_begin, expert_end = rank_begin * experts_per_rank, rank_end * experts_per_rank
        per_thread_rank = [[0] * K_NUM_RANKS_PER_SM for _ in range(K_NUM_THREADS)]
        per_thread_rdma = [[0] * rdma_per_sm for _ in range(K_NUM_THREADS)]
        for thread_id in range(K_NUM_THREADS):
            for i in range(thread_id, num_tokens, K_NUM_THREADS):
                is_in_rank = [0] * K_NUM_RANKS_PER_SM
                is_in_rdma_rank = [0] * rdma_per_sm
                for expert_idx in topk[i]:
                    if expert_begin <= expert_idx < expert_end:
                        local_rank = expert_idx // experts_per_rank - rank_begin
                        is_in_rank[local_rank] += 1
                        is_in_rdma_rank[local_rank // NUM_MAX_NVL_PEERS] += 1
                for j in range(rank_end - rank_begin):
                    is_token_in_rank[i, rank_begin + j] = is_in_rank[j] > 0
                    per_thread_rank[thread_id][j] += int(is_in_rank[j] > 0)
                for j in range(rdma_end - rdma_begin):
                    per_thread_rdma[thread_id][j] += int(is_in_rdma_rank[j] > 0)
        # __syncthreads()
        for thread_id in range(K_NUM_THREADS):
            if rank_begin + thread_id < rank_end:
                num_tokens_per_rank[rank_begin + thread_id] = sum(per_thread_rank[i][thread_id] for i in range(K_NUM_THREADS))
            if has_rdma and rdma_begin + thread_id < rdma_end:
                num_tokens_per_rdma_rank[rdma_begin + thread_id] = sum(per_thread_rdma[i][thread_id] for i in range(K_NUM_THREADS))

    return num_sms, num_tokens_per_rank, num_tokens_per_rdma_rank, num_tokens_per_expert, is_token_in_rank


def reference(topk_idx: torch.Tensor, num_ranks: int, num_experts: int):
    """Same semantics written directly with PyTorch ops."""
    num_tokens = topk_idx.size(0)
    valid = topk_idx >= 0
    num_tokens_per_expert = torch.bincount(topk_idx[valid], minlength=num_experts).to(torch.int32)
    rank_idx = torch.where(valid, topk_idx // (num_experts // num_ranks), torch.full_like(topk_idx, num_ranks))
    hit = torch.zeros((num_tokens, num_ranks + 1), dtype=torch.bool)
    hit.scatter_(1, rank_idx, True)
    is_token_in_rank = hit[:, :num_ranks]
    num_tokens_per_rank = is_token_in_rank.sum(dim=0).to(torch.int32)
    return num_tokens_per_rank, num_tokens_per_expert, is_token_in_rank


def check(name: str, topk_idx: torch.Tensor, num_ranks: int, num_experts: int, verbose: bool) -> None:
    num_sms, per_rank, per_rdma, per_expert, in_rank = simulate_kernel(topk_idx, num_ranks, num_experts)
    ref_rank, ref_expert, ref_in_rank = reference(topk_idx, num_ranks, num_experts)
    assert torch.equal(per_rank, ref_rank)
    assert torch.equal(per_expert, ref_expert)
    assert torch.equal(in_rank, ref_in_rank)
    print(f"[{name}] num_sms(blocks) = {num_sms}, rdma = {per_rdma}")
    if verbose:
        print("  num_tokens_per_expert =", per_expert.tolist())
        print("  num_tokens_per_rank   =", per_rank.tolist())
        print("  is_token_in_rank      =", in_rank.int().tolist())
    print(f"  sum(per_expert) = {int(per_expert.sum())}, sum(per_rank) = {int(per_rank.sum())}  PASS")


def main() -> None:
    small = torch.tensor([[0, 3], [2, 3], [5, -1], [7, 1], [4, 5]], dtype=torch.int64)
    check("small: 8 experts, 4 ranks", small, num_ranks=4, num_experts=8, verbose=True)

    torch.manual_seed(0)
    num_tokens, num_experts, num_ranks, num_topk = 1024, 256, 8, 8
    scores = torch.rand((num_tokens, num_experts))
    topk_idx = scores.topk(num_topk, dim=1).indices
    topk_idx[torch.rand((num_tokens, num_topk)) < 0.05] = -1
    check("random: 256 experts, 8 ranks", topk_idx, num_ranks=num_ranks, num_experts=num_experts, verbose=False)


if __name__ == "__main__":
    main()
```

模拟输出张量初始化为 `-1`，用来对应 host 端 `torch::empty` 不初始化的情况：如果某个位置没被任何 block 写到，会留下 `-1`，和参考结果对不上，断言失败。

运行：

```bash
python3 dispatch_layout_ref.py
```

本机（macOS，PyTorch CPU）实际输出：

```text
[small: 8 experts, 4 ranks] num_sms(blocks) = 3, rdma = None
  num_tokens_per_expert = [1, 1, 1, 2, 1, 2, 0, 1]
  num_tokens_per_rank   = [2, 2, 2, 1]
  is_token_in_rank      = [[1, 1, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, 1], [0, 0, 1, 0]]
  sum(per_expert) = 9, sum(per_rank) = 7  PASS
[random: 256 experts, 8 ranks] num_sms(blocks) = 65, rdma = None
  sum(per_expert) = 7789, sum(per_rank) = 5295  PASS
```

第一组和第 7 节手算的结果逐项一致；第二组是单机 8 卡、256 experts 的典型配置，65 个 block（64 个 expert block + 1 个 rank block）。

在有 GPU 的 DeepEP V1 环境里，可以把 `buffer.get_dispatch_layout(topk_idx.cuda(), num_experts)` 的返回值和上面 `reference()` 的结果对比，作为真实 kernel 的运行时验证。

## 10. 速记卡

```text
grid:
  ceil(E/4) 个 expert block + ceil(P/8) 个 rank block
  sm_id == blockIdx.x，不是物理 SM；单机只有 1 个 rank block

expert block:
  shared [256][4]，行 = thread_id，列 = 局部 expert（expert_idx - expert_begin_idx）
  每线程写自己的行 -> __syncthreads -> 线程 k 加第 k 列
  -> num_tokens_per_expert（token-expert 路由数，不去重）

rank block:
  寄存器 is_in_rank[8] 标记一个 token 命中哪些 rank
  -> 直接写 is_token_in_rank[i, r]（每行由唯一线程写）
  -> shared [256][8] 累加 (is_in_rank > 0) -> __syncthreads -> 线程 k 加第 k 列
  -> num_tokens_per_rank（token-rank 去重计数）

单机：
  num_tokens_per_rdma_rank = nullptr / None，RDMA 分支算了但不写回

要点：
  每个 block 都扫全部 token，换来无原子操作、无跨 block 同步
  输出是 torch::empty，靠 block 区间划分保证每个元素都被写到
  E 必须能被 P 整除（host 端不检查）
```

## 11. 自测题与答案

1. **E = 64、P = 8 时启动多少个 block？其中 `blockIdx.x = 16` 的 block 做什么？**
   答：ceil(64/4) + ceil(8/8) = 16 + 1 = 17 个。`blockIdx.x = 16` 时 `16*4 = 64 >= 64`，expert 区间为空，它是唯一的 rank block，`rank_begin = (16-16)*8 = 0`，负责 rank 0–7。
2. **`num_tokens_per_expert_per_thread[37][2]` 在 block 5 里存的是什么？**
   答：线程 37 扫过的那些 token 中，路由到全局 expert `5*4 + 2 = 22` 的次数。
3. **归约阶段为什么只有线程 0–3 工作？换成线程 252–255 可以吗？**
   答：每列需要一个线程负责，用 `thread_id` 直接当列号最简单，条件 `expert_begin_idx + thread_id < expert_end_idx` 自然选出了前 4 个线程。换成别的线程在逻辑上可以，但要额外做下标换算，没有好处。
4. **token 的 topk 是 `[6, 7]`，E = 8、P = 4，这个 token 让 `num_tokens_per_rank` 和 `num_tokens_per_expert` 分别增加多少？**
   答：6/2 = 3，7/2 = 3，都在 rank 3。`num_tokens_per_rank[3]` 加 1；`num_tokens_per_expert[6]` 和 `[7]` 各加 1，expert 计数一共加 2。
5. **单机 8 卡时 Python 端拿到的 `num_tokens_per_rdma_rank` 是什么？kernel 里 RDMA 相关代码还执行吗？**
   答：是 `None`。kernel 里局部数组和 shared 数组照样声明、清零，但 `rdma_rank_end_idx = 8/8 = 1`、`rdma_rank_begin_idx = 0`，累加循环会执行，只是最后写回那段因为指针是 `nullptr` 被跳过，结果被丢弃。

## 12. 学习进度

- [x] DeepEP V1 normal：layout → rank dispatch → local expert metadata → handle → combine
- [x] 区分 rank-major 通信布局、expert-major 计算布局与 gate 加权
- [x] DeepEP V1 low-latency：定容 dispatch、packed expert 输入、handle 回程元数据、weighted combine 与 hook
- [x] 机内 `get_dispatch_layout` kernel：block 分工、per-thread 计数 + 按列归约、token-rank 去重、单机 RDMA 分支

### 下一知识点

机内 `intranode_dispatch` 的 notify 阶段：各 rank 如何交换 `num_tokens_per_rank`，得到 `rank_prefix_matrix` 和 `channel_prefix_matrix`，进而确定每个 token 在接收端 `recv_x` 里的行号。
