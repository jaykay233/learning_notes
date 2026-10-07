# DeepEP V1：机内 `notify_dispatch` 与 CPU 端接收计数握手

> 承接 [08-deepep-v1-barrier-block.md](./08-deepep-v1-barrier-block.md)。07 在每张卡上各自算出了 `num_tokens_per_rank` / `num_tokens_per_expert` / `is_token_in_rank`，08 讲了跨 GPU 的 `barrier_block`。本文讲 intranode dispatch 的第一个 kernel `notify_dispatch`：8 张卡如何借两次 barrier 交换计数，算出「我要收多少 token、每个本地 expert 收多少、每个 channel 写到哪一行」，以及 CPU 如何通过 pinned mapped 内存拿到这些数，好去分配 `recv_x`。
>
> 代码对照：本地 DeepEP checkout `/Users/saboxu/Downloads/communication/DeepEP`。kernel 在 `csrc/kernels/legacy/intranode.cu`（L25–L173），依赖函数在 `csrc/kernels/legacy/utils.cuh`，CPU 端在 `csrc/legacy/buffer.hpp`（计数器分配约 L152–L161，调用与轮询约 L528–L599）。

```text
本次讲解位置
章节：communication / DeepEP V1
小节：intranode dispatch 的 notify 阶段
知识点：block 0 两次 barrier 交换计数 → rank_prefix_matrix / 接收总数 / 每 expert 接收数；block 1..P 计算 channel_prefix_matrix；pinned mapped 计数器 + −1 哨兵的 CPU 轮询
上次：机内 barrier_block（8×8 分布式信号矩阵、+T/−T、warp 投票、kSyncOnly）
下次：intranode dispatch kernel 本体（发送/接收 block、环形队列、head/tail 流控、send_head）
PTX：shfl.sync.bfly（__shfl_xor_sync）、elect.sync；CUDA Runtime：cudaHostAllocMapped、cudaHostGetDevicePointer
```

## 为什么现在讲这个

dispatch 要把 token 写进 `recv_x`，但 `recv_x` 是 **CPU 端用 `torch::empty({num_recv_tokens, hidden})` 分配的**，而 `num_recv_tokens` 取决于其他 7 张卡要发给我多少 token。这个数在本卡上算不出来，只有各卡交换过计数才知道。于是有两个必须解决的问题：

1. **GPU 之间要交换计数，并且算出每个 token 的落点**。只知道总数不够：dispatch kernel 里每个接收 warp 必须知道「rank s 的 channel c 发来的第 k 个 token 写到 `recv_x` 第几行」，否则只能用原子操作抢位置，结果不确定，combine 也就没法按原顺序送回去。`rank_prefix_matrix` 和 `channel_prefix_matrix` 就是为此预先算好的两张前缀和表。
2. **GPU 算出的数要尽快交给 CPU**。如果用 `cudaMemcpy` + `cudaStreamSynchronize`，要等整条 stream 空下来，开销大，也没法和后续工作重叠。DeepEP 让 kernel 直接把结果写进 CPU 内存（pinned mapped），CPU 用 −1 作哨兵忙等。

不搞清楚这一阶段，就读不懂 dispatch kernel 里 `rank_offset`、`channel_start_offset`、`recv_channel_offset` 这些偏移从哪里来，也看不懂 `buffer.hpp` 里那段 `while (true)` 轮询为什么用 `>= 0` 判断。

## 0. 心智模型

```text
每张卡输入（07 算好）：num_tokens_per_rank[P]、num_tokens_per_expert[E]、is_token_in_rank[T][P]

block 0（每张卡 1 个，128 线程）
  barrier（只对齐）
  线程 t：把 "我 → rank t 的 token 数" 和 "我 → rank t 各本地 expert 的 token 数"
          通过 NVLink 写进 GPU t 的 buffer
  barrier（带 fence，保证写入可见）
  读自己 buffer：第 i 行 = rank i 发给我的数
    → 按列做前缀和 → rank_prefix_matrix[i][me] = rank 0..i 发给我的累计数
    → 最后一行 = 接收总数              → 写进 CPU 内存 moe_recv_counter
    → 每个本地 expert 求和并对齐       → 写进 CPU 内存 moe_recv_expert_counter[]
  清零后续 dispatch 要用的队列元数据，再 barrier

block 1..P（block 1+d 负责目标 rank d）
  按 channel 数 "本卡发往 rank d 的 token 数"，做前缀和 → channel_prefix_matrix[d][c]

CPU：先把计数器置 −1 → 启动 kernel → 轮询直到全部 >= 0 → 分配 recv_x
```

一句话：**block 0 负责「别人发给我多少」，block 1..P 负责「我发给别人的，每个 channel 占多少」。**

## 1. 术语与常量

| 名字 | 位置 | 值 / 含义 |
|---|---|---|
| `P` / `kNumRanks` | 模板参数 | 机内 rank 数，≤ 8 |
| `E` / `num_experts` | 参数 | 全局 expert 数，必须能被 P 整除 |
| `num_experts_per_rank` | kernel 内 | E / P，本地 expert 数，≤ 128 |
| `kNumThreads` | host 端 L166 | 128，每个 block 的线程数 |
| grid | host 端 L170 | `1 + P` 个 block |
| `num_channels` | `buffer.hpp` L437 | `config.num_sms / 2`，默认 20 / 2 = 10 |
| `buffer_ptrs[r]` | 08 / `buffer.hpp` | GPU r 上 NVLink buffer 的起始地址（IPC 映射后本进程可用） |
| `per_rank_buffer` | kernel 内 | 某张卡 buffer 开头的 P×P 个 int |
| `per_expert_buffer` | kernel 内 | 紧跟其后的 P×(E/P) 个 int |
| `moe_recv_counter` / `_mapped` | `buffer.hpp` L153–L155 | 同一块 CPU 内存的 CPU 地址 / GPU 端地址 |
| `moe_recv_expert_counter` / `_mapped` | `buffer.hpp` L158–L161 | 同上，1024 个 int（`LEGACY_NUM_MAX_LOCAL_EXPERTS`） |
| `LEGACY_NUM_CPU_TIMEOUT_SECS` | `compiled.cuh` | 100，CPU 轮询超时秒数 |

## 2. 完整代码

### 2.1 kernel 与 host 启动（`csrc/kernels/legacy/intranode.cu` L25–L173）

```cpp
template <int kNumRanks>
__global__ void notify_dispatch(const int* num_tokens_per_rank,
                                int* moe_recv_counter_mapped,
                                const int* num_tokens_per_expert,
                                int* moe_recv_expert_counter_mapped,
                                int num_experts,
                                int num_tokens,
                                int num_channels,
                                const bool* is_token_in_rank,
                                int* channel_prefix_matrix,
                                int* rank_prefix_matrix_copy,
                                int num_memset_int,
                                int expert_alignment,
                                void** buffer_ptrs,
                                int** barrier_signal_ptrs,
                                int rank) {
    auto sm_id = static_cast<int>(blockIdx.x);
    auto thread_id = static_cast<int>(threadIdx.x), num_threads = static_cast<int>(blockDim.x);
    auto lane_id = thread_id % 32, warp_id = thread_id / 32, num_warps = num_threads / 32;

    if (sm_id == 0) {
        // Barrier first
        barrier_block<kNumRanks, true>(barrier_signal_ptrs, rank);

        int *per_rank_buffer, *per_expert_buffer;
        if (thread_id < kNumRanks) {
            per_rank_buffer = static_cast<int*>(buffer_ptrs[thread_id]);
            per_expert_buffer = per_rank_buffer + kNumRanks * kNumRanks;
        }

        // After this loop:
        //  - `per_rank_buffer[rank][i, j]` means the number of tokens from rank i to rank j
        //  - `per_expert_buffer[rank][i, j]` means the number of tokens from rank i to local expert j
        int num_experts_per_rank = num_experts / kNumRanks;
        if (thread_id < kNumRanks) {
            per_rank_buffer[rank * kNumRanks + thread_id] = num_tokens_per_rank[thread_id];
            #pragma unroll
            for (int i = 0; i < num_experts_per_rank; ++i)
                per_expert_buffer[rank * num_experts_per_rank + i] = num_tokens_per_expert[thread_id * num_experts_per_rank + i];
        }

        // Wait for all ranks to be finished
        barrier_block<kNumRanks>(barrier_signal_ptrs, rank);

        // Sum per-rank counts and return to CPU
        // Also pre-compute the prefix sum for data sending
        auto local_per_rank_buffer = static_cast<int*>(buffer_ptrs[rank]);
        if (thread_id < kNumRanks) {
            #pragma unroll
            for (int i = 1; i < kNumRanks; ++i)
                local_per_rank_buffer[i * kNumRanks + thread_id] += local_per_rank_buffer[(i - 1) * kNumRanks + thread_id];
            if (thread_id == rank)
                *moe_recv_counter_mapped = local_per_rank_buffer[(kNumRanks - 1) * kNumRanks + rank];
        }

        // Sum per-experts counts and return to CPU
        auto local_per_expert_buffer = local_per_rank_buffer + kNumRanks * kNumRanks;
        if (thread_id < num_experts_per_rank) {
            int sum = 0;
            #pragma unroll
            for (int i = 0; i < kNumRanks; ++i)
                sum += local_per_expert_buffer[i * num_experts_per_rank + thread_id];
            sum = (sum + expert_alignment - 1) / expert_alignment * expert_alignment;
            moe_recv_expert_counter_mapped[thread_id] = sum;
        }
        __syncthreads();

        // Copy rank size prefix matrix to another tensor
        #pragma unroll
        for (int i = thread_id; i < kNumRanks * kNumRanks; i += num_threads)
            rank_prefix_matrix_copy[i] = local_per_rank_buffer[i];

        // Extra memset for later communication queue
        #pragma unroll
        for (int i = thread_id; i < num_memset_int; i += num_threads)
            local_per_expert_buffer[i] = 0;

        // Barrier
        barrier_block<kNumRanks>(barrier_signal_ptrs, rank);
    } else {
        int dst_rank = sm_id - 1;
        for (int channel_id = warp_id; channel_id < num_channels; channel_id += num_warps) {
            int token_start_idx, token_end_idx;
            get_channel_task_range(num_tokens, num_channels, channel_id, token_start_idx, token_end_idx);

            // Iterate over tokens
            int count = 0;
            for (int64_t i = token_start_idx + lane_id; i < token_end_idx; i += 32)
                count += is_token_in_rank[i * kNumRanks + dst_rank];
            count = warp_reduce_sum(count);
            if (elect_one_sync())
                channel_prefix_matrix[dst_rank * num_channels + channel_id] = count;
        }
        __syncthreads();

        // Pre-compute prefix sum for all channels
        if (thread_id == 0) {
            #pragma unroll
            for (int i = 1; i < num_channels; ++i)
                channel_prefix_matrix[dst_rank * num_channels + i] += channel_prefix_matrix[dst_rank * num_channels + i - 1];
        }
    }
}

void notify_dispatch(const int* num_tokens_per_rank,
                     int* moe_recv_counter_mapped,
                     int num_ranks,
                     const int* num_tokens_per_expert,
                     int* moe_recv_expert_counter_mapped,
                     int num_experts,
                     int num_tokens,
                     const bool* is_token_in_rank,
                     int* channel_prefix_matrix,
                     int* rank_prefix_matrix_copy,
                     int num_memset_int,
                     int expert_alignment,
                     void** buffer_ptrs,
                     int** barrier_signal_ptrs,
                     int rank,
                     cudaStream_t stream,
                     int num_channels) {
#define NOTIFY_DISPATCH_LAUNCH_CASE(ranks)        \
    LAUNCH_KERNEL(&cfg,                           \
                  notify_dispatch<ranks>,         \
                  num_tokens_per_rank,            \
                  moe_recv_counter_mapped,        \
                  num_tokens_per_expert,          \
                  moe_recv_expert_counter_mapped, \
                  num_experts,                    \
                  num_tokens,                     \
                  num_channels,                   \
                  is_token_in_rank,               \
                  channel_prefix_matrix,          \
                  rank_prefix_matrix_copy,        \
                  num_memset_int,                 \
                  expert_alignment,               \
                  buffer_ptrs,                    \
                  barrier_signal_ptrs,            \
                  rank);                          \
    break

    constexpr int kNumThreads = 128;
    EP_HOST_ASSERT(num_experts % num_ranks == 0);
    EP_HOST_ASSERT(num_experts / num_ranks <= kNumThreads and num_ranks <= kNumThreads);

    SETUP_LAUNCH_CONFIG(1 + num_ranks, kNumThreads, stream);
    SWITCH_RANKS(NOTIFY_DISPATCH_LAUNCH_CASE);
#undef NOTIFY_DISPATCH_LAUNCH_CASE
}
```

### 2.2 依赖的设备函数（`csrc/kernels/legacy/utils.cuh`）

```cpp
__forceinline__ __device__ void get_channel_task_range(int num_tokens, int num_sms, int sm_id, int& token_start_idx, int& token_end_idx) {
    int num_tokens_per_sm = ceil_div(num_tokens, num_sms);
    token_start_idx = min(num_tokens_per_sm * sm_id, num_tokens);
    token_end_idx = min(token_start_idx + num_tokens_per_sm, num_tokens);
}

__device__ __forceinline__ uint32_t elect_one_sync() {
#ifndef DISABLE_SM90_FEATURES
    uint32_t pred = 0;
    asm volatile(
        "{\n"
        ".reg .b32 %%rx;\n"
        ".reg .pred %%px;\n"
        "      elect.sync %%rx|%%px, %1;\n"
        "@%%px mov.s32 %0, 1;\n"
        "}\n"
        : "+r"(pred)
        : "r"(0xffffffff));
    return pred;
#else
    return get_lane_id() == 0;
#endif
}

// Unified reduction function
template <int kNumLanesPerGroup, bool kIntergroupReduce, typename T, typename Op>
__forceinline__ __device__ T warp_reduce(T value, Op op) {
    EP_STATIC_ASSERT(kNumLanesPerGroup == 32 or kNumLanesPerGroup == 16 or kNumLanesPerGroup == 8 or kNumLanesPerGroup == 4 or
                         kNumLanesPerGroup == 2 or kNumLanesPerGroup == 1,
                     "Invalid number of lanes");
    constexpr uint32_t mask = 0xffffffff;
    if constexpr (kIntergroupReduce) {
        if constexpr (kNumLanesPerGroup <= 1)
            value = op(value, __shfl_xor_sync(mask, value, 1));
        if constexpr (kNumLanesPerGroup <= 2)
            value = op(value, __shfl_xor_sync(mask, value, 2));
        if constexpr (kNumLanesPerGroup <= 4)
            value = op(value, __shfl_xor_sync(mask, value, 4));
        if constexpr (kNumLanesPerGroup <= 8)
            value = op(value, __shfl_xor_sync(mask, value, 8));
        if constexpr (kNumLanesPerGroup <= 16)
            value = op(value, __shfl_xor_sync(mask, value, 16));
    } else {
        if constexpr (kNumLanesPerGroup >= 32)
            value = op(value, __shfl_xor_sync(mask, value, 16));
        if constexpr (kNumLanesPerGroup >= 16)
            value = op(value, __shfl_xor_sync(mask, value, 8));
        if constexpr (kNumLanesPerGroup >= 8)
            value = op(value, __shfl_xor_sync(mask, value, 4));
        if constexpr (kNumLanesPerGroup >= 4)
            value = op(value, __shfl_xor_sync(mask, value, 2));
        if constexpr (kNumLanesPerGroup >= 2)
            value = op(value, __shfl_xor_sync(mask, value, 1));
    }
    return value;
}

// Convenience aliases
template <int kNumLanesPerGroup = 32, bool kIntergroupReduce = false, typename T>
__forceinline__ __device__ T warp_reduce_sum(T value) {
    return warp_reduce<kNumLanesPerGroup, kIntergroupReduce, T>(value, ReduceSum<T>{});
}
```

`barrier_block` 的完整代码见 08。

### 2.3 CPU 端计数器的分配（`csrc/legacy/buffer.hpp` `Buffer` 构造函数 L152–L161）

```cpp
        // MoE counter
        CUDA_RUNTIME_CHECK(cudaMallocHost(&moe_recv_counter, sizeof(int64_t), cudaHostAllocMapped));
        CUDA_RUNTIME_CHECK(cudaHostGetDevicePointer(&moe_recv_counter_mapped, const_cast<int*>(moe_recv_counter), 0));
        *moe_recv_counter = -1;

        // MoE expert-level counter
        CUDA_RUNTIME_CHECK(cudaMallocHost(&moe_recv_expert_counter, sizeof(int) * LEGACY_NUM_MAX_LOCAL_EXPERTS, cudaHostAllocMapped));
        CUDA_RUNTIME_CHECK(cudaHostGetDevicePointer(&moe_recv_expert_counter_mapped, const_cast<int*>(moe_recv_expert_counter), 0));
        for (int i = 0; i < LEGACY_NUM_MAX_LOCAL_EXPERTS; ++i)
            moe_recv_expert_counter[i] = -1;
```

成员声明（L70–L75）：`volatile int* moe_recv_counter`、`int* moe_recv_counter_mapped`、`volatile int* moe_recv_expert_counter`、`int* moe_recv_expert_counter_mapped`。

### 2.4 CPU 端调用与轮询（`csrc/legacy/buffer.hpp` `intranode_dispatch` L528–L599）

```cpp
        // Barrier or send sizes
        // To clean: channel start/end offset, head and tail
        int num_memset_int = num_channels * num_ranks * 4;
        if (cached_mode) {
            num_recv_tokens = cached_num_recv_tokens;
            rank_prefix_matrix = cached_rank_prefix_matrix.value();
            channel_prefix_matrix = cached_channel_prefix_matrix.value();

            // Copy rank prefix matrix and clean flags
            intranode::cached_notify_dispatch(
                rank_prefix_matrix.data_ptr<int>(), num_memset_int, buffer_ptrs_gpu, barrier_signal_ptrs_gpu, rank, num_ranks, comm_stream);
        } else {
            rank_prefix_matrix = torch::empty({num_ranks, num_ranks}, dtype(torch::kInt32).device(torch::kCUDA));
            channel_prefix_matrix = torch::empty({num_ranks, num_channels}, dtype(torch::kInt32).device(torch::kCUDA));

            // Send sizes
            // Meta information:
            //  - Size prefix by ranks, shaped as `[num_ranks, num_ranks]`
            //  - Size prefix by experts (not used later), shaped as `[num_ranks, num_local_experts]`
            // NOTES: no more token dropping in this version
            *moe_recv_counter = -1;
            for (int i = 0; i < num_local_experts; ++i)
                moe_recv_expert_counter[i] = -1;
            EP_HOST_ASSERT(num_ranks * (num_ranks + num_local_experts) * sizeof(int) <= num_nvl_bytes);
            intranode::notify_dispatch(num_tokens_per_rank->data_ptr<int>(),
                                       moe_recv_counter_mapped,
                                       num_ranks,
                                       num_tokens_per_expert->data_ptr<int>(),
                                       moe_recv_expert_counter_mapped,
                                       num_experts,
                                       num_tokens,
                                       is_token_in_rank.data_ptr<bool>(),
                                       channel_prefix_matrix.data_ptr<int>(),
                                       rank_prefix_matrix.data_ptr<int>(),
                                       num_memset_int,
                                       expert_alignment,
                                       buffer_ptrs_gpu,
                                       barrier_signal_ptrs_gpu,
                                       rank,
                                       comm_stream,
                                       num_channels);

            if (num_worst_tokens > 0) {
                // No CPU sync, just allocate the worst case
                num_recv_tokens = num_worst_tokens;

                // Must be forward with top-k stuffs
                EP_HOST_ASSERT(topk_idx.has_value());
                EP_HOST_ASSERT(topk_weights.has_value());
            } else {
                // Synchronize total received tokens and tokens per expert
                auto start_time = std::chrono::high_resolution_clock::now();
                while (true) {
                    // Read total count
                    num_recv_tokens = static_cast<int>(*moe_recv_counter);

                    // Read per-expert count
                    bool ready = (num_recv_tokens >= 0);
                    for (int i = 0; i < num_local_experts and ready; ++i)
                        ready &= moe_recv_expert_counter[i] >= 0;

                    if (ready)
                        break;

                    // Timeout check
                    if (std::chrono::duration_cast<std::chrono::seconds>(std::chrono::high_resolution_clock::now() - start_time).count() >
                        LEGACY_NUM_CPU_TIMEOUT_SECS)
                        throw std::runtime_error("DeepEP error: CPU recv timeout");
                }
                num_recv_tokens_per_expert_list = std::vector<int>(moe_recv_expert_counter, moe_recv_expert_counter + num_local_experts);
            }
        }
```

## 3. block 分工

grid = `1 + P` 个 block，每个 128 线程。以 P = 8 为例：

| block | 线程中真正干活的 | 负责 |
|---|---|---|
| 0 | 线程 0..7（rank 维度）、线程 0..E/P−1（expert 维度）、全部 128 线程（拷贝 / 清零） | 交换计数、算 `rank_prefix_matrix`、写 CPU 计数器、清零队列元数据 |
| 1 + d（d = 0..7） | 4 个 warp，每个 warp 负责若干 channel | 算 `channel_prefix_matrix` 的第 d 行 |

两组 block 互不依赖：block 1..P 只读本卡的 `is_token_in_rank`，不需要等 barrier，所以和 block 0 并行。

host 端两个断言决定了上限：

- `num_experts % num_ranks == 0`：expert 必须均分，否则 `num_experts_per_rank` 截断后尾部 expert 没人统计，host 直接报错（而不是静默丢 token）。
- `num_experts / num_ranks <= 128`：block 0 用 `thread_id` 当本地 expert 编号，128 个线程最多覆盖 128 个本地 expert。`buffer.hpp` L469 还有一个 `<= 1024` 的断言（CPU 计数器数组大小），实际上限取更紧的 128。

## 4. block 0 逐段

### 4.1 第一次 barrier：只对齐（L47）

```cpp
barrier_block<kNumRanks, true>(barrier_signal_ptrs, rank);
```

`kSyncOnly = true`：本卡此刻还没写任何要给对端读的数据，只需要确认「所有卡都进入了这一轮 notify」，避免本卡往对端 buffer 写计数时，对端上一个 kernel 还在用这块 buffer。

### 4.2 把计数写进对端 buffer（L49–L64）

线程 t（t < P）拿到 GPU t 的 buffer 地址，把「我（rank）→ rank t」的数写进去：

| 写入位置（在 GPU t 上） | 写入值 | 含义 |
|---|---|---|
| `per_rank_buffer[rank * P + t]` | `num_tokens_per_rank[t]` | rank 发给 t 的 token 数 |
| `per_expert_buffer[rank * (E/P) + i]` | `num_tokens_per_expert[t * (E/P) + i]` | rank 发给 t 的第 i 个本地 expert 的 token 数 |

注意每个线程只写一格 `per_rank_buffer`。8 张卡做完后，**GPU r 的 `per_rank_buffer` 只有第 r 列被填满**：`[i][r]` 由 rank i 写入，表示 rank i 发给 r 的数。其他列不保证有意义。`per_expert_buffer` 则被完整填满：第 i 行是 rank i 发给本卡各本地 expert 的数。

这里的「去重」在 07 已经做完：`num_tokens_per_rank[t]` 统计的是「至少有一个 topk expert 在 rank t 上的 token 数」，一个 token 选了 rank t 的两个 expert 也只算一次；`num_tokens_per_expert` 不去重，每个 expert 各算一次。

### 4.3 第二次 barrier：带 fence（L67）

默认 `kSyncOnly = false`，先 `fence.acq_rel.sys` + `__syncthreads()` 再发信号（见 08 第 4.1 节）。保证 4.2 写进对端的数在对端通过 barrier 后一定可见。

### 4.4 列前缀和与接收总数（L71–L78）

线程 t 负责第 t 列，从上往下累加：

```text
local_per_rank_buffer[i][t] += local_per_rank_buffer[i-1][t]    (i = 1..P-1)
```

只有第 `rank` 列有意义，所以只有 `thread_id == rank` 的结果被使用：

- `rank_prefix_matrix[i][rank]` = rank 0..i 发给本卡的累计 token 数。dispatch 里 rank s 发来的 token 从 `recv_x` 第 `rank_prefix_matrix[s-1][rank]` 行开始（s = 0 时从 0 开始）。
- 最后一行 `[P-1][rank]` = 本卡接收总数，写进 `*moe_recv_counter_mapped`。

### 4.5 每个本地 expert 的接收数（L81–L89）

线程 t（t < E/P）负责本地 expert t，把 `per_expert_buffer` 第 t 列的 P 个数加起来，再向上对齐到 `expert_alignment` 的倍数：

```text
sum = Σ_i local_per_expert_buffer[i][t]
sum = ceil(sum / expert_alignment) * expert_alignment
moe_recv_expert_counter_mapped[t] = sum
```

对齐是为了下游 grouped GEMM：每个 expert 的 token 段长度是 alignment 的倍数时，GEMM kernel 可以按固定的块大小切分。`sum = 0` 时对齐后仍为 0（`(0 + a − 1) / a = 0`），照样写入。

这些 per-expert 计数最后变成 `num_recv_tokens_per_expert_list` 返回给 Python。

### 4.6 拷出、清零、第三次 barrier（L90–L103）

1. `__syncthreads()`：等 4.4 的列前缀和全部写完。
2. 把本卡 buffer 开头的 P×P 拷到 `rank_prefix_matrix_copy`（就是 host 端的 `rank_prefix_matrix` tensor）。它会进 handle，cached 模式和 combine 都要用。
3. 从 `local_per_expert_buffer` 起清零 `num_memset_int = C × P × 4` 个 int。这块区域在 dispatch kernel 里正好是 4 张 `[C][P]` 的元数据表：`start_offset`、`end_offset`、`head_idx`、`tail_idx`。`per_expert_buffer` 读完就没用了，和队列元数据共用同一段内存。
4. 第三次 barrier：保证所有卡都清零完毕，后面的 dispatch kernel 才能开始往对端写 offset 和 tail。dispatch 用「值为 0 = 还没写」判断，清零必须先于对端写入。

## 5. block 1..P：`channel_prefix_matrix`（L104–L126）

`dst_rank = sm_id − 1`。128 线程 = 4 个 warp，warp w 负责 channel w, w+4, w+8, …（10 个 channel 时：warp 0 负责 0/4/8，warp 1 负责 1/5/9，warp 2 负责 2/6，warp 3 负责 3/7）。

对每个 channel：

1. `get_channel_task_range` 把 `num_tokens` 均分成 C 段，channel c 负责 `[c × ceil(T/C), min((c+1) × ceil(T/C), T))`。
2. **一个 warp 数一个 channel**：lane l 数 token `start+l, start+l+32, …` 中有多少个 `is_token_in_rank[i][dst]` 为真。分到同一个 channel 的 32 个 lane 各数一部分，所以要归约。
3. `warp_reduce_sum`：蝶形归约，`__shfl_xor_sync` 依次用偏移 16、8、4、2、1，每一步 lane l 与 lane `l ^ offset` 交换并相加。5 步后 **32 个 lane 都持有总和**。
4. `elect_one_sync()`：32 个 lane 值相同，只需一个 lane 写入，避免 32 次重复写同一地址。SM90 上用 `elect.sync` 硬件指令，其他架构退化为 `lane_id == 0`。
5. `__syncthreads()` 后线程 0 对第 dst 行做前缀和：`channel_prefix_matrix[dst][c]` = channel 0..c 中发往 dst 的累计 token 数。

dispatch 发送端用它告诉接收端「本 channel 在我这一段里从第几个开始、到第几个结束」：`start = c > 0 ? cpm[dst][c−1] : 0`，`end = cpm[dst][c]`。

`warp_reduce_sum` 用 4 个 lane 演示（偏移 2、1）：

| 步骤 | lane 0 | lane 1 | lane 2 | lane 3 |
|---|---|---|---|---|
| 初值 | 1 | 0 | 2 | 1 |
| xor 2：与 lane^2 相加 | 1+2=3 | 0+1=1 | 2+1=3 | 1+0=1 |
| xor 1：与 lane^1 相加 | 3+1=4 | 1+3=4 | 3+1=4 | 1+3=4 |

## 6. CPU 拿到结果：pinned mapped 内存 + −1 哨兵

### 6.1 两个指针，同一块 CPU 内存

`cudaMallocHost(..., cudaHostAllocMapped)` 分配页锁定的 CPU 内存，并允许 GPU 直接访问；`cudaHostGetDevicePointer` 取出 GPU 端访问这块内存用的地址（最后的 flags 参数必须为 0）。

| 变量 | 是什么 | 谁用 |
|---|---|---|
| `moe_recv_counter` | CPU 地址（`volatile int*`） | CPU：置 −1、轮询 |
| `moe_recv_counter_mapped` | 同一块内存的 GPU 端地址 | kernel：`*moe_recv_counter_mapped = ...` |

数据**只存在 CPU 内存里**。GPU 写 `_mapped` 时经 PCIe 写到 CPU 内存（zero-copy）。开启 UVA 的 64 位系统上两个地址数值通常相同，分成两个变量主要是标明用途。`const_cast` 是为了去掉 `volatile`；CPU 端声明成 `volatile`，是为了让编译器每次轮询都真正读内存。

### 6.2 −1 哨兵和 `>= 0` 判断

1. CPU 每次 dispatch 前把计数器全部置 −1（L549–L551）。
2. kernel 写入 ≥ 0 的结果。0 是合法结果（一个 token 都没收到），所以哨兵必须是 −1，判断必须是 `>= 0`。
3. CPU 循环检查：总数 ≥ 0，且前 `num_local_experts` 个 expert 计数全部 ≥ 0。`for` 条件里的 `and ready` 让循环在遇到第一个 −1 时立刻停止。

**为什么每个 expert 都要单独检查**：总数由 `thread_id == rank` 的线程写，每个 expert 的计数由不同线程分别写，到达 CPU 的先后顺序没有保证。看到总数不代表 expert 计数也到了。

**什么情况下永远等不到**：某个计数器根本没有线程去写，比如本地 expert 超过 128 个。host 断言已经挡住了这种情况；如果绕过了，CPU 会在 100 秒后抛出 `DeepEP error: CPU recv timeout`。

### 6.3 `num_worst_tokens > 0`：跳过 CPU 同步

调用者传入 `num_worst_tokens > 0` 时，CPU 不等 kernel，直接按最坏情况分配 `recv_x[num_worst_tokens, hidden]`。好处是整个 dispatch 没有 CPU-GPU 同步，可以录进 CUDA graph。代价：

- 要求同时传 `topk_idx` / `topk_weights`（L576–L577）。
- `num_recv_tokens_per_expert_list` 为空。
- 多出来的行由 dispatch kernel 把 `recv_topk_idx` 填成 −1（dispatch kernel L535–L545），下游据此跳过。
- 调用者必须保证足够大，测试里用的是 `num_tokens * num_ranks`（`tests/legacy/test_intranode.py`）。

### 6.4 internode 版本的差别

`internode_dispatch`（`buffer.hpp` 约 L1049–L1108）多一个 `moe_recv_rdma_counter`（本卡经 RDMA 从各节点收到的 token 数），轮询条件变成三类计数都 ≥ 0。另外 internode kernel 写入前会先 `while (ld_volatile_global(ptr) != -1);` 等 CPU 完成重置，intranode 版本没有这一步。

## 7. 数值追踪：4 rank、8 expert、2 channel

输入沿用 07 的 rank 0，另外三张卡各给几组 topk（每个 rank 2 个本地 expert：rank r 拥有 expert 2r、2r+1）：

| rank | topk_idx | num_tokens_per_rank | num_tokens_per_expert |
|---|---|---|---|
| 0 | `[0,3] [2,3] [5,-1] [7,1] [4,5]` | `[2, 2, 2, 1]` | `[1,1,1,2,1,2,0,1]` |
| 1 | `[1,6] [0,2] [3,4]` | `[2, 2, 1, 1]` | `[1,1,1,1,1,0,1,0]` |
| 2 | `[6,7] [2,5] [0,1] [4,6]` | `[1, 1, 2, 2]` | `[1,1,1,0,1,1,2,1]` |
| 3 | `[3,5] [1,7]` | `[1, 1, 1, 1]` | `[0,1,0,1,0,1,0,1]` |

**rank 1 的 block 0**

4.2 之后，GPU 1 的 `per_rank_buffer` 第 1 列 = 各 rank 的 `num_tokens_per_rank[1]` = `[2, 2, 1, 1]`。列前缀和：

| i | 写入值（rank i → 1） | 前缀和 `rank_prefix_matrix[i][1]` | 物理含义 |
|---|---|---|---|
| 0 | 2 | 2 | rank 0 的 token 占 `recv_x` 第 0–1 行 |
| 1 | 2 | 4 | rank 1 的占第 2–3 行 |
| 2 | 1 | 5 | rank 2 的占第 4 行 |
| 3 | 1 | 6 | rank 3 的占第 5 行 |

接收总数 = 6。

`per_expert_buffer`（rank 1 的本地 expert 是全局 expert 2、3）：

| 来源 rank | → 本地 expert 0（全局 2） | → 本地 expert 1（全局 3） |
|---|---|---|
| 0 | 1 | 2 |
| 1 | 1 | 1 |
| 2 | 1 | 0 |
| 3 | 0 | 1 |
| 合计 | 3 | 4 |

`expert_alignment = 1` 时写入 `[3, 4]`；`expert_alignment = 4` 时写入 `[4, 4]`（3 向上对齐到 4，4 不变）。注意 3 + 4 = 7 > 6：rank 0 的 token 1（topk `[2,3]`）两个 expert 都在 rank 1 上，按 rank 只算 1 个 token，按 expert 算 2 次。

**rank 1 的 block 1..4**

rank 1 有 3 个 token，C = 2，`ceil(3/2) = 2`：channel 0 = token 0–1，channel 1 = token 2。

| token | topk | 去往的 rank |
|---|---|---|
| 0 | `[1, 6]` | 0, 3 |
| 1 | `[0, 2]` | 0, 1 |
| 2 | `[3, 4]` | 1, 2 |

| dst | channel 0 计数 | channel 1 计数 | 前缀和后 `channel_prefix_matrix[dst]` |
|---|---|---|---|
| 0 | 2 | 0 | `[2, 2]` |
| 1 | 1 | 1 | `[1, 2]` |
| 2 | 0 | 1 | `[0, 1]` |
| 3 | 1 | 0 | `[1, 1]` |

## 8. 验证：CPU 重放脚本

当前机器是 macOS，没有 NVIDIA GPU，不能运行 DeepEP。下面的脚本只依赖 Python 标准库，按 kernel 的写入位置和顺序重放 notify 阶段（包括「线程 t 写 GPU t 的 buffer」「只有第 rank 列有效」「蝶形归约」），并用一个线程模拟 GPU 分几次写入 mapped 内存、主线程按 `buffer.hpp` 的逻辑轮询。它验证的是数据语义，**不能**代替在 GPU 上验证内存序。

保存为 `notify_dispatch_ref.py`：

```python
import threading
import time

WARP_SIZE = 32


def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def get_channel_task_range(num_tokens: int, num_channels: int, channel_id: int) -> tuple[int, int]:
    per = ceil_div(num_tokens, num_channels)
    start = min(per * channel_id, num_tokens)
    end = min(start + per, num_tokens)
    return start, end


def warp_reduce_sum(lane_values: list[int]) -> list[int]:
    """Butterfly reduction with __shfl_xor_sync offsets 16, 8, 4, 2, 1."""
    v = list(lane_values)
    for offset in (16, 8, 4, 2, 1):
        v = [v[lane] + v[lane ^ offset] for lane in range(WARP_SIZE)]
    return v


def layout(topk_idx: list[list[int]], num_experts: int, num_ranks: int):
    """Same results as get_dispatch_layout (note 07)."""
    experts_per_rank = num_experts // num_ranks
    num_tokens = len(topk_idx)
    per_expert = [0] * num_experts
    per_rank = [0] * num_ranks
    is_token_in_rank = [[False] * num_ranks for _ in range(num_tokens)]
    for t, row in enumerate(topk_idx):
        for e in row:
            if e < 0:
                continue
            per_expert[e] += 1
            is_token_in_rank[t][e // experts_per_rank] = True
        for r in range(num_ranks):
            per_rank[r] += is_token_in_rank[t][r]
    return per_rank, per_expert, is_token_in_rank


def notify_dispatch(inputs, num_experts: int, num_channels: int, expert_alignment: int):
    num_ranks = len(inputs)
    epr = num_experts // num_ranks
    assert num_experts % num_ranks == 0 and epr <= 128

    # buffer_ptrs[r]: [per_rank_buffer P*P | per_expert_buffer P*epr], lives on GPU r
    buffers = [[0] * (num_ranks * num_ranks + num_ranks * epr) for _ in range(num_ranks)]

    # Block 0, before the 2nd barrier: rank `me`, thread t writes into GPU t's buffer
    for me, (per_rank, per_expert, _) in enumerate(inputs):
        for t in range(num_ranks):
            buffers[t][me * num_ranks + t] = per_rank[t]
            for i in range(epr):
                buffers[t][num_ranks * num_ranks + me * epr + i] = per_expert[t * epr + i]

    results = []
    for me in range(num_ranks):
        buf = buffers[me]
        # Block 0, after the barrier: column-wise inclusive prefix sum, thread t owns column t
        for t in range(num_ranks):
            for i in range(1, num_ranks):
                buf[i * num_ranks + t] += buf[(i - 1) * num_ranks + t]
        rank_prefix_matrix = [buf[i * num_ranks:(i + 1) * num_ranks] for i in range(num_ranks)]
        moe_recv_counter = buf[(num_ranks - 1) * num_ranks + me]

        expert_counter = []
        for t in range(epr):
            s = sum(buf[num_ranks * num_ranks + i * epr + t] for i in range(num_ranks))
            expert_counter.append((s + expert_alignment - 1) // expert_alignment * expert_alignment)

        # Blocks 1..P: block (1 + dst) fills row dst; warp w handles channels w, w + 4, ...
        is_token_in_rank = inputs[me][2]
        num_tokens = len(is_token_in_rank)
        channel_prefix_matrix = [[0] * num_channels for _ in range(num_ranks)]
        for dst in range(num_ranks):
            for c in range(num_channels):
                start, end = get_channel_task_range(num_tokens, num_channels, c)
                lanes = [0] * WARP_SIZE
                for lane in range(WARP_SIZE):
                    for i in range(start + lane, end, WARP_SIZE):
                        lanes[lane] += is_token_in_rank[i][dst]
                reduced = warp_reduce_sum(lanes)
                assert len(set(reduced)) == 1
                channel_prefix_matrix[dst][c] = reduced[0]
            for c in range(1, num_channels):
                channel_prefix_matrix[dst][c] += channel_prefix_matrix[dst][c - 1]

        results.append((rank_prefix_matrix, moe_recv_counter, expert_counter, channel_prefix_matrix))
    return results


def cpu_poll_demo(final_total: int, final_experts: list[int]) -> None:
    """Pinned mapped host memory: GPU threads write at different times, CPU polls for >= 0."""
    moe_recv_counter = [-1]
    moe_recv_expert_counter = [-1] * len(final_experts)

    def gpu_block0() -> None:
        time.sleep(0.02)
        moe_recv_counter[0] = final_total
        for i, v in enumerate(final_experts):
            time.sleep(0.01)
            moe_recv_expert_counter[i] = v

    gpu = threading.Thread(target=gpu_block0)
    gpu.start()
    polls, snapshots = 0, []
    while True:
        polls += 1
        num_recv_tokens = moe_recv_counter[0]
        ready = num_recv_tokens >= 0
        i = 0
        while i < len(final_experts) and ready:
            ready &= moe_recv_expert_counter[i] >= 0
            i += 1
        snap = (num_recv_tokens, tuple(moe_recv_expert_counter))
        if not snapshots or snapshots[-1] != snap:
            snapshots.append(snap)
        if ready:
            break
        time.sleep(0.001)
    gpu.join()
    for total, experts in snapshots:
        print(f"  CPU sees total={total:>2}, experts={list(experts)}")
    print(f"  ready after {len(snapshots)} distinct states -> allocate recv_x[{num_recv_tokens}, hidden]")


def main() -> None:
    num_ranks, num_experts, num_channels, alignment = 4, 8, 2, 1
    topk_by_rank = [
        [[0, 3], [2, 3], [5, -1], [7, 1], [4, 5]],  # rank 0 (same tokens as note 07)
        [[1, 6], [0, 2], [3, 4]],                   # rank 1
        [[6, 7], [2, 5], [0, 1], [4, 6]],           # rank 2
        [[3, 5], [1, 7]],                           # rank 3
    ]
    inputs = [layout(t, num_experts, num_ranks) for t in topk_by_rank]
    for r, (per_rank, per_expert, _) in enumerate(inputs):
        print(f"rank {r}: num_tokens_per_rank={per_rank}, num_tokens_per_expert={per_expert}")

    results = notify_dispatch(inputs, num_experts, num_channels, alignment)
    rpm, total, experts, cpm = results[1]
    print("\nrank 1 after notify_dispatch:")
    print("  rank_prefix_matrix (row i = prefix over source ranks 0..i):")
    for row in rpm:
        print(f"    {row}")
    print(f"  moe_recv_counter = {total}")
    print(f"  moe_recv_expert_counter (alignment=1) = {experts}")
    print(f"  channel_prefix_matrix [dst][channel] = {cpm}")

    for me in range(num_ranks):
        recv = sum(inputs[src][0][me] for src in range(num_ranks))
        assert results[me][1] == recv
        assert results[me][0][num_ranks - 1][me] == recv
    print("\ncheck: moe_recv_counter == sum of num_tokens_per_rank[*][me] on every rank  PASS")

    aligned = notify_dispatch(inputs, num_experts, num_channels, 4)[1][2]
    print(f"rank 1 moe_recv_expert_counter with expert_alignment=4: {aligned}")

    print("\nCPU poll loop on rank 1:")
    cpu_poll_demo(total, experts)


if __name__ == "__main__":
    main()
```

运行：

```bash
python3 notify_dispatch_ref.py
```

本机（macOS，Python 3.14）实际输出：

```text
rank 0: num_tokens_per_rank=[2, 2, 2, 1], num_tokens_per_expert=[1, 1, 1, 2, 1, 2, 0, 1]
rank 1: num_tokens_per_rank=[2, 2, 1, 1], num_tokens_per_expert=[1, 1, 1, 1, 1, 0, 1, 0]
rank 2: num_tokens_per_rank=[1, 1, 2, 2], num_tokens_per_expert=[1, 1, 1, 0, 1, 1, 2, 1]
rank 3: num_tokens_per_rank=[1, 1, 1, 1], num_tokens_per_expert=[0, 1, 0, 1, 0, 1, 0, 1]

rank 1 after notify_dispatch:
  rank_prefix_matrix (row i = prefix over source ranks 0..i):
    [0, 2, 0, 0]
    [0, 4, 0, 0]
    [0, 5, 0, 0]
    [0, 6, 0, 0]
  moe_recv_counter = 6
  moe_recv_expert_counter (alignment=1) = [3, 4]
  channel_prefix_matrix [dst][channel] = [[2, 2], [1, 2], [0, 1], [1, 1]]

check: moe_recv_counter == sum of num_tokens_per_rank[*][me] on every rank  PASS
rank 1 moe_recv_expert_counter with expert_alignment=4: [4, 4]

CPU poll loop on rank 1:
  CPU sees total=-1, experts=[-1, -1]
  CPU sees total= 6, experts=[-1, -1]
  CPU sees total= 6, experts=[3, -1]
  CPU sees total= 6, experts=[3, 4]
  ready after 4 distinct states -> allocate recv_x[6, hidden]
```

- `rank_prefix_matrix` 只有第 1 列非零，对应「GPU r 只有第 r 列被写入」。脚本里其他列是初始化的 0；真实 GPU 上那些位置的内容不保证有意义。
- CPU 轮询经历了「总数到了、expert 计数还没到」的中间状态，说明只检查总数是不够的。

## 9. 常见错误与症状

| 错误理解 / 写法 | 正确理解 | 可观察到的症状 |
|---|---|---|
| 以为 `rank_prefix_matrix` 整个矩阵都有效 | 本卡只有第 `rank` 列有意义 | 用其他列算偏移会得到随机数，`recv_x` 写错行 |
| 以为 `moe_recv_counter_mapped` 指向 GPU 显存 | 数据在 CPU 内存，`_mapped` 只是 GPU 访问它的地址 | 误以为需要 `cudaMemcpy` 才能读，加了多余同步 |
| 哨兵用 0、判断用 `> 0` | 哨兵 −1，判断 `>= 0` | 某个 expert 收到 0 个 token 时 CPU 永远等不到，100 秒后 `CPU recv timeout` |
| 只轮询总数 | 总数和每个 expert 计数都要 ≥ 0 | 偶发：`num_recv_tokens_per_expert_list` 里出现 −1，下游 grouped GEMM 尺寸错误 |
| E 不能被 P 整除 / 本地 expert > 128 | host 断言会拦截 | `EP_HOST_ASSERT` 失败，直接报错 |
| 第二次 barrier 也用 `kSyncOnly = true` | 写完对端数据后必须带 fence | 不会卡死；对端偶尔读到旧计数，接收行数算错 |
| 去掉最后的 barrier | 清零必须先于所有卡的 dispatch 写入 | dispatch 接收端把旧 offset/tail 当成新值，读到错误数据或越界 |
| `num_worst_tokens` 给得太小 | 必须 ≥ 真实接收数，测试用 `num_tokens * num_ranks` | 写越界，非法地址访问或静默覆盖相邻显存 |

## 10. 速记卡

```text
grid = 1 + P 个 block，128 线程

block 0：
  barrier<SyncOnly>
  线程 t：GPU t.per_rank[me][t] = 我→t 的 token 数；GPU t.per_expert[me][*] = 我→t 各本地 expert 的数
  barrier（fence）
  本卡第 me 列前缀和 → rank_prefix_matrix[i][me] = rank 0..i 发给我的累计数
  total = [P-1][me] → *moe_recv_counter_mapped
  线程 t：Σ_i per_expert[i][t]，向上对齐 → moe_recv_expert_counter_mapped[t]
  拷出 P×P；清零 C×P×4 个 int（dispatch 的 start/end/head/tail）；barrier

block 1+d：warp 一个 channel，lane 跨步计数 → warp_reduce_sum → elect_one 写；thread 0 前缀和
  channel_prefix_matrix[d][c] = 我发往 d 的 token 中，channel 0..c 的累计数

CPU：置 −1 → launch → while(总数 >= 0 且每个 expert >= 0) → 分配 recv_x
  num_worst_tokens > 0：不等，直接按最坏情况分配（CUDA graph 友好）
限制：E % P == 0；E/P <= 128；超时 100 s
```

## 11. 自测题与答案

1. **GPU 3 的 `per_rank_buffer` 里，哪些格子是有意义的？`[1][3]` 是谁写的、表示什么？**
   答：只有第 3 列。`[1][3]` 是 rank 1 的 block 0 线程 3 通过 NVLink 写进来的，表示 rank 1 发给 rank 3 的 token 数。

2. **rank 2 的 dispatch 接收端，收 rank 3 发来的 token 时从 `recv_x` 第几行开始？用什么表达式？**
   答：`rank_prefix_matrix[2][2]`，即 rank 0、1、2 发给 rank 2 的累计数。一般式是 `src > 0 ? rank_prefix_matrix[src−1][rank] : 0`。

3. **某个本地 expert 一个 token 都没收到，`moe_recv_expert_counter[i]` 会被写入吗？CPU 会卡住吗？**
   答：会写入 0（对齐公式对 0 仍得 0），写入没有条件。CPU 判断用 `>= 0`，0 视为就绪，不会卡住。

4. **`warp_reduce_sum` 之后为什么还要 `elect_one_sync`？去掉会怎样？**
   答：蝶形归约后 32 个 lane 都持有同一个总和，去掉的话 32 个 lane 会往同一地址写同一个值，结果正确但多了 31 次无用写入。`elect_one_sync` 只让一个 lane 写。

5. **`moe_recv_counter` 和 `moe_recv_counter_mapped` 有什么区别？数据存在哪里？**
   答：同一块 pinned mapped CPU 内存的两个地址：前者给 CPU 用（置 −1、轮询），后者给 kernel 写。数据只存在 CPU 内存里，GPU 写入经 PCIe 直达；UVA 下两个地址数值通常相同。

## 12. 学习进度

- [x] DeepEP V1 normal：layout → rank dispatch → local expert metadata → handle → combine
- [x] 区分 rank-major 通信布局、expert-major 计算布局与 gate 加权
- [x] DeepEP V1 low-latency：定容 dispatch、packed expert 输入、handle 回程元数据、weighted combine 与 hook
- [x] 机内 `get_dispatch_layout` kernel：block 分工、per-thread 计数 + 按列归约、token-rank 去重、单机 RDMA 分支
- [x] 机内 `barrier_block`：分布式 8×8 信号矩阵、+T/−T 成对原子操作、warp 投票、`kSyncOnly`、`<= 0` 的原因
- [x] 机内 `notify_dispatch`：两次 barrier 交换计数、`rank_prefix_matrix` 只有本列有效、expert 对齐、`channel_prefix_matrix`、pinned mapped 计数器与 −1 哨兵、`num_worst_tokens`

### 下一知识点

intranode dispatch kernel 本体：偶数 block 发送、奇数 block 接收；每个（channel, 对端 rank）一个环形队列，head/tail 流控；`send_head` 记录 channel 内序号；接收端如何用 `rank_prefix_matrix` + `channel_prefix_matrix` + channel 内序号算出 `recv_x` 行号。见 [10-deepep-v1-intranode-dispatch-kernel.md](./10-deepep-v1-intranode-dispatch-kernel.md)。
