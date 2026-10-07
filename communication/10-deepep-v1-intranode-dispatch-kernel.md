# DeepEP V1：机内 `dispatch` kernel —— 环形队列、head/tail 流控与 `send_head`

> 承接 [09-deepep-v1-notify-dispatch.md](./09-deepep-v1-notify-dispatch.md)。09 算好了「每个来源 rank 的 token 从 `recv_x` 第几行开始」（`rank_prefix_matrix`）、「每个 channel 在这一段里占多少」（`channel_prefix_matrix`），CPU 也按接收总数分配好了 `recv_x`。本文讲真正搬数据的 `dispatch` kernel：每张卡把 token 通过 NVLink 推进目标卡的环形队列，目标卡再搬进 `recv_x` 的最终行。顺带整理了读这个 kernel 时需要的 GPU 背景：`__launch_bounds__`、grid/block/SM、HBM/L1/shared memory、release/acquire、named barrier。
>
> 代码对照：本地 DeepEP checkout `/Users/saboxu/Downloads/communication/DeepEP`。kernel 在 `csrc/kernels/legacy/intranode.cu`（L211–L624），工具函数在 `csrc/kernels/legacy/utils.cuh`、`buffer.cuh`、`launch.cuh`，host 端在 `csrc/legacy/buffer.hpp`（约 L602–L665）。

```text
本次讲解位置
章节：communication / DeepEP V1
小节：intranode dispatch 的数据搬运阶段
知识点：偶数 block 发、奇数 block 收；每个（channel, 对端 rank）一个接收端环形队列；Buffer<T> 切分布局；发送端分批 + release tail；接收端 acquire tail + 释放 head；send_head 记录 channel 内序号；recv_x 行号 = rank 偏移 + channel 起点 + channel 内序号
上次：intranode notify_dispatch（rank_prefix_matrix、channel_prefix_matrix、pinned mapped 计数器）
下次：intranode combine kernel（如何用 send_head / recv_channel_offset 把 expert 输出按原顺序送回并求和）
PTX：st.release.sys.global、ld.acquire.sys.global、st.global.L1::no_allocate、bar.sync a, b（named barrier）、cp.async.bulk（TMA 1D）
```

## 为什么现在讲这个

notify 阶段只交换了「数量」，真正的数据（每个 token 约 14 KB 的 hidden 向量）还在原卡上。把它们送到目标卡，要同时解决四个问题：

1. **不用原子操作也能确定落点**。8 张卡、每卡 10 个 channel 同时往同一个 `recv_x` 写，如果用原子计数器抢行号，结果顺序每次都不同，combine 就没法把 expert 输出送回原 token。DeepEP 的做法是：发送端按 token 顺序给每个 token 编 channel 内序号，接收端按「rank 偏移 + channel 起点 + 序号」落位，完全确定。
2. **缓冲区比数据小**。发送端不知道对端每次新分配的 `recv_x` 的地址，只能写进 Buffer 初始化时 IPC 映射好的固定 NVLink buffer。这块 buffer 给每个（channel, 来源 rank）只留 `num_recv_buffer_tokens`（默认 256）个槽位，token 再多也只能循环使用，于是需要 head/tail 流控：写满就等，读完就释放。
3. **跨 GPU 的内存序**。发送端先写数据、后写 tail。如果对端先看到新 tail、后看到数据，就会读到旧数据。这类 bug 不会卡死，只会偶尔算错。必须用 `st.release.sys` / `ld.acquire.sys` 配对。
4. **combine 要能原路返回**。dispatch 要给 combine 留下「token t 发给 rank r 时排在 channel 内第几个」（`send_head`），以及「每个 channel 在来源段内从第几行开始」（`recv_channel_offset`）。

不读懂这个 kernel，就没法解释 `Config` 里 `num_max_nvl_chunked_send_tokens` / `num_max_nvl_chunked_recv_tokens` 这两个参数的含义，也读不懂 combine。

## 0. 心智模型

```text
每张卡启动 num_sms = 20 个 block、每 block 768 线程（24 warp）
  block 2c   = channel c 的发送端     block 2c+1 = channel c 的接收端
  block 内按对端 rank 分组：线程 [96r, 96r+96) = 3 个 warp 负责对端 rank r

队列：每个（channel c, 发送 rank S → 接收 rank R）一个，全部放在 R 的显存里
  元数据：start_offset / end_offset / head / tail（各一个 int）
  数据环：N = num_recv_buffer_tokens 个槽位，每槽一个 token 的 x / src_idx / topk / scales

发送 rank S 的 block 2c、第 R 组：               接收 rank R 的 block 2c+1、第 S 组：
  写 start/end offset（编码 -v-1）     ──────►    等 offset 非 0，解码
  按 token 顺序遍历本 channel 的 token
    等空位 ≥ 一批（读 head）           ◄──────    处理完一批后写 head（释放槽位）
    一批最多 num_max_send_tokens 个：
      send_head[t][R] = 序号
      槽位 = 序号 % N，写 x / src_idx / topk(本地化) / scales  ──NVLink──►  环形队列
    bar.sync（只同步本组 3 个 warp）
    st.release.sys tail = 累计序号     ──────►    ld.acquire.sys 读 tail
                                                  把 [head, tail) 搬进 recv_x[rank偏移 + channel起点 + 序号]
```

一句话：**发送端编号、接收端按号落位，环形队列靠 head/tail 两个单调计数器流控。**

## 1. 术语与常量

| 名字 | 位置 | 值 / 含义 |
|---|---|---|
| `kNumThreads` | host L576 | 768，每 block 线程数（24 warp） |
| `kNumTMABytesPerWarp` | host L577 | 8192，每 warp 的 TMA 中转 shared memory |
| `smem_size` | host L579 | 8192 × 24 = 196608 字节（192 KB），每 block 动态 shared memory |
| `num_sms` | `Config`，默认 20 | grid 大小，必须为偶数 |
| `num_channels`（C） | kernel L244 | `num_sms / 2` = 10 |
| `num_threads_per_rank` | kernel L243 | 768 / P；P = 8 时为 96（3 个 warp） |
| `num_max_send_tokens` | `Config.num_max_nvl_chunked_send_tokens` | 一批最多发几个 token；8 卡 dispatch 默认 6 |
| `num_recv_buffer_tokens`（N） | `Config.num_max_nvl_chunked_recv_tokens` | 每个环形队列的槽位数；8 卡默认 256 |
| `hidden_int4` | host L653 | `hidden × 元素字节数 / 16`；hidden 7168 的 bf16 为 896 |
| `is_sender` | kernel L239 | `sm_id % 2 == 0` |
| `responsible_channel` | kernel L247 | `sm_id / 2` |
| `responsible_rank` | kernel L245 | `thread_id / num_threads_per_rank`：发送 block 里是目标 rank，接收 block 里是来源 rank |
| `cached_channel_tail_idx` | 发送端 L333 | 本队列累计已发 token 数（channel 内序号计数器），只增不减 |
| `cached_channel_head_idx` | 接收端 L447 | 本队列累计已收 token 数 |

`Config` 的默认值来自 `deep_ep/buffers/legacy.py`：`num_sms = 20`；`get_dispatch_config(8)` 返回 `Config(Buffer.num_sms, 6, 256, 6, 128)`，前两个数就是 NVLink 的 send / recv chunk。

## 2. 完整代码

### 2.1 kernel 与 host 启动（`csrc/kernels/legacy/intranode.cu` L211–L624）

```cpp
template <int kNumRanks, int kNumThreads, int kNumTMABytesPerWarp>
__global__ void __launch_bounds__(kNumThreads, 1) dispatch(int4* recv_x,
                                                           float* recv_x_scales,
                                                           int* recv_src_idx,
                                                           topk_idx_t* recv_topk_idx,
                                                           float* recv_topk_weights,
                                                           int* recv_channel_offset,
                                                           int* send_head,
                                                           const int4* x,
                                                           const float* x_scales,
                                                           const topk_idx_t* topk_idx,
                                                           const float* topk_weights,
                                                           const bool* is_token_in_rank,
                                                           const int* channel_prefix_matrix,
                                                           int num_tokens,
                                                           int num_worst_tokens,
                                                           int hidden_int4,
                                                           int num_topk,
                                                           int num_experts,
                                                           int num_scales,
                                                           int scale_token_stride,
                                                           int scale_hidden_stride,
                                                           void** buffer_ptrs,
                                                           int rank,
                                                           int num_max_send_tokens,
                                                           int num_recv_buffer_tokens) {
    const auto num_sms = static_cast<int>(gridDim.x), sm_id = static_cast<int>(blockIdx.x);
    const auto thread_id = static_cast<int>(threadIdx.x), lane_id = get_lane_id();
    const bool is_sender = sm_id % 2 == 0;
    EP_DEVICE_ASSERT(num_sms % 2 == 0);

    // Several warps are response for a single rank
    const auto num_threads_per_rank = kNumThreads / kNumRanks;
    const auto num_channels = num_sms / 2;
    const auto responsible_rank = (static_cast<int>(thread_id)) / num_threads_per_rank;
    // Even-numbered blocks for sending, odd-numbered blocks for receiving.
    const auto responsible_channel = sm_id / 2;

    int num_experts_per_rank = num_experts / kNumRanks;
    EP_DEVICE_ASSERT(num_experts_per_rank > 0 or num_topk == 0);
    EP_DEVICE_ASSERT(num_topk <= 32);
    EP_DEVICE_ASSERT((topk_idx == nullptr) == (topk_weights == nullptr));
    EP_DEVICE_ASSERT((recv_topk_idx == nullptr) == (recv_topk_weights == nullptr));

    // Calculate pointers by the specific layout
    // `rank_prefix_matrix`: kNumRanks * kNumRanks * sizeof(int)
    auto ptr = reinterpret_cast<void*>(static_cast<int8_t*>(buffer_ptrs[is_sender ? responsible_rank : rank]) +
                                       kNumRanks * kNumRanks * sizeof(int));
    int target_rank = is_sender ? rank : responsible_rank;
    auto num_channels_total = num_channels * kNumRanks;
    auto channel_rank_offset = responsible_channel * kNumRanks + target_rank;

    // Channel buffer metadata
    // Senders are responsible for tails, and receivers are responsible for heads
    // Stored on the receiver side
    // The retired signals are actually boolean flags, but to align with 16 bytes, we make it `int64_t`
    // `start_offset`: kNumChannels * kNumRanks * sizeof(int)
    // `end_offset`: kNumChannels * kNumRanks * sizeof(int)
    // `head_idx`: kNumChannels * kNumRanks * sizeof(int)
    // `tail_idx`: kNumChannels * kNumRanks * sizeof(int)
    auto channel_start_offset = Buffer<int>(ptr, num_channels_total, channel_rank_offset);
    auto channel_end_offset = Buffer<int>(ptr, num_channels_total, channel_rank_offset);
    auto channel_head_idx = Buffer<int>(ptr, num_channels_total, channel_rank_offset);
    auto channel_tail_idx = Buffer<int>(ptr, num_channels_total, channel_rank_offset);

    // Channel data buffers, stored on the receiver side
    // `x_buffers`: kNumChannels * kNumRanks * num_recv_buffer_tokens * hidden_int4 * sizeof(int4)
    // `src_idx_buffers`: kNumChannels * kNumRanks * num_recv_buffer_tokens * sizeof(int)
    // `topk_idx_buffers`: kNumChannels * kNumRanks * num_recv_buffer_tokens * num_topk * sizeof(topk_idx_t)
    // `topk_weights_buffers`: kNumChannels * kNumRanks * num_recv_buffer_tokens * num_topk * sizeof(float)
    // `x_scales_buffers`: kNumChannels * kNumRanks * num_recv_buffer_tokens * num_scales * sizeof(float)
    auto channel_x_buffers = Buffer<int4>(
        ptr, num_channels_total * num_recv_buffer_tokens * hidden_int4, channel_rank_offset * num_recv_buffer_tokens * hidden_int4);
    auto channel_src_idx_buffers =
        Buffer<int>(ptr, num_channels_total * num_recv_buffer_tokens, channel_rank_offset * num_recv_buffer_tokens);
    auto channel_topk_idx_buffers = Buffer<topk_idx_t>(
        ptr, num_channels_total * num_recv_buffer_tokens * num_topk, channel_rank_offset * num_recv_buffer_tokens * num_topk);
    auto channel_topk_weights_buffers =
        Buffer<float>(ptr, num_channels_total * num_recv_buffer_tokens * num_topk, channel_rank_offset * num_recv_buffer_tokens * num_topk);
    auto channel_x_scales_buffers = Buffer<float>(
        ptr, num_channels_total * num_recv_buffer_tokens * num_scales, channel_rank_offset * num_recv_buffer_tokens * num_scales);

    // TMA stuffs
#ifndef DISABLE_SM90_FEATURES
    extern __shared__ __align__(1024) uint8_t smem_buffer[];
    auto half_hidden_int4 = hidden_int4 / 2;
    auto half_hidden_bytes = half_hidden_int4 * static_cast<int>(sizeof(int4));
    auto tma_buffer = smem_buffer + (thread_id / 32) * kNumTMABytesPerWarp;
    auto tma_mbarrier = reinterpret_cast<uint64_t*>(tma_buffer + half_hidden_bytes);
    uint32_t tma_phase = 0;
    if (elect_one_sync()) {
        mbarrier_init(tma_mbarrier, 1);
        fence_barrier_init();
        EP_DEVICE_ASSERT(hidden_int4 % 2 == 0 and half_hidden_bytes + sizeof(uint64_t) <= kNumTMABytesPerWarp);
    }
    __syncwarp();
#endif

    if (is_sender) {
        // Workers for sending
        constexpr int num_send_warps = kNumThreads / 32;
        constexpr int num_send_warps_per_rank = num_send_warps / kNumRanks;
        const auto send_thread_id = thread_id;
        const auto send_warp_id_in_rank = send_thread_id % num_threads_per_rank / 32;
        EP_DEVICE_ASSERT(kNumRanks <= 32);
        EP_DEVICE_ASSERT(num_send_warps % kNumRanks == 0);

        // Send offset by `-value - 1`, e.g. 0 -> -1, 1 -> -2
        // NOTES: this is for distinguishing zero tokens
        if (send_warp_id_in_rank == 0 and elect_one_sync()) {
            int value = responsible_channel > 0 ? channel_prefix_matrix[responsible_rank * num_channels + responsible_channel - 1] : 0;
            st_relaxed_sys_global(channel_start_offset.buffer(), -value - 1);
            value = channel_prefix_matrix[responsible_rank * num_channels + responsible_channel];
            st_relaxed_sys_global(channel_end_offset.buffer(), -value - 1);
        }
        __syncwarp();

        // Get tasks
        int token_start_idx, token_end_idx;
        get_channel_task_range(num_tokens, num_channels, responsible_channel, token_start_idx, token_end_idx);

        // Iterate over all tokens and send by chunks
        int cached_channel_tail_idx = 0;
        for (int64_t token_idx = token_start_idx; token_idx < token_end_idx;) {
            // Check destination queue emptiness, or wait a buffer to be released (rare cases)
            // NOTES: the head index received by different warps may not be the same
            auto start_time = clock64();
            if (elect_one_sync()) {
                while (true) {
                    // NOTES: we only consider the worst case, because counting the real numbers are time-consuming
                    int num_used_slots = cached_channel_tail_idx - ld_volatile_global(channel_head_idx.buffer());
                    if (num_recv_buffer_tokens - num_used_slots >= num_max_send_tokens)
                        break;

                    // Rare cases to loop again
                    if (clock64() - start_time > LEGACY_NUM_TIMEOUT_CYCLES) {
                        printf("DeepEP timeout for dispatch senders, rank %d, responsible_channel = %d\n", rank, responsible_channel);
                        trap();
                    }
                }
            }
            __syncwarp();

            int chunk_token_idx = 0;
            while (chunk_token_idx < num_max_send_tokens and token_idx < token_end_idx) {
                // NOTES: for the same token, the warp assigned to save `send_head` may be different from the warp assigned to send the
                // following data
                if (token_idx % num_send_warps_per_rank == send_warp_id_in_rank and elect_one_sync())
                    send_head[token_idx * kNumRanks + responsible_rank] =
                        is_token_in_rank[token_idx * kNumRanks + responsible_rank] ? cached_channel_tail_idx : -1;

                // Skip if not selected
                if (not is_token_in_rank[token_idx * kNumRanks + responsible_rank]) {
                    token_idx++;
                    continue;
                }

                // Get an empty slot
                int dst_slot_idx = (cached_channel_tail_idx++) % num_recv_buffer_tokens;
                if (cached_channel_tail_idx % num_send_warps_per_rank == send_warp_id_in_rank) {
                    // Copy data
                    auto shifted_channel_x_buffers = channel_x_buffers.buffer() + dst_slot_idx * hidden_int4;
                    auto shifted_x = x + token_idx * hidden_int4;
                    UNROLLED_WARP_COPY(5, lane_id, hidden_int4, shifted_channel_x_buffers, shifted_x, __ldg, st_na_global);

                    // Copy source index
                    if (elect_one_sync())
                        channel_src_idx_buffers[dst_slot_idx] = static_cast<int>(token_idx);

                    // Copy `topk_idx` and `topk_weights` with transformed index
                    if (lane_id < num_topk) {
                        // Top-k index
                        int recv_expert_begin = responsible_rank * num_experts_per_rank,
                            recv_expert_end = (responsible_rank + 1) * num_experts_per_rank;
                        auto idx_value = __ldg(topk_idx + token_idx * num_topk + lane_id);
                        idx_value = (idx_value >= recv_expert_begin and idx_value < recv_expert_end) ? idx_value - recv_expert_begin : -1;
                        channel_topk_idx_buffers[dst_slot_idx * num_topk + lane_id] = idx_value;

                        // Top-k weights
                        auto weight_value = __ldg(topk_weights + token_idx * num_topk + lane_id);
                        weight_value = (idx_value >= 0) ? weight_value : 0.0f;
                        channel_topk_weights_buffers[dst_slot_idx * num_topk + lane_id] = weight_value;
                    }

                    // Copy `x_scales`
                    #pragma unroll
                    for (int i = lane_id; i < num_scales; i += 32) {
                        auto offset = token_idx * scale_token_stride + i * scale_hidden_stride;
                        channel_x_scales_buffers[dst_slot_idx * num_scales + i] = __ldg(x_scales + offset);
                    }
                }

                // Move token index
                chunk_token_idx++, token_idx++;
            }

            // Move tail index
            // NOTES: here all warps should share the same new tail
            asm volatile("bar.sync %0, %1;" ::"r"(responsible_rank), "r"(num_threads_per_rank));
            if (send_warp_id_in_rank == 0 and elect_one_sync())
                st_release_sys_global(channel_tail_idx.buffer(), cached_channel_tail_idx);
        }
    } else {
        // Workers for receiving and copying into buffer
        constexpr int num_recv_warps = kNumThreads / 32;
        constexpr int num_recv_warps_per_rank = num_recv_warps / kNumRanks;
        const auto recv_thread_id = thread_id;
        const auto recv_thread_id_in_rank = recv_thread_id % num_threads_per_rank;
        const auto recv_warp_id_in_rank = recv_thread_id_in_rank / 32;
        EP_DEVICE_ASSERT(kNumRanks <= 32);
        EP_DEVICE_ASSERT(recv_thread_id >= 0 and num_recv_warps % kNumRanks == 0);

        // Calculate offset first
        auto rank_prefix_matrix = static_cast<int*>(buffer_ptrs[rank]);
        int rank_offset = responsible_rank > 0 ? rank_prefix_matrix[(responsible_rank - 1) * kNumRanks + rank] : 0;

        // Receive channel offset
        int total_offset, num_tokens_to_recv;
        if (elect_one_sync()) {
            while ((total_offset = ld_volatile_global(channel_start_offset.buffer())) == 0)
                ;
            while ((num_tokens_to_recv = ld_volatile_global(channel_end_offset.buffer())) == 0)
                ;
            total_offset = -total_offset - 1, num_tokens_to_recv = -num_tokens_to_recv - 1;
            if (recv_warp_id_in_rank == 0)
                recv_channel_offset[responsible_rank * num_channels + responsible_channel] = total_offset;
            num_tokens_to_recv -= total_offset;
        }
        total_offset = __shfl_sync(0xffffffff, total_offset, 0);
        total_offset += rank_offset;
        num_tokens_to_recv = __shfl_sync(0xffffffff, num_tokens_to_recv, 0);

        // Shared tail indices for different warps
        __shared__ volatile int shared_channel_tail_idx[kNumRanks];

        auto start_time = clock64();
        int cached_channel_head_idx = 0, cached_channel_tail_idx = 0;
        while (num_tokens_to_recv > 0) {
            // NOTES: unlike the sender, the receiver must ensure that the tail indices hold by different warps are the same
            while (recv_thread_id_in_rank == 0) {
                cached_channel_tail_idx = ld_acquire_sys_global(channel_tail_idx.buffer());

                // Ready to copy
                if (cached_channel_head_idx != cached_channel_tail_idx) {
                    shared_channel_tail_idx[responsible_rank] = cached_channel_tail_idx;
                    break;
                }

                // Timeout check
                if (clock64() - start_time > LEGACY_NUM_TIMEOUT_CYCLES) {
                    printf("DeepEP timeout for dispatch receivers, rank %d, responsible_channel = %d, tokens remained: %d\n",
                           rank,
                           responsible_channel,
                           num_tokens_to_recv);
                    trap();
                }
            }

            // Synchronize queue tail
            asm volatile("bar.sync %0, %1;" ::"r"(responsible_rank), "r"(num_threads_per_rank));
            cached_channel_tail_idx = shared_channel_tail_idx[responsible_rank];

            // Copy data
            int num_recv_tokens = cached_channel_tail_idx - cached_channel_head_idx;
            for (int chunk_idx = recv_warp_id_in_rank; chunk_idx < num_recv_tokens; chunk_idx += num_recv_warps_per_rank) {
                int token_idx_in_buffer = (cached_channel_head_idx + chunk_idx) % num_recv_buffer_tokens;
                auto shifted_buffer_x_int4 = channel_x_buffers.buffer() + token_idx_in_buffer * hidden_int4;
                auto shifted_recv_x_int4 = recv_x + static_cast<int64_t>(total_offset + chunk_idx) * hidden_int4;
#ifndef DISABLE_SM90_FEATURES
                #pragma unroll
                for (int i = 0; i < 2; ++i) {
                    tma_store_wait<0>();
                    if (elect_one_sync()) {
                        tma_load_1d(tma_buffer, shifted_buffer_x_int4 + i * half_hidden_int4, tma_mbarrier, half_hidden_bytes);
                        mbarrier_arrive_and_expect_tx(tma_mbarrier, half_hidden_bytes);
                        mbarrier_wait(tma_mbarrier, tma_phase);
                        tma_store_1d(tma_buffer, shifted_recv_x_int4 + i * half_hidden_int4, half_hidden_bytes, false);
                    }
                }
                __syncwarp();
#else
                UNROLLED_WARP_COPY(5, lane_id, hidden_int4, shifted_recv_x_int4, shifted_buffer_x_int4, ld_nc_global, st_na_global);
#endif
            }

            // Copy `src_idx`
            #pragma unroll 4
            for (int chunk_idx = cached_channel_head_idx + recv_thread_id_in_rank; chunk_idx < cached_channel_tail_idx;
                 chunk_idx += 32 * num_recv_warps_per_rank)
                recv_src_idx[total_offset + chunk_idx - cached_channel_head_idx] =
                    ld_nc_global(channel_src_idx_buffers.buffer() + chunk_idx % num_recv_buffer_tokens);

            // Copy `topk_idx` and `topk_weights`
            #pragma unroll 4
            for (int idx = recv_thread_id_in_rank; idx < num_recv_tokens * num_topk; idx += 32 * num_recv_warps_per_rank) {
                int chunk_idx = idx / num_topk, token_topk_idx = idx % num_topk;
                int token_idx_in_buffer = (cached_channel_head_idx + chunk_idx) % num_recv_buffer_tokens;
                auto recv_idx = static_cast<int64_t>(total_offset + chunk_idx) * num_topk + token_topk_idx;
                auto buffer_idx = token_idx_in_buffer * num_topk + token_topk_idx;
                recv_topk_idx[recv_idx] = ld_nc_global(channel_topk_idx_buffers.buffer() + buffer_idx);
                recv_topk_weights[recv_idx] = ld_nc_global(channel_topk_weights_buffers.buffer() + buffer_idx);
            }

            // Copy `x_scales`
            #pragma unroll 4
            for (int i = recv_thread_id_in_rank; i < num_recv_tokens * num_scales; i += 32 * num_recv_warps_per_rank) {
                int chunk_idx = i / num_scales, scales_idx = i % num_scales;
                int token_idx_in_buffer = (cached_channel_head_idx + chunk_idx) % num_recv_buffer_tokens;
                recv_x_scales[static_cast<int64_t>(total_offset + chunk_idx) * num_scales + scales_idx] =
                    ld_nc_global(channel_x_scales_buffers.buffer() + token_idx_in_buffer * num_scales + scales_idx);
            }

            // Move queue
            cached_channel_head_idx += num_recv_tokens;
            total_offset += num_recv_tokens;
            asm volatile("bar.sync %0, %1;" ::"r"(responsible_rank), "r"(num_threads_per_rank));
            if (recv_warp_id_in_rank == num_recv_warps_per_rank - 1 and elect_one_sync())
                st_relaxed_sys_global(channel_head_idx.buffer(), cached_channel_head_idx);

            // Exit
            num_tokens_to_recv -= num_recv_tokens;
        }
    }

    // Clean unused `recv_topk_idx` as -1
    if (num_worst_tokens > 0) {
        auto rank_prefix_matrix = static_cast<int*>(buffer_ptrs[rank]);
        const auto num_recv_tokens = rank_prefix_matrix[(kNumRanks - 1) * kNumRanks + rank];
        const auto clean_start = num_recv_tokens * num_topk + sm_id * kNumThreads;
        const auto clean_end = num_worst_tokens * num_topk;
        const auto clean_stride = num_sms * kNumThreads;
        #pragma unroll
        for (int i = clean_start + thread_id; i < clean_end; i += clean_stride)
            recv_topk_idx[i] = -1;
    }
}

void dispatch(void* recv_x,
              float* recv_x_scales,
              int* recv_src_idx,
              topk_idx_t* recv_topk_idx,
              float* recv_topk_weights,
              int* recv_channel_offset,
              int* send_head,
              const void* x,
              const float* x_scales,
              const topk_idx_t* topk_idx,
              const float* topk_weights,
              const bool* is_token_in_rank,
              const int* channel_prefix_matrix,
              int num_tokens,
              int num_worst_tokens,
              int hidden_int4,
              int num_topk,
              int num_experts,
              int num_scales,
              int scale_token_stride,
              int scale_hidden_stride,
              void** buffer_ptrs,
              int rank,
              int num_ranks,
              cudaStream_t stream,
              int num_sms,
              int num_max_send_tokens,
              int num_recv_buffer_tokens) {
    constexpr int kNumThreads = 768;
    constexpr int kNumTMABytesPerWarp = 8192;
#ifndef DISABLE_SM90_FEATURES
    constexpr int smem_size = kNumTMABytesPerWarp * (kNumThreads / 32);
#endif

    // Make sure never OOB
    EP_HOST_ASSERT(static_cast<int64_t>(num_scales) * scale_hidden_stride < std::numeric_limits<int>::max());

#define DISPATCH_LAUNCH_CASE(ranks)                                      \
    {                                                                    \
        auto kernel = dispatch<ranks, kNumThreads, kNumTMABytesPerWarp>; \
        SET_SHARED_MEMORY_FOR_TMA(kernel);                               \
        LAUNCH_KERNEL(&cfg,                                              \
                      kernel,                                            \
                      reinterpret_cast<int4*>(recv_x),                   \
                      recv_x_scales,                                     \
                      recv_src_idx,                                      \
                      recv_topk_idx,                                     \
                      recv_topk_weights,                                 \
                      recv_channel_offset,                               \
                      send_head,                                         \
                      reinterpret_cast<const int4*>(x),                  \
                      x_scales,                                          \
                      topk_idx,                                          \
                      topk_weights,                                      \
                      is_token_in_rank,                                  \
                      channel_prefix_matrix,                             \
                      num_tokens,                                        \
                      num_worst_tokens,                                  \
                      hidden_int4,                                       \
                      num_topk,                                          \
                      num_experts,                                       \
                      num_scales,                                        \
                      scale_token_stride,                                \
                      scale_hidden_stride,                               \
                      buffer_ptrs,                                       \
                      rank,                                              \
                      num_max_send_tokens,                               \
                      num_recv_buffer_tokens);                           \
    }                                                                    \
    break

    // Even-numbered blocks for sending, odd-numbered blocks for receiving.
    EP_HOST_ASSERT(num_sms % 2 == 0);
    SETUP_LAUNCH_CONFIG(num_sms, kNumThreads, stream);
    SWITCH_RANKS(DISPATCH_LAUNCH_CASE);
#undef DISPATCH_LAUNCH_CASE
}
```

### 2.2 `Buffer<T>`：顺序切分一块大内存（`csrc/kernels/legacy/buffer.cuh`）

```cpp
template <typename dtype_t>
struct Buffer {
private:
    uint8_t* ptr;

public:
    int64_t total_bytes;

    __device__ __forceinline__ Buffer() : ptr(nullptr), total_bytes(0) {}

    __device__ __forceinline__ Buffer(void*& gbl_ptr, int num_elems, int offset = 0) {
        total_bytes = num_elems * sizeof(dtype_t);
        ptr = static_cast<uint8_t*>(gbl_ptr) + offset * sizeof(dtype_t);
        gbl_ptr = static_cast<uint8_t*>(gbl_ptr) + total_bytes;
    }

    __device__ __forceinline__ Buffer advance_also(void*& gbl_ptr) {
        gbl_ptr = static_cast<uint8_t*>(gbl_ptr) + total_bytes;
        return *this;
    }

    __device__ __forceinline__ dtype_t* buffer() { return reinterpret_cast<dtype_t*>(ptr); }

    __device__ __forceinline__ dtype_t& operator[](int idx) { return buffer()[idx]; }
};
```

### 2.3 拷贝宏与内存序原语（`csrc/kernels/legacy/utils.cuh`）

```cpp
#define UNROLLED_WARP_COPY(UNROLL_FACTOR, LANE_ID, N, DST, SRC, LD_FUNC, ST_FUNC)                                                     \
    {                                                                                                                                 \
        constexpr int kLoopStride = 32 * (UNROLL_FACTOR);                                                                             \
        typename std::remove_reference<decltype(LD_FUNC((SRC) + 0))>::type unrolled_values[(UNROLL_FACTOR)];                          \
        auto __src = (SRC);                                                                                                           \
        auto __dst = (DST);                                                                                                           \
        for (int __i = (LANE_ID); __i < ((N) / kLoopStride) * kLoopStride; __i += kLoopStride) {                                      \
            _Pragma("unroll") for (int __j = 0; __j < (UNROLL_FACTOR); ++__j) unrolled_values[__j] = LD_FUNC(__src + __i + __j * 32); \
            _Pragma("unroll") for (int __j = 0; __j < (UNROLL_FACTOR); ++__j) ST_FUNC(__dst + __i + __j * 32, unrolled_values[__j]);  \
        }                                                                                                                             \
        {                                                                                                                             \
            int __i = ((N) / kLoopStride) * kLoopStride + (LANE_ID);                                                                  \
            _Pragma("unroll") for (int __j = 0; __j < (UNROLL_FACTOR); ++__j) {                                                       \
                if (__i + __j * 32 < (N)) {                                                                                           \
                    unrolled_values[__j] = LD_FUNC(__src + __i + __j * 32);                                                           \
                }                                                                                                                     \
            }                                                                                                                         \
            _Pragma("unroll") for (int __j = 0; __j < (UNROLL_FACTOR); ++__j) {                                                       \
                if (__i + __j * 32 < (N)) {                                                                                           \
                    ST_FUNC(__dst + __i + __j * 32, unrolled_values[__j]);                                                            \
                }                                                                                                                     \
            }                                                                                                                         \
        }                                                                                                                             \
    }

__device__ __forceinline__ void st_release_sys_global(const int* ptr, int val) {
    asm volatile("st.release.sys.global.s32 [%0], %1;" ::"l"(ptr), "r"(val) : "memory");
}

__device__ __forceinline__ int ld_acquire_sys_global(const int* ptr) {
    int ret;
    asm volatile("ld.acquire.sys.global.s32 %0, [%1];" : "=r"(ret) : "l"(ptr));
    return ret;
}

// `st.global.L1::no_allocate` will be translated into `ST.E.NA.[width]` in SASS
#ifndef DISABLE_AGGRESSIVE_PTX_INSTRS
#define ST_NA_FUNC "st.global.L1::no_allocate"
#else
#define ST_NA_FUNC "st.global"
#endif

template <>
__device__ __forceinline__ void st_na_global(const int4* ptr, const int4& value) {
    asm volatile(ST_NA_FUNC ".v4.s32 [%0], {%1, %2, %3, %4};" ::"l"(ptr), "r"(value.x), "r"(value.y), "r"(value.z), "r"(value.w));
}
```

### 2.4 启动配置（`csrc/kernels/legacy/launch.cuh`，SM90 路径）

```cpp
#define SETUP_LAUNCH_CONFIG(num_sms, num_threads, stream)                       \
    cudaLaunchConfig_t cfg = {(num_sms), (num_threads), 0, stream, nullptr, 0}; \
    cudaLaunchAttribute attr[2];                                                \
    attr[0].id = cudaLaunchAttributeCooperative;                                \
    attr[0].val.cooperative = 1;                                                \
    attr[1].id = cudaLaunchAttributeClusterDimension;                           \
    attr[1].val.clusterDim.x = (num_sms % 2 == 0 ? 2 : 1);                      \
    attr[1].val.clusterDim.y = 1;                                               \
    attr[1].val.clusterDim.z = 1;                                               \
    cfg.attrs = attr;                                                           \
    cfg.numAttrs = 2

#define SET_SHARED_MEMORY_FOR_TMA(kernel)                                                                                \
    EP_HOST_ASSERT(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size) == cudaSuccess); \
    cfg.dynamicSmemBytes = smem_size;
```

### 2.5 host 端分配输出并调用（`csrc/legacy/buffer.hpp` L602–L665）

```cpp
        // Allocate new tensors
        auto recv_x = torch::empty({num_recv_tokens, hidden}, x.options());
        auto recv_src_idx = torch::empty({num_recv_tokens}, dtype(torch::kInt32).device(torch::kCUDA));
        auto recv_topk_idx = std::optional<torch::Tensor>(), recv_topk_weights = std::optional<torch::Tensor>(),
             recv_x_scales = std::optional<torch::Tensor>();
        auto recv_channel_prefix_matrix = torch::empty({num_ranks, num_channels}, dtype(torch::kInt32).device(torch::kCUDA));
        auto send_head = torch::empty({num_tokens, num_ranks}, dtype(torch::kInt32).device(torch::kCUDA));

        // Assign pointers
        topk_idx_t* recv_topk_idx_ptr = nullptr;
        float* recv_topk_weights_ptr = nullptr;
        float* recv_x_scales_ptr = nullptr;
        if (topk_idx.has_value()) {
            recv_topk_idx = torch::empty({num_recv_tokens, num_topk}, topk_idx->options());
            recv_topk_weights = torch::empty({num_recv_tokens, num_topk}, topk_weights->options());
            recv_topk_idx_ptr = recv_topk_idx->data_ptr<topk_idx_t>();
            recv_topk_weights_ptr = recv_topk_weights->data_ptr<float>();
        }
        if (x_scales.has_value()) {
            recv_x_scales = x_scales->dim() == 1 ? torch::empty({num_recv_tokens}, x_scales->options())
                                                 : torch::empty({num_recv_tokens, num_scales}, x_scales->options());
            recv_x_scales_ptr = static_cast<float*>(recv_x_scales->data_ptr());
        }

        // Dispatch
        EP_HOST_ASSERT(
            num_ranks * num_ranks * sizeof(int) +                                                                     // Size prefix matrix
                num_channels * num_ranks * sizeof(int) +                                                              // Channel start offset
                num_channels * num_ranks * sizeof(int) +                                                              // Channel end offset
                num_channels * num_ranks * sizeof(int) * 2 +                                                          // Queue head and tail
                num_channels * num_ranks * config.num_max_nvl_chunked_recv_tokens * hidden * recv_x.element_size() +  // Data buffer
                num_channels * num_ranks * config.num_max_nvl_chunked_recv_tokens * sizeof(int) +                     // Source index buffer
                num_channels * num_ranks * config.num_max_nvl_chunked_recv_tokens * num_topk * sizeof(topk_idx_t) +   // Top-k index buffer
                num_channels * num_ranks * config.num_max_nvl_chunked_recv_tokens * num_topk * sizeof(float) +        // Top-k weight buffer
                num_channels * num_ranks * config.num_max_nvl_chunked_recv_tokens * sizeof(float) * num_scales        // FP8 scale buffer
            <= num_nvl_bytes);
        intranode::dispatch(recv_x.data_ptr(),
                            recv_x_scales_ptr,
                            recv_src_idx.data_ptr<int>(),
                            recv_topk_idx_ptr,
                            recv_topk_weights_ptr,
                            recv_channel_prefix_matrix.data_ptr<int>(),
                            send_head.data_ptr<int>(),
                            x.data_ptr(),
                            x_scales_ptr,
                            topk_idx_ptr,
                            topk_weights_ptr,
                            is_token_in_rank.data_ptr<bool>(),
                            channel_prefix_matrix.data_ptr<int>(),
                            num_tokens,
                            num_worst_tokens,
                            static_cast<int>(hidden * recv_x.element_size() / sizeof(int4)),
                            num_topk,
                            num_experts,
                            num_scales,
                            scale_token_stride,
                            scale_hidden_stride,
                            buffer_ptrs_gpu,
                            rank,
                            num_ranks,
                            comm_stream,
                            config.num_sms,
                            config.num_max_nvl_chunked_send_tokens,
                            config.num_max_nvl_chunked_recv_tokens);
```

kernel 参数 `recv_channel_offset` 在 host 端叫 `recv_channel_prefix_matrix`，是同一个 tensor。

## 3. 角色分工：每个线程「是谁、负责哪条通道」

一个线程的身份由三个量决定：

| 变量 | 由什么决定 | 含义 |
|---|---|---|
| `is_sender` = `sm_id % 2 == 0` | block 编号奇偶 | 发还是收 |
| `responsible_channel` = `sm_id / 2` | block 编号 | 处理哪一段 token（第几个 channel），本身不区分发收 |
| `responsible_rank` = `thread_id / 96` | 线程编号 | 对端 rank：发送 block 里是目标 rank，接收 block 里是来源 rank；可以等于自己 |

**channel 是对 token 的切分**：本卡 `num_tokens` 个 token 用 `get_channel_task_range` 均分成 C 段连续区间。每个 channel 配一对 block：

| block（`sm_id`） | 0 | 1 | 2 | 3 | … | 18 | 19 |
|---|---|---|---|---|---|---|---|
| 角色 | 发 | 收 | 发 | 收 | … | 发 | 收 |
| `responsible_channel` | 0 | 0 | 1 | 1 | … | 9 | 9 |

SM90 路径的启动配置把 cluster 大小设成 2（`num_sms` 为偶数时），block `2c` 和 `2c+1` 属于同一个 cluster，硬件会把它们放在同一个 GPC 上同时调度。

**block 内按对端 rank 分组**（P = 8）：

| `thread_id` | 0–95 | 96–191 | … | 672–767 |
|---|---|---|---|---|
| warp | 0–2 | 3–5 | … | 21–23 |
| `responsible_rank` | 0 | 1 | … | 7 |

所以每对（channel, 对端 rank）有 3 个 warp 专门负责。一个具体例子（rank 3 上）：

| 线程位置 | 做什么 |
|---|---|
| block 4（发送端，channel 2），线程 200（`responsible_rank` = 2） | 把 rank 3 第 2 段 token 中发往 rank 2 的那些，写进 rank 2 显存里的队列（channel 2，来源 3） |
| block 5（接收端，channel 2），线程 200（`responsible_rank` = 2） | 从 rank 3 自己显存里的队列（channel 2，来源 2）取数据，写进 `recv_x` 里 rank 2 那一段的 channel 2 部分 |

**这 3 个 warp 的分工**（第 6、7 节细讲）：

| 阶段 | 发送端 | 接收端 |
|---|---|---|
| 轮询 | 每个 warp 各自读 head 判断空位 | 只有组内 0 号线程读 tail，经 shared memory + `bar.sync` 广播 |
| 记录 | `send_head`：按 `token_idx % 3` 轮流 | – |
| 搬 x | 按 `tail % 3` 轮流，每个 token 只由一个 warp 拷 | 按 `chunk_idx % 3` 轮流 |
| 小元数据 | 拷 x 的那个 warp 顺带拷 | 96 个线程一起分摊 |
| 发布 | `bar.sync` 后 warp 0 写 tail | `bar.sync` 后最后一个 warp 写 head |

用 3 个 warp 而不是 1 个，是为了让每个队列同时有 3 个 token 在搬，吞吐约提高 3 倍。3 是 `768 / P / 32` 算出来的，P = 4 时就是 6 个 warp。

## 4. 缓冲区布局

### 4.1 `Buffer<T>` 做了什么

构造函数第一个参数是 `void*&`，**按引用传入**：

1. 返回的视图指向 `gbl_ptr + offset × sizeof(T)`；
2. 把外面的 `ptr` 推进 `num_elems × sizeof(T)`，留给下一个表。

```cpp
auto channel_start_offset = Buffer<int>(ptr, num_channels_total, channel_rank_offset);
```

等价于：

```cpp
int* start_offset_table = (int*)ptr;                         // 整张表 [C][P]
int* channel_start_offset = &start_offset_table[c * P + target_rank];
ptr = (char*)ptr + C * P * sizeof(int);                      // 跳过这张表
```

所以 `channel_start_offset.buffer()` 指向**单个 int**，不是整张表的开头。L271–L274 四行参数完全相同，得到的却是四个不同的地址：

| 语句（P = 8，C = 10，channel 2，target_rank 3） | 构造时 `ptr` | 得到的地址 | 构造后 `ptr` |
|---|---|---|---|
| `channel_start_offset` | B | B + 76 | B + 320 |
| `channel_end_offset` | B + 320 | B + 396 | B + 640 |
| `channel_head_idx` | B + 640 | B + 716 | B + 960 |
| `channel_tail_idx` | B + 960 | B + 1036 | B + 1280 |

每张表 C × P = 80 个 int = 320 字节；`channel_rank_offset = 2 × 8 + 3 = 19`，19 × 4 = 76。四张表形状相同、格子编号相同，相当于把「一个通道的 start/end/head/tail」这个结构体拆成了四张表（SoA）。

### 4.2 接收 rank R 的 NVLink buffer 全貌

```text
buffer_ptrs[R] ─►
[rank_prefix_matrix  P*P int]        ← notify 写入；ptr 从这之后开始
[start_offset        C*P int]  ┐
[end_offset          C*P int]  │ notify 末尾清零的 C*P*4 个 int
[head_idx            C*P int]  │
[tail_idx            C*P int]  ┘
[x_buffers           C*P*N*hidden_int4 int4]   每个 (channel, 来源) 一个长度 N 的环
[src_idx_buffers     C*P*N int]
[topk_idx_buffers    C*P*N*topk topk_idx_t]
[topk_weights        C*P*N*topk float]
[x_scales            C*P*N*scales float]
```

host 端 L627–L637 的断言就是把这张图的字节数加起来，要求不超过 `num_nvl_bytes`。以 P = 8、C = 10、N = 256、hidden 7168 bf16 为例，光 `x_buffers` 就是 `80 × 256 × 14336 ≈ 293.6 MB`。

### 4.3 发送端和接收端怎样指到同一格

```cpp
auto ptr = ... buffer_ptrs[is_sender ? responsible_rank : rank] + P*P*sizeof(int);
int target_rank = is_sender ? rank : responsible_rank;
auto channel_rank_offset = responsible_channel * kNumRanks + target_rank;
```

| 角色 | 基地址 | 格子编号 |
|---|---|---|
| 发送端：rank S 发给 R | `buffer_ptrs[R]`（对端显存，NVLink 远程写） | `c × P + S` |
| 接收端：rank R 收 S 的 | `buffer_ptrs[R]`（自己显存） | `c × P + S` |

两边按同样顺序构造同样大小的 `Buffer`，所以每张表的起点一致，最终指向 R 的显存里（channel c，来源 S）那一格。数据和 head/tail 都放在接收端，注释里写的 "Stored on the receiver side" 就是这个意思。

### 4.4 为什么是 `[channel][rank]`，和 `channel_prefix_matrix` 的 `[rank][channel]` 相反

| 结构 | 下标 | 外层维度对应 |
|---|---|---|
| `channel_prefix_matrix` | `dst × C + c` | notify 里一个 block 负责一个 dst rank，写出一整行 |
| buffer 里的表和环 | `c × P + rank` | dispatch 里一个 block 负责一个 channel |

规律是：**外层维度 = 一个 block 负责的东西**，这样同一个 block 访问的数据在内存里连续。接收 block `2c+1` 用到的 8 个 head/tail 是相邻的 8 个 int；channel c 的 8 个环拼成一整块连续区域，不同 channel（不同 SM）各用各的区域。代码里没有注释说明原因，这种局部性带来的性能差别应该不大，可以理解成一致的布局习惯；正确性上两种排法都可以，只要收发双方一致。

## 5. TMA 准备与 shared memory（L293–L307）

每个 warp 分到 `kNumTMABytesPerWarp = 8192` 字节 shared memory：前半个 hidden 作为 TMA 中转区，紧接着放 8 字节 mbarrier。这个数是 host 端写死的常量，经过三步生效：

1. host：`smem_size = 8192 × 24 = 196608`（192 KB）；
2. `SET_SHARED_MEMORY_FOR_TMA`：`cudaFuncSetAttribute(..., cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size)` 把动态 shared memory 上限从默认 48 KB 调高，并设置 `cfg.dynamicSmemBytes`；
3. kernel：`extern __shared__ smem_buffer[]` 就是这 192 KB，warp w 从 `w × 8192` 开始。

**为什么每个 token 分两半搬**：一行 bf16 是 14336 字节，8 KB 放不下。为什么不给每个 warp 14 KB？因为 24 × 14 KB ≈ 336 KB，超过 H800 单 block 227 KB 的上限。227 KB / 24 ≈ 9.4 KB，一整行怎么都放不进，只能切开。代码固定切两半（`for i < 2`）。

| 方案 | 每 warp | 24 warp 合计 | 是否 ≤ 227 KB |
|---|---|---|---|
| 整行一次搬 | 14336 + 8 | ≈ 336 KB | 否 |
| 两半（现在） | 8192 | 192 KB | 是 |

断言 `half_hidden_bytes + 8 <= 8192`：hidden 7168 bf16 时 7168 + 8 = 7176，满足。按这个断言反推，bf16 下 hidden 大约不能超过 8176；hidden = 8192 的 bf16 会得到 8192 + 8 > 8192，按公式会触发断言（这是从代码推算的，没有在 GPU 上验证）。FP8 每元素 1 字节，空间宽松得多。编译时定义 `DISABLE_SM90_FEATURES` 则不用 TMA，接收端退回 `UNROLLED_WARP_COPY`（L492）。

## 6. 发送端逐段（L309–L412）

### 6.1 告诉接收端本 channel 的范围（L320–L325）

组内 warp 0 选一个 lane，从 `channel_prefix_matrix[dst]` 取本 channel 的起止位置，编码成 `-value-1` 写进对端：

```text
start = c > 0 ? cpm[dst][c-1] : 0     → 写 -start-1
end   = cpm[dst][c]                    → 写 -end-1
```

编码是为了区分「0 个 token」和「还没写」：notify 末尾把这块清成了 0，接收端看到 0 就继续等；起点 0 会编码成 −1，不会和 0 混淆。

### 6.2 本 channel 的 token 区间（L330）

`get_channel_task_range(num_tokens, C, c)`：每段 `ceil(T/C)` 个，channel c 负责 `[c × ceil(T/C), min(...+ceil(T/C), T))`。token 少于 channel 数时，后面的 channel 区间为空，start = end，接收端 `num_tokens_to_recv = 0` 直接跳过循环。

### 6.3 等空位（L338–L352）

```cpp
int num_used_slots = cached_channel_tail_idx - ld_volatile_global(channel_head_idx.buffer());
if (num_recv_buffer_tokens - num_used_slots >= num_max_send_tokens) break;
```

每个 warp 选一个 lane，用 volatile 读对端的 head。剩余空位能放下「最坏情况的一整批」才继续。注释说只按最坏情况算，是因为精确统计这一批实际会发几个 token 太费时间。3 个 warp 各自读，读到的 head 可能不同，但每个 warp 只在确认有空位后才写，所以安全。

### 6.4 分批发送：为什么要 `chunk_token_idx < num_max_send_tokens`（L354–L405）

```cpp
while (chunk_token_idx < num_max_send_tokens and token_idx < token_end_idx)
```

- `token_idx < token_end_idx`：不超出本 channel 区间。
- `chunk_token_idx < num_max_send_tokens`：
  1. **与 6.3 的空位检查对应**：只确认过有 `num_max_send_tokens` 个空位，多发就会覆盖接收端还没读走的槽位。`chunk_token_idx` 只在真正发送时加 1（跳过的 token 在 L364 `continue`），所以它精确等于这一批占用的槽位数。
  2. **定期发布 tail**：tail 只在每批结束时更新。如果一口气发完整个 channel 才更新，队列比 token 少时会死锁：发送端写满后等空位，接收端看不到 tail 不会读、不会释放，双方互等。即使不死锁，也失去了「接收端搬上一批、发送端写下一批」的流水线。
  3. 逐个 token 发布也不好：每次发布要一次 named barrier 加一次跨 NVLink 的 release 写。按批是折中。

### 6.5 记录 `send_head`（L358–L360）

```cpp
if (token_idx % num_send_warps_per_rank == send_warp_id_in_rank and elect_one_sync())
    send_head[token_idx * kNumRanks + responsible_rank] =
        is_token_in_rank[token_idx * kNumRanks + responsible_rank] ? cached_channel_tail_idx : -1;
```

- 写入值：发给这个 rank 的 token 写 channel 内序号（自增**之前**的 tail），不发的写 −1。放在跳过判断之前，所以本 channel 的每个 token 都写一次。
- 谁写：3 个 warp 都跑这个循环，按 `token_idx % 3` 选一个 warp，再 `elect_one_sync` 选一个 lane，每个 token 只写一次。
- 注释提醒：写 `send_head` 的 warp（按 `token_idx % 3`）和拷数据的 warp（按 `tail % 3`）不一定是同一个。每个 warp 维护同一份 tail 计数，两种分工互不冲突。
- 用途：`send_head` 是本卡的输出 tensor `[num_tokens, num_ranks]`，dispatch 里不再读，留给 combine（第 10 节）。

例：channel 负责 token `[10, 16)`，目标 rank 2，发往 rank 2 的是 11、13、14：

| token | 发给 rank 2？ | 写 `send_head` 的 warp（`token % 3`） | 写入值 |
|---|---|---|---|
| 10 | 否 | 1 | −1 |
| 11 | 是 | 2 | 0 |
| 12 | 否 | 0 | −1 |
| 13 | 是 | 1 | 1 |
| 14 | 是 | 2 | 2 |
| 15 | 否 | 0 | −1 |

### 6.6 分配槽位、决定谁拷（L369–L370）

```cpp
int dst_slot_idx = (cached_channel_tail_idx++) % num_recv_buffer_tokens;
if (cached_channel_tail_idx % num_send_warps_per_rank == send_warp_id_in_rank) {
```

后置自增：用自增前的序号对 N 取模得槽位；用自增后的值对 3 取模决定哪个 warp 拷。3 个 warp 都执行 L369，所以 tail 计数一致；只有一个进入 if 块，另外两个直接去处理下一个 token。

| 序号（自增前） | 槽位（N = 256） | 自增后 `% 3` | 拷贝的 warp |
|---|---|---|---|
| 0 | 0 | 1 | 1 |
| 1 | 1 | 2 | 2 |
| 2 | 2 | 0 | 0 |

### 6.7 打包一个 token（L371–L400）

一个槽位分布在 5 个并列数组里，下标都是 `dst_slot_idx`：

| 内容 | 每 token 大小（hidden 7168, topk 8） | 谁拷 | 说明 |
|---|---|---|---|
| `x` | 14336 B（bf16） | 32 个 lane，`UNROLLED_WARP_COPY` | 真正的数据，占传输量 99% 以上 |
| `src_idx` | 4 B | 1 个 lane | token 在源 rank 的下标，combine 送回用 |
| `topk_idx` | 8 × 8 B | 前 `num_topk` 个 lane | 转成目标 rank 的本地 expert 编号，非本地记 −1 |
| `topk_weights` | 8 × 4 B | 同上 | 对应 expert 非本地则置 0 |
| `x_scales` | 56 × 4 B（仅 FP8） | 32 个 lane 按步长 32 循环 | 按 stride 读，支持非连续 scales |

**x 的地址**：`x` 已被 `reinterpret_cast` 成 `const int4*`，指针运算以 16 字节为单位。`x` 是行优先连续的 `[num_tokens, hidden]`，按 int4 看就是 `[num_tokens][hidden_int4]`，所以第 `token_idx` 行起点是 `x + token_idx × hidden_int4`。槽位同理：`channel_x_buffers.buffer() + dst_slot_idx × hidden_int4`。hidden 7168 bf16：`hidden_int4 = 7168 × 2 / 16 = 896`，token 5 的起点是 `x + 4480`（以 int4 计），即第 71680 字节。

**`UNROLLED_WARP_COPY(5, ...)` 展开**：

```cpp
int4 vals[5];
for (int i = lane_id; i < (N / 160) * 160; i += 160) {
    for (int j = 0; j < 5; ++j) vals[j] = __ldg(src + i + j * 32);       // 先连发 5 个读
    for (int j = 0; j < 5; ++j) st_na_global(dst + i + j * 32, vals[j]); // 再连发 5 个写
}
int i = (N / 160) * 160 + lane_id;                                       // 尾巴带边界检查
for (int j = 0; j < 5; ++j) if (i + j * 32 < N) vals[j] = __ldg(src + i + j * 32);
for (int j = 0; j < 5; ++j) if (i + j * 32 < N) st_na_global(dst + i + j * 32, vals[j]);
```

- 一轮：每 lane 5 个 int4 = 80 字节，32 lane = 2560 字节 = 160 个 int4。
- 下标 `i + j × 32`：同一个 j 上 32 个 lane 访问连续 512 字节，可以合并访存。
- 先读 5 个再写 5 个：5 个读请求同时在路上，延迟互相重叠。
- 896 个 int4：主循环 5 轮搬 800 个；尾巴 `i = 800 + lane`，j = 0、1、2 覆盖 800–895，j = 3、4 越界跳过。每 lane 共 28 个。
- `__ldg`：只读缓存路径读本地 `x`。`st_na_global`：`st.global.L1::no_allocate`，目标在对端 GPU，本 SM 不会再读，不占 L1。

**topk 本地化**（256 expert、8 rank，目标 rank 2 拥有 `[64, 96)`）：

| topk 位置 | 原始 expert | 原始权重 | 转换后 expert | 转换后权重 |
|---|---|---|---|---|
| 0 | 3 | 0.4 | −1 | 0 |
| 1 | 70 | 0.3 | 6 | 0.3 |
| 2 | 95 | 0.2 | 31 | 0.2 |
| 3 | 130 | 0.1 | −1 | 0 |

`lane_id < num_topk`：一个 lane 负责一个 topk 位置。lane ≥ `num_topk` 如果也执行，会读到下一个 token 的 topk、写到下一个槽位，所以必须挡掉。`num_topk <= 32`（L251 device 断言）保证一个 warp 一次覆盖。

**为什么 topk 和 scales 不用 `UNROLLED_WARP_COPY`**：数据只有几个到几十个元素，一条指令就够，展开没有收益；topk 每个元素要转换，权重还依赖同一位置编号的转换结果，宏只能原样拷贝；scales 按 stride 读，源不一定连续，宏要求连续。

### 6.8 发布 tail（L409–L411）

```cpp
asm volatile("bar.sync %0, %1;" ::"r"(responsible_rank), "r"(num_threads_per_rank));
if (send_warp_id_in_rank == 0 and elect_one_sync())
    st_release_sys_global(channel_tail_idx.buffer(), cached_channel_tail_idx);
```

1. `bar.sync responsible_rank, 96`：等本组 3 个 warp 都写完这一批（数据是轮流拷的，warp 0 不能只看自己）。named barrier 原理见 9.5。
2. 组内 warp 0 的一个 lane 用 `st.release.sys.global` 把累计序号写进接收端显存的 tail。release：之前的写入全部可见后，这次写入才可见；`.sys`：作用范围包括其他 GPU。PTX 的 release 是累积的，经过 `bar.sync` 后其他 warp 的写入也在保证范围内。

## 7. 接收端逐段（L413–L533）

### 7.1 算出写入起点（L424–L441）

1. `rank_offset = src > 0 ? rank_prefix_matrix[src−1][rank] : 0`，从自己 buffer 开头读（notify 写进去的，只用本列）。
2. 选一个 lane 用 volatile 轮询 `start_offset` / `end_offset` 直到非 0，解码 `-v-1`。
3. 组内 warp 0 把 channel 起点写进 `recv_channel_offset[src][c]`，留给 combine。
4. `num_tokens_to_recv = end − start`；`total_offset = rank_offset + start`；`__shfl_sync` 广播给整个 warp。

### 7.2 读 tail：为什么只让一个线程读并广播（L448–L471）

```cpp
while (recv_thread_id_in_rank == 0) {
    cached_channel_tail_idx = ld_acquire_sys_global(channel_tail_idx.buffer());
    if (cached_channel_head_idx != cached_channel_tail_idx) {
        shared_channel_tail_idx[responsible_rank] = cached_channel_tail_idx;
        break;
    }
    ...
}
asm volatile("bar.sync %0, %1;" ::"r"(responsible_rank), "r"(num_threads_per_rank));
cached_channel_tail_idx = shared_channel_tail_idx[responsible_rank];
```

组内 0 号线程用 acquire 读到新 tail 后写进 shared memory，`bar.sync` 后 3 个 warp 读到**同一个** tail。注释强调「接收端必须保证不同 warp 持有的 tail 相同」：3 个 warp 按 `chunk_idx` 分工，tail 不一致就会漏拷或重拷，head 也没法一致推进。发送端没有这个要求，是因为每个 warp 只用 head 判断「能不能写」，读到旧 head 只会多等一会儿。

### 7.3 搬 `[head, tail)`，行号怎么来（L473–L521）

```cpp
int token_idx_in_buffer = (cached_channel_head_idx + chunk_idx) % num_recv_buffer_tokens;
auto shifted_recv_x_int4 = recv_x + static_cast<int64_t>(total_offset + chunk_idx) * hidden_int4;
...
cached_channel_head_idx += num_recv_tokens;
total_offset += num_recv_tokens;
```

`total_offset` 初值是 `rank_offset + start`，之后每轮和 head 一起加 `num_recv_tokens`，所以始终有 `total_offset = rank_offset + start + head`：

```text
recv_x 行号 = total_offset + chunk_idx
           = rank_offset + channel 起点 + (head + chunk_idx)
                                          └── channel 内序号 ──┘
槽位        = (head + chunk_idx) % N      ← 槽位取模，行号不取模
```

「channel 内第几个」没有存成数组：发送端用 `cached_channel_tail_idx` 计数，接收端用 `cached_channel_head_idx` 跟踪，两边一一对应；唯一落盘的是发送端的 `send_head`。

| 内容 | 分工 | 搬运方式 |
|---|---|---|
| `x` | 按 token 在 3 个 warp 间轮流（`chunk_idx += 3`） | SM90：TMA 两半，global → shared → global；否则 `UNROLLED_WARP_COPY` |
| `src_idx`、`topk_idx`、`topk_weights`、`x_scales` | 96 个线程按步长 96 分摊 | `ld_nc_global` 读 |

TMA 每一半的顺序：`tma_store_wait<0>()` 等上一次 store 读完中转区 → `tma_load_1d` 把半行搬进 shared memory → `mbarrier_wait` 等到齐 → `tma_store_1d` 写到 `recv_x`。两半串行，但同一时刻还有其他 warp 在搬别的 token，SM 上并发足够。

### 7.4 释放槽位（L524–L531）

`bar.sync` 后，组内**最后一个** warp 用 `st_relaxed_sys_global` 写 head。发送端在 6.3 读到新 head 就知道这些槽位可以复用。收满 `num_tokens_to_recv` 后退出。

## 8. `num_worst_tokens > 0` 的收尾（L535–L545）

`recv_x` 按最坏情况分配时（见 09 第 6.3 节），真实行数之后的行没有数据。所有 block 的所有线程按 `num_sms × 768` 的步长，把 `recv_topk_idx` 中 `[真实行数 × topk, num_worst_tokens × topk)` 填成 −1，下游看到 −1 就跳过。真实行数从自己 buffer 开头的 `rank_prefix_matrix[P−1][rank]` 读。

## 9. 读这个 kernel 需要的 GPU 背景

### 9.1 `__launch_bounds__(768, 1)`

```text
__launch_bounds__(maxThreadsPerBlock, minBlocksPerMultiprocessor)
```

告诉编译器「每 block 最多 768 线程，每 SM 至少放下 1 个 block」，编译器据此限制每线程寄存器数，保证能启动。H800 每 SM 65536 个寄存器：65536 / 768 ≈ 85，按分配粒度实际约 80 个。

| 情况 | 每线程寄存器 | 768 线程需求 | 结果 |
|---|---|---|---|
| 不写，编译器分 128 个 | 128 | 98304 > 65536 | 启动失败：`too many resources requested for launch` |
| `__launch_bounds__(768, 1)` | ≤ 约 80 | ≤ 65536 | 能启动，多出的变量 spill 到 local memory |
| 假如写 `(768, 2)` | ≤ 约 40 | 2 × 768 × 40 ≤ 65536 | 大量 spill，更慢 |

`notify_dispatch` 只有 128 线程（65536 / 128 = 512，远高于单线程上限 255），不写也不会超。

### 9.2 grid、block、SM

| 软件概念 | 是什么 | 对应硬件 |
|---|---|---|
| grid | 一次 kernel 启动的全部 block | 整张 GPU |
| block | 一组线程，共享 shared memory，可 `__syncthreads` | 被调度到某一个 SM，执行完前不迁移 |
| warp | 32 线程同步执行 | SM 内的 warp 调度器 |

grid 大小由程序员定，和 SM 数无关；放不下的 block 排队。DeepEP 写 `sm_id = blockIdx.x`、`num_sms = gridDim.x`，严格说是 block 编号和 block 数，block 0 不一定在 0 号 SM 上；这么命名是因为 dispatch 的设计就是一个 block 独占一个 SM。

H800（sm_90）每 SM 上限：32 个 block、2048 线程、65536 寄存器、228 KB shared memory（单 block 最多 227 KB）。一个 SM 同时放几个 block = 各项资源允许数的最小值：

| kernel | 每 block 线程 | 每 SM 最多 block | 主要约束 |
|---|---|---|---|
| `get_dispatch_layout` | 256 | 8 | 线程数 2048 / 256 |
| `notify_dispatch` | 128 | 16 | 线程数 2048 / 128 |
| `dispatch` / `combine` | 768 | 1 | shared memory 2 × 192 KB > 228 KB；寄存器 2 × 768 × 80 > 65536 |

### 9.3 `buffer_ptrs` 指向的是 HBM，不是 shared memory

分配器叫 `SharedMemoryAllocator`，这里的 shared 是「跨进程、跨 GPU 共享的显存」。它在普通模式下就是 `cudaMalloc`（fabric 模式用 `cuMemCreate`），再用 `cudaIpcGetMemHandle` / `cudaIpcOpenMemHandle` 把对端显存映射进本进程，最后把 `void*[8]` 拷进 `buffer_ptrs_gpu` 传给 kernel（`buffer.hpp` L134–L250，`csrc/utils/shared_memory.hpp`）。

| | `buffer_ptrs` 指向的 buffer | `smem_buffer`（`extern __shared__`） |
|---|---|---|
| 物理位置 | GPU 显存（HBM） | SM 片上 SRAM |
| 大小 | `num_nvl_bytes`，几百 MB 到 GB 级 | 每 block 192 KB |
| 谁能访问 | 同机 8 张卡所有进程（IPC 映射后） | 本 block |
| 生命周期 | Buffer 对象存在期间 | kernel 结束失效 |
| 用途 | 跨 GPU 传数据和信号 | TMA 中转 |

`buffer_ptrs[rank]` 是本卡 HBM；`buffer_ptrs[r]`（r ≠ rank）是第 r 张卡的 HBM，读写走 NVLink。

### 9.4 L1 不一致，所以需要 volatile / acquire / release

```text
SM 0: [block A] ─► L1（SM 0 独有）─┐
                                  ├─► L2（全 GPU 共享，一致性汇合点）─► HBM
SM 1: [block C] ─► L1（SM 1 独有）─┘
```

L1 是 global memory 的缓存，不能直接寻址。同一 SM 上的 block 共用一个 L1；不同 SM 都能访问同一地址，但各自可能在 L1 里缓存一份副本，**L1 之间不同步**。NVIDIA GPU 的 L1 对 global 写是直写，写入会落到 L2，所以典型问题是另一个 SM 读到自己 L1 里的旧副本，以及写入顺序被重排。dispatch 里每个 SM 只有一个 block，「跨 block」就等于「跨 SM」。

| 原语 | 位置 | 作用 |
|---|---|---|
| `st_na_global`（`L1::no_allocate`） | 发送端写数据 | 不在本 SM 的 L1 里占位（本 SM 不会再读；而且 192 KB 给了 shared memory，L1 本来就小） |
| `st_release_sys_global` | 发送端写 tail | 之前的写入先可见，tail 后可见 |
| `ld_acquire_sys_global` | 接收端读 tail | 读到新 tail 后，后续读取一定看到对应数据 |
| `ld_volatile_global` | 轮询 offset / head | 每次真正读内存，不用寄存器或缓存里的旧值 |
| `st_relaxed_sys_global` | 写 offset / head | 只要求写入最终可见、不被优化掉 |

`.sys` 作用范围是整个系统（含其他 GPU、CPU），NVLink 对端需要它；`.gpu` 只覆盖本 GPU。

### 9.5 named barrier：`bar.sync a, b` 只等 96 个线程

```text
bar.sync  a,  b
          │   └─ 参与线程数，凑够就放行（32 的倍数，硬件按 warp 计数）
          └───── 屏障编号 0–15，每 block 16 个独立硬件计数器
```

warp 执行到 `bar.sync a, b`，a 号计数器加 32 并等待；到达 b 就放行所有在等 a 号屏障的 warp 并清零。不执行这条指令的 warp 完全不受影响。`__syncthreads()` 就是 `bar.sync 0`、人数为整个 block。

dispatch 里第 r 组用 r 号屏障、人数 96，8 组互不干扰：

```text
时刻  warp 3 (组1)      warp 4 (组1)      warp 5 (组1)      warp 0 (组0)
t1    bar.sync 1,96     拷贝中            拷贝中            正常工作
      屏障1: 32/96
t2    等待              bar.sync 1,96     拷贝中            bar.sync 0,96
                        屏障1: 64/96                        屏障0: 32/96（与屏障1无关）
t3    放行              放行              bar.sync 1,96 → 96/96，全部放行
t4    warp 3 发布 tail
```

约束：编号只有 0–15（这里 ≤ 8 组够用）；0 号和 `__syncthreads` 共用，dispatch 主循环里没有 `__syncthreads`，所以不冲突；**同组所有 warp 必须执行相同次数的 `bar.sync`**，这也是发送端 3 个 warp 都要跑完整 token 循环的原因之一（另一个原因是每个 warp 都要算出每个 token 的序号）。

### 9.6 自旋等待要求所有 block 同时驻留

block 之间没有内置同步，只能在 global memory 上轮询。如果接收 block 在空转，发送 block 却因为 SM 被占满一直没被调度，就会死锁。DeepEP 的保证：

- `num_sms`（默认 20）远小于 H800 的 132 个 SM；
- `__launch_bounds__(768, 1)`：一个 SM 一个 block；
- SM90 路径用 cooperative launch（`cudaLaunchAttributeCooperative`），整个 grid 必须一次性全部驻留，否则启动失败。

剩下的 SM 可以留给同时运行的计算 kernel，这就是 DeepEP 计算与通信重叠的基础。

## 10. 输出汇总与 combine 如何使用

| 输出 | 在哪一侧写 | 内容 | 后续用途 |
|---|---|---|---|
| `recv_x` | 接收端 | 按（来源 rank, channel, 序号）排列的 token | 给本地 expert 计算 |
| `recv_x_scales` | 接收端 | FP8 缩放系数 | 反量化 |
| `recv_topk_idx` / `recv_topk_weights` | 接收端 | 本地 expert 编号 / 权重，非本地 −1 / 0 | 把行分给本地 expert、加权 |
| `recv_src_idx` | 接收端 | token 在来源 rank 的下标 | combine 送回原位 |
| `recv_channel_offset[src][c]` | 接收端 | channel c 在来源 src 段内的起点 | combine 定位 |
| `send_head[token][dst]` | 发送端 | channel 内序号，不发为 −1 | combine 按序号取回结果 |

combine 的流向相反：rank 2 按 dispatch 时同样的顺序把 expert 输出写回本卡的队列，token 14 的结果会出现在序号 2 的位置。本卡读 `send_head` 就知道该等哪个序号（`intranode.cu` L920–L926）：

```cpp
                // Read expected head
                int expected_head = -1;
                if (lane_id < kNumRanks)
                    expected_head = ld_nc_global(send_head + token_idx * kNumRanks + lane_id);

                auto start_time = clock64();
                while (__any_sync(0xffffffff, channel_tail_idx[lane_id] <= expected_head and expected_head >= 0)) {
```

每个 lane 对应一个来源 rank，只要还有某个相关 rank 的 tail 没超过 `expected_head` 就继续等；−1 表示当初没发，不用等。combine 细节留到下一篇。

## 11. 限制汇总

| 量 | 限制 | 在哪里检查 | 违反时 |
|---|---|---|---|
| `num_tokens` | 无专门上限（int；发送端 `token_idx` 是 int64） | 只检查 `is_token_in_rank` 形状 `[x.size(0), num_ranks]` 且连续 | 实际受 `recv_x` 显存限制；token 再多只是环形队列多转几圈 |
| `num_topk` | normal ≤ 32；low-latency ≤ 11 | normal：dispatch/combine 的 device 断言（`intranode.cu` L251、L730，`internode.cu` L519、L1754）；LL：host 断言（`internode_ll.cu` L493/L501、L1166/L1179） | normal 在 kernel 运行时断言失败（CUDA 错误）；LL 调用时直接报错 |
| 本地 expert 数 | ≤ 128 | `notify_dispatch` host 断言 | 直接报错 |
| `num_sms` | 偶数 | host + device 断言 | 报错 |
| hidden（SM90 TMA 路径） | `hidden_int4` 为偶数，半行 + 8 B ≤ 8192 | device 断言 L304 | kernel 断言失败 |
| NVLink buffer | 第 4.2 节各项之和 ≤ `num_nvl_bytes` | host 断言 L627 | 报错，需要调大 buffer 或调小 `Config` 的 recv chunk |

## 12. 数值追踪

### 12.1 小例子：4 rank、8 expert、2 channel、环长 4、一批 3

输入与 09 相同。rank 0 的 5 个 token 分成 channel 0 = token 0–2、channel 1 = token 3–4：

| token | topk | 去往 | `send_head[token]`（dst 0..3） |
|---|---|---|---|
| 0 | `[0, 3]` | 0, 1 | `[0, 0, -1, -1]` |
| 1 | `[2, 3]` | 1 | `[-1, 1, -1, -1]` |
| 2 | `[5, -1]` | 2 | `[-1, -1, 0, -1]` |
| 3 | `[7, 1]` | 0, 3 | `[0, -1, -1, 0]`（channel 1 重新从 0 编号） |
| 4 | `[4, 5]` | 2 | `[-1, -1, 0, -1]` |

rank 1 的 `recv_x`（`rank_prefix_matrix[*][1] = [2, 4, 5, 6]`）：

| 行 | 来源 (rank, token) | 行号来历 | `recv_topk_idx` | `recv_topk_weights` |
|---|---|---|---|---|
| 0 | (0, 0) | rank 偏移 0 + channel 0 起点 0 + 序号 0 | `[-1, 1]` | `[0.0, 0.4]` |
| 1 | (0, 1) | 0 + 0 + 1 | `[0, 1]` | `[0.6, 0.4]` |
| 2 | (1, 1) | 2 + 0 + 0 | `[-1, 0]` | `[0.0, 0.4]` |
| 3 | (1, 2) | 2 + 1（rank 1 channel 1 起点）+ 0 | `[1, -1]` | `[0.6, 0.0]` |
| 4 | (2, 1) | 4 + 0 + 0 | `[0, -1]` | `[0.6, 0.0]` |
| 5 | (3, 0) | 5 + 0 + 0 | `[1, -1]` | `[0.6, 0.0]` |

（权重统一取 `[0.6, 0.4]` 便于观察：非本地位置被置 0。）

### 12.2 环形队列绕圈：N = 8、一批 3、某队列共 7 个 token

| 批次 | 空位检查 | 发送序号 | 槽位 | 结束后 tail | 接收端 |
|---|---|---|---|---|---|
| 1 | 8 − (0 − 0) = 8 ≥ 3 | 0, 1, 2 | 0, 1, 2 | 3 | 读到 3，搬序号 0–2，head → 3 |
| 2 | 设 head = 0：8 − 3 = 5 ≥ 3 | 3, 4, 5 | 3, 4, 5 | 6 | 搬 3–5，head → 6 |
| 3 | 设 head = 3：8 − 3 = 5 ≥ 3 | 6 | 6 | 7 | 搬 6 |

第 3 批只发 1 个，是 `token_idx` 先到了 `token_end_idx`。若 N = 4 而一条队列要传 17 个 token，序号 4 会复用槽位 0、序号 8 再复用槽位 0……共绕 4 圈，行号仍按不取模的序号计算。

### 12.3 接收端一轮的行号（承接 6 节的发送例子）

源 rank 0 的 channel 1 发给 rank 2 的序号 0–3（token 11、14、15、19），rank 2 上 `rank_offset = 100`、channel 起点 30、N = 2：

| 轮次 | 读到的 tail | head | `total_offset` | 搬的 `chunk_idx` | 槽位 | 写入行 |
|---|---|---|---|---|---|---|
| 1 | 2 | 0 | 130 | 0, 1 | 0, 1 | 130, 131 |
| 2 | 4 | 2 | 132 | 0, 1 | 0, 1（复用） | 132, 133 |

## 13. 验证：多线程模拟环形队列

当前机器是 macOS，没有 NVIDIA GPU，不能运行 DeepEP。下面的脚本只依赖 Python 标准库：每个（rank, channel, 对端）的发送端和接收端各是一个线程，队列放在「接收 rank」的字典里，按 kernel 的逻辑做 offset 编码、空位检查、分批、`send_head`、topk 本地化、tail/head 推进。它验证的是算法语义（落点、顺序、绕圈、`send_head` 往返、分批对死锁的作用），**不能**代替在 GPU 上验证内存序和 NVLink 行为；Python 的 GIL 让赋值天然有序，相当于假设 release/acquire 已经正确。

保存为 `intranode_dispatch_sim.py`：

```python
import random
import threading
import time

TIMEOUT_S = 3.0


def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def get_channel_task_range(num_tokens: int, num_channels: int, channel_id: int) -> tuple[int, int]:
    per = ceil_div(num_tokens, num_channels)
    start = min(per * channel_id, num_tokens)
    return start, min(start + per, num_tokens)


class Queue:
    """One ring per (channel, src rank), stored in the receiver's NVLink buffer."""

    def __init__(self, n: int) -> None:
        self.start_offset = 0  # encoded -value-1, 0 means "not written yet"
        self.end_offset = 0
        self.head = 0          # written by the receiver
        self.tail = 0          # written by the sender (st.release in CUDA)
        self.x = [None] * n
        self.src_idx = [None] * n
        self.topk_idx = [None] * n
        self.topk_w = [None] * n
        self.max_tail = 0


def wait_until(cond, what: str) -> None:
    start = time.monotonic()
    while not cond():
        if time.monotonic() - start > TIMEOUT_S:
            raise TimeoutError(what)
        time.sleep(0)


def run(topk_by_rank, weights_by_rank, num_experts, num_channels, ring, chunk, publish_per_chunk=True):
    P = len(topk_by_rank)
    epr = num_experts // P
    assert chunk <= ring

    # ---- layout + notify (notes 07 and 09) ----
    in_rank = [[[False] * P for _ in t] for t in topk_by_rank]
    for r, rows in enumerate(topk_by_rank):
        for t, row in enumerate(rows):
            for e in row:
                if e >= 0:
                    in_rank[r][t][e // epr] = True
    rank_prefix = [[0] * P for _ in range(P)]  # rank_prefix[i][dst]: tokens from ranks 0..i to dst
    for dst in range(P):
        acc = 0
        for i in range(P):
            acc += sum(in_rank[i][t][dst] for t in range(len(in_rank[i])))
            rank_prefix[i][dst] = acc
    chan_prefix = []  # chan_prefix[src][dst][c]
    for src in range(P):
        n = len(in_rank[src])
        m = []
        for dst in range(P):
            acc, row = 0, []
            for c in range(num_channels):
                s, e = get_channel_task_range(n, num_channels, c)
                acc += sum(in_rank[src][t][dst] for t in range(s, e))
                row.append(acc)
            m.append(row)
        chan_prefix.append(m)

    # ---- outputs ----
    queues = [{(c, src): Queue(ring) for c in range(num_channels) for src in range(P)} for _ in range(P)]
    recv_x = [[None] * rank_prefix[P - 1][dst] for dst in range(P)]
    recv_src_idx = [[None] * rank_prefix[P - 1][dst] for dst in range(P)]
    recv_topk = [[None] * rank_prefix[P - 1][dst] for dst in range(P)]
    recv_w = [[None] * rank_prefix[P - 1][dst] for dst in range(P)]
    recv_channel_offset = [[[None] * num_channels for _ in range(P)] for _ in range(P)]
    send_head = [[[None] * P for _ in in_rank[src]] for src in range(P)]
    errors = []

    def sender(me: int, c: int, dst: int) -> None:
        q = queues[dst][(c, me)]                       # buffer_ptrs[responsible_rank], slot (c, rank)
        start_val = chan_prefix[me][dst][c - 1] if c > 0 else 0
        q.start_offset = -start_val - 1
        q.end_offset = -chan_prefix[me][dst][c] - 1
        tok, tok_end = get_channel_task_range(len(in_rank[me]), num_channels, c)
        tail = 0
        while tok < tok_end:
            wait_until(lambda: ring - (tail - q.head) >= chunk, f"sender {me}->{dst} c{c} waits for space")
            sent = 0
            while sent < chunk and tok < tok_end:
                send_head[me][tok][dst] = tail if in_rank[me][tok][dst] else -1
                if not in_rank[me][tok][dst]:
                    tok += 1
                    continue
                slot = tail % ring
                tail += 1
                q.x[slot] = (me, tok)                  # stands for x[tok] (hidden_int4 int4s)
                q.src_idx[slot] = tok
                b, e = dst * epr, (dst + 1) * epr
                idx = [ex - b if b <= ex < e else -1 for ex in topk_by_rank[me][tok]]
                q.topk_idx[slot] = idx
                q.topk_w[slot] = [w if i >= 0 else 0.0 for i, w in zip(idx, weights_by_rank[me][tok])]
                sent += 1
                tok += 1
            if publish_per_chunk:
                q.tail = tail                          # bar.sync + st_release_sys_global
                q.max_tail = tail
        q.tail = tail
        q.max_tail = tail

    def receiver(me: int, c: int, src: int) -> None:
        q = queues[me][(c, src)]                       # buffer_ptrs[rank], slot (c, responsible_rank)
        rank_offset = rank_prefix[src - 1][me] if src > 0 else 0
        wait_until(lambda: q.start_offset != 0 and q.end_offset != 0, "receiver waits for offsets")
        start, end = -q.start_offset - 1, -q.end_offset - 1
        recv_channel_offset[me][src][c] = start
        remaining, total_offset, head = end - start, rank_offset + start, 0
        while remaining > 0:
            wait_until(lambda: q.tail != head, f"receiver {me}<-{src} c{c} waits for tail")
            tail = q.tail                              # ld_acquire_sys_global
            n = tail - head
            for k in range(n):
                slot = (head + k) % ring
                recv_x[me][total_offset + k] = q.x[slot]
                recv_src_idx[me][total_offset + k] = q.src_idx[slot]
                recv_topk[me][total_offset + k] = q.topk_idx[slot]
                recv_w[me][total_offset + k] = q.topk_w[slot]
            head += n
            total_offset += n
            q.head = head                              # st_relaxed_sys_global, frees slots
            remaining -= n

    def guarded(fn, *args):
        try:
            fn(*args)
        except TimeoutError as exc:
            errors.append(str(exc))

    threads = []
    for me in range(P):
        for c in range(num_channels):
            for peer in range(P):
                threads.append(threading.Thread(target=guarded, args=(sender, me, c, peer)))
                threads.append(threading.Thread(target=guarded, args=(receiver, me, c, peer)))
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if errors:
        return None, errors

    return dict(rank_prefix=rank_prefix, chan_prefix=chan_prefix, recv_x=recv_x, recv_src_idx=recv_src_idx,
                recv_topk=recv_topk, recv_w=recv_w, recv_channel_offset=recv_channel_offset,
                send_head=send_head, in_rank=in_rank, queues=queues), []


def verify(out, topk_by_rank, num_experts, num_channels) -> None:
    P = len(topk_by_rank)
    epr = num_experts // P
    in_rank, send_head = out["in_rank"], out["send_head"]
    for dst in range(P):
        expected = [(src, t) for src in range(P) for t in range(len(in_rank[src])) if in_rank[src][t][dst]]
        assert out["recv_x"][dst] == expected, (dst, out["recv_x"][dst], expected)
        for row, (src, t) in enumerate(expected):
            assert out["recv_src_idx"][dst][row] == t
            for e, local in zip(topk_by_rank[src][t], out["recv_topk"][dst][row]):
                assert local == (e - dst * epr if e >= 0 and e // epr == dst else -1)
    for src in range(P):
        n = len(in_rank[src])
        for t in range(n):
            c = next(c for c in range(num_channels) if get_channel_task_range(n, num_channels, c)[0] <= t
                     < get_channel_task_range(n, num_channels, c)[1])
            for dst in range(P):
                h = send_head[src][t][dst]
                if not in_rank[src][t][dst]:
                    assert h == -1
                    continue
                rank_offset = out["rank_prefix"][src - 1][dst] if src > 0 else 0
                row = rank_offset + out["recv_channel_offset"][dst][src][c] + h
                assert out["recv_x"][dst][row] == (src, t)


def main() -> None:
    P, E, C = 4, 8, 2
    topk = [
        [[0, 3], [2, 3], [5, -1], [7, 1], [4, 5]],
        [[1, 6], [0, 2], [3, 4]],
        [[6, 7], [2, 5], [0, 1], [4, 6]],
        [[3, 5], [1, 7]],
    ]
    weights = [[[0.6, 0.4] for _ in rows] for rows in topk]
    out, err = run(topk, weights, E, C, ring=4, chunk=3)
    assert not err, err
    verify(out, topk, E, C)
    print("[small] rank 1 recv_x rows (src_rank, src_token):")
    for row, item in enumerate(out["recv_x"][1]):
        print(f"  row {row}: {item}  recv_topk_idx={out['recv_topk'][1][row]}  recv_topk_weights={out['recv_w'][1][row]}")
    print("[small] rank 0 send_head[token][dst]:")
    for t, row in enumerate(out["send_head"][0]):
        print(f"  token {t}: {row}")
    print("[small] recv order, recv_topk, send_head round trip  PASS")

    rng = random.Random(0)
    big_topk, big_w = [], []
    for r in range(P):
        rows = []
        for _ in range(rng.randint(30, 50)):
            rows.append(rng.sample(range(E), 2))
        big_topk.append(rows)
        big_w.append([[0.5, 0.5] for _ in rows])
    out, err = run(big_topk, big_w, E, C, ring=4, chunk=3)
    assert not err, err
    verify(out, big_topk, E, C)
    max_tail = max(q.max_tail for qs in out["queues"] for q in qs.values())
    print(f"[random] tokens per rank = {[len(t) for t in big_topk]}, ring = 4 slots, chunk = 3")
    print(f"[random] longest queue carried {max_tail} tokens through 4 slots "
          f"(wrapped {max_tail // 4} times)  PASS")

    out, err = run(big_topk, big_w, E, C, ring=4, chunk=3, publish_per_chunk=False)
    stuck_senders = any("waits for space" in e for e in err)
    stuck_receivers = any("waits for tail" in e for e in err)
    print(f"[no per-chunk tail] deadlock detected: {out is None}; "
          f"senders stuck on space: {stuck_senders}; receivers stuck on tail: {stuck_receivers}")


if __name__ == "__main__":
    main()
```

运行：

```bash
python3 intranode_dispatch_sim.py
```

本机（macOS，Python 3.14）实际输出（约 3 秒，最后一项要等超时）：

```text
[small] rank 1 recv_x rows (src_rank, src_token):
  row 0: (0, 0)  recv_topk_idx=[-1, 1]  recv_topk_weights=[0.0, 0.4]
  row 1: (0, 1)  recv_topk_idx=[0, 1]  recv_topk_weights=[0.6, 0.4]
  row 2: (1, 1)  recv_topk_idx=[-1, 0]  recv_topk_weights=[0.0, 0.4]
  row 3: (1, 2)  recv_topk_idx=[1, -1]  recv_topk_weights=[0.6, 0.0]
  row 4: (2, 1)  recv_topk_idx=[0, -1]  recv_topk_weights=[0.6, 0.0]
  row 5: (3, 0)  recv_topk_idx=[1, -1]  recv_topk_weights=[0.6, 0.0]
[small] rank 0 send_head[token][dst]:
  token 0: [0, 0, -1, -1]
  token 1: [-1, 1, -1, -1]
  token 2: [-1, -1, 0, -1]
  token 3: [0, -1, -1, 0]
  token 4: [-1, -1, 0, -1]
[small] recv order, recv_topk, send_head round trip  PASS
[random] tokens per rank = [42, 49, 48, 30], ring = 4 slots, chunk = 3
[random] longest queue carried 17 tokens through 4 slots (wrapped 4 times)  PASS
[no per-chunk tail] deadlock detected: True; senders stuck on space: True; receivers stuck on tail: True
```

- `[small]`：`recv_x` 顺序等于「按来源 rank、再按来源 token 顺序」，`recv_topk_idx` 是本地编号；对每个发出的 token，`rank 偏移 + recv_channel_offset + send_head` 都能找回它所在的行。
- `[random]`：环长只有 4，最长的队列传了 17 个 token，绕了 4 圈，结果仍然正确。
- `[no per-chunk tail]`：改成只在 channel 末尾发布 tail，发送端写满 4 个槽位后等空位、接收端等 tail，互相卡死，验证了 6.4 节的结论。连跑 3 次结果一致。

## 14. 常见错误与症状

| 错误理解 / 写法 | 正确理解 | 可观察到的症状 |
|---|---|---|
| 以为 `responsible_channel` 决定发还是收 | 发收由 `is_sender`（`sm_id` 奇偶）决定 | 读代码时把接收端的 `responsible_rank` 当成目标 rank，偏移全推错 |
| 以为 L271–L274 四个指针相同 | `Buffer` 按引用推进 `ptr`，四个地址各差一张表 | 自己写类似代码时四个字段互相覆盖 |
| 以为 `buffer_ptrs` 是 shared memory | 是 HBM，经 IPC 映射可跨 GPU 访问 | 误判容量和可见范围，以为只能 block 内使用 |
| 去掉分批限制，一次发完再写 tail | 每批 ≤ `num_max_send_tokens`，每批发布一次 | 队列比 token 少时双方互等，约 100 秒后 `DeepEP timeout for dispatch senders/receivers` 并 trap |
| tail 用普通写或 relaxed 写 | 发送端 release、接收端 acquire | 不会卡死；偶尔读到旧槽位数据，`recv_x` 某些行内容错乱，难复现 |
| 接收端每个 warp 各自读 tail | 一个线程读、shared memory + `bar.sync` 广播 | 漏拷或重拷，head 推进不一致，最终越界或超时 |
| `topk` 拷贝去掉 `lane_id < num_topk` | 只让前 `num_topk` 个 lane 干活 | 写坏相邻槽位的 topk，下游 expert 分配错乱 |
| 发送端 3 个 warp 只跑自己负责的 token | 都要跑完整循环，`bar.sync` 次数必须一致 | named barrier 永远凑不齐 96，卡死后超时 |
| 以为 `send_head` 只是调试信息 | combine 靠它知道该等哪个序号 | combine 等错位置，结果加错或超时 |
| `num_sms` 设成奇数 | 必须偶数，发收成对 | host 断言失败 |
| hidden 太大（bf16 超过约 8176）走 TMA 路径 | 半行 + 8 B 必须 ≤ 8192 | device 断言失败 |

## 15. 速记卡

```text
grid = num_sms(20)，768 线程，__launch_bounds__(768,1)，192 KB dynamic smem，cooperative + cluster 2
block 2c 发、2c+1 收；channel = 本卡 token 的第 c 段
block 内：responsible_rank = thread_id / 96，每个对端 rank 3 个 warp

队列（在接收 rank R 显存）：(channel c, 来源 S) → 元数据 start/end/head/tail + N 槽数据环
  发送：buffer_ptrs[R] + 格子 c*P+S；接收：buffer_ptrs[R] + 格子 c*P+S
  Buffer<T>(ptr, 表大小, 格子)：返回 ptr+格子，并把 ptr 推过整张表

发送：
  写 -start-1 / -end-1
  for 本 channel token，分批：
    等 N - (tail - head) >= chunk
    每批 ≤ chunk 个：send_head[t][R] = tail 或 -1；slot = tail++ % N；tail%3 的 warp 拷 x/src/topk(本地化)/scales
    bar.sync R,96 → warp0 st.release.sys tail
接收：
  rank_offset = rpm[S-1][me]；等 offset，start 写 recv_channel_offset；total = rank_offset + start
  loop：线程0 ld.acquire tail → smem → bar.sync；搬 [head,tail) 到 recv_x[total + i]（TMA 两半）
        head += n, total += n；bar.sync → 最后一个 warp 写 head
行号 = rank 偏移 + channel 起点 + channel 内序号；槽位 = 序号 % N
```

## 16. 自测题与答案

1. **rank 3 的 block 7、线程 300 在做什么？（P = 8）**
   答：block 7 是奇数，接收端；`responsible_channel = 7 / 2 = 3`；`responsible_rank = 300 / 96 = 3`。它负责从 rank 3 自己显存里（channel 3，来源 rank 3）的队列取数据，也就是「自己发给自己」的那条队列，写进 `recv_x` 里 rank 3 那一段的 channel 3 部分。

2. **为什么空位检查用的数字必须和分批上限一致？只改其中一个会怎样？**
   答：空位检查只保证有 `num_max_send_tokens` 个空槽。分批上限更大，就会覆盖接收端没读走的槽位，数据被破坏；分批上限更小，只是更频繁地发布 tail，正确但效率低。两者一致才是「刚好不越界」。

3. **接收端读到 tail = 6、自己 head = 3、`total_offset = 133`，N = 4。这一轮要搬哪几个槽位、写到哪几行？**
   答：`num_recv_tokens = 3`，`chunk_idx = 0, 1, 2`。槽位 `(3+0)%4 = 3`、`(3+1)%4 = 0`、`(3+2)%4 = 1`；行号 133、134、135。之后 head = 6，`total_offset` = 136。

4. **发送端 tail 为什么必须用 release 写？接收端为什么必须用 acquire 读？**
   答：发送端先写槽位数据、后写 tail。release 保证对端看到新 tail 时，之前的数据写入已经可见；acquire 保证读到新 tail 后，后续读槽位不会被提前到读 tail 之前。少了任何一边，接收端都可能读到旧数据，而且不会报错。

5. **`send_head[14][2] = 2` 表示什么？combine 怎么用？**
   答：token 14 发往 rank 2 时，是它所在 channel 发往 rank 2 的第 2 个（从 0 开始）。combine 时 rank 2 按同样顺序把 expert 输出写回本卡队列，本卡等这条队列的 tail 超过 2，再到槽位 `2 % N` 取出 token 14 的结果。

## 17. 学习进度

- [x] DeepEP V1 normal：layout → rank dispatch → local expert metadata → handle → combine
- [x] 区分 rank-major 通信布局、expert-major 计算布局与 gate 加权
- [x] DeepEP V1 low-latency：定容 dispatch、packed expert 输入、handle 回程元数据、weighted combine 与 hook
- [x] 机内 `get_dispatch_layout` kernel：block 分工、per-thread 计数 + 按列归约、token-rank 去重、单机 RDMA 分支
- [x] 机内 `barrier_block`：分布式 8×8 信号矩阵、+T/−T 成对原子操作、warp 投票、`kSyncOnly`、`<= 0` 的原因
- [x] 机内 `notify_dispatch`：两次 barrier 交换计数、`rank_prefix_matrix` 只有本列有效、expert 对齐、`channel_prefix_matrix`、pinned mapped 计数器与 −1 哨兵、`num_worst_tokens`
- [x] 机内 `dispatch` kernel：发收 block 分工、`Buffer<T>` 布局、环形队列与 head/tail 流控、分批发布、`send_head`、TMA 两半搬运、release/acquire、named barrier、`__launch_bounds__` 与 SM 资源

### 下一知识点

intranode `combine` kernel：`cached_notify_combine` 如何修补 `send_head` 中的 −1，combine 的发送端（原接收端）如何按 `recv_channel_offset` 把 expert 输出写回，接收端（原发送端）如何用 `send_head` 等待每个来源 rank 的结果并按 topk 权重求和。
