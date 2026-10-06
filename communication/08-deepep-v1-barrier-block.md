# DeepEP V1：机内跨 GPU barrier `barrier_block` 逐行拆解

> 承接 [07-deepep-v1-get-dispatch-layout-kernel.md](./07-deepep-v1-get-dispatch-layout-kernel.md)。07 算出了 `num_tokens_per_rank` 等布局信息；接下来 intranode dispatch 的 notify 阶段要把这些计数写到其他 GPU 的缓冲区里，再等所有 GPU 都写完才能读。这个「等所有 GPU 都写完」靠的就是本文的 `barrier_block`。
>
> 代码对照：本地 DeepEP checkout `/Users/saboxu/Downloads/communication/DeepEP`。函数在 `csrc/kernels/legacy/utils.cuh`（约 L514–L544），信号区的分配在 `csrc/legacy/buffer.hpp` 的 `Buffer` 构造函数（约 L100–L145）和 `sync()`（约 L232–L252），调用点在 `csrc/kernels/legacy/intranode.cu`。

```text
本次讲解位置
章节：communication / DeepEP V1
小节：机内同步原语 barrier_block
知识点：8×8 信号矩阵按行分布在 8 张 GPU 上；+TAG/−TAG 成对原子操作；warp 投票等待；kSyncOnly；thread_id 兼任对端 rank 编号
上次：机内 get_dispatch_layout kernel（block 分工、per-thread 计数、token-rank 去重）
下次：intranode notify_dispatch 如何用两次 barrier 交换计数，得到 rank_prefix_matrix 和 channel_prefix_matrix
PTX：fence.acq_rel.sys、ld.volatile.global、system scope 原子操作（atomicAdd_system）
```

## 为什么现在讲这个

intranode 的每个通信 kernel 都要先跨 GPU 对齐一次。以 `notify_dispatch` 为例：每张卡先把自己的 `num_tokens_per_rank` 通过 NVLink **直接写进其他 7 张卡的缓冲区**，然后每张卡读自己缓冲区里收到的 8 份计数求前缀和。如果中间不同步：

1. **读到旧数据或半成品**：rank 0 开始求前缀和时，rank 5 的计数可能还没写到，或者写了但还不可见，得到的接收行数是错的。这类 bug 不会卡死，只会偶尔算错，极难排查。
2. **覆盖还在被使用的缓冲区**：上一个 kernel 里对端可能还在读这块缓冲区，这边已经开始写新的计数。

用 NCCL 或 CPU 同步太慢，而且要打断 kernel。`barrier_block` 用不到 40 行代码，在 kernel 内部通过 NVLink 的系统级原子操作完成同步，不依赖 CPU，信号区也不用每次清零。它是后面所有 intranode notify / cached notify kernel 的基础，读懂它，才能看懂这些 kernel 中「写完 → barrier → 读」的结构。

## 0. 心智模型

```text
信号区：逻辑上是 8×8 的 int 矩阵 S，第 r 行 S_r[0..7] 物理上放在 GPU r 的显存里

rank r 到达 barrier：
  线程 t（t < kNumRanks）：
    在自己这一行加：S_r[t] += TAG       "我又到了一次"
    在对方那一行减：S_t[r] -= TAG       通过 NVLink 告诉 rank t："r 到了"

rank r 等待：
  线程 t 反复读 S_r[t]，warp 投票：8 格全部 <= 0 就放行

不变量：S_r[t] = TAG × (r 到达次数 − t 到达次数)
  S_r[t] <= 0  ⇔  rank t 到达的次数不少于 r  ⇔  t 已经到了
```

一句话：**每张卡只盯自己那一行，自己那一行全部 ≤ 0，就说明其他所有卡都到了。**

## 1. 术语与常量

| 名字 | 定义位置 | 值 / 含义 |
|---|---|---|
| `LEGACY_NUM_MAX_NVL_PEERS` | `compiled.cuh` | 8，一个节点内 NVLink 互联的最大 GPU 数 |
| `LEGACY_FINISHED_SUM_TAG` | `compiled.cuh` | 1024，每次到达加减的步长，下文简写为 T |
| `LEGACY_NUM_TIMEOUT_CYCLES` | `compiled.cuh` | 200G 个时钟周期，约 100 秒 |
| `kNumRanks` | 模板参数 | 参与同步的 GPU 数，≤ 8 |
| `kSyncOnly` | 模板参数，默认 `false` | `true` 时跳过开头的 fence + `__syncthreads()` |
| `rank` | 函数参数 | 本卡在节点内的编号 0–7（单机时等于全局 rank，多机时必须是 `nvl_rank`） |
| `S_r[t]` | 本文记号 | GPU r 上信号区的第 t 个 int |

## 2. 完整代码

### 2.1 `barrier_block` 及其依赖（`csrc/kernels/legacy/utils.cuh`）

```cpp
__device__ __forceinline__ void memory_fence() {
    asm volatile("fence.acq_rel.sys;" ::: "memory");
}

__device__ __forceinline__ int ld_volatile_global(const int* ptr) {
    int ret;
    asm volatile("ld.volatile.global.s32 %0, [%1];" : "=r"(ret) : "l"(ptr));
    return ret;
}

template <int kNumRanks, bool kSyncOnly = false>
__forceinline__ __device__ void barrier_block(int** barrier_signal_ptrs, int rank) {
    auto thread_id = static_cast<int>(threadIdx.x);

    // For non-sync-only cases, the memory operations by other threads in the block must be visible to the `sys` scope
    if constexpr (not kSyncOnly) {
        memory_fence();
        __syncthreads();
    }

    // Add self-ranks, sub other ranks
    if (thread_id < kNumRanks) {
        atomicAdd_system(barrier_signal_ptrs[rank] + thread_id, LEGACY_FINISHED_SUM_TAG);
        atomicSub_system(barrier_signal_ptrs[thread_id] + rank, LEGACY_FINISHED_SUM_TAG);
    }
    EP_DEVICE_ASSERT(kNumRanks <= blockDim.x);

    // Check timeout
    auto start_time = clock64();
    while (true) {
        auto value = thread_id < kNumRanks ? ld_volatile_global(barrier_signal_ptrs[rank] + thread_id) : 0;
        if (__all_sync(0xffffffff, value <= 0))
            break;

        if (clock64() - start_time > LEGACY_NUM_TIMEOUT_CYCLES and thread_id < kNumRanks) {
            printf("DeepEP timeout check failed: rank = %d, thread = %d, value = %d)\n", rank, thread_id, value);
            trap();
        }
    }
    __syncthreads();
}
```

常量（`csrc/kernels/legacy/compiled.cuh`）：

```cpp
#define LEGACY_FINISHED_SUM_TAG 1024
#define LEGACY_NUM_TIMEOUT_CYCLES 200000000000ull  // 200G cycles ~= 100s
```

### 2.2 信号区的分配（`csrc/legacy/buffer.hpp`）

`Buffer` 的成员：

```cpp
void* buffer_ptrs[LEGACY_NUM_MAX_NVL_PEERS] = {nullptr};
void** buffer_ptrs_gpu = nullptr;
int* barrier_signal_ptrs[LEGACY_NUM_MAX_NVL_PEERS] = {nullptr};
int** barrier_signal_ptrs_gpu = nullptr;
```

构造函数：一次分配，切出四段。

```cpp
int64_t barrier_signal_bytes = LEGACY_NUM_MAX_NVL_PEERS * sizeof(int);
int64_t buffer_ptr_bytes = LEGACY_NUM_MAX_NVL_PEERS * sizeof(void*);
int64_t barrier_signal_ptr_bytes = LEGACY_NUM_MAX_NVL_PEERS * sizeof(int*);

if (num_nvl_bytes > 0) {
    // Local IPC: alloc local memory and set local IPC handles
    shared_memory_allocator.malloc(&buffer_ptrs[nvl_rank],
                                   num_nvl_bytes + barrier_signal_bytes + buffer_ptr_bytes + barrier_signal_ptr_bytes);
    shared_memory_allocator.get_mem_handle(&ipc_handles[nvl_rank], buffer_ptrs[nvl_rank]);
    buffer_ptrs_gpu = reinterpret_cast<void**>(static_cast<uint8_t*>(buffer_ptrs[nvl_rank]) + num_nvl_bytes + barrier_signal_bytes);

    // Set barrier signals
    barrier_signal_ptrs[nvl_rank] = reinterpret_cast<int*>(static_cast<uint8_t*>(buffer_ptrs[nvl_rank]) + num_nvl_bytes);
    barrier_signal_ptrs_gpu =
        reinterpret_cast<int**>(static_cast<uint8_t*>(buffer_ptrs[nvl_rank]) + num_nvl_bytes + barrier_signal_bytes + buffer_ptr_bytes);

    // No need to synchronize, will do a full device sync during `sync`
    CUDA_RUNTIME_CHECK(cudaMemsetAsync(barrier_signal_ptrs[nvl_rank], 0, barrier_signal_bytes, comm_stream));
}
```

`sync()`：打开对端的 IPC 句柄，补齐其他 7 个指针，再把指针数组拷到显存。

```cpp
for (int i = 0, offset = rdma_rank * num_nvl_ranks; i < num_nvl_ranks; ++i) {
    EP_HOST_ASSERT(all_gathered_handles[offset + i].has_value());
    auto handle_str = std::string(all_gathered_handles[offset + i].value());
    EP_HOST_ASSERT(handle_str.size() == sizeof(shared_memory::MemHandle));
    if (offset + i != rank) {
        std::memcpy(&ipc_handles[i], handle_str.c_str(), sizeof(shared_memory::MemHandle));
        shared_memory_allocator.open_mem_handle(&buffer_ptrs[i], &ipc_handles[i]);
        barrier_signal_ptrs[i] = reinterpret_cast<int*>(static_cast<uint8_t*>(buffer_ptrs[i]) + num_nvl_bytes);
    } else {
        EP_HOST_ASSERT(std::memcmp(&ipc_handles[i], handle_str.c_str(), sizeof(shared_memory::MemHandle)) == 0);
    }
}

// Copy all buffer and barrier signal pointers to GPU
CUDA_RUNTIME_CHECK(cudaMemcpy(buffer_ptrs_gpu, buffer_ptrs, sizeof(void*) * LEGACY_NUM_MAX_NVL_PEERS, cudaMemcpyHostToDevice));
CUDA_RUNTIME_CHECK(cudaMemcpy(barrier_signal_ptrs_gpu, barrier_signal_ptrs, sizeof(int*) * LEGACY_NUM_MAX_NVL_PEERS, cudaMemcpyHostToDevice));
CUDA_RUNTIME_CHECK(cudaDeviceSynchronize());
```

`shared_memory_allocator.malloc` 默认是 `cudaMalloc`，开启 fabric 时是 `cuMemCreate` + `cuMemMap`（`csrc/utils/shared_memory.hpp`），两者都能导出 IPC 句柄给同节点其他 GPU 映射。

## 3. 信号区的大小与布局

### 3.1 每张卡上的 NVLink 缓冲区

```
buffer_ptrs[nvl_rank]
│
├─ [0, num_nvl_bytes)        NVLink 数据缓冲区（dispatch / combine 队列、notify 计数）
├─ int[8]        32 B        信号区 S_nvl_rank        ← barrier_signal_ptrs[nvl_rank]
├─ void*[8]      64 B        buffer_ptrs 的显存副本   ← buffer_ptrs_gpu
└─ int*[8]       64 B        barrier_signal_ptrs 的显存副本 ← barrier_signal_ptrs_gpu（kernel 拿到的就是它）
```

### 3.2 两层结构：指针数组 + 分布式矩阵

`barrier_signal_ptrs` 是 **8 个指针**，每个指针指向**一张卡上的 `int[8]`**：

```
barrier_signal_ptrs: [ p0,  p1,  p2, ..., p7 ]      int*[8]，64 B
                       │    │    │        │
                       ▼    ▼    ▼        ▼
                    GPU0  GPU1  GPU2 ... GPU7       每张卡上 int[8]，32 B
                    S_0   S_1   S_2      S_7
```

- 逻辑上是 8×8 的 int 矩阵，`barrier_signal_ptrs[x][y]` 就是第 x 行第 y 列 `S_x[y]`。
- 物理上按行分散在 8 张卡上，每张卡只存自己那一行。
- `p_nvl_rank` 是本地地址，其他 7 个是 IPC 映射进来的对端地址，读写它们会走 NVLink。
- 两层长度都固定为 8，跟实际卡数无关。4 卡时指针数组后 4 项保持 `nullptr`，每行只用前 4 格；`thread_id < kNumRanks` 保证不会碰到它们。
- 信号区只在构造函数里清零一次，后面不再清零（原因见第 6 节）。

## 4. 逐段拆解

### 4.1 发布之前写的数据（`kSyncOnly = false` 时）

```cpp
if constexpr (not kSyncOnly) {
    memory_fence();      // fence.acq_rel.sys
    __syncthreads();
}
```

| 问题 | 回答 |
|---|---|
| 做什么 | 每个线程做一次系统范围的 acquire-release fence，然后 block 内同步 |
| 谁执行 | block 内全部线程 |
| 为什么需要 fence | 保证本线程在 barrier 之前的写操作（包括通过 NVLink 写进对端缓冲区的数据）排在后面的信号原子操作之前，对所有 GPU 可见 |
| 为什么需要 `__syncthreads()` | 发信号的只有线程 0–7，但写数据的可能是任何线程。必须等所有线程都做完 fence，线程 0–7 才能告诉对端「我到了」 |
| 做错的症状 | 对端通过 barrier 后读到旧数据，结果偶尔错误，不会卡死 |

### 4.2 发信号：加自己这一行，减对方那一行

```cpp
if (thread_id < kNumRanks) {
    atomicAdd_system(barrier_signal_ptrs[rank] + thread_id, TAG);   // S_rank[t] += T（本地）
    atomicSub_system(barrier_signal_ptrs[thread_id] + rank, TAG);   // S_t[rank] -= T（走 NVLink 写到 GPU t）
}
```

**这里的 `thread_id` 被当成对端 rank 编号**：线程 t 专门负责本卡和 rank t 之间的配对。两行代码里 `rank` 和 `thread_id` 只是互换了位置。

4 卡、本卡 rank = 2 时：

| 线程 | 加（本卡 GPU 2） | 减（对端卡） |
|---|---|---|
| 0 | `S_2[0] += T` | GPU 0 上 `S_0[2] -= T` |
| 1 | `S_2[1] += T` | GPU 1 上 `S_1[2] -= T` |
| 2 | `S_2[2] += T` | GPU 2 上 `S_2[2] -= T`（同一格先加后减，不变） |
| 3 | `S_2[3] += T` | GPU 3 上 `S_3[2] -= T` |
| 4–255 | 不执行 | 不执行 |

`_system` 后缀表示原子操作的作用范围是整个系统。同一个格子会被本卡和对端卡同时加减，GPU 范围的原子操作保证不了跨卡的原子性。

### 4.3 等待：warp 0 轮询 + warp 投票

```cpp
auto value = thread_id < kNumRanks ? ld_volatile_global(barrier_signal_ptrs[rank] + thread_id) : 0;
if (__all_sync(0xffffffff, value <= 0))
    break;
```

- 线程 t 读本卡的 `S_rank[t]`，检查 rank t 有没有来把这一格减回 ≤ 0。
- `ld.volatile` 强制每次都从显存读。不加的话编译器可能把值缓存在寄存器里，永远看不到对端的修改。
- `value` 是每个线程自己的寄存器变量。`__all_sync(mask, pred)` 是 warp 级投票：mask 里每个线程提交自己的 `pred`，**全部为真**才对所有线程返回真。8 个线程各拿一格，合起来判断的就是一整行。
- 线程 8–31 没有对应的 rank，赋值 0，判断恒为真，等于投赞成票，不影响结果。
- **只有 warp 0 真正在轮询。** 其他 warp 的 `value` 都是 0，第一次就跳出循环，然后停在最后的 `__syncthreads()` 等 warp 0。

rank 0 等待 rank 7 时，warp 0 里的情况：

| lane | 读的位置 | value | value <= 0 |
|---|---|---|---|
| 0–6 | `S_0[0..6]` | 0 | 真 |
| 7 | `S_0[7]` | T | **假** |
| 8–31 | 不读 | 0 | 真 |
| `__all_sync` 结果 | | | **假**，整个 warp 继续循环 |

等价的串行写法：

```cpp
bool all_arrived = true;
for (int t = 0; t < kNumRanks; ++t)
    all_arrived &= (S_me[t] <= 0);
```

### 4.4 超时与收尾

- 超时：约 100 秒后，还在等的线程打印 `rank`、`thread`、`value` 并 `trap()`，让 kernel 报错退出而不是永远卡住。`value / 1024` 是本卡比 rank `thread` 多到达的次数，可以据此判断是哪张卡没跟上。
- 最后的 `__syncthreads()`：warp 0 确认所有 rank 都到了，整个 block 一起放行。

## 5. 执行追踪

### 5.1 两个 rank

| 事件 | 操作 | S_0 | S_1 | 结果 |
|---|---|---|---|---|
| 初始 | — | [0, 0] | [0, 0] | — |
| rank 0 先到 | 线程 0：`S_0[0]` 先加后减不变；线程 1：`S_0[1] += T`，`S_1[0] -= T` | [0, **T**] | [**−T**, 0] | rank 0 读到 `S_0[1] = T > 0`，等待 |
| rank 1 后到 | 线程 0：`S_1[0] += T`，`S_0[1] -= T`；线程 1：自己的格子不变 | [0, **0**] | [**0**, 0] | 两边都 ≤ 0，各自放行 |

### 5.2 八个 rank：只看 rank 0 这一行

| 时刻 | 事件 | S_0 | rank 0 能否通过 |
|---|---|---|---|
| 初始 | — | [0, 0, 0, 0, 0, 0, 0, 0] | — |
| 1 | rank 0 到达，格子 1–7 各加 T | [0, T, T, T, T, T, T, T] | 否，7 格 > 0 |
| 2 | rank 3 到达，来减 `S_0[3]` | [0, T, T, **0**, T, T, T, T] | 否，还剩 6 格 |
| 3 | rank 1、2、4、5、6 陆续到达 | [0, 0, 0, 0, 0, 0, 0, **T**] | 否，只差 rank 7 |
| 4 | rank 7 到达，来减 `S_0[7]` | [0, 0, 0, 0, 0, 0, 0, **0**] | **通过** |

通过条件：

\[
\text{rank } me \text{ 通过} \iff \forall t \in [0, \text{kNumRanks}):\ S_{me}[t] \le 0
\]

### 5.3 不同 rank 放行的时刻不同

- **最后到的 rank 一来就能通过。** 其他 rank 已经把它这一行减成了 `[−T, …, 0, …, −T]`，它加上 +T 后每格都是 0，第一次检查就满足。
- **先到的 rank 要等最后一个到达**，再各自在轮询中读到满足条件，放行时间有先后。

所以 barrier 保证的是：**任何 rank 通过时，所有 rank 都已经到达**；但不保证大家同时离开。

## 6. 两个关键设计

### 6.1 `+T/−T` 成对：信号区不用清零就能重复使用

每次 barrier，本卡对 `S_me[t]` 加一次 T，rank t 对它减一次 T。所有 rank 都通过后，每格正好回到 0。整个生命周期只需要构造函数里 `cudaMemsetAsync` 清零一次，省掉了每次 barrier 后的清零和清零本身需要的额外同步。

### 6.2 用 `<= 0` 而不是 `== 0`：允许对端跑得快

rank 1 通过第 k 个 barrier 后，可能马上进入第 k+1 个，又来减一次 `S_0[1]`。如果 rank 0 这时还没读到第 k 个 barrier 的结果：

```text
rank 0 到达 k   : S_0 = [0,  T], S_1 = [-T, 0]
rank 1 到达 k   : S_0 = [0,  0], S_1 = [ 0, 0]
rank 1 通过 k，到达 k+1 : S_0 = [0, -T], S_1 = [ T, 0]
rank 0 才去检查第 k 个 barrier：S_0[1] = -T
```

- `<= 0`：rank 0 通过第 k 个 barrier，进入第 k+1 个时加 T，`S_0[1]` 回到 0，rank 1 也随之放行，计数正确。
- `== 0`：rank 0 卡在第 k 个 barrier；rank 1 在第 k+1 个 barrier 等 rank 0。两边互相等待，**死锁**，约 100 秒后超时 `trap()`。

第 8 节的模拟脚本复现了这个场景。

## 7. `kSyncOnly` 与 `thread_id` 映射的前提

### 7.1 `kSyncOnly = true` 只少做第一步

| 步骤 | `kSyncOnly = false` | `kSyncOnly = true` |
|---|---|---|
| `memory_fence()` + `__syncthreads()` | 执行 | **跳过** |
| 加自己这一行、减对方那一行 | 执行 | 执行 |
| warp 0 轮询 | 执行 | 执行 |
| 最后的 `__syncthreads()` | 执行 | 执行 |

`kSyncOnly = true` 的意思是只要求大家到齐，不负责发布之前写的数据。适合在 kernel 开头、还没写任何需要对端看到的数据时使用。

`notify_dispatch`（`intranode.cu` 第 45–103 行，只由 `sm_id == 0` 的 block 执行）里三次调用正好体现了这个区别：

| 位置 | 调用 | 为什么这样用 |
|---|---|---|
| 第 47 行，开头 | `barrier_block<kNumRanks, true>` | 确认所有卡都准备好了（上一轮对缓冲区的使用已经结束），再开始往对端写。此时本卡还没写东西，不需要 fence |
| 第 60–63 行之后，第 67 行 | `barrier_block<kNumRanks>` | 线程 t 刚把本卡的计数写进 GPU t 的缓冲区（`buffer_ptrs[thread_id]`），必须 fence 发布后再通知对端；对端通过后才能读到完整的计数 |
| 第 103 行，结尾 | `barrier_block<kNumRanks>` | 本卡刚清零了后续通信队列要用的区域，发布后再让大家进入 dispatch 主体 |

`cached_notify_dispatch`（第 177–192 行）是同样的结构：开头 `<kNumRanks, true>`，写完、清完之后 `<kNumRanks>`。另外 `intranode::barrier`（第 11–23 行）把 `barrier_block` 单独包成一个 1 个 block、32 个线程的 kernel，供 host 端在 `Buffer::destroy` 等场合做一次全卡同步。

### 7.2 `thread_id` 兼任对端 rank 编号的前提

1. **线程数 ≥ rank 数。** 每个对端 rank 都要有一个线程负责。第 529 行 `EP_DEVICE_ASSERT(kNumRanks <= blockDim.x)` 显式检查。
2. **配对线程最好都在 warp 0。** `__all_sync` 只在一个 warp 内投票。`kNumRanks ≤ 8` 时线程 0–7 都在 warp 0，由 warp 0 看全一整行。
3. **各卡对 rank 的编号一致，范围 0 到 `kNumRanks − 1`。** `barrier_signal_ptrs[thread_id]` 要求「线程 t」「指针数组第 t 项」「GPU t」是同一个对象。单机时 `rank == nvl_rank` 自然满足；多机时 internode kernel 都传 `nvl_rank`（例如 `internode.cu` 第 149 行 `barrier_block<LEGACY_NUM_MAX_NVL_PEERS, true>(barrier_signal_ptrs, nvl_rank)`）。

每个 rank 配一个线程不是唯一写法。也可以让线程 0 循环处理 8 个 rank，但那样 8 次跨 NVLink 的原子操作和 8 次轮询读都变成串行，barrier 延迟更高。DeepEP 选择并行发出，代价是 `thread_id` 必须同时承担 rank 编号的含义。

### 7.3 每张卡同一时刻只能有一个 block 使用

每张卡只有一行信号区。同一张卡上如果有两个 block 同时调用 `barrier_block`，会对同一格各加一次 T，计数就乱了。DeepEP 中都由固定的那个 block 调用（`notify_dispatch` 中是 `sm_id == 0`，`cached_notify_*` 是单 block kernel）。它也只同步调用它的这个 block，同一张卡上的其他 block 不受影响。

## 8. 验证：用 8 个 Python 线程模拟 8 张 GPU

当前机器是 macOS，没有 NVIDIA GPU，不能运行 DeepEP。下面的脚本只依赖 Python 标准库：每个线程扮演一张 GPU，用加锁的加法模拟 system scope 原子操作，按 `barrier_block` 的逻辑发信号和投票。它验证的是算法语义，**不能**代替在 GPU 上验证内存序和 NVLink 行为。

保存为 `barrier_block_sim.py`：

```python
import random
import threading
import time

NUM_MAX_NVL_PEERS = 8
WARP_SIZE = 32
TAG = 1024  # LEGACY_FINISHED_SUM_TAG


class SignalMatrix:
    """S[r][t]: the int[8] signal slots living on GPU r (row r), slot t."""

    def __init__(self) -> None:
        self.rows = [[0] * NUM_MAX_NVL_PEERS for _ in range(NUM_MAX_NVL_PEERS)]
        self.lock = threading.Lock()

    def atomic_add_system(self, row: int, col: int, value: int) -> None:
        with self.lock:
            self.rows[row][col] += value

    def ld_volatile(self, row: int, col: int) -> int:
        return self.rows[row][col]


def signal(S: SignalMatrix, rank: int, num_ranks: int) -> None:
    # Lanes thread_id < kNumRanks run these two atomics in parallel on the GPU
    for thread_id in range(num_ranks):
        S.atomic_add_system(rank, thread_id, TAG)   # barrier_signal_ptrs[rank] + thread_id
        S.atomic_add_system(thread_id, rank, -TAG)  # barrier_signal_ptrs[thread_id] + rank


def all_arrived(S: SignalMatrix, rank: int, num_ranks: int, strict_zero: bool = False) -> bool:
    values = [S.ld_volatile(rank, t) if t < num_ranks else 0 for t in range(WARP_SIZE)]
    return all((v == 0) if strict_zero else (v <= 0) for v in values)  # __all_sync


def barrier_block(S: SignalMatrix, rank: int, num_ranks: int, timeout_s: float = 10.0) -> None:
    signal(S, rank, num_ranks)
    start = time.monotonic()
    while not all_arrived(S, rank, num_ranks):
        if time.monotonic() - start > timeout_s:
            raise RuntimeError(f"timeout: rank = {rank}, row = {S.rows[rank]}")
        time.sleep(0)


def stress_test(num_ranks: int = 8, num_barriers: int = 200) -> None:
    S = SignalMatrix()
    arrive = [[0.0] * num_ranks for _ in range(num_barriers)]
    leave = [[0.0] * num_ranks for _ in range(num_barriers)]

    def worker(rank: int) -> None:
        rng = random.Random(rank)
        for k in range(num_barriers):
            time.sleep(rng.random() * 1e-4)  # uneven work before each barrier
            arrive[k][rank] = time.monotonic()
            barrier_block(S, rank, num_ranks)
            leave[k][rank] = time.monotonic()

    threads = [threading.Thread(target=worker, args=(r,)) for r in range(num_ranks)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    for k in range(num_barriers):
        assert min(leave[k]) >= max(arrive[k]), f"barrier {k}: someone left before everyone arrived"
    assert all(v == 0 for row in S.rows for v in row), S.rows
    print(f"[stress] {num_ranks} ranks x {num_barriers} barriers: "
          f"no rank left early, final matrix all zero  PASS")


def scripted_fast_peer() -> None:
    """2 ranks; rank 1 enters the next barrier before rank 0 polls the current one."""
    for strict_zero in (False, True):
        S = SignalMatrix()
        signal(S, 0, 2)                       # rank 0 arrives at barrier k
        print(f"  after rank 0 arrives k : S_0 = {S.rows[0][:2]}, S_1 = {S.rows[1][:2]}")
        signal(S, 1, 2)                       # rank 1 arrives at barrier k
        print(f"  after rank 1 arrives k : S_0 = {S.rows[0][:2]}, S_1 = {S.rows[1][:2]}")
        assert all_arrived(S, 1, 2, strict_zero)  # rank 1 passes barrier k
        signal(S, 1, 2)                       # rank 1 arrives at barrier k+1
        print(f"  after rank 1 arrives k+1: S_0 = {S.rows[0][:2]}, S_1 = {S.rows[1][:2]}")
        passed = all_arrived(S, 0, 2, strict_zero)  # rank 0 finally polls barrier k
        cond = "== 0" if strict_zero else "<= 0"
        print(f"[fast peer] condition {cond}: rank 0 passes barrier k? {passed}")


def main() -> None:
    stress_test()
    scripted_fast_peer()


if __name__ == "__main__":
    main()
```

运行：

```bash
python3 barrier_block_sim.py
```

本机（macOS，Python 3）实际输出：

```text
[stress] 8 ranks x 200 barriers: no rank left early, final matrix all zero  PASS
  after rank 0 arrives k : S_0 = [0, 1024], S_1 = [-1024, 0]
  after rank 1 arrives k : S_0 = [0, 0], S_1 = [0, 0]
  after rank 1 arrives k+1: S_0 = [0, -1024], S_1 = [1024, 0]
[fast peer] condition <= 0: rank 0 passes barrier k? True
  after rank 0 arrives k : S_0 = [0, 1024], S_1 = [-1024, 0]
  after rank 1 arrives k : S_0 = [0, 0], S_1 = [0, 0]
  after rank 1 arrives k+1: S_0 = [0, -1024], S_1 = [1024, 0]
[fast peer] condition == 0: rank 0 passes barrier k? False
```

- `[stress]`：8 个线程各连续过 200 次 barrier，每次到达前随机耽搁一段时间。断言每个 barrier 里最早离开的 rank 也晚于最后到达的 rank，并且结束后 8×8 矩阵全为 0，验证了 barrier 语义和「不用清零可重复使用」。连跑 5 次均通过。
- `[fast peer]`：复现第 6.2 节的场景。`<= 0` 时 rank 0 能通过；换成 `== 0` 时 rank 0 被卡住，而 rank 1 已经在下一个 barrier 等 rank 0，形成死锁。

## 9. 常见错误与症状

| 错误 | 正确做法 | 可观察到的症状 |
|---|---|---|
| 写了要给对端读的数据后用 `kSyncOnly = true` | 写完数据后用默认版本，先 fence 再发信号 | 不会卡死；对端偶尔读到旧计数，接收行数算错，难以复现 |
| 多机时把全局 rank 传给 `barrier_block` | 传 `nvl_rank`（0–7） | `barrier_signal_ptrs[11]` 越界，非法地址访问或乱写内存 |
| 同一张卡上两个 block 同时调用 | 每张卡只让一个固定的 block 使用信号区 | 本卡的格子被加了两次 T，对端只减一次，永远 > 0，约 100 秒后超时 trap |
| block 线程数少于 `kNumRanks` | 保证 `blockDim.x >= kNumRanks` | `EP_DEVICE_ASSERT` 失败；去掉 assert 的话部分 rank 没人负责，对端等到超时 |
| 轮询用普通读而不是 `ld.volatile` | 用 `ld_volatile_global` | 编译器把值缓存在寄存器里，对端已经减到 0 也看不到，表现为假超时 |
| 等待条件写成 `== 0` | 用 `<= 0` | 对端连续进入下一个 barrier 时互相等待，死锁后超时 |
| 某个 rank 少调用一次 barrier（代码分支不一致） | 所有 rank 的 barrier 次数必须相同 | 其他 rank 的对应格子停在 T，超时信息里 `value = 1024`，`thread` 就是没跟上的 rank |

## 10. 速记卡

```text
布局：
  每张卡 NVLink 缓冲区末尾：int[8] 信号区 + void*[8] + int*[8]（指针的显存副本）
  barrier_signal_ptrs = int*[8]，第 t 项指向 GPU t 的 int[8]
  逻辑 8×8 矩阵，第 r 行在 GPU r 上；只在构造时清零一次

到达：
  [kSyncOnly=false] fence.acq_rel.sys + __syncthreads   发布之前写的数据
  线程 t：S_me[t] += T（本地），S_t[me] -= T（NVLink）

等待：
  线程 t 读 S_me[t]（ld.volatile），线程 8..31 给 0
  __all_sync(全部 <= 0) -> break；只有 warp 0 真在轮询，其余 warp 在最后的 __syncthreads 等

不变量：
  S_r[t] = T × (r 到达次数 − t 到达次数)
  <= 0 ⇔ t 已到；用 <= 而不是 == 是为了允许对端先进入下一个 barrier

约束：
  blockDim.x >= kNumRanks；rank 必须是 0..7 的节点内编号；每卡同时只能一个 block 用
  超时约 100s，value/1024 = 本卡比对端多到达的次数
```

## 11. 自测题与答案

1. **`barrier_signal_ptrs` 是长度为 8 的 int 数组吗？每张卡上实际存了多少字节的信号？**
   答：不是。它是 8 个指针（`int*[8]`），第 t 个指向 GPU t 上的 `int[8]`。每张卡只存自己那一行，8 × 4 B = 32 B；指针数组本身和它的显存副本各 64 B。
2. **4 卡、rank 1 的线程 3 在发信号阶段改了哪两个格子，分别在哪张卡上？**
   答：本卡 GPU 1 上的 `S_1[3] += T`，以及通过 NVLink 改 GPU 3 上的 `S_3[1] -= T`。
3. **8 卡中 rank 5 最后一个到达，它要等多久？为什么？**
   答：基本不用等。其他 7 个 rank 已经把 `S_5` 中对应的格子减成了 −T，rank 5 加 T 后全部为 0，第一次轮询就通过。
4. **`value` 只是一个数，`__all_sync` 怎么判断一整行？**
   答：warp 里每个线程有自己的 `value`，线程 t 读 `S_me[t]`。`__all_sync` 汇总 32 个线程的判断，全部为真才返回真。8 个线程各负责一格，合起来就是一整行；线程 8–31 给 0，恒为真。
5. **`notify_dispatch` 开头那次 barrier 为什么可以用 `kSyncOnly = true`，第二次为什么不行？**
   答：开头时本卡还没写任何需要对端读的数据，只需对齐。第二次之前，线程 t 已经把计数写进了 GPU t 的缓冲区，必须先 fence 让这些写入对所有 GPU 可见，再发信号；否则对端通过 barrier 后可能读到旧数据。

## 12. 学习进度

- [x] DeepEP V1 normal：layout → rank dispatch → local expert metadata → handle → combine
- [x] 区分 rank-major 通信布局、expert-major 计算布局与 gate 加权
- [x] DeepEP V1 low-latency：定容 dispatch、packed expert 输入、handle 回程元数据、weighted combine 与 hook
- [x] 机内 `get_dispatch_layout` kernel：block 分工、per-thread 计数 + 按列归约、token-rank 去重、单机 RDMA 分支
- [x] 机内 `barrier_block`：分布式 8×8 信号矩阵、+T/−T 成对原子操作、warp 投票、`kSyncOnly`、`<= 0` 的原因

### 下一知识点

intranode `notify_dispatch`：`sm_id == 0` 的 block 如何借助两次 `barrier_block` 交换各卡的 `num_tokens_per_rank` / `num_tokens_per_expert`，得到 `rank_prefix_matrix` 和 CPU 端的接收计数；其余 block 如何按 channel 算出 `channel_prefix_matrix`。
