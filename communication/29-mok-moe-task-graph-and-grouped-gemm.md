# MoK 的 MoE 计算编排：任务表、CLC 抢活与专家行段映射

> 承接 [02-mok-scheduling-and-buffers.md](./02-mok-scheduling-and-buffers.md)。上一篇讲的是"一张 schedule 怎么产生、buffer 怎么复用"，这一篇讲**这张 schedule 怎么变成 GPU 上的 tile 级计算**：任务表怎么算、cluster 怎么领活、一个 cluster 索引怎么反算到"第几个专家的第几行块"。
>
> 代码基准：`mixture-of-kittens` 本地仓库。
>
> - `mok/functional.py`：`build_schedule` / `forward` / `backward`
> - `csrc/megakernel/types.cuh`：`globals_fwd::grid()`
> - `csrc/megakernel/forward.cuh`：通信段 + 计算段的任务分派
> - `csrc/megakernel/grouped_gemm.cuh`：专家行段 → `tile_coord`
> - `csrc/megakernel/swiglu.cuh`、`csrc/megakernel/dispatch_combine.cuh`
> - 复算脚本：[29_mok_task_table_sim.py](./29_mok_task_table_sim.py)

---

```text
本次讲解位置
章节：communication
小节：MoK megakernel 的 MoE 计算编排
知识点：任务表公式、CLC 抢活、compute_cluster_idx -> (expert, row block, col block) 的整数算术
上次：02 笔记 —— 设备端 schedule、pull/push 方向、macrobatch 环形 buffer
下次：dispatch_kernel 的 MXFP8 转置量化与 e8m0 scale 布局
指令：PTX clusterlaunchcontrol.try_cancel.async / query_cancel.is_canceled（Cluster Launch Control）
```

## 1. 为什么现在讲这个

上一篇把 MoK 的**输入侧**讲清楚了：device 端 schedule 生成 `schedule_peer_rank` / `schedule_peer_token_idx` 两张表，把每个 token 落到本地专家缓冲区的一个确定行号上。但剩下最关键的一步没讲：**这张表怎么驱动计算**。

MoE 训练真正的难点不是 GEMM 本身，而是这三个事实同时成立：

1. 每个 rank 收到的 token 总数 `num_tokens` 只有在 device 端算完 schedule 才知道，host 端拿不到（拿到就要同步）。
2. 每个专家的长度都不一样，且随时会倾斜。
3. grouped GEMM 需要一个确定的 tile 循环边界：M 方向有多少个 256 行块、每个块属于哪个专家。

传统做法在这里必须二选一：要么 host 端等 token 数（D2H 同步 + 动态 launch），要么用 atomic 动态排队（多一个全局计数器、要轮询、任务分配不确定）。MoK 的答案是第三种：

> **把整层 MoE 编译成一张"任务表"，任务数用整数公式从 `num_tokens` 现算，tile 坐标用整数除法从 cluster 索引反算，负载不均交给硬件的 cluster launch control 处理。**

这就是为什么你在 `forward.cuh` 里看到的不是"每个专家一次 launch"，而是一长串 `if (compute_cluster_idx < ...)`。那些整数除法和前缀和不是魔法，是**用算术替代索引数组**：不建表，就不会有建表的 kernel、建表的同步、建表的 buffer。

不懂这套算术，你会误以为 `grid` 就是任务数、误以为 swizzle 是对整个矩阵做的、也会算错每个 barrier 的 `required_count`。这些误解直接对应调试时的"形状对但数值错"和"偶发死锁"。

## 2. 心智模型：一段通信 + 两段计算

一层 MoE 的前向（`functional.forward`）只 launch **一个** kernel。这个 kernel 的 CTA 被分成三段角色，靠 `clusterIdx().x` 与 `num_comm_sms` 划分：

```text
grid.x = CLUSTER_SIZE * (shared_tasks + num_minibatches * minibatch_tasks) + num_comm_sms
         └─────────────── 计算段（cluster，2 CTA 一组）────────────────┘   └ 通信 CTA ┘

cluster_idx ∈ [0, comm_clusters)                      -> 通信 cluster：dispatch / combine
compute_idx = cluster_idx - comm_clusters ∈ [0, 上界)  -> 计算 cluster：领一个 tile 任务
```

计算段内部再分两截：

```text
[0, shared_tasks)                          共享专家：任务数编译期就能算（只依赖 num_local_tokens）
[shared_tasks, ...)                        路由专家：按 (macrobatch, minibatch) 分组，任务数依赖 num_tokens
```

关键区分：

| 名字 | 是什么 | 数量由谁决定 |
|---|---|---|
| **task** | 一个 cluster 做的一次工作：一个 256×256 输出 tile，或 6 个 128×128 SwiGLU tile | device 端整数公式 |
| **cluster** | 一个 CUDA cluster（`CLUSTER_SIZE=2` 个 CTA），`cta_rank` 0/1 各算 128 行 | 物理资源 |
| **grid** | launch 的 CTA 总数 | **capacity 上界**，不是任务数 |
| **slot** | 一个 barrier 计数器的下标 | 由 task 的索引空间决定 |

## 3. 术语、尺寸与不变量

先固定常量（`types.cuh:1-42`），后面所有数字都基于它们：

| 常量 | 值 | 含义 |
|---|---|---|
| `MLP_Mb` | 256 | GEMM 输出 tile 的 M，一个 cluster 把它拆成 2×128 行给两个 CTA |
| `MLP_Nb` | 256 | GEMM 输出 tile 的 N |
| `MLP_BF16_Kb` / `MLP_FP8_Kb` | 64 / 128 | K 方向分块（BF16 / MXFP8） |
| `MLP_LOAD_PIPE_DEPTH` | 6 | A/B 操作数的 TMA 流水级数 |
| `SWIGLU_Mb` / `SWIGLU_Nb` | 128 / 128 | SwiGLU tile |
| `SWIGLU_FWD_PIPE_DEPTH` | 3 | SwiGLU 前向流水级数 |
| `CLUSTER_SIZE` | 2 | 一个 cluster 的 CTA 数 |
| `DISPATCH_Mb` / `DISPATCH_Nb` | 128 / 512 | dispatch 每次搬的 chunk |
| `COMBINE_Mb` / `COMBINE_Nb` | 16 / 1024 | combine 每次搬的 chunk（更细，7 级流水） |
| `EXPERT_PADDING` | 256 | schedule 给每个专家段补齐到的行数（`scheduler.cuh:12`） |

四条不变量，后面反复使用：

1. `hidden_size % 256 == 0`、`num_local_tokens % 256 == 0`（`functional.py:138-146`）。
2. **每个专家段的行数都是 256 的倍数**。因此专家 `e` 的起始行 = `tokens_per_expert[0..e)` 的前缀和，不需要任何"行 → 专家"的索引数组。
3. 任务数只依赖 `num_tokens` 和编译期常量，所以 device 端可以现算（这是"零 CPU 同步"的前提）。
4. 一个 cluster 的两个 CTA 各负责 128 行；全局行号写成 `2 * tile_row_block + cta_rank`。

## 4. 任务表怎么算

### 4.1 公式

共享专家段（`types.cuh:139-152`），注意这里用的是 `num_local_tokens`，不含路由信息：

```text
shared_row_blocks        = num_local_tokens / 256
shared_gate_up_tasks     = shared_row_blocks * (I / 256)          # gate 和 up 各一份
shared_swiglu_tiles      = (num_local_tokens/128) * (I/128)
shared_swiglu_tasks      = ceil(shared_swiglu_tiles / (2 * 3))     # CLUSTER_SIZE * SWIGLU_FWD_PIPE_DEPTH
shared_down_tasks        = shared_row_blocks * (H / 256)
shared_tasks             = 2*shared_gate_up_tasks + shared_swiglu_tasks + shared_down_tasks
```

路由专家段，**以 minibatch 为单位**（不是 macrobatch —— 粒度就是重叠粒度）：

```text
mb_gate_up_tasks         = (minibatch_size/256) * (I/256)          # gate 和 up 各一份
mb_swiglu_tiles          = (minibatch_size/128) * (I/128)
mb_swiglu_tasks          = ceil(mb_swiglu_tiles / 6)
mb_down_tasks            = (minibatch_size/256) * (H/256)
minibatch_tasks          = 2*mb_gate_up_tasks + mb_swiglu_tasks + mb_down_tasks
```

### 4.2 数值（EP=8, H=7168, I=2048, E_local=32, topk=8, T_local=8192）

`schedule_capacity = T_local * topk * max(2, ceil(EP * 0.5)) = 8192 * 8 * 4 = 262144`。
均衡路由下每 rank 收到 `8*8192*8/8 = 65536` 个 token，每专家 2048 个（正好 256 的倍数，padding 为 0）。

| 项 | 值 |
|---|---|
| `shared_row_blocks` | 32 |
| `shared_gate_up_tasks`（每份） | 32 × 8 = 256 |
| `shared_swiglu_tasks` | ceil(1024 / 6) = 171 |
| `shared_down_tasks` | 32 × 28 = 896 |
| **`shared_tasks`** | 2×256 + 171 + 896 = **1579** |
| `mb_gate_up_tasks`（每份） | 16 × 8 = 128 |
| `mb_swiglu_tasks` | ceil(512 / 6) = 86 |
| `mb_down_tasks` | 16 × 28 = 448 |
| **`minibatch_tasks`** | 256 + 86 + 448 = **790** |
| `true_minibatches` | ceil(65536 / 4096) = 16 |
| **真实计算 cluster 数** | 1579 + 16 × 790 = **14239** |

### 4.3 grid 是上界，不是任务数

这里是最容易看错的地方。`globals_fwd::grid()` 用的是 **`schedule_peer_rank.cols()`（capacity）**，不是 `num_tokens`：

```text
num_minibatches(grid) = ceil(262144 / 4096) = 64        # 真实只有 16
grid.x = 2 * (1579 + 64 * 790) + 40 = 104318 CTA         # = 52159 个 cluster
```

于是 launch 出来 52159 个 cluster，但只有 14239 个该干活，**37900 个 cluster 会被取消掉**（下一节的 CLC）。这个上界之所以松，是因为 `schedule_capacity_multiplier=0.5` 意味着"最坏情况下一个 rank 拿走一半的 token"，capacity 是均值的 4 倍。

要理解的关键：**开大 grid 的代价在 launch 队列里，不在 SM 上**。还没启动的 cluster 不占 SM、不耗执行资源，被取消只是队列操作。真正决定性能的是"实际跑起来的 14239 个 cluster 怎么排布"。

## 5. CLC：上界 grid 怎么变成任务队列

### 5.1 指令语义

Blackwell 的 Cluster Launch Control 两条指令（`third_party/ThunderKittens/include/ops/thread/util/util.cuh:257-287`）：

```text
clusterlaunchcontrol.try_cancel.async.shared::cta.mbarrier::complete_tx::bytes.multicast::cluster::all.b128
clusterlaunchcontrol.query_cancel.is_canceled.pred.b128   +   get_first_ctaid.v4.b32
```

语义是：**取消一个"尚未启动"的 cluster，并接管它的坐标**。

- 发起方是 cluster 级操作，**整个 cluster 只允许一个线程发**（`forward.cuh:148-153`：`cta_rank == 0`、warp 1、`warp::elect_leader()`）。
- 结果 handle 通过 `multicast::cluster::all` 写到 cluster 内每个 CTA 的 shared memory，同时 complete 一个 mbarrier 的 tx bytes。所以代码里是 `tma::expect_bytes(schedule_arrived[stage], sizeof(clc_handle))` 再 `wait`。
- `query` 是一次性的：注释明确写 "Calling this again after failure is undefined behavior"，所以每个 handle 只能查一次。

### 5.2 握手与计数

```cpp
// forward.cuh:146-160（节选）
if (warpgroup::groupid() == config::NUM_CONSUMERS && warpgroup::warpid() == 1 && warp::elect_leader()) {
    if (cta_rank == 0) {
        wait(schedule_finished[clc_stage], ((task_iter + CLC_PIPE_DEPTH) / CLC_PIPE_DEPTH) % 2);
        clc::schedule(clc_handle[clc_stage], schedule_arrived[clc_stage]);
    }
    tma::expect_bytes(schedule_arrived[clc_stage], sizeof(clc_handle[clc_stage]));
}
...
wait(schedule_arrived[clc_stage], (task_iter / config::CLC_PIPE_DEPTH) % 2);
const auto schedule = clc::query(clc_handle[clc_stage]);
cluster_idx = schedule.success ? static_cast<int>(schedule.x / config::CLUSTER_SIZE) : -1;
__syncwarp();
warp::tma::cluster::arrive(schedule_finished[clc_stage], 0);
```

两个 semaphore 的 count 不是随便写的：

| semaphore | count | 为什么 |
|---|---|---|
| `schedule_arrived` | `1` | handle 通过 multicast 到达，`expect_bytes` 一次收齐 |
| `schedule_finished` | `CLUSTER_SIZE * NUM_WARPS` = 2×8 = **16** | cluster 内**每个 warp** 都要在 `query` 之后 arrive，保证所有 warp 都拿到新 `cluster_idx`，才允许复用这个 handle 发下一次 `try_cancel` |

`CLC_PIPE_DEPTH = 1` 意味着没有重叠（先 query 完才发下一次），这是刻意的：query 之后才能知道下一个 index，没法预取。

### 5.3 三种退出路径

```text
1. cluster_idx >= true_num_clusters        -> 主循环开始前就 return（多开的 cluster）
2. query 返回 success == false             -> 队列空了，cluster_idx = -1，循环退出
3. 拿到 >= true_num_clusters 的 index       -> 退出主循环，进入 drain
```

drain 段（`forward.cuh:270-291`）是一段看着奇怪但必要的代码：

```cpp
// CLC drain for no-op threadblocks
if (cluster_idx >= 0 && warp::laneid() == 0) {
    const int stage = warpid();
    ... clc::schedule(...); while (true) { wait(...); query(...); if (!schedule.success) break; ... }
}
```

含义：已经退出主循环的 cluster **继续 `try_cancel` 直到队列排空**。按实现和注释推断，这样做是为了把 launch 队列里剩下的待取消 cluster 全部消费掉；`CLC_DRAIN_PIPE_DEPTH = 8` 正好对应 8 个 warp 并行排空，让 kernel 尽快结束。

> 确定 vs 推断：`try_cancel` 的语义、两点握手、count 值都是代码里能验证的；"drain 是为了让 kernel 能退出/加快退出"是按注释 `CLC drain for no-op threadblocks` 做的推断，不同 CUDA 版本的具体行为需要实测确认。

### 5.4 为什么敢让"哪个 cluster 算哪个 tile"不确定

CLC 让物理 cluster 与任务的对应关系变成运行时决定。MoK 的结果仍然 bitwise deterministic，靠的不是 CLC，而是另外三条：

1. 每个输出 tile **只有一个写者**（一个 task 独占它负责的 256×256 区域）。
2. 跨 task 的加法（wgrad）由 `routed_buffers_done` 按 macrobatch 串行化，加法顺序固定（`grouped_gemm.cuh:285-288` 的注释明确写了这一点）。
3. schedule 本身是确定性排序（上一篇讲过：按 peer rank 再按 token 顺序）。

## 6. 从 cluster 索引反算 tile 坐标

这一段是整篇的核心。取 `compute_cluster_idx = 5629`，一步步走。

### 6.1 先算 (macrobatch, minibatch, 任务族)

```text
compute_cluster_idx = 5629
减 shared_tasks     : 5629 - 1579 = 4050
ordered = 4050 / 790 = 5   ,  task_idx = 4050 - 5*790 = 100
100 < mb_gate_up_tasks(128)  ->  routed gate，task_idx 仍是 100
num_macrobatches = 1，所以 ordered=5 < last_mb_minibatches(16) -> macrobatch=0, minibatch=5
```

把三个量写进表：

| 量 | 表达式 | 值 |
|---|---|---|
| `macrobatch_idx` | 分支判定 | 0 |
| `minibatch_idx` | `ordered` | 5 |
| `global_minibatch_idx` | `macrobatch*(macrobatch_size/minibatch) + minibatch` | 0×32 + 5 = 5 |
| `task_idx` | 任务族内偏移 | 100 |

### 6.2 再用前缀和找专家（`grouped_gemm.cuh:96-110`）

```text
mb_row_blocks              = 4096/256 = 16
first_rb                   = global_minibatch_idx * 16 = 80        # 本 minibatch 覆盖全局行块 [80, 96)
macrobatch_row_block_offset = 0
每专家行块数（均衡时）       = 2048/256 = 8
```

循环每个专家，取 **(专家段) ∩ (minibatch 行窗口)** 作为 swizzle 域：

| expert | 段区间 `[offset, offset+8)` | `lo = max(80, offset)` | `hi = min(96, offset+8)` | `rb = hi-lo` | `tasks = rb*8` | `task_idx` 之后 |
|---|---|---|---|---|---|---|
| 0..9 | [0,80) | 80 | ≤80 | 0 | 0 | 100 |
| 10 | [80,88) | 80 | 88 | 8 | 64 | 100 ≥ 64 → **36** |
| 11 | [88,96) | 88 | 96 | 8 | 64 | 36 < 64 → **命中** |

所以这个 cluster 算的是**专家 11**。

### 6.3 swizzle 到 (row block, col block)

`get_swizzled_2d_idx<8>(num_rows=8, num_cols=8, idx=36)`（`third_party/ThunderKittens/include/common/util.cuh:367`，`ROW_MAJOR=true`）：

```text
supergroup_numel    = num_rows * 8 = 64
supergroup_idx      = 36 / 64 = 0
supersection_cols   = (8/8)*8 = 8 ; supersection_numel = 64
36 < 64  ->  row_idx = (36 % 64) / 8 = 4
             col_idx = 0*8 + 36 % 8   = 4
supergroup_idx 为偶数 -> 不翻转
```

| 量 | 值 |
|---|---|
| `swizzled` | (4, 4) |
| `tile_coord.x` = `lo + 4 - 0` | 92 |
| `tile_coord.y` | 4 |
| `tile_coord.z` | 11（专家） |
| 覆盖的 routed 行 | 92×256 … 93×256 = **[23552, 23808)** |
| 覆盖的 I 列 | 4×256 … 5×256 = **[1024, 1280)** |
| 写回时的 128 行 tile | `2*92 + cta_rank` = 184 / 185 |

交叉验证：专家 11 的段是 `[88*256, 96*256) = [22528, 24576)`，23552 落在里面 ✓。更上层再验证：第 23552 行在 `schedule_peer_rank` 里的那个值，就是"这个 tile 的第 0 行来自哪个 rank 的哪个 token"。

### 6.4 swizzle 在优化什么

`SUPERGROUP_SIZE = 8` 的蛇形（boustrophedon）映射：在一个 8 行块 × 8 列块的方阵里，任务先沿列走、再换列；`supergroup_idx` 为奇数时行序整体翻转，使相邻列 supergroup 的行号首尾相接。这样同一个 wave 里跑的 tile 集中在 8×8 的方块内，A 操作数（routed 激活）和 B 操作数（专家权重）的工作集都能压进 L2。

注意 swizzle 的域是 **(专家 ∩ minibatch) 的交集矩形**，不是整个矩阵。如果一个专家跨了多个 minibatch（`tokens_per_expert` 不是 `minibatch_size` 的倍数，或者专家特别热），每个 minibatch 会各自算一份 `rb`，交集矩阵变窄，swizzle 域也随之变窄。这是"grouped"的固有代价：**负载均衡和 L2 局部性只能在小矩形内做**。

## 7. 五段数据流的计数器契约

一段话：**MoK 的融合边界不是 kernel 边界，而是计数器。** 每个生产者写完一批数据，对某个 slot 做一次 `red.release.gpu.add`；消费者用 `ld.relaxed.gpu` 轮询到阈值，再 `fence.acquire.gpu`（`csrc/megakernel/utils.cuh:1-12`）。

前向的五个计数器（用 4.2 的数值）：

| 计数器 | 索引空间（slot） | 生产者 | 消费者 | `required_count` | 值 |
|---|---|---|---|---|---|
| `x_routed_ready` | 每个 global minibatch | dispatch（每 128×512 chunk 一次） | gate/up GEMM 读 A 前 | `ceil(minibatch/128) * ceil(H/512)` | 32 × 14 = **448** |
| `gate_up_tile_ready` | 每个父 256×256 tile | gate GEMM + up GEMM | SwiGLU 读 gate/up 前 | `2 * CLUSTER_SIZE` | **4** |
| `hidden_row_block_ready` | 每个 256 行块 | SwiGLU（每 128×128 tile 一次） | down GEMM 读 A 前 | `(256/128) * (I/128)` | 2 × 16 = **32** |
| `y_routed_ready` | 每个 global minibatch | down GEMM（每个输出 task 一次） | combine 读之前 | `(minibatch/256) * (H/256) * CLUSTER_SIZE` | 16 × 28 × 2 = **896** |
| `y_routed_done` | 每个 128 行块 | combine（`ARRIVE_BY_ROW`） | **上一代** macrobatch 的 down GEMM epilogue 写之前 | `(128/16) * ceil(H/1024)` | 8 × 7 = **56** |

四个值得停下来看的点：

**(a) `x_routed_ready` 的 448 不是"整个 macrobatch 的 chunk 数"。** 如果写成 7168（= 65536/128×14），gate GEMM 就会一直等而不动 —— 因为下一个 macrobatch 的 dispatch 还没跑，永远凑不满。slot 的选择直接决定了重叠粒度：按 minibatch 计数，dispatch 推进到第 6 个 minibatch，第 5 个 minibatch 的 GEMM 就能开跑。

**(b) `gate_up_tile_ready` 是 4。** SwiGLU 需要 gate 和 up 两个 GEMM 的同一个 tile 都完成，每个 GEMM 由 `CLUSTER_SIZE=2` 个 CTA 各 arrive 一次：

```cpp
// swiglu.cuh:106
barrier_wait(gate_up_tile_ready, gate_up_tile_ready_base_index + parent_task_idx, 2 * config::CLUSTER_SIZE);
```

**(c) shared 与 routed 共用同一批计数器数组，靠 base index 拼接。** down GEMM 的 `input_row_block_ready_base_index = shared_row_blocks = 32`，SwiGLU 的 `gate_up_tile_ready_base_index = shared_gate_up_tasks = 256`。这样 shared expert 和 routed expert 用同一套代码路径，只是索引偏移不同。

**(d) `y_routed_done` 的语义是"上一代 ring 已经读完"。** 因为 `y_routed` 只有 `macrobatch_size` 行，会被下一轮覆盖。代码里写着 `previous_macrobatch_offset = (macrobatch_idx + 1) * macrobatch_size` —— 叫 "previous" 却用 `+1`，因为**macrobatch 是按倒序执行的**：先算最后一个 macrobatch，所以在执行序里"上一代"是 ring 索引更大的那个。

## 8. 为什么 router 加权放在 epilogue

dispatch 只搬运，不乘权重；combine 只按 `orig_token*topk + k` 归位，也不乘权重。真正的加权在 forward 末尾的一个小 kernel 里（`csrc/utils.cuh:43-93`）：

```text
output[t] = y_shared[t] + Σ_k router_weight[t][k] * combine_buffer[t*topk + k]
```

原因很直接：**dispatch 乘了会破坏语义**。route 的方向是 token→expert，同一个 token 的 top-k 份权重不同；如果 dispatch 时乘上权重，expert 算出来的 `down(silu(gate(x*w)))` 就不是 `w * down(...)`，MLP 是非线性的，权重不能在入口乘。反过来，combine 时如果加权，也只能加权在"某一份 k"上，但 combine 是按 slot 精确回填的，加法留给 epilogue 更省事（一次读 topk+1 行向量，`rv_fl` 累加，`TOKENS_PER_CTA=2`）。

反向是镜像（`csrc/utils.cuh:140-174`）：`d_x = d_x_shared + Σ_k d_x_routed[t*topk+k]`，**没有权重**，因为权重在反向 combine 时已经乘进 `d_y` 了（`backward.cuh:133` 用 `dispatch_kernel<true>`，即 `SCALE_ROWS=true`）：

- BF16 路径：在搬到 shared staging 之后逐元素乘 `router_weights_smem`（`dispatch_combine.cuh:117-128`）。
- MXFP8 路径：把权重传给 `mxfp8::quantize_tile<..., SCALE_ROWS=true>`，在量化时一并乘掉（`dispatch_combine.cuh:104-105`）。

所以 dgrad 之后的每一段都只处理"已经带权重的 d_y"，不需要再回头查 router。

## 9. backward：镜像 + replay

backward 的任务表比 forward 复杂，因为它要处理"哪些激活还活着"：

```text
true_num_clusters = comm_clusters + shared_tasks
                  + num_minibatches * minibatch_bwd_tasks          # 每个 minibatch 的 dgrad + swiglu bwd + wgrad
                  + num_replay_minibatches * minibatch_replay_tasks # 第 1..N 个 macrobatch 补跑前向
                  + num_macrobatches * wgrad_tasks                  # 每个 macrobatch 一轮 wgrad
```

三档任务（`backward.cuh:186-207` 的 `is_cta_local_task` 判定逻辑与之对应）：

| 档 | 内容 | 数据来源 |
|---|---|---|
| `saved` | macrobatch 0 的 dgrad + swiglu bwd + wgrad | forward 存下来的 `gate/up/hidden` |
| `replay` | macrobatch ≥1 的 dispatch + gate GEMM + up GEMM + SwiGLU | 用**同一张 schedule** 重跑，只为重新产出 gate/up/hidden |
| `bwd` | macrobatch ≥1 的 dgrad + swiglu bwd + wgrad | 上一档刚重算出来的激活 |

为什么可以只存 macrobatch 0？因为 forward 的输出（`y_routed`）和输入（`x_fp8_routed`）都是 ring buffer，跨 macrobatch 会被覆盖；真正需要长期保留的只有 `x`（用户传的）和 schedule。其余激活重算比存储便宜 —— 这就是 `functional.recompute_forward_context` 的底层机制，也是 README 里"manual activation checkpointing"那段。

反向的计数器（`backward.cuh:568-572` 创建）只是把前向的依赖图反过来：

```text
d_y_routed_ready      : 反向 combine(dispatch<true>) -> dgrad down
d_hidden_ready        : dgrad down             -> swiglu bwd
d_gate_up_ready       : swiglu bwd             -> dgrad gate/up 与 wgrad gate/up
d_x_routed_ready      : dgrad gate/up          -> 反向 dispatch
replayed_x_routed_ready / replayed_gate_up_ready / replayed_hidden_ready : replay 三段内部
routed_buffers_done   : 整个 macrobatch 的所有 routed 任务 + combine -> 允许下一轮覆盖 ring
```

`routed_buffers_done` 的 `required_count` 是动态算的（`backward.cuh:50-52`）：

```text
CLUSTER_SIZE * (num_minibatches_of(mb) * minibatch_bwd_tasks + wgrad_tasks) + num_combine_tasks_of(mb)
```

这是整份代码里少数"每个 macrobatch 的期望值都不一样"的计数器，因为最后一个 macrobatch 的 minibatch 数可能是残缺的。

## 10. 可运行复现

下面这段脚本在 CPU 上完整复算第 4~7 节的每一个数字，不需要 GPU。完整版（含 dispatch/combine chunk、双 macrobatch 排序）见 [29_mok_task_table_sim.py](./29_mok_task_table_sim.py)。

```python
"""Reconstruct the MoK forward task table and tile mapping on CPU."""

import math
from dataclasses import dataclass

MLP_Mb, MLP_Nb = 256, 256
SWIGLU_Mb, SWIGLU_Nb = 128, 128
SWIGLU_FWD_PIPE_DEPTH = 3
CLUSTER_SIZE = 2
SUPERGROUP_SIZE = 8
DISPATCH_Mb, DISPATCH_Nb = 128, 512
COMBINE_Mb, COMBINE_Nb = 16, 1024
EXPERT_PADDING = 256


@dataclass(frozen=True)
class Cfg:
    ep_size: int
    hidden: int
    intermediate: int
    num_local_experts: int
    topk: int
    num_local_tokens: int
    minibatch_size: int
    macrobatch_size: int
    num_comm_sms: int
    schedule_capacity_multiplier: float = 0.5

    @property
    def schedule_capacity_factor(self) -> int:
        return max(2, math.ceil(self.ep_size * self.schedule_capacity_multiplier))

    @property
    def schedule_capacity(self) -> int:
        return self.num_local_tokens * self.topk * self.schedule_capacity_factor


def swizzled_2d_idx(num_rows: int, num_cols: int, linear_idx: int,
                    supergroup: int = SUPERGROUP_SIZE) -> tuple[int, int]:
    """kittens::get_swizzled_2d_idx, ROW_MAJOR = true."""
    supergroup_numel = num_rows * supergroup
    supergroup_idx = linear_idx // supergroup_numel
    supersection_cols = (num_cols // supergroup) * supergroup
    supersection_numel = num_rows * supersection_cols
    if linear_idx < supersection_numel:
        row_idx = (linear_idx % supergroup_numel) // supergroup
        col_idx = supergroup_idx * supergroup + linear_idx % supergroup
    else:
        remainder = linear_idx - supersection_numel
        finalsection_cols = num_cols - supersection_cols
        row_idx = remainder // finalsection_cols
        col_idx = supersection_cols + remainder % finalsection_cols
    if supergroup_idx & 1:
        row_idx = num_rows - row_idx - 1
    return row_idx, col_idx


def balanced_tokens_per_expert(cfg: Cfg, num_tokens_unpadded: int) -> list[int]:
    """Even routing, then pad every expert segment up to 256 rows."""
    base, rest = divmod(num_tokens_unpadded, cfg.num_local_experts)
    per_expert = [base + (1 if i < rest else 0) for i in range(cfg.num_local_experts)]
    return [math.ceil(n / EXPERT_PADDING) * EXPERT_PADDING for n in per_expert]


def fwd_task_table(cfg: Cfg, tokens_per_expert: list[int]) -> dict:
    num_tokens = sum(tokens_per_expert)
    shared_rb = cfg.num_local_tokens // MLP_Mb
    int_cb = cfg.intermediate // MLP_Nb
    hid_cb = cfg.hidden // MLP_Nb

    shared_gate_up = shared_rb * int_cb
    shared_swiglu_tiles = (cfg.num_local_tokens // SWIGLU_Mb) * (cfg.intermediate // SWIGLU_Nb)
    shared_swiglu = math.ceil(shared_swiglu_tiles / (CLUSTER_SIZE * SWIGLU_FWD_PIPE_DEPTH))
    shared_down = shared_rb * hid_cb
    shared_tasks = 2 * shared_gate_up + shared_swiglu + shared_down

    mb_rb = cfg.minibatch_size // MLP_Mb
    mb_gate_up = mb_rb * int_cb
    mb_swiglu_tiles = (cfg.minibatch_size // SWIGLU_Mb) * (cfg.intermediate // SWIGLU_Nb)
    mb_swiglu = math.ceil(mb_swiglu_tiles / (CLUSTER_SIZE * SWIGLU_FWD_PIPE_DEPTH))
    mb_down = mb_rb * hid_cb
    minibatch_tasks = 2 * mb_gate_up + mb_swiglu + mb_down

    cap_minibatches = math.ceil(cfg.schedule_capacity / cfg.minibatch_size)
    grid = CLUSTER_SIZE * (shared_tasks + cap_minibatches * minibatch_tasks) + cfg.num_comm_sms

    minibatches_per_mb = cfg.macrobatch_size // cfg.minibatch_size
    num_macrobatches = math.ceil(num_tokens / cfg.macrobatch_size)
    true_minibatches = math.ceil(num_tokens / cfg.minibatch_size)
    last_mb_minibatches = true_minibatches - (num_macrobatches - 1) * minibatches_per_mb

    return dict(num_tokens=num_tokens, shared_gate_up=shared_gate_up,
                shared_swiglu=shared_swiglu, shared_down=shared_down,
                shared_tasks=shared_tasks, mb_gate_up=mb_gate_up, mb_swiglu=mb_swiglu,
                mb_down=mb_down, minibatch_tasks=minibatch_tasks,
                cap_minibatches=cap_minibatches, grid=grid,
                comm_clusters=cfg.num_comm_sms // CLUSTER_SIZE,
                true_clusters=(cfg.num_comm_sms // CLUSTER_SIZE) + shared_tasks
                              + true_minibatches * minibatch_tasks,
                num_macrobatches=num_macrobatches, true_minibatches=true_minibatches,
                last_mb_minibatches=last_mb_minibatches,
                minibatches_per_mb=minibatches_per_mb)


def resolve_fwd_cluster(cfg: Cfg, tokens_per_expert: list[int], t: dict,
                        compute_cluster_idx: int) -> dict:
    if not 0 <= compute_cluster_idx < t["shared_tasks"] + t["true_minibatches"] * t["minibatch_tasks"]:
        return {"kind": "out_of_range"}

    if compute_cluster_idx < t["shared_tasks"]:
        if compute_cluster_idx < t["shared_gate_up"]:
            return {"kind": "shared_gate", "task_idx": compute_cluster_idx}
        if compute_cluster_idx < 2 * t["shared_gate_up"]:
            return {"kind": "shared_up", "task_idx": compute_cluster_idx - t["shared_gate_up"]}
        if compute_cluster_idx < 2 * t["shared_gate_up"] + t["shared_swiglu"]:
            return {"kind": "shared_swiglu",
                    "task_idx": compute_cluster_idx - 2 * t["shared_gate_up"]}
        return {"kind": "shared_down",
                "task_idx": compute_cluster_idx - 2 * t["shared_gate_up"] - t["shared_swiglu"]}

    ordered = (compute_cluster_idx - t["shared_tasks"]) // t["minibatch_tasks"]
    task_idx = (compute_cluster_idx - t["shared_tasks"]) - ordered * t["minibatch_tasks"]
    if ordered < t["last_mb_minibatches"]:
        macrobatch_idx, minibatch_idx = t["num_macrobatches"] - 1, ordered
    else:
        idx = ordered - t["last_mb_minibatches"]
        macrobatch_idx = t["num_macrobatches"] - 2 - idx // t["minibatches_per_mb"]
        minibatch_idx = idx % t["minibatches_per_mb"]

    int_cb = cfg.intermediate // MLP_Nb
    hid_cb = cfg.hidden // MLP_Nb
    if task_idx < t["mb_gate_up"]:
        family = "routed_gate"
    elif task_idx < 2 * t["mb_gate_up"]:
        family = "routed_up"
        task_idx -= t["mb_gate_up"]
    elif task_idx < 2 * t["mb_gate_up"] + t["mb_swiglu"]:
        return {"kind": "routed_swiglu", "macrobatch_idx": macrobatch_idx,
                "minibatch_idx": minibatch_idx,
                "task_idx": task_idx - 2 * t["mb_gate_up"]}
    else:
        family = "routed_down"
        task_idx -= 2 * t["mb_gate_up"] + t["mb_swiglu"]

    col_blocks = int_cb if family != "routed_down" else hid_cb
    global_mb = macrobatch_idx * (cfg.macrobatch_size // cfg.minibatch_size) + minibatch_idx
    mb_rb = cfg.minibatch_size // MLP_Mb
    first_rb = global_mb * mb_rb
    mb_rb_offset = macrobatch_idx * (cfg.macrobatch_size // MLP_Mb)

    offset = 0
    for expert_idx, n in enumerate(tokens_per_expert):
        expert_rb = n // MLP_Mb
        lo = max(first_rb, offset)
        rb = max(0, min(first_rb + mb_rb, offset + expert_rb) - lo)
        num_tasks = rb * col_blocks
        if task_idx < num_tasks:
            sr, sc = swizzled_2d_idx(rb, col_blocks, task_idx)
            return {"kind": family, "macrobatch_idx": macrobatch_idx,
                    "minibatch_idx": minibatch_idx, "expert_idx": expert_idx,
                    "intersect_row_blocks": rb, "col_blocks": col_blocks,
                    "tile_row_block": lo + sr - mb_rb_offset, "tile_col_block": sc,
                    "first_routed_row": (lo + sr) * MLP_Mb}
        task_idx -= num_tasks
        offset += expert_rb
    return {"kind": "no_tile"}


def counter_contract(cfg: Cfg) -> dict:
    hid_cb = cfg.hidden // MLP_Nb
    d_cb = math.ceil(cfg.hidden / DISPATCH_Nb)
    c_cb = math.ceil(cfg.hidden / COMBINE_Nb)
    return {
        "x_routed_ready[minibatch]": (cfg.minibatch_size // DISPATCH_Mb) * d_cb,
        "gate_up_tile_ready[tile]": 2 * CLUSTER_SIZE,
        "hidden_row_block_ready[row_block]": (MLP_Mb // SWIGLU_Mb) * (cfg.intermediate // SWIGLU_Nb),
        "y_routed_ready[minibatch]": (cfg.minibatch_size // MLP_Mb) * hid_cb * CLUSTER_SIZE,
        "y_routed_done[row_block_128]": (MLP_Mb // CLUSTER_SIZE // COMBINE_Mb) * c_cb,
    }


def main() -> None:
    cfg = Cfg(ep_size=8, hidden=7168, intermediate=2048, num_local_experts=32,
              topk=8, num_local_tokens=8192, minibatch_size=4096,
              macrobatch_size=131072, num_comm_sms=40)
    tpe = balanced_tokens_per_expert(cfg, cfg.num_local_tokens * cfg.topk)

    print("schedule_capacity =", cfg.num_local_tokens, "*", cfg.topk, "*",
          cfg.schedule_capacity_factor, "=", cfg.schedule_capacity)
    t = fwd_task_table(cfg, tpe)
    for k, v in t.items():
        print(f"{k:>22} = {v}")
    print("tokens_per_expert[0:2] =", tpe[0:2])

    c = t["shared_tasks"] + 5 * t["minibatch_tasks"] + 100
    print(f"\nresolve compute_cluster_idx={c}")
    for k, v in resolve_fwd_cluster(cfg, tpe, t, c).items():
        print(f"  {k:>22} = {v}")

    print("\ncounter contracts")
    for k, v in counter_contract(cfg).items():
        print(f"  {k:>34} = {v}")


main()
```

运行（把上面的脚本存成任意文件名，例如 `/tmp/mok_task_table.py`）：

```bash
python3 /tmp/mok_task_table.py
```

期望输出（本机 Python 3.12 实测，与第 4~7 节的所有数字一致）：

```text
schedule_capacity = 8192 * 8 * 4 = 262144
            num_tokens = 65536
        shared_gate_up = 256
         shared_swiglu = 171
           shared_down = 896
          shared_tasks = 1579
            mb_gate_up = 128
             mb_swiglu = 86
               mb_down = 448
       minibatch_tasks = 790
       cap_minibatches = 64
                  grid = 104318
         comm_clusters = 20
         true_clusters = 14239
      num_macrobatches = 1
      true_minibatches = 16
   last_mb_minibatches = 16
    minibatches_per_mb = 32
tokens_per_expert[0:2] = [2048, 2048]

resolve compute_cluster_idx=5629
                    kind = routed_gate
          macrobatch_idx = 0
           minibatch_idx = 5
              expert_idx = 11
    intersect_row_blocks = 8
              col_blocks = 8
          tile_row_block = 92
          tile_col_block = 4
        first_routed_row = 23552

counter contracts
           x_routed_ready[minibatch] = 448
            gate_up_tile_ready[tile] = 4
   hidden_row_block_ready[row_block] = 32
           y_routed_ready[minibatch] = 896
        y_routed_done[row_block_128] = 56
```

`grid = 104318` 里只有 `2 × 14239 + 40` 个 CTA 真的干活，差值就是 `cancelled_clusters = 37900`（本节脚本没有打印这一项，完整版 [29_mok_task_table_sim.py](./29_mok_task_table_sim.py) 会打印，并额外算 dispatch/combine 的 chunk 数）：

```bash
cd ~/Downloads/learning_notes/communication && python3 29_mok_task_table_sim.py
```

```text
  cancelled_clusters = 37900
              dispatch_tasks = 7168          # 每个 chunk 128 KiB
               combine_tasks = 4096          # 每个 task 7 个 16x1024 chunk
                routed_bytes = 939524096     # 一个 macrobatch 的 routed 激活字节数
```

再补一个 padding 的例子：未补齐的 `num_tokens = 150000`，32 个专家各补到 256 的倍数后变成 155648（多 3.8%），于是 `num_macrobatches = 2`、`true_minibatches = 38`、`last_mb_minibatches = 6`，路由任务的发射顺序变成（同一脚本实测）：

```text
ordered_minibatch= 0 -> macrobatch=1 minibatch=0    # 先发最后一个 macrobatch
ordered_minibatch= 5 -> macrobatch=1 minibatch=5
ordered_minibatch= 6 -> macrobatch=0 minibatch=0    # 再倒着发更早的
ordered_minibatch=36 -> macrobatch=0 minibatch=30
```

这就是"通信先动、计算跟上"的排布：通信 SM 从最后一个 macrobatch 的 dispatch 开始，计算 cluster 立刻有活干。

## 11. 常见错误与症状

| 错误 | 症状 | 真实规则 |
|---|---|---|
| 以为 `grid` 就是任务数 | 估算 kernel 数/overhead 时高估 3~4 倍 | grid 按 capacity 上界算，真实任务数在 device 端 |
| 按"整个 macrobatch"算 `x_routed_ready` 的 `required_count` | GEMM 永远等不满，kernel 挂住（只在大 token 数下出现） | slot 是 global minibatch，值 = 448 |
| 忘记每个专家补齐到 256 | 专家之间 token 串味：形状对、数值错、topk 越靠后错得越多 | `EXPERT_PADDING=256`，段起点是前缀和 |
| 以为 swizzle 作用于整个 (M, N) | 手算 tile 坐标不一致，尤其专家长度不等时 | swizzle 域是 (专家 ∩ minibatch) 交集矩形 |
| `required_count` 少乘 `CLUSTER_SIZE` | 提前放行 → 偶发数值错误（读到半张 tile） | 一个 GEMM task = 2 个 CTA 各 arrive 一次 |
| 认为 CLC 保证"cluster i 恒算 task i" | 拿它做性能归因会错位 | CLC 只是硬件任务队列；确定性靠单写者 + 串行化加法 |
| 改了 `minibatch_size` 没保持 `macrobatch_size` 的整除关系 | Python 层直接 `ValueError` | `macrobatch_size % minibatch_size == 0` |
| 用 BF16 权重走 MXFP8 分支 | 类型检查失败（tuple vs tensor） | 权重是 tuple 才走 MXFP8 路径 |

## 12. 自检题

**Q1.** EP=8、`T_local=8192`、`topk=8`、`minibatch_size=4096`、H=7168。`x_routed_ready` 一个 slot 的 `required_count` 是多少？为什么不是 7168？

**A1.** 448 = `ceil(4096/128) × ceil(7168/512)` = 32 × 14。7168 是整个 macrobatch（65536 行）的 chunk 总数；如果用它，slot 要等到该 macrobatch 的**所有** dispatch 完成才放行，而那要等下一个 macrobatch 的通信，形成环形等待。按 minibatch 计数才能让 dispatch 和 GEMM 真正重叠。

**Q2.** grid 里那 37900 个被取消的 cluster 会占 SM 吗？为什么还要 drain？

**A2.** 不会。`try_cancel` 取消的是**尚未启动**的 cluster，取消发生在 launch 队列里，不耗执行资源。drain 是为了把队列里剩余的待取消 cluster 全部消费掉（按注释 `CLC drain for no-op threadblocks` 的推断），让 kernel 能干净结束而不是留残余 launch。

**Q3.** 给你 `compute_cluster_idx`，怎么判断落在哪个专家？为什么不用一张预先算好的"row block → expert"表？

**A3.** 线性扫描 `tokens_per_expert` 的前缀和，求 (专家段 ∩ minibatch 行窗口) 与 `task_idx` 的关系，命中就停下。不用表的原因：表要在 device 端写（需要额外 kernel + 同步），而扫描 32 次整数比较每个 task 只做一次（cluster 粒度，不是 thread 粒度），代价可以忽略；更重要的是——**不建表就不会有建表的生命周期问题**，这与"零 CPU 同步"是同一个设计取向。

**Q4.** wgrad 用 `store_add_async` 累加，为什么结果还是 deterministic？

**A4.** 加法顺序被固定：同一专家的 wgrad 跨 macrobatch 累加，而 macrobatch 之间由 `routed_buffers_done` 串行化（`grouped_gemm.cuh:285-288` 注释明确说明了这一点）。并行的是不同输出 tile 的写者，同一 tile 的累加序列只有一个。

**Q5.** 如果把 `hidden_row_block_ready` 的 `required_count` 从 32 改成 16，会发生什么？

**A5.** down GEMM 会在 SwiGLU 只写完一半列块（或一半行）时就开跑，读到未写的 A 操作数 → 数值错但不崩；因为 32 = (256/128)×(I/128) = 2 行 tile × 16 列 tile，改成 16 相当于只等一半。这类错误最难查：形状对、随机性错、复跑可能"好了"（因为时序变了）。

## 13. 验证边界

本节的所有数字来自 **CPU 端复算**（`29_mok_task_table_sim.py`，可随时重跑），属于静态验证。真实 kernel 无法在 macOS 上跑，需要：

- NVIDIA SM100 / SM103（GB200/GB300 NVL72）
- CUDA 13.0+、PyTorch 2.10+（symmetric memory + `torch.distributed._symmetric_memory`）

真实运行验证（需要多卡 `torchrun` 且每卡独占一个 GPU）：

```bash
torchrun --standalone --nproc-per-node=8 -m pytest -s tests/test_functional.py::test_forward_bf16
torchrun --standalone --nproc-per-node=8 -m pytest -s tests/test_functional.py::test_forward_mxfp8
```

如果要验证本文的算术而不是数值，可以只跑 schedule 相关的测试（`tests/test_functional.py::test_build_schedule`），它检查的正是 `num_tokens`、`tokens_per_expert` 的 256 对齐与 deterministic 排序。
