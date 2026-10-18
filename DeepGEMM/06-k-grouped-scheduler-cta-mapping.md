# DeepGEMM K-grouped scheduler：CTA 如何分配 M/N tile 与 K 段

## 本次讲解位置

```text
章节：DeepGEMM grouped GEMM 与 scheduler
小节：K-grouped 的 persistent CTA 调度
知识点：一个 scheduler task 对应一个 (group, M tile, N tile)，K 分块在 task 内归约
上次：K-grouped 的连续 K arena、分组边界、padding 与每组独立输出
下次：M-grouped scheduler 如何处理每组不同的 M tile 数，以及 MegaMoE 动态 expert 工作量
PTX：本节跟踪 scheduler 到 SM100 TMA / tcgen05 mainloop 的索引；重点是 CTA 工作单元，不推导指令编码
```

### 为什么现在讲这个

上一节已经知道每个 expert 的 K 段各自归约，但这还不能回答 GPU 上“到底派了多少 CTA、CTA 的工作单位是什么”。关键设计是：**K-grouped 的一个调度任务不是一个 K block，而是一个 group 的一个输出 M/N tile；该 CTA 在任务内部循环这个组自己的 K blocks。**看懂这个边界，才能避免把 K blocks 错当成调度任务、错误计算并行度，或漏掉输出的 group 维而让 expert 结果互相覆盖。

## 1. 心智模型：先挑输出 tile，再在组内扫完整个 K

记号：

- `BLOCK_M`、`BLOCK_N`：一个 CTA 的输出 tile 尺寸；
- `BLOCK_K`：mainloop 每轮处理的 K 深度；
- `num_m_blocks = ceil(M / BLOCK_M)`；
- `num_n_blocks = ceil(N / BLOCK_N)`；
- `num_blocks = num_m_blocks * num_n_blocks`：**一个 group 的输出 tile 数**。

因为 K-grouped 要求所有组共享 M、N，所以每个 group 有相同数量的输出 tile。整体逻辑任务依次是：

```text
group 0: (Mtile0,Ntile0), (Mtile0,Ntile1), ...
group 1: (Mtile0,Ntile0), (Mtile0,Ntile1), ...
...
```

每个任务内部再按该组的 `current_shape_k` 循环：

```text
选定 (group=e, M tile, N tile)
    -> K block 0 -> K block 1 -> ... -> K block (e 的末块)
    -> 写 D[e, 该 M tile, 该 N tile]
```

因此，调度任务总数是 `G * num_m_blocks * num_n_blocks`，而不是再乘上 K block 数。K 长度不同会让每个任务的内部循环次数不同，但不会改变该组的 M/N 输出 tile 数。

## 2. Persistent scheduler 如何把任务分给 CTA

K-grouped 的 `Scheduler::get_next_block()` 使用 persistent 方式。源码的核心索引关系是：

```text
next_block_idx = current_iter * kNumSMs + blockIdx.x
```

`current_iter` 每次取新任务就前进一次。于是初始任务按 SM/CTA 交错分配：

```text
CTA 0: task 0, task kNumSMs, task 2*kNumSMs, ...
CTA 1: task 1, task kNumSMs+1, ...
...
```

这让 grid 可以按 SM 数量启动并由 CTA 持续领取后续输出 tile，而不是为每一个 group/tile 固定启动一个单独的大 grid。

K-grouped 的分组边界通过阈值判断：第 `e` 组拥有全局任务区间

```text
[e * num_blocks, (e + 1) * num_blocks)
```

当 CTA 的 `next_block_idx` 越过当前区间，scheduler 增加 `current_group_idx`，推进本组 K 起点/长度，然后把组内任务编号交给 `get_swizzled_block_idx()` 得到 M/N tile 坐标。swizzle 会改变组内 tile 的遍历顺序，但不会改变任务属于哪个 group。

## 3. group 切换时更新哪些状态

K-grouped scheduler 维护几个容易混淆的量：

| 状态 | 含义 | 用在什么地方 |
|---|---|---|
| `current_group_idx` | 当前输出属于哪个 group/expert | 选择组的 K 范围；写 D 的第三维 group index |
| `current_k_start` | 当前组在 A/B 共享 K arena 中的物理起点 | TMA/MMA 的 A、B K 地址 |
| `current_shape_k` | 当前组 scheduler/mainloop 采用的 K 范围长度 | 计算本输出 tile 内要走多少 K block |
| `current_sf_k_cumsum` | scale-factor K 布局中的当前组起点 | 找 SFA/SFB 对应的 scale tile |
| `m_block_idx`, `n_block_idx` | 当前输出 tile 的 M、N tile 坐标 | 计算输入 tile 与输出 tile 的 M/N 地址 |

普通 contiguous layout 中，构造下一个 group 时，scheduler 先把 `current_k_start` 加上上一个 group 的 `current_shape_k`，再从 `grouped_layout[current_group_idx]` 读出新组物理 K 长度。

PSUM layout 的边界编码不同：`grouped_layout[e]` 是逻辑 K end；新组起点是前一逻辑 end 对齐后的地址，当前组长度是新 end 减去该起点。不要把两种 layout 的元数据都当成“每组长度”。

同时，SF arena 的起点单独按 `kSFKSpan` 更新，不能拿 token K offset 直接代替 scale-factor K offset。

## 4. 从逻辑 task 到 TMA/MMA/输出地址

对一个 `(group=e, m_tile, n_tile)` task，device kernel 大致按以下坐标计算：

```text
M 起点 = m_block_idx * BLOCK_M
N 起点 = n_block_idx * BLOCK_N
K 起点 = current_k_start
```

每个 K block `q` 的输入 K 起点是：

```text
A/B K 起点(q) = current_k_start + q * BLOCK_K
```

mainloop 的循环次数是：

```text
num_total_k_blocks = max(1, ceil(current_shape_k / BLOCK_K))
```

循环中 TMA 加载当前 M/N tile 与 K block，SFA/SFB 根据各自 scale K 起点搬运；UMMA 对当前输出 tile 累加。K loop 结束后，epilogue 写入：

```text
D[current_group_idx, M tile 范围, N tile 范围]
```

K-grouped 的输出描述符是 3D `[G,M,N]`。K-grouped epilogue 不把 `group_idx` 折进 M 坐标；group index 作为独立输出维传给 store。这样 `D₀`、`D₁` 等组不会落到同一块 `[M,N]` 输出上。

同步方面，TMA/scale-transpose/UMMA 的 barrier 负责保证当前 task 的数据已到 SMEM、UMMA 完成后 accumulator 才可由 epilogue 消费。scheduler 只决定“是谁的哪个 tile”；它本身不替代数据搬运和 producer-consumer 同步协议。

## 5. 具体 CTA 轨迹

设：

```text
M=256, N=512
BLOCK_M=128, BLOCK_N=256  -> num_m_blocks=2, num_n_blocks=2
num_blocks=2*2=4 个输出 tile / group
BLOCK_K=128
```

三个 group 的物理 K 起点/长度为：

| group | `current_k_start` | `current_shape_k` | K blocks |
|---|---:|---:|---:|
| 0 | 0 | 384 | 3 |
| 1 | 384 | 640 | 5 |
| 2 | 1024 | 256 | 2 |

总输出 tile task 数是 `3 * 4 = 12`。假设 `kNumSMs=4`，persistent CTA 按 `task_id = iter*4 + blockIdx.x` 取任务。为便于手算，以下组内 M/N 映射先用 row-major；真实代码会通过 `get_swizzled_block_idx()` 做 swizzle。

| CTA | task ids | 各任务归属（示意 M/N tile） | 对应 K block 次数 |
|---:|---|---|---|
| 0 | 0, 4, 8 | g0 `(m0,n0)`；g1 `(m0,n0)`；g2 `(m0,n0)` | 3、5、2 |
| 1 | 1, 5, 9 | g0 `(m0,n1)`；g1 `(m0,n1)`；g2 `(m0,n1)` | 3、5、2 |
| 2 | 2, 6, 10 | g0 `(m1,n0)`；g1 `(m1,n0)`；g2 `(m1,n0)` | 3、5、2 |
| 3 | 3, 7, 11 | g0 `(m1,n1)`；g1 `(m1,n1)`；g2 `(m1,n1)` | 3、5、2 |

CTA 0 的详细 K 访问起点：

```text
task 0 -> group 0：0, 128, 256
 task 4 -> group 1：384, 512, 640, 768, 896
 task 8 -> group 2：1024, 1152
```

这里任务 0、4、8 分别对三个 group 的同一个示意 M/N tile 做计算，但每个 task 都重新使用对应 group 的 K 起点，并写向不同的 `D[group]`。CTA 内不会把 g0 的累加器接着用于 g1。

## 6. 可运行的 Python scheduler 语义模型

下面完整脚本模拟 K-grouped 的 task 编号、persistent CTA 交错和 task 内部 K loop。为了易读，组内 tile 用 row-major；实际 DeepGEMM 会 swizzle tile 次序，真实 layout、TMA、SF 与同步均未由该 CPU 模型实现。

```python
def ceil_div(x, y):
    return (x + y - 1) // y


def trace_k_grouped_scheduler(m_blocks, n_blocks, num_sms,
                              group_k_ranges, block_k):
    num_groups = len(group_k_ranges)
    num_blocks = m_blocks * n_blocks
    total_tasks = num_groups * num_blocks
    cta_tasks = [[] for _ in range(num_sms)]

    for cta_id in range(num_sms):
        task_id = cta_id
        while task_id < total_tasks:
            group_id = task_id // num_blocks
            local_tile_id = task_id % num_blocks
            m_tile = local_tile_id // n_blocks
            n_tile = local_tile_id % n_blocks
            k_start, shape_k = group_k_ranges[group_id]
            num_k_blocks = max(1, ceil_div(shape_k, block_k))
            k_offsets = [k_start + q * block_k for q in range(num_k_blocks)]

            cta_tasks[cta_id].append(
                (task_id, group_id, m_tile, n_tile, k_offsets)
            )
            task_id += num_sms

    return cta_tasks


if __name__ == "__main__":
    ranges = [(0, 384), (384, 640), (1024, 256)]
    assignments = trace_k_grouped_scheduler(
        m_blocks=2,
        n_blocks=2,
        num_sms=4,
        group_k_ranges=ranges,
        block_k=128,
    )

    for cta_id, tasks in enumerate(assignments):
        print(f"CTA {cta_id}:")
        for task_id, group_id, m_tile, n_tile, k_offsets in tasks:
            print(
                f"  task={task_id}, group={group_id}, "
                f"MN=({m_tile},{n_tile}), K_offsets={k_offsets}"
            )
```

预期输出：

```text
CTA 0:
  task=0, group=0, MN=(0,0), K_offsets=[0, 128, 256]
  task=4, group=1, MN=(0,0), K_offsets=[384, 512, 640, 768, 896]
  task=8, group=2, MN=(0,0), K_offsets=[1024, 1152]
CTA 1:
  task=1, group=0, MN=(0,1), K_offsets=[0, 128, 256]
  task=5, group=1, MN=(0,1), K_offsets=[384, 512, 640, 768, 896]
  task=9, group=2, MN=(0,1), K_offsets=[1024, 1152]
CTA 2:
  task=2, group=0, MN=(1,0), K_offsets=[0, 128, 256]
  task=6, group=1, MN=(1,0), K_offsets=[384, 512, 640, 768, 896]
  task=10, group=2, MN=(1,0), K_offsets=[1024, 1152]
CTA 3:
  task=3, group=0, MN=(1,1), K_offsets=[0, 128, 256]
  task=7, group=1, MN=(1,1), K_offsets=[384, 512, 640, 768, 896]
  task=11, group=2, MN=(1,1), K_offsets=[1024, 1152]
```

运行命令：

```bash
python3 k_grouped_scheduler_trace.py
```

## 7. 常见误解与症状

| 误解/错误 | 实际语义 | 可见后果 |
|---|---|---|
| 把每个 K block 当成独立 scheduler task | task 是 `(group, M tile, N tile)`；K blocks 是 task 内循环 | CTA 任务数和并行度估算错误，可能重复或漏做输出 tile |
| task 编号只解出 M/N，不解 group | global task 区间按 group 分段 | 不同 expert 的数据会串读，或输出被覆盖 |
| 切换 group 时只更新 group index，不更新 K/SF 起点 | 新组 A/B 与 SF 都需要各自的 arena offset | 组边界处数据/scale 错配，误差集中在某些 expert |
| 认为 N 个 groups 就需要 N 次 kernel launch | scheduler 可在单个 persistent launch 内调度多组 tile | 对 launch 数、CTA 工作分配与负载均衡理解错误 |
| 把演示代码的 row-major tile 顺序当成真实顺序 | 真正 tile 次序经过 `get_swizzled_block_idx()` | 对 CTA 到 M/N 坐标映射预测错误，但 group 区间语义仍不变 |

## 8. 源码阅读路线与验证边界

1. `/Users/saboxu/Downloads/DeepGEMM/deep_gemm/include/deep_gemm/scheduler/gemm.cuh`：`get_next_k_group()` 维护组 K 范围；构造函数为 K-grouped 设置 `num_blocks=num_m_blocks*num_n_blocks`；`get_next_block()` 通过 `next_block_idx` 区分 group 并映射 M/N tile。
2. `/Users/saboxu/Downloads/DeepGEMM/deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_gemm_1d1d.cuh`：TMA warp 对每个 scheduler task 循环 K blocks；UMMA warp 对相同输出 tile 累加；epilogue 根据 `current_group_idx` 写 3D D 的 group 维。
3. `/Users/saboxu/Downloads/DeepGEMM/csrc/jit_kernels/impls/sm100_fp8_fp4_gemm_1d1d.hpp`：建立 `KGroupedContiguous` 描述符和 `[G,M,N]` TMA 输出描述符。

源码可用下面的命令定位：

```bash
rg -n "get_next_k_group|get_next_block|current_k_start|num_total_k_blocks|kCDWithGroupOffset" \
  /Users/saboxu/Downloads/DeepGEMM/deep_gemm/include/deep_gemm/scheduler/gemm.cuh \
  /Users/saboxu/Downloads/DeepGEMM/deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_gemm_1d1d.cuh
```

CPU trace 可以验证本节简化模型的 task 分配与 K 地址算术；要验证真实输出、TMA/barrier 与 tcgen05 路径，仍需目标 GPU、对应构建环境和 DeepGEMM 测试。

## 本章进度

- [x] 说明一个任务对应 group + M/N tile，K blocks 在任务内部循环
- [x] 用 3 个 group、2×2 输出 tiles、4 个 CTA 展示 persistent task 分配
- [x] 提供可运行 Python trace，并明确 swizzle 和真实硬件路径的验证边界

下一知识点：对照 M-grouped scheduler，理解不同 group 的 M tile 数变化后，global task 边界为什么不再是固定的 `group_idx * num_blocks`。

## 自测（含答案）

1. **一个 K-grouped scheduler task 代表什么？**  答：一个 group 的一个输出 M/N tile；该 CTA 在 task 内循环本组所需的 K blocks。
2. **调度任务总数是什么？**  答：`num_groups * ceil(M/BLOCK_M) * ceil(N/BLOCK_N)`，不乘 K block 数。
3. **为什么切换 group 时要同时更新 K 起点和 SF K 起点？**  答：A/B 与 SFA/SFB 各自有 K 轴分组布局，物理偏移单位和边界元数据不同。
4. **CTA 0 处理 group 0 后能否沿用它的 TMEM accumulator 给 group 1？**  答：不能；每个 task 的 group 输出独立，accumulator 生命周期/结果不能串组。
5. **简化脚本输出的 row-major tile 次序和真实 DeepGEMM 是否完全一致？**  答：不保证；真实 scheduler 经 `get_swizzled_block_idx()` 改变组内 tile 顺序，但 group/task 分段规律相同。
