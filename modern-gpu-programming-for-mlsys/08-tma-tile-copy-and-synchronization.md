# 08 TMA Tile Copy、Swizzle 与完成通知

## 课程位置

课程：

- <https://github.com/mlc-ai/modern-gpu-programming-for-mlsys>

章节：

- `chapter_tma`
- 中文标题：《异步数据搬运：TMA》

本篇覆盖：

```text
一个 thread 如何描述整个 tile
TMA 如何写入 swizzled layout
用 3D TMA 搬运多个 swizzle atoms
128-byte swizzle 与 row layout
如何等待 TMA load 完成
如何等待 TMA store 完成
为什么 load 使用 mbarrier，而 store 使用 commit / wait group
```

下一小节尚未展开：

```text
把 TMA 放进 pipeline
```

## 目标

TMA 解决的问题不是单纯把 copy 指令写短，而是：

```text
让一个 thread 提交整个多维 tile 的异步搬运
让硬件负责地址生成、边界处理和 swizzle
让数据搬运与 Tensor Core 计算重叠
```

本篇重点回答：

```text
tensor map descriptor 和单次 copy 参数分别包含什么
为什么一个 thread 就能发起整个 tile copy
TMA 如何把逻辑 tile 写成 MMA 所需的 swizzled layout
为什么一行 128 个 fp16 会变成 256 bytes
为什么 3D TMA 要增加 group 维
TMA box 的宽度限制是什么
同样的 SWIZZLE_128B 为什么仍可能产生 bank conflict
mbarrier 的 arrival count 和 pending bytes 如何交接 load
commit group / wait group 如何保护 TMA store 的 source buffer
为什么 load 和 store 使用不同的完成机制
```

## 一、一个 thread 如何描述整个 tile

### 普通 SIMT copy

传统 copy 通常由多个 threads 共同完成：

```text
thread 0: GMEM[base + 0] -> SMEM[base + 0]
thread 1: GMEM[base + 1] -> SMEM[base + 1]
thread 2: GMEM[base + 2] -> SMEM[base + 2]
...
```

每个 thread 都要：

```text
计算自己的 global memory 地址
计算自己的 shared memory 地址
执行 load / store
```

TMA 把描述粒度提升到 tile：

```text
一个 elected thread:
    提交 tensor descriptor
    + tile 起始坐标
    + shared memory 目标地址
    -> TMA engine

TMA engine:
    遍历 tile
    计算全部地址
    完成 GMEM -> SMEM 搬运
```

这里的核心不是“一个 thread 搬完所有字节”，而是：

```text
一个 thread 只提交任务
TMA engine 负责实际搬运
```

### Tensor map descriptor

Tensor map descriptor 可以理解为 global tensor 的说明书，通常包含：

```text
dtype
rank
global tensor shape
global tensor strides
tile / box shape
swizzle mode
interleave
OOB fill
L2 promotion
```

其中大部分内容在一个 kernel 内可以重复使用。

### 单次 copy 参数

每次 TMA 指令额外给出：

```text
tile 在 global tensor 中的起始坐标
SMEM 目标地址
completion 机制
```

可以概括为：

```text
descriptor：整个 tensor 如何组织
instruction arguments：这一次从哪里开始搬、搬到哪里
```

### 具体例子

假设：

```text
A shape = (128, 256)
dtype   = bf16
box     = (64, 64)
start   = (64, 64)
```

这次 copy 覆盖：

```text
row: 64 .. 127
col: 64 .. 127
```

数据量：

```text
64 * 64 * 2 = 8192 bytes
```

descriptor 中可以复用：

```text
shape = (128, 256)
box   = (64, 64)
dtype = bf16
```

每次变化的是：

```text
coord    = (64, 64)
smem_dst = A_smem + stage * 8192
```

### SIMT 中的 elected thread

发起 TMA 时，warp 仍处于 SIMT 模型：

```cpp
if (elect_one_sync()) {
    tma_load(
        desc = A_desc,
        coord = {m0, k0},
        smem = A_smem + stage * A_stage_bytes,
        barrier = full[stage]
    );
}
```

只有被选中的 lane 提交 TMA 请求，其他 lanes 在这条指令期间被屏蔽。

这个屏蔽只持续到请求提交完成。请求提交后：

```text
发起 TMA 的 warp 可以继续执行
其他 warps 也可以继续执行
TMA engine 在后台异步搬运
```

不能由所有 lanes 重复发起同一份 copy，否则可能产生重复搬运。

### 发起不等于完成

下面这个判断是错误的：

```text
TMA 指令返回
=> SMEM 数据已经就绪
```

正确流程是：

```text
producer 提交 TMA
-> producer 继续执行其他工作
TMA engine 后台搬运
-> 搬运完成后更新 completion object
consumer 等待完成
-> 才能安全读取 SMEM
```

## 二、TMA 如何写入 swizzled layout

### 先看一个单 atom tile

考虑：

```text
tile shape = (8, 64)
dtype      = fp16
```

一行的大小：

```text
64 * 2 bytes = 128 bytes
```

一个 16-byte sector 中包含：

```text
16 / 2 = 8 个 fp16
```

所以 tile 可以看成：

```text
8 rows * 8 logical sectors
```

逻辑元素列到 sector 的换算：

```text
logical_sector = col // 8
```

### Linear layout

没有 swizzle 时：

```text
physical_sector = logical_sector
```

地址为：

```text
smem_addr =
    base
    + row * 128
    + logical_sector * 16
```

### SWIZZLE_128B

启用 `SWIZZLE_128B` 后：

```text
physical_sector =
    logical_sector XOR (row % 8)
```

地址为：

```text
smem_addr =
    base
    + row * 128
    + (logical_sector XOR (row % 8)) * 16
```

完整映射：

```text
logical sector:   0 1 2 3 4 5 6 7
row 0:            0 1 2 3 4 5 6 7
row 1:            1 0 3 2 5 4 7 6
row 2:            2 3 0 1 6 7 4 5
row 3:            3 2 1 0 7 6 5 4
row 4:            4 5 6 7 0 1 2 3
row 5:            5 4 7 6 1 0 3 2
row 6:            6 7 4 5 2 3 0 1
row 7:            7 6 5 4 3 2 1 0
```

例如：

```text
row = 1
logical_sector = 0

physical_sector = 0 XOR 1 = 1
```

逻辑上第 1 行第 0 个 sector 的数据，物理上写到第 1 个 sector。

### 为什么不改变逻辑内容

Swizzle 只改变物理排列：

```text
逻辑矩阵不变
元素值不变
元素所在的物理位置改变
```

如果 TMA 使用 swizzle 写入，而 MMA 按 linear layout 读取，Tensor Core 会把物理 sector 中的数据解释成错误的逻辑元素。

下面三者必须一致：

```text
TMA descriptor 的 swizzle mode
shared-memory tile 的物理布局
MMA 指令读取 shared memory 时使用的 layout
```

### dtype 只改变 sector 中的元素数

XOR 映射作用于 16-byte sector：

```text
fp16: 8 个元素 / sector
bf16: 8 个元素 / sector
fp32: 4 个元素 / sector
```

dtype 不改变 sector 编号之间的 XOR 关系。

## 三、用 3D TMA 搬运多个 swizzle atoms

### 2D tile 超过 128 bytes

考虑一个 `fp16` slice：

```text
shape = (16, 128)
```

这里的 `128` 是元素数，不是字节数。

每一行占用：

```text
128 elements * 2 bytes = 256 bytes
```

而 `SWIZZLE_128B` 的最内层连续宽度最多是 128 bytes，所以一行不能直接作为一个 128-byte swizzle 单元。

### 增加 group 维

把列拆成两个 groups：

```text
group 0: columns 0 .. 63    = 128 bytes
group 1: columns 64 .. 127  = 128 bytes
```

定义：

```text
group     = col // 64
inner_col = col % 64
```

二维逻辑视图：

```text
original[row, col]
```

变成三维逻辑视图：

```text
view3d[group, row, inner_col]
```

形状为：

```text
(group=2, row=16, inner_col=64)
```

这只是 reshape 解释方式，不会预先移动 global memory 中的数据。

假设原始整体矩阵是 row-major `(16, 256)`：

```text
original_offset =
    row * 256 + col
=
    row * 256
    + group * 64
    + inner_col
```

所以三维视图对应的元素 stride 是：

```text
group     stride = 64 elements = 128 bytes
row       stride = 256 elements = 512 bytes
inner_col stride = 1 element = 2 bytes
```

### Tensor map

按课程中的逻辑维度顺序：

```text
dtype   = fp16
rank    = 3

shape   = (2, 16, 64)
strides = (64, 256, 1)  # 单位是元素

box     = (2, 16, 64)
swizzle = SWIZZLE_128B
```

最内层：

```text
64 个 fp16 = 128 bytes
```

现在满足 swizzle 宽度限制。

不同 API 对维度顺序的约定可能相反，这里按课程的逻辑顺序描述。

### Atom 数量

每个 group：

```text
16 rows * 128 bytes
```

一个 swizzle atom：

```text
8 rows * 128 bytes
```

所以每个 group 包含两个 atoms，整个 tile 包含四个 atoms：

```text
group 0:
    rows 0 .. 7   -> atom 0
    rows 8 .. 15  -> atom 1

group 1:
    rows 0 .. 7   -> atom 2
    rows 8 .. 15  -> atom 3
```

总数据量：

```text
4 atoms * 8 rows * 128 bytes = 4096 bytes
```

原始数据量：

```text
16 rows * 128 elements * 2 bytes = 4096 bytes
```

### Atom 内的 swizzle

每一个 atom 内仍使用：

```text
physical_sector =
    logical_sector XOR (local_row % 8)
```

这里的 `local_row` 是 atom 内部的行号，不是整个 tile 的行号。

例如：

```text
tile row 0 和 tile row 8
```

在各自的 atom 内都有：

```text
0 % 8 = 0
8 % 8 = 0
```

它们使用相同的 XOR 模式，但位于不同的 atom 基址。

## 四、TMA box 宽度限制

TMA descriptor 中的 `box` 表示一次 copy 的 tile shape：

```text
box = (box_dim_0, box_dim_1, ..., box_dim_n)
```

每个维度通常以元素为单位。

最内层连续维度换算成字节：

```text
inner_box_bytes =
    innermost_box_dim * sizeof(dtype)
```

对于 swizzled TMA：

```text
SWIZZLE_128B -> inner bytes <= 128
SWIZZLE_64B  -> inner bytes <= 64
SWIZZLE_32B  -> inner bytes <= 32
```

对应 fp16：

```text
SWIZZLE_128B -> 最多 64 个 fp16
SWIZZLE_64B  -> 最多 32 个 fp16
SWIZZLE_32B  -> 最多 16 个 fp16
```

### 有效与无效 box

f16 二维 box：

```text
box = (16, 128)
inner bytes = 128 * 2 = 256 bytes
```

对 `SWIZZLE_128B` 无效。

增加 group 维后：

```text
box = (2, 16, 64)
inner bytes = 64 * 2 = 128 bytes
```

有效。

这里的“宽度”特指最内层连续维度的字节数。外层维度不受这个 128-byte 限制。

## 五、128-byte swizzle 与 row layout

### Bank sector

Shared memory 有 32 个 banks，每个 bank 每周期访问 4 bytes：

```text
32 banks * 4 bytes = 128 bytes
```

一个 16-byte sector 占 4 个 banks。把 32 个 banks 分成 8 个 bank sectors：

```text
S0 -> banks 0 .. 3
S1 -> banks 4 .. 7
...
S7 -> banks 28 .. 31
```

如果一次并行访问覆盖 `S0 .. S7`，每个 bank 只用一次。

### 16x16 sector grid

考虑：

```text
16 rows * 16 sectors
每 sector = 16 bytes
每 row = 256 bytes
```

令：

```text
span      = sector_col // 8
local_col = sector_col % 8
```

### 256-byte row stride

如果逻辑行保持 256-byte stride：

```text
bank_sector =
    local_col XOR ((2 * row + span) % 8)
```

固定 `span`：

```text
row:       0  1  2  3  4  5  6  7
2 * row:   0  2  4  6  8 10 12 14
mod 8:     0  2  4  6  0  2  4  6
```

连续 8 行只得到四个不同的 key。若 `local_col = 0`：

```text
row 0 -> S0
row 1 -> S2
row 2 -> S4
row 3 -> S6
row 4 -> S0
row 5 -> S2
row 6 -> S4
row 7 -> S6
```

结果是：

```text
S0、S2、S4、S6 各被访问两次
```

这就是 2-way bank conflict。

这个 256-byte row layout 主要用于解释 row stride 对 swizzle 地址的影响，不表示它是推荐的单个 `SWIZZLE_128B` 二维 TMA box。

### 128-byte grouped layout

把逻辑坐标写成：

```text
(group, row, local_col)
```

group 内部相邻行的距离为：

```text
8 sectors = 128 bytes
```

此时：

```text
bank_sector =
    local_col XOR (row % 8)
```

连续八行：

```text
row 0 -> S0
row 1 -> S1
row 2 -> S2
row 3 -> S3
row 4 -> S4
row 5 -> S5
row 6 -> S6
row 7 -> S7
```

八个 sectors 刚好覆盖：

```text
8 * 4 = 32 banks
```

没有 bank conflict。

### 结论

同样使用 `SWIZZLE_128B`：

```text
256-byte row stride:
    8 rows 只覆盖 4 个 bank sectors
    2-way conflict

128-byte grouped layout:
    8 rows 覆盖 8 个 bank sectors
    无 conflict
```

关键不只是有没有 swizzle，还包括 row 在内存中的 stride。

## 六、如何等待 TMA load 完成

### mbarrier 跟踪两类状态

TMA load 常使用 shared memory 中的 `mbarrier`：

```text
arrival count
pending transaction bytes
```

一个 phase 只有在两者都归零时才算完成：

```text
arrival count == 0
and
pending bytes == 0
```

含义分别是：

```text
arrival count：多少个 producer 已到达或已登记
pending bytes：TMA engine 还有多少字节未搬完
```

### 具体例子

同时加载 A、B：

```text
A = 2048 bytes
B = 2048 bytes
total = 4096 bytes
```

初始化 barrier：

```text
expected arrival count = 1
```

producer 执行：

```text
mbarrier.arrive.expect_tx(4096)
```

状态：

| 时刻 | arrival count | pending bytes | phase 完成 |
|---|---:|---:|---|
| 初始化 | 1 | 0 | 否 |
| `arrive.expect_tx(4096)` 后 | 0 | 4096 | 否 |
| A 的 2048 bytes 完成 | 0 | 2048 | 否 |
| B 的 2048 bytes 完成 | 0 | 0 | 是 |

barrier 不需要区分 A 和 B，只关心总共登记的 bytes 是否全部到齐。

### Producer

```cpp
mbarrier_init(barrier, 1);

if (elect_one_sync()) {
    mbarrier_arrive_expect_tx(barrier, 4096);

    tma_load(A, ..., barrier);
    tma_load(B, ..., barrier);
}
```

producer 发起后可以继续做其他工作。

### Consumer

```cpp
while (!mbarrier_try_wait(barrier, phase)) {
    // wait
}

// A、B 均已完整到达
use(A_smem, B_smem);
```

提前读取可能得到旧数据、半份新数据或尚未完成写入的数据。

### Phase

`mbarrier` 可以重复使用。一个 phase 完成后，相位翻转：

```text
phase 0 完成
-> 下一次等待 phase 1

phase 1 完成
-> 下一次等待 phase 0
```

常见的 pipeline 会为每个 stage 建立：

```text
full[stage]
empty[stage]
```

其中：

```text
full[stage]：TMA 数据已经到达
empty[stage]：计算已经用完该 stage
```

## 七、如何等待 TMA store 完成

### 方向与问题都反转

TMA store：

```text
shared memory -> global memory
```

主要问题从“目标什么时候可读”变成：

```text
source buffer 什么时候可以覆盖
```

假设 epilogue 已经写好了输出 tile：

```text
Dsmem
```

随后发起 TMA store：

```text
Dsmem -> global D
```

store 是异步的。如果马上覆盖 `Dsmem`，TMA engine 可能读到新旧混合的数据。

### Bulk async group

典型流程：

```text
发起一个或多个 TMA stores

cp.async.bulk.commit_group

cp.async.bulk.wait_group 0

复用 Dsmem
```

### commit_group

连续发起多个 stores：

```text
TMA store A
TMA store B
```

执行：

```text
cp.async.bulk.commit_group
```

会把此前尚未提交的 bulk async operations 组成一个 group：

```text
Group 0:
    TMA store A
    TMA store B
```

### wait_group

```text
cp.async.bulk.wait_group 0
```

表示等待此前提交的 groups 全部完成。

```text
Group 0 未完成 -> 继续等待
Group 0 完成   -> source SMEM 可复用
```

`wait_group N` 表示最多允许最近 N 个 groups 仍处于 pending：

```text
wait_group 0：全部完成
wait_group 1：允许最近一个 group 仍在途
```

### 具体流程

输出 tile：

```text
64 * 64 * 2 = 8192 bytes
```

时间线：

```text
t0: epilogue 写完 Dsmem
t1: 发起 TMA store
t2: commit_group
t3: TMA engine 异步写 GMEM
t4: 其他工作继续执行
t5: wait_group 0 返回
t6: Dsmem 可复用
```

## 八、为什么 load 用 mbarrier，store 用 commit / wait

### 完成信号给谁看

TMA load 通常是跨线程交接：

```text
producer warp:
    发起 TMA load

consumer warp / warpgroup:
    等待 SMEM 数据
    执行 MMA
```

`mbarrier` 位于 shared memory 中，可以被 CTA 内其他 threads 等待，因此很适合：

```text
producer -> 更新 mbarrier
consumer -> 等待 mbarrier
```

bulk async group 是某个 thread 自己发起的异步操作队列：

```text
commit 了哪些 bulk stores
还有多少组未完成
```

其他 warp 不能直接等待另一个 thread 的 bulk group。若需要通知其他 warp，可以在 issuer 的 `wait_group` 之后再叠加普通 barrier 或 mbarrier。

### 保护目标可读，还是保护 source 可复用

TMA load 关心：

```text
目标 SMEM 是否完整可读
```

因此需要统计：

```text
arrival count
pending transaction bytes
```

TMA store 关心：

```text
source SMEM 是否已经读完
```

issuer 不需要知道目标 GMEM 写了多少字节。它只需要知道：

```text
自己提交的 store groups 是否全部完成
```

因此使用：

```text
commit group
wait group
```

### 跟踪 bytes，还是跟踪 groups

```text
mbarrier：
    pending transaction bytes
    还差多少 bytes 才完整

bulk group：
    还有多少组异步操作未完成
```

### 对比表

| 项目 | TMA load | TMA store |
|---|---|---|
| 数据方向 | GMEM -> SMEM | SMEM -> GMEM |
| TMA engine 角色 | producer | SMEM 的 consumer |
| 谁等待 | 使用 SMEM 的 consumer | 发起 store 的 issuer |
| 等待什么 | 数据是否完整到达 | source 是否已经读完 |
| 跟踪单位 | transaction bytes | committed groups |
| 典型机制 | `mbarrier` | `commit_group` / `wait_group` |
| 失效后果 | 读到不完整数据 | 覆盖尚未读走的 source |

### 决定规则

```text
完成信号要给其他 threads，
并且需要确认已经到达多少 bytes：
    使用 mbarrier

只是发起者等待自己的异步 store 完成，
从而判断 source buffer 何时可复用：
    使用 bulk async group
```

两种机制不是任意可替换的标准路径。store 完成后如果还要通知其他 threads，可以在 `wait_group` 之后再叠加共享 barrier。

## 九、放到推理系统中

### Prefill GEMM

典型 overlap：

```text
当前 K tile 正在做 Tensor Core 计算
同时 elected thread 为下一 stage 发起 A/B tile TMA load
```

TMA load 完成后通过 `full[stage]` 通知 MMA consumer。

### Attention

Q、K、V tile 可以使用 tensor tile 描述。若 `head_dim = 128`、dtype 为 `fp16`，一行大小是：

```text
128 * 2 = 256 bytes
```

可以拆成两个 128-byte groups，再分别形成 swizzle atoms。

### GEMM Epilogue

输出 tile 先写入 shared memory：

```text
Dsmem -> global D
```

使用 TMA store 后，需要：

```text
commit_group
wait_group 0
```

确认 source 已读完，才能覆盖 `Dsmem`。

## 十、当前进度

`chapter_tma` 已完成：

```text
一个 thread 如何描述整个 tile
TMA tensor map descriptor 与单次 copy 参数
TMA 如何写入 swizzled layout
128-byte swizzle 的 XOR 地址映射
一行元素数到字节数的换算
3D TMA 搬运多个 swizzle atoms
TMA box 最内层宽度限制
128-byte grouped layout 与 256-byte row stride
bank sector 与 2-way conflict
TMA load 的 mbarrier completion
arrival count 与 pending transaction bytes
phase 翻转与 stage reuse
TMA store 的 commit group / wait group
source buffer 的复用条件
load 与 store 同步机制不同的原因
```

下一小节：

```text
把 TMA 放进 pipeline
```
