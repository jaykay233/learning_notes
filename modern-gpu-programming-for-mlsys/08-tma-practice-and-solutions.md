# 08 TMA 自测题与答案

配套章节：

- [08 TMA Tile Copy、Swizzle 与完成通知](08-tma-tile-copy-and-synchronization.md)

本篇覆盖 `chapter_tma` 的核心自测题：

```text
tensor map descriptor
单次 copy 参数
3D TMA
swizzle 地址计算
bank conflict
mbarrier load
commit / wait store
full / empty pipeline
ldmatrix 判定
TMA descriptor 与 MMA descriptor 契约
pipeline 性能分析
```

## 一、TMA 的任务划分

### 题目

解释下面四件事分别由谁负责：

```text
tensor map descriptor
单次 TMA copy 的参数
发起 TMA 的 elected thread
TMA engine
```

并说明为什么“TMA 指令返回”不代表 SMEM 数据已经可读。

### 答案

```text
tensor map descriptor：
    描述 GMEM 张量的全局 shape、strides、dtype、box、swizzle、OOB fill 等

单次 copy 参数：
    指定这次从哪个 tile 坐标开始搬、写到哪个 SMEM 地址、使用什么完成机制

elected thread：
    负责提交 TMA 请求，避免多个 lane 重复发起同一份 copy

TMA engine：
    根据 descriptor 和坐标生成地址，完成 GMEM -> SMEM 搬运、边界处理和 swizzle
```

TMA 指令返回只表示“请求已经提交”，不表示数据已经到达。数据完成后
通过 `mbarrier` 的 transaction bytes 更新通知 consumer，所以必须等待
对应 phase 完成才能读取 SMEM。

## 二、3D TMA descriptor

### 题目

有一个 row-major 的 `fp16` 矩阵：

```text
shape   = (16, 256)
strides = (256, 1)   # 元素为单位
```

现在要搬运前 128 列：

```text
rows 0..15
cols 0..127
```

每行是：

```text
128 * 2 = 256 bytes
```

不能直接作为 `SWIZZLE_128B` 的二维 box。

请写出：

```text
shape
strides
box
最内层字节数
swizzle atom 数量
```

### 答案

拆成两个 group：

```text
group 0: col 0..63
group 1: col 64..127
```

转换为：

```text
(group, row, inner_col)
```

得到：

```text
shape   = (2, 16, 64)
strides = (64, 256, 1)
box     = (2, 16, 64)
```

stride 推导：

```text
group 之间相差 64 个元素
    -> group stride = 64

row 之间相差原来一行的 256 个元素
    -> row stride = 256

inner_col 连续
    -> inner_col stride = 1
```

最内层字节数：

```text
64 * 2 = 128 bytes
```

Swizzle atom 数量：

```text
每个 group：
    16 rows * 128 bytes = 2048 bytes

每个 atom：
    8 rows * 128 bytes = 1024 bytes

每 group 有 2 个 atoms
总共有 2 * 2 = 4 个 atoms
```

如果按 CUDA API 填写，global strides 通常要换算成 bytes。

## 三、Swizzle 地址计算

### 题目

`SWIZZLE_128B` 的一个 8x8 sector atom 中：

```text
row = 5
logical_sector = 3
每行 = 128 bytes
每 sector = 16 bytes
```

使用：

```text
physical_sector =
    logical_sector XOR (row % 8)
```

求：

1. `physical_sector`
2. 相对于 atom 基址的字节偏移
3. 为什么 swizzle 不会改变逻辑矩阵的元素值

### 答案

```text
physical_sector =
    3 XOR 5
= 6
```

字节偏移：

```text
offset =
    row * 128
    + physical_sector * 16
=
    5 * 128
    + 6 * 16
=
    640 + 96
=
736 bytes
```

Swizzle 不改变逻辑矩阵的值，它只改变数据在 SMEM 中的物理位置。
只要 TMA 写入和 MMA 读取使用同一套 swizzle 规则，逻辑上仍然读回
同一个元素。

## 四、Bank conflict

### 题目

解释为什么下面两种布局在同样使用 `SWIZZLE_128B` 时，bank conflict
情况不同：

```text
256-byte row stride
128-byte grouped layout
```

请写出连续 8 行对应的 bank sector，或者说明它们分别覆盖了多少个
bank sector。

### 答案

Shared memory 有 32 个 banks，每周期每个 bank 访问 4 bytes：

```text
32 * 4 = 128 bytes
```

一个 16-byte sector 占 4 个 banks，所以有 8 个 bank sectors：

```text
S0 ... S7
```

### 256-byte row stride

```text
span      = sector_col // 8
local_col = sector_col % 8

bank_sector =
    local_col XOR ((2 * row + span) % 8)
```

固定 `span = 0`、`local_col = 0` 时：

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

只覆盖 `S0、S2、S4、S6`，每个被访问两次，因此是
2-way bank conflict。

### 128-byte grouped layout

group 内部相邻行的距离是：

```text
8 sectors * 16 bytes = 128 bytes
```

映射变成：

```text
bank_sector =
    local_col XOR (row % 8)
```

固定 `local_col = 0`：

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

八行覆盖全部八个 bank sectors，没有 bank conflict。

## 五、TMA load 的 mbarrier

### 题目

一次 TMA load 同时搬运：

```text
A = 3072 bytes
B = 2048 bytes
```

初始 barrier：

```text
expected arrival count = 1
```

请写出：

1. `arrive.expect_tx` 应该登记多少 bytes
2. `arrival count` 和 `pending bytes` 分别在什么时候变化
3. barrier 在什么条件下完成
4. TMA 完成后 consumer 为什么不能继续等待旧 phase

### 答案

总字节数：

```text
3072 + 2048 = 5120 bytes
```

所以：

```text
mbarrier.arrive.expect_tx(5120)
```

状态变化：

```text
初始：
    arrival = 1
    pending = 0

arrive.expect_tx(5120) 后：
    arrival = 0
    pending = 5120

A 的 3072 bytes 完成：
    arrival = 0
    pending = 2048

B 的 2048 bytes 完成：
    arrival = 0
    pending = 0
```

A、B 的实际完成顺序可能交换，但 barrier 只关心所有登记 bytes
是否归零。

完成条件：

```text
arrival count == 0
且
pending transaction bytes == 0
```

每完成一个 phase，phase 会翻转。consumer 必须等待本次使用的 phase。
如果继续等待旧 phase，旧 phase 已经完成，wait 可能立即返回，导致
读取尚未完成的新数据或旧数据。

## 六、Load 与 store 的同步机制

### 题目

回答：

1. 为什么 TMA load 通常使用 `mbarrier`？
2. 为什么 TMA store 通常使用 `commit_group` 和 `wait_group`？
3. 两种机制分别跟踪什么单位？
4. 如果 TMA store 完成后还要通知其他 warp，应该怎么做？

### 答案

TMA load 使用 `mbarrier`，因为：

```text
producer 发起 TMA
consumer 等待数据完整到达
完成信号需要让其他 threads 看见
还必须统计已经到达多少 bytes
```

TMA store 使用 `commit_group / wait_group`，因为：

```text
issuer 等待自己提交的 bulk async stores
目标是确认 source SMEM 何时可以覆盖
不需要统计目标 GMEM 已写多少 bytes
```

跟踪单位：

```text
mbarrier：
    transaction bytes

bulk async group：
    committed groups
```

如果 TMA store 完成后还要通知其他 warp：

```text
issuer 先执行 wait_group
issuer 再 arrive 一个普通 barrier 或 mbarrier
其他 warp 等待该 barrier
```

其他 warp 不能直接等待 issuer 的 bulk async group。

## 七、双 stage pipeline

### 题目

有 6 个 K tiles，SMEM 有两个 stages。请写出：

```text
k0 k1 k2 k3 k4 k5
```

分别落在哪个 stage。

然后回答：

1. 为什么 `k2` 不能直接覆盖 stage 0？
2. `full[stage]` 和 `empty[stage]` 分别保证什么？
3. stage 被第二次、第三次复用时，phase 为什么要翻转？
4. 要完全遮住 TMA，steady state 需要满足什么时间条件？

### 答案

stage 映射：

```text
k0 -> stage 0
k1 -> stage 1
k2 -> stage 0
k3 -> stage 1
k4 -> stage 0
k5 -> stage 1
```

也就是：

```text
stage = k % 2
```

`k2` 不能直接覆盖 stage 0，因为 consumer 可能还在读取 `k0`。
producer 必须先等待：

```text
empty[0]
```

含义：

```text
full[stage]：
    TMA 已完成，stage 中的数据可以被 consumer 读取

empty[stage]：
    consumer 已用完 stage，producer 可以覆盖
```

phase 翻转是因为同一个 `mbarrier` 被重复使用。每次 stage 被复用时，
对应的是下一个 phase。若不切换 phase，会把上一次的完成状态误认为
本次数据已经就绪。

完全遮住 TMA 的条件：

```text
TMA time(下一 tile)
<=
MMA time(当前 tile)
```

## 八、TMA 之后是否需要 ldmatrix

### 题目

分别判断下面三种情况是否还需要显式 `ldmatrix`，并说明原因：

```text
TMA + mma.sync
TMA + WGMMA SS
TMA + tcgen05.mma
```

另外说明 `tcgen05.ld` 与 `ldmatrix` 的区别。

### 答案

| 路径 | 是否需要显式 `ldmatrix` | 原因 |
|---|---|---|
| TMA + `mma.sync` | 需要 | A/B operands 必须位于 registers |
| TMA + WGMMA SS | 通常不需要 | Tensor Core 通过 matrix descriptor 直接读取 SMEM |
| TMA + `tcgen05.mma` | 通常不需要 | SMEM A/B 可以通过 matrix descriptor 消费 |
| TMEM accumulator | 使用 `tcgen05.ld` | 把 TMEM 结果取回 registers |

两者的区别：

```text
ldmatrix：
    SMEM -> registers
    同时完成 lane 之间的 fragment 分发

tcgen05.ld：
    TMEM -> registers
    不负责 SMEM 到 lane 的 ldmatrix 分发
```

核心判断标准是：下一条 Tensor Core 指令从哪里读取 A/B。如果要求
register fragments，就需要显式寄存器分发；如果支持直接读取 SMEM，
就通过 descriptor 描述 SMEM 布局。

## 九、Descriptor 契约

### 题目

解释下面这条链：

```text
GMEM
  -> TMA descriptor
SMEM
  -> MMA descriptor
Tensor Core
```

回答：

1. 两类 descriptor 各自描述什么？
2. 必须满足的核心布局等式是什么？
3. descriptor 不匹配时，为什么经常是静默的数值错误，而不是直接崩溃？
4. base address alignment 和 swizzle phase 有什么区别？

### 答案

TMA descriptor 描述：

```text
GMEM shape
GMEM strides
dtype
box shape
swizzle mode
OOB fill
```

MMA descriptor 描述：

```text
SMEM 起点
leading dimension stride
stride dimension
swizzle mode
base offset / swizzle phase
```

核心等式：

```text
TMA 写出的布局
==
SMEM 中的实际布局
==
MMA descriptor 声明的布局
```

不匹配时经常不会越界，因为地址可能仍在合法 SMEM 范围内。问题变成
“用错误的坐标解释正确范围内的数据”，于是会读到别的元素，表现为
局部 K/N tile 错位、数值缓慢偏离或大范围结果错误，而不是直接崩溃。

对齐和 phase 的区别：

```text
alignment：
    决定起始地址是否满足 byte alignment 要求

base offset / swizzle phase：
    决定从 atom 的哪个相对相位解释 swizzle
```

地址满足 128-byte 或 1024-byte 对齐，不代表 swizzle phase 一定正确。

## 十、Pipeline 设计题

### 题目

假设：

```text
单个 TMA tile 的稳态吞吐时间 = 0.8 us
单个 MMA tile 的计算时间 = 0.5 us
```

回答：

1. 双 stage 稳态下大致由谁决定吞吐？
2. 继续增加 stages 能否把吞吐降到 `0.5 us/tile`？
3. 增加 stages 主要解决 latency 还是 throughput？
4. 增加 stages 的代价是什么？

### 答案

双 stage 稳态下，吞吐由较慢的一侧决定：

```text
steady-state throughput
= max(TMA, MMA)
= max(0.8, 0.5)
= 0.8 us/tile
```

所以 Tensor Core 理想利用率是：

```text
0.5 / 0.8 = 62.5%
```

继续增加 stages 不能把吞吐直接降到 `0.5 us/tile`。如果 `0.8 us`
是 TMA 的持续吞吐瓶颈，增加深度只能隐藏启动延迟、抖动和其他等待，
不能凭空提高 TMA engine 或内存带宽。

增加 stages 主要解决 latency，而不是提高瓶颈 throughput。代价是：

```text
SMEM 占用增加
pipeline 状态增加
barrier 和 phase 管理更复杂
可能降低 occupancy
```

要真正达到 `0.5 us/tile`，需要提高 TMA 的有效吞吐，例如增加数据
复用、降低每 tile 的搬运量、调整 tile 形状，或者提高 TMA / 内存
路径的并行度。
