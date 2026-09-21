# 09 Blackwell Tensor Core 与 tcgen05.mma

## 课程位置

课程：

- <https://github.com/mlc-ai/modern-gpu-programming-for-mlsys>

章节：

- `chapter_tensor_cores`
- 中文标题：《Blackwell Tensor Core: `tcgen05.mma`》

本篇已经覆盖：

```text
tcgen05.mma 的单 thread 发起语义
tcgen05.mma 指令字段
TMEM accumulator
tcgen05.commit 与 mbarrier
tcgen05.commit + mbarrier 和 TMA load / store 的差异
cta_group 的基本操作范围
cta_group::2 的 CTA pair 资源访问边界
cta_group::1, M=128 的直接 accumulator 映射
cta_group::1, M=64 的 Layout F
cta_group::2, M=128 dense A 的 Layout B
cta_group::2, M=256 的 accumulator 切分
```

## 目标

`chapter_tma` 结束在 A/B tile 到达 SMEM。本篇从这里继续：

```text
A/B 在 SMEM
  -> tcgen05.mma
C/D 在 TMEM
  -> tcgen05.ld
registers
```

重点回答：

```text
tcgen05.mma 为什么是 tile-level MMA
为什么一个 elected thread 就能发起
d-tmem、a-desc、b-desc、idesc 分别表示什么
enable-input-d 如何控制 K 循环累加
accumulator 为什么从 registers 移到 TMEM
C[m,n] 如何映射到 TLane / TCol
tcgen05.commit 为什么还要配合 mbarrier
为什么不直接使用 TMA store 的 commit_group / wait_group
cta_group::1 和 cta_group::2 的资源范围如何变化
```

## 一、tcgen05.mma 的执行模型

### 不是每个 thread 执行一次标量乘加

`tcgen05.mma` 是一条 tile-level Tensor Core 指令：

```text
一个 elected thread 发起完整 tile MMA
硬件执行 CTA 或 CTA pair 范围的矩阵运算
结果写入 TMEM
```

它和前几代指令的执行范围不同：

| 指令 | 执行范围 | A/B 位置 | C/D 位置 | 发起方式 |
|---|---|---|---|---|
| `mma.sync` | warp | registers | registers | warp 内 lanes 参与 |
| `wgmma.mma_async` | warpgroup | registers 或 SMEM | registers | warpgroup 发起 |
| `tcgen05.mma` | CTA 或 CTA pair | 通常为 SMEM | TMEM | 一个 elected thread 发起 |

核心变化是：

```text
mma.sync：
    fragment 分发到 lanes，再执行 warp-level MMA

tcgen05.mma：
    一个 thread 提交 tile-level MMA，
    硬件负责 CTA 或 CTA pair 范围的协作执行
```

### 常见指令形式

```text
tcgen05.mma.cta_group.kind
    [d-tmem], a-desc, b-desc, idesc,
    {disable-output-lane}, enable-input-d {, scale-input-d};
```

字段作用：

| 字段 | 作用 |
|---|---|
| `cta_group` | 使用当前 CTA 或 CTA pair 的资源 |
| `kind` | 选择 A/B 的数据类型家族 |
| `d-tmem` | C/D accumulator 的 TMEM 起始地址 |
| `a-desc` | A 在 SMEM 中的地址与布局 |
| `b-desc` | B 在 SMEM 中的地址与布局 |
| `idesc` | 描述 M、N、K、具体类型、major mode 等 |
| `disable-output-lane` | 选择哪些 TMEM lanes 不更新 |
| `enable-input-d` | 选择覆盖 D 还是执行 `A x B + D` |
| `scale-input-d` | 可选地缩放已有 D |

`kind::f16` 只表示浮点家族。具体是 `fp16` 还是 `bf16`，以及
accumulator 是 `f16` 还是 `f32`，由 `idesc` 选择。

### K 循环累加

第一次 K iteration：

```text
enable-input-d = false

D = A x B
```

后续 K iterations：

```text
enable-input-d = true

D = A x B + D
```

所有 K iterations 更新同一个 TMEM accumulator region：

```text
iteration 0:
    D = partial product 0

iteration 1:
    D = partial product 1 + D

iteration 2:
    D = partial product 2 + D

...

最后一次 iteration 后:
    TMEM 中才是完整 C tile
```

### 发起完成不等于结果完成

```text
tcgen05.mma 发起成功
!= TMEM 中的结果已经完成
```

典型完成路径：

```text
elected thread:
    issue tcgen05.mma
    tcgen05.commit [mbarrier]

consumer:
    wait mbarrier
    tcgen05.fence::after_thread_sync
    tcgen05.ld
    tcgen05.wait::ld
    use registers
```

## 二、TMEM Accumulator

### 从 register accumulator 到 TMEM

Ampere 和 Hopper 的 accumulator 长期位于 registers：

```text
mma.sync:
    C fragment 分散在各 lanes 的 registers 中

wgmma:
    accumulator 也在 registers 中
```

accumulator 越大，长期占用的 registers 越多。Blackwell 将长期存活的
accumulator 移到 TMEM：

```text
SMEM 中的 A/B
  -> tcgen05.mma
TMEM 中的 C/D
  -> tcgen05.ld
registers
```

### TMEM 的形状

课程给出的 `sm_100a` TMEM 模型：

```text
128 Lane rows
512 Col columns
每个 (TLane, TCol) 是一个 32-bit cell
```

总容量：

```text
128 * 512 * 4 bytes
= 256 KiB
```

TMEM 是 CTA-scoped memory space，不是普通 thread 使用通用
load/store 随意访问的地址空间。它通过 `tcgen05` 指令路径访问：

```text
tcgen05.mma:
    Tensor Core 更新 TMEM accumulator

tcgen05.ld:
    TMEM -> registers

tcgen05.st:
    registers -> TMEM

tcgen05.cp:
    SMEM 等路径 -> TMEM
```

### C[m,n] 到 TLane / TCol

最简单的直接映射是：

```text
cta_group::1
M = 128
```

此时：

```text
C[m, n]
  -> TLane = m
  -> TCol  = n
```

例如：

```text
C[0, 0]   -> TMEM(TLane=0,   TCol=0)
C[5, 3]   -> TMEM(TLane=5,   TCol=3)
C[127, 7] -> TMEM(TLane=127, TCol=7)
```

如果 `M=128`、`N=16`，accumulator 占：

```text
128 个 Lane rows
16 个 Col columns
```

实际映射会随以下条件变化：

```text
cta_group
M 的大小
dense A 还是 structured-sparse A
普通 tcgen05.mma 还是 tcgen05.mma.ws
```

### 为什么重要

寄存器 accumulator 的问题：

```text
register 压力随 accumulator 增大
主循环中要一直保留 C fragment
```

TMEM accumulator 的特点：

```text
主循环期间 C/D 不占 registers
Tensor Core 直接累加到 TMEM
registers 可以更多用于数据路径和 epilogue
```

代价是：

```text
需要分配 TMEM columns
需要管理 TLane / TCol 地址
需要保证 MMA 写入 layout 和 tcgen05.ld 读取 layout 一致
必须在正确时机回收 TMEM
```

布局问题也从：

```text
C fragment 位于哪个 lane 的哪个 register
```

变成了：

```text
C[m, n] 位于哪个 CTA 的哪个 TLane / TCol
```

## 三、tcgen05 为什么使用 commit 与 mbarrier

### commit 不是开始执行

```text
tcgen05.mma
    开始执行 MMA

tcgen05.commit
    把此前已经发出的 tcgen05 异步操作绑定到 mbarrier
```

当被绑定的 tcgen05 操作全部完成时，硬件 arrive 对应 mbarrier。

```text
tcgen05 异步执行完成
  -> tcgen05.commit 绑定的 mbarrier
  -> 其他 warp 可以等待这个 mbarrier
```

### 三种路径的完成机制

TMA load：

```text
TMA load 指令直接带 mbarrier
  -> TMA engine 按 transaction bytes 更新 mbarrier
  -> consumer 等待 mbarrier
```

TMA store：

```text
发起一个或多个 bulk stores
  -> cp.async.bulk.commit_group
  -> cp.async.bulk.wait_group
  -> source SMEM 可以复用
```

`tcgen05.mma`：

```text
一个 thread 发出 tcgen05.mma
  -> tcgen05.commit [mbarrier]
  -> MMA 完成后 arrive mbarrier
  -> consumer 等待 mbarrier
  -> tcgen05.fence::after_thread_sync
  -> tcgen05.ld
```

因此：

```text
tcgen05.commit + mbarrier
```

在结构上更接近：

```text
TMA load + mbarrier
```

而不是：

```text
TMA store + commit_group + wait_group
```

TMA load 可以直接在指令上指定 mbarrier；`tcgen05.mma` 则通过
`tcgen05.commit` 将完成事件转接到 mbarrier。

### 为什么不直接使用 wait_group

决定同步机制的首先是完成信号的消费者：

```text
TMA store:
    issuer 自己等待 source SMEM 可以复用
    -> issuer-local completion
    -> commit_group / wait_group 很合适

tcgen05.mma:
    MMA 由一个 elected thread 发起
    accumulator 由其他 warp 或 warpgroup 读取
    完成状态必须跨 threads 可见
    -> shared completion
    -> 需要 mbarrier
```

MMA 完成后要交给消费者的是“计算已经完成”，而不是“还差多少
bytes 没有到达”。所以这里没有 transaction bytes 语义，而是：

```text
commit 建立异步操作组
mbarrier 接收完成通知
consumer 等待 mbarrier
```

### 统一判断规则

先问两个问题：

```text
1. 谁在等待完成？
2. 完成后要把哪块数据交给谁？
```

| 场景 | 完成信号的消费者 | 典型机制 |
|---|---|---|
| TMA load | 使用 SMEM 的其他 warps | mbarrier + transaction bytes |
| TMA store | 发起 store 的 issue thread | commit_group / wait_group |
| `tcgen05.mma` | 读取 TMEM 的其他 warps | `tcgen05.commit` + mbarrier |
| `tcgen05.ld` | 使用 registers 的当前 warp | `tcgen05.wait::ld` |

一句话总结：

> `tcgen05.commit` 不替代 `mbarrier`，而是把 `tcgen05.mma` 的异步
> 完成事件接到 `mbarrier` 上。它对应的是 TMA load 的跨 warp 完成
> 通知路径，而不是 TMA store 的 issuer-local `commit_group /
> wait_group` 路径。

## 四、cta_group 的基本操作范围

`cta_group` 决定 MMA 使用哪些 CTA 资源，但不改变“单 thread 发起”的
语义。

### cta_group::1

```text
cta_group::1
```

只使用当前 CTA 的 SMEM 和 TMEM：

```text
CTA 0:
    SMEM -> tcgen05.mma -> TMEM
```

最简单的例子：

```text
M = 128
C[m, n] -> TLane=m, TCol=n
```

对于 `f16` / `bf16`，`cta_group::1` 支持：

```text
N = 8..256，以 8 为步长
```

### cta_group::2

```text
cta_group::2
```

使用一个 CTA pair 的资源。pair 中两个 CTA 位于同一个 cluster，
`%cluster_ctarank` 只有最低位不同，一个是偶数 CTA，一个是奇数 CTA。

```text
CTA pair:
    even CTA + odd CTA
```

MMA 可以访问 pair 中两个 CTA 的 TMEM。通常只由 pair 中的一个
thread 发起，例如由 even CTA 中的一个 elected thread 发起。
peer CTA 必须保持 active。

```text
even CTA + odd CTA
        |
        v
cooperative tcgen05.mma
        |
        +-> even CTA TMEM
        |
        +-> odd CTA TMEM
```

对于 `f16` / `bf16`，`cta_group::2` 支持：

```text
N = 16..256，以 16 为步长
```

`cta_group` 只决定操作范围。A/B 如何分布到两个 CTA 的 SMEM，
以及 accumulator 如何映射到两个 CTA 的 TMEM，还需要继续看 M 的
大小和具体 MMA 形式。

### cta_group::2 能访问哪些资源

这里不能把 `cta_group::2` 理解成“MMA 拿到一个能指向两个 CTA
任意 SMEM 地址的指针”。它表达的是 operation scope：

```text
MMA 是一个 CTA-pair scope 的操作
A/B operand 和 D accumulator 按 pair layout 使用两张 CTA 的资源
```

PTX 对 operand 的描述是：

```text
A: Tensor Memory OR Shared Memory
B: Shared Memory of the current CTA and optionally in peer CTA
D: Tensor Memory
```

重点区别如下：

| 资源 | `cta_group::2` MMA | `tcgen05.ld/st` | 普通 cluster DSMEM |
|---|---|---|---|
| A | TMEM 或 SMEM，按 pair operand layout 取数 | 不涉及 | 可访问 peer SMEM |
| B | SMEM，当前 CTA，且 PTX 明确允许 peer CTA | 不涉及 | 可访问 peer SMEM |
| D/C | 写入当前 CTA 和 peer CTA 的 TMEM | 只能访问当前 CTA 的 TMEM | 不涉及 |
| Register file | 不跨 CTA 共享 | 不跨 CTA 共享 | 不跨 CTA 共享 |
| Cluster barrier | 可同步 pair 或整个 cluster | 不涉及 | 需要同步 |

SMEM descriptor 描述的是 matrix 在 **current CTA** 的 shared memory
中的位置。它不是可以任意编码 peer CTA 地址的远程指针。`cta_group::2`
下的 peer 访问由固定的 CTA-pair data path 和 operand layout 决定。

普通 cluster DSMEM 是另一套能力。cluster 内 CTA 可以使用
`shared::cluster` 和 `mapa` 访问 peer SMEM：

```text
mapa.shared::cluster.u32 peer_addr, local_addr, peer_rank;
ld.shared::cluster.u32 r, [peer_addr];
```

这种访问可以覆盖整个 cluster，而不局限于 `cta_group::2` 所选的 pair。
它要求 peer CTA 仍然 active，并且需要正确的 cluster synchronization。

一句话记忆：

```text
cta_group::2 MMA:
    operand 可以由 CTA pair 的两张 SMEM/TMEM 共同参与
    但 descriptor 不是任意远程 SMEM 指针

tcgen05.ld/st:
    只能访问当前 CTA 的 TMEM

普通 cluster DSMEM:
    可以通过 shared::cluster/mapa 访问 peer SMEM
    但需要显式同步，和 MMA operand layout 无关
```

## 五、cta_group::1, M=128 的直接 accumulator 映射

这是 `tcgen05.mma` 最简单的一种 accumulator 映射，也是理解其他
layout 的基准。

### 资源属于同一个 CTA

```text
cta_group::1
```

表示 MMA 只使用当前 CTA 的资源：

```text
current CTA SMEM
  -> tcgen05.mma
current CTA TMEM
```

### 逻辑坐标直接映射到 TMEM 坐标

当：

```text
cta_group = cta_group::1
M = 128
```

TMEM 的 128 个 Lane rows 刚好容纳 M 的 128 行，因此映射是：

```text
C[m, n]
  -> TLane = m
  -> TCol  = n
```

完整范围：

```text
m = 0..127
n = 0..N-1
```

示例：

| 逻辑元素 | CTA | TLane | TCol |
|---|---:|---:|---:|
| `C[0, 0]` | 当前 CTA | 0 | 0 |
| `C[5, 3]` | 当前 CTA | 5 | 3 |
| `C[127, 7]` | 当前 CTA | 127 | 7 |

如果 `N=16`，accumulator 使用：

```text
128 个 Lane rows
16 个 Col columns
```

也就是 TMEM 中的一个 `128 x 16` 区域。N 增加时，逻辑矩阵沿
TMEM columns 方向增长。

### 为什么这个映射重要

后续 layout 都可以看作对这个直接映射的重排：

```text
cta_group::1, M=128:
    一个 CTA，一行 M 对应一行 TLane

cta_group::1, M=64:
    M 行拆成多个 group，使用 Lane stride

cta_group::2:
    M 和 N 还需要在 CTA pair 之间分配
```

`d-tmem` 给出 accumulator region 的起始地址；逻辑坐标 `(m, n)`
再由当前选择的 layout 映射到具体的 `(CTA, TLane, TCol)`。
epilogue 使用 `tcgen05.ld` 时，必须采用和 MMA 写回相兼容的
TMEM 地址与 load shape，否则读出的逻辑 C tile 会错位。

## 六、cta_group::1, M=64 的 Layout F

这一节对应普通 `tcgen05.mma`，不是 weight-stationary 的 `.ws` 形式。
`.ws` 使用 Layout E，两者的 accumulator placement 不同。

### 为什么不能直接 `TLane = m`

TMEM 有 128 个 Lane rows，但 `M=64` 不直接占用 `Lane 0..63`。
普通 MMA 的 TMEM data path 分为四个 32-lane region：

```text
Lane 0..31
Lane 32..63
Lane 64..95
Lane 96..127
```

`M=64` 被拆成四个 16-row group，每个 group 放入一个 32-lane
region 的一半。当前 tile 因此只使用 half datapath。

### Layout F 公式

设 lane alignment：

```text
a = 0 或 16
```

逻辑坐标到 TMEM 坐标的映射是：

```text
group        = m // 16
row_in_group = m % 16
TLane        = group * 32 + a + row_in_group
TCol         = n
```

`a=0` 时：

```text
m = 0..15   -> lanes   0..15
m = 16..31  -> lanes  32..47
m = 32..47  -> lanes  64..79
m = 48..63  -> lanes  96..111
```

`a=16` 时使用互补的 Lane：

```text
m = 0..15   -> lanes  16..31
m = 16..31  -> lanes  48..63
m = 32..47  -> lanes  80..95
m = 48..63  -> lanes 112..127
```

例如：

```text
m = 37, a = 0
group        = 37 // 16 = 2
row_in_group = 37 % 16 = 5
TLane        = 2 * 32 + 0 + 5 = 69

C[37, n] -> TMEM(TLane=69, TCol=n)
```

### lane alignment 的意义

`a=0` 和 `a=16` 的 Lane placement 互不重叠，因此两张独立的
`M=64` accumulator 可以共用同一组 TMEM columns：

```text
32-lane region:
    [ a=0 的 16 行 ][ a=16 的 16 行 ]
```

Layout F 只使用一半 datapath，所以 A、D 和 sparsity metadata 等
矩阵必须使用一致的同组 lane alignment。epilogue 通过
`tcgen05.ld` 读取 accumulator 时，也要采用与之相容的 Lane
alignment 和 load shape。

一句话记忆：

```text
M=128:
    TLane = m

M=64 Layout F:
    TLane = (m // 16) * 32 + (m % 16) + a
    a = 0 或 16
```

## 七、cta_group::2, M=256 的 accumulator 切分

`M=256` 超过了单个 CTA 的 128 个 Lane rows，因此 accumulator
必须在 CTA pair 的两块独立 TMEM 上存放。

### CTA pair 中 M 如何切分

课程采用的映射是沿 M 方向连续切分：

```text
even CTA: logical rows 0..127
odd CTA:  logical rows 128..255
```

每个 CTA 在自己的本地 TMEM 中使用：

```text
128 个 Lane rows
N 个 Col columns
```

物理上这是两个独立的 `128 x N` TMEM region：

```text
even CTA TMEM          odd CTA TMEM
128 x N accumulator    128 x N accumulator
```

逻辑上它们组成一个 `256 x N` accumulator tile。

### 坐标公式

对于逻辑元素 `C[m, n]`：

```text
if m < 128:
    CTA   = even
    TLane = m
else:
    CTA   = odd
    TLane = m - 128

TCol = n
```

具体例子：

| 逻辑元素 | CTA | 本地 TLane | TCol |
|---|---|---:|---:|
| `C[0, 7]` | even | 0 | 7 |
| `C[127, 7]` | even | 127 | 7 |
| `C[128, 7]` | odd | 0 | 7 |
| `C[255, 7]` | odd | 127 | 7 |

### 与 `tcgen05.ld` 的关系

两个 `128 x N` region 位于不同 CTA 的 TMEM 中，不是一个可以
由单个 CTA 任意读取的连续地址空间。`tcgen05.ld/st` 只能访问
当前 CTA 自己的 TMEM，因此 pair 的两侧需要各自完成自己那 128
行的 epilogue，或者通过额外路径交换结果。

A/B tile 如何分布到两个 CTA 的 SMEM，不属于 accumulator layout
本身；它由具体 kernel 的 operand 切分和 descriptor 决定。
`cta_group::2` MMA 通常由 pair 中的一个 elected thread 发起，
但写入结果会覆盖 pair 中两个 CTA 的 TMEM。

## 八、cta_group::2, M=128 dense A 的 Layout B

```text
本次讲解位置
章节：chapter_tensor_cores
小节：How cta_group Sets the Operation Scope
知识点：cta_group::2, M=128 dense A 的 Layout B
上次：cta_group::2, M=256 的 CTA pair accumulator 切分
下次：block-scaled MMA 的 SFA/SFB 跨 CTA pair 放置
PTX：9.7.18.10.5.2 Layout B (M = 128 + cta_group::2 + Dense A matrix)
```

上一节的 `M=256` accumulator 把 M 平均分给 CTA pair：

```text
even CTA: M rows 0..127
odd CTA:  M rows 128..255
```

每个 CTA 仍然使用完整的 N 方向，所以本地 TMEM 形状是
`128 x N`。

`cta_group::2, M=128` 不同：逻辑 tile 的总 M 只有 128，
但两个 CTA 仍然共同参与运算。因此每个 CTA 只拿 64 个 M
rows，同时把 N 方向拆成两半，再映射到自己的四个 32-lane
region。

### CTA pair 先沿 M 方向切分

对逻辑元素 `C[m, n]`：

```text
even CTA: m = 0..63
odd CTA:  m = 64..127
```

定义 CTA 内的局部行号：

```text
m_local = m % 64
```

因此：

| 逻辑 M 行 | CTA | `m_local` |
|---|---|---:|
| `0..63` | even | `0..63` |
| `64..127` | odd | `0..63` |

### 每个 CTA 内再把 N 折进 Lane 轴

设 N 的有效宽度为 `N`。在当前 CTA 内，N 被分成上下两个区间：

```text
lower half: n = 0 .. N/2-1
upper half: n = N/2 .. N-1
```

`m_local` 的 64 行再分成两个 32-row group：

```text
local group 0: m_local = 0..31
local group 1: m_local = 32..63
```

这两组 M 行和两个 N half 组成一个 `2 x 2` 映射，正好占满
当前 CTA 的四个 32-lane region：

| N 区间 | CTA 内局部 M 行 | TMEM Lane |
|---|---|---|
| `0 ... N/2-1` | `0..31` | `0..31` |
| `0 ... N/2-1` | `32..63` | `32..63` |
| `N/2 ... N-1` | `0..31` | `64..95` |
| `N/2 ... N-1` | `32..63` | `96..127` |

所以坐标公式是：

```text
CTA = even,  if m < 64
      odd,   if m >= 64

m_local = m % 64

TLane = m_local,       if n < N/2
        64 + m_local,  if n >= N/2

TCol = n
```

这里的 `m_local` 和 `TLane` 的关系是连续的：

```text
lower N half:
    m_local 0..63 -> TLane 0..63

upper N half:
    m_local 0..63 -> TLane 64..127
```

### 具体坐标例子

取 `N=16`，则：

```text
N/2 = 8

lower half: n = 0..7
upper half: n = 8..15
```

| 逻辑元素 | CTA | `m_local` | N half | TLane | TCol |
|---|---|---:|---|---:|---:|
| `C[10, 3]` | even | 10 | lower | 10 | 3 |
| `C[10, 11]` | even | 10 | upper | 74 | 11 |
| `C[70, 3]` | odd | 6 | lower | 6 | 3 |
| `C[70, 11]` | odd | 6 | upper | 70 | 11 |

例如 `C[70, 11]`：

```text
m >= 64                 -> odd CTA
m_local = 70 % 64 = 6
n = 11 >= N/2 = 8       -> upper N half
TLane = 64 + 6 = 70
TCol  = 11
```

### 和 M=256 的核心区别

```text
M=256:
    每个 CTA 保存 128 个 M rows
    每个 CTA 使用完整的 N
    本地 TMEM 形状为 128 x N

M=128, Layout B, dense A:
    每个 CTA 只保存 64 个 M rows
    N 的 lower / upper half 折进 Lane 的 lower / upper 64 lanes
    本地仍然占满 128 个 Lane rows，但每个 Lane row 对应的逻辑 M 不同
```

因此不能把上一节的 `TLane = m` 或 `TLane = m % 128` 直接套到
Layout B。必须先根据 M 选择 CTA，再根据 N 选择 Lane half。

### 适用范围和 epilogue 限制

这个映射只适用于：

```text
cta_group::2
M = 128
dense A
Layout B
```

PTX 中对应的 Layout B 条目还给出 `2 x 2` placement 和
lane alignment `0`。如果是 structured-sparse A，
`cta_group::2, M=128` 改用 Layout C，上面公式不能复用。

另外，Layout B 只描述 MMA 如何把 accumulator 写进 CTA pair。
CTA pair 中每个 CTA 的 TMEM 仍然是独立的；epilogue 做
`tcgen05.ld/st` 时，每个 CTA 必须按自己的 CTA 编号、`TLane`
和 `TCol` 读取，并使用与布局相容的合法 copy atom。不能把
pair 的两块 TMEM 当成一块连续的 `256 x N` 地址空间来寻址。

一句话记忆：

```text
M=256:
    M 在 pair 上各分 128 行，N 不折

M=128 dense A Layout B:
    M 在 pair 上各分 64 行，
    N 的 lower / upper half 折到 Lane 0..63 / 64..127
```

## 九、当前进度

### 本章剩余知识点

`chapter_tensor_cores` 主目录对应 6 个 accumulator /
data-path 知识点：

```text
[x] cta_group::1, M=128 的直接映射
[x] cta_group::1, M=64，非 .ws 的 Layout F
[x] cta_group::2, M=256，M rows 在 CTA pair 上连续切分
[x] cta_group::2, M=128 dense A 的 Layout B
[ ] block-scaled MMA 的 SFA/SFB 跨 CTA pair 放置
[ ] tcgen05 指令之间的 scope / layout / completion 三层契约
```

其中 sparse A 会把 `cta_group::2, M=128` 的 accumulator layout
从 Layout B 改为 Layout C，因此需要和 dense A 对比理解。

学完 M128 dense A 的 Layout B 后，本章还剩 2 个主知识点。

`chapter_tensor_cores` 正在进行中：

```text
tcgen05.mma 的单 thread 发起语义
tcgen05.mma 指令字段
d-tmem、a-desc、b-desc、idesc
enable-input-d 与 K 循环累加
TMEM accumulator
C[m,n] -> TLane / TCol
tcgen05.mma 的异步完成
tcgen05.commit + mbarrier
与 TMA load / store 同步机制的差异
cta_group::1 与 cta_group::2 的基本资源范围
cta_group::2 的 CTA pair 资源访问边界
cta_group::1, M=128 的直接 accumulator 映射
cta_group::1, M=64 的 Layout F
cta_group::2, M=256 的 CTA pair accumulator 切分
cta_group::2, M=128 dense A 的 Layout B
```

下一知识点：

```text
block-scaled MMA 的 SFA/SFB 跨 CTA pair 放置
```
