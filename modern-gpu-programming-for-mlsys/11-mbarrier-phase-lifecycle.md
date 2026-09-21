# Async Coordination: mbarrier 与异步交接

本篇已经覆盖：

- `mbarrier` 的 arrival、pending count、phase 与 phase parity
- 一个 phase 完成的条件
- phase 完成后为什么自动进入下一轮
- 软件为什么只记录 parity 的 `0 / 1`
- 为什么复用 barrier 时必须切换等待的 parity
- threads 写 SMEM 后交给 async proxy 读取时的 fence 与 thread sync
- `full[stage]` / `empty[stage]` 如何通过所有权交接复用同一个 stage

## 一、本次讲解位置

```text
本次讲解位置
章节：chapter_async_barriers
小节：mbarrier / How phases distinguish consecutive uses
知识点：mbarrier 的 phase 生命周期与 parity
上次：tcgen05.wait::ld/st 的异步完成边界
下次：Common synchronization handoffs -> threads 写 SMEM 后交给异步硬件读取
PTX：9.7.15.16.12 mbarrier.init
     9.7.15.16.16 mbarrier.arrive
     9.7.15.16.19 mbarrier.test_wait / mbarrier.try_wait
```

上一课解决的是同一个线程内部的完成顺序：

```text
tcgen05.ld/st 发出
tcgen05.wait::ld/st 等待当前线程此前同类操作完成
```

这一课解决另一个问题：

```text
producer 和 consumer 不是同一个线程
它们如何判断某一次数据交接已经完成
```

承担这个交接的对象就是 `mbarrier`。

## 二、mbarrier 最少需要跟踪哪些状态

理解 phase 生命周期，只需要先抓住四个状态：

| 状态 | 含义 |
|---|---|
| expected arrivals | 当前 phase 一共需要多少次 arrive |
| pending arrivals | 当前 phase 还缺多少次 arrive |
| phase parity | 当前处理的是第几轮，只保留 `phase % 2` |
| tx-count | TMA load 还需要完成多少 transaction bytes |

初始化时：

```text
pending arrivals = expected arrivals
phase parity = 0
```

例如：

```text
mbarrier.init(bar, 1)
```

表示该 barrier 的每个 phase 需要一次 arrival，初始状态是：

```text
pending arrivals = 1
phase parity = 0
```

`mbarrier` 位于 shared memory，不是一个只能使用一次的一次性事件。
它可以反复被 reuse，每完成一轮就自动进入下一 phase。

## 三、phase 什么时候完成

对于普通 arrive，phase 完成条件是：

```text
pending arrivals == 0
```

对于 TMA load，还要同时满足：

```text
pending arrivals == 0
tx-count == 0
```

也就是说：

```text
arrive
    表示参与生产者已经提交了 arrival

tx-count 归零
    表示 TMA engine 已经完成约定的数据传输
```

当前 phase 完成后：

```text
pending arrivals 重新回到 expected arrivals
phase parity ^= 1
```

具体变化是：

```text
完成 phase 0 -> 进入 phase 1
完成 phase 1 -> 进入 phase 0
完成 phase 0 -> 进入 phase 1
...
```

相位可以一直循环，硬件具体执行到第几千轮并不重要。软件通常只记录
当前要等待的 parity：

```text
phase = 0 或 1
```

## 四、具体例子：一次 TMA load 的 phase 0

假设：

```text
expected arrivals = 1
TMA transfer bytes = 4096
```

初始状态：

```text
pending arrivals = 1
tx-count         = 0
phase parity     = 0
```

producer 执行：

```text
mbarrier.arrive.expect_tx(bar, 4096)
```

状态变成：

```text
pending arrivals = 0
tx-count         = 4096
phase parity     = 0
```

此时 arrival 已经到齐，但数据还没有到齐，所以 phase 0 尚未完成。

当 TMA engine 完成 2048 bytes：

```text
pending arrivals = 0
tx-count         = 2048
phase parity     = 0
```

当 TMA engine 完成全部 4096 bytes：

```text
pending arrivals = 0
tx-count         = 0
phase parity     = 1
```

这一刻 phase 0 完成，barrier 自动进入 phase 1。

consumer 原本执行：

```text
mbarrier.try_wait(bar, 0)
```

等待的是 phase 0。phase 0 完成后，这次 wait 才能返回。

consumer 成功消费这块数据后，下一轮应当等待 phase 1：

```text
phase = 0

第 1 次使用：
    try_wait(bar, phase)
    phase ^= 1

第 2 次使用：
    try_wait(bar, phase)
    phase ^= 1
```

展开就是：

```text
第 1 次等待 phase 0
第 2 次等待 phase 1
第 3 次等待 phase 0
第 4 次等待 phase 1
```

## 五、为什么旧 parity 会造成误判

假设同一个 barrier 已完成过 phase 0，硬件 parity 当前是 1。

此时：

```text
try_wait(bar, 0)
```

可能在下一轮 TMA 尚未完成时立即成功，因为 phase 0 确实已经完成过。
consumer 就可能读到尚未写完的新 SMEM tile。

因此每次 reuse 都必须区分：

```text
上一次 phase 0 的完成状态
这一次 phase 0 的新完成状态
```

phase parity 的作用就是让 consumer 明确表达：

```text
我现在等待的是当前这一轮
不是上一轮残留的完成状态
```

记忆方式：

```text
barrier 可以复用
completion 不能原地复用
parity 用来区分相邻两轮
```

## 六、双 stage pipeline 为什么只用一个 phase_tma

双 stage TMA pipeline 通常有两个 barrier：

```text
full[0] 跟踪 stage 0
full[1] 跟踪 stage 1
```

两个 barrier 各自独立地从 phase 0 开始。软件用一个共享的
`phase_tma` 表示当前经过 circular buffer 的第几个 round：

```text
stage = iteration % 2
try_wait(full[stage], phase_tma)

if stage == 1:
    phase_tma ^= 1
```

前四次 iteration：

| iteration | stage | 等待的 parity | stage barrier 完成后 | `phase_tma` 完成后 |
|---:|---:|---:|---:|---:|
| 0 | 0 | 0 | 1 | 0 |
| 1 | 1 | 0 | 1 | 1 |
| 2 | 0 | 1 | 0 | 1 |
| 3 | 1 | 1 | 0 | 0 |

为什么 `phase_tma` 不是在 iteration 0 后就翻转？

因为 iteration 0 只完成了 stage 0。当前这一轮 circular buffer 还包含
stage 1，必须等两个 stage 都访问完，才进入下一轮：

```text
stage 0 phase 0
stage 1 phase 0
-------- 当前 round 完成 --------
stage 0 phase 1
stage 1 phase 1
```

这里重要的是：

```text
每个 stage barrier 自己维护硬件 phase
phase_tma 是软件记录的当前 round parity
两者描述相关但不同层次的状态
```

## 七、不要和 swizzle phase 混淆

这里的 mbarrier phase：

```text
表示 barrier 的第几次复用
parity 在 0 和 1 之间翻转
```

shared memory swizzle 中的 phase：

```text
表示地址落在哪个 swizzle atom / phase
用于生成正确的 shared memory 地址
```

二者名字相同，但解决的是完全不同的问题。

## 八、完整判断流程

看到 `mbarrier.try_wait` 时，按下面顺序检查：

```text
1. 当前 producer 是谁
2. 当前 consumer 是谁
3. 当前等待的是哪个 stage barrier
4. 这是该 barrier 的第几轮 reuse
5. 当前 iteration 应该等待 parity 0 还是 1
6. phase 在什么时候翻转
7. phase 完成是否还依赖 TMA tx-count
```

一句话记忆：

```text
arrive 减少 pending count
TMA complete-tx 减少 tx-count
两个计数都归零，当前 phase 才完成
phase 完成后 parity 翻转
consumer 必须等待当前 round 对应的 parity
```

## 九、threads -> async hardware：fence.proxy.async 与 thread sync

```text
本次讲解位置
章节：chapter_async_barriers
小节：Common synchronization handoffs / Threads to asynchronous hardware
知识点：generic proxy 写入 SMEM 后交给 async proxy 读取
上次：mbarrier 的 phase 生命周期与 parity
下次：Using barriers to reuse a stage -> full/empty barrier
PTX：9.7.15.4 membar / fence
     fence.proxy.async.shared::cta
```

这一课只解决一个具体问题：

```text
普通 threads 把数据写进 SMEM
随后由 TMA engine 读取并 store 到 GMEM
如何保证 TMA 看到的是完整、正确的新数据
```

课程中的 epilogue 路径是：

```text
TMEM
  -> registers
  -> Dsmem
  -> TMA store
  -> GMEM
```

例如输出 tile 是：

```text
Dsmem: 128 x 128 fp16
```

每个 thread 负责写一行：

```python
Tx.copy(
    Dsmem[warp_id * 32 + lane_id, 0:BLK_N],
    Dreg_f16[:],
)
```

当 `tid = 37` 时，它负责写：

```text
Dsmem[37, 0:128]
```

随后只由一个 thread 发出 TMA store：

```python
if tid == 0:
    Tx.copy_async(
        D[m_st : m_st + BLK_M, n_st : n_st + BLK_N],
        Dsmem[:, :],
        dispatch="tma_auto",
    )
```

问题不在 TMA 指令本身，而在发出 TMA store 之前，需要满足两个不同条件。

### 一、条件一：所有 thread 都写完了

`tid == 0` 只知道自己的那一行已经写完，不知道其他 127 个 thread 是否完成。

可能出现：

```text
tid 0   已经写完 Dsmem[0, :]     -> 准备发 TMA
tid 100 还在写 Dsmem[100, :]      -> 尚未完成
```

如果此时 TMA store 已经启动，TMA engine 可能读到：

```text
完整的 row 0
旧的或部分更新的 row 100
```

所以需要 thread synchronization。课程使用：

```python
T.cuda.warpgroup_sync(10)
```

它让 warpgroup 内参与写入的 128 个 threads 都到达同步点之后，
`tid == 0` 才能继续发出 TMA store。

这一层解决的是：

```text
执行顺序
所有 writer 是否都已经完成写入
```

### 二、条件二：generic proxy 的写入对 async proxy 可见

普通 thread 执行的 shared memory load / store 属于：

```text
generic proxy
```

TMA engine 访问 SMEM 属于：

```text
async proxy
```

即使 thread 已经执行完 `st.shared`，也不能只凭普通 thread synchronization
就假设 TMA async proxy 一定能观察到这次写入。两个 proxy 之间还需要显式
建立顺序：

```python
T.ptx.fence.proxy_async("shared::cta")
```

它对应 PTX：

```text
fence.proxy.async.shared::cta;
```

含义是：

```text
当前 thread 此前通过 generic proxy 对 shared::cta 的写入
在后续通过 async proxy 的访问之前建立顺序
```

注意 fence 是“当前 thread 的写入发布”，它不会等待其他 thread：

```text
tid 37:
    写 Dsmem[37, :]
    fence.proxy.async
```

这只能说明 tid 37 自己的写入已经对 async proxy 建立顺序，不能让 tid 0
知道 tid 37 是否已经执行到 fence。

### 三、两个条件必须同时满足

课程中的正确顺序是：

```python
Tx.copy(Dsmem[warp_id * 32 + lane_id, 0:BLK_N], Dreg_f16[:])

T.ptx.fence.proxy_async("shared::cta")
T.cuda.warpgroup_sync(10)

if tid == 0:
    Tx.copy_async(
        D[m_st : m_st + BLK_M, n_st : n_st + BLK_N],
        Dsmem[:, :],
        dispatch="tma_auto",
    )
```

可以拆成：

```text
thread 写 SMEM
    -> generic proxy 产生数据

fence.proxy.async
    -> 发布当前 thread 的写入给 async proxy

warpgroup_sync
    -> 等所有 writer 都完成“写入 + fence”

tid 0 发 TMA store
    -> TMA engine 读取完整且可见的 Dsmem
```

两者不能互相替代：

| 机制 | 解决的问题 |
|---|---|
| `fence.proxy.async` | generic proxy 到 async proxy 的可见性与顺序 |
| `warpgroup_sync` | 所有 writer 是否已经完成 |

因此：

```text
只有 fence，没有 sync
    -> tid 0 可能提前发 TMA，读到其他 thread 尚未写完的数据

只有 sync，没有 fence
    -> 所有 thread 确实写完了
       但 TMA async proxy 仍可能看不到这些 generic 写入
```

### 四、TMA store 发出后，还要等待它完成

`threads -> TMA` 表示数据已经从 SMEM 交给 TMA engine。TMA store 开始后，
`Dsmem` 仍然不能被下一轮 epilogue 覆盖，因为 TMA engine 可能还在读取。

课程使用 bulk async group 等待 store 完成：

```python
T.ptx.cp_async.bulk.commit_group()
T.ptx.cp_async.bulk.wait_group(0)
```

TVM 中这个 intrinsic 的签名是：

```python
ptx_cp_async_bulk_wait_group(n=0, read=True)
```

因此上面的代码默认会生成：

```text
cp.async.bulk.wait_group.read 0
```

`0` 表示最多允许 0 个 bulk async group 继续 pending。`.read` 限定等待的
是 tensormap 和 source 的读取完成，而不是 destination 的写入完成。

对于课程里的 TMA store，这里的等待目的正好是：

```text
TMA engine 已经读完 Dsmem
-> 这块 buffer 可以给下一轮 epilogue 覆盖
```

如果还需要等待完整的 destination write 和可见性，要使用：

```python
T.ptx.cp_async.bulk.wait_group(0, read=False)
```

它生成 `cp.async.bulk.wait_group 0`，等待 source read、destination write
以及相应可见性都完成。

课程再用一次同步，把“tid 0 已经确认 TMA store 完成”传播给整个
warpgroup：

```python
T.cuda.warpgroup_sync(10)
```

完整写回顺序可以记成：

```text
1. 所有 thread 写 Dsmem
2. 每个 thread 执行 fence.proxy.async
3. warpgroup_sync
4. tid 0 发出 TMA store
5. tid 0 commit_group + wait_group(0)
6. warpgroup_sync
7. 下一轮可以复用 Dsmem
```

### 五、和 TMA load 的方向对比

TMA load 的方向是：

```text
GMEM -> TMA async proxy -> SMEM -> threads 读取
```

所以 consumer 主要等待：

```text
mbarrier 当前 phase 完成
```

TMA store 的方向是：

```text
threads -> generic proxy 写 SMEM -> TMA async proxy 读取 -> GMEM
```

所以 producer 在交给 TMA 之前需要：

```text
fence.proxy.async + thread synchronization
```

随后等待 TMA store 完成使用：

```text
commit_group + wait_group
```

一句话记忆：

```text
fence 负责让 TMA 看见 thread 写的 SMEM
sync 负责让 TMA issuer 知道所有 thread 都写完了
wait_group 负责确认 TMA 已经读完 SMEM
```

## 十、Using barriers to reuse a stage

```text
本次讲解位置
章节：chapter_async_barriers
小节：Using barriers to reuse a stage
知识点：full[stage] / empty[stage] 与 stage 所有权交接
上次：threads 写 SMEM 后通过 fence.proxy.async 与 thread sync 交给 TMA
下次：chapter_clc -> 静态 persistent scheduler 的局限
PTX：mbarrier.arrive / mbarrier.try_wait.parity
```

上一课解决的是：

```text
thread 写进 SMEM 的数据
什么时候能被 TMA async proxy 看见
```

这一课继续解决：

```text
一个 SMEM stage 被反复复用时
producer 什么时候可以覆盖它
consumer 什么时候可以读取它
```

只靠 `stage = k % S` 不够。stage 编号相同，只能说明两次使用落在
同一块内存，不能说明上一轮 consumer 已经读完。

### 一、full 和 empty 分别表示什么

每个 stage 使用一对 barrier：

| barrier | 所有权方向 | 完成时代表 |
|---|---|---|
| `full[stage]` | producer -> consumer | 数据已经写入，consumer 可以读 |
| `empty[stage]` | consumer -> producer | 数据已经用完，producer 可以覆盖 |

它们在 steady state 中构成一个循环：

```text
EMPTY / 可写
    |
    | producer 获得所有权
    v
producer 写入 stage
    |
    | full[stage] 完成
    v
FULL / 可读
    |
    | consumer 获得所有权
    v
consumer 读取 stage
    |
    | arrive empty[stage]
    v
EMPTY / 可写
```

一句话记忆：

```text
wait 是获取所有权
arrive 是归还所有权
full 归还给 consumer
empty 归还给 producer
```

### 二、TMA load pipeline 中的具体协议

以两级 pipeline 为例：

```text
S = 2
stage = k % 2
```

producer 侧负责 TMA load：

```text
Producer:
    如果需要复用 stage：
        wait empty[stage]

    issue TMA load -> SMEM[stage]
    full[stage] 在 TMA 完成后完成当前 phase
```

consumer 侧负责 MMA 或 `tcgen05.mma`：

```text
Consumer:
    wait full[stage]
    使用 SMEM[stage] 中的数据
    arrive empty[stage]
```

它们不是同一个线程。`full` 把数据从 producer 交给 consumer，
`empty` 把 buffer 从 consumer 还给 producer。

四个 K tile 的 stage 映射仍然是：

```text
k0 -> stage 0
k1 -> stage 1
k2 -> stage 0
k3 -> stage 1
```

但是 `k2 -> stage 0` 之前必须满足：

```text
consumer 已经读完 k0
producer 已经观察到 empty[0]
```

否则 producer 可能覆盖 consumer 仍在读取的 k0 数据。

### 三、expected arrival count 取决于谁负责报告完成

barrier 初始化时指定的 expected arrival count，必须等于每个 phase
实际会发生的 arrival 次数。

| barrier | 常见 arrival 来源 |
|---|---|
| `full[stage]` | TMA issuer 的 arrival，以及 TMA 的 complete-tx |
| `empty[stage]` | consumer 完成读取后的一个或多个 arrival |

例如：

```text
full[stage] init expected count = 1
    producer 只有一个 elected thread 发起 TMA

empty[stage] init expected count = 128
    consumer warpgroup 的 128 个 thread 都要报告读完
```

也可以只让一个 elected thread 在确认整个 consumer 操作完成后
`arrive empty[stage]`：

```text
empty[stage] init expected count = 1
    consumer 只有一个 elected thread 负责归还 stage
```

关键不是固定写 `1` 或 `128`，而是：

```text
wait 方要求的完成条件
必须和 arrive 方实际报告的次数完全一致
```

expected count 太小，可能提前完成，consumer 会读到未完成数据。
expected count 太大，phase 永远无法完成，wait 会一直阻塞。

如果 consumer 本身是异步操作，例如 `tcgen05.mma`，elected thread
不能只在“发出 MMA”时立刻归还 stage。它必须先确保 MMA 已经完成，
或通过 `tcgen05.commit` 把完成事件关联到 barrier，再产生对应的
`empty` arrival。这里的“用完”必须表示硬件已经不需要继续读取
这块 SMEM，而不是仅仅表示指令已经发出。

### 四、tcgen05.commit 如何产生 empty arrival

`tcgen05.commit` 不是一次立即发生的软件 arrival。它建立的是：

```text
当此前发出的异步 tcgen05 operations 完成后
由硬件对指定 mbarrier 执行一次 mbarrier::arrive::one
```

因此：

```text
tcgen05.commit [mbar]
```

可以理解成向硬件注册一个延迟完成通知：

```text
commit 指令执行
    -> 只是建立关联
    -> 此时 pending count 还没有减少

tcgen05 operations 完成
    -> 硬件执行 mbarrier.arrive(mbar, 1)
    -> pending count 减 1
```

假设 `empty[s]` 表示 MMA 已经不再需要这块 SMEM：

```python
mbarrier_init(empty[s], expected_count=1)

if elected:
    tcgen05_mma(A_smem[s], B_smem[s], accumulator)
    tcgen05_commit(empty[s])
```

这里不需要再额外执行：

```python
mbarrier_arrive(empty[s])
```

因为 `tcgen05.commit` 已经会在 MMA 完成后提供一次 arrival。producer
仍然通过下面的 wait 获得 stage 所有权：

```text
wait empty[s], phase
```

这里必须核对 arrival 次数：

```text
empty[s] expected count = 1
    commit 提供 1 次 arrival
    不需要软件 arrive

empty[s] expected count = 128
    commit 只提供 1 次 arrival
    还需要另外 127 次 arrival
```

如果把 `commit` 提供的那次 arrival 和软件 `mbarrier.arrive` 重复记账，
可能导致 phase 提前完成，或者让后续轮次的 pending count 与原计划
错位。

标准 `tcgen05.commit` 等待此前异步 tcgen05 operations 的完整完成。
较新的 PTX 还提供：

```text
tcgen05.commit...sync_restrict::shared::read::mma::a
```

这个变体在 A operand 已经从 SMEM 读取完成时触发 arrival，不等待整个
MMA 最终完成。它适合只保护 A 对应 SMEM stage 的复用，不能用来表示
TMEM accumulator 已经可以读取。

### 五、full 和 empty 要分别跟踪 phase parity

`full[stage]` 和 `empty[stage]` 是两个独立的 `mbarrier`，因此也有
两套独立的 phase。不能共用一个 parity 变量。

等待 parity `p` 的含义是：

```text
等待该 barrier 的第 p 个 parity 对应的 phase 完成
```

以 stage 0 为例：

| stage 0 使用轮次 | consumer 等 `full[0]` | producer 覆盖前等 `empty[0]` |
|---:|---:|---:|
| 第 1 次，k0 | parity 0 | 初始就是 free，通常不等 |
| 第 2 次，k2 | parity 1 | parity 0 |
| 第 3 次，k4 | parity 0 | parity 1 |

这里有两个容易混淆的点：

```text
full[0] 第 1 次等待 parity 0
    TMA 完成第一次 fill 时，phase 0 完成

empty[0] 第 1 次真正等待 parity 0
    consumer 第一次用完 k0 后，phase 0 完成

producer 第 2 次使用 stage 0 时
    等待的就是这次 empty[0] phase 0 完成
```

prologue 第一次填满 stage 时，stage 本来就处于 free 状态，因此可以：

```text
直接填充
```

或者在初始化后预先完成一次 `empty[stage]`，明确表示所有 stage
初始均为空闲。两种实现都必须保证“第一次填充前不等待一个尚未
发生过的 consumer release”。

### 六、和上一课的边界如何连接

上一课的 `fence.proxy.async` 和 `warpgroup_sync` 解决的是：

```text
threads 已经写好的 SMEM
如何安全地交给 TMA 读取
```

这一课的 `full / empty` 解决的是：

```text
一块 stage 在 producer 和 consumer 之间
如何在多轮 iteration 中反复交接所有权
```

因此完整的理解顺序是：

```text
1. producer 获得 free stage
2. producer 写完 SMEM
3. fence + thread sync
4. TMA 或 consumer 读取这块数据
5. full / empty 对应的 phase 报告完成
6. 另一方获得所有权
```

最后不要混淆两类 barrier：

| 对象 | 用途 |
|---|---|
| `full[stage]` / `empty[stage]` | SMEM stage 的数据就绪与 buffer 归还 |
| `bar.sync 10` 这类 named barrier | 让一组 thread 在代码位置同步 |

前者是放在 SMEM 中、带 phase 和 arrival 协议的对象；后者只是
named barrier slot。二者解决的问题不同。

## 十一、当前进度

`chapter_async_barriers` 的知识点：

```text
[x] mbarrier
[x] How phases distinguish consecutive uses
[x] Common synchronization handoffs
[x] Using barriers to reuse a stage
```

已经覆盖：

```text
mbarrier 的 arrival / pending count / phase / parity
phase 完成条件
phase 完成后自动进入下一轮
软件侧 phase parity 的翻转
复用 barrier 时等待旧 parity 的风险
双 stage pipeline 中 stage barrier 与 phase_tma 的关系
generic proxy 与 async proxy 的区别
threads 写 SMEM 后通过 fence.proxy.async 发布给 TMA
warpgroup_sync 保证所有 SMEM writer 已完成
TMA store 使用 commit_group / wait_group 等待源 buffer 可复用
TMA load 与 TMA store 两种交接方向的区别
full[stage] / empty[stage] 的 producer-consumer 所有权协议
empty barrier 的初始 free 状态
full / empty 各自的 expected arrival count
full / empty 两套独立的 phase parity
tcgen05.commit 作为 deferred hardware arrival
commit arrival 与软件 mbarrier.arrive 不能重复记账
sync_restrict::shared::read::mma::a 的提前释放语义
chapter_async_barriers 完成
```

下一知识点：

```text
chapter_clc
-> 静态 persistent scheduler 的局限
```
