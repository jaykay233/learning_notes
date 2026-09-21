# Async Coordination: mbarrier 与异步交接

本篇已经覆盖：

- `mbarrier` 的 arrival、pending count、phase 与 phase parity
- 一个 phase 完成的条件
- phase 完成后为什么自动进入下一轮
- 软件为什么只记录 parity 的 `0 / 1`
- 为什么复用 barrier 时必须切换等待的 parity
- threads 写 SMEM 后交给 async proxy 读取时的 fence 与 thread sync

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

含义是：

```text
commit_group
    把此前发出的 TMA stores 归入一个 group

wait_group(0)
    等所有已提交 group 完成
```

`wait_group(0)` 返回后，TMA engine 已经读完 `Dsmem`，这块 buffer 才能
被下一轮写入复用。

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

## 十、当前进度

`chapter_async_barriers` 的知识点：

```text
[x] mbarrier
[x] How phases distinguish consecutive uses
[x] Common synchronization handoffs
[ ] Using barriers to reuse a stage
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
```

下一知识点：

```text
chapter_async_barriers
-> Using barriers to reuse a stage
-> full / empty barrier 与 stage 所有权交接
```
