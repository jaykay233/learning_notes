# Async Coordination: mbarrier 的 phase 生命周期

本篇已经覆盖：

- `mbarrier` 的 arrival、pending count、phase 与 phase parity
- 一个 phase 完成的条件
- phase 完成后为什么自动进入下一轮
- 软件为什么只记录 parity 的 `0 / 1`
- 为什么复用 barrier 时必须切换等待的 parity

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

## 九、当前进度

`chapter_async_barriers` 的知识点：

```text
[x] mbarrier
[x] How phases distinguish consecutive uses
[ ] Common synchronization handoffs
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
```

下一知识点：

```text
chapter_async_barriers
-> Common synchronization handoffs
-> threads 写 SMEM 后交给异步硬件读取
```
