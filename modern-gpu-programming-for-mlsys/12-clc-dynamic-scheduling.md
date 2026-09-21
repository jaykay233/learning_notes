# Cluster Launch Control: 静态 persistent scheduler 的局限

本篇开始学习 `chapter_clc`。这一课只回答一个问题：

```text
为什么静态 persistent scheduler 会产生 launch tail？
```

CLC 请求本身、shared memory response 和 `mbarrier` 完成通知留到下一课。

## 一、本次讲解位置

```text
本次讲解位置
章节：chapter_clc
小节：The limits of a static persistent scheduler
知识点：静态 persistent scheduler 为什么会产生 launch tail
上次：full/empty stage ownership 与 tcgen05.commit arrival
下次：One CLC request -> clusterlaunchcontrol.try_cancel.async
PTX：本知识点先讲调度模型，不涉及新 PTX 指令
```

上一章解决的是异步操作的进度交接：

```text
TMA load / store、tcgen05 等异步操作
如何通过 mbarrier、phase 和 stage ownership 与 thread 交接
```

本章把视角从“一块 tile 如何完成计算”提升到:

```text
多个 output tiles
如何分配给已经驻留的 CTA 或 CTA cluster
```

## 二、Persistent worker 与静态分配

Persistent kernel 通常只启动少量长期运行的 worker。这里的 worker
可以是一个 CTA，也可以是一个 thread block cluster。

每个 worker 连续处理多个 output tiles，从而减少以下开销：

```text
反复启动 CTA
反复执行公共准备工作
反复初始化部分 kernel 状态
```

最简单的 persistent scheduler 使用静态分配。假设有 12 个 tiles 和
4 个 workers:

```text
worker 0: tile 0, 4, 8
worker 1: tile 1, 5, 9
worker 2: tile 2, 6, 10
worker 3: tile 3, 7, 11
```

常见实现就是:

```text
next_tile = worker_id + iteration * worker_count
```

这种 scheduler 的优点是：

| 特性 | 静态 persistent scheduler |
|---|---|
| 下一块 tile 的来源 | 简单公式计算 |
| 动态调度开销 | 几乎没有 |
| 是否需要软件工作队列 | 不一定需要 |
| 对运行环境的假设 | workers 能同时启动，tile 成本接近 |

只要这些假设成立，静态分配通常已经足够。

## 三、worker 延迟启动带来 launch tail

kernel 实际能同时运行多少个 workers，并不总能提前准确知道。其他
kernel 可能正在占用一部分 SM，或者当前 kernel 的资源需求使硬件
只能先驻留其中一部分 CTAs。

假设开始时只能运行 worker 0、1、2，worker 3 要过一段时间才能启动:

| 阶段 | 正在执行的 worker | 尚待处理的静态任务 |
|---|---|---|
| 开始 | 0、1、2 | worker 3 的 `3, 7, 11` |
| worker 0-2 完成 | 0、1、2 退出 | `3, 7, 11` |
| 尾部 | 只有 3 | `3, 7, 11` |

时间线可以表示为:

```text
worker 0:  0 ---- 4 ---- 8 ---- 结束
worker 1:  1 ---- 5 ---- 9 ---- 结束
worker 2:  2 ---- 6 ---- 10 --- 结束
worker 3:                        3 ---- 7 ---- 11 ---- 结束
                                 ^
                                 很长的 launch tail
```

问题不在于 worker 3 的计算能力更弱，而在于静态 scheduler 在
kernel 开始前就固定了:

```text
worker 0 必须处理 tile 0, 4, 8
worker 1 必须处理 tile 1, 5, 9
worker 2 必须处理 tile 2, 6, 10
worker 3 必须处理 tile 3, 7, 11
```

即使 worker 0、1、2 已经空闲，它们也不知道 worker 3 还有哪些任务，
因此不能自动接手 `3, 7, 11`。

## 四、tile 成本不同也会导致不均衡

即使所有 workers 同时启动，静态分配仍可能不均衡。假设每块 tile
耗时如下:

| tile | 耗时 |
|---:|---:|
| 0、1、2 | 10 us |
| 3 | 40 us |
| 4、5、6 | 10 us |
| 7 | 40 us |
| 8、9、10 | 10 us |
| 11 | 40 us |

那么各 worker 的总耗时约为:

| worker | tiles | 总耗时 |
|---|---|---:|
| 0 | 0、4、8 | 30 us |
| 1 | 1、5、9 | 30 us |
| 2 | 2、6、10 | 30 us |
| 3 | 3、7、11 | 120 us |

worker 0、1、2 结束后，worker 3 还要继续运行很久。边界 mask、
不同 shape 的尾部、融合 epilogue、稀疏计算和输入数据差异，都可能
让 tile 成本不相等。

静态 scheduler 看不到实时完成情况，因此无法执行:

```text
worker 0 已经空闲
把 worker 3 尚未开始的一块 tile 交给 worker 0
```

## 五、CTA launch queue 不是软件工作队列

这里的 `launch queue` 指硬件保存尚未启动 CTA 或 cluster 的队列。
它和程序自己实现的软件工作队列不是同一个对象。

| 队列 | 保存内容 | 谁负责调度 | 典型开销 |
|---|---|---|---|
| CTA launch queue | 尚未启动的 grid coordinate | GPU grid scheduler | 不需要程序原子操作 |
| 软件工作队列 | 程序定义的任务编号 | kernel 自己 | atomic、同步、全局内存通信 |

传统动态调度经常需要自行维护软件工作队列。它虽然能适应实时进度，
但会引入额外通信和 contention。

CLC 的目标是让已经运行的 CTA 或 cluster:

```text
取消一个尚未启动的 launch
接管它的 coordinate
```

这样不需要程序自己保存所有待办任务，也能让先完成的 worker 继续
领取 pending work。

## 六、CLC 如何改变上面的例子

CLC kernel 的 launch grid 仍然覆盖全部 tiles。这里假设:

```text
blockIdx.x = 0 ... 11
blockIdx.x = i 的 CTA 原本负责 tile i
```

如果当前只能容纳 3 个 CTAs:

```text
正在运行：CTA 0、CTA 1、CTA 2
launch queue：CTA 3 ... CTA 11
```

当 CTA 0 算完 tile 0，它不直接退出，而是向硬件请求一份 pending
work。如果硬件取消尚未启动的 CTA 3，并把 coordinate `3` 返回，
CTA 0 就可以处理 tile 3。

每个 coordinate 仍只处理一次:

```text
coordinate 3 由原本的 CTA 3 处理
或者
CTA 3 在启动前被取消，coordinate 3 交给一个正在运行的 worker
```

本课只建立调度模型。下一课再展开:

```text
clusterlaunchcontrol.try_cancel.async
16-byte response 如何写入 shared memory
mbarrier 如何报告 response 完成
query_cancel 如何读取 coordinate
```

## 七、什么时候需要 CLC

静态 scheduler 和 CLC scheduler 可以共用同一个 tile 计算主体，它们
只在“下一块 tile 从哪里来”这一点不同:

```text
静态调度：根据 worker ID 和迭代次数计算下一块 coordinate
CLC 调度：由硬件返回一个尚未启动的 CTA 或 cluster coordinate
```

因此选择标准可以概括为:

| 运行条件 | 更适合的调度方式 |
|---|---|
| 可用 SM 稳定、tile 成本接近 | 静态 scheduler |
| worker 启动时间不确定 | 动态调度更有利 |
| tile 成本差异明显 | 动态调度更有利 |
| 不能承受动态调度开销 | 静态 scheduler |

在推理系统中，这种情况对应到 GEMM、Attention 或其他 tiled kernel
时，重点是让所有已驻留 worker 尽可能保持忙碌，避免只剩少数 worker
处理 launch tail。

## 八、当前进度

`chapter_clc` 的知识点:

```text
[x] The limits of a static persistent scheduler
[ ] One CLC request -> clusterlaunchcontrol.try_cancel.async
[ ] Overlap the request with the current tile
[ ] When to use CLC
```

已经覆盖:

```text
persistent worker 为什么连续处理多个 tiles
静态 grid-stride 分配方式
worker 延迟启动造成 launch tail
不同 tile 成本造成 worker 负载不均衡
CTA launch queue 与软件工作队列的区别
CLC 取消 pending launch 并接管 coordinate 的基本模型
```

下一知识点:

```text
chapter_clc
-> One CLC request -> clusterlaunchcontrol.try_cancel.async
```
