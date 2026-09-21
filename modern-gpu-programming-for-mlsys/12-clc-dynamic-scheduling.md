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

## 八、一次 CLC 请求

### 本次讲解位置

```text
本次讲解位置
章节：chapter_clc
小节：一次 CLC 请求
知识点：clusterlaunchcontrol.try_cancel.async 的提交、16-byte response 与 mbarrier 完成通知
上次：静态 persistent scheduler 的 launch tail 与工作不均衡
下次：把 CLC 请求与当前 tile 的计算重叠
PTX：9.7.15.18 clusterlaunchcontrol.try_cancel
     9.7.15.19 clusterlaunchcontrol.query_cancel
```

上一课已经说明静态 scheduler 为什么会产生 launch tail。CLC 允许
运行中的 CTA 或 cluster 请求取消一个尚未启动的 launch，并接管其
coordinate。本课只讲一次请求内部发生了什么。

### 一、请求的对象是 pending cluster launch

PTX 的核心指令是:

```text
clusterlaunchcontrol.try_cancel.async.mbarrier::complete_tx::bytes.b128 [addr], [mbar];
```

各部分的含义如下:

| 部分 | 含义 |
|---|---|
| `clusterlaunchcontrol.try_cancel` | 请求取消一个尚未开始的 cluster launch |
| `.async` | 异步提交请求，发出后线程继续执行 |
| `.mbarrier::complete_tx::bytes` | 完成后用 `complete-tx` 更新指定 `mbarrier` |
| `.b128` | 结果是一条 16-byte opaque response |
| `[addr]` | response 写入的 shared memory 地址，必须 16-byte 自然对齐 |
| `[mbar]` | 用于报告异步完成的 `mbarrier` |

这里的 `cancel` 不是杀掉正在运行的 CTA。被取消对象必须还没有开始
执行，因此没有执行状态需要迁移。硬件取消成功后，只把原 launch 的
coordinate 返回给请求者。

### 二、response 是不透明的 16-byte handle

硬件不会直接把一个普通整数写回寄存器，而是在 shared memory 中写
一条 16-byte 记录:

```text
[addr] + 0  ... [addr] + 15
+--------------------------------+
|      16-byte opaque handle     |
+--------------------------------+
```

应用代码不能自行解释这些 bit。它必须把整条 handle 读到 16-byte
register，再交给:

```text
clusterlaunchcontrol.query_cancel.is_canceled
```

只有当结果 predicate 为 true 时，才可以执行:

```text
clusterlaunchcontrol.query_cancel.get_first_ctaid
```

取得被取消 cluster 第一个 CTA 的 `(x, y, z)` coordinate。

例如使用一维 launch:

```text
grid = 12 CTAs
blockIdx.x = i 的 CTA 负责 tile i
CTA 0 当前计算 tile 0
```

CTA 0 发出请求后，硬件可能取消原来的 CTA 3:

```text
被取消的 coordinate:
(x, y, z) = (3, 0, 0)

tile 映射:
tile = x = 3
```

这样 CTA 0 后续可以计算 tile 3。

如果请求失败，`get_first_ctaid` 的结果无效。PTX 还规定，一个 CTA
一旦观察到 `try_cancel` 失败，再次发出 `try_cancel` 的行为是未
定义的。因此失败后应结束取任务循环，而不是继续重试。

### 三、异步完成必须通过 mbarrier 观察

指令发出后，线程可以继续执行，但此时 response 可能还没有写完:

```text
try_cancel 提交
    -> 线程继续执行
    -> response 尚未可用
    -> 不能读取 [addr]
```

CLC 使用与 TMA load 相似的完成机制。请求需要同时满足:

```text
arrival 条件
    发起请求的一方报告约定次数的 arrival

transaction 条件
    硬件写完 16-byte response
    通过 complete-tx 减去 16 bytes
```

`mbarrier` 的 phase 完成条件可以写成:

```text
pending arrivals == 0
&&
pending transaction bytes == 0
```

只选择一个 thread 发起请求时，概念上可以写成下面的伪代码:

```text
mbarrier.init(bar, 1)

elected thread:
    mbarrier.arrive.expect_tx(bar, 16)
    clusterlaunchcontrol.try_cancel... [response], [bar]

hardware:
    response 写完
    complete_tx(bar, 16)
```

这里需要分清两件事:

```text
try_cancel 指令负责 complete-tx(response bytes)
arrival 仍需由发起方按 mbarrier 协议提供
```

因此不能只初始化 barrier 而不提供对应 arrival。否则即使 response
已经写完，phase 也可能无法完成。

### 四、一次请求的状态变化

| 阶段 | 执行者 | `mbarrier` 状态变化 | 是否可以读 response |
|---|---|---|---|
| 初始化 | 一个 thread | `pending arrival = 1` | 否 |
| 登记请求 | elected thread | `arrival` 减少，`tx-count += 16` | 否 |
| 发出请求 | elected thread | 指令异步执行 | 否 |
| 写入 response | 硬件 async proxy | response 写入 SMEM | 否 |
| 报告完成 | 硬件 | `tx-count -= 16` | phase 完成后才可以 |
| 读取 handle | worker threads | 不变 | 是 |
| 查询结果 | worker threads | 不变 | 由 `is_canceled` 决定 |

这里仍然沿用上一章的 phase 思想:

```text
等待 CLC barrier 完成
不是只等 response 写完
还要等约定的 arrival 条件满足
```

### 五、多个 thread 会产生多个独立请求

通常只选择一个 thread 提交请求:

```text
if elected:
    issue one request
```

如果多个 thread 都执行这条指令，它们会产生多个独立取消请求。
此时必须准备多个 response 位置，并分别计入:

```text
request count
response buffer count
mbarrier arrival count
mbarrier tx-count
```

例如两个请求，每个 response 16 bytes:

```text
response 0: 16 bytes
response 1: 16 bytes

tx-count 需求 = 16 + 16 = 32 bytes
arrival 需求 = 2
```

否则可能出现:

```text
barrier 等两个 arrival，但只有一个请求
两个请求只准备一个 response buffer
response 写入位置互相覆盖
tx-count 只登记 16，却会完成 32 bytes
```

### 六、response 写入与普通线程读取跨越 proxy

CLC response 由硬件通过 async proxy 写入 shared memory，普通 thread
则通过 generic proxy 读取 handle:

```text
硬件 async proxy 写 response
        |
    mbarrier 完成通知
        |
thread generic proxy 读 handle
        |
query_cancel 解析 handle
```

等待 `mbarrier` 能确认异步操作已经完成，但跨 proxy 的缓冲区复用
还要遵守 PTX 的 fence 要求。读取完 response 后，代码需要建立本轮
generic read 已经结束的顺序，再允许下一轮 async write 覆盖同一个
`[addr]`。

在本课只需要先记住:

```text
CLC response 不是普通 thread 写的
CLC 完成不能只靠普通 ld.shared 判断
必须使用 mbarrier 完成通知和 query_cancel
```

### 七、跨 proxy 到底是什么

这里的 `proxy` 不是 VPN、物理 memory bank 或 cache level。它是 PTX
对“这一次 memory access 走哪套访问路径”的抽象。不同 proxy 访问
同一个 shared memory 地址时，完成通知不代表跨 proxy 的读写顺序
已经自动安全。

| Proxy | 典型操作 |
|---|---|
| generic proxy | 普通 thread `ld.shared`、`st.shared` |
| async proxy | TMA、`cp.async.bulk`，以及 CLC response 这类异步硬件写入 |

CLC 的路径是:

```text
async proxy 写入 16-byte CLC response
    -> complete-tx 更新 mbarrier
    -> generic proxy thread 读取 response
    -> query_cancel 解析 response
```

这里有两项独立责任:

```text
mbarrier
    回答“异步操作完成了吗？”

proxy fence
    回答“跨 proxy 的读写顺序安全吗？”
    特别是那个 response buffer 能否被下一轮覆盖
```

可以用邮箱类比:

```text
async proxy: 硬件把信放进邮箱
generic proxy: thread 从邮箱取信
mbarrier: 确认信已经放进邮箱
proxy fence: 确认上一封信已经读完，才能清空并覆盖邮箱
```

如果下一轮异步请求会复用同一个 `[addr]`，只等待上一轮 `mbarrier`
完成还不够。generic proxy 的读取必须先形成可被 async proxy 观察
的顺序，之后下一轮 async write 才能安全覆盖这块 response buffer。

TMA 也会遇到方向相反的同类问题:

```text
普通 thread 通过 generic proxy 写 SMEM
    -> fence.proxy.async
    -> async TMA 读取 SMEM
```

也就是说，`mbarrier` 只负责发布“完成”，proxy fence 负责建立跨
访问路径的顺序。两者不能互相替代。

相关 PTX 位置:

```text
8.6 Proxies
9.7.10.28.2 Async Proxy
9.7.15.18 clusterlaunchcontrol.try_cancel
9.7.15.19 clusterlaunchcontrol.query_cancel
```

PTX 特别说明，`try_cancel` 对 `mbarrier` operand 的访问通过
generic proxy 完成，而 CLC response 由 async proxy 异步写入。
这正是 CLC response 会出现跨 proxy 顺序问题的原因。

### 八、本课边界

这里已经走完一条请求的生命周期:

```text
提交 try_cancel
等待 mbarrier
读取 16-byte handle
query_cancel.is_canceled
成功时 get_first_ctaid
失败时结束取任务
```

下一课要解决:

```text
什么时候发请求
为什么要提前发
请求等待期间 worker 应该做什么
```

答案是提前发出下一块工作的请求，让 grid scheduler 的延迟与当前
tile 的计算重叠。

## 九、当前进度

`chapter_clc` 的知识点:

```text
[x] The limits of a static persistent scheduler
[x] One CLC request -> clusterlaunchcontrol.try_cancel.async
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
clusterlaunchcontrol.try_cancel.async 的 16-byte response
CLC request 的 mbarrier arrival 与 complete-tx 完成条件
clusterlaunchcontrol.query_cancel 的 is_canceled 与 get_first_ctaid
多个 thread 提交 CLC request 时的 response 与 barrier 计数
CLC response 的 async-proxy 写入与 generic-proxy 读取
generic proxy 与 async proxy 的抽象含义
CLC response 的跨 proxy 读写顺序
mbarrier 完成通知与 proxy fence 的分工
```

下一知识点:

```text
chapter_clc
-> Overlap the request with the current tile
```
