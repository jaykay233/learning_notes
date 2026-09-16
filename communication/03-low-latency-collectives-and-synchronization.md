# 低延迟 GPU Collective：同步、Sentinel、Credit、Multicast 与 LL128

> 承接 [01-proxy-gdaki-gpi.md](./01-proxy-gdaki-gpi.md) 和
> [02-mok-scheduling-and-buffers.md](./02-mok-scheduling-and-buffers.md)。
> 本文整理单 NVLink scale-up domain 内的小消息 collective 优化，重点回答：
> 为什么 barrier 会成为关键路径、Sentinel 怎样替代 flag、双缓冲 credit 怎样
> 消除全局同步，以及 Multicast、LL128 atomic 和 `multimem.ld_reduce` 分别
> 处在什么层次。
>
> 主要参考：
> - [Every Microsecond Matters: Achieving Near Speed-of-Light Latency in GPU Collectives](https://arxiv.org/abs/2607.16100v1)
> - [NCCL Device API: Multimem Device Kernel](https://docs.nvidia.com/deeplearning/nccl/archives/nccl_2297/user-guide/docs/usage/deviceapi.html#multimem-device-kernel)
> - [PTX ISA: `multimem.ld_reduce`, `multimem.st`, `multimem.red`](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-multimem)
> - [NCCL 2.28 Device API 与 Copy Engine Collectives](https://developer.nvidia.com/blog/fusing-communication-and-compute-with-new-device-api-and-copy-engine-collectives-in-nvidia-nccl-2-28/)

---

## 0. 一句话理解

小消息 collective 的瓶颈通常不是带宽，而是“搬数据之外的等待”：

```text
数据写入后等 fence
-> 单独写 ready flag
-> 对端轮询 flag
-> 全体 rank barrier
-> 确认旧 buffer 可以复用
-> 再做归约和广播
```

低延迟设计的主线是把这些步骤折叠起来：

```text
数据到达通知 -> 让数据本身携带
buffer 复用许可 -> 用 peer credit 代替全局 barrier
归约与扇出 -> 尽量下放到 fabric
算法选择 -> 小消息 one-shot，中等消息 two-shot
```

优化目标是让每次 collective 更接近硬件数据路径，而不是只换一种 ring/tree。

---

## 1. 为什么 decode 会把 collective 变成延迟题

张量并行把一层模型拆到多张 GPU。每层的局部结果要通过 AllReduce 合并：

```text
GPU 0: partial output 0
GPU 1: partial output 1
GPU 2: partial output 2
GPU 3: partial output 3

AllReduce
-> 四张 GPU 都得到 sum(partial 0..3)
```

不同阶段的通信形状不同：

| 阶段 | 消息特征 | 主要瓶颈 |
|---|---|---|
| Prefill | token 多，消息大 | 带宽 |
| Decode | 每步 token 少，消息小且频繁 | 延迟、同步、kernel 启动 |

Decode 的小 AllReduce 又位于 token 生成关键路径：

```text
上一层 GEMM
-> AllReduce
-> 下一层依赖完整结果
```

它不像离线梯度同步那样容易藏在长时间计算背后。层数、请求并发数和生成
token 数都会反复放大每次多出来的微秒。

因此优化目标从：

```text
每秒搬多少字节
```

转成：

```text
一次 collective 最少经过几次同步
```

---

## 2. 1.404 μs 的 Speed-of-Light 是什么

这里的 Speed-of-Light（SoL）不是信号在 NVLink 或机架中传播的物理时间，
而是论文为一个 128B cache line AllReduce 定义的理想数据路径下界。

论文的简化模型是：

```text
输入从 L2 进入 SM
-> 远端写到 peer scratch
-> 回到 SM 完成归约
```

测得：

```text
L2 RTT        约 0.306 μs
remote store  约 0.792 μs
SoL           2 × 0.306 + 0.792 = 1.404 μs
```

模型假设包括：

- 消息已经命中 L2。
- 向所有 peer 的 remote store 同时发出。
- 所有贡献同时到达。
- 归约计算不占可见时间。
- 最终 output store 只计算发射，不等待真正落盘。

所以 1.404 μs 不是所有 collective 的通用硬下界，而是一把延迟账单的尺子：

```text
实际延迟 - 1.404 μs
= 协议、同步、指令调度和 kernel 执行留下的空间
```

它的价值是帮助定位“还可以从哪里删时间”，而不是证明某条路径已经物理极限。

---

## 3. 为什么数据必须先于 flag 可见

普通发布模式：

```cpp
data[0] = new_value;
flag = 1;
```

生产者认为自己是先写数据，再写旗标：

```text
数据准备好了
-> 通知消费者
```

但消费者可能看到：

```text
flag == 1
data[0] == old_value
```

于是它会读取尚未到达或已经过期的数据。

### 3.1 为什么不同地址可能乱序

`data` 和 `flag` 是两个不同地址。GPU 编程模型通常保证：

- 对同一个地址的写入具有 coherence。
- 不同地址的写入如果没有明确内存序，对另一个线程或设备的可见顺序没有保证。

实际路径还会经过：

```text
warp 发出 store
-> LSU
-> L2 / memory partition
-> NVLink
-> peer L2
-> peer 读线程
```

两条写可能进入不同队列、不同 cache line 或不同 fabric 路径。编译器也可能
在没有依赖关系的两个写之间调整顺序。

因此问题不是“发送线程的代码顺序”，而是“对端实际观察到的顺序”。

### 3.2 fence 或 release/acquire 建立 happens-before

传统 CUDA 写法：

```cpp
// producer
data[0] = new_value;
__threadfence();
flag = 1;

// consumer
while (flag != 1) {
}
__threadfence();
value = data[0];
```

现代 C++/CUDA 原子语义可以表达为：

```cpp
// producer
data[0] = new_value;
flag.store(1, std::memory_order_release);

// consumer
while (flag.load(std::memory_order_acquire) != 1) {
}
value = data[0];
```

它建立的关系是：

```text
data 写入 happens-before flag release
flag acquire happens-before data 读取
```

结论：

```text
消费者一旦看到 flag == 1
就必须能看到此前的 data
```

### 3.3 为什么这一步在小消息里很贵

fence 不是“空指令”。它要求先前的访问达到规定的可见顺序，后续 flag 写不能
越过它。生产者可能要等待 remote store 已经到达足够的一致性观察点。

流程因此变成：

```text
搬 payload
-> 等待 fence
-> 单独写 flag
-> 对端轮询 flag
-> 对端读 payload
```

消息越小时，payload 搬运越快，等待和通知占的比例越大。这也是 LL 和
Sentinel 想解决的核心问题。

---

## 4. Sentinel：让数据本身充当到达标志

Sentinel 是一个“哨兵值”或“预留标记值”。

普通模式：

```text
buffer = 旧值
flag = 0

buffer = 新值
fence
flag = 1
```

Sentinel 模式：

```text
初始化:
buffer = SENTINEL

发送端:
buffer = 真实数据

接收端:
while buffer 仍然是 SENTINEL:
    继续等

确认已经不是 SENTINEL
-> 读取数据
```

状态变成：

```text
buffer == SENTINEL  -> 数据没到
buffer != SENTINEL  -> 数据已写入
```

数据位置本身同时承担 payload 和到达通知，不再需要单独的 ready flag。

### 4.1 为什么论文可以用 `-NaN`

正常模型计算通常不应该出现 NaN。因此论文可以用一种负 NaN 的 bit pattern
作为“不可能出现的值”。

但实现不能直接做浮点比较：

```cpp
if (x != -NaN) {
}
```

IEEE 754 中任何 NaN 都不等于自身，浮点比较会产生反直觉结果。实现通常要比较整数
bit pattern：

```cpp
uint32_t bits = bit_cast<uint32_t>(x);
if (bits != NEGATIVE_NAN_BITS) {
    // 数据到达
}
```

### 4.2 为什么它保留完整 payload

LL 常见布局：

```text
8B 真实数据 + 8B flag
-> 一次 16B atomic store
```

其中只有一半宽度承载有效 payload。

Sentinel：

```text
128B 全部承载 payload
数据值从 sentinel 变成真实值
```

因此它不像 LL 那样把有效 payload 带宽减半。

### 4.3 代价与正确性边界

Sentinel 需要额外满足：

- 合法输入不能等于 sentinel bit pattern。
- 每轮复用前必须重新填满 sentinel。
- 预处理和重置本身会消耗内存带宽。
- 如果模型数值异常真的产生 NaN，可能和协议标记冲突。
- 接收端必须确认所有需要的数据都已经不再是 sentinel。

最后一点很重要。如果只观察一个哨兵位，却直接假设整段数据全部可见，不同地址
写入的可见顺序问题仍然可能回来。安全实现要么逐元素确认，要么由发送端提供
块级顺序保证。

---

## 5. 双缓冲加 credit：把全局 barrier 换成局部许可

### 5.1 先看单 buffer 的问题

假设 rank A 不断向 rank B 的 buffer 写数据：

```text
A 写入 B.buffer
B 读取 B.buffer
```

只有一个 buffer 时：

```text
A 想写第 t+1 轮
-> 必须确认 B 已读完第 t 轮
```

最直接的解决方式就是全局 barrier：

```text
所有 rank 都读完第 t 轮
-> 大家才能开始第 t+1 轮
```

只要有一个慢 rank，所有人都会被它拖住。barrier 本身也占微秒级时间。

### 5.2 双缓冲解决“边写边读”

改成：

```text
buffer 0
buffer 1
```

第 0 轮用 buffer 0，第 1 轮用 buffer 1，第 2 轮再复用 buffer 0。

这样写入下一块时，不需要立刻等待上一块读完。但第 2 轮覆盖 buffer 0 前，
仍然必须知道：

```text
对端真的已经消费过 buffer 0
```

这个许可就是 credit。

### 5.3 credit 是什么

最直白的 credit 是显式计数器：

```text
初始:
buffer 0、buffer 1 可复用

消费者读完 buffer 0
-> 返回一个 credit

生产者消耗 credit
-> 获得覆盖 buffer 0 的资格
```

credit 用完时，生产者必须停。于是慢消费者自然给快生产者施加 backpressure。

### 5.4 论文更巧的做法：反向数据充当回执

假设两个 rank A、B 双向交换，每边有两块接收区域：

```text
A_to_B[0], A_to_B[1]
B_to_A[0], B_to_A[1]
```

第 0 轮：

```text
A 把 A0 写入 A_to_B[0]
B 把 B0 写入 B_to_A[0]
```

双方读取第 0 轮数据。

第 1 轮：

```text
A 把 A1 写入 A_to_B[1]
B 把 B1 写入 B_to_A[1]
```

因为第 1 轮使用另一块 buffer，双方不需要先等待第 0 轮确认。

现在假设协议要求：

```text
B 必须先消费 A0
才能产生并发送 B1
```

那么 A 收到 `B1` 时就知道：

```text
B 已经进入第 1 轮
=> B 已经消费 A0
=> A 可以在第 2 轮重新覆盖原来的 buffer 0
```

反过来：

```text
B 收到 A1
=> A 已经消费 B0
=> B 可以覆盖原来的 buffer 0
```

于是：

```text
B1 = 新数据 + “A0 已消费”的隐含 credit
A1 = 新数据 + “B0 已消费”的隐含 credit
```

同步没有彻底消失，而是折叠进数据依赖。

### 5.5 为什么快的一方不能无限超前

如果没有 credit，A 可能连续写很多轮，覆盖 B 尚未读取的数据。

有了双缓冲和 credit：

```text
A 想继续写
-> 必须等到 B 下一轮的数据返回
```

A 的等待不再来自“所有 rank 的全局 barrier”，而是来自“我依赖 B 的下一次发送”。

它把同步边界从：

```text
整个 communicator 的集合事件
```

缩小成：

```text
A 与 B 之间的局部依赖
```

### 5.6 使用条件

这种隐含 credit 成立需要协议保证：

- 对端的下一轮发送确实依赖上一轮消费完成。
- 双缓冲数量或 credit 数量有限。
- 每个 peer 的进度可以独立表达。
- 真实实现还要处理 epoch、phase 和错误恢复。

如果下一轮发送与上一轮消费没有依赖关系，就必须使用显式 ack/counter。

---

## 6. fabric 是什么

fabric 通常译为“互连网络”或“网络织物”。它不是单个芯片，而是把 GPU、
链路和交换芯片组织起来的整套通信系统。

```text
NVLink   = 一段链路
NVSwitch = 交换节点
fabric   = 链路、交换机、路由、缓冲和硬件能力的整体
```

一次跨 GPU 访问可以简化为：

```text
SM
-> 本地 L2
-> 本地 NVLink
-> NVSwitch / fabric
-> 对端 NVLink
-> 对端 L2
-> 对端内存
```

### 6.1 scale-up fabric

典型技术是 NVLink、NVSwitch 和 NVLink SHARP：

```text
GPU 可以直接 load/store peer memory
支持 multicast 和部分硬件归约
延迟低，位于同一 scale-up domain
```

### 6.2 scale-out fabric

典型技术是 InfiniBand、RoCE、Ethernet：

```text
GPU
-> PCIe / C2C
-> NIC
-> 交换网络
-> 对端 NIC
-> 对端 GPU
```

即使 NVLink fabric 很快，也无法消除 scale-out 的 NIC、网络协议、拥塞控制和
远端内存路径。论文的低延迟 kernel 主要限定在单个 NVLink domain。

---

## 7. Multicast、SHARP 与 LL128 atomic

这三个词处在不同层次：

```text
Multicast      负责一份数据扇出给多个 peer
SHARP 归约      负责在 fabric 路径中合并数据
LL128 atomic   负责一种具体的数据布局与归约协议
```

### 7.1 unicast 与 multicast

普通 unicast：

```text
rank 0 -> rank 1
rank 0 -> rank 2
rank 0 -> rank 3
```

rank 0 需要发起多次远端操作。

Multicast：

```text
rank 0 对 multimem 地址写一次
-> fabric 负责复制
-> rank 1、2、3 都收到
```

multimem 地址是一块逻辑地址，它映射到多张 GPU 上布局相同的副本。

### 7.2 NVLink SHARP 为什么能减少开销

普通软件归约：

```text
读取 peer 0 的值
读取 peer 1 的值
读取 peer 2 的值
SM 自己相加
```

硬件归约：

```text
发起 multimem load-reduce
-> fabric 路径合并多个副本
-> 调用者得到归约结果
```

好处包括：

- 减少 SM 发出的 load/store 指令。
- 减少每个 GPU 的 L2 往返。
- 部分加法在 NVSwitch/fabric 路径完成。
- rank 数增加时扩展性更好。

但 multicast 有固定启动成本。2 卡时它甚至可能慢于 unicast，更多 GPU 时才
开始占优。

### 7.3 LL128 atomic 是什么

LL128 atomic 把 128B cache line 组织成：

```text
[count, sum0, sum1, sum2, ..., sum30]
```

概念上对应：

- 第 1 个元素作为计数或到达 flag。
- 后面 31 个元素承载 payload。
- 每个 rank 用向量 atomic add 累计自己的贡献。
- 当 `count == rank_num` 时，表示所有贡献已到。
- 此时后面的元素是最终归约值。

和 LL 相比：

```text
LL:
8B payload + 8B flag
有效 payload 比例约 50%

LL128 atomic:
1 个计数元素 + 31 个 payload 元素
有效比例约 31/32
```

它保留了更高比例的 payload，并减少独立 flag buffer。

限制也很明显：

- 依赖硬件对 cache line 级 vector atomic 的支持。
- 浮点 atomic 到达顺序不确定。
- 结果不保证严格复现。
- 通常只适用于硬件支持的加法和特定数据类型。

---

## 8. one-shot 与 two-shot

### 8.1 one-shot：少一次同步

四个 rank 的 one-shot：

```text
每个 rank 把自己的完整贡献发送给所有 peer
每个 rank 收到全部贡献后本地归约
```

示意：

```text
rank 0 收到 A1、A2、A3
-> 计算 A0 + A1 + A2 + A3

rank 1 收到 A0、A2、A3
-> 计算 A0 + A1 + A2 + A3
```

优点：

- 通信阶段少。
- 小消息时延迟低。
- Multicast 可以替代显式逐个 peer 发送。

缺点：

- 每个 rank 的远端操作和 scratch 使用量随 rank 数增长。
- 消息变大后复制成本明显。

### 8.2 two-shot：ReduceScatter + AllGather

two-shot 拆成两个阶段：

```text
ReduceScatter:
rank 0 得到 slice 0 的全局和
rank 1 得到 slice 1 的全局和
rank 2 得到 slice 2 的全局和
rank 3 得到 slice 3 的全局和

AllGather:
各 rank 交换归约完成的 slice
-> 每个 rank 拥有完整 AllReduce 结果
```

相比 one-shot：

- 多一个通信阶段。
- 通信和 scratch 压力更均匀。
- 中等消息开始受带宽限制时通常更划算。

论文实验中的切换边界是：

```text
4 rank、小于约 1 MiB:
one-shot

4 rank、约 1-2 MiB:
two-shot

更多 rank 或更大消息:
继续根据实测选择 LL、Sentinel、LL128 atomic、multicast
或现有 symmetric kernel
```

没有单一 kernel 覆盖所有消息区间。

---

## 9. `multimem.ld_reduce` 只做 Reduce，不自动做 AllReduce

这一点很容易被 `ld_reduce` 的名字误导。

PTX 对 `multimem.ld_reduce` 的定义是：

```text
从 multimem 地址对应的所有内存副本加载
-> 按指定操作归约
-> 把结果返回到发起线程的寄存器
```

它不会自动把结果写到所有 rank。

### 9.1 四副本示例

假设：

```text
rank 0 的副本 = a
rank 1 的副本 = b
rank 2 的副本 = c
rank 3 的副本 = d
```

rank 0 执行：

```ptx
multimem.ld_reduce.global.add.f32 result, [mm_ptr];
```

含义是：

```text
读取 a、b、c、d
-> 在 fabric 路径归约
-> a + b + c + d 只返回 rank 0 的 result 寄存器
```

这时其他 rank 没有结果：

```text
rank 0: 有结果
rank 1: 没有
rank 2: 没有
rank 3: 没有
```

所以单独一次 `ld_reduce` 是 Reduce。

### 9.2 怎样补齐 AllReduce 的 “all”

第一种方式：

```text
所有 rank 都执行 multimem.ld_reduce
-> 每个 rank 都返回同一个总和
```

示意：

```text
rank 0: ld_reduce -> a+b+c+d
rank 1: ld_reduce -> a+b+c+d
rank 2: ld_reduce -> a+b+c+d
rank 3: ld_reduce -> a+b+c+d
```

“all” 来自每个 rank 都发起相同的归约读取，不是硬件偷偷广播。

第二种方式：

```text
一个 rank 执行 ld_reduce 得到总和
-> 再用 multimem.st 把总和写回所有副本
```

`multimem.st` 的语义是：

```text
把一个值写到 multimem 地址对应的所有 GPU 副本
```

因此完整阶段是：

```text
1. 所有 rank 写入各自贡献
2. barrier 或 release/acquire 保证贡献可见
3. 每个 rank 执行 ld_reduce
   或者由一个 rank 执行 ld_reduce
4. 如果只有部分 rank 执行，则需要 multimem.st 广播结果
```

NCCL 文档的简化示例是：

```cpp
T v = multimem_sum(mmPtr + o);
multimem_st(mmPtr + o, v);
```

其中：

```text
multimem_sum -> multimem.ld_reduce
multimem_st  -> 广播存储
```

生产实现还需要 phase、epoch、barrier 或版本号，避免把上一轮写回的结果当成本轮
输入。函数名中的 `ld_reduce` 只描述了数据移动和归约语义，并不代表一条指令独立
完成整个集合通信。

---

## 10. 各机制的职责不能混在一起

| 机制 | 解决的问题 | 是否独立完成 collective |
|---|---|---|
| fence / release-acquire | 保证 data 先于 flag 可见 | 否 |
| LL | 数据和 flag 打包提交 | 否 |
| Sentinel | 数据值本身表示是否到达 | 否 |
| 双缓冲 + credit | 旧 buffer 何时可安全覆盖 | 否 |
| Multicast | 一份数据扇出到多个副本 | 否 |
| `multimem.ld_reduce` | 多副本归约并返回调用者 | 只完成 Reduce |
| `multimem.st` | 把结果写到所有副本 | 只完成广播 |
| one-shot / two-shot | 选择 collective 的算法阶段 | 是，但内部仍需上述原语 |

可以把一次 AllReduce 拆成四类问题：

```text
到达通知:
data 是否已经可读

buffer 复用:
旧 slot 是否可以覆盖

归约:
多个 rank 的值怎样相加

分发:
最终结果怎样让所有 rank 都拿到
```

低延迟实现的本质就是让一个操作同时回答多个问题：

```text
LL:
数据和到达通知一起提交

Sentinel:
payload 本身兼任到达通知

credit:
下一轮数据兼任旧 buffer 的消费回执

multimem:
fabric 同时承担扇出和部分归约
```

---

## 11. 收益数字与适用边界

论文报告的代表性结果：

```text
4 张 GB200、小消息 AllReduce
NCCL ring 11.0 μs
-> 新实现 2.37 μs

vLLM inter-token latency
-> 降低约 7%-13%

2 卡、128B
-> 距离 1.404 μs SoL 约 7%

64 卡
-> 距离 SoL 扩大到约 70%
```

性能数字需要和边界一起读：

- 只适用于单个 NVLink scale-up domain。
- 不消除 scale-out NIC 和交换网络瓶颈。
- LL128 atomic 主要支持部分浮点类型和加法。
- 浮点 atomic 到达顺序不固定，结果可能不确定。
- 公开代码是论文作者的研究分支，不等于所有 NCCL 稳定接口。
- 每百万 token 成本下降是使用机型小时价换算的模型估算，与端到端 ITL 实测
  不是同一种证据强度。

适用判断可以压缩成四个问题：

```text
1. 消息是否位于单个 NVLink domain？
2. 消息是否足够小？
3. collective 是否位于 token 关键路径？
4. 业务是否接受对应数据类型和确定性边界？
```

四项同时成立，低延迟 collective 才容易进入生产收益区。

---

## 12. 一页速记

```text
为什么需要同步:
payload 和 flag 是不同地址
消费者可能先看到 flag，后看到 data
需要 fence 或 release/acquire 建立 happens-before

为什么小消息贵:
fence、flag 轮询和全局 barrier 不搬 payload
但会占微秒级关键路径

Sentinel:
预填不可能输入值
数据值变化本身就是到达通知
保留完整 payload
代价是哨兵冲突和重置

双缓冲 + credit:
双缓冲允许边写下一块、边读上一块
credit 说明旧块何时可以覆盖
反向下一轮数据可以隐含充当上一轮 ACK
快 rank 被 peer 数据依赖限速，不能无限超前

fabric:
NVLink、NVSwitch、路由和硬件能力的整体
Multicast 负责扇出
SHARP / multimem.ld_reduce 负责部分归约

LL128 atomic:
128B cache line 中第一个元素做计数
其余元素做向量 atomic add
payload 利用率高
但浮点结果可能不确定

one-shot:
每个 rank 把贡献发到所有 peer
本地归约
少同步，适合小消息

two-shot:
ReduceScatter + AllGather
多一个阶段，但通信和 scratch 更均衡
适合中等消息

multimem.ld_reduce:
只是 Reduce
结果只返回调用者
每个 rank 都调用，或者 ld_reduce + multimem.st
才组成 AllReduce

最终主线:
让数据携带同步语义
让 peer credit 替代全局 barrier
让 fabric 承担扇出和归约
按消息大小和 rank 数选择 kernel
```

---

## 13. 和 MoK 的共通点

MoK 和低延迟 collective 优化的对象不同，但设计思想有共通处：

```text
MoK:
把 schedule、buffer 生命周期和通信发起放到设备端
用固定 workspace、ring slot 和细粒度 barrier
避免 CPU 和全局屏障进入 MoE 热路径

低延迟 collective:
把到达通知和 buffer 许可折叠进数据/fabric
避免单独 flag 和全局 barrier
```

两者都在追问：

```text
谁拥有状态
谁发出就绪信号
什么时候可以覆盖旧数据
哪些等待真正搬运了 payload
```

最终优化单位不是单个 memory copy 或单个 atomic，而是整段依赖关系。
