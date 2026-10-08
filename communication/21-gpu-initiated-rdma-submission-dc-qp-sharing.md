# GPU-initiated RDMA：SM payload、doorbell、WQE fetch、完成与 DC/QP 共享

> 承接 [01-proxy-gdaki-gpi.md](./01-proxy-gdaki-gpi.md)。01 已经回答
> Proxy、GDAKI、GPI 分别由谁提交，以及 QP 粒度为什么会变化；本文沿着
> GPU-initiated RDMA 的一次真实提交继续向下走，钉死这些常见问题：
> SM 写的到底是什么，谁真正把工作交给 NIC，doorbell 为什么有两层，
> NIC 为什么要 fetch WQE，CQE 什么时候产生，UAR 为什么可能映射不了，
> 以及 DC、DCI、DCT、QP sharing 分别在节省什么。
>
> 主要参考：
> - *GPU-Initiated Communication: Dissecting Down to the Bone*，arXiv:2610.01380v1
> - 本地论文：`/Users/saboxu/Downloads/AF分离/2610.01380v1.pdf`
> - [mlx5 WQE/doorbell 与 GPUDirect RDMA 相关公开资料](https://docs.nvidia.com/networking/)

```text
本次讲解位置
章节：communication / GPU-initiated RDMA
小节：从 SM payload 到 NIC 提交、取 WQE、完成，以及连接资源模型
知识点：GPU-submitted / proxy-submitted、dbrec / UAR doorbell、WQE fetch / CQE、DCI / DCT / AV、QP sharing
上次：Proxy / GDAKI / GPI 的队列粒度与 QP 所有权（01）
下次：mini-gda / mini-proxy / NVSHMEM 的 issue、RTT 与消息率边界
原语：global store、fence、WQE、doorbell record、UAR MMIO、DMA、CQE、DCI / DCT
```

## 为什么现在讲这个

01 把“谁提交、按什么粒度建队列”讲清后，继续读 GPU-initiated RDMA 会立刻遇到一组很容易混在一起的词：SM、payload、WQE、doorbell、dbrec、UAR、CQE、DCI、DCT、QP sharing。

它们不在同一层：

```text
数据面:       source payload
控制面:       WQE / doorbell / dbrec / UAR
传输资源面:   QP / DCI / DCT / AV
完成面:       CQE / counter
线程协作面:   QP sharing / slot reservation / publication
```

如果把这些放在一句话里，就会误以为“GPU doorbell 就是在写自己的内存”，或者误以为“BlueFlame 和 DC 都是在省 QP”。本文先沿一次 put 的硬件路径向下走，再把资源模型单独抽出来。

---

## 0. 一句话模型

一次 GPU-initiated RDMA 可以压缩成下面这条链：

```text
SM 产生 payload
  -> GPU / CPU 发布 WQE
  -> GPU / CPU 写 UAR doorbell
  -> NIC 知道 SQ 有新工作
  -> NIC DMA fetch WQE 和 payload
  -> 网络传输
  -> 对端 NIC DMA 写目标内存
  -> 完成记录回写
  -> 发起端 poll CQE / counter
```

关键区分是：

```text
GPU-initiated
  谁在语义上决定要通信

GPU-submitted / proxy-submitted
  谁真正构造工作的最后提交动作并通知 NIC

QP / DCI / DCT
  网络连接状态如何组织

QP sharing
  多个 GPU 线程怎样共同使用一个发送队列
```

这四件事可以任意组合，不能互相替代。

---

## 1. “SM 写 source payload”到底指什么

这里的 `source` 是相对于一次 RDMA 操作而言的发送端，`payload` 是真正要传输的用户数据字节。

例如 rank 0 要把一个 tensor 分片发给 rank 1：

```cuda
__global__ void produce(float* sendbuf, const float* input) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    sendbuf[i] = input[i];
}
```

执行 `sendbuf[i] = input[i]` 的 CUDA 线程运行在 SM 上，所以这句话是在说：

```text
SM 上的 CUDA 线程
  -> 通过普通 global store
  -> 把 payload 写进本地 GPU memory
```

它不等于 SM 直接把字节发送到网络。正常情况下，后面的路径是：

```text
SM 写 sendbuf
WQE 记录 src=sendbuf, dst=远端 buffer, size=N
doorbell 通知 NIC
NIC DMA 读取 sendbuf
NIC 发包
```

因此：

| 对象 | 是什么 | 谁来写 |
|---|---|---|
| source payload | 真正被传输的业务数据 | SM 上的 CUDA kernel，或更早的生产 kernel |
| WQE | 告诉 NIC 如何执行一次发送 | GPU 或 CPU proxy |
| doorbell | 通知 NIC 有新 WQE | GPU 或 CPU proxy |
| 网络包 | NIC 对 payload 的传输封装 | NIC |

这里的 `source payload` 也取决于操作方向：

| 操作 | source 在哪里 | 谁读 source |
|---|---|---|
| 本地 RDMA WRITE | 本端 sendbuf | 本端 NIC |
| 本地 RDMA READ | 远端 buffer | 远端 NIC 读取后回传 |
| NVSHMEM put | 本端对称 buffer | 本端 NIC |
| NVSHMEM get | 远端对称 buffer | 远端 NIC 读取后回传 |

MoE dispatch 中，“源端 token buffer 已经是 payload，NIC 负责把它发到专家端”就属于第一种。源端 SM 只是把 token 放进本地 buffer，并没有亲手把 token 推过 PCIe 和网络。

---

## 2. 谁真正提交给 NIC

论文把“谁发起”与“谁提交”拆开，这是理解 CPU proxy 的关键。

| 名称 | 含义 | 典型动作 |
|---|---|---|
| GPU-initiated | GPU kernel 在语义上决定要通信 | 路由后在 kernel 内触发 put/get |
| GPU-submitted | GPU 自己构造 WQE 并写 NIC 的 UAR doorbell | SM 完成最后提交 |
| proxy-submitted | GPU 写 descriptor 或 doorbell value，CPU proxy 代替提交 | CPU 构造 WQE / 写 UAR |
| CPU-assisted doorbell | GPU 构造 WQE，CPU 只转发 doorbell | CPU 不做完整 WQE 构造 |

因此下面两句话可以同时成立：

```text
这是一次 GPU-initiated 通信。
它仍然是 proxy-submitted。
```

前者描述通信意图的来源，后者描述热路径的最后提交者。GPU-initiated 不等于完全没有 CPU。

对“GPU-submitted”而言，SM 自己完成：

```text
构造 WQE
推进 GPU memory 中的 producer index
执行 fence
通过 PCIe MMIO 写 NIC 的 UAR doorbell
```

对“proxy-submitted”而言，SM 可能只完成：

```text
构造逻辑 descriptor
把一个新值写进 mailbox / doorbell value
```

之后 CPU progress thread 看到它，再代 GPU 构造 WQE、更新所需状态并写 UAR。

---

## 3. Doorbell 有两层，不能混成一块内存

这一段是最容易产生“明明只更新了 GPU 自己，NIC 怎么知道”这个问题的地方。

### 3.1 Doorbell record：普通内存中的生产进度

`doorbell record` 常缩写为 `dbrec`。它是 GPU memory 或 host memory 中的一块普通状态记录，通常表示：

```text
发送队列的 producer index 已经推进到哪里
```

只写 dbrec，NIC 不会立刻被“叫醒”。dbrec 更像记录当前水位，不是门铃按钮。

### 3.2 UAR doorbell：NIC PCIe BAR 内的硬件寄存器

`UAR` 是 User Access Region，来自 NIC 的一段 PCIe BAR 映射。用户态进程通过写 UAR 区域通知 NIC：

```text
这个 SQ 有新工作
```

这个写操作是 MMIO，不是普通 HBM 写，也不是 DMA。

常规 mlx5 风格路径可以写成：

```text
1. SM 把 WQE 写进 GPU memory 里的 SQ
2. SM 更新 GPU memory 中的 dbrec / producer index
3. fence，保证 WQE 和 dbrec 对 NIC 可见
4. SM 通过 PCIe MMIO 写 NIC UAR doorbell
5. NIC 得知 SQ 有新工作
```

所以“GPU 更新自己的 doorbell”如果指的是 dbrec，那么它只是一个内存状态。NIC 真正收到通知，依赖后面单独的 UAR 写。

如果 UAR 反向映射不可用，可以让 CPU 执行第 4 步：

```text
GPU 写 WQE 和状态
CPU 读到新的 doorbell value
CPU 把它写进 NIC UAR
```

这就是 CPU-assisted doorbell。它不是完整 CPU proxy，因为 WQE 仍由 GPU 构造。

---

## 4. NIC 为什么还要 fetch WQE

`fetch WQE` 不是 GPU 从 NIC 取 WQE，而是：

```text
NIC DMA engine 从 SQ 所在内存读取 WQE
```

SQ 可以位于 host memory，也可以位于 NIC 能 DMA 访问的 GPU memory。写 UAR doorbell 只负责通知，并不要求 doorbell 本身携带完整 WQE。

典型顺序：

```text
SM 写 WQE 到 GPU memory SQ
SM 写 UAR doorbell
NIC 收 doorbell
NIC DMA 从 SQ fetch WQE
NIC 根据 WQE 执行 RDMA 操作
```

### 4.1 BlueFlame 为什么可以跳过 fetch

BlueFlame 把小型 WQE 直接写进 NIC 的 UAR / BlueFlame 区域：

```text
常规路径:
WQE 留在内存 -> doorbell -> NIC 再 DMA 读 WQE

BlueFlame:
SM 直接把最多 64B WQE 写进 NIC 提交区
NIC 不必再回 GPU 读这份小型 WQE
```

它省掉的是一次 NIC 到 GPU memory 的 WQE fetch，不是 RDMA payload 的 DMA。

---

## 5. GPU 自己有 DMA 吗

有。GPU 有 copy engine，可以执行：

- `cudaMemcpyAsync`
- GPU P2P copy
- 一些显式的 device-to-device 或 device-to-host 拷贝

但在 GPU-initiated RDMA 的典型数据路径里，真正从 GPU memory 读取 RDMA payload 的是 NIC 的 DMA engine：

```text
GPU compute engine / SM:
  产生并写入 payload

NIC DMA engine:
  根据 WQE 读取 payload
  把 payload 交给网络协议栈和 wire

GPU copy engine:
  主要不是这条 RDMA put 的数据搬运者
```

还要区分：

| 动作 | 机制 |
|---|---|
| SM 写 source payload | global store 到 GPU memory |
| SM 写 NIC UAR doorbell | PCIe MMIO write |
| NIC 读取 GPU SQ | NIC DMA read |
| NIC 读取 GPU payload | NIC DMA read，GPUDirect RDMA / DMA-BUF |
| 对端 NIC 写 GPU memory | NIC DMA write |

所以“GPU 有 DMA”不能推出“这次通信由 GPU copy engine 搬运 payload”。判断谁搬数据，要看具体操作和 WQE 的 data path。

---

## 6. WQE 与 CQE：fetch 不等于 complete

### 6.1 WQE 的基本组织

mlx5 WQE 常见以 16B segment、64B basic block 组织。论文讨论到的几种大小：

| WQE 类型 | 大小 | 说明 |
|---|---:|---|
| RC pointer WQE | 48B | 不内联 payload，NIC 后面单独读 source buffer |
| DC pointer WQE | 64B | 比 RC 多 16B AV，用来动态指定目标 DCT |
| RC inline 28B | 64B | 在一个 basic block 内携带 28B payload |
| RC inline 92B | 128B | 跨两个 basic block，携带 92B payload |

control segment 是 16B，四个 4B word，通常包含：

```text
dword 0: index / opcode
dword 1: QPN / size
dword 2: flags
dword 3: imm / unused
```

doorbell 常写 control segment 的前 8B，也就是前两个 dword，告诉 NIC 先到哪个 index 找哪种 WQE。

pointer WQE 体积小，但 NIC 还要再读 payload；inline WQE 体积大，却可能省掉一次 payload DMA。小消息优化经常就是在这两个成本之间选。

### 6.2 CQE 是完成，不是 fetch

`post CQE` 发生在 WQE 对应的 RDMA 操作完成之后，而不是 NIC 刚 fetch 到 WQE 之后。

```text
NIC fetch WQE
  -> NIC 读取 payload
  -> 数据发送并得到协议要求的完成状态
  -> NIC 写 CQE 到 CQ
  -> SM poll CQE
```

如果是 unsignaled WQE，它不会单独产生 CQE。后续一个 signaled WQE 可以间接证明此前的 unsignaled 工作已经完成，这也是集合通信里常见的 batching 优化。

CQE 与 fetch 混淆会造成两个错误推论：

1. 误以为 doorbell 一写完，发送就完成了。
2. 误以为看到 WQE 被取走，就可以安全覆盖 source buffer。

正确边界是：source buffer 至少要活到 NIC 已经完成对该 payload 的读取；若要求远程可见或操作整体完成，还要看 completion 语义。

---

## 7. UAR 为什么可能无法映射给 GPU

NIC 能 DMA 访问 GPU memory，和 GPU 能写 NIC BAR，是两个方向的权限。

```text
NIC -> GPU memory:
  GPUDirect RDMA / DMA-BUF 解决

GPU -> NIC UAR:
  需要把 NIC MMIO BAR page 映射到 GPU 虚拟地址空间
```

典型步骤可以抽象为：

```text
mlx5dv_devx_alloc_uar
  -> 得到 NIC UAR
cuMemHostRegister
  -> 把 BAR page 注册成可映射的 I/O memory
cuMemHostGetDevicePointer
  -> 得到 GPU kernel 可使用的 device pointer
```

把第三方 PCIe BAR 映射进 GPU 地址空间有安全和隔离风险，因此 NVIDIA kernel module 可能要求：

```text
PeerMappingOverride=1
```

如果禁止映射，直接 GPU-submitted doorbell 就不可用。此时有两种降级：

| 模式 | GPU 做什么 | CPU 做什么 |
|---|---|---|
| CPU-assisted doorbell | 构造 WQE | 只转发 doorbell value 到 UAR |
| 完整 CPU proxy | 构造 descriptor | 构造 WQE 并提交 |

因此 `cpu_proxy` 确实可以是“UAR 不能反向映射时”的兼容路径。不过它不是唯一原因，旧硬件、驱动限制、状态所有权和调试路径也会选择 CPU proxy。

---

## 8. DC、DCI、DCT、AV：为什么要减少持久连接状态

### 8.1 RC 的问题

RC QP 把连接状态固定在 QP context 中：

```text
本地 QP
  -> 固定绑定远端 QPN / endpoint
```

如果 rank A 要直接发往很多 rank，每个目标通常要有自己的 QP。系统越大，持久连接状态越容易膨胀。

### 8.2 DC 把目标选择从 QP context 搬到 WQE

DC 是 Dynamically Connected transport。它拆成两端：

| 名词 | 所在端 | 作用 |
|---|---|---|
| DCI | 发起端 NIC | Dynamically Connected Initiator，可复用的动态发送上下文 |
| DCT | 目标端 NIC | Dynamically Connected Target，可被多个 DCI 访问的目标端点 |
| AV | WQE 内部 | Address Vector，指定这次使用哪个 DCT / destination |

RC：

```text
QP context 已经告诉我发给谁
WQE 主要负责 op、raddr、size
```

DC：

```text
少量 DCI 是可复用发送上下文
每个 WQE 额外带 AV
AV 决定这次发给哪个 DCT
```

这解释了“每个 WQE 多带 AV，为什么能节省 QP”：

```text
持久状态: 为很多 destination 各保留一个 RC QP
动态状态: 少量 DCI + 每个 WQE 临时选一个 AV/DCT
```

代价是 WQE 从 48B 增至 64B，并且频繁切换 destination 时有 NIC 内部选择成本。收益是连接数量不再随 destination 数量线性爆炸。

### 8.3 AV 和 raddr 不是一个东西

```text
AV       = 发给哪个远端 DCT / endpoint
raddr    = 写到远端内存的哪个虚拟地址
size     = 写多少字节
```

AV 是路由目标，raddr 是目标内存地址。

---

## 9. Sharing QP 与 DC 不是同一个优化

两者经常一起出现，但解决的问题正交。

### 9.1 Sharing QP：多个线程共享一个 SQ

场景：

```text
一个 GPU 上的多个 warp / thread
都要向同一个 peer 发送
如果每个线程各建 QP，资源浪费且难管理
```

共享 QP 后，多个线程共同使用一条发送队列。难点从“建多少 QP”变成“怎样安全地向同一个 SQ 追加 WQE”。

常见发布协议：

```text
1. 每个线程原子 reserve 连续 slot
2. 各线程并行构造自己的 WQE
3. 按 reservation order 发布就绪状态
4. 某个线程或其他 lane 发现连续区间 ready
5. 只写一次 doorbell，发布整批 WQE
```

核心风险是慢 lane：

```text
线程 0 的 slot 已 ready
线程 1 的 slot 还没 ready
如果此时 doorbell 越过了线程 1
NIC 会发现空洞或读到未完成 WQE
```

因此不能简单让每个线程各自写 UAR。要么由 leader 在确认整批 ready 后发布，要么使用 cooperative warp publication，让同一批 slot 的发布动作协同完成。

### 9.2 DC：一个发送上下文服务多个 destination

DC 解决的是：

```text
一个 rank 要发给很多 peer
不希望为每个 peer 永久保留完整连接状态
```

QP sharing 解决的是：

```text
很多线程发给同一个 peer
不希望重复创建 QP
```

两者可以组合：

```text
少量 DCI
  -> 多个线程共享同一个 DCI / SQ
  -> WQE 内的 AV 决定每笔工作去哪个 DCT
```

所以 DC 和 QP sharing 都可能减少 QP 数量，但减少的是不同维度：

| 优化 | 减少什么 | 主要代价 |
|---|---|---|
| QP sharing | 同 destination 多线程造成的重复 QP | slot reservation、有序发布、慢 lane 协调 |
| DC | destination 数量增加造成的持久 QP / 连接状态 | WQE +16B AV、目标切换成本 |

---

## 10. 内存排序：为什么 payload 必须先于 flag 可见

普通 producer-consumer 写法：

```cpp
payload[0] = value;
flag = 1;
```

程序顺序是 payload 在前，flag 在后，但对另一个设备实际可观察到的顺序未必如此。两笔 store 可能经过不同 cache line、L2 slice、PCIe / NVLink 路径，也可能被编译器重排。

如果 peer 先看到 `flag == 1`，却读到旧 payload，就发生了发布顺序错误。

正确语义是：

```text
payload store
  -> release / fence
flag store

peer:
flag acquire
  -> payload read
```

GPU-initiated RDMA 还需要第二层排序：

```text
WQE 和 source payload stores
  -> 必须先对 NIC 可见
  -> 再写 UAR doorbell
```

否则 NIC 可能先收到 doorbell，再读到尚未完成的 WQE 或旧 payload。

论文中的成本对照：

| 排序方式 | 约成本 | 场景 |
|---|---:|---|
| GPU-scope release fence | 约 0.70 us | queue / consumer 都在 GPU scope 内 |
| system-scope fence | 约 2.62 us | 需要让 host、NIC 或其他 system-scope 设备观察到顺序 |

这也是为什么 mini-gda 不只是“少写几条指令”，而是要把 fence scope、WQE 可见性和 doorbell 发布共同缩短。

---

## 11. 论文里的性能数字怎样对上这条路径

不同设计的 issue、完成和 RTT 不是一回事：

| 实现 | issue 延迟 | put + completion | RTT | 备注 |
|---|---:|---:|---:|---|
| mini-gda | 0.70 us | 4.03 us | 6.85 us | GPU 直接提交 |
| GDAKI inline | 1.60 us | 6.02 us | 10.50 us | 更完整、更重的 GPU 提交路径 |
| NVSHMEM public | 5.31 us | 11.10 us | 21.50 us | 通用公共实现 |
| tuned mini-proxy | 0.13 us enqueue | 4.10 us | 5.89 us | 需要专用 CPU core |

这里要特别注意：

```text
mini-proxy 的 enqueue 更便宜
不等于完整通信路径一定更便宜
```

GPU 只把 descriptor 写进 mailbox，所以 enqueue 看起来很快；后面的 CPU polling、WQE 构造、doorbell 和网络完成仍然存在，而且需要占用一个专用 CPU core。

消息率方面，论文给出：

- mini-gda 峰值约 `260M msg/s`，需要 QP 并行和 doorbell batching。
- send-only NIC 在约 `242M msg/s` 时可扩展到 32,768 active QP。
- send + receive 时常降到约 `74M msg/s`。
- all-to-all 在约 3,000 connections 时吞吐损失约 59%。
- 即使 kernel 不执行通信，dormant communication code 也可能让有效吞吐下降 27% 到 37%。

这些数字说明提交路径不是唯一瓶颈。QP 状态、completion、连接拓扑和代码占用都会进入总成本。

---

## 12. 最容易混淆的七件事

1. **GPU-initiated 不等于 GPU-submitted。** 前者是通信意图来自 GPU，后者是 GPU 真正完成 doorbell 提交。
2. **doorbell record 不等于 UAR doorbell。** dbrec 是内存状态，UAR 是通知 NIC 的 MMIO 区域。
3. **写 UAR 不等于 WQE 已经被 NIC 取走。** UAR 只负责通知，NIC 之后才可能 fetch WQE。
4. **fetch WQE 不等于 post CQE。** fetch 是取命令，CQE 是操作完成后的回执。
5. **BlueFlame 不等于 DC。** BlueFlame 改的是 WQE 的提交方式，DC 改的是目标连接状态的组织方式。
6. **Sharing QP 不等于 DC。** 前者处理多线程共享同一 SQ，后者处理一个发送上下文服务多个 destination。
7. **GPU 有 copy engine 不等于 RDMA payload 由 copy engine 搬。** 典型 GPUDirect RDMA 路径仍是 NIC DMA 访问 GPU memory。

---

## 13. 自检问题

### Q1：SM 写了 source payload 后，数据是谁跨 PCIe 搬走的？

通常是 NIC DMA engine。SM 只把数据写入 GPU memory；NIC 根据 WQE 中的 source address、size 和 op，通过 GPUDirect RDMA 读取 payload。

### Q2：只更新 GPU memory 中的 dbrec，NIC 会知道有新工作吗？

不会单独知道。dbrec 只是生产进度记录，还需要写 NIC UAR doorbell。若 GPU 不能写 UAR，就由 CPU-assisted 或完整 proxy 代写。

### Q3：`fetch WQE` 和 `post CQE` 的先后关系是什么？

NIC 先 fetch WQE 获得发送命令，执行 RDMA 操作，协议完成后才 post CQE。fetch 成功只说明命令被取走，不说明 payload 已发送或完成。

### Q4：DC 为什么能把大量 RC QP 省下来？

RC 把目标固定在 QP context 中；DC 把目标放进每条 WQE 的 AV，让少量 DCI 动态选择大量 DCT。代价是每条 WQE 多 16B，并有目标切换成本。

### Q5：多个线程共享一个 QP 时为什么不能各写各的 doorbell？

共享 SQ 时，后面的 WQE 只有在前面连续区间都发布后才能被 NIC 安全消费。每个线程各自 doorbell 可能让 NIC 越过尚未 ready 的 slot，因此需要 slot reservation、按序发布或 cooperative publication，并把 doorbell batching 到最后。

---

## 14. 一页速记

```text
payload:
  SM 把真实数据写进本地 GPU memory
  网络包由 NIC 生成，不是 SM 直接生成

submission:
  GPU-submitted = GPU 构造 WQE 并写 UAR doorbell
  proxy-submitted = GPU 写 descriptor，CPU 真正提交
  CPU-assisted = WQE 由 GPU 构造，CPU 只转发 doorbell

doorbell:
  dbrec 是 GPU memory 中的 producer index
  UAR doorbell 是 NIC BAR 内的 MMIO 通知点
  写 UAR 后，NIC 才 fetch WQE

completion:
  fetch WQE != complete
  completion 后 NIC post CQE / counter
  unsignaled WQE 可批量回收

UAR mapping:
  NIC DMA 读写 GPU memory 不代表 GPU 能写 NIC BAR
  BAR 反向映射可能被安全策略禁止
  禁止时用 CPU-assisted 或 CPU proxy

resource model:
  RC: destination 固定在 QP context
  DC: destination 放进 WQE 的 AV
  DCI: 动态发送上下文
  DCT: 动态目标上下文
  QP sharing: 多线程共享一个 SQ
  DC 与 QP sharing 可以叠加，但不是同一个优化

ordering:
  payload -> release/fence -> flag
  WQE/payload -> fence -> UAR doorbell
  GPU-scope release 比 system-scope fence 便宜，但只能覆盖相应 scope
```

---

## 参考论文段落的阅读顺序

如果回看论文，建议按下面顺序对照本文：

```text
1. GPU-initiated vs GPU-submitted vs proxy-submitted
2. source payload 到 WQE / doorbell 的提交链
3. NIC WQE fetch、CQE 和 completion 边界
4. UAR mapping、CPU-assisted doorbell 与 PeerMappingOverride
5. WQE 格式：RC / DC / inline 大小
6. DCI / DCT / AV 与动态连接
7. QP sharing、slot reservation 和 cooperative publication
8. fence scope 与 mini-gda / mini-proxy 的实测数字
```
