# 10 TMEM Allocation Lifecycle

## 课程位置

课程：

- <https://github.com/mlc-ai/modern-gpu-programming-for-mlsys>

章节：

- `chapter_tmem`
- 中文标题：《Tensor Memory (TMEM)》

本篇已经覆盖：

```text
TMEM 的 128 Lane x 512 Column 与 32-bit cell
TMEM 按 Column 分配，每列包含全部 128 Lanes
tcgen05.alloc 的 warp-collective 语义
allocated_addr 与 TMEM buffer layout
连续 allocation 的 nCols 单调不增约束
relinquish_alloc_permit 与 tcgen05.dealloc
cta_group::2 的 CTA pair allocation 契约
```

## 目标

`chapter_tensor_cores` 已经说明 C、SFA、SFB 如何映射到 TMEM。
本篇往前一步，回答这些数据在进入 TMEM 前需要先解决什么问题：

```text
TMEM 从哪里来
谁负责申请
申请多少
地址如何变得对其他 warp 可见
什么时候必须释放
```

## 一、TMEM 的物理容量

```text
本次讲解位置
章节：chapter_tmem
小节：The TMEM Allocation Lifecycle
知识点：TMEM allocation 与 deallocation 生命周期
上次：tcgen05 指令之间的 scope / layout / completion 三层契约
下次：Which TMEM Lanes Each Warp Can Access
PTX：9.7.18.1.2 Tensor Memory Allocation；9.7.18.7.1 tcgen05.alloc / dealloc / relinquish_alloc_permit
```

TMEM 是一个二维地址空间：

```text
128 Lanes
每列最多 512 Columns
每个 (Lane, Column) cell = 32 bits = 4 bytes
```

完整容量是：

```text
128 * 512 * 4 bytes
= 262,144 bytes
= 256 KiB
```

`Lane` 是 TMEM 的地址坐标，不是 thread 的 lane ID。

分配只发生在 Column 维度：

```text
申请若干 Columns
每个 Column 自动包含全部 128 Lanes
```

例如申请 256 Columns：

```text
128 Lanes * 256 Columns * 4 bytes
= 131,072 bytes
= 128 KiB
```

一个 `float32` 的 `(128, 256)` TMEM buffer 正好占用 256 Columns。

## 二、warp-collective allocation

常见代码形式是：

```python
pool = T.SMEMPool()
tmem_addr = pool.alloc((1,), "uint32")
pool.commit()

if warp_id == 0:
    T.ptx.tcgen05.alloc(
        T.address_of(tmem_addr),
        n_cols=256,
        cta_group=1,
    )
```

这里有几个关键点：

```text
warp_id == 0 选择整个 warp 0
不是 lane_id == 0
tcgen05.alloc 是 warp-collective
同一个 warp 中的所有线程必须传入相同的 n_cols
```

`tcgen05.alloc` 是阻塞指令。如果暂时没有足够多的空闲列，它会等待，
直到所请求的 Column 数量可以分配。

分配成功后，指令把 TMEM 地址写到 `tmem_addr`。这个地址槽位于
shared memory：

```text
tcgen05.alloc
    -> 获得 TMEM base address
    -> 写入 SMEM 中的 tmem_addr
```

其他 warp 不能假设这个写入自动可见。读取 `tmem_addr` 前，需要用
合适的 fence 和 CTA synchronization 建立跨线程可见性。

## 三、把地址绑定到 TMEM buffer

拿到地址后，可以声明逻辑 buffer：

```python
tmem = T.decl_buffer(
    (128, 256),
    "float32",
    scope="tmem",
    allocated_addr=tmem_addr[0],
    layout=TileLayout(
        S[(128, 256) : (1@TLane, 1@TCol)]
    ),
)
```

`allocated_addr` 把 buffer 绑定到 allocation 返回的地址。
`layout` 描述逻辑坐标到 TMEM 坐标的映射。

这里使用最简单的 identity layout：

```text
logical m -> TLane
logical n -> TCol
```

所以：

```text
tmem[10, 130]
    -> allocation 内的 (TLane, TCol) = (10, 130)
```

如果换一个 TMEM layout，`m` 和 `n` 仍可写成逻辑坐标，但底层可能映射
到不同的 `TLane`、`TCol`，甚至加入 replication。

## 四、合法的 allocation size

课程重点使用的非 exclusive allocation size 是：

```text
32, 64, 128, 256, 512 Columns
```

PTX 还规定了连续 allocation 的顺序约束：

```text
后续 allocation 请求的 nCols
不能大于前面 allocation 请求的 nCols
```

例如：

```text
256 -> 128   合法
128 -> 256   非法
```

因此 kernel 在排列 allocation 时，需要提前确定最大需求，按不增的顺序
安排。不能先申请小块，再在程序后面扩大申请。

补充 PTX 细节：

```text
普通 allocation:
    nCols 是 32 到 512 范围内的 2 的幂

.exclusive allocation:
    nCols 是 32 的倍数
    sm_100f 最大 512
    sm_107f 最大 576
```

本课程当前以 32、64、128、256、512 这条主线为准。

## 五、释放生命周期

不再需要 TMEM 时，需要按顺序执行：

```python
if warp_id == 0:
    T.ptx.tcgen05.relinquish_alloc_permit(cta_group=1)
    T.ptx.tcgen05.dealloc(tmem_addr[0], n_cols=256, cta_group=1)
```

两个操作的含义不同：

```text
relinquish_alloc_permit:
    声明当前 CTA 不会再申请新的 TMEM allocation
    执行后不能再次调用 tcgen05.alloc

dealloc:
    释放之前申请的 Columns
```

每个通过 `tcgen05.alloc` 获得的 allocation 都必须在 kernel 退出前
显式释放。

释放前还必须确认所有异步操作已经完成：

```text
tcgen05.mma
tcgen05.ld
tcgen05.st
tcgen05.cp
其他访问该 TMEM 的操作
```

否则可能出现仍在写入 TMEM，但 allocation 已经被释放的时序错误。

## 六、cta_group::2 的 allocation

`cta_group::1` 只需要当前 CTA 中一个 warp 执行 allocation 和
deallocation。

`cta_group::2` 要求：

```text
CTA pair 两边各有一个 warp
执行相同的 tcgen05.alloc 或 tcgen05.dealloc
先到的一边等待 peer CTA
```

同一个 kernel 中，所有带 `cta_group` 的 tcgen05 指令必须使用相同的
值。不能先用 `cta_group::2` 申请，再用 `cta_group::1` 访问。

`cta_group::2` 也不会把两个 CTA 的 TMEM 合并成一个连续的
`128 x 1024` 地址空间。每个 CTA 仍然维护自己的本地 TMEM
allocation，只是 allocation 和部分访问操作需要成对协调。

## 七、常见错误定位

| 现象 | 检查点 |
|---|---|
| 另一个 warp 读到无效 `tmem_addr` | allocation 完成后是否建立 fence 与 CTA synchronization |
| 同一个 warp 内线程传入不同 `nCols` | `tcgen05.alloc` 必须 warp-collective |
| 先申请 128，再申请 256 报错或行为非法 | 后续 allocation 的列数不能增加 |
| kernel 退出时资源泄漏 | 每个 allocation 是否都执行了匹配的 dealloc |
| dealloc 后仍发生写操作 | MMA、load、store、copy 是否都已完成 |
| CTA pair 一边卡住 | 两个 peer CTA 是否都执行相同的 collective allocation |
| 混用不同 `cta_group` | 当前 kernel 内所有 tcgen05 指令的 qualifier 是否一致 |

一句话记忆：

```text
alloc 决定从哪里开始
layout 决定逻辑坐标怎么落进去
dealloc 决定什么时候归还
```

## 八、Which TMEM Lanes Each Warp Can Access

```text
本次讲解位置
章节：chapter_tmem
小节：Which TMEM Lanes Each Warp Can Access
知识点：warpgroup 内四个 warp 的固定 TLane 访问窗口
上次：TMEM allocation 与 deallocation 生命周期
下次：How tcgen05.ld and tcgen05.st Move Data
PTX：9.7.18.8 Tensor Memory and Register Load/Store Instructions；9.7.18.8.1 Access restrictions
```

先区分两件容易混淆的事：

```text
allocation:
    整个 CTA 当前拥有哪些 TMEM Columns

lane access restriction:
    CTA 内的某个 warp 能读取或写入哪些 TMEM Lanes
```

TMEM 在资源归属上属于 CTA。`tcgen05.alloc` 申请的是 Columns，而且一列
会包含全部 128 Lanes。可是 `tcgen05.ld` 和 `tcgen05.st` 并不会让 CTA
里的任意 warp 访问全部 128 个 Lanes。

一个 warpgroup 中的四个 warp 各自拥有固定的 32-Lane 访问窗口：

| warpgroup 内的 warp ID | 可访问的 TMEM Lanes |
|---|---|
| 0 | 0-31 |
| 1 | 32-63 |
| 2 | 64-95 |
| 3 | 96-127 |

如果令：

```text
warp_in_group = warp_id % 4
```

那么这个 warp 的窗口是：

```text
first_lane = 32 * warp_in_group
last_lane  = first_lane + 31
```

也可以直接判断某个 TMEM Lane 是否属于该 warp：

```text
lane // 32 == warp_in_group
```

这里的 `warp_in_group` 是 warpgroup 内的相对编号，不是整个 thread block
的绝对 `warp_id`。例如绝对 `warp_id = 5`：

```text
warpgroup       = 5 // 4 = 1
warp_in_group   = 5 % 4 = 1
可访问 Lanes    = 32-63
```

四个 warp 的区别只在 Lane 窗口；它们都可以访问该 allocation 内的所有
Columns。也就是说：

```text
Lane 维:
    warp 0 -> 0-31
    warp 1 -> 32-63
    warp 2 -> 64-95
    warp 3 -> 96-127

Column 维:
    四个 warp 都覆盖当前 allocation 的全部 Columns
```

这里的“都能访问所有 Columns”仍然受 allocation 边界约束。某个 warp
可以访问分配给自己窗口的 Lanes，但不能借此访问 CTA 尚未申请或已经
释放的 Columns。

以一个 identity layout 为例：

```python
TileLayout(
    S[(128, 256) : (1@TLane, 1@TCol)]
)
```

映射关系是：

```text
logical m -> TLane
logical n -> TCol
```

于是：

```text
C[10, 130]
    -> TLane = 10
    -> TCol  = 130
    -> 只有 warp 0 能通过 tcgen05.ld/st 访问

C[74, 130]
    -> TLane = 74
    -> TCol  = 130
    -> 只有 warp 2 能通过 tcgen05.ld/st 访问

C[110, 7]
    -> TLane = 110
    -> TCol  = 7
    -> 只有 warp 3 能通过 tcgen05.ld/st 访问
```

所以，当 accumulator 跨越全部 128 Lanes 时，需要四个 warp 协作读取：

| warp | 负责读取的逻辑行 |
|---|---|
| 0 | `m = 0..31` |
| 1 | `m = 32..63` |
| 2 | `m = 64..95` |
| 3 | `m = 96..127` |

每个 warp 读取自己的 32-Lane 窗口。四个 warp 合起来，才形成“warpgroup
读取了一整块 TMEM accumulator”的结果。

这也能消除一个常见误解：

```text
tcgen05.mma 能写满整个 TMEM tile
!=
任意单个 warp 能读取整个 TMEM tile
```

`tcgen05.mma` 是 warp-level collective 指令，结果可以覆盖很宽的
`TLane x TCol` 区域；但后续用 `tcgen05.ld` 把结果取回寄存器时，仍然
必须遵守上面按 warp 划分的 Lane 窗口。

最后要区分两类 lane：

```text
thread lane ID / laneid:
    warp 内 32 个线程的编号

TMEM TLane:
    Tensor Memory 的 128 个逻辑 Lane 坐标
```

本小节的规则决定“哪个 warp 可以碰哪些 TMEM Lanes”。至于 warp 内每个
线程具体拿到哪些寄存器元素，则由 `tcgen05.ld/st` 的 shape、num 和寄存器
分布规则决定，留到下一个知识点。

一句话记忆：

```text
Column 看 allocation
Lane 看 warp 在 warpgroup 中的位置
```

## 九、TMEM 与寄存器之间的数据通路（合并课）

```text
本次讲解位置
章节：chapter_tmem
小节：How tcgen05.ld and tcgen05.st Move Data；Shape and Repeat Factor；Packing and Unpacking 16-Bit Data；Waiting for Asynchronous Loads and Stores
知识点：TMEM 到寄存器的搬运、shape/num 数据量、16-bit pack/unpack 与异步等待
上次：warpgroup 内四个 warp 的固定 32-Lane TMEM 访问窗口
下次：chapter_async_barriers -> mbarrier 与 phase 生命周期
PTX：9.7.18.8 Tensor Memory and Register Load/Store Instructions；9.7.18.8.2 Packing and Unpacking；9.7.18.8.3 tcgen05.ld；9.7.18.8.4 tcgen05.st；9.7.18.8.5 tcgen05.wait
```

这一课把四个原本分散的小点合成一条数据通路：

```text
访问哪个 TMEM 窗口
    -> 一条指令搬运多少数据
    -> 数据怎样进入 warp 内各线程的寄存器
    -> 16-bit 数据是否需要 pack/unpack
    -> 什么时候可以安全读取或覆盖这些数据
```

### 一、方向与 warp-collective 语义

两条指令的方向正好相反：

```text
tcgen05.ld:
    TMEM -> registers

tcgen05.st:
    registers -> TMEM
```

它们都是 warp-collective 指令：

```text
同一个 warp 中所有 32 个线程
    执行同一条 tcgen05.ld/st

所有线程提供同一个 [taddr]
    [taddr] 是整个 warp 操作的 base address
```

如果 warp 内不同线程传入不同的 `taddr`，行为没有定义。硬件根据每条
线程的 `%laneid`，把 TMEM 数据分发到各线程自己的寄存器，或者把各线程
的寄存器写回对应的 TMEM cells。

典型形式：

```text
tcgen05.ld.sync.aligned.16x128b.x4.b32
    {r0, r1, r2, r3, r4, r5, r6, r7}, [taddr];
```

这里：

```text
一个 warp 共同执行一次 ld
每个线程得到自己的 r0-r7
[taddr] 对 warp 内所有线程相同
```

`tcgen05.st` 只是反向：

```text
tcgen05.st.sync.aligned.16x64b.x4.b32
    [taddr], {r0, r1, r2, r3};
```

上一课的 Lane 窗口限制仍然生效。一个 warp 只能在自己的 32-Lane 窗口
内执行这些访问。

### 二、shape 与 num 决定搬运量

`.shape` 不是 MMA 的 `M x N x K`。它是 TMEM 与寄存器之间的数据搬运
形状，主要回答：

```text
有多少个 TMEM Lanes 参与
每个 Lane 的基础数据宽度是多少
```

`.num` 是基础形状的重复次数：

```text
.x1  -> 重复 1 次
.x2  -> 重复 2 次
.x4  -> 重复 4 次
.x8  -> 重复 8 次
```

统一用 32-bit cell 计数：

```text
cells_moved
    = shape_lanes
    * shape_bits_per_lane / 32
    * num

registers_per_thread
    = cells_moved / 32
```

以：

```text
tcgen05.ld.sync.aligned.16x128b.x4.b32
```

为例：

```text
16 lanes
128 bits / lane = 4 个 32-bit cells
x4 repetition

cells_moved = 16 * 4 * 4 = 256
registers_per_thread = 256 / 32 = 8
```

这正好对应寄存器列表 `{r0, ..., r7}`。

| 指令形式 | 每 Lane 数据 | 总 cells | 每线程寄存器 |
|---|---:|---:|---:|
| `.16x128b.x1` | 128 bit | 64 | 2 |
| `.16x128b.x2` | 256 bit | 128 | 4 |
| `.16x128b.x4` | 512 bit | 256 | 8 |
| `.16x128b.x8` | 1024 bit | 512 | 16 |

也可以用更常见的 `.32x32b` 做快速心算：

```text
.32x32b.x1:
    32 lanes * 1 cell * 1 = 32 cells
    每线程 1 个 32-bit register

.32x32b.x4:
    32 lanes * 1 cell * 4 = 128 cells
    每线程 4 个 32-bit registers
```

`.shape` 和 `.num` 决定搬了**多少**数据，但不必单独决定每个寄存器
对应哪个逻辑矩阵元素。最后的 fragment mapping 还取决于具体指令形状
和寄存器 fragment 定义。

可以先用一条简单规则检查：

```text
指令写的寄存器数量
    必须等于 shape/num 计算出的每线程寄存器数量
```

### 三、pack 与 unpack

TMEM cell 和 `tcgen05.ld/st` 的寄存器操作数都以 32 bit 为基本单位，
但实际数据可能是 16-bit：

```text
tcgen05.ld + .pack::16b:
    两个相邻 TMEM Column 中的 16-bit 数据
    -> 一个 32-bit register

tcgen05.st + .unpack::16b:
    一个 32-bit register
    -> 两个相邻 TMEM Column 中的 16-bit 数据
```

可以把它理解成：

```text
TMEM 中:
    Column 2k     = 16-bit value low
    Column 2k + 1 = 16-bit value high

ld.pack:
    register = (high, low)

st.unpack:
    (high, low) -> Column 2k, Column 2k + 1
```

pack/unpack 只改变搬运时的数据组织方式：

```text
不改变 TMEM allocation
不改变一个 Column 仍然包含 128 Lanes
不改变 Column 的分配单位
```

它解决的是“16-bit 逻辑数据怎样放进 32-bit 搬运单元”，不是重新设计
TMEM 地址空间。

### 四、异步完成与 wait

`tcgen05.ld` 和 `tcgen05.st` 都是异步指令。发出指令不等于数据已经可用
或已经写完。

加载后的正确顺序：

```text
tcgen05.ld ...
tcgen05.wait::ld.sync.aligned;
使用目标寄存器
```

存储后的正确顺序：

```text
tcgen05.st ...
tcgen05.wait::st.sync.aligned;
再依赖这次 TMEM 写入已经完成
```

两个 wait 分别覆盖当前线程之前发出的所有相关操作：

```text
tcgen05.wait::ld:
    等待当前线程此前所有 tcgen05.ld

tcgen05.wait::st:
    等待当前线程此前所有 tcgen05.st
```

但 wait 只解决当前线程自己的异步完成顺序。如果另一个线程或 warp
接下来要消费这份数据，还需要：

```text
合适的 tcgen05.fence
CTA / warpgroup 级别的线程同步
发布者与消费者之间的内存顺序
```

所以不能把：

```text
tcgen05.wait::st
```

误认为：

```text
另一个 warp 已经能看到新数据
```

### 五、完整检查顺序

读一段 TMEM kernel 时，按下面四步检查最稳定：

```text
1. allocation:
    分配了多少 Columns，何时释放

2. lane window:
    当前 warp 能访问哪 32 个 Lanes

3. movement:
    shape/num 实际产生多少个寄存器
    是否需要 pack/unpack

4. completion:
    ld 后是否 wait::ld
    st 后是否 wait::st
    跨线程消费是否补齐 fence + synchronization
```

一句话记忆：

```text
taddr 选起点
shape 选基础搬运块
num 决定重复几份
laneid 决定线程拿到哪些寄存器
wait 决定什么时候可以使用
```

## 十、异步等待：tcgen05.wait::ld 与 tcgen05.wait::st

```text
本次讲解位置
章节：chapter_tmem
小节：Waiting for Asynchronous Loads and Stores
知识点：tcgen05.ld/st 的异步完成边界
上次：tcgen05.ld/st 的 shape/num 与 16-bit pack/unpack
下次：chapter_async_barriers -> mbarrier 与 phase 生命周期
PTX：9.7.18.8.3 tcgen05.ld；9.7.18.8.4 tcgen05.st；9.7.18.8.5 tcgen05.wait
```

这一节只解决一个问题：

```text
tcgen05.ld/st 已经把指令发出去了
什么时候才能认为数据真的搬完了
```

### 一、发出指令不等于操作完成

普通 shared memory load 通常可以按同步 load 来理解：

```text
ld.shared r0, [addr]
下一条指令使用 r0
```

`tcgen05.ld` 不是这种模型。它是异步 collective load：

```text
tcgen05.ld ... {r0, ...}, [taddr]
```

这条指令只是告诉硬件：

```text
从 [taddr] 开始的 TMEM 区域
搬一部分数据到当前 warp 各线程的寄存器
```

指令发射完成后，数据搬运仍可能还在后台进行。因此必须有明确边界，
告诉硬件和编译器：

```text
之前的 tcgen05.ld 已经完成
现在可以安全使用目标寄存器
```

这个边界就是：

```text
tcgen05.wait::ld.sync.aligned;
```

存储方向同理：

```text
tcgen05.st ... [taddr], {r0, ...}
```

它表示把寄存器数据写向 TMEM。即使指令已经发射，写入也可能仍在
后台进行。后续 MMA 或其他 tcgen05 操作如果马上复用 `[taddr]`，就可能
与前一次 store 重叠。此时需要：

```text
tcgen05.wait::st.sync.aligned;
```

一句话区分：

```text
tcgen05.ld/st:
    发起异步数据搬运

tcgen05.wait::ld/st:
    等待此前同类异步搬运完成
```

### 二、wait::ld 保证什么

典型顺序：

```text
tcgen05.ld.sync.aligned.32x32b.x2.b32
    {r0, r1}, [taddr];

tcgen05.wait::ld.sync.aligned;

// 这里再使用 r0、r1
```

`tcgen05.wait::ld` 的 PTX 语义是：

```text
执行线程阻塞
直到该线程此前发出的所有 tcgen05.ld 都已完成
```

它覆盖的是“此前发出的 load”，不是未来准备发出的 load：

```text
ld A
ld B
wait::ld       // A 和 B 都被等待
ld C
wait::ld       // 等待 C
```

因此可以把一次 wait 理解成一次 drain，也就是把此前积压的 load
全部排空。wait 之后如果又发出了新的 load，就还需要新的 wait。

它不会等待 `tcgen05.st`：

```text
wait::ld  -> 只管此前 tcgen05.ld
wait::st  -> 只管此前 tcgen05.st
```

具体到坐标例子。假设 identity layout：

```text
C[m, n] -> (TLane = m, TCol = n)
```

某个 warp 执行：

```text
tcgen05.ld.sync.aligned.32x32b.x1.b32
    {r0}, [taddr + column_offset];

tcgen05.wait::ld.sync.aligned;

float value = r0;
```

`wait::ld` 返回后，当前线程才把 `r0` 当作这次 load 已经完成的
结果来使用。

### 三、wait::st 保证什么

假设当前 warp 把寄存器中的 `P` 写入 TMEM，随后 MMA 要读取这块
`P`：

```text
tcgen05.st.sync.aligned.16x128b.x1.b32
    [p_taddr], {r0, r1};

tcgen05.wait::st.sync.aligned;

// 在这个线程的后续 tcgen05 顺序中
// 可以依赖前面的 P store 已经完成
```

`tcgen05.wait::st` 的 PTX 语义是：

```text
执行线程阻塞
直到该线程此前发出的所有 tcgen05.st 都已完成
```

如果没有这个边界，后面对同一个 TMEM 区域的 MMA 或其他写入可能
与前一次 `tcgen05.st` 竞争。典型后果是：

```text
MMA 读到了旧 P
新的 TMEM write 覆盖了尚未落盘的 P
结果依赖调度时序，偶发错误
```

所以 `wait::st` 的核心用途是：

```text
在复用或消费刚写入的 TMEM 区域前
先确认之前的 store 已经完成
```

### 四、为什么是 wait，而不是 mbarrier

`tcgen05.ld/st` 使用 `tcgen05.wait::ld/st` 完成本地等待。

这与另外两类异步机制不同：

| 异步操作 | 主要完成机制 |
|---|---|
| `tcgen05.ld` / `tcgen05.st` | `tcgen05.wait::ld` / `tcgen05.wait::st` |
| `tcgen05.mma` / `tcgen05.cp` | `tcgen05.commit` 到 mbarrier |
| TMA load | mbarrier 的 transaction completion |
| TMA store | `cp.async.bulk.commit_group` / `wait_group` |

这里的区别来自指令设计：

```text
tcgen05.ld/st 是 warp 在 TMEM 与自己的寄存器之间搬运数据
    -> 直接使用 wait 排空当前线程的同类操作

tcgen05.mma/cp 是更大的异步引擎操作
    -> 使用 commit 把完成事件发布到 mbarrier
```

不要看到一个异步操作就默认它一定用 mbarrier。先看指令的完成协议。

### 五、wait 不等于跨线程可见

这是最容易混淆，也最重要的一点。

`tcgen05.wait::st` 只说明：

```text
执行该 wait 的线程
此前发出的 tcgen05.st
已经完成
```

它不自动说明：

```text
另一个 warp 已经观察到这次写入
另一个线程可以跳过同步直接消费
```

跨线程 handoff 还需要组成一条顺序链：

```text
生产者线程:
    发出 tcgen05 操作
    wait 本地完成
    tcgen05.fence::before_thread_sync

线程同步:
    barrier / mbarrier / 其他执行顺序建立机制

消费者线程:
    tcgen05.fence::after_thread_sync
    发出后续 tcgen05 操作
```

可以把三层边界记成：

```text
issue:
    指令已经发出

wait:
    当前线程此前同类操作已经完成

fence + thread synchronization:
    把完成顺序扩展到另一个线程或 warp
```

因此：

```text
tcgen05.wait::st
!=
所有线程都已经看到新的 TMEM 内容
```

`wait::ld` 也有相同边界。它保证当前线程的 load 已完成，但如果另一个
线程依赖这次 load 的结果，仍然需要合适的 fence 和线程同步。

### 六、warp-collective 的 .sync.aligned

`tcgen05.ld/st` 的 `.sync.aligned` 要求整个 warp 执行同一条指令。
`tcgen05.wait::ld/st` 同样是：

```text
tcgen05.wait::ld.sync.aligned;
tcgen05.wait::st.sync.aligned;
```

它们的等待语义可以拆成两层：

```text
每个线程先等待自己此前同类 tcgen05 操作完成
然后 warp 内所有线程在 wait 指令处同步
```

所以 wait 返回时，不只是某一个 lane 完成了自己的 load 或 store，
整个 warp 也已经共同越过了这个完成边界。

这也是为什么不能在条件分支里只让部分 lane 执行 wait：

```text
if (lane_id < 16)
    tcgen05.wait::ld.sync.aligned;   // 错误用法
```

`tcgen05.ld/st`、`tcgen05.wait` 都是 `.sync.aligned` warp-collective
指令，执行条件必须对整个 warp 一致。

### 七、和课程代码的对应关系

课程里的 TIRx 写法：

```python
Tx.wg.copy_async(reg_wg, tmem[:, :BLK_N])
T.ptx.tcgen05.wait.ld()
```

对应：

```text
Tx.wg.copy_async -> tcgen05.ld
tcgen05.wait.ld  -> tcgen05.wait::ld.sync.aligned
```

反向写回：

```python
Tx.wg.copy_async(tmem_as_f16, reg)
T.ptx.tcgen05.wait.st()
```

对应：

```text
Tx.wg.copy_async -> tcgen05.st
tcgen05.wait.st  -> tcgen05.wait::st.sync.aligned
```

看到 `copy_async` 时，不要把它当成普通同步 copy。先找紧随其后的
`wait.ld()` 或 `wait.st()`，再检查跨 warp 的 fence 和 barrier。

### 八、完整判断流程

遇到一段 TMEM 代码时，按下面顺序检查：

```text
1. 这是 tcgen05.ld 还是 tcgen05.st

2. 谁会在之后使用结果，或者复用同一 TMEM 区域

3. 如果是同一个执行线程：
       ld -> wait::ld
       st -> wait::st

4. 如果是另一个 warp 或线程：
       是否补齐 tcgen05.fence 和跨线程同步

5. wait 之后是否又发起了新的同类操作
       如果有，是否需要新的 wait
```

一句话记忆：

```text
tcgen05.ld/st 负责发射搬运
tcgen05.wait::ld/st 负责排空本地同类搬运
tcgen05.fence + 线程同步负责扩展到其他线程
```

## 十一、当前进度

`chapter_tmem` 的知识点：

```text
[x] The TMEM Allocation Lifecycle
[x] Which TMEM Lanes Each Warp Can Access
[x] How tcgen05.ld and tcgen05.st Move Data
[x] Shape and Repeat Factor
[x] Packing and Unpacking 16-Bit Data
[x] Waiting for Asynchronous Loads and Stores
```

已经覆盖：

```text
TMEM 128 Lanes x 512 Columns
32-bit TMEM cell
256 KiB 总容量
按 Column allocation
tcgen05.alloc 的 warp-collective 语义
tmem_addr 的 SMEM 可见性
allocated_addr 与 layout
allocation size 与单调不增约束
relinquish_alloc_permit
tcgen05.dealloc
cta_group::2 allocation 契约
warpgroup 内四个 warp 的固定 32-Lane TMEM 访问窗口
CTA allocation 边界与 warp Lane 访问限制的区别
tcgen05.ld/st 的 warp-collective 数据通路
shape 与 num 的 data volume / register count 计算
16-bit pack/unpack 语义
tcgen05.ld/st 的异步 issue / completion 边界
tcgen05.wait::ld/st 的 per-thread 完成语义
tcgen05.wait、tcgen05.fence 与跨线程线程同步的区别
chapter_tmem 完成
```

下一知识点：

```text
chapter_async_barriers -> mbarrier 与 phase 生命周期
```
