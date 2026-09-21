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

## 八、当前进度

`chapter_tmem` 的知识点：

```text
[x] The TMEM Allocation Lifecycle
[ ] Which TMEM Lanes Each Warp Can Access
[ ] How tcgen05.ld and tcgen05.st Move Data
[ ] Shape and Repeat Factor
[ ] Packing and Unpacking 16-Bit Data
[ ] Waiting for Asynchronous Loads and Stores
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
```

下一知识点：

```text
chapter_tmem -> Which TMEM Lanes Each Warp Can Access
```
