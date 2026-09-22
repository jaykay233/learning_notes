# TIRx Layout API：`S[...]`、`R[...]`、offset 与命名轴

这篇笔记开始学习 `chapter_tirx_layout_api`。目前完整讲清前七个
知识点：`S[...] + R[...] + offset` 的语义与计算、命名轴的坐标空间，
`apply()` 的三种输入形式，以及 `2 x 128 x 112` accumulator 的
`TLane / TCol` 映射、scale-factor atom 沿 `TLane` 的复制，以及
`tcgen05.mma` D/F datapath 的 row 到 Lane 映射，最后通过
`tcgen05_atom_layout` 把 TMEM fragment 分布到 warpgroup 的线程寄存器。
先看：

```text
S[...] + R[...] + offset
```

目标不是背 API，而是能从逻辑坐标算出基础物理坐标，并准确区分
base mapping、physical replicas 和固定平移。

## 本次讲解位置

```text
本次讲解位置
章节：chapter_tirx_layout_api
小节：TileLayout
知识点：S[...] shard、R[...] replica 与固定 offset 的组合
上次：chapter_intro_tirx 完成
下次：chapter_tirx_layout_api -> 命名轴：laneid / warpid / m / TLane / TCol
PTX：无直接 PTX 指令；结果会映射到 thread、warp、register 或 TMEM 坐标
```

上一章的 kernel 已经使用过：

```python
TileLayout(S[(128, 512) : (1@TLane, 1@TCol)])
TileLayout(S[(128, BLK_N) : (1@tid_in_wg, 1)])
```

这一章要回答：这些记号到底描述了什么，`apply()` 又如何计算它们。

### 本章知识点清单

```text
[x] TileLayout 的 S[...]、R[...] 与 offset
[x] 命名轴：laneid / warpid / m / TLane / TCol
[x] apply() 的三种输入形式与 flatten / decompose
[x] TMEM accumulator layout 示例
[x] scale-factor layout 中的 replication
[x] tmem_datapath_layout：D/F datapath
[x] tcgen05_atom_layout 的 instr_shape / tensor_shape / dtype
[ ] wg_local_layout
[ ] ComposeLayout 与 shared-memory swizzle
```

本篇目前完成前七项。后续知识点会在同一章新的小节中继续展开。

## 一、这个 API 要解决什么问题

普通 Shape-Stride 只告诉我们：

```text
某一行、某一列的数据在线性存储中的地址是什么
```

GPU kernel 还需要描述更多坐标：

```text
数据属于哪个 lane？
数据属于哪个 warp？
数据在 register fragment 的哪个局部 slot？
数据在 TMEM 的哪个 Lane 和 Col？
同一份数据是否要复制到多个物理位置？
整个 tile 是否需要整体平移？
```

因此，一个 layout item 不只包含 stride，还要包含 axis：

```text
(extent, stride, axis)
```

例如：

```text
4 @ laneid
```

表示这一维每前进一步，沿 `laneid` 轴移动 4。

### 心智模型

先把 `TileLayout` 看成一个产生物理坐标集合的函数：

```text
逻辑坐标 x
-> S[...] 产生基础坐标 D(x)
-> R[...] 枚举额外副本 offsets
-> offset 平移所有结果
-> 得到物理坐标集合
```

完整公式是：

```text
L(x) = { D(x) + r + O | r in R }
```

其中：

```text
D(x): S[...] 给出的基础位置
r:    R[...] 枚举出的副本 offset
O:    固定 offset
```

没有 `R[...]` 时，可以把副本集合看成只包含一个零 offset：

```text
L(x) = { D(x) + O }
```

这里有一个非常重要的 API 边界：

```text
layout.apply(x) 只返回 D(x) + O
layout.apply(x) 不枚举 R[...]
```

Replica 信息保存在：

```python
layout.replica
```

它由实际使用这个 layout 的 tile operation 处理。`apply()` 返回的是
base coordinate，不是完整的 physical coordinate set。

## 二、三个组成部分

### 2.1 `S[...]`：Shard

```python
S[shape : strides]
```

`S[...]` 定义逻辑 tile 的基础映射：

```text
逻辑索引如何拆成多个 iter
每个 iter 的 stride 是多少
每个 stride 作用在哪个命名轴上
```

它是逻辑索引相关的。修改 `S[...]` 会改变：

```text
哪个物理坐标接收哪个逻辑元素
```

### 2.2 `R[...]`：Replica

```python
R[replica_shape : replica_strides]
```

`R[...]` 描述同一逻辑元素的额外物理副本。它不增加新的逻辑元素，
也不改变 `apply()` 的基础坐标。

例如：

```python
R[2 : 4@warpid]
```

枚举出两个副本 offset：

```text
r0 = 0 @ warpid
r1 = 4 @ warpid
```

它表示同一个逻辑元素还要在沿 `warpid` 轴相差 4 的位置出现一次。

### 2.3 Offset：固定平移

```python
5@warpid
```

表示整个 layout 沿 `warpid` 轴平移 5。Offset 不产生副本，它只会让
`apply()` 的结果整体移动。

### 2.4 三者对比

| 组成 | 是否依赖逻辑索引 | 是否增加物理副本 | 是否改变 `apply()` 结果 |
|---|---|---|---|
| `S[...]` | 是 | 否 | 是 |
| `R[...]` | 否 | 是 | 否 |
| offset | 否 | 否 | 是，整体平移 |

可以压缩成：

```text
S: 把逻辑元素放到哪里
R: 同一元素还要复制到哪里
offset: 所有位置整体移动多少
```

## 三、命名轴的单位

当前示例使用以下 axes：

| Axis | 含义 |
|---|---|
| `laneid` | thread 在 warp 中的 lane ID |
| `warpid` | CTA 中的 warp ID |
| `m` | 默认线性物理轴，具体含义由 buffer scope 决定 |

`m` 的语义依赖 buffer scope：

```text
全局或 shared buffer -> 线性存储地址或局部地址
register-backed local buffer -> 当前 thread 的局部线性位置
```

轴名本身属于 layout 语义：

```text
laneid=5 和 m=5 不是同一个物理位置
warpid=1 和 laneid=1 也不是同一个坐标
```

同一轴可以出现多次。多个 iter 对同一轴产生的贡献必须相加，而不是
覆盖。本节的示例会具体展示这一点。

对于 `m`、`TCol` 这类存储轴，stride 以 buffer element 为单位。
只有当元素宽度是 32 bits 时，沿 `TCol` 前进一个元素才恰好对应一个
32-bit hardware TMEM Col。8-bit 或 16-bit buffer 可能存在多个
元素打包进同一个 hardware Col 的情况。

## 四、完整可运行代码

下面的脚本可以在没有 GPU 的 macOS 环境运行。它只构造并查询
`TileLayout`，不 launch CUDA kernel。

文件名：

```text
tirx_layout_s_r_offset.py
```

完整代码：

```python
from tvm.tirx.layout import R, S, TileLayout, laneid, warpid


def main():
    layout = TileLayout(
        S[(8, 2, 4, 2) : (4 @ laneid, 1 @ warpid, 1 @ laneid, 1)]
        + R[2 : 4 @ warpid]
        + 5 @ warpid
    )

    base = layout.apply(1, 3, shape=[8, 16])

    print("layout:", layout)
    print("base coordinate dict:", base)
    print("laneid:", base["laneid"])
    print("warpid:", base["warpid"])
    print("m:", base["m"])

    replica_warps = [
        base["warpid"] + replica_index * 4
        for replica_index in range(2)
    ]
    print("replica warps:", replica_warps)

    print("shard:", layout.shard)
    print("replica:", layout.replica)
    print("offset:", layout.offset)


if __name__ == "__main__":
    main()
```

## 五、逐行映射

### 5.1 构造 shard

```python
S[(8, 2, 4, 2) : (4 @ laneid, 1 @ warpid, 1 @ laneid, 1)]
```

拆成四个 iters：

| Iter | Extent | Stride | Axis |
|---:|---:|---:|---|
| 0 | 8 | 4 | `laneid` |
| 1 | 2 | 1 | `warpid` |
| 2 | 4 | 1 | `laneid` |
| 3 | 2 | 1 | `m` |

最后一个 stride 没有显式写出 axis，因此默认使用 `m`。

Shard 能表示的逻辑元素总数是：

```text
8 x 2 x 4 x 2 = 128
```

调用端指定的逻辑 shape 是：

```text
8 x 16 = 128
```

两者元素数量相同，但维度拆分方式不同。所以 `apply()` 先按照
`[8, 16]` flatten 逻辑坐标，再按照 shard extents
`[8, 2, 4, 2]` 拆分。

### 5.2 加入 replica

```python
+ R[2 : 4 @ warpid]
```

它表示 Replica iter：

```text
extent = 2
stride = 4
axis   = warpid
```

枚举出的副本 offset 是：

```text
0 @ warpid
4 @ warpid
```

Replica 不改变逻辑 shape，也不改变 `apply()`。它只告诉后续 tile
operation：同一个逻辑元素还有一份位于沿 `warpid` 方向偏移 4 的位置。

### 5.3 加入固定 offset

```python
+ 5 @ warpid
```

把 shard 和 replica 的 `warpid` 坐标整体加 5。

因此示例元素的最终 physical locations 是：

```text
base + 0 = warpid 5
base + 4 = warpid 9
```

### 5.4 `layout.shard`、`layout.replica` 和 `layout.offset`

```python
print("shard:", layout.shard)
print("replica:", layout.replica)
print("offset:", layout.offset)
```

它们分别暴露：

| 属性 | 类型 | 含义 |
|---|---|---|
| `layout.shard` | tuple of `Iter` | `S[...]` 的四个基础 iters |
| `layout.replica` | tuple of `Iter` | `R[...]` 的副本 iters |
| `layout.offset` | axis 到整数的字典 | 固定平移 |

## 六、完整数值执行追踪

查询元素：

```text
logical coordinate = (1, 3)
logical shape      = [8, 16]
```

### 6.1 Flatten

按 row-major 顺序：

```text
flat = row * 16 + col
     = 1 * 16 + 3
     = 19
```

### 6.2 按 shard extents 拆分

Shard extents 是：

```text
(8, 2, 4, 2)
```

拆分过程：

| Iter | 计算 | Coordinate | Remainder |
|---:|---|---:|---:|
| 0 | `19 // 16` | 1 | 3 |
| 1 | `3 // 8` | 0 | 3 |
| 2 | `3 // 2` | 1 | 1 |
| 3 | `1 // 1` | 1 | 0 |

因此：

```text
(c0, c1, c2, c3) = (1, 0, 1, 1)
```

### 6.3 每个分量产生物理贡献

| Iter | Coordinate | Stride | Axis | Contribution |
|---:|---:|---:|---|---|
| 0 | 1 | 4 | `laneid` | `4 @ laneid` |
| 1 | 0 | 1 | `warpid` | `0 @ warpid` |
| 2 | 1 | 1 | `laneid` | `1 @ laneid` |
| 3 | 1 | 1 | `m` | `1 @ m` |

属于同一 axis 的贡献相加：

```text
laneid = 4 + 1 = 5
warpid = 0
m      = 1
```

所以 shard 基础坐标是：

```text
D(1, 3) = {laneid: 5, warpid: 0, m: 1}
```

### 6.4 加入固定 offset

```text
O = {warpid: 5}
```

因此：

```text
D(1, 3) + O = {laneid: 5, warpid: 5, m: 1}
```

这正是：

```python
layout.apply(1, 3, shape=[8, 16])
```

返回的 base coordinate。

### 6.5 枚举 replica

Replica offsets 是：

```text
0 @ warpid
4 @ warpid
```

在 base coordinate 上分别加入：

```text
warpid = 5 + 0 = 5
warpid = 5 + 4 = 9
```

所以 `(1, 3)` 的完整物理位置集合是：

```text
{(laneid=5, warpid=5, m=1),
 (laneid=5, warpid=9, m=1)}
```

但是 `apply()` 只返回：

```text
{"m": 1, "warpid": 5, "laneid": 5}
```

它不会返回 `warpid=9` 的副本。副本枚举属于 `layout.replica` 的
下游消费者。

对于整个 `(8, 16)` tile，基础映射可以写成：

```text
laneid = 4 * i + (floor(j / 2) mod 4)
warpid = floor(j / 8) + 5
m      = j mod 2
```

其中：

```text
i in [0, 8)
j in [0, 16)
```

Replica 再让每个位置在 `warpid` 轴额外出现一次，偏移为 4。

## 七、`S`、`R`、offset 修改后的差异

| 修改 | `apply()` 是否变化 | physical copies 是否变化 | 预期现象 |
|---|---|---|---|
| 修改 `S[...]` 的 stride 或 axis | 是 | 通常不变 | 逻辑元素映射到不同 lane / warp / local slot |
| 删除 `R[...]` | 否 | 从 2 份变成 1 份 | base coordinate 不变，但副本消费者少一份数据 |
| 修改 `R[2 : 4@warpid]` 为 `R[2 : 8@warpid]` | 否 | 仍是 2 份，间隔变成 8 | 副本位置变成 `warpid=5` 和 `13` |
| 把 offset 从 `5@warpid` 改成 `7@warpid` | 是 | 不增加副本 | 所有位置整体平移 |
| 改成 `R[4 : 32@TLane]` | 否 | 4 份 | 基础坐标不变，额外出现四个 TMEM Lane window |

这张表也解释了常见混淆：

```text
R 看得见副本数量，但 apply() 看不见 R 的坐标增量。
offset 看得见整体平移，但它不增加副本。
```

## 八、常见错误与可观察症状

| 错误 | 可观察症状 | 优先检查 |
|---|---|---|
| 以为 `apply()` 会返回所有 replica 坐标 | 只处理 base location，部分 warp 或 TMEM window 没有数据 | `layout.replica` 的下游 consumer |
| 把 `R[...]` 当作额外逻辑 shape | flatten 尺寸计算错误，出现不存在的逻辑元素 | 逻辑 shape 只由 shard 的逻辑域决定 |
| 把固定 offset 当作复制 | 数据整体错位，但没有多出副本 | offset 和 replica 的语义 |
| 混用 `laneid` 与 `TLane` | `apply()` 能得到坐标，但硬件消费者对错位置读取 | axis 名称和 buffer scope |
| 同一 axis 的贡献只保留最后一个 | lane 或 warp 映射少加一部分，行列有规律错位 | 同轴贡献是否全部相加 |
| 没有传 `shape=` | 输入坐标被当成 shard coordinate，或拆分结果偏差 | `apply(*logical, shape=...)` 的调用形式 |
| 把 element stride 当成 byte stride | 8-bit 或 16-bit buffer 出现 2 倍、4 倍地址偏差 | buffer dtype 和 hardware Col 打包规则 |
| 修改 layout 却只改一侧 consumer | 编译可能成功，数值整块错位 | producer layout 与 consumer layout 是否一致 |

在推理 kernel 中，这类错误常见于：

```text
GEMM accumulator 的 row / column 错位
epilogue 读出错误的 TMEM window
block-scaled GEMM 的 scale factors 少复制或多复制
Attention 中 Q/K/V tile 的 lane ownership 不一致
```

## 九、验证命令和预期输出

### 环境

这个脚本只需要 TVM，不需要 CUDA GPU：

```bash
python -m pip install apache-tvm==0.26.0
python tirx_layout_s_r_offset.py
```

本机实际使用的是：

```text
/Users/saboxu/Documents/ChatGPT/mlc学习/mlc/bin/python
TVM 0.26.dev246
```

`0.26.dev246` 与课程要求的 `0.26.0` 在本篇使用的 `TileLayout` API
上兼容；预计输出为：

```text
layout: T.TileLayout(T.S[(8, 2, 4, 2):(4 @ Axis.laneid, 1 @ Axis.warpid, 1 @ Axis.laneid, 1)] + T.R[2:4 @ Axis.warpid] + 5 @ Axis.warpid)
base coordinate dict: {"m": 1, "warpid": 5, "laneid": 5}
laneid: 5
warpid: 5
m: 1
replica warps: [5, 9]
shard: (T.Iter(8, 4, "laneid"), T.Iter(2, 1, "warpid"), T.Iter(4, 1, "laneid"), T.Iter(2, 1, "m"))
replica: (T.Iter(2, 4, "warpid"),)
offset: {tirx.Axis(name="warpid"): 5}
```

不同 TVM 版本可能改变对象字符串的显示形式。需要核对的稳定结果
是：

```text
base coordinate = laneid 5, warpid 5, m 1
replica warps   = 5, 9
```

本机运行边界：

```text
可以实际运行: TileLayout 构造、apply()、shard / replica / offset 查询
不能实际运行: 需要 Blackwell 的 tcgen05 kernel
```

## 十、自测题

### 1. `S[...]`、`R[...]` 和 offset 分别回答什么问题？

答：

```text
S[...]：逻辑元素的基础物理坐标是什么？
R[...]：同一个逻辑元素还要出现在哪些额外位置？
offset：整个 layout 要整体平移多少？
```

### 2. `R[2 : 4@warpid]` 枚举出的两个 offset 是什么？

答：

```text
r0 = 0 @ warpid
r1 = 4 @ warpid
```

副本数量是 2，副本之间的 `warpid` 间隔是 4。

### 3. 为什么 `apply()` 不会返回 replica 坐标？

答：`apply()` 的职责是计算：

```text
D(x) + O
```

也就是逻辑元素的基础物理坐标。Replica iters 保存在
`layout.replica` 中，由实际使用该 layout 的 tile operation 枚举和
消费。

### 4. 逻辑坐标 `(1, 3)` 在 shape `[8, 16]` 中 flatten 后是多少？

答：

```text
flat = 1 * 16 + 3 = 19
```

它再按照 shard extents `(8, 2, 4, 2)` 拆成：

```text
(1, 0, 1, 1)
```

### 5. 为什么最后得到的 `warpid` 是 5，而副本位置是 5 和 9？

答：shard 的 `warpid` 基础贡献是 0，固定 offset 是 5，所以：

```text
base warpid = 0 + 5 = 5
```

Replica 再枚举 `0@warpid` 和 `4@warpid`，因此完整副本位置是：

```text
5 + 0 = 5
5 + 4 = 9
```

## 十一、命名轴：`laneid`、`warpid`、`m`、`TLane`、`TCol`

### 本次讲解位置

```text
本次讲解位置
章节：chapter_tirx_layout_api
小节：命名轴
知识点：laneid / warpid / m / TLane / TCol 分别表示什么
上次：TileLayout 的 S[...]、R[...] 与 offset
下次：apply() 的三种输入形式与 flatten / decompose
PTX：无直接 PTX 指令；这些轴分别对应 thread、warp、线性存储与 TMEM 坐标
```

上一课已经知道 `S[...]` 的每个 iter 都由：

```text
(extent, stride, axis)
```

组成。现在要解决一个更基础的问题：

```text
axis 只是一个名字，还是实际代表不同的硬件资源？
为什么 laneid=5 和 TLane=5 不能视为同一个位置？
为什么默认轴 m 有时表示地址，有时表示 register slot？
```

答案是：**axis 是坐标空间的名字，不是普通的维度标签。**

### 心智模型

普通 shape-stride 最终只产生一个线性地址：

```text
地址 = 16 * row + col
```

命名轴 layout 产生的是一组坐标：

```text
{
    laneid: 5,
    warpid: 2,
    m: 1,
}
```

同一个整数 `5` 只有在 axis 也相同时才表示同一个物理位置：

```text
laneid=5  != TLane=5
warpid=5  != laneid=5
m=5       != TCol=5
```

可以把它想成多维坐标：

```text
(laneid=5, warpid=2, m=1)
```

和普通坐标 `(x=5, y=2)` 一样，坐标值和坐标轴必须一起看。

### 本节重点轴

| Axis | 坐标含义 | 典型范围或单位 |
|---|---|---|
| `laneid` | thread 在当前 warp 中的 lane ID | `[0, 32)` |
| `warpid` | warp 在当前 CTA 中的编号 | 由当前 CTA 的 warp 数决定 |
| `m` | 默认线性物理轴 | 单位由 buffer scope 决定 |
| `TLane` | TMEM 的硬件 Lane 坐标 | `[0, 128)` |
| `TCol` | TMEM 的 Column 坐标 | 以 buffer element 计，不一定等于 hardware Col |

它们回答的是不同层级的位置：

```text
laneid / warpid
-> thread 和 warp 拥有哪份数据

m
-> 在某种线性存储中位于哪个 element 或 local slot

TLane / TCol
-> 在 TMEM 的二维存储坐标中位于哪里
```

### `m` 的语义由 buffer scope 决定

`m` 不是固定等于 global address，也不是固定等于 register slot。

| Buffer scope | `m` 的常见含义 |
|---|---|
| global memory | 线性 element 下标 |
| shared memory | shared memory 内的线性 element 下标，swizzle 前的基础地址 |
| register-backed local buffer | 当前 thread 自己的局部线性位置或 fragment slot |

例如：

```python
S[(8, 16) : (16 @ m, 1 @ m)]
```

对逻辑坐标 `(1, 3)` 产生：

```text
m = 1 * 16 + 3 * 1 = 19
```

这里的 `m=19` 是普通 row-major 线性位置。它最终是全局地址、shared memory
地址，还是某个 thread 的 register slot，由绑定 layout 的 buffer scope
决定。

### `laneid` 和 `TLane` 不是一回事

```text
laneid
-> 执行线程的 lane 身份

TLane
-> TMEM 的存储 Lane 坐标
```

一个 warp 有 32 个执行 lane：

```text
laneid in [0, 32)
```

TMEM 有 128 条硬件 Lane：

```text
TLane in [0, 128)
```

所以 `laneid=5` 描述“第 5 个线程 lane 拥有或处理这个元素”，而 `TLane=5`
描述“这个元素存放在 TMEM 的第 5 条 Lane”。编译器可以让 thread lane 5
读取 TMEM Lane 5，但两个 axis 的身份并没有被合并。

同理，`warpid` 是 CTA 范围内的 warp 编号；`wid_in_wg` 才是 warp 在
warpgroup 内的相对编号。二者不能因为数值相同而互换。

### `TCol` 的单位

`TCol` 的 stride 以 buffer element 为单位。一个 32-bit hardware Col
能容纳多少个元素，取决于 dtype：

| dtype 宽度 | 每个 32-bit hardware Col 的元素数 | `TCol=3` 对应的 hardware Col |
|---:|---:|---:|
| 32-bit | 1 | 3 |
| 16-bit | 2 | `3 // 2 = 1` |
| 8-bit | 4 | `3 // 4 = 0` |

因此在 fp16 或 fp8 scale-factor buffer 中：

```text
TCol=3
```

不代表 hardware Col 编号一定是 3。必须结合 dtype 计算元素打包关系。

### 完整可运行代码

下面的脚本只构造和查询 layout，不 launch CUDA kernel，可以在当前 macOS
环境运行。

文件名：

```text
tirx_named_axes.py
```

完整代码：

```python
from tvm.tirx.layout import S, TileLayout, laneid, warpid, m, TLane, TCol


def main():
    frag = TileLayout(
        S[(8, 2, 4, 2) : (4 @ laneid, 1 @ warpid, 1 @ laneid, 1)]
    )
    tmem = TileLayout(
        S[(128, 256) : (1 @ TLane, 1 @ TCol)]
    )
    linear = TileLayout(
        S[(8, 16) : (16 @ m, 1 @ m)]
    )

    print("frag shard:", frag.shard)
    print("tmem shard:", tmem.shard)
    print("linear shard:", linear.shard)

    frag_coord = frag.apply(1, 3, shape=[8, 16])
    tmem_coord = tmem.apply(17, 130, shape=[128, 256])
    linear_coord = linear.apply(1, 3, shape=[8, 16])

    print("frag (1, 3):", frag_coord)
    print("tmem (17, 130):", tmem_coord)
    print("linear (1, 3):", linear_coord)

    print("frag laneid:", frag_coord["laneid"])
    print("frag warpid:", frag_coord["warpid"])
    print("frag m:", frag_coord["m"])
    print("tmem TLane:", tmem_coord["TLane"])
    print("tmem TCol:", tmem_coord["TCol"])
    print("linear m:", linear_coord["m"])


if __name__ == "__main__":
    main()
```

### 代码逐段解释

第一组 layout：

```python
frag = TileLayout(
    S[(8, 2, 4, 2) : (4 @ laneid, 1 @ warpid, 1 @ laneid, 1)]
)
```

它描述一个 register fragment：

```text
第一个 iter: 4 @ laneid
第二个 iter: 1 @ warpid
第三个 iter: 1 @ laneid
第四个 iter: 1，默认 axis=m
```

前两个 laneid iter 属于同一个 axis，所以它们的贡献必须相加：

```text
laneid_total = c0 * 4 + c2 * 1
```

第二组 layout：

```python
tmem = TileLayout(
    S[(128, 256) : (1 @ TLane, 1 @ TCol)]
)
```

它描述 TMEM 中的二维 tile：

```text
logical row    -> TLane
logical column -> TCol
```

这里的 axis 是存储坐标，不表示线程 ownership。

第三组 layout：

```python
linear = TileLayout(
    S[(8, 16) : (16 @ m, 1 @ m)]
)
```

它描述普通 row-major 线性布局：

```text
逻辑 row    -> m 上 stride 16
逻辑 column -> m 上 stride 1
```

此时：

```text
m(1, 3) = 1 * 16 + 3 * 1 = 19
```

### 完整数值执行追踪

#### 1. Register fragment：`(1, 3)`

逻辑 shape：

```text
[8, 16]
```

先 flatten：

```text
flat = 1 * 16 + 3 = 19
```

再按 shard extents `(8, 2, 4, 2)` 分解：

```text
(c0, c1, c2, c3) = (1, 0, 1, 1)
```

逐项乘 stride：

| Iter | Coordinate | Stride | Axis | Contribution |
|---:|---:|---:|---|---|
| 0 | 1 | 4 | `laneid` | `4 @ laneid` |
| 1 | 0 | 1 | `warpid` | `0 @ warpid` |
| 2 | 1 | 1 | `laneid` | `1 @ laneid` |
| 3 | 1 | 1 | `m` | `1 @ m` |

合并同一个 axis 的贡献：

```text
laneid = 4 + 1 = 5
warpid = 0
m      = 1
```

结果为：

```text
{"m": 1, "warpid": 0, "laneid": 5}
```

#### 2. TMEM layout：`(17, 130)`

逻辑 shape 和 shard extents 都是 `(128, 256)`，因此先 flatten：

```text
flat = 17 * 256 + 130
     = 4352 + 130
     = 4482
```

再拆回：

```text
4482 = 17 * 256 + 130
```

所以：

```text
TLane = 17
TCol  = 130
```

结果为：

```text
{"TCol": 130, "TLane": 17}
```

这里 `TCol=130` 是 buffer element 坐标。若 dtype 是 fp16，它对应的
hardware Col 和元素内位置还需要按 2 个元素/Col 计算。

#### 3. 普通线性 layout：`(1, 3)`

```text
m = 1 * 16 + 3 * 1
  = 19
```

结果为：

```text
{"m": 19}
```

同样是 `m` 轴，如果这个 buffer 在当前 thread 的 local scope 中，那么
`19` 是该 thread 的局部位置；如果 buffer 在 shared 或 global scope
中，那么 `19` 是线性 element 下标。

### 常见错误与可观察症状

| 错误 | 可观察症状 | 优先检查 |
|---|---|---|
| 认为 `laneid=5` 和 `TLane=5` 是同一位置 | layout 查询看似正常，实际读错线程数据或 TMEM 区域 | axis 名称是否一致 |
| 把 `warpid` 当作 warpgroup 内编号 | warpgroup 中第二个 warp 被错误解释为全局 warp 编号 | 应使用 `wid_in_wg` 时是否误用了 `warpid` |
| 把 `TCol` 当作 byte offset | 8-bit、16-bit buffer 出现 2 倍或 4 倍地址偏差 | dtype、elements per hardware Col |
| 把 `m` 固定理解成 global address | local fragment 的 register slot 被错误映射 | buffer scope |
| 同一个 axis 的多个贡献只保留最后一个 | lane 或 warp 映射呈规律性错位 | 同轴贡献是否全部相加 |
| 修改 axis 名但数值没变 | 编译可能通过，producer 与 consumer 读错位置 | 两侧 layout 的 axis 名称和取值是否一致 |

在推理 kernel 中，典型表现是：

```text
GEMM fragment 中某些 lane 拿到错误元素
TMEM accumulator 的 row 或 column 整块错位
scale-factor 读取到相邻元素或错误 byte
epilogue 写回时 warpgroup 内的 ownership 不一致
```

### 验证命令和预期输出

运行：

```bash
/Users/saboxu/Documents/ChatGPT/mlc学习/mlc/bin/python tirx_named_axes.py
```

当前环境实测输出：

```text
frag shard: (T.Iter(8, 4, "laneid"), T.Iter(2, 1, "warpid"), T.Iter(4, 1, "laneid"), T.Iter(2, 1, "m"))
tmem shard: (T.Iter(128, 1, "TLane"), T.Iter(256, 1, "TCol"))
linear shard: (T.Iter(8, 16, "m"), T.Iter(16, 1, "m"))
frag (1, 3): {"m": 1, "warpid": 0, "laneid": 5}
tmem (17, 130): {"TCol": 130, "TLane": 17}
linear (1, 3): {"m": 19}
frag laneid: 5
frag warpid: 0
frag m: 1
tmem TLane: 17
tmem TCol: 130
linear m: 19
```

本机运行边界：

```text
可以运行: TileLayout 构造、axis 查询、apply() 坐标计算
不能运行: 需要 Blackwell TMEM / tcgen05 硬件执行的 kernel
```

### 自测题

#### 1. 为什么 `laneid=5` 和 `TLane=5` 不是同一个位置？

答：`laneid` 是 warp 内执行 thread 的坐标，`TLane` 是 TMEM
存储 Lane 的坐标。二者属于不同坐标空间，只有 axis 名称相同才表示同一位置。

#### 2. `m` 表示 global memory address 吗？

答：不一定。`m` 是默认线性物理轴；具体含义由 buffer scope 决定：

```text
global  -> 线性 element 下标
shared  -> shared memory 线性 element 下标
register-backed local -> 当前 thread 的局部 slot
```

#### 3. 同一个 `laneid` 在 shard 中出现两次，应该相加还是覆盖？

答：相加。每个 iter 都产生一个贡献：

```text
laneid = c_first * stride_first + c_second * stride_second
```

当前例子中：

```text
laneid = 1 * 4 + 1 * 1 = 5
```

#### 4. fp16 的 `TCol=3` 能不能直接当作 hardware Col 3？

答：不能。fp16 每个 32-bit hardware Col 容纳 2 个元素，因此：

```text
hardware_col = TCol // 2
```

`TCol=3` 位于 hardware Col 1 中。

#### 5. 逻辑坐标 `(1, 3)` 在 `S[(8, 16) : (16@m, 1@m)]` 中得到的 `m` 是多少？

答：

```text
m = 1 * 16 + 3 * 1 = 19
```

## 十二、`apply()` 的三种输入形式与 flatten / decompose

### 本次讲解位置

```text
本次讲解位置
章节：chapter_tirx_layout_api
小节：正向映射
知识点：apply() 的三种输入形式与 flatten / decompose
上次：命名轴 laneid / warpid / m / TLane / TCol
下次：TMEM accumulator layout 示例
PTX：无直接 PTX 指令；结果供后续 tile operation 和 lowering 使用
```

上一课已经知道，layout 的物理坐标来自每个 iter 的：

```text
(extent, stride, axis)
```

这一课进一步解决一个很容易混淆的问题：

```python
layout.apply(1, 3, shape=[8, 16])
layout.apply(19)
layout.apply(1, 0, 1, 1)
```

为什么这三个看起来不同的调用会得到同一个坐标？它们并不是三个功能，
而是从同一条映射流水线的三个位置进入。

### 学习目标

学完后应该能够：

1. 说清 logical coordinate、linear coordinate 和 shard coordinate 的区别。
2. 判断一次 `apply()` 调用是否执行 flatten，是否执行 decompose。
3. 用逻辑 shape 和 shard extents 手算中间值。
4. 解释为什么 `TileLayout` 本身不能自动推断逻辑 shape。
5. 判断 `apply()` 的返回值是否包含 `R[...]` 产生的 replica。

### 心智模型

完整映射可以画成一条流水线：

```text
logical coordinate + logical shape
    |
    | flatten，row-major
    v
linear coordinate q
    |
    | decompose，按 shard extents
    v
shard coordinate (c0, c1, ..., cn-1)
    |
    | 每个分量乘 stride，并按 axis 合并
    v
physical base coordinate D
    |
    | 加固定 offset O
    v
layout.apply() 的返回值
```

`apply()` 的三个入口分别位于：

```text
logical coordinate + shape -> 从流水线最前面进入
linear coordinate          -> 从 flatten 之后进入
shard coordinate           -> 从 decompose 之后进入
```

因此三种入口的职责可以压缩为：

| 调用形式 | 输入处于哪个阶段 | 是否 flatten | 是否 decompose |
|---|---|---:|---:|
| `apply(*logical_coord, shape=...)` | 逻辑坐标 | 是 | 是 |
| `apply(linear_coord)` | 一维线性索引 | 否 | 是 |
| `apply(*shard_coord)` | 各 shard iter 的坐标 | 否 | 否 |

三者最后都会执行相同的：

```text
stride + axis
-> 合并 physical coordinate
-> 加 offset
```

### 三个坐标分别是什么

继续使用前面的 layout：

```python
layout = TileLayout(
    S[(8, 2, 4, 2) : (4 @ laneid, 1 @ warpid, 1 @ laneid, 1)]
    + R[2 : 4 @ warpid]
    + 5 @ warpid
)
```

它包含四个 shard iters：

```text
(extent, stride, axis)
(8,      4,      laneid)
(2,      1,      warpid)
(4,      1,      laneid)
(2,      1,      m)
```

#### Logical coordinate

逻辑坐标是业务或 tile 语义使用的坐标。例如，把 layout 表示的数据解释为
一个 `8 x 16` 矩阵时：

```text
(row=1, col=3)
```

就是一个 logical coordinate。

逻辑 shape 不来自 `TileLayout`，而是调用方在解释逻辑 tensor 时提供的：

```python
shape = [8, 16]
```

同一个 `TileLayout` 也可以被解释为：

```python
shape = [16, 8]
```

因为两者的元素总数都是：

```text
8 * 16 = 128
16 * 8 = 128
```

但同一个 coordinate `(1, 3)` 在两种解释下的 row-major flatten 值不同。
这也是本章最重要的边界之一：

```text
TileLayout 描述 128 个元素的物理映射；
logical shape 决定 128 个元素如何被用户命名和 flatten。
```

#### Linear coordinate

线性坐标是 row-major flatten 后的单个索引：

```text
q in [0, 128)
```

例如：

```text
logical (1, 3), shape [8, 16]
-> q = 1 * 16 + 3
     = 19
```

`19` 就是 linear coordinate。

#### Shard coordinate

Shard coordinate 是 `q` 根据 shard extents：

```text
(8, 2, 4, 2)
```

拆分后的四个分量：

```text
(c0, c1, c2, c3)
```

例如：

```text
q = 19
-> (c0, c1, c2, c3) = (1, 0, 1, 1)
```

这些分量不是新的逻辑维度。它们只是 shard 的四个 iter 分别取什么值。

### 完整可运行代码

下面的脚本不需要 GPU，只需要 TVM。它同时调用三个入口，并用一个
不同的逻辑 shape 证明逻辑 shape 不会从 `TileLayout` 自动推断。

文件名：

```text
tirx_apply_inputs.py
```

完整代码：

```python
from tvm.tirx.layout import R, S, TileLayout, laneid, warpid


def main():
    layout = TileLayout(
        S[(8, 2, 4, 2) : (4 @ laneid, 1 @ warpid, 1 @ laneid, 1)]
        + R[2 : 4 @ warpid]
        + 5 @ warpid
    )

    logical_coord = (1, 3)
    logical_shape = [8, 16]

    flat = logical_coord[0] * logical_shape[1] + logical_coord[1]
    shard_coord = (1, 0, 1, 1)

    by_logical = layout.apply(*logical_coord, shape=logical_shape)
    by_linear = layout.apply(flat)
    by_shard = layout.apply(*shard_coord)

    print("logical + shape:", by_logical)
    print("linear:", by_linear)
    print("shard coords:", by_shard)
    print("same result:", by_logical == by_linear == by_shard)

    alternate_shape = [16, 8]
    by_alternate_logical = layout.apply(
        *logical_coord,
        shape=alternate_shape,
    )
    print("alternate shape [16, 8]:", by_alternate_logical)

    try:
        layout.apply(*logical_coord)
    except Exception as err:
        first_line = str(err).splitlines()[0]
        print("missing shape error:", type(err).__name__, first_line)


if __name__ == "__main__":
    main()
```

### 三个阶段对应的代码

#### 入口一：logical coordinate + shape

```python
by_logical = layout.apply(1, 3, shape=[8, 16])
```

它执行：

```text
flatten -> decompose -> stride / axis -> offset
```

调用者必须明确告诉 `apply()`：目前这两个整数应解释为 `[8, 16]` 中的
`(row, col)`。`TileLayout` 只知道 shard extents 是 `(8, 2, 4, 2)`，
并不知道外部逻辑 tensor 是 `8 x 16`、`16 x 8` 还是别的一种 reshape。

#### 入口二：linear coordinate

```python
by_linear = layout.apply(19)
```

`19` 已经完成了 flatten，所以这个入口不执行 flatten，只执行：

```text
decompose -> stride / axis -> offset
```

它不需要 `shape`，因为一维线性索引不再依赖逻辑 tensor 的维数和每维
大小。调用这个入口的前提是：调用者已经正确算出了逻辑坐标的 row-major
linear index。

#### 入口三：shard coordinate

```python
by_shard = layout.apply(1, 0, 1, 1)
```

四个参数已经分别对应四个 shard iters，所以这个入口既不 flatten，
也不 decompose。它只执行：

```text
stride / axis -> offset
```

这条路径适合：

```text
调试 shard iter 到 physical coordinate 的映射
直接验证某个 iter 组合是否产生期望的 lane / warp / m
判断错误来自 flatten / decompose，还是来自 stride / axis
```

### Python 层如何选择入口

当前 TVM 的简化分发逻辑是：

```python
if len(coord) == 1:
    return LayoutApplyLinear(self, coord[0])
if shape is None:
    return LayoutApply(self, coord)
return LayoutApplyWithShape(self, coord, shape)
```

因此：

```text
一个参数              -> 一律按 linear coordinate 处理
多个参数且无 shape    -> 按 shard coordinate 处理
多个参数且有 shape    -> 先按 logical shape flatten，再 decompose
```

这也解释了一个常见陷阱：

```python
layout.apply(19)
```

和：

```python
layout.apply(19, shape=[...])
```

在当前实现中不会进入同一分支。前者把 `19` 当 linear coordinate；
后者只有一个参数，分发逻辑仍优先把它当 linear coordinate，`shape`
并不会自动把单参数改写成多维逻辑坐标。

### 完整数值追踪一：逻辑 shape `[8, 16]`

查询：

```text
logical coordinate = (1, 3)
logical shape      = [8, 16]
```

#### 1. Flatten

row-major flatten：

```text
flat = row * shape[1] + col
     = 1 * 16 + 3
     = 19
```

#### 2. Decompose

Shard extents：

```text
(8, 2, 4, 2)
```

从组合基数最大的维度开始拆：

| 步骤 | 当前值 | Extent | 商/Coordinate | 余数/Remainder |
|---:|---:|---:|---:|---:|
| `c0` | 19 | 8 | `19 // 16 = 1` | `19 % 16 = 3` |
| `c1` | 3 | 2 | `3 // 8 = 0` | `3 % 8 = 3` |
| `c2` | 3 | 4 | `3 // 2 = 1` | `3 % 2 = 1` |
| `c3` | 1 | 2 | `1 // 1 = 1` | `0` |

得到：

```text
(c0, c1, c2, c3) = (1, 0, 1, 1)
```

#### 3. 计算 physical base coordinate

| Iter | Coordinate | Stride | Axis | Contribution |
|---:|---:|---:|---|---|
| 0 | 1 | 4 | `laneid` | `4 @ laneid` |
| 1 | 0 | 1 | `warpid` | `0 @ warpid` |
| 2 | 1 | 1 | `laneid` | `1 @ laneid` |
| 3 | 1 | 1 | `m` | `1 @ m` |

同一个 axis 的贡献相加：

```text
laneid = 1 * 4 + 1 * 1
       = 5
warpid = 0 * 1
       = 0
m      = 1 * 1
       = 1
```

所以：

```text
D(1, 3) = {laneid: 5, warpid: 0, m: 1}
```

#### 4. 加固定 offset

```text
O = {warpid: 5}
```

得到：

```text
D(1, 3) + O = {laneid: 5, warpid: 5, m: 1}
```

这就是三个入口共同返回的 base coordinate：

```text
{"m": 1, "warpid": 5, "laneid": 5}
```

它仍然不包含 `R[2 : 4@warpid]` 枚举出的副本 `warpid=9`。

### 完整数值追踪二：同一个 `(1, 3)`，逻辑 shape 改成 `[16, 8]`

现在不修改 layout，只把调用方对数据的逻辑解释改成：

```python
shape = [16, 8]
```

#### 1. Flatten

```text
flat = 1 * 8 + 3
     = 11
```

同一个逻辑坐标 `(1, 3)` 从 `19` 变成了 `11`。

#### 2. Decompose

| 步骤 | 当前值 | Extent | Coordinate | Remainder |
|---:|---:|---:|---:|---:|
| `c0` | 11 | 8 | `11 // 16 = 0` | `11` |
| `c1` | 11 | 2 | `11 // 8 = 1` | `3` |
| `c2` | 3 | 4 | `3 // 2 = 1` | `1` |
| `c3` | 1 | 2 | `1 // 1 = 1` | `0` |

得到：

```text
(c0, c1, c2, c3) = (0, 1, 1, 1)
```

#### 3. 合成坐标

```text
laneid = 0 * 4 + 1 * 1
       = 1
warpid = 1 * 1
       = 1
m      = 1
```

再加固定 offset：

```text
warpid = 1 + 5
       = 6
```

最终结果：

```text
{"m": 1, "warpid": 6, "laneid": 1}
```

对比表：

| Logical shape | Flatten | Shard coordinate | Base before offset | Final base |
|---|---:|---|---|---|
| `[8, 16]` | `1 * 16 + 3 = 19` | `(1, 0, 1, 1)` | `laneid=5, warpid=0, m=1` | `laneid=5, warpid=5, m=1` |
| `[16, 8]` | `1 * 8 + 3 = 11` | `(0, 1, 1, 1)` | `laneid=1, warpid=1, m=1` | `laneid=1, warpid=6, m=1` |

它直接证明：

```text
layout 相同，不代表 logical coordinate 的 flatten 结果相同。
```

在真实 kernel 中，如果 producer 按 `[8, 16]` 解释 tile，而 consumer 按
`[16, 8]` 解释同一个 coordinate，两边可能都不会报错，却会访问不同的
warp 和 lane。

### 用等价性检查定位错误

对于合法输入，下面三个调用必须等价：

```python
layout.apply(1, 3, shape=[8, 16])
layout.apply(19)
layout.apply(1, 0, 1, 1)
```

可以利用这一点做分层检查：

| 检查 | 如果失败，优先怀疑 |
|---|---|
| 手算 `flat` 是否等于代码中的 linear coordinate | 逻辑 shape、坐标顺序或 flatten 规则 |
| `apply(linear)` 与 `apply(*shard_coord)` 是否相同 | decompose 或 shard extents |
| `apply(logical, shape=...)` 与 `apply(linear)` 是否相同 | flatten、shape 传递或调用入口 |
| `apply(*shard_coord)` 是否等于手算 coordinate | stride、axis 或 offset |

这是一种很适合推理 kernel 的调试顺序：

```text
先分离 flatten
再分离 decompose
最后检查 stride / axis / offset
```

它比直接看最终生成的 CUDA 更容易定位布局偏差发生在哪一层。

### `apply()` 的执行边界

`apply()` 只做坐标计算。它不会：

```text
launch thread
移动数据
检查业务层的 row / col bounds
枚举 replica
建立 producer / consumer 同步
```

因此它适合在 Python 中做静态验证和算子开发调试，但不能单独证明
硬件数据路径正确。`apply()` 返回正确坐标之后，还需要 tile operation、
lowering 和同步协议各自满足约束。

### 常见错误与可观察症状

| 错误 | 为什么错 | 可观察症状 | 优先检查 |
|---|---|---|---|
| 认为 `TileLayout` 保存了 logical shape | shape 来自调用方的逻辑解释，不来自 shard extents | 同一 coordinate 在 reshape 后落到不同 warp / lane | 调用双方是否使用同一个 input shape |
| `apply(*logical_coord)` 忘记 `shape=` | 多参数无 shape 会被当作 shard coordinate | rank 不同时报 coordinate size 错误；rank 相同时可能静默产生错误坐标 | 输入参数个数和 `shape=` |
| 把 `apply(19)` 当成逻辑坐标 | 单参数入口一律按 linear coordinate 处理 | 以为在查询 `(1, 9)` 或 `(0, 19)`，实际在查询已 flatten 的 `19` | 先手算 `flat` |
| 给 `apply(19)` 再传 `shape` 期望改变语义 | 当前分发优先匹配单参数 linear 入口 | shape 没有按预期参与 flatten | 使用多参数逻辑坐标入口 |
| 把 shard coordinate 当成逻辑坐标 | shard 分量属于 iter，不属于业务 tensor 轴 | stride / axis 组合看似正确但坐标来源错误 | 输入处于流水线的哪一阶段 |
| 认为 `apply()` 会返回所有副本 | `apply()` 只返回 base + offset | 只处理一个位置，另一个 replica 没有数据 | `layout.replica` 的下游 consumer |
| 假定越界坐标会自动报错 | `apply()` 是映射函数，不承担业务 bounds check | 编译通过，但 physical coordinate 超出预期范围 | 调用前验证 logical domain |
| logical shape 的元素数不等于 shard 域 | 映射仍可能计算，但输入解释和 layout 契约不一致 | 部分物理位置不用，或越界访问 | `prod(shape)` 与 `prod(shard extents)` |
| 只看最终 coordinate，不核对中间量 | 最终值错误可能来自 flatten、decompose 或 stride | 只能猜是“layout 错”，无法定位 | 三入口等价性检查 |

在推理 kernel 中，这类错误常表现为：

```text
GEMM epilogue 按错误的 row / col reshape 读 accumulator
Attention 的 tile reshape 与 Q/K/V layout 不一致
量化 scale factor 按错误的扁平索引选择 block
同一 tile 的 producer 和 consumer 对 logical shape 的理解不同
debug 时 linear coordinate 手算正确，但传进 apply() 的入口错误
```

### 验证命令和预期输出

运行：

```bash
/Users/saboxu/Documents/ChatGPT/mlc学习/mlc/bin/python tirx_apply_inputs.py
```

本机 TVM `0.26.dev246` 实测输出：

```text
logical + shape: {"m": 1, "warpid": 5, "laneid": 5}
linear: {"m": 1, "warpid": 5, "laneid": 5}
shard coords: {"m": 1, "warpid": 5, "laneid": 5}
same result: True
alternate shape [16, 8]: {"m": 1, "warpid": 6, "laneid": 1}
missing shape error: InternalError Check failed: coord.size() == shard.size() (2 vs. 4) : Coordinate size must match the number of shard axes
```

需要核对的稳定结果是：

```text
三个入口在 (1, 3), [8, 16] 下结果相同
同一个 (1, 3) 在 [16, 8] 下变成 laneid=1, warpid=6, m=1
多参数逻辑坐标漏掉 shape 时，坐标个数与 shard 个数不匹配
```

错误信息中的：

```text
coord.size() == shard.size() (2 vs. 4)
```

表示调用者传了两个 coordinate，但 layout 有四个 shard iters。它正好
说明多参数无 `shape` 的分支期待的是 shard coordinate。

本机运行边界：

```text
可以运行: TileLayout 构造、三个 apply() 入口、flatten / decompose 结果对比
不能运行: 需要 NVIDIA GPU 或 Blackwell TMEM 的 kernel launch
```

### 本课结论

把整条流水线再压缩一次：

```text
apply(*logical, shape=...)
    = flatten + decompose + mapping

apply(linear)
    = decompose + mapping

apply(*shard)
    = mapping
```

其中三个入口最终都执行：

```text
mapping = 按 axis 合并 coordinate * stride + 加 offset
```

最关键的工程结论是：

```text
physical layout 正确，不代表 logical shape 解释正确；
两者必须分别验证，并保持 producer / consumer 一致。
```

### 自测题

#### 1. `apply(19)` 会执行 flatten 吗？

答：不会。`19` 已经是一个 linear coordinate，当前 Python 分发会把它
直接送入 `LayoutApplyLinear`。它仍然会执行 decompose 和 physical
coordinate 合成。

#### 2. `(1, 3)` 在逻辑 shape `[16, 8]` 下的 linear coordinate 是多少？

答：

```text
flat = 1 * 8 + 3
     = 11
```

再按 `(8, 2, 4, 2)` 拆成：

```text
(0, 1, 1, 1)
```

#### 3. 为什么 `apply(1, 3)` 不传 `shape` 会报错？

答：多参数且没有 `shape` 时，`apply()` 把 `(1, 3)` 解释为 shard
coordinate。当前 layout 有四个 shard iters，因此输入只有两个坐标，
不满足：

```text
coord.size() == shard.size()
```

#### 4. 为什么同一个 `TileLayout` 可以对应多种 logical shape？

答：`TileLayout` 描述的是固定大小线性域到物理坐标的映射。当前 shard
域有：

```text
8 * 2 * 4 * 2 = 128
```

个元素。只要调用方提供与这个域相符的逻辑解释，`(8, 16)`、`(16, 8)`
等 shape 都可以 flatten 到这个线性域，但不同 shape 会改变 coordinate
到 linear index 的对应关系。

#### 5. 三个入口都返回同一个坐标，能否证明整个 kernel 的 replica 和同步也正确？

答：不能。它们只证明 flatten、decompose、stride、axis 和 offset 这一
条坐标计算链路一致。Replica 枚举、数据移动、producer / consumer
同步以及硬件执行仍需分别验证。

## 十三、TMEM accumulator layout 示例：`2 x 128 x 112`

### 本次讲解位置

```text
本次讲解位置
章节：chapter_tirx_layout_api
小节：示例：Blackwell Tensor Memory
知识点：2 x 128 x 112 accumulator 映射到 TLane / TCol
上次：apply() 的三种输入形式与 flatten / decompose
下次：scale-factor layout 中的 replication
PTX：tcgen05.mma 的 d-tmem 坐标；本课不讨论具体指令编码
```

上一课解决了 `apply()` 如何从逻辑坐标计算物理坐标。这一课把同样的方法
用于一个具体问题：

```text
一个 shape 为 (2, 128, 112) 的 accumulator 逻辑 tile，
如何放到 TMEM 的 TLane / TCol 坐标中？
```

这里的重点不是背一个固定 layout，而是分清：

```text
逻辑 tensor 的 a / lane / col
物理 TMEM 的 TLane / TCol
```

### 先建立 TMEM 的物理图

在 `sm_100a` 上，一个 CTA 的 TMEM 可以画成：

```text
           TCol
             0        ...        511
TLane  0  +------+------+------+------+
          | cell | cell | ...  | cell |
       1  +------+------+------+------+
          | cell | cell | ...  | cell |
       ...                              ...
     127  +------+------+------+------+
```

每个坐标最多对应一个 32-bit cell：

```text
TLane in [0, 128)
TCol  in [0, 512)
```

本课要描述的 accumulator 只使用其中一部分：

```text
TLane in [0, 128)
TCol  in [0, 224)
```

因此它没有填满 TMEM，只是在 128 条 Lane 上占用前 224 个 columns。

### 心智模型

把逻辑 tile 想成两个并排的 `128 x 112` 区域：

```text
逻辑 a = 0
  +--------------------------+
  | TLane 0..127             |
  | TCol  0..111             |
  +--------------------------+

逻辑 a = 1
  +--------------------------+
  | TLane 0..127             |
  | TCol  112..223           |
  +--------------------------+
```

映射关系可以直接写成：

```text
TLane = lane
TCol  = 112 * a + col
```

其中：

| 坐标 | 逻辑范围 | 物理含义 |
|---|---:|---|
| `a` | `[0, 2)` | 外层区域编号 |
| `lane` | `[0, 128)` | 当前区域内的 M row |
| `col` | `[0, 112)` | 当前区域内的 N column |
| `TLane` | `[0, 128)` | TMEM 硬件 Lane |
| `TCol` | `[0, 224)` | TMEM element Column 坐标 |

`a` 的语义由调用方决定。它可以表示一个外层 stage、region 或别的逻辑分组；
`TileLayout` 本身只规定映射，不会自动赋予 `a` 这个业务含义。

### 完整可运行代码

下面的脚本可以在没有 NVIDIA GPU 的 macOS 上运行。它只构造 layout、
查询 span 和计算坐标，不执行真正的 `tcgen05.mma`。

文件名：

```text
tirx_tmem_accumulator_layout.py
```

完整代码：

```python
from tvm.tirx.layout import S, TCol, TileLayout, TLane


def main():
    layout = TileLayout(
        S[(2, 128, 112) : (112 @ TCol, 1 @ TLane, 1 @ TCol)]
    )
    logical_shape = [2, 128, 112]

    print("layout:", layout)
    print("size:", layout.size())
    print("span TLane:", layout.span("TLane"))
    print("span TCol:", layout.span("TCol"))

    coords = [
        (0, 0, 0),
        (0, 17, 30),
        (1, 17, 30),
        (1, 127, 111),
    ]
    for a, lane, col in coords:
        physical = layout.apply(a, lane, col, shape=logical_shape)
        print(f"({a}, {lane}, {col}) ->", physical)

    from_logical = layout.apply(1, 17, 30, shape=logical_shape)
    from_linear = layout.apply(16270)
    print("linear 16270 ->", from_linear)
    print("logical == linear:", from_logical == from_linear)


if __name__ == "__main__":
    main()
```

### 逐行解释

#### Shard extents

```python
S[(2, 128, 112) : (112 @ TCol, 1 @ TLane, 1 @ TCol)]
```

三个 iters 分别是：

| Iter | Extent | Stride | Axis | 含义 |
|---:|---:|---:|---|---|
| 0 | 2 | 112 | `TCol` | 每进入一个外层区域，TCol 前进 112 |
| 1 | 128 | 1 | `TLane` | M row 直接映射到 128 条 Lane |
| 2 | 112 | 1 | `TCol` | 区域内 column 连续排列 |

逻辑域大小是：

```text
2 * 128 * 112 = 28672
```

物理范围是：

```text
128 * 224 = 28672
```

所以这个 layout 是一对一映射，没有 replication，也没有空洞。

#### `layout.size()` 和 `layout.span(...)`

```python
layout.size()
layout.span("TLane")
layout.span("TCol")
```

分别得到：

```text
size      = 28672
TLane span = 128
TCol span  = 224
```

需要区分：

```text
size       -> 逻辑元素总数
TLane span -> 该 axis 上需要的坐标范围
TCol span  -> 该 axis 上需要的坐标范围
```

`TCol span = 224` 不表示 TMEM 只有 224 个 columns；硬件容量仍是
128 x 512。它只表示这个 layout 实际使用了 `[0, 224)`。

#### `apply()` 的调用

```python
layout.apply(a, lane, col, shape=[2, 128, 112])
```

这里逻辑 shape 和 shard extents 都是 `(2, 128, 112)`，因此：

```text
logical coordinate
-> flatten
-> decompose
-> 又得到同一个 (a, lane, col)
```

这看似绕了一圈，但它保持了上一课的统一模型。若调用者已经拥有 shard
coordinate，也可以直接调用：

```python
layout.apply(a, lane, col)
```

本例中两者结果相同，因为逻辑 shape 恰好等于 shard extents。

### 数值追踪一：`(0, 17, 30)`

外层区域：

```text
a = 0
```

计算：

```text
TLane = lane
      = 17

TCol = 112 * a + col
     = 112 * 0 + 30
     = 30
```

结果：

```text
{"TLane": 17, "TCol": 30}
```

它位于第一个 `128 x 112` 区域。

### 数值追踪二：`(1, 17, 30)`

只把外层区域从 `0` 改成 `1`：

```text
TLane = 17

TCol = 112 * 1 + 30
     = 112 + 30
     = 142
```

结果：

```text
{"TLane": 17, "TCol": 142}
```

两个逻辑坐标的 `lane` 和 `col` 相同，但属于不同的 `a`，因此物理
`TCol` 相差 112。

如果从逻辑坐标完整走一遍前面学过的 flatten / decompose：

```text
flat = ((1 * 128) + 17) * 112 + 30
     = (128 + 17) * 112 + 30
     = 145 * 112 + 30
     = 16240 + 30
     = 16270
```

按 shard extents `(2, 128, 112)` 分解：

| 步骤 | 当前值 | 组合基数 | Coordinate | Remainder |
|---:|---:|---:|---:|---:|
| `c0` | 16270 | `128 * 112 = 14336` | `16270 // 14336 = 1` | `1934` |
| `c1` | 1934 | `112` | `1934 // 112 = 17` | `30` |
| `c2` | 30 | `1` | `30` | `0` |

得到：

```text
(c0, c1, c2) = (1, 17, 30)
```

再计算：

```text
TLane = 17 * 1
      = 17

TCol = 1 * 112 + 30 * 1
     = 142
```

所以下面的调用结果相同：

```python
layout.apply(1, 17, 30, shape=[2, 128, 112])
layout.apply(16270)
```

#### 为什么是 `16270`？

因为它是以 `(2, 128, 112)` 为逻辑 shape、按 row-major 顺序展开后的
linear coordinate。若把逻辑 shape 改成 `(128, 2, 112)`，同一个坐标
三元组即使数值相同，也会被解释成另一个逻辑元素；这就是上一课强调
“逻辑 shape 不来自 layout”的具体表现。

### 边界检查：`(1, 127, 111)`

最后一个合法坐标：

```text
TLane = 127
TCol  = 112 * 1 + 111
      = 223
```

所以物理范围是半开区间：

```text
TLane in [0, 128)
TCol  in [0, 224)
```

`TCol=224` 已经越界，不属于这个 layout。

### 为什么 extent 可以是 112

TMEM layout 并没有要求每一维必须是 2 的幂。这里可以直接写：

```python
112 @ TCol
```

原因有两层：

```text
物理层：TMEM 按 Lane / Col 坐标寻址，不是要求逻辑 extent 必须对齐到 128。
布局层：TileLayout 只计算 coordinate，112 是一个合法的迭代范围。
```

使用两个 `112` column 区域的原因是设计选择，不是硬件强制：

```text
2 * 112 = 224 columns
```

这样还留下：

```text
512 - 224 = 288 columns
```

可给其他 accumulator stage、scale factors、workspace 或其他 TMEM 数据。

在 block-scaled FP8 GEMM 中，这种“不要盲目占满 256 columns”的选择很常见：
一个 kernel 可能同时需要多个 accumulator 区域和额外的 SFA/SFB 区域。
本课只讲 accumulator 的基础坐标；SFA/SFB 的 replica 是下一课内容。

### 与简单 `128 x N` layout 的关系

如果不需要把两个区域并排，只写一个普通 accumulator：

```python
TileLayout(
    S[(128, 224) : (1 @ TLane, 1 @ TCol)]
)
```

它就是：

```text
TLane = lane
TCol  = col
```

两者占用的物理 columns 都是 224。区别在于逻辑坐标的解释：

| Layout | 逻辑坐标 | 映射 |
|---|---|---|
| `S[(128,224):(1@TLane,1@TCol)]` | `(lane,col)` | `TCol=col` |
| `S[(2,128,112):(112@TCol,1@TLane,1@TCol)]` | `(a,lane,col)` | `TCol=112*a+col` |

第一种把一个 `128 x 224` tile 看成一个整体；第二种明确把 224 个 columns
分成两个逻辑区域，每个区域 112 columns。

### 数据路径与同步边界

本课只构造 layout，不搬运数据。真实 GEMM 路径是：

```text
A/B: GMEM -> SMEM
     tcgen05.mma
C:   SMEM operands -> TMEM accumulator
     tcgen05.commit + mbarrier wait
     tcgen05.ld
     寄存器
```

`TileLayout` 只回答：

```text
逻辑 accumulator 元素 (a, lane, col)
对应哪个 (TLane, TCol)？
```

它不负责：

```text
分配 TMEM columns
启动 tcgen05.mma
保证 MMA 已完成
把 TMEM 数据读取到寄存器
处理 CTA pair
```

因此 coordinate 正确只是必要条件，不是整个 kernel 正确的充分条件。

### 常见错误与可观察症状

| 错误 | 为什么错 | 可观察症状 | 优先检查 |
|---|---|---|---|
| 认为 `a` 自动等于 pipeline stage | layout 不知道 stage 语义 | 把两个逻辑区域当成不能重叠的 stage，实际语义由 kernel 决定 | `a` 的调用方语义 |
| 把 `TCol=223` 当成越界 | 两个 112 column 区域覆盖 `[0,224)` | 合法边界被误判，或者边界元素被排除 | `(2 - 1) * 112 + 112` |
| 认为 `TCol` 一定到 511 | 511 是硬件最大坐标，不是这个 layout 的使用范围 | 为未使用 columns 多分配或错误索引 | `layout.span("TCol")` |
| 以为 112 必须补齐到 128 | TileLayout 支持非 2 次幂 extent | 地址范围被放大，allocator 与 layout 不一致 | extent 和实际 allocation size |
| 用逻辑 `col=130` 查询此 layout | 该 layout 的内层 N extent 只有 112 | flatten 可能给出看似合法但并非预期区域的坐标 | logical shape 和 bounds |
| 把 `TCol` 当成 byte offset | `TCol` 以 element 为单位 | fp16/fp8 数据出现 2x/4x 打包误差 | dtype 和 hardware cell 宽度 |
| 认为 `TileLayout` 会分配 TMEM | 它只描述坐标映射 | layout 正确但实际 allocation 不足或地址基址不同 | `tcgen05.alloc` 和 base address |
| 忽略 `R[...]` | 本布局没有 replica，不代表其他 TMEM layout 也没有 | 下一课的 scale factor 数据只复制一份或不复制 | `layout.replica` |

在推理 kernel 中，这类坐标错误通常表现为：

```text
GEMM 输出部分列正确、部分列整块错位
两个 accumulator 区域互相覆盖或只写到一个区域
epilogue 读取的 TMEM column 与 MMA 写入位置不一致
allocator 留出的 TMEM columns 少于 layout 实际使用的范围
block-scaled GEMM 的 scale 区域覆盖 accumulator
```

### 验证命令和预期输出

运行：

```bash
/Users/saboxu/Documents/ChatGPT/mlc学习/mlc/bin/python tirx_tmem_accumulator_layout.py
```

本机 TVM `0.26.dev246` 实测输出：

```text
layout: T.TileLayout(T.S[(2, 128, 112):(112 @ Axis.TCol, 1 @ Axis.TLane, 1 @ Axis.TCol)])
size: 28672
span TLane: 128
span TCol: 224
(0, 0, 0) -> {"TLane": 0, "TCol": 0}
(0, 17, 30) -> {"TLane": 17, "TCol": 30}
(1, 17, 30) -> {"TLane": 17, "TCol": 142}
(1, 127, 111) -> {"TLane": 127, "TCol": 223}
linear 16270 -> {"TLane": 17, "TCol": 142}
logical == linear: True
```

本机运行边界：

```text
可以运行: TileLayout 构造、size / span 查询、apply() 坐标计算
不能运行: tcgen05.mma、tcgen05.ld 以及需要 Blackwell 的实际 kernel
```

### 本课结论

这个 layout 可以用一行记住：

```text
TLane = lane
TCol  = 112 * a + col
```

它描述的是：

```text
两个逻辑区域
每个区域 128 个 Lane、112 个 columns
两个区域在 TCol 轴上首尾相接
总计覆盖 128 x 224 个 TMEM cell
```

最重要的工程边界是：

```text
layout 只描述“逻辑元素放哪里”，
不决定 a 的业务语义，也不负责 TMEM 分配和同步。
```

### 自测题

#### 1. 逻辑坐标 `(0, 17, 30)` 映射到哪里？

答：

```text
TLane = 17
TCol  = 112 * 0 + 30
      = 30
```

结果是 `{"TLane": 17, "TCol": 30}`。

#### 2. 逻辑坐标 `(1, 17, 30)` 为什么和上一题相差 112 个 columns？

答：它属于外层区域 `a=1`。外层 iter 的 stride 是：

```text
112 @ TCol
```

因此：

```text
TCol = 112 * 1 + 30 = 142
```

#### 3. 这个 layout 使用了多少 TMEM columns？

答：使用：

```text
2 * 112 = 224
```

个 element columns，范围是 `[0, 224)`；不是 256，也不是完整的 512。

#### 4. extent 为 112 是否需要补齐到 128？

答：不需要。`TileLayout` 不要求 extent 是 2 的幂。112 是合法的
column 区域大小，是否补齐取决于具体 kernel 的 allocator 和对齐选择。

#### 5. `TileLayout` 构造完成后，是否已经为 TMEM 分配了空间？

答：没有。它只描述逻辑坐标到 `TLane / TCol` 的映射。实际 TMEM
allocation、基址和生命周期仍由 `tcgen05.alloc`、地址转换以及
`tcgen05.dealloc` 等机制负责。

## 十四、Scale Factor 的 replication：`R[4 : 32@TLane]`

### 本次讲解位置

```text
本次讲解位置
章节：chapter_tirx_layout_api
小节：Scale Factor 布局
知识点：32 x sf_per_mma atom 中 R[4 : 32@TLane] 的四份副本
上次：2 x 128 x 112 accumulator 映射到 TLane / TCol
下次：tmem_datapath_layout 的 datapath / rows / cols 参数
PTX：tcgen05.cp.32x128b.warpx4 的目标布局；本课不讨论指令编码
```

上一课描述的是一个一对一的 accumulator layout：

```text
每个逻辑 accumulator 元素
-> 一个 TLane / TCol 坐标
```

Scale factor 不是一对一。Block-scaled MMA 需要同一组逻辑 scale factors
被多个 TMEM partition 看见，因此要把它的物理数据复制几份。本课只讲
这个复制模式：

```python
scale = TileLayout(
    S[(32, sf_per_mma) : (1@TLane, 1@TCol)]
    + R[4 : 32@TLane]
)
```

目标是回答：

```text
R[4 : 32@TLane] 到底复制几次？
每个逻辑坐标会出现哪些物理 TLane？
为什么 apply() 没有自动返回四份？
```

### 先建立心智模型

先暂时只看 scale factor 的一个基础区域：

```text
32 行 scale
x
sf_per_mma 个连续的 8-bit scale 元素
```

设：

```text
SF_PER_MMA = 4
```

基础 shard 是：

```text
S[(32, 4) : (1@TLane, 1@TCol)]
```

它只覆盖：

```text
TLane = 0..31
TCol  = 0..3
```

可以画成：

```text
partition 0
TLane  0  [s0 s1 s2 s3]
TLane  1  [s0 s1 s2 s3]
...
TLane 31  [s0 s1 s2 s3]
```

可是 TMEM 的 128 条 Lane 被分成四个 32-lane window：

```text
partition 0: TLane   0..31
partition 1: TLane  32..63
partition 2: TLane  64..95
partition 3: TLane  96..127
```

如果 scale factor 只存在于 partition 0，其余三个 partition 在本地
窗口里看不到它。解决方式是沿 `TLane` 再放置三份物理副本：

```text
copy 0: 原位置
copy 1: TLane + 32
copy 2: TLane + 64
copy 3: TLane + 96
```

这就是：

```python
R[4 : 32@TLane]
```

要强调：

```text
逻辑上仍然只有一个 scale factor；
物理 TMEM 中出现四份相同字节。
```

### 拆解 `R[4 : 32@TLane]`

`R` 表示 replica。这个 iter 的语法是：

```text
R[extent : stride @ axis]
```

因此：

| 部分 | 值 | 含义 |
|---|---:|---|
| extent | `4` | 枚举四个 `q` |
| stride | `32` | 每次在目标轴上前进 32 |
| axis | `TLane` | 复制沿 TMEM Lane 轴发生 |

枚举结果是：

```text
q = 0 -> offset = 0 * 32 = 0
q = 1 -> offset = 1 * 32 = 32
q = 2 -> offset = 2 * 32 = 64
q = 3 -> offset = 3 * 32 = 96
```

所以对于基础坐标：

```text
TLane = r
TCol  = s
```

四份物理坐标是：

```text
TLane = r + 32 * q
TCol  = s
q     in {0, 1, 2, 3}
```

`q=0` 对应的就是 base 本身，因此总共是四份，不是“base 再加四份”。

### 不要把重复出现的数字混成同一种含义

当 `sf_per_mma=4` 时，layout 中会出现两组容易混淆的 `32` 和 `4`：

```python
S[(32, sf_per_mma) : (1@TLane, 1@TCol)] + R[4 : 32@TLane]
     ^       ^                                     ^
     |       |                                     |
     |       +-- 每个 TLane 上的 4 个 TCol 元素    |
     |                                             |
     +-- 基础 shard 有 32 条 TLane                 |
                                                   |
                                                   +-- 四份物理副本
```

具体区分：

| 数字 | 所在位置 | 作用 |
|---:|---|---|
| `32` | `S[...]` 第一个 extent | 基础 shard 的 `TLane` 行数 |
| `4` | `S[...]` 第二个 extent | 每个 Lane 上的 `TCol` 元素数 |
| `4` | `R[...]` 的 extent | replica 数量 |
| `32` | `R[...]` 的 stride | 相邻副本在 `TLane` 上相隔多少 |

它们恰好重复出现，但不是同一个维度的参数。

### 完整可运行代码

文件名：

```text
tirx_scale_factor_replication.py
```

完整代码：

```python
from tvm.tirx.layout import R, S, TCol, TileLayout, TLane

SF_PER_MMA = 4


def enumerate_scale_copies(layout, r, s):
    """Enumerate the four physical copies for one logical (r, s)."""
    base = dict(layout.apply(r, s))
    replica = layout.replica[0]
    axis = replica.axis.name
    copies = []

    for q in range(int(replica.extent)):
        coord = dict(base)
        coord[axis] += q * int(replica.stride)
        copies.append((q, coord))

    return base, copies


def main():
    layout = TileLayout(
        S[(32, SF_PER_MMA) : (1 @ TLane, 1 @ TCol)]
        + R[4 : 32 @ TLane]
    )

    print("layout:", layout)
    print("shard:", layout.shard)
    print("replica:", layout.replica)
    print("size:", layout.size())
    print("span TLane:", layout.span("TLane"))
    print("span TCol:", layout.span("TCol"))

    for r, s in [(0, 0), (3, 2), (31, 3)]:
        base, copies = enumerate_scale_copies(layout, r, s)
        print(f"logical ({r}, {s})")
        print("  base:", base)
        for q, coord in copies:
            print(f"  q={q}:", coord)
        print("  hardware:", {"TCol": s // 4, "byte": s % 4})


if __name__ == "__main__":
    main()
```

### 逐段解释

#### 构造基础 shard

```python
S[(32, SF_PER_MMA) : (1 @ TLane, 1 @ TCol)]
```

两个 iter 是：

```text
(32, 1, TLane)
(4,  1, TCol)
```

因此基础映射只回答：

```text
TLane = r
TCol  = s
```

其中：

```text
r in [0, 32)
s in [0, 4)
```

#### 添加 replica

```python
R[4 : 32 @ TLane]
```

它不依赖逻辑坐标 `(r, s)`，而是在基础坐标之外枚举额外物理位置。

#### 查询 `layout.apply(r, s)`

```python
base = dict(layout.apply(r, s))
```

这里再次强调上一课的结论：

```text
layout.apply() 只返回 D(x) + O
不会枚举 R
```

所以 `(3,2)` 的 `apply()` 返回：

```text
{"TCol": 2, "TLane": 3}
```

不是自动返回四个坐标。

#### 读取 replica 描述

```python
replica = layout.replica[0]
```

打印结果是：

```text
T.Iter(4, 32, "TLane")
```

即：

```text
extent = 4
stride = 32
axis   = TLane
```

#### 枚举 `q`

```python
for q in range(int(replica.extent)):
    coord = dict(base)
    coord[axis] += q * int(replica.stride)
```

这一步只是教学代码中手工枚举 layout 声明的副本：

```text
coord[TLane] = r + q * 32
```

真实 tile 操作负责按 `layout.replica` 生成或搬运这些副本。`R` 本身
不会在 Python 里自动修改数据。

### 数值追踪：逻辑坐标 `(3, 2)`

先算 base：

```text
TLane = r = 3
TCol  = s = 2
```

再算四个副本：

| q | TLane 计算 | TLane | TCol |
|---:|---|---:|---:|
| 0 | `3 + 32 * 0` | 3 | 2 |
| 1 | `3 + 32 * 1` | 35 | 2 |
| 2 | `3 + 32 * 2` | 67 | 2 |
| 3 | `3 + 32 * 3` | 99 | 2 |

对应 partition：

```text
TLane  3 -> partition 0
TLane 35 -> partition 1
TLane 67 -> partition 2
TLane 99 -> partition 3
```

四个副本的 `TCol` 都保持为 `2`。Replica 只改变 `TLane`。

### 边界追踪：逻辑坐标 `(31, 3)`

这是基础 shard 中最后一个合法坐标：

```text
r = 31
s = 3
```

四个副本是：

```text
q=0 -> TLane 31
q=1 -> TLane 63
q=2 -> TLane 95
q=3 -> TLane 127
```

它们分别是四个 partition 的最后一个 Lane。结果没有越界：

```text
TLane 127 是合法坐标
TLane 128 才是这个 layout 的越界位置
```

### 8-bit scale 的硬件打包

课程在这里用了一个容易忽略的细节：

```text
TCol 仍然以 buffer element 为单位。
```

当 scale factor 是 8-bit 时，四个连续元素打包进一个 32-bit hardware
TMEM cell：

```text
hardware TCol = s // 4
byte position = s % 4
```

这里 `sf_per_mma=4`，所以 `s=0..3` 正好组成一个 hardware cell：

| 逻辑 s | hardware TCol | byte |
|---:|---:|---:|
| 0 | 0 | 0 |
| 1 | 0 | 1 |
| 2 | 0 | 2 |
| 3 | 0 | 3 |

对于 `(3,2)`：

```text
TLane         = 3
logical TCol  = 2
hardware TCol = 2 // 4 = 0
byte          = 2 % 4 = 2
```

复制后，同一个 byte 出现在：

```text
(TLane 3,  hardware TCol 0, byte 2)
(TLane 35, hardware TCol 0, byte 2)
(TLane 67, hardware TCol 0, byte 2)
(TLane 99, hardware TCol 0, byte 2)
```

这就是 block-scaled MMA 能从四个本地 32-lane window 读取同一组 scale
factor 的原因。

### `size()` 和 `span()` 为什么都打印 128

本课输出中：

```text
size: 128
span TLane: 128
```

两个 `128` 的含义不同：

```text
size = 32 * 4 = 128
       -> 逻辑元素数

span TLane = 32 + 3 * 32 = 128
           -> replica 后实际覆盖的 TLane 坐标范围
```

如果去掉 replica：

```python
TileLayout(S[(32, 4) : (1@TLane, 1@TCol)])
```

那么：

```text
size       = 128
span TLane = 32
TCol span  = 4
```

所以两个 `128` 相等只是本组参数造成的巧合，不代表 replica 没有增加
物理坐标范围。

### 数据路径与执行边界

真实 block-scaled 路径可以概括为：

```text
scale factors:
GMEM -> SMEM
     -> tcgen05.cp.32x128b.warpx4
     -> TMEM 四个 32-lane windows
     -> block-scaled tcgen05.mma 从本地 partition 读取
```

这里有三层不同责任：

| 层 | 负责什么 |
|---|---|
| `TileLayout + R[...]` | 描述基础位置和副本目标坐标 |
| `tcgen05.cp ... .warpx4` | 实际把数据搬运并复制到 TMEM |
| `tcgen05.mma` | 按 TMEM 地址读取 scale factors 并执行 MMA |

`layout.apply()` 只完成第一层中的 base coordinate 查询。它不会：

```text
复制寄存器或 SMEM 数据
发射 tcgen05.cp
保证复制已经完成
启动 block-scaled tcgen05.mma
保证 MMA 的同步契约
```

具体 cp 的 payload 宽度由完整 tile layout 决定。本课这个
`32 x sf_per_mma` atom 只描述复制模式，不等于整条硬件搬运指令的全部
数据宽度。

### 常见错误与可观察症状

| 错误 | 为什么错 | 可观察症状 | 优先检查 |
|---|---|---|---|
| 只用 `layout.apply(r,s)` | 它只返回 base coordinate | partition 1..3 读到零、旧数据或错误 scale | 是否枚举了 `layout.replica` |
| 把 replica 数理解成 1 + 4 | `q=0` 已经是 base | 多发一份，目标可能越界或覆盖相邻数据 | `q in {0,1,2,3}` |
| 枚举 `q=1..4` | 跳过了 base 并多出 `+128` | 第一份缺失，最后一份超出 128 Lane | replica extent 和 stride |
| 将复制步长应用到 `TCol` | `R` 的 axis 是 `TLane` | scale 出现在错误列，四个 partition 仍没有本地副本 | `R[...]` 的 axis |
| 认为四个逻辑 scale 就是四个 hardware columns | 8-bit 四个元素打包进一个 32-bit cell | TCol 被放大四倍，byte 地址错误 | `s // 4`、`s % 4` |
| 认为 `span TLane = 128` 表示逻辑元素增加四倍 | `size()` 仍是 128 | 错误分配逻辑 buffer 或在 GMEM 中重复存四份 | `size()` 与 `span()` 的区别 |
| 把 atom 当成整条 `.32x128b` payload | atom 只是局部复制模式 | cp 宽度、列数和外层布局不一致 | 外层 M / K-block iters |

在量化推理 kernel 中，这类错误通常表现为：

```text
只有部分 warp partition 的 block-scaled GEMM 结果正确
partition 1..3 的 scale 全部相同或为零
FP8 / FP4 GEMM 的输出误差突然增大
scale factor 跨 byte 边界打包，出现数量级错误
tcgen05.mma 读到尚未完成的 tcgen05.cp 数据
```

### 验证命令和预期输出

运行：

```bash
/Users/saboxu/Documents/ChatGPT/mlc学习/mlc/bin/python tirx_scale_factor_replication.py
```

本机 TVM `0.26.dev246` 实测输出：

```text
layout: T.TileLayout(T.S[(32, 4):(1 @ Axis.TLane, 1 @ Axis.TCol)] + T.R[4:32 @ Axis.TLane])
shard: (T.Iter(32, 1, "TLane"), T.Iter(4, 1, "TCol"))
replica: (T.Iter(4, 32, "TLane"),)
size: 128
span TLane: 128
span TCol: 4
logical (0, 0)
  base: {'TCol': 0, 'TLane': 0}
  q=0: {'TCol': 0, 'TLane': 0}
  q=1: {'TCol': 0, 'TLane': 32}
  q=2: {'TCol': 0, 'TLane': 64}
  q=3: {'TCol': 0, 'TLane': 96}
  hardware: {'TCol': 0, 'byte': 0}
logical (3, 2)
  base: {'TCol': 2, 'TLane': 3}
  q=0: {'TCol': 2, 'TLane': 3}
  q=1: {'TCol': 2, 'TLane': 35}
  q=2: {'TCol': 2, 'TLane': 67}
  q=3: {'TCol': 2, 'TLane': 99}
  hardware: {'TCol': 0, 'byte': 2}
logical (31, 3)
  base: {'TCol': 3, 'TLane': 31}
  q=0: {'TCol': 3, 'TLane': 31}
  q=1: {'TCol': 3, 'TLane': 63}
  q=2: {'TCol': 3, 'TLane': 95}
  q=3: {'TCol': 3, 'TLane': 127}
  hardware: {'TCol': 0, 'byte': 3}
```

本机运行边界：

```text
可以运行: TileLayout / Iter 查询、apply()、手工 replica 枚举
不能运行: tcgen05.cp、tcgen05.mma 以及需要 Blackwell 的实际 kernel
```

### 本课结论

这一行：

```python
R[4 : 32@TLane]
```

完整含义是：

```text
生成四个副本
副本编号 q = 0, 1, 2, 3
每个副本在 TLane 上增加 q * 32
TCol 不变
q=0 就是 base 本身
```

最重要的工程结论是：

```text
布局声明 replica；
tile 操作生成 replica；
MMA 消费本地 partition 中的 replica。
```

### 自测题

#### 1. `R[4 : 32@TLane]` 产生哪些偏移？

答：

```text
q=0 -> 0
q=1 -> 32
q=2 -> 64
q=3 -> 96
```

总共四份，其中偏移 0 是 base。

#### 2. 逻辑坐标 `(3, 2)` 的四份物理坐标是什么？

答：

```text
(TLane 3,  TCol 2)
(TLane 35, TCol 2)
(TLane 67, TCol 2)
(TLane 99, TCol 2)
```

#### 3. 逻辑坐标 `(31, 3)` 为什么最后一份落在 `TLane 127`？

答：

```text
TLane = 31 + 32 * 3
      = 31 + 96
      = 127
```

它正好是第四个 partition 的最后一个 Lane。

#### 4. 8-bit scale 的逻辑 `s=2` 对应哪个 hardware TCol 和 byte？

答：

```text
hardware TCol = 2 // 4 = 0
byte          = 2 % 4 = 2
```

四个连续 8-bit element `s=0..3` 共用 `hardware TCol=0`。

#### 5. 为什么 `layout.apply(3,2)` 没有返回四个坐标？

答：`apply()` 只返回基础映射 `D(x)+O`，不枚举 replica。四份副本的
描述保存在 `layout.replica` 中，由消费这个 layout 的 tile 操作负责
生成或搬运。

## 十五、`tmem_datapath_layout`：逻辑 M 如何映射到物理 `TLane`

### 本次讲解位置

```text
本次讲解位置
章节：chapter_tirx_layout_api
小节：常用 Layout 构造函数
知识点：tmem_datapath_layout 的 datapath / rows / cols 与 D/F row mapping
上次：scale-factor atom 中 R[4 : 32@TLane] 的四份副本
下次：tcgen05_atom_layout 的 instr_shape / tensor_shape / dtype
PTX：PTX ISA §9.7.16.10.5 的 tcgen05.mma datapath enumeration
```

上一课解决的是“同一份逻辑数据要不要复制到多个 `TLane`”。本课解决另一个
问题：

```text
一个逻辑 accumulator 的第 r 行
应该由 tcgen05.mma 写到哪条物理 TLane？
```

这个映射取决于 MMA 使用的 datapath。TIRx 提供：

```python
tmem_datapath_layout(datapath, rows, cols)
```

它不是分配 TMEM，也不启动 MMA。它根据 datapath 返回一个
`TileLayout`，描述逻辑 `(row, col)` 对应的 `TLane / TCol`。

### 为什么不能全部用恒等映射

最直观的 M=128 layout 是：

```text
TLane = row
TCol  = col
```

这在 `datapath="D"` 下成立。但 M=64 的 non-`.ws` MMA 使用 half
datapath，物理 Lane 不是简单地只写 `0..63`。它把 64 个逻辑 row 分成
四个 16-row 组，分散放进四个 32-lane partition：

```text
逻辑 rows  0..15 -> 物理 TLane   0..15
逻辑 rows 16..31 -> 物理 TLane  32..47
逻辑 rows 32..47 -> 物理 TLane  64..79
逻辑 rows 48..63 -> 物理 TLane  96..111
```

每个 32-lane partition 只使用低 16 条 Lane。

如果这个映射写错，MMA 的写入位置与后续 `tcgen05.ld` 的读取位置就会
不一致。GEMM 结果可能出现整段 16-row 错位，甚至读到未写入的 Lane。

### 当前支持的两种 datapath

当前 TIRx 工厂支持：

| `datapath` | rows / M | MMA 范围 | 映射思想 |
|---|---:|---|---|
| `"D"` | 128 | `cta_group::1` full datapath | `TLane = row` |
| `"F"` | 64 | non-`.ws` half datapath | 四个 16-row slab 分散到 32-lane partitions |

PTX 还定义其他 layout 名称，但当前工厂只实现 `D` 和 `F`。传入 `"A"`、
`"B"`、`"C"`、`"E"`、`"G"` 会报 unknown datapath，而不是自动选择
近似映射。

这里必须区分：

```text
datapath 是 MMA 的硬件行映射契约；
rows / cols 是它要描述的 accumulator 逻辑 shape。
```

`datapath` 和 `rows` 不能随意组合：

```text
"D" 只接受 rows=128
"F" 只接受 rows=64
```

`cols` 是逻辑 column extent。它可以是实际 N tile 大小，例如 112、224、
256；`TCol` 仍以 buffer element 为单位。

### 等价的显式 TileLayout

工厂没有引入新的 layout 类型。它只是生成常见的 `TileLayout`。

#### Datapath D

```python
tmem_datapath_layout("D", 128, cols)
```

等价于：

```python
TileLayout(
    S[(128, cols) : (1@TLane, 1@TCol)]
)
```

映射：

```text
TLane = row
TCol  = col
```

#### Datapath F

```python
tmem_datapath_layout("F", 64, cols)
```

等价于：

```python
TileLayout(
    S[(4, 16, cols) : (32@TLane, 1@TLane, 1@TCol)]
)
```

逻辑 row 先分解成：

```text
w     = row // 16
intra = row % 16
```

再计算：

```text
TLane = 32 * w + intra
TCol  = col
```

其中：

```text
w     in [0, 4)
intra in [0, 16)
```

这是“四个 slab”的来源。

### 完整可运行代码

文件名：

```text
tirx_tmem_datapath_layout.py
```

完整代码：

```python
from tvm.tirx.layout import tmem_datapath_layout


def show(name, layout, shape, coords):
    print(f"{name} layout:", layout)
    print(f"{name} shard:", layout.shard)
    print(f"{name} size:", layout.size())
    print(f"{name} span TLane:", layout.span("TLane"))
    print(f"{name} span TCol:", layout.span("TCol"))
    for r, c in coords:
        print(f"{name} ({r}, {c}) ->", layout.apply(r, c, shape=shape))


def expect_error(datapath, rows, cols):
    try:
        tmem_datapath_layout(datapath, rows, cols)
    except ValueError as exc:
        print(f"error ({datapath}, {rows}, {cols}):", exc)


def main():
    d = tmem_datapath_layout("D", 128, 112)
    f = tmem_datapath_layout("F", 64, 112)

    show(
        "D",
        d,
        [128, 112],
        [(0, 0), (15, 7), (16, 9), (127, 31)],
    )
    show(
        "F",
        f,
        [64, 112],
        [(0, 0), (15, 7), (16, 9), (31, 13), (32, 17), (63, 23)],
    )

    expect_error("D", 64, 16)
    expect_error("F", 128, 16)
    expect_error("A", 128, 16)


if __name__ == "__main__":
    main()
```

### 逐段解释

#### `datapath="D"`

```python
d = tmem_datapath_layout("D", 128, 112)
```

工厂检查：

```text
datapath 必须是 D
rows 必须等于 128
```

然后返回：

```text
T.TileLayout(
    T.S[(128, 112) : (1 @ Axis.TLane, 1 @ Axis.TCol)]
)
```

这就是 M=128 的 full datapath，逻辑 row `0..127` 对应物理
`TLane 0..127`。

#### `datapath="F"`

```python
f = tmem_datapath_layout("F", 64, 112)
```

工厂检查：

```text
datapath 必须是 F
rows 必须等于 64
```

返回的 shard 有三个 iter：

```text
T.Iter(4,   32, "TLane")
T.Iter(16,   1, "TLane")
T.Iter(112,  1, "TCol")
```

三个 extent 是 `(4, 16, 112)`，但调用方的逻辑 shape 仍是 `(64, 112)`。
`TileLayout` 的 shard extent 与逻辑 shape 不需要逐个相等，因为 `apply()`
可以先 flatten 逻辑 `(64,112)`，再按内部 shard 分解。

#### 为什么 `F.apply` 要传 `shape`

下面这个调用是不完整的：

```python
f.apply(row, col)
```

`F` 的内部 shard 有三个 iter，而这里只提供两个 coordinate，因此会报：

```text
Coordinate size must match the number of shard axes: 2 vs. 3
```

应该传逻辑 shape：

```python
f.apply(row, col, shape=[64, 112])
```

或者自己提供三个 shard coordinate：

```python
f.apply(row // 16, row % 16, col)
```

第一种调用保留了逻辑 `(row, col)` 语义，也是课程示例使用的方式。

#### 为什么 `rows` 不能写错

```python
tmem_datapath_layout("F", 128, 112)
```

会得到：

```text
datapath='F' expects rows=64, got 128
```

因为 F 描述的是一个 M=64 的 half datapath。给它 128 行并不能自动变成
D，也不能自动拆成两个 F tile。

### D 与 F 的逐 row 对比

对同一个 `TCol=col`，两种 datapath 的部分映射如下：

| 逻辑 row | D 的 `TLane` | F 的 `TLane` | 说明 |
|---:|---:|---:|---|
| 0 | 0 | 0 | 第一个 slab 起点 |
| 15 | 15 | 15 | 第一个 slab 终点 |
| 16 | 16 | 32 | F 开始第二个 slab |
| 31 | 31 | 47 | 第二个 slab 终点 |
| 32 | 32 | 64 | F 开始第三个 slab |
| 47 | 47 | 79 | 第三个 slab 终点 |
| 48 | 48 | 96 | F 开始第四个 slab |
| 63 | 63 | 111 | F 最后一个合法 row |
| 64 | 64 | 非法 | F 只有 64 行 |
| 127 | 127 | 非法 | D 的最后一行 |

一个具体计算，F 的 `row=45`：

```text
w     = 45 // 16 = 2
intra = 45 % 16 = 13

TLane = 32 * 2 + 13
      = 64 + 13
      = 77
```

所以：

```text
逻辑 (45, col)
-> 物理 (TLane=77, TCol=col)
```

### F 的 Lane hole

F 只使用：

```text
0..15
32..47
64..79
96..111
```

没有使用：

```text
16..31
48..63
80..95
112..127
```

因此：

```text
logical rows  = 4 * 16 = 64
active lanes  = 4 * 16 = 64
span TLane    = 112
```

`span("TLane")=112` 表示最高使用坐标是 `111`，不是表示
`TLane 0..111` 全部被使用。

这正是 span 与 active set 的区别：

```text
span   描述坐标上界包络
layout 的映射结果决定哪些坐标真正被使用
```

### 它与 scale-factor replica 的区别

上一课的：

```python
R[4 : 32@TLane]
```

表示同一个逻辑 scale factor 被复制到四个物理位置。

本课 F 的：

```text
S[(4, 16, cols) : (32@TLane, 1@TLane, 1@TCol)]
```

不是复制。不同的逻辑 row 被分配到不同的物理 Lane：

```text
row 0 和 row 16 是不同的逻辑数据
它们分别去 TLane 0 和 TLane 32
```

对比：

| 机制 | 逻辑数据是否重复 | 目的 |
|---|---|---|
| `R[...]` replica | 是 | 多个 partition 获得同一份数据 |
| F datapath | 否 | 把一个 M=64 tile 分散到 half datapath |

### 数据路径与契约边界

真实路径中：

```text
tcgen05.mma
  -> 按 datapath layout 写 TMEM accumulator

tcgen05.ld / tcgen05.st atom
  -> 使用匹配的 register tile layout 读 TMEM
```

`tmem_datapath_layout` 只负责生产端的 row -> Lane 映射描述。它不负责：

```text
选择 tcgen05.mma 的 datapath 指令形式
分配 TMEM columns
提交或等待 MMA
生成 tcgen05.ld 的 register mapping
验证 producer 与 consumer 的完整等价性
```

工程上必须满足：

```text
MMA 的 datapath
== tmem_datapath_layout 的 datapath
== tcgen05.ld atom 期望的 TMEM row mapping
```

在 GEMM 中，D/F 不匹配可能表现为：

```text
输出 rows 16..31、32..47、48..63 出现整段错位
M=64 结果只填满错误的一半 Lane
epilogue 从空白 Lane 读取数据
同一次 MMA 的部分 warp partition 结果正确、其他 partition 错误
```

在 block-scaled 量化 GEMM 中，还会进一步导致 scale factor 的本地窗口
与 accumulator row 对应错误，产生整块数值偏差。

### 常见错误与可观察症状

| 错误 | 为什么错 | 可观察症状 | 优先检查 |
|---|---|---|---|
| `("D", 64, cols)` | D 是 M=128 full datapath | 工厂抛出 rows 不匹配 | datapath 与 M |
| `("F", 128, cols)` | F 只支持 M=64 | 工厂抛出 rows 不匹配 | 是否需要用 D |
| 用 `f.apply(row, col)` | F shard 有三个 iter | 报 coordinate size 2 vs. 3 | 是否传 logical shape |
| 认为 F 使用 `TLane 0..63` | F 分散到四段 16-row slab | 后三个 slab 的读取地址错误 | `r // 16`、`r % 16` |
| 认为 F 的 span 112 表示连续使用 112 Lane | 四段之间和末尾有 hole | 误分配或读取未使用 Lane | active lane 集合 |
| 把 F 当成 scale replica | F 映射不同逻辑 row，不复制 | 数据被错误重复或 row 被覆盖 | `R[...]` 与 S[...] 的区别 |
| D/F 与 MMA 形式不一致 | row mapping 是硬件契约 | MMA 写入与 readback 不匹配 | PTX datapath 与 lowering |
| 把 `cols` 当成 hardware column 数 | `TCol` 以 buffer element 为单位 | fp16/fp8 打包地址出现倍数错误 | dtype 与 hardware cell |

### 验证命令和预期输出

运行：

```bash
/Users/saboxu/Documents/ChatGPT/mlc学习/mlc/bin/python tirx_tmem_datapath_layout.py
```

本机 TVM `0.26.dev246` 实测输出：

```text
D layout: T.TileLayout(T.S[(128, 112):(1 @ Axis.TLane, 1 @ Axis.TCol)])
D shard: (T.Iter(128, 1, "TLane"), T.Iter(112, 1, "TCol"))
D size: 14336
D span TLane: 128
D span TCol: 112
D (0, 0) -> {"TCol": 0, "TLane": 0}
D (15, 7) -> {"TCol": 7, "TLane": 15}
D (16, 9) -> {"TCol": 9, "TLane": 16}
D (127, 31) -> {"TCol": 31, "TLane": 127}
F layout: T.TileLayout(T.S[(4, 16, 112):(32 @ Axis.TLane, 1 @ Axis.TLane, 1 @ Axis.TCol)])
F shard: (T.Iter(4, 32, "TLane"), T.Iter(16, 1, "TLane"), T.Iter(112, 1, "TCol"))
F size: 7168
F span TLane: 112
F span TCol: 112
F (0, 0) -> {"TCol": 0, "TLane": 0}
F (15, 7) -> {"TCol": 7, "TLane": 15}
F (16, 9) -> {"TCol": 9, "TLane": 32}
F (31, 13) -> {"TCol": 13, "TLane": 47}
F (32, 17) -> {"TCol": 17, "TLane": 64}
F (63, 23) -> {"TCol": 23, "TLane": 111}
error (D, 64, 16): tmem_datapath_layout: datapath='D' expects rows=128, got 64
error (F, 128, 16): tmem_datapath_layout: datapath='F' expects rows=64, got 128
error (A, 128, 16): tmem_datapath_layout: unknown datapath 'A'; supported: ['D', 'F']
```

本机运行边界：

```text
可以运行: datapath 工厂、TileLayout 查询、D/F 坐标推导和参数校验
不能运行: 真实 tcgen05.mma 写入和需要 Blackwell 的 kernel
```

### 本课结论

D：

```text
rows = 128
TLane = row
TCol  = col
```

F：

```text
rows = 64
w     = row // 16
intra = row % 16
TLane = 32 * w + intra
TCol  = col
```

最重要的边界是：

```text
datapath 是 MMA 写入 TMEM 的 row mapping 契约。
工厂只生成布局描述，不执行 MMA，也不代替 tcgen05.ld 的匹配验证。
```

### 自测题

#### 1. `datapath="D"`、`rows=112` 会发生什么？

答：抛出参数错误。D 要求 `rows=128`，不能通过缩小 rows 得到 M=112
的 D datapath。

#### 2. `datapath="F"`、逻辑 `row=45` 映射到哪条 `TLane`？

答：

```text
w     = 45 // 16 = 2
intra = 45 % 16 = 13
TLane = 32 * 2 + 13 = 77
```

所以映射到 `TLane=77`。

#### 3. F 的 `span("TLane")` 为什么是 112？

答：F 最高使用 `TLane=111`，坐标上界加一得到 112。它不表示
`0..111` 都连续被使用，四段 slab 之间仍有 hole。

#### 4. F 是否把同一行数据复制到四个 slab？

答：不是。F 把不同的逻辑 row 分配到四个 slab，每个逻辑 row 只有一个
物理位置。复制是 `R[...]` replica 的语义。

#### 5. 为什么 `f.apply(16, 9)` 不能直接调用？

答：F 的内部 shard 有三个 iter，而两个参数只提供两个坐标。应传逻辑
shape，例如 `f.apply(16, 9, shape=[64, 112])`。

## 十六、`tcgen05_atom_layout`：从 TMEM atom 到线程寄存器

### 本次讲解位置

```text
本次讲解位置
章节：chapter_tirx_layout_api
小节：常用 Layout 构造函数
知识点：tcgen05_atom_layout 的 instr_shape / tensor_shape / dtype
上次：tmem_datapath_layout 的 D/F datapath
下次：wg_local_layout 的行到 tid_in_wg 映射
PTX：PTX ISA §9.7.18.2.3 Data Movement Shape / §9.7.18.8.3 tcgen05.ld
```

上一课描述的是生产者端：

```text
tcgen05.mma
-> 按 datapath D/F 把 accumulator 写入 TMEM
```

本课描述消费者端的一半：

```text
tcgen05.ld
-> 把一个 TMEM fragment 搬到 warpgroup 的寄存器
```

`tcgen05_atom_layout` 返回的 `TileLayout` 不是 TMEM 地址，而是寄存器侧的
分布。它回答：

```text
逻辑坐标 (row, col)
由哪个 warp、哪个 lane 的哪个局部元素位置持有？
```

### 先建立心智模型

`tcgen05.ld` 和 `tcgen05.st` 不是由单个线程搬运整个矩阵，而是由 warp
执行 collective data movement：

```text
每个 warp 操作自己可见的一部分 TMEM lanes
每个 lane 从若干连续 TMEM columns 读取数据
读出的 b32 值进入该 lane 的寄存器
```

PTX 用两个 qualifier 描述一次搬运：

```text
.shape  一次 atom 的基础形状
.num    沿 column 方向重复多少次
```

例如：

```text
tcgen05.ld.sync.aligned.16x128b.x4.b32
```

可以拆成：

```text
.16x128b  基础 atom: 16 lanes x 128 bits
.x4       重复 4 次
.b32      每个寄存器是 32-bit
```

`tcgen05_atom_layout()` 接收逻辑 fragment 形状后，会反推出应当使用的
`.xN`，并在 `TileLayout` 中描述结果如何分布到 `wid_in_wg / laneid / m`。

### 三个参数分别控制什么

```python
tcgen05_atom_layout(instr_shape, tensor_shape, dtype)
```

#### `instr_shape`

它是一个 PTX data-movement atom 的基础形状，格式为：

```text
lane x size-in-bits
```

当前 TIRx 支持：

| `instr_shape` | 每个 warp 访问的 lanes | 每个 lane 每次搬运的 bits |
|---|---:|---:|
| `"32x32b"` | 32 | 32 |
| `"16x64b"` | 16 | 64 |
| `"16x128b"` | 16 | 128 |
| `"16x256b"` | 16 | 256 |

`instr_shape` 不表示整个矩阵的 M、N，也不表示最终 `.xN`。它只定义一次
基础搬运的粒度。

#### `tensor_shape`

它是逻辑 register fragment 的完整形状，单位为 **buffer element**：

```text
tensor_shape = (rows, K)
```

当前实现允许：

| `instr_shape` | 允许的 `rows` |
|---|---|
| `"32x32b"` | 128 |
| `"16x64b"` | 64 或 128 |
| `"16x128b"` | 64 或 128 |
| `"16x256b"` | 64 或 128 |

`K` 是每行有多少个逻辑元素。函数会根据 `K` 推导 `.xN`，而不是由调用方
直接传 `.xN`。

#### `dtype`

`dtype` 决定一个 32-bit TMEM cell 或寄存器中能容纳多少元素：

```text
32-bit dtype: 1 element per 32-bit cell
16-bit dtype: 2 elements per 32-bit cell
```

当前只接受 16-bit 或 32-bit 类型，例如：

```text
float32, float16, bfloat16
```

因此相同 `instr_shape` 下，fp16 的 `K` 可以是 fp32 的两倍，二者仍使用
相同的 `.xN`。

### `.xN` 如何从 `tensor_shape` 和 `dtype` 推导

定义：

```text
elem_per_32b = 32 / dtype_bits
```

于是：

```text
fp32: elem_per_32b = 1
fp16: elem_per_32b = 2
bf16: elem_per_32b = 2
```

每个 atom repetition 覆盖的 column factor 是：

| `instr_shape` | fp32 column factor | fp16/bf16 column factor |
|---|---:|---:|
| `"32x32b"` | 1 | 2 |
| `"16x64b"` | 2 | 4 |
| `"16x128b"` | 4 | 8 |
| `"16x256b"` | 8 | 16 |

然后：

```text
rep = K / per_rep_column_factor
```

这里的 `rep` 就是 PTX 指令中的 `.xN`。

例如：

```text
16x128b, fp32, K=128

elem_per_32b = 1
per_rep_cols = 4 * 1 = 4
rep = 128 / 4 = 32
-> .16x128b.x32
```

```text
16x128b, fp16, K=256

elem_per_32b = 2
per_rep_cols = 4 * 2 = 8
rep = 256 / 8 = 32
-> .16x128b.x32
```

两者都对应：

```text
tcgen05.ld.sync.aligned.16x128b.x32.b32
```

不同的是，fp16 的两个相邻逻辑 column 可以进入同一个 32-bit 寄存器。

### 合法 `.xN`

推导出的 `rep` 必须属于 PTX 允许的集合：

| `instr_shape` | 允许的 `.xN` |
|---|---|
| `"32x32b"` | `1, 2, 4, 8, 16, 32, 64, 128` |
| `"16x64b"` | `1, 2, 4, 8, 16, 32, 64, 128` |
| `"16x128b"` | `1, 2, 4, 8, 16, 32, 64` |
| `"16x256b"` | `1, 2, 4, 8, 16, 32` |

所以下面的构思不成立：

```text
K=12, instr_shape="16x128b", fp32
per_rep_cols = 4
rep = 3
```

虽然 12 能被 4 整除，但 PTX 没有 `.x3`，构造函数会直接拒绝。

### 完整可运行代码

文件名：

```text
tirx_tcgen05_atom_layout.py
```

完整代码：

```python
from tvm.tirx.layout import tcgen05_atom_layout


def show(label, instr_shape, tensor_shape, dtype, coords):
    layout = tcgen05_atom_layout(instr_shape, tensor_shape, dtype)
    print(f"{label} shape={tensor_shape} dtype={dtype}")
    print("  layout =", layout)
    print("  shard  =", layout.shard)
    for row, col in coords:
        print(f"  ({row}, {col}) ->", layout.apply(row, col, shape=tensor_shape))


def expect_error(label, instr_shape, tensor_shape, dtype):
    try:
        tcgen05_atom_layout(instr_shape, tensor_shape, dtype)
    except ValueError as exc:
        print(f"  error {label}: {exc}")


def main():
    show(
        "fp32",
        "16x128b",
        (64, 128),
        "float32",
        [(0, 0), (1, 1), (15, 15), (19, 101), (63, 127)],
    )
    show(
        "fp16",
        "16x128b",
        (64, 256),
        "float16",
        [(0, 0), (1, 2), (15, 30), (19, 202), (63, 254)],
    )

    print("parameter checks:")
    expect_error("32x32b rows=64", "32x32b", (64, 32), "float32")
    expect_error("bad K", "16x128b", (64, 129), "float32")
    expect_error("unsupported rep", "16x128b", (64, 12), "float32")


if __name__ == "__main__":
    main()
```

### 逐段解释

#### fp32 调用

```python
tcgen05_atom_layout("16x128b", (64, 128), "float32")
```

参数含义是：

```text
基础 atom: 16 lanes x 128 bits
逻辑 fragment: 64 rows x 128 fp32 columns
```

推导：

```text
elem_per_32b = 32 / 32 = 1
per_rep_cols = 4 * 1 = 4
rep = 128 / 4 = 32
```

因此这是一个 `.16x128b.x32` fragment。

返回的 shard 是：

```text
(4, 2, 8, 32, 4)
:(1 @ wid_in_wg, 1, 4 @ laneid, 2, 1 @ laneid)
```

重要 axes：

```text
wid_in_wg  逻辑 row 属于哪个 warp
laneid     warp 内哪个 lane 持有该 row
m          该 lane 的局部寄存器或元素槽位
```

#### fp16 调用

```python
tcgen05_atom_layout("16x128b", (64, 256), "float16")
```

参数含义是：

```text
基础 atom: 16 lanes x 128 bits
逻辑 fragment: 64 rows x 256 fp16 columns
```

推导：

```text
elem_per_32b = 32 / 16 = 2
per_rep_cols = 4 * 2 = 8
rep = 256 / 8 = 32
```

它同样使用 `.16x128b.x32`。

返回的 shard 在 fp32 版本末尾多了一层：

```text
(4, 2, 8, 32, 4, 2)
:(1 @ wid_in_wg, 2, 4 @ laneid, 4, 1 @ laneid, 1)
```

最后这个：

```text
(2, 1, m)
```

表示两个相邻 fp16 元素打包进一个 32-bit 寄存器。

### 用 `16x128b` 追踪一个真实坐标

先定义 fp32 视角的 column：

```text
q = col // elem_per_32b
```

对于 fp32：

```text
q = col
```

对于 fp16：

```text
q = col // 2
half = col % 2
```

`16x128b` 的映射公式是：

```text
wid    = row // 16
lane   = 4 * (row % 8) + (q % 4)
reg32  = ((row // 8) % 2) + 2 * (q // 4)
```

#### 追踪 `(19, 101)`，fp32

```text
wid  = 19 // 16
     = 1

q    = 101

lane = 4 * (19 % 8) + (101 % 4)
     = 4 * 3 + 1
     = 13

reg32 = ((19 // 8) % 2) + 2 * (101 // 4)
      = (2 % 2) + 2 * 25
      = 0 + 50
      = 50
```

结果是：

```text
(row=19, col=101)
-> wid_in_wg=1, laneid=13, m=50
```

这正好解释前面打印出的：

```python
{"laneid": 13, "m": 50, "wid_in_wg": 1}
```

#### 追踪 `(19, 202)`，fp16

```text
wid  = 1

q    = 202 // 2
     = 101

half = 202 % 2
     = 0

lane = 4 * (19 % 8) + (101 % 4)
     = 13

reg32 = ((19 // 8) % 2) + 2 * (101 // 4)
      = 50

m = 2 * reg32 + half
  = 100
```

所以：

```text
(row=19, col=202)
-> wid_in_wg=1, laneid=13, m=100
```

如果读取下一个 fp16 元素：

```text
(row=19, col=203)
-> wid_in_wg=1, laneid=13, m=101
```

这里：

```text
m=100 和 m=101
```

对应同一个 32-bit register 的低半和高半。

### `32x32b` 为什么特殊

`"32x32b"` 的返回路径不同：

```python
TileLayout(S[(128, cols) : (1 @ tid_in_wg, 1@m)])
```

它表示：

```text
128 个 warpgroup 线程各自负责一行
每个线程持有该行的 cols 个元素
```

因此它不显式使用：

```text
wid_in_wg + laneid
```

而是直接使用：

```text
tid_in_wg
```

这也说明 `instr_shape` 选的不只是“每次读多少 bit”，还会选择对应的
thread fragment 组织。

### 与 datapath layout 的关系

本课的 atom layout 和上一课的 datapath layout 位于数据路径的不同阶段：

| 阶段 | API | 描述对象 |
|---|---|---|
| MMA 写 TMEM | `tmem_datapath_layout` | 逻辑 row 到 TMEM `TLane` |
| ld/st 搬运 | `tcgen05_atom_layout` | TMEM fragment 到线程寄存器 |

生产中必须保证：

```text
MMA 写入的 datapath
+ TMEM buffer 的 TileLayout
+ tcgen05.ld/st atom 的读取形状
```

三者相容。atom layout 本身不会重新排列 MMA 写下来的 TMEM rows，也不会
执行 `tcgen05.ld`；它只描述寄存器结果应如何分布。

### 常见错误与可观察症状

| 错误 | 为什么错 | 可观察症状 | 优先检查 |
|---|---|---|---|
| 把 `rows` 当成整个 CTA 的 M | fragment rows 由 atom 类型约束 | `32x32b` rows=64 直接报错 | 是否应使用 `.16x*b` |
| 把 `K` 写成 byte 数 | API 的 column 是 element 数 | 推导出的 rep 大一倍或小一倍 | dtype 和 element 单位 |
| 把 fp16 的 K 写成 fp32 的值 | fp16 每个 cell 容纳两个元素 | 寄存器数量少一半，后半列丢失 | `elem_per_32b` |
| K 不能被 per-rep factor 整除 | 无法形成整数次 atom repeat | 构造函数抛出不可整除错误 | `per_rep_cols` |
| rep 不在 PTX 表中 | PTX 没有任意 `.xN` | 抛出 unsupported rep | 合法 `.xN` 集合 |
| 把 `m` 当成物理 TMEM Col | `m` 是线程局部槽位 | 寄存器索引与 TMEM 地址混淆 | `TLane / TCol` 与 `m` |
| atom shape 与 datapath 不匹配 | 两侧 row/lane 契约不同 | readback 错位、16-row slab 错 | datapath 与 atom 组合 |
| 忘记 16-bit packing 顺序 | 低半和高半顺序错误 | 相邻两列交换或数值拼接错误 | `col // 2`、`col % 2` |

### 验证命令和预期输出

运行：

```bash
/Users/saboxu/Documents/ChatGPT/mlc学习/mlc/bin/python tirx_tcgen05_atom_layout.py
```

本机 TVM `0.26.dev246` 实测输出：

```text
fp32 shape=(64, 128) dtype=float32
  layout = T.TileLayout(T.S[(4, 2, 8, 32, 4):(1 @ Axis.wid_in_wg, 1, 4 @ Axis.laneid, 2, 1 @ Axis.laneid)])
  shard  = (T.Iter(4, 1, "wid_in_wg"), T.Iter(2, 1, "m"), T.Iter(8, 4, "laneid"), T.Iter(32, 2, "m"), T.Iter(4, 1, "laneid"))
  (0, 0) -> {"laneid": 0, "m": 0, "wid_in_wg": 0}
  (1, 1) -> {"laneid": 5, "m": 0, "wid_in_wg": 0}
  (15, 15) -> {"laneid": 31, "m": 7, "wid_in_wg": 0}
  (19, 101) -> {"laneid": 13, "m": 50, "wid_in_wg": 1}
  (63, 127) -> {"laneid": 31, "m": 63, "wid_in_wg": 3}
fp16 shape=(64, 256) dtype=float16
  layout = T.TileLayout(T.S[(4, 2, 8, 32, 4, 2):(1 @ Axis.wid_in_wg, 2, 4 @ Axis.laneid, 4, 1 @ Axis.laneid, 1)])
  shard  = (T.Iter(4, 1, "wid_in_wg"), T.Iter(2, 2, "m"), T.Iter(8, 4, "laneid"), T.Iter(32, 4, "m"), T.Iter(4, 1, "laneid"), T.Iter(2, 1, "m"))
  (0, 0) -> {"laneid": 0, "m": 0, "wid_in_wg": 0}
  (1, 2) -> {"laneid": 5, "m": 0, "wid_in_wg": 0}
  (15, 30) -> {"laneid": 31, "m": 14, "wid_in_wg": 0}
  (19, 202) -> {"laneid": 13, "m": 100, "wid_in_wg": 1}
  (63, 254) -> {"laneid": 31, "m": 126, "wid_in_wg": 3}
parameter checks:
  error 32x32b rows=64: tcgen05_atom_layout '32x32b' expects rows ∈ (128,), got 64
  error bad K: tcgen05_atom_layout cols=129 not divisible by the per-rep column factor 4 for instr_shape='16x128b' dtype=float32; valid cols are k * 4 for k in (1, 2, 4, 8, 16, 32, 64)
  error unsupported rep: tcgen05_atom_layout inferred rep=3 (from cols=12) is not in the PTX Table 49 supported set for 16x128b: (1, 2, 4, 8, 16, 32, 64)
```

本机运行边界：

```text
可以运行: atom layout 构造、rep 推导、register coordinate 查询和参数校验
不能运行: 真实 tcgen05.ld/st 及需要 Blackwell 的 kernel
```

### 本课结论

固定公式：

```text
elem_per_32b = 32 / dtype_bits
per_rep_cols = instr_column_factor * elem_per_32b
rep = K / per_rep_cols
```

固定边界：

```text
instr_shape 定义 atom 粒度
tensor_shape 定义逻辑 fragment 的 rows 和 element columns
dtype 决定 16-bit packing
rep 是推导结果，不是输入参数
```

最重要的区别是：

```text
TMEM TLane / TCol 描述数据在 Tensor Memory 中的位置
atom layout 的 wid_in_wg / laneid / m 描述数据在线程寄存器中的位置
```

### 自测题

#### 1. `tcgen05_atom_layout("16x128b", (64, 256), "float16")` 的 `.xN` 是多少？

答：

```text
elem_per_32b = 32 / 16 = 2
per_rep_cols = 4 * 2 = 8
rep = 256 / 8 = 32
```

所以是 `.16x128b.x32`。

#### 2. 为什么 fp16 的 `K=256` 和 fp32 的 `K=128` 可以使用同一个 `.x32`？

答：因为 fp16 每个 32-bit 位置容纳两个元素，每个 repetition 覆盖 8 个
fp16 columns；fp32 每个位置只容纳一个元素，每个 repetition 覆盖 4 个
fp32 columns。两者的 column 数是 2:1，但 rep 都是 32。

#### 3. `tensor_shape` 的 rows 在 `"32x32b"` 下为什么只能是 128？

答：`"32x32b"` 对应的 thread-row 组织要求一个 warpgroup 的 128 个线程
各负责一行。因此逻辑 fragment 的 rows 必须是 128。

#### 4. `m=100` 和 `m=101` 在 fp16 下一定属于不同寄存器吗？

答：不一定。`m` 是 element 级局部坐标。对 fp16：

```text
register = m // 2
half     = m % 2
```

`m=100` 和 `m=101` 对应同一个 register 的低半和高半。

#### 5. 为什么 `K=12`、`"16x128b"`、`"float32"` 不合法？

答：

```text
per_rep_cols = 4
rep = 12 / 4 = 3
```

PTX 没有 `.x3`，合法重复次数只能从表中的 `.x1, .x2, .x4, ...` 选择。

## 十七、当前进度

`chapter_tirx_layout_api` 的知识点：

```text
[x] TileLayout 的 S[...]、R[...] 与 offset
[x] 命名轴：laneid / warpid / m / TLane / TCol
[x] apply() 的三种输入形式与 flatten / decompose
[x] TMEM accumulator layout 示例
[x] scale-factor layout 中的 replication
[x] tmem_datapath_layout：D/F datapath
[x] tcgen05_atom_layout 的 instr_shape / tensor_shape / dtype
[ ] wg_local_layout
[ ] ComposeLayout 与 shared-memory swizzle
```

本篇目前完成前七项。已经覆盖：

```text
TileLayout 可以表示 lane、warp、register、TMEM 等命名轴坐标
S[...] 定义逻辑索引相关的基础映射
R[...] 定义与逻辑索引无关的额外物理副本
固定 offset 平移所有物理坐标，但不产生副本
每个 iter 是 (extent, stride, axis)
多个 iter 对同一 axis 的贡献需要相加
apply() 只返回 D(x) + O，不枚举 replica
layout.shard / layout.replica / layout.offset 暴露三个组成部分
用 (1, 3)、shape [8, 16] 完整追踪了 flatten、decompose 和坐标合成
R[2 : 4@warpid] 为示例元素生成 warpid 5 和 9 两个位置
axis 是坐标空间名称，不同 axis 上的相同整数不是同一物理位置
laneid 表示 warp 内执行 thread 的 lane ID
warpid 表示 CTA 内 warp ID，和 warpgroup 内的 wid_in_wg 不同
m 是默认线性轴，其含义由 global / shared / register-backed local scope 决定
TLane 表示 TMEM 的硬件 Lane 坐标
TCol 以 buffer element 为单位，dtype 决定多少个元素打包进 hardware Col
同一个 axis 的多个 shard iter 贡献必须相加
用 frag、TMEM 和普通 m layout 对比了三种坐标空间
apply() 支持 logical + shape、linear、shard coordinate 三种入口
logical + shape 会执行 flatten 和 decompose
linear coordinate 只执行 decompose 和 coordinate 合成
shard coordinate 只执行 stride / axis / offset
单参数 apply(coord) 在当前实现中按 linear coordinate 处理
TileLayout 不保存调用方的 logical shape
逻辑 shape [8, 16] 下 (1, 3) flatten 为 19
逻辑 shape [16, 8] 下 (1, 3) flatten 为 11
同一个逻辑坐标在不同 logical shape 下会产生不同 warp / lane coordinate
可以用三入口等价性分层定位 flatten、decompose、stride / axis 错误
用 (2, 128, 112) accumulator layout 完成 TMEM 坐标推导
TMEM physical coordinate 使用 TLane / TCol，TLane 范围是 [0, 128)
(2, 128, 112) 将两个 128 x 112 区域并排放置在 TCol 0..223
outer region 的 112@TCol stride 产生 TCol = 112 * a + col
TMEM layout 不要求 extent 是 2 的幂，112 column 区域无需补齐到 128
layout.apply() 返回基础 TLane / TCol，不分配 TMEM，也不发射 tcgen05 指令
layout 与 allocation / MMA / wait / tcgen05.ld 是四个独立契约
scale-factor atom 使用 S[(32, sf_per_mma):(1@TLane,1@TCol)] 描述基础位置
R[4 : 32@TLane] 枚举 q=0..3，偏移分别是 0、32、64、96
q=0 就是 base 本身，因此总数是四份
四个副本的 TLane 是 r + 32*q，TCol 保持 s 不变
同一个 scale 出现在四个 32-lane TMEM partition 的本地窗口
8-bit scale 的 logical TCol s 打包为 hardware TCol s//4 和 byte s%4
layout.apply() 只返回 base，replica 保存在 layout.replica
layout 描述复制目标，tcgen05.cp 执行数据搬运，MMA 消费本地副本
tmem_datapath_layout(datapath, rows, cols) 返回 tcgen05.mma 的 TMEM row mapping
datapath=D 要求 rows=128，TLane=row，TCol=col
datapath=F 要求 rows=64，把四个 16-row slab 放到 Lane 0/32/64/96 起点的低 16 Lane
F 的 TLane=32*(row//16)+row%16，映射不是复制
F 使用 64 条 active Lane，但 span TLane 是 112，因为 111 是最高 Lane
F.apply(row,col) 需要 shape=[64,cols]，因为内部 shard 有三个 iter
datapath 必须同时匹配 MMA 写入形式和 tcgen05.ld atom 的读取映射
tcgen05_atom_layout 将 TMEM fragment 的搬运形状映射到线程寄存器
instr_shape 是 lane x bits 的 PTX data-movement atom
tensor_shape 与 dtype 共同决定每个 32-bit cell 容纳多少元素
per_rep_cols = instr_column_factor * (32 / dtype_bits)
rep = K / per_rep_cols，rep 就是 tcgen05.ld/st 的 .xN
16x128b 的 fp32 K=128 与 fp16 K=256 都使用 .x32
16x128b 的 fp16 把相邻两个元素打包进一个 32-bit register
16x128b 的 lane 不直接等于行号，映射按 row group 和 32-bit word 选择
16x128b 的 reg32 = ((row // 8) % 2) + 2 * (q // 4)
16x128b 的 fp16 half = col % 2，决定低半或高半
rows=64 与 rows=128 的行数约束会随 instr_shape 改变
合法 .xN 集合取决于 instr_shape，非法 rep 不能通过补零静默绕过
layout 只描述 register mapping，不负责 tcgen05.ld/st 发射或 barrier 同步
```

下一知识点：

```text
chapter_tirx_layout_api
-> wg_local_layout 的行到 tid_in_wg 映射
```
