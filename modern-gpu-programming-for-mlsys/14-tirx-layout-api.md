# TIRx Layout API：`S[...]`、`R[...]`、offset 与命名轴

这篇笔记开始学习 `chapter_tirx_layout_api`。目前完整讲清前两个
知识点：如何读取和计算：

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
[ ] apply() 的三种输入形式与 flatten / decompose
[ ] TMEM accumulator layout 示例
[ ] scale-factor layout 中的 replication
[ ] tmem_datapath_layout / tcgen05_atom_layout / wg_local_layout
[ ] ComposeLayout 与 shared-memory swizzle
```

本篇目前完成前两项。后续知识点会在同一章新的小节中继续展开。

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

## 十二、当前进度

`chapter_tirx_layout_api` 的知识点：

```text
[x] TileLayout 的 S[...]、R[...] 与 offset
[x] 命名轴：laneid / warpid / m / TLane / TCol
[ ] apply() 的三种输入形式与 flatten / decompose
[ ] TMEM accumulator layout 示例
[ ] scale-factor layout 中的 replication
[ ] tmem_datapath_layout / tcgen05_atom_layout / wg_local_layout
[ ] ComposeLayout 与 shared-memory swizzle
```

已经覆盖：

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
```

下一知识点：

```text
chapter_tirx_layout_api
-> apply() 的三种输入形式与 flatten / decompose
```
