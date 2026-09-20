# 02 Replication 与 Offset

前面的 layout 通常是一对一映射：

```text
一个逻辑元素 -> 一个物理位置
```

但硬件中的数据路径会出现两种额外情况：

```text
同一个逻辑元素被复制到多个物理位置
基础映射整体增加一个固定偏移
```

对应两种 layout 修饰：

```text
R[...]  Replication
O[...]  Offset
```

## Replication

语法：

```text
R[n : s@axis]
```

含义：

- 产生 `n` 份副本
- 副本编号 `r = 0 ... n-1`
- 第 `r` 份在 `@axis` 上增加 `r*s`

例如：

```text
R[4 : 32@TLane]
```

四个副本的偏移分别是：

```text
r=0 -> 0@TLane
r=1 -> 32@TLane
r=2 -> 64@TLane
r=3 -> 96@TLane
```

`r=0` 对应基础位置本身，所以总副本数是四份，不是额外再加三份。

## TMEM Scale Factor 广播

Block-scaled MMA 需要把 scale factors 放入 TMEM。以 `.warpx4` 为例：

```text
base = S[(32, ...) : (1@TLane, ...)]
layout = base + R[4 : 32@TLane]
```

如果某个 scale factor 在基础 tile 中位于：

```text
TLane = 7
```

复制后的位置是：

```text
r=0:  7 + 0*32  = 7
r=1:  7 + 1*32  = 39
r=2:  7 + 2*32  = 71
r=3:  7 + 3*32  = 103
```

逻辑数据仍然只有一份，但物理上出现在四个 TMEM partitions。

对应的四组 TMEM lanes：

```text
partition 0: TLane 0 ... 31
partition 1: TLane 32 ... 63
partition 2: TLane 64 ... 95
partition 3: TLane 96 ... 127
```

## Offset

语法：

```text
O[s@axis]
```

含义是只在目标轴上增加一个固定偏移：

```text
O[1@gpuid_x]
```

如果基础位置是：

```text
(gpuid_x=0, gpuid_y=1)
```

增加 offset 后：

```text
(gpuid_x=1, gpuid_y=1)
```

这里没有产生副本，逻辑元素依然只有一个物理位置。

## GPU Mesh 示例

基础布局：

```text
base = S[(2, 4, 8) : (1@gpuid_y, 8@m, 1@m)]
```

逻辑坐标：

```text
(y=1, row=2, col=3)
```

基础映射：

```text
gpuid_y = 1
m = 2*8 + 3 = 19
```

增加 replication：

```text
base + R[2 : 1@gpuid_x]
```

结果：

```text
devices = {(0, 1), (1, 1)}
local offset = 19
```

同一个逻辑元素出现在 `gpuid_x=0` 和 `gpuid_x=1` 两张设备上。

增加 offset：

```text
base + O[1@gpuid_x]
```

结果：

```text
device = (1, 1)
local offset = 19
```

它只把基础位置平移到 `gpuid_x=1`，没有复制。

## R、O、S 与 stride 的区别

| 概念 | 作用 |
|---|---|
| `S[...]` | 定义逻辑元素到物理位置的基础映射 |
| `stride` | 在同一个基础映射中，坐标变化时移动多少 |
| `R[...]` | 增加副本维度，一个逻辑位置变成多个物理位置 |
| `O[...]` | 不增加副本，只给最终位置加固定偏移 |

对比：

```text
base + R[2 : 1@gpuid_x]
-> 2 份物理副本

base + O[1@gpuid_x]
-> 1 份物理数据，位置整体移动
```

## 常见错误

### 1. 漏掉原始副本

```text
R[4 : 32@TLane]
```

共有四个可能位置：

```text
r=0
r=1
r=2
r=3
```

`r=0` 是基础位置，不能省略。

### 2. 把 Offset 当成 Replication

```text
O[1@gpuid_x]
```

不表示“额外多一份”，而是：

```text
原来的一份移动到偏移后的位置
```

### 3. 把副本坐标当成新的逻辑数据

Replication 不增加不同的逻辑元素：

```text
同一个逻辑元素
-> 同一份值
-> 多个物理副本
```

## 本节结论

```text
R 负责复制到哪些位置
O 负责整体平移到哪里
```

Replication 常用于 TMEM multicast、跨 partition 广播和多设备复制；
Offset 常用于表达固定设备坐标或布局平移。
