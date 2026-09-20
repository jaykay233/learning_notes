# 03 练习题与订正

本页保留命名轴、Register Fragment 以及 Replication / Offset 的自测题，
并记录第一次作答中的问题和正确推导。

## 第 1 题：概念区分

分别解释：

```text
stride
@axis
R[...]
O[...]
```

并回答：

```text
为什么 1@TLane 和 1@TCol 的 stride 都是 1，结果却不同？
```

### 正确理解

| 概念 | 正确含义 |
|---|---|
| `stride` | 沿目标轴走一步，位置变化多少 |
| `@axis` | 在哪个物理轴上移动 |
| `R[...]` | 产生副本，一个逻辑位置对应多个物理位置 |
| `O[...]` | 固定偏移，一个逻辑位置仍然只对应一个物理位置 |

`1@TLane` 和 `1@TCol` 的步长相同，但目标轴不同：

```text
1@TLane -> 改变 TLane
1@TCol  -> 改变 TCol
```

因此不能合并成一个线性偏移。

## 第 2 题：命名轴计算

给定：

```text
S[(8, 4, 2) : (4@laneid, 1@laneid, 1@reg)]
```

求逻辑元素：

```text
(row=6, col=5)
```

### 第一次作答

```text
laneid = 4*6 + 5*1
reg = 1
```

### 错误原因

`col=5` 不能直接进入 lane 计算。

shape `(8, 4, 2)` 表示：

```text
row
column_pair
element_in_pair
```

列方向必须先拆分：

```text
column_pair = col // 2
slot        = col % 2
```

### 正确推导

```text
c0 = row      = 6
c1 = col // 2 = 2
c2 = col % 2  = 1
```

前两个坐标进入 `@laneid`：

```text
laneid = 6*4 + 2 = 26
```

第三个坐标进入 `@reg`：

```text
reg = 1
```

答案：

```text
laneid = 26
reg    = 1
```

## 第 3 题：Replication 计算

给定：

```text
base = S[(32, ...) : (1@TLane, ...)]
layout = base + R[4 : 32@TLane]
```

如果基础位置是：

```text
TLane = 7
```

求复制后的位置。

### 第一次作答

```text
39, 71, 103
```

### 错误原因

漏掉了副本编号 `r=0` 对应的基础位置。

`R[4 : 32@TLane]` 表示总共有四份：

```text
r = 0, 1, 2, 3
```

### 正确结果

```text
r=0:  7 + 0*32  = 7
r=1:  7 + 1*32  = 39
r=2:  7 + 2*32  = 71
r=3:  7 + 3*32  = 103
```

答案：

```text
7, 39, 71, 103
```

## 第 4 题：R 和 O 的区别

基础位置：

```text
(gpuid_x=0, gpuid_y=1)
```

分别求：

```text
base + R[2 : 1@gpuid_x]
base + O[1@gpuid_x]
```

### 正确结果

Replication：

```text
(gpuid_x=0, gpuid_y=1)
(gpuid_x=1, gpuid_y=1)
```

共两份物理副本。

Offset：

```text
(gpuid_x=1, gpuid_y=1)
```

只有一份物理数据，只是位置发生偏移。

### 第一次作答的问题

只写了“2 份”，没有区分：

```text
R -> 2 份
O -> 1 份
```

## 第 5 题：同 stride，不同物理轴

比较：

```text
A = S[(2, 4) : (4@m, 1@m)]
B = S[(2, 4) : (4@TLane, 1@TCol)]
```

逻辑元素：

```text
(row=1, col=3)
```

### A 的结果

```text
1*4@m + 3*1@m
= 4@m + 3@m
= 7@m
```

### B 的结果

```text
1*4@TLane + 3*1@TCol
= 4@TLane + 3@TCol
```

也就是：

```text
(TLane=4, TCol=3)
```

### 为什么不能合并

`@m` 是一根线性地址轴，同一轴上的偏移可以相加：

```text
4@m + 3@m = 7@m
```

`@TLane` 和 `@TCol` 是两根不同的物理轴，所以必须保留为二元坐标：

```text
4@TLane + 3@TCol
```

不能写成：

```text
4 + 3 = 7
```

## 复习检查

可以只看下面几项检查是否真正掌握：

```text
shape 和 stride 分别回答什么问题
为什么 1@TLane 与 1@TCol 不能相加
为什么 fragment layout 要把 col 拆成 pair 和 slot
R[4:...] 为什么包含基础位置
R 和 O 对物理副本数量的影响
不同命名轴何时可以合并，何时不能合并
```
