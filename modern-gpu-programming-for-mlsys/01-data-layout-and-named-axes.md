# 01 数据布局与命名轴

## Shape-Stride 模型

一个 layout 可以用 shape 和 stride 表示：

```text
S[(e0, e1, ..., en-1) : (s0, s1, ..., sn-1)]
```

对于线性逻辑索引 `x`：

```text
(c0, c1, ..., cn-1) = unflatten(x; e0, e1, ..., en-1)
f(x) = c0*s0 + c1*s1 + ... + cn-1*sn-1
```

其中：

- shape 决定逻辑索引如何拆成多个坐标
- stride 决定每个坐标变化时物理位置移动多少

例如：

```text
S[(4, 2, 2, 4) : (16, 4, 8, 1)]
```

逻辑索引拆分后：

```text
c0 = i // 2
c1 = i % 2
c2 = j // 4
c3 = j % 4
```

最终映射为：

```text
f(i, j) = c0*16 + c1*4 + c2*8 + c3*1
```

这里的 `(16, 4, 8, 1)` 表示四个坐标在同一个线性地址空间 `@m`
中的 stride。

## 为什么需要命名轴

普通的 global memory 或 shared memory 可以用一个线性地址轴表示：

```text
@m
```

但 TMEM 和 register fragment 不是一维地址：

```text
TMEM:             TLane × TCol
Register fragment: laneid × reg slot
```

仅有一个整数地址无法区分这些物理位置，所以 layout 使用命名轴：

```text
stride@axis
```

例如：

```text
1@TLane
1@TCol
```

读法是：

```text
1@TLane: 沿 TLane 轴移动 1
1@TCol:  沿 TCol 轴移动 1
```

二者 stride 相同，但目标物理维度不同。

## 常见命名轴

| 命名轴 | 含义 |
|---|---|
| `@m` | 普通线性内存地址轴 |
| `@TLane` | TMEM 的 lane 坐标，范围通常为 0 到 127 |
| `@TCol` | TMEM 的 column 坐标，每列是一个 32-bit cell |
| `@laneid` | warp 内 lane ID，范围 0 到 31 |
| `@reg` | 当前 lane 内的 fragment slot |

`@reg` 表示 layout 中的 lane-local slot，不一定等于一个完整的硬件寄存器。
例如两个 `fp16` 或 `bf16` 元素可以打包进一个 32-bit register。

## TMEM 的二维映射

一个 `128×256` accumulator tile 可以写成：

```text
S[(128, 256) : (1@TLane, 1@TCol)]
```

逻辑索引先拆分：

```text
(row, col) = unflatten(x; 128, 256)
```

再映射为：

```text
f(x) = row@TLane + col@TCol
```

例如：

```text
(row=37, col=100)
-> (TLane=37, TCol=100)
```

这不是：

```text
37 + 100 = 137
```

因为 `@TLane` 和 `@TCol` 是两个不同的物理轴。

## stride 与命名轴的区别

```text
stride: 在目标轴上走一步，物理位置变化多少
@axis:  这一步发生在哪个物理轴上
```

例如：

```text
S[(2, 4) : (4@m, 1@m)]
```

两个坐标都进入 `@m`：

```text
(row=1, col=3)
-> 1*4@m + 3*1@m
-> 7@m
```

如果改成：

```text
S[(2, 4) : (4@TLane, 1@TCol)]
```

虽然 stride 数值相同，但结果变成二维坐标：

```text
(row=1, col=3)
-> 4@TLane + 3@TCol
-> (TLane=4, TCol=3)
```

不能把 `4@TLane + 3@TCol` 合并成 `7`，因为两个值位于不同坐标轴。

## Register Fragment

一个 `8×8` tile 的 fragment layout：

```text
S[(8, 4, 2) : (4@laneid, 1@laneid, 1@reg)]
```

三个坐标分别表示：

```text
(row, column_pair, element_in_pair)
```

矩阵坐标到坐标的转换：

```text
row         = row
column_pair = col // 2
slot        = col % 2
```

映射到物理位置：

```text
laneid = row*4 + column_pair
reg    = slot
```

### 为什么必须拆 column

一个 `8×8` tile 有 64 个逻辑元素，warp 有 32 个 lanes，因此每个 lane
需要持有 2 个元素。

这个 fragment 让每个 lane 持有相邻的两个 column：

```text
col 0, 1 -> pair 0, slot 0/1
col 2, 3 -> pair 1, slot 0/1
col 4, 5 -> pair 2, slot 0/1
col 6, 7 -> pair 3, slot 0/1
```

因此同一行中的四个 lane 分别负责四个 column pair：

```text
row r, pair 0 -> lane 4r
row r, pair 1 -> lane 4r + 1
row r, pair 2 -> lane 4r + 2
row r, pair 3 -> lane 4r + 3
```

shape `(8, 4, 2)` 正好描述了：

```text
8 rows
× 4 column pairs
× 2 elements per pair
```

如果直接把 `col` 当 lane 坐标的一部分，就无法表示“一个 lane 同时持有
两个 column”的映射。

## 具体例子：`(row=6, col=5)`

对于 `8×8` row-major tile：

```text
x = row*8 + col
  = 6*8 + 5
  = 53
```

用 shape `(8, 4, 2)` 反展平：

```text
c0 = 53 // (4*2) = 6
c1 = (53 // 2) % 4 = 2
c2 = 53 % 2 = 1
```

所以：

```text
(row, column_pair, slot) = (6, 2, 1)
```

再代入 stride：

```text
laneid = 6*4 + 2 = 26
reg    = 1
```

最终物理位置：

```text
laneid = 26
reg    = 1
```

常见错误是直接计算：

```text
laneid = 6*4 + 5
```

这里把 `col=5` 误当成了 `column_pair=5`。实际上：

```text
column_pair = 5 // 2 = 2
slot        = 5 % 2  = 1
```

## 本节结论

```text
shape 决定逻辑索引拆成哪些坐标
stride 决定坐标变化时在目标轴上移动多少
命名轴决定 stride 最终进入哪个物理维度
不同命名轴不能直接合并成标量
```

命名轴让 layout 能同时描述线程、lane、register slot 和 TMEM 坐标，
这是理解 Tensor Core 数据路径的基础。
