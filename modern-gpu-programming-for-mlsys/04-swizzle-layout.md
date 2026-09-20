# 04 Swizzle Layout

## 目标

这一节记录 shared memory 的 bank conflict，以及 XOR swizzle 如何改变
物理地址排列，让同行访问和同列访问都能分散到不同 banks。

核心公式：

```text
bank = (addr // 4) % 32
```

简化例子使用 8 个 banks：

```text
bank = elem_index % 8
```

## Bank Conflict

现代 NVIDIA GPU 的 shared memory 被划分为 32 个 memory banks，每个 bank
通常对应一个 4-byte word。

一个 warp 发起 shared memory 访问时：

```text
同一 bank + 不同地址 -> bank conflict
同一 bank + 相同地址 -> broadcast
```

bank conflict 会让硬件拆分访问，降低有效带宽。

对于普通 row-major `8×8` tile：

```text
bank = logical_col
```

按行访问：

```text
row 0 -> banks 0 1 2 3 4 5 6 7
```

无冲突。

按列访问：

```text
column 3 -> 8 个元素都位于 bank 3
```

产生 8-way bank conflict。

## XOR Swizzle

XOR swizzle 使用行号和列号计算物理 column：

```text
mapped_col = logical_col XOR row
bank       = mapped_col
```

以 8 个 banks 为例：

```text
row 0: 0 1 2 3 4 5 6 7
row 1: 1 0 3 2 5 4 7 6
row 2: 2 3 0 1 6 7 4 5
row 3: 3 2 1 0 7 6 5 4
row 4: 4 5 6 7 0 1 2 3
row 5: 5 4 7 6 1 0 3 2
row 6: 6 7 4 5 2 3 0 1
row 7: 7 6 5 4 3 2 1 0
```

## 表格如何计算

先写出 row 和 col 的二进制：

```text
0 = 000
1 = 001
2 = 010
3 = 011
4 = 100
5 = 101
6 = 110
7 = 111
```

每一格的 bank 都计算：

```text
bank = col XOR row
```

### `row 0`

```text
col XOR 0 = col
```

所以：

```text
row 0: 0 1 2 3 4 5 6 7
```

### `row 1`

`row 1 = 001`，异或 `1` 会翻转最低位：

```text
0 XOR 1 = 1
1 XOR 1 = 0
2 XOR 1 = 3
3 XOR 1 = 2
4 XOR 1 = 5
5 XOR 1 = 4
6 XOR 1 = 7
7 XOR 1 = 6
```

所以：

```text
row 1: 1 0 3 2 5 4 7 6
```

效果是相邻两个元素互换。

### `row 2`

`row 2 = 010`，异或 `2` 会翻转第二位：

```text
0 XOR 2 = 2
1 XOR 2 = 3
2 XOR 2 = 0
3 XOR 2 = 1
4 XOR 2 = 6
5 XOR 2 = 7
6 XOR 2 = 4
7 XOR 2 = 5
```

所以：

```text
row 2: 2 3 0 1 6 7 4 5
```

效果是每两个元素组成一组，然后交换相邻两组。

### `row 3`

`row 3 = 011`，会同时翻转最低两位：

```text
0 XOR 3 = 3
1 XOR 3 = 2
2 XOR 3 = 1
3 XOR 3 = 0
4 XOR 3 = 7
5 XOR 3 = 6
6 XOR 3 = 5
7 XOR 3 = 4
```

所以：

```text
row 3: 3 2 1 0 7 6 5 4
```

### `row 4` 到 `row 7`

```text
row 4: col XOR 4 -> 4 5 6 7 0 1 2 3
row 5: col XOR 5 -> 5 4 7 6 1 0 3 2
row 6: col XOR 6 -> 6 7 4 5 2 3 0 1
row 7: col XOR 7 -> 7 6 5 4 3 2 1 0
```

## 为什么列访问不再冲突

查看逻辑 column `0`：

```text
row 0: 0 XOR 0 = bank 0
row 1: 0 XOR 1 = bank 1
row 2: 0 XOR 2 = bank 2
row 3: 0 XOR 3 = bank 3
row 4: 0 XOR 4 = bank 4
row 5: 0 XOR 5 = bank 5
row 6: 0 XOR 6 = bank 6
row 7: 0 XOR 7 = bank 7
```

8 个元素分别进入 8 个不同 banks，因此没有 conflict。

查看逻辑 column `3`：

```text
row 0: 3 XOR 0 = 3
row 1: 3 XOR 1 = 2
row 2: 3 XOR 2 = 1
row 3: 3 XOR 3 = 0
row 4: 3 XOR 4 = 7
row 5: 3 XOR 5 = 6
row 6: 3 XOR 6 = 5
row 7: 3 XOR 7 = 4
```

得到：

```text
3 2 1 0 7 6 5 4
```

仍然覆盖全部 `0...7`。

## 为什么行访问也不冲突

固定 row 后，例如 `row 2`：

```text
2 3 0 1 6 7 4 5
```

这也是一个 `0...7` 的排列，因此一行访问仍然覆盖全部 banks。

XOR swizzle 的关键性质是：

```text
固定 row 时，col XOR row 是 col 的排列
固定 col 时，col XOR row 是 row 的排列
```

排列意味着每个 bank 只出现一次，所以行访问和列访问都能避免重复命中。

## 逻辑位置和物理位置

Swizzle 不改变逻辑数据：

```text
逻辑元素 (row=2, col=3) 仍然是 (2,3)
```

它只改变 shared memory 中的物理位置：

```text
mapped_col = 3 XOR 2
           = 1

physical bank = 1
```

写入方和读取方必须采用同一套 swizzle 规则。否则会产生错误的地址映射，
读取到的元素就可能不再对应原来的逻辑坐标。

## 与命名轴、stride、R/O 的区别

| 概念 | 作用 |
|---|---|
| 命名轴 | 描述物理位置由哪些坐标组成 |
| stride | 描述沿坐标移动时位置变化多少 |
| `R[...]` | 产生多个物理副本 |
| `O[...]` | 给基础位置增加固定偏移 |
| swizzle | 改变同一个 shared memory tile 内部的物理排列 |

Swizzle 描述的是 tile 内部的地址重排，不是复制，也不是整体平移。

## 真实 GPU 上的补充

上面的 `8×8`、8 banks 是为了看清 XOR 规律而使用的简化图。

真实 NVIDIA shared memory 通常使用：

```text
32 banks
4-byte bank word
bank = (byte_address // 4) % 32
```

实际 Tensor Core kernel 常见 32 B、64 B、128 B swizzle。TMA、WGMMA
matrix descriptor 和其他 shared memory 消费者必须按相同的 swizzle mode
解释数据。

## 结论

```text
普通 row-major：行访问好，列访问容易冲突
XOR swizzle：用 row 和 col 异或重排物理 column
固定行：覆盖全部 banks
固定列：覆盖全部 banks
```

Swizzle 的目标不是改变矩阵内容，而是在不破坏行访问的情况下，让列访问
或多方向 tile 访问更均匀地分散到 shared memory banks。
