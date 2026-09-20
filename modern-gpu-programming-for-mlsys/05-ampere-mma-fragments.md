# 05 Ampere MMA Fragment 与 ldmatrix

## 目标

这一节记录 `mma.sync`、Tensor Core fragment 和 `ldmatrix` 之间的关系。
重点是回答：

```text
mma.sync.aligned.m16n8k16 表示什么
每个 lane 的 A/B/C/D fragment 分别放在哪里
为什么 B fragment 和 A fragment 的 lane 映射不同
ldmatrix.x1 / .x2 / .x4 分别搬运多少数据
为什么 B 通常使用 ldmatrix.x2.trans
```

这些内容是从 shared memory 到 Tensor Core 的数据路径基础。

## mma.sync 的基本形式

常见的 Ampere fp16/bf16 MMA 指令类似：

```text
mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
```

其中：

```text
m16n8k16  M=16, N=8, K=16
row.col   A 按 row-major 解释，B 按 col-major 解释
f32       累加器 C/D 是 fp32
f16.f16   A/B 操作数是 fp16
```

### `.aligned` 是什么

`.aligned` 不是矩阵维度，也不是要求数据地址按某个大小对齐。

它表示：

```text
warp 中的所有线程都必须执行同一条 mma 指令
```

如果同一个 warp 中的线程走了不同控制流，一部分线程执行 MMA，另一部分
不执行，那么行为是未定义的。实际 kernel 中应保证 MMA 位于 warp-uniform
的控制流中。

不要把以下两个词混淆：

```text
.aligned  执行约束：warp 内所有线程执行同一条指令
.row.col  A/B 矩阵的逻辑布局约定
```

## MMA 形状不是运行时随便选的

`m16n8k16` 中的三个数是 Tensor Core 指令要求的固定 tile 形状：

```text
A: 16 x 16
B: 16 x 8
C: 16 x 8
D: 16 x 8
```

不同 dtype、指令家族和 GPU 架构支持的形状不同。常见例子：

| 数据类型 | 常见 `mma.sync` 形状 |
|---|---|
| fp16 / bf16 | `m16n8k8`, `m16n8k16` |
| tf32 | `m16n8k4`, `m16n8k8` |
| fp64 | `m8n8k4`, `m16n8k8` |
| int8 | `m16n8k16`, `m16n8k32` |
| fp8 | `m16n8k32` 等较新组合 |

这里列出的是常见组合，不是“所有 PTX 版本和所有 GPU 都支持”的清单。
实际使用时必须同时检查 dtype、PTX 版本、目标架构和 PTX ISA 语法。

`mma.sync` 的形状也不能直接拿 `wmma` 的形状套用。例如 `wmma` 还有：

```text
m16n16k16
m8n32k16
m32n8k16
```

它们属于不同的编程接口和 fragment 约定。

## 从 GEMM 到 fragment

对于：

```text
C = A @ B
```

当形状是 `m16n8k16` 时：

```text
A is M x K = 16 x 16
B is K x N = 16 x 8
C is M x N = 16 x 8
```

这里的 `B` 明确写成：

```text
B[k, n]
```

其中 `k` 是 K 方向坐标，`n` 是 N 方向坐标。

使用下面的 lane 坐标：

```text
g = laneid // 4
t = laneid % 4
```

后面四组 fragment 都可以用 `g` 和 `t` 表示。

## C/D fragment

对于 fp32 累加器，每个 lane 持有 4 个 fp32 元素：

```text
c0 = C[g,     2t]
c1 = C[g,     2t+1]
c2 = C[g+8,   2t]
c3 = C[g+8,   2t+1]
```

以 `laneid = 5` 为例：

```text
g = 5 // 4 = 1
t = 5 % 4 = 1
```

所以 lane 5 持有：

```text
C[1, 2]
C[1, 3]
C[9, 2]
C[9, 3]
```

每四个 lane 覆盖同一组两列：

```text
t = 0 -> columns 0, 1
t = 1 -> columns 2, 3
t = 2 -> columns 4, 5
t = 3 -> columns 6, 7
```

`g` 则负责行坐标的低半部分 `g` 和高半部分 `g+8`。

## A fragment

对于 fp16/bf16，每个 A 元素是 16-bit。两个元素可以打包进一个 32-bit
寄存器，所以每个 lane 的 A fragment 使用 4 个寄存器：

```text
r0 = [A[g,   2t],   A[g,   2t+1]]
r1 = [A[g+8, 2t],   A[g+8, 2t+1]]
r2 = [A[g,   2t+8], A[g,   2t+9]]
r3 = [A[g+8, 2t+8], A[g+8, 2t+9]]
```

可以发现：

```text
r0, r1 负责 columns 0 ... 7
r2, r3 负责 columns 8 ... 15

r0, r2 负责 rows 0 ... 7
r1, r3 负责 rows 8 ... 15
```

这就是为什么一个 `16 x 16` 的 A 矩阵可以被拆成四个 `8 x 8` 子块。

## B fragment

`m16n8k16` 的 B 是 `16 x 8`。每个 lane 持有 4 个 fp16/bf16 元素，
打包进 2 个 32-bit 寄存器：

```text
r0 = [B[2t,   g], B[2t+1, g]]
r1 = [B[2t+8, g], B[2t+9, g]]
```

以 `laneid = 5` 为例：

```text
g = 1
t = 1
```

所以 lane 5 持有：

```text
r0 = [B[2, 1],  B[3, 1]]
r1 = [B[10, 1], B[11, 1]]
```

为什么 B 的映射和 A 不一样？

因为在 B fragment 中：

```text
组号 g      选择 N 方向的 column
组内编号 t  选择 K 方向相邻的两个 row
```

一组四个 lane 合起来覆盖 B 的一个完整 column：

```text
lane 0..3   -> B 的 column 0
lane 4..7   -> B 的 column 1
lane 8..11  -> B 的 column 2
...
lane 28..31 -> B 的 column 7
```

以 `g=0` 为例：

```text
lane 0: B[0, 0], B[1, 0], B[8, 0],  B[9, 0]
lane 1: B[2, 0], B[3, 0], B[10, 0], B[11, 0]
lane 2: B[4, 0], B[5, 0], B[12, 0], B[13, 0]
lane 3: B[6, 0], B[7, 0], B[14, 0], B[15, 0]
```

因此四个 lane 共同覆盖 column 0 的全部 16 个 K 值。

MMA 是一条 warp 级协同指令。每个 lane 不一定要在自己的寄存器里拥有
计算目标 C 元素所需的全部 A/B 值，硬件会在 warp 内部完成 fragment
之间的数据路由。

## 为什么需要 ldmatrix

如果使用普通 shared memory load，每个 lane 都要自己计算地址，并且需要
手工把数据排成 MMA fragment。`ldmatrix` 把“加载矩阵行”和“按 fragment
分发数据”合并为一条 warp 级指令。

对于 fp16/bf16，一个基本的 `8 x 8` 矩阵大小是：

```text
8 * 8 = 64 个 b16 元素
64 * 2 Byte = 128 Byte
```

32 个 lane 平均接收：

```text
128 Byte / 32 lane = 4 Byte/lane
```

也就是每个 lane 得到：

```text
1 个 32-bit 寄存器
```

## x1、x2、x4 的含义

`.x1`、`.x2`、`.x4` 不是矩阵的 M/N/K，也不是每个 lane 加载 1、2、4
个元素。它们表示一条 `ldmatrix` 指令同时处理几个 `8 x 8` 矩阵。

| 后缀 | `8 x 8` 矩阵数 | 每 lane 得到 | 总数据量 | 提供有效行地址的 lane |
|---|---:|---:|---:|---|
| `.x1` | 1 | 1 个 32-bit 寄存器 | 128 Byte | lane 0 ... 7 |
| `.x2` | 2 | 2 个 32-bit 寄存器 | 256 Byte | lane 0 ... 15 |
| `.x4` | 4 | 4 个 32-bit 寄存器 | 512 Byte | lane 0 ... 31 |

可以记成：

```text
.x1 -> r0
.x2 -> r0, r1
.x4 -> r0, r1, r2, r3
```

每个 lane 提供的是一个矩阵行的起始 shared memory 地址。对于 `.x4`，
四个 `8 x 8` 矩阵的 32 行地址分别由 32 个 lane 提供。

每行包含 8 个 b16 元素，也就是 16 Byte，所以行地址通常需要满足
16 Byte 对齐。具体约束应以对应 PTX 版本的说明为准。

## 使用 ldmatrix.x4 加载 A

把 `16 x 16` 的 A 拆成四个 `8 x 8` 子块：

```text
A00 = rows 0..7,   columns 0..7
A10 = rows 8..15,  columns 0..7
A01 = rows 0..7,   columns 8..15
A11 = rows 8..15,  columns 8..15
```

一种常见的地址安排是：

```text
lane 0..7   -> A00 的 8 行地址
lane 8..15  -> A10 的 8 行地址
lane 16..23 -> A01 的 8 行地址
lane 24..31 -> A11 的 8 行地址
```

使用：

```text
ldmatrix.sync.aligned.m8n8.x4.shared.b16
```

最终得到：

```text
r0 -> A00 fragment
r1 -> A10 fragment
r2 -> A01 fragment
r3 -> A11 fragment
```

展开为：

```text
r0 = [A[g,   2t],   A[g,   2t+1]]
r1 = [A[g+8, 2t],   A[g+8, 2t+1]]
r2 = [A[g,   2t+8], A[g,   2t+9]]
r3 = [A[g+8, 2t+8], A[g+8, 2t+9]]
```

恰好就是 A fragment 需要的布局。

## 使用 ldmatrix.x2.trans 加载 B

B 的逻辑形状是：

```text
B[K, N] = 16 x 8
```

可以拆成：

```text
B_top    = rows 0..7,  columns 0..7
B_bottom = rows 8..15, columns 0..7
```

一种常见地址安排是：

```text
lane 0..7  -> B_top 的 8 行地址
lane 8..15 -> B_bottom 的 8 行地址
```

使用：

```text
ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16
```

这里 `.trans` 会对每一个 `8 x 8` 子块执行转置，而不是把整个 `16 x 8`
整体转置。

最终得到：

```text
r0 = [B[2t,   g], B[2t+1, g]]
r1 = [B[2t+8, g], B[2t+9, g]]
```

这正是 `m16n8k16` 所需的 B fragment。

## 与 swizzle 的关系

`ldmatrix` 接收的是 shared memory 行地址，因此它可以和 shared memory
swizzle 配合使用：

```text
生产者按照 swizzle 规则写入 shared memory
ldmatrix 使用同一套规则计算每个 lane 的行地址
硬件再按照 fragment layout 分发到寄存器
```

生产者、消费者和 descriptor 必须使用一致的 swizzle mode。否则元素虽然
被加载进寄存器，逻辑坐标可能已经完全错位。

## 推理系统中的应用

在推理系统的 GEMM、Attention 和 KV Cache kernel 中，典型路径是：

```text
global memory
    -> shared memory
    -> ldmatrix
    -> mma.sync
    -> accumulator fragment
    -> epilogue
```

因此下面这些错误经常表现在性能数字或结果正确性上：

```text
swizzle 地址算错
ldmatrix 提供的行地址不对
A/B fragment 的 lane 映射搞反
accumulator 的行列坐标错位
K tile 边界处理错误
```

## 结论

```text
m16n8k16 是固定的 MMA tile 形状
.aligned 要求整个 warp 执行同一条 mma 指令
A/B/C/D 各有固定的 lane fragment 映射
.x1/.x2/.x4 表示 ldmatrix 一次搬运几个 8x8 矩阵
.trans 表示每个 8x8 子块加载时转置
A 常用 ldmatrix.x4
B 常用 ldmatrix.x2.trans
```
