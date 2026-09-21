# QoQ / QServe：W4A8KV4、渐进分组量化与 SmoothAttention

> 本文讨论 QServe 提出的 QoQ 量化算法。核心问题是：如何同时获得 W4 的显存带宽优势、A8 的 INT8 Tensor Core 吞吐，以及 KV4 Attention 的低带宽访问。
>
> 参考资料：
> - [QServe: W4A8KV4 Quantization and System Co-design for Efficient LLM Serving](https://arxiv.org/abs/2405.04532)
> - [mit-han-lab/omniserve](https://github.com/mit-han-lab/omniserve)
> - QServe 论文中的 QoQ、W4A8 GEMM、KV4 Attention 与 SmoothAttention 章节

---

## 0. 先记结论

QServe 不是把所有计算都变成 4-bit。它采用下面的组合：

```text
权重：4-bit 存储
激活：8-bit 计算
KV cache：4-bit 存储
GEMM：INT8 Tensor Core
```

因此，`W4A8KV4` 应该拆开理解：

| 记号 | 含义 |
|---|---|
| `W4` | Weight 以 4-bit 形式存储，降低权重显存和内存带宽 |
| `A8` | Activation 使用 8-bit，矩阵乘法走 INT8 Tensor Core |
| `KV4` | KV cache 使用 4-bit，降低 decode 阶段 Attention 的 KV 带宽 |

QoQ 负责连接 `W4` 和 `A8`：

```text
4-bit 权重存储
  -> 在寄存器中还原为近似 8-bit 整数
  -> 和 8-bit 激活做 INT8 Tensor Core GEMM
```

准确地说，不是“把 INT4 权重直接喂给 INT8 Tensor Core”，而是：

```text
INT4 weight
  -> INT8 weight operand
  -> INT8 x INT8 Tensor Core
  -> INT32 accumulator
  -> 应用 scale
  -> FP16 output
```

---

## 1. 为什么普通 INT4 量化没有形成端到端加速

量化理论上可以减少内存流量，也可以提高 Tensor Core 算力，但前提是反量化路径足够便宜。

### 1.1 常见精度组合

| 配置 | 权重存储 | 激活 | 主计算 | 主要瓶颈 |
|---|---:|---:|---|---|
| W4A16 | INT4 | FP16 | FP16 Tensor Core | INT4 要在 CUDA Core 上还原成 FP16 |
| W8A8 | INT8 | INT8 | INT8 Tensor Core | 权重内存流量大于 W4 |
| W4A4 | INT4 | INT4 | INT4 Tensor Core | partial sum 反量化昂贵，精度差，寄存器压力高 |
| W4A8 | INT4 | INT8 | INT8 Tensor Core | 需要便宜地把 INT4 权重还原成 INT8 |

### 1.2 W4A16

W4A16 的流程通常是：

```text
INT4 weight
  -> CUDA Core 反量化成 FP16
  -> FP16 Tensor Core GEMM
```

小 batch 时，GEMM 是 memory-bound。W4 权重减少了内存流量，因此 W4A16 可能更快。

大 batch 时，GEMM 逐渐变成 compute-bound。FP16 Tensor Core 的峰值吞吐低于 INT8 Tensor Core，因此 W4A16 的收益会下降。

### 1.3 W8A8

W8A8 的主循环可以一直使用 INT8 Tensor Core：

```text
INT8 activation x INT8 weight
  -> INT32 accumulate
  -> epilogue 中应用 scale
```

它的计算吞吐高，但权重和激活都占 8-bit，内存流量大于 W4。

### 1.4 W4A4

W4A4 看起来最理想：

```text
W4 的带宽
+ 4-bit Tensor Core 的算力
```

现实中有两个问题。

第一，量化精度很难保持。权重和激活都只有 16 个离散级别，通常需要 per-group scale 和 zero point。

第二，反量化不能完全留在 GEMM epilogue。为了让 per-group scale 生效，reduction 过程中可能需要反复处理 partial sum：

```text
INT4 x INT4
  -> INT32 partial sum
  -> 转 FP32
  -> 乘 group scale
  -> 累加
```

这个 INT32 到 FP32 的转换和 scale 操作发生在 CUDA Core 上，而且位于 GEMM 主循环内部。它还会同时占用 FP32 和 INT32 寄存器，降低 active warp 数量。

论文强调，A100 上 CUDA Core 的 FP32 峰值吞吐远低于 INT4 Tensor Core。一次昂贵的反量化操作，可能等价于几十次 Tensor Core MAC，从而抵消 4-bit 的收益。

### 1.5 W4A8 的目标

W4A8 试图同时获得：

```text
小 batch：W4 权重减少内存流量
大 batch：A8 使用 INT8 Tensor Core 提高算力
```

前提是必须解决 INT4 到 INT8 的转换成本。

---

## 2. QoQ 的核心：渐进式分组量化

QoQ 的全称是 `Quattuor-Octō-Quattuor`，即 4-8-4。

它把权重量化拆成两级：

```text
原始 FP16/BF16 权重 W
  |
  | 第一级：per-channel 对称 INT8
  | 范围限制到 [-119, 119]
  v
中间 INT8 权重 Q8
  |
  | 第二级：per-group 非对称 INT4
  v
Q4 + group zero point + group scale
```

### 2.1 第一级：per-channel INT8

第一级使用 per-channel 对称 INT8：

```text
Q8 = round(W / s0)
```

其中：

```text
s0 = max(abs(W), dim=channel) / 119
```

然后做保护范围截断：

```text
Q8 = clip(Q8, -119, 119)
```

正常情况下，对称 INT8 可以使用 `[-127, 127]`。QoQ 特意缩到 `[-119, 119]`，原因会在下一节解释。

这一级的中间结果 `Q8` 不必长期保存。它主要用于生成第二级量化参数。

### 2.2 第二级：per-group INT4

对每一组 `Q8`，再做非对称 INT4 量化：

```text
s1 = (Q8_group.max() - Q8_group.min()) / 15
z4 = round(-Q8_group.min() / s1)
Q4 = round(Q8_group / s1 + z4)
```

其中：

| 参数 | 含义 | 典型宽度 |
|---|---|---:|
| `Q4` | 最终存储的 4-bit 权重 | 4-bit |
| `z4` | group zero point | 4-bit |
| `s1` | group scale | 8-bit |
| `s0` | 第一级 per-channel scale | FP16 |

运行时使用第二级参数还原近似 INT8 权重：

```text
W8_hat = (Q4 - z4) * s1
```

这里的 `W8_hat` 是一个 8-bit 整数权重操作数，和第一级生成的 `Q8` 很接近，但不完全相等。

完整 GEMM 路径可以写成：

```text
X8 = quantize_to_int8(X)

W8_hat = (Q4 - z4) * s1

Y_int32 = X8 @ W8_hat

Y = Y_int32 * (sx * s0)   // epilogue
```

最终输出通常是 FP16。

### 2.3 为什么需要第一级 INT8

如果把普通 INT4 权重量化为：

```text
W_fp16 -> Q4 + FP16 scale
```

那么计算前通常需要：

```text
Q4 + FP16 scale -> FP16 weight
FP16 weight x FP16 activation -> FP16 Tensor Core
```

这就是 W4A16 的路径，反量化会落在 CUDA Core 上。

QoQ 改成：

```text
W_fp16
  -> 第一级 INT8：为第二级提供 8-bit 整数参考网格
  -> 第二级 INT4：把 INT8 参考值重新压缩为 4-bit
```

运行时：

```text
Q4 -> W8_hat
```

得到的是整数，不需要先转成 FP16。随后可以直接执行：

```text
INT8 activation x INT8 weight -> INT8 Tensor Core
```

这就是“存储按 W4 算，计算按 W8A8 算”的来源。

---

## 3. 保护范围为什么是 `[-119, 119]`

第二级 INT4 量化会有取整误差。如果第一级 INT8 值直接接近 127，第二级反量化后可能超过 INT8 最大值：

```text
原始 Q8 = 120
INT4 反量化后 = 128
```

`128` 已经超出有符号 INT8 的范围 `[-128, 127]`。

最直接的修复方式是在反量化后做 saturate，但饱和操作会显著降低 GEMM 吞吐。QServe 的解决方式是提前限制第一级范围。

论文给出的保护范围是：

```text
Q8 in [-119, 119]
```

这样第二级反量化后的整数可以稳定落在 INT8 范围内，不需要在主循环中增加饱和处理。

因此，`[-119, 119]` 不是普通量化中的自然范围，而是为后面的整数计算和寄存器级并行专门设计的约束。

---

## 4. “主循环完全没有反量化”需要修正

更准确的说法是：

> QoQ 消除了主循环中昂贵的浮点反量化和 partial sum 反量化，但主循环仍然保留廉价的整数解包与重建。

主循环中仍然存在：

```text
INT4 -> INT8 unpack
INT8 reconstruction
```

但它们使用整数操作，并且可以并行处理四个值。例如 NVIDIA GPU 可以用一条 `vadd4` 处理四个 INT8 加法。

系统实现中还有两个关键优化。

### 4.1 Subtraction after multiplication

直觉上，从 UINT4 还原成 SINT8 需要先减去 zero point，再做后续计算：

```text
W8 = Q4 - z4
```

但 zero point 操作也可以调整计算顺序，使它尽量不落在主循环热路径中。

对于 per-channel 权重，zero point 的减法可以移动到 GEMM epilogue。

对于 per-group 权重，因为 zero point 按 group 不同，不能完全移动，但可以：

```text
先做乘法
再做减法
```

这个顺序允许四个 INT8 值一起处理，利用寄存器级并行降低开销。

### 4.2 寄存器级并行

QoQ 会重排每 32 个 UINT4 权重，使解包时可以同时处理多个值：

```text
w0, w1, ..., w31
  -> w0, w16, w1, w17, ...
```

这样可以用少量位操作把多个 UINT4 并行解包成 UINT8，避免逐元素标量处理。

所以主循环的成本模型是：

```text
有整数解包
有整数重建
没有昂贵的 FP16/FP32 反量化
没有 partial sum 反量化
```

---

## 5. 计算感知权重重排

### 5.1 `ldmatrix` 为什么不能直接使用

Tensor Core GEMM intrinsic 要求每个线程拿到特定交错布局的数据。

如果存储类型和计算类型相同，例如：

```text
INT8 storage
INT8 compute
```

那么 `ldmatrix` 可以自动完成布局分发给每个线程。

但 W4A8 是：

```text
INT4 storage
INT8 compute
```

`ldmatrix` 按字节数而不是元素数分发数据，会出现错位：

```text
thread 0 拿到 thread 0 和 thread 1 要用的数据
thread 1 拿到 thread 2 和 thread 3 要用的数据
```

结果就是数据布局和计算需求不匹配，无法直接使用 `ldmatrix`。

普通地址计算会带来两个问题：

1. 指针运算在 CUDA Core 上执行，成本不可忽略。
2. 非连续访问无法达到最高的 128-bit load 带宽。

### 5.2 离线重排

QServe 的做法是把静态权重按照计算时的访问顺序重新排列。

它将 GEMM 划分成多个 `32 x 32` tile。每个线程需要 32 个输入通道，因此把：

```text
32 x 4-bit = 128-bit
```

通道拼成一个连续的 128-bit 字。

因为权重是静态的，这个重排可以在模型加载或离线阶段完成，不进入运行时热路径。

重排的目标不是改变数值，而是：

```text
消除主循环中的指针运算
保证每个线程 128-bit 连续访问
保持 Tensor Core 需要的线程布局
```

scale 和 zero point 也采用类似重排，减少反量化阶段的额外地址计算。

---

## 6. KV4 Attention

Decode 阶段的 Attention 接近批量 GEMV：

```text
score = Q K^T
```

对于每个 head，Query 通常只有一个 token。Attention 的计算强度大约是：

```text
1 MAC / element
```

因此，KV cache 的读取带宽是主要瓶颈。

从 KV8 降到 KV4 的理论收益是：

```text
KV cache 传输量减半
Attention 峰值性能理论上提升约 2x
```

### 6.1 为什么朴素 KV4 不一定更快

KV4 需要反量化。一个朴素实现可能需要：

```text
mask
shift
int -> float
mul
sub
```

每个元素大约需要 5 个 ALU 操作。叠加到 Attention kernel 后，A100 上的 KV4 Attention 可能从 memory-bound 变成 compute-bound。

QServe 做了几项优化：

| 优化 | 作用 |
|---|---|
| 将 FP32 算术替换为 FP16 | 提高 CUDA Core 的有效计算上限 |
| 使用位技巧处理 KV4 | 将反量化算术从约 5 ops 降到约 2 ops |
| 预取 scale 和 zero point | 减少地址计算与访存等待 |
| 简化控制流 | 降低 fused kernel 的指令开销 |

### 6.2 KV cache 量化粒度

vLLM 和 TensorRT-LLM 的 KV8 常见配置是：

```text
per-tensor
static quantization
```

QServe 使用：

```text
per-head
dynamic quantization
```

每个 head 单独保存 FP16 scale 和 zero point。降低到 4-bit 后，只有更细粒度的量化参数才能维持精度。

QServe 还支持：

- page-based KV cache；
- in-flight batching；
- scale 和 zero point 随 KV page 一起管理。

---

## 7. SmoothAttention

### 7.1 它解决什么问题

论文观察到：

```text
Value cache 没有明显的 outlier 模式
Key cache 有固定的 outlier channel
```

这些 Key outlier 可能比普通通道大十倍到几十倍。

INT4 只有 16 个量化级别。如果量化范围必须覆盖这些 outlier，普通通道的量化精度就会被拖累。

### 7.2 成对缩放保持数学等价

Attention score 为：

```text
score = Q K^T
```

对每个通道引入缩放系数 `lambda_i`：

```text
Q' = Q Lambda
K' = K Lambda^-1
```

那么：

```text
Q' K'^T
= (Q Lambda) (K Lambda^-1)^T
= Q Lambda Lambda^-1 K^T
= Q K^T
```

在没有量化的情况下，Attention score 完全不变。

但是：

```text
K 被量化
Q 不量化
```

所以可以让 Key 中难量化的 outlier 变小，把对应的数值范围转移到 Query 上。

例如：

```text
原来的 K_i 最大值：100
lambda_i = sqrt(100) = 10

新的 K_i：K_i / 10
新的 Q_i：Q_i * 10
```

Key 的量化范围变小，Query 虽然变大，但它仍然是 FP16，并不进入 KV4 量化。

### 7.3 缩放系数怎么选

论文使用：

```text
lambda_i = max(abs(K_i))^alpha
```

实践中：

```text
alpha = 0.5
```

即：

```text
lambda_i = sqrt(max(abs(K_i)))
```

### 7.4 为什么需要处理 RoPE

Q 和 K 都会经过 RoPE。RoPE 会把同一 head 内的通道：

```text
i
i + D / 2
```

配成一对。如果这两个通道使用不同的缩放，缩放和 RoPE 交换顺序后结果会改变。

因此 SmoothAttention 增加约束：

```text
lambda_i = lambda_(i + D/2)
```

论文将它们都取成配对通道中的较大值：

```text
lambda_i
= lambda_(i + D/2)
= max(max(abs(K_i)), max(abs(K_(i + D/2))))^0.5
```

这样缩放可以融合到前面的权重中：

```text
W_Q := Lambda W_Q
W_K := Lambda^-1 W_K
```

不需要增加额外的 Attention kernel。

---

## 8. SmoothAttention 和 SmoothQuant 的关系

两者的底层思想相同：

```text
插入一对互逆缩放
保持线性变换的数学结果
把量化难度转移到另一个张量
```

SmoothQuant：

```text
X W
= (X / s) (s W)
```

把激活 `X` 的 outlier 转移给权重 `W`。

SmoothAttention：

```text
Q K^T
= (Q Lambda) (K Lambda^-1)^T
```

把 Key `K` 的量化难度转移给 Query `Q`。

| 维度 | SmoothQuant | SmoothAttention |
|---|---|---|
| 优化对象 | Linear 层激活 `X` | Attention Key `K` |
| 转移目标 | Weight `W` | Query `Q` |
| 目标侧是否量化 | Weight 也量化 | Query 不量化 |
| 核心约束 | 激活与权重之间需要平衡 | 只需保证 K 更适合量化 |
| RoPE 处理 | 不涉及 | 需要配对通道 `i` 与 `i + D/2` |
| 主要目的 | 改善 W8A8 激活量化 | 改善 KV4 的 Key 量化 |

核心区别是：

```text
SmoothQuant 转移给另一个也会量化的张量
SmoothAttention 转移给不量化的 Query
```

所以 SmoothAttention 不需要在量化的 Q 和 K 之间寻找平衡。它只需要处理 Key，把 K 的量化难度转给 FP16 Query。

---

## 9. 存储开销

“W4”不是严格的每个元素只占 4 bit。每个 group 还需要保存第二级量化参数：

```text
Q4: 4 bit / element
z4: 4 bit / group
s1: 8 bit / group
s0: FP16 / channel
```

如果 group size 是 128，那么第二级参数的额外开销大约是：

```text
(4 + 8) / 128
= 0.09375 bit / element
```

因此，忽略对齐和第一级 per-channel scale，权重存储大约是：

```text
4 + 0.094
= 4.094 bit / element
```

仍然接近 4-bit，但必须把量化元数据计入系统设计。

---

## 10. 最终心智模型

QServe 可以拆成三条优化线。

### 10.1 GEMM 小 batch

```text
W4 权重减少内存流量
```

小 batch 时 GEMM 主要受权重读取带宽限制。

### 10.2 GEMM 大 batch

```text
A8 + INT8 Tensor Core 提供更高计算吞吐
```

大 batch 时 GEMM 逐渐变成 compute-bound。

### 10.3 Attention decode

```text
KV4 减少 KV cache 带宽
SmoothAttention 减少 KV4 量化误差
KV4 kernel 优化保证 Attention 保持 memory-bound
```

连接这三条线的是 QoQ：

```text
4-bit 权重存储
+ 8-bit 中间表示
+ INT8 Tensor Core
+ 便宜的整数还原
```

最准确的总结是：

> QoQ 用 4-bit 解决存储和带宽问题，用 8-bit 整数计算解决 Tensor Core 吞吐问题，再用渐进分组量化和系统级数据重排把两者接起来。

论文报告的端到端提升需要带上下文理解。相对 TensorRT-LLM 的最好配置，A100 上大约提升 `1.2x` 到 `2.4x`，L40S 上大约提升 `1.47x` 到 `3.47x`。`3.5x` 是 L40S 上的最高值，不是所有模型和配置的统一提升。
