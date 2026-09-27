# Flash Attention：Tile 分解、Online Softmax 与 Conditional Rescaling

这篇笔记开始 `chapter_flash_attention`。上一章已经把 TMA、软件流水线、
persistent scheduling、warp specialization 和 two-CTA cooperative MMA
整理完成。FlashAttention 把同一套数据路径移到 attention 中：

```text
QK^T MMA -> softmax -> PV MMA
```

前两部分先讲最基础、也是后面所有 FA4 优化共同依赖的两个知识点：
不保存完整 score matrix，如何按 K/V block 计算等价的 attention 输出；
以及 FA4 如何用 `delta`、阈值 8 和 `acc_scale` 减少 TMEM 中 `O` 的
重缩放次数。

## 本次讲解位置

```text
本次讲解位置
章节：chapter_flash_attention
小节：算法结构
知识点：Q/K/V tile 分解，以及逐行 row_max / row_sum / O 的 online softmax 更新
上次：chapter_gemm_advanced 完成：Multi-Consumer Warp Specialization
下次：FA4 conditional rescaling：阈值 8、delta 与 acc_scale
PTX：本节先建立算法和数值状态；后续映射到 tcgen05.mma、
     tcgen05.ld / st 与 mbarrier 交接
```

### chapter_flash_attention 进度清单

```text
[x] 1. Tile 分解与 online softmax 三状态
[x] 2. FA4 conditional rescaling、delta 与 acc_scale
[ ] 3. S / P / O 的 TMEM layout 与分时复用
[ ] 4. QK^T MMA、softmax、PV MMA 的数据路径
[ ] 5. Warp 角色、register 分配与 barrier 分工
[ ] 6. Q / K / V pipeline 时间线
[ ] 7. Correction、最终归一化与 epilogue
[ ] 8. Causal mask、GQA、tile scheduling 与验证
```

## 一、问题：完整 attention 为什么会产生平方级中间量

单头 self-attention 输入：

```text
Q: [L, d]
K: [L, d]
V: [L, d]
```

标准计算是：

```text
S = Q @ K.T
P = softmax(S / sqrt(d))
O = P @ V
```

`S` 和 `P` 的 shape 都是：

```text
[L, L]
```

如果 `L=4096`，并且每个 score 用 fp32 保存：

```text
4096 * 4096 = 16,777,216 elements
16,777,216 * 4 bytes = 67,108,864 bytes
                     = 64 MiB
```

这还只是单个 head 的 `S`。一个 `32 heads` 的 batch item 就需要：

```text
64 MiB * 32 = 2048 MiB = 2 GiB
```

完整保存 `S` 的问题不只是容量。即使放得下，也要经历：

```text
MMA 写 S 到 GMEM
softmax 从 GMEM 读 S
softmax 写 P 到 GMEM
PV MMA 从 GMEM 读 P
```

中间 traffic 随 `L^2` 增长。

FlashAttention 的目标不是改变 attention 的数学结果，而是避免把完整
`S` 和 `P` 长期放进显存。它按 block 读取 K/V，并在每个 query row 上
维护少量 running state。

## 二、心智模型：一个 query block 流式吸收所有 K/V block

把序列维切成：

```text
BLOCK_M: 每次处理多少个 query rows
BLOCK_N: 每次处理多少个 key/value positions
```

对固定 Q block：

```text
Q_block: [BLOCK_M, d]
```

依次读取：

```text
K_block: [BLOCK_N, d]
V_block: [BLOCK_N, d]
```

每个 K/V block 产生：

```text
S_block = Q_block @ K_block.T
        [BLOCK_M, BLOCK_N]
```

注意这里只需要当前 `[BLOCK_M, BLOCK_N]` score tile，而不是完整
`[L, L]` score matrix。

不过，softmax 需要整行信息：

```text
行最大值
指数之和
指数加权后的 V
```

这些信息必须跨 K/V blocks 保存。FlashAttention 为每个 query row 保存
三个状态：

| 状态 | 数学含义 | 作用 |
|---|---|---|
| `row_max` | 当前作为指数参考值的最大值 | 防止 `exp(score)` 溢出 |
| `row_sum` | 已处理 positions 的指数和 | softmax 分母 |
| `O` | 已处理 positions 的指数加权 V | 尚未除分母的输出 |

核心不变量是：

```text
row_sum_i = sum_j exp(s_ij - row_max_i)
O_i       = sum_j exp(s_ij - row_max_i) * v_j
```

这里求和范围是所有已经处理的 K/V positions。

最后：

```text
output_i = O_i / row_sum_i
```

如果后续 block 发现更大的 score，参考值会变化。旧状态必须整体乘一个
尺度系数，才能和新 block 的贡献放到同一尺度上。

## 三、完整可运行代码

下面的程序用 NumPy 在 CPU 上验证算法。它不是 GPU kernel，只验证
FlashAttention 的分块状态更新和最终结果是否与标准 attention 相等。

文件名：`flash_attention_online_softmax_demo.py`

```python
import math

import numpy as np


def standard_attention(Q, K, V):
    d = Q.shape[-1]
    scores = Q @ K.T / math.sqrt(d)
    shifted = scores - scores.max(axis=-1, keepdims=True)
    weights = np.exp(shifted)
    weights = weights / weights.sum(axis=-1, keepdims=True)
    return weights, weights @ V


def tiled_attention_trace(Q, K, V, block_m=2, block_n=2):
    d = Q.shape[-1]
    scale = 1.0 / math.sqrt(d)
    outputs = np.empty((Q.shape[0], V.shape[1]), dtype=np.float64)
    traces = []

    for q_start in range(0, Q.shape[0], block_m):
        q_end = min(q_start + block_m, Q.shape[0])
        Qb = Q[q_start:q_end]
        q_len = q_end - q_start

        O = np.zeros((q_len, V.shape[1]), dtype=np.float64)
        row_max = np.full((q_len,), -np.inf, dtype=np.float64)
        row_sum = np.zeros((q_len,), dtype=np.float64)

        for kv_start in range(0, K.shape[0], block_n):
            kv_end = min(kv_start + block_n, K.shape[0])
            Kb = K[kv_start:kv_end]
            Vb = V[kv_start:kv_end]

            S = Qb @ Kb.T * scale
            tile_max = S.max(axis=-1)
            new_max = np.maximum(row_max, tile_max)
            alpha = np.exp(row_max - new_max)
            P = np.exp(S - new_max[:, None])

            old_O = O.copy()
            row_sum = row_sum * alpha + P.sum(axis=-1)
            O = old_O * alpha[:, None] + P @ Vb
            row_max = new_max

            traces.append(
                {
                    "q_block": (q_start, q_end),
                    "kv_block": (kv_start, kv_end),
                    "tile_max": tile_max.copy(),
                    "alpha": alpha.copy(),
                    "row_max": row_max.copy(),
                    "row_sum": row_sum.copy(),
                    "O": O.copy(),
                }
            )

        outputs[q_start:q_end] = O / row_sum[:, None]

    return outputs, traces


def main():
    Q = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [2.0, 1.0],
            [0.0, 2.0],
        ]
    )
    K = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [2.0, 0.0],
            [0.0, 2.0],
        ]
    )
    V = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [2.0, 1.0],
            [1.0, 2.0],
        ]
    )

    weights, reference = standard_attention(Q, K, V)
    tiled, traces = tiled_attention_trace(Q, K, V)

    np.set_printoptions(precision=6, suppress=True)
    print("full attention weights:")
    print(weights)
    print("standard output:")
    print(reference)
    print("tiled output:")
    print(tiled)
    print(f"max error: {np.abs(reference - tiled).max():.16g}")

    for idx, trace in enumerate(traces):
        print(
            f"trace {idx}: "
            f"q={trace['q_block']} kv={trace['kv_block']} "
            f"tile_max={trace['tile_max']} "
            f"alpha={trace['alpha']} "
            f"row_max={trace['row_max']} "
            f"row_sum={trace['row_sum']} "
            f"O={trace['O'].tolist()}"
        )


if __name__ == "__main__":
    main()
```

运行：

```bash
python flash_attention_online_softmax_demo.py
```

预期输出：

```text
full attention weights:
[[0.249112 0.12283  0.505229 0.12283 ]
 [0.12283  0.249112 0.12283  0.505229]
 [0.15137  0.074636 0.622624 0.15137 ]
 [0.043418 0.178588 0.043418 0.734577]]
standard output:
[[1.382399 0.873717]
 [0.873717 1.382399]
 [1.547988 1.      ]
 [0.86483  1.691159]]
tiled output:
[[1.382399 0.873717]
 [0.873717 1.382399]
 [1.547988 1.      ]
 [0.86483  1.691159]]
max error: 6.661338147750939e-16
trace 0: q=(0, 2) kv=(0, 2) tile_max=[0.707107 0.707107] alpha=[0. 0.] row_max=[0.707107 0.707107] row_sum=[1.493069 1.493069] O=[[1.0, 0.49306869139523984], [0.49306869139523984, 1.0]]
trace 1: q=(0, 2) kv=(2, 4) tile_max=[1.414214 1.414214] alpha=[0.493069 0.493069] row_max=[1.414214 1.414214] row_sum=[1.979302 1.979302] O=[[2.7361854258294542, 1.7293502033026427], [1.7293502033026427, 2.7361854258294542]]
trace 2: q=(2, 4) kv=(0, 2) tile_max=[1.414214 1.414214] alpha=[0. 0.] row_max=[1.414214 1.414214] row_sum=[1.493069 1.243117] O=[[1.0, 0.49306869139523984], [0.24311673443421425, 1.0]]
trace 3: q=(2, 4) kv=(2, 4) tile_max=[2.828427 2.828427] alpha=[0.243117 0.243117] row_max=[2.828427 2.828427] row_sum=[1.606107 1.361328] O=[[2.486233468868429, 1.6061067189721905], [1.1773172396858687, 2.302222480996171]]
```

运行环境只要求 Python 和 NumPy。这个结果说明 tiled 版本与完整
softmax 的误差只有浮点舍入量级。

## 四、算法逐阶段展开

### 1. 初始化一个 Q block 的状态

```python
O = np.zeros((q_len, V.shape[1]))
row_max = np.full((q_len,), -np.inf)
row_sum = np.zeros((q_len,))
```

`row_max` 必须从负无穷开始，不是 0。原因会在常见错误中展开。

### 2. 计算当前 score tile

```python
S = Qb @ Kb.T * scale
scale = 1.0 / math.sqrt(d)
```

`S` 的 shape 是：

```text
[BLOCK_M, BLOCK_N]
```

它是完整 `QK^T` 中当前 query rows 和当前 key columns 对应的一小块。

### 3. 求当前 tile 的逐行最大值

```python
tile_max = S.max(axis=-1)
```

对第 `i` 行：

```text
tile_max_i = max_j S[i, j]
```

这只是当前 K/V block 内的最大值，不是整行最大值。

### 4. 合并旧参考值和新 tile 最大值

```python
new_max = np.maximum(row_max, tile_max)
```

如果新 tile 出现更大的 score：

```text
new_max_i > row_max_i
```

旧的指数和旧 output 都是相对于 `row_max_i` 计算的，必须先缩放。

### 5. 计算旧状态到新尺度的系数

```python
alpha = np.exp(row_max - new_max)
```

数学上：

```text
alpha_i = exp(r_old - r_new)
```

因为：

```text
r_new >= r_old
alpha_i <= 1
```

如果参考值没有变化，则：

```text
alpha_i = exp(0) = 1
```

### 6. 计算当前 block 的未归一化权重

```python
P = np.exp(S - new_max[:, None])
```

这是当前 block 相对于最新参考值的权重。每一行最大的那个元素会变成：

```text
exp(0) = 1
```

### 7. 更新 softmax 分母

```python
row_sum = row_sum * alpha + P.sum(axis=-1)
```

展开为：

```text
new_row_sum_i =
    old_row_sum_i * exp(r_old_i - r_new_i)
    + sum_j P[i, j]
```

旧部分先换到新尺度，再加上当前 block 的贡献。

### 8. 更新未归一化 output

```python
old_O = O.copy()
O = old_O * alpha[:, None] + P @ Vb
```

这里不能只缩放 `row_sum` 而忘记缩放 `O`。两者必须使用同一个
`alpha`。

### 9. 所有 K/V blocks 完成后归一化

```python
outputs[q_start:q_end] = O / row_sum[:, None]
```

不要在每次 K/V block 后提前归一化。`O` 和 `row_sum` 只有在全部
positions 处理完成后才是最终 softmax 的分子和分母。

## 五、具体数值 trace

取第一个 Q block：

```text
Q_block rows 0, 1
```

### 第一个 K/V block：positions 0, 1

计算得到：

```text
tile_max = [0.707107, 0.707107]
row_max  = [0.707107, 0.707107]
alpha    = [0, 0]
```

因为这是第一个 block，旧 `row_sum` 和旧 `O` 都是 0。虽然
`alpha=exp(-inf - 0.707107)=0`，乘 0 后旧状态仍然是 0。

第 0 行当前 block 的权重为：

```text
S[0] = [0.707107, 0]
P[0] = exp([0, -0.707107])
     = [1, 0.49306869139523984]
row_sum[0] = 1 + 0.49306869139523984
           = 1.4930686913952398
```

当前 block 的未归一化输出为：

```text
O[0] = P[0] @ V[0:2]
     = [1, 0.49306869139523984]
```

此时如果错误地立刻归一化，会得到：

```text
[0.66976264, 0.33023736]
```

但后面还有 positions 2、3，不能在这里结束。

### 第二个 K/V block：positions 2, 3

新 tile 的最大值变大：

```text
old row_max = 0.707107
tile_max    = 1.414214
new row_max = 1.414214
```

计算 rescale 系数：

```text
alpha = exp(0.707107 - 1.414214)
      = exp(-0.707107)
      = 0.49306869139523984
```

旧分母先缩放：

```text
old row_sum * alpha
= 1.4930686913952398 * 0.49306869139523984
= 0.7361854258294542
```

第二个 block 在当前尺度下的第 0 行权重为：

```text
P[0] = [1, 0.24311673443421425]
sum(P[0]) = 1.2431167344342143
```

新分母：

```text
row_sum[0] = 0.7361854258294541 + 1.2431167344342143
           = 1.9793021602636682
```

旧 output 使用同一个 `alpha` 缩放，再叠加当前 block：

```text
old_O[0] * alpha
= [1, 0.49306869139523984] * 0.49306869139523984
= [0.49306869139523984, 0.24311673443421428]

new block contribution
= [1, 0.24311673443421425] @ V[2:4]
= [2.2431167344342144, 1.4862334688684284]

O[0]
= [2.7361854258294542, 1.7293502033026427]
```

最后除以行分母：

```text
O[0] / row_sum[0]
= [2.7361854258294542, 1.7293502033026427]
   / 1.9793021602636682
= [1.3823990499080543, 0.8737171302194058]
```

这与完整 softmax 的第 0 行输出一致。

## 六、与完整 attention 的对比

| 维度 | 完整 attention | tiled online attention |
|---|---|---|
| 完整 `S` 的存放 | `[L, L]` | 只保留 `[BLOCK_M, BLOCK_N]` |
| softmax 输入 | 一整行 | 逐个 K/V block |
| 行最大值 | 一次性求完整行的 max | `row_max` 随 block 更新 |
| 分母 | 一次性 sum | `row_sum` 增量累加 |
| output | 一次性 `P @ V` | `O` 增量累加 |
| 数值稳定性 | 减去整行 max | 每轮 merge 新旧参考值 |
| 数学结果 | 标准 attention | 与完整 attention 等价 |

这里“等价”指数学结果。浮点计算顺序不同，所以末位会有舍入差异。

## 七、常见错误与可观察症状

| 错误 | 破坏的状态关系 | 可观察症状 |
|---|---|---|
| `row_max` 初始化为 0 | 负 score 也错误地变成小于 1 的指数 | 第一块结果偏小，甚至整行输出错误 |
| 新的最大值出现后只更新 `row_max` | 旧 `row_sum` 与旧 `O` 仍在旧尺度 | 输出随 K/V block 顺序变化 |
| 只缩放 `row_sum`，不缩放 `O` | softmax 分子与分母尺度不一致 | 输出整体比例错误 |
| 每处理一个 block 就归一化 | 提前丢掉后续 positions 的贡献 | 越靠后的 block 贡献越被重复缩放 |
| 使用 `exp(S)` 而不减参考值 | score 较大时溢出 | 出现 `inf`、`nan`，或整行退化为均匀分布 |
| 把 tile max 当作完整行 max | 仍需要后续 block 的贡献 | 第一块看似正确，长序列上误差逐渐增大 |
| 自然指数和 base-2 指数混用 | scale 因子不一致 | 输出误差大，且随 `sqrt(d)` 改变 |
| 忘记 causal mask | 被 mask 的 positions 仍进入分母 | 输出泄露未来 token |

调试时先按 block 顺序打印：

```text
tile_max
row_max
alpha
row_sum
O
```

如果最终结果错误，但每个 block 的 `row_sum` 和 `O` 都与
full-row reference 的局部版本一致，问题通常在最后的归一化或 layout；
如果从第二个 block 开始偏离，优先检查 `alpha` 是否同时作用于
`row_sum` 和 `O`。

## 八、与 FA4 代码的对应关系

本节的通用算法对应课程伪代码中的：

```text
row_max
row_sum
O
candidate_max
acc_scale
P = exp((S - new_reference) * scale_log2)
row_sum = row_sum * acc_scale + rowsum(P)
O = O * acc_scale + P @ V_block
```

FA4 在这里做了两个进一步的实现选择：

1. 指数使用 base-2，引入 `scale_log2 = log2(e) / sqrt(d)`。
2. 不要求每一轮都把 `row_max` 更新为真实最大值，而是在新旧参考值
   差距超过阈值时再执行 rescale。

因此下一节会把这个通用版：

```text
new_max = max(row_max, tile_max)
alpha = exp(row_max - new_max)
```

替换为 FA4 的：

```text
delta = (row_max - candidate_max) * scale_log2
delta < -8 时切换参考值，否则保留旧参考值
```

## 九、自测题与答案

### 1. 为什么 FlashAttention 仍然需要 `row_max`，却不需要保存完整 `S`？

答：softmax 的数值稳定性需要逐行参考值，但参考值可以随着 K/V block
增量更新。完整 `S` 对后续已经处理的内容不再需要，只需要保留
`row_max`、`row_sum` 和累计的 `O`。

### 2. 新 block 的 `tile_max` 比旧 `row_max` 大时，为什么要缩放旧状态？

答：旧 `row_sum` 和旧 `O` 使用旧参考值计算。新的指数参考值更大后，
旧状态要乘：

```text
alpha = exp(old_row_max - new_row_max)
```

才能和新 block 的贡献处于同一尺度。

### 3. 为什么不能只更新 `row_sum`，不更新 `O`？

答：`row_sum` 和 `O` 分别是 softmax 的分母和分子。参考值改变时，
两者都必须乘同一个 `alpha`。只改一个会破坏分子与分母的相对比例。

### 4. tiled 版本什么时候执行最终除法？

答：只有当前 Q block 的所有 K/V blocks 都处理完成后，才执行
`O / row_sum`。每个 K/V block 结束时都只是 partial state。

### 5. 本节的 tiled 结果是否应该与标准 attention 完全逐位相等？

答：不应该要求逐位相等。两者数学等价，但浮点加法和乘法的顺序不同，
通常只应满足很小的数值误差。本文例子得到的 max error 是
`6.661338147750939e-16`。

## 十、FA4 conditional rescaling：delta、阈值 8 与 acc_scale

### 本次讲解位置

```text
本次讲解位置
章节：chapter_flash_attention
小节：算法结构 / 重缩放与结果写回
知识点：Conditional rescaling、delta、阈值 8、acc_scale 与两级重缩放筛选
上次：Tile 分解与 online softmax 的 row_max / row_sum / O 三状态
下次：S / P / O 的 TMEM layout 与分时复用
PTX：tcgen05.ld / tcgen05.st 读取和写回 O；
     statistics named barrier 传递逐行 acc_scale
```

### 一、这一节解决的问题

基础 online softmax 每读一个 K/V block，都会做：

```text
candidate_max = max(row_max, rowmax(S))
new_ref = candidate_max
alpha = exp(row_max - new_ref)
row_sum = row_sum * alpha + sum(P)
O = O * alpha + P @ V
```

这种做法总是把参考值推进到当前真实最大值，因此只要出现更大的
score，就必须重缩放已经累计的 `O`。在 FlashAttention 的 GPU kernel
中，`O` 位于 TMEM，重缩放不是一次简单的寄存器乘法，而是：

```text
O in TMEM
    -> tcgen05.ld
O in registers
    -> multiply by acc_scale
O in registers
    -> tcgen05.st
O in TMEM
```

如果每个 K/V block 都触发这条路径，会增加 TMEM load、register
压力、乘法、TMEM store 和同步开销。

FA4 的关键观察是：

> 指数参考值不必每一轮都等于真实最大值。只要 `row_sum` 和 `O` 使用
> 同一个参考值，最终 `O / row_sum` 的比值就正确。

因此可以先保留旧参考值。只有候选最大值比旧参考值高出太多时，才
切换到候选值并执行 correction。

这不是近似答案。不同参考值会让 `P`、`row_sum` 和 `O` 同时乘上一个
公共尺度，最终归一化时这个尺度会被约掉。真正会引入误差的是指数近似
和浮点舍入，不是选择哪一个公共参考值。

### 二、数学定义

代码使用 base-2 指数：

```text
scale_log2 = log2(e) / sqrt(d)

exp((s - r) / sqrt(d))
= 2 ** ((s - r) * scale_log2)
```

定义：

```text
r_old        = 当前已经使用的指数参考值
candidate    = max(r_old, rowmax(S))
delta        = (r_old - candidate) * scale_log2
```

因为：

```text
candidate >= r_old
```

所以：

```text
delta <= 0
```

`-delta` 表示候选参考值比旧参考值高出了多少个 base-2 exponent units。

FA4 使用：

```text
rescale_threshold = 8
```

判断规则是：

```text
delta >= -8：保留旧参考值，acc_scale = 1
delta <  -8：切换到 candidate，acc_scale = exp2(delta)
```

### 三、为什么阈值是 8

如果继续使用旧参考值，当前 block 中最大的 `P` 元素最多可能是：

```text
max(P)
= 2 ** ((candidate - r_old) * scale_log2)
= 2 ** (-delta)
```

当 `-delta = 8` 时：

```text
max(P) = 2 ** 8 = 256
```

也就是说，允许新 block 相对于旧参考值最多放大 256 倍。

如果切换到候选参考值，旧状态需要乘：

```text
acc_scale = 2 ** delta = 2 ** -8 = 1 / 256
```

所以阈值 8 的两边是同一个尺度的两种表达：

```text
继续使用旧参考值：
    新 block 的 P 最大可以到旧尺度的 256 倍

切换到新参考值：
    旧 row_sum 和旧 O 要缩小到新尺度的 1/256
```

阈值取 8 是在两个成本之间折中：

```text
阈值越大：
    越少触发 O correction
    但 P 的指数范围越大，精度和溢出风险越大

阈值越小：
    P 的范围更受控
    但更频繁地执行 TMEM -> registers -> TMEM
```

### 四、三个分支

| 情况 | `new_ref` | `acc_scale` | 当前 block 的 `P` | 旧 `O` 的处理 |
|---|---|---:|---|---|
| 第一个 K/V block | `candidate_max` | `1` | `2 ** ((S - candidate) * scale_log2)` | 没有旧 `O`，直接初始化 |
| `delta >= -8` | 保留 `r_old` | `1` | `2 ** ((S - r_old) * scale_log2)` | 直接累加，不读回 `O` |
| `delta < -8` | `candidate_max` | `2 ** delta` | `2 ** ((S - candidate) * scale_log2)` | 先缩放旧 `O`，再累加 |

统一状态更新可以写成：

```text
row_max_safe = 0 if new_ref == -inf else new_ref
P = exp2((S - row_max_safe) * scale_log2)

row_sum = row_sum * acc_scale + sum(P)
O = O * acc_scale + P @ V_block
```

第一块需要单独处理，因为此时没有旧状态。后续两个分支都能套用同一个
状态更新公式，只是 `acc_scale` 是 `1` 或一个更小的正数。

还要处理全 mask 行的边界情况。如果旧参考值和当前 block 最大值都是
`-inf`，直接计算 `S - new_ref` 会得到 `-inf - (-inf)`。当前实现使用
`row_max_safe = 0`，让全部被 mask 的 scores 产生零权重。

### 五、完整可运行示例

下面的程序逐行模拟一个 query row 的 conditional rescaling。它不是 GPU
kernel，而是用于验证阈值分支、指数尺度和 `O` 更新公式。

文件名：`flash_attention_conditional_rescale_demo.py`

```python
import math

import numpy as np


def conditional_rescale_step(
    row_ref,
    row_sum,
    O,
    S,
    V,
    scale_log2=1.0,
    rescale_threshold=8.0,
    is_first=False,
):
    candidate_max = max(row_ref, float(np.max(S)))

    if is_first:
        new_ref = candidate_max
        acc_scale = 1.0
        decision = "first_block"
    else:
        delta = (row_ref - candidate_max) * scale_log2
        if delta >= -rescale_threshold:
            new_ref = row_ref
            acc_scale = 1.0
            decision = "keep_old_reference"
        else:
            new_ref = candidate_max
            acc_scale = 2.0 ** delta
            decision = "switch_to_candidate_max"

    safe_ref = 0.0 if new_ref == -math.inf else new_ref
    P = np.exp2((S - safe_ref) * scale_log2)
    block_O = P @ V
    new_row_sum = row_sum * acc_scale + float(np.sum(P))
    new_O = block_O if is_first else O * acc_scale + block_O

    return {
        "candidate_max": candidate_max,
        "delta": None if is_first else (row_ref - candidate_max) * scale_log2,
        "decision": decision,
        "new_ref": new_ref,
        "acc_scale": acc_scale,
        "P": P.tolist(),
        "block_O": block_O.tolist(),
        "row_sum": new_row_sum,
        "O": new_O.tolist(),
    }


def print_step(label, result):
    print(label)
    for key in (
        "candidate_max",
        "delta",
        "decision",
        "new_ref",
        "acc_scale",
        "P",
        "block_O",
        "row_sum",
        "O",
    ):
        print(f"{key}: {result[key]}")
    print()


def main():
    V = np.array([[1.0, 0.0], [0.0, 1.0]])

    keep = conditional_rescale_step(
        row_ref=2.0,
        row_sum=3.0,
        O=np.array([4.0, 6.0]),
        S=np.array([5.0, 4.0]),
        V=V,
    )
    print_step("case 1: delta >= -8, keep old reference", keep)

    switch = conditional_rescale_step(
        row_ref=2.0,
        row_sum=3.0,
        O=np.array([4.0, 6.0]),
        S=np.array([11.0, 10.0]),
        V=V,
    )
    print_step("case 2: delta < -8, switch reference", switch)


if __name__ == "__main__":
    main()
```

运行：

```bash
python flash_attention_conditional_rescale_demo.py
```

预期输出：

```text
case 1: delta >= -8, keep old reference
candidate_max: 5.0
delta: -3.0
decision: keep_old_reference
new_ref: 2.0
acc_scale: 1.0
P: [8.0, 4.0]
block_O: [8.0, 4.0]
row_sum: 15.0
O: [12.0, 10.0]

case 2: delta < -8, switch reference
candidate_max: 11.0
delta: -9.0
decision: switch_to_candidate_max
new_ref: 11.0
acc_scale: 0.001953125
P: [1.0, 0.5]
block_O: [1.0, 0.5]
row_sum: 1.505859375
O: [1.0078125, 0.51171875]
```

### 六、逐项数值推导

初始状态：

```text
row_ref = 2
row_sum = 3
O = [4, 6]
V = [[1, 0],
     [0, 1]]
scale_log2 = 1
rescale_threshold = 8
```

#### Case 1：`S = [5, 4]`

候选最大值：

```text
candidate = max(2, max(5, 4)) = 5
```

有符号差距：

```text
delta = (2 - 5) * 1 = -3
```

因为：

```text
-3 >= -8
```

所以保留旧参考值：

```text
new_ref = 2
acc_scale = 1
```

当前 block 继续相对于旧参考值计算：

```text
P = [2 ** (5 - 2), 2 ** (4 - 2)]
  = [8, 4]
```

这里 `P` 可以大于 1，这并不错误。它只是因为当前 block 暂时使用了
一个比真实最大值更小的参考值。最终 `O / row_sum` 会把公共尺度消掉。

当前 block 对 output 的贡献：

```text
block_O = P @ V
        = [8, 4]
```

更新分母：

```text
row_sum = 3 * 1 + (8 + 4)
        = 15
```

更新 output：

```text
O = [4, 6] * 1 + [8, 4]
  = [12, 10]
```

这条路径没有读取或写回 TMEM 中的旧 `O`，只执行普通累加。

#### Case 2：`S = [11, 10]`

候选最大值：

```text
candidate = max(2, 11) = 11
```

有符号差距：

```text
delta = (2 - 11) * 1 = -9
```

因为：

```text
-9 < -8
```

所以切换到候选参考值，并计算旧状态的转换系数：

```text
new_ref = 11
acc_scale = 2 ** -9
          = 1 / 512
          = 0.001953125
```

当前 block 的权重改为相对于 `11` 计算：

```text
P = [2 ** (11 - 11), 2 ** (10 - 11)]
  = [1, 0.5]
```

当前 block 的贡献：

```text
block_O = [1, 0.5]
```

旧分母先换到新尺度：

```text
old row_sum * acc_scale
= 3 / 512
= 0.005859375
```

再加入当前 block：

```text
row_sum = 0.005859375 + 1.5
        = 1.505859375
```

旧 output 也使用完全相同的 `acc_scale`：

```text
old O * acc_scale
= [4, 6] / 512
= [0.0078125, 0.01171875]
```

加入当前 block：

```text
O = [0.0078125, 0.01171875] + [1, 0.5]
  = [1.0078125, 0.51171875]
```

`row_sum` 和 `O` 都换到了 `new_ref = 11` 的尺度，因此后续 block
可以继续累加；最后的 `O / row_sum` 仍然是正确的 attention 输出。

### 七、真实 kernel 的两级筛选

Softmax 的 `row_sum` 位于 WG0/WG1 的 registers 中，直接在更新时乘
`acc_scale`。`O` 位于 TMEM 中，由 WG2 负责 correction。

Softmax 将逐行 `acc_scale` 写入 `sScale`：

```text
softmax WG0/WG1
    row_sum *= acc_scale
    sScale[row] = acc_scale
    statistics named barrier arrive

WG2
    wait statistics named barrier
    read acc_scale
    rescale O when needed
    p_o_rescale.arrive
    softmax_corr.empty.arrive
```

第一级筛选是逐行阈值判断：

```python
should_rescale = T.Select(acc_scale < T.float32(1.0), 1, 0)
```

只要 `acc_scale < 1`，说明这一行确实切换了参考值，需要 correction。

第二级筛选发生在 WG2 的一个 warp 内。WG2 有 128 个 threads，每个 warp
负责 32 行。由于 TMEM load/store 是 warp-collective 操作，不能只让
32 行中任意几个 lane 单独执行：

```python
any_needs_rescale = T.ptx.any_sync(0xFFFFFFFF, should_rescale)

if any_needs_rescale != 0:
    # 当前 warp 的 32 行统一走 TMEM -> registers -> TMEM
    # 其中 acc_scale == 1 的行只是乘 1
```

这两级筛选的效果是：

```text
某个 warp 的 32 行全部 acc_scale = 1
    -> 完全跳过 O 的 TMEM load / multiply / store

某个 warp 只要有一行 acc_scale < 1
    -> 该 warp 的 32 行统一执行数据路径
      不需要变化的行乘以 1
```

即使跳过数据操作，barrier arrivals 也不能跳过。因为 WG2 还要完成：

```text
p_o_rescale.arrive(i_q)
```

这允许 PV MMA 继续使用初始化或重缩放后的 `O`。

以及：

```text
softmax_corr.empty.arrive(1 - i_q)
```

这允许 softmax 重新写入对应的 `sScale` slot。

因此最准确的 mental model 是：

```text
conditional rescaling
= 先决定是否需要换尺度
+ 再决定当前 warp 是否真的搬运 O
+ 无论是否搬运，都继续推进生产者和消费者之间的 barrier 协议
```

### 八、常见错误与可观察症状

| 错误 | 破坏的约束 | 可观察症状 |
|---|---|---|
| 把 `delta` 当自然指数尺度 | 阈值 8 的含义改变 | 重缩放次数和指数范围与设计不符 |
| 认为保留旧参考值是近似计算 | 公共参考尺度可以在最终除法中约掉 | 误以为只有更新真实最大值才正确 |
| 切换参考值但只缩放 `row_sum` | softmax 分子与分母尺度不一致 | 行输出整体缩放错误 |
| 当前 block 的 `P` 仍使用旧参考值 | `P` 与已经 rescale 的旧状态不在同一尺度 | 新 block 贡献比例错误 |
| 阈值判断后跳过 barrier arrival | pipeline 的下一个等待者永远得不到信号 | kernel 卡住或超时 |
| `any_sync` 为 0 时跳过整个 WG2 | 其他 warp 的等待协议也无法完成 | 后续 MMA 或 softmax 停滞 |
| warp 内只让需要 rescale 的 lane 执行 | TMEM load/store 是 warp-collective | 编译失败、结果不完整或行为未定义 |

### 九、自测题与答案

#### 1. 保留旧参考值时，为什么 `P` 可以大于 1？

答：参考值只是所有指数共同减去的偏移量。只要旧的 `row_sum`、旧的
`O` 和当前 block 的 `P` 都使用同一个旧参考值，最终 `O / row_sum`
仍然正确。`P` 大于 1 只是说明当前 block 的最大 score 高于这个暂时
保留的参考值。

#### 2. 阈值 8 对应的最大尺度差是多少？

答：

```text
2 ** 8 = 256
```

也就是最多允许新 block 相对于旧参考值放大 256 倍。

#### 3. `acc_scale` 为什么必须同时作用于 `row_sum` 和 `O`？

答：它们分别是 softmax 的未归一化分母和分子。只缩放其中一个会改变
两者比例，最终 `O / row_sum` 就不再对应同一个 softmax 分布。

#### 4. 如果某个 warp 的 32 行都不需要 correction，WG2 可以完全不做事吗？

答：不可以。它可以跳过 TMEM load、multiply 和 store，但必须继续执行
`p_o_rescale.arrive` 与 `softmax_corr.empty.arrive`，否则 PV MMA
无法继续，softmax 也无法复用 `sScale` slot。

#### 5. 为什么 WG2 不是只让真正需要重缩放的那些 lane 执行？

答：`O` 的 correction 使用 `tcgen05.ld` 和 `tcgen05.st`，它们都是
warp-collective tile operations。只要当前 warp 有一行需要重缩放，该
warp 的 32 行就统一执行；`acc_scale = 1` 的行只是乘 1。

## 十一、下一知识点

下一步进入 `S / P / O` 的 TMEM layout 与分时复用：

```text
S、P、O 分别使用哪些 TMEM rows / columns
fp32 accumulator 与 fp16 MMA operand 如何共享同一块物理 TMEM
两个 Q stages 如何轮流复用 S、P、O regions
哪些 region 会重叠，哪些 barrier 防止过早读取或覆盖
```

这会连接前面的 online softmax 数学更新和后面 QKᵀ / PV MMA、epilogue
的具体数据路径。
