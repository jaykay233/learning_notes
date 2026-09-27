# Flash Attention：Tile 分解与 Online Softmax 三状态

这篇笔记开始 `chapter_flash_attention`。上一章已经把 TMA、软件流水线、
persistent scheduling、warp specialization 和 two-CTA cooperative MMA
整理完成。FlashAttention 把同一套数据路径移到 attention 中：

```text
QK^T MMA -> softmax -> PV MMA
```

本节先讲最基础、也是后面所有 FA4 优化共同依赖的一个知识点：
不保存完整 score matrix，如何按 K/V block 计算等价的 attention 输出。

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
[ ] 2. FA4 conditional rescaling、delta 与 acc_scale
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

## 下一知识点

下一步进入 FA4 的 conditional rescaling：

```text
delta = (row_max_old - candidate_max) * scale_log2
delta >= -8：保留旧参考值，acc_scale = 1
delta < -8：切换到 candidate_max，acc_scale = exp2(delta)
```

重点解释阈值为什么取 8、旧 `O` 是否需要在 TMEM 与 registers 之间
往返，以及 WG2 如何只重缩放真正需要 correction 的行。
