# Flash Attention：Tile 分解、Online Softmax 与 Conditional Rescaling

这篇笔记开始 `chapter_flash_attention`。上一章已经把 TMA、软件流水线、
persistent scheduling、warp specialization 和 two-CTA cooperative MMA
整理完成。FlashAttention 把同一套数据路径移到 attention 中：

```text
QK^T MMA -> softmax -> PV MMA
```

目前已经讲完 attention tile 分解、FA4 conditional rescaling、`S/P/O`
的 TMEM layout、QKᵀ MMA、softmax 和 PV MMA 数据路径。接下来需要把
这些数据路径映射到具体执行者：CTAs 中的哪些 warpgroups 和 warps
负责 softmax、correction、TMA 与 MMA issue。

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
[x] 3. S / P / O 的 TMEM layout 与分时复用
[x] 4. QK^T MMA、softmax、PV MMA 的数据路径
    [x] 4.1 QK^T MMA：SMEM 中的 Q/K -> TMEM 中的 S
    [x] 4.2 Softmax：S TMEM -> registers -> P TMEM
    [x] 4.3 PV MMA：P TMEM + V SMEM -> O TMEM
[ ] 5. Warp 角色、register 分配与 barrier 分工
    [x] 5.1 Warp 角色地图
    [ ] 5.2 Register 分配与 setmaxnreg
    [ ] 5.3 Barrier 分工与角色交接
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

## 十一、S / P / O 的 TMEM layout 与分时复用

### 本次讲解位置

```text
本次讲解位置
章节：chapter_flash_attention
小节：TMEM 布局与复用
知识点：S / P / O 的 512-column 划分，以及 fp16 P 对 fp32 S 后半块的分时复用
上次：Conditional rescaling、delta、阈值 8 与 acc_scale
下次：QK^T MMA、softmax、PV MMA 的数据路径
PTX：tcgen05.mma、tcgen05.ld / tcgen05.st、tcgen05.commit、
     tcgen05.wait::ld / wait::st
```

### 为什么现在讲这个

前面的课程已经说明 `S`、`P`、`O` 各自的数学作用，但仅知道三种 tile 的
含义，仍然无法回答一个关键问题：FA4 为什么能在固定的 512-column TMEM
中同时支持两个 Q stages。

32-bit fp32 view 与 16-bit fp16 view 的 alias，正是这个容量问题的
答案。`S` 和 `O` 已经占满 512 columns，如果不知道同一个 physical
column 可以按两个 fp16 slots 重新索引，就会误以为 `P` 必须再申请
128 个独立 columns；如果不知道逻辑 slot `s` 映射到
`column s // 2`、`half s % 2`，也无法推导 `P0` 为什么恰好占用
`[64, 128)` 并覆盖 `S0` 的后半部分。

因此，本知识点不是单纯介绍一个 dtype 转换，而是把逻辑 tile 的形状、
TMEM 的物理容量和 `P` 对 `S` 的分时复用连接起来。掌握它之后，才能
判断哪些 region 可以重叠、哪些数据必须长期保留，以及后续 barrier
究竟在保护哪一段物理存储。

### 一、这一节解决的问题

FA4 kernel 为每个 CTA 申请：

```text
128 TMEM rows × 512 physical columns
每个 physical column = 32 bits
```

Q pipeline 有两个 stages，分别由 WG0 和 WG1 处理。每个 stage 至少需要：

| buffer | logical shape | fp32 physical columns |
|---|---:|---:|
| `S` | `128 × 128` fp32 scores | 128 |
| `O` | `128 × 128` fp32 accumulator | 128 |

`S` 和 `O` 合计：

```text
2 stages × (128 + 128) = 512 columns
```

也就是说，仅 `S` 和 `O` 就已经把 512-column allocation 全部用完。
如果 `P` 再单独申请一块 fp32 空间，需要额外的 256 columns；即使
`P` 用 fp16，也需要额外的 128 physical columns。两者都放不下。

解法不是扩大 TMEM，而是利用 softmax 的数据生命周期：

```text
QK^T MMA 写出完整 S
        ↓
softmax 将完整 S 读入 registers
        ↓
S 在 TMEM 中的旧内容不再需要
        ↓
把 fp16 P 写回 S 的后半部分
```

因此，`P` 与 `S` 的物理范围重叠，但它们不是同时活着。这种关系是
**分时复用**，不是把 `S` 和 `P` 同时塞进同一列。

### 二、physical column 与 fp16 alias

源码先建立 fp32 buffer：

```text
tmem: 128 × 512 fp32
```

然后执行 `move_base_to(0)`，让第二个 buffer 从同一个物理起点开始：

```text
tmem_as_f16: 128 × 1024 fp16
```

两个 buffer 每行包含的总 bits 相同：

```text
tmem:         512 × 32 bits = 16384 bits
tmem_as_f16: 1024 × 16 bits = 16384 bits
```

所以 `tmem_as_f16` 不是额外申请的一块 TMEM，而是同一块 bits 的 fp16
索引方式。一个 32-bit physical column 包含两个 fp16 slots：

```text
physical column p
┌────────────────┬────────────────┐
│ fp16 slot 2p   │ fp16 slot 2p+1 │
└────────────────┴────────────────┘
```

映射公式是：

```text
fp16 logical slot s
    -> physical column s // 2
    -> fp16 half     s % 2
```

例如：

```text
slot 128 -> physical column 64, low half
slot 129 -> physical column 64, high half
slot 130 -> physical column 65, low half
slot 131 -> physical column 65, high half
```

### 三、源码中的三个 views

课程完整布局定义如下：

```python
tmem_pool = T.TMEMPool(
    pool,
    total_cols=N_COLS_TMEM,
    cta_group=CTA_GROUP,
    tmem_addr=tmem_addr,
    alloc_warp=12,
    dealloc_warp=0,
)
tmem = tmem_pool.alloc((128, N_COLS_TMEM), "float32")
tmem_pool.move_base_to(0)
tmem_as_f16 = tmem_pool.alloc((128, N_COLS_TMEM * 2), "float16")
tmem_pool.commit()

S_region = T.meta_var(
    tmem.rearrange("m (s n) -> s m n", n=MMA_N)
)
O_region = S_region
P_region = T.meta_var(
    tmem_as_f16.rearrange("m (s two n) -> s two m n", two=2, n=MMA_N)
)
```

其中：

```text
N_COLS_TMEM = 512
MMA_N = BLK_N = 128
stage 数量 = 2
```

`S_region` 把 fp32 第二维 `512` 分解成：

```text
4 × 128 fp32 blocks
```

于是：

```text
S0 = S_region[0]
S1 = S_region[1]
O0 = S_region[2]
O1 = S_region[3]
```

`P_region` 把 fp16 第二维 `1024` 分解成：

```text
4 个 128-column blocks × 2 halves × 128 fp16 values
```

这里的“4”不是四个 Q pipeline stages，而是 fp16 alias 覆盖整块 512 physical
columns 后得到的四个 128-column blocks。它们分别对应底层已经存在的
`S0 / S1 / O0 / O1`：

| `s` | fp16 logical slots | physical columns | fp32 区域 |
|---:|---:|---:|---|
| 0 | `[0, 256)` | `[0, 128)` | `S0` |
| 1 | `[256, 512)` | `[128, 256)` | `S1` |
| 2 | `[512, 768)` | `[256, 384)` | `O0` |
| 3 | `[768, 1024)` | `[384, 512)` | `O1` |

`two=2` 再把每个 256-fp16-slot block 分成两个 128-fp16-slot 的 half。例如
`s=0` 的 high half 使用 fp16 slots `[128, 256)`，对应 physical columns
`[64, 128)`，因此它就是 `P0`。同理，`s=1` 的 high half 对应 `P1`。
`s=2`、`s=3` 对应 O accumulator，不会拿来保存 P。因此 Q pipeline 仍然只有
两个 stages，`q_stage` 的取值仍只有 0 和 1。

源码实际使用每个 stage 的 high half：

```text
P0 = P_region[0, 1, :, :]
P1 = P_region[1, 1, :, :]
```

因此各区域的物理 column 范围是：

| Region | 数据 | 逻辑来源 | 物理 columns |
|---|---|---|---:|
| `S0` | 128 个 fp32 scores | `tmem[:, 0:128]` | `[0, 128)` |
| `P0` | 128 个 fp16 weights | `tmem_as_f16[:, 128:256]` | `[64, 128)` |
| `S1` | 128 个 fp32 scores | `tmem[:, 128:256]` | `[128, 256)` |
| `P1` | 128 个 fp16 weights | `tmem_as_f16[:, 384:512]` | `[192, 256)` |
| `O0` | 128 个 fp32 accumulators | `tmem[:, 256:384]` | `[256, 384)` |
| `O1` | 128 个 fp32 accumulators | `tmem[:, 384:512]` | `[384, 512)` |

用一张物理列图表示：

```text
columns:  0        63 64        127 128       191 192       255 256       383 384       511
          |----------|-------------|-----------|-------------|-------------|-------------|
S/O:       S0 first  S0 second     S1 first    S1 second     O0            O1
alias:                P0                         P1
```

`stage` 的下标来自 Q pipeline：

```text
WG0 处理 q_stage 0 -> 使用 S0 / P0 / O0
WG1 处理 q_stage 1 -> 使用 S1 / P1 / O1
```

### 四、完整可运行地址计算脚本

下面的程序不依赖 CUDA，用纯 Python 复现课程中的 view 分解和 physical
column 映射。文件名：
`flash_attention_tmem_layout_demo.py`

```python
from __future__ import annotations


PHYSICAL_COLS = 512
FP16_LOGICAL_COLS = PHYSICAL_COLS * 2
MMA_N = 128
STAGES = 2


def fp16_slot_to_physical(slot: int) -> tuple[int, int]:
    if not 0 <= slot < FP16_LOGICAL_COLS:
        raise ValueError(
            f"fp16 slot {slot} outside [0, {FP16_LOGICAL_COLS})"
        )
    return slot // 2, slot % 2


def fp32_region(stage: int, kind: str) -> tuple[int, int]:
    if stage not in (0, 1):
        raise ValueError("stage must be 0 or 1")
    if kind == "S":
        start = stage * MMA_N
        return start, start + MMA_N
    if kind == "O":
        start = (STAGES + stage) * MMA_N
        return start, start + MMA_N
    raise ValueError(kind)


def p_slot(stage: int, n: int, half: int = 1) -> int:
    if stage not in (0, 1):
        raise ValueError("stage must be 0 or 1")
    if not 0 <= n < MMA_N:
        raise ValueError(f"n must be in [0, {MMA_N})")
    if half not in (0, 1):
        raise ValueError("half must be 0 or 1")
    return stage * (2 * MMA_N) + half * MMA_N + n


def p_physical(stage: int, n: int) -> tuple[int, int]:
    return fp16_slot_to_physical(p_slot(stage, n))


print("regions")
for stage in range(STAGES):
    for kind in ("S", "P", "O"):
        if kind == "P":
            lo = p_physical(stage, 0)[0]
            hi = p_physical(stage, MMA_N - 1)[0]
            print(f"{kind}{stage}: physical columns [{lo}, {hi + 1})")
        else:
            lo, hi = fp32_region(stage, kind)
            print(f"{kind}{stage}: physical columns [{lo}, {hi})")

print("\nfp16 alias trace")
for stage in range(STAGES):
    for n in (0, 1, 2, 3, 126, 127):
        slot = p_slot(stage, n)
        col, half = fp16_slot_to_physical(slot)
        print(
            f"P{stage}[n={n:3d}] -> fp16 slot {slot:4d} "
            f"-> physical column {col:3d}, half {half}"
        )

print("\noverlap check")
for stage in range(STAGES):
    s_lo, s_hi = fp32_region(stage, "S")
    p_lo = p_physical(stage, 0)[0]
    p_hi = p_physical(stage, MMA_N - 1)[0] + 1
    overlap = range(max(s_lo, p_lo), min(s_hi, p_hi))
    print(f"S{stage} intersect P{stage} = [{overlap.start}, {overlap.stop})")
```

运行：

```bash
python3 flash_attention_tmem_layout_demo.py
```

预期输出：

```text
regions
S0: physical columns [0, 128)
P0: physical columns [64, 128)
O0: physical columns [256, 384)
S1: physical columns [128, 256)
P1: physical columns [192, 256)
O1: physical columns [384, 512)

fp16 alias trace
P0[n=  0] -> fp16 slot  128 -> physical column  64, half 0
P0[n=  1] -> fp16 slot  129 -> physical column  64, half 1
P0[n=  2] -> fp16 slot  130 -> physical column  65, half 0
P0[n=  3] -> fp16 slot  131 -> physical column  65, half 1
P0[n=126] -> fp16 slot  254 -> physical column 127, half 0
P0[n=127] -> fp16 slot  255 -> physical column 127, half 1
P1[n=  0] -> fp16 slot  384 -> physical column 192, half 0
P1[n=  1] -> fp16 slot  385 -> physical column 192, half 1
P1[n=  2] -> fp16 slot  386 -> physical column 193, half 0
P1[n=  3] -> fp16 slot  387 -> physical column 193, half 1
P1[n=126] -> fp16 slot  510 -> physical column 255, half 0
P1[n=127] -> fp16 slot  511 -> physical column 255, half 1

overlap check
S0 intersect P0 = [64, 128)
S1 intersect P1 = [192, 256)
```

### 五、stage 0 的完整执行顺序

以 `S0` 和 `P0` 为例，一个 K/V block 中的数据流是：

```text
1. QK^T MMA 写 S0
   physical columns [0, 128)

2. tcgen05.commit 通知 s_ready

3. softmax wait s_ready
   通过四次 tcgen05.ld 将完整 S0 读入 registers

4. softmax 在 registers 中计算 fp16 P0

5. tcgen05.st 将 P0 写入
   physical columns [64, 128)
   这会覆盖 S0 的后 64 columns

6. tcgen05.wait::st 确认 P0 stores 完成

7. softmax 通知 p_o_rescale / p_ready_2

8. PV MMA 读取 P0 并更新 O0

9. P0 已消费后，后续 QK^T MMA 才能重新使用
   S0/P0 共用的物理区域
```

这里每一步都同时涉及三类约束：

| 约束 | 本课中的含义 |
|---|---|
| Layout | 同一个 logical `(row, n)` 由哪个 view、哪个 physical column 保存 |
| Lifetime | `S0` 的旧 fp32 内容何时失去价值，`P0` 何时可以覆盖它 |
| Synchronization | 哪个 consumer 必须等哪次 completion，何时才能读或覆盖 |

`tcgen05.ld` 和 `tcgen05.st` 是异步 issue 的。等待 `s_ready` 只表示
QKᵀ MMA 已经写完 `S`；它不表示 softmax 的 TMEM loads 已经完成。
同理，softmax 发出 TMEM stores 后，必须以 `tcgen05.wait::st()` 确认
这些 stores 完成，才能向 PV MMA 报告 `P` 已可用。

`P0` 写完后，PV MMA 和下一轮 QKᵀ MMA 也不能只依赖 Python 源码的
先后顺序。课程 kernel 由 WG3 warp 0 的同一个 issuing thread 按固定
序列发出两类 MMA，lowering 必须保留必要的 `tcgen05` 依赖，确保下一
轮写 `S0` 不会先于本轮消费完 `P0`。

### 六、常见错误与可观察症状

| 错误 | 破坏的约束 | 可观察症状 |
|---|---|---|
| 认为 `tmem_as_f16` 是第二块 1024-column allocation | 总容量超过 512 columns | allocation 冲突、越界或 kernel 无法按预期启动 |
| 把 `P0` 按 fp32 计算为 128 physical columns | 误以为 `P0` 占用 `[0, 128)` 或 `[128, 256)` | 地址推导和覆盖范围全部错误 |
| 将 `P_region[i_q, 1]` 改成未同步的另一个 half | producer 与 consumer view 不一致 | PV MMA 读到旧数据、零值或错误 scores |
| `P` 覆盖前没有把完整 `S` 读入 registers | 后半 scores 被提前破坏 | attention 后半 keys 的贡献错误或归零 |
| 写完 `P` 后省略 `tcgen05.wait::st()` | PV MMA 可能读取尚未完成的 stores | 偶发错误、部分列错误、数据竞争 |
| 当前 `P` 尚未消费就让下一轮 QKᵀ 覆盖同一区域 | write-after-read hazard | 输出随调度和时序变化，表现为非确定性错误 |
| 把 overlap 当成“两个 tile 同时有效” | 混淆分时复用与并行存储 | 错误地同时读取 S/P，或漏掉必须的 barrier |

### 七、自测题与答案

#### 1. 为什么 `P` 不需要第三块 128-column 独立空间？

答：softmax 先把完整 `S` 读入 registers，之后 TMEM 中的旧 `S` 不再
需要。每个 stage 的 `P` 是 `128` 个 fp16 values，只占
`128 × 16 / 32 = 64` 个 physical columns，因此可以覆盖同一个 stage
中 `S` 的后 64 columns。

#### 2. `P0` 的逻辑列 `n=10` 对应哪个 physical column 和 half？

答：

```text
P_region[0, 1, :, 10]
    -> tmem_as_f16[:, 128 + 10]
    -> fp16 slot 138
    -> physical column 138 // 2 = 69
    -> half 138 % 2 = 0
```

所以它位于 physical column 69 的 low half。

#### 3. `S0` 和 `P0` 的物理范围重叠，为什么不会互相破坏？

答：因为它们的生命周期经过同步后错开。QKᵀ MMA 先把 `S0` 写入
`[0, 128)`，softmax 等到 `s_ready` 后把完整 `S0` 读走；只有在这之后
`P0` 才覆盖 `[64, 128)`。重叠的是存储位置，不是同时有效的数据。

#### 4. 下一轮 QKᵀ MMA 重新写 `S0` 前，至少必须保证什么？

答：必须保证本轮 PV MMA 已经消费完 `P0`，因为 `P0` 位于 `S0` 的后
半部分。否则下一轮 QKᵀ 写 `[0, 128)` 时可能覆盖尚未被 PV MMA 读取的
`P0`。

#### 5. 为什么不能只根据源码先后顺序判断 `S`、`P`、`O` 的复用安全？

答：QKᵀ MMA、TMEM load/store 和 PV MMA 都可能异步执行。源码顺序只
描述 issuing thread 发出的顺序，不自动证明 Tensor Core stores、
softmax loads 或 MMA reads 已经完成。必须结合 `s_ready`、
`tcgen05.wait::st()`、`p_o_rescale`、`p_ready_2` 以及 `tcgen05`
的 issuing 依赖共同判断。

## 十二、QKᵀ MMA 数据路径

### 本次讲解位置

```text
本次讲解位置
章节：chapter_flash_attention
小节：QKᵀ MMA 与 PV MMA / QKᵀ MMA
知识点：SMEM 中的 Q、K 如何经过 tcgen05.mma 生成 TMEM 中的 S
上次：S / P / O 的 TMEM layout 与分时复用
下次：TMEM 中的 S 如何读入 registers 并计算 P
PTX：tcgen05.mma、tcgen05.commit、tcgen05.wait
```

### 为什么现在讲这个

上一节已经知道 `S0` 位于 physical columns `[0, 128)`、`S1` 位于
`[128, 256)`，但地址正确并不代表数据会正确到达。QKᵀ MMA 是第一个真正向
这些 `S` region 写入数据的 producer，它同时涉及 SMEM 输入、TMEM 输出和
异步完成通知。

如果只知道 `Tx.warp.gemm_async` 的名字，很容易误以为函数返回时 `S` 已经
可用，直接在下一行读取；也可能误以为 `K` 需要先复制一份转置矩阵，或者
把 `s_ready.arrive` 理解成普通线程写完数据后的软件到达。这些误解都会导致
错误结果、偶发数据竞争或 barrier 死锁。本节把“谁发起、硬件读哪里、写到
哪里、何时通知消费者”连接成一条完整的数据路径。

### 一、心智模型

固定一个 Q stage 和一个 K block，QKᵀ MMA 完成的是：

```text
Q_block [128, HEAD_DIM] @ K_block^T [HEAD_DIM, 128]
    -> S [128, 128]
```

矩阵元素之间的关系是：

```text
S[row, col] = sum_d Q[row, d] * K[col, d]
```

这里：

| 名称 | 含义 |
|---|---|
| `row` | Q tile 中的 query 行 |
| `col` | 当前 K tile 中的 key 行，也是 S 的列 |
| `d` | head dimension |
| `q_stage` | 当前 Q pipeline slot，取值 0 或 1 |
| `kv_stage` | 当前 K/V pipeline slot |
| `S_region[q_stage]` | 当前 MMA 的 TMEM 输出区域 |

Q 和 K 都保存在 SMEM。矩阵计算由 Blackwell Tensor Core 执行，结果直接写入
TMEM。普通 threads 不负责逐元素执行这个矩阵乘法，也不把 S 从 Tensor Core
逐项搬运到 TMEM。

### 二、课程中的真实调用

课程代码对这一过程的表达是：

```python
Tx.warp.gemm_async(
    S_region[q_stage, :, :],
    Q_smem[q_stage, 0:BLK_M, 0:HEAD_DIM],
    K_smem[kv_stage, 0:BLK_N, 0:HEAD_DIM],
    dispatch="tcgen05",
    cta_group=CTA_GROUP,
)
if T.ptx.elect_sync():
    s_ready.arrive(q_stage)
```

这里的参数含义是：

| 参数 | 作用 | 所在位置 |
|---|---|---|
| `S_region[q_stage]` | MMA 的目的地 | TMEM |
| `Q_smem[q_stage]` | A operand | SMEM |
| `K_smem[kv_stage]` | B operand | SMEM |
| `dispatch="tcgen05"` | 选择 Blackwell MMA 路径 | hardware dispatch |
| `cta_group=CTA_GROUP` | 选择单 CTA 或 CTA-pair 范围 | MMA scope |

`K_smem` 仍然按 `[BLK_N, HEAD_DIM]` 保存，并不需要软件预先生成一份转置
矩阵。`gemm_async` 知道这是 `Q @ K^T`，SMEM descriptor 会告诉 Tensor Core
怎样遍历 K tile。多复制一次转置矩阵只会浪费 SMEM 和搬运带宽。

Q、K 能安全作为 operand 读取，还必须先满足：

```text
q_load.full 确认 Q_smem[q_stage] 已由 TMA 写入
kv_load.full 确认 K_smem[kv_stage] 已由 TMA 写入
```

这两个条件由 MMA issuing warp 在发起 MMA 前等待。源码顺序本身不能证明
TMA 已经完成。

### 三、Scope、layout 与 dispatch

这个 tile operation 的执行边界如下：

```text
Scope：WG3 warp 0 中的 elected lane 发起
输入： SMEM 中的 Q tile 和 K tile
输出： TMEM 中的 S tile
Dispatch：tcgen05.mma
交接：s_ready -> softmax warpgroup
```

`Tx.warp.gemm_async` 是 tile primitive，不等价于 32 个 lanes 各自计算一部分
标量乘法。底层 `tcgen05.mma` 具有 single-thread issue 语义：一个 elected
lane 发出矩阵级操作，Tensor Core 随后异步执行。

`S_region[q_stage, :, :]` 的逻辑 shape 是 `[128, 128]`。它不申请新的
TMEM，而是上一节已经建立的 fp32 view：

```text
q_stage = 0 -> S0 使用 physical columns [0, 128)
q_stage = 1 -> S1 使用 physical columns [128, 256)
```

QKᵀ MMA 每次为当前 K/V block **重新产生** S，不累加到上一轮的 S。PV MMA
才会把结果累加到长期存在的 `O` accumulator。

### 四、`s_ready.arrive` 不是普通软件到达

`s_ready` 是跟踪 Tensor Core 完成事件的 `TCGen05Bar`。这里的
`s_ready.arrive(q_stage)` 应理解为发出 `tcgen05.commit`：

```text
1. elected lane 发起 QKᵀ MMA
2. elected lane 执行 commit
3. Tensor Core 继续异步写 S
4. S 写完后，硬件向 s_ready[q_stage] 报告完成
5. softmax warpgroup 等待到对应 phase 后读取 S
```

因此下面两条时间线并不相同：

```text
gemm_async 返回
    != S 已经写完

线程执行到 s_ready.arrive
    != 线程已经写完 S
```

前者只表示 MMA 已提交给硬件；后者只表示 issuing lane 已登记完成通知。
只有 `s_ready` 的硬件 phase 发生翻转，才表示 QKᵀ MMA 的结果可以安全读取。
如果所有 lanes 都执行 `s_ready.arrive`，barrier 的 expected arrival count
也会错误。

### 五、完整可运行数据路径模拟

下面的脚本不依赖 CUDA，用纯 Python 计算一个小型 `QK^T`，并复现课程中的
stage-to-column 映射和完成通知顺序。文件名：
`flash_attention_qk_mma_trace.py`

```python
from __future__ import annotations


MMA_N = 128


def qk_mma(
    q: list[list[int]],
    k: list[list[int]],
) -> list[list[int]]:
    rows_q = len(q)
    rows_k = len(k)
    head_dim_q = len(q[0])
    head_dim_k = len(k[0])

    if head_dim_q != head_dim_k:
        raise ValueError("Q and K must have the same head dimension")
    if rows_q > MMA_N or rows_k > MMA_N:
        raise ValueError("demo tile exceeds the real MMA_N")

    result = [[0] * rows_k for _ in range(rows_q)]
    for row in range(rows_q):
        for col in range(rows_k):
            result[row][col] = sum(
                q[row][d] * k[col][d]
                for d in range(head_dim_q)
            )
    return result


def s_region_column(q_stage: int, n: int) -> int:
    if q_stage not in (0, 1):
        raise ValueError("q_stage must be 0 or 1")
    if not 0 <= n < MMA_N:
        raise ValueError("S column must be in [0, 128)")
    return q_stage * MMA_N + n


class ReadyBarrierTrace:
    def __init__(self) -> None:
        self.phase = 0
        self.committed = False
        self.completed = False

    def issue_and_commit(self) -> None:
        self.committed = True
        self.completed = False
        print("WG3 elected lane: issue tcgen05.mma")
        print("WG3 elected lane: tcgen05.commit -> s_ready")
        print("consumer: wait(phase=0) would spin")

    def tensor_core_complete(self) -> None:
        if not self.committed:
            raise RuntimeError("complete before commit")
        self.completed = True
        self.phase ^= 1
        print("Tensor Core: S write complete; s_ready phase -> 1")

    def wait(self, expected_phase: int) -> None:
        if self.phase == expected_phase:
            raise RuntimeError("barrier phase has not advanced")
        print("WG0/WG1: s_ready wait returned; S may be read")


Q = [
    [1, 2, 0],
    [0, 1, 2],
]
K = [
    [1, 0, 1],
    [0, 1, 1],
    [2, 1, 0],
]

S = qk_mma(Q, K)

print("S = Q @ K^T")
for row, values in enumerate(S):
    print(f"S row {row}: {values}")

print("\nS[0, 2] calculation")
print("= Q[0,0]*K[2,0] + Q[0,1]*K[2,1] + Q[0,2]*K[2,2]")
print(f"= {Q[0][0]}*{K[2][0]} + {Q[0][1]}*{K[2][1]} + {Q[0][2]}*{K[2][2]}")
print(f"= {S[0][2]}")

print("\nstage 0 column mapping")
print(f"S0[0, 2] -> physical column {s_region_column(0, 2)}")

print("\nstage 1 column mapping")
print(f"S1[1, 2] -> physical column {s_region_column(1, 2)}")

print("\nbarrier trace")
ready = ReadyBarrierTrace()
ready.issue_and_commit()
ready.tensor_core_complete()
ready.wait(expected_phase=0)
```

运行：

```bash
python3 flash_attention_qk_mma_trace.py
```

预期输出：

```text
S = Q @ K^T
S row 0: [1, 2, 4]
S row 1: [2, 3, 1]

S[0, 2] calculation
= Q[0,0]*K[2,0] + Q[0,1]*K[2,1] + Q[0,2]*K[2,2]
= 1*2 + 2*1 + 0*0
= 4

stage 0 column mapping
S0[0, 2] -> physical column 2

stage 1 column mapping
S1[1, 2] -> physical column 130

barrier trace
WG3 elected lane: issue tcgen05.mma
WG3 elected lane: tcgen05.commit -> s_ready
consumer: wait(phase=0) would spin
Tensor Core: S write complete; s_ready phase -> 1
WG0/WG1: s_ready wait returned; S may be read
```

这个脚本验证的是矩阵公式、TMEM 列映射和完成顺序。它不执行真实
`tcgen05.mma`；真实 kernel 需要 Blackwell 硬件、TIRx 和完整的
`flash_attention4.py`。

### 六、完整数据路径

把上面的部分连起来，一次 QKᵀ MMA 的完整顺序是：

```text
TMA: Q tile -> Q_smem[q_stage]
TMA: K tile -> K_smem[kv_stage]
        |
        v
q_load.full / kv_load.full 分别确认 Q、K 已到达
        |
        v
WG3 warp 0 的 elected lane 发起 Tx.warp.gemm_async
输入: Q_smem[q_stage] + K_smem[kv_stage]
输出: S_region[q_stage] in TMEM
        |
        v
elected lane 执行 s_ready.arrive(q_stage)
底层: tcgen05.commit
        |
        v
Tensor Core 异步计算并写 S
        |
        v
硬件翻转 s_ready[q_stage] 的 phase
        |
        v
WG0 或 WG1 的 softmax 等待成功
        |
        v
softmax 可以通过 tcgen05.ld 读取 S
```

这里最重要的一点是：`gemm_async` 负责提交计算，`s_ready` 负责证明计算已经
完成并允许消费者读取结果；二者缺一不可。

### 七、常见错误与可观察症状

| 错误 | 为什么错 | 可观察症状 |
|---|---|---|
| 认为 `gemm_async` 返回后 S 已可读 | 它只表示异步 MMA 已提交 | 读到旧值、零值或部分更新结果 |
| 软件复制一份 K 转置后再传给 MMA | `K_smem` 已可直接作为 `K^T` operand | 浪费 SMEM 和带宽，甚至转置两次 |
| 所有 lanes 都执行 `s_ready.arrive` | barrier arrival count 与初始化不一致 | 提早放行或永久等待 |
| 把 `q_stage` 与 `kv_stage` 对调 | 选择错误的 operand 和 S region | operand 越界或 S 写到错误 stage |
| 在 QKᵀ MMA 内提前乘 `1/sqrt(d)`，softmax 又乘一次 | scaling 被重复应用 | score 过小、attention 分布异常 |
| Q/K TMA 尚未完成就发 MMA | SMEM descriptor 读取未完成数据 | 非确定性错误或 kernel hang |

QKᵀ MMA 产生的是原始 dot-product scores。课程把 softmax 的
`1/sqrt(d)` 缩放折入后面的 `scale_log2 = log2(e) / sqrt(d)`，本节不需要
额外缩放 S。

### 八、自测题与答案

#### 1. 为什么 `Q_block [128, HEAD_DIM]` 与 `K_block [128, HEAD_DIM]` 产生的是 `[128, 128]` S tile？

答：`K_block` 被解释成转置后的 `[HEAD_DIM, 128]`。矩阵乘法得到
`[128, HEAD_DIM] @ [HEAD_DIM, 128] = [128, 128]`，行是 query，列是当前
K block 中的 key。

#### 2. 谁执行 QKᵀ MMA，哪些 threads 不执行？

答：WG3 warp 0 中由 `elect_sync()` 选出的一个 lane 发起。其他 lanes
不各自提交一份 MMA；Tensor Core 执行矩阵级操作。

#### 3. 为什么不需要先把 `K_smem` 转置成 `[HEAD_DIM, BLK_N]`？

答：SMEM descriptor 和 `gemm_async` 的 B operand 语义已经描述如何把
`[BLK_N, HEAD_DIM]` 的行解释为 `K^T` 的列。软件复制转置矩阵不是必需步骤。

#### 4. `s_ready.arrive(q_stage)` 为什么不能理解成“这个线程写完 S”？

答：S 由 Tensor Core 异步写入，不由 issuing lane 写。`arrive` 的底层
作用是 `tcgen05.commit`，硬件只有在 MMA 完成后才真正向 barrier 报告
完成。

#### 5. QKᵀ MMA 为什么不累加到上一轮的 S，而 PV MMA 要累加到 O？

答：每个 K/V block 的 scores 是相互独立的中间结果，新的 S 可以覆盖旧
S；O 则是所有 K/V blocks 的累计输出，因此 PV MMA 要持续累加到同一块
`O_region`，期间还可能执行 correction。

## 十三、Softmax 数据路径：S TMEM -> registers -> P TMEM

### 本次讲解位置

```text
本次讲解位置
章节：chapter_flash_attention
小节：两次 MMA 之间的 Softmax
知识点：TMEM 中的 S 如何读入 registers，完成逐行 softmax，再以 fp16 P 写回 TMEM
上次：SMEM 中的 Q/K 经过 QKᵀ MMA 生成 TMEM 中的 S
下次：TMEM 中的 P 与 SMEM 中的 V 经过 PV MMA 累加到 O
PTX：tcgen05.ld、tcgen05.st、tcgen05.wait::st、mbarrier
```

### 为什么现在讲这个

QKᵀ MMA 完成后，TMEM 中只有原始 score tile `S`。`S` 不能直接交给
PV MMA，因为 PV MMA 需要的 operand 是逐行归一化过程中的指数结果
`P = exp2((S - row_max_safe) * scale_log2)`。

这一步同时引入两个容易出错的状态变化。第一，完整的一行 `S` 要先从
TMEM 读进 registers，之后才能用 fp16 `P` 覆盖同一块 TMEM；如果只读了
前半行就开始写回，后半行 scores 会被破坏。第二，`P` 是异步 TMEM store，
函数返回不代表数据已经可见；PV MMA 必须等 `tcgen05.wait.st()` 和对应
barrier 完成后才能读取。本节的软硬件边界是：Tensor Core 产出 `S`，
128 个 softmax threads 在 registers 中计算 `P`，Tensor Core 再从 TMEM
读取 `P` 做 PV MMA。

### 一、心智模型

对一个 128×128 score tile，softmax 的执行结构是：

```text
128 rows × 128 columns
        |
        | one warpgroup: 128 threads
        v
thread r owns logical row r
        |
        | 4 × tcgen05.ld
        v
thread r's registers: s_chunk[0:128]
        |
        | row max / exp2 / fp16 cast
        v
thread r's registers: fp32 P and packed fp16 P
        |
        | 4 × tcgen05.st
        v
P in TMEM, readable by PV MMA
```

`S` 到 `P` 的路径是：

```text
S TMEM fp32
    -> registers fp32
    -> registers fp32 P
    -> registers packed fp16 P
    -> TMEM fp16 P
```

它不是：

```text
S TMEM -> TMEM 原地 softmax -> PV MMA
```

TMEM 不执行 softmax。`max`、FMA、`exp2`、cast 和逐行求和都由
softmax warpgroup 的 CUDA cores 完成。

### 二、线程、layout 与四个 chunk

默认 `GQA_RATIO=1` 时，当前 score tile 有 128 行，softmax warpgroup
也有 128 个 threads，因此：

```text
tid_in_wg = r
row r -> thread r
```

每个 thread 最终要持有自己这一行的 128 个 fp32 scores。源码不是用一个
巨大的 TMEM load 一次取完整行，而是使用：

```python
SOFTMAX_LD_CHUNK = 32

for chunk_idx in T.unroll(BLK_N // SOFTMAX_LD_CHUNK):
    Tx.wg.copy_async(
        s_chunk[
            :, chunk_idx * SOFTMAX_LD_CHUNK : (chunk_idx + 1) * SOFTMAX_LD_CHUNK
        ],
        S_region[
            wg_id, :,
            chunk_idx * SOFTMAX_LD_CHUNK : (chunk_idx + 1) * SOFTMAX_LD_CHUNK,
        ],
    )
```

这里的 `BLK_N=128`，所以循环执行四次：

| chunk | TMEM columns | register fragment |
|---:|---:|---:|
| 0 | `[0, 32)` | `s_chunk[0:32]` |
| 1 | `[32, 64)` | `s_chunk[32:64]` |
| 2 | `[64, 96)` | `s_chunk[64:96]` |
| 3 | `[96, 128)` | `s_chunk[96:128]` |

`Tx.wg.copy_async` 在这条 TMEM-to-register 路径上 lower 成
`tcgen05.ld`。四个 chunks 是 load 粒度，不是四次独立的 softmax。
四次读取完成后，每个 thread 的 registers 中仍保留完整 128 个 scores，
后续 max、exp 和 sum 都针对整行执行。

当 `GQA_RATIO != 1` 时，thread 到 packed query row 的映射还要经过
`seq_pos_in_wg = tid_in_wg // GQA_RATIO`。上面的“thread `r` 处理 row
`r`”只描述默认比例。

### 三、课程代码中的完整 softmax 主路径

先等待 QKᵀ MMA 完成：

```python
s_ready.wait(wg_id, score_epoch.phase)
```

读取完整 S row，并计算本轮候选最大值：

```python
for chunk_idx in range(BLK_N // SOFTMAX_LD_CHUNK):
    tmem_load(
        s_chunk,
        chunk_idx * SOFTMAX_LD_CHUNK,
        tmem(wg_id * MMA_N + chunk_idx * SOFTMAX_LD_CHUNK),
        SOFTMAX_LD_CHUNK,
    )

if apply_mask:
    apply_causal_mask(s_chunk, m_block_idx, i_kv)

if is_first:
    reduce_max_128(tile_max, s_chunk)
else:
    row_max_old = row_max
    tile_max[0] = row_max_old
    reduce_max_128(tile_max, s_chunk, accum=True)
```

根据 FA4 的阈值 8 决定是否保留旧参考值：

```python
row_max_new = tile_max[0]
row_max_safe = T.if_then_else(
    tile_max[0] == NEG_INF,
    T.float32(0.0),
    tile_max[0],
)

if is_first:
    acc_scale = T.float32(1.0)
else:
    acc_scale_ = (row_max_old - row_max_safe) * scale_log2
    if acc_scale_ >= -rescale_threshold:
        row_max_new = row_max_old
        row_max_safe = row_max_old
        acc_scale = T.float32(1.0)
    else:
        acc_scale = T.ptx.exp2(acc_scale_)

row_max = row_max_new
```

把 base-2 exponent 的输入写成 FMA，再计算 fp32 `P`、打包成 fp16。
下面先只保留 hardware `exp2` 路径；FA4 的实际 kernel 会按 pair 选择
`exp2` 或三次多项式近似，但两者产生同一个逻辑 `P`，不改变本节的数据路径：

```python
Tx.wg.fma(
    s_chunk,
    s_chunk,
    scale_log2,
    -row_max_safe * scale_log2,
)

for fragment_idx in T.unroll(4):
    for pair_index in T.unroll(BLK_N // 4 // 2):
        index = T.meta_var(fragment_idx * BLK_N // 4 + 2 * pair_index)
        s_chunk[index] = T.ptx.exp2(s_chunk[index])
        s_chunk[index + 1] = T.ptx.exp2(s_chunk[index + 1])

    Tx.wg.cast(
        p_chunk[
            :,
            fragment_idx * BLK_N // 4 : (fragment_idx + 1) * BLK_N // 4,
        ],
        s_chunk[
            :,
            fragment_idx * BLK_N // 4 : (fragment_idx + 1) * BLK_N // 4,
        ],
    )
```

最后把 fp16 `P` 分块写回 TMEM。Non-causal 路径先写三块，再写最后一块：

```python
P_SPLIT_Q = 3

for i in T.unroll(P_SPLIT_Q):
    Tx.wg.copy_async(
        P_region[wg_id, 1, :, i * BLK_N // 4 : (i + 1) * BLK_N // 4],
        p_chunk[:, i * BLK_N // 4 : (i + 1) * BLK_N // 4],
    )

T.ptx.tcgen05.wait.st()
p_o_rescale.arrive(wg_id)

for i in T.unroll(4 - P_SPLIT_Q):
    Tx.wg.copy_async(
        P_region[
            wg_id,
            1,
            :,
            (P_SPLIT_Q + i) * BLK_N // 4 : (P_SPLIT_Q + i + 1) * BLK_N // 4,
        ],
        p_chunk[
            :,
            (P_SPLIT_Q + i) * BLK_N // 4 : (P_SPLIT_Q + i + 1) * BLK_N // 4,
        ],
    )

T.ptx.tcgen05.wait.st()
p_ready_2.arrive(wg_id)
```

这里两次 `wait.st()` 对应两个不同的交接点：

```text
前三块写完 -> p_o_rescale -> 第一段 PV MMA 可以开始
第四块写完 -> p_ready_2   -> 第二段 PV MMA 可以开始
```

Causal 路径使用 `P_SPLIT_Q=2`，按 `64 + 64` 交接；non-causal 路径按
`96 + 32` 交接。这个差异只改变第一段 PV MMA 能多早启动，不改变
softmax 每行仍处理 128 个 columns。

`row_sum` 必须在 WG2 读走当前 `acc_scale` 之后更新：

```python
softmax_corr.empty.wait(wg_id, softmax_epoch.phase)

if is_first:
    reduce_sum_128(row_sum, s_chunk)
else:
    row_sum[0] = row_sum[0] * acc_scale
    reduce_sum_128(row_sum, s_chunk, accum=True)
```

行和使用的是 registers 中的 fp32 `P`，不是已经 cast 成 fp16 的
`p_chunk`。这一点决定了 denominator 保留 fp32 精度。

### 四、完整可运行数据路径模拟

下面的脚本不依赖 CUDA。它复现一行的四个读 chunk、FA4 阈值判断、
base-2 指数、fp32 行和、非因果 `96 + 32` 写回和完成通知。文件名：
`flash_attention_softmax_trace.py`

```python
from __future__ import annotations

import struct


BLK_N = 128
SOFTMAX_LD_CHUNK = 32
SCALE_LOG2 = 1.0
RESCALE_THRESHOLD = 8.0
P_SPLIT_Q = 3


def exp2(value: float) -> float:
    return 2.0 ** value


def to_fp16(value: float) -> float:
    return struct.unpack("e", struct.pack("e", value))[0]


def softmax_step(
    scores: list[float],
    row_max_old: float,
    row_sum_old: float,
    is_first: bool,
) -> tuple[list[float], list[float], float, float, float]:
    if len(scores) != BLK_N:
        raise ValueError("expected one full 128-column score row")

    if is_first:
        candidate_max = max(scores)
        row_max_new = candidate_max
        row_max_safe = 0.0 if candidate_max == -float("inf") else candidate_max
        acc_scale = 1.0
    else:
        candidate_max = max(row_max_old, max(scores))
        row_max_safe = 0.0 if candidate_max == -float("inf") else candidate_max
        delta = (row_max_old - row_max_safe) * SCALE_LOG2
        if delta >= -RESCALE_THRESHOLD:
            row_max_new = row_max_old
            row_max_safe = row_max_old
            acc_scale = 1.0
        else:
            row_max_new = candidate_max
            acc_scale = exp2(delta)

    p_fp32 = [
        exp2((score - row_max_safe) * SCALE_LOG2)
        for score in scores
    ]
    p_fp16 = [to_fp16(value) for value in p_fp32]

    contribution = sum(p_fp32)
    if is_first:
        row_sum_new = contribution
    else:
        row_sum_new = row_sum_old * acc_scale + contribution

    return p_fp16, p_fp32, row_max_new, row_sum_new, acc_scale


scores = [float((column % 4) - 1) for column in range(BLK_N)]
row_max_old = 3.0
row_sum_old = 4.0

p_fp16, p_fp32, row_max_new, row_sum_new, acc_scale = softmax_step(
    scores,
    row_max_old,
    row_sum_old,
    is_first=False,
)

print("logical rows: 128")
print("thread mapping: thread r handles row r")
print()

for chunk_idx in range(BLK_N // SOFTMAX_LD_CHUNK):
    start = chunk_idx * SOFTMAX_LD_CHUNK
    end = start + SOFTMAX_LD_CHUNK
    print(
        f"tcgen05.ld chunk {chunk_idx}: "
        f"columns [{start}, {end}) -> registers [{start}, {end})"
    )

print()
print("candidate max =", max(scores))
print("row_max_old =", row_max_old)
print("row_max_safe =", row_max_old)
print("delta = (3.0 - 3.0) * 1.0 = 0.0")
print("delta >= -8.0: keep old reference")
print("acc_scale = 1.0")
print()

for index in (0, 1, 2, 3, 127):
    exponent_input = (scores[index] - row_max_old) * SCALE_LOG2
    print(
        f"P[{index}] = exp2(({scores[index]:g} - 3.0) * 1.0) "
        f"= exp2({exponent_input:g}) = {p_fp32[index]}"
    )

print()
print("fp32 contribution =", sum(p_fp32))
print("row_sum_new = 4.0 * 1.0 +", sum(p_fp32), "=", row_sum_new)
print()

for chunk_idx in range(P_SPLIT_Q):
    start = chunk_idx * SOFTMAX_LD_CHUNK
    end = start + SOFTMAX_LD_CHUNK
    print(
        f"tcgen05.st first group chunk {chunk_idx}: "
        f"P columns [{start}, {end}) -> TMEM"
    )

print("tcgen05.wait.st -> all first-group stores complete")
print("p_o_rescale.arrive(stage)")

chunk_idx = P_SPLIT_Q
start = chunk_idx * SOFTMAX_LD_CHUNK
end = start + SOFTMAX_LD_CHUNK
print(
    f"tcgen05.st remaining chunk {chunk_idx}: "
    f"P columns [{start}, {end}) -> TMEM"
)
print("tcgen05.wait.st -> remaining store complete")
print("p_ready_2.arrive(stage)")
print("softmax_corr.empty.wait(stage) -> fp32 row_sum may advance")
print("row_sum =", row_sum_new)
print("P[0:4] in fp16 TMEM =", p_fp16[0:4])
```

运行：

```bash
python3 flash_attention_softmax_trace.py
```

预期输出：

```text
logical rows: 128
thread mapping: thread r handles row r

tcgen05.ld chunk 0: columns [0, 32) -> registers [0, 32)
tcgen05.ld chunk 1: columns [32, 64) -> registers [32, 64)
tcgen05.ld chunk 2: columns [64, 96) -> registers [64, 96)
tcgen05.ld chunk 3: columns [96, 128) -> registers [96, 128)

candidate max = 2.0
row_max_old = 3.0
row_max_safe = 3.0
delta = (3.0 - 3.0) * 1.0 = 0.0
delta >= -8.0: keep old reference
acc_scale = 1.0

P[0] = exp2((-1 - 3.0) * 1.0) = exp2(-4) = 0.0625
P[1] = exp2((0 - 3.0) * 1.0) = exp2(-3) = 0.125
P[2] = exp2((1 - 3.0) * 1.0) = exp2(-2) = 0.25
P[3] = exp2((2 - 3.0) * 1.0) = exp2(-1) = 0.5
P[127] = exp2((2 - 3.0) * 1.0) = exp2(-1) = 0.5

fp32 contribution = 30.0
row_sum_new = 4.0 * 1.0 + 30.0 = 34.0

tcgen05.st first group chunk 0: P columns [0, 32) -> TMEM
tcgen05.st first group chunk 1: P columns [32, 64) -> TMEM
tcgen05.st first group chunk 2: P columns [64, 96) -> TMEM
tcgen05.wait.st -> all first-group stores complete
p_o_rescale.arrive(stage)
tcgen05.st remaining chunk 3: P columns [96, 128) -> TMEM
tcgen05.wait.st -> remaining store complete
p_ready_2.arrive(stage)
softmax_corr.empty.wait(stage) -> fp32 row_sum may advance
row_sum = 34.0
P[0:4] in fp16 TMEM = [0.0625, 0.125, 0.25, 0.5]
```

这个脚本验证的是行映射、FA4 阈值路径、fp32 行和和分段交接。它不执行
真实 `tcgen05.ld` 或 `tcgen05.st`；真实 kernel 需要 Blackwell 硬件、
TIRx 和完整的 `flash_attention4.py`。

### 五、为什么 P 要覆盖回 S 的 TMEM

Stage 0 的 `S0` 占用 fp32 physical columns `[0, 128)`。softmax 把整行
读进 registers 后，`P0` 的 128 个 fp16 values 会写到 physical columns
`[64, 128)`：

```text
P_region[0, 1, :, n]
    -> physical column 64 + n // 2
```

因此 `P0[:, 0]` 和 `P0[:, 1]` 共用 physical column 64 的两个 fp16
半格，`P0[:, 2]` 和 `P0[:, 3]` 共用 column 65。最终 `P0` 只占 64 个
物理 columns。

这份空间复用有严格顺序：

```text
1. QKᵀ MMA 写完整 S0
2. softmax 等待 s_ready
3. 四次 tcgen05.ld 读完整 S0 到 registers
4. 计算 P
5. tcgen05.st 用 P0 覆盖 S0 的物理 columns [64, 128)
6. tcgen05.wait.st 证明 store 完成
7. barrier 放行 PV MMA
```

如果第 3 步未完成就开始第 5 步，尚未读出的 `S0[:, 64:128]` 会被 `P0`
覆盖。症状通常不是统一错误，而是部分 score columns 计算正确、另一部分
出现随机错误，最终 attention 输出也只在这些 columns 对应的 V rows 上
异常。

### 六、常见错误与可观察症状

| 错误 | 为什么错 | 可观察症状 |
|---|---|---|
| 一次 `tcgen05.ld` 后就把 32 个 values 当成整行 softmax | 每行仍有另外 96 个 columns 未读入 | 行最大值和 denominator 都偏小，P 错误 |
| 未读完 S 就执行 `tcgen05.st` | P 会覆盖尚未读取的 S columns | 部分 scores 随机损坏，结果不可复现 |
| 把四个 load chunks 当成四段独立 softmax | softmax max 和 sum 必须覆盖整行 | 得到四个局部归一化片段 |
| `tcgen05.st` 后不执行 `tcgen05.wait.st()` | store 仍异步进行 | PV MMA 读到旧值或部分新值 |
| 用 fp16 `p_chunk` 累计 `row_sum` | denominator 应保留 fp32 精度 | 长 K/V loop 中归一化误差逐渐增大 |
| `p_o_rescale` 对 `GQA_RATIO != 1` 却按 256 arrivals 理解 | GQA 路径使用 pairwise named barriers | 同步比例解释错误，进而误判死锁原因 |
| `softmax_corr.empty` 后仍重写上一轮 `acc_scale` | WG2 可能尚未消费该行 statistics | statistics 丢失或 O correction 使用错误 scale |

### 七、自测题与答案

#### 1. 一个 softmax warpgroup 有 128 个 threads，为什么能让 thread `r` 负责 row `r`？

答：默认 `GQA_RATIO=1` 时，一个 S tile 正好有 128 行，warpgroup 也正好
有 128 个 threads。每个 thread 在自己的 registers 中保留一整行 128 个
fp32 scores，因此无需跨 thread 交换 row max 和 row sum。

#### 2. 为什么`tcgen05.ld`分成四个 32-column chunks，但 softmax 不是四次局部计算？

答：四个 chunks 只控制单次 TMEM load 的 register tuple 大小。四次读取
完成后，每个 thread 持有完整 128 个 scores，随后对整行做一次 max、
exp2 和 sum。

#### 3. 为什么 `P0` 能覆盖 `S0`，而不会丢失数据？

答：`P0` 写入之前，softmax 已把完整 `S0` 读入 registers。`P0` 只需要
128 个 fp16 values，两两打包后占 64 个物理 columns，因此可以覆盖
`S0` 的后 64 个 fp32 columns。被覆盖的数据已经不再需要。

#### 4. 为什么前三块写完后要 `tcgen05.wait.st()` 再 `p_o_rescale.arrive`？

答：`p_o_rescale` 只证明前三块 `P` 可被第一段 PV MMA 读取。若不先等
TMEM stores 完成就发 arrival，PV MMA 可能读到未写完的 `P`。

#### 5. `row_sum` 更新为什么要等待 `softmax_corr.empty`？

答：当前 `acc_scale` 还要由 WG2 读取，用于决定是否重缩放 TMEM 中的旧
`O`。Softmax 在 WG2 确认对应 `sScale` slot 已被消费之前，不应推进到下一
个 phase 并重写该状态。等待完成后，softmax 才用 registers 中仍保留的
fp32 `P` 更新 `row_sum`。

## 十四、PV MMA 数据路径：P TMEM + V SMEM -> O TMEM

### 本次讲解位置

```text
本次讲解位置
章节：chapter_flash_attention
小节：PV MMA
知识点：TMEM 中的 P 与 SMEM 中的 V 分两段累加到 TMEM 中的 O
上次：softmax 把 TMEM 中的 S 转成 TMEM 中的 fp16 P
下次：Warp 角色、register 分配与 barrier 分工
PTX：tcgen05.mma、tcgen05.commit、mbarrier
```

### 为什么现在讲这个

Softmax 只产生了当前 K/V block 的权重 `P`，还没有把它作用到 value
`V`。PV MMA 必须计算：

```text
O_block = P_block @ V_block
```

第一个 K/V block 的乘积用于初始化长期累加器 `O`；后续 block 的乘积必须
继续累加到同一块 `O`。如果所有 N 个 KV blocks 都使用 `accum=false`，
最后只会保留最后一个 block；如果第一轮第二个 MMA segment 也使用
`accum=false`，它会把第一段刚算出的 partial sum 覆盖掉。

本节还要说明 `P` 和 `V` 为什么能从不同 memory spaces 直接进入同一条 MMA，
以及 `p_o_rescale`、`p_ready_2`、`o_ready` 分别证明哪一部分数据或
accumulator 已经可用。

### 一、心智模型

设 `HEAD_DIM=d`，当前 K/V block 的长度是 `BLK_N=128`：

```text
P_block: [128, 128]         tensor cores 的 A operand
V_block: [128, d]           tensor cores 的 B operand
O_tile:  [128, d]           accumulator，位于 TMEM

P_block @ V_block -> O_tile
```

一个输出元素是：

```text
O[row, col] = sum_{k=0}^{127} P[row, k] * V[k, col]
```

其中 `k` 是当前 K/V block 内的归约位置。PV MMA 执行时：

```text
读取 P：TMEM
读取 V：SMEM
累加 O：TMEM
```

`P` 不需要重新搬回 SMEM，`V` 也不需要先搬进 TMEM。Blackwell 的
`tcgen05.mma` 支持 TMEM operand，因此可以形成：

```text
QKᵀ MMA: SMEM + SMEM -> TMEM
Softmax:  TMEM -> registers -> TMEM
PV MMA:  TMEM + SMEM -> TMEM
```

这就是 FA4 能把两次 MMA 和 register-softmax 串在片上完成的关键。

### 二、Operand、scope 与物理位置

课程中的调用是：

```python
K_SPLIT = T.meta_var((4 if is_causal else 6) * MMA_K)

Tx.warp.gemm_async(
    O_region[SMEM_PIPE_DEPTH_Q + i_q, :, :],
    P_region[i_q, 1, :, 0:K_SPLIT],
    V_smem[kv_stage, 0:K_SPLIT, 0:HEAD_DIM],
    transB=True,
    accum=should_accumulate,
    dispatch="tcgen05",
    cta_group=CTA_GROUP,
)

p_ready_2.wait(i_q, phase_tmem)

Tx.warp.gemm_async(
    O_region[SMEM_PIPE_DEPTH_Q + i_q, :, :],
    P_region[i_q, 1, :, K_SPLIT:BLK_N],
    V_smem[kv_stage, K_SPLIT:BLK_N, 0:HEAD_DIM],
    transB=True,
    accum=True,
    dispatch="tcgen05",
    cta_group=CTA_GROUP,
)
```

每个参数的角色是：

| 参数 | 数据位置 | 含义 |
|---|---|---|
| `O_region[...]` | TMEM | fp32 accumulator |
| `P_region[i_q, 1, ...]` | TMEM | fp16 attention weights |
| `V_smem[...]` | SMEM | fp16 value tile |
| `transB=True` | descriptor 语义 | 让 SMEM operand 按 PV MMA 需要的遍历方式解释 |
| `accum` | MMA 状态 | 覆盖初始化或继续累加到 `O` |
| `dispatch="tcgen05"` | 指令路径 | Blackwell Tensor Core MMA |
| `cta_group` | scope | 单 CTA 或两 CTA cooperative MMA |

`transB=True` 不是因为目标数学公式变成了 `P @ V^T`。逻辑计算仍然是
`P @ V`；该标志配合 descriptor，告诉 lowering 如何从 V 的 SMEM layout
读取 B operand。

默认 `CTA_GROUP=1` 时，WG3 warp 0 中由一个 elected lane 发起该
warp-scoped tile operation；Tensor Core 执行实际的矩阵乘加。

### 三、K_SPLIT 为什么不是固定的 128

`P` 的四个 32-column chunks 是分批写回 TMEM 的。若 PV MMA 必须等待全部
128 列写完，Tensor Core 会长时间空转。FA4 因此把 inner-K 分成两段：

| 路径 | `K_SPLIT` | 分段 | 第一段含义 |
|---|---:|---|---|
| Causal | 64 | 64 + 64 | 与 causal mask 的处理粒度对齐 |
| Non-causal | 96 | 96 + 32 | 前三块 P 写完就提前启动 |

`MMA_K=16`。Non-causal 的第一段是：

```text
96 / 16 = 6 个 MMA K-steps
```

第二段是：

```text
(128 - 96) / 16 = 2 个 MMA K-steps
```

Causal 的两段各为：

```text
64 / 16 = 4 个 MMA K-steps
```

第一段使用：

```python
P[:, 0:K_SPLIT] @ V[0:K_SPLIT, :]
```

第二段使用：

```python
P[:, K_SPLIT:128] @ V[K_SPLIT:128, :]
```

两段相加仍等于完整的：

```python
P[:, 0:128] @ V[0:128, :]
```

拆分的意义只是让 Tensor Core 提前处理已经准备好的一部分 P，不改变
attention 数学。

### 四、accum 的三段状态

`O` 的生命周期比单个 KV block 长。设当前 CTA 正在处理 KV block `t`：

```text
t = 0: 旧 O 不存在，第一段必须初始化 O
t > 0: 旧 O 存在，第一段必须继续累加
```

实际标志如下：

| 场景 | 第一段 `accum` | 第二段 `accum` | 原因 |
|---|---:|---:|---|
| 第一个 KV block | `False` | `True` | 第一段初始化；第二段累加到第一段结果 |
| 后续 KV block | `True` | `True` | 两段都累加到旧 O |

这里最容易误解的一点是：第一个 K/V block 的第二段也不能使用
`accum=False`。第一段已经产生了 `O_partial`，若第二段覆盖它，就只剩
`P[:, 96:128] @ V[96:128, :]`，前面的 96 个归约位置全部丢失。

在底层，一个 segment 又由多个 `MMA_K=16` steps 组成。以第一个 KV
block 的第一段为例：

```text
ki = 0: accum = False，初始化 O
ki = 1: accum = True，累加
ki = 2: accum = True，累加
...
```

所以 `accum=False` 只适用于整块 `O` 的第一次 MMA step，不是该 segment
内每个 step 都覆盖。

### 五、P 的 fp16 地址怎样映射到 TMEM columns

`P` 的逻辑 shape 是 `[128, 128]`，但 fp16 两个 values 打包在一个
32-bit TMEM cell 中。源码中 A operand 的列地址为：

```text
i_q * MMA_N + MMA_N // 2 + ki * (MMA_K // 2)
```

代入 `MMA_N=128`、`MMA_K=16`：

```text
physical column = i_q * 128 + 64 + ki * 8
```

Stage 0 的 P 地址是：

| `ki` | 覆盖的逻辑 P columns | 起始 physical column |
|---:|---:|---:|
| 0 | `[0, 16)` | 64 |
| 1 | `[16, 32)` | 72 |
| 2 | `[32, 48)` | 80 |
| 3 | `[48, 64)` | 88 |
| 4 | `[64, 80)` | 96 |
| 5 | `[80, 96)` | 104 |
| 6 | `[96, 112)` | 112 |
| 7 | `[112, 128)` | 120 |

Stage 0 的完整 `P0` 占用 physical columns `[64, 128)`。Stage 1 加上
基础偏移 128，占用 `[192, 256)`。因此：

```text
non-causal 第一段: ki = 0..5 -> physical columns [64, 112)
non-causal 第二段: ki = 6..7 -> physical columns [112, 128)
```

### 六、PV MMA 的完整 barrier 协议

第一段开始前需要两组条件：

```text
kv_load.full
    -> 完整 V block 已进入 SMEM

p_o_rescale
    -> P[:, 0:K_SPLIT] 已在 TMEM
    -> 旧 O 已完成必要的 rescale，或首轮可以直接初始化
```

默认单 CTA 路径中，`p_o_rescale` 的 expected arrival count 是 256：

| Producer | arrivals | 证明 |
|---|---:|---|
| Softmax warpgroup | 128 | 前 `K_SPLIT` 个 P columns 已写入 TMEM |
| WG2 | 128 | O slot 已初始化、已 rescale，或确认无需 rescale |

第一段发出后，MMA warp 等待：

```text
p_ready_2
    -> P[:, K_SPLIT:128] 已写入 TMEM
```

`p_ready_2` 的 expected arrival count 是 128，由 softmax warpgroup
提供。第二段不需要再次等待 `kv_load.full`，因为完整 V 在第一段开始前
已经确认到达 SMEM。

完整的稳态顺序是：

```text
softmax 写 P[:, 0:K_SPLIT]
WG2 准备旧 O
        |
        v
p_o_rescale 放行
        |
        v
第一段 PV MMA: accum=should_accumulate
        |
        v
p_ready_2 放行
        |
        v
第二段 PV MMA: accum=True
```

最后一个 K/V block 的尾段 PV MMA 完成后，elected lane 执行：

```python
commit(o_ready, i_q)
```

`o_ready` 表示对应 Q stage 的 `O` 已经完成所有累积，可以由 WG2 做最终
normalization 或 epilogue。它不是“O 的第一个部分写完了”，而是“这个
output tile 不再接受新的 PV MMA”。

### 七、完整可运行的分段累加模拟

下面的脚本用 `2×4` 的 `P`、`4×2` 的 `V` 和 `K_SPLIT=2` 模拟两个
KV blocks。它完整展示第一轮初始化、第二轮 rescale、两段累加和
`o_ready`。文件名：`flash_attention_pv_mma_trace.py`

```python
from __future__ import annotations


def matmul(a: list[list[int]], b: list[list[int]]) -> list[list[int]]:
    rows_a = len(a)
    inner = len(a[0])
    cols_b = len(b[0])

    if len(b) != inner:
        raise ValueError("shape mismatch")

    result = [[0] * cols_b for _ in range(rows_a)]
    for row in range(rows_a):
        for col in range(cols_b):
            result[row][col] = sum(
                a[row][k] * b[k][col]
                for k in range(inner)
            )
    return result


def add_into(dst: list[list[int]], src: list[list[int]]) -> None:
    for row in range(len(dst)):
        for col in range(len(dst[0])):
            dst[row][col] += src[row][col]


def scale(values: list[list[int]], factor: float) -> list[list[int]]:
    return [
        [int(value * factor) for value in row]
        for row in values
    ]


P0 = [
    [1, 2, 3, 4],
    [0, 1, 1, 0],
]
V0 = [
    [1, 0],
    [0, 1],
    [1, 1],
    [2, 0],
]
K_SPLIT = 2

print("iteration 0")
print("kv_load.full -> V0 is in SMEM")
print("p_o_rescale -> P0[:, :2] is in TMEM; O0 may initialize")

part1 = matmul(
    [row[:K_SPLIT] for row in P0],
    V0[:K_SPLIT],
)
print("accum=False")
print("part1 =", part1)

O = [row[:] for row in part1]
print("O after part1 =", O)
print("p_ready_2 -> P0[:, 2:] is in TMEM")

part2 = matmul(
    [row[K_SPLIT:] for row in P0],
    V0[K_SPLIT:],
)
print("accum=True")
print("part2 =", part2)
add_into(O, part2)
print("O after part2 =", O)
print("full product =", matmul(P0, V0))
print()

P1 = [
    [1, 0, 0, 1],
    [0, 1, 1, 0],
]
V1 = [
    [1, 0],
    [0, 1],
    [1, 0],
    [0, 1],
]
acc_scale = 2.0

print("iteration 1")
print("softmax sends acc_scale =", acc_scale)
O = scale(O, acc_scale)
print("WG2 rescales old O =", O)
print("p_o_rescale -> part1 may accumulate")

part1 = matmul(
    [row[:K_SPLIT] for row in P1],
    V1[:K_SPLIT],
)
print("accum=True")
print("part1 =", part1)
add_into(O, part1)
print("O after part1 =", O)

part2 = matmul(
    [row[K_SPLIT:] for row in P1],
    V1[K_SPLIT:],
)
print("p_ready_2 -> part2 may accumulate")
print("part2 =", part2)
add_into(O, part2)
print("O after part2 =", O)
print()
print("after final KV block: tcgen05.commit -> o_ready")
print("epilogue waits o_ready and reads O")
```

运行：

```bash
python3 flash_attention_pv_mma_trace.py
```

预期输出：

```text
iteration 0
kv_load.full -> V0 is in SMEM
p_o_rescale -> P0[:, :2] is in TMEM; O0 may initialize
accum=False
part1 = [[1, 2], [0, 1]]
O after part1 = [[1, 2], [0, 1]]
p_ready_2 -> P0[:, 2:] is in TMEM
accum=True
part2 = [[11, 3], [1, 1]]
O after part2 = [[12, 5], [1, 2]]
full product = [[12, 5], [1, 2]]

iteration 1
softmax sends acc_scale = 2.0
WG2 rescales old O = [[24, 10], [2, 4]]
p_o_rescale -> part1 may accumulate
accum=True
part1 = [[1, 0], [0, 1]]
O after part1 = [[25, 10], [2, 5]]
p_ready_2 -> part2 may accumulate
part2 = [[0, 1], [1, 0]]
O after part2 = [[25, 11], [3, 5]]

after final KV block: tcgen05.commit -> o_ready
epilogue waits o_ready and reads O
```

脚本验证了分段乘加、初始化与累加规则，以及 rescale 必须发生在第一段
PV MMA 之前。它不执行真实 `tcgen05.mma`；真实 kernel 需要 Blackwell
硬件、TIRx 和完整的 `flash_attention4.py`。

### 八、常见错误与可观察症状

| 错误 | 为什么错 | 可观察症状 |
|---|---|---|
| 第一个 KV block 的总结果只用第一段计算 | 丢了 `P[:, K_SPLIT:] @ V[K_SPLIT:]` | O 的所有列都偏小，且误差随 block 内容变化 |
| 第一轮第二段使用 `accum=False` | 覆盖第一段产生的 partial O | O 只反映后 32 或 64 个 K positions |
| 后续轮第一段使用 `accum=False` | 丢掉所有之前 KV blocks 的贡献 | 最终只保留最后一个 KV block 的 attention |
| 未等 `p_ready_2` 就发第二段 | 后 32 个 P columns 可能尚未写完 | 非确定性错误集中在最后 32 个 K positions |
| 未等 `p_o_rescale` 就发第一段 | P 前半或重缩放后的 O 尚未准备好 | 首段读到旧 P 或未 rescale 的 O |
| 认为 `transB=True` 表示要计算 `P @ V^T` | 它描述 SMEM operand 遍历方式 | 数学 shape 理解错误，进而误判切片方向 |
| 最后一个 KV block 后没有 `commit(o_ready)` | epilogue 不知道 O 已完成 | WG2/epilogue 永久等待或读到未完成 O |

### 九、自测题与答案

#### 1. 为什么 `P` 在 TMEM，而 `V` 仍在 SMEM，两者可以直接做 MMA？

答：Blackwell `tcgen05.mma` 支持 TMEM operand。当前调用把 TMEM 中的
`P` 作为 A operand，把 SMEM 中的 `V` 作为 B operand，输出和 accumulator
仍放在 TMEM 的 `O_region`。

#### 2. Non-causal 为什么选择 `96 + 32`，而不是始终 `128 + 0`？

答：Softmax 已按四个 32-column chunks 写 P。前三块写完时，PV MMA 就能
处理前 96 个 K positions。其余 32 positions 由 `p_ready_2` 单独放行，
这样 Tensor Core 不必等待全部 P writeback。

#### 3. 为什么第一个 K/V block 的第二段必须使用 `accum=True`？

答：第一段已经以 `accum=False` 初始化了 `O_partial`。第二段需要使用
`O_partial + P[:, K_SPLIT:] @ V[K_SPLIT:]`，所以必须累加，不能覆盖。

#### 4. `p_o_rescale` 与 `p_ready_2` 分别证明什么？

答：`p_o_rescale` 证明 P 的前 `K_SPLIT` columns 已就绪，并且旧 O 已
初始化、已完成 rescale 或确认无需 rescale。`p_ready_2` 只证明 P 的
剩余 columns 已经写入 TMEM，不再等待 O 状态。

#### 5. `o_ready` 在什么时候报告，为什么它和 `p_ready_2` 不是一回事？

答：`o_ready` 在某个 Q stage 的最后一个 K/V block 的尾段 PV MMA
完成后，通过 `tcgen05.commit` 报告，表示 O 已完成全部累积。
`p_ready_2` 只表示当前 block 的剩余 P columns 已可用于第二段 MMA。

## 十五、Warp 角色地图

### 本次讲解位置

```text
本次讲解位置
章节：chapter_flash_attention
小节：Warp 角色与 Scope
知识点：5.1 CTA 内的 warpgroup/warp/thread 角色地图
上次：PV MMA 数据路径
下次：Register 分配与 setmaxnreg
PTX：角色本身不是一条 PTX 指令；它决定 TMA、tcgen05.mma 与
     mbarrier 操作分别由哪些 warp 发出
```

### 为什么现在讲这个

前面的 QKᵀ MMA、softmax 和 PV MMA 已经回答了“数据从哪来、经过哪条
路径、写回哪里”，但还没有回答“谁发出指令、谁执行计算、谁负责搬运”。
如果只看到 `wg_id == 0`，很容易把 warpgroup 0 误认成 warp 0；如果
不知道同一条 softmax 分支覆盖 128 个线程，就会给 barrier 设置错误的
arrival count，或者把一个必须由整个 CTA 到达的 `cta_sync()` 放进单个
warpgroup 分支，导致其他线程永远到不了同步点。

本节先建立稳定的 thread 归属模型。它不深入 register 数量，也不展开每个
barrier 的 wait/arrive 协议，只回答一个具体问题：

```text
对于 CTA 中任意一个 thread，
怎样从它的 (wg_id, warp_id, lane_id)
判断它属于哪个角色？
```

### 一、先建立层级：CTA -> warpgroup -> warp -> lane

当前 non-causal FlashAttention kernel 的一个 CTA 有 512 个 threads：

```text
1 CTA
  = 4 warpgroups
  = 16 warps
  = 512 threads
```

一个 warpgroup 包含 4 个 warps，每个 warp 有 32 个 lanes，所以：

```text
128 threads / warpgroup
= 4 warps / warpgroup
= 32 lanes / warp
```

`tid_in_wg` 是线程在所属 warpgroup 内的编号，范围为 `0..127`：

```text
tid_in_wg = warp_id * 32 + lane_id
```

CTA 内的全局 thread id 则是：

```text
tid = wg_id * 128 + warp_id * 32 + lane_id
```

反解公式为：

```text
wg_id   = tid // 128
warp_id = (tid % 128) // 32
lane_id = tid % 32
```

这里必须区分三个变量：

| 名称 | 范围 | 含义 |
|---|---:|---|
| `wg_id` | `0..3` | 当前 thread 属于哪个 warpgroup |
| `warp_id` | `0..3` | 当前 warp 在该 warpgroup 内的序号 |
| `lane_id` | `0..31` | 当前 thread 在该 warp 内的 lane 序号 |

因此“warp 0”并不唯一，必须同时说明 warpgroup。例如：

```text
(wg_id=0, warp_id=0) -> CTA 内第 0 个 warp
(wg_id=3, warp_id=0) -> CTA 内第 12 个 warp
```

### 二、逻辑角色地图

标准 non-causal 路径的角色分配如下：

| 执行者 | 角色 | 负责的数据路径 |
|---|---|---|
| WG0 | softmax stage 0 | 读取 TMEM `S0`，写回 TMEM `P0` |
| WG1 | softmax stage 1 | 读取 TMEM `S1`，写回 TMEM `P1` |
| WG2 | correction / non-causal epilogue | 按需 rescale TMEM `O`，最终归一化并写 SMEM |
| WG3, warp 0 | MMA issue | 发起 QKᵀ MMA 和 PV MMA |
| WG3, warp 1 | TMA load | 将 Q、K、V 从 GMEM 搬到 SMEM |
| WG3, warp 2 | TMA store | 将最终 O 从 SMEM 搬回 GMEM |
| WG3, warp 3 | idle / 辅助 | 不承担当前路径的主要 issue 工作 |

映射到 thread 范围：

```text
WG0: threads [0, 128)       -> softmax stage 0
WG1: threads [128, 256)     -> softmax stage 1
WG2: threads [256, 384)     -> correction / non-causal epilogue
WG3: threads [384, 512)
  warp 0: threads [384, 416) -> MMA issue
  warp 1: threads [416, 448) -> TMA load
  warp 2: threads [448, 480) -> TMA store
  warp 3: threads [480, 512) -> idle / 辅助
```

这张图描述的是当前 non-causal 路径的执行分工，不等价于“这些线程亲自
完成所有算术”。例如，WG3 warp 0 只负责提交 MMA；矩阵乘加由 Tensor
Core 执行。TMA load/store warp 也只负责提交描述符和操作，搬运由 TMA
engine 执行。

### 三、TIRx 源码中的角色声明

TIRx 用 `specialize(...).role(...)` 描述角色与 warp 的绑定：

```python
sp = txl.specialize(chain_dispatch=True)

r_softmax = sp.role(
    "softmax",
    warps=[0, 1, 2, 3, 4, 5, 6, 7],
    regs=softmax_regs,
)
r_correction = sp.role(
    "correction",
    warps=[8, 9, 10, 11],
    regs=correction_regs,
)
wg3 = sp.warpgroup(
    "wg3",
    warps=range(12, 16),
    regs=other_regs,
)
r_mma = sp.role(
    "mma",
    warps=[12],
    group=wg3,
)
r_load = sp.role(
    "load",
    warps=[13],
    group=wg3,
)
r_store = sp.role(
    "store",
    warps=[14],
    group=wg3,
)
r_idle = sp.role(
    "idle",
    warps=[15],
    group=wg3,
)
```

源码里的 `softmax` role 覆盖 warps `0..7`，也就是 WG0 和 WG1 两个
warpgroups。kernel 再用 `wg_id` 选择 Q stage：

```python
wg_id = T.warpgroup_id([4])
warp_id = T.warp_id_in_wg([4])
```

因此：

```text
wg_id == 0, warp_id == 0..3 -> softmax stage 0
wg_id == 1, warp_id == 0..3 -> softmax stage 1
wg_id == 2, warp_id == 0..3 -> correction / epilogue
wg_id == 3, warp_id == 0    -> MMA
wg_id == 3, warp_id == 1    -> TMA load
wg_id == 3, warp_id == 2    -> TMA store
wg_id == 3, warp_id == 3    -> idle / 辅助
```

`warp_id` 是 warpgroup 内的 warp 序号，不是全局 warp 序号。源码写
`warps=[12]`，对应的是全局第 12 个 warp，也就是 WG3 内
`warp_id == 0`。

### 四、完整可运行的角色映射脚本

下面的脚本完整枚举 512 个 threads，并验证上面的 thread 范围。文件名：
`flash_attention_warp_role_map.py`

```python
from __future__ import annotations

from collections import defaultdict


WG_THREADS = 128
WARP_THREADS = 32
WG_COUNT = 4


def classify(wg_id: int, warp_id: int) -> str:
    if wg_id == 0:
        return "softmax stage 0 (WG0)"
    if wg_id == 1:
        return "softmax stage 1 (WG1)"
    if wg_id == 2:
        return "correction / non-causal epilogue (WG2)"

    if wg_id == 3 and warp_id == 0:
        return "MMA issue (WG3 warp 0)"
    if wg_id == 3 and warp_id == 1:
        return "TMA load (WG3 warp 1)"
    if wg_id == 3 and warp_id == 2:
        return "TMA store (WG3 warp 2)"
    if wg_id == 3 and warp_id == 3:
        return "idle / helper (WG3 warp 3)"

    raise AssertionError((wg_id, warp_id))


def main() -> None:
    ranges: dict[str, list[int]] = defaultdict(list)

    for tid in range(WG_COUNT * WG_THREADS):
        wg_id = tid // WG_THREADS
        tid_in_wg = tid % WG_THREADS
        warp_id = tid_in_wg // WARP_THREADS
        lane_id = tid_in_wg % WARP_THREADS

        assert tid == wg_id * WG_THREADS + warp_id * WARP_THREADS + lane_id
        ranges[classify(wg_id, warp_id)].append(tid)

    expected = {
        "softmax stage 0 (WG0)": (0, 128),
        "softmax stage 1 (WG1)": (128, 256),
        "correction / non-causal epilogue (WG2)": (256, 384),
        "MMA issue (WG3 warp 0)": (384, 416),
        "TMA load (WG3 warp 1)": (416, 448),
        "TMA store (WG3 warp 2)": (448, 480),
        "idle / helper (WG3 warp 3)": (480, 512),
    }

    print("CTA role map")
    for role, threads in ranges.items():
        actual = (threads[0], threads[-1] + 1)
        assert actual == expected[role], (role, actual)
        print(f"{role:42s}: threads [{actual[0]:3d}, {actual[1]:3d})")

    examples = [129, 300, 430]
    print()
    print("example decoding")
    for tid in examples:
        wg_id = tid // WG_THREADS
        tid_in_wg = tid % WG_THREADS
        warp_id = tid_in_wg // WARP_THREADS
        lane_id = tid_in_wg % WARP_THREADS
        print(
            f"tid={tid}: wg_id={wg_id}, warp_id={warp_id}, "
            f"lane_id={lane_id}, role={classify(wg_id, warp_id)}"
        )


if __name__ == "__main__":
    main()
```

运行：

```bash
python3 flash_attention_warp_role_map.py
```

预期输出：

```text
CTA role map
softmax stage 0 (WG0)                     : threads [  0, 128)
softmax stage 1 (WG1)                     : threads [128, 256)
correction / non-causal epilogue (WG2)    : threads [256, 384)
MMA issue (WG3 warp 0)                    : threads [384, 416)
TMA load (WG3 warp 1)                     : threads [416, 448)
TMA store (WG3 warp 2)                    : threads [448, 480)
idle / helper (WG3 warp 3)                : threads [480, 512)

example decoding
tid=129: wg_id=1, warp_id=0, lane_id=1, role=softmax stage 1 (WG1)
tid=300: wg_id=2, warp_id=1, lane_id=12, role=correction / non-causal epilogue (WG2)
tid=430: wg_id=3, warp_id=1, lane_id=14, role=TMA load (WG3 warp 1)
```

### 五、具体解码：`tid=430` 为什么属于 TMA load

逐步计算：

```text
tid = 430

wg_id
  = 430 // 128
  = 3

tid_in_wg
  = 430 % 128
  = 46

warp_id
  = 46 // 32
  = 1

lane_id
  = 46 % 32
  = 14
```

所以：

```text
(wg_id, warp_id, lane_id) = (3, 1, 14)
```

该线程位于 WG3 的 warp 1，因此属于 TMA load warp。它不会执行 softmax
的 128-thread collective；它所在的 warp 在满足 elected-lane 条件后提交
TMA load。

再比较 `tid=129`：

```text
wg_id   = 129 // 128 = 1
tid_in_wg = 129 % 128 = 1
warp_id = 1 // 32 = 0
lane_id = 1 % 32 = 1
```

所以它属于：

```text
WG1 -> softmax stage 1
```

这两个例子说明：`wg_id` 决定大角色块，`warp_id` 在 WG3 内继续细分
MMA、TMA load、TMA store 和 idle。

### 六、角色范围与 per-thread 数据视图

角色不仅决定控制流，也决定该线程看到的索引语义：

| 角色 | 常用的本地索引 | 原因 |
|---|---|---|
| WG0/WG1 softmax | `tid_in_wg` | softmax 按 128-thread warpgroup 分行 |
| WG2 correction | `tid_in_wg` 或行块索引 | correction 以 warpgroup 范围处理 O |
| WG3 load | `elect_sync()` | 一个 warp 只需要一个 lane 提交 TMA |
| WG3 MMA | `elect_sync()` | `tcgen05.mma` 是 single-thread issue |
| WG3 store | `elect_sync()` | TMA store 也只由一个 elected lane 提交 |

这一点会直接影响 buffer indexing。例如 softmax 中按行定位时，应该使用
warpgroup 内的 `tid_in_wg`，而不是 CTA 全局 `tid`，否则 WG1 会把
线程映射到错误的 128 行范围内。

### 七、常见错误与可观察症状

| 错误 | 为什么错 | 可观察症状 |
|---|---|---|
| 把 `wg_id == 0` 当成“warp 0” | warp 0 在每个 warpgroup 中都存在 | 角色分支覆盖 128 个线程，功能范围错误 |
| 用全局 `tid` 直接代替 `tid_in_wg` | softmax 的本地行范围应为 `0..127` | WG1 访问越界或读写错误 row |
| 认为 512 个线程都执行 softmax | 只有 WG0/WG1 负责 softmax | register 预算和 barrier arrival 全部估算错误 |
| 把 WG0 和 WG1 当成同一个 stage | 它们分别服务 Q stage 0 和 stage 1 | S/P/O slot 混用，出现跨 stage 数据污染 |
| 只用 `warp_id` 区分 WG3 角色 | WG0 的 warp 0 与 WG3 的 warp 0 不同 | MMA、TMA、softmax 分支互相串线 |
| 把 TMA/MMA issue 写成整个 warp 都提交 | 多数硬件指令由 elected lane 提交 | 重复提交、barrier 多到达或结果不确定 |
| 在单个 warpgroup 分支里调用 CTA-wide `cta_sync()` | 其他 warpgroup 到不了同一个同步点 | kernel 挂死，等待永远不结束 |

### 八、自测题与答案

#### 1. CTA 内有 512 个 threads，如何快速算出 `wg_id`、`warp_id`、`lane_id`？

答：

```text
wg_id   = tid // 128
warp_id = (tid % 128) // 32
lane_id = tid % 32
```

例如 `tid=430` 得到 `(wg_id, warp_id, lane_id) = (3, 1, 14)`。

#### 2. WG1 内的 `warp_id=2, lane_id=7`，它的 CTA 全局 `tid` 是多少？

答：

```text
tid = 1 * 128 + 2 * 32 + 7
    = 128 + 64 + 7
    = 199
```

它属于 WG1 的 softmax stage 1。

#### 3. 为什么 TIRx 角色声明中的 `warps=[12]` 对应 WG3 的 `warp_id=0`？

答：全局 warp 12 是第 `12 // 4 = 3` 个 warpgroup，即 WG3；其
warpgroup 内序号为 `12 % 4 = 0`。因此源码写 `warps=[12]`，执行时
对应 `wg_id == 3 and warp_id == 0`。

#### 4. WG3 warp 0 与 WG0 warp 0 都能写成“warp 0”，为什么不能只看 `warp_id`？

答：`warp_id` 只在所属 warpgroup 内唯一。WG0 warp 0 负责 softmax
stage 0，WG3 warp 0 负责 MMA issue；它们的 CTA 线程范围分别是
`[0, 32)` 和 `[384, 416)`。必须同时检查 `wg_id`。

#### 5. 如果 softmax 行索引错误地使用全局 `tid`，最先会出现什么现象？

答：WG0 在 `tid=0..127` 时可能看起来正确，但 WG1 的 `tid=128..255`
会超出预期的本地行范围。常见结果是越界、写错 SMEM/TMEM row，或者
两个 softmax stage 互相覆盖彼此的数据；错误通常从第二个 Q stage 开始
出现。

## 十六、下一知识点

下一步进入 `5.2 Register 分配与 setmaxnreg`：解释为什么 WG0/WG1
需要更多 registers，WG2 和 WG3 为什么可以释放 registers，以及
`setmaxnreg` 如何让四个 warpgroups 共享同一个 65,536-register CTA
budget。
