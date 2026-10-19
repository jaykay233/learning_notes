# FlashKDA：从 Delta Rule 到 K1/K2 数据流

> 本笔记对应本机 `/Users/saboxu/Downloads/FlashKDA` 的源码快照 `7afb9f4`（2026-10-19 阅读）。实现会变化；下面的 chunk 大小、数据类型和 warp 映射均以该快照为准。

## 本次讲解位置

```text
章节：算子开发 / 状态空间与线性注意力算子
小节：FlashKDA v1：chunkwise KDA 的 K1 prepare → K2 recurrence
知识点：KDA chunk 内的 L / INV / Mqk，以及跨 chunk recurrent state 如何推进
上次：13-flashmla-sparse-attention-reading-guide.md 的 sparse decode、split-KV 与 softmax merge
下次：沿 K2 MMA warp 的 16×16 tile 深挖 state update、寄存器转置和 TMA pipeline
PTX：不是单条 PTX 指令专题；实现重点是 SM80 16×8×16 BF16 MMA atom 与 SM90 TMA
```

## 为什么现在讲这个

前一节的 FlashMLA 可以把长 KV 分给多个 split，再合并 partial softmax；每个 split 主要是在一组给定 KV 上做 attention。FlashKDA 的困难不同：**序列的状态有时间依赖**，后一块必须接着前一块产生的 state 算，不能把所有 chunk 都当成彼此独立的 attention split。与此同时，chunk 内仍可把 token 维矩阵化并并行计算。

FlashKDA 因此把工作切成两种并行度：K1 让每个 chunk 独立预处理，尽量铺开 token 并行；K2 让每个 `(sequence, head)` CTA 按 chunk 顺序递推 state。理解这项切分，就能避免把它误读成 softmax attention 或“多个 chunk 各算一份最后取平均”，也能解释为什么它用了 TMA / warp specialization，却不需要 FlashMLA 那种 split-softmax 合并。

## 先建立心智模型

把 KDA 想成“每个 head 持有一张可更新的记忆矩阵”，而不是把所有历史 K/V token 永久堆在一个 cache 里：

```text
输入 token (q, k, v, gate, beta)
       │
       ├─ K1：每个 16-token chunk 独立做 gate/decay、构造小矩阵并求 INV
       │       token/chunk 并行；把中间结果写到 workspace
       ▼
       workspace ───────────────┐
                               ▼
初始 state ───────────→ K2：每个 (sequence, head) 按 chunk 串行递推
                               ├─ 用当前 state 算本 chunk 的输出
                               ├─ 更新 state，交给下一个 chunk
                               └─ 写 output / 可选 final_state
```

- **KDA（Kimi Delta Attention）属于 gated delta-rule / linear attention 路线。** 它不是对所有历史 token 做 softmax；长历史的信息由 state 汇总。
- 每个 head 的 state 形状是 `[V, K]`。当前 FlashKDA 限定 `K = V = 128`，所以是 `[128,128]`。
- 一个 CTA 处理 `CHUNK=16` 个连续 token。chunk 内先变成小的 16×16 下三角线性系统；chunk 与 chunk 之间仍然保留 recurrent 顺序。
- `beta` 输入是 sigmoid **之前的 logit**；kernel 内部做 sigmoid。`g` 输入也处于 gate activation 之前，门控还结合 `A_log`、`dt_bias` 和 `lower_bound`。

## API 与边界

仓库 README / `flash_kda/__init__.py` 的接口语义：

| 张量 | 形状 | 类型 / 语义 |
|---|---|---|
| `q, k, v` | `[B,T,H,128]` | BF16；Q/K 会做 L2 normalize |
| `g` | `[B,T,H,128]` | BF16，gate activation 前的输入 |
| `beta` | `[B,T,H]` | BF16 logits，内部 sigmoid |
| `A_log` | `[H]` | FP32 |
| `dt_bias` | `[H,128]` | FP32 |
| `out` | `[B,T,H,128]` | BF16，调用方提供的输出 buffer |
| `initial_state/final_state` | `[B,H,V,K]`，变长时 `[N,H,V,K]` | 可 BF16 / FP32；内部 state 运算驻留 BF16 |
| `cu_seqlens` | `[N+1]` | 可选；提供时要求 `B=1`，定义 N 个变长序列 |

当前代码还要求 CUDA contiguous 张量、CUDA 12.9+、PyTorch 2.4+ 和 SM90+；README 的编译目标例子为 `90a,100a`。**注意**：K1/K2 的 MMA 使用 SM80 风格的 MMA atom，不代表整套 kernel 可在 SM80 运行；当前实现依赖 SM90 TMA 路径。

## 数学：chunk 内如何变成矩阵乘法

以下把 gate 写成已经激活后的 `g`，使用 base-2 指数。源码中 gate 预处理等价于：

```text
g_act = lower_bound * log2(e)
        * sigmoid(exp(A_log) * (g_input + dt_bias))
```

它是逐 token、逐 key 维度计算，通常为非正数。于是 `2**g` 表示 state 在 key 维度上的衰减。`G` 表示 chunk 内沿 token 轴的前缀和，下面所有 `exp2` 都是逐元素操作：

```text
G[t, :]             = cumsum(g_act, token_axis)
g_total[:]          = G[CHUNK-1, :]
K_decay[t, :]        = K[t, :] * 2**G[t, :]
Q_decay[t, :]        = Q[t, :] * 2**G[t, :] * scale
K_inv[t, :]          = K[t, :] * 2**(-G[t, :])
K_restored[t, :]     = K_inv[t, :] * 2**g_total[:]
```

其中 `K_decay / Q_decay / K_inv / K_restored` 形状都是 `[16,128]`，`g_total` 是 `[128]`。`K_restored` 把 chunk 内逆衰减的 key 换算到 chunk 末尾 state 的时间坐标。

接着构造两个 chunk 矩阵：

```text
L   = tril(K_decay @ K_inv.T, diagonal=-1) * sigmoid(beta)[:, None]
Mqk = tril(Q_decay @ K_inv.T, diagonal=0)
INV = (I + L)^(-1)
```

- `L` 只保留**严格下三角**，因此 token `t` 只依赖 chunk 内更早的 token；它编码了 delta update 的因果依赖。
- `INV` 把 chunk 内顺序依赖改写成一次小型三角系统求解：`U = INV @ v_adjusted`。`I+L` 是单位下三角矩阵，可以用 forward substitution 求逆/求解。
- `Mqk` 保留下三角（含对角），把 chunk 中各 token 的 update 投影到各个 query 上。

对输入 state `S`，参考实现中的核心数据流可以写成：

```text
v_adjusted = (V - K_decay @ S.T) * sigmoid(beta)[:, None]
U           = INV @ v_adjusted
out_chunk   = Q_decay @ S.T + Mqk @ U

delta       = (K_restored.T @ U).T
S_new       = S * 2**g_total[None, :] + delta
```

矩阵维度逐项核对：

```text
K_decay                 [C,K] @ S.T [K,V]       -> [C,V]
INV                     [C,C] @ v_adjusted [C,V] -> [C,V]
Q_decay                 [C,K] @ S.T [K,V]        -> [C,V]
Mqk                     [C,C] @ U [C,V]          -> [C,V]
(K_restored.T @ U).T    ([K,C] @ [C,V]).T        -> [V,K]
```

所以 `out_chunk` 是 `[C,V]`，更新后的 state 仍为 `[V,K]`。注意 state 更新里的 `2**g_total` 按 key 维度缩放，即沿 `S` 的列维广播。

### K1 如何求 16×16 的 INV

当前源码不是旧版 Neumann 级数。它保持 `L` 的 fp32 精度，把 16×16 的 `I+L` 分为两个 8×8 对角块做 FP32 forward substitution，再通过两个 BF16 HMMA 矩阵乘把非对角块合并；最后得到 BF16 `INV`。这样既利用下三角结构，也把矩阵尺寸限制在 16×16。仓库 deep-dive 说明，较大的 chunk 会加重 `exp(cumsum(g))` 的动态范围与矩阵求逆成本；`CHUNK=16` 是本版本的设计选择，不是 KDA 数学要求永远固定 16。

## 从源码看 K1 → workspace → K2

入口在 `csrc/smxx/fwd_launch.cu`：K1 grid 为 `(total_tiles, H)`，K2 grid 为 `(N,H)`；两个 kernel 按顺序在同一 CUDA stream launch。每个 kernel 的 grid 维度就是理解并行度的第一线索。

### K1：每个 chunk 独立 prepare

在 `csrc/smxx/fwd_kernel1.cuh`：

1. **CTA 找到 sequence / chunk。** 定长 batch 直接算映射；变长模式先用 tile prefix，把 `global_tile_idx` 二分映射到 sequence 和局部 chunk。prefix 统计每条序列的 `ceil(seq_len/16)`；`total_tiles` 按上界分配，落到有效 chunk 之外的 CTA 会提前 return。
2. **TMA 把输入 tile 搬到 shared memory。** 一个 K1 CTA 准备一个 head 的一个 16-token 块：Q、K、g、beta 和该 head 的 `dt_bias`。发起 TMA 的是 CTA 线程 0；其余线程可在 TMA 传输期间先算 `exp(A_log)`，随后用 transaction barrier 等待数据可见。
3. **线程做 Q/K L2 normalize 与 gate 前缀和。** K1 的 256 threads 将工作分配到行/列；对每个 key feature 得到长度 16 的 gate prefix `G` 和 `g_total`，尾块无效 token 的 gate 设 0、K 清零。
4. **形成 decay 后的矩阵。** 算出 `K_decay`、`Q_decay`、`K_inv`、`K_restored`。随后一个 warp 算 fp32 `L`，另一个 warp 算 bf16 `Mqk`；按三角形 mask 处理后求 `INV`。
5. **写中间 workspace。** 用 TMA store 输出六项：`K_decay`、`Q_decay`、`K_restored`、`g_total`、`INV`、`Mqk`。K2 不需要重新做门控累积和矩阵逆。

K1 的 shared storage 用 union 覆盖生命周期不重叠的缓冲区：先存 `q/k/g`，随后复用空间存 decay 矩阵和 `L/INV/Mqk`。这是用更复杂的生命周期管理换 shared-memory 容量与 occupancy。

### K2：每个 sequence/head 顺序推进 state

`csrc/smxx/fwd_kernel2.cuh` 的 K2 grid 是 `(N,H)`，每个 CTA 只持有一份 `[128,128]` state，并在 `for (t=0; t<t_tiles; ++t)` 中逐 chunk 循环。当前 chunk 的 state 更新完成前，下个 chunk 不可开始，因此这部分不能简单拆成独立 chunk CTAs。

默认 192 threads 分成三种角色：

| Warp | 数量 | 角色 | 工作 |
|---|---:|---|---|
| MMA | 4 | consumer / compute | 读 shared-memory tiles，做矩阵乘、输出和 state update |
| LOAD_QKG | 1 | producer | 用 TMA 加载 V、beta 和 K1 workspace；初始化时还加载 state |
| STORE | 1 | output consumer | 把 output tile 写回 global memory，并在需要时写 final state |

实际采用 3 个 input stages、2 个 output stages。LOAD warp 通过 `producer_acquire` 取得空 pipeline stage，发起 TMA；MMA warp `consumer_wait` 确认对应数据到 shared memory 后再读，用完后 `consumer_release` 释放 stage。MMA 计算后的 output stage 通过 store pipeline 交给 STORE warp 写回。也就是说，warp specialization 不改变 state 的依赖顺序，而是在一条 CTA 的 chunk 循环里重叠“加载下一份 tile”和“计算当前 tile”。

每个 chunk 在数学上的计算次序是：

```text
读取 V 与 K1 workspace
→ 用当前 state 算旧状态对 V 的预测，并形成 residual
→ 乘 beta，再用 INV 得到 U
→ Q_decay @ state.T + Mqk @ U 得到当前 chunk output
→ K_restored.T @ U 并加上衰减后的 state，得到下一份 state
```

U 会在 MMA 路径的寄存器片段之间转置传递，源码注释说明通过 `MOVM_T` 避免将 U 写回 shared memory 再读出。完整块输出用 TMA store；最后一个不满 16 token 的尾块走手动 store，只写有效行，避免覆盖变长 batch 中下一个 sequence 的数据。若需要 `final_state`，CTA 在所有 chunk 后写最终 state。

### 变长序列的 tile 映射

假设 `cu_seqlens=[0,19,24]`，`CHUNK=16`：第一条序列 19 个 token 对应 2 个 tile，第二条 5 个 token 对应 1 个 tile；prefix 是 `[0,2,3]`。K1 的 `global_tile_idx=2` 落在第二条序列：`seq_idx=1`，`local_t=2-2=0`，`bos=19`，所以它读第二条序列的第一个 chunk `[19,24)`。此例 `total_tiles=ceil(24/16)+2=4` 是上界，`global_tile_idx=3` 映射到第二条序列的局部 chunk 1 后发现越界并退出。K2 则直接由 `blockIdx.x=seq_idx` 选择序列，并对该序列的 chunk 做顺序循环。

## 和 FlashMLA、Triton / TMA 概念对照

| 项目 | FlashKDA（本快照） | FlashMLA sparse decode（前一节） |
|---|---|---|
| 算法 | gated delta-rule / recurrent linear attention | 给定稀疏索引的 softmax attention |
| 长历史表示 | 每个 head 的 `[V,K]` state | KV cache 中显式 token（可 paged/量化） |
| chunk / split | chunk 间有 state 依赖，K2 顺序递推 | KV 可 split，partial softmax 需要合并 |
| 数学核心 | 下三角 `INV`、`U`、state update | softmax score、value reduction、LSE |
| 本仓库实现 | CUTLASS/CuTe CUDA，不是 Triton kernel | 多种 CUDA / 其他平台的 kernel family |

| 硬件机制 | 在 FlashKDA 里是否用到？ | 应如何理解 |
|---|---|---|
| TMA | **是** | K1/K2 对输入、workspace、output、state 建了 TMA descriptor；K2 用 TMA pipeline |
| Warp specialization | **是，K2 明确使用** | 4 MMA + 1 LOAD + 1 STORE warp；K1 则以一个 CTA 内的线程/warp 分工完成 prepare |
| Block cluster / CGA 多 CTA | **没有看到** | grid 中 CTA 彼此独立；源码出现 `ClusterTransactionBarrier` 类型不等于创建了多-CTA cluster |
| PDL | **没有看到** | 两个 kernel 在同一 stream 顺序 launch；本路径未用 programmatic dependent launch |
| TMEM / tcgen05 | **没有看到** | 不要把 Blackwell tcgen05 示例套过来；K2 明确使用 `SM80_16x8x16_F32BF16BF16F32_TN` MMA atom，累加片段走 CUTLASS/CuTe 路径 |

TMA 本身是 SM90 能力，但使用 TMA 不代表一定要 cluster、PDL 或 TMEM。FlashKDA 的重点是：**SM90 TMA 搬运 + shared-memory pipeline + 常规 warp MMA/state recurrence**，而不是 Blackwell multi-CTA/TMEM 数据流。

## 两 token 数值轨迹：亲手算出 output 与新 state

为了能手算，把不支持的真实维度缩成 `K=V=1`、`CHUNK=2`。这里只验证公式的数据流，**不代表 FlashKDA kernel 接受该 shape**。Q/K 已归一化，取：

```text
q = [[1], [1]]       k = [[1], [1]]
v = [[3], [4]]       S = [[2]]
g_act = [-1, -1]      beta_logit = [0, 0]      scale = 1
```

先算门控、衰减和 key 变换：`sigmoid(0)=0.5`，累计 gate `G=[-1,-2]`，`g_total=-2`。

| token t | `G[t]` | `2**G[t]` | `2**(-G[t])` | `k_decay[t]` | `q_decay[t]` | `k_inv[t]` | `k_restored[t]=k_inv*2**g_total` |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | -1 | 0.5 | 2 | 0.5 | 0.5 | 2 | 0.5 |
| 1 | -2 | 0.25 | 4 | 0.25 | 0.25 | 4 | 1 |

接着逐项做小矩阵：

```text
K_decay @ K_inv.T = [[1,   2],
                     [0.5, 1]]
L = strict_lower(...) * beta[row]
  = [[0,    0],
     [0.25, 0]]
I + L = [[1,    0],
         [0.25, 1]]
INV = (I + L)^-1 = [[1,     0],
                    [-0.25, 1]]

Mqk = tril(Q_decay @ K_inv.T)
    = [[1,   0],
       [0.5, 1]]
```

对旧 state 做 prediction/residual：

```text
K_decay @ S.T = [[0.5], [0.25]] * 2 = [[1], [0.5]]
V - prediction = [[3], [4]] - [[1], [0.5]] = [[2], [3.5]]
v_adjusted = [[2], [3.5]] * [[0.5], [0.5]] = [[1], [1.75]]
U = INV @ v_adjusted = [[1], [-0.25*1 + 1.75]] = [[1], [1.5]]
```

最后分别得到输出与状态：

```text
Q_decay @ S.T = [[0.5], [0.25]] * 2 = [[1], [0.5]]
Mqk @ U = [[1], [0.5*1 + 1.5]] = [[1], [2]]
out_chunk = [[1], [0.5]] + [[1], [2]] = [[2], [2.5]]

delta = (K_restored.T @ U).T = (0.5*1 + 1*1.5) = [[2]]
S_new = 2**(-2) * [[2]] + [[2]] = [[2.5]]
```

这条轨迹强调：chunk 第二个 token 的结果已经包含第一个 token 的因果影响（由 `L/INV` 编码），而 chunk 结束后产生的 `S_new=2.5` 才交给下一个 chunk。

## 完整可运行示例

脚本 `operators/code/flashkda/kda_scalar_trace.py` 只使用 Python 标准库，可直接运行。以下是完整内容：

```python
#!/usr/bin/env python3
"""A tiny, standard-library-only trace of one FlashKDA chunk recurrence.

This uses K=V=1 and CHUNK=2 to make every intermediate value visible. It is a
mathematical teaching example, not a supported FlashKDA kernel shape.
"""
from math import exp
from typing import List, Sequence

Matrix = List[List[float]]


def transpose(a: Matrix) -> Matrix:
    return [list(row) for row in zip(*a)]


def matmul(a: Matrix, b: Matrix) -> Matrix:
    if not a or not b or len(a[0]) != len(b):
        raise ValueError("incompatible matrix shapes")
    bt = transpose(b)
    return [[sum(x * y for x, y in zip(row, col)) for col in bt] for row in a]


def lower_triangle(a: Matrix, include_diagonal: bool) -> Matrix:
    result: Matrix = []
    for i, row in enumerate(a):
        result.append([
            value if (j <= i if include_diagonal else j < i) else 0.0
            for j, value in enumerate(row)
        ])
    return result


def inverse_unit_lower_from_l(l: Matrix) -> Matrix:
    """Solve (I + L) X = I by forward substitution; L is strict-lower."""
    n = len(l)
    a = [[l[i][j] + float(i == j) for j in range(n)] for i in range(n)]
    inverse = [[0.0 for _ in range(n)] for _ in range(n)]
    for col in range(n):
        for row in range(n):
            rhs = float(row == col) - sum(
                a[row][k] * inverse[k][col] for k in range(row)
            )
            inverse[row][col] = rhs / a[row][row]
    return inverse


def scale_rows(a: Matrix, factors: Sequence[float]) -> Matrix:
    return [[value * factors[i] for value in row] for i, row in enumerate(a)]


def add(a: Matrix, b: Matrix) -> Matrix:
    return [[x + y for x, y in zip(ra, rb)] for ra, rb in zip(a, b)]


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + exp(-x))


def show(name: str, value: object) -> None:
    print(f"{name} = {value}")


def main() -> None:
    # Two tokens, one key/value feature. q and k are already L2-normalized.
    q: Matrix = [[1.0], [1.0]]
    k: Matrix = [[1.0], [1.0]]
    v: Matrix = [[3.0], [4.0]]
    g_after_activation = [-1.0, -1.0]  # base-2 log decay per token
    beta_logits = [0.0, 0.0]
    state: Matrix = [[2.0]]  # [V, K]
    scale = 1.0

    beta = [sigmoid(x) for x in beta_logits]
    g_cumsum: List[float] = []
    running = 0.0
    for gate in g_after_activation:
        running += gate
        g_cumsum.append(running)
    g_total = g_cumsum[-1]

    decay = [2.0 ** x for x in g_cumsum]
    inverse_decay = [2.0 ** (-x) for x in g_cumsum]
    k_decayed = scale_rows(k, decay)
    q_decayed = [[row[0] * factor * scale] for row, factor in zip(q, decay)]
    k_inv = scale_rows(k, inverse_decay)
    k_restored = scale_rows(k_inv, [2.0 ** g_total] * len(k_inv))

    raw_k_interaction = matmul(k_decayed, transpose(k_inv))
    l = lower_triangle(raw_k_interaction, include_diagonal=False)
    l = scale_rows(l, beta)
    mqk = lower_triangle(matmul(q_decayed, transpose(k_inv)), include_diagonal=True)
    inv = inverse_unit_lower_from_l(l)

    old_state_prediction = matmul(k_decayed, transpose(state))
    residual = [[x - y for x, y in zip(rv, pv)] for rv, pv in zip(v, old_state_prediction)]
    v_adjusted = scale_rows(residual, beta)
    u = matmul(inv, v_adjusted)

    direct_output = matmul(q_decayed, transpose(state))
    correction = matmul(mqk, u)
    output = add(direct_output, correction)

    delta_state_kv = matmul(transpose(k_restored), u)  # [K, V]
    delta_state = transpose(delta_state_kv)             # [V, K]
    decayed_state = scale_rows(state, [2.0 ** g_total])
    next_state = add(decayed_state, delta_state)

    show("beta", [round(x, 6) for x in beta])
    show("g_cumsum", g_cumsum)
    show("k_decayed", k_decayed)
    show("k_inv", k_inv)
    show("k_restored", k_restored)
    show("L", l)
    show("INV", inv)
    show("Mqk", mqk)
    show("v_adjusted", v_adjusted)
    show("U", u)
    show("output", output)
    show("delta_state", delta_state)
    show("next_state", next_state)

    assert output == [[2.0], [2.5]]
    assert next_state == [[2.5]]
    print("trace check: PASS")


if __name__ == "__main__":
    main()
```

运行：

```bash
python3 operators/code/flashkda/kda_scalar_trace.py
```

预期末尾：

```text
output = [[2.0], [2.5]]
delta_state = [[2.0]]
next_state = [[2.5]]
trace check: PASS
```

## 常见误读与可观察后果

1. **把它当作 softmax attention。** FlashKDA 没有对历史 token 的 `QK^T` 做 softmax；它用 `INV` 解 chunk 内 delta-rule 依赖。如果照 softmax 写 reference，输出会在整个序列上系统性不同。
2. **把 chunk 当成可独立的 split-K。** K1 的 prepare 可以并行，但 K2 处理下个 chunk 要等当前 `state_new`。若各 chunk 使用相同初始 state，跨 chunk 输出和 final_state 都会错。
3. **忘记 beta 是 logit。** API 输入 `beta`，内部才 sigmoid；调用前先 sigmoid 会变成 `sigmoid(sigmoid(x))`，更新幅度通常偏小。
4. **把 `g` 当作直接乘法 gate 或自然指数。** 源码先把 gate 参数变换到 base-2 的指数域并沿 token 轴累加。把 `2**G` 写成 `exp(G)` 会改变 decay 强度；不累计则会丢失 chunk 内时间位置。
5. **忽略 state 的转置与维度。** 存储为 `[V,K]`，预测要乘 `S.T`，而 `(K_restored.T @ U)` 结果需转回 `[V,K]`。错了通常会 shape error；非方阵泛化时尤其容易暴露。
6. **认为 TMA 就意味着 block cluster、PDL 或 TMEM。** 这些是不同机制。源码中的 CTA grid 和普通同 stream kernel launches 没有形成多-CTA cluster/PDL；K2 的 MMA 也不是 tcgen05/TMEM。按错误架构去找 barrier 或 profiler 指标，会对不上源码和 Nsight 轨迹。
7. **以为 K1/K2 能合并一定更快。** 仓库作者报告拆分后比早期融合版至少快 15%，原因是 K1 的 chunk 并行度远高于 state recurrence；但这是项目 deep-dive 的 benchmark 结论，不是本机验证结果，也不保证所有 shape 都相同。

## 如何验证与本次边界

在任何有 Python 3 的环境运行上面的纯 Python 标量 trace：

```bash
python3 operators/code/flashkda/kda_scalar_trace.py
```

这只验证本文缩维数学流程的中间结果与自检，不验证 CUDA kernel。FlashKDA 的真实回归入口是其仓库 `bash tests/test.sh`，需要 CUDA 12.9+、PyTorch 2.4+ 和 SM90+ NVIDIA GPU。本次环境为 macOS arm64，且没有 PyTorch、`nvcc` 或 `nvidia-smi`，因此这里完成的是源码静态阅读与 CPU 数学示例验证，**没有声称编译或运行过 FlashKDA GPU kernel**。

主要源码锚点（FlashKDA `7afb9f4`）：

- `flash_kda/__init__.py`：Python API 文档与 workspace 调用。
- `csrc/flash_kda.cpp`：dtype、contiguous、shape、K=128 与 varlen 校验。
- `csrc/smxx/fwd_launch.cu`：workspace / TMA descriptor、tile prefix、K1/K2 grid 与 launch 顺序。
- `csrc/smxx/fwd_kernel1.cuh`：K1 的 varlen tile 映射、TMA 输入、Q/K normalize、gate scan、L/INV/Mqk 和 workspace store。
- `csrc/smxx/fwd_kernel2.cuh`：K2 的 warp roles、state load、TMA pipeline、chunk recurrence、output/state store。
- `tests/torch_ref.py`：可读的 chunk 参考计算与数值约定。
- `docs/20260420-flashkda-v1-deep-dive.md`：CHUNK、K1/K2 分工、精度与性能设计说明。

## 自我检查（含答案）

1. K1 和 K2 分别沿什么维度并行？
   - **答：** K1 的 CTA 对应 head × chunk，可同时 prepare 多个 chunk；K2 的 CTA 对应 sequence × head，chunk 维在 CTA 内顺序递推。
2. `L` 为什么严格下三角，而 `Mqk` 包含对角线？
   - **答：** `L` 表示一个 token 对更早 token 更新的因果依赖；`Mqk` 还要包含当前 token 自身对当前输出的贡献。
3. `INV` 在 `U = INV @ v_adjusted` 里做什么？
   - **答：** 它是 `(I+L)^-1`，将一个 chunk 内的因果三角耦合转换为矩阵形式的解。
4. FlashKDA 是否像 FlashMLA 那样对 split 的 softmax partial 做 merge？
   - **答：** 不做。它不是 softmax attention；K2 通过 state 把 chunk 依赖接起来。
5. 代码中有 TMA、warp specialization，就能推断有 cluster / PDL / TMEM 吗？
   - **答：** 不能。当前路径明确有 TMA 与 K2 warp roles，但没有看到多-CTA cluster、PDL 或 tcgen05/TMEM 使用。

## 进度

已经覆盖：

- [x] FlashKDA 的 KDA 数学、16-token chunk 内三角系统、K1 prepare → workspace → K2 recurrent state 的执行数据流及 TMA/warp specialization 边界。

下一知识点：

- [ ] 逐段映射 K2 中 4 个 MMA warps 的 16×16 tile：U 如何留在寄存器里、state 如何分块读改写，以及 input/output pipeline 如何重叠。
