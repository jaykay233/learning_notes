# DeepGEMM K-grouped contiguous GEMM：数据流与 MoE 权重反向

## 本次讲解位置

```text
章节：DeepGEMM grouped GEMM 与 scheduler
小节：K 维连续分组与独立归约
知识点：K-grouped contiguous 的数据布局、分组边界和输出数据流
上次：M-grouped contiguous / masked 的数据组织，以及不同 expert 的 M 有何不同
下次：Scheduler 如何把 K-group 元数据映射到 CTA 的 M/N tile 与 K 范围
PTX：SM100 FP8 实现最终走 TMA 与 tcgen05/UMMA 矩阵乘路径；本节重点是分组数据语义
```

### 为什么现在讲这个

M-grouped 已经说明“各 expert 的 token 行数不同”可以通过 M 维分组处理；但 MoE 权重反向有另一种形态：每个 expert 的输入 token 数不同，**变化的是 K 归约长度，而每个 expert 要产出的权重梯度矩阵尺寸相同**。如果只看到物理上连续的 K 数据，就容易把它误认为一次更长的普通 GEMM，从而把不同 expert 的梯度错误地加在一起。本节要弄清楚：连续存储不等于共享同一个 reduction，组边界会保留每个独立 GEMM 的语义。

## 1. 心智模型：一条 K 方向数据带，多组独立 GEMM

标准矩阵乘的记号是：

```text
A_e[M, K_e] @ B_e[K_e, N] = D_e[M, N]
```

这里 `e` 是 expert/group 编号。K-grouped 的不变量是：

- 所有组的 `M` 相同；
- 所有组的 `N` 相同；
- 每组的 `K_e` 可以不同；
- 每组独立产生一个 `[M, N]` 输出 `D_e`。

DeepGEMM 的 TN contiguous API 实际把 A、B 的 K 轴放在第 0 维，输入分别像 `[sum_K, M]` 和 `[sum_K, N]`，所以每组数学上执行 `A_e.T @ B_e`。它们共享连续的物理 arena，但 group 边界告诉 kernel 哪一段 K 属于哪一个 GEMM。

```text
A arena: [A₀ 的 K 段][A₁ 的 K 段][A₂ 的 K 段]
B arena: [B₀ 的 K 段][B₁ 的 K 段][B₂ 的 K 段]

计算：D₀ = A₀ᵀ @ B₀
      D₁ = A₁ᵀ @ B₁
      D₂ = A₂ᵀ @ B₂
```

每组有自己的累加器/输出 tile。K 从一组切换到另一组时，不把前一组的累加值带过去。

## 2. 为什么 MoE 权重反向会需要它

每个 expert 的权重梯度可以写为：

```text
dW_e = X_eᵀ @ dY_e
```

若 expert `e` 收到 `T_e` 个 token，则：

```text
X_e : [T_e, hidden_in]
dY_e: [T_e, hidden_out]
dW_e: [hidden_in, hidden_out]
```

因此 `M=hidden_in`、`N=hidden_out` 对所有 expert 固定，`K=T_e` 随 token 数变化。这正是 K-grouped：每个 expert 的归约长度不同，结果矩阵大小一致。DeepGEMM README 将 K-axis grouped API 的目标描述为 MoE weight backward。

## 3. 从输入到输出的数据流

```text
每个 expert 的 X_e 与 dY_e（token 行一一对应）
                 │
                 ├─ 按 expert 的 K 段顺序写进 A/B 连续 arena
                 ├─ 保存每组真实 K 长度 ks_cpu
                 ├─ 保存物理对齐布局 grouped_layout
                 └─ FP8 路径准备对应的 SFA/SFB
                              │
                              ▼
               启动一个 K-grouped GEMM kernel
                              │
           scheduler 为每组选择 M/N 输出 tile 和 K 范围
                              │
                 TMA 搬入该组 A/B 与 scale tile
                              │
              Tensor Core 在该组内沿 K 分块累加
                              │
                              ▼
                  写到 D[group, M, N]
```

分层看代码时要分清：

1. **Host/API 层**：检查张量和分组布局，准备/转换 scale，选择 kernel。主要入口为 `k_grouped_fp8_gemm_tn_contiguous`。
2. **Scheduler 层**：用 `grouped_layout` 算当前组的 K 起点、长度，并把 CTA 的 M/N tile 归属到当前 group。
3. **Device kernel 层**：加载当前组的数据；对该组内部的 K block 进行乘加；写入该组的输出切片。

本节不把所有设备端 warp、barrier 和 stage 展开；这些属于下一步 scheduler/mainloop 细读的范围。

## 4. 元数据与 padding：真实 K 和物理 K 是两回事

K 长度可能需要按硬件 tile 对齐。Host 侧 `ks_cpu` 表示逻辑上的真实 K 长度；`grouped_layout` 描述 kernel 使用的物理分组布局。

普通 contiguous 布局中，`grouped_layout[e]` 是第 `e` 组**对齐后的 K 长度**，各组依次占据 arena；PSUM layout 中它存的是逻辑 K 结束位置，下一组从前一边界按 `k_alignment` 对齐后开始。因此：

- 逻辑有效 K（如 `ks_cpu`）描述真实 token 数；
- 物理对齐长度/边界决定这组在 arena 中的地址范围，也可能决定 scheduler/mainloop 处理的 padded K 范围；
- 普通对齐布局中的 padding 不是有效 token，但 A/B padding 必须为 0，scale padding 也必须有效，使额外乘加贡献为 0；
- FP8/FP4 的 scale-factor 布局也必须与对应组的 K 位置匹配。

具体约束会因 dtype、recipe 和架构不同。当前 API 对常规 K alignment 有 `% 128 == 0` 的检查；不要把下文为了手算选用的 4 行 alignment 当成生产硬件配置。

## 5. 数字轨迹：K=[3, 2, 4]

设三个组的真实 K 长度为 `[3, 2, 4]`。为了让地址容易手算，**玩具例子**假设按 4 对齐：

| 组 | 真实 K | 物理 K 容量 | arena 起点 | 有效范围 | 对齐空位 |
|---|---:|---:|---:|---|---|
| 0 | 3 | 4 | 0 | `[0, 3)` | `[3, 4)` |
| 1 | 2 | 4 | 4 | `[4, 6)` | `[6, 8)` |
| 2 | 4 | 4 | 8 | `[8, 12)` | 无 |

A、B 两个 arena 用完全一致的 K 分段：

```text
物理 K 索引：  0  1  2  3 | 4  5  6  7 | 8  9 10 11
所属组：       0  0  0 pad| 1  1 pad pad| 2  2  2  2
```

例如每行只有一个数（`M=N=1`）：

```text
A₀=[1,2,3]       B₀=[10,20,30]  -> D₀=1×10+2×20+3×30=140
A₁=[4,5]         B₁=[2,3]       -> D₁=4×2+5×3=23
A₂=[6,7,8,9]     B₂=[1,1,1,1]  -> D₂=6+7+8+9=30
```

输出必须是三个独立结果 `[140, 23, 30]`。若把整个 arena 当成一次普通长点积，则会得到 `140+23+30=193` 这一个数，丢失了每个 expert 各自的权重梯度。这就是“连续存储”与“合并归约”的区别。

## 6. 可运行的 Python 语义模型

以下脚本只模拟**K 分段存储与分组归约语义**，不模拟 CUDA、TMA、FP8 scale 或性能；alignment=4 仅为方便打印。

保存为 `k_grouped_trace.py` 后运行：

```python
def align(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def pack_and_compute_k_grouped(a_groups, b_groups, alignment=4):
    """Scalar M=N=1 model of contiguous K-grouped GEMM."""
    if len(a_groups) != len(b_groups):
        raise ValueError("A/B must have the same number of groups")

    ks_cpu = [len(group) for group in a_groups]
    if any(k != len(b_groups[i]) for i, k in enumerate(ks_cpu)):
        raise ValueError("A and B K lengths must match within every group")

    padded_ks = [align(k, alignment) for k in ks_cpu]
    grouped_layout = list(padded_ks)
    total_k = sum(padded_ks)
    a_arena = [0] * total_k
    b_arena = [0] * total_k

    offsets = []
    offset = 0
    for a_group, b_group, padded_k in zip(a_groups, b_groups, padded_ks):
        offsets.append(offset)
        for k_idx, (a_row, b_row) in enumerate(zip(a_group, b_group)):
            a_arena[offset + k_idx] = a_row[0]
            b_arena[offset + k_idx] = b_row[0]
        offset += padded_k

    # Each group starts a fresh accumulator and reduces only its valid K rows.
    outputs = []
    for k, start in zip(ks_cpu, offsets):
        accumulator = 0
        for k_idx in range(k):
            accumulator += a_arena[start + k_idx] * b_arena[start + k_idx]
        outputs.append(accumulator)

    return a_arena, b_arena, ks_cpu, grouped_layout, offsets, outputs


a_groups = [
    [[1], [2], [3]],
    [[4], [5]],
    [[6], [7], [8], [9]],
]
b_groups = [
    [[10], [20], [30]],
    [[2], [3]],
    [[1], [1], [1], [1]],
]

a_arena, b_arena, ks_cpu, grouped_layout, offsets, outputs = (
    pack_and_compute_k_grouped(a_groups, b_groups, alignment=4)
)
print("A arena:", a_arena)
print("B arena:", b_arena)
print("ks_cpu:", ks_cpu)
print("grouped_layout (padded K):", grouped_layout)
print("group offsets:", offsets)
print("independent D values:", outputs)
print("one naive full dot (wrong grouping):", sum(outputs))
```

预期输出：

```text
A arena: [1, 2, 3, 0, 4, 5, 0, 0, 6, 7, 8, 9]
B arena: [10, 20, 30, 0, 2, 3, 0, 0, 1, 1, 1, 1]
ks_cpu: [3, 2, 4]
grouped_layout (padded K): [4, 4, 4]
group offsets: [0, 4, 8]
independent D values: [140, 23, 30]
one naive full dot (wrong grouping): 193
```

真实 DeepGEMM 输入是矩阵而非标量：A arena 为 `[sum_K, M]`、B arena 为 `[sum_K, N]`，输出 D 为 `[G, M, N]`；每一组内部执行矩阵乘累加。这个模型仅把矩阵元素压缩成标量，以暴露 group offset、有效 K 和 accumulator reset 的语义。

## 7. 与 M-grouped 的区别

| 问题 | M-grouped | K-grouped |
|---|---|---|
| 每组变化的长度 | 输出/输入行维 M | reduction 维 K |
| 典型表达式 | `A_e[M_e,K] @ B_e[K,N]` | `A_e[M,K_e] @ B_e[K_e,N]` |
| 各组输出 | 行数可能不同，常可拼接到一个 M 长带 | 每组固定 `[M,N]`，保留 group 维 |
| MoE 典型用途 | 前向：expert 收到不同 token 行数 | 权重反向：expert 的梯度归约 token 数不同 |

## 8. 常见错误及症状

| 错误 | 为什么错 | 可能看到的结果 |
|---|---|---|
| 将 arena 上所有 K 行做一次普通 GEMM | 会把多个 expert 的贡献累加到一个输出，而非各自 `D_e` | 输出只有一份，expert 权重梯度互相串加 |
| 用 `grouped_layout` 的 padded K 当作真实 token 数 | 对齐空位不是有效样本 | 结果受垃圾 padding 影响，或者误差只在组边界附近出现 |
| A、B 使用不同的 K 边界/顺序 | 同一 K 行必须对应同一个 token 样本 | 梯度数值不对，但形状仍可能完全合法 |
| 忽略 FP8/FP4 的 scale 分组布局 | scale 必须对应所属组和 K block | 反量化幅度异常，常在分组边界附近出现 |
| 以为一个 kernel launch 就代表一个 GEMM 结果 | 一个 launch 可以调度许多独立 group 输出 | 把 launch 数误当作数学 GEMM 数，理解错输出语义 |

## 9. 源码定位与验证

当前仓库可按这条路径阅读：

1. Host API：`/Users/saboxu/Downloads/DeepGEMM/csrc/apis/gemm.hpp` 中 `k_grouped_fp8_gemm_tn_contiguous`（约 320 行）。看输入 shape、`ks_cpu` / `grouped_layout` 检查、SFA/SFB 转换和实现分派。
2. SM100 launch：`/Users/saboxu/Downloads/DeepGEMM/csrc/jit_kernels/impls/sm100_fp8_fp4_gemm_1d1d.hpp` 中 `sm100_k_grouped_fp8_gemm_1d1d`（约 325 行）。看 `GemmType::KGroupedContiguous`、TMA descriptor 和 `[G,M,N]` 输出描述符。
3. Scheduler：`/Users/saboxu/Downloads/DeepGEMM/deep_gemm/include/deep_gemm/scheduler/gemm.cuh` 中 `get_next_k_group()` 与 `get_next_block()` 的 K-grouped 分支。看如何推进 group 起点/长度并将 M/N tile 分给当前组。
4. 测试数据生成：`/Users/saboxu/Downloads/DeepGEMM/tests/generators.py` 中 `generate_k_grouped_contiguous`（约 551 行）。看真实 K、对齐 K、arena 与 reference 输出如何构造。

可运行上面的 CPU 语义模型验证分段和输出；真实 GPU kernel 的正确性仍需对应 CUDA/Blackwell 环境与 DeepGEMM 测试，CPU 模型不证明 TMA、scale 布局或硬件性能。

```bash
python3 k_grouped_trace.py
```

## 本章进度

- [x] 讲清 K-grouped 中变化的是 K 归约长度，M/N 与各组输出形状固定
- [x] 用 `[3,2,4]` 追踪 K 段偏移、对齐 padding 与每组独立结果
- [x] 给出可运行 CPU 语义模型，并指出它不验证 GPU/TMA/SF 硬件路径

下一知识点：沿 scheduler 的 `get_next_k_group()` 和 `get_next_block()`，追踪 K-group 边界如何影响 CTA tile 调度。

## 自测（含答案）

1. **K-grouped 中每个 group 变化的是哪个维度？**  答：K，即该组的归约长度；M、N 固定。
2. **K 段物理连续，是否意味着各组共享一个 accumulator？**  答：不是；每组独立归约、独立写 `[M,N]` 输出。
3. **MoE 权重反向为什么自然形成 K-grouped？**  答：`dW_e=X_eᵀ@dY_e` 的 K 是该 expert token 数，各 expert token 数不同，而权重梯度矩阵形状相同。
4. **真实 K 和对齐 K 分别由什么概念表达？**  答：真实长度由 `ks_cpu` 等逻辑长度描述；`grouped_layout` 描述物理分组/对齐布局，具体编码取决于普通或 PSUM layout。
5. **若把所有组当作一次长点积，会发生什么？**  答：得到各组结果之和的一份输出，而不是每个 expert 各自的结果。
