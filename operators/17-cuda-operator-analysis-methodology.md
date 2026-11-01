# 算子分析方法论：从分类到 tile 反推（GELU / MoE dW 两个案例）

## 本次讲解位置

```text
章节：算子开发（operators）
小节：CUDA 算子的分析方法论
知识点：① 分类 → ② roofline → ③ 下标三分法 → ④ tile 反推 → ⑤ block 内分工
        → ⑥ 搬运与流水 → ⑦ 边界统一化 → ⑧ 负载均衡 → ⑨ 正确性 → ⑩ 诊断循环
上次：16 · 官方 FlashAttention 的 varlen 与 paged KV
下次：3×TF32 拆分在无 TF32 设备（T4/sm_75）上的替代路径，或持久化 kernel + 动态取 tile
PTX：本节不是指令课；涉及 sm_75 的 LDS.128、64 KB/SM 共享内存、cp.async 不可用
```

### 为什么现在讲这个

写算子最常见的两种失败，都不是「代码写不出来」，而是**优化错了对象**和**结构判断错了**：

- **优化错对象**：MoE dW 那道题，第一反应是「输出 dW 有 134 MB，瓶颈肯定在写」。
  一算 roofline 就发现写只值 0.42 ms，而分块不充分带来的**重复读**值 3.7–7 ms —— 差一个数量级。
  （顺带一提：这个 0.42 ms 正是把「0.04 ms」写错后靠重算抓出来的，见 §2 的账目。）
  没有 roofline 这一层，这个错误无法自查，只会越优化越慢。
- **结构判断错**：把 ragged 的分组维 `g` 当成普通的 tiling 维度，对全部 `M_total` 行做一次归约，
  得到的是一个 K×N 矩阵而不是 G 个，**答案的结构就是错的**，性能再好也零分。

所以本节给的不是「某个算子的写法」，而是一条**固定顺序的决策链**：先知道该跑多快，再决定归约维放哪、tile 多大、block 里怎么分工，最后才是流水和边界。
GELU（纯 elementwise）与 MoE dW（ragged contraction）这两个案例放在链子的两端，正好能看出同一套流程怎么自动剪枝。

---

## 0. 全景：十个阶段

```text
① 算子分类  →  ② Roofline 算上限  →  ③ 下标三分法(定位并行/归约/复用)
      ↓                                        ↓
⑨ 正确性清单 ← ⑧ 负载均衡 ← ⑦ 边界统一化 ← ⑥ 搬运与流水 ← ⑤ block 内分工 ← ④ tile 尺寸反推
      ↓
⑩ 诊断循环(ncu 指标 → 病因 → 药方)，回到对应阶段
```

**唯一不能压缩的顺序是 ①② 最优先。** ③④⑤ 之间是互相约束、需要联合求解的（第 3、4、5 节会说明），但 ①② 必须最先做完。

---

## 1. 阶段①：算子分类

分类不看名字，看三个结构特征：**输出尺寸关系、有没有归约、操作数如何复用**。

| 类别 | 特征 | 例子 | 主矛盾 |
|---|---|---|---|
| Elementwise | 输出=输入尺寸，1:1，无复用 | GELU、ReLU、add | 纯带宽，只能优化访存 |
| Reduction | 输出比输入小一维 | sum、softmax、layernorm | 跨线程归约 + 带宽 |
| Contraction / GEMM 型 | 输出 = 两操作数各自一个维度 | matmul、`dW = XᵀY` | 计算量 vs 分块重复读 |
| Ragged / Grouped | 归约维长度可变 | MoE grouped GEMM、dW | 上面 + 负载不均 |
| Stencil | 输出依赖邻域 | conv、pooling | 复用靠 halo / tile |

**为什么必须先分类**：Elementwise 的手段（向量化、合并访存）在 GEMM 上几乎无效；GEMM 的 tile/寄存器分块用在 elementwise 上是纯浪费。归类错了，后面九个阶段全部用错。

---

## 2. 阶段②：Roofline —— 先算出「该跑多快」

三个数就够了：

```text
计算时间下限 T_c = 总 FLOP   / 峰值算力
访存时间下限 T_m = 总字节数  / 峰值带宽
机器平衡点   I   = 峰值算力  / 峰值带宽        [FLOP/Byte]

算术强度 AI = 总 FLOP / 总字节数
AI > I  → 计算受限（可以放心做计算换访存）
AI < I  → 带宽受限（一切优化都必须减少字节数）
```

以 **T4（sm_75）** 为例：8.1 TFLOPS、320 GB/s、40 SM、4 MB L2 → `I = 25.3 FLOP/Byte`。
（T4 的 FP32 = 40 SM × 64 核 × 1.59 GHz × 2 = 8.14 TFLOPS。）

### 两个案例的 roofline

GELU，N = 5×10⁷：

| 项 | 数值 | 推导 |
|---|---|---|
| FLOP | ~1.0–1.5 G | 每元素含 `erff` 约 20–30 op |
| 字节 | 400 MB | 读 200 MB + 写 200 MB |
| AI | ≈ 3 FLOP/B | ≪ 25.3 |
| T_c | 0.12–0.19 ms | 1.5e9 / 8.1e12 |
| **T_m** | **1.25 ms** | 400 MB / 320 GB/s |

→ **带宽受限，下限 1.25 ms，优化空间只有「减少/合并字节」**。`erff` 用了多少指令都不重要。

MoE dW，G=64、M=32768、K=1024、N=512：

| 项 | 数值 | 推导 |
|---|---|---|
| FLOP | 34.4 G | 2 × 32768 × 1024 × 512 |
| 最小字节 | 335 MB | X 134 + dY 67 + dW 134 |
| AI | **103 FLOP/B** | ≫ 25.3 |
| **T_c** | **4.24 ms** | 34.4e9 / 8.1e12 |
| T_m | 1.05 ms | 335 MB / 320 GB/s |

→ **计算受限，下限 4.24 ms**。写 dW 的 134 MB 只值 **0.42 ms**（`134 MB / 320 GB/s`），
占计算下限的 10% —— 是笔可观的账，但**不是瓶颈**。
真正要防的是分块之后的**重复读**（第 4 节）。

### 这一步的额外收益：知道朴素实现有多差

MoE 的 thread-per-output 版本，`X[t][k]` 被 N=512 个线程各读一遍、`dY[t][n]` 被 K=1024 个线程各读一遍：

```
134 MB × 512 + 67 MB × 1024 ≈ 137 GB  →  137 GB / 320 GB/s ≈ 430 ms
```

430 ms vs 4.24 ms，**差 100 倍**。这个数字直接告诉你：**分块是必需项，不是可选项**。

---

## 3. 阶段③：下标三分法 —— 「选并行维度」的完整版

把算子里每一个下标列出来，逐个归类。这是全套方法里最可操作的一步。

```
dW[g][k][n] = Σ_t  X[t][k] · dY[t][n]
     ↑  ↑  ↑         ↑
     |  |  |         └─ 归约维
     |  └──┴─ 独立维(输出)
     └─ 分组维
```

| 下标类型 | 判据 | 处理 | 禁忌 |
|---|---|---|---|
| 独立维 | 不出现在求和里，输出要它 | 并行（blockIdx.x / .y） | — |
| 分组维 | 改它的值会**换掉整段参与求和的元素** | 并行，放最外层 grid.z | 不要塞进 block 内共享归约 |
| 归约维 | 出现在 Σ 里 | 串行累加，层级待定（见下） | 除非必要，不要跨 block |

### 3.1 分组维 vs 独立维

区别只有一句话：**分组维会把归约域切成几段，独立维只在同一个归约域里移动下标。**

判定测试：

> 改变这个下标的值，会不会改变「参与求和的是哪些元素」？
>
> - 会 → 分组维。`g` 从 0 变到 2，求和范围从 token `[0,2)` 换成 `[2,3)`。
> - 不会 → 独立维。`k` 变到 1，求和永远是那 3 个 token，只是取第 0 列还是第 1 列。

辅助判据（处理边界时用）：**每个下标值对应的输出是否落在独立的、互不交叠的连续块里**。
`g` 对应 `dW + g*K*N`，是分立的 G 个矩阵，块间跳一大步；`k`、`n` 只是同一矩阵内的偏移。

落到代码上的差异：

| 方面 | 独立维 `k, n` | 分组维 `g` |
|---|---|---|
| 映射 | `blockIdx.y / .x` | `blockIdx.z`（最外层） |
| 下标来源 | 纯算术 `blockIdx.y * TILE` | **必须读显存** `group_offsets[g]` |
| 归约长度 | 所有 tile 一样，均匀 | 每个值不同，参差，可为 0 |
| 边界 | 只在张量末尾 | **在张量中间**（每段分界）+ 空段 |
| 输出地址 | 小步长偏移 | 大步长跳跃 `g*K*N` |
| 搞错的后果 | 慢一点 / bank conflict / 边界错 | **结果完全错**：算进别的专家的 token |
| 影响负载均衡 | 不影响 | **直接影响**（长短段 block 耗时差几倍） |

最后两行是最要紧的：独立维搞错通常只是性能问题，**分组维搞错是正确性问题**。

### 3.2 它其实是三档光谱

模糊感来自「分组」有强弱之分：

**① 纯独立维** —— 完全均匀、无间接寻址。

```
A[i][j] = B[i][j] + 1
```

**② 均匀分组维** —— 结构上分组（每个值有自己的数据切片和输出块），但**长度相同、地址可闭式算出**，可以当独立维压平处理。

```cuda
// batched GEMM：base = b * K * N，闭式，不需要任何 offset 数组
C[b][k][n] = Σ_m A[b][k][m] * B[b][m][n]
// 压平：g = flatIdx / tilesPerGroup，零分支
```

**③ 参差分组维（ragged）** —— 长度不一、需要 `group_offsets` 间接寻址、可能有空组。
**MoE 的专家属于这一类**，必须特殊处理：读 offset、异构归约长度、空组写零、按长度排序调度。

前面说的「按段长降序启动 block」「选择性 split-K」「空组写零」**只对第 ③ 类有意义**；套到第 ② 类是纯浪费（均匀分组根本没有尾巴）。

### 3.3 一个容易搞混的点：ragged 不一定落在归约维

MoE 题目开头专门强调的正是这一点：

- **前向 grouped GEMM**：ragged 的是**输出行**（每专家输出不同行数，归约维 K 固定）。
- **反向 dW**：ragged 变成了**归约维**（每专家归约不同 token 数，输出固定 K×N）。

但**无论哪个方向，`g` 都是分组维**。分类看的是「哪个下标切分数据和输出空间」，不是「哪个下标参差」。
这个区别决定组织方式：前向是「归约域固定、输出范围参差」，反向是「输出固定、归约域参差」，两者的 tile 形状和负载均衡策略完全不同。

### 3.4 归约维的三个层级 —— 真正的分叉点

| 层级 | 做法 | 代价 | 何时用 |
|---|---|---|---|
| 线程内 | 单线程循环累加 | 无额外代价 | 归约长度 ≤ 几百，**首选** |
| block 内 | shared memory + 树形归约 | 一次 sync + smem | 线程内装不下，或需要更高并行度 |
| 跨 block | split-K + atomicAdd / 两趟 reduce | **额外写一整份输出 + 原子流量** | 只在负载严重倾斜时**选择性**使用 |

判据：

```
跨 block 归约的成本 ∝ 输出大小
跨 block 归约的收益 ∝ 归约长度的减少

值得 split-K  ⟺  输出字节数 ≪ 归约带来的收益
```

MoE 这道题输出 134 MB，**任何全局 split-K 都要多写/多次原子累加 134 MB**，几乎必然把计算受限变成访存受限。所以默认不 split，只在单专家 token 数越过阈值时对那一个专家开 split。

### 3.5 并行度够不够的闸门检查

选完并行维度，立刻数工作单元：

```
单元数 = ∏(独立维的 tile 数) × 分组维大小
要求：  单元数 ≥ 4~8 × SM 数 × 每 SM 驻留 block 数
```

- 只按 `g` 并行 → 只有 **64** 个 block，而 T4 有 40 SM × 2 block = 80 个槽位，**一轮都填不满**（0.8 波），**直接否掉**
- 加上 k/n tile → 64 × 8 × 4 = **2048** 个单元，够用
- GELU → 5e7/4 = 1250 万个线程，远超需求（此时反向要担心**单元太多导致发射开销**，用网格跨步循环把 block 数压到 65535 以内）

**这就是「先确定并行维度」的量化标准**，但它发生在归约层级决策之后。

---

## 4. 阶段④：Tile 尺寸 —— 用公式反推，不要试

tile 尺寸是**唯一能算出来的参数**。分块把「重复读」引进来了，重复倍数取决于 tile 大小。

对 contraction 型，`Tk × Tn` 的 tile：

```
X  被读 N/Tn 遍  →  (N/Tn) · M·K · 4 B
dY 被读 K/Tk 遍  →  (K/Tk) · M·N · 4 B
dW 写一遍        →  M_out · 4 B   （相对小，先忽略）

总读字节 = 4·M·K·N·(1/Tn + 1/Tk)
```

要求 `T_m ≤ T_c`：

```
4·M·K·N·(1/Tn + 1/Tk) / BW  ≤  2·M·K·N / P
⇒  2·(1/Tn + 1/Tk)  ≤  BW/P  =  1/I
⇒  1/Tn + 1/Tk  ≤  1/(2I)          ← 判据（只依赖机器平衡点 I）
```

T4 的 `I = 25.3` → 阈值 `1/(2×25.3) = 0.0198`：

| tile | `1/Tk + 1/Tn` | 判定 |
|---|---|---|
| 64×64 | 0.03125 | ✗ 超阈值 58% → 只能跑出 memory bound |
| **128×128** | **0.015625** | ✓ 有 21% 余量，T4 的最小可行值 |
| 256×128 | 0.011719 | ✓ 但每线程累加器 128 个，寄存器装不下 |

所以「**128×128 正好落在交叉点**」是算出来的。代入实际字节验证：

```
X  读 4 遍 → 134 × 4 = 536 MB
dY 读 8 遍 →  67 × 8 = 536 MB
dW 写 1 遍 → 134 MB
合计 1206 MB → /320 GB/s = 3.77 ms   <   T_c = 4.24 ms ✓（89% 余量很薄）
```

**余量这么薄，说明「靠 L2 把重复读吃掉」是必需的，不是加分项。** 这引出 T4 的一个红利：

```
单个专家一个 tile 的切片：X 512×1024×4 = 2 MB，dY 512×512×4 = 1 MB，合计 3 MB
T4 L2 = 4 MB  →  一个专家的 32 个 block 挨着跑，重复读大部分命中 L2
```

前提是同一时刻别在跑太多个专家：2 block/SM × 40 SM = 80 个并发 block ≈ 2.5 个专家 = 7.5 MB，略超 4 MB，属于「部分命中」。
结论：**`grid.z = 专家`，让同一专家的 block 连续发射**，在 T4 上从「小优化」升级为「必做」。

### 同一套公式在其他类别上的退化形式

| 类别 | 重复倍数 | 退化后的「tile」参数 |
|---|---|---|
| Elementwise | 无重复 | tile 无意义，唯一参数是**向量宽度**（float / float4） |
| Reduction | `N/Tn`（N 是输出个数） | 同上公式，但只剩一项 |
| Stencil | `(T+halo)/T` | halo 由卷积核大小决定 |
| Ragged | 同 contraction，但 Tk 方向的 `M` 变成每段长度 `len` | tile 不变，**负载均衡另算**（第 8 节） |

### 定完 tile 必须过的三个约束

```
寄存器： 累加器数 = (Tk/线程micro_K) × (Tn/线程micro_N)  ≤ 128 − 开销(~30)
共享内存：2 × 2 × CHUNK_T × (TILE+PAD) × 4 B × 驻留 block 数  ≤ 64/96/228 KB
整除性： K、N 是否 tile 的倍数（不是就要做边界，见第 7 节）
```

---

## 5. 阶段⑤：block 内部分工 + `__launch_bounds__`

block 有了 `128×128` 的输出，要分给 256 个线程。四条硬约束按顺序过。

### 约束 1：每线程必须持多个累加器

一个线程只算 1 个输出 → 每次 FMA 都要重读操作数 → 就是 thread-per-output 灾难。
**累加器数就是复用倍数。**

### 约束 2：累加器数由寄存器预算封顶

```
T4：65536 reg / SM
__launch_bounds__(256, 2)  →  65536 / (256 × 2) = 128 reg/thread
累加器 ≤ 128 − (寻址 + 切换开销 ≈ 30)  →  ≤ 64 个
```

`8×8 = 64` 恰好是上限。这是**这道题在「64 累加器 × T4 只有 64K 寄存器」下的唯一解**，不是随便挑的。

### 约束 3：内层循环的计算:访存 ≥ 4:1

```
内层每次迭代：读 a[8] + b[8] = 2 条 LDS.128（拆成 4 条 8-float 读取）
              做 8×8 = 64 次 FMA
比例 = 64 / (16 个元素) = 4:1
```

比例掉到 2:1 以下，shared memory 带宽会先饱和，FMA 单元开始闲置。

### 约束 4：2 的幂分解 + 对齐

`TILE=128` 拆成 `16 线程 × micro 8`：

```cuda
kk0 = (tid % 16) * 8      // 0,8,...,120 —— 全是 4 的倍数
→ float4 加载天然 16 B 对齐 → 一条 LDS.128 取 8 个 float
```

micro tile 若不是 4 的倍数，只能用 4 条 LDS.32，**shared 带宽直接翻 4 倍**。

### 5.1 `__launch_bounds__(maxThreads, minBlocks)` 到底在做什么

它是**给编译器看的编译期承诺**，不改运行时行为。

```cuda
__launch_bounds__(256, 2)
//                ↑  ↑
//                |  └ 请保证每个 SM 至少能驻留 2 个这样的 block
//                └ 每个 block 最多 256 线程
```

- 第一个参数：编译器提前知道「最多 256 线程」，才能推算出每线程的寄存器预算。不给它，编译器按最坏情况（1024 线程/block）保守压低寄存器。
- 第二个参数：真正起作用的部分。`65536 / (256×2) = 128`，逼编译器把每线程寄存器压到 128 以内，压力部分 spill 到 local memory。

不写会怎样：

| 每线程寄存器 | 可驻留 block/SM | warp 数 | 占用率 |
|---|---|---|---|
| 168（编译器自作主张） | 1 | 8 | 12.5% |
| **128（被 `,2` 卡住）** | **2** | **16** | **25%** |
| 64 | 4 | 32 | 50% |

为什么不是 1 或 4：

- **1 个 block**：8 warp 太少。`LDS` 有 20–30 cycle 延迟，16 次 FMA 之间插着一次 LDS，8 warp 藏不住。
- **4 个 block**：需要 ≤ 64 reg/thread，而**光累加器就 64 个**，必然大规模 spill，比降占用率更糟。
- **2 个 block**：唯一「不 spill 且吞吐可接受」的点。

### 5.2 它只解决寄存器，另一半在共享内存

**能否真正驻留 2 个 block，还要看 smem 够不够**，两个条件必须同时满足：

| `CHUNK_T`（双缓冲） | 每 block smem | 2 block 合计 | 结果 |
|---|---|---|---|
| 16 | 33.8 KB | 67.6 KB | ✗ 超过 64 KB → 只能 1 block，`,2` 白设 |
| **8** | **16.5 KB** | **33.0 KB** | ✓ 两个约束同时满足 |

每 block smem 推导：`2(双缓冲) × 8(token) × 132(stride) × 2(X,Y) × 4B = 16896 B`。

### 5.3 必须验证：有没有 spill

```bash
nvcc -O3 -arch=sm_75 -Xptxas -v moe_expert_dw.cu -c
```

看输出里的：

```text
ptxas info: Used 106 registers, ...
            0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
```

**`spill stores` 必须为 0。** 若不为 0：
(a) `MICRO_N` 8→4（累加器 32 个），代价是计算:访存比降到 2:1；
(b) 去掉第二个参数，接受 1 block/SM，用占用率换寄存器。

### 5.4 两个常见误解

- 它**不会**强制调度器真的跑 2 个 block；驻留数还取决于 smem、block 总数、硬件策略。
- 它**不是** `-maxrregcount`：后者对整个编译单元生效，`__launch_bounds__` 是 per-kernel 且优先级更高。多个 kernel 需求不同时用后者之外的这个。

---

## 6. 阶段⑥：数据搬运与流水 —— 唯一的隐藏延迟战场

计算:访存比达标后，剩下的时间全花在**等**上。

### 6.1 合并访存：铁律是「同一个 warp 内地址连续」

```cuda
idx = tid + it * blockDim.x;   // ✓ warp 内 idx 连续 → 地址连续 → 完美合并
```

反例（很常见）：想「每线程读一个 float4」，写成 `idx = tid*4 + it`，warp 内地址间隔 16 B，
**每轮只用 1/4 个 sector，效率 25%**。要读 float4 就必须让 `tid` 映射到连续 float4 下标。

### 6.2 双缓冲：别让 load 和 compute 串行

| 架构 | 手段 | 注意 |
|---|---|---|
| sm_80+（A100/H100） | `cp.async` 异步拷贝 | 硬件级，最省寄存器 |
| **sm_75（T4）** | **手写：先把 global 读数发到寄存器 → 算当前块 → 再写 smem** | 需要 `2×CHUNK_T` 的寄存器做中转 |
| 通用 | 至少 double buffer，两个 buffer 交替 | sync 次数容易写多 |

T4 上的副作用值得记一笔：中转用的 `rx/ry` 每个线程 8 个寄存器，**正是它把寄存器预算从「宽裕」推到「刚好 128」**。

### 6.3 同步次数最小化

初版直觉会写成 `load → sync → compute → sync`，每 stage 2 次 sync。可以降到 1 次，依据：

> 算 `cur` 和写 `nxt` 是**两个不同 buffer**，不冲突；
> 而 `nxt` 上一轮被读完之后，**上一轮结尾的 sync 已经保证了安全**。

**每次写 sync 都要问自己：「这次 sync 保护的到底是哪一对读/写」。** 答不出来就说明多写了。

### 6.4 bank conflict

判据：**同一 warp 内落到同一个 bank 的访问次数 > 1**。

- `stride = TILE + 4 = 132` 这种 padding 只解决**跨行**冲突。
- `LDS.128` 的 bank 宽度问题另说：一个 warp 里 16 个线程按 `kk0` 步长 8 float（32 B）读，
  8 个线程一轮只覆盖 256 B 中的 128 B → **2 路冲突**，两条 `LDS.128`（`a0`、`a1`）各命中一次。
- 注意：这种冲突**不在行之间，而在同一行内**，所以「按行 XOR swizzle」这一类手法**对它无效**。
  真正的修法是改 micro tile 沿 k 的持有方式（让一轮内 8 个线程读连续 16 B 块），或把 `micro_K` 降到 4。
  **改完必须用 ncu 的 `shared load bank conflict` 指标验证，不要凭直觉判定已修好。**

---

## 7. 阶段⑦：边界与特殊情形统一化 —— 用「填零」消灭分支

性价比最高的一个技巧，三种用途几乎覆盖所有边界问题。

### 用法 1：消掉 ragged 边界

```cuda
// ✗ 分支版：内层循环带条件，无法完全展开
for (int t = 0; t < len; ++t) { if (...) acc += ...; }

// ✓ 填零版：固定 CHUNK_T 次迭代，越界处 smem 里是 0
#pragma unroll
for (int t = 0; t < CHUNK_T; ++t) { ... 64 次 FMA ... }
```

代价：最多 `CHUNK_T−1 = 7` 次空迭代，`len=512` 时约 **1.4%**。
换到的是**完全展开、无分支、可向量化的 FMA 主体**。

### 用法 2：消掉「空组」特判

空专家 `len = 0` → `nStages = 0` → 归约循环整个跳过 → `acc` 保持 0 → **写回路径自然产出全零**。
不需要单独 memset kernel，也不需要 `if (is_empty)` 分支。
配合题目「dW 未初始化，每个元素都必须写」的要求，一条特殊路径变成了零成本。

### 用法 3：K/N 非整 tile

**load 端填零、store 端判断** 是标准分工：填零让内层循环满；写不能越界，所以 store 端保留边界检查。

---

## 8. 阶段⑧：负载均衡 —— 只看最长，不看平均

ragged/grouped 类算子的专属阶段；一般算子里的对应物是「尾波」。

### 判据

```
单 block 耗时 ≈ L × (tile 的 FMA 数) / (单 SM 算力 / 驻留 block 数)
总时长下限 ≈ max( 总计算量 / 总算力 , 最长单 block 的耗时 )
```

**是否要处理，取决于最长段，与平均段无关。**

MoE 性能配置的实际数字（T4，2 block/SM，`L × 0.324 µs`）：

| 最长段 L | 单 block 耗时 | 判定 |
|---|---|---|
| 512（平均） | 166 µs | 远小于总时长 4.24 ms |
| 4096 | 1.33 ms | 仍可被其他 block 填平 |
| 8192 | 2.65 ms | 接近，开始危险 |
| **>13000** | **> 4.24 ms** | **成为硬上限，整体被拖慢** |

推导：单 block FLOP = `2 × 128 × 128 × L`；SM 峰值的一半（与另一 block 共享）= 101 GFLOP/s；
`L=512` 时 `16.8 MFLOP / 101 GFLOP/s = 166 µs`，故 `166/512 ≈ 0.324 µs/token`。

### 成本递增的四种手段

| 手段 | 成本 | 效果 |
|---|---|---|
| **按段长降序启动 block** | ≈ 0（一个小 kernel 排个序） | 长任务先跑、短任务填尾，通常吃掉大半尾巴 |
| 选择性 split-K | 只对超阈值专家付原子/多写代价 | 削掉单点尖峰 |
| 持久化 kernel + 动态取 tile | 需全局计数器 + tile 序列表 | 最彻底，同时治尾波 |
| 全局 split-K | 输出大小 × 份数，往往灾难性 | 一般不要 |

**比例判断**：`最长段 / 平均段` < 4 时手段 1 就够；> 10 才考虑 2 或 3。

---

## 9. 阶段⑨：正确性清单

这类题的分往往丢在正确性而不是性能上。固定清单：

- [ ] **输出全部写满**（尤其未初始化 buffer、空组）
- [ ] **累加精度**：float32 起步；归约长度 > 1e4 考虑 Kahan 或分段 double
- [ ] **别名 / 原地**：`Y == X` 时读写顺序
- [ ] **边界**：N=1、K=1、tile 比整个维度大、G 组全空
- [ ] **下标溢出**：`M×K`、`G×K×N` 用 `size_t`（本题 `G·K·N = 3.4e7` 安全，但到 1e8 就该警惕）
- [ ] **确定性**：用了 `atomicAdd` 后浮点累加顺序不定 → 不可复现（评测若比 bit-exact 会挂）
- [ ] **精度语义**：题目禁止 TF32/FP16 时，不能顺手用张量核加速

---

## 10. 阶段⑩：诊断循环

**前提：先有 naive baseline 的数字。** 没有 baseline，你不知道任何优化到底有没有用。

| ncu 症状 | 病因 | 回哪个阶段 |
|---|---|---|
| SM Busy < 60%，有明显 bubble | 流水没重叠 | ⑥ 双缓冲 |
| `long_scoreboard` stall 高 | global load 延迟暴露 | ⑥ 预取 / ⑤ 提占用率 |
| `shared load bank conflict` > 0 | 跨行 stride 或 LDS.128 宽度 | ⑥ §6.4 |
| `spill stores` > 0 | tile 太大 / launch_bounds 太紧 | ⑤ 降 micro tile |
| DRAM Throughput > 80% 且 AI 低 | **真的**访存受限 | ④ 加大 tile / ⑥ 靠 L2 |
| SM Busy 高但 FMA 占比低 | 指令开销、分支、地址计算 | ⑦ 填零法 / 展开 |
| 最后几波空转 | 负载不均 | ⑧ 排序 / split-K |
| Achieved Occupancy < 25% | 寄存器或 smem 卡住 | ⑤ 调 launch_bounds / ⑥ 减 CHUNK_T |
| 实际时间 ≈ 访存下限但远高于计算下限 | **分类错了** | 回 ① 重新归类 |

---

## 11. 案例 A：GELU（elementwise，走最短路径）

`GELU(x) = 0.5·x·(1 + erf(x/√2))`，精确 erf 形式，不用 tanh 近似。

完整文件：[code/cuda/gelu_erf.cu](code/cuda/gelu_erf.cu)。核心部分：

```cuda
constexpr float kInvSqrt2 = 0.70710678118654752440f;   // 1/sqrt(2)

__device__ __forceinline__ float gelu_scalar(float x) {
    return 0.5f * x * (1.0f + erff(x * kInvSqrt2));
}

// 最简版：一线程一元素。正确性最好写。
__global__ void gelu_naive(const float* __restrict__ X, float* __restrict__ Y, int N) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    Y[i] = gelu_scalar(X[i]);
}

// 向量化版：一次搬 16 B，全局访存事务数降到 1/4
__global__ void __launch_bounds__(256)
gelu_vec4(const float4* __restrict__ X4, float4* __restrict__ Y4, int n4) {
    const int stride = gridDim.x * blockDim.x;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += stride) {
        const float4 v = X4[i];
        float4 r;
        r.x = gelu_scalar(v.x);
        r.y = gelu_scalar(v.y);
        r.z = gelu_scalar(v.z);
        r.w = gelu_scalar(v.w);
        Y4[i] = r;                      // 16 B 连续写
    }
}
```

`extern "C" void solve(...)` 里做三件事：算 `n4 = N/4`、按 `n4` 开 grid（上限 65535，靠网格跨步循环吃完）、再用 `gelu_tail` 处理 `N % 4` 的 0–3 个尾巴元素。

**方法论在这题上自动剪枝的结果**：

| 阶段 | 结论 |
|---|---|
| ① 分类 | elementwise，无归约、无复用 |
| ② roofline | 带宽受限，下限 1.25 ms，`erff` 多贵都不重要 |
| ③ 下标三分法 | 只有 `i` 一个下标，纯独立维，**没有归约维可放** |
| ④ tile | 退化：不给 tile，只给**向量宽度** float4（对齐由 `cudaMalloc` 的 256 B 保证） |
| ⑤ block 分工 | 退化：线程之间零交互，没有累加器 |
| ⑥ 流水 | 退化：只保留「合并访存」一条 |
| ⑦ 边界 | `N % 4` 用独立 tail kernel，而不是让主循环带条件 |
| ⑧ 负载均衡 | 均匀，无需处理 |
| ⑩ 诊断 | 只看 DRAM Throughput 是否接近 100% |

**一维算子做完 ②④⑥ 就到头了**，这就是「同一套流程在不同类别上自动剪枝」的直观例子。

### 数值校验（可用 CPU 复算）

| x | `erf(x/√2)` | `0.5·x·(1+…)` | 题面期望 |
|---|---|---|---|
| 0.0 | 0.0 | 0.0 | 0.0 |
| 1.0 | 0.682689 | 0.841345 | 0.8413 |
| −1.0 | −0.682689 | −0.158655 | −0.1587 |
| 2.0 | 0.954500 | 1.954500 | 1.9545 |
| −3.0 | −0.997300 | **−0.004050** | **−0.0041** |

---

## 12. 案例 B：MoE 专家权重梯度（ragged contraction，全程用满）

完整文件：[code/cuda/moe_expert_dw.cu](code/cuda/moe_expert_dw.cu)（含 naive 基线 + 分块版 + 排序 kernel）。

### 12.1 核心决策回顾

| 阶段 | 决策 | 依据 |
|---|---|---|
| ① 分类 | ragged contraction | 有归约 + 归约维长度参差 |
| ② roofline | 计算受限，T_c = 4.24 ms | AI = 103 ≫ I = 25.3 |
| ③ 并行 | `grid = (nTile, kTile, g)`，`t` 留线程内 | 只按 g 并行仅 64 单元；split-K 要写 134 MB |
| ④ tile | 128×128 | `1/128+1/128 = 0.0156 ≤ 1/(2I) = 0.0198` |
| ⑤ block 分工 | 256 线程 = 16×16，micro 8×8 = 64 累加器 | 128 reg/thread 的封顶；内层 4:1 |
| ⑥ 流水 | `CHUNK_T=8` 双缓冲，手写寄存器中转，1 sync/stage | sm_75 无 `cp.async`；2 block/SM 需 ≤ 32 KB |
| ⑦ 边界 | load 端填零，store 端判断 | 内层循环完全展开；空专家自然写零 |
| ⑧ 均衡 | 按段长降序重排 `blockIdx.z` | 阈值 L > 13000 才需要 split-K |
| ⑨ 正确性 | 空专家写零、`size_t` 下标、无 atomic | — |

### 12.2 关键代码：一次 stage 的三段

```cuda
for (int st = 0; st < nStages; ++st) {          // 空专家 nStages==0，整个跳过
    const int  cur     = st & 1;
    const int  nxt     = cur ^ 1;
    const bool hasNext = (st + 1 < nStages);

    // (1) 先把下一 chunk 的 global 读数发到寄存器（sm_75 没有 cp.async）
    float rx[LOAD_ITERS], ry[LOAD_ITERS];
    if (hasNext) { /* 4 次迭代，idx = tid + it*256，越界填 0 */ }

    // (2) 算当前 chunk：每 token 64 次 FMA 只配 4 条 LDS.128
    #pragma unroll
    for (int t = 0; t < CHUNK_T; ++t) { /* a[8] × b[8] → 64 次 fmaf */ }

    // (3) 写入另一块 buffer；cur 与 nxt 不冲突，只需一次 sync
    if (hasNext) { /* 写 sX[nxt]、sY[nxt] */ __syncthreads(); }
}
```

第 (3) 段是安全性的关键，值得单独确认一遍：

- `st=0`：算 `sX[0]`，写 `sX[1]`，sync。
- `st=1`：算 `sX[1]`，写 `sX[0]` —— 此时所有线程都已离开 `sX[0]`，因为 `st=0` 结尾的 sync。
- 最后一轮 `hasNext=false`，不写、不 sync，循环结束。

### 12.3 数值轨迹：block `(g=0, kTile=0, nTile=0)` 的 thread 0

`kk0 = 0`、`nn0 = 0`。`s=0`、`len=2000`。

| 步骤 | 内容 | 具体值 |
|---|---|---|
| stage 数 | `⌈2000/8⌉` | 250 |
| stage 0 载入 | X 行 0–7 的 k∈[0,128) | `sX[r][0..127]`，8×128 个 float |
| stage 0 thread 0 | 读 `a = sX[0][t][0..7]`、`b = sY[0][t][0..7]` | 16 个 float |
| stage 0 thread 0 | `acc[i][j] += a[i]*b[j]` | 64 次 FMA |
| 250 stage 累计 | `acc[0][0] = Σ_{t=0}^{1999} X[t][0]·dY[t][0]` | 2000 次累加 |
| 写回 | `dW[0][0][0] = acc[0][0]`（`float4` 一次写 4 个） | — |

一个 block 的总量：`128 × 128 × 2000 = 3.28e7` 次 FMA = 65.5 MFLOP。
T4 单 SM 峰值 202.5 GFLOP/s，与另一 block 共享 → `65.5e6 / 101e9 ≈ 648 µs`。

### 12.4 全量账目（性能配置）

| 项 | 数值 |
|---|---|
| block 数 | `64 × (1024/128) × (512/128) = 2048` |
| 并发 block（2/SM × 40 SM） | 80 |
| 波次 | 2048 / 80 = **25.6** |
| 每 block 读 global | 64 stage × 8 KB = 512 KB（两 block 合计每 SM 每 stage 16 KB） |
| 全量读 | 1.07 GB（X 4 遍 + dY 8 遍） |
| 折算所需 DRAM 带宽 | `1.07 GB / 4.24 ms ≈ 253 GB/s` = T4 峰值的 **79%** |
| 结论 | 计算受限但带宽余量只剩 21% → **必须靠 L2 命中那 3 MB/专家**，否则直接掉进 memory bound |

这一行「79%」是整道题最脆的地方：单看 roofline 结论是「计算受限」，但实际 DRAM 需求已经贴在峰值的 80%，任何效率折损（事务不理想、L2 抖动）都会让实测掉到 6–8 ms 而不是 4.2 ms。

### 12.5 静态校验与运行边界

本机（macOS，无 NVIDIA GPU、无 `nvcc`）**只能做静态校验**：

```text
已完成：代码结构、下标推导、寄存器/smem 预算、字节与 FLOP 账目、边界路径
未完成：编译、spill 检查、实测耗时、ncu 指标、与 CPU 参考实现的数值比对
```

在有 T4 的机器上按顺序做三步：

```bash
# 1. 编译 + 看寄存器和 spill
nvcc -O3 -arch=sm_75 -Xptxas -v moe_expert_dw.cu -c

# 2. 与参考实现比对（建议用 double 在 CPU 或 GPU 上算 dW 作为 ground truth）
#    重点覆盖：G 组全空、len=0、K/N 非 128 倍数、N 非 4 倍数

# 3. profile，对照第 10 节的表
ncu --set full --kernel-name moe_dw_tiled ./bench
#    先看：DRAM Throughput、SM Busy、shared bank conflict、spill stores、波尾
```

**一个必须实测确认的怀疑点**：`sX[cur][t][kk0]` 的 `LDS.128` 在 §6.4 分析中有 2 路 bank conflict，
但**它是分析结论，不是实测结论**——如果 `ncu` 报的 `shared load bank conflict` 接近 0，
说明编译器做了我没预料到的重排，此时不要为了「消除冲突」去改代码。

---

## 13. 速查：三步压缩版

熟练之后可以把十阶段压成三行：

```text
1. 分类 → 定主矛盾（带宽 / 计算 / 归约）
2. Roofline 算下限 → 知道当前实现差几倍 → 判断分块值不值得
3. 下标三分法 → 独立维并行、归约维定层级（默认线程内）
   → 公式反推 tile → 寄存器反推 micro tile → 双缓冲
   → 填零统一边界 → 排序治尾部 → 清单保正确 → ncu 定位
```

以及一句总纲：**③④⑤ 是联合求解，不是三步串联。**

- ③ 定**归约维在哪一层**（决定并行结构上限）
- ④ 由 roofline 定 **tile 多大**（决定 grid 有多少并行单元）
- ⑤ 由寄存器预算反推**每线程算多少**（决定 block 内怎么映射）

先按 ③ 定结构 → 按 ④ 算一个可行 tile → 按 ⑤ 验证寄存器装不装得下；
装不下就退回 ④ 减 tile，或退回 ③ 改归约层级。**唯一不能压缩的是 ①② 必须最先做。**

---

## 14. 自检

**Q1.** 某算子输出 4 MB、每个输出要归约 10⁴ 个元素、算术强度 5 FLOP/B。该不该 split-K？

<details><summary>答案</summary>

先算访存：跨 block 归约每份要**多写一遍 4 MB**（atomic 或中间 buffer）。设 split-K 为 S 份，
额外写 `4 MB × S`，换来每 block 归约长度 `10⁴/S`。当 S=8 时额外 32 MB 写 ≈ 0.1 ms（320 GB/s）；
而 AI = 5 FLOP/B，远低于平衡点 25.3 → **这本来是带宽受限的算子，加写量雪上加霜**。
所以**除非归约长度导致并行度严重不足**（例如只有几十个 block），否则不 split。
真正的判据是「并行单元数够不够」，不是「归约长不长」。

</details>

**Q2.** 为什么 MoE 这题的 `g` 是分组维而不是独立维？如果把它当独立维写，会出什么错？

<details><summary>答案</summary>

判据：改 `g` 会**换掉整段参与求和的 token 范围** `[s_g, e_g)`，而改 `k/n` 不会。
若当独立维处理（忽略分段、对全部 `M_total` 行做一次归约），得到的是**一个 K×N 矩阵**而不是 G 个，
输出的结构就错了；即便只算一个专家，也会把其他专家的 token 算进来，数值全错。
（注意：均匀分组维如 batched GEMM 可以压平，因为地址闭式、长度相等。）

</details>

**Q3.** `__launch_bounds__(256, 2)` 在 T4 上把每线程寄存器压到多少？如果出现 `spill stores`，为什么「减少累加器」比「去掉第二个参数」更常见？

<details><summary>答案</summary>

`65536 / (256 × 2) = 128` reg/thread。
出现 spill 说明 128 不够。去掉第二个参数会让编译器放松到 168+，此时只能驻留 **1 个 block = 8 warp**，
`LDS` 的 20–30 cycle 延迟根本藏不住，等于用占用率换寄存器，通常净亏；
而把 `MICRO_N` 8→4 只把计算:访存比从 4:1 降到 2:1（仍不算瓶颈），却能把累计器从 64 降到 32，
**代价更小**。所以常规做法是先降 micro tile。

</details>

**Q4.** 双缓冲为什么每 stage 只需要一次 `__syncthreads()`？如果写回时直接写 `dW` 而不经过 smem，还能省掉哪类同步？

<details><summary>答案</summary>

因为「算 `cur`」和「写 `nxt`」作用于**两个不同 buffer**，互不冲突；而 `nxt` 的上一轮读取已由上一轮结尾的
sync 完成隔离。若写回直接打到 global（本题就是），写回阶段根本不需要 sync——sync 只用于保护 smem 的生产者/消费者关系。

</details>

**Q5.** 用真实数字说明：为什么这题「写 134 MB 的 dW」不是瓶颈，而「分块不足带来的重复读」才是？

<details><summary>答案</summary>

写：134 MB / 320 GB/s = **0.42 ms**，只占计算下限 4.24 ms 的 10% —— 写不是瓶颈。
分块不足（64×64 tile）：X 读 8 遍 1072 MB + dY 读 16 遍 1072 MB = 2.14 GB → **6.7 ms**，直接超过计算下限。
128×128 把它压到 1.2 GB → 3.77 ms，才回到计算受限。
差别的来源是**重复读的倍数由 tile 尺寸决定**，而写的量由问题规模决定、与 tile 无关。

</details>

---

## 关联

- 完整可运行代码：[code/cuda/gelu_erf.cu](code/cuda/gelu_erf.cu)、[code/cuda/moe_expert_dw.cu](code/cuda/moe_expert_dw.cu)
- GPU layout、Tensor Core、pipeline 细节：[../modern-gpu-programming-for-mlsys/](../modern-gpu-programming-for-mlsys/)
- Grouped GEMM 与 MoE dispatch 数据流：[../DeepGEMM/05-k-grouped-gemm-dataflow.md](../DeepGEMM/05-k-grouped-gemm-dataflow.md)、[../DeepGEMM/06-k-grouped-scheduler-cta-mapping.md](../DeepGEMM/06-k-grouped-scheduler-cta-mapping.md)
- Triton 版的同类分块与 K-loop：[02-triton-matmul-blocked-kernel.md](02-triton-matmul-blocked-kernel.md)
- 异步拷贝与流水（sm_80+ 对照）：[08-triton-gluon-async-tma-and-pipelining.md](08-triton-gluon-async-tma-and-pipelining.md)
