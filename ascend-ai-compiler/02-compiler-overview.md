# 02 · AI 编译器概论（Graph vs Kernel）

> 原课：「AI 编译器概论」+ 大纲「1.1 框架图」。  
> 读法与 [01](./01-cpp-foundations.md) 相同：每个块按 **为什么 → 怎样工作 → 效果 → 用在哪 / 怎么连**。  
> 缩写对照：[glossary.md](./glossary.md)。

---

## 0. 本讲要串起的故事

你已经有一个训好的模型文件。芯片却只认「某地址上的指令与数据」。中间缺的不是又一个训练框架，而是 **编译器**：

```text
框架模型（算什么）
    → Frontend：统一 IR（Intermediate Representation，中间表示）+ 与硬件无关的图优化
    → Backend：绑硬件的调度 / 内存 / kernel / codegen
    → 目标芯片上可执行
```

昇腾把后半段收成 **ATC（Ascend Tensor Compiler，昇腾张量编译器）/ GE（Graph Engine，图引擎） → `.om`（离线模型）→ AscendCL（运行时 API）**；道理与 TensorRT、tpu-mlir 同一张骨架。

---

## 1. 为什么需要 AI 编译器

### 1.1 问题


| 训练侧给你的                   | 芯片要的                    |
| ------------------------ | ----------------------- |
| Conv、MatMul、Relu 的图 + 权重 | 具体 ISA、地址、DMA、哪块 Cube 算 |
| 可能来自 TF / PyTorch / ONNX | 一种（或少数）可加载格式            |
| 关心数值正确                   | 还关心延迟、带宽、片上内存           |


若没有编译器：每来一个新前端 × 新 NPU，都要从头手写「映射 + 优化 + 代码生成」→ 大纲说的 **IR 碎片化**。

### 1.2 编译器实际在做的四件事

1. **统一**：多前端 → 一种（或少数）中间图
2. **图优化**：融合、DCE（Dead Code Elimination，死码消除）、代数化简…（少算、少搬）  
3. **硬件映射**：选 kernel、切分、分配内存与流  
4. **产出**：AOT（Ahead-Of-Time，提前/离线编译）文件或 JIT（Just-In-Time，即时编译）缓存码  



### 1.3 效果与用在哪

效果：同一套优化可复用；换芯片时主要换 Backend。  
用在哪：边缘部署、云推理、芯片原厂工具链——本课全程。

---



## 2. 框架图详解（原大纲 1.1）



### 2.1 总图

```text
  模型输入层：TensorFlow / PyTorch / ONNX
              │
              ▼
┌── Compiler Frontend ────────────────┐
│  High-level Graph IR                │
│  Algebraic simplification           │
│  Operator fusion · DCE              │
│  → Optimized Computation Graph      │
└─────────────────┬───────────────────┘
                  │
                  ▼
┌── Compiler Backend ─────────────────┐
│  Memory allocation                  │
│  Memory latency hiding              │
│  Loop-oriented opt · Parallelization│
│  Auto-tuning · Kernel libraries     │
│  Low-level IR & Code generation     │
└─────────────────┬───────────────────┘
                  │
                  ▼
         CPU / GPU / ASIC(NPU/TPU)
```



### 2.2 模型输入层——为什么单独画

**为什么：** 业务不会只活在一种框架里。  
**怎样：** Parser 读图与权重，映射成编译器认识的 Op。  
**效果：** 后面优化与 Backend 不用关心「这是 TF 还是 ONNX」。  
**用在哪：** ATC 的 `--framework`、onnx 导出；见 [08](./08-frontend-parser.md)。

### 2.3 Frontend 与 High-level IR

**为什么要硬件无关 IR：**  
若一上来就按某 NPU 的 buffer 名写死，换代芯片或换 GPU 时整图作废；也难做「Conv+BN+Relu → 一个 fused op」这类通吃优化。

**怎样工作：**


| 优化名（大纲）                  | 在干什么                  | 直觉效果        |
| ------------------------ | --------------------- | ----------- |
| Algebraic simplification | `x+0→x`、`x*1→x`、强度削弱等 | 少算          |
| Operator fusion          | 相邻 op 合成一个            | 少写回内存、少启动开销 |
| DCE（死码消除 / 无效计算剔除）       | 删没有使用者的计算             | 少算          |


**效果：** 输出 Optimized Graph，仍偏「算什么」，尚未钉死「在哪个 Core 的哪块 UB（Unified Buffer，片上统一缓冲）」。  
**用在哪：** MLIR 的 canonicalize（规范化）/ CSE（Common Subexpression Elimination，公共子表达式消除）；GE 的图准备/原图优化。

### 2.4 Backend 各子项（逐项四问缩写）

**Memory allocation**  
为什么：片上小、片外慢，必须规划 tensor 地址与复用。  
效果：减少 DDR 往返。→ [10](./10-memory-schedule-aoe.md)

**Memory latency hiding**  
为什么：DMA 有延迟；计算与搬运重叠才能藏延迟。  
效果：流水更高。→ [05](./05-ascend-npu-hardware.md)

**Loop-oriented opt / Parallelization**  
为什么：大张量要切块、多核并行。  
效果：吃满 Cube/多核。偏 kernel 与调度交界。

**Auto-tuning**  
为什么：手工搜 tile/切分组合爆炸。  
效果：用搜索+实测找较优策略。昇腾 **AOE**（自动调优；含 SGAT 子图调优 / OPAT 算子调优，以官方文档名为准）。

**Kernel libraries**  
为什么：成熟算子（Conv/GEMM）有手写极致实现。  
效果：编译器「认出来就调库」而不是每次从零 codegen。cuDNN / DNNL / 昇腾算子库。

**Low-level IR & Codegen**  
为什么：最终要落到指令或可加载二进制。  
效果：`.om`、`.so`、可执行任务描述等。

### 2.5 昇腾落点对照


| 框架图       | 昇腾                       |
| --------- | ------------------------ |
| 输入层       | ONNX/TF → Parser         |
| Frontend  | 图准备、原图优化、Infershape      |
| Backend   | 拆分、子图/整图优化、内存与 stream、编译 |
| 产物        | `.om`                    |
| 运行        | **AscendCL** / msame     |
| Auto-tune | **AOE**                  |


---



## 3. Graph 级 vs Kernel 级



### 3.1 为什么要区分

招聘与排错时常混：「慢了是该融图层，还是该改 CUDA/AscendC？」  
两层优化目标不同，工具也不同。

### 3.2 怎样理解


|     | Graph                | Kernel            |
| --- | -------------------- | ----------------- |
| 视角  | 整栋楼户型                | 单间施工图             |
| 例子  | Conv+BN+Relu 融合；删死支路 | 循环分块、流水、向量化       |
| 谁做  | Pass / GE（图引擎）       | 手写 kernel 或自动调优生成 |
| 输入  | IR 图                 | 已选定的一个（融合）算子      |




### 3.3 效果与用在哪

- 图优做得好 → kernel 再强也少干活  
- 图已最优仍慢 → 才下钻 kernel / AOE op 调优

本课 Lab1 练 Graph；Profiling 后可能走到 Kernel/AOE。

---



## 4. AOT vs JIT



### 4.1 为什么出现两种


| 需求                | 更合适     |
| ----------------- | ------- |
| 车规/安防：上线延迟稳定、可预分析 | **AOT** |
| 动态 shape、快速试验     | JIT 或混合 |




### 4.2 怎样工作

- **AOT：** 部署前编译 → 固化文件（昇腾 `.om`）→ 运行只加载执行  
- **JIT：** 运行中按 shape/设备编译 → 常有缓存



### 4.3 效果

AOT：首帧可控、可做严肃性能/精度门禁；改模型要重编译。  
JIT：灵活；冷启动可能卡。

### 4.4 用在哪

本课 Lab2–6 主路径 **ATC AOT**。动态 batch/分辨率是 AOT 里的「多档编译」，不是否定 AOT。

---



## 5. 主流方案坐标系（为什么提它们）

不是要你都会用，而是面试/读论文时有锚点：


| 方案          | 一句话             | 和本课             |
| ----------- | --------------- | --------------- |
| TVM         | 自动调优 + 多后端      | 对照 Auto-tune 思想 |
| XLA         | 融合强，生态偏谷歌       | 对照图优化           |
| TensorRT    | 极致，绑 NVIDIA     | 对照「厂商闭源图编译器」    |
| tpu-mlir    | MLIR Dialect 路线 | **原课**          |
| CANN/ATC/GE | 昇腾官方            | **本课**          |


共同骨架仍是 §2 那张图。

---



## 6. 串到后面课程

```text
02 建立地图
 → 03/04 用 MLIR 把 Frontend 优化「白盒化」练一遍（Lab1）
 → 05/06 懂芯片与环境，才懂 Backend 为什么那些优化
 → 07–10 在昇腾上走通同一张图
 → 11–13 量化与 profiling = 精度/性能工程化
 → 14–17 真实 CV/LLM 业务
```

---



## 7. 练习

1. 不看讲义，默画框架图四层，并写出昇腾对应名词。
2. 各举一例：Algebraic simplification / Fusion / DCE。
3. 延迟差，你先查「图是否没融」还是「单算子 kernel」？为什么？
4. 为何边缘更爱 AOT？动态分辨率时 AOT 怎么仍能用？

下一讲：[03](./03-mlir-dialect-op-type.md) · 上一讲：[01](./01-cpp-foundations.md)