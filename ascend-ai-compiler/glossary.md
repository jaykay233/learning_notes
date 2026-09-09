# 术语英中对照（昇腾 AI 编译器课）

> 缩写首次难记时查本表。讲义正文里也会尽量在**首次出现**写「英文全称（中文）」。

## 编译器通用

| 缩写 / 词 | 英文 | 中文 |
|---|---|---|
| **IR** | Intermediate Representation | 中间表示 |
| **Pass** | Pass（compiler pass） | 编译器一遍变换/分析插件 |
| **Frontend** | Frontend | 编译器前端（导入 + 高阶图优化） |
| **Backend** | Backend | 编译器后端（硬件相关优化与代码生成） |
| **Graph IR** | Graph Intermediate Representation | 图级中间表示 |
| **DCE** | Dead Code Elimination | 死码消除 / 无效计算剔除 |
| **CSE** | Common Subexpression Elimination | 公共子表达式消除 |
| **Canonicalize** | Canonicalization | 规范化（把 IR 收到标准形态） |
| **Fusion** | Operator fusion | 算子融合 |
| **Algebraic simplification** | Algebraic simplification | 代数化简 |
| **AOT** | Ahead-Of-Time (compilation) | 提前编译 / 离线编译 |
| **JIT** | Just-In-Time (compilation) | 即时编译 |
| **Kernel** | Kernel | 算子在硬件上的具体实现例程 |
| **Codegen** | Code generation | 代码生成 |
| **SSA** | Static Single Assignment | 静态单赋值（每个值名只赋值一次） |
| **Lowering** | Lowering | 降级（从高抽象 IR 降到低抽象 IR） |
| **DMA** | Direct Memory Access | 直接存储器访问（数据搬运） |
| **NPU** | Neural Processing Unit | 神经网络处理器 |
| **TPU** | Tensor Processing Unit | 张量处理器（谷歌等；大纲里与 ASIC 并列） |
| **ASIC** | Application-Specific Integrated Circuit | 专用集成电路 |
| **ISA** | Instruction Set Architecture | 指令集架构 |

## MLIR 相关

| 缩写 / 词 | 英文 | 中文 |
|---|---|---|
| **MLIR** | Multi-Level Intermediate Representation | 多层级中间表示（造编译器的基础设施） |
| **Dialect** | Dialect | 方言（一组 Op/Type 的命名空间） |
| **Op** | Operation | 运算 / 操作
| **Type** | Type | 类型 |
| **Attribute** | Attribute | 属性（编译期常量标注） |
| **Pattern / RewritePattern** | Rewrite pattern | 重写模式（匹配子图并替换） |
| **GreedyPatternRewrite** | Greedy pattern rewrite | 贪婪式反复套用 Pattern 直到收敛 |
| **PM** | Pass Manager | Pass 管理器 / 管道调度器 |
| **TOSA** | Tensor Operator Set Architecture | 张量算子集架构（标准中层张量算子；方言 `tosa`） |
| **SCF** | Structured Control Flow | 结构化控制流（方言 `scf`，如 `scf.for` / `scf.if`） |
| **Linalg** | Linear Algebra dialect（惯用名） | 线性代数/循环级中层方言之一 |
| **TableGen** | TableGen | LLVM/MLIR 的声明式代码生成（`.td` 文件） |
| **RTTI** | Run-Time Type Information | 运行时类型信息（课里指 `isa`/`cast`/`dyn_cast`） |
| **CRTP** | Curiously Recurring Template Pattern | 奇异递归模板模式（静态多态） |
| **Visitor** | Visitor pattern | 访问者模式（遍历与处理分离） |

## 昇腾 / CANN

| 缩写 / 词 | 英文 | 中文 |
|---|---|---|
| **CANN** | Compute Architecture for Neural Networks | 昇腾异构计算架构（工具与运行时全家桶） |
| **ATC** | Ascend Tensor Compiler | 昇腾张量编译器（模型 → `.om`） |
| **GE** | Graph Engine | 图引擎（ATC 内图准备/拆分/优化/编译的核心） |
| **AOE** | Ascend Optimization Engine（课内常用说法；以官方文档名为准） | 昇腾自动调优引擎（子图/算子调优） |
| **SGAT** | SubGraph Auto Tune | 子图自动调优 |
| **OPAT** | Operator Auto Tune | 算子自动调优 |
| **om** | Offline Model（`.om` 文件） | 离线模型（AOT 产物） |
| **AscendCL** | Ascend Computing Language | 昇腾计算语言/运行时 API（加载执行 om） |
| **msame** | （工具名） | 官方常用的 om 快速推理/验证工具 |
| **UB** | Unified Buffer | 统一缓冲（AI Core 片上缓冲；代际细节以文档为准） |
| **AI Core** | AI Core | 昇腾主计算核（含 Cube/Vector 等） |
| **AI CPU** | AI CPU | 处理部分不适合 AI Core 的算子/控制 |
| **soc_version** | SoC version | 芯片型号标识（ATC 必填类参数） |
| **PTQ** | Post-Training Quantization | 训练后量化 |
| **QAT** | Quantization-Aware Training | 量化感知训练 |

## 量化

| 缩写 / 词 | 英文 | 中文 |
|---|---|---|
| **INT8** | 8-bit integer | 8 位整数量化 |
| **W8A8** | Weights 8-bit, Activations 8-bit | 权重 8 位、激活 8 位 |
| **W4A16** | Weights 4-bit, Activations 16-bit | 权重 4 位、激活 16 位 |
| **scale / zp** | scale / zero-point | 缩放因子 / 零点 |
| **Per-tensor** | Per-tensor quantization | 整张量共用一套量化参数 |
| **Per-channel** | Per-channel quantization | 按通道各自一套参数 |
| **KL** | Kullback–Leibler divergence（校准） | KL 散度（一种校准选阈值方法） |
| **MinMax** | Min-max calibration | 用最小最大值定范围 |
| **Percentile** | Percentile calibration | 用分位数丢掉极端值再定范围 |

## LLM / 部署

| 缩写 / 词 | 英文 | 中文 |
|---|---|---|
| **MHA** | Multi-Head Attention | 多头注意力 |
| **GQA** | Grouped-Query Attention | 分组查询注意力 |
| **MQA** | Multi-Query Attention | 多查询注意力 |
| **KV Cache** | Key-Value Cache | 键值缓存 |
| **Prefill** | Prefill | 预填充（一次处理提示词） |
| **Decode** | Decode | 解码（逐步生成 token） |
| **TTFT** | Time To First Token | 首 token 时延 |
| **NMS** | Non-Maximum Suppression | 非极大值抑制（检测后处理） |
| **FPS** | Frames Per Second | 帧率 |

## 原课算能对照（只为读旧大纲）

| 原课用语 | 英文线索 | 本课昇腾对应 |
|---|---|---|
| TopDialect | Top-level dialect | Parser 后高阶中间图 |
| TpuDialect | TPU/NPU dialect | GE 硬件侧 / om |
| BModel | Binary model | `.om` |
| LayerGroup | Layer grouping | 内存编排 / 子图调度 / AOE |
| tpu-mlir | TPU MLIR toolchain | CANN + ATC + GE |
