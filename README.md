# learning_notes

AI 编译器、GPU 编程、推理系统与模型架构的学习笔记，按主题分目录：

| 目录 | 侧重 |
|---|---|
| [`models/`](models/) | 模型架构（DeepSeek / GLM / Kimi / Qwen …） |
| [`ascend-ai-compiler/`](ascend-ai-compiler/) | 昇腾版 AI 编译器 / 部署讲义（CANN、ATC、GE、om） |
| [`ascendc/`](ascendc/) | AscendC 算子：**CPU 孪生**环境 + `add_custom`（macOS 用 Colima） |
| [`mlc-tvm/`](mlc-tvm/) | MLC / TVM：TensorIR、Schedule、端到端模型与 tensorization |
| [`modern-gpu-programming-for-mlsys/`](modern-gpu-programming-for-mlsys/) | GPU layout、Tensor Core、pipeline、WGMMA 与 Blackwell TMEM |
| [`cuda-graph/`](cuda-graph/) | 推理 CUDA Graph、memory-saver、SGLang vs Inductor Trees |
| [`communication/`](communication/) | GPU 通信提交、Proxy / GDAKI / GPI、MoK 调度，以及低延迟 collective |
| [`speculative-decoding/`](speculative-decoding/) | 投机解码收益（GPU vs LPU） |
| [`torch_compile/`](torch_compile/) | `torch.compile` / Dynamo / FX / AOTAutograd / Dispatcher |
| [`quantization/`](quantization/) | LLM 低比特量化、W4A8KV4、QoQ、KV4 Attention 与 SmoothAttention |

## 目录总览

```
ascend-ai-compiler/          # 原「AI 编译器」课 → 昇腾 CANN 全量改写
├── 00-syllabus-checklist.md
├── glossary.md
├── 01 … 18
├── labs/lab1…lab6
└── codes/lab1/              # Lab1：inputs / expected / Pass / MLIR 工具
ascendc/                     # AscendC CPU 孪生（Colima + CANN 9.x）
├── README.md
├── scripts/setup_linux_cpu.sh
├── docker/Dockerfile
└── examples/add_custom/     # vector add；run.sh -r cpu -v Ascend910B1
mlc-tvm/                     # MLC / TVM 课程笔记
├── README.md
├── 01-tensor-program-abstraction.md
├── 02-tensorir-case-study.md
└── 03-end-to-end-model.md
modern-gpu-programming-for-mlsys/  # MLSys GPU 编程课程笔记
├── README.md
├── 01-data-layout-and-named-axes.md
├── 02-replication-and-offset.md
├── 03-practice-and-corrections.md
├── 04-swizzle-layout.md
├── 05-ampere-mma-fragments.md
├── 06-warp-tile-and-k-pipeline.md
├── 07-hopper-wgmma-and-blackwell-tmem.md
├── 08-tma-tile-copy-and-synchronization.md
├── 08-tma-practice-and-solutions.md
├── 09-tensor-cores-tcgen05.md
├── 10-tmem-allocation-lifecycle.md
├── 11-mbarrier-phase-lifecycle.md
├── 12-clc-dynamic-scheduling.md
├── 13-tirx-first-kernel.md
├── 14-tirx-layout-api.md
├── 15-gemm-basics.md
├── 16-gemm-async-tma.md
├── 17-gemm-software-pipeline.md
├── 18-gemm-persistent-kernel.md
└── 19-gemm-warp-specialization.md
quantization/                # LLM 低比特量化与推理系统
└── 01-qoq-w4a8kv4.md       # QoQ / QServe：W4A8KV4、重排与 SmoothAttention
cuda-graph/                  # CUDA Graph 基础、SGLang 与 VMM
communication/               # Proxy、GDAKI/GPI、MoK、collective
speculative-decoding/        # 投机解码
torch_compile/               # Dynamo、AOTAutograd、Dispatcher
models/                      # DeepSeek / GLM / Kimi / Qwen 架构笔记
```

完整昇腾讲义目录与学习顺序见 [ascend-ai-compiler/README.md](ascend-ai-compiler/README.md)。

```
cuda-graph/
├── 01-basics-and-memory-saver.md              # Graph 是什么；region / pause / resume
├── 02-sglang-cudagraph-vs-inductor-trees.md   # 录图粒度；CudaGraphRunner vs CUDAGraph Trees
└── 03-flavors-vmm-and-hijack.md               # 几种外壳对照；VMM；劫持换分配器
```

```
communication/
├── 01-proxy-gdaki-gpi.md                    # CPU Proxy / GDAKI / GPI；QP、WQE、提交路径
├── 02-mok-scheduling-and-buffers.md          # MoK：device schedule、pull/push、ring token buffer
└── 03-low-latency-collectives-and-synchronization.md  # Sentinel / credit / Multicast / LL128 / multimem
```

```
speculative-decoding/
└── 01-gpu-vs-lpu-sram.md   # verify(K)≪K×decode(1)；大 SRAM/低算力为何吃不满收益
```

```
torch_compile/
├── 01-dynamo-and-fx.md           # Dynamo 调用过程；FX；Script/LazyTensor 对照
├── 02-aot-autograd.md            # AOTAutograd：joint 图、Proxy 四步追踪、反向捕获、partition
└── 03-dispatcher-and-modes.md    # DispatchKey / Mode / TLS / 拦截边界 / AOT 衔接
```

```
models/
├── deepseek_v4/
│   ├── architecture.md   # DeepSeek-V4：MQA / CSA / HCA、MoE、mHC
│   ├── mhc.md            # mHC（流形约束超连接）详解与伪代码
│   └── mqa.md            # MQA / Attention Layer 详解
├── glm-5.3/
│   ├── architecture.md   # GLM-5.3 text-only：MLA + DSA + MoE
│   ├── mla.md            # MLA latent attention
│   └── dsa.md            # DSA 动态稀疏注意力
├── glm-5.3-flash/
│   ├── architecture.md            # GLM-5.3-Flash / Glm5Next：KDA / MLA / mHC
│   └── radix_linear_attention.md  # RadixLinearAttention / KDA gated delta
├── kimi-k3/
│   └── architecture.md   # Kimi-K3：MLA / KDA / MoE / AttnRes
├── qwen-3.8/
│   └── architecture.md   # Qwen3.8-2.4T-A95B：GDN + Gated GQA + MoE（无 QSA）
└── qwen3.8-flash-next/
    ├── architecture.md   # Qwen3.8-Flash-Next 总览（Qwen4 预览）
    ├── qsa.md            # Qwen Sparse Attention（c4 indexer → 稀疏 GQA）
    └── ple.md            # N-gram / PLE（hash 查表、门控注入 HC）
```

## 昇腾 AI 编译器讲义

原课大纲 **全部条目** 均有对应课件（算能工具名 → ATC/GE/om）。入口与校对表：

- [ascend-ai-compiler/README.md](ascend-ai-compiler/README.md)
- [00-syllabus-checklist.md](ascend-ai-compiler/00-syllabus-checklist.md)
- [glossary.md](ascend-ai-compiler/glossary.md)（缩写英中对照）

| 阶段 | 内容 |
|---|---|
| 基础 | C++ 基石、编译器概论、MLIR Dialect/Pass、Lab1 |
| 硬件环境 | 昇腾存储层次、CANN 环境、Lab2 ResNet50 |
| CANN | 工作流、Parser、GE、AOE/内存调度、Lab3 ONNX→om |
| 量化性能 | INT8、校准、Lab4、Profiling |
| CV / LLM | YOLO+Stream、Attention、KV、Qwen、Lab5/6 |
| 求职 | 岗位与面试题方向 |

## AscendC（CPU 孪生）

与讲义里的 CANN 部署互补：这里练 **自定义算子** 的 CPU 域调试（无 NPU 也可）。macOS 走 **Colima headless Docker**，容器内装 CANN Toolkit。

入口：[ascendc/README.md](ascendc/README.md)

| 项 | 说明 |
|---|---|
| 环境 | Colima + `ascendc-cpu-dev`；`source /usr/local/Ascend/cann/set_env.sh`（CANN 9.x） |
| 示例 | [`examples/add_custom`](ascendc/examples/add_custom/)：`bash run.sh -r cpu -v Ascend910B1` |
| 已验证 | aarch64 **9.2.0-beta.1** Toolkit + 910b-ops；CPU twin `max_abs_diff=0.0` |

注意：Toolkit **不能**在 macOS 原生安装；`.run` 放 `ascendc/docker/cann-packages/`（不入库）。

## 模型对照（速查）

| 目录 | 模型 | 主干注意力 | 残差 / 额外容量 | 备注 |
|---|---|---|---|---|
| `deepseek_v4/` | DeepSeek-V4 | MQA + CSA/HCA | **mHC** | 压缩稀疏检索 |
| `glm-5.3/` | GLM-5.3 text | **MLA + DSA** | 单流 | 与 Flash 不是同一套 |
| `glm-5.3-flash/` | GLM-5.3-Flash | **KDA + MLA**（hybrid） | **mHC** | `Glm5Next*` |
| `kimi-k3/` | Kimi-K3 | **KDA + MLA** | **AttnRes** | 对称 KDA 等 |
| `qwen-3.8/` | Qwen3.8-2.4T-A95B | **GDN + Gated GQA** | 单流 | **无 QSA / 无 PLE** |
| `qwen3.8-flash-next/` | Qwen3.8-Flash-Next | **GDN + QSA** | **HC 4 路 + PLE ~51B** | `qwen4_exp`；≠ 旧 Qwen3-Next-80B |

易混：

- **Qwen3.8** ≠ Qwen3-**8B** dense；本仓是 **2.4T-A95B** MoE。
- **Qwen3.8-Flash-Next** ≠ 旧 `Qwen3-Next-80B-A3B`；是 Qwen4 架构预览。
- **GLM-5.3**（MLA+DSA text）≠ **GLM-5.3-Flash**（KDA hybrid + mHC）。

## 专题索引

按机制跨模型跳转：

| 主题 | 文档 |
|---|---|
| CUDA Graph / memory-saver / SGLang Runner | [01](cuda-graph/01-basics-and-memory-saver.md)、[02](cuda-graph/02-sglang-cudagraph-vs-inductor-trees.md)、[03 · 几种外壳与 VMM](cuda-graph/03-flavors-vmm-and-hijack.md) |
| GPU 通信提交 / Proxy / GDAKI / GPI | [communication/01](communication/01-proxy-gdaki-gpi.md) |
| MoE 训练 / MoK / pull-push / buffer | [communication/02](communication/02-mok-scheduling-and-buffers.md) |
| 小消息 collective / Sentinel / credit / SHARP / LL128 | [communication/03](communication/03-low-latency-collectives-and-synchronization.md) |
| 投机解码（GPU vs LPU） | [speculative-decoding/01](speculative-decoding/01-gpu-vs-lpu-sram.md) |
| AscendC CPU 孪生 / Colima | [ascendc/README.md](ascendc/README.md)、[add_custom](ascendc/examples/add_custom/) |
| MLC / TVM / TensorIR / Tensorization | [mlc-tvm/README.md](mlc-tvm/README.md) |
| GPU layout / Tensor Core / WGMMA / TMEM | [modern-gpu-programming-for-mlsys/README.md](modern-gpu-programming-for-mlsys/README.md) |
| LLM 量化 / W4A8KV4 / QoQ / KV4 / SmoothAttention | [quantization/01-qoq-w4a8kv4.md](quantization/01-qoq-w4a8kv4.md) |
| torch.compile / Dynamo / FX / AOTAutograd / Dispatcher | [01](torch_compile/01-dynamo-and-fx.md)、[02](torch_compile/02-aot-autograd.md)、[03 · Dispatcher/Mode](torch_compile/03-dispatcher-and-modes.md) |
| MLA | [glm-5.3/mla.md](models/glm-5.3/mla.md)、[glm-5.3-flash/architecture.md](models/glm-5.3-flash/architecture.md)、[kimi-k3/architecture.md](models/kimi-k3/architecture.md) |
| DSA（token/KPool 稀疏） | [glm-5.3/dsa.md](models/glm-5.3/dsa.md) |
| QSA（block 粗选 → token 展开） | [qwen3.8-flash-next/qsa.md](models/qwen3.8-flash-next/qsa.md) |
| Linear / gated delta（KDA / GDN / Radix） | [glm-5.3-flash/radix_linear_attention.md](models/glm-5.3-flash/radix_linear_attention.md)、[qwen-3.8/architecture.md](models/qwen-3.8/architecture.md)、[kimi-k3/architecture.md](models/kimi-k3/architecture.md) |
| mHC / HC / Gated Residual | [deepseek_v4/mhc.md](models/deepseek_v4/mhc.md)、[qwen3.8-flash-next/architecture.md](models/qwen3.8-flash-next/architecture.md)（HC；无 Sinkhorn） |
| AttnRes | [kimi-k3/architecture.md](models/kimi-k3/architecture.md) |
| PLE / N-gram embedding | [qwen3.8-flash-next/ple.md](models/qwen3.8-flash-next/ple.md) |
| MQA / CSA / HCA | [deepseek_v4/mqa.md](models/deepseek_v4/mqa.md)、[deepseek_v4/architecture.md](models/deepseek_v4/architecture.md) |

---

## MLC / TVM

围绕 MLC 课程 notebook 与 TVMScript / TensorIR 调度展开。当前已落盘：

| 文档 | 内容 |
|---|---|
| [01-tensor-program-abstraction.md](mlc-tvm/01-tensor-program-abstraction.md) | Tensor Expression、IRModule、Schedule、build/run |
| [02-tensorir-case-study.md](mlc-tvm/02-tensorir-case-study.md) | Buffer、循环、sblock、axis、函数属性与变换 |
| [03-end-to-end-model.md](mlc-tvm/03-end-to-end-model.md) | Relax 计算图、`call_tir`、DPS 与端到端执行 |

完整学习环境、API 差异和后续计划见 [mlc-tvm/README.md](mlc-tvm/README.md)。

## Modern GPU Programming for MLSys

围绕 GPU 数据布局、Tensor Core 数据路径与推理 kernel 展开。当前已落盘 01-19：

```text
layout / named axes / replication / offset
shared memory swizzle
Ampere mma.sync / ldmatrix
warp tile / K pipeline / cp.async
Hopper WGMMA / Blackwell TMEM
tcgen05 / SFA/SFB / scale_vec
TMA tile copy / swizzle / pipeline / descriptor / mbarrier / bulk group
Blackwell tcgen05.mma / TMEM / commit + mbarrier / cta_group
Warp specialization / TMA producer / MMA consumer / writeback
tma2mma、mma2tma、mma2ld、ld2mma 四条 barrier 交接
cta_group::2 的 CTA pair 资源访问边界
cta_group::1, M=128 的直接 accumulator 映射
cta_group::1, M=64 的 Layout F
cta_group::2, M=256 的 CTA pair accumulator 切分
cta_group::2, M=128 dense A 的 Layout B
block-scaled MMA 的 SFA/SFB 跨 CTA pair 放置
tcgen05 指令之间的 scope / layout / completion 三层契约
chapter_tensor_cores 完成
TMEM allocation / deallocation 生命周期
tcgen05.alloc 的 warp-collective 语义
TMEM 地址可见性、allocation size 限制与 cta_group::2 契约
TMEM warpgroup Lane 访问窗口与 CTA allocation 边界
TMEM tcgen05.ld/st、shape/num、pack/unpack 与异步等待
chapter_tmem 完成
mbarrier 的 arrival、pending count 与 phase parity
phase 完成后进入下一轮，consumer 等待当前 round 的 parity
双 stage pipeline 中 stage barrier 与 phase_tma 的关系
threads 写 SMEM 后通过 fence.proxy.async 交接给 TMA
generic proxy 与 async proxy 的可见性边界
chapter_async_barriers 完成
静态 persistent scheduler 的 launch tail
worker 延迟启动与 tile 成本不均衡
CTA launch queue 与软件工作队列的区别
CLC 取消 pending launch 并接管 coordinate 的基本模型
clusterlaunchcontrol.try_cancel.async 的 16-byte response
CLC request 的 mbarrier arrival 与 complete-tx 完成条件
clusterlaunchcontrol.query_cancel 的 is_canceled 与 get_first_ctaid
多个 thread 提交 CLC request 时的 response 与 barrier 计数
CLC response 的 async-proxy 写入与 generic-proxy 读取
generic proxy 与 async proxy 的抽象含义
CLC response 的跨 proxy 读写顺序
mbarrier 完成通知与 proxy fence 的分工
在计算当前 tile 前提交 try_cancel
用当前 tile 计算隐藏 grid scheduler 延迟
先请求、再计算、最后等待的单请求软件流水
CLC 单 outstanding request 的 buffer / barrier 复用约束
静态 scheduler 在稳定、均匀、短 tile 场景中的优势
CLC 在 worker 启动与 tile 成本不确定时的收益
CLC 的 request / barrier / fence / query 额外开销
CLC 的 sm_100+ 硬件与编译目标边界
H100/H200 不支持 CLC，但支持 thread block cluster
把 CLC 封装为动态 tile scheduler，并保持 mainloop 不变
chapter_clc 完成
TIRx Scope / Layout / Dispatch
单 tile GEMM 的 A/B 与 D 数据路径
TIRx SMEMPool、mbarrier 与 TMEM 生命周期
chapter_intro_tirx 第一个单 tile GEMM Kernel
TIRx 编译、Executable 调用与 PyTorch 数值验证
chapter_intro_tirx 完成
TileLayout 的 S[...]、R[...] 与固定 offset
S[...] 的逻辑索引相关基础映射
R[...] 的逻辑索引无关物理副本
offset 的整体平移但不增加副本
apply() 只返回 D(x) + O
(1, 3)、shape [8, 16] 的 flatten、decompose 与坐标推导
chapter_tirx_layout_api 第一个知识点完成：S[...]、R[...] 与 offset
命名轴 laneid / warpid / m / TLane / TCol
不同 axis 是不同坐标空间，相同整数值不能直接合并
m 的语义由 buffer scope 决定，TLane / TCol 描述 TMEM 存储坐标
chapter_tirx_layout_api 第二个知识点完成：命名轴
apply() 的 logical + shape、linear、shard coordinate 三种入口
logical shape 决定 flatten，shard extents 决定 decompose
TileLayout 不保存 logical shape，同一个 (1, 3) 在不同 shape 下映射不同
单参数 apply(coord) 按 linear coordinate 处理
用三入口等价性分层定位 flatten、decompose 与 stride / axis 错误
chapter_tirx_layout_api 第三个知识点完成：apply() 输入形式
(2, 128, 112) accumulator layout 的两个 128 x 112 TMEM 区域
TLane 承载 128 个逻辑 M rows，TCol 覆盖 [0, 224)
TCol = 112 * a + col，非 2 的幂 extent 112 无需补齐
layout 只描述 TMEM 坐标，不负责 allocation、MMA 或 tcgen05.ld
chapter_tirx_layout_api 第四个知识点完成：TMEM accumulator layout
scale-factor atom 的 R[4 : 32@TLane] 在 TLane 上生成四份副本
副本偏移为 0、32、64、96，q=0 就是 base 本身
同一个 scale factor 位于四个 32-lane TMEM partition 的本地窗口
8-bit logical TCol s 打包到 hardware TCol s//4 和 byte s%4
layout.apply() 只返回 base，layout.replica 描述四份副本
chapter_tirx_layout_api 第五个知识点完成：scale-factor replication
tmem_datapath_layout 根据 datapath 返回 tcgen05.mma 的 TMEM row mapping
datapath=D 使用 rows=128，恒等映射 TLane=row
datapath=F 使用 rows=64，四个 16-row slab 分散到 Lane 0/32/64/96
F 的公式 TLane=32*(row//16)+row%16，不复制逻辑数据
F 只使用 64 条 active Lane，span TLane=112 来自最高 Lane 111
chapter_tirx_layout_api 第六个知识点完成：tmem_datapath_layout D/F
tcgen05_atom_layout 将 TMEM fragment 的搬运形状映射到线程寄存器
instr_shape 是 lane x bits 的 atom，tensor_shape 和 dtype 推导 .xN
16x128b 的 fp32 K=128 与 fp16 K=256 都使用 .x32
fp16 的两个相邻元素打包进一个 32-bit register 的低半和高半
chapter_tirx_layout_api 第七个知识点完成：tcgen05_atom_layout
wg_local_layout 把逻辑 row 映射到 tid_in_wg，把同一行的 columns 映射到局部 m
默认 rows=128 时，完整 warpgroup 的每个 thread 负责一行
warp 0/1/2/3 分别负责 rows 0..31 / 32..63 / 64..95 / 96..127
wg_local_layout(rows=128).shard 与 32x32b fp32 fragment 的 shard 相同
wg_local_layout 不校验 atom、不处理 dtype packing，也不执行 allocation 或 copy
rows=64 时只生成 tid_in_wg 0..63，剩下 64 个线程没有该 layout 的数据坐标
chapter_tirx_layout_api 第八个知识点完成：wg_local_layout
ComposeLayout 将 affine TileLayout 与 XOR swizzle 组合起来
M/B/S 是 bit count，不是 bytes
(8,64) fp16 的 128B swizzle 使用 M=B=S=3
low 不参与 XOR，保证连续 16-byte vector 保持成组
x=m>>M，source=(x>>S)&mask，x2=x^source，addr=(x2<<M)|low
(8,64)、i=5、j=0 的逐位路径为 320 -> 40 -> 5 -> 45 -> 360
同一路径的 bank 从 0 变为 20
j=0 时八行地址为 72*i，bank 为 0,4,8,...,28
无 swizzle 时八行都落 bank 0
chapter_tirx_layout_api 第九个知识点完成：ComposeLayout 与 shared-memory swizzle
GEMM 的 A(M,K) / B(N,K) / D(M,N) 约定
单 tile baseline 的 GMEM -> SMEM -> TMEM -> registers -> GMEM 数据路径
chapter_gemm_basics 第 1 个知识点完成：单 Tile Baseline
K-loop 按 BLK_K 切分 K，并用 accum=False / True 控制 TMEM 首轮覆写与后续累加
tcgen05.commit 关联异步 MMA 完成，mbarrier phase 每轮翻转
chapter_gemm_basics 第 2 个知识点完成：K-Loop 累加与 MMA barrier phase
spatial tiling 用 M/N 二维 grid 把输出切成多个 CTA-owned tiles
每个 CTA 以 m_st / n_st 选择 A/B rows 和 D columns，K-loop 与 spatial tiling 正交
chapter_gemm_basics 第 3 个知识点完成：空间 Tiling（Multi-CTA）
hgemm_v4 用单线程 TMA 替换 CTA 协作的 A/B load
mbarrier.arrive.expect_tx 同时登记 thread arrival 与 32768 transaction bytes
try_wait 等待 TMA A/B tiles 全部到达后才能启动 MMA
Step 4 仍立即等待 TMA，真正的 Load / Compute overlap 留到后续 pipeline
chapter_gemm_async 第 4 步完成：TMA Async Load
hgemm_v5 使用 PIPE_DEPTH=2 的双缓冲，并在 prologue 预取前两个 stages
stage = k % PIPE_DEPTH 控制 operand slot，k + PIPE_DEPTH 回填已释放 slot
每个 stage 有自己的 TMA barrier，phase_tma 在 ring wrap 时翻转
所有 K iterations 共用一个 TMEM accumulator 和 mma_bar，phase_mma 每轮翻转
回填前必须等待 MMA 完成，避免覆盖仍在读取的 operand stage
chapter_gemm_async 第 5 步完成：Software Pipeline（PIPE_DEPTH=2）
hgemm_v6 用固定数量的 persistent CTA 顺序处理多块 output tile
bx = T.cta_id([SM_COUNT]) 把二维 tile grid 改成一维 persistent worker grid
ClusterPersistentScheduler2D 统一生成 tile 坐标并按 l2_group_size 调整访问顺序
1024 块 output tile 分给 148 个 CTA：136 个处理 7 块，12 个处理 6 块
TMEM、SMEM allocation 和 barriers 在 CTA 生命周期内只初始化一次
K_TILES=64、PIPE_DEPTH=2 时 barrier completion 次数为偶数，块边界可安全重置 phase
chapter_gemm_async 第 6 步完成：Persistent Kernel + Tile Scheduler
```

完整文档索引、当前进度和硬件环境说明见
[modern-gpu-programming-for-mlsys/README.md](modern-gpu-programming-for-mlsys/README.md)。

---

## Quantization

围绕 LLM 低比特推理量化与系统协同设计展开：

| 文档 | 内容 |
|---|---|
| [01-qoq-w4a8kv4.md](quantization/01-qoq-w4a8kv4.md) | QServe 的 QoQ：W4A8KV4、渐进式分组量化、计算感知权重重排、KV4 Attention 与 SmoothAttention |

---

## Communication

| 文档 | 内容 |
|---|---|
| [01-proxy-gdaki-gpi.md](communication/01-proxy-gdaki-gpi.md) | 传统 CPU Proxy、QP 爆炸、GDAKI 提交模式；BlueFlame 与 GPI 的队列所有权差异 |
| [02-mok-scheduling-and-buffers.md](communication/02-mok-scheduling-and-buffers.md) | MoK 的设备端 schedule、dispatch pull / combine push、minibatch overlap、macrobatch ring buffer，以及与 DeepEP Buffer 的对照 |
| [03-low-latency-collectives-and-synchronization.md](communication/03-low-latency-collectives-and-synchronization.md) | 小消息 collective 的 memory ordering、Sentinel、双缓冲 credit、fabric / SHARP、LL128 atomic，以及 `multimem.ld_reduce` 与 AllReduce 的关系 |

## DeepSeek-V4

基于 [SGLang](https://github.com/sgl-project/sglang) 源码梳理。

| 文档 | 内容 |
|---|---|
| [architecture.md](models/deepseek_v4/architecture.md) | 模块树、MQA / CSA / HCA、MoE、KV 与压缩状态 |
| [mhc.md](models/deepseek_v4/mhc.md) | mHC 动机、公式、Sinkhorn、层内数据流与伪代码 |
| [mqa.md](models/deepseek_v4/mqa.md) | MQA 投影形状、单 KV head、RoPE/nope、CSA/HCA 与 Attention 数据流 |

## GLM-5.3

Text-only 主干（`GlmMoeDsaForCausalLM` 一脉）。**不要**和 `glm-5.3-flash/` 混读。

| 文档 | 内容 |
|---|---|
| [architecture.md](models/glm-5.3/architecture.md) | MLA / DSA / MoE 主干结构 |
| [mla.md](models/glm-5.3/mla.md) | MLA 投影、latent KV cache、RoPE/nope |
| [dsa.md](models/glm-5.3/dsa.md) | DSA indexer、top-k、skip 复用、KPool 与 MLA 关系 |

## GLM-5.3-Flash

`Glm5Next*`：hybrid **KDA + MLA**，可选 mHC / 多模态。

| 文档 | 内容 |
|---|---|
| [architecture.md](models/glm-5.3-flash/architecture.md) | KDA、MLA、DSA、mHC、多模态入口与 DFlash capture |
| [radix_linear_attention.md](models/glm-5.3-flash/radix_linear_attention.md) | `RadixLinearAttention` 层壳、KDA gated delta 递推、prefill chunk / decode、conv/SSM cache |

## Kimi-K3

| 文档 | 内容 |
|---|---|
| [architecture.md](models/kimi-k3/architecture.md) | 配置一览、MLA/KDA 混排、MoE、AttnRes（bank + prefix）、forward / logits |

## Qwen3.8

**Qwen3.8-2.4T-A95B**（`qwen3_5_moe_text`）。Hybrid = GDN + **Gated GQA**，**没有** QSA / PLE。

| 文档 | 内容 |
|---|---|
| [architecture.md](models/qwen-3.8/architecture.md) | 配置、GDN/Gated GQA 混排、Sparse MoE |

## Qwen3.8-Flash-Next

Qwen4 架构预览（`qwen4_exp`）：GDN + **QSA**，**Gated Residual（HC×4）**，**N-gram / PLE ~51B**。

| 文档 | 内容 |
|---|---|
| [architecture.md](models/qwen3.8-flash-next/architecture.md) | 配置总览、GDN/QSA 混排、HC Mix/Combine、PLE 入口 |
| [qsa.md](models/qwen3.8-flash-next/qsa.md) | c4 压缩 indexer、`update_compressed_index_cache`、expand_blocks / unfinished 尾巴、稀疏 GQA、MTP IndexShare |
| [ple.md](models/qwen3.8-flash-next/ple.md) | PLE 动机、可学习 N-gram 表、int64 hash、`gate(norm(Q),norm(K))`、注入 HC、host offload |

要点备忘：

- **QSA**：先 top-k **block** → `expand_blocks` 成 token index → 读**原始** K/V；未满 4-token 尾巴强制并入。
- **PLE**：预训练学好的查表（非 token emb 拷贝）；推理冻结，可放 host；hash = 乘数 + XOR + 素数模。

---

## License

[MIT](LICENSE)
