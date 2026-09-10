# 几种「CUDA Graph」分别是什么

> 承接 [01-basics-and-memory-saver.md](./01-basics-and-memory-saver.md)、[02-sglang-cudagraph-vs-inductor-trees.md](./02-sglang-cudagraph-vs-inductor-trees.md)。  
> 一句话：**底层只有一种 CUDA Graph；上面是三套外壳（谁录图、怎么切段、缓冲怎么分配）。**

---

## 1. 底层只有一种 Graph

NVIDIA 能力：录一串 GPU 操作 → instantiate → replay。  
下面这些名字都不是换了一种 Graph ISA，而是 **谁驱动录图 / 显存从哪来 / 能不能腾物理页**。

| 名字 | 本质 | 多出来的东西 |
|---|---|---|
| **Inductor CUDAGraph Trees** | `torch.compile` 自动给编译 partition 套 Graph | 私有 memory pool、树状路径、checkpoint |
| **Breakable / Full（SGLang）** | 框架显式 `torch.cuda.graph`（BCG 可切段） | eager 缝、`@eager_on_graph`；或整 forward 一张图 |
| **memory-saver 的 cuda graph** | 仍用上面某套在录图，但 **分配走 VMM** | 图相关缓冲可 `pause` / `resume` |

关系直觉：

```text
CUDA Graph（驱动能力）
    ├─ Inductor Trees      → 编译器自动录 + 池/树
    ├─ Breakable / Full    → SGLang 显式录（可切段）
    └─ memory-saver        → 给「正在用的那套录图」换分配器（VMM）
                             以便 pause/resume（不是第四种算法）
```

---

## 2. Piecewise vs Breakable（都是分段，造法不同）

两边最终都是：`[Graph 段] → [eager 缝] → [Graph 段] → …`（常在 attention 等处断开）。

| | **Piecewise（tc_piecewise / PCG）** | **Breakable（BCG）** |
|---|---|---|
| 怎么切 | `torch.compile` 整网 trace → FX → 按 `split_ops` 切开 | Capture 跑 forward 时遇 `@eager_on_graph` / `break_graph()` → 当场断段 |
| 要不要编译器 | 要 | 不要 |
| 建图成本 | 编译常占大半时间 | 无 compile，建图往往快很多 |
| 失败模式 | 不可 trace / 新 kernel 未注册 | Capture 段里撞上非法 op；与 memory-saver **cuda graph 模式**不兼容 |

「Capture 时非法 op」：stream 在 capturing 时不能做 sync、部分 host 回调、当场 JIT 等；BCG 用 eager 缝把它们挪出图。漏标仍会报 `operation not permitted when stream is capturing`。

「与 memory-saver cuda graph 不兼容」：两套都想接管 capture / 图内分配入口，文档要求不要同时开 BCG 与 `SGLANG_MEMORY_SAVER_CUDA_GRAPH`（二选一）。

---

## 3. memory-saver：劫持后换一种 malloc 实现

**核心判断：就是劫持分配，换成 VMM 实现；Graph 本身还是普通 CUDA Graph。**

典型路径：`LD_PRELOAD`（或同类钩子）劫持 `cudaMalloc`：

- 在 `region(tag=...)` / `cuda_graph(...)` 里 → 走自定义 **VMM** 分配，打 tag  
- 作用域外 → 仍走原生 CUDA 分配  

因此「memory-saver 的 cuda graph」≈：**录图 API 长得像 `torch.cuda.graph`，但捕获期相关缓冲不走普通 `cudaMalloc`。**

---

## 4. VMM 是什么、和一般 Graph 差在哪

**VMM = Virtual Memory Management**（CUDA Driver 虚拟内存 API），不是 Graph 的另一种格式。

一般 `cudaMalloc`：一次拿到「虚拟地址 + 物理页」绑死的一块。

VMM 拆开：

1. **Reserve** 虚拟地址（`cuMemAddressReserve`）  
2. **Create** 物理内存（`cuMemCreate`）  
3. **Map** 物理页映射到该地址（`cuMemMap`）

于是：

| 操作 | 效果 |
|---|---|
| `pause` | Unmap + 释放**物理**页；**虚拟地址保留** |
| `resume` | 新物理页 Map 回**同一虚拟地址** |

Graph 记录的是捕获时的 **虚拟地址**。地址不变 → 不必重录图，却能暂时把物理显存借走（训推同卡等场景）。


| | **一般 CUDA Graph** | **memory-saver 包过的录图** |
|---|---|---|
| 录/放 | 普通 capture / replay | 同样在录 Graph |
| 图内显存 | caching allocator / `cudaMalloc` | **VMM**，可按 tag 管理 |
| 解决的问题 | 减 CPU launch | **额外**解决「图占着几 GB 腾不出去」 |
| 不解决 | 显存腾挪 | 不替代 Trees / BCG 的切段逻辑 |

---

## 5. 选型速记

| 你想要… | 用 |
|---|---|
| 编译器自动录图 + 路径树 | Inductor CUDAGraph Trees（SGLang 外层 Runner 场景通常关掉，见 02） |
| Serving 整 forward / 可切段 | SGLang Full / Breakable |
| 图还能 replay，物理显存能腾 | memory-saver（`region` + `pause`/`resume`；录图走其 `cuda_graph`） |
| BCG + memory-saver cuda graph | **不要叠**（文档明确） |

---

## 6. 相关文档

- [01 · 基础与 region/pause/resume](./01-basics-and-memory-saver.md)  
- [02 · CudaGraphRunner vs Inductor Trees](./02-sglang-cudagraph-vs-inductor-trees.md)  
- [Awesome-ML-SYS-Tutorial · cuda-graph](https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/tree/main/torch/cuda-graph)  
- [torch_memory_saver](https://github.com/fzyzcjy/torch_memory_saver/)  
- [SGLang Breakable CUDA Graph](https://docs.sglang.io/docs/advanced_features/breakable_cuda_graph)
