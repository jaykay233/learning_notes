# SGLang CudaGraphRunner 与 Inductor CUDAGraph Trees

> 承接 [01-basics-and-memory-saver.md](./01-basics-and-memory-saver.md)。  
> 依据：SGLang `cuda_graph_runner.py` / Breakable CUDA Graph、[PyTorch CUDAGraph Trees](https://docs.pytorch.org/docs/main/user_guide/torch_compiler/torch.compiler_cudagraph_trees.html)、[SGLang BCG 博客](https://www.sglang.io/blog/breakable-cuda-graph)。

---

## 1. SGLang 怎么录：整 forward，不是按层循环

### 1.1 默认 Decode：`CudaGraphRunner`

对每个要覆盖的 **batch size**（以及 decode / speculative 等 mode）录图：

1. 准备该 bs 的 **静态 input/output buffer**
2. `with cuda_graph(...): run_once()`
3. `run_once` = 一次完整 **`model.forward(...)`**

粒度是：

```text
多档 bs ×（可能多个 forward mode）→ 多张图
每张图覆盖「整网 forward」，不是 for layer in layers: capture(layer)
```

Runtime：把真实 batch **pad** 到最近已捕获档，copy 进静态 buffer，再 `graph.replay()`。

### 1.2 变体：Full / Breakable / Piecewise

| 模式 | 怎么录 | 和「每层」的关系 |
|---|---|---|
| **Full CUDA Graph** | 一次 forward → 一张完整图 | 整网，无 eager 段 |
| **Breakable (BCG)** | 仍从一次 `model.forward` 开始；在 **attention / mamba** 等处自动断图 | 多段 segment；断点在 attention 边界，**不是**业务上「每层各录一张」 |
| **tc_piecewise（偏 prefill）** | `torch.compile` 切段后再由 SGLang 侧 capture | 分段由编译/注意力边界决定 |

BCG 动机：标准整图性能最好，但 attention metadata、部分自定义 kernel、调试等需要 **eager 缝**；在 capture 过程中插入 break，兼容性更好，构建也往往比纯 compile 切段更快（见官方 BCG 博客）。

Prefill 后端选择示意：`tc_piecewise` / `breakable` / `full` / `disabled`（以当时 SGLang 配置为准）。

---

## 2. 两套「谁在管 CUDA Graph」

| | **SGLang `CudaGraphRunner`** | **Inductor CUDAGraph Trees** |
|---|---|---|
| **驱动方** | Serving 框架显式 `torch.cuda.graph` / BCG | `torch.compile` → Inductor 自动包在编译 partition 外 |
| **录什么** | 整段（或 BCG 多段）`model.forward`；静态 buffer 由 runner 准备 | FX/Inductor **partition**；按执行路径长成 **tree/forest** |
| **形状** | 预先枚举 `cuda_graph_bs`；pad 到档 | 运行中按 shape/路径分支；可多 root |
| **显存** | SGLang static tensor + `graph_pool` | Caching allocator **私有池** + liveness / **checkpoint**；路径间共享池 |
| **输入** | 业务侧写入 runner 静态地址再 replay | 常先 **copy** 到图内静态地址再 replay |
| **懂不懂 serving** | 懂 decode padding、KV、attn metadata、TP barrier | 通用编译路径，不内建 SGLang batch/KV 契约 |

CUDAGraph Trees 要点（官方文档）：

- 多段图 **共享同一 memory pool**，避免段间拷中间结果、并按 `max(路径显存)` 复用。
- 支持树状分支：replay A 后可走 B 或 B'，不必死成单一序列。
- Replay 后 pool 状态要 checkpoint，才能安全录/跑另一分支。

---

## 3. 为什么不能「叠着用」

两边都想当 **stream capture + 显存地址** 的主人：

1. **禁止嵌套 capture**  
   Runner 外层已 `begin capture` 时，Trees 再 capture / 做 capturing 下非法的分配或同步 →  
   `operation not permitted when stream is capturing`（如 [SGLang #7918](https://github.com/sgl-project/sglang/issues/7918)）。

2. **两套 pool / 指针约定**  
   Trees：私有池 + checkpoint。Runner：静态 buffer + 自有 pool。  
   都假定「图内指针由我稳定管理」→ 叠用 pool tracking 错乱。

3. **生命周期不同**  
   - Runner：启动时对固定 bs **主动录完**，之后只 replay。  
   - Trees：**lazy** 录路径，含 warmup、再录子节点。  
   包在 runner 的 capture/replay 里时，Trees 的懒逻辑对不上。

4. **输入 copy-in 语义冲突**  
   两套静态地址与 in-place 约定叠加 → 双倍拷贝或写错缓冲。

---

## 4. SGLang 的实际组合方式

`CudaGraphRunner` 开 `--enable-torch-compile` 时，默认：

```text
mode = max-autotune-no-cudagraphs
（可用环境变量 SGLANG_TORCH_COMPILE_MODE 覆盖）
```

含义：

```text
✅ Inductor：算子融合 / autotune（不要 Trees）
✅ CudaGraphRunner：外层整 forward（或 BCG）CUDA Graph
❌ Inductor CUDAGraph Trees + CudaGraphRunner 双重 capture
```

Piecewise / `tc_piecewise` 是另一条整合线：SGLang compile backend **切开** forward，再由 SGLang **按段** capture——仍不是把官方 Trees 原样嵌进 Runner。

失败时官方提示常见选项：减小 `--mem-fraction-static` / `--cuda-graph-max-bs`、关掉 `--enable-torch-compile`、或（不推荐）`--disable-cuda-graph`。

---

## 5. 选型速记

| 场景 | 倾向 |
|---|---|
| LLM decode，形状可枚举 | `CudaGraphRunner` Full / 默认路径 |
| Attention 难进图、要调试 | BCG / `--debug-cuda-graph` |
| Prefill 要 compile 切段 | `tc_piecewise` 等 prefill backend |
| 只要融合、不要第二套 Graph | `max-autotune-no-cudagraphs` + 外层 Runner |
| 训推同卡腾显存 | memory-saver `region` + `pause`/`resume`（见 01） |

---

## 6. 相关代码与文档（入口）

| 资源 | 说明 |
|---|---|
| `sglang/.../cuda_graph_runner.py` | Decode CudaGraphRunner、`patch_model` + compile mode |
| `breakable_cuda_graph*` | BCG 分段 capture/replay |
| `prefill_cuda_graph_runner` / `tc_piecewise_*` | Prefill 后端 |
| [torch/_inductor/cudagraph_trees.py](https://github.com/pytorch/pytorch/blob/main/torch/_inductor/cudagraph_trees.py) | Trees 实现与 pool 语义 |
| [01 · 基础与 memory-saver](./01-basics-and-memory-saver.md) | Graph 是什么、region/pause/resume |
| [03 · 几种外壳与 VMM](./03-flavors-vmm-and-hijack.md) | Trees / BCG / memory-saver 对照；劫持换分配器 |
