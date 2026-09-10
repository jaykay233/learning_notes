# CUDA Graph 基础与 torch-memory-saver

> 依据：[Awesome-ML-SYS-Tutorial · cuda-graph](https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/tree/main/torch/cuda-graph)、[torch_memory_saver](https://github.com/fzyzcjy/torch_memory_saver/)。  
> SGLang 侧录图 / 与 Inductor 关系见 [02-sglang-cudagraph-vs-inductor-trees.md](./02-sglang-cudagraph-vs-inductor-trees.md)。

---

## 1. 什么是 CUDA Graph

把一串 GPU 操作（kernel launch、memcpy、memset 等）录成一张 **DAG**：节点是操作，边是依赖。

- **不用 Graph**：CPU 为每个小 kernel 单独 launch → 短 kernel 多时，启动开销累积成瓶颈。
- **用 Graph**：定义并实例化后，CPU **一次 launch** 整图；结构不变时可反复 replay，无需重录。

较新的 CUDA 还有条件节点等，允许图内局部分支/循环而不把控制权交回 CPU。

一句话：**用「整图提交」换掉「频繁 CPU↔GPU 交互」。**

---

## 2. 为什么推理大量用、训练很少用

### 推理适合

1. 模型结构固定（层序、连接不变）。
2. 给定 batch / 最大序列等约束后，计算图形状可预先定死。
3. decode 等路径高度重复 → 录一次、replay 无数次，收益大。

因此 SGLang 等框架会为 CUDA Graph **预留显存**（常见量级 1～数 GB，视 bs 档与模型而定）。

### 训练不适合（相对）

1. 优化器、梯度裁剪、LR schedule 等控制流更动态。
2. 梯度累积使「何时 step / 清梯度」改变图形态。
3. Dropout 等随机性难静态捕获。
4. 激活与梯度导致 **显存分配模式动态**；Graph 捕获时要钉死内存需求与指针。

---

## 3. Graph 对显存的硬约束

CUDA Graph **不管理**虚拟地址布局；它记录的是捕获时各 kernel 引用的 **GPU 虚拟地址（指针）**。

Replay 时仍用这些指针。若中间 `cudaFree` 再分配，即使大小相同，**虚拟地址也可能变** → 图失效或结果错误。

因此：要长期 replay，相关 tensor 的地址必须在图生命周期内保持稳定（或用下面的 VMM 技巧在「同地址」上换物理页）。

---

## 4. torch-memory-saver：region / pause / resume

目标：在 **不破坏 Graph 已记录指针** 的前提下，暂时把物理显存让给别人（典型：训推同卡 / RL colocate）。

机制：用 CUDA Driver **虚拟内存 API**，把 **虚拟地址** 与 **物理页** 分开管。

| API | 何时用 | 做什么 |
|---|---|---|
| `region(tag=...)` | **分配时** | 作用域内的 malloc 走自定义路径，打上 tag |
| `pause(tag=...)` | 推理 idle / 交给训练 | unmap + 释放**物理**页；**虚拟地址保留** |
| `resume(tag=...)` | 再开推理 / 再 replay | 在**同一虚拟地址**上 remap 新物理页 |

### 4.1 `region` 是什么

**不是**预先划好的一块连续显存池，而是 **context 作用域**：

```python
with torch_memory_saver.region(tag="kv_cache"):
    a = torch.empty(...)   # 归 tag=kv_cache
    b = torch.empty(...)   # 也归 tag=kv_cache
# 离开后普通分配，不受管
```

实现上接近 thread-local：`enter` → `is_interesting_region=true` + `current_tag`；劫持的 `cudaMalloc` 看开关决定走 VMM 还是原生路径。

同一 tag 下可以是多块分散分配；`pause("kv_cache")` 卸掉该 tag 下所有块的物理页。

常见 tag 示意：`kv_cache`、权重、`cuda_graph`（捕获产生的中间缓冲）。

### 4.2 典型时间线

```text
启动
  └─ region / cuda_graph(...) 分配并捕获（地址钉死）
稳态推理（反复 replay）          ← 不用 pause/resume

切到训练 / 暂停引擎
  └─ pause(tag=...)              ← 腾物理显存

训练结束，再 serving
  └─ resume(tag=...)             ← 地址不变，Graph 仍可 replay
  └─ （按需）重建 KV / 恢复权重内容
```

只做纯推理、不与训练抢卡时：主要用 `region`（或 `cuda_graph`）保证指针稳定；**不会**每个 decode step 都 pause。

### 4.3 和 `torch.compile` 的对比（概念层）


| | CUDA Graph（手工 / Runner） | `torch.compile` |
|---|---|---|
| 抽象 | 偏底层：捕获 GPU 操作序列 | 偏上层：Dynamo → Inductor 等 |
| 动态性 | 结构 / 形状 / 地址要稳 | 可 graph break，动态形状更灵活 |
| 目标 | 主要砍 CPU launch | 融合、访存、可选底层再用 Graph |

`mode="reduce-overhead"` 等会更倾向让 Inductor 启用 Graph；与 SGLang 自管 Graph 如何共存，见文档 02。

---

## 5. 延伸阅读

- [Optimizing Memory Usage in verl](https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial)（教程文内链）
- SGLang：`enable_memory_saver`、`SGLANG_MEMORY_SAVER_CUDA_GRAPH`；相关讨论 [PR #7873](https://github.com/sgl-project/sglang/pull/7873)、[#10217](https://github.com/sgl-project/sglang/pull/10217)
- 下一篇：[02 · SGLang CudaGraphRunner 与 Inductor CUDAGraph Trees](./02-sglang-cudagraph-vs-inductor-trees.md)
- 对照总览：[03 · 几种外壳、VMM 与劫持分配](./03-flavors-vmm-and-hijack.md)
