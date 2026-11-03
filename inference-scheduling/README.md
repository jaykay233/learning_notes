# Inference Scheduling

推理框架的 token budget、running/waiting 与 KV 抢占、DP attention 混合 P/D 执行、EPLB、跨 rank 同步、带分层 KV 的集群性能仿真、SSD 前缀 KV 的重要性加载、位置无关 KV 复用，以及单机 CPU-GPU MoE 专家卸载。

## 文档索引

| 文档 | 内容 |
|---|---|
| [01-vllm-v1-token-budget-and-spec-slots.md](01-vllm-v1-token-budget-and-spec-slots.md) | vLLM V1 的 `max_num_scheduled_tokens`、`max_num_batched_tokens`、per-request draft slot、数值 trace 与 CPU 仿真 |
| [02-sglang-dp-attention-prefill-delayer.md](02-sglang-dp-attention-prefill-delayer.md) | SGLang attention DP 的本地 P/D 选择、`--enable-prefill-delayer`、MLP/EP collective 同步与数值 trace |
| [03-tensorrt-llm-generation-first-and-mixed-batch.md](03-tensorrt-llm-generation-first-and-mixed-batch.md) | TLLM V2 的 generation-first、统一 token budget、chunked prefill 混批与 PD 分离边界 |
| [04-tensorrt-llm-kv-cache-preemption.md](04-tensorrt-llm-kv-cache-preemption.md) | KV allocation 失败后的 suspend、HELD、recompute pause、self-evict 阶梯 |
| [05-tensorrt-llm-ep-eplb-and-dispatch-layout.md](05-tensorrt-llm-ep-eplb-and-dispatch-layout.md) | EPLB 的 logical expert → global slot/rank 映射，EP rank-major、expert-major 与 combine |
| [06-kv-preemption-requester-vs-victim.md](06-kv-preemption-requester-vs-victim.md) | 区分 KV requester 与 victim：FCFS、priority、SGLang length retraction、GNE 与 MAX_UTILIZATION |
| [07-compositional-inference-cluster-simulator.md](07-compositional-inference-cluster-simulator.md) | 组合式离散事件仿真：局部 compute/transfer profile、分层 KV 不变量、P-D handoff、1P-1D 标定复用与策略对比 |
| [08-pdscope-prefill-decode-moe-offload.md](08-pdscope-prefill-decode-moe-offload.md) | PDScope：单机 MoE 专家 offload、PhasePer 分相预测、AdaptSched 跨层调度与异步 I/O |
| [09-impress-importance-aware-prefix-kv.md](09-impress-importance-aware-prefix-kv.md) | IMPRESS：probe head 重要性识别、Jaccard 门控、radix node 内 KV 重排与 score 缓存 |
| [10-mepic-exact-vs-approximate-kv-reuse.md](10-mepic-exact-vs-approximate-kv-reuse.md) | MEPIC：位置无关 chunk 复用、首 block 重算、共享 tail 与精确性边界 |
| [11-pd-resource-allocation-and-simulator.md](11-pd-resource-allocation-and-simulator.md) | P/D 资源分配：有效 Prefill 吞吐、连续实例数、整数配置搜索与 benchmark-calibrated 离散事件仿真 |
| [examples/vllm_v1_token_budget.py](examples/vllm_v1_token_budget.py) | 可运行的预算模拟：两个请求、14 个普通 token、6 个 draft slot、20 个输入 slot |
| [examples/sglang_dp_prefill_delayer.py](examples/sglang_dp_prefill_delayer.py) | 可运行的 DP attention 调度模拟：三个 rank 的 mixed/all 状态与 idle units |
| [examples/tensorrt_llm_generation_first_budget.py](examples/tensorrt_llm_generation_first_budget.py) | generation-first 与 chunked prefill 两种预算分配 trace |
| [examples/tensorrt_llm_preemption_ladder.py](examples/tensorrt_llm_preemption_ladder.py) | token budget、suspend、HELD、recompute pause 与 self-evict 决策 trace |
| [examples/tensorrt_llm_eplb_dispatch.py](examples/tensorrt_llm_eplb_dispatch.py) | logical top-k 经 EPLB 重映射后的 rank/slot、combine 和通信布局数值 trace |
| [examples/kv_preemption_victim_policy.py](examples/kv_preemption_victim_policy.py) | 六个 CPU trace，对照 requester、victim、self-preempt 与 waiting 的不同结果 |
| [examples/impress_importance_tiering.py](examples/impress_importance_tiering.py) | Jaccard 门控、共享重要 token、score 缓存、节点内重排与 K/V 向量数的 CPU trace |
| [examples/mepic_position_independent_kv.py](examples/mepic_position_independent_kv.py) | MEPIC 的 block 占用、精确/近似路径与 RoPE、上下文误差来源的 CPU trace |
| [examples/pd_capacity_simulator.py](examples/pd_capacity_simulator.py) | P/D 容量公式、Prefill/Decode/KV 请求生命周期、P95/P99 与候选配置对比的 CPU 离散事件仿真 |

## 当前进度

- [x] vLLM V1 running-first 与 chunked prefill 共用 token budget
- [x] `max_num_scheduled_tokens` 与 `max_num_batched_tokens` 的两本预算账
- [x] 投机解码 per-request draft slots 与各算法额外 slot 表
- [x] SGLang DP attention 的 prefill-first、prefill delayer 与 MLP/EP 同步
- [x] TensorRT-LLM generation-first、token budget、chunked prefill 与 PD batch 边界
- [x] TensorRT-LLM KV allocation 失败后的 suspend/preempt/recompute/self-evict 阶梯
- [x] TensorRT-LLM EP/EPLB 的 logical expert 到 physical slot，以及 rank-major/expert-major dispatch
- [x] KV allocation 失败时 requester 与 victim 的角色分离，以及跨框架 victim policy
- [x] LLM 集群的组合式性能建模：局部服务时间、KV 容量预留、资源排队与跨拓扑复用
- [x] PDScope 的 prefill/decode 分相专家预测、跨层代价模型与静态图异步搬运
- [x] IMPRESS 的 probe head 识别、相似度回退、KV 重排与重要性感知缓存
- [x] MEPIC 的位置无关 chunk 复用、首 block 重算与精确性边界
- [x] P/D 资源数量公式、TTFT 下的有效 Prefill 吞吐、TPOT 下的 Decode batch 与离散事件仿真
- [ ] TensorRT-LLM 的 ADP router、phase intent 与 dummy request

## 已经覆盖

- V1 不区分独立的 prefill/decode token budget；两者都以请求当前欠算 token 数进入统一预算。
- running 请求优先于 waiting；running 中仍可能同时包含 decode 和 prefill chunk。
- `max_num_scheduled_tokens` 只统计签发的普通 token；`max_num_batched_tokens` 还包含 drafter 追加 slot。
- 当前实现按实际调度请求逐个扣 `draft_slots`，不再按 `max_num_seqs` 做固定全局预留。
- P-EAGLE、PARD、串行 draft model、DFlash/DSpark 的额外 slot 数不同。
- `0914ed2e81` 将固定全局预留改成 per-scheduled-request 的自适应预算。
- SGLang attention DP 允许各 rank 拥有不同的本地 P/D batch，但 MLP/MoE/EP collective 仍要求所有 rank 同轮参与。
- `--enable-prefill-delayer` 在 `mixed` 状态延迟新 prefill，等待更多 rank 对齐或超时放行，用来减少单个重 prefill 与其他 rank 轻 decode 造成的空泡。
- EP dispatch/combine 使用逻辑 expert top-k 和当前物理映射；空 DP rank 也必须以 idle batch 或零 token 参与。
- TensorRT-LLM V2 默认 generation/decode first，但 chunked prefill 可以用剩余预算与 generation 同轮混批。
- PD 分离后，P 侧 chunked prefill 仍属于 context worker 的 batch，D 侧 decode 属于另一个 worker/engine。
- token budget 不足先 `STOP`；KV target/draft allocation 失败才进入 suspend、recompute pause、self-evict 阶梯。
- GPU-only cache tier 的 suspend 可能只产生 `HELD` pages，真正释放 KV 的 recompute pause 会把请求打回 context。
- EPLB 不改变 router 的 logical top-k；它改变 logical expert 到 global slot/rank 的当前映射，以及后续 dispatch 和 per-rank load。
- placement 与专家权重必须来自同一版本，否则调度和 collective 都能完成，但会把 token 送到旧权重的物理 expert。
- KV allocation 失败只说明谁是 requester；victim 由 FCFS、priority、length 或 capacity policy 单独选择。
- vLLM FCFS 选择 running 尾部，priority 选择最大 `(priority, arrival_time)`；只有 requester 自己被选中时才 self-preempt。
- SGLang decode retraction 是 batch 级检查，默认 length 顺序优先 retract 短输出、长输入的请求。
- TensorRT-LLM GNE 不暂停已启动请求；MAX_UTILIZATION 可从 active 尾部暂停，V2 当前强制使用该 policy。
- 集群仿真把路由策略、局部 compute/transfer 服务模型和共享资源队列分开；换 worker 数时复用兼容 profile，重新组合事件顺序。
- 分层 KV 的搬运必须满足目标容量先预留、源副本使用中受保护、目标副本传输完成后才可见。
- Qwen3/L20Z/TP2 上的 1.34% 是 QA E2E MAPE；不能直接外推到 H800、其他 TP 或不同推理引擎。
- PDScope 的跷跷板来自 prefill 的权重复用与 decode 的单 token 搬运差异；同一层专家加载成本在 decode 中无法被后续 token 摊薄。
- PhasePer 分别为 prefill 与 decode 预测下一层专家，预测只决定预取和放置；真实 gate 输出后必须覆盖草稿路由。
- AdaptSched 联合评估当前层放置与下一层预取；CPU 既保存权重，也可以在代价更低时执行冷专家。
- PDScope 是单机 CPU-GPU 专家调度，不是 P/D 分离、EP、MegaMoE 或 DeepGEMM kernel 优化；它也不减少 KV cache。
- IMPRESS 优化共享前缀从 SSD/CPU 到 GPU 的加载，不减少完整 KV 的总容量；完整副本仍保留在磁盘。
- probe head 只负责估计重要 token 集合；被选 token 的 K/V 仍需从全部 head 读取。
- probe 相似度低于相对阈值时必须回退到完整加载，否则省 I/O 的代价可能是个别 head 的 attention 失真。
- KV reorder 只能在 radix node 内进行；chunk 级读取是否受益取决于重要 token 是否被重排到更少的 chunk。
- score 由访问频率和重要比例共同决定，单纯 LRU/LFU 会把热但低重要比例的 chunk 留在 GPU。
- IMPRESS 与 HiSparse 正交：前者减少前缀复用时的慢层 I/O，后者选择 decode 每一步访问的 KV block。
- MEPIC 首次完整计算 chunk 时等价于普通 prefill；正常复用只重算每个 chunk 的第一个 block，因此是近似路径。
- fused RoPE 可以精确调整缓存 K 的绝对位置，但不能恢复左侧上下文变化导致的 raw K/V 差异。
- 若业务要求逐 token 数值等价，需要重算完整 chunk，或在近似路径外增加误差检测与回退。
- `TP_total` 统计输入和输出 token 的总量；P/D 比例由 `L_in/L_out` 乘上 `TP_D/TP_P,eff` 得到。
- TTFT 通过 M/M/1 的服务率 `μ=TP_P,max/L_in` 进入模型，目标延迟越严格，有效 Prefill 吞吐越低。
- Decode 吞吐必须取满足 TPOT 的 batch 点；连续实例数向上取整后还要重新计算容量，`3P4D` 约为 4.875 MTPM，不能严格覆盖 5 MTPM。
- 离散事件仿真器把 P 队列、D batch、KV transfer、请求生命周期和 P95/P99 分开记录，适合在解析公式给出的候选范围内搜索整数配置。
- 仿真的准确性由 P/D profile、KV 传输模型和请求 trace 决定；有限 trace 的吞吐读数包含启动与 drain，不是稳态容量本身。

## 下一知识点

把 P/D simulator 接入真实的 Prefill/Decode/KV profile 与请求 trace，验证模型外推误差。比较 radix cache、IMPRESS、MEPIC 与 HiSparse 在请求、共享 prefix、可复用 chunk 和 decode block 四层的索引与复用边界。TensorRT-LLM ADP 顺延到后续笔记。

## 源码与论文快照

- vLLM（01）：`c1dc2e9cf992995953548c392f8c67db7283495b`
- SGLang（02）：`af9bb2a32d4a313d20a629ac1b2cb8e6ef245b95`
- vLLM（06）：`bf4b3b5cb86e1b4ccff23906c0dae46309c935e8`
- SGLang（06）：`635430d07378e2b12381353781a23b0704969909`
- TensorRT-LLM：`0372799d1543ef5920c1318b829cf127800dbcb9`
- 自适应预算修正：`0914ed2e81`
- IMPRESS / HyperInfer：`91df7a2fc12581c6780a6485c5b50e7fd9ef9d0e`
- MEPIC：arXiv `2512.16822v1`
- P/D 资源分配：arXiv `2603.04716v1`；公众号 `LLM软硬协同优化`，2026-09-11
- 依赖：Python 3.10+；示例仅需标准库，不需要 GPU
