# Inference Scheduling

推理框架 scheduler 的 token budget、running-first、DP attention 混合 P/D 执行与跨 rank 同步。

## 文档索引

| 文档 | 内容 |
|---|---|
| [01-vllm-v1-token-budget-and-spec-slots.md](01-vllm-v1-token-budget-and-spec-slots.md) | vLLM V1 的 `max_num_scheduled_tokens`、`max_num_batched_tokens`、per-request draft slot、数值 trace 与 CPU 仿真 |
| [02-sglang-dp-attention-prefill-delayer.md](02-sglang-dp-attention-prefill-delayer.md) | SGLang attention DP 的本地 P/D 选择、`--enable-prefill-delayer`、MLP/EP collective 同步与数值 trace |
| [03-tensorrt-llm-generation-first-and-mixed-batch.md](03-tensorrt-llm-generation-first-and-mixed-batch.md) | TLLM V2 的 generation-first、统一 token budget、chunked prefill 混批与 PD 分离边界 |
| [04-tensorrt-llm-kv-cache-preemption.md](04-tensorrt-llm-kv-cache-preemption.md) | KV allocation 失败后的 suspend、HELD、recompute pause、self-evict 阶梯 |
| [05-tensorrt-llm-ep-eplb-and-dispatch-layout.md](05-tensorrt-llm-ep-eplb-and-dispatch-layout.md) | EPLB 的 logical expert → global slot/rank 映射，EP rank-major、expert-major 与 combine |
| [examples/vllm_v1_token_budget.py](examples/vllm_v1_token_budget.py) | 可运行的预算模拟：两个请求、14 个普通 token、6 个 draft slot、20 个输入 slot |
| [examples/sglang_dp_prefill_delayer.py](examples/sglang_dp_prefill_delayer.py) | 可运行的 DP attention 调度模拟：三个 rank 的 mixed/all 状态与 idle units |
| [examples/tensorrt_llm_generation_first_budget.py](examples/tensorrt_llm_generation_first_budget.py) | generation-first 与 chunked prefill 两种预算分配 trace |
| [examples/tensorrt_llm_preemption_ladder.py](examples/tensorrt_llm_preemption_ladder.py) | token budget、suspend、HELD、recompute pause 与 self-evict 决策 trace |
| [examples/tensorrt_llm_eplb_dispatch.py](examples/tensorrt_llm_eplb_dispatch.py) | logical top-k 经 EPLB 重映射后的 rank/slot、combine 和通信布局数值 trace |

## 当前进度

- [x] vLLM V1 running-first 与 chunked prefill 共用 token budget
- [x] `max_num_scheduled_tokens` 与 `max_num_batched_tokens` 的两本预算账
- [x] 投机解码 per-request draft slots 与各算法额外 slot 表
- [x] SGLang DP attention 的 prefill-first、prefill delayer 与 MLP/EP 同步
- [x] TensorRT-LLM generation-first、token budget、chunked prefill 与 PD batch 边界
- [x] TensorRT-LLM KV allocation 失败后的 suspend/preempt/recompute/self-evict 阶梯
- [x] TensorRT-LLM EP/EPLB 的 logical expert 到 physical slot，以及 rank-major/expert-major dispatch
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

## 下一知识点

转入 TensorRT-LLM：沿 `ADPRouter`、`_balance_adp_requests()`、`_ADPForwardIntent` 和 dummy request 路径，解释 attention DP 如何协商本轮 phase，以及它与 SGLang prefill delayer 的差异。

## 源码快照

- vLLM：`c1dc2e9cf992995953548c392f8c67db7283495b`
- SGLang：`af9bb2a32d4a313d20a629ac1b2cb8e6ef245b95`
- TensorRT-LLM：`0372799d1543ef5920c1318b829cf127800dbcb9`
- 自适应预算修正：`0914ed2e81`
- 依赖：Python 3.10+；示例仅需标准库，不需要 GPU
