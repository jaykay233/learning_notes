# Inference Scheduling

推理框架 scheduler 的 token budget、running-first、DP attention 混合 P/D 执行与跨 rank 同步。

## 文档索引

| 文档 | 内容 |
|---|---|
| [01-vllm-v1-token-budget-and-spec-slots.md](01-vllm-v1-token-budget-and-spec-slots.md) | vLLM V1 的 `max_num_scheduled_tokens`、`max_num_batched_tokens`、per-request draft slot、数值 trace 与 CPU 仿真 |
| [02-sglang-dp-attention-prefill-delayer.md](02-sglang-dp-attention-prefill-delayer.md) | SGLang attention DP 的本地 P/D 选择、`--enable-prefill-delayer`、MLP/EP collective 同步与数值 trace |
| [examples/vllm_v1_token_budget.py](examples/vllm_v1_token_budget.py) | 可运行的预算模拟：两个请求、14 个普通 token、6 个 draft slot、20 个输入 slot |
| [examples/sglang_dp_prefill_delayer.py](examples/sglang_dp_prefill_delayer.py) | 可运行的 DP attention 调度模拟：三个 rank 的 mixed/all 状态与 idle units |

## 当前进度

- [x] vLLM V1 running-first 与 chunked prefill 共用 token budget
- [x] `max_num_scheduled_tokens` 与 `max_num_batched_tokens` 的两本预算账
- [x] 投机解码 per-request draft slots 与各算法额外 slot 表
- [x] SGLang DP attention 的 prefill-first、prefill delayer 与 MLP/EP 同步
- [ ] TensorRT-LLM 的 scheduler、ADP router、phase intent 与 dummy request
- [ ] KV cache allocation 失败后的抢占与 recompute
- [ ] prefill/decode 混合 batch 的 KV block 分配边界

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

## 下一知识点

转入 TensorRT-LLM：沿 `ADPRouter`、`_balance_adp_requests()`、`_ADPForwardIntent` 和 dummy request 路径，解释 attention DP 如何协商本轮 phase，以及它与 SGLang prefill delayer 的差异。

## 源码快照

- vLLM：`c1dc2e9cf992995953548c392f8c67db7283495b`
- SGLang：`af9bb2a32d4a313d20a629ac1b2cb8e6ef245b95`
- 自适应预算修正：`0914ed2e81`
- 依赖：Python 3.10+；示例仅需标准库，不需要 GPU
