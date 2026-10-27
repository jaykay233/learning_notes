# Inference Scheduling

推理框架 V1 scheduler 的统一 token budget、running-first、投机解码 slot 预算与 KV cache 抢占。

## 文档索引

| 文档 | 内容 |
|---|---|
| [01-vllm-v1-token-budget-and-spec-slots.md](01-vllm-v1-token-budget-and-spec-slots.md) | vLLM V1 的 `max_num_scheduled_tokens`、`max_num_batched_tokens`、per-request draft slot、数值 trace 与 CPU 仿真 |
| [examples/vllm_v1_token_budget.py](examples/vllm_v1_token_budget.py) | 可运行的预算模拟：两个请求、14 个普通 token、6 个 draft slot、20 个输入 slot |

## 当前进度

- [x] vLLM V1 running-first 与 chunked prefill 共用 token budget
- [x] `max_num_scheduled_tokens` 与 `max_num_batched_tokens` 的两本预算账
- [x] 投机解码 per-request draft slots 与各算法额外 slot 表
- [ ] KV cache allocation 失败后的抢占与 recompute
- [ ] prefill/decode 混合 batch 的 KV block 分配边界

## 已经覆盖

- V1 不区分独立的 prefill/decode token budget；两者都以请求当前欠算 token 数进入统一预算。
- running 请求优先于 waiting；running 中仍可能同时包含 decode 和 prefill chunk。
- `max_num_scheduled_tokens` 只统计签发的普通 token；`max_num_batched_tokens` 还包含 drafter 追加 slot。
- 当前实现按实际调度请求逐个扣 `draft_slots`，不再按 `max_num_seqs` 做固定全局预留。
- P-EAGLE、PARD、串行 draft model、DFlash/DSpark 的额外 slot 数不同。
- `0914ed2e81` 将固定全局预留改成 per-scheduled-request 的自适应预算。

## 下一知识点

沿 `KVCacheManager.allocate_slots()` 返回 `None` 的路径，解释 running 请求如何被抢占、为何回到 waiting、KV block 如何释放，以及 recompute 的调度和性能含义。

## 源码快照

- vLLM：`c1dc2e9cf992995953548c392f8c67db7283495b`
- 自适应预算修正：`0914ed2e81`
- 依赖：Python 3.10+；示例仅需标准库，不需要 GPU
