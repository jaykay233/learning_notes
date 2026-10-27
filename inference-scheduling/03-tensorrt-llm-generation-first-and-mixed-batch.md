# TensorRT-LLM 的 generation-first、chunked prefill 与混批边界

> 源码快照：TensorRT-LLM `0372799d1543ef5920c1318b829cf127800dbcb9`（2026-10-27）。
> 本文讨论 PyTorch backend 默认的 V2 scheduler。核心区别是“谁先拿 token budget”和“同一轮能否同时执行 prefill/decode”，不要把二者混成同一个问题。

```text
本次讲解位置
章节：inference-scheduling
小节：TensorRT-LLM V2 scheduler 的预算顺序与 batch 边界
知识点：generation/decode first、context 剩余预算、chunked prefill 混批、PD 分离
上次：vLLM V1 running-first 与 SGLang attention-DP 混合 P/D
下次：KV allocation 失败后的 suspend/preempt/recompute 阶梯
PTX：不适用
```

## 为什么现在讲这个

前面比较 vLLM V1、SGLang 和 TensorRT-LLM 时，`running-first`、`generation-first`、`prefill-first` 很容易被理解成“本轮只运行某一类请求”。TLLM 的 V2 scheduler 恰好说明这个理解不对：它先让 generation 请求占用预算，再把剩余预算给 context；如果开启 chunked prefill，context chunk 和 generation 仍会进入同一个 `ScheduledRequests`，同一次 forward 执行。

PD 分离又把这个边界切开了。context worker 上的 chunked prefill 仍然属于 P 侧 batch，不会因为服务端配置了 `generation_first` 就跑到 decode worker 的 batch 里。把本地 scheduler 顺序、聚合混批、PD 编排顺序分开之后，才能正确回答“decode first 到底 first 在哪里”。

## 1. 心智模型：优先级顺序不等于 batch 内容

一轮 TLLM scheduling 可以写成：

```text
active_requests
  -> Phase 1: generation / disagg 先申请 token budget 与 KV
  -> Phase 2: context / encoder 使用剩余 budget 与 pages
  -> SchedulerOutput(context_requests, generation_requests, ...)
  -> 组成本轮 ScheduledRequests
  -> 同一次 model forward
```

`ScheduledRequests` 把 context 和 generation 放进不同列表，是因为两类请求的 attention、sampling 和 KV 行为不同，不代表它们必须分两轮执行。源码中对应的字段是：

- `context_requests_chunking`
- `context_requests_last_chunk`
- `generation_requests`

V2 scheduler 默认开启，`use_kv_cache_manager_v2=True`。`scheduler_v2.py::_schedule_loop()` 明确分成：

```text
Phase 1: generation / disagg only
Phase 2: schedule deferred context / encoder requests
```

因此这里的 generation-first 是预算优先级：

1. 已进入 generation 的请求先拿 token budget 和 KV pages。
2. context 请求只能用剩余预算。
3. 如果 chunked prefill 开启，context 可以被切到剩余预算大小，并与 generation 混在同一轮。
4. 如果没有 chunked prefill，一个完整 context 放不下就继续等待，不会为了塞入 generation 而擅自切块。

TLLM 的 `enable_chunked_prefill` 默认是 `False`。所以“generation-first 可以和 chunked prefill 混批”成立，但前半句不是默认部署的完整状态。

## 2. 预算到底怎么记

V2 的 `BudgetTracker` 同时限制 `max_num_tokens` 和 `max_num_requests`。generation 请求的预算是：

```text
generation_tokens = beam_width + draft_token_length
```

投机解码的 draft 部分也占同一个 `max_num_tokens`，不是额外免费预算。分配 target 和 draft KV 时，`_try_allocate_generation()` 还要求两边同时成功；draft 侧失败会回滚 target 侧增长。

context 请求会先走 `prepare_context`，prefix cache 命中后 `context_remaining_length` 缩小，预算按实际还要计算的 token 数扣，而不是按原始 prompt 长度扣。chunk 调度还会考虑：

- 剩余 token budget；
- chunk alignment；
- 最后一个 chunk 的 draft tokens；
- encoder replay rows；
- 多模态边界。

这里先抓住对理解调度最重要的两条：

```text
完整 context 放不下且未开 chunking -> 等待
部分 context 能放下且已开 chunking -> 本轮切一个 chunk
```

token budget 不够只会让本轮停止继续准入。它本身不会触发 KV 抢占。抢占要等 KV page allocation 失败才进入另一条路径，下一篇专门拆。

## 3. 三个框架的本地选择顺序

| 框架/模式 | 本地优先级 | chunk 能否与 decode 混批 | 容易混淆的配置 |
|---|---|---|---|
| vLLM V1 | running 请求先调度，running 中可以同时有 decode 和 prefill continuation | 可以 | `running-first` 不是“只允许 decode” |
| TensorRT-LLM V2 | generation 先占预算，context 用剩余预算 | 开启 chunked prefill 后可以 | local `generation-first` 不是 generation-only |
| SGLang | 默认主路径先尝试构造新 prefill，否则推进 decode | 取决于 chunked prefill、mixed chunk 和 attention-DP 配置 | `--prefill-decode-interval=N` 是周期性压 prefill、推进 N 轮 decode，不是永久改成 decode-only |

SGLang 的 `--prefill-decode-interval` 容易被叫成“decode first 开关”，更准确的说法是 prefill/decode cadence：一轮 extend/prefill 后，接下来若干轮优先推进 running decode，然后重新允许 prefill。它和 TLLM 的 phase ordering 不是同一层机制。

## 4. PD 分离：P 和 D 不是一个 scheduler batch

普通聚合部署中，TLLM 的 generation-first 由本地 `KVCacheV2Scheduler` 执行。PD 分离部署中还有另一个同名概念：

```text
DisaggServerConfig.schedule_style = context_first | generation_first
```

这个 `schedule_style` 决定服务编排和 KV transfer 的准备顺序，默认是 `context_first`。它不是把 P 侧 context 和 D 侧 generation 塞进同一个 scheduler batch 的开关。

典型 PD 部署是两套独立 worker/engine：

```text
P worker: prefill / chunked prefill / context batch
          -> 发送 KV
D worker: generation / decode batch
          -> 接收 KV 后继续生成
```

于是：

- P 侧一个 chunk 仍然可以和 P 侧其他 prefill/chunk 混批。
- D 侧通常主要跑 decode。
- 跨 P/D 没有同一个 local scheduler batch。
- 即使 P、D 同机甚至共卡，只要它们是独立 engine/worker，也不能仅凭 `schedule_style=generation_first` 把两侧请求合成一次 forward。

vLLM 和 SGLang 的典型 PD 分离也一样：P/D 是不同实例之间的 KV 交接边界。聚合时三者对 prefill/decode 的优先级不同；分离后“chunk 跟 prefill 在一起、decode 跟 generation 在一起”的边界一致。

## 5. 数值 trace

令 `max_num_tokens=16`：

```text
G1: beam=1, draft=1 -> 2 tokens
G2: beam=1, draft=0 -> 1 token
C1: remaining context=20
C2: remaining context=8
```

先执行 Phase 1，G1 和 G2 一共占 3 个 token，剩余 13 个给 context。

| 配置 | context 决策 | 本轮 batch | 使用 budget |
|---|---|---|---:|
| chunked=True | C1 取 13-token chunk，C2 等待 | G1 + G2 + C1 chunk | 16 |
| chunked=False | C1 的 20 token 放不下；C2 的 8 token 放得下 | G1 + G2 + C2 | 11 |

第二种留下 5 个 token 未使用，不是 scheduler 忘了填，而是 C1 不能被切开，C2 又已经完整放进去。恰好 5 是剩余预算，不足以放另一个完整请求。这也解释了为什么 chunked prefill 能提高混合负载利用率，但会增加一次请求被拆成多轮 forward 的状态管理成本。

## 6. 常见错误

**把 generation-first 理解成本轮只有 generation。** 症状是看到日志里同时有 context chunk 和 generation，误以为配置没生效。实际生效点只是 generation 先扣预算。

**把 `enable_chunked_prefill` 当成默认开启。** TLLM 这个字段默认是 `False`。在未开启 chunking 时，长 prefill 放不进剩余预算就只能等待，不能产生 partial context chunk。

**把 PD 的 `schedule_style=generation_first` 当成聚合 scheduler 的 generation-first。** 前者属于 disagg server 的编排策略，后者是 worker 内 `_schedule_loop()` 的两阶段准入。两者名字相同，作用域不同。

**把 token budget 不足当成 KV 不足。** 预算不足的直接结果是本轮 STOP；KV 页不足才进入 suspend、recompute pause、self-evict 这些路径。

**忽略 draft tokens。** generation 的预算不是恒等于 1；投机解码下是 beam width 加 draft length，最后一个 context chunk 也可能额外带 draft KV。

## 7. 自检

**问题 1：TLLM V2 先调度 generation，为什么还能和 prefill chunk 混批？**

答：generation 只是先扣预算。Phase 2 用剩余预算调度 context，开启 chunked prefill 后，context 被裁成可放入的 chunk，最后与 generation 一起写入同一个 `ScheduledRequests` 并执行 forward。

**问题 2：未开启 chunked prefill 时，context 放不进剩余预算会怎样？**

答：这个 context 请求本轮等待，完整 token 数必须能放进剩余预算。scheduler 不会自动把它切成 partial chunk。

**问题 3：PD 分离后，P 侧的 chunk 和 D 侧的 decode 在同一 batch 吗？**

答：不在。它们是独立 worker/engine。chunk 与 P 侧其他 prefill work 同 batch，decode 在 D 侧推进，两侧通过 KV transfer 和请求编排衔接。

**问题 4：SGLang 的 `--prefill-decode-interval` 是不是永久切到 decode-first？**

答：不是。它设置一轮 prefill 后连续推进多少轮 decode 的节奏，计数结束后重新尝试 prefill。它不等价于 TLLM 本地 scheduler 的两阶段准入。

**问题 5：`max_num_tokens` 是否只限制 prefill？**

答：不是。generation 的 beam 和 draft tokens、context chunk、最后一个 context chunk 的 draft 部分都从这里扣。它是整轮 forward 的输入 token 预算。

## 8. 可运行 artifact

完整代码见 [`examples/tensorrt_llm_generation_first_budget.py`](examples/tensorrt_llm_generation_first_budget.py)。它只模拟 scheduling 决策，不依赖 GPU、CUDA 或 TensorRT-LLM wheel。

```python
"""TensorRT-LLM V2 scheduler: generation-first budget, not generation-only."""

from dataclasses import dataclass


@dataclass
class Request:
    request_id: str
    kind: str
    remaining_tokens: int
    beam_width: int = 1
    draft_tokens: int = 0

    def generation_cost(self) -> int:
        # V2 scheduler: committed generation tokens = beam width + draft tokens.
        return self.beam_width + self.draft_tokens


@dataclass
class Plan:
    generation: list[tuple[str, int, int]]
    context: list[tuple[str, int, int]]
    deferred: list[str]
    used_tokens: int
    remaining: dict[str, int]


def schedule(
    requests: list[Request], max_num_tokens: int, chunked_prefill: bool
) -> Plan:
    used = 0
    generation: list[tuple[str, int, int]] = []
    context: list[tuple[str, int, int]] = []
    deferred: list[str] = []

    # Phase 1: generation requests claim the token budget first.
    for req in requests:
        if req.kind != "generation":
            continue
        cost = req.generation_cost()
        if used + cost > max_num_tokens:
            break
        used += cost
        generation.append((req.request_id, cost, cost))

    # Phase 2: only the remaining budget is available to context requests.
    for req in requests:
        if req.kind != "context":
            continue
        free = max_num_tokens - used
        if free == 0:
            deferred.append(req.request_id)
            continue

        if chunked_prefill:
            take = min(req.remaining_tokens, free)
        else:
            if req.remaining_tokens > free:
                deferred.append(req.request_id)
                continue
            take = req.remaining_tokens

        if take <= 0:
            deferred.append(req.request_id)
            continue

        used += take
        context.append((req.request_id, take, req.remaining_tokens))
        req.remaining_tokens -= take

    return Plan(
        generation=generation,
        context=context,
        deferred=deferred,
        used_tokens=used,
        remaining={r.request_id: r.remaining_tokens for r in requests},
    )


def print_plan(title: str, plan: Plan) -> None:
    print(title)
    print("phase1 generation:", ", ".join(f"{rid}:{take}/{cost}" for rid, take, cost in plan.generation))
    print("phase2 context:   ", ", ".join(f"{rid}:{take}/{total}" for rid, take, total in plan.context))
    print("deferred:         ", ", ".join(plan.deferred) or "-")
    print(f"used tokens:       {plan.used_tokens}")
    print("remaining:        ", ", ".join(f"{rid}={left}" for rid, left in plan.remaining.items()))
    print()


for chunking in (True, False):
    requests = [
        Request("G1", "generation", remaining_tokens=1, beam_width=1, draft_tokens=1),
        Request("G2", "generation", remaining_tokens=1, beam_width=1, draft_tokens=0),
        Request("C1", "context", remaining_tokens=20),
        Request("C2", "context", remaining_tokens=8),
    ]
    result = schedule(requests, max_num_tokens=16, chunked_prefill=chunking)
    print_plan("chunked prefill" if chunking else "without chunking", result)
```

运行：

```bash
python3 inference-scheduling/examples/tensorrt_llm_generation_first_budget.py
```

预期输出：

```text
chunked prefill
phase1 generation: G1:2/2, G2:1/1
phase2 context:    C1:13/20
deferred:          C2
used tokens:       16
remaining:         G1=1, G2=1, C1=7, C2=8

without chunking
phase1 generation: G1:2/2, G2:1/1
phase2 context:    C2:8/8
deferred:          C1
used tokens:       11
remaining:         G1=1, G2=1, C1=20, C2=0
```

关键源码：

- `tensorrt_llm/_torch/pyexecutor/scheduler/scheduler_v2.py`：`_schedule_loop()`、`_try_schedule_generation()`、`_try_schedule_context_chunked()`
- `tensorrt_llm/_torch/pyexecutor/scheduler/scheduler.py`：`ScheduledRequests`
- `tensorrt_llm/llmapi/llm_args.py`：`use_kv_cache_manager_v2`、`enable_chunked_prefill`
- `tensorrt_llm/llmapi/disagg_utils.py`：`DisaggServerConfig.schedule_style`
