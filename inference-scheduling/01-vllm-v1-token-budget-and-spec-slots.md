# vLLM V1 的 token budget 与投机解码 slot 预算

> 源码快照：vLLM `c1dc2e9cf992995953548c392f8c67db7283495b`（2026-10-19）。
> 本文只讨论 V1 scheduler 在进入 model runner 前的预算账本；KV cache 分配失败的抢占与 recompute 留给下一节。

```text
本次讲解位置
章节：inference-scheduling
小节：vLLM V1 的 token budget 与投机解码 slot 预算
知识点：max_num_scheduled_tokens、max_num_batched_tokens 和 per-request draft_slots
上次：vLLM V1 running-first 与 chunked prefill 共用 token budget
下次：KV cache allocation 失败后的抢占与 recompute
PTX：不适用
```

## 为什么现在讲这个

上一节只说了 V1 把 prefill chunk 和 decode 都折算成“本轮要计算多少 token”，由 running 请求先吃统一预算。这个模型在普通解码里够用，但在投机解码里会漏掉一类输入行：drafter 可能给每个请求插入额外的 query/draft slot，而这些 slot 不在 `SchedulerOutput.num_scheduled_tokens` 里。

如果不把这两笔账分开，就会把 `max_num_scheduled_tokens` 误当成最终模型输入行数，从而低估 batch 的实际占用；反过来，若沿用旧版 `max_num_batched_tokens - draft_slots * max_num_seqs` 的全局预留，又会在只调度少量请求时高估损失，人为缩小 token budget，导致 batch 欠填充和更高的 TTFT。本节讲清当前 V1 的逐请求扣减规则。

## 1. 心智模型：两本预算账本

V1 没有一个独立的 prefill budget，也没有一个独立的 decode budget。每个请求只维护“已经算到哪”和“应该算到哪”：

```text
demand = 请求当前的 token 上界 - 已计算 token 数
```

Scheduler 每轮把 demand 截成 `n`，并同时维护两项容量：

| 账本 | 初始值 | 记录什么 | 谁消费 |
|---|---:|---|---|
| `token_budget` | `max_num_scheduled_tokens` | 本轮 scheduler 签发、需要模型推进的普通 token 数 | scheduler 的 running/waiting 两段循环 |
| `input_budget` | `max_num_batched_tokens` | 本轮模型输入 batch 的 slot 容量，包括 drafter 追加的 slot | model runner 的输入行布局 |

因此：

```text
sum(num_scheduled_tokens) <= max_num_scheduled_tokens
sum(num_scheduled_tokens + draft_slots_per_request) <= max_num_batched_tokens
```

`draft_slots_per_request` 是每个被调度请求可能追加的输入 slot 上界，不是全局常数。普通解码路径没有投机时它为 0，两项预算退化成一个 token budget。

`max_num_batched_tokens` 表示模型输入 batch 的容量上界。它不表示 KV cache 容量，也不保证本轮 batch 会被填满。

## 2. 两个配置各自的含义

`vllm/config/scheduler.py` 的定义已经把差异写得很直接：

```text
max_num_batched_tokens:
    Maximum number of tokens that can be processed in a single iteration.

max_num_scheduled_tokens:
    Maximum number of tokens that the scheduler may issue in a single iteration.
    This is usually equal to max_num_batched_tokens, but can be smaller in cases
    when the model might append tokens into the batch (such as speculative decoding).
```

当前实现里，未显式设置 `max_num_scheduled_tokens` 时，scheduler 会回退到 `max_num_batched_tokens`；启用投机解码时，`VllmConfig._set_max_num_scheduled_tokens()` 也显式把它设为相同值。默认相等不等于两者永远同义：

| 场景 | 结果 |
|---|---|
| 无投机，`max_scheduled == max_batched` | 两本账完全等价 |
| P-EAGLE，`K=4`，每个已调度请求额外 3 个 slot | `n` 受 `token_budget` 和 `input_budget - 3` 双重约束 |
| 手动把 `max_scheduled` 设小 | scheduler 会主动少签发 token，模型输入容量仍可更大 |
| 手动把 `max_scheduled` 设大 | `input_budget` 先成为实际约束，不会突破模型 batch 容量 |

这也解释了为什么“默认相等”不会让投机 batch 越界：调度每个请求时，会从 `input_budget` 额外扣掉该请求的 draft slots。

## 3. Scheduler 的精确扣减路径

### 3.1 每轮开始时建立两本账

源码位置：`vllm/v1/core/sched/scheduler.py:641-644`。

```python
token_budget = self.max_num_scheduled_tokens
spec = self.vllm_config.speculative_config
draft_slots = spec.max_num_new_slots_for_drafting if spec is not None else 0
input_budget = self.scheduler_config.max_num_batched_tokens
```

`draft_slots` 在本轮开始时算一次。它是当前投机方法对“每个已调度请求”的额外 slot 上界。

### 3.2 running 请求先调度

V1 会先扫描 `self.running`。这不表示“decode first”；running 中同时可能有 decode 请求和尚未完成的 prefill chunk。它表示已经在运行的请求优先于 waiting queue。

源码位置：`vllm/v1/core/sched/scheduler.py:712-767`。

```python
while req_index < len(self.running) and token_budget > 0:
    request = self.running[req_index]
    if input_budget <= draft_slots:
        break

    num_new_tokens = (
        request.num_tokens_with_spec
        + request.num_output_placeholders
        - request.num_computed_tokens
    )
    num_new_tokens = min(
        num_new_tokens, token_budget, input_budget - draft_slots
    )
```

`num_new_tokens` 原本是该请求“还欠模型多少 token”。三道截断分别处理：

1. `token_budget`：本轮 scheduler 还能签发多少普通 token。
2. `input_budget - draft_slots`：给这个请求的 draft slot 留出空间后，还剩多少普通输入位置。
3. 后续还会叠加 max model length、Mamba block 对齐、encoder budget 等请求级约束。

如果 `input_budget <= draft_slots`，剩余容量连一个普通 token 加该请求的额外 slot 都放不下，因此不能再调度下一个请求。

### 3.3 提交请求时扣账

源码位置：`vllm/v1/core/sched/scheduler.py:905-912`；waiting 请求在 `1428-1430` 采用同一记账方式。

```python
num_scheduled_tokens[request_id] = num_new_tokens
token_budget -= num_new_tokens
input_budget -= num_new_tokens + draft_slots
```

两本账的扣减差异是：

```text
token_budget -= n
input_budget -= n + draft_slots
```

waiting 队列的候选预算也使用同一个上界：

```python
request_token_budget = min(token_budget, input_budget - draft_slots)
```

所以 running-first 并不改变预算单位；它只决定谁先拿预算。running 吃剩下的容量会留给 waiting；若本轮发生抢占，waiting 准入会暂停，先处理 preemption 路径。

### 3.4 同一轮内抢占会退还两笔预算

源码位置：`vllm/v1/core/sched/scheduler.py:869-875`。

```python
restored = num_scheduled_tokens.pop(preempted_req_id)
token_budget += restored
input_budget += restored + draft_slots
```

如果某个已经在本轮被标记调度的请求又被抢占，它之前扣掉的普通 token 和 draft slot 都要退回当前轮账本。预算只属于当前 step，每轮 `schedule()` 重新初始化。

### 3.5 最后检查什么

源码位置：`vllm/v1/core/sched/scheduler.py:1474-1488`。

```python
total_num_scheduled_tokens = sum(num_scheduled_tokens.values())
assert total_num_scheduled_tokens <= self.max_num_scheduled_tokens
assert token_budget >= 0
assert input_budget >= 0
```

注意 `total_num_scheduled_tokens` 只统计普通签发 token，不包含 draft slot。例如本节后面的 trace 中它是 14，而输入预算实际预留了 20。

## 4. 投机方法的额外 slot 表

源码位置：`vllm/config/speculative.py:1904-1945`。令 `K = num_speculative_tokens`：

| 算法 | Method | Parallel | 每请求额外 slot | 原因 |
|---|---|---:|---:|---|
| EAGLE3 | `eagle3` | 否 | 0 | 复用已有 query slot |
| P-EAGLE | `eagle3` | 是 | `K - 1` | 一个 query 已存在，只追加其余并行 query |
| DFlash | `dflash` | 是 | 0 | 在独立 draft pass 中生成，不进入 target batch |
| DSpark | `dspark` | 是 | 0 | 同上 |
| MTP | `mtp` | 否 | 0 | 现有 batch 布局已覆盖 |
| N-gram | `ngram` | 否 | 0 | 不增加 target batch 行 |
| 串行 draft model | `draft_model` | 否 | 1 | 自回归 draft 输入保留一个未切分 token |
| PARD | `draft_model` | 是 | `K` | 现有输入不整体平移，K 个 query 位置都要额外 slot |

DFlash/DSpark 的额外 slot 为 0，但它们仍要执行独立 draft pass。该 pass 的 query 行数由 `num_lookahead_tokens` 约束，配置初始化会按

```text
max_draft_seqs = max_num_batched_tokens // num_lookahead_tokens
```

收紧 `max_num_active_seqs`，避免 target batch 看似没超，但 draft pass 一次容纳不了全部活跃请求。源码见 `vllm/config/vllm.py:2363-2390`。

表中的 K 是配置上限。调度阶段用它计算保守的 per-request reserve；不要把本步动态选择的较小验证长度当成已经不需要预留。

## 5. 数值 trace：14 个普通 token 如何占满 20 个输入 slot

条件：

```text
max_num_batched_tokens  = 20
max_num_scheduled_tokens = 20
method                    = P-EAGLE
K                         = 4
draft_slots_per_request   = K - 1 = 3
两个请求的 demand          = 10, 10
```

初始账本：

| 时刻 | `token_budget` | `input_budget` |
|---|---:|---:|
| 初始 | 20 | 20 |

请求 0：

```text
n0 = min(10, token_budget=20, input_budget-draft_slots=20-3)
   = 10

token_budget = 20 - 10 = 10
input_budget = 20 - (10 + 3) = 7
```

请求 1：

```text
n1 = min(10, token_budget=10, input_budget-draft_slots=7-3)
   = min(10, 10, 4)
   = 4

token_budget = 10 - 4 = 6
input_budget = 7 - (4 + 3) = 0
```

最终：

```text
scheduler 签发 token = 10 + 4 = 14
额外的 draft slot   = 3 + 3 = 6
输入 slot 预留      = 14 + 6 = 20
```

这和 vLLM 的 CPU scheduler 测试一致：`tests/v1/core/test_scheduler.py:2138-2159` 在 `max_num_batched_tokens=20`、P-EAGLE `K=4`、两个 prompt 各 10 token 时，断言调度结果为：

```python
{"0": 10, "1": 4}
```

这里的“最终输入 slot”是 scheduler 为本轮预留的容量账本。实际 worker 是否填满 20 行，还取决于输入构造和 padding；不能用 `total_num_scheduled_tokens=14` 推算完整模型 batch 形状。

## 6. 为什么改成逐请求扣减

旧实现会先做全局上界预留：

```text
old_max_num_scheduled_tokens
  = max_num_batched_tokens
    - draft_slots_per_request * max_num_seqs
```

这个公式的问题是把 `max_num_seqs` 当成“本轮一定会调度这么多请求”。实际 batch 可能只有一两个请求，于是大量输入容量被根本不会发生的 draft slot 提前扣掉。

例如：

```text
max_num_batched_tokens = 8192
draft_slots_per_request = 3
max_num_seqs = 128

旧版 scheduler cap = 8192 - 3 * 128 = 7808
新版默认 cap     = 8192
```

新版把额外 slot 记在实际被调度请求的账上：

```text
每调度一个请求，input_budget -= n + draft_slots
```

当请求很多时，两者最终仍会被 `max_num_batched_tokens` 卡住；当请求很少时，新版不会为未调度的请求空扣 slot。提交 `0914ed2e81` 正是把这种全局预留改成按实际调度请求自适应预算的修正。

## 7. 完整可运行示例

文件：`inference-scheduling/examples/vllm_v1_token_budget.py`

```python
#!/usr/bin/env python3
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class RequestDemand:
    req_id: str
    demand: int
    state: str


@dataclass(frozen=True)
class BudgetConfig:
    max_num_scheduled_tokens: int
    max_num_batched_tokens: int
    draft_slots_per_request: int


@dataclass(frozen=True)
class Assignment:
    req_id: str
    scheduled_tokens: int
    draft_slots: int
    reserved_input_slots: int


@dataclass(frozen=True)
class StepTrace:
    assignments: Sequence[Assignment]
    scheduled_token_sum: int
    draft_slot_sum: int
    reserved_input_slot_sum: int
    remaining_scheduled_budget: int
    remaining_input_budget: int


def simulate_schedule(
    config: BudgetConfig,
    requests: Sequence[RequestDemand],
) -> StepTrace:
    if config.max_num_scheduled_tokens <= 0:
        raise ValueError("max_num_scheduled_tokens must be positive")
    if config.max_num_batched_tokens <= 0:
        raise ValueError("max_num_batched_tokens must be positive")
    if config.draft_slots_per_request < 0:
        raise ValueError("draft_slots_per_request cannot be negative")

    token_budget = config.max_num_scheduled_tokens
    input_budget = config.max_num_batched_tokens
    draft_slots = config.draft_slots_per_request

    # vLLM V1 schedules running requests before waiting requests. Both kinds
    # consume the same token budget; the order is the only distinction here.
    ordered_requests = (
        tuple(req for req in requests if req.state == "running")
        + tuple(req for req in requests if req.state == "waiting")
    )

    assignments: list[Assignment] = []
    for request in ordered_requests:
        if request.demand <= 0:
            continue
        if token_budget == 0 or input_budget <= draft_slots:
            break

        num_new_tokens = min(
            request.demand,
            token_budget,
            input_budget - draft_slots,
        )
        if num_new_tokens <= 0:
            break

        assignments.append(
            Assignment(
                req_id=request.req_id,
                scheduled_tokens=num_new_tokens,
                draft_slots=draft_slots,
                reserved_input_slots=num_new_tokens + draft_slots,
            )
        )
        token_budget -= num_new_tokens
        input_budget -= num_new_tokens + draft_slots

    return StepTrace(
        assignments=tuple(assignments),
        scheduled_token_sum=sum(a.scheduled_tokens for a in assignments),
        draft_slot_sum=sum(a.draft_slots for a in assignments),
        reserved_input_slot_sum=sum(a.reserved_input_slots for a in assignments),
        remaining_scheduled_budget=token_budget,
        remaining_input_budget=input_budget,
    )


def main() -> None:
    config = BudgetConfig(
        max_num_scheduled_tokens=20,
        max_num_batched_tokens=20,
        draft_slots_per_request=3,
    )
    requests = (
        RequestDemand(req_id="req-0", demand=10, state="running"),
        RequestDemand(req_id="req-1", demand=10, state="waiting"),
    )

    trace = simulate_schedule(config, requests)
    for item in trace.assignments:
        print(
            f"{item.req_id}: scheduled={item.scheduled_tokens} "
            f"draft_slots={item.draft_slots} "
            f"reserved_input_slots={item.reserved_input_slots}"
        )

    print(f"scheduled_token_sum={trace.scheduled_token_sum}")
    print(f"draft_slot_sum={trace.draft_slot_sum}")
    print(f"reserved_input_slot_sum={trace.reserved_input_slot_sum}")
    print(f"remaining_scheduled_budget={trace.remaining_scheduled_budget}")
    print(f"remaining_input_budget={trace.remaining_input_budget}")

    assert tuple(
        (item.req_id, item.scheduled_tokens) for item in trace.assignments
    ) == (("req-0", 10), ("req-1", 4))
    assert trace.scheduled_token_sum == 14
    assert trace.draft_slot_sum == 6
    assert trace.reserved_input_slot_sum == 20
    assert trace.remaining_scheduled_budget == 6
    assert trace.remaining_input_budget == 0


if __name__ == "__main__":
    main()
```

执行边界：

```bash
python3 inference-scheduling/examples/vllm_v1_token_budget.py
```

预期输出：

```text
req-0: scheduled=10 draft_slots=3 reserved_input_slots=13
req-1: scheduled=4 draft_slots=3 reserved_input_slots=7
scheduled_token_sum=14
draft_slot_sum=6
reserved_input_slot_sum=20
remaining_scheduled_budget=6
remaining_input_budget=0
```

这是 CPU 上的预算模型，不需要 GPU 或安装 vLLM。它验证的是 scheduler 的算术，不验证真实 model runner 的 CUDA 输入布局。

### 代码块映射

| 代码块 | 对应的 scheduler 行为 |
|---|---|
| `BudgetConfig` | `max_num_scheduled_tokens`、`max_num_batched_tokens`、方法特定的 draft slot 上界 |
| `ordered_requests` | running-first：先扫描 running，再扫描 waiting |
| `min(demand, token_budget, input_budget - draft_slots)` | 请求级 token 上限与额外 slot 预留 |
| `token_budget -= n` | 只扣普通签发 token |
| `input_budget -= n + draft_slots` | 同时扣普通输入行与 drafter 追加 slot |
| `reserved_input_slot_sum` | `total_num_scheduled_tokens + 已调度请求数 * draft_slots` 的等价账本 |

示例省略了 KV cache allocation、preemption、prefix cache、encoder budget、Mamba 对齐和 max model length。这些步骤可能进一步缩小 `n`，但不改变本节的两本账关系。

### 作用域与执行边界

- 预算计算发生在 EngineCore 的 scheduler 进程，模型 forward 之前。
- running-first、等待队列和预算扣减由 scheduler 单线程顺序执行，不涉及 GPU warp 或 PTX 同步。
- `SchedulerOutput.num_scheduled_tokens` 把每个请求的 `n` 交给 worker；额外 slot 由具体投机路径在输入构造阶段体现。
- 预算只活在一次 `schedule()` 调用内。下一轮重新从配置值初始化，不继承上一轮剩余额度。
- KV cache 分配失败时可能在同一轮触发抢占；如果 victim 已在当前轮被调度，两本账都会恢复，然后重新安排。

## 8. 常见错误与可观察症状

| 错误理解 | 实际机制 | 可观察症状 |
|---|---|---|
| 把 `max_num_scheduled_tokens` 当成最终模型输入行数 | 它不含每请求 draft slot | 估算 batch shape 偏小，和 worker 的输入容量/CUDA Graph padding 对不上 |
| 用 `draft_slots * max_num_seqs` 做固定全局预留 | 当前实现按实际调度请求扣减 | 少量请求时 token budget 被人为压低，TTFT 变差，`total_num_scheduled_tokens` 远低于配置 |
| 看到 P-EAGLE `K=4` 就按每请求 4 个额外 slot 记账 | 已有 query 复用，额外是 `K-1=3` | 过度预留，batch 欠填 |
| 认为 PARD 和 P-EAGLE 的额外 slot 相同 | PARD 每请求额外 `K`，P-EAGLE 是 `K-1` | 预算少算或多算一个 slot |
| 认为 DFlash/DSpark 的额外 slot 为 0 就等于没有 draft 成本 | 它们走独立 draft pass，用 lookahead 和 active-request 上限约束 | 只看 target batch 容量会误判可同时解码的请求数 |
| 认为 `max_num_scheduled_tokens < max_num_batched_tokens` 会自动把差额变成 draft 空间 | 差额只是 scheduler 少签发的额度，不是自动分配 | 配置了更小 cap 却期待更多投机 slot，结果仍是低填充 |
| 把 worker 内局部的 `max_num_scheduled_tokens` 当成同一个配置字段 | worker 局部变量是当前 batch 的最大单请求 query 长度 | 查日志或读 profile 时把两个层次混在一起 |

## 9. 自检

### Q1. 为什么 V1 需要在 `max_num_batched_tokens` 之外再有 `max_num_scheduled_tokens`？

**答：** V1 的 scheduler 统一签发本轮可推进的普通 token，而 drafter 可能在同一 batch 增补额外 slot。前者限制 scheduler 的签发量，后者限制 worker 的最终输入容量。两者分开后，调度器可以先按普通 token 工作，再按实际调度到的请求扣额外 slot。

### Q2. `SchedulerOutput.total_num_scheduled_tokens=14`，`draft_slots=3`，调度了 2 个请求，模型输入容量账本用了多少？

**答：** `14 + 2 * 3 = 20`。`total_num_scheduled_tokens` 只包含普通签发 token，不包含每请求的额外 slot。

### Q3. 为什么当前默认 `max_num_scheduled_tokens == max_num_batched_tokens` 仍然不会突破输入容量？

**答：** 因为每成功调度一个请求，`input_budget` 会扣掉 `n + draft_slots`。默认相等只是让 scheduler 初始拥有和 worker 一样的容量；per-request 扣减保证最终输入预算不为负。

### Q4. P-EAGLE `K=4`、PARD `K=4`、串行 draft model 的 per-request draft slot 分别是多少？

**答：** P-EAGLE 是 3，PARD 是 4，串行 draft model 是 1。DFlash/DSpark 在 target batch 中是 0，但它们有独立的 draft pass 容量约束。

### Q5. 如果手动把 `max_num_scheduled_tokens` 设成 8192，`max_num_batched_tokens` 设成 4096，谁先限制本轮？

**答：** `max_num_batched_tokens` 先限制。`token_budget` 允许 8192 个普通 token，但 `input_budget` 只有 4096，扣除请求的 draft slot 后不足以继续签发；调度不会突破输入容量。

## 10. 源码索引与下一知识点

| 内容 | 位置 |
|---|---|
| 两个配置字段 | `vllm/config/scheduler.py:48-60` |
| 默认值回退 | `vllm/v1/core/sched/scheduler.py:137-141` |
| 投机配置归一化 | `vllm/config/vllm.py:2335-2390` |
| 两本预算初始化 | `vllm/v1/core/sched/scheduler.py:641-644` |
| running 预算截断 | `vllm/v1/core/sched/scheduler.py:712-767` |
| waiting 预算截断 | `vllm/v1/core/sched/scheduler.py:967-974`、`1173-1224` |
| per-request 扣减 | `vllm/v1/core/sched/scheduler.py:905-912`、`1428-1430` |
| 抢占退还预算 | `vllm/v1/core/sched/scheduler.py:869-875` |
| per-method draft slots | `vllm/config/speculative.py:1904-1945` |
| CPU scheduler 回归测试 | `tests/v1/core/test_scheduler.py:2138-2199` |
| 自适应预算修正 | commit `0914ed2e81` |

下一节沿着 `kv_cache_manager.allocate_slots()` 返回 `None` 的路径，解释抢占 victim 如何选择、request 何时回 waiting、已算 KV 是否丢失以及 preemption recompute 的实际代价。
