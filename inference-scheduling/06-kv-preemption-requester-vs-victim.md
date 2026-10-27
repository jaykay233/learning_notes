# KV 抢占：requester 与 victim 不是同一个角色

> 源码快照：vLLM `bf4b3b5cb86e1b4ccff23906c0dae46309c935e8`、SGLang `635430d07378e2b12381353781a23b0704969909`、TensorRT-LLM `0372799d1543ef5920c1318b829cf127800dbcb9`（2026-10-27）。
> 本文只讲一个判断：谁发现 KV 不够，与谁被选为抢占对象，是两个独立的调度决定。具体的状态机和 KV 分层见上一篇。

```text
本次讲解位置
章节：inference-scheduling
小节：KV allocation failure 中的 requester 与 victim
知识点：等待、抢占他人、self-preempt 的判定边界
上次：TensorRT-LLM EP/EPLB 的 logical expert → physical slot 映射
下次：TensorRT-LLM ADP router、phase intent 与 dummy request
PTX：不适用
```

## 为什么现在讲这个

上一篇把 TensorRT-LLM 的 KV 压力路径拆成 suspend、recompute pause 和 self-evict，但“走到 self-evict”不等于“一开始就把当前请求选成 victim”。如果把这两件事混在一起，就会从日志里读出错误因果：例如请求 B 需要新页，实际被暂停的是尾部的 C，B 却继续解码；或者 GNE 下根本没有 victim，只是新请求没有获准进入 batch。

这个区分直接决定三个可观察行为：谁的状态被重置、谁的 KV 被释放、下一轮谁先获得重新调度机会。下面统一使用两个词：

- **requester**：本次需要新增 KV pages、但当前容量不足的请求或 decode batch。
- **victim**：调度器实际选中，要求它暂停或放弃 KV 的请求。

`requester` 和 `victim` 可以相同，但这不是默认规则。

## 1. 心智模型：先处理“谁来付”，再处理“怎么恢复”

KV 分配失败的共同起点是：

```text
requester needs more pages
  -> free pages enough?
       yes: allocate, continue
       no : apply policy
              no preemption        -> requester waits / is not admitted
              choose victim V
                V != requester     -> release/suspend V, retry requester
                V == requester     -> requester gives up its own pages and stops this pass
```

这里有两条常被混为一谈的路径。

```text
waiting request admission
  new request tries to enter running batch

running request incremental allocation
  an already-running request needs its next decode/context pages
```

vLLM FCFS 的普通路径主要在后一条路径做抢占。SGLang 除了 decode batch 的 retraction，还可能在准入等待请求时按 priority 主动抢占低优先级 running 请求。TensorRT-LLM 则先由 capacity policy 决定是否允许抢占。

“自己回退”确实存在，但它对应的是：

```text
policy(requester, running candidates) == requester
```

而不是：

```text
requester has a KV shortage
```

后者只是触发条件，前者才是 victim 选择结果。

## 2. 三个框架分别怎么选

### vLLM V1：running allocation 失败后按策略选 victim

vLLM V1 在 RUNNING 循环里调用 `kv_cache_manager.allocate_slots()`。返回 `None` 后：

- FCFS policy 选择 `self.running[-1]`，即 running 队列尾部；
- priority policy 选择 `max(self.running, key=lambda r: (r.priority, r.arrival_time))`。vLLM 中 priority 数值越小越优先，所以这里选数值最大的低优先级请求；同 priority 时选更晚到达的请求；
- 如果 victim 不是 requester，移除 victim 后重试 allocation；
- 如果 victim 就是 requester，停止当前 scheduling pass。

`_preempt_request()` 会释放请求的 blocks、把 `num_computed_tokens` 重置为 0，并把它 prepend 到 waiting。也就是说，vLLM 里被抢的请求确实回到 waiting，并从 context 重新计算；触发者不一定是被抢者。

源码：[scheduler.py 的 victim 选择](https://github.com/vllm-project/vllm/blob/bf4b3b5cb86e1b4ccff23906c0dae46309c935e8/vllm/v1/core/sched/scheduler.py#L915-L970)、[`_preempt_request()`](https://github.com/vllm-project/vllm/blob/bf4b3b5cb86e1b4ccff23906c0dae46309c935e8/vllm/v1/core/sched/scheduler.py#L1773-L1812)。

一个新 waiting 请求因为显存不足进不来时，不会仅凭“新请求需要 KV”就让 FCFS scheduler 立即踢掉 running 请求；它先在 waiting 里等。priority 在这里首先影响等待队列顺序和 running victim 的选择，不等于任意新请求都能无条件抢占已有 running 请求。

### SGLang：decode retraction 是 batch 级决策

SGLang 在 decode 前调用 `check_decode_mem()` 检查下一轮整批请求是否放得下。失败后进入 `retract_decode()`，按 `_get_decode_retraction_order()` 的顺序从最不偏好的请求开始 retract。默认 `length` 策略的 key 是：

```text
(output_len, -input_len)
```

代码对 key 做 reverse sort，然后从列表尾部 pop。其结果优先 retract：

- output 更短的请求；
- output 相同时，input 更长的请求。

这与“当前哪个请求申请新页”不是同一件事：decode 容量检查针对 batch，requester 应理解为“decode batch”，随后从 batch 成员中选择 victim。被 retract 的请求回到队列，可能需要重新 prefill，或从已有 KV backup 恢复。

SGLang 还支持 priority 准入抢占。`preempt_to_schedule()` 在等待请求的 priority 高于 running 请求且超过阈值时，可以释放低优先级 running 请求，让新的高优先级请求进入。这是等待请求主动触发 victim 的路径，和默认 FCFS 的等待行为不同。

源码：[`retract_decode()`](https://github.com/sgl-project/sglang/blob/635430d07378e2b12381353781a23b0704969909/python/sglang/srt/managers/schedule_batch.py#L3451-L3520)、[默认长度顺序](https://github.com/sgl-project/sglang/blob/635430d07378e2b12381353781a23b0704969909/python/sglang/srt/managers/schedule_batch.py#L3540-L3580)、[priority 准入抢占](https://github.com/sgl-project/sglang/blob/635430d07378e2b12381353781a23b0704969909/python/sglang/srt/managers/schedule_policy.py#L1642-L1715)。

### TensorRT-LLM：先看 capacity policy 是否允许抢占

TensorRT-LLM 不能脱离 `CapacitySchedulerPolicy` 回答这个问题。

`GUARANTEED_NO_EVICT` 的语义是：已经开始的请求不会被暂停。调度器在准入时按请求完成所需 blocks 做预留，因此不应先启动一个之后必然因 KV 不足而被踢掉的请求。容量不够时，新请求保持 waiting，不产生 victim。

`MAX_UTILIZATION` 则会尽量填满 batch，容量不足时从 active request 队列尾部暂停一个 started request。尾部可能是别的请求，也可能正好是当前 requester；所以代码路径相同，观感却可能是“抢占别人”或“自己回退”。

PyTorch 的 `KVCacheV2Scheduler` 在当前快照中只支持 `MAX_UTILIZATION`；即使传入其他 policy，也会改成 MAX。V2 的 `_try_schedule_generation()` 先尝试 suspend/recompute pause，最后才 self-evict。GNE 的“不抢占、保持 waiting”应归到 capacity scheduler 语义，不能直接套到 V2 的强制 MAX 路径上。

源码：[GNE 与 MAX 的 capacity policy 分发](https://github.com/NVIDIA/TensorRT-LLM/blob/0372799d1543ef5920c1318b829cf127800dbcb9/cpp/tensorrt_llm/batch_manager/capacityScheduler.cpp#L514-L640)、[V2 的 allocation → eviction → recompute → self-evict 顺序](https://github.com/NVIDIA/TensorRT-LLM/blob/0372799d1543ef5920c1318b829cf127800dbcb9/tensorrt_llm/_torch/pyexecutor/scheduler/scheduler_v2.py#L1470-L1535)、[V2 只接受 MAX_UTILIZATION](https://github.com/NVIDIA/TensorRT-LLM/blob/0372799d1543ef5920c1318b829cf127800dbcb9/tensorrt_llm/_torch/pyexecutor/scheduler/scheduler_v2.py#L207-L212)。

## 3. 横向对照

| 路径 | 谁触发容量检查 | victim 选择 | requester 是否可能成为 victim | 典型状态变化 |
|---|---|---|---|---|
| vLLM V1 FCFS | running request 的 allocation 返回 `None` | `running[-1]` | 是，requester 正好在尾部时 | victim 清空 KV，`num_computed_tokens=0`，回到 waiting |
| vLLM V1 priority | running request 的 allocation 返回 `None` | 最大 `(priority, arrival_time)` | 是，requester 本身是最大键时 | 同上 |
| SGLang decode length | 下一轮 decode batch 不满足容量 | 最短 output、最长 input 优先 | 不按单个 requester 判断；任一 batch 成员可被选出 | 被 retract 的请求回队，之后重算或恢复 |
| SGLang priority 准入 | waiting 高优先级请求无法进入 | 低优先级 running，达到阈值才抢 | requester 通常仍是等待者，不是 running victim | running victim 被释放，高优先级请求获准 |
| TensorRT-LLM GNE | 新请求无法满足完成所需 blocks | 不选 | 否 | 新请求继续 waiting，running 不被暂停 |
| TensorRT-LLM MAX_UTILIZATION | active batch 容量不足 | 从 started active 队列尾部暂停 | 是，从尾部扫描时可能命中自己 | 暂停/搬走 victim，之后 resume 或 recompute |

这张表的核心不是背策略名，而是确认三件事：抢占是否存在、victim 如何选、requester 是否拥有豁免权。三个答案都不由“谁报了 allocation failure”单独决定。

## 4. 完整可运行模拟

文件：`inference-scheduling/examples/kv_preemption_victim_policy.py`

这个程序只模拟调度决策，不分配 CUDA pages，也不替代三个框架的真实状态机。它把 victim 选择从请求状态中拆出来，使同一个 requester 在 FCFS、priority、length 和 tail policy 下得到不同结果。

### 完整代码

```python
"""Compare who triggers KV preemption with who is selected as its victim.

This is a CPU-only policy trace. It models scheduler decisions, not CUDA page
allocation or the full request state machines of vLLM, SGLang, or TensorRT-LLM.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Literal


@dataclass
class Request:
    rid: str
    pages: int
    priority: int = 0
    arrival: int = 0
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class Pool:
    capacity: int
    used: int

    @property
    def free(self) -> int:
        return self.capacity - self.used

    def fits(self, pages: int) -> bool:
        return pages <= self.free

    def release(self, pages: int) -> None:
        self.used -= pages

    def allocate(self, pages: int) -> None:
        self.used += pages


@dataclass
class Outcome:
    requester: str
    scheduled: bool
    requester_state: str
    victims: list[str] = field(default_factory=list)
    waiting: list[str] = field(default_factory=list)
    events: list[str] = field(default_factory=list)


def preempt(
    victim: Request,
    running: list[Request],
    waiting: list[Request],
    pool: Pool,
    events: list[str],
) -> None:
    running.remove(victim)
    pool.release(victim.pages)
    waiting.insert(0, victim)
    events.append(
        f"preempt {victim.rid}: release {victim.pages} pages, free={pool.free}"
    )


def vllm_running_allocation(
    requester: Request,
    running: list[Request],
    pool: Pool,
    need: int,
    choose_victim: Callable[[list[Request]], Request],
    *,
    policy_name: str,
) -> Outcome:
    """Model vLLM V1's running-request allocate_slots retry loop."""
    events = [
        f"{policy_name}: requester={requester.rid}, need={need}, "
        f"free={pool.free}"
    ]
    waiting: list[Request] = []
    victims: list[str] = []

    while not pool.fits(need):
        if not running:
            events.append("no running victim remains")
            break
        victim = choose_victim(running)
        preempt(victim, running, waiting, pool, events)
        victims.append(victim.rid)
        if victim.rid == requester.rid:
            events.append("requester selected itself: stop current scheduling pass")
            return Outcome(
                requester.rid,
                False,
                "self_preempted, requeued at waiting front",
                victims,
                [req.rid for req in waiting],
                events,
            )

    if not pool.fits(need):
        events.append("allocation still fails")
        return Outcome(
            requester.rid,
            False,
            "cannot schedule this pass",
            victims,
            [req.rid for req in waiting],
            events,
        )

    pool.allocate(need)
    events.append(f"allocate {need} pages after victim selection: free={pool.free}")
    return Outcome(
        requester.rid,
        True,
        "keeps running and is scheduled this pass",
        victims,
        [req.rid for req in waiting],
        events,
    )


def sg_decode_retraction(
    running: list[Request],
    pool: Pool,
    need: int,
    *,
    retraction_policy: Literal["length", "priority"] = "length",
) -> Outcome:
    """Model SGLang's batch-level decode capacity check and retraction loop."""
    events = [f"decode batch needs {need} pages, free={pool.free}"]
    waiting: list[Request] = []
    victims: list[str] = []

    def length_key(req: Request) -> tuple[int, int]:
        return (req.output_tokens, -req.input_tokens)

    while not pool.fits(need) and len(running) > 1:
        if retraction_policy == "priority":
            # Lower numeric priority is more important. Break ties by the
            # least-preferred length order described below.
            sorted_indices = sorted(
                range(len(running)),
                key=lambda i: (
                    running[i].priority,
                    *length_key(running[i]),
                ),
                reverse=True,
            )
        else:
            sorted_indices = sorted(
                range(len(running)),
                key=lambda i: length_key(running[i]),
                reverse=True,
            )
        victim = running[sorted_indices[-1]]
        preempt(victim, running, waiting, pool, events)
        victims.append(victim.rid)

    if not pool.fits(need):
        events.append("even one remaining decode row does not fit")
        return Outcome(
            "decode_batch",
            False,
            "decode batch cannot make progress",
            victims,
            [req.rid for req in waiting],
            events,
        )

    pool.allocate(need)
    events.append(f"allocate {need} pages for the remaining batch: free={pool.free}")
    return Outcome(
        "decode_batch",
        True,
        "remaining running rows continue decoding",
        victims,
        [req.rid for req in waiting],
        events,
    )


def tllm_capacity_policy(
    requester: Request,
    running: list[Request],
    pool: Pool,
    need: int,
    *,
    policy: Literal["GNE", "MAX_UTIL"],
) -> Outcome:
    """Model only the victim-selection contract of TensorRT-LLM policies."""
    events = [f"{policy}: requester={requester.rid}, need={need}, free={pool.free}"]
    if pool.fits(need):
        pool.allocate(need)
        events.append(f"allocate {need} pages: free={pool.free}")
        return Outcome(requester.rid, True, "scheduled", [], [], events)

    if policy == "GNE":
        events.append("GUARANTEED_NO_EVICT: do not pause another request")
        return Outcome(
            requester.rid,
            False,
            "not admitted; remains in the waiting queue",
            [],
            [requester.rid],
            events,
        )

    # MAX_UTILIZATION gives up one active request from the tail. If the
    # requester is itself at the tail, it becomes the victim.
    ordered_started = [req for req in running if req.pages > 0]
    if not ordered_started:
        events.append("MAX_UTILIZATION: no started request to pause")
        return Outcome(requester.rid, False, "deferred", [], [], events)

    victim = ordered_started[-1]
    waiting: list[Request] = []
    events.append("MAX_UTILIZATION: scan started requests from the tail")
    preempt(victim, running, waiting, pool, events)
    if victim.rid == requester.rid:
        return Outcome(
            requester.rid,
            False,
            "requester paused itself at the active tail",
            [victim.rid],
            [req.rid for req in waiting],
            events,
        )
    if not pool.fits(need):
        events.append("one tail pause is insufficient; defer without more victims")
        return Outcome(
            requester.rid,
            False,
            "remains active, deferred",
            [victim.rid],
            [req.rid for req in waiting],
            events,
        )

    pool.allocate(need)
    events.append(f"allocate {need} pages: free={pool.free}")
    return Outcome(
        requester.rid,
        True,
        "keeps running and is scheduled this pass",
        [victim.rid],
        [req.rid for req in waiting],
        events,
    )


def show(title: str, outcome: Outcome) -> None:
    print(title)
    for event in outcome.events:
        print(f"  {event}")
    print(f"  requester={outcome.requester}")
    print(f"  scheduled={outcome.scheduled}")
    print(f"  requester_state={outcome.requester_state}")
    print(f"  victims={outcome.victims}")
    print(f"  waiting={outcome.waiting}\n")


def fcfs_victim(running: list[Request]) -> Request:
    return running[-1]


def priority_victim(running: list[Request]) -> Request:
    return max(running, key=lambda req: (req.priority, req.arrival))


show(
    "1. vLLM FCFS: requester B evicts tail request C",
    vllm_running_allocation(
        requester=Request("B", pages=3),
        running=[Request("A", pages=2), Request("B", pages=3), Request("C", pages=4)],
        pool=Pool(capacity=9, used=9),
        need=2,
        choose_victim=fcfs_victim,
        policy_name="vLLM FCFS",
    ),
)

show(
    "2. vLLM FCFS: requester is the tail and preempts itself",
    vllm_running_allocation(
        requester=Request("C", pages=3),
        running=[Request("A", pages=2), Request("B", pages=3), Request("C", pages=3)],
        pool=Pool(capacity=8, used=8),
        need=2,
        choose_victim=fcfs_victim,
        policy_name="vLLM FCFS",
    ),
)

show(
    "3. vLLM priority: C is the lowest-priority victim, not tail D",
    vllm_running_allocation(
        requester=Request("B", pages=3, priority=5),
        running=[
            Request("A", pages=2, priority=0),
            Request("C", pages=4, priority=9),
            Request("B", pages=3, priority=5),
            Request("D", pages=2, priority=1),
        ],
        pool=Pool(capacity=11, used=11),
        need=2,
        choose_victim=priority_victim,
        policy_name="vLLM priority",
    ),
)

show(
    "4. SGLang length retraction: C has the least output and longest input",
    sg_decode_retraction(
        running=[
            Request("A", pages=2, input_tokens=100, output_tokens=5),
            Request("B", pages=3, input_tokens=1_000, output_tokens=20),
            Request("C", pages=4, input_tokens=2_000, output_tokens=2),
        ],
        pool=Pool(capacity=9, used=9),
        need=2,
    ),
)

show(
    "5. TensorRT-LLM GNE: requester waits without taking a victim",
    tllm_capacity_policy(
        requester=Request("B", pages=3),
        running=[Request("A", pages=2), Request("C", pages=7)],
        pool=Pool(capacity=9, used=9),
        need=2,
        policy="GNE",
    ),
)

show(
    "6. TensorRT-LLM MAX_UTILIZATION: pause tail C for requester B",
    tllm_capacity_policy(
        requester=Request("B", pages=3),
        running=[Request("A", pages=2), Request("B", pages=3), Request("C", pages=4)],
        pool=Pool(capacity=9, used=9),
        need=2,
        policy="MAX_UTIL",
    ),
)
```

### 运行方式

```bash
python3 inference-scheduling/examples/kv_preemption_victim_policy.py
```

### 预期输出

```text
1. vLLM FCFS: requester B evicts tail request C
  vLLM FCFS: requester=B, need=2, free=0
  preempt C: release 4 pages, free=4
  allocate 2 pages after victim selection: free=2
  requester=B
  scheduled=True
  requester_state=keeps running and is scheduled this pass
  victims=['C']
  waiting=['C']

2. vLLM FCFS: requester is the tail and preempts itself
  vLLM FCFS: requester=C, need=2, free=0
  preempt C: release 3 pages, free=3
  requester selected itself: stop current scheduling pass
  requester=C
  scheduled=False
  requester_state=self_preempted, requeued at waiting front
  victims=['C']
  waiting=['C']

3. vLLM priority: C is the lowest-priority victim, not tail D
  vLLM priority: requester=B, need=2, free=0
  preempt C: release 4 pages, free=4
  allocate 2 pages after victim selection: free=2
  requester=B
  scheduled=True
  requester_state=keeps running and is scheduled this pass
  victims=['C']
  waiting=['C']

4. SGLang length retraction: C has the least output and longest input
  decode batch needs 2 pages, free=0
  preempt C: release 4 pages, free=4
  allocate 2 pages for the remaining batch: free=2
  requester=decode_batch
  scheduled=True
  requester_state=remaining running rows continue decoding
  victims=['C']
  waiting=['C']

5. TensorRT-LLM GNE: requester waits without taking a victim
  GNE: requester=B, need=2, free=0
  GUARANTEED_NO_EVICT: do not pause another request
  requester=B
  scheduled=False
  requester_state=not admitted; remains in the waiting queue
  victims=[]
  waiting=['B']

6. TensorRT-LLM MAX_UTILIZATION: pause tail C for requester B
  MAX_UTIL: requester=B, need=2, free=0
  MAX_UTILIZATION: scan started requests from the tail
  preempt C: release 4 pages, free=4
  allocate 2 pages: free=2
  requester=B
  scheduled=True
  requester_state=keeps running and is scheduled this pass
  victims=['C']
  waiting=['C']

```

## 5. 逐块解释

`Request` 只保留决定策略的字段：pages、priority、arrival、input/output length。`Pool` 维护 capacity 和 used，`free` 是两者之差。

`preempt()` 是所有策略共用的“付款”动作：

```python
running.remove(victim)
pool.release(victim.pages)
waiting.insert(0, victim)
```

它先把 victim 移出 running，再释放 pages，最后放入 waiting。因此 requester 是否能继续，要看释放后 `pool.fits(need)`，而不是看“曾经失败过”。

`vllm_running_allocation()` 对应 vLLM 的 retry 结构。它在 `while not pool.fits(need)` 内调用 `choose_victim()`，并在 victim 等于 requester 时立即停止。这个停止条件非常关键：自己的 old KV 虽然被释放了，但当前请求已经进入 preempted/waiting 语义，不能把这次 allocation 当作成功。

`sg_decode_retraction()` 的 requester 写为 `decode_batch`。默认 length 策略计算：

```python
length_key = (req.output_tokens, -req.input_tokens)
sorted_indices = sorted(
    range(len(running)),
    key=lambda i: length_key(running[i]),
    reverse=True,
)
victim = running[sorted_indices[-1]]
```

所以列表末尾是最不偏好的成员：output 最短；如果 output 相同，则 `-input` 最小，也就是 input 最长。

`tllm_capacity_policy()` 明确分成 GNE 和 MAX_UTIL。GNE 只返回“没有 victim，requester 留在 waiting”；MAX_UTIL 从 started active 队列尾部暂停。这个模拟针对 policy contract；真实 TensorRT-LLM V2 还会按上一篇的 suspend、recompute 和 self-evict 阶梯处理 KV tier。

## 6. 六个场景的数值 trace

所有场景的 requester 都需要 2 pages：

| 场景 | 初始容量 | victim | 释放后 free | requester 结果 |
|---|---:|---|---:|---|
| vLLM FCFS，B 请求，C 是尾部 | 9 used / 9 | C，4 pages | `0+4=4` | 再分配 2，剩 2；B 继续 |
| vLLM FCFS，C 自己请求 | 8 used / 8 | C，3 pages | `0+3=3` | 不重新分配；C 本轮失败并回 waiting |
| vLLM priority，C 是最大 priority 键 | 11 used / 11 | C，4 pages | `0+4=4` | 再分配 2，剩 2；尾部 D 不被选 |
| SGLang length，C 短输出长输入 | 9 used / 9 | C，4 pages | `0+4=4` | batch 分配 2，剩 2；A/B 继续 |
| TensorRT-LLM GNE | 9 used / 9 | 无 | 0 | requester 不获准，继续 waiting |
| TensorRT-LLM MAX_UTIL | 9 used / 9 | 尾部 C，4 pages | `0+4=4` | 再分配 2，剩 2；B 继续 |

第二个场景最容易被误读。C 的 3 pages 被释放后，pool 确实有 3 pages free，但 scheduler 已经选择中断 C 的当前 decode pass。free pages 要等后续重新调度，不能反向证明本轮 C 已经被成功调度。

## 7. 常见误区与可观察症状

**把 allocation failure 直接等同 self-preempt。**

症状：日志显示请求 B 申请失败，随后出现 `preempt C`，但下一轮 B 仍在生成。这里 B 是 requester，C 才是 victim。若只盯报错请求，会错误认为 B 的 `num_computed_tokens` 被清零。

**认为尾部 victim 永远等于 self-victim。**

症状：FCFS 或 MAX_UTIL 选择了 `running[-1]`，但 requester 位于中间。只有 requester 自己也在尾部，或前面的候选都不可用时，才会出现自己回退。

**把 `running → waiting` 当成所有框架的统一状态转移。**

症状：vLLM 的 victim 确实会被清空 computed tokens 并放回 waiting；TensorRT-LLM 的暂停请求还可能处于 suspend/HELD/paused 路径，GNE 则是新请求从未获准进入 running。只靠“waiting 数量增加”判断抢占会漏掉这些差异。

**给 SGLang decode retraction 强行指定一个请求 requester。**

症状：一次 capacity failure 连续 retract 两个请求，或者被 retract 的是与触发检查无关的成员。容量检查针对整个 decode batch，victim 顺序才针对单个成员。

**把 priority 的大小方向跨框架复用。**

vLLM 文档明确 smaller priority 更优先；SGLang 有 `schedule_low_priority_values_first` 配置决定方向。看到数值大不等于在所有框架里都代表低优先级，应先确认配置和比较方向。

**把 GNE 当成“先启动，缺页时再等”。**

GNE 的关键是启动请求获得完成所需容量保证，因此正常状态下不会靠中途抢占腾页。要观察的是新请求能否获准，而不是有没有 self-preempt 日志。

## 8. 自测

**1. B 请求新 KV page 失败，日志却显示 C 被 pause，B 怎么可能还继续？**

因为 B 是 requester，C 是 policy 选出的 victim。释放 C 的 pages 后，B 重试 allocation 成功；vLLM FCFS 在 B 不在尾部时就是这条路径。

**2. vLLM FCFS 中 requester 什么时候会 self-preempt？**

选择规则是 `running[-1]`。如果 requester 正好是该尾部请求，`_preempt_request()` 会重置它并停止本轮；反之会先牺牲其他尾部请求。

**3. 为什么 self-preempt 释放了自己的 pages，却不能在同一轮继续分配？**

因为 victim 选择已经改变了 requester 的请求状态：vLLM 把它设为 preempted、清零 computed tokens、放入 waiting。释放 pages 是状态转移的结果，不是让原 decode pass 原地复活的许可。

**4. SGLang 默认 length retraction 优先抢哪类请求？**

优先抢 output 更短的请求；output 相同时，input 更长的请求更先被 retract。这样更偏向保留已经生成较多结果的请求。

**5. TensorRT-LLM 的 GNE 和 MAX_UTILIZATION 最本质的区别是什么？**

GNE 不发生已经启动请求的 pause，容量不足时新请求等待；MAX_UTILIZATION 为了提高 batch 利用率，可以从 active 尾部暂停请求，随后通过 resume 或 recompute 恢复。

## 9. 一句话结论

**KV 不够只说明谁在申请；victim 由调度策略选择。requester 可以抢别人，也可以等，只有在策略选中自己时才发生 self-preempt。**
