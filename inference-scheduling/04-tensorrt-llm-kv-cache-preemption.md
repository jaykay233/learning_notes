# TensorRT-LLM V2 的 KV 压力抢占阶梯

> 源码快照：TensorRT-LLM `0372799d1543ef5920c1318b829cf127800dbcb9`（2026-10-27）。
> 本文讨论 PyTorch backend 默认 V2 scheduler。这里的“抢占”不是取消请求，而是暂停执行、让出 KV，之后从可恢复点继续。

```text
本次讲解位置
章节：inference-scheduling
小节：TensorRT-LLM V2 的 KV pressure fallback ladder
知识点：suspend、HELD、recompute pause、self-evict、victim 顺序
上次：generation-first 与 chunked prefill 混批
下次：EPLB 逻辑专家到物理 expert 的映射
PTX：不适用
```

## 为什么现在讲这个

上一篇把 scheduling 分成 token budget 和 KV page 两类资源，但没有解释 allocation 失败后请求到底怎么继续。这里最容易错的是把所有“这一轮没排上”都叫成抢占：token budget 不够时，V2 scheduler 直接 `STOP`，不会为了腾 token 去暂停别的请求；只有 target/draft KV allocation 失败，才进入 suspend、recompute pause、self-evict 这条阶梯。

把这条阶梯拆清楚，才能预测一个请求是短暂等下一轮、回退成 context 重新 prefill，还是当前 generation 自己让出 GPU pages。它也能解释为什么多级 KV cache、PD 分离和 connector transfer 会改变抢占行为。

## 1. 心智模型：先看缺什么资源

一轮 generation 准入按下面的顺序走：

```text
token budget
  -> target + draft KV allocation
  -> ordinary suspend / eviction
  -> destructive recompute pause
  -> self-evict current generation
  -> STOP this scheduling pass
```

四个结果对应不同状态：

| 位置 | 失败条件 | 结果 | 是否释放别人的 KV |
|---|---|---|---|
| token budget | 当前 generation cost 放不进本轮预算 | 立即 `STOP` | 否 |
| KV allocation | target 或 draft pool 不够 | 进入 fallback ladder | 可能 |
| ordinary suspend | 需要更多 GPU pages | 暂停 victim；有下级 tier 时搬走 pages | 是，但保留可恢复性 |
| recompute pause | suspend 仍腾不出空间 | 丢掉 victim 的 KV，请求回 context | 是，破坏性释放 |
| self-evict | 前面都失败 | 暂停当前请求并停止本轮 | 释放当前请求自己的 pages |

`_try_schedule_generation()` 先算 generation cost：

```text
beam_width + draft_token_length
```

`budget.can_fit_tokens()` 失败时直接返回 `ScheduleAction.STOP`。源码没有在这里调用 `_try_evict_for_gen()`，所以预算不足不会触发 KV 抢占。预算不足和显存不足是两条不同的失败路径。

## 2. target 与 draft pool 必须原子成功

`_try_allocate_generation()` 先尝试 target KV manager。target 成功后，如果还有 joint draft manager，再尝试 draft manager；draft 失败就回滚 target 的增长，并统一报告 allocation failure。

这个“原子性”很重要：

```text
target 成功，draft 失败
  -> revert target
  -> 对 caller 表现为一次普通 KV allocation failure
  -> 后续 victim 选择和回滚逻辑只有一套
```

如果不回滚，target 已经增长了 pages，但本轮又不能执行 generation，下一轮会看到一对不一致的 target/draft capacity。

## 3. 第一层：ordinary suspend 与 victim 顺序

generation allocation 失败后，`_try_evict_for_gen()` 从 `req_it_end - 1` 向前扫描，也就是优先选择队列尾部较新的 started request。受害者要满足：

- 已经执行过，至少不是尚未开始的 first context chunk；
- 当前 KV cache 仍在 GPU active；
- 不在 `inflight_request_ids`；
- 没有 pending connector load；
- 不在当前已调度或受保护的集合中。

代码每次只 suspend 一个 victim，然后立刻重试 allocation。成功后停止，不再多牺牲一个请求。

“较新优先”不是随便选的。较老的请求往往已经生成更多 token，丢掉或搬走它们的 KV 成本更高；从队尾开始有助于先释放近期请求的 pages，同时减少连续 thrash。

### suspend 不一定真的释放 pages

`suspend_request()` 同时处理 main 和 draft KV manager。如果 GPU 下面还有 cache tier，suspend 可以把 GPU 上的 residency 让出来，页面仍能沿 tier 恢复；这时普通 suspend 就能解决压力。

如果 GPU 是最后一级 cache tier，suspend 后页面会处于 `HELD` 状态。请求暂时不执行，但这些 pages 仍不能被别的请求覆盖，因此 free page 数没有增加。此时继续 suspend 更多请求可能只会得到更多 `HELD`，不能完成 allocation。

这个区别是下面例子里最重要的观察：

```text
tier_below_gpu=True   -> suspend B -> moved 5 pages -> allocation ok
tier_below_gpu=False  -> suspend B -> HELD
                      -> suspend A -> HELD
                      -> still no free pages
```

## 4. 第二层：destructive recompute pause

ordinary suspend 扫完仍失败后，`_try_recompute_pause_for_gen()` 才考虑 destructive preemption。候选者仍然优先从尾部找，但不能是 inflight、pending connector load、`GENERATION_TO_COMPLETE`，也不能是缺少 replay 数据的已完成多模态 prefill。

真正释放 pages 的动作是：

```text
kv_cache_manager.preempt_request(victim)
  -> 释放 victim 的 KV
  -> victim 从 generation 回到 context
  -> executor 之后重新 prefill
  -> 新结果再次进入 generation
```

因此 recompute pause 与普通 suspend 的语义不同：前者保 KV，后者丢 KV。抢占的代价是重复 prefill，换来的是一次真正可复用的 GPU page。

`preempt_request()` 还可能因为 connector 正在保存 victim 的 KV 而返回 false。源码这时会停止继续找第二个 victim；先等这次 save 完成，避免页面被新请求覆盖后，旧 KV 被错误发布到另一个请求的 hash 下。

recompute pause 成功后，scheduler 会立刻重试 allocation。如果仍不够，而且 cache manager 允许继续 eviction，可以再走一次 ordinary suspend；仍不够才继续找下一个 recompute 候选。

## 5. 最后一层：self-evict 当前 generation

如果 suspending victims 和 recompute pause 都没有解决问题，`_try_schedule_generation()` 会检查当前请求自己是否 active，并且没有 pending connector load。满足条件时执行 self-evict：

```text
_suspend_request(current)
  -> append current to evicted
  -> STOP this pass
```

这不是请求失败，也不是取消。它的作用是让当前 generation 暂时让出 GPU pages，使其他请求有机会恢复；当前请求会在后续 pass 里重新参与 scheduling。

源码还避免对已经 suspended 的请求重复 self-evict，因为再次 suspend 是 no-op，不会产生新的 free pages。

## 6. context phase 的抢占边界不同

Phase 2 的 context allocation 失败后，会调用 `preempt_for_pages(req)`。它构造受保护集合：

```text
scheduled_gen + scheduled_ctx + current context request
```

然后从 active request 尾部向前挑一个可 destructive preempt 的 started request。context 路径有两个重要限制：

1. `kv_cache_manager.has_cache_tier_below_gpu` 为 true 时，`_try_preempt_for_pages()` 直接返回 false。下级 tier 能用较便宜的 suspend 保住 KV，就不先做破坏性 preempt。
2. `enable_recompute_pause=False` 时不能抢占。disaggregated generation worker 通过 KV transfer 收到 context KV，没有本地 prefill 输入可重放，所以在 `_util.py` 中明确传了 `enable_recompute_pause=not is_disagg`。

如果 context preemption 成功，调度循环会停止当前 Phase 2，并把刚释放页面的请求记到 `preempted_ids`。下一次扫描会跳过它，防止后面的 context 请求把它牺牲出来的 pages 又抢走。否则同一个请求可能每轮都抢占、每轮都排不上，形成 thrash。

## 7. 用例子看完整决策链

下面模型只模拟 scheduler 的决策，不分配真实 CUDA pages。它故意把两种暂停分开：

- `suspend()`：有下级 tier 时搬走 pages；GPU-only 时只标 `HELD`。
- `recompute_pause()`：真正从 pool 里删掉 pages，把 victim 的 state 改成 `context`。

### 完整可运行代码

文件名：`inference-scheduling/examples/tensorrt_llm_preemption_ladder.py`

"""A CPU model of TensorRT-LLM V2's KV-pressure fallback ladder.

This is a decision trace, not a replacement for KVCacheManagerV2. It models the
important distinction between suspension (pages may move down a cache tier)
and recompute preemption (pages are released, so the request must prefill again).
"""

from dataclasses import dataclass


@dataclass
class Pool:
    capacity: int
    used: int

    @property
    def free(self) -> int:
        return self.capacity - self.used

    def fits(self, amount: int) -> bool:
        return amount <= self.free

    def release(self, amount: int) -> None:
        self.used -= amount


@dataclass
class Victim:
    request_id: str
    pages: int
    started: bool = True
    active_gpu: bool = True
    tier_below_gpu: bool = False
    inflight: bool = False
    connector_loading: bool = False
    suspended: bool = False
    state: str = "generation"


def eligible(victim: Victim) -> bool:
    return (
        victim.started
        and victim.active_gpu
        and not victim.inflight
        and not victim.connector_loading
    )


def recompute_eligible(victim: Victim) -> bool:
    # A previously suspended victim can still be destructively preempted when
    # the cache manager permits eviction; suspension alone did not free pages.
    return victim.started and not victim.inflight and not victim.connector_loading


def suspend(victim: Victim, pool: Pool, events: list[str]) -> bool:
    if not eligible(victim):
        return False
    victim.suspended = True
    victim.active_gpu = False
    if victim.tier_below_gpu:
        pool.release(victim.pages)
        events.append(f"suspend {victim.request_id}: moved {victim.pages} pages, free={pool.free}")
        return True
    # With GPU as the last cache level, suspended pages are HELD and cannot be
    # reclaimed, so this operation does not make room for another request.
    events.append(f"suspend {victim.request_id}: HELD, free={pool.free}")
    return False


def recompute_pause(victim: Victim, pool: Pool, events: list[str]) -> None:
    pool.release(victim.pages)
    victim.suspended = False
    victim.active_gpu = False
    victim.state = "context"
    events.append(
        f"recompute {victim.request_id}: released {victim.pages}, "
        f"state=context, free={pool.free}"
    )


def try_generation(
    *,
    need: int,
    pool: Pool,
    victims: list[Victim],
    budget_ok: bool,
    enable_recompute: bool,
    current_pages: int = 0,
) -> tuple[bool, list[str]]:
    events: list[str] = []
    if not budget_ok:
        events.append("token budget check failed: STOP before KV allocation")
        return False, events

    events.append(f"allocate target+draft: need={need}, free={pool.free}")
    if pool.fits(need):
        pool.used += need
        events.append(f"allocation ok: free={pool.free}")
        return True, events

    events.append("allocation failed: enter suspend/evict ladder")
    for victim in reversed(victims):  # newest first
        if suspend(victim, pool, events) and pool.fits(need):
            pool.used += need
            events.append(f"allocation ok: free={pool.free}")
            return True, events

    if enable_recompute:
        for victim in reversed(victims):
            if not recompute_eligible(victim):
                continue
            recompute_pause(victim, pool, events)
            if pool.fits(need):
                pool.used += need
                events.append(f"allocation ok: free={pool.free}")
                return True, events
    else:
        events.append("recompute pause disabled: cannot destructively reclaim KV")

    if current_pages:
        pool.release(current_pages)
        events.append(f"self-evict current generation: released {current_pages}, free={pool.free}")

    events.append("STOP: no allocation in this iteration")
    return False, events


def show(title: str, result: tuple[bool, list[str]]) -> None:
    ok, events = result
    print(title)
    for event in events:
        print(" ", event)
    print(f"  result: {'scheduled' if ok else 'waiting'}\n")


show(
    "budget exhausted",
    try_generation(
        need=9,
        pool=Pool(capacity=8, used=7),
        victims=[Victim("A", pages=2), Victim("B", pages=5)],
        budget_ok=False,
        enable_recompute=True,
    ),
)

show(
    "KV page exhaustion with a tier below GPU",
    try_generation(
        need=4,
        pool=Pool(capacity=8, used=7),
        victims=[Victim("A", pages=2), Victim("B", pages=5, tier_below_gpu=True)],
        budget_ok=True,
        enable_recompute=True,
    ),
)

show(
    "KV page exhaustion on a GPU-only cache tier",
    try_generation(
        need=4,
        pool=Pool(capacity=8, used=7),
        victims=[Victim("A", pages=2), Victim("B", pages=5)],
        budget_ok=True,
        enable_recompute=True,
    ),
)

show(
    "GPU-only tier with recompute disabled",
    try_generation(
        need=4,
        pool=Pool(capacity=8, used=7),
        victims=[Victim("A", pages=2), Victim("B", pages=3)],
        budget_ok=True,
        enable_recompute=False,
        current_pages=2,
    ),
)

运行：

```bash
python3 inference-scheduling/examples/tensorrt_llm_preemption_ladder.py
```

实际输出：

```text
budget exhausted
  token budget check failed: STOP before KV allocation
  result: waiting

KV page exhaustion with a tier below GPU
  allocate target+draft: need=4, free=1
  allocation failed: enter suspend/evict ladder
  suspend B: moved 5 pages, free=6
  allocation ok: free=2
  result: scheduled

KV page exhaustion on a GPU-only cache tier
  allocate target+draft: need=4, free=1
  allocation failed: enter suspend/evict ladder
  suspend B: HELD, free=1
  suspend A: HELD, free=1
  recompute B: released 5, state=context, free=6
  allocation ok: free=2
  result: scheduled

GPU-only tier with recompute disabled
  allocate target+draft: need=4, free=1
  allocation failed: enter suspend/evict ladder
  suspend B: HELD, free=1
  suspend A: HELD, free=1
  recompute pause disabled: cannot destructively reclaim KV
  self-evict current generation: released 2, free=3
  STOP: no allocation in this iteration
  result: waiting
```

### 逐段对照

第一段先触发 token budget 检查。尽管 pool 也不够，代码没有扫描 victim，直接 `STOP`。

第二段从尾部选择 B。B 有下级 cache tier，`suspend(B)` 把 5 个 pages 从 GPU pool 移走，free 从 1 变成 6；申请 4 个 pages 后剩 2。

第三段仍是先 suspend B、A，但 GPU-only 的 suspend 只产生 `HELD`，free 一直是 1。ordinary 路径失败后，recompute pause 选择 B，释放 5 个 pages，把 B 打回 context，再成功为当前 generation 分配。

第四段关闭 recompute pause，HELD pages 无法被破坏性回收。当前 generation 最终 self-evict，释放自己的 2 个 pages，但本轮仍停止。

## 8. 常见错误

**把 token budget 不足当成抢占触发器。** 症状是觉得“本轮预算满了就应该踢一个 decode 请求”；实际源码直接 `STOP`，不为预算牺牲 KV。

**把 suspend 当成一定释放显存。** 只有存在可落地的下级 tier 或 cache manager 能真正回收 residency 时才腾出 GPU pages。GPU-only 的 `HELD` 状态看起来像暂停，实际不能给新请求使用。

**认为 recompute pause 只是普通 delay。** 它丢掉了 KV，之后必须重新 prefill。观察到 victim 从 generation 回到 context、prompt 被重放，才说明走了这条路径。

**从队首找 victim。** V2 的 ordinary 和 destructive 路径都强调从尾部、较新请求开始，并避开 inflight、connector load 和当前 batch 的受保护请求。

**以为抢占会取消请求。** 请求仍在 active set，只是暂停或 reset；如果观察到 request id 消失、客户端收到取消，那不是这里描述的 preemption。

**在 PD generation worker 上期待 local recompute。** 该 worker 没有可重放的 prefill 输入，源码会关闭 recompute pause。它需要依赖 KV transfer、容量配置或降并发解决压力。

## 9. 自检

**问题 1：generation token budget 放不下时，会扫描 victim 吗？**

答：不会。`budget.can_fit_tokens()` 先返回 false，函数直接 `ScheduleAction.STOP`。

**问题 2：suspend 和 recompute pause 的核心区别是什么？**

答：suspend 尽量保住 KV，有下级 tier 时可以搬走 pages；recompute pause 破坏性释放 KV，请求回 context 后还要重新 prefill。

**问题 3：GPU-only cache tier 上连续 suspend B、A 为什么可能仍然没有 free pages？**

答：suspended pages 进入 `HELD`，仍被原请求占有，不能被新请求覆盖，所以 free 计数不变。

**问题 4：为什么 victim 通常从队尾选？**

答：优先暂停较新的请求，避免先丢掉快完成、已积累大量 KV 的较老请求，并减少反复 thrash。

**问题 5：context preemption 成功后，为什么 Phase 2 要停止并保护刚释放的 pages？**

答：这些 pages 是为一个正在等待的 context 请求付了 recompute 代价换来的；如果继续让后面的 context 分配，就可能再次把该请求挤掉，形成每轮抢占但永远排不上的循环。

## 10. 源码路径

- `tensorrt_llm/_torch/pyexecutor/scheduler/scheduler_v2.py:1455`：`_try_schedule_generation()`，budget 检查与阶梯入口。
- 同文件 `:1534`：`_try_allocate_generation()`，target/draft 原子分配与 target 回滚。
- 同文件 `:1571`：`_suspend_request()`，main/draft manager 的暂停。
- 同文件 `:1588`：`_try_preempt_for_pages()`，context phase 的破坏性抢占边界。
- 同文件 `:1838`、`:1852`：ordinary suspend 与 recompute candidate 的资格判断。
- 同文件 `:1899`：`_try_evict_for_gen()`，从尾部 suspend victim。
- 同文件 `:1940`：`_try_recompute_pause_for_gen()`，释放 KV 并回 context。
- `tensorrt_llm/_torch/pyexecutor/_util.py:4403`：disagg worker 设置 `enable_recompute_pause=False`。
