# P/D 资源分配与离散事件仿真：从吞吐配平到 SLO 搜索

- 论文：SLO-Aware Compute Resource Allocation for Prefill-Decode Disaggregated LLM Inference
- arXiv：[2603.04716](https://arxiv.org/abs/2603.04716)
- 公众号解读：[PD分离场景中PD实例比例的计算方法](https://mp.weixin.qq.com/s/WosdLM0F9s4QJgYoDdviKw)
- 论文单位：金山云
- 实验基线：DeepSeek-V3.1-Terminus、NVIDIA H200、SGLang v0.5.8
- 示例：[pd_capacity_simulator.py](examples/pd_capacity_simulator.py)

```text
本次讲解位置
章节：inference-scheduling
小节：P/D 分离资源规划
知识点：有效 Prefill 吞吐、P/D 连续实例数、整数配置搜索与离散事件仿真
上次：从总 token 吞吐和单实例吞吐推导 P/D 数量比
下次：用真实 P/D benchmark 曲线和请求 trace 替换示例参数，验证 P95/P99
PTX：不适用；本节讨论集群级队列、服务时间和请求生命周期
```

## 为什么现在讲这个

前面的公式能算出连续值 `N_P*` 和 `N_D*`，但它只回答“平均负载下 P/D 大约是什么比例”。它不能回答：

- `3P4D` 和 `4P4D` 的 TTFT P95/P99 分别是多少；
- 请求突发到达时，P 或 D 队列会不会积压；
- Decode batch 变化后，TPOT 和总吞吐如何一起变化；
- KV 传输是否会把原本平衡的配置重新变成下游瓶颈；
- 连续值向上取整后，哪个整数配置在成本和 SLO 上更合适。

因此，容量规划需要两层模型：

```text
解析公式：快速给出候选 P/D 范围
离散事件仿真：在候选配置上模拟请求排队、批处理和 SLO 分布
```

解析公式负责缩小搜索空间，离散事件仿真负责判断候选配置是否真的满足尾延迟。

---

## 1. 公式解决的问题

设：

| 符号 | 含义 | 单位 |
|---|---|---|
| `L_in` | 平均输入长度 | token/request |
| `L_out` | 平均输出长度 | token/request |
| `TP_total` | 目标总 token 吞吐，包含输入和输出 | token/s |
| `TP_P` | 单个 P 实例的有效 Prefill 吞吐 | input token/s |
| `TP_D` | 单个 D 实例的 Decode 吞吐 | output token/s |
| `N_P` | P 实例数 | 个 |
| `N_D` | D 实例数 | 个 |
| `N_req` | 请求数 | 个 |
| `T_total` | 完成全部请求的时间 | s |

`TP_total` 统计的是完整请求 token：

\[
TP_{total}
= \frac{N_{req}(L_{in}+L_{out})}{T_{total}}
\]

P 侧处理全部输入所需时间为：

\[
T_P=\frac{N_{req}L_{in}}{N_P TP_P}
\]

D 侧生成全部输出所需时间为：

\[
T_D=\frac{N_{req}L_{out}}{N_D TP_D}
\]

P/D 分离后，两个阶段可以流水执行。若两个阶段完全配平：

\[
T_P=T_D=T_{total}
\]

代入 P 侧时间：

\[
\frac{N_{req}L_{in}}{N_P TP_P}
=
\frac{N_{req}(L_{in}+L_{out})}{TP_{total}}
\]

约掉 `N_req` 并移项：

\[
N_P
=
\frac{TP_{total}L_{in}}
{(L_{in}+L_{out})TP_P}
\]

同理：

\[
N_D
=
\frac{TP_{total}L_{out}}
{(L_{in}+L_{out})TP_D}
\]

两式相除得到连续 P/D 比例：

\[
\frac{N_P}{N_D}
=
\frac{L_{in}}{L_{out}}
\cdot
\frac{TP_D}{TP_P}
\]

这组公式没有把 SLO 直接放进比例。SLO 只通过 `TP_P` 和 `TP_D` 进入模型：TTFT 决定 P 的有效吞吐，TPOT 决定 D 的 batch 和吞吐。

---

## 2. TTFT 如何约束 P 的有效吞吐

单 P 实例的最大输入 token 吞吐记为 `TP_{P,max}`。把 P 实例看成一个排队服务节点，则它的服务率是：

\[
\mu=\frac{TP_{P,max}}{L_{in}}
\]

单位是请求/秒。设实际请求到达率为 `λ`，系统利用率为：

\[
\rho=\frac{\lambda}{\mu}
\]

M/M/1 模型把排队时间加服务时间写成：

\[
T_s=\frac{1}{\mu-\lambda}
\]

论文把 TTFT 拆成：

\[
TTFT=T_s+T_{overhead}
\]

因此：

\[
T_s=TTFT-T_{overhead}
\]

代入 `T_s`：

\[
\frac{1}{\mu-\lambda}=TTFT-T_{overhead}
\]

利用 `μ=TP_{P,max}/L_in`，可以得到目标 TTFT 下的有效 Prefill 吞吐：

\[
TP_{P,eff}
=
TP_{P,max}
-
\frac{L_{in}}{TTFT-T_{overhead}}
\]

也可以先计算目标 TTFT 下的最大请求率：

\[
\lambda_{max}=\mu-\frac{1}{TTFT-T_{overhead}}
\]

再得到：

\[
TP_{P,eff}=\lambda_{max}L_{in}
\]

两个条件不能忽略：

```text
TTFT > T_overhead
TP_P,eff > 0
```

`TTFT` 越严格，第二项越大，P 的有效吞吐越低，最终需要的 P 实例越多。

### M/M/1 在这里只是近似

M/M/1 的完整假设是：

- 请求按泊松过程到达；
- 服务时间服从指数分布；
- 一个服务节点一次处理一个请求；
- 队列无限；
- `λ < μ` 时系统稳定。

真实 Prefill 常见的行为并不完全符合这些假设：

- 输入长度可能固定或呈多峰分布，服务时间不一定指数分布；
- chunked prefill 会把一个请求拆成多个计算片段；
- continuous batching 会同时推进多个请求；
- 请求到达可能突发；
- 平均值不能保证 P95/P99。

因此，`TP_{P,eff}` 是容量估算值，不是对单个请求 TTFT 的承诺。

---

## 3. TPOT 如何决定 D 的吞吐

Decode 侧不直接用 M/M/1 算一个固定服务率，而是通过 benchmark 得到：

```text
batch size → TPOT
batch size → Decode 吞吐
```

先找到满足：

\[
TPOT(B^*) \leq TPOT_{target}
\]

的最大 batch `B*`，然后取该点的 Decode 吞吐：

\[
TP_D=Throughput(B^*)
\]

如果 benchmark 直接给出每个 decode step 的时间 `t_D(B)`，也可以用：

\[
TP_D(B)=\frac{B}{t_D(B)}
\]

这里不能只看 `B` 变大后吞吐上升。batch 越大，TPOT 也可能越大，超过 SLO 的吞吐点不能用于容量规划。

---

## 4. 论文场景的数值轨迹

论文参数：

```text
TP_total = 5 MTPM = 5,000,000 token/min
L_in     = 6,144
L_out    = 512
TTFT     = 2 s
TPOT     = 20 ms
TP_P,max = 28,300 token/s
T_overhead = 0.1 s
```

先算可用排队和计算时间：

```text
S = 2 - 0.1 = 1.9 s
```

算 P 的有效吞吐：

```text
TP_P,eff = 28,300 - 6,144 / 1.9
         = 28,300 - 3,233.684
         = 25,066.316 token/s
```

把总吞吐换算成 token/s：

```text
TP_total = 5,000,000 / 60
         = 83,333.333 token/s
```

P 侧需要处理的输入 token 吞吐：

```text
83,333.333 × 6,144 / (6,144 + 512)
= 76,923.077 token/s
```

连续 P 实例数：

```text
N_P* = 76,923.077 / 25,066.316
     = 3.069
```

D 侧需要处理的输出 token 吞吐：

```text
83,333.333 × 512 / (6,144 + 512)
= 6,410.256 token/s
```

论文给出的 Decode 有效吞吐约为：

```text
TP_D ≈ 1,700 token/s
```

连续 D 实例数：

```text
N_D* = 6,410.256 / 1,700
     = 3.771
```

连续 P/D 比例：

```text
N_P* / N_D* = 3.069 / 3.771
            = 0.814
```

论文正文使用约 `0.82:1`，数值差异来自 `TP_P,eff` 取整和图表读数。

### 整数配置不能直接按连续比例拍板

如果要求从公式上严格覆盖 `5 MTPM`：

```text
ceil(3.069) = 4 P
ceil(3.771) = 4 D
```

得到 `4P4D`。

论文实际选择了 `3P4D`。按 P 侧容量估算：

```text
3 × 25,000 = 75,000 input token/s
```

对应请求率：

```text
75,000 / 6,144 = 12.207 request/s
```

每请求总 token 数：

```text
6,144 + 512 = 6,656
```

理论总吞吐约为：

```text
12.207 × 6,656 × 60
= 4.875 MTPM
```

论文实测约 `4.8 MTPM`。因此 `3P4D` 是一个接近目标的近似配置，不是严格覆盖 `5 MTPM` 的配置。

`4P4D` 的连续容量估算约为：

```text
min(
  4 × 25,000 / 6,144,
  4 × 1,700 / 512
) × 6,656 × 60
≈ 5.30 MTPM
```

实际是否达到这个值，要看 D 的 batch 形成、请求长度分布和 SLO 尾延迟。下面的仿真会展示这种差异。

---

## 5. 从公式到 simulator

只把上面的公式包装成一个 UI，得到的仍然是静态计算器。要模拟真实系统，需要加入事件时钟和请求生命周期。

```text
请求到达
  ↓
进入 P 队列
  ↓
Prefill 完成
  ↓
KV Cache 传输
  ↓
进入 D 队列
  ↓
Decode step
  ├─ 未完成：回到下一次 Decode step
  └─ 完成：记录 TTFT、TPOT、完成时间和 token 数
```

最小仿真器需要的输入：

| 输入 | 说明 |
|---|---|
| 请求 trace | 到达时间、`L_in`、`L_out`，可选 prefix cache 命中长度 |
| P profile | 输入 token 数对应的 Prefill 时间或吞吐 |
| D profile | batch size 对应的 step time、TPOT 和吞吐 |
| KV profile | KV 大小、传输延迟、有效带宽和并发限制 |
| 分配方案 | `N_P`、`N_D`、最大 decode batch、chunk size |
| SLO | TTFT、TPOT、目标总吞吐、要求的百分位 |

仿真器维护的核心状态：

```text
P worker: busy, queue
D worker: busy, current batch, queue
Request: 到达时间、剩余输出 token、first-token 时间、完成时间
KV transfer: 传输中的请求、传输完成事件
```

对每个请求记录：

```text
TTFT = first_token_time - arrival_time
TPOT = (completion_time - first_token_time) / (L_out - 1)
```

资源规划不能只看平均值。一个配置可能有不错的平均 TTFT，但 P99 已经堆积；也可能平均吞吐接近目标，但 Decode batch 已经超过 TPOT 上限。

---

## 6. 可运行的最小仿真器

下面使用 Python 标准库实现一个 CPU-only 离散事件仿真器。它不执行模型，只维护时间、队列、worker、batch 和请求状态。

完整文件：[`examples/pd_capacity_simulator.py`](examples/pd_capacity_simulator.py)

```python
#!/usr/bin/env python3
"""CPU-only P/D capacity and queueing simulator.

The simulator is deliberately small:
- one request occupies one prefill worker while its prompt is processed;
- prefill and decode workers have FIFO queues;
- decode workers process a batch one token per step;
- decode step time comes from an interpolated benchmark profile.

It is a capacity-planning tool, not a kernel-level performance model.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from heapq import heappop, heappush
from math import ceil, nan
import random
import statistics


@dataclass(frozen=True)
class Request:
    request_id: int
    arrival_s: float
    input_tokens: int
    output_tokens: int


@dataclass
class RequestState:
    request: Request
    remaining_output: int
    first_token_s: float | None = None
    completion_s: float | None = None


@dataclass
class DecodeProfile:
    """Piecewise-linear (batch size, step time) benchmark curve."""

    points: tuple[tuple[int, float], ...]

    def __post_init__(self) -> None:
        if not self.points or self.points[0][0] != 1:
            raise ValueError("decode profile must start at batch size 1")
        if any(a[0] >= b[0] or a[1] >= b[1] for a, b in zip(self.points, self.points[1:])):
            raise ValueError("decode profile must be strictly increasing")

    @property
    def max_batch(self) -> int:
        return self.points[-1][0]

    def step_time_s(self, batch_size: int) -> float:
        if batch_size < 1 or batch_size > self.max_batch:
            raise ValueError(f"batch size {batch_size} outside profile")
        for (b0, t0), (b1, t1) in zip(self.points, self.points[1:]):
            if b0 <= batch_size <= b1:
                ratio = (batch_size - b0) / (b1 - b0)
                return t0 + ratio * (t1 - t0)
        return self.points[0][1]


@dataclass(frozen=True)
class Allocation:
    prefill_workers: int
    decode_workers: int
    prefill_tokens_per_s: float
    decode_profile: DecodeProfile
    kv_transfer_s: float
    max_decode_batch: int

    def __post_init__(self) -> None:
        if self.prefill_workers < 1 or self.decode_workers < 1:
            raise ValueError("worker count must be positive")
        if not 1 <= self.max_decode_batch <= self.decode_profile.max_batch:
            raise ValueError("decode batch cap outside profile")


@dataclass
class PrefillWorker:
    busy: bool = False
    queue: deque[RequestState] = field(default_factory=deque)


@dataclass
class DecodeWorker:
    busy: bool = False
    batch: list[RequestState] = field(default_factory=list)
    queue: deque[RequestState] = field(default_factory=deque)


@dataclass(frozen=True)
class Metrics:
    completed_requests: int
    throughput_tokens_per_min: float
    ttft_mean_s: float
    ttft_p95_s: float
    ttft_p99_s: float
    tpot_mean_s: float
    tpot_p95_s: float
    tpot_p99_s: float
    elapsed_s: float


def percentile(values: list[float], q: float) -> float:
    if not values:
        return nan
    ordered = sorted(values)
    index = max(0, ceil(q * len(ordered)) - 1)
    return ordered[index]


def make_poisson_trace(
    request_count: int,
    request_rate_per_s: float,
    input_tokens: int,
    output_tokens: int,
    seed: int,
) -> list[Request]:
    rng = random.Random(seed)
    now_s = 0.0
    trace: list[Request] = []
    for request_id in range(request_count):
        now_s += rng.expovariate(request_rate_per_s)
        trace.append(Request(request_id, now_s, input_tokens, output_tokens))
    return trace


class Simulator:
    def __init__(self, trace: list[Request], allocation: Allocation):
        self.trace = trace
        self.allocation = allocation
        self.now_s = 0.0
        self.events: list[tuple[float, int, str, object]] = []
        self.event_seq = 0
        self.states = {
            request.request_id: RequestState(request, request.output_tokens)
            for request in trace
        }
        self.prefill_workers = [
            PrefillWorker() for _ in range(allocation.prefill_workers)
        ]
        self.decode_workers = [
            DecodeWorker() for _ in range(allocation.decode_workers)
        ]
        self.round_robin = 0

    def push(self, at_s: float, kind: str, payload: object) -> None:
        heappush(self.events, (at_s, self.event_seq, kind, payload))
        self.event_seq += 1

    def choose_prefill_worker(self) -> int:
        count = len(self.prefill_workers)
        start = self.round_robin % count
        self.round_robin += 1

        def load(index: int) -> int:
            worker = self.prefill_workers[index]
            return int(worker.busy) + len(worker.queue)

        return min(range(count), key=lambda i: (load(i), (i - start) % count))

    def choose_decode_worker(self) -> int:
        count = len(self.decode_workers)
        start = self.round_robin % count
        self.round_robin += 1

        def load(index: int) -> int:
            worker = self.decode_workers[index]
            return int(worker.busy) + len(worker.batch) + len(worker.queue)

        return min(range(count), key=lambda i: (load(i), (i - start) % count))

    def schedule_prefill(self, worker_index: int, at_s: float) -> None:
        worker = self.prefill_workers[worker_index]
        if worker.busy or not worker.queue:
            return
        state = worker.queue.popleft()
        worker.busy = True
        duration_s = state.request.input_tokens / self.allocation.prefill_tokens_per_s
        self.push(at_s + duration_s, "prefill_done", (worker_index, state.request.request_id))

    def schedule_decode(self, worker_index: int, at_s: float) -> None:
        worker = self.decode_workers[worker_index]
        if worker.busy:
            return
        while worker.queue and len(worker.batch) < self.allocation.max_decode_batch:
            worker.batch.append(worker.queue.popleft())
        if not worker.batch:
            return

        batch_size = len(worker.batch)
        step_s = self.allocation.decode_profile.step_time_s(batch_size)
        worker.busy = True
        batch_ids = tuple(state.request.request_id for state in worker.batch)
        self.push(at_s + step_s, "decode_step", (worker_index, batch_ids))

    def dispatch_arrival(self, state: RequestState, at_s: float) -> None:
        worker_index = self.choose_prefill_worker()
        self.prefill_workers[worker_index].queue.append(state)
        self.schedule_prefill(worker_index, at_s)

    def on_prefill_done(self, worker_index: int, request_id: int, at_s: float) -> None:
        worker = self.prefill_workers[worker_index]
        worker.busy = False
        self.push(at_s + self.allocation.kv_transfer_s, "kv_done", request_id)
        self.schedule_prefill(worker_index, at_s)

    def on_kv_done(self, request_id: int, at_s: float) -> None:
        worker_index = self.choose_decode_worker()
        state = self.states[request_id]
        self.decode_workers[worker_index].queue.append(state)
        self.schedule_decode(worker_index, at_s)

    def on_decode_step(
        self, worker_index: int, batch_ids: tuple[int, ...], at_s: float
    ) -> None:
        worker = self.decode_workers[worker_index]
        active = {state.request.request_id: state for state in worker.batch}
        next_batch: list[RequestState] = []

        for request_id in batch_ids:
            state = active[request_id]
            if state.first_token_s is None:
                state.first_token_s = at_s
            state.remaining_output -= 1
            if state.remaining_output == 0:
                state.completion_s = at_s
            else:
                next_batch.append(state)

        worker.batch = next_batch
        worker.busy = False
        self.schedule_decode(worker_index, at_s)

    def run(self) -> Metrics:
        for request in self.trace:
            self.push(request.arrival_s, "arrival", request.request_id)

        while self.events:
            at_s, _, kind, payload = heappop(self.events)
            self.now_s = at_s
            if kind == "arrival":
                state = self.states[int(payload)]
                self.dispatch_arrival(state, at_s)
            elif kind == "prefill_done":
                worker_index, request_id = payload  # type: ignore[misc]
                self.on_prefill_done(int(worker_index), int(request_id), at_s)
            elif kind == "kv_done":
                self.on_kv_done(int(payload), at_s)
            elif kind == "decode_step":
                worker_index, batch_ids = payload  # type: ignore[misc]
                self.on_decode_step(int(worker_index), batch_ids, at_s)
            else:
                raise AssertionError(f"unknown event: {kind}")

        completed = [state for state in self.states.values() if state.completion_s is not None]
        ttft = [
            state.first_token_s - state.request.arrival_s
            for state in completed
            if state.first_token_s is not None
        ]
        tpot = [
            (state.completion_s - state.first_token_s) / (state.request.output_tokens - 1)
            for state in completed
            if state.first_token_s is not None and state.request.output_tokens > 1
        ]
        total_tokens = sum(
            state.request.input_tokens + state.request.output_tokens
            for state in completed
        )
        elapsed_s = max(state.completion_s for state in completed) - self.trace[0].arrival_s

        return Metrics(
            completed_requests=len(completed),
            throughput_tokens_per_min=total_tokens / elapsed_s * 60,
            ttft_mean_s=statistics.mean(ttft),
            ttft_p95_s=percentile(ttft, 0.95),
            ttft_p99_s=percentile(ttft, 0.99),
            tpot_mean_s=statistics.mean(tpot),
            tpot_p95_s=percentile(tpot, 0.95),
            tpot_p99_s=percentile(tpot, 0.99),
            elapsed_s=elapsed_s,
        )


def static_estimate(
    total_tokens_per_min: float,
    input_tokens: int,
    output_tokens: int,
    prefill_max_tokens_per_s: float,
    ttft_s: float,
    overhead_s: float,
    decode_tokens_per_s: float,
) -> tuple[float, float, float, float]:
    """Return effective prefill rate, continuous P count, D count, P/D ratio."""
    queue_time_s = ttft_s - overhead_s
    effective_prefill = prefill_max_tokens_per_s - input_tokens / queue_time_s
    total_tokens_per_s = total_tokens_per_min / 60
    n_prefill = (
        total_tokens_per_s * input_tokens
        / ((input_tokens + output_tokens) * effective_prefill)
    )
    n_decode = (
        total_tokens_per_s * output_tokens
        / ((input_tokens + output_tokens) * decode_tokens_per_s)
    )
    return effective_prefill, n_prefill, n_decode, n_prefill / n_decode


def run_repeats(
    request_count: int,
    request_rate_per_s: float,
    seeds: tuple[int, ...],
    allocation: Allocation,
    input_tokens: int,
    output_tokens: int,
) -> Metrics:
    results = [
        Simulator(
            make_poisson_trace(
                request_count,
                request_rate_per_s,
                input_tokens,
                output_tokens,
                seed,
            ),
            allocation,
        ).run()
        for seed in seeds
    ]
    return Metrics(
        completed_requests=int(statistics.median(r.completed_requests for r in results)),
        throughput_tokens_per_min=statistics.median(
            r.throughput_tokens_per_min for r in results
        ),
        ttft_mean_s=statistics.median(r.ttft_mean_s for r in results),
        ttft_p95_s=statistics.median(r.ttft_p95_s for r in results),
        ttft_p99_s=statistics.median(r.ttft_p99_s for r in results),
        tpot_mean_s=statistics.median(r.tpot_mean_s for r in results),
        tpot_p95_s=statistics.median(r.tpot_p95_s for r in results),
        tpot_p99_s=statistics.median(r.tpot_p99_s for r in results),
        elapsed_s=statistics.median(r.elapsed_s for r in results),
    )


def main() -> None:
    input_tokens = 6144
    output_tokens = 512
    ttft_s = 2.0
    tpot_s = 0.020
    overhead_s = 0.1
    prefill_max = 28_300.0
    total_tokens_per_min = 5_000_000.0

    # Illustrative benchmark curve: tpot_s is reached around batch 34.
    decode_profile = DecodeProfile(
        points=(
            (1, 0.01900),
            (8, 0.01924),
            (16, 0.01952),
            (24, 0.01976),
            (32, 0.01996),
            (34, 0.02000),
        )
    )
    effective_prefill, n_prefill, n_decode, ratio = static_estimate(
        total_tokens_per_min=total_tokens_per_min,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        prefill_max_tokens_per_s=prefill_max,
        ttft_s=ttft_s,
        overhead_s=overhead_s,
        decode_tokens_per_s=34 / tpot_s,
    )
    request_rate = total_tokens_per_min / 60 / (input_tokens + output_tokens)
    print(
        "static: "
        f"prefill_eff={effective_prefill:.0f} token/s, "
        f"Np*={n_prefill:.3f}, Nd*={n_decode:.3f}, ratio={ratio:.3f}"
    )
    print(
        f"trace: requests={5000}, rate={request_rate:.3f} req/s, "
        f"input={input_tokens}, output={output_tokens}\n"
    )
    print(
        "config | throughput(MTPM) | TTFT p95/p99(s) | TPOT p95/p99(ms)"
    )
    print("-------+------------------+-------------------+------------------")

    for np, nd in ((3, 3), (3, 4), (4, 4), (4, 5)):
        allocation = Allocation(
            prefill_workers=np,
            decode_workers=nd,
            prefill_tokens_per_s=25_000.0,
            decode_profile=decode_profile,
            kv_transfer_s=overhead_s,
            max_decode_batch=34,
        )
        metrics = run_repeats(
            request_count=5000,
            request_rate_per_s=request_rate,
            seeds=(1, 2, 3),
            allocation=allocation,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
        print(
            f"{np}P{nd}D  | "
            f"{metrics.throughput_tokens_per_min / 1e6:16.3f} | "
            f"{metrics.ttft_p95_s:7.3f}/{metrics.ttft_p99_s:7.3f} | "
            f"{metrics.tpot_p95_s * 1e3:7.2f}/{metrics.tpot_p99_s * 1e3:7.2f}"
        )


if __name__ == "__main__":
    main()
```

运行命令：

```bash
python3 inference-scheduling/examples/pd_capacity_simulator.py
```

本次运行输出：

```text
static: prefill_eff=25066 token/s, Np*=3.069, Nd*=3.771, ratio=0.814
trace: requests=5000, rate=12.520 req/s, input=6144, output=512

config | throughput(MTPM) | TTFT p95/p99(s) | TPOT p95/p99(ms)
-------+------------------+-------------------+------------------
3P3D  |            3.899 |  99.251/102.108 |   20.00/  20.00
3P4D  |            4.740 |  12.831/ 13.429 |   19.94/  19.94
4P4D  |            4.877 |   1.654/  2.238 |   20.00/  20.00
4P5D  |            4.878 |   0.753/  0.870 |   19.87/  19.90
```

这里的 throughput 是有限长度 trace 的整体完成速率，包含启动和 drain 阶段，不要把它当成论文实测值或稳态容量的精确读数。这个结果主要说明两件事：

1. `3P3D` 的 P 侧持续过载，排队长到 TTFT P95 超过 99 秒；
2. `4P4D` 的 TTFT P95 约 `1.65 s`，但 P99 约 `2.24 s`，已经越过 `2 s` SLO；增加到 `4P5D` 后 P99 降到约 `0.87 s`，说明积压主要发生在 D 侧。

TPOT 都接近 `20 ms`，原因是仿真器把 decode batch 上限设成了 profile 中满足 TPOT 的 `34`。如果允许 batch 继续增大，吞吐可能上升，但 TPOT 会越过 SLO。

这个对比也说明 P/D 扩容要按瓶颈选择。`4P4D` 的 TTFT 尾部主要被 D 队列拖长，继续增加到 `5P4D` 未必有用；先确认队列堆在哪一侧，再决定加 P 还是加 D。

### 仿真器中的服务时间

Prefill 目前使用确定性服务时间：

\[
t_P=\frac{L_{in}}{TP_{P,eff}}
\]

Decode 从 benchmark 曲线插值：

\[
t_D=t_D(B)
\]

每完成一个 decode step，每个活跃请求生成一个 token。这个细节很重要：Decode TPOT 不是“总吞吐除以请求数”，而是由 batch 和每步计算时间共同决定。

真实 Prefill 可以替换成：

- 指数分布服务时间，匹配 M/M/1；
- 实测输入长度直方图；
- chunked prefill 的 segment 事件；
- 带 prefix cache 的 `L_eff`。

真实 Decode 可以替换成：

- 上下文长度与 batch 的二维 profile；
- CUDA graph bucket；
- MTP 每步多个 token；
- 最大 batch、最大 token 数和显存约束。

---

## 7. 如何用 simulator 搜索 P/D 数量

推荐的搜索流程：

```text
1. 用解析公式得到 N_P*、N_D* 和连续比例
2. 在连续值附近枚举整数配置
3. 对每个配置回放 trace 或多随机种子运行
4. 过滤 TTFT/TPOT 百分位和吞吐不达标的配置
5. 在剩余配置中按成本、利用率和 SLO 余量排序
6. 用真实集群压测最终候选
```

候选生成示例：

```python
for n_p in range(max(1, floor(n_p_star) - 1),
                 ceil(n_p_star) + 3):
    for n_d in range(max(1, floor(n_d_star) - 1),
                     ceil(n_d_star) + 3):
        run_candidate(n_p, n_d)
```

过滤条件不要只写平均值：

```text
P95 TTFT ≤ TTFT_SLO
P95 TPOT ≤ TPOT_SLO
总吞吐 ≥ 目标吞吐
```

如果生产目标包含严格的 P99 或突发流量，还应增加：

- 多组随机种子；
- 峰值到达率；
- KV 传输抖动；
- P 或 D 实例故障；
- `+1` 冗余；
- 显存容量约束；
- Prefix Cache 命中率变化。

---

## 8. 常见误区

| 误区 | 现象 | 正确做法 |
|---|---|---|
| 直接把 `N_P*/N_D*` 四舍五入 | `3.08` 被写成 `3P`，但 P 侧实际不够 | 用整数容量约束重新计算，并重新模拟 |
| 只按 `L_in/L_out` 配 P/D | 忽略两阶段吞吐差异，P 可能严重不足 | 乘上 `TP_D/TP_P,eff` |
| 使用峰值 `TP_P,max` 做 TTFT 规划 | 平均 TTFT 看似合格，P99 已经超限 | 先计算 `TP_P,eff`，再做尾延迟仿真 |
| 用 Decode 的理论峰值吞吐 | 可能违反 TPOT | 取满足 TPOT 的 batch 点 |
| 只看吞吐，不看队列 | 目标附近的配置尾延迟抖动很大 | 输出 P95/P99、backlog 和利用率 |
| 把 CPU 仿真当成模型执行 | 期待仿真时间和模型大小强相关 | 仿真只执行事件和服务模型 |
| 忽略 KV 传输 | P/D 计算配平后，handoff 仍可能成为瓶颈 | 把 KV 作为独立资源事件建模 |

---

## 9. 自检

**Q1：为什么 P/D 比例公式中会有 `TP_D / TP_P,eff`？**

因为 P 和 D 处理的 token 类型不同，单实例处理速度也不同。`L_in/L_out` 只描述工作量比例，吞吐比值把“同样多的 token 需要多少实例”折算进来。

**Q2：为什么 M/M/1 中的 `μ` 不是 token/s？**

因为 M/M/1 的到达率 `λ` 和服务率 `μ` 都以请求/秒为单位。`TP_{P,max}/L_in` 把单实例的 token 处理能力换算成了请求处理能力。

**Q3：为什么 `3P4D` 可能达不到 `5 MTPM`？**

连续公式给出 `N_P*≈3.08`。取 3 个 P 实例时，P 侧容量约为 `4.875 MTPM`，所以它只能逼近目标，不能严格覆盖目标。论文实测约 `4.8 MTPM` 与这个判断一致。

**Q4：仿真器为什么不能只用总吞吐判断配置是否足够？**

因为同一总吞吐下，请求长度分布、到达突发和 D batch 形成方式不同，TTFT/TPOT 的尾部会明显不同。容量足够不等于每个 SLO 百分位都达标。

**Q5：如果要把这个 simulator 用在真实部署上，第一步应该补什么？**

先补真实的 P/D profile 和请求 trace，尤其是 Prefill 时间随 token 数变化的曲线、Decode step time 随 batch 和上下文变化的曲线、KV handoff 的延迟和带宽。没有这些数据，仿真器只能给出结构性趋势。
