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
