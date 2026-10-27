#!/usr/bin/env python3
from dataclasses import dataclass
from enum import Enum


class Mode(Enum):
    PREFILL = "P"
    DECODE = "D"


@dataclass(frozen=True)
class RankPlan:
    rank: int
    ready_at: int
    prefill_tokens: int = 64
    decode_tokens: int = 2
    decode_context_len: int = 64


@dataclass
class RankState:
    plan: RankPlan
    prefill_done: bool = False


@dataclass(frozen=True)
class StepTrace:
    step: int
    status: str
    allowed: bool
    modes: tuple[Mode, ...]
    local_work: tuple[int, ...]
    global_tokens: int
    max_work: int
    idle_units: int


def attention_pairs(plan: RankPlan, mode: Mode) -> int:
    if mode is Mode.PREFILL:
        n = plan.prefill_tokens
        return n * (n + 1) // 2
    return plan.decode_tokens * plan.decode_context_len


def prefill_status(offers: tuple[bool, ...]) -> str:
    if all(offers):
        return "all"
    if not any(offers):
        return "none"
    return "mixed"


def decide_prefill(
    offers: tuple[bool, ...],
    delayed_count: int,
    enabled: bool,
    max_delay_passes: int,
) -> tuple[bool, int, str]:
    status = prefill_status(offers)
    if not enabled:
        return True, 0, status

    # Mirrors the key mixed-state guard in SGLang PrefillDelayer:
    # prev_delayed_count < max_delay_passes - 1 -> delay one more pass.
    if status == "mixed" and delayed_count < max_delay_passes - 1:
        return False, delayed_count + 1, status
    return True, 0, status


def simulate(
    plans: tuple[RankPlan, ...],
    *,
    enabled: bool,
    max_delay_passes: int = 3,
) -> tuple[StepTrace, ...]:
    if not plans:
        raise ValueError("plans cannot be empty")
    if max_delay_passes <= 0:
        raise ValueError("max_delay_passes must be positive")

    states = [RankState(plan=plan) for plan in plans]
    delayed_count = 0
    trace: list[StepTrace] = []

    for step in range(16):
        offers = tuple(
            not state.prefill_done and step >= state.plan.ready_at for state in states
        )
        allowed, delayed_count, status = decide_prefill(
            offers,
            delayed_count,
            enabled,
            max_delay_passes,
        )

        modes = tuple(
            Mode.PREFILL if allowed and offer else Mode.DECODE
            for offer in offers
        )
        local_work = tuple(
            attention_pairs(state.plan, mode) for state, mode in zip(states, modes)
        )
        global_tokens = sum(
            state.plan.prefill_tokens
            if mode is Mode.PREFILL
            else state.plan.decode_tokens
            for state, mode in zip(states, modes)
        )
        max_work = max(local_work)
        idle_units = max_work * len(local_work) - sum(local_work)
        trace.append(
            StepTrace(
                step=step,
                status=status,
                allowed=allowed,
                modes=modes,
                local_work=local_work,
                global_tokens=global_tokens,
                max_work=max_work,
                idle_units=idle_units,
            )
        )

        if allowed:
            for state, offer in zip(states, offers):
                if offer:
                    state.prefill_done = True
        if all(state.prefill_done for state in states):
            break
    else:
        raise RuntimeError("simulation did not converge")

    return tuple(trace)


def print_trace(label: str, trace: tuple[StepTrace, ...]) -> None:
    print(label)
    header = (
        "step status allow rank_modes            "
        "local_work             global_tokens idle_units"
    )
    print(header)
    for item in trace:
        modes = " ".join(
            f"{mode.value}:{work}" for mode, work in zip(item.modes, item.local_work)
        )
        works = "/".join(str(work) for work in item.local_work)
        print(
            f"{item.step:<4} {item.status:<6} "
            f"{'yes' if item.allowed else 'no ':<5} "
            f"{modes:<23} {works:<21} "
            f"{item.global_tokens:<13} {item.idle_units}"
        )
    print(f"total_idle_units={sum(item.idle_units for item in trace)}")
    print()


def main() -> None:
    plans = (
        RankPlan(rank=0, ready_at=0),
        RankPlan(rank=1, ready_at=1),
        RankPlan(rank=2, ready_at=2),
    )

    immediate = simulate(plans, enabled=False)
    delayed = simulate(plans, enabled=True, max_delay_passes=3)

    print_trace("without delayer", immediate)
    print_trace("with delayer, max_delay_passes=3", delayed)

    assert tuple(item.modes for item in immediate) == (
        (Mode.PREFILL, Mode.DECODE, Mode.DECODE),
        (Mode.DECODE, Mode.PREFILL, Mode.DECODE),
        (Mode.DECODE, Mode.DECODE, Mode.PREFILL),
    )
    assert sum(item.idle_units for item in immediate) == 11712

    assert tuple(item.modes for item in delayed) == (
        (Mode.DECODE, Mode.DECODE, Mode.DECODE),
        (Mode.DECODE, Mode.DECODE, Mode.DECODE),
        (Mode.PREFILL, Mode.PREFILL, Mode.PREFILL),
    )
    assert sum(item.idle_units for item in delayed) == 0
    assert sum(item.global_tokens for item in immediate) == sum(
        item.global_tokens for item in delayed
    )


if __name__ == "__main__":
    main()
