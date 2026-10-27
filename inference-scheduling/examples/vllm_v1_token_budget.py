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
