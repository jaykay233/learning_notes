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
