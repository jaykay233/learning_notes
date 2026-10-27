"""Trace logical experts through EPLB placement, EP dispatch, and combine."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Placement:
    slots_by_rank: tuple[tuple[int, ...], ...]
    slots_per_rank: int

    def __post_init__(self) -> None:
        assert all(len(row) == self.slots_per_rank for row in self.slots_by_rank)

    def slots_for(self, expert_id: int) -> list[int]:
        slots: list[int] = []
        for rank, row in enumerate(self.slots_by_rank):
            for local_slot, placed_expert in enumerate(row):
                if placed_expert == expert_id:
                    slots.append(rank * self.slots_per_rank + local_slot)
        if not slots:
            raise KeyError(f"logical expert {expert_id} has no physical slot")
        return slots

    @property
    def ep_size(self) -> int:
        return len(self.slots_by_rank)


@dataclass(frozen=True)
class Route:
    token: int
    top_k_slot: int
    logical_expert: int
    global_slot: int
    destination_rank: int
    local_slot: int


def route_topk(
    topk_logical: list[list[int]],
    placement: Placement,
    *,
    offset_by_ep_rank: bool = False,
    source_rank: int = 0,
) -> list[Route]:
    routes: list[Route] = []
    round_robin_key: dict[int, int] = {}

    for token, experts in enumerate(topk_logical):
        for top_k_slot, expert in enumerate(experts):
            replicas = placement.slots_for(expert)
            key = round_robin_key.get(expert, 0)
            if offset_by_ep_rank:
                key += source_rank
            global_slot = replicas[key % len(replicas)]
            round_robin_key[expert] = key + 1
            routes.append(
                Route(
                    token=token,
                    top_k_slot=top_k_slot,
                    logical_expert=expert,
                    global_slot=global_slot,
                    destination_rank=global_slot // placement.slots_per_rank,
                    local_slot=global_slot % placement.slots_per_rank,
                )
            )
    return routes


def combine(routes: list[Route]) -> list[int]:
    # The toy expert result depends only on the logical identity and token/top-k
    # position. That makes an incorrect physical mapping visible in the sum.
    combined = [0] * (max(route.token for route in routes) + 1)
    for route in routes:
        toy_expert_output = route.token * 100 + route.logical_expert * 10 + route.top_k_slot
        combined[route.token] += toy_expert_output
    return combined


before = Placement(
    slots_per_rank=2,
    slots_by_rank=((0, 1), (2, 3), (4, 5), (6, 7)),
)
after = Placement(
    slots_per_rank=2,
    slots_by_rank=((0, 6), (2, 3), (4, 5), (1, 7)),
)
topk_logical = [[6, 2], [1, 2], [6, 0]]

print("router logical top-k (EPLB does not rewrite these IDs):", topk_logical)
for label, placement in (("before EPLB", before), ("after EPLB", after)):
    routes = route_topk(topk_logical, placement)
    print(f"\n{label}")
    for route in routes:
        print(
            f"  token={route.token} k={route.top_k_slot} logical={route.logical_expert} "
            f"-> global_slot={route.global_slot} rank={route.destination_rank} "
            f"local_slot={route.local_slot}"
        )
    counts = [0] * placement.ep_size
    for route in routes:
        counts[route.destination_rank] += 1
    print("  destination route counts:", counts)
    print("  combine result:", combine(routes))

ep_size = after.ep_size
max_tokens_per_rank = 3
hidden = 4
local_experts = after.slots_per_rank
rank_major_shape = (ep_size, max_tokens_per_rank, hidden)
expert_major_shape = (local_experts, ep_size * max_tokens_per_rank, hidden)
print("\nNCCL EP LL rank-major dispatch shape:", rank_major_shape, "->", rank_major_shape[0] * rank_major_shape[1], "rows")
print("DeepEP LL expert-major rank-0 shape:", expert_major_shape, "->", expert_major_shape[0] * expert_major_shape[1], "rows")
print("rank-0 recv_expert_count for local logical experts (0, 6): [1, 2]")
