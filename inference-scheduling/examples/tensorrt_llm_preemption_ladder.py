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
