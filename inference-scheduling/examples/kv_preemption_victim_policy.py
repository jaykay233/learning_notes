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
