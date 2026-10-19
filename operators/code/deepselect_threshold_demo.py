#!/usr/bin/env python3
"""Tiny CPU model of DeepSelect's threshold-and-reconstruct TopK strategy.

This is a teaching model, not a performance implementation. It uses Python sort
for each exact reconstruction; the CUDA kernel uses radix selection in shared
memory and PTX-assisted candidate compaction.
"""

from __future__ import annotations

from itertools import chain
from typing import Iterable, Sequence

Item = tuple[int, int]  # (value, original_index)


def threshold_topk(
    seed: Sequence[Item],
    blocks: Sequence[Sequence[Item]],
    k: int,
    *,
    largest: bool,
    candidate_limit: int,
    trace: bool = False,
) -> list[Item]:
    """Return exact TopK after an exact seed selection and threshold filtering.

    The values in this demo are unique, so the pivot/tie-quota corner case is
    deliberately excluded. Production kernels must handle equal values.
    """
    all_items = list(seed) + list(chain.from_iterable(blocks))
    values = [value for value, _ in all_items]
    assert len(values) == len(set(values)), "demo expects unique values"
    assert 0 < k <= len(seed)
    assert candidate_limit > 0

    def exact_topk(items: Iterable[Item]) -> list[Item]:
        return sorted(items, key=lambda item: item[0], reverse=largest)[:k]

    def passes(value: int, threshold: int) -> bool:
        return value > threshold if largest else value < threshold

    survivor = exact_topk(seed)  # exact within seed; not k random picks
    threshold = survivor[-1][0]
    incoming: list[Item] = []
    if trace:
        direction = "largest" if largest else "smallest"
        print(f"{direction}: seed survivor={survivor}, threshold={threshold}")

    for block_id, block in enumerate(blocks):
        hits = [item for item in block if passes(item[0], threshold)]
        incoming.extend(hits)
        if trace:
            print(
                f"  block {block_id}: values={[v for v, _ in block]}, "
                f"hits={hits}, incoming={incoming}"
            )

        if len(incoming) >= candidate_limit:
            survivor = exact_topk(chain(survivor, incoming))
            threshold = survivor[-1][0]
            incoming.clear()
            if trace:
                print(
                    f"    reconstruct: survivor={survivor}, "
                    f"new threshold={threshold}"
                )

    if incoming:
        survivor = exact_topk(chain(survivor, incoming))
        threshold = survivor[-1][0]
        if trace:
            print(f"  final reconstruct: survivor={survivor}, threshold={threshold}")

    return survivor


def main() -> None:
    largest_seed = [(1, 0), (2, 1)]
    largest_blocks = [
        [(3, 2), (9, 3)],
        [(4, 4), (10, 5)],
        [(8, 6), (0, 7)],
    ]
    got_largest = threshold_topk(
        largest_seed, largest_blocks, 2,
        largest=True, candidate_limit=2, trace=True,
    )
    ref_largest = sorted(
        chain(largest_seed, chain.from_iterable(largest_blocks)),
        key=lambda item: item[0], reverse=True,
    )[:2]
    assert got_largest == ref_largest == [(10, 5), (9, 3)]
    print(f"  result={got_largest}; reference={ref_largest}\n")

    smallest_seed = [(8, 0), (9, 1)]
    smallest_blocks = [
        [(7, 2), (1, 3)],
        [(6, 4), (0, 5)],
        [(2, 6), (10, 7)],
    ]
    got_smallest = threshold_topk(
        smallest_seed, smallest_blocks, 2,
        largest=False, candidate_limit=2, trace=True,
    )
    ref_smallest = sorted(
        chain(smallest_seed, chain.from_iterable(smallest_blocks)),
        key=lambda item: item[0],
    )[:2]
    assert got_smallest == ref_smallest == [(0, 5), (1, 3)]
    print(f"  result={got_smallest}; reference={ref_smallest}")
    print("PASS: thresholded selection matches full-sort references")


if __name__ == "__main__":
    main()
