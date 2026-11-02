#!/usr/bin/env python3
"""CPU-only trace of the core IMPRESS decisions.

The script does not run an LLM or read KV tensors. It reproduces the
decision arithmetic used by the note:

1. estimate the similarity threshold for important-token sets;
2. compare probe-head sets and choose a shared representative set;
3. rank KV chunks by access frequency times important ratio;
4. recover logical token order after one radix-node-local reorder;
5. compare the vector count of the full-K and probe-assisted paths.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Iterable


def expected_random_jaccard(k: int, n: int) -> float:
    """Expected Jaccard of two independently chosen k-of-n token sets."""
    if not (1 <= k <= n):
        raise ValueError("require 1 <= k <= n")
    p = k / n
    return p / (2.0 - p)


def similarity_threshold(k: int, n: int, alpha: float = 0.6) -> float:
    """IMPRESS threshold: random-baseline Jaccard raised to alpha."""
    return expected_random_jaccard(k, n) ** alpha


def mean_pairwise_jaccard(token_sets: Iterable[set[int]]) -> float:
    sets = list(token_sets)
    if len(sets) < 2:
        raise ValueError("need at least two probe-head sets")
    values = [
        len(a & b) / len(a | b)
        for a, b in combinations(sets, 2)
        if a or b
    ]
    return sum(values) / len(values)


def shared_probe_tokens(token_sets: Iterable[set[int]], count: int) -> set[int]:
    """Take the tokens that appear in the most probe-head top-k sets."""
    counts: dict[int, int] = {}
    for token_set in token_sets:
        for token in token_set:
            counts[token] = counts.get(token, 0) + 1
    ranked = sorted(counts, key=lambda token: (-counts[token], token))
    return set(ranked[:count])


def chunk_score(access_frequency: float, important_ratio: float) -> float:
    if access_frequency < 0:
        raise ValueError("access_frequency must be non-negative")
    if not 0.0 <= important_ratio <= 1.0:
        raise ValueError("important_ratio must be in [0, 1]")
    return access_frequency * important_ratio


@dataclass(frozen=True)
class ReorderedNode:
    logical_tokens: tuple[str, ...]
    importance: tuple[float, ...]
    storage_order: tuple[int, ...]
    reordered_tokens: tuple[str, ...]
    restore_mapping: tuple[int, ...]

    @classmethod
    def from_node(
        cls,
        logical_tokens: tuple[str, ...],
        importance: tuple[float, ...],
    ) -> "ReorderedNode":
        if len(logical_tokens) != len(importance):
            raise ValueError("token and importance lengths differ")

        # Stable sort keeps the original order when scores tie.
        storage_order = tuple(
            sorted(range(len(logical_tokens)), key=lambda i: (-importance[i], i))
        )
        reordered_tokens = tuple(logical_tokens[i] for i in storage_order)

        # restore_mapping[original_pos] = position in reordered storage.
        restore_mapping_list = [0] * len(logical_tokens)
        for storage_pos, original_pos in enumerate(storage_order):
            restore_mapping_list[original_pos] = storage_pos

        return cls(
            logical_tokens=logical_tokens,
            importance=importance,
            storage_order=storage_order,
            reordered_tokens=reordered_tokens,
            restore_mapping=tuple(restore_mapping_list),
        )

    def restore(self) -> tuple[str, ...]:
        return tuple(
            self.reordered_tokens[storage_pos] for storage_pos in self.restore_mapping
        )


def print_case(label: str, probe_sets: list[set[int]], k: int, n: int) -> None:
    similarity = mean_pairwise_jaccard(probe_sets)
    threshold = similarity_threshold(k, n)
    accepted = similarity >= threshold
    print(f"{label}: probe_sets={probe_sets}")
    print(f"  mean_jaccard={similarity:.6f}, threshold={threshold:.6f}")
    print(f"  use_shared_importance={accepted}")
    if accepted:
        print(f"  shared_tokens={sorted(shared_probe_tokens(probe_sets, k))}")
    else:
        print("  action=fallback: load the full prefix KV set")


def main() -> None:
    n, k, alpha = 8, 2, 0.6
    baseline = expected_random_jaccard(k, n)
    threshold = similarity_threshold(k, n, alpha)
    print("== Similarity gate ==")
    print(f"n={n}, k={k}, p={k / n:.6f}")
    print(f"expected_random_jaccard={baseline:.6f}")
    print(f"threshold={baseline:.6f} ** {alpha} = {threshold:.6f}")

    print_case("similar probes", [{0, 3}, {0, 2}, {0, 3}], k, n)
    print_case("dissimilar probes", [{0, 3}, {1, 4}, {2, 5}], k, n)

    print("\n== Chunk admission ==")
    scores = {
        "chunk1": chunk_score(access_frequency=1.5, important_ratio=0.5),
        "chunk2": chunk_score(access_frequency=1.0, important_ratio=1.0),
    }
    for name, score in scores.items():
        print(f"{name}: score={score:.2f}")
    selected = max(scores, key=scores.get)
    print(f"one GPU slot -> cache {selected}")

    print("\n== Radix-node-local reorder ==")
    node = ReorderedNode.from_node(
        logical_tokens=("t0", "t1", "t2", "t3"),
        importance=(0.9, 0.2, 0.2, 0.7),
    )
    print(f"storage_order={node.storage_order}")
    print(f"reordered_tokens={node.reordered_tokens}")
    print(f"restore_mapping={node.restore_mapping}")
    print(f"restored_tokens={node.restore()}")
    assert node.restore() == node.logical_tokens

    before_chunks = (("t0", "t1"), ("t2", "t3"))
    after_chunks = (
        tuple(node.reordered_tokens[0:2]),
        tuple(node.reordered_tokens[2:4]),
    )
    print(f"chunks_before={before_chunks}")
    print("important t0,t3 touch 2 chunks before reorder")
    print(f"chunks_after={after_chunks}")
    print("important t0,t3 share 1 chunk after reorder")

    print("\n== K/V vector-count model ==")
    tokens, heads, probe_heads = 4, 32, 3
    important_tokens = 1
    full_path = heads * (tokens + important_tokens)
    probe_path = tokens * probe_heads + 2 * important_tokens * heads
    print("full K + selected V = H*(n+k) = "
          f"{heads}*({tokens}+{important_tokens}) = {full_path} vectors")
    print("probe K + selected K/V = n*P + 2*k*H = "
          f"{tokens}*{probe_heads}+2*{important_tokens}*{heads} = {probe_path} vectors")
    print(f"reduction={full_path / probe_path:.3f}x")


if __name__ == "__main__":
    main()
