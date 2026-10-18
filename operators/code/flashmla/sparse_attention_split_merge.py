from math import exp, log
from typing import List, Sequence, Tuple

Partial = Tuple[float, float, float]  # (m, l, u)


def make_partial(q: float, keys: Sequence[float], values: Sequence[float],
                 indices: Sequence[int]) -> Partial:
    """对一个 index 子集计算稳定的 softmax 统计量 (m, l, u)。"""
    scores = [q * keys[index] for index in indices]
    m = max(scores)
    weights = [exp(score - m) for score in scores]
    l = sum(weights)
    u = sum(weight * values[index] for weight, index in zip(weights, indices))
    return m, l, u


def merge_partials(partials: Sequence[Partial]) -> Tuple[float, float]:
    """返回 (attention output, log-sum-exp)，等价于一次性计算全部 token。"""
    m = max(partial[0] for partial in partials)
    l = sum(partial_l * exp(partial_m - m)
            for partial_m, partial_l, _ in partials)
    u = sum(partial_u * exp(partial_m - m)
            for partial_m, _, partial_u in partials)
    return u / l, m + log(l)


def main() -> None:
    q = 1.0
    keys = [0.0, log(2.0), log(3.0), log(4.0)]
    values = [10.0, 20.0, 30.0, 40.0]
    sparse_indices = [0, 1, 2, 3]  # 真实场景由上游 indexer 提供
    chunks = [sparse_indices[:2], sparse_indices[2:]]

    partials: List[Partial] = []
    for split_id, indices in enumerate(chunks):
        partial = make_partial(q, keys, values, indices)
        partials.append(partial)
        m, l, u = partial
        print(
            f"split {split_id}: m={m:.6f}, l={l:.6f}, "
            f"u={u:.6f}, local_output={u / l:.6f}"
        )

    output, lse = merge_partials(partials)
    print(f"merged output={output:.6f}, lse={lse:.6f}")

    # 与未切分的直接计算互相校验。
    direct = make_partial(q, keys, values, sparse_indices)
    assert abs(output - direct[2] / direct[1]) < 1e-12
    assert abs(lse - (direct[0] + log(direct[1]))) < 1e-12


if __name__ == "__main__":
    main()
