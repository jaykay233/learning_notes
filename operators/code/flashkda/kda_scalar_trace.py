#!/usr/bin/env python3
"""A tiny, standard-library-only trace of one FlashKDA chunk recurrence.

This uses K=V=1 and CHUNK=2 to make every intermediate value visible. It is a
mathematical teaching example, not a supported FlashKDA kernel shape.
"""
from math import exp
from typing import List, Sequence

Matrix = List[List[float]]


def transpose(a: Matrix) -> Matrix:
    return [list(row) for row in zip(*a)]


def matmul(a: Matrix, b: Matrix) -> Matrix:
    if not a or not b or len(a[0]) != len(b):
        raise ValueError("incompatible matrix shapes")
    bt = transpose(b)
    return [[sum(x * y for x, y in zip(row, col)) for col in bt] for row in a]


def lower_triangle(a: Matrix, include_diagonal: bool) -> Matrix:
    result: Matrix = []
    for i, row in enumerate(a):
        result.append([
            value if (j <= i if include_diagonal else j < i) else 0.0
            for j, value in enumerate(row)
        ])
    return result


def inverse_unit_lower_from_l(l: Matrix) -> Matrix:
    """Solve (I + L) X = I by forward substitution; L is strict-lower."""
    n = len(l)
    a = [[l[i][j] + float(i == j) for j in range(n)] for i in range(n)]
    inverse = [[0.0 for _ in range(n)] for _ in range(n)]
    for col in range(n):
        for row in range(n):
            rhs = float(row == col) - sum(
                a[row][k] * inverse[k][col] for k in range(row)
            )
            inverse[row][col] = rhs / a[row][row]
    return inverse


def scale_rows(a: Matrix, factors: Sequence[float]) -> Matrix:
    return [[value * factors[i] for value in row] for i, row in enumerate(a)]


def add(a: Matrix, b: Matrix) -> Matrix:
    return [[x + y for x, y in zip(ra, rb)] for ra, rb in zip(a, b)]


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + exp(-x))


def show(name: str, value: object) -> None:
    print(f"{name} = {value}")


def main() -> None:
    # Two tokens, one key/value feature. q and k are already L2-normalized.
    q: Matrix = [[1.0], [1.0]]
    k: Matrix = [[1.0], [1.0]]
    v: Matrix = [[3.0], [4.0]]
    g_after_activation = [-1.0, -1.0]  # base-2 log decay per token
    beta_logits = [0.0, 0.0]
    state: Matrix = [[2.0]]  # [V, K]
    scale = 1.0

    beta = [sigmoid(x) for x in beta_logits]
    g_cumsum: List[float] = []
    running = 0.0
    for gate in g_after_activation:
        running += gate
        g_cumsum.append(running)
    g_total = g_cumsum[-1]

    decay = [2.0 ** x for x in g_cumsum]
    inverse_decay = [2.0 ** (-x) for x in g_cumsum]
    k_decayed = scale_rows(k, decay)
    q_decayed = [[row[0] * factor * scale] for row, factor in zip(q, decay)]
    k_inv = scale_rows(k, inverse_decay)
    k_restored = scale_rows(k_inv, [2.0 ** g_total] * len(k_inv))

    raw_k_interaction = matmul(k_decayed, transpose(k_inv))
    l = lower_triangle(raw_k_interaction, include_diagonal=False)
    l = scale_rows(l, beta)
    mqk = lower_triangle(matmul(q_decayed, transpose(k_inv)), include_diagonal=True)
    inv = inverse_unit_lower_from_l(l)

    old_state_prediction = matmul(k_decayed, transpose(state))
    residual = [[x - y for x, y in zip(rv, pv)] for rv, pv in zip(v, old_state_prediction)]
    v_adjusted = scale_rows(residual, beta)
    u = matmul(inv, v_adjusted)

    direct_output = matmul(q_decayed, transpose(state))
    correction = matmul(mqk, u)
    output = add(direct_output, correction)

    delta_state_kv = matmul(transpose(k_restored), u)  # [K, V]
    delta_state = transpose(delta_state_kv)             # [V, K]
    decayed_state = scale_rows(state, [2.0 ** g_total])
    next_state = add(decayed_state, delta_state)

    show("beta", [round(x, 6) for x in beta])
    show("g_cumsum", g_cumsum)
    show("k_decayed", k_decayed)
    show("k_inv", k_inv)
    show("k_restored", k_restored)
    show("L", l)
    show("INV", inv)
    show("Mqk", mqk)
    show("v_adjusted", v_adjusted)
    show("U", u)
    show("output", output)
    show("delta_state", delta_state)
    show("next_state", next_state)

    assert output == [[2.0], [2.5]]
    assert next_state == [[2.5]]
    print("trace check: PASS")


if __name__ == "__main__":
    main()
