#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把 ols.cu 的索引逻辑逐行搬进 numpy，不需要 GPU 就能验证数学。

用途不是「模拟性能」，而是在没有 nvcc / T4 的机器上抓索引错误。
它第一次跑出来就把 Cholesky 的对角更新漏掉了（i > j 应为 i >= j），
症状是判题误差从 3e-4 跳到 5e+1 —— 这种错误在 CUDA 上不会崩、不会 NaN，
只看 max abs diff 也只会觉得「精度不太行」。

对应关系：
  gram() 的循环            ↔ ols.cu 阶段 1（线程 (i,j) 算 A_ij）
  gj_solve() / chol_solve() ↔ ols.cu 阶段 2（两种解法都实现了，用于对照）
  A 用 (32, 32) 的布局，与 Gauss-Jordan 版的 smem 一致；
  所有标量运算都显式走 f32，否则 numpy 会悄悄用 float64 把误差抹掉。

运行：python ols_emul.py
"""

from __future__ import annotations

import numpy as np

f32 = np.float32
MAXP = 32
STRIDE = MAXP          # GJ 只做行访问，不需要 +1 填充
CHUNK = 256


def gram(X: np.ndarray, y: np.ndarray):
    """阶段 1：A = XᵀX，b = Xᵀy。float 段内 + double 段间。"""
    X = np.ascontiguousarray(X, dtype=f32)
    y = np.ascontiguousarray(y, dtype=f32)
    n, p = X.shape
    assert 1 <= p <= MAXP and n >= 1

    A = np.zeros((MAXP, STRIDE), dtype=f32)
    b = np.zeros(MAXP, dtype=f32)

    for i in range(p):
        for j in range(p):
            acc = 0.0                          # double：段间累加
            for k0 in range(0, n, CHUNK):
                pa = f32(0.0)                  # float：段内累加
                for k in range(k0, min(k0 + CHUNK, n)):
                    pa = f32(pa + f32(X[k, i] * X[k, j]))
                acc += float(pa)
            A[i, j] = f32(acc)
    for j in range(p):
        acc = 0.0
        for k0 in range(0, n, CHUNK):
            pb = f32(0.0)
            for k in range(k0, min(k0 + CHUNK, n)):
                pb = f32(pb + f32(X[k, j] * y[k]))
            acc += float(pb)
        b[j] = f32(acc)
    return A, b, p


def gj_solve(A0: np.ndarray, b0: np.ndarray, p: int) -> np.ndarray:
    """阶段 2（ols.cu 的选择）：Gauss-Jordan 直接消 [A|b]。零归约。"""
    A = A0.copy()
    b = b0.copy()
    for c in range(p):
        inv = f32(1.0) / A[c, c]               # 2a 归一化第 c 行
        for j in range(c, p):
            A[c, j] = f32(A[c, j] * inv)
        b[c] = f32(b[c] * inv)

        for i in range(p):                      # 2b 消去其余各行
            if i == c:
                continue
            f = A[i, c]
            for j in range(c + 1, p):
                A[i, j] = f32(A[i, j] - f32(f * A[c, j]))
            b[i] = f32(b[i] - f32(f * b[c]))
        # 不写回 A[i][c]：第 c 列此后再不被读，写它反而构成竞争
    return b[:p]


def chol_solve(A0: np.ndarray, b0: np.ndarray, p: int) -> np.ndarray:
    """对照实现：Cholesky + 两次三角回代。慢在 2p 次串行归约上，不在 FLOP。"""
    A = A0.copy()
    b = b0.copy()
    for c in range(p):
        d = A[c, c]
        A[c, c] = f32(np.sqrt(d if d > 1e-30 else 1e-30))
        dcc = A[c, c]
        for i in range(c + 1, p):               # 列归一化
            A[i, c] = f32(A[i, c] / dcc)
        for j in range(c + 1, p):               # rank-1 更新，注意 i >= j
            for i in range(j, p):
                A[i, j] = f32(A[i, j] - f32(A[i, c] * A[j, c]))

    z = np.zeros(MAXP, f32)
    res = np.zeros(MAXP, f32)
    for r in range(p):                          # 前代
        v = sum(float(A[r, k] * z[k]) for k in range(r))
        z[r] = f32(f32(b[r] - f32(v)) / A[r, r])
    for r in range(p - 1, -1, -1):              # 回代
        v = sum(float(A[k, r] * res[k]) for k in range(r + 1, p))
        res[r] = f32(f32(z[r] - f32(v)) / A[r, r])
    return res[:p]


def judge(got: np.ndarray, want: np.ndarray) -> float:
    """题面口径：abs tol 1e-2、rel tol 1e-2，返回 max 归一化误差，< 1 即通过。"""
    d = np.abs(got.astype(np.float64) - want)
    return float(np.max(d / (1e-2 + 1e-2 * np.abs(want))))


def reference(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    Xd, yd = X.astype(np.float64), y.astype(np.float64)
    return np.linalg.solve(Xd.T @ Xd, Xd.T @ yd)


def main() -> None:
    rng = np.random.default_rng(0)

    print("解法对照（同一份 Gram 输出，两条不同的求解路径）")
    print(f"{'n×p':>10} {'κ(X)':>8} {'κ²·ε':>9} | {'GJ':>10} {'Cholesky':>10}")
    print("-" * 60)
    ok = True
    for n, p in [(1, 1), (5, 1), (7, 3), (8, 8), (20, 5), (32, 32),
                 (64, 32), (200, 16), (1000, 8), (1000, 32)]:
        X = rng.uniform(-1000.0, 1000.0, (n, p))
        y = rng.uniform(-1000.0, 1000.0, n)
        A, b, pp = gram(X, y)
        want = reference(X, y)
        kappa = np.linalg.cond(X.astype(np.float64))
        e_gj = judge(gj_solve(A, b, pp), want)
        e_ch = judge(chol_solve(A, b, pp), want)
        ok &= e_gj < 1.0
        print(f"{n:>5}×{p:<4} {kappa:8.2f} {kappa**2 * 1.2e-7:9.2e} | "
              f"{e_gj:10.2e} {e_ch:10.2e}  {'PASS' if e_gj < 1.0 else 'FAIL'}")

    print("\n近共线输入的失效边界（决定 float32 够不够用）：")
    print(f"{'ε':>9} {'κ(X)':>10} {'κ²·ε':>9} | {'GJ':>10} {'Cholesky':>10}")
    print("-" * 60)
    for eps in (1e-1, 3e-2, 1e-2, 3e-3, 1e-3):
        n = p = 32
        X = rng.uniform(-1000.0, 1000.0, (n, p))
        X[:, -1] = X[:, 0] * (1.0 + eps) + rng.normal(0.0, abs(eps) * 1000.0, n)
        y = rng.uniform(-1000.0, 1000.0, n)
        A, b, pp = gram(X, y)
        want = reference(X, y)
        kappa = np.linalg.cond(X.astype(np.float64))
        e_gj = judge(gj_solve(A, b, pp), want)
        e_ch = judge(chol_solve(A, b, pp), want)
        print(f"{eps:9.0e} {kappa:10.1f} {kappa**2 * 1.2e-7:9.2e} | "
              f"{e_gj:10.2e} {e_ch:10.2e}")

    print("\n随机形状全部通过 ✓" if ok else "\n存在失败 ✗")


if __name__ == "__main__":
    main()
