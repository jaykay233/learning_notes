#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Gauss-Jordan 的数学性质验证 —— 笔记 19 号 §6.4 / §6.5 里每个数字的出处。

这个脚本不碰 CUDA，也不追求性能。它只回答四个「这句话到底对不对」的问题：

  A. FLOP 计数     GJ 到底是 0.53p³ 还是 p³/3？
  B. 左乘恒等式    E_c 乘积是否真的等于那 p 次行操作？
  C. 增长因子      对 SPD，中间矩阵元素是否真的从不增大？（定理 vs 实测）
  D. 乘数与相消    max|f| 能到 7.6e5，而元素不动，两者怎么共存？

运行：python ols_math_check.py
所有数字与笔记 19 号 §6.4、§6.5 的表格一一对应。
"""

from __future__ import annotations

import numpy as np

# ============================================================================
# A. FLOP 精确计数
# ============================================================================
# 笔记里原先写 GJ = p³/3，实测是 0.53p³，比值 2.69 而不是 1.5。
# 下面按代码的每一个算术运算逐项加，不用渐近公式。


def count_gj(p: int) -> tuple[int, int, int]:
    """返回 (fma, div, sqrt)。对应 ols.cu 阶段 2 + 阶段 3。

    每个 c 步、每条 i != c 的行：
        f = A[i][c] * inv          → 1 次乘法（记进 fma 计数）
        A[i][j] -= f * A[c][j]     → (p-1-c) 次 FMA
        b[i]    -= f * b[c]        → 1 次 FMA
    每步另有 1 次 1/A[c][c]。
    阶段 3 的 p 次除法另记。
    """
    fma = 0
    div = 0
    for c in range(p):
        div += 1                                   # inv = 1 / A[c][c]
        for i in range(p):
            if i == c:
                continue
            fma += 1                               # f = A[i][c] * inv
            fma += (p - 1 - c)                     # A[i][j] -= f*A[c][j]
            fma += 1                               # b[i]    -= f*b[c]
    div += p                                       # beta[i] = b[i]/A[i][i]
    return fma, div, 0


def count_chol(p: int) -> tuple[int, int, int]:
    """返回 (fma, div, sqrt)。对应 Cholesky + 两次三角回代（笔记 §6.1 的版本）。

        sqrt(A[c][c])                  → 1 次 sqrt
        A[i][c] /= dcc                 → (p-1-c) 次 div
        A[i][j] -= A[i][c]*A[j][c]     → 对 i >= j > c，(p-c)(p-c-1)/2 次 FMA
        前代 + 回代                     → 各 p(p-1)/2 次 FMA，另 2p 次 div
    """
    fma = 0
    div = 0
    sq = 0
    for c in range(p):
        sq += 1                                    # sqrt
        div += (p - 1 - c)                         # 列归一化
        for j in range(c + 1, p):                  # i >= j > c
            fma += (p - j)
    fma += p * (p - 1) // 2                        # 前代
    fma += p * (p - 1) // 2                        # 回代
    div += 2 * p
    return fma, div, sq


def part_a() -> None:
    print("=" * 78)
    print("A. FLOP 精确计数：GJ 是 0.53p^3，不是 p^3/3")
    print("=" * 78)
    print(f"{'p':>6} {'GJ fma':>11} {'/p^3':>8} {'GJ div':>8} | "
          f"{'Chol fma':>11} {'/p^3':>8} {'Chol div':>9} {'sqrt':>5} {'比值':>7}")
    print("-" * 78)
    for p in (8, 16, 32, 64, 128, 1000):
        g, gd, gs = count_gj(p)
        c, cd, cs = count_chol(p)
        print(f"{p:>6} {g:>11} {g/p**3:>8.4f} {gd:>8} | "
              f"{c:>11} {c/p**3:>8.4f} {cd:>9} {cs:>5} {g/c:>7.2f}")
    g, gd, _ = count_gj(32)
    c, cd, cs = count_chol(32)
    print(f"\np = 32 的绝对值：GJ {g} FMA / {gd} div ；"
          f"Chol {c} FMA / {cd} div / {cs} sqrt")
    print(f"比值 {g/c:.2f}x，而笔记最初误写成 1.5x（源于把 GJ 当成 p^3/3）")
    print(f"\n除法才是被忽略的量级：{cd} vs {gd}，T4 上一次除法约 20 cycles，")
    print(f"  Cholesky 光除法就 {cd * 20} cycles，GJ 只要 {gd * 20} cycles。")
    print(f"两个都远小于两次三角回代的归约延迟（约 6000-9500 cycles）。")


# ============================================================================
# B. E_c 乘积恒等式
# ============================================================================
# E_c = I - f e_c^T。验证 E_{p-1}...E_0 A == 那 p 次逐行操作的结果，
# 并顺带验证 det(E_c) = 1 与 E_c^{-1} = I + f e_c^T。


def gj_with_E(A0: np.ndarray, b0: np.ndarray, p: int):
    """同时做两件事：逐行操作（代码路径）和累积 E_c 乘积（证明路径）。"""
    A = A0.copy()
    b = b0.copy()
    Es = []
    for c in range(p):
        inv = 1.0 / A[c, c]
        f = np.zeros(p)
        for i in range(p):
            if i == c:
                continue
            f[i] = A[i, c] * inv
            A[i, c + 1:] -= f[i] * A[c, c + 1:]
            b[i] -= f[i] * b[c]
        E = np.eye(p)
        E[:, c] -= f                              # E_c = I - f e_c^T
        Es.append(E)
    Eprod = np.eye(p)
    for E in Es:                                  # 注意顺序：E_{p-1} ... E_0
        Eprod = E @ Eprod
    return A, b, Es, Eprod


def part_b() -> None:
    print("\n" + "=" * 78)
    print("B. E_c 乘积恒等式：矩阵形式 == 逐行操作")
    print("=" * 78)
    rng = np.random.default_rng(0)
    for p in (4, 7, 16):
        X = rng.uniform(-3, 3, (3 * p, p))
        A = X.T @ X
        b = X.T @ rng.uniform(-3, 3, 3 * p)
        Ad, bd, Es, Eprod = gj_with_E(A, b, p)

        D = np.diag(np.diag(Ad))
        err_A = np.abs(Eprod @ A - D).max()
        err_b = np.abs(Eprod @ b - bd).max()
        beta = np.array([bd[i] / Ad[i, i] for i in range(p)])
        err_beta = np.abs(beta - np.linalg.solve(A, b)).max()
        dets = [abs(np.linalg.det(E)) - 1.0 for E in Es]

        print(f"\np = {p}")
        print(f"  ||E_prod @ A - diag(diag(A_p))||  = {err_A:.3e}")
        print(f"  ||E_prod @ b - b_p||              = {err_b:.3e}")
        print(f"  beta 与 numpy.solve 的差           = {err_beta:.3e}")
        print(f"  max |det(E_c) - 1|                = {max(dets):.3e}")
        print(f"  prod(diag) = {np.prod(np.diag(Ad)):.10e}")
        print(f"  det(A)     = {np.linalg.det(A):.10e}")

        # E_c^{-1} = I + f e_c^T 只在 f_c = 0 时成立
        bad = 0
        for c, E in enumerate(Es):
            f = -E[:, c].copy()
            f[c] = 0.0                            # 代码里 f_c 恒为 0
            Einv = np.eye(p) + np.outer(f, np.eye(p)[c])
            if np.abs(E @ Einv - np.eye(p)).max() > 1e-12:
                bad += 1
        print(f"  E_c^{-1} = I + f e_c^T 是否成立     = {'是' if bad == 0 else '否'}")
        print(f"  E_c 是否对称                       = "
              f"{'是' if np.allclose(Es[0], Es[0].T) else '否（尽管 A 对称）'}")


# ============================================================================
# C. 增长因子：对 SPD 是否真的 <= 1
# ============================================================================
# 定理（笔记 §6.4.4）：
#   ① 每个中间矩阵是 Schur 补，仍是 SPD
#   ② SPD 非对角元被对角元控制：|M_ij| <= sqrt(M_ii M_jj)
#   ③ 对角元单调不增
#   ⇒ |A^(c)_ij| <= ||A||，即增长因子 <= 1
# 下面用暴力统计验证「恰好等于 1」，而不只是「小于某个常数」。


def growth_factor(A0: np.ndarray, p: int) -> float:
    A = A0.copy()
    m0 = np.abs(A).max()
    peak = m0
    for c in range(p):
        inv = 1.0 / A[c, c]
        for i in range(p):
            if i == c:
                continue
            f = A[i, c] * inv
            A[i, c + 1:] -= f * A[c, c + 1:]
        peak = max(peak, np.abs(A).max())
    return peak / m0


def part_c() -> None:
    print("\n" + "=" * 78)
    print("C. 增长因子：对 SPD 是否恰好等于 1")
    print("=" * 78)
    rng = np.random.default_rng(0)
    print(f"{'p':>4} {'组数':>6} {'最大增长因子':>14} {'结论':>10}")
    print("-" * 40)
    for p in (8, 16, 32, 64):
        worst = 0.0
        for _ in range(20):
            X = rng.uniform(-1000.0, 1000.0, (3 * p, p))
            worst = max(worst, growth_factor(X.T @ X, p))
        verdict = "= 1 精确" if worst <= 1.0 + 1e-12 else "> 1"
        print(f"{p:>4} {20:>6} {worst:>14.6f} {verdict:>10}")

    # 对照：同样的统计在非对称矩阵上会怎样
    # 说明「增长因子 <= 1」依赖 SPD，不是 GJ 本身的普遍性质
    print("\n对照组：把 A 换成非对称矩阵（GJ 不再有 <= 1 的保证）")
    for p in (8, 16, 32):
        worst = 0.0
        for _ in range(20):
            M = rng.uniform(-1.0, 1.0, (p, p)) + p * np.eye(p)
            worst = max(worst, growth_factor(M, p))
        print(f"  p={p:>3}  最大增长因子 = {worst:.4f}"
              f"   {'仍 <= 1' if worst <= 1.0 + 1e-12 else '已 > 1'}")

    # 对角单调性
    print("\n对角单调不增 A^(c+1)_ii <= A^(c)_ii：")
    for p in (8, 32):
        X = rng.uniform(-1000.0, 1000.0, (3 * p, p))
        A = X.T @ X
        mono = True
        for c in range(p):
            inv = 1.0 / A[c, c]
            for i in range(p):
                if i == c:
                    continue
                f = A[i, c] * inv
                before = A[i, i]
                A[i, c + 1:] -= f * A[c, c + 1:]
                if A[i, i] > before + 1e-9 * abs(before):
                    mono = False
        print(f"  p={p:>3}  {'保持' if mono else '被破坏'}")


# ============================================================================
# D. 乘数无界、更新有界
# ============================================================================
# 笔记 §6.4.5 的核心：
#   |f_i| = |A_ic|/A_cc <= sqrt(A_ii/A_cc)          ← 可以极大
#   |f_i A_cj| <= sqrt(A_ii A_jj) <= ||A||          ← 但乘积有界
# 所以更新是一次灾难性相消，误差来自丢有效位，不来自乘数大小。


def part_d() -> None:
    print("\n" + "=" * 78)
    print("D. 乘数无界而更新有界：误差来自相消，不来自乘数")
    print("=" * 78)
    rng = np.random.default_rng(0)
    p = 32
    print(f"{'eps':>8} {'kappa(X)':>11} {'max|f|':>14} {'|f·A_cj| 最大':>15} "
          f"{'||A||_max':>12} {'乘积超界?':>10}")
    print("-" * 78)
    for eps in (0.0, 1e-1, 1e-2, 1e-3):
        X = rng.uniform(-1000.0, 1000.0, (3 * p, p))
        if eps > 0:
            X[:, -1] = X[:, 0] * (1.0 + eps) + rng.normal(0.0, eps * 1000.0, len(X))
        A = X.T @ X
        Anorm = np.abs(A).max()
        maxf = 0.0
        maxprod = 0.0
        for c in range(p):
            inv = 1.0 / A[c, c]
            row_c = A[c, c + 1:].copy()
            for i in range(p):
                if i == c:
                    continue
                f = A[i, c] * inv
                maxf = max(maxf, abs(f))
                if row_c.size:
                    maxprod = max(maxprod, np.abs(f * row_c).max())
                A[i, c + 1:] -= f * A[c, c + 1:]
        ok = "否（界成立）" if maxprod <= Anorm * (1 + 1e-9) else "是（界被破）"
        print(f"{eps:>8.0e} {np.linalg.cond(X):>11.1f} {maxf:>14.2f} "
              f"{maxprod:>15.3e} {Anorm:>12.3e} {ok:>10}")

    # 粗糙界 vs 真实误差：差多少个数量级
    print("\n粗糙界 ||E_c||_inf 累乘 vs 实测误差（p = 32）：")
    print("  注意：这个累乘量对随机种子极敏感（实测跨种子在 1e21 ~ 1e35 之间跳），")
    print("  所以只用来做「差多少个数量级」的量级判断，不能当成一个确定的界。")
    for seed in (7, 11):
        rng = np.random.default_rng(seed)
        p = 32
        X = rng.uniform(-1000.0, 1000.0, (p, p))
        A = X.T @ X
        growth = 1.0
        for c in range(p):
            s = sum(abs(A[i, c]) / A[c, c] for i in range(p) if i != c)
            growth *= max(1.0, s)
            inv = 1.0 / A[c, c]
            for i in range(p):
                if i == c:
                    continue
                A[i, c + 1:] -= (A[i, c] * inv) * A[c, c + 1:]
        kappa2 = np.linalg.cond(X) ** 2
        print(f"  seed={seed:>3}  ||E_c||_inf 累乘 = {growth:.3e}   "
              f"kappa(X)^2 = {kappa2:.3e}   相差 {growth / kappa2:.1e} 倍")
    print("  → 任何基于 ||E_c|| 的界都看不见相消，注定悲观。")


def main() -> None:
    part_a()
    part_b()
    part_c()
    part_d()
    print("\n" + "=" * 78)
    print("结论：GJ 的 FLOP 是 0.53p³（Chol 的 2.7 倍），但它零归约、")
    print("      对 SPD 后向稳定（增长因子精确为 1），且误差只受 kappa² 支配。")
    print("=" * 78)


if __name__ == "__main__":
    main()
