#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""OLS 单 kernel 版的对拍脚本。

参考实现用 float64 解正规方程，不用 numpy.linalg.lstsq：
  - lstsq 走 SVD，与本题的「正规方程 + Cholesky」不是同一条数值路径，
    比出来的差值里混了算法差异，定位不了 kernel 的 bug。
  - 用 float64 解正规方程，与 C++ 那边是同一条公式，只剩精度差，
    这样超差就一定是我们这边的问题。

用法：
    nvcc -O3 -arch=sm_75 -Xptxas -v -shared -Xcompiler -fPIC ols.cu -o libols.so
    python ols_check.py
"""

from __future__ import annotations

import argparse
import ctypes
import sys

import numpy as np


class OlsLib:
    def __init__(self, path: str):
        self.lib = ctypes.CDLL(path)
        self.lib.solve.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int, ctypes.c_int,
        ]
        self.lib.solve.restype = None

    def run(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        n, p = X.shape
        Xc = np.ascontiguousarray(X, dtype=np.float32)
        yc = np.ascontiguousarray(y, dtype=np.float32)
        beta = np.zeros(p, dtype=np.float32)
        self.lib.solve(
            ctypes.c_void_p(Xc.ctypes.data),
            ctypes.c_void_p(yc.ctypes.data),
            ctypes.c_void_p(beta.ctypes.data),
            n, p,
        )
        return beta


def reference(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """float64 解正规方程，与 kernel 同一条公式。"""
    Xd = X.astype(np.float64)
    yd = y.astype(np.float64)
    A = Xd.T @ Xd
    b = Xd.T @ yd
    return np.linalg.solve(A, b)


def margin(X: np.ndarray, p: int) -> float:
    """κ(X)² × ε_fp32，用来预测误差上界。"""
    s = np.linalg.svd(X.astype(np.float64), compute_uv=False)
    kappa = s[0] / s[-1]
    return kappa, kappa ** 2 * 1.2e-7


def rel_err(got: np.ndarray, want: np.ndarray) -> float:
    """判题口径：abs tol 1e-2 与 rel tol 1e-2 取宽的那个。"""
    d = np.abs(got.astype(np.float64) - want)
    return float(np.max(d / (1e-2 + 1e-2 * np.abs(want))))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lib", default="./libols.so")
    ap.add_argument("--big", action="store_true", help="加上 n=100000 的慢用例")
    args = ap.parse_args()

    try:
        lib = OlsLib(args.lib)
    except OSError as e:
        print(f"加载 {args.lib} 失败: {e}")
        print("编译：nvcc -O3 -arch=sm_75 -Xptxas -v -shared -Xcompiler -fPIC "
              "ols.cu -o libols.so")
        return 1

    rng = np.random.default_rng(0)
    ok_all = True

    print("=" * 84)
    print("正确性：gpu kernel vs float64 正规方程")
    print("=" * 84)
    print(f"{'n×p':>12}  {'κ(X)':>9}  {'预测上界':>9}  {'判题口径误差':>12}  "
          f"{'max|β|':>9}  结果")

    cases = [(32, 32), (64, 32), (1000, 32), (1000, 16), (1000, 8),
             (7, 3), (1, 1), (5, 1), (100, 2), (17, 17)]
    if args.big:
        cases.append((100_000, 32))

    for n, p in cases:
        # 与题面一致：[-1000, 1000] 均匀分布
        X = rng.uniform(-1000.0, 1000.0, (n, p))
        y = rng.uniform(-1000.0, 1000.0, n)
        got = lib.run(X, y)
        want = reference(X, y)
        kappa, bound = margin(X, p)
        e = rel_err(got, want)
        ok = e < 1.0
        ok_all &= ok
        print(f"{n:>6}×{p:<5}  {kappa:9.2f}  {bound:9.2e}  {e:12.2e}  "
              f"{np.abs(want).max():9.2e}  {'PASS' if ok else 'FAIL'}")

    # ---- 边界 ----
    print("\n边界：")
    for name, (n, p) in {"n=p=1": (1, 1), "p=1": (50, 1),
                         "n=p=32": (32, 32)}.items():
        X = rng.uniform(-1000, 1000, (n, p))
        y = rng.uniform(-1000, 1000, n)
        got = lib.run(X, y)
        want = reference(X, y)
        e = rel_err(got, want)
        nan = not np.all(np.isfinite(got))
        print(f"  {name:10s} 误差={e:.2e}  NaN/Inf={'YES ✗' if nan else 'no'}")
        if nan or e >= 1.0:
            ok_all = False

    # ---- 病态输入：这是「正规方程把条件数平方」的直接检验 ----
    print("\n病态（近共线的列，检验 §精度 的推导）：")
    n, p = 32, 32
    X = rng.uniform(-1000, 1000, (n, p))
    for eps in (1e-2, 1e-3, 1e-4):
        Xp = X.copy()
        Xp[:, -1] = Xp[:, 0] * (1.0 + eps) + rng.normal(0, eps, n)
        y = rng.uniform(-1000, 1000, n)
        got = lib.run(Xp, y)
        want = reference(Xp, y)
        kappa, bound = margin(Xp, p)
        e = rel_err(got, want)
        print(f"  ε={eps:<6g} κ={kappa:10.1f}  预测={bound:8.1e}  "
              f"实测={e:8.2e}  {'PASS' if e < 1.0 else 'FAIL'}")

    # ---- 已知答案：y = X·beta_true 时 beta 应精确还原 ----
    print("\n已知答案（y = X·β_true）：")
    for n, p in [(64, 8), (100, 32), (1000, 32)]:
        bt = rng.normal(0, 1, p)
        X = rng.uniform(-1000, 1000, (n, p))
        y = (X.astype(np.float64) @ bt).astype(np.float32)
        got = lib.run(X, y)
        e = rel_err(got, bt)
        print(f"  {n:>6}×{p:<4} 判题口径误差={e:.2e}  "
              f"{'PASS' if e < 1.0 else 'FAIL'}")
        ok_all &= e < 1.0

    print("\n" + "=" * 84)
    print("全部通过 ✓" if ok_all else "存在失败 ✗")
    print("=" * 84)
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
