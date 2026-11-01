#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Muon 单步更新的对拍脚本：与本题规格的 PyTorch 实现逐元素比对。

本脚本只对齐「本题规格」，不碰 torch.optim.Muon，也不碰 SVD 极因子。
原因：三者的定义不同，混在一起对拍会得到错误结论。

    本题规格 (R1)  5 步 NS + 系数 a,b,c + eps + 缩放 sqrt(max(M,N))
    PyTorch Muon   同一迭代，但缩放 0.2*sqrt(max(M,N))  → 与 R1 差恒定 5 倍
    精确极因子     U V^T                              → 与 R1 差 1e-2 量级

判对错只看第 1 列。

三条实现路径
-----------
1. `spec(dtype=float64)`   规格的参考实现，float64，作为 ground truth
2. `emulate()`             规格的 float32 实现，**按 CUDA 内核的运算顺序**写
3. `MuonCUDA`              通过 ctypes 调用编译好的 libmuon.so

对拍时 (2) vs (1) 给出「公式理解是否正确 + float32 舍入多少」，
(3) vs (1) 给出「内核实现是否正确」。若 (3) 错而 (2) 对，问题一定在内核。

用法
----
    # 只跑 PyTorch 两条路径（无需 GPU，可在 macOS 上验证公式）
    python muon_reference_check.py

    # 加上 CUDA 内核
    nvcc -O3 -arch=sm_75 -shared -Xcompiler -fPIC muon.cu -o libmuon.so
    python muon_reference_check.py --lib ./libmuon.so

    # 只看某个形状的逐步奇异值轨迹
    python muon_reference_check.py --trace 1024 768
"""

from __future__ import annotations

import argparse
import ctypes
import math
import sys

import torch

# ---------------------------------------------------------------------------
# 规格常量。这 4 项是整道题的「定义」，改动任何一项都会整体改变结果。
# 若某个用例失败，最先应该怀疑这里，而不是内核。
# ---------------------------------------------------------------------------
COEF_A, COEF_B, COEF_C = 3.4445, -4.7750, 2.0315
EPS = 1e-7          # 全零用例必需：没有它 0/0 = NaN
STEPS = 5
V_MODE = 0          # 0: V = momentum*Buf + G
                    # 1: V = momentum*Buf + (1 - momentum)*G   （PyTorch lerp 写法）

TORCH_F32 = torch.float32
TORCH_F64 = torch.float64


# ===========================================================================
# 1. 规格参考实现（ground truth）
# ===========================================================================
def spec(G: torch.Tensor, Buf: torch.Tensor, momentum: float,
         dtype=TORCH_F64, steps: int = STEPS, eps: float = EPS,
         v_mode: int = V_MODE) -> torch.Tensor:
    """按题面文字逐句实现。返回 float32 以便与内核比对。

    数学上等价于一个只作用在奇异值上的标量映射：
        out = sqrt(max(M,N)) * U @ p^steps(S / ||S||_F) @ V^T
    因为 A = X X^T = U S^2 U^T 对 U 对角化，整个迭代在 U/V 基下完全解耦。
    这一点让所有数值直觉都可以退化成一维函数 p 的性质。
    """
    G = G.to(dtype)
    Buf = Buf.to(dtype)

    # ① 混合
    V = momentum * Buf + (G if v_mode == 0 else (1.0 - momentum) * G)

    M, N = V.shape
    tr = M > N
    # M > N 时在转置上迭代：Gram 矩阵落在 min(M,N) 维度。
    # 这个等价是精确的：(V V^T)^k V 与 ((V^T V)^k V^T)^T 恒等。
    X = V.T.contiguous() if tr else V.clone()

    # ② Frobenius 归一化，使 sigma_max <= 1（因为 sigma_max <= ||V||_F）
    X = X / (torch.linalg.norm(X) + eps)

    # ③ 5 步 quintic Newton-Schulz
    for _ in range(steps):
        Am = X @ X.T
        Bm = COEF_B * Am + COEF_C * (Am @ Am)
        X = COEF_A * X + Bm @ X

    if tr:
        X = X.T.contiguous()

    # ④ 缩放
    return (X * math.sqrt(max(M, N))).to(torch.float32)


def emulate(G: torch.Tensor, Buf: torch.Tensor, momentum: float) -> torch.Tensor:
    """规格的 float32 实现，运算顺序与 CUDA 内核一致。

    与内核逐段对应：
        combine_kernel          → V = momentum*Buf + G
        frob_sq_kernel (double) → ||V||_F
        scale_kernel            → X /= (||V||_F + eps)
        sgemm ×3                → Am = X X^T, Am@Am, Bm@X
        axpby_kernel            → Bm = c*A2 + b*Am
        融合 epilogue           → X = a*Xold + (Bm@X)
    """
    G = G.to(TORCH_F32)
    Buf = Buf.to(TORCH_F32)

    V = momentum * Buf + (G if V_MODE == 0 else (1.0 - momentum) * G)

    M, N = V.shape
    tr = M > N
    X = V.T.contiguous() if tr else V.clone()

    # 内核用 double 累加 Frobenius 范数，这里保持一致
    nrm = torch.linalg.norm(X.to(TORCH_F64)).item()
    X = X / float(nrm + EPS)

    for _ in range(STEPS):
        Am = X @ X.T
        A2 = Am @ Am
        Bm = COEF_C * A2 + COEF_B * Am          # 注意顺序：c*A2 + b*A
        X = COEF_A * X + Bm @ X

    if tr:
        X = X.T.contiguous()
    return (X * math.sqrt(max(M, N))).to(TORCH_F32)


# ===========================================================================
# 2. CUDA 内核入口
# ===========================================================================
class MuonCUDA:
    """extern "C" void solve(const float* G, const float* Buf, float* out,
                            int M, int N, float momentum)"""

    def __init__(self, path: str):
        self.lib = ctypes.CDLL(path)
        self.lib.solve.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int, ctypes.c_int, ctypes.c_float,
        ]
        self.lib.solve.restype = None

    def __call__(self, G: torch.Tensor, Buf: torch.Tensor, momentum: float) -> torch.Tensor:
        G = G.contiguous()
        Buf = Buf.contiguous()
        M, N = G.shape
        Gc = G.cuda()
        Bc = Buf.cuda()
        out = torch.empty(M, N, device="cuda", dtype=TORCH_F32)
        self.lib.solve(
            ctypes.c_void_p(Gc.data_ptr()),
            ctypes.c_void_p(Bc.data_ptr()),
            ctypes.c_void_p(out.data_ptr()),
            M, N, float(momentum),
        )
        torch.cuda.synchronize()
        return out.cpu()


# ===========================================================================
# 3. 诊断：把「max abs diff」翻译成「哪里错了」
# ===========================================================================
def report(got: torch.Tensor, want: torch.Tensor, band: int = 64,
           tag: str = "", tol: float = 2e-4) -> bool:
    """打印误差分布。返回是否通过。

    关键的两个探针：
      * 按行带统计 max diff   → 尾部成片错（转置写回漏行）一眼可见
      * 「want 非零但 got 为 0」→ 最典型的「没写」症状，并给出行范围
    """
    d = (got.float() - want.float()).abs()
    M = want.shape[0]
    mx = d.max().item()
    ok = mx < tol * max(1.0, want.abs().max().item())

    print(f"\n  [{tag}] max={mx:.3e}  mean={d.mean().item():.3e}  "
          f"→ {'PASS' if ok else 'FAIL'}")

    # 探针 1：未写区域
    miss = (got == 0) & (want != 0)
    if miss.any():
        idx = miss.nonzero()
        rows = idx[:, 0].unique()
        print(f"  ⚠ {miss.sum().item()} 个元素 want 非零而 got 为 0")
        print(f"    首个位置 (row, col) = ({idx[0, 0].item()}, {idx[0, 1].item()})")
        print(f"    涉及行 {rows.min().item()} … {rows.max().item()}"
              f"（{rows.numel()} 行，占 {rows.numel() / M:.1%}）")

    # 探针 2：按行带定位
    rows_max = d.amax(dim=1)
    nb = (M + band - 1) // band
    bad = []
    for i in range(nb):
        r = rows_max[i * band:(i + 1) * band].max().item()
        if r > tol:
            bad.append((i * band, min((i + 1) * band, M), r))

    if not bad:
        print(f"    全部 {nb} 个行带（每带 {band} 行）误差 < {tol:.0e} ✓")
    else:
        print(f"    超出容差的行带 {len(bad)}/{nb}:")
        for lo, hi, r in bad[:10]:
            flag = "  ← 整带为 0" if got[lo:hi].count_nonzero() == 0 else ""
            print(f"      行 [{lo:5d}, {hi:5d})  max_diff={r:.3e}{flag}")
        if len(bad) > 10:
            print(f"      ... 还有 {len(bad) - 10} 个")
    return ok


def with_small_tolerance(want: torch.Tensor, got: torch.Tensor) -> bool:
    """float32 路径之间的严格比对：相对 1e-5。"""
    scale = max(1.0, want.abs().max().item())
    return (got - want).abs().max().item() < 1e-5 * scale


# ===========================================================================
# 4. 逐步轨迹：把「标量映射」这件事直接打出来
# ===========================================================================
def trace(M: int, N: int, momentum: float = 0.9) -> None:
    """打印归一化后每一步的奇异值区间。

    预期形态：第 0 步区间很宽（例如 [1e-3, 1]），之后逐步收拢到 1 附近，
    但不会精确收敛到 1 —— 这是本题规格的核心，不是 bug。
    """
    torch.manual_seed(0)
    G = torch.randn(M, N)
    Buf = torch.randn(M, N)
    V = momentum * Buf + G
    tr = M > N
    X = (V.T.contiguous() if tr else V.clone()).double()
    X = X / (torch.linalg.norm(X) + EPS)

    print(f"\nM={M} N={N} tr={tr}  工作矩阵 {X.shape[0]}x{X.shape[1]}")
    print(f"  step  sigma_min      sigma_max      sigma_max 距 1")
    for s in range(STEPS + 1):
        sv = torch.linalg.svdvals(X)
        print(f"  {s:4d}  {sv.min().item():.4e}   {sv.max().item():.6f}"
              f"      {abs(sv.max().item() - 1.0):.4f}")
        if s == STEPS:
            break
        Am = X @ X.T
        Bm = COEF_B * Am + COEF_C * (Am @ Am)
        X = COEF_A * X + Bm @ X

    print("\n  标量映射的性质（对奇异值逐步相扣，所以这些数字决定一切）：")
    p = lambda s: COEF_A * s + COEF_B * s ** 3 + COEF_C * s ** 5
    dp = lambda s: COEF_A + 3 * COEF_B * s ** 2 + 5 * COEF_C * s ** 4
    print(f"    p(0)={p(0.0):.4f}   p(1)={p(1.0):.4f}   p(0.868)={p(0.868):.4f}")
    print(f"    p'(0)={dp(0.0):.4f}  p'(1)={dp(1.0):.4f}  min p'≈{min(dp(s/1000) for s in range(1, 1500)):.4f}")
    print("  轨迹演示：", " → ".join(f"{v:.3f}" for v in _orbit(1.0, 6)))
    print("  小奇异值：", " → ".join(f"{v:.3e}" for v in _orbit(1e-3, 5)))


def _orbit(s0: float, n: int):
    out, s = [s0], s0
    for _ in range(n):
        s = COEF_A * s + COEF_B * s ** 3 + COEF_C * s ** 5
        out.append(s)
    return out


# ===========================================================================
# 5. 主流程
# ===========================================================================
SHAPES = [
    (1024, 1024),   # 方阵 → 不转置
    (1024,  768),   # M > N → 走转置路径（历史 bug 都在这里）
    ( 768, 1024),   # M < N → 不转置，但 R/C 与上一行完全相同
    ( 512,  512),
    (  37,   91),   # 两个方向都不对齐 128
    (   1,    1),   # 退化
    (   1,  256),   # M < N 极端
    ( 256,    1),   # M > N 极端 → 转置
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lib", default=None, help="libmuon.so 路径；给了才会测 CUDA 内核")
    ap.add_argument("--momentum", type=float, default=0.9)
    ap.add_argument("--tol", type=float, default=2e-4)
    ap.add_argument("--trace", nargs=2, type=int, metavar=("M", "N"),
                    help="只打印该形状的逐步奇异值轨迹然后退出")
    args = ap.parse_args()

    # 重要：float32 矩阵乘默认可能走 TF32（Ampere+），会把误差放大到 1e-3 量级，
    # 从而把「实现错误」和「精度选择错误」混在一起。对拍前必须关掉。
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        torch.set_float32_matmul_precision("highest")
    except Exception:
        pass

    if args.trace:
        trace(args.trace[0], args.trace[1], args.momentum)
        return 0

    dev = MuonCUDA(args.lib) if args.lib else None
    momentum = args.momentum
    torch.manual_seed(0)

    print("=" * 78)
    print(f"规格：a={COEF_A} b={COEF_B} c={COEF_C}  eps={EPS}  steps={STEPS}  "
          f"V_MODE={V_MODE}")
    print("=" * 78)

    # ---------------- 第 1 组：float32 仿真 vs float64 参考 ----------------
    print("\n第 1 组  float32 仿真 vs float64 参考")
    print("         （验证公式理解；不涉及 CUDA）")
    all_ok = True
    for (M, N) in SHAPES:
        G = torch.randn(M, N)
        Buf = torch.randn(M, N)
        want64 = spec(G, Buf, momentum, dtype=TORCH_F64)
        got32 = emulate(G, Buf, momentum)
        all_ok &= report(got32, want64, band=max(1, min(64, M)),
                         tag=f"{M}x{N} f32仿真", tol=args.tol)

    # ---------------- 第 2 组：切断 momentum ----------------
    print("\n第 2 组  隔离测试（Buf=0, momentum=0 → V=G）")
    print("         （把 NS 迭代与 V 组合/转置的嫌疑分开）")
    for (M, N) in [(1024, 768), (768, 1024), (1024, 1024)]:
        G = torch.randn(M, N)
        Z = torch.zeros(M, N)
        want = spec(G, Z, 0.0, dtype=TORCH_F64)
        got = emulate(G, Z, 0.0)
        all_ok &= report(got, want, band=max(1, min(64, M)),
                         tag=f"{M}x{N} 隔离", tol=args.tol)

    # ---------------- 第 3 组：边界用例 ----------------
    print("\n第 3 组  边界用例")
    edge = {
        "全零 G,Buf": (torch.zeros(64, 48), torch.zeros(64, 48)),
        "G 全零 (Buf 非零)": (torch.zeros(64, 48), torch.randn(64, 48)),
        "1e-8 量级": (torch.randn(64, 48) * 1e-8, torch.randn(64, 48) * 1e-8),
        "单元素": (torch.randn(1, 1), torch.randn(1, 1)),
    }
    for name, (G, Buf) in edge.items():
        want = spec(G, Buf, momentum, dtype=TORCH_F64)
        got = emulate(G, Buf, momentum)
        d = (got - want).abs().max().item()
        nan = torch.isnan(got).any().item()
        print(f"  {name:20s} max_diff={d:.3e}  "
              f"非零项={got.count_nonzero().item():5d}  "
              f"{'NaN ✗' if nan else 'no NaN ✓'}  期望非零={want.count_nonzero().item()}")
        if nan:
            all_ok = False

    # 1e-8 量级的期望行为：必然有 ~EPS/||V|| 的相对偏差，这是规格不是 bug
    G = torch.randn(64, 48) * 1e-8
    Buf = torch.randn(64, 48) * 1e-8
    V = momentum * Buf + G
    ratio = EPS / (torch.linalg.norm(V).item() + EPS)
    print(f"  1e-8 用例：||V||_F={torch.linalg.norm(V).item():.3e}  "
          f"eps/(||V||+eps)={ratio:.3%}  ← 归一化被 eps 稀释的幅度")

    # ---------------- 第 4 组：CUDA 内核 ----------------
    if dev is None:
        print("\n第 4 组  跳过（未提供 --lib）")
        print("         编译：nvcc -O3 -arch=sm_75 -shared -Xcompiler -fPIC muon.cu -o libmuon.so")
    else:
        print("\n第 4 组  CUDA 内核 vs float64 参考")
        for (M, N) in SHAPES:
            G = torch.randn(M, N)
            Buf = torch.randn(M, N)
            want = spec(G, Buf, momentum, dtype=TORCH_F64)
            got = dev(G, Buf, momentum)
            all_ok &= report(got, want, band=max(1, min(64, M)),
                             tag=f"{M}x{N} CUDA", tol=args.tol)

        # 与 float32 仿真比：两者都是 f32，差异应只剩舍入
        print("\n         CUDA vs float32 仿真（只该剩舍入，相对 1e-5）")
        for (M, N) in [(1024, 1024), (1024, 768), (768, 1024)]:
            G = torch.randn(M, N)
            Buf = torch.randn(M, N)
            a = dev(G, Buf, momentum)
            b = emulate(G, Buf, momentum)
            d = (a - b).abs().max().item()
            tight = with_small_tolerance(b, a)
            print(f"  {M}x{N}: max_diff={d:.3e}  {'✓' if tight else '✗ 差异超出舍入范围'}")
            if not tight:
                report(a, b, band=max(1, min(64, M)), tag=f"{M}x{N} cuda-vs-emul")
            all_ok &= tight

    print("\n" + "=" * 78)
    print("全部通过 ✓" if all_ok else "存在失败 ✗ （先看上面标了 ⚠ 或「整带为 0」的行带）")
    print("=" * 78)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
