#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Muon 两个版本的对拍 + 计时。

用途：
  1. 正确性 —— muon_opt.cu（优化版）与 muon.cu（基线，已验证）逐元素比对。
     两者都通过了，优化版才算可用。
  2. 计时 —— 验证「SYRK 省 33%」是否真的兑现。

对拍对象用基线内核而不是 float64 参考，原因是：
  两个版本走同一条 float32 路径，只差 tiling 与 kernel 数量，
  互相比对时容差可以收紧到「只剩累加顺序差异」。用 float64 参考会让
  两者的误差都被 5 步迭代的误差放大掩盖，看不出优化版是否引入了新错误。

用法：
    nvcc -O3 -arch=sm_75 -shared -Xcompiler -fPIC muon.cu     -o libmuon.so
    nvcc -O3 -arch=sm_75 -shared -Xcompiler -fPIC muon_opt.cu -o libmuon_opt.so
    python bench_muon.py
"""

from __future__ import annotations

import argparse
import ctypes
import math
import sys
import time

import torch


class MuonLib:
    def __init__(self, path: str):
        self.path = path
        self.lib = ctypes.CDLL(path)
        self.lib.solve.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int, ctypes.c_int, ctypes.c_float,
        ]
        self.lib.solve.restype = None

    def run(self, G: torch.Tensor, Buf: torch.Tensor, momentum: float) -> torch.Tensor:
        M, N = G.shape
        Gc = G.contiguous().cuda()
        Bc = Buf.contiguous().cuda()
        out = torch.empty(M, N, device="cuda", dtype=torch.float32)
        self.lib.solve(
            ctypes.c_void_p(Gc.data_ptr()),
            ctypes.c_void_p(Bc.data_ptr()),
            ctypes.c_void_p(out.data_ptr()),
            M, N, float(momentum),
        )
        torch.cuda.synchronize()
        return out.cpu()

    def time_ms(self, G: torch.Tensor, Buf: torch.Tensor, momentum: float,
                iters: int = 20, warmup: int = 5) -> float:
        M, N = G.shape
        Gc = G.contiguous().cuda()
        Bc = Buf.contiguous().cuda()
        out = torch.empty(M, N, device="cuda", dtype=torch.float32)
        gp, bp, op = (ctypes.c_void_p(Gc.data_ptr()),
                      ctypes.c_void_p(Bc.data_ptr()),
                      ctypes.c_void_p(out.data_ptr()))
        for _ in range(warmup):
            self.lib.solve(gp, bp, op, M, N, float(momentum))
        torch.cuda.synchronize()

        # CUDA events：比 time.time() 精确，且不受 host 侧抖动影响
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            self.lib.solve(gp, bp, op, M, N, float(momentum))
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / iters


def diag(a: torch.Tensor, b: torch.Tensor, tag: str) -> bool:
    """a=新版, b=基线。返回是否在容差内。"""
    d = (a.float() - b.float()).abs()
    mx = d.max().item()
    scale = max(1.0, b.abs().max().item())
    rel = mx / scale
    ok = rel < 3e-5          # 只该剩累加顺序差异

    print(f"  {tag:20s} max={mx:.3e}  rel={rel:.3e}  "
          f"mean={d.mean().item():.3e}  {'PASS' if ok else 'FAIL'}")

    if not ok:
        M = b.shape[0]
        band = max(1, min(64, M))
        rows = d.amax(dim=1)
        bad = []
        for i in range((M + band - 1) // band):
            r = rows[i * band:(i + 1) * band].max().item()
            if r > 1e-3 * scale:
                bad.append((i * band, min((i + 1) * band, M), r))
        for lo, hi, r in bad[:8]:
            print(f"      行 [{lo:5d},{hi:5d}) max={r:.3e}")
        miss = (a == 0) & (b != 0)
        if miss.any():
            idx = miss.nonzero()
            print(f"      ⚠ {miss.sum().item()} 个元素基线非零而新版为 0，"
                  f"首个 ({idx[0,0].item()}, {idx[0,1].item()})")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="./libmuon.so")
    ap.add_argument("--opt", default="./libmuon_opt.so")
    ap.add_argument("--momentum", type=float, default=0.9)
    ap.add_argument("--iters", type=int, default=20)
    args = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        torch.set_float32_matmul_precision("highest")
    except Exception:
        pass

    try:
        base = MuonLib(args.base)
        opt = MuonLib(args.opt)
    except OSError as e:
        print(f"加载 .so 失败: {e}")
        print("编译命令：")
        print("  nvcc -O3 -arch=sm_75 -shared -Xcompiler -fPIC muon.cu     -o libmuon.so")
        print("  nvcc -O3 -arch=sm_75 -shared -Xcompiler -fPIC muon_opt.cu -o libmuon_opt.so")
        return 1

    torch.manual_seed(0)
    mom = args.momentum

    print("=" * 78)
    print("正确性：muon_opt vs muon（基线，已与 float64 参考对拍通过）")
    print("=" * 78)
    ok_all = True
    for (M, N) in [(1024, 1024), (1024, 768), (768, 1024),
                   (512, 512), (37, 91), (1, 1), (1, 256), (256, 1)]:
        G = torch.randn(M, N)
        Buf = torch.randn(M, N)
        a = opt.run(G, Buf, mom)
        b = base.run(G, Buf, mom)
        ok_all &= diag(a, b, f"{M}x{N}")

    # 边界
    print("\n边界：")
    for name, (G, Buf) in {
        "全零": (torch.zeros(64, 48), torch.zeros(64, 48)),
        "1e-8": (torch.randn(64, 48) * 1e-8, torch.randn(64, 48) * 1e-8),
    }.items():
        a = opt.run(G, Buf, mom)
        b = base.run(G, Buf, mom)
        nan = torch.isnan(a).any().item()
        d = (a - b).abs().max().item()
        print(f"  {name:10s} max={d:.3e}  NaN={'YES ✗' if nan else 'no'}  "
              f"非零={a.count_nonzero().item()}")
        if nan:
            ok_all = False

    print("\n" + "=" * 78)
    print(f"计时：M=N=1024，momentum={mom}，{args.iters} 次取平均")
    print("=" * 78)
    M = N = 1024
    G = torch.randn(M, N)
    Buf = torch.randn(M, N)

    t_base = base.time_ms(G, Buf, mom, args.iters)
    t_opt = opt.time_ms(G, Buf, mom, args.iters)
    print(f"  基线 muon.cu      {t_base:7.3f} ms")
    print(f"  优化 muon_opt.cu  {t_opt:7.3f} ms")
    print(f"  加速比            {t_base / t_opt:7.3f}x   "
          f"（减少 {100 * (1 - t_opt / t_base):.1f}%）")

    # 和理论值对照
    print(f"\n  理论：每 iter 165(A) + 165(A²) + 330(B@X) + 20(镜像) ≈ 680 µs")
    print(f"        5 iter ≈ 3.40 ms；基线 5 × 990 ≈ 4.95 ms")
    print(f"  实测每 iter：基线 {t_base / 5 * 1000:.0f} µs，"
          f"优化 {t_opt / 5 * 1000:.0f} µs")

    print("\n" + "=" * 78)
    print("全部通过 ✓" if ok_all else "存在失败 ✗")
    print("=" * 78)
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
