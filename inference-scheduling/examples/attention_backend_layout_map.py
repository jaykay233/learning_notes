#!/usr/bin/env python3
"""Attention backend 的 KV 布局 / 架构门槛 / spec-decode 分派对照表。

只用标准库，不需要 GPU。默认运行三张表：

    1. NHD / HND / LHBNC 三种布局的 stride，以及「取一个子块要分成几段」
    2. decode 与 prefill 的算术强度，以及相对 roofline 拐点（ridge）的倍率
    3. vLLM attention backend 的布局声明与架构门槛（源码快照表）

带 --scan 时第 3 张表改为从 vLLM 源码现场抽取，用来核对换版本后的事实。

    python3 attention_backend_layout_map.py
    python3 attention_backend_layout_map.py --scan /path/to/vllm
"""

import argparse
import itertools
import os
import re
import sys

# ---------------------------------------------------------------------------
# 1. 布局：逻辑形状恒为 [L, B, H, N, C]，值是该顺序映射到物理（内存）顺序的排列
#    与 vllm/v1/kv_cache_layout.py 的 KVCacheLayout 一致。
# ---------------------------------------------------------------------------
LOGICAL = ("L", "B", "H", "N", "C")

LAYOUTS = {
    "NHD   = LBNHC": (0, 1, 3, 2, 4),  # L, B, N, H, C
    "HND   = LBHNC": (0, 1, 2, 3, 4),  # L, B, H, N, C
    "LHBNC": (0, 2, 1, 3, 4),  # L, H, B, N, C（head 跨 block 最外）
}

# 典型形状：kv_head 数、每 page token 数、head_dim，以及要扫的历史 page 数
DEFAULT_PAGE = {"H": 8, "N": 16, "C": 128}
DEFAULT_BLOCKS = 4
ELEM_BYTES = 2  # fp16 / bf16


def strides(layout_perm, sizes):
    """返回 {dim: stride_in_elements}，按「最内层 stride=1」反向累乘。"""
    stride = {}
    acc = 1
    for dim_index in reversed(layout_perm):
        dim = LOGICAL[dim_index]
        stride[dim] = acc
        acc *= sizes[dim]
    return stride


def segments(layout_perm, sizes, vary, fixed):
    """把子块展开成字节偏移集合，返回 (段数, 每段字节数)。

    vary:  参与变化的逻辑维，例如 ("N", "C") 表示「一个 head 的整页 token」
    fixed: 固定的逻辑维取值，例如 {"B": 0, "L": 0, "H": 3}
    """
    stride = strides(layout_perm, sizes)
    offsets = []
    ranges = [range(sizes[dim]) for dim in vary]
    base = sum(fixed.get(dim, 0) * stride[dim] for dim in LOGICAL if dim not in vary)
    for coords in itertools.product(*ranges):
        offsets.append(base + sum(c * stride[dim] for c, dim in zip(coords, vary)))
    offsets.sort()
    runs = 1
    for prev, cur in zip(offsets, offsets[1:]):
        if cur != prev + 1:
            runs += 1
    return runs, (len(offsets) // runs) * ELEM_BYTES


def report_layouts(page, elem_bytes, num_blocks):
    sizes = dict(page)
    sizes.update({"L": 1, "B": num_blocks})
    page_bytes = page["H"] * page["N"] * page["C"] * elem_bytes
    print("=" * 78)
    print(f"[1] KV 布局与子块分段   page=[H={page['H']}, N={page['N']}, C={page['C']}] "
          f"{elem_bytes}B/elem, 历史 {num_blocks} 页")
    print("=" * 78)
    print(f"一页总字节 = H*N*C*elem = {page['H']}*{page['N']}*{page['C']}*{elem_bytes}"
          f" = {page_bytes} B = {page_bytes / 1024:.0f} KiB（三种布局都是一整块连续内存）")
    print()

    for name, perm in LAYOUTS.items():
        st = strides(perm, sizes)
        print(f"{name}")
        print("  stride(byte)  " + "  ".join(
            f"{d}={st[d] * elem_bytes}" for d in ("H", "B", "N", "C")))

        patterns = [
            ("decode 单页: 1 head × N × C", ("N", "C"), {"B": 0, "L": 0, "H": 3}),
            (f"decode 扫 {num_blocks} 页: 1 head × B × N × C", ("B", "N", "C"),
             {"L": 0, "H": 3}),
            ("prefill 单页: 1 token × H × C", ("H", "C"), {"B": 0, "L": 0, "N": 5}),
            ("搬单页: H × N × C", ("H", "N", "C"), {"B": 0, "L": 0}),
        ]
        for label, vary, fixed in patterns:
            runs, seg_bytes = segments(perm, sizes, vary, fixed)
            print(f"  {label:<32} -> {runs:>2} 段 × 每段 {seg_bytes:>5} B")
        print()

    print("判据：段数越少、每段越长，DRAM 请求数越少。整页搬运时三者完全相同，")
    print("      差别只出现在 kernel 内层循环沿哪个维展开、以及是否跨 page。")
    print()


# ---------------------------------------------------------------------------
# 2. 算术强度：一个 decode step 里每个 KV 字节被乘加几次
# ---------------------------------------------------------------------------
def report_roofline(tflops, bw_tbs, gqa, elem_bytes):
    ridge = tflops * 1e12 / (bw_tbs * 1e12)
    print("=" * 78)
    print(f"[2] 算术强度与 roofline   {tflops:.0f} TFLOPS / {bw_tbs:.2f} TB/s")
    print("=" * 78)
    print("推导：L 个 query token、S 个 KV token、H_q=G*H_kv 个 query head、head_dim D")
    print("  FLOPs = 2*L*S*D*H_q (QK^T) + 2*L*S*D*H_q (P·V) = 4*L*S*D*G*H_kv")
    print(f"  KV 字节 = 2(K,V)*S*D*H_kv*{elem_bytes}B = {2 * elem_bytes}*S*D*H_kv")
    print(f"  => 强度 = 4*L*S*D*G*H_kv / ({2 * elem_bytes}*S*D*H_kv)"
          f" = {4 / (2 * elem_bytes):.1f}*L*G  FLOP/B")
    print(f"  ridge = {tflops:.0f}e12 / {bw_tbs}e12 = {ridge:.0f} FLOP/B "
          f"(超过即算力 bound)")
    print()
    header = "  ".join(f"L={L:<5}" for L in (1, 2, 4, 8))
    print(f"{'模型 (H_q/H_kv)':<24}{'G':>4}   {header}")
    for model, g in (
        ("Llama-3-8B  32/8", 4),
        ("Llama-3-70B 64/8", 8),
        ("Mistral-7B  32/8", 4),
        ("DeepSeek MLA", 1),
    ):
        cells = []
        for L in (1, 2, 4, 8):
            ai = 4 * L * g / (2 * elem_bytes)
            cells.append(f"{ai:>5.0f}{'*' if ai >= ridge else ' '}")
        print(f"{model:<24}{g:>4}   " + "".join(f"{c:>7}" for c in cells))
    print()
    need = {g: ridge * (2 * elem_bytes) / (4 * g) for g in (1, 4, 8, 16)}
    print("跨过 ridge 所需的 query token 数 L（*=算力 bound）：")
    print("  " + "  ".join(f"G={g}: L>={L:.0f}" for g, L in need.items()))
    print()
    print("结论：投机解码把 L 从 1 抬到 γ+1，乘上 GQA 的复用，仍远低于 ridge；")
    print("      强度变高只是减少带宽余量，不改变「decode 是带宽 bound」这一性质。")
    print()


# ---------------------------------------------------------------------------
# 3. backend 对照表（源码快照）
# ---------------------------------------------------------------------------
SNAPSHOT = [
    # name, vendor, arch gate, 负责阶段, 布局, spec-as-decode
    ("trtllm-gen", "NVIDIA", "SM100/103/107",
     "decode+prefill", "强制 HND", "是（device 端裁 draft）"),
    ("XQA", "NVIDIA", "SM90 / SM120 家族",
     "decode (MQA/GQA, beam)", "HND 原生；TMA 支持 NHD", "是（SPEC_DEC 分支）"),
    ("FlashInfer native", "NVIDIA", "SM80+",
     "decode+prefill", "任意（stride-aware）", "否 → 退回 prefill 路径"),
    ("FA3 / FA4", "NVIDIA", "SM90 / SM100",
     "prefill 为主", "任意（stride-aware）", "不切分，varlen 一把跑"),
    ("aiter (AITER)", "AMD", "gfx9 CDNA3+",
     "decode (asm) + prefill", "LHBNC / SHUFFLE", "是（MTP kernel）"),
    ("hpc (hpc-ops)", "Tencent", "SM90 only",
     "decode(MTP)+prefill", "LBNHC = NHD", "是（mtp<=3）"),
    ("TurboQuant", "vLLM", "见 backend",
     "decode", "LBNHC = NHD", "否"),
]


def report_backends():
    print("=" * 78)
    print("[3] attention backend 对照（源码快照）")
    print("=" * 78)
    widths = (18, 9, 18, 22, 24, 24)
    header = ("内核", "厂商", "架构门槛", "负责阶段", "KV 布局", "spec-as-decode")
    print("  ".join(str(h).ljust(w) for h, w in zip(header, widths)))
    print("-" * 78)
    for row in SNAPSHOT:
        print("  ".join(str(c).ljust(w) for c, w in zip(row, widths)))
    print()
    print("读法：布局这一列和 spec 这一列不成正比。hpc 有 MTP kernel 却用 NHD，")
    print("      所以「spec decode ⇒ HND」是伪相关；真正决定布局的是 kernel 内层循环。")
    print()


# ---------------------------------------------------------------------------
# 4. --scan：从源码现场抽取 supported_kv_cache_layouts()
# ---------------------------------------------------------------------------
LAYOUT_RE = re.compile(r"KVCacheLayout\.([A-Z]+)")


def scan(root):
    backend_dir = os.path.join(root, "vllm", "v1", "attention", "backends")
    if not os.path.isdir(backend_dir):
        print(f"!! 找不到 {backend_dir}", file=sys.stderr)
        return 1
    print("=" * 78)
    print(f"[4] 现场扫描 {backend_dir}")
    print("=" * 78)
    hits = 0
    for dirpath, _, filenames in os.walk(backend_dir):
        for fn in sorted(filenames):
            if not fn.endswith(".py"):
                continue
            path = os.path.join(dirpath, fn)
            with open(path, encoding="utf-8") as fh:
                src = fh.read()
            idx = src.find("def supported_kv_cache_layouts")
            if idx < 0:
                continue
            body = src[idx:idx + 1200]
            # 只保留到下一个同级 def 为止，避免把别处的方法算进来
            nxt = body.find("\n    def ", 10)
            if nxt > 0:
                body = body[:nxt]
            names = []
            for m in LAYOUT_RE.finditer(body):
                if m.group(1) not in names:
                    names.append(m.group(1))
            rel = os.path.relpath(path, backend_dir)
            layout_txt = ", ".join(names) if names else "(未声明 = 任意布局)"
            print(f"{rel:<52} {layout_txt}")
            hits += 1
    print()
    print(f"共 {hits} 个 backend 声明了布局偏好；未声明者读 stride，不读布局名。")
    print()
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scan", metavar="VLLM_ROOT",
                    help="vLLM 仓库根目录，从源码现场抽取布局声明")
    ap.add_argument("--tflops", type=float, default=989.0, help="fp16 峰值 TFLOPS")
    ap.add_argument("--bw", type=float, default=3.35, help="HBM 带宽 TB/s")
    ap.add_argument("--gqa", type=int, default=8, help="GQA 组大小 G")
    ap.add_argument("--blocks", type=int, default=DEFAULT_BLOCKS,
                    help="模拟的历史 page 数")
    args = ap.parse_args()

    report_layouts(DEFAULT_PAGE, ELEM_BYTES, args.blocks)
    report_roofline(args.tflops, args.bw, args.gqa, ELEM_BYTES)
    if args.scan:
        return scan(args.scan)
    report_backends()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
