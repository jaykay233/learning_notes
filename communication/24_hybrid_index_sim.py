#!/usr/bin/env python3
"""Encode/decode hybrid rank and global token index g."""

from __future__ import annotations


def main() -> None:
    num_scaleout, num_scaleup = 2, 4
    num_experts, max_tok = 16, 8
    e_per_scaleout = num_experts // num_scaleout
    e_per_rank = num_experts // (num_scaleout * num_scaleup)

    def split_expert(e: int):
        so = e // e_per_scaleout
        su = (e % e_per_scaleout) // e_per_rank
        return so, su

    src_so, src_su, t = 0, 1, 5
    src_rank = src_so * num_scaleup + src_su
    g = src_rank * max_tok + t
    assert g // max_tok == src_rank and g % max_tok == t
    assert g // (max_tok * num_scaleup) == src_so

    topk = [3, 12]
    print(f"src_rank={src_rank} (so={src_so}, su={src_su}) t={t} g={g}")
    for e in topk:
        so, su = split_expert(e)
        print(f"  expert {e:2d} -> scaleout={so} scaleup={su}  "
              f"{'local-so' if so == src_so else 'cross-so'}")

    g2 = 13
    print(f"decode g={g2}: rank={g2 // max_tok} t={g2 % max_tok} "
          f"so={g2 // (max_tok * num_scaleup)}")


if __name__ == "__main__":
    main()
