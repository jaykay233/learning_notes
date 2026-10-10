#!/usr/bin/env python3
"""Map MoE EP forward ops to their DeepEP backward duals."""

from __future__ import annotations


DUAL = {
    "dispatch": "combine",  # dispatch_backward
    "combine": "dispatch",  # combine_backward
}


def backward_comm(forward_op: str) -> str:
    return DUAL[forward_op]


def main() -> None:
    print("EP training backward dual (same kernels, reuse handle):")
    for fwd in ("dispatch", "combine"):
        print(f"  forward {fwd:8s} -> backward calls {backward_comm(fwd)}")

    print("product roles:")
    print("  inference MoE  -> EP (not PP)")
    print("  training MoE   -> EP forward + dual backward")
    print("  PP/Engram/CP   -> experimental side paths")


if __name__ == "__main__":
    main()
