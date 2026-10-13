"""Minimal Blackwell demo: D = A @ B + C using tcgen05.copy.

This pedagogical single-CTA kernel keeps one 128x128 output tile and one K tile.
It deliberately uses ordinary global loads and is not a performance kernel.
"""

import torch
import triton.experimental.gluon as gluon
import triton.experimental.gluon.language as gl
from triton.experimental.gluon.language.nvidia.blackwell import (
    TensorMemoryLayout,
    allocate_tensor_memory,
    fence_async_shared,
    mbarrier,
    tcgen05_commit,
    tcgen05_copy,
    tcgen05_mma,
)


@gluon.jit
def accumulate_one_tile_kernel(a_ptr, b_ptr, c_ptr, d_ptr):
    BM: gl.constexpr = 128
    BN: gl.constexpr = 128
    BK: gl.constexpr = 64

    # 4 contiguous values/thread, a 4x8 lane tile, and four warps over M.
    reg_layout: gl.constexpr = gl.BlockedLayout(
        [1, 4], [4, 8], [4, 1], [1, 0]
    )
    rm = gl.arange(0, BM, gl.SliceLayout(1, reg_layout))
    rn = gl.arange(0, BN, gl.SliceLayout(0, reg_layout))
    rk_a = gl.arange(0, BK, gl.SliceLayout(0, reg_layout))
    rk_b = gl.arange(0, BK, gl.SliceLayout(1, reg_layout))

    a = gl.load(a_ptr + rm[:, None] * BK + rk_a[None, :])
    b = gl.load(b_ptr + rk_b[:, None] * BN + rn[None, :])
    c = gl.load(c_ptr + rm[:, None] * BN + rn[None, :])

    a_layout: gl.constexpr = gl.NVMMASharedLayout.get_default_for(
        [BM, BK], gl.float16
    )
    b_layout: gl.constexpr = gl.NVMMASharedLayout.get_default_for(
        [BK, BN], gl.float16
    )
    c_layout: gl.constexpr = gl.NVMMASharedLayout.get_default_for(
        [BM, BN], gl.float32
    )
    a_smem = gl.allocate_shared_memory(gl.float16, [BM, BK], a_layout)
    b_smem = gl.allocate_shared_memory(gl.float16, [BK, BN], b_layout)
    c_smem = gl.allocate_shared_memory(gl.float32, [BM, BN], c_layout)
    acc_tmem = allocate_tensor_memory(
        gl.float32, [BM, BN], TensorMemoryLayout([BM, BN], col_stride=1)
    )

    done = gl.allocate_shared_memory(
        gl.int64, [1], gl.constexpr(mbarrier.MBarrierLayout())
    )
    mbarrier.init(done, count=1)

    # Generic-proxy SMEM stores become visible to the async tensor pipe.
    a_smem.store(a)
    b_smem.store(b)
    c_smem.store(c)
    fence_async_shared()

    # One ordered tensor-pipe sequence: seed accumulator with C, then add A@B.
    tcgen05_copy(c_smem, acc_tmem)
    tcgen05_mma(a_smem, b_smem, acc_tmem, use_acc=True)
    tcgen05_commit(done)
    mbarrier.wait(done, phase=0)
    mbarrier.invalidate(done)

    # Only after the wait may ordinary threads read the completed accumulator.
    out = acc_tmem.load()
    out = gl.convert_layout(out, reg_layout)
    gl.store(d_ptr + rm[:, None] * BN + rn[None, :], out)


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("This demo requires a CUDA GPU with Blackwell support")
    if torch.cuda.get_device_capability()[0] < 10:
        raise RuntimeError("This demo requires NVIDIA compute capability 10.x or newer")

    torch.manual_seed(0)
    a = torch.randn((128, 64), device="cuda", dtype=torch.float16)
    b = torch.randn((64, 128), device="cuda", dtype=torch.float16)
    c = torch.randn((128, 128), device="cuda", dtype=torch.float32)
    d = torch.empty_like(c)

    accumulate_one_tile_kernel[(1,)](a, b, c, d, num_warps=4)
    reference = a.float() @ b.float() + c
    torch.testing.assert_close(d, reference, atol=5e-3, rtol=1e-2)
    max_abs_error = (d - reference).abs().max().item()
    print(f"D = A @ B + C: check passed; max_abs_error={max_abs_error:.6g}")


if __name__ == "__main__":
    main()
