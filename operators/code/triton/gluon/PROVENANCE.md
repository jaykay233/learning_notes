# Triton Gluon tutorial provenance

This directory contains a study archive of the official Triton Gluon tutorial
series.  The archive is kept beside the learning notes so the lessons remain
self-contained when the upstream source tree is unavailable.

- Upstream repository: <https://github.com/triton-lang/triton>
- Upstream tutorial index: <https://triton-lang.org/main/getting-started/tutorials/gluon/index.html>
- Fixed source revision: `41846bc965e51306d574f0dab721039f97f24996`
- Retrieved: 2026-10-13
- Archived path: `operators/code/triton/gluon/tutorials/`

## Archived files

| File | Main topic | Runtime requirement |
|---|---|---|
| `01-intro.py` | Gluon JIT, programs, grids and a first memcpy | CUDA |
| `02-layouts.py` | distributed and shared-memory layouts | CUDA |
| `03-async-copy.py` | `cp.async`, commit groups and pipelining | Ampere or newer |
| `04-tma.py` | tensor descriptors, TMA and mbarriers | Hopper or newer |
| `05-wgmma.py` | WGMMA / MMAv3 | Hopper |
| `06-tcgen05.py` | tensor memory and tcgen05 MMA | Blackwell |
| `07-persistence.py` | persistent tiles and pipelined persistent GEMM | Hopper or newer |
| `08-warp-specialization.py` | warp-specialized producer/MMA/epilogue kernels | Hopper or newer |
| `09-tma-gather-scatter.py` | TMA gather/scatter | Blackwell |
| `10-tcgen05-copy.py` | asynchronous shared-to-tensor-memory copy | Blackwell |
| `11-tcgen05-mma-scaled.py` | block-scaled MMA and quantized GEMM | Blackwell |
| `12-cluster-launch-control.py` | CLC dynamic work distribution | Blackwell SM100+ |
| `13-conv-im2col.py` | TMA im2col and implicit-GEMM convolution | Hopper or newer |
| `14-multicta.py` | CGA, distributed shared memory and 2CTA MMA | Hopper+; full MMA path Blackwell |
| `conftest.py` | pytest path setup used by the upstream tutorials | Python only |

The original Triton license is preserved as [`LICENSE`](LICENSE).  These files
are reference material; the explanatory Markdown lessons live under
`operators/07-*.md` through `operators/12-*.md`.

The local machine used to create these notes did not have a Triton runtime or a
CUDA GPU.  The archive and Markdown were checked statically only; benchmark
numbers quoted from the upstream tutorials are not local measurements.
