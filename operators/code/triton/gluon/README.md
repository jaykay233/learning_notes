# Gluon tutorial archive

This is a fixed-revision archive of the official Gluon tutorials, used by the
six Gluon lessons in `operators/`:

1. [Programming model and layouts](../../../07-triton-gluon-programming-model-and-layouts.md)
2. [Async copy, TMA and pipelining](../../../08-triton-gluon-async-tma-and-pipelining.md)
3. [WGMMA and tcgen05](../../../09-triton-gluon-wgmma-and-tcgen05.md)
4. [Persistent kernels and warp specialization](../../../10-triton-gluon-persistent-warp-specialization.md)
5. [Blackwell dataflow and convolution](../../../11-triton-gluon-blackwell-dataflow-and-convolution.md)
6. [Multi-CTA and the capstone GEMM](../../../12-triton-gluon-multicta-and-capstone.md)

The examples import the upstream modules directly.  They require a compatible
Triton checkout, PyTorch and the GPU generation named in the table in
[`PROVENANCE.md`](PROVENANCE.md).  Start with `pytest` from this directory, for
example:

```bash
cd operators/code/triton/gluon/tutorials
pytest 01-intro.py -q
```

The archived source is pinned to the revision recorded in `PROVENANCE.md`; do
not silently mix it with a different Triton API version when debugging layout
or synchronization errors.
