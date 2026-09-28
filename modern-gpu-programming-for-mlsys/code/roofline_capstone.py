from dataclasses import dataclass


PEAK_COMPUTE_TFLOPS = 2000.0
HBM_BANDWIDTH_TBPS = 8.0
TB = 1e12
TFLOP = 1e12
FP16_BYTES = 2


@dataclass(frozen=True)
class RooflineResult:
    name: str
    flops: float
    bytes_moved: float
    arithmetic_intensity: float
    memory_roof_tflops: float
    attainable_tflops: float
    bottleneck: str


def analyze(
    name: str,
    flops: float,
    bytes_moved: float,
    peak_compute_tflops: float = PEAK_COMPUTE_TFLOPS,
    hbm_bandwidth_tbps: float = HBM_BANDWIDTH_TBPS,
) -> RooflineResult:
    if flops <= 0:
        raise ValueError("flops must be positive")
    if bytes_moved <= 0:
        raise ValueError("bytes_moved must be positive")

    arithmetic_intensity = flops / bytes_moved
    memory_roof_tflops = hbm_bandwidth_tbps * TB * arithmetic_intensity / TFLOP
    attainable_tflops = min(peak_compute_tflops, memory_roof_tflops)
    bottleneck = "compute" if peak_compute_tflops <= memory_roof_tflops else "memory"

    return RooflineResult(
        name=name,
        flops=flops,
        bytes_moved=bytes_moved,
        arithmetic_intensity=arithmetic_intensity,
        memory_roof_tflops=memory_roof_tflops,
        attainable_tflops=attainable_tflops,
        bottleneck=bottleneck,
    )


def print_result(result: RooflineResult) -> None:
    print(result.name)
    print(f"  flops              = {result.flops:.6e}")
    print(f"  bytes              = {result.bytes_moved:.6e}")
    print(f"  arithmetic_intensity = {result.arithmetic_intensity:.6f} FLOP/byte")
    print(f"  memory_roof        = {result.memory_roof_tflops:.6f} TFLOP/s")
    print(f"  attainable         = {result.attainable_tflops:.6f} TFLOP/s")
    print(f"  bottleneck         = {result.bottleneck}")
    print()


def main() -> None:
    ridge_point = PEAK_COMPUTE_TFLOPS / HBM_BANDWIDTH_TBPS
    print("B200 rounded roofline inputs")
    print(f"  peak_compute = {PEAK_COMPUTE_TFLOPS:.1f} TFLOP/s")
    print(f"  hbm_bandwidth = {HBM_BANDWIDTH_TBPS:.1f} TB/s")
    print(f"  ridge_point = {ridge_point:.1f} FLOP/byte")
    print()

    gemm_n = 4096
    gemm_flops = 2 * gemm_n**3
    gemm_bytes = 3 * FP16_BYTES * gemm_n**2

    blk_m = 128
    blk_n = 128
    blk_k = 64
    stage_flops = 2 * blk_m * blk_n * blk_k
    stage_bytes = FP16_BYTES * (blk_m * blk_k + blk_k * blk_n)

    seq_len = 4096
    head_dim = 128
    attention_flops = 4 * seq_len**2 * head_dim
    materialized_bytes = 8 * seq_len**2 + 8 * seq_len * head_dim
    flash_prefill_bytes = 8 * seq_len * head_dim

    decode_query_count = 1
    decode_flops = 4 * decode_query_count * seq_len * head_dim
    decode_bytes = 4 * seq_len * head_dim

    results = [
        analyze(
            f"GEMM ideal square: N={gemm_n}, fp16",
            gemm_flops,
            gemm_bytes,
        ),
        analyze(
            f"GEMM CTA stage: Bm={blk_m}, Bn={blk_n}, Bk={blk_k}, fp16",
            stage_flops,
            stage_bytes,
        ),
        analyze(
            f"Attention materialized: S={seq_len}, D={head_dim}, fp16",
            attention_flops,
            materialized_bytes,
        ),
        analyze(
            f"Flash Attention prefill: S={seq_len}, D={head_dim}, fp16",
            attention_flops,
            flash_prefill_bytes,
        ),
        analyze(
            f"Decode one token: S={seq_len}, D={head_dim}, fp16 KV cache",
            decode_flops,
            decode_bytes,
        ),
    ]

    for result in results:
        print_result(result)

    assert results[0].bottleneck == "compute"
    assert results[1].bottleneck == "memory"
    assert results[2].bottleneck == "memory"
    assert results[3].bottleneck == "compute"
    assert results[4].bottleneck == "memory"
    print("classification assertions: PASS")


if __name__ == "__main__":
    main()
