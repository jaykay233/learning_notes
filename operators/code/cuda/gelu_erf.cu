// GELU（erf 精确形式）：Y[i] = 0.5 * X[i] * (1 + erf(X[i] / sqrt(2)))
//
// 这是 elementwise 算子，没有任何归约、也没有操作数复用。
// 因此：分类结论 = 纯带宽受限，优化手段只有「向量化 + 无分支 + 网格跨步」。
//
// 编译示例：
//   nvcc -O3 -arch=sm_75 gelu_erf.cu -c
// 判定是否带宽受限：算子算术强度 ≈ 2*4B/8B = 1 FLOP/Byte
//   远小于 T4 的机器平衡点 8.1 TFLOPS / 320 GB/s ≈ 25 FLOP/Byte

#include <cuda_runtime.h>
#include <math.h>
#include <stddef.h>

namespace {

// erf(x / sqrt(2)) 里的 1/sqrt(2)
constexpr float kInvSqrt2 = 0.70710678118654752440f;

__device__ __forceinline__ float gelu_scalar(float x) {
    // 0.5 * x * (1 + erf(x * 1/sqrt2))
    return 0.5f * x * (1.0f + erff(x * kInvSqrt2));
}

// ---------------------------------------------------------------- 最简版
// 一个线程一个元素。正确性最好写，但每个线程只搬 4 字节，
// 单次访存事务的利用率取决于相邻线程是否连续 —— 这里恰好是连续的。
__global__ void gelu_naive(const float* __restrict__ X, float* __restrict__ Y, int N) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    Y[i] = gelu_scalar(X[i]);
}

// ------------------------------------------------------------ 向量化版本
// 每个线程一次搬 16 字节（float4），全局访存事务数降到 1/4。
// 用网格跨步循环，所以 grid 大小与 N 解耦，不必开满 1e8/4/256 = 97657 个 block。
__global__ void __launch_bounds__(256)
gelu_vec4(const float4* __restrict__ X4, float4* __restrict__ Y4, int n4) {
    const int stride = gridDim.x * blockDim.x;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += stride) {
        const float4 v = X4[i];
        float4 r;
        r.x = gelu_scalar(v.x);
        r.y = gelu_scalar(v.y);
        r.z = gelu_scalar(v.z);
        r.w = gelu_scalar(v.w);
        Y4[i] = r;
    }
}

// N 不是 4 的倍数时的尾部（0~3 个元素）
__global__ void gelu_tail(const float* __restrict__ X, float* __restrict__ Y,
                          int start, int N) {
    const int i = start + blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) Y[i] = gelu_scalar(X[i]);
}

}  // namespace

// X、Y 是 device 指针
extern "C" void solve(const float* X, float* Y, int N) {
    constexpr int kThreads = 256;

    // cudaMalloc 返回 256B 对齐，reinterpret_cast 到 float4 的 16B 对齐有保证。
    const int n4 = N / 4;
    if (n4 > 0) {
        int blocks = (n4 + kThreads - 1) / kThreads;
        if (blocks > 65535) blocks = 65535;  // 网格跨步循环会吃完剩余部分
        gelu_vec4<<<blocks, kThreads>>>(reinterpret_cast<const float4*>(X),
                                        reinterpret_cast<float4*>(Y), n4);
    }

    const int rem = N - n4 * 4;
    if (rem > 0) {
        const int blocks = (rem + kThreads - 1) / kThreads;
        gelu_tail<<<blocks, kThreads>>>(X, Y, n4 * 4, N);
    }

    cudaDeviceSynchronize();
}

// 未使用的朴素版保留作对照：同一份 N 下它比 vec4 慢，
// 差距正是「每线程 4B 访存 vs 16B 访存」体现出的指令数与事务数差异。
extern "C" void solve_naive(const float* X, float* Y, int N) {
    constexpr int kThreads = 256;
    const int blocks = (N + kThreads - 1) / kThreads;
    gelu_naive<<<blocks, kThreads>>>(X, Y, N);
    cudaDeviceSynchronize();
}
