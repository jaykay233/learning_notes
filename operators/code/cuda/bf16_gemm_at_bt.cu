// Dense GeMM (bf16): C = A @ B.T
// A: (M, K) row-major, B: (N, K) row-major, C: (M, N) row-major
// C[i,j] = sum_k A[i,k] * B[j,k]
//
// 需要 sm_80+（Ampere 及以上）的 BF16 Tensor Core / WMMA。
// 编译示例：nvcc -O3 -arch=sm_80 bf16_gemm_at_bt.cu

#include <stdint.h>
#include <cuda_bf16.h>
#include <mma.h>

using namespace nvcuda;

namespace {

constexpr int WM = 16;
constexpr int WN = 16;
constexpr int WK = 16;

__device__ __forceinline__ __nv_bfloat16 f2bf16(float x) {
    return __float2bfloat16(x);
}

__device__ __forceinline__ float bf162f(__nv_bfloat16 x) {
    return __bfloat162float(x);
}

// 边界 / 非 16 对齐：一线程算一个 C[i,j]
__global__ void gemm_bt_scalar(const __nv_bfloat16* __restrict__ A,
                               const __nv_bfloat16* __restrict__ B,
                               __nv_bfloat16* __restrict__ C,
                               int64_t M, int64_t N, int64_t K) {
    const int64_t j = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t i = (int64_t)blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= M || j >= N) return;

    float acc = 0.f;
    const __nv_bfloat16* a_row = A + i * K;
    const __nv_bfloat16* b_row = B + j * K;
    for (int64_t k = 0; k < K; ++k) {
        acc += bf162f(a_row[k]) * bf162f(b_row[k]);
    }
    C[i * N + j] = f2bf16(acc);
}

// 每个 warp 计算 16x16 的 C tile；利用：
//   B^T 的 col-major(ld=K) 恰好等于 row-major 的 B
__global__ void gemm_bt_wmma(const __nv_bfloat16* __restrict__ A,
                             const __nv_bfloat16* __restrict__ B,
                             __nv_bfloat16* __restrict__ C,
                             int64_t M, int64_t N, int64_t K) {
    // blockDim.x = 32（单 warp）
    const int warp_m = (int)blockIdx.y;
    const int warp_n = (int)blockIdx.x;
    const int i0 = warp_m * WM;
    const int j0 = warp_n * WN;
    if (i0 >= M || j0 >= N) return;

    wmma::fragment<wmma::matrix_a, WM, WN, WK, __nv_bfloat16, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WM, WN, WK, __nv_bfloat16, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, WM, WN, WK, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    for (int64_t k0 = 0; k0 < K; k0 += WK) {
        const __nv_bfloat16* a_ptr = A + (int64_t)i0 * K + k0;
        const __nv_bfloat16* b_ptr = B + (int64_t)j0 * K + k0;
        wmma::load_matrix_sync(a_frag, a_ptr, (unsigned)K);
        // col_major + ld=K：元素 (r,c) -> B[(j0+c)*K + (k0+r)] = B^T[k0+r, j0+c]
        wmma::load_matrix_sync(b_frag, b_ptr, (unsigned)K);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    __shared__ float Cs[WM][WN];
    wmma::store_matrix_sync(&Cs[0][0], c_frag, WN, wmma::mem_row_major);
    __syncwarp();

    const int lane = (int)threadIdx.x;
    for (int idx = lane; idx < WM * WN; idx += 32) {
        const int r = idx / WN;
        const int c = idx % WN;
        const int64_t ii = (int64_t)i0 + r;
        const int64_t jj = (int64_t)j0 + c;
        if (ii < M && jj < N) {
            C[ii * N + jj] = f2bf16(Cs[r][c]);
        }
    }
}

inline bool divisible_by_16(int64_t M, int64_t N, int64_t K) {
    return ((M | N | K) & 15) == 0;
}

}  // namespace

extern "C" void run_kernel(const __nv_bfloat16* A,
                           const __nv_bfloat16* B,
                           __nv_bfloat16* C,
                           int64_t M,
                           int64_t N,
                           int64_t K) {
    if (A == nullptr || B == nullptr || C == nullptr) return;
    if (M <= 0 || N <= 0 || K <= 0) return;

    // 评测尺寸均能被 16 整除；走 WMMA Tensor Core
    if (divisible_by_16(M, N, K)) {
        dim3 block(32);  // 1 warp
        dim3 grid((unsigned)((N + WN - 1) / WN),
                  (unsigned)((M + WM - 1) / WM));
        gemm_bt_wmma<<<grid, block>>>(A, B, C, M, N, K);
        return;
    }

    // 兜底：尺寸不规则时用标量核
    dim3 block(16, 16);
    dim3 grid((unsigned)((N + block.x - 1) / block.x),
              (unsigned)((M + block.y - 1) / block.y));
    gemm_bt_scalar<<<grid, block>>>(A, B, C, M, N, K);
}
