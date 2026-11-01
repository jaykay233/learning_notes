// MoE 专家权重梯度：dW[g] = X[s:e]^T * dY[s:e]
//
//   dW[g][k][n] = sum_{t=s}^{e-1} X[t][k] * dY[t][n]
//   s = group_offsets[g], e = group_offsets[g+1]
//
// 这是 ragged contraction：
//   - 独立维 k、n  → 并行（blockIdx.y / blockIdx.x）
//   - 分组维 g     → 并行，最外层（blockIdx.z），长度参差、可能为 0
//   - 归约维 t     → 线程内串行，不做 split-K
//
// 性能配置：G=64, M_total=32768, K=1024, N=512，两个空专家。
// 在 T4（sm_75, 8.1 TFLOPS, 320 GB/s, 40 SM, 4 MB L2）上：
//   计算下限 = 2*32768*1024*512 / 8.1e12 = 4.24 ms
//   访存下限 = (134 + 67 + 134) MB / 320 GB/s = 1.05 ms
//   算术强度 = 34.4 GFLOP / 335 MB = 103 FLOP/B > 25 FLOP/B  → 计算受限
//
// 编译示例：
//   nvcc -O3 -arch=sm_75 -Xptxas -v moe_expert_dw.cu -c
// 关注输出里的 "0 bytes spill stores"：spill 说明 128 reg/thread 不够。

#include <cuda_runtime.h>
#include <stddef.h>

namespace {

constexpr int MAX_G     = 128;
constexpr int TILE      = 128;                       // TILE_K = TILE_N = 128
constexpr int THREADS   = 256;
constexpr int MICRO_K   = 8;                         // 每线程沿 K 的累加器数
constexpr int MICRO_N   = 8;                         // 每线程沿 N 的累加器数
constexpr int CHUNK_T   = 8;                         // 每个 stage 搬入的 token 数
constexpr int PAD       = 4;                         // 行跨度补齐，错开 bank
constexpr int SSTRIDE   = TILE + PAD;                // 132 floats = 528 B
constexpr int LOAD_ITERS = (CHUNK_T * TILE) / THREADS;   // = 4

__device__ int g_order[MAX_G];

// ---------------------------------------------------------------- 基线版
// 一个线程算一个 dW[g][k][n]，自己把整段 token 串着累加。
// 用于说明「为什么必须分块」：X[t][k] 被 N=512 个线程各读一遍，
// dY[t][n] 被 K=1024 个线程各读一遍。
//   访存 ≈ 134 MB*512 + 67 MB*1024 ≈ 137 GB → 137 GB / 320 GB/s ≈ 430 ms
// 而分块版的计算下限只有 4.24 ms，差 100 倍。
__global__ void moe_dw_naive(const float* __restrict__ X,
                             const float* __restrict__ dY,
                             const int* __restrict__ group_offsets,
                             float* __restrict__ dW,
                             int K, int N) {
    const int g = blockIdx.z;
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    const int k = blockIdx.y * blockDim.y + threadIdx.y;
    if (k >= K || n >= N) return;

    const int s = group_offsets[g];
    const int e = group_offsets[g + 1];

    float acc = 0.0f;
    for (int t = s; t < e; ++t) {
        acc += X[(size_t)t * K + k] * dY[(size_t)t * N + n];
    }
    dW[(size_t)g * K * N + (size_t)k * N + n] = acc;   // 空专家也写 0
}

// ------------------------------------------------ 按段长降序排专家（治倾斜）
// blockIdx.z 是最慢的维度，序号小的先发射；长段先跑，短段填尾。
__global__ void build_expert_order(const int* __restrict__ group_offsets, int G) {
    if (threadIdx.x != 0) return;

    int len[MAX_G];
    for (int i = 0; i < G; ++i) {
        len[i]    = group_offsets[i + 1] - group_offsets[i];
        g_order[i] = i;
    }
    for (int i = 0; i < G; ++i) {          // 选择排序，O(G^2)，G<=128，一次性开销
        int bi = i;
        for (int j = i + 1; j < G; ++j) {
            if (len[j] > len[bi]) bi = j;
        }
        const int tl = len[i];     len[i] = len[bi];     len[bi] = tl;
        const int tg = g_order[i]; g_order[i] = g_order[bi]; g_order[bi] = tg;
    }
}

// ---------------------------------------------------------------- 分块版
// grid = (ceil(N/128), ceil(K/128), G)，block = 256 线程。
// 每 block 算 128x128 的 dW 子块，沿 token 维分 stage 归约。
__global__ void __launch_bounds__(THREADS, 2)
moe_dw_tiled(const float* __restrict__ X,
             const float* __restrict__ dY,
             const int* __restrict__ group_offsets,
             float* __restrict__ dW,
             int K, int N, bool useOrder) {
    // 双缓冲：2 * 8 * 132 * 2(X,Y) * 4B = 16896 B，两 block/SM 共 33 KB < 64 KB
    __shared__ float sX[2][CHUNK_T][SSTRIDE];
    __shared__ float sY[2][CHUNK_T][SSTRIDE];

    const int gid   = useOrder ? g_order[blockIdx.z] : blockIdx.z;
    const int kTile = blockIdx.y * TILE;
    const int nTile = blockIdx.x * TILE;
    const int s     = group_offsets[gid];
    const int len   = group_offsets[gid + 1] - s;      // 空专家 => 0

    const int tid = threadIdx.x;
    const int kk0 = (tid % 16) * MICRO_K;              // 0,8,...,120，4 对齐
    const int nn0 = (tid / 16) * MICRO_N;              // 0,8,...,120

    // 64 个累加器，全部留在寄存器里。这是复用倍数的上限。
    float acc[MICRO_K][MICRO_N];
#pragma unroll
    for (int i = 0; i < MICRO_K; ++i)
#pragma unroll
        for (int j = 0; j < MICRO_N; ++j) acc[i][j] = 0.0f;

    const int nStages = (len + CHUNK_T - 1) / CHUNK_T;  // 空专家 => 0 次

    // prologue：搬 chunk 0。越界（token 超出本段 / 列超出 K,N）一律填 0，
    // 这样内层循环永远是满的 CHUNK_T 次，不需要在 FMA 里带条件。
    if (nStages > 0) {
#pragma unroll
        for (int it = 0; it < LOAD_ITERS; ++it) {
            const int idx = tid + it * THREADS;   // warp 内 idx 连续 -> 合并访存
            const int r   = idx >> 7;             // token 行 0..7
            const int c   = idx & (TILE - 1);     // 列 0..127
            float vx = 0.0f, vy = 0.0f;
            if (r < len) {
                if (kTile + c < K) vx = X [(size_t)(s + r) * K + kTile + c];
                if (nTile + c < N) vy = dY[(size_t)(s + r) * N + nTile + c];
            }
            sX[0][r][c] = vx;
            sY[0][r][c] = vy;
        }
        __syncthreads();
    }

    for (int st = 0; st < nStages; ++st) {
        const int  cur     = st & 1;
        const int  nxt     = cur ^ 1;
        const bool hasNext = (st + 1 < nStages);
        const int  tBase   = (st + 1) * CHUNK_T;

        // (1) 先把下一 chunk 的 global 读数发到寄存器（sm_75 没有 cp.async）
        float rx[LOAD_ITERS], ry[LOAD_ITERS];
        if (hasNext) {
#pragma unroll
            for (int it = 0; it < LOAD_ITERS; ++it) {
                const int idx = tid + it * THREADS;
                const int r   = idx >> 7;
                const int c   = idx & (TILE - 1);
                const int t   = tBase + r;
                rx[it] = 0.0f;
                ry[it] = 0.0f;
                if (t < len) {
                    if (kTile + c < K) rx[it] = X [(size_t)(s + t) * K + kTile + c];
                    if (nTile + c < N) ry[it] = dY[(size_t)(s + t) * N + nTile + c];
                }
            }
        }

        // (2) 算当前 chunk：每个 token 64 次 FMA 只配 4 条 LDS.128，比例 4:1
#pragma unroll
        for (int t = 0; t < CHUNK_T; ++t) {
            const float* pa = &sX[cur][t][kk0];
            const float* pb = &sY[cur][t][nn0];
            const float4 a0 = *reinterpret_cast<const float4*>(pa);
            const float4 a1 = *reinterpret_cast<const float4*>(pa + 4);
            const float4 b0 = *reinterpret_cast<const float4*>(pb);
            const float4 b1 = *reinterpret_cast<const float4*>(pb + 4);
            const float a[MICRO_K] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
            const float b[MICRO_N] = {b0.x, b0.y, b0.z, b0.w, b1.x, b1.y, b1.z, b1.w};
#pragma unroll
            for (int i = 0; i < MICRO_K; ++i)
#pragma unroll
                for (int j = 0; j < MICRO_N; ++j)
                    acc[i][j] = fmaf(a[i], b[j], acc[i][j]);
        }

        // (3) 写入另一块 buffer。算 cur、写 nxt 是两个不同 buffer；
        //     nxt 上一轮被读完之后，上一轮结尾的 sync 已保证安全。
        //     所以每 stage 只需要这一次 __syncthreads()。
        if (hasNext) {
#pragma unroll
            for (int it = 0; it < LOAD_ITERS; ++it) {
                const int idx = tid + it * THREADS;
                const int r   = idx >> 7;
                const int c   = idx & (TILE - 1);
                sX[nxt][r][c] = rx[it];
                sY[nxt][r][c] = ry[it];
            }
            __syncthreads();
        }
    }

    // 写回：空专家 nStages==0，acc 保持 0，这里自然写出全零
    // （dW 未初始化，每个元素都必须写，空专家不能跳过）
    const size_t gBase  = (size_t)gid * K * N;
    const int    nnBase = nTile + nn0;
#pragma unroll
    for (int i = 0; i < MICRO_K; ++i) {
        const int kk = kTile + kk0 + i;
        if (kk >= K) break;
        float* dst = dW + gBase + (size_t)kk * N + nnBase;
        if (((N & 3) == 0) && (nnBase + MICRO_N <= N)) {
            *reinterpret_cast<float4*>(dst)     =
                make_float4(acc[i][0], acc[i][1], acc[i][2], acc[i][3]);
            *reinterpret_cast<float4*>(dst + 4) =
                make_float4(acc[i][4], acc[i][5], acc[i][6], acc[i][7]);
        } else {
#pragma unroll
            for (int j = 0; j < MICRO_N; ++j) {
                if (nnBase + j < N) dst[j] = acc[i][j];
            }
        }
    }
}

}  // namespace

// X、dY、group_offsets、dW 是 device 指针
extern "C" void solve(const float* X, const float* dY, const int* group_offsets, float* dW,
                      int G, int M_total, int K, int N) {
    (void)M_total;

    const bool useOrder = (G <= MAX_G);
    if (useOrder) build_expert_order<<<1, 32>>>(group_offsets, G);

    const dim3 grid((N + TILE - 1) / TILE,
                    (K + TILE - 1) / TILE,
                    G);
    moe_dw_tiled<<<grid, THREADS>>>(X, dY, group_offsets, dW, K, N, useOrder);

    cudaDeviceSynchronize();
}

// 基线入口，用于实测对比（同一份输入下应显著慢于 solve）
extern "C" void solve_naive(const float* X, const float* dY, const int* group_offsets,
                            float* dW, int G, int M_total, int K, int N) {
    (void)M_total;
    const dim3 block(16, 16);
    const dim3 grid((N + 15) / 16, (K + 15) / 16, G);
    moe_dw_naive<<<grid, block>>>(X, dY, group_offsets, dW, K, N);
    cudaDeviceSynchronize();
}
