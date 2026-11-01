// Muon 优化器单步更新 —— 优化版（SYRK + 融合 epilogue）
//
// ============================ 状态：未对拍 ==================================
// 本文件尚未在 GPU 上编译/运行过。数学与索引经过逐行审计，但未经实测。
// 使用前务必跑：
//     nvcc -O3 -arch=sm_75 -Xptxas -v -shared -Xcompiler -fPIC muon_opt.cu -o libmuon.so
//     python muon_reference_check.py --lib ./libmuon.so
// 已验证的基线版本见同目录 muon.cu（已通过 PyTorch float64 参考对拍）。
// ==========================================================================
//
// 相比 muon.cu 的三处改动，收益全部来自「少算」，不来自指令级优化：
//
//   1. XXᵀ 与 A@A 走 SYRK：只对 ti <= tj 的 tile 发 block。
//      1024x1024 时 64 块 -> 36 块，轮数 ceil(64/40)=2 降到 ceil(36/40)=1，
//      这两个 GEMM 各自时间直接砍半。
//      注意：FLOP 只省 44% 而非 50%，因为对角 tile 整块算完整。
//
//   2. B = c*A² + b*A 融进 A@A 的 epilogue（FUSE_LIN），
//      省掉原本独立的 axpby kernel（读 8 MB + 写 4 MB）。
//
//   3. 上三角 -> 下三角用独立的 mirror_kernel（32x32 smem 转置）。
//      不能靠 epilogue 双写：转置位置的写跨行，单次写退化成 32 次事务，
//      2 MB * 32 = 64 MB/GEMM，比省下的还多。
//
// 每 iter 预期：165(A) + 165(A²) + 330(B@X) + 20(镜像) ≈ 660 µs
// 5 iter 总计约 3.4 ms，对比基线 5.3 ms，约 -36%。
//
// 顺序约束（易错点）：
//   syrk(上三角) -> mirror -> syrk(fused, 读 Am) -> mirror -> B@X
//   若把 fused SYRK 排在 mirror 之前，Lin[i][j] (i>j) 读到未初始化数据，
//   数值错但不会崩。

#include <cuda_runtime.h>
#include <math.h>
#include <stddef.h>

#ifndef MUON_V_MODE
#define MUON_V_MODE 0
#endif
static const int   kVMode = MUON_V_MODE;
static const float kEps   = 1e-7f;
static const float kCoefA =  3.4445f;
static const float kCoefB = -4.7750f;
static const float kCoefC =  2.0315f;
static const int   kSteps = 5;

namespace {

constexpr int TILE     = 128;
constexpr int THREADS  = 256;
constexpr int MICRO    = 8;
constexpr int BK       = 8;
constexpr int PAD      = 4;
constexpr int SSTRIDE  = TILE + PAD;
constexpr int LOAD_IT  = (TILE * BK) / THREADS;

float*  g_ws    = nullptr;
size_t  g_wsCap = 0;
double* g_normD = nullptr;

float* get_ws(size_t need) {
    if (g_wsCap < need) {
        if (g_ws) cudaFree(g_ws);
        cudaMalloc(reinterpret_cast<void**>(&g_ws), need * sizeof(float));
        g_wsCap = need;
    }
    return g_ws;
}
double* get_norm_dev() {
    if (!g_normD) cudaMalloc(reinterpret_cast<void**>(&g_normD), sizeof(double));
    return g_normD;
}

// ---------------------------------------------------------------- elementwise
__global__ void combine_kernel(const float* __restrict__ G, const float* __restrict__ Buf,
                               float* __restrict__ out, long long n,
                               float momentum, float gs) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const long long stride = (long long)gridDim.x * blockDim.x;
    for (; i < n; i += stride) out[i] = momentum * Buf[i] + gs * G[i];
}

__global__ void scale_kernel(float* __restrict__ x, long long n, float s) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const long long stride = (long long)gridDim.x * blockDim.x;
    for (; i < n; i += stride) x[i] *= s;
}

// double 累加：float 累加 1024^2 项会丢有效位，而该范数用于全局归一化。
__global__ void frob_sq_kernel(const float* __restrict__ x, long long n,
                               double* __restrict__ acc) {
    double s = 0.0;
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const long long stride = (long long)gridDim.x * blockDim.x;
    for (; i < n; i += stride) { const double v = (double)x[i]; s += v * v; }
    __shared__ double sh[256];
    sh[threadIdx.x] = s;
    __syncthreads();
    for (int off = blockDim.x >> 1; off > 0; off >>= 1) {
        if (threadIdx.x < off) sh[threadIdx.x] += sh[threadIdx.x + off];
        __syncthreads();
    }
    if (threadIdx.x == 0) atomicAdd(acc, sh[0]);
}

__global__ void copy_scale_kernel(const float* __restrict__ x, float* __restrict__ out,
                                  long long n, float s) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const long long stride = (long long)gridDim.x * blockDim.x;
    for (; i < n; i += stride) out[i] = s * x[i];
}

// -------------------------------------------------- 转置（G/Buf -> 工作矩阵）
// 强制一维 grid 自己解码 M/N 方向：二维 grid 下「kernel 把 blockIdx.x 当 M、
// grid 却按 N 开」会只写一部分行，现象是「头部全对、尾部成片为 0」，
// max abs diff 只从 4.85 掉到 4.00，极难定位。本项目在这一点上错过两次。
__global__ void transpose_combine_kernel(const float* __restrict__ G,
                                         const float* __restrict__ Buf,
                                         float* __restrict__ out,
                                         int M, int N, float momentum, float gs) {
    __shared__ float tile[32][33];
    const int nTilesN = (N + 31) >> 5;
    const int bi = (blockIdx.x / nTilesN) * 32;
    const int bj = (blockIdx.x % nTilesN) * 32;
    const int tx = threadIdx.x, ty = threadIdx.y;
    for (int a = ty; a < 32; a += blockDim.y) {
        const int m = bi + a, n = bj + tx;
        float v = 0.0f;
        if (m < M && n < N)
            v = momentum * Buf[(size_t)m * N + n] + gs * G[(size_t)m * N + n];
        tile[a][tx] = v;
    }
    __syncthreads();
    for (int a = ty; a < 32; a += blockDim.y) {
        const int n = bj + a, m = bi + tx;
        if (n < N && m < M) out[(size_t)n * M + m] = tile[tx][a];
    }
}

// 注意：本 kernel 阶段 1 的索引约定与上面那个不同（这里是 [N][M]），
// 但各自与自己的阶段 2 自洽。看起来不对称，推下来是对的。
__global__ void transpose_scale_kernel(const float* __restrict__ x,
                                       float* __restrict__ out,
                                       int M, int N, float s) {
    __shared__ float tile[32][33];
    const int nTilesN = (N + 31) >> 5;
    const int bi = (blockIdx.x / nTilesN) * 32;
    const int bj = (blockIdx.x % nTilesN) * 32;
    const int tx = threadIdx.x, ty = threadIdx.y;
    for (int a = ty; a < 32; a += blockDim.y) {
        const int n = bj + a, m = bi + tx;
        tile[a][tx] = (n < N && m < M) ? x[(size_t)n * M + m] : 0.0f;
    }
    __syncthreads();
    for (int a = ty; a < 32; a += blockDim.y) {
        const int m = bi + a, n = bj + tx;
        if (m < M && n < N) out[(size_t)m * N + n] = s * tile[tx][a];
    }
}

// ------------------------------------------------------------- 上三角 -> 下三角
// 32x32 smem 分块 transpose-copy，读写两侧都保持合并。
// grid = nt32 * nt32；下三角块的 block 立刻 return，空转开销可忽略。
__global__ void mirror_kernel(float* __restrict__ C, int R) {
    __shared__ float tile[32][33];
    const int nt32 = (R + 31) >> 5;
    const int bi = (blockIdx.x / nt32) * 32;    // 上三角的行
    const int bj = (blockIdx.x % nt32) * 32;    // 上三角的列
    if (bi > bj) return;

    const int tx = threadIdx.x, ty = threadIdx.y;

    // 读上三角块 C[bi+a][bj+tx]；a = ty 连续 -> 沿列合并
    for (int a = ty; a < 32; a += blockDim.y) {
        const int gi = bi + a, gj = bj + tx;
        tile[a][tx] = (gi < R && gj < R) ? C[(size_t)gi * R + gj] : 0.0f;
    }
    __syncthreads();

    // 写 C[bj+a][bi+tx] = tile[tx][a]，只写下三角（gi > gj）
    for (int a = ty; a < 32; a += blockDim.y) {
        const int gi = bj + a, gj = bi + tx;
        if (gi < R && gj < R && gi > gj) C[(size_t)gi * R + gj] = tile[tx][a];
    }
}

// -------------------------------------------------------------- 三角索引解码
// 上三角（含对角）的线性 blockIdx -> (ti, tj)。nt <= 8，循环开销可忽略。
__device__ __forceinline__ void tri_decode(int bid, int nt, int& ti, int& tj) {
    ti = 0;
    int off = 0;
    while (ti < nt - 1 && off + (nt - ti) <= bid) { off += nt - ti; ++ti; }
    tj = ti + (bid - off);
}

// --------------------------------------------------------- stage load/prefetch
template <bool TRANS_B>
__device__ __forceinline__ void stage_load(const float* __restrict__ A,
                                           const float* __restrict__ B,
                                           float* __restrict__ sA,
                                           float* __restrict__ sB,
                                           int i0, int j0, int k0,
                                           int M, int N, int K, int tid) {
    for (int it = 0; it < LOAD_IT; ++it) {
        const int idx = tid + it * THREADS;      // 0..1023
        {   // A: -> sA[k][i]，转置进 smem 让内层按 i 连续取
            const int r = idx >> 3, c = idx & 7;
            const int gi = i0 + r, gk = k0 + c;
            sA[c * SSTRIDE + r] = (gi < M && gk < K) ? A[(size_t)gi * K + gk] : 0.0f;
        }
        if (TRANS_B) {                            // B: N x K -> sB[k][j]
            const int r = idx >> 3, c = idx & 7;
            const int gj = j0 + r, gk = k0 + c;
            sB[c * SSTRIDE + r] = (gj < N && gk < K) ? B[(size_t)gj * K + gk] : 0.0f;
        } else {                                  // B: K x N -> sB[k][j]
            const int r = idx >> 7, c = idx & 127;
            const int gk = k0 + r, gj = j0 + c;
            sB[r * SSTRIDE + c] = (gk < K && gj < N) ? B[(size_t)gk * N + gj] : 0.0f;
        }
    }
}

// sm_75 没有 cp.async，只能先把 global 读数发到寄存器做软件流水。
template <bool TRANS_B>
__device__ __forceinline__ void stage_prefetch(const float* __restrict__ A,
                                               const float* __restrict__ B,
                                               int i0, int j0, int k0,
                                               int M, int N, int K, int tid,
                                               float* __restrict__ rx,
                                               float* __restrict__ ry) {
    for (int it = 0; it < LOAD_IT; ++it) {
        const int idx = tid + it * THREADS;
        {
            const int r = idx >> 3, c = idx & 7;
            const int gi = i0 + r, gk = k0 + c;
            rx[it] = (gi < M && gk < K) ? A[(size_t)gi * K + gk] : 0.0f;
        }
        if (TRANS_B) {
            const int r = idx >> 3, c = idx & 7;
            const int gj = j0 + r, gk = k0 + c;
            ry[it] = (gj < N && gk < K) ? B[(size_t)gj * K + gk] : 0.0f;
        } else {
            const int r = idx >> 7, c = idx & 127;
            const int gk = k0 + r, gj = j0 + c;
            ry[it] = (gk < K && gj < N) ? B[(size_t)gk * N + gj] : 0.0f;
        }
    }
}

template <bool TRANS_B>
__device__ __forceinline__ void stage_commit(float* __restrict__ sA,
                                             float* __restrict__ sB,
                                             int tid,
                                             const float* __restrict__ rx,
                                             const float* __restrict__ ry) {
    for (int it = 0; it < LOAD_IT; ++it) {
        const int idx = tid + it * THREADS;
        if (TRANS_B) {
            const int r = idx >> 3, c = idx & 7;
            sA[c * SSTRIDE + r] = rx[it];
            sB[c * SSTRIDE + r] = ry[it];
        } else {
            const int rA = idx >> 3, cA = idx & 7;
            const int rB = idx >> 7, cB = idx & 127;
            sA[cA * SSTRIDE + rA] = rx[it];
            sB[rB * SSTRIDE + cB] = ry[it];
        }
    }
}

// ------------------------------------------------------------ 归约主体（共用）
// SYRK 与完整 GEMM 的「沿 K 分 stage 双缓冲累加」完全相同，抽出来。
// 内层每 token 64 次 FMA 只配 4 条 LDS.128，计算:访存 = 4:1。
template <bool TRANS_B>
__device__ __forceinline__ void accumulate_tile(const float* __restrict__ A,
                                                const float* __restrict__ B,
                                                float (&acc)[MICRO][MICRO],
                                                int i0, int j0, int K,
                                                int M, int N, int tid,
                                                float (*sA)[BK][SSTRIDE],
                                                float (*sB)[BK][SSTRIDE]) {
    const int mi0 = (tid & 15) * MICRO;
    const int ni0 = (tid >> 4) * MICRO;
    const int nStage = (K + BK - 1) / BK;

    stage_load<TRANS_B>(A, B, &sA[0][0][0], &sB[0][0][0], i0, j0, 0, M, N, K, tid);
    __syncthreads();

    for (int st = 0; st < nStage; ++st) {
        const int  cur     = st & 1;
        const int  nxt     = cur ^ 1;
        const bool hasNext = (st + 1 < nStage);

        float rx[LOAD_IT], ry[LOAD_IT];
        if (hasNext)
            stage_prefetch<TRANS_B>(A, B, i0, j0, (st + 1) * BK, M, N, K, tid, rx, ry);

#pragma unroll
        for (int t = 0; t < BK; ++t) {
            const float* pa = &sA[cur][t][mi0];
            const float* pb = &sB[cur][t][ni0];
            const float4 a0 = *reinterpret_cast<const float4*>(pa);
            const float4 a1 = *reinterpret_cast<const float4*>(pa + 4);
            const float4 b0 = *reinterpret_cast<const float4*>(pb);
            const float4 b1 = *reinterpret_cast<const float4*>(pb + 4);
            const float av[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
            const float bv[8] = {b0.x, b0.y, b0.z, b0.w, b1.x, b1.y, b1.z, b1.w};
#pragma unroll
            for (int i = 0; i < MICRO; ++i)
#pragma unroll
                for (int j = 0; j < MICRO; ++j)
                    acc[i][j] = fmaf(av[i], bv[j], acc[i][j]);
        }

        // cur 在算、nxt 在写，两个 buffer 不冲突；nxt 上一轮读完后已由上一轮
        // 结尾的 sync 隔离。所以每 stage 只需一次 __syncthreads()。
        if (hasNext) {
            stage_commit<TRANS_B>(&sA[nxt][0][0], &sB[nxt][0][0], tid, rx, ry);
            __syncthreads();
        }
    }
    // epilogue 只写 global、不读 smem，因此不需要收尾 sync。
}

// ============================== SYRK（只算上三角） ==============================
// C[i][j] = sum_k A[i][k] * B[j][k]，对 blockIdx 解出的 ti <= tj 计算。
// FUSE_LIN = true：写成 coef_acc*acc + coef_lin*Lin[i][j]，
//   把 B = c*A² + b*A 融进 A@A 的 epilogue，省掉独立的 axpby kernel。
template <bool FUSE_LIN>
__global__ void __launch_bounds__(THREADS, 2)
syrk_kernel(const float* __restrict__ A, const float* __restrict__ B,
            const float* __restrict__ Lin, float* __restrict__ C,
            int R, int K, int nt, float coef_acc, float coef_lin) {
    __shared__ float sA[2][BK][SSTRIDE];
    __shared__ float sB[2][BK][SSTRIDE];

    int ti, tj;
    tri_decode(blockIdx.x, nt, ti, tj);
    const int i0 = ti * TILE;
    const int j0 = tj * TILE;

    float acc[MICRO][MICRO];
#pragma unroll
    for (int i = 0; i < MICRO; ++i)
#pragma unroll
        for (int j = 0; j < MICRO; ++j) acc[i][j] = 0.0f;

    accumulate_tile<true>(A, B, acc, i0, j0, K, R, R, threadIdx.x, sA, sB);

    const int tid = threadIdx.x;
    const int gi0 = i0 + (tid & 15) * MICRO;
    const int gj0 = j0 + (tid >> 4) * MICRO;
    const bool v4  = ((R & 3) == 0) && (gj0 + MICRO <= R);

#pragma unroll
    for (int i = 0; i < MICRO; ++i) {
        const int gi = gi0 + i;
        if (gi >= R) break;
        float* dst = C + (size_t)gi * R + gj0;
        if (v4) {
            if (FUSE_LIN) {
                const float* lp = Lin + (size_t)gi * R + gj0;
                const float4 l0 = *reinterpret_cast<const float4*>(lp);
                const float4 l1 = *reinterpret_cast<const float4*>(lp + 4);
                *reinterpret_cast<float4*>(dst) =
                    make_float4(coef_acc * acc[i][0] + coef_lin * l0.x,
                                coef_acc * acc[i][1] + coef_lin * l0.y,
                                coef_acc * acc[i][2] + coef_lin * l0.z,
                                coef_acc * acc[i][3] + coef_lin * l0.w);
                *reinterpret_cast<float4*>(dst + 4) =
                    make_float4(coef_acc * acc[i][4] + coef_lin * l1.x,
                                coef_acc * acc[i][5] + coef_lin * l1.y,
                                coef_acc * acc[i][6] + coef_lin * l1.z,
                                coef_acc * acc[i][7] + coef_lin * l1.w);
            } else {
                *reinterpret_cast<float4*>(dst) =
                    make_float4(coef_acc * acc[i][0], coef_acc * acc[i][1],
                                coef_acc * acc[i][2], coef_acc * acc[i][3]);
                *reinterpret_cast<float4*>(dst + 4) =
                    make_float4(coef_acc * acc[i][4], coef_acc * acc[i][5],
                                coef_acc * acc[i][6], coef_acc * acc[i][7]);
            }
        } else {
#pragma unroll
            for (int j = 0; j < MICRO; ++j) {
                if (gj0 + j < R) {
                    float v = coef_acc * acc[i][j];
                    if (FUSE_LIN) v += coef_lin * Lin[(size_t)gi * R + gj0 + j];
                    dst[j] = v;
                }
            }
        }
    }
}

// ========================== 完整 GEMM（X2 = a*X + B@X） ==========================
template <bool TRANS_B, bool FUSE_XOLD>
__global__ void __launch_bounds__(THREADS, 2)
sgemm_kernel(const float* __restrict__ A, const float* __restrict__ B,
             const float* __restrict__ Xold, float* __restrict__ C,
             int M, int N, int K, float alpha) {
    __shared__ float sA[2][BK][SSTRIDE];
    __shared__ float sB[2][BK][SSTRIDE];

    const int i0  = blockIdx.y * TILE;
    const int j0  = blockIdx.x * TILE;
    const int tid = threadIdx.x;

    float acc[MICRO][MICRO];
#pragma unroll
    for (int i = 0; i < MICRO; ++i)
#pragma unroll
        for (int j = 0; j < MICRO; ++j) acc[i][j] = 0.0f;

    accumulate_tile<TRANS_B>(A, B, acc, i0, j0, K, M, N, tid, sA, sB);

    const int  gi0 = i0 + (tid & 15) * MICRO;
    const int  gj0 = j0 + (tid >> 4) * MICRO;
    const bool v4  = ((N & 3) == 0) && (gj0 + MICRO <= N);

#pragma unroll
    for (int i = 0; i < MICRO; ++i) {
        const int gi = gi0 + i;
        if (gi >= M) break;
        float* dst = C + (size_t)gi * N + gj0;
        if (v4) {
            if (FUSE_XOLD) {
                // X2 = a*Xold + acc：省一次 4 MB 读 + 4 MB 写，5 iter 共 40 MB
                const float* src = Xold + (size_t)gi * N + gj0;
                const float4 o0 = *reinterpret_cast<const float4*>(src);
                const float4 o1 = *reinterpret_cast<const float4*>(src + 4);
                *reinterpret_cast<float4*>(dst) =
                    make_float4(alpha * o0.x + acc[i][0], alpha * o0.y + acc[i][1],
                                alpha * o0.z + acc[i][2], alpha * o0.w + acc[i][3]);
                *reinterpret_cast<float4*>(dst + 4) =
                    make_float4(alpha * o1.x + acc[i][4], alpha * o1.y + acc[i][5],
                                alpha * o1.z + acc[i][6], alpha * o1.w + acc[i][7]);
            } else {
                *reinterpret_cast<float4*>(dst) =
                    make_float4(acc[i][0], acc[i][1], acc[i][2], acc[i][3]);
                *reinterpret_cast<float4*>(dst + 4) =
                    make_float4(acc[i][4], acc[i][5], acc[i][6], acc[i][7]);
            }
        } else {
#pragma unroll
            for (int j = 0; j < MICRO; ++j) {
                if (gj0 + j < N) {
                    float v = acc[i][j];
                    if (FUSE_XOLD) v += alpha * Xold[(size_t)gi * N + gj0 + j];
                    dst[j] = v;
                }
            }
        }
    }
}

inline dim3 gemm_grid(int m, int n) {
    return dim3((unsigned)((n + TILE - 1) / TILE), (unsigned)((m + TILE - 1) / TILE));
}

}  // namespace

extern "C" void solve(const float* G, const float* Buf, float* output,
                      int M, int N, float momentum) {
    if (!G || !Buf || !output || M <= 0 || N <= 0) return;

    // R = min(M,N)，C = max(M,N)。M > N 时转置，让 Gram 落在较小维度
    // （精确等价：(V Vᵀ)^k V ≡ ((Vᵀ V)^k Vᵀ)ᵀ，只省 FLOP）。
    const int  R  = (M < N) ? M : N;
    const int  C  = (M < N) ? N : M;
    const bool tr = (M > N);

    float* ws = get_ws((size_t)2 * R * C + (size_t)2 * R * R);
    float* X  = ws;                             // R x C
    float* X2 = X  + (size_t)R * C;             // R x C
    float* Am = X2 + (size_t)R * C;             // R x R
    float* Bm = Am + (size_t)R * R;             // R x R

    const float gs = (kVMode == 0) ? 1.0f : (1.0f - momentum);

    // ---- (1) V = momentum*Buf + gs*G ----
    if (tr) {
        const dim3 blk(32, 8);
        const unsigned nb = (unsigned)(((M + 31) / 32) * ((N + 31) / 32));
        transpose_combine_kernel<<<nb, blk>>>(G, Buf, X, M, N, momentum, gs);
    } else {
        const long long n = (long long)M * N;
        long long b = (n + 255) / 256;
        if (b > 4096) b = 4096;
        combine_kernel<<<(unsigned)b, 256>>>(G, Buf, X, n, momentum, gs);
    }

    // ---- (2) X /= (||X||_F + eps) ----
    {
        const long long n = (long long)R * C;
        double* dn = get_norm_dev();
        cudaMemset(dn, 0, sizeof(double));
        frob_sq_kernel<<<512, 256>>>(X, n, dn);
        double h = 0.0;
        cudaMemcpy(&h, dn, sizeof(double), cudaMemcpyDeviceToHost);
        const float inv = 1.0f / ((float)sqrt(h) + kEps);   // 全零时 X 仍为 0
        scale_kernel<<<512, 256>>>(X, n, inv);
    }

    // ---- (3) 5 步 Newton-Schulz ----
    const int nt      = (R + TILE - 1) / TILE;
    const int ntri    = nt * (nt + 1) / 2;              // 上三角 tile 数
    const int nt32    = (R + 31) / 32;
    const unsigned nm = (unsigned)(nt32 * nt32);
    const dim3 mblk(32, 8);

    for (int s = 0; s < kSteps; ++s) {
        // A = X Xᵀ（只算上三角）-> 镜像成完整对称矩阵
        syrk_kernel<false><<<ntri, THREADS>>>(X, X, nullptr, Am, R, C, nt, 1.0f, 0.0f);
        mirror_kernel<<<nm, mblk>>>(Am, R);

        // B = c*A² + b*A（只算上三角，axpby 融进 epilogue）-> 镜像
        // Lin 必须指向已经镜像完成的 Am；顺序颠倒会读到未初始化的下三角。
        syrk_kernel<true><<<ntri, THREADS>>>(Am, Am, Am, Bm, R, R, nt, kCoefC, kCoefB);
        mirror_kernel<<<nm, mblk>>>(Bm, R);

        // X2 = a*X + B@X（输出 R x C，完整 GEMM，axpy 融进 epilogue）
        sgemm_kernel<false, true><<<gemm_grid(R, C), THREADS>>>(
            Bm, X, X, X2, R, C, R, kCoefA);

        float* t = X; X = X2; X2 = t;
    }

    // ---- (4) output = sqrt(max(M,N)) * X ----
    // 题面「缩放后 RMS = 1」是启发式；实测 1024x1024 为 0.910，
    // 因为 5 步后 sigma_max 落在 1.20 而非 1。不要为此改缩放因子。
    const float sc = sqrtf((float)((M > N) ? M : N));
    if (tr) {
        const dim3 blk(32, 8);
        const unsigned nb = (unsigned)(((M + 31) / 32) * ((N + 31) / 32));
        transpose_scale_kernel<<<nb, blk>>>(X, output, M, N, sc);
    } else {
        const long long n = (long long)M * N;
        long long b = (n + 255) / 256;
        if (b > 4096) b = 4096;
        copy_scale_kernel<<<(unsigned)b, 256>>>(X, output, n, sc);
    }

    cudaDeviceSynchronize();
}
