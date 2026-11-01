#include <cuda_runtime.h>

// ============================================================================
// Ordinary Least Squares:  beta = inv(X^T X) * X^T y
//
// Single kernel, single block, all data in shared memory, one launch.
//   p = n_features <= 32, so p*p <= 1024 = one block of 1024 threads.
//   For p = n = 32 the whole computation is ~4.5e4 FMA (~0.4us on one SM),
//   while a kernel launch costs 3-5us. Three kernels would spend 90% of the
//   wall clock on launches, so everything is fused into one.
//
// Solver is Gauss-Jordan, not Cholesky + triangular solves. The reason is
// serial depth, not FLOP. At p=32 both are ~0.1us of arithmetic, but
// Cholesky's two triangular solves need 2p back-to-back reductions of
// length <= p (a 5-level __shfl chain each, ~100-150 cycles), i.e. 4-6us of
// pure latency with 63 of 64 warps idle. Gauss-Jordan is p steps, each step
// p*p-way parallel, zero reductions.
//
// This version does NOT normalise the pivot row. Instead the multiplier is
// scaled: f = A[i][c] / A[c][c]. Then A ends up diagonal, and beta[i] is
// b[i] / A[i][i] at the end. Why: during step c no thread writes row c at
// all, so every thread can read A[c][c] with no read/write hazard and no
// warp-level sync. It also costs p __syncthreads() instead of 2p.
//
// Accumulation in the Gram matrix is float inside a segment of 256 terms and
// double across segments. Plain float gives a relative error of about
// eps*sqrt(n) = 3.8e-5 at n=1e5; the normal equations square the condition
// number, so that becomes kappa^2 * 3.8e-5, which busts the 1e-2 tolerance.
// Segmenting replaces n by 256: eps*sqrt(256) = 1.9e-6. Full double is not an
// option: T4 FP64 runs at 1/32 of FP32, and p*p*n = 1e8 DFMA on one SM is
// 31ms while the loads only need 1.6ms.
//
// Range: p > 32 is not supported. solve() returns without touching beta.
//
// Measured accuracy (per-index emulation, code/cuda/ols_emul.py). Metric is
// max|dbeta| / (1e-2 + 1e-2*|beta|), must be < 1, so each value may grow by
// 1/value before it fails:
//     n x p = 32 x 32    kappa=133   err 4.7e-3
//     n x p = 8  x 8     kappa=105   err 9.5e-3
//     n x p = 1000 x 32  kappa=1.4   err 2.9e-6
// Near-collinear sweep (last column := X[:,0]*(1+eps) + noise):
//     kappa=118  err 5.0e-2  pass
//     kappa=394  err 8.1e-3  pass
//     kappa=380  err 0.55    borderline
//     kappa=2881 err 1.3e+2  fail
// Safe region is kappa(X) < ~400; random 32x32 uniform matrices come out at
// kappa = 30..130. This matches err ~ kappa^2 * eps, so it predicts pass/fail
// for a given test case without running it.
// ============================================================================

#define OLS_MAXP    32
#define OLS_THREADS 1024
#define OLS_CHUNK   256

__global__ void ols_solve_kernel(const float* X,
                                 const float* y,
                                 float* beta,
                                 int n, int p)
{
    __shared__ float A[OLS_MAXP * OLS_MAXP];
    __shared__ float b[OLS_MAXP];

    const int tid = threadIdx.x;
    const int row = tid >> 5;          // 0..31 : row of A, also warp id
    const int col = tid & 31;          // 0..31 : column of A, also lane id

    // ------------------------------------------------------------------
    // Stage 1: A = X^T X and b = X^T y
    // Thread (row, col) computes A[row][col]. Warp 0 additionally computes
    // b[col] using the xj = X[:, col] it is already loading.
    // Within a warp col varies, so X[k][col] is 32 consecutive floats (one
    // 128B line, fully coalesced) and X[k][row] is one broadcast address.
    // ------------------------------------------------------------------
    {
        double accA = 0.0;
        double accB = 0.0;
        const int ownsB = (row == 0);

        if (row < p && col < p) {
            for (int k0 = 0; k0 < n; k0 += OLS_CHUNK) {
                const int k1 = (k0 + OLS_CHUNK < n) ? (k0 + OLS_CHUNK) : n;
                float partA = 0.0f;
                float partB = 0.0f;
                for (int k = k0; k < k1; ++k) {
                    const float xj = X[k * p + col];
                    partA += X[k * p + row] * xj;
                    if (ownsB) partB += xj * y[k];
                }
                accA += (double)partA;
                if (ownsB) accB += (double)partB;
            }
            A[row * OLS_MAXP + col] = (float)accA;
            if (ownsB) b[col] = (float)accB;
        }
    }
    __syncthreads();

    // ------------------------------------------------------------------
    // Stage 2: Gauss-Jordan elimination of [A | b], without normalising
    // the pivot row. Step c: invariant is that columns < c are already
    // cleared, so only j > c needs updating.
    //   reads  : A[c][c] (every thread, broadcast), A[i][c], A[c][j], b[c]
    //   writes : A[i][j] and b[i] for i != c only
    // Row c is written by nobody during step c, so there is no read/write
    // hazard inside a warp. One __syncthreads() per step is enough: it
    // publishes this step's writes before step c+1 reads them.
    // A[i][c] is deliberately left uninitialised-to-zero: column c is never
    // read again, and clearing it from the (i, c) thread would race with the
    // other threads of row i that are reading it as the multiplier.
    // ------------------------------------------------------------------
    for (int c = 0; c < p; ++c) {
        const float inv = 1.0f / A[c * OLS_MAXP + c];

        if (row != c && row < p) {
            const float f = A[row * OLS_MAXP + c] * inv;
            if (col > c && col < p)
                A[row * OLS_MAXP + col] -= f * A[c * OLS_MAXP + col];
            if (col == 0)
                b[row] -= f * b[c];
        }
        __syncthreads();
    }

    // ------------------------------------------------------------------
    // Stage 3: A is diagonal now, so beta[i] = b[i] / A[i][i].
    // ------------------------------------------------------------------
    if (row == 0 && col < p) beta[col] = b[col] / A[col * OLS_MAXP + col];
}

// X, y, beta are device pointers
extern "C" void solve(const float* X, const float* y, float* beta,
                      int n_samples, int n_features)
{
    if (X == 0 || y == 0 || beta == 0) return;
    if (n_samples <= 0 || n_features <= 0) return;
    if (n_features > OLS_MAXP) return;      // out of range for this version
    ols_solve_kernel<<<1, OLS_THREADS>>>(X, y, beta, n_samples, n_features);
}
