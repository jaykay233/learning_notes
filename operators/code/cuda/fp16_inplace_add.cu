// FP16 原地加法：A += B
// 评测入口：extern "C" void run_kernel(__half* A, const __half* B, int64_t numel);

#include <stdint.h>
#include <cuda_fp16.h>

// 每个线程按 grid-stride 处理多个元素；用 half2 一次加两个 fp16，提高带宽利用率。
__global__ void add_inplace_fp16_kernel(__half* __restrict__ A,
                                        const __half* __restrict__ B,
                                        int64_t numel) {
    const int64_t tid = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    const int64_t stride = (int64_t)blockDim.x * (int64_t)gridDim.x;

    // 成对处理：A[2i], A[2i+1] += B[2i], B[2i+1]
    const int64_t n2 = numel >> 1;  // numel / 2
    __half2* A2 = reinterpret_cast<__half2*>(A);
    const __half2* B2 = reinterpret_cast<const __half2*>(B);

    for (int64_t i = tid; i < n2; i += stride) {
        A2[i] = __hadd2(A2[i], B2[i]);
    }

    // numel 为奇数时，最后一个元素单独处理（本题测试用例均为偶数，但仍需覆盖）
    if ((numel & 1) && tid == 0) {
        const int64_t last = numel - 1;
        A[last] = __hadd(A[last], B[last]);
    }
}

extern "C" void run_kernel(__half* A, const __half* B, int64_t numel) {
    if (A == nullptr || B == nullptr || numel <= 0) {
        return;
    }

    // 256 线程/block 在多数 GPU 上占用率较好
    constexpr int kThreads = 256;

    // 按“成对元素”数量估 grid；再用 grid-stride，不必一次覆盖全部
    const int64_t n2 = (numel + 1) >> 1;
    int64_t blocks64 = (n2 + kThreads - 1) / kThreads;

    // 限制上限，避免过大 grid；剩余靠 kernel 内 stride 循环吃完
    // 现代 GPU gridDim.x 上限远大于此；65536 对本题 ~1.5e8 元素已足够
    constexpr int64_t kMaxBlocks = 65536;
    if (blocks64 > kMaxBlocks) {
        blocks64 = kMaxBlocks;
    }
    if (blocks64 < 1) {
        blocks64 = 1;
    }

    const dim3 block(kThreads);
    const dim3 grid(static_cast<unsigned int>(blocks64));

    add_inplace_fp16_kernel<<<grid, block>>>(A, B, numel);
}
