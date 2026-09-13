/**
 * Host: CPU twin uses ICPU_RUN_KF; NPU uses aclrtLaunch path (stubbed by macros).
 */
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include "data_utils.h"

#if defined(ASCENDC_CPU_DEBUG) || defined(__CCE_KT_TEST__)
#include "tikicpulib.h"
extern "C" __global__ __aicore__ void add_custom(GM_ADDR x, GM_ADDR y, GM_ADDR z);
#else
#include "acl/acl.h"
// Generated / toolkit launch header name varies by CANN version; keep declaration.
extern "C" __global__ __aicore__ void add_custom(GM_ADDR x, GM_ADDR y, GM_ADDR z);
#endif

namespace {
constexpr uint32_t kBlockDim = 8;
constexpr size_t kElems = 8 * 2048;
constexpr size_t kBytes = kElems * sizeof(uint16_t); // half

int EnvInt(const char *name, int fallback) {
  const char *v = std::getenv(name);
  if (v == nullptr || v[0] == '\0') {
    return fallback;
  }
  return std::atoi(v);
}
} // namespace

int32_t main(int32_t argc, char *argv[]) {
  (void)argc;
  (void)argv;

#if defined(ASCENDC_CPU_DEBUG) || defined(__CCE_KT_TEST__)
  uint8_t *x = static_cast<uint8_t *>(AscendC::GmAlloc(kBytes));
  uint8_t *y = static_cast<uint8_t *>(AscendC::GmAlloc(kBytes));
  uint8_t *z = static_cast<uint8_t *>(AscendC::GmAlloc(kBytes));

  if (!ReadFile("./input/input_x.bin", kBytes, x, kBytes) ||
      !ReadFile("./input/input_y.bin", kBytes, y, kBytes)) {
    return 1;
  }

  AscendC::SetKernelMode(KernelMode::AIV_MODE);

  const int warmup = EnvInt("ASCENDC_BENCH_WARMUP", 1);
  const int iters = EnvInt("ASCENDC_BENCH_ITERS", 5);
  for (int i = 0; i < warmup; ++i) {
    ICPU_RUN_KF(add_custom, kBlockDim, x, y, z);
  }

  using clock = std::chrono::steady_clock;
  const auto t0 = clock::now();
  for (int i = 0; i < iters; ++i) {
    ICPU_RUN_KF(add_custom, kBlockDim, x, y, z);
  }
  const auto t1 = clock::now();
  const double ms_total =
      std::chrono::duration<double, std::milli>(t1 - t0).count();
  const double ms_avg = ms_total / static_cast<double>(iters);
  const double elems_per_s =
      (static_cast<double>(kElems) * 1e3) / ms_avg; // elems / second

  WriteFile("./output/output_z.bin", z, kBytes);
  AscendC::GmFree(static_cast<void *>(x));
  AscendC::GmFree(static_cast<void *>(y));
  AscendC::GmFree(static_cast<void *>(z));
  std::printf("CPU twin run done. output -> ./output/output_z.bin\n");
  std::printf(
      "timing: kernel_wall_ms_avg=%.3f  total_ms=%.3f  iters=%d  warmup=%d  "
      "elems=%zu  elems_per_s=%.0f\n",
      ms_avg, ms_total, iters, warmup, kElems, elems_per_s);
  std::printf(
      "note: wall time is CPU-twin host timing (incl. twin runtime), not NPU.\n");
  return 0;
#else
  std::printf("NPU path: wire aclInit / aclrtMalloc / aclrtlaunch_add_custom per your CANN sample.\n");
  std::printf("For CPU twin, configure -DRUN_MODE=cpu and rebuild.\n");
  return 0;
#endif
}
