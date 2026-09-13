/**
 * Host: CPU twin uses ICPU_RUN_KF; NPU uses aclrtLaunch path (stubbed by macros).
 */
#include <cstdint>
#include <cstdio>

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
  ICPU_RUN_KF(add_custom, kBlockDim, x, y, z);

  WriteFile("./output/output_z.bin", z, kBytes);
  AscendC::GmFree(static_cast<void *>(x));
  AscendC::GmFree(static_cast<void *>(y));
  AscendC::GmFree(static_cast<void *>(z));
  std::printf("CPU twin run done. output -> ./output/output_z.bin\n");
  return 0;
#else
  std::printf("NPU path: wire aclInit / aclrtMalloc / aclrtlaunch_add_custom per your CANN sample.\n");
  std::printf("For CPU twin, configure -DCMAKE_ASC_RUN_MODE=cpu and rebuild.\n");
  return 0;
#endif
}
