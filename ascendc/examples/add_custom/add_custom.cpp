/**
 * AscendC vector add kernel (half).
 * CPU twin: macros like __aicore__ become empty; do not overload solely on them.
 */
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2;
constexpr int32_t TILE_LENGTH = 256;

class KernelAdd {
public:
  __aicore__ inline KernelAdd() {}

  __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, uint32_t totalLength) {
    this->totalLength = totalLength;
    xGm.SetGlobalBuffer((__gm__ half *)x, totalLength);
    yGm.SetGlobalBuffer((__gm__ half *)y, totalLength);
    zGm.SetGlobalBuffer((__gm__ half *)z, totalLength);
    pipe.InitBuffer(inQueueX, BUFFER_NUM, TILE_LENGTH * sizeof(half));
    pipe.InitBuffer(inQueueY, BUFFER_NUM, TILE_LENGTH * sizeof(half));
    pipe.InitBuffer(outQueueZ, BUFFER_NUM, TILE_LENGTH * sizeof(half));
  }

  __aicore__ inline void Process() {
    int32_t loopCount = (totalLength + TILE_LENGTH - 1) / TILE_LENGTH;
    for (int32_t i = 0; i < loopCount; ++i) {
      uint32_t len = TILE_LENGTH;
      if ((i + 1) * TILE_LENGTH > totalLength) {
        len = totalLength - i * TILE_LENGTH;
      }
      CopyIn(i, len);
      Compute(len);
      CopyOut(i, len);
    }
  }

private:
  __aicore__ inline void CopyIn(int32_t progress, uint32_t len) {
    AscendC::LocalTensor<half> xLocal = inQueueX.AllocTensor<half>();
    AscendC::LocalTensor<half> yLocal = inQueueY.AllocTensor<half>();
    AscendC::DataCopy(xLocal, xGm[progress * TILE_LENGTH], len);
    AscendC::DataCopy(yLocal, yGm[progress * TILE_LENGTH], len);
    inQueueX.EnQue(xLocal);
    inQueueY.EnQue(yLocal);
  }

  __aicore__ inline void Compute(uint32_t len) {
    AscendC::LocalTensor<half> xLocal = inQueueX.DeQue<half>();
    AscendC::LocalTensor<half> yLocal = inQueueY.DeQue<half>();
    AscendC::LocalTensor<half> zLocal = outQueueZ.AllocTensor<half>();
    AscendC::Add(zLocal, xLocal, yLocal, len);
    outQueueZ.EnQue(zLocal);
    inQueueX.FreeTensor(xLocal);
    inQueueY.FreeTensor(yLocal);
  }

  __aicore__ inline void CopyOut(int32_t progress, uint32_t len) {
    AscendC::LocalTensor<half> zLocal = outQueueZ.DeQue<half>();
    AscendC::DataCopy(zGm[progress * TILE_LENGTH], zLocal, len);
    outQueueZ.FreeTensor(zLocal);
  }

private:
  AscendC::TPipe pipe;
  AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX;
  AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueY;
  AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueZ;
  AscendC::GlobalTensor<half> xGm;
  AscendC::GlobalTensor<half> yGm;
  AscendC::GlobalTensor<half> zGm;
  uint32_t totalLength = 0;
};

extern "C" __global__ __aicore__ void add_custom(GM_ADDR x, GM_ADDR y, GM_ADDR z) {
  // 每核处理一段：示例固定总长 8*2048，按 block_idx 切分
  constexpr uint32_t TOTAL = 8 * 2048;
  uint32_t blockDim = AscendC::GetBlockNum();
  uint32_t blockIdx = AscendC::GetBlockIdx();
  uint32_t perBlock = TOTAL / blockDim;
  KernelAdd op;
  op.Init(x + blockIdx * perBlock * sizeof(half),
          y + blockIdx * perBlock * sizeof(half),
          z + blockIdx * perBlock * sizeof(half), perBlock);
  op.Process();
}
