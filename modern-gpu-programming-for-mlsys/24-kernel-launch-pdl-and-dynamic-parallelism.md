# 二十四、GPU Kernel Launch：PDL 与 Dynamic Parallelism

## 本次讲解位置

```text
本次讲解位置
章节：CUDA 执行机制补充（不属于教材 chapter_flash_attention）
小节：CUDA Programming Guide 4.5 Programmatic Dependent Launch；
     4.20 CUDA Dynamic Parallelism
知识点：区分 host launch、GPU dispatch、PDL 与 device-side child launch
上次：Define the Timing Boundary：host timer 与 CUDA event 的测量边界
下次：warm-up、repeat、rounds 与样本稳定性
PTX：无；讨论 CUDA Runtime / Device Runtime 的 launch 与依赖契约
```

## 为什么现在讲这个

前面区分了 CPU 框架算子分发、CPU 调用 CUDA launch API、GPU 调度 kernel
和 GPU 执行 kernel。现在遇到的关键判断是：一个小算子慢，究竟是 CPU
提交开销、GPU kernel 间的依赖等待，还是工作本身太多？

如果把这几层统称为“kernel launch 开销”，容易得出两个错误结论：以为
PDL 会消除 CPU 下发开销，或以为 Dynamic Parallelism 是 PDL 的另一种写法。
本节把两个机制放到同一条时间线上，说明该用 CUDA Graph、PDL，还是
device-side launch，并指出每种机制仍然要求什么同步。

## 一、先把四个边界分开

```text
框架收到 operator 调用
    -> CPU dispatcher 选择实现、准备参数
        -> CPU 调用 CUDA launch API，提交 GPU 工作
            -> GPU dispatch / 调度 CTA
                -> kernel 在 SM 上执行
```

| 名称 | 所在位置 | 典型工作 |
|---|---|---|
| 算子分发 | CPU 框架层 | Dispatcher、参数检查、选择实现；一个 operator 可对应多个 kernel |
| host launch | CPU / CUDA Runtime | `kernel<<<...>>>()` 或 `cudaLaunchKernelEx` 提交一个 kernel |
| GPU dispatch | GPU 执行层 | GPU 让 CTA 获得执行资源并开始运行 |
| kernel execution | GPU SM | CTA 中的线程执行 kernel body |

因此，口语中的 “kernel launch overhead” 有时指 host API 提交开销，有时指
GPU 侧 kernel 启动/调度延迟。讨论优化前要先说清计时边界。

## 二、PDL 的心智模型：有依赖，但不必等到最后一刻才启动 B

假设同一 CUDA stream 中先提交 A，再提交依赖 A 结果的 B。普通 stream
语义下，B 要等 A 完整结束。PDL（Programmatic Dependent Launch）允许 B
在 A 还没结束时开始调度；B 可以先做不依赖 A 的准备工作，再在读取 A
结果前等待依赖完成。

```text
时间 ───────────────────────────────────────────────────────────>

普通 stream
A:  [生成结果──────────────收尾]
B:                             [GPU 启动/调度][前置工作][读 A]

PDL
A:  [生成结果][trigger][──────收尾]
B:                       [GPU 启动/调度][独立前置工作][等待][读 A]
```

PDL 不是把 host launch API 搬到 GPU 上，也不保证两个 kernel 一定并发。
CPU 仍然提交 A 和 B；PDL 让 GPU 有机会把 B 的启动/调度延迟及 B 的独立
前置工作，藏在 A 尚未结束的时间里。CUDA 文档明确将这种机会描述为
opportunistic：能否实际并发，取决于资源和调度。

### PDL 的三个动作

1. **Primary A** 在适当位置调用 `cudaTriggerProgrammaticLaunchCompletion()`。
   它通知 GPU：secondary B 可以开始调度。该调用要由 A 的每个 CTA 执行；
   下方示例只有一个 CTA，因此该 CTA 中的线程都会调用。
2. **Host** 在同一个 stream 中提交 B，并在 B 的 extensible launch config
   上设置 `cudaLaunchAttributeProgrammaticStreamSerialization`。
3. **Secondary B** 在使用 A 的输出前调用
   `cudaGridDependencySynchronize()`，或采用等价的正确依赖同步。trigger
   本身不代表 A 写入的数据已经对 B 可见。

触发点应放在“B 可以安全开始调度”的位置；不要因为 B 已经开始运行就
假设它可以读取 A 的任意输出。数据依赖仍由同步边界保护。

### 哪段 launch latency 被隐藏

PDL 不缩短 CPU 调用 launch API 的耗时。它争取重叠的是 GPU 侧 B 的启动/
调度，以及 B 的独立前置工作。

用一个纯示意的时间 trace：

```text
A: GPU 执行到 t=6 us 时发出 trigger，之后还需 4 us 收尾
B: GPU 启动/调度需 3 us，独立前置工作需 2 us
```

普通串行时，A 到 `t=10 us` 才结束，B 才开始它的 `3 + 2 us` 前置段，
所以 B 的依赖工作最早约在 `t=15 us` 开始。

PDL 下，B 从 `t=6 us` 起步，约 `t=11 us` 完成启动和前置段；A 在
`t=10 us` 完成，因此 B 通过依赖同步后约在 `t=11 us` 进入依赖工作。
这个理想化例子把最多 `4 us` 重叠掉了。数字只是解释关键路径，不是
硬件性能承诺；如果没有独立前置工作、A 收尾很短或资源不足，收益会变小
甚至没有。

## 三、Dynamic Parallelism 的心智模型：GPU 运行时决定并启动子任务

Dynamic Parallelism（CUDA Dynamic Parallelism，CDP）允许 GPU 上执行的
parent kernel 依据 device 数据，直接从 device code 发起 child kernel：

```text
CPU 提交 parent kernel
    -> parent 在 GPU 上检查数据
        -> parent 决定是否及如何启动 child grid
            -> child 在 GPU 上执行
```

```cpp
__global__ void child_kernel(int *count, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        atomicAdd(count, 1);
    }
}

__global__ void parent_kernel(int *count, int n) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        int block_size = 128;
        int grid_size = (n + block_size - 1) / block_size;
        child_kernel<<<grid_size, block_size>>>(count, n);
    }
}
```

这里 parent 的一个 GPU thread 根据 `n` 配置 child grid。CPU 仍然必须
启动 parent；只是 CPU 不需要先知道每个 child 的具体 launch 配置。
Dynamic Parallelism 依赖 CUDA Device Runtime，编译通常需要 relocatable
device code（例如 `nvcc -rdc=true`）。

child launch 不会自动带来并发保证，也不是免费的。它适合子任务取决于
GPU 上刚算出的数据、任务结构不规则且让 CPU 逐项回读再下发代价很高的
场景。若任务列表已知，批量 host launch 或 CUDA Graphs 往往更直接。

## 四、机制对照：不要把“由谁发起”与“何时开始”混为一谈

| 机制 | 谁提交/发起 kernel | 解决什么问题 | 不解决什么 |
|---|---|---|---|
| 普通 host launch | CPU | 把 kernel 放入 stream | 不自动减少 CPU API 提交次数 |
| CUDA Graphs | CPU 建图/实例化并 replay | 降低重复的 host 提交开销，固定工作流可整体 replay | 不会自动消除 kernel 间数据依赖 |
| PDL | CPU 提交 A、B；GPU 按 programmatic dependency 提前调度 B | 重叠 B 的启动和独立前置工作与 A 的尾部 | 不消除 host API 成本，也不取消数据同步 |
| Dynamic Parallelism | GPU 上的 parent kernel 发起 child | 让 GPU 根据 device 数据动态生成工作 | 不保证并发，device-side launch 有成本 |

一个好记的分类方法：

```text
反复提交相同工作流的 CPU 开销       -> CUDA Graphs
同一 stream 中依赖 kernel 的等待间隙 -> PDL
GPU 运行时才知道要生成哪些子任务     -> Dynamic Parallelism
```

PDL 与 Dynamic Parallelism 可以组合，但概念上彼此独立：PDL 控制一对已
提交 kernel 的依赖启动边界；CDP 改变 child kernel launch 的发起方。

## 五、完整可运行示例

下面同一个 `.cu` 文件分别验证 PDL 的依赖结果与 CDP 的 parent-child
launch。PDL 测试只验证正确性；它不试图证明 kernel 一定并发，因为并发是
机会性行为。

文件名：`pdl_dynamic_parallelism_demo.cu`

```cpp
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t error__ = (call);                                         \
        if (error__ != cudaSuccess) {                                         \
            std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__,          \
                         cudaGetErrorString(error__));                        \
            std::exit(EXIT_FAILURE);                                          \
        }                                                                     \
    } while (0)

constexpr int kN = 256;
constexpr int kBlockSize = 128;

__global__ void pdl_primary(int *produced, int *tail, int n) {
    int i = threadIdx.x;
    if (i < n) {
        produced[i] = i + 1;
    }

    __syncthreads();

    // Every thread in every primary CTA must reach the programmatic trigger.
    cudaTriggerProgrammaticLaunchCompletion();

    // Independent tail work: the secondary kernel does not read this array.
    if (i < n) {
        tail[i] = n - i;
    }
}

__global__ void pdl_secondary(const int *produced, int *result, int n) {
    int i = threadIdx.x;
    int independent_value = 2 * i;

    // All threads in this one-CTA grid reach the dependency point uniformly.
    cudaGridDependencySynchronize();

    if (i < n) {
        result[i] = produced[i] + independent_value;
    }
}

__global__ void cdp_child(int *count, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        atomicAdd(count, 1);
    }
}

__global__ void cdp_parent(int *count, int n) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        int grid_size = (n + kBlockSize - 1) / kBlockSize;
        cdp_child<<<grid_size, kBlockSize>>>(count, n);
    }
}

void run_pdl_demo() {
    int *d_produced = nullptr;
    int *d_tail = nullptr;
    int *d_result = nullptr;
    int h_result[kN] = {};
    cudaStream_t stream = nullptr;

    CUDA_CHECK(cudaStreamCreate(&stream));
    CUDA_CHECK(cudaMalloc(&d_produced, kN * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_tail, kN * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_result, kN * sizeof(int)));

    pdl_primary<<<1, kN, 0, stream>>>(d_produced, d_tail, kN);
    CUDA_CHECK(cudaGetLastError());

    cudaLaunchAttribute attribute{};
    attribute.id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attribute.val.programmaticStreamSerializationAllowed = 1;

    cudaLaunchConfig_t config{};
    config.gridDim = dim3(1, 1, 1);
    config.blockDim = dim3(kN, 1, 1);
    config.dynamicSmemBytes = 0;
    config.stream = stream;
    config.attrs = &attribute;
    config.numAttrs = 1;

    CUDA_CHECK(cudaLaunchKernelEx(&config, pdl_secondary, d_produced,
                                  d_result, kN));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaMemcpy(h_result, d_result, kN * sizeof(int),
                          cudaMemcpyDeviceToHost));

    for (int i = 0; i < kN; ++i) {
        int expected = 3 * i + 1;
        if (h_result[i] != expected) {
            std::fprintf(stderr,
                         "PDL mismatch at %d: got %d, expected %d\n",
                         i, h_result[i], expected);
            std::exit(EXIT_FAILURE);
        }
    }

    std::printf("PDL: PASS (result[0]=%d, result[%d]=%d)\n",
                h_result[0], kN - 1, h_result[kN - 1]);

    CUDA_CHECK(cudaFree(d_result));
    CUDA_CHECK(cudaFree(d_tail));
    CUDA_CHECK(cudaFree(d_produced));
    CUDA_CHECK(cudaStreamDestroy(stream));
}

void run_cdp_demo() {
    int *d_count = nullptr;
    int h_count = -1;
    cudaStream_t stream = nullptr;

    CUDA_CHECK(cudaStreamCreate(&stream));
    CUDA_CHECK(cudaMalloc(&d_count, sizeof(int)));
    CUDA_CHECK(cudaMemsetAsync(d_count, 0, sizeof(int), stream));

    // The CPU launches only the parent. The parent launches the child grid.
    cdp_parent<<<1, 1, 0, stream>>>(d_count, kN);
    CUDA_CHECK(cudaGetLastError());

    // Parent completion is nested after its child grid; stream sync also waits.
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaMemcpy(&h_count, d_count, sizeof(int),
                          cudaMemcpyDeviceToHost));

    if (h_count != kN) {
        std::fprintf(stderr, "CDP count mismatch: got %d, expected %d\n",
                     h_count, kN);
        std::exit(EXIT_FAILURE);
    }

    std::printf("Dynamic Parallelism: PASS (child count=%d)\n", h_count);

    CUDA_CHECK(cudaFree(d_count));
    CUDA_CHECK(cudaStreamDestroy(stream));
}

int main() {
    run_pdl_demo();
    run_cdp_demo();
    return 0;
}
```

### 编译与运行

当前示例同时用了 PDL 与 device-side child launch。PDL 需要 compute
capability 9.0 或更新设备；编译目标和本机 GPU 必须匹配。Dynamic
Parallelism 使用 device runtime 与 relocatable device code。

```bash
nvcc -std=c++17 -arch=sm_90 -rdc=true \
  pdl_dynamic_parallelism_demo.cu -lcudadevrt \
  -o pdl_dynamic_parallelism_demo
./pdl_dynamic_parallelism_demo
```

预期输出：

```text
PDL: PASS (result[0]=1, result[255]=766)
Dynamic Parallelism: PASS (child count=256)
```

若设备不是 `sm_90`，应改成受支持且匹配本机设备的目标；若设备低于
compute capability 9.0，不能验证 PDL 的重叠路径。程序的 PDL 断言只验证
数据依赖和结果正确，不验证时间重叠；实际收益需在支持设备上用 CUDA
event/Nsight Systems 测量，并把 host submission 与 GPU stream 时间分开。

本次写作环境为 macOS ARM，未检测到 `nvcc` 或 NVIDIA GPU 工具，因此示例
无法在本机编译和运行。上面的预期输出是根据代码路径推导的；本次完成了
API 文档核对与静态检查，未完成 CUDA 编译或硬件运行验证。

## 六、逐段执行 trace

### PDL 示例，`kN = 256`

| 阶段 | 执行位置与范围 | 数据/依赖 |
|---|---|---|
| Host 提交 `pdl_primary` | CPU，stream 中第一个 launch | CPU API 提交成本仍存在 |
| `produced[i] = i + 1` | 一个 CTA 的 256 个线程 | 写 global memory，结果供 secondary 读取 |
| `__syncthreads()` | primary CTA 全体线程 | 确保本 CTA 的 produced 写入先于该 CTA 的 trigger |
| `cudaTrigger...()` | 该示例 primary CTA 的线程 | 允许 secondary 开始调度，不代表输出已可读 |
| 写 `tail[i]` | primary CTA | 与 secondary 无数据依赖的尾部工作 |
| Host 提交 secondary | CPU，仍是同一个 stream | launch attribute 开启 programmatic serialization |
| 计算 `2 * i` | secondary CTA | 独立前置工作，不读 primary 的输出 |
| `cudaGridDependencySynchronize()` | secondary grid，全体线程一致到达 | 等 primary 完成并使结果可用 |
| `result[i] = produced[i] + 2*i` | secondary CTA | `i+1+2i=3i+1` |

手算两个坐标：

```text
i = 0:
produced[0] = 0 + 1 = 1
result[0]   = 1 + 2*0 = 1

i = 255:
produced[255] = 255 + 1 = 256
result[255]   = 256 + 2*255 = 766
```

host 最终检查全部 `256` 个元素；如果同步被删掉，secondary 可能在 primary
输出对它可见之前就读取，产生非确定性错误。通过一次成功运行不能证明
同步是多余的。

### Dynamic Parallelism 示例，`kN = 256`

```text
CPU launch cdp_parent<<<1, 1>>>
  parent grid: 1 CTA x 1 thread
    child grid size = ceil(256 / 128) = 2 CTAs
      child CTA 0: i = 0..127，128 次 atomicAdd
      child CTA 1: i = 128..255，128 次 atomicAdd
  child 结束，count = 128 + 128 = 256
parent/child 嵌套工作完成
host stream synchronize 后读回 count
```

`count` 的初值由同一 stream 上的 `cudaMemsetAsync` 设为 0。child 的
每个有效线程原子加 1，所以读回值可预期为 256。不能把 parent launch
返回误认为 child 已经完成；示例用 stream synchronization 建立 host 读回
边界。CUDA Dynamic Parallelism 的 parent-child completion 是嵌套语义，
而 parent kernel 内部读取 child 写入数据还受当前 CDP 版本的 device
memory consistency 规则约束；不要据此推断 parent 可在任意位置读回 child
结果。

## 七、常见混淆与症状

| 混淆 | 实际情况 | 可能症状 |
|---|---|---|
| “PDL 把 CPU launch 移到 GPU” | PDL 的 A/B 仍由 CPU 提交 | host API 延迟没有下降，误判为 PDL 无效 |
| “trigger 后 B 可以读 A 的所有结果” | trigger 只允许 B 提前调度；依赖同步仍必要 | 结果偶发错、时序敏感 |
| “PDL 保证 A、B 并发” | 并发是机会性行为，受资源和调度影响 | 正确性测试通过，但 Nsight 看不到 overlap |
| “Dynamic Parallelism 是 PDL” | CDP 改 launch 发起者；PDL 改同 stream 依赖启动边界 | 选错机制，增加无意义的 device-side launch |
| “operator dispatch 等于 kernel launch” | 一个 operator 可能无 kernel、一个 kernel 或多个 kernel | benchmark 边界与优化对象不一致 |
| “device-side launch 必定比 host launch 快” | child launch 也有 runtime/调度成本 | 小任务变慢，递归层次增加后开销显著 |

## 八、自测题与答案

### 1. PDL 是否减少 CPU 调用 `cudaLaunchKernelEx` 的时间？

答：不减少。它尝试重叠 GPU 侧 secondary kernel 的启动/调度与 primary
仍在执行的时间。要降低重复 host 提交开销，应评估 CUDA Graphs。

### 2. A 调用 `cudaTriggerProgrammaticLaunchCompletion()` 后，B 能立刻读取 A 写的数据吗？

答：不能这样假设。B 必须在读依赖数据前使用
`cudaGridDependencySynchronize()` 或其他正确同步机制。

### 3. Dynamic Parallelism 中，谁 launch child kernel？CPU 是否完全不参与？

答：parent kernel 中的 GPU thread 发起 child launch。CPU 仍先 launch
parent，并负责外层工作流和资源生命周期。

### 4. 哪种机制适合固定、重复 replay 的多 kernel 序列？哪种适合 device 数据才决定的子任务？

答：固定重复序列优先评估 CUDA Graphs；运行时由 GPU 数据决定的子任务可
评估 Dynamic Parallelism。若已知的 A/B 依赖链中有可重叠的独立前置工作，
再评估 PDL。

## 参考

- [CUDA Programming Guide: Programmatic Dependent Launch and Synchronization](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/programmatic-dependent-launch.html)
- [CUDA Programming Guide: CUDA Dynamic Parallelism](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/dynamic-parallelism.html)
