# AscendC 最小示例：vector add（CPU 孪生 / NPU）

## 编译运行（需已 source CANN set_env.sh）

```bash
bash run.sh -r cpu -v Ascend910B1
```

| 参数 | 含义 |
|---|---|
| `-r cpu` | CPU 孪生（默认） |
| `-r npu` / `-r sim` | 上板 / 仿真（需对应环境） |
| `-v Ascend910B1` | `SOC_VERSION`（Atlas A2 / 910B 一代示例） |
| `-a dav-2201` | 兼容旧写法，映射为 `Ascend910B1` |

期望：`CPU twin run done`，打印 `timing: kernel_wall_ms_avg=...`，`max_abs_diff=0.0`。

计时可调（默认 warmup=1、iters=5）：

```bash
ASCENDC_BENCH_WARMUP=2 ASCENDC_BENCH_ITERS=20 bash run.sh -r cpu -v Ascend910B1
```

注意：这是 **CPU 孪生墙钟时间**（含 twin runtime），只能作相对对比，**不是** NPU 真实性能。

## 文件

| 文件 | 作用 |
|---|---|
| `add_custom.cpp` | Kernel |
| `main.cpp` | Host：CPU 用 `ICPU_RUN_KF` |
| `data_utils.h` | 读写 bin |
| `scripts/gen_data.py` | 造输入与真值 |
| `scripts/verify_result.py` | 对比 output / golden |
| `CMakeLists.txt` | `ascendc_library` + host 可执行文件 |
| `run.sh` | 一键 cmake / build / 跑 / 校验（含 `LD_LIBRARY_PATH`） |

## 说明

对齐 CANN 8.x/9.x Kernel Launch / CPU 孪生：`include(ascendc.cmake)` + `ascendc_library(... SHARED ...)`。  
若本机 CANN 的 cmake 布局略有差异，以 `$ASCEND_HOME_PATH` 下 `tools/tikcpp/ascendc_kernel_cmake` 与官方样例为准。
