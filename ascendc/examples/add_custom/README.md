# AscendC 最小示例：vector add（CPU 孪生 / NPU）

## 编译运行（需已 source CANN set_env.sh）

```bash
bash run.sh -r cpu -a dav-2201
```

- `-r cpu`：CPU 孪生  
- `-r npu`：上板（需驱动与设备）  
- `-a`：`CMAKE_ASC_ARCHITECTURES`，如 `dav-2201`

## 文件

| 文件 | 作用 |
|---|---|
| `add_custom.cpp` | Kernel |
| `main.cpp` | Host：CPU 用 `ICPU_RUN_KF`，NPU 用 aclrtLaunch |
| `data_utils.h` | 读写 bin |
| `scripts/gen_data.py` | 造输入与真值 |
| `CMakeLists.txt` | 工程 |

## 说明

本骨架对齐官方 Kernel Launch / CPU 孪生写法。若你本机 CANN 版本的 cmake 宏名略有差异，以  
`$ASCEND_HOME_PATH` 下 `ascendc.cmake` / 官方样例为准微调。
