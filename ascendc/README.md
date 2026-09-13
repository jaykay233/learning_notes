# AscendC · CPU 孪生开发环境

> 目标：在 **无 NPU** 的机器上用 CANN Toolkit 做 AscendC **CPU 域调测**（孪生调试），验证 kernel 逻辑后再上板。  
> 官方说明：[CPU 域孪生调试](https://asc.gitcode.com/guide/programming_guide/debug_and_tuning/functional_debug/cpu_twin_debug.html)

本目录已在 **macOS arm64 + Colima + Docker** 上跑通：`add_custom` CPU 孪生 `max_abs_diff=0.0`（CANN **9.2.0-beta.1** aarch64 + 910b-ops）。

---

## 0. 先看清本机约束

| 项目 | 状态 |
|---|---|
| 官方 CANN Toolkit | **仅 Linux**（x86_64 / aarch64），**不能**在 macOS 原生安装 |
| NPU 驱动 | CPU 孪生 **不需要** |
| 本目录能做什么 | 工程骨架、安装脚本、Docker；**真编译在 Linux/容器内** |

推荐路径（三选一）：

1. **远程 / 云上 Ubuntu** → `scripts/setup_linux_cpu.sh`  
2. **本机 Colima（headless Docker，推荐）** → 见下节  
3. **Linux 虚拟机**（UTM / Parallels / VirtualBox）

### macOS：Colima

```bash
export PATH="$HOME/.local/bin:$PATH"   # 或 brew install colima docker
colima start --cpu 4 --memory 8 --vm-type=vz   # Apple Silicon
docker context use colima
docker info   # OS 应为 Linux
```

Toolkit `.run` 装在 **容器内的 Linux**，不是 macOS 原生。

---

## 1. CPU 孪生是什么

```text
NPU 模式:  Kernel 编成 Device 代码 → 上板执行
CPU 模式:  Kernel 当 Host 程序链 CPU 调测库 → 本机跑通 + gdb/printf
```

CANN 8.x/9.x 工程侧用官方 `ascendc_library`，配置：

```bash
cmake -B build -DRUN_MODE=cpu -DSOC_VERSION=Ascend910B1
```

| `SOC_VERSION`（示例） | 常见对应 |
|---|---|
| `Ascend910B1` / `Ascend910B2` … | Atlas A2 / 800I A2（样例默认） |
| 以 `npu-smi info` 的 Name 加 `Ascend` 前缀为准 | 上板时必须匹配 |

旧文档里的 `CMAKE_ASC_ARCHITECTURES=dav-2201` 对应 910B 一代；本示例 `run.sh` 用 `-v Ascend910B1`。

---

## 2. Linux 上装环境（CPU 调测最小集）

### 2.1 系统依赖

- OS：Ubuntu 20.04/22.04 等官方支持发行版  
- gcc/g++ ≥ 7.3、cmake ≥ 3.16、Python 3.7–3.11 + pip  

### 2.2 安装 CANN Toolkit + ops（无需驱动）

1. 从 [昇腾社区 CANN 下载](https://www.hiascend.com/software/cann) 或 OBS 获取匹配架构包。aarch64 示例（9.2.0-beta.1）：

```bash
# Toolkit ~1.4GB
wget https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%209.2.T1/Ascend-cann-toolkit_9.2.0-beta.1_linux-aarch64.run
# 910B ops（与 dav-2201 / Ascend910B* 配套）~2.3GB
wget https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%209.2.T1/Ascend-cann-910b-ops_9.2.0-beta.1_linux-aarch64.run

chmod +x Ascend-cann-*.run
./Ascend-cann-toolkit_9.2.0-beta.1_linux-aarch64.run --quiet --install --install-path=/usr/local/Ascend --install-for-all
./Ascend-cann-910b-ops_9.2.0-beta.1_linux-aarch64.run --quiet --install --install-path=/usr/local/Ascend --install-for-all
```

2. 环境变量（CANN **9.x** 路径；每次开终端或写入 `~/.bashrc`）：

```bash
# 9.x root 默认：
source /usr/local/Ascend/cann/set_env.sh
# 若脚本在 set -u 下报 unbound variable，可先：set +u; source ...; set -u

# 旧版 toolkit 布局仍可能是：
# source $HOME/Ascend/ascend-toolkit/latest/set_env.sh
```

`set_env` 后应有非空的 `ASCEND_HOME_PATH`（9.2 常见为 `/usr/local/Ascend/cann-9.2.0-beta.1`）。

依赖检查：

```bash
bash scripts/setup_linux_cpu.sh
```

---

## 3. 编译运行示例（`examples/add_custom`）

在 **已 source set_env.sh 的 Linux / 容器** 上：

```bash
cd examples/add_custom
bash run.sh -r cpu -v Ascend910B1
```

等价手动：

```bash
cmake -B build -DRUN_MODE=cpu -DSOC_VERSION=Ascend910B1 -DCMAKE_BUILD_TYPE=Debug
cmake --build build -j
# run.sh 会设置 cpudebug / tikicpulib / simulator 的 LD_LIBRARY_PATH
./build/add_custom_cpu
```

期望：8 核 CPU twin `SUCCESS`，`verify_result.py` 报 `max_abs_diff=0.0`。

调试：

```bash
gdb --args ./build/add_custom_cpu
(gdb) set follow-fork-mode child   # CPU 调测常多进程
```

---

## 4. Docker（本机 macOS + Colima）

已验证流程：镜像 `ascendc-cpu-dev`，容器名 `ascendc-dev`，Toolkit/ops 装在容器层。

```bash
# 构建基础镜像（不含 Toolkit；也可 --build-arg CANN_RUN=...）
cd ascendc
docker build -t ascendc-cpu-dev -f docker/Dockerfile .

docker run -d --name ascendc-dev -v "$PWD":/workspace -w /workspace ascendc-cpu-dev sleep infinity

# 把 .run 放到 docker/cann-packages/ 后，在容器内安装（见 §2.2）
docker exec -it ascendc-dev bash
source /usr/local/Ascend/cann/set_env.sh
cd examples/add_custom && bash run.sh -r cpu -v Ascend910B1
```

离线包目录：`docker/cann-packages/`（`*.run` 已 gitignore，勿提交）。

若 Docker Hub 拉不动，给 Colima 内 `dockerd` 配上与宿主机一致的 HTTP(S)_PROXY。

---

## 5. 目录结构

```text
ascendc/
├── README.md
├── .gitignore                # build / cceprint / *.run 等
├── scripts/setup_linux_cpu.sh
├── docker/
│   ├── Dockerfile
│   └── cann-packages/        # 放官方 .run（不入库）
└── examples/add_custom/      # vector add + run.sh
```

---

## 6. 常见坑

| 现象 | 处理 |
|---|---|
| macOS 上 `cmake` 找不到 AscendC | 正常；进 Colima 容器 |
| `ASCEND_HOME_PATH` 空 | `source /usr/local/Ascend/cann/set_env.sh`（9.x） |
| 链接缺 `aclrt*` | CPU 模式需 `tools/cpudebug` 的 `ascendc_acl_stub` 等；用本目录 `CMakeLists.txt` / `run.sh` |
| 运行缺 `libpem_davinci.so` | `LD_LIBRARY_PATH` 加上 `aarch64-linux/simulator/dav_2201/lib`（`run.sh` 已带） |
| 仅 BuiltIn 关键字区分的重载冲突 | CPU 模式关键字被置空，勿靠 `__aicore__` 区分同名函数 |
| 与 NPU 结果不一致 | CPU 孪生偏功能；性能/部分精度以 NPU 为准 |

本地 PDF 参考：`~/Downloads/books/CANN ... Ascend C算子开发指南.pdf`。
