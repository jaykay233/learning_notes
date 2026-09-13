# AscendC · CPU 孪生开发环境

> 目标：在 **无 NPU** 的机器上用 CANN Toolkit 做 AscendC **CPU 域调测**（孪生调试），验证 kernel 逻辑后再上板。  
> 官方说明：[CPU 域孪生调试](https://asc.gitcode.com/guide/programming_guide/debug_and_tuning/functional_debug/cpu_twin_debug.html)

---

## 0. 先看清本机约束

当前仓库所在机器是 **macOS arm64**。

| 项目 | 状态 |
|---|---|
| 官方 CANN Toolkit | **仅 Linux**（x86_64 / aarch64），**不能**在 macOS 原生安装 |
| NPU 驱动 | CPU 孪生 **不需要** |
| 本目录能做什么 | 放工程骨架、安装脚本、Docker 配方；**真编译需 Linux + Toolkit** |

推荐路径（三选一）：

1. **远程 / 云上 Ubuntu**（最省事）→ 跑 `scripts/setup_linux_cpu.sh`  
2. **本机 headless Docker（推荐 Colima，无需 Docker Desktop GUI）** + 本目录 `docker/Dockerfile`  
3. **Linux 虚拟机**（UTM / Parallels / VirtualBox）

### macOS：Colima（无 GUI Docker）

Mac 上没有 Linux 的 `dockerd`，但可以用 **Colima = Lima 轻量 Linux VM + Docker**，全程 CLI：

```bash
# 已装到 ~/.local/bin 时可直接用；否则 brew install colima docker
export PATH="$HOME/.local/bin:$PATH"
colima start --cpu 4 --memory 8 --vm-type=vz   # Apple Silicon
docker context use colima
docker info   # OS 应为 Linux
```

然后在仓库根目录按 `docker/` 里说明 build/run；Toolkit `.run` 仍装在 **容器/VM 内的 Linux**，不是 macOS 原生。

---

## 1. CPU 孪生是什么

```text
NPU 模式:  Kernel 编成 Device 代码 → 上板执行
CPU 模式:  Kernel 当 Host 程序链 CPU 调测库 → 本机跑通 + gdb/printf
```

CMake 关键：

```bash
cmake -B build \
  -DCMAKE_ASC_RUN_MODE=cpu \
  -DCMAKE_ASC_ARCHITECTURES=dav-2201   # 按目标芯片改，见下表
```

| `CMAKE_ASC_ARCHITECTURES` | 常见对应 |
|---|---|
| `dav-2201` | Atlas A2 / 800I A2 等（样例默认） |
| `dav-3510` | Ascend 950 系列（以你 CANN 版本文档为准） |

`SOC_VERSION`（旧工程 `run.sh -v`）形如 `Ascend910B1`，以 `npu-smi info` 的 Name 为准；纯 CPU 调测可先用文档示例值。

---

## 2. Linux 上装环境（CPU 调测最小集）

### 2.1 系统依赖

- OS：Ubuntu 20.04/22.04 等官方支持发行版  
- gcc/g++ ≥ 7.3  
- cmake ≥ 3.16  
- Python 3.7–3.11.x + pip  

### 2.2 安装 CANN Toolkit（无需驱动）

1. 从 [昇腾社区 / CANN 下载](https://www.hiascend.com/software/cann) 获取与架构匹配的  
   `Ascend-cann-toolkit_*_linux-x86_64.run` 或 `*_linux-aarch64.run`  
2. 安装示例：

```bash
chmod +x Ascend-cann-toolkit_*.run
# 非 root 常见装到 $HOME/Ascend
./Ascend-cann-toolkit_*.run --install --install-path=$HOME/Ascend
```

3. 环境变量（每次开终端或写入 `~/.bashrc`）：

```bash
# 路径按实际安装位置改
source $HOME/Ascend/ascend-toolkit/latest/set_env.sh
# 或 root 默认：
# source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh

export ASCEND_HOME_PATH=${ASCEND_HOME_PATH:-$HOME/Ascend/ascend-toolkit/latest}
export ASCEND_INSTALL_PATH=${ASCEND_INSTALL_PATH:-$ASCEND_HOME_PATH}
```

一键检查依赖 + 打印下一步：本目录

```bash
bash scripts/setup_linux_cpu.sh
```

---

## 3. 编译运行示例（`examples/add_custom`）

在 **已 source set_env.sh 的 Linux** 上：

```bash
cd examples/add_custom
bash run.sh -r cpu -a dav-2201
# 或手动：
# cmake -B build -DCMAKE_ASC_RUN_MODE=cpu -DCMAKE_ASC_ARCHITECTURES=dav-2201
# cmake --build build -j
# ./build/add_custom_cpu
```

期望：生成可执行文件，CPU 上跑完 vector add，并与 `scripts/gen_data.py` 真值对齐（见示例 README）。

调试：

```bash
gdb --args ./build/add_custom_cpu
(gdb) set follow-fork-mode child   # CPU 调测常多进程
```

也可用 kernel 里 `printf` 打中间量。

---

## 4. Docker（本机 macOS + Colima）

当前环境已验证：**CANN 9.2.0-beta.1**（aarch64 Toolkit + 910b-ops）装在容器 `ascendc-dev` 内。

```bash
export PATH="$HOME/.local/bin:$PATH"
docker exec -it ascendc-dev bash
# 容器内：
source /usr/local/Ascend/cann/set_env.sh
cd examples/add_custom && bash run.sh -r cpu -v Ascend910B1
```

离线包可放 `docker/cann-packages/`（已 `.gitignore` 忽略 `*.run`）。官方 OBS 直链示例：

- Toolkit: `https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%209.2.T1/Ascend-cann-toolkit_9.2.0-beta.1_linux-aarch64.run`
- 910B ops: `.../Ascend-cann-910b-ops_9.2.0-beta.1_linux-aarch64.run`

重建镜像时可用 `--build-arg CANN_RUN=...`；现有容器是运行时 `docker exec` 安装的，重启容器后 Toolkit 仍在（写在容器层）。

---

## 5. 目录结构

```text
ascendc/
├── README.md                 # 本文件
├── scripts/
│   └── setup_linux_cpu.sh    # Linux 依赖检查 + 环境提示
├── docker/
│   ├── Dockerfile
│   └── cann-packages/.gitkeep
└── examples/
    └── add_custom/           # 最小 vector add（CPU/NPU 双路径骨架）
```

---

## 6. 常见坑

| 现象 | 处理 |
|---|---|
| macOS 上 `cmake` 找不到 AscendC | 正常；换 Linux/容器 |
| `ASCEND_HOME_PATH` 空 | 先 `source .../set_env.sh` |
| 仅 BuiltIn 关键字区分的重载冲突 | CPU 模式关键字被置空，勿靠 `__aicore__` 区分同名函数 |
| 与 NPU 结果不一致 | CPU 孪生偏功能/流程；性能与部分精度以 NPU 为准 |

本地 PDF 参考：`~/Downloads/books/CANN ... Ascend C算子开发指南.pdf`。
