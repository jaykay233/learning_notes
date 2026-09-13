#!/usr/bin/env bash
# AscendC CPU 孪生：Linux 依赖检查 + 环境提示
# 用法：bash scripts/setup_linux_cpu.sh
set -euo pipefail

echo "==> OS"
uname -a
if [[ "$(uname -s)" != "Linux" ]]; then
  echo "ERROR: 当前不是 Linux（$(uname -s)）。"
  echo "CANN Toolkit 不能在 macOS/Windows 原生安装。"
  echo "请用：远程 Ubuntu / VM / Docker（见 ../README.md）。"
  exit 1
fi

echo
echo "==> 工具链"
need_ok=1
check() {
  local name="$1" cmd="$2"
  if command -v "$cmd" >/dev/null 2>&1; then
    echo "  OK  $name: $($cmd --version 2>&1 | head -1)"
  else
    echo "  MISSING  $name ($cmd)"
    need_ok=0
  fi
}
check gcc gcc
check g++ g++
check cmake cmake
check python3 python3
check pip3 pip3

echo
echo "==> 建议安装依赖（Ubuntu/Debian）"
cat <<'EOF'
  sudo apt-get update
  sudo apt-get install -y gcc g++ make cmake net-tools \
    python3 python3-dev python3-pip
  pip3 install --user numpy decorator sympy cffi protobuf scipy requests absl-py
EOF

echo
echo "==> CANN Toolkit"
CANDIDATES=(
  "$HOME/Ascend/ascend-toolkit/latest"
  "/usr/local/Ascend/ascend-toolkit/latest"
  "${ASCEND_HOME_PATH:-}"
  "${ASCEND_INSTALL_PATH:-}"
)
found=""
for p in "${CANDIDATES[@]}"; do
  [[ -z "$p" ]] && continue
  if [[ -f "$p/set_env.sh" ]]; then
    found="$p"
    break
  fi
done

if [[ -n "$found" ]]; then
  echo "  发现 Toolkit: $found"
  echo "  请执行: source \"$found/set_env.sh\""
else
  echo "  未发现已安装的 Toolkit。"
  echo "  1) 从 https://www.hiascend.com/software/cann 下载匹配架构的 .run"
  echo "  2) chmod +x Ascend-cann-toolkit_*.run"
  echo "  3) ./Ascend-cann-toolkit_*.run --install --install-path=\$HOME/Ascend"
  echo "  4) source \$HOME/Ascend/ascend-toolkit/latest/set_env.sh"
fi

echo
echo "==> 验证命令（Toolkit 已 source 后）"
cat <<'EOF'
  echo "ASCEND_HOME_PATH=$ASCEND_HOME_PATH"
  ls "$ASCEND_HOME_PATH/tools" 2>/dev/null | head
  cd examples/add_custom && bash run.sh -r cpu -a dav-2201
EOF

if [[ "$need_ok" -eq 0 ]]; then
  echo
  echo "还有缺失依赖，请先按上面 apt/pip 安装。"
  exit 2
fi

echo
echo "依赖检查通过。下一步：安装/source Toolkit，再编 examples/add_custom。"
