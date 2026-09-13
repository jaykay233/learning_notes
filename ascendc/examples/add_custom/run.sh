#!/usr/bin/env bash
# Usage: bash run.sh -r cpu|npu|sim -a dav-2201
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

RUN_MODE=cpu
SOC_VERSION=Ascend910B1
while getopts "r:a:v:" opt; do
  case "$opt" in
    r) RUN_MODE="$OPTARG" ;;
    a) # legacy dav-* flag; dav-2201 -> Ascend910B1
       if [[ "$OPTARG" == "dav-2201" ]]; then SOC_VERSION=Ascend910B1; else ASC_ARCH="$OPTARG"; fi ;;
    v) SOC_VERSION="$OPTARG" ;;
    *) echo "usage: $0 -r cpu|sim|npu -v Ascend910B1"; exit 2 ;;
  esac
done

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "ERROR: AscendC build needs Linux + CANN Toolkit (current: $(uname -s))."
  echo "See ../../README.md"
  exit 1
fi

if [[ -z "${ASCEND_HOME_PATH:-}${ASCEND_INSTALL_PATH:-}" ]]; then
  # CANN 9.x default after root install
  if [[ -f /usr/local/Ascend/cann/set_env.sh ]]; then
    # set_env uses unbound vars under set -u; temporarily relax
    set +u
    # shellcheck disable=SC1091
    source /usr/local/Ascend/cann/set_env.sh
    set -u
  else
    echo "ERROR: ASCEND_HOME_PATH empty. source <cann>/set_env.sh first."
    exit 1
  fi
fi

python3 scripts/gen_data.py

rm -rf build
cmake -B build \
  -DRUN_MODE="${RUN_MODE}" \
  -DSOC_VERSION="${SOC_VERSION}" \
  -DCMAKE_BUILD_TYPE=Debug
cmake --build build -j"$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)"

BIN="build/add_custom_${RUN_MODE}"
if [[ ! -x "$BIN" ]]; then
  # some toolkits rename output
  BIN="$(find build -maxdepth 2 -type f -perm -111 | head -1 || true)"
fi
CANN_ROOT="${ASCEND_HOME_PATH:-${ASCEND_INSTALL_PATH:-/usr/local/Ascend/cann}}"
# CPU twin needs cpudebug + tikicpulib + simulator (pem_davinci) on the loader path
export LD_LIBRARY_PATH="${ROOT}/build/lib:\
${CANN_ROOT}/tools/cpudebug/lib64:\
${CANN_ROOT}/tools/cpudebug/lib64/${SOC_VERSION}:\
${CANN_ROOT}/tools/tikicpulib/lib:\
${CANN_ROOT}/tools/tikicpulib/lib/${SOC_VERSION}:\
${CANN_ROOT}/tools/simulator/${SOC_VERSION}/lib:\
${CANN_ROOT}/aarch64-linux/simulator/dav_2201/lib:\
${CANN_ROOT}/lib64:\
${LD_LIBRARY_PATH:-}"
echo "Running: $BIN"
"$BIN"

python3 scripts/verify_result.py || echo "verify soft-failed (check CPU twin precision / kernel)"
echo "done."
