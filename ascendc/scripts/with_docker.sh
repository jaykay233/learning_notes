#!/usr/bin/env bash
# Ephemeral AscendC Docker run (no always-on container).
#
# Usage (from ascendc/ or repo):
#   bash scripts/with_docker.sh                         # interactive shell
#   bash scripts/with_docker.sh make                    # run default: add_custom cpu twin
#   bash scripts/with_docker.sh -- cmd...               # arbitrary command in container
#   bash scripts/with_docker.sh --stop-colima -- make   # after run, colima stop
#
# Image: ascendc-cpu-dev:cann92 (Toolkit baked in). Falls back to ascendc-cpu-dev:latest.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PATH="${HOME}/.local/bin:${PATH}"
export DOCKER_HOST="${DOCKER_HOST:-unix://${HOME}/.colima/default/docker.sock}"

IMAGE="${ASCENDC_DOCKER_IMAGE:-ascendc-cpu-dev:cann92}"
STOP_COLIMA=0
ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --stop-colima) STOP_COLIMA=1; shift ;;
    --image) IMAGE="$2"; shift 2 ;;
    --) shift; ARGS+=("$@"); break ;;
    make) ARGS+=("make"); shift; break ;;
    *) ARGS+=("$@"); break ;;
  esac
done

ensure_colima() {
  if ! command -v colima >/dev/null 2>&1; then
    echo "ERROR: colima not found. export PATH=\$HOME/.local/bin:\$PATH" >&2
    exit 1
  fi
  if ! colima status 2>&1 | grep -qi "is running"; then
    echo "==> colima start (vz)..."
    colima start --cpu 4 --memory 8 --vm-type=vz
  fi
  if ! docker info >/dev/null 2>&1; then
    echo "ERROR: docker not reachable via $DOCKER_HOST" >&2
    exit 1
  fi
}

pick_image() {
  if docker image inspect "$IMAGE" >/dev/null 2>&1; then
    return
  fi
  if docker image inspect ascendc-cpu-dev:latest >/dev/null 2>&1; then
    echo "WARN: $IMAGE missing; using ascendc-cpu-dev:latest (may lack Toolkit)" >&2
    IMAGE=ascendc-cpu-dev:latest
    return
  fi
  echo "ERROR: no image. Build or: docker commit <container> ascendc-cpu-dev:cann92" >&2
  exit 1
}

run_default_example() {
  # shellcheck disable=SC2016
  cat <<'EOS'
set +u
if [ -f /usr/local/Ascend/cann/set_env.sh ]; then
  source /usr/local/Ascend/cann/set_env.sh
fi
set -e
cd /workspace/examples/add_custom
bash run.sh -r cpu -v Ascend910B1
EOS
}

ensure_colima
pick_image

DOCKER_ARGS=(
  run --rm
  -v "${ROOT}:/workspace"
  -w /workspace
)

if [[ ${#ARGS[@]} -eq 0 ]]; then
  echo "==> interactive: $IMAGE (exit to stop container)"
  docker "${DOCKER_ARGS[@]}" -it "$IMAGE" bash
elif [[ ${#ARGS[@]} -eq 1 && "${ARGS[0]}" == "make" ]]; then
  echo "==> one-shot: add_custom CPU twin"
  docker "${DOCKER_ARGS[@]}" "$IMAGE" bash -lc "$(run_default_example)"
else
  echo "==> one-shot: ${ARGS[*]}"
  docker "${DOCKER_ARGS[@]}" "$IMAGE" bash -lc \
    'set +u; [ -f /usr/local/Ascend/cann/set_env.sh ] && source /usr/local/Ascend/cann/set_env.sh; set -e; '"${ARGS[*]}"
fi

if [[ "$STOP_COLIMA" -eq 1 ]]; then
  echo "==> colima stop"
  colima stop
fi
