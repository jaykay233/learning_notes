#!/usr/bin/env bash
# Usage: ./scripts/check_diff.sh /path/to/lab1-opt
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OPT="${1:?usage: $0 /path/to/lab1-opt}"
PIPE='builtin.module(func.func(lab1-eliminate-identity))'

fail=0
for inp in "$ROOT"/inputs/0{1,2,3,4}_*.mlir; do
  base="$(basename "$inp")"
  exp="$ROOT/expected/$base"
  out="$(mktemp)"
  echo "==> $base"
  "$OPT" "$inp" --pass-pipeline="$PIPE" -o "$out" || { echo "PASS FAILED on $base"; fail=1; continue; }
  # 粗对比：期望中不应再出现 lab.identity（01-03）；04 本就没有
  if grep -q 'lab.identity' "$out"; then
    echo "FAIL: still has lab.identity"
    cat "$out"
    fail=1
  else
    echo "OK: no lab.identity left (compare dataflow with expected/$base manually if needed)"
  fi
  rm -f "$out"
done
exit "$fail"
