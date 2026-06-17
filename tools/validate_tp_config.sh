#!/usr/bin/env bash
# Validate launcher tensor-parallel / GPU selection consistency (dry checks).
set -euo pipefail

ROOT=${STABLE_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}
LAUNCHER="$ROOT/launcher.sh"

if [[ ! -x "$LAUNCHER" && ! -f "$LAUNCHER" ]]; then
  echo "launcher not found: $LAUNCHER" >&2
  exit 2
fi

bash -n "$ROOT/build.sh" "$LAUNCHER" "$ROOT/tools/validate_profiles.sh"

run_case() {
  local name=$1
  shift
  echo "CASE $name"
  if "$@"; then
    echo "  OK"
  else
    echo "  FAIL" >&2
    return 1
  fi
}

fail_case() {
  local name=$1
  shift
  echo "CASE $name (expect failure)"
  if "$@"; then
    echo "  FAIL: expected non-zero exit" >&2
    return 1
  fi
  echo "  OK (failed as expected)"
}

export MODEL_DIR=${MODEL_DIR:-/tmp/vllm-dummy-model}
mkdir -p "$MODEL_DIR"
touch "$MODEL_DIR/config.json"

errors=0

run_case "tp2 matches two GPUs" env GPU_DEVICES=0,1 TP_SIZE=2 PROFILE= \
  bash "$LAUNCHER" --print-config >/dev/null || errors=$((errors + 1))

run_case "tp4 matches four GPUs" env GPU_DEVICES=0,1,2,3 TP_SIZE=4 \
  PROFILE=qwen27b/tp4/normal/int4/fp16kv-128K-mtp3-text-only.env \
  bash "$LAUNCHER" --print-config >/dev/null || errors=$((errors + 1))

fail_case "tp2 with four GPUs rejected" env GPU_DEVICES=0,1,2,3 TP_SIZE=2 \
  bash "$LAUNCHER" --print-config >/dev/null || errors=$((errors + 1))

if (( errors > 0 )); then
  echo "validate_tp_config: $errors case group(s) failed" >&2
  exit 1
fi

echo "validate_tp_config: all checks passed"
