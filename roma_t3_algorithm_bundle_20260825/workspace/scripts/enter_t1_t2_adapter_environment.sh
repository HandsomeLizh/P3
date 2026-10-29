#!/usr/bin/env bash
# Source this file to load the Task 3 base plus the optional T1/T2 adapter.

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "[ERROR] This file must be sourced:" >&2
  echo "        source $0" >&2
  exit 1
fi

set +u

TASK_ADAPTER_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export T3_ADAPTER_WORKSPACE_ROOT="$(cd "${TASK_ADAPTER_SCRIPT_DIR}/.." && pwd)"
TASK_ADAPTER_NATIVE_ROOT="${T3_NATIVE_RUNTIME_ROOT:-${T3_ADAPTER_WORKSPACE_ROOT}/offline_deploy/runtime/native_orin}"
TASK_ADAPTER_USED_NATIVE=false

if [[ "$(uname -m)" == "aarch64" && \
      -f "${TASK_ADAPTER_NATIVE_ROOT}/ros2_humble/install/setup.bash" && \
      -f "${TASK_ADAPTER_NATIVE_ROOT}/t3_gridmap/install/setup.bash" && \
      -f "${TASK_ADAPTER_NATIVE_ROOT}/venv/bin/activate" ]]; then
  export T3_NATIVE_RUNTIME_ROOT="${TASK_ADAPTER_NATIVE_ROOT}"
  source "${T3_ADAPTER_WORKSPACE_ROOT}/offline_deploy/scripts/enter_native_environment.sh"
  TASK_ADAPTER_USED_NATIVE=true
  export T3_ADAPTER_BASE_MODE=native
else
  source "${T3_ADAPTER_WORKSPACE_ROOT}/scripts/enter_workspace.sh"
  export T3_ADAPTER_BASE_MODE=workspace
fi

if [[ -z "${T3_ADAPTER_RUNTIME_ROOT:-}" ]]; then
  if [[ "${TASK_ADAPTER_USED_NATIVE}" == true ]]; then
    export T3_ADAPTER_RUNTIME_ROOT="${TASK_ADAPTER_NATIVE_ROOT}/t1_t2_adapters"
  else
    export T3_ADAPTER_RUNTIME_ROOT="${T3_ADAPTER_WORKSPACE_ROOT}/offline_deploy/runtime/t1_t2_adapters"
  fi
fi

TASK_ADAPTER_LOCAL_SETUP="${T3_ADAPTER_RUNTIME_ROOT}/install/local_setup.bash"
TASK_ADAPTER_OVERLAY_STATUS=not-built
if [[ "${T3_ADAPTER_SKIP_OVERLAY:-0}" != "1" && \
      -f "${TASK_ADAPTER_LOCAL_SETUP}" ]]; then
  source "${TASK_ADAPTER_LOCAL_SETUP}"
  TASK_ADAPTER_OVERLAY_STATUS=loaded
elif [[ -f "${TASK_ADAPTER_LOCAL_SETUP}" ]]; then
  TASK_ADAPTER_OVERLAY_STATUS=skipped-for-build
fi

echo "[adapter] workspace=${T3_ADAPTER_WORKSPACE_ROOT}"
echo "[adapter] runtime=${T3_ADAPTER_RUNTIME_ROOT}"
echo "[adapter] overlay=${TASK_ADAPTER_OVERLAY_STATUS}"
