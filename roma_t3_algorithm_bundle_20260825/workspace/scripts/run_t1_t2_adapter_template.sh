#!/usr/bin/env bash
set -e
set -o pipefail
set +u

TASK_ADAPTER_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TASK_ADAPTER_WORKSPACE="$(cd "${TASK_ADAPTER_SCRIPT_DIR}/.." && pwd)"
TASK_ADAPTER_CONFIG="${TASK_ADAPTER_WORKSPACE}/src/t3_upstream_adapters/config/interface_mapping.yaml"
TASK_ADAPTER_LAUNCH_FILE="${T3_ADAPTER_LAUNCH_FILE:-integrated_t1_t2.launch.py}"
TASK_UPSTREAM_PROFILE="${T3_UPSTREAM_PROFILE:-t1_t2}"

TASK_ADAPTER_USER_ARGUMENTS=()
while (( $# > 0 )); do
  case "$1" in
    --profile)
      if (( $# < 2 )); then
        echo "[ERROR] --profile requires standard, t1, t2 or t1_t2." >&2
        exit 2
      fi
      TASK_UPSTREAM_PROFILE="$2"
      shift 2
      ;;
    --profile=*)
      TASK_UPSTREAM_PROFILE="${1#--profile=}"
      shift
      ;;
    *)
      TASK_ADAPTER_USER_ARGUMENTS+=("$1")
      shift
      ;;
  esac
done

case "${TASK_UPSTREAM_PROFILE}" in
  standard)
    TASK_DEFAULT_ENABLE_T1=false
    TASK_DEFAULT_ENABLE_T2=false
    ;;
  t1)
    TASK_DEFAULT_ENABLE_T1=true
    TASK_DEFAULT_ENABLE_T2=false
    ;;
  t2)
    TASK_DEFAULT_ENABLE_T1=false
    TASK_DEFAULT_ENABLE_T2=true
    ;;
  t1_t2)
    TASK_DEFAULT_ENABLE_T1=true
    TASK_DEFAULT_ENABLE_T2=true
    ;;
  *)
    echo "[ERROR] Unknown upstream profile: ${TASK_UPSTREAM_PROFILE}" >&2
    echo "        Choose standard, t1, t2 or t1_t2." >&2
    exit 2
    ;;
esac

source "${TASK_ADAPTER_SCRIPT_DIR}/enter_t1_t2_adapter_environment.sh"

case "${TASK_ADAPTER_LAUNCH_FILE}" in
  integrated_t1_t2.launch.py|adapters_only.launch.py)
    ;;
  *)
    echo "[ERROR] T3_ADAPTER_LAUNCH_FILE must be integrated_t1_t2.launch.py" >&2
    echo "        or adapters_only.launch.py." >&2
    exit 2
    ;;
esac

if ! ros2 pkg prefix t3_upstream_adapters >/dev/null 2>&1; then
  echo "[ERROR] The adapter is not built in this environment." >&2
  echo "        Run: bash ${TASK_ADAPTER_SCRIPT_DIR}/build_t1_t2_adapter_template.sh" >&2
  exit 1
fi
adapter_section_has_placeholder() {
  local section="$1"
  awk -v wanted="${section}:" '
    $0 == wanted { inside = 1; next }
    inside && /^[^[:space:]#][^:]*:/ { inside = 0 }
    inside { print }
  ' "${TASK_ADAPTER_EFFECTIVE_CONFIG}" | grep -q 'CHANGE_ME_'
}

TASK_ADAPTER_HAS_PARAM_FILE=false
TASK_ADAPTER_HAS_T1_SWITCH=false
TASK_ADAPTER_HAS_T2_SWITCH=false
TASK_ADAPTER_EFFECTIVE_CONFIG="${TASK_ADAPTER_CONFIG}"
TASK_EFFECTIVE_ENABLE_T1="${TASK_DEFAULT_ENABLE_T1}"
TASK_EFFECTIVE_ENABLE_T2="${TASK_DEFAULT_ENABLE_T2}"
for TASK_ADAPTER_ARGUMENT in "${TASK_ADAPTER_USER_ARGUMENTS[@]}"; do
  if [[ "${TASK_ADAPTER_ARGUMENT}" == adapter_params_file:=* ]]; then
    TASK_ADAPTER_HAS_PARAM_FILE=true
    TASK_ADAPTER_EFFECTIVE_CONFIG="${TASK_ADAPTER_ARGUMENT#adapter_params_file:=}"
  elif [[ "${TASK_ADAPTER_ARGUMENT}" == enable_t1_adapter:=* ]]; then
    TASK_ADAPTER_HAS_T1_SWITCH=true
    TASK_EFFECTIVE_ENABLE_T1="${TASK_ADAPTER_ARGUMENT#enable_t1_adapter:=}"
  elif [[ "${TASK_ADAPTER_ARGUMENT}" == enable_t2_adapter:=* ]]; then
    TASK_ADAPTER_HAS_T2_SWITCH=true
    TASK_EFFECTIVE_ENABLE_T2="${TASK_ADAPTER_ARGUMENT#enable_t2_adapter:=}"
  fi
done
for TASK_ADAPTER_SWITCH in \
  "${TASK_EFFECTIVE_ENABLE_T1}" "${TASK_EFFECTIVE_ENABLE_T2}"; do
  if [[ "${TASK_ADAPTER_SWITCH}" != true && \
        "${TASK_ADAPTER_SWITCH}" != false ]]; then
    echo "[ERROR] Adapter enable switches must be true or false." >&2
    exit 2
  fi
done
if [[ ! -f "${TASK_ADAPTER_EFFECTIVE_CONFIG}" ]]; then
  echo "[ERROR] Adapter parameter file is missing:" >&2
  echo "        ${TASK_ADAPTER_EFFECTIVE_CONFIG}" >&2
  exit 1
fi
TASK_ADAPTER_ARGUMENTS=("${TASK_ADAPTER_USER_ARGUMENTS[@]}")
if [[ "${TASK_ADAPTER_HAS_PARAM_FILE}" == false ]]; then
  TASK_ADAPTER_ARGUMENTS=(
    "adapter_params_file:=${TASK_ADAPTER_CONFIG}"
    "${TASK_ADAPTER_ARGUMENTS[@]}"
  )
fi
if [[ "${TASK_ADAPTER_HAS_T1_SWITCH}" == false ]]; then
  TASK_ADAPTER_ARGUMENTS+=(
    "enable_t1_adapter:=${TASK_DEFAULT_ENABLE_T1}"
  )
fi
if [[ "${TASK_ADAPTER_HAS_T2_SWITCH}" == false ]]; then
  TASK_ADAPTER_ARGUMENTS+=(
    "enable_t2_adapter:=${TASK_DEFAULT_ENABLE_T2}"
  )
fi

if [[ "${TASK_EFFECTIVE_ENABLE_T1}" == true ]] && \
   adapter_section_has_placeholder t1_fusion_adapter; then
  echo "[WARN] Enabled Task 1 adapter still has CHANGE_ME fields in the YAML." >&2
fi
if [[ "${TASK_EFFECTIVE_ENABLE_T2}" == true ]] && \
   adapter_section_has_placeholder t2_scene_adapter; then
  echo "[WARN] Enabled Task 2 adapter still has CHANGE_ME fields in the YAML." >&2
fi

echo "[adapter] upstream profile=${TASK_UPSTREAM_PROFILE} "\
"(t1=${TASK_EFFECTIVE_ENABLE_T1}, t2=${TASK_EFFECTIVE_ENABLE_T2})"

exec ros2 launch t3_upstream_adapters \
  "${TASK_ADAPTER_LAUNCH_FILE}" "${TASK_ADAPTER_ARGUMENTS[@]}"
