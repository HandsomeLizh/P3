#!/usr/bin/env bash
# Source this file: source scripts/enter_workspace.sh

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "[ERROR] Use: source $0" >&2
  exit 1
fi

set +u

T3_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export T3_WS="$(cd "${T3_SCRIPT_DIR}/.." && pwd)"
export PRACTICE_WS="${T3_WS}"
export ROS_SETUP_FILE="${ROS_SETUP_FILE:-/opt/ros/humble/setup.bash}"
export ROS_PYTHON_EXECUTABLE=/usr/bin/python3

if [[ ! -f "${ROS_SETUP_FILE}" ]]; then
  echo "[ERROR] ROS setup not found: ${ROS_SETUP_FILE}" >&2
  return 1
fi
if [[ ! -x "${ROS_PYTHON_EXECUTABLE}" ]]; then
  echo "[ERROR] Python not found: ${ROS_PYTHON_EXECUTABLE}" >&2
  return 1
fi

export T3_ALGORITHM_ROOT="${T3_ALGORITHM_ROOT:-${T3_WS}/third_party/RoMa}"
if [[ ! -f "${T3_ALGORITHM_ROOT}/roma_vo.py" ]]; then
  echo "[ERROR] RoMa source not found: ${T3_ALGORITHM_ROOT}/roma_vo.py" >&2
  return 1
fi

export TORCH_HOME="${TORCH_HOME:-${T3_WS}/models/torch}"
export T3_OUTPUT_ROOT="${T3_OUTPUT_ROOT:-${HOME}/P3/output}"
export T3_OUTPUT_DIR="${T3_OUTPUT_DIR:-${T3_OUTPUT_ROOT}}"
mkdir -p "${T3_OUTPUT_ROOT}"
export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_cyclonedds_cpp}"
export PYTHONUNBUFFERED=1

case ":${PYTHONPATH:-}:" in
  *":${T3_ALGORITHM_ROOT}:"*) ;;
  *) export PYTHONPATH="${T3_ALGORITHM_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" ;;
esac

source "${ROS_SETUP_FILE}"
if [[ -f "${T3_WS}/install/setup.bash" ]]; then
  source "${T3_WS}/install/setup.bash"
else
  echo "[WARN] Workspace is not built yet; run: bash scripts/build.sh" >&2
fi

echo "[t3] workspace=${T3_WS}"
echo "[t3] python=${ROS_PYTHON_EXECUTABLE}"
echo "[t3] roma=${T3_ALGORITHM_ROOT}"
echo "[t3] output_root=${T3_OUTPUT_ROOT}"
echo "[t3] mode=T5 RoMa VO (LiDAR off, IMU off)"

unset T3_SCRIPT_DIR
