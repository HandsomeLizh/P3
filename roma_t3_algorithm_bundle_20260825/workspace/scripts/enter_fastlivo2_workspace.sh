#!/usr/bin/env bash
# Source this file; do not execute it in a child shell.

set +u

TASK_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TASK_WORKSPACE_ROOT="$(cd "${TASK_SCRIPT_DIR}/.." && pwd)"
TASK_DEPENDENCY_ROOT="${TASK_WORKSPACE_ROOT}/../t3_gridmap_v2_ws_dependencies/runtime_sources/third_party"
TASK_BUNDLED_FASTLIVO_WS="${TASK_WORKSPACE_ROOT}/third_party/fastlivo2"
if [[ ! -d "${TASK_BUNDLED_FASTLIVO_WS}" ]]; then
  TASK_BUNDLED_FASTLIVO_WS="${TASK_DEPENDENCY_ROOT}/fastlivo2"
fi
FASTLIVO_WS="${FASTLIVO_WS:-${TASK_BUNDLED_FASTLIVO_WS}}"

TASK_BUNDLED_FASTLIO_WS="${TASK_WORKSPACE_ROOT}/third_party/fastlio2"
if [[ ! -d "${TASK_BUNDLED_FASTLIO_WS}" ]]; then
  TASK_BUNDLED_FASTLIO_WS="${TASK_DEPENDENCY_ROOT}/fastlio2"
fi
FASTLIO_WS="${FASTLIO_WS:-${TASK_BUNDLED_FASTLIO_WS}}"

TASK_ROS_SETUP="${ROS_SETUP_FILE:-/opt/ros/humble/setup.bash}"
if [[ ! -f "${TASK_ROS_SETUP}" ]]; then
  echo "[ERROR] ROS setup file not found: ${TASK_ROS_SETUP}" >&2
  return 1 2>/dev/null || exit 1
fi

unset AMENT_PREFIX_PATH || true
unset CMAKE_PREFIX_PATH || true
unset COLCON_PREFIX_PATH || true
source "${TASK_ROS_SETUP}"

TASK_PCL_SETUP="${FASTLIO_WS}/rolling_overlay/opt/ros/humble/share/pcl_ros/local_setup.bash"
if ! ros2 pkg prefix pcl_ros >/dev/null 2>&1 && [[ -f "${TASK_PCL_SETUP}" ]]; then
  source "${TASK_PCL_SETUP}"
fi

# FAST-LIVO2's Ouster path still compiles against the Livox message package;
# reuse the already pinned FAST-LIO2 dependency overlay without launching it.
if [[ ! -f "${FASTLIO_WS}/install/setup.bash" ]]; then
  echo "[ERROR] FAST-LIO2 dependency overlay is missing: ${FASTLIO_WS}/install" >&2
  return 1 2>/dev/null || exit 1
fi
source "${FASTLIO_WS}/install/setup.bash"

TASK_FASTLIVO_INSTALL="${FASTLIVO_INSTALL_BASE:-${FASTLIVO_WS}/install}"
TASK_FASTLIVO_BINARY="${TASK_FASTLIVO_INSTALL}/lib/fast_livo/fastlivo_mapping"
if [[ ! -f "${TASK_FASTLIVO_INSTALL}/setup.bash" || ! -x "${TASK_FASTLIVO_BINARY}" ]]; then
  echo "[ERROR] FAST-LIVO2 install is incomplete: ${TASK_FASTLIVO_INSTALL}" >&2
  return 1 2>/dev/null || exit 1
fi

export LD_LIBRARY_PATH="${FASTLIVO_WS}/local/lib:${TASK_FASTLIVO_INSTALL}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
source "${TASK_FASTLIVO_INSTALL}/setup.bash"
if [[ -f "${TASK_WORKSPACE_ROOT}/install/setup.bash" ]]; then
  source "${TASK_WORKSPACE_ROOT}/install/setup.bash"
fi
export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_cyclonedds_cpp}"

if ! ros2 pkg prefix fast_livo >/dev/null 2>&1; then
  echo "[ERROR] fast_livo is not discoverable after loading ${TASK_FASTLIVO_INSTALL}." >&2
  return 1 2>/dev/null || exit 1
fi

echo "[fastlivo2] workspace=${FASTLIVO_WS}"
echo "[fastlivo2] install=${TASK_FASTLIVO_INSTALL}"
echo "[fastlivo2] executable=${TASK_FASTLIVO_BINARY}"

unset TASK_SCRIPT_DIR TASK_DEPENDENCY_ROOT TASK_BUNDLED_FASTLIVO_WS
unset TASK_BUNDLED_FASTLIO_WS TASK_ROS_SETUP TASK_PCL_SETUP
unset TASK_FASTLIVO_INSTALL TASK_FASTLIVO_BINARY
