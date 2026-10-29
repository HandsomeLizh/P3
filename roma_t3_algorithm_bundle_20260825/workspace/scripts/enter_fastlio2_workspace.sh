#!/usr/bin/env bash
# Source this file; do not execute it in a child shell.

set +u

TASK_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TASK_WORKSPACE_ROOT="$(cd "${TASK_SCRIPT_DIR}/.." && pwd)"
TASK_BUNDLED_FASTLIO_WS="${TASK_WORKSPACE_ROOT}/third_party/fastlio2"
FASTLIO_WS="${FASTLIO_WS:-${TASK_BUNDLED_FASTLIO_WS}}"
TASK_ROS_DISTRO="${ROS_DISTRO:-}"
if [[ -z "${TASK_ROS_DISTRO}" ]]; then
  if [[ -n "${ROS_SETUP_FILE:-}" ]]; then
    TASK_ROS_DISTRO=humble
  elif [[ -f /opt/ros/humble/setup.bash ]]; then
    TASK_ROS_DISTRO=humble
  elif [[ -f /opt/ros/jazzy/setup.bash ]]; then
    TASK_ROS_DISTRO=jazzy
  elif [[ -f /opt/ros/rolling/setup.bash ]]; then
    TASK_ROS_DISTRO=rolling
  else
    echo "[ERROR] No ROS 2 setup.bash was found under /opt/ros." >&2
    return 1 2>/dev/null || exit 1
  fi
fi

TASK_ROS_SETUP="${ROS_SETUP_FILE:-/opt/ros/${TASK_ROS_DISTRO}/setup.bash}"
if [[ ! -f "${TASK_ROS_SETUP}" ]]; then
  echo "[ERROR] ROS setup file not found: ${TASK_ROS_SETUP}" >&2
  return 1 2>/dev/null || exit 1
fi

unset AMENT_PREFIX_PATH || true
unset CMAKE_PREFIX_PATH || true
unset COLCON_PREFIX_PATH || true
source "${TASK_ROS_SETUP}"

TASK_LOCAL_PCL_SETUP="${FASTLIO_WS}/rolling_overlay/opt/ros/${TASK_ROS_DISTRO}/share/pcl_ros/local_setup.bash"
if ! ros2 pkg prefix pcl_ros >/dev/null 2>&1 && [[ -f "${TASK_LOCAL_PCL_SETUP}" ]]; then
  source "${TASK_LOCAL_PCL_SETUP}"
fi

TASK_LOCAL_PREFIX="${FASTLIO_LOCAL_PREFIX:-${FASTLIO_WS}/local}"
export LD_LIBRARY_PATH="${TASK_LOCAL_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export CMAKE_PREFIX_PATH="${TASK_LOCAL_PREFIX}:${CMAKE_PREFIX_PATH:-}"

fastlio_install_complete() {
  local candidate="$1"
  [[ -f "${candidate}/setup.bash" ]] &&
    [[ -x "${candidate}/fast_lio/lib/fast_lio/fastlio_mapping" ]] &&
    [[ -e "${candidate}/livox_ros_driver2/lib/liblivox_ros_driver2.so" ]]
}

TASK_INSTALL_BASE="${FASTLIO_INSTALL_BASE:-${FASTLIO_WS}/install}"
if ! fastlio_install_complete "${TASK_INSTALL_BASE}"; then
  if [[ -n "${FASTLIO_INSTALL_BASE:-}" ]]; then
    echo "[ERROR] FASTLIO_INSTALL_BASE is incomplete: ${TASK_INSTALL_BASE}" >&2
    return 1 2>/dev/null || exit 1
  fi

  TASK_RECORDED_INSTALL_BASE=""
  if [[ -f "${FASTLIO_WS}/FASTLIO2_BUILD_INFO.txt" ]]; then
    TASK_RECORDED_INSTALL_BASE="$(
      sed -n 's/^INSTALL_BASE=//p' \
        "${FASTLIO_WS}/FASTLIO2_BUILD_INFO.txt" | tail -n 1
    )"
  fi
  if [[ -n "${TASK_RECORDED_INSTALL_BASE}" ]] && \
     fastlio_install_complete "${TASK_RECORDED_INSTALL_BASE}"; then
    TASK_INSTALL_BASE="${TASK_RECORDED_INSTALL_BASE}"
  elif fastlio_install_complete "${FASTLIO_WS}/install_cmake316"; then
    # Compatibility result used only by the current old Rolling host.
    TASK_INSTALL_BASE="${FASTLIO_WS}/install_cmake316"
  else
    echo "[ERROR] A complete FAST-LIO2 install was not found under ${FASTLIO_WS}." >&2
    return 1 2>/dev/null || exit 1
  fi
fi

TASK_CURRENT_ARCH="$(uname -m)"
TASK_RECORDED_ARCH=""
if [[ -f "${FASTLIO_WS}/FASTLIO2_BUILD_INFO.txt" ]]; then
  TASK_RECORDED_ARCH="$(
    sed -n 's/^ARCHITECTURE=//p' \
      "${FASTLIO_WS}/FASTLIO2_BUILD_INFO.txt" | tail -n 1
  )"
fi
if [[ -n "${TASK_RECORDED_ARCH}" && \
      "${TASK_RECORDED_ARCH}" != "${TASK_CURRENT_ARCH}" ]]; then
  echo "[ERROR] FAST-LIO2 was built for ${TASK_RECORDED_ARCH}," >&2
  echo "        but this machine is ${TASK_CURRENT_ARCH}. Rebuild it natively." >&2
  return 1 2>/dev/null || exit 1
fi

TASK_FASTLIO_BINARY="${TASK_INSTALL_BASE}/fast_lio/lib/fast_lio/fastlio_mapping"
if command -v file >/dev/null 2>&1; then
  TASK_BINARY_DESCRIPTION="$(file -Lb "${TASK_FASTLIO_BINARY}")"
  if [[ "${TASK_CURRENT_ARCH}" == "aarch64" && \
        "${TASK_BINARY_DESCRIPTION}" == *"x86-64"* ]]; then
    echo "[ERROR] The selected FAST-LIO2 executable is x86-64, not aarch64." >&2
    echo "        Do not copy x86 build/install directories to the Orin." >&2
    return 1 2>/dev/null || exit 1
  fi
  if [[ "${TASK_CURRENT_ARCH}" == "x86_64" && \
        "${TASK_BINARY_DESCRIPTION}" == *"ARM aarch64"* ]]; then
    echo "[ERROR] The selected FAST-LIO2 executable is aarch64, not x86-64." >&2
    return 1 2>/dev/null || exit 1
  fi
fi

source "${TASK_INSTALL_BASE}/setup.bash"
export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_cyclonedds_cpp}"

if ! ros2 pkg prefix fast_lio >/dev/null 2>&1 || \
   ! ros2 pkg prefix livox_ros_driver2 >/dev/null 2>&1; then
  echo "[ERROR] FAST-LIO2 packages are not discoverable after loading ${TASK_INSTALL_BASE}." >&2
  return 1 2>/dev/null || exit 1
fi

echo "[fastlio2] FASTLIO_WS=${FASTLIO_WS}"
echo "[fastlio2] ROS_DISTRO=${ROS_DISTRO:-unknown}"
echo "[fastlio2] architecture=${TASK_CURRENT_ARCH}"
echo "[fastlio2] install=${TASK_INSTALL_BASE}"
echo "[fastlio2] fast_lio=$(ros2 pkg prefix fast_lio)"
echo "[fastlio2] livox_ros_driver2=$(ros2 pkg prefix livox_ros_driver2)"
