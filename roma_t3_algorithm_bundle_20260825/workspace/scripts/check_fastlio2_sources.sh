#!/usr/bin/env bash
set -e
set -o pipefail
set +u

TASK_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TASK_WORKSPACE_ROOT="$(cd "${TASK_SCRIPT_DIR}/.." && pwd)"
TASK_FASTLIO_WS="${1:-${FASTLIO_WS:-${TASK_WORKSPACE_ROOT}/third_party/fastlio2}}"
TASK_SOURCE_DIR="${TASK_FASTLIO_WS}/src"
TASK_SOURCE_INFO="${TASK_FASTLIO_WS}/FASTLIO2_SOURCE_INFO.txt"
TASK_ERRORS=0

repository_has_git_metadata() {
  local repository_dir="$1"
  [[ -d "${repository_dir}/.git" || -f "${repository_dir}/.git" ]]
}

check_portable_snapshot() {
  local required_path
  local required_paths=(
    "${TASK_SOURCE_INFO}"
    "${TASK_SOURCE_DIR}/Livox-SDK2/CMakeLists.txt"
    "${TASK_SOURCE_DIR}/livox_ros_driver2/CMakeLists.txt"
    "${TASK_SOURCE_DIR}/livox_ros_driver2/package_ROS2.xml"
    "${TASK_SOURCE_DIR}/FAST_LIO_ROS2/package.xml"
    "${TASK_SOURCE_DIR}/FAST_LIO_ROS2/src/laserMapping.cpp"
    "${TASK_SOURCE_DIR}/FAST_LIO_ROS2/include/ikd-Tree/ikd_Tree.h"
    "${TASK_SOURCE_DIR}/FAST_LIO_ROS2/include/ikd-Tree/ikd_Tree.cpp"
  )

  if ! grep -qx 'TYPE=portable-source-only' "${TASK_SOURCE_INFO}" 2>/dev/null; then
    echo "[ERROR] Portable FAST-LIO2 source marker is missing: ${TASK_SOURCE_INFO}" >&2
    exit 1
  fi
  for required_path in "${required_paths[@]}"; do
    if [[ ! -f "${required_path}" ]]; then
      echo "[ERROR] Portable FAST-LIO2 source is incomplete: ${required_path}" >&2
      TASK_ERRORS=$((TASK_ERRORS + 1))
    fi
  done
  if (( TASK_ERRORS > 0 )); then
    echo "[ERROR] Portable FAST-LIO2 source verification failed (${TASK_ERRORS})." >&2
    exit 1
  fi
  echo "[OK] Portable FAST-LIO2 source snapshot is complete: ${TASK_FASTLIO_WS}"
}

TASK_PORTABLE_SNAPSHOT=false
if [[ -f "${TASK_SOURCE_INFO}" ]] && \
   ! repository_has_git_metadata "${TASK_SOURCE_DIR}/Livox-SDK2" && \
   ! repository_has_git_metadata "${TASK_SOURCE_DIR}/livox_ros_driver2" && \
   ! repository_has_git_metadata "${TASK_SOURCE_DIR}/FAST_LIO_ROS2"; then
  TASK_PORTABLE_SNAPSHOT=true
fi

if [[ "${TASK_PORTABLE_SNAPSHOT}" == true ]]; then
  check_portable_snapshot
  exit 0
fi

check_repository() {
  local repository_name="$1"
  local expected_commit="$2"
  local repository_dir="${TASK_SOURCE_DIR}/${repository_name}"
  if [[ ! -d "${repository_dir}/.git" && ! -f "${repository_dir}/.git" ]]; then
    echo "[ERROR] Git checkout is missing: ${repository_dir}" >&2
    TASK_ERRORS=$((TASK_ERRORS + 1))
    return
  fi

  local actual_commit
  actual_commit="$(git -C "${repository_dir}" rev-parse HEAD 2>/dev/null || true)"
  if [[ "${actual_commit}" != "${expected_commit}" ]]; then
    echo "[ERROR] ${repository_name}: expected ${expected_commit}, got ${actual_commit:-unknown}" >&2
    TASK_ERRORS=$((TASK_ERRORS + 1))
    return
  fi

  local dirty_lines
  dirty_lines="$(
    git -C "${repository_dir}" status --porcelain 2>/dev/null |
      sed '/^?? COLCON_IGNORE$/d'
  )"
  if [[ -n "${dirty_lines}" ]]; then
    echo "[ERROR] ${repository_name} contains uncommitted source changes:" >&2
    echo "${dirty_lines}" >&2
    TASK_ERRORS=$((TASK_ERRORS + 1))
    return
  fi
  echo "[OK] ${repository_name}=${actual_commit}"
}

if ! command -v git >/dev/null 2>&1; then
  echo "[ERROR] git is required to verify the bundled sources." >&2
  exit 1
fi

check_repository Livox-SDK2 08f523c930b2f0ba1e98a6afaa8d7476bf479908
check_repository livox_ros_driver2 4a1def929e5b59c7a8122d19fce6efba581ce9f7
check_repository FAST_LIO_ROS2 2fffc570a25d0df172720bac034fbdb6a13d2162
check_repository \
  FAST_LIO_ROS2/include/ikd-Tree \
  e2e3f4e9d3b95a9e66b1ba83dc98d4a05ed8a3c4

if (( TASK_ERRORS > 0 )); then
  echo "[ERROR] Bundled FAST-LIO2 source verification failed (${TASK_ERRORS})." >&2
  exit 1
fi

echo "[OK] Bundled FAST-LIO2 sources are complete and pinned: ${TASK_FASTLIO_WS}"
