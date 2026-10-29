#!/usr/bin/env bash
set -e
set -o pipefail
set +u

# A reproducible, process-isolated FAST-LIO2 workspace.  The SDK is installed
# below the workspace, so this script does not write to /usr/local.
TASK_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TASK_WORKSPACE_ROOT="$(cd "${TASK_SCRIPT_DIR}/.." && pwd)"
TASK_BUNDLED_FASTLIO_WS="${TASK_WORKSPACE_ROOT}/third_party/fastlio2"
FASTLIO_WS="${1:-${FASTLIO_WS:-${TASK_BUNDLED_FASTLIO_WS}}}"
TASK_SOURCE_DIR="${FASTLIO_WS}/src"
TASK_SOURCE_INFO="${FASTLIO_WS}/FASTLIO2_SOURCE_INFO.txt"
TASK_LOCAL_PREFIX="${FASTLIO_LOCAL_PREFIX:-${FASTLIO_WS}/local}"
TASK_SDK_BUILD_BASE="${FASTLIO_SDK_BUILD_BASE:-${FASTLIO_WS}/build_sdk2}"
TASK_BUILD_BASE="${FASTLIO_BUILD_BASE:-${FASTLIO_WS}/build}"
TASK_INSTALL_BASE="${FASTLIO_INSTALL_BASE:-${FASTLIO_WS}/install}"
TASK_LOG_BASE="${FASTLIO_LOG_BASE:-${FASTLIO_WS}/log}"
TASK_BUILD_INFO="${FASTLIO_BUILD_INFO_PATH:-${FASTLIO_WS}/FASTLIO2_BUILD_INFO.txt}"
TASK_OFFLINE_ONLY="${FASTLIO_OFFLINE:-0}"
mkdir -p "$(dirname "${TASK_BUILD_INFO}")"
if [[ -n "${FASTLIO_BUILD_JOBS:-}" ]]; then
  TASK_BUILD_JOBS="${FASTLIO_BUILD_JOBS}"
else
  TASK_BUILD_JOBS="$(nproc)"
  if (( TASK_BUILD_JOBS > 8 )); then
    TASK_BUILD_JOBS=8
  fi
fi
if [[ ! "${TASK_BUILD_JOBS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "[ERROR] FASTLIO_BUILD_JOBS must be a positive integer." >&2
  exit 1
fi
if [[ "${TASK_OFFLINE_ONLY}" != "0" && "${TASK_OFFLINE_ONLY}" != "1" ]]; then
  echo "[ERROR] FASTLIO_OFFLINE must be 0 or 1." >&2
  exit 1
fi

repository_has_git_metadata() {
  local repository_dir="$1"
  [[ -d "${repository_dir}/.git" || -f "${repository_dir}/.git" ]]
}

TASK_PORTABLE_SNAPSHOT=false
if [[ -f "${TASK_SOURCE_INFO}" ]] && \
   grep -qx 'TYPE=portable-source-only' "${TASK_SOURCE_INFO}" && \
   ! repository_has_git_metadata "${TASK_SOURCE_DIR}/Livox-SDK2" && \
   ! repository_has_git_metadata "${TASK_SOURCE_DIR}/livox_ros_driver2" && \
   ! repository_has_git_metadata "${TASK_SOURCE_DIR}/FAST_LIO_ROS2"; then
  TASK_PORTABLE_SNAPSHOT=true
fi

if [[ "${TASK_PORTABLE_SNAPSHOT}" == true ]]; then
  for TASK_REQUIRED_SOURCE in \
    "${TASK_SOURCE_DIR}/Livox-SDK2/CMakeLists.txt" \
    "${TASK_SOURCE_DIR}/livox_ros_driver2/CMakeLists.txt" \
    "${TASK_SOURCE_DIR}/livox_ros_driver2/package_ROS2.xml" \
    "${TASK_SOURCE_DIR}/FAST_LIO_ROS2/package.xml" \
    "${TASK_SOURCE_DIR}/FAST_LIO_ROS2/src/laserMapping.cpp" \
    "${TASK_SOURCE_DIR}/FAST_LIO_ROS2/include/ikd-Tree/ikd_Tree.h" \
    "${TASK_SOURCE_DIR}/FAST_LIO_ROS2/include/ikd-Tree/ikd_Tree.cpp"; do
    if [[ ! -f "${TASK_REQUIRED_SOURCE}" ]]; then
      echo "[ERROR] Portable FAST-LIO2 source is incomplete: ${TASK_REQUIRED_SOURCE}" >&2
      exit 1
    fi
  done
  echo "[setup] Using the bundled portable FAST-LIO2 source snapshot."
fi

LIVOX_SDK2_URL="https://github.com/Livox-SDK/Livox-SDK2.git"
LIVOX_SDK2_COMMIT="08f523c930b2f0ba1e98a6afaa8d7476bf479908"
LIVOX_DRIVER_URL="https://github.com/Livox-SDK/livox_ros_driver2.git"
LIVOX_DRIVER_COMMIT="4a1def929e5b59c7a8122d19fce6efba581ce9f7"
FASTLIO_URL="https://github.com/Ericsii/FAST_LIO_ROS2.git"
FASTLIO_COMMIT="2fffc570a25d0df172720bac034fbdb6a13d2162"
IKD_TREE_COMMIT="e2e3f4e9d3b95a9e66b1ba83dc98d4a05ed8a3c4"

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
    echo "[ERROR] No ROS 2 installation was found under /opt/ros." >&2
    exit 1
  fi
fi
TASK_ROS_SETUP="${ROS_SETUP_FILE:-/opt/ros/${TASK_ROS_DISTRO}/setup.bash}"
if [[ ! -f "${TASK_ROS_SETUP}" ]]; then
  echo "[ERROR] ROS setup file not found: ${TASK_ROS_SETUP}" >&2
  exit 1
fi

TASK_CMAKE_BIN="${FASTLIO_CMAKE_BIN:-$(command -v cmake || true)}"
if [[ -z "${TASK_CMAKE_BIN}" || ! -x "${TASK_CMAKE_BIN}" ]]; then
  echo "[ERROR] cmake was not found. Run install_fastlio2_prereqs.sh first." >&2
  exit 1
fi
TASK_CMAKE_BIN="$(readlink -f "${TASK_CMAKE_BIN}")"
export PATH="$(dirname "${TASK_CMAKE_BIN}"):/usr/bin:/bin:${PATH}"

for command_name in git colcon nproc; do
  if ! command -v "${command_name}" >/dev/null 2>&1; then
    echo "[ERROR] Required command is missing: ${command_name}" >&2
    exit 1
  fi
done

mkdir -p "${TASK_SOURCE_DIR}"

checkout_pinned_repo() {
  local repository_name="$1"
  local repository_url="$2"
  local repository_commit="$3"
  local recursive_mode="$4"
  local repository_dir="${TASK_SOURCE_DIR}/${repository_name}"

  if [[ "${TASK_PORTABLE_SNAPSHOT}" == true ]]; then
    if [[ ! -d "${repository_dir}" ]]; then
      echo "[ERROR] Portable FAST-LIO2 source is missing: ${repository_dir}" >&2
      exit 1
    fi
    echo "[setup] Portable source: ${repository_name}"
    return 0
  fi

  if [[ ! -d "${repository_dir}/.git" ]]; then
    if [[ "${TASK_OFFLINE_ONLY}" == "1" ]]; then
      echo "[ERROR] Offline FAST-LIO2 source is missing: ${repository_dir}" >&2
      exit 1
    fi
    if [[ "${recursive_mode}" == "recursive" ]]; then
      git clone --recursive "${repository_url}" "${repository_dir}"
    else
      git clone "${repository_url}" "${repository_dir}"
    fi
  else
    local dirty_lines
    dirty_lines="$(git -C "${repository_dir}" status --porcelain | sed '/^?? COLCON_IGNORE$/d')"
    if [[ -n "${dirty_lines}" ]]; then
      echo "[ERROR] Refusing to change a modified dependency checkout:" >&2
      echo "        ${repository_dir}" >&2
      echo "${dirty_lines}" >&2
      exit 1
    fi
    if ! git -C "${repository_dir}" cat-file -e "${repository_commit}^{commit}" 2>/dev/null; then
      if [[ "${TASK_OFFLINE_ONLY}" == "1" ]]; then
        echo "[ERROR] Pinned commit is absent from the offline checkout: ${repository_name} ${repository_commit}" >&2
        exit 1
      fi
      git -C "${repository_dir}" fetch --tags origin
    fi
  fi

  git -C "${repository_dir}" checkout --detach "${repository_commit}"
  if [[ "${recursive_mode}" == "recursive" ]]; then
    git -C "${repository_dir}" submodule sync --recursive
    if [[ "${TASK_OFFLINE_ONLY}" == "1" ]]; then
      if git -C "${repository_dir}" submodule status --recursive | grep -Eq '^[+-U]'; then
        echo "[ERROR] Offline submodule is missing or at the wrong revision: ${repository_name}" >&2
        exit 1
      fi
    else
      git -C "${repository_dir}" submodule update --init --recursive
    fi
  fi
}

checkout_pinned_repo Livox-SDK2 "${LIVOX_SDK2_URL}" "${LIVOX_SDK2_COMMIT}" normal
checkout_pinned_repo livox_ros_driver2 "${LIVOX_DRIVER_URL}" "${LIVOX_DRIVER_COMMIT}" normal
checkout_pinned_repo FAST_LIO_ROS2 "${FASTLIO_URL}" "${FASTLIO_COMMIT}" recursive

if [[ "${TASK_PORTABLE_SNAPSHOT}" != true ]]; then
  TASK_ACTUAL_IKD_COMMIT="$(git -C "${TASK_SOURCE_DIR}/FAST_LIO_ROS2/include/ikd-Tree" rev-parse HEAD)"
  if [[ "${TASK_ACTUAL_IKD_COMMIT}" != "${IKD_TREE_COMMIT}" ]]; then
    echo "[ERROR] Unexpected ikd-Tree revision: ${TASK_ACTUAL_IKD_COMMIT}" >&2
    exit 1
  fi
fi

# Livox-SDK2 is a plain CMake project rather than a ROS package.
touch "${TASK_SOURCE_DIR}/Livox-SDK2/COLCON_IGNORE"
"${TASK_CMAKE_BIN}" \
  -S "${TASK_SOURCE_DIR}/Livox-SDK2" \
  -B "${TASK_SDK_BUILD_BASE}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="${TASK_LOCAL_PREFIX}"
"${TASK_CMAKE_BIN}" --build "${TASK_SDK_BUILD_BASE}" --parallel "${TASK_BUILD_JOBS}"
"${TASK_CMAKE_BIN}" --install "${TASK_SDK_BUILD_BASE}"
touch "${TASK_SDK_BUILD_BASE}/COLCON_IGNORE"
touch "${TASK_LOCAL_PREFIX}/COLCON_IGNORE"

# The upstream driver selects its ROS manifest at build time.  package.xml is
# intentionally ignored by that repository, so copying it does not dirty git.
cp -f \
  "${TASK_SOURCE_DIR}/livox_ros_driver2/package_ROS2.xml" \
  "${TASK_SOURCE_DIR}/livox_ros_driver2/package.xml"

unset AMENT_PREFIX_PATH || true
unset CMAKE_PREFIX_PATH || true
unset COLCON_PREFIX_PATH || true
source "${TASK_ROS_SETUP}"

# Compatibility fallback for the already provisioned Ubuntu 20.04 test host.
# A normal Humble machine obtains pcl_ros from apt and never enters this branch.
if ! ros2 pkg prefix pcl_ros >/dev/null 2>&1; then
  TASK_LOCAL_PCL_SETUP="${FASTLIO_PCL_SETUP:-${FASTLIO_WS}/rolling_overlay/opt/ros/${TASK_ROS_DISTRO}/share/pcl_ros/local_setup.bash}"
  if [[ -f "${TASK_LOCAL_PCL_SETUP}" ]]; then
    source "${TASK_LOCAL_PCL_SETUP}"
  else
    echo "[ERROR] ROS package pcl_ros is missing." >&2
    echo "        Run scripts/install_fastlio2_prereqs.sh first." >&2
    exit 1
  fi
fi

export CMAKE_PREFIX_PATH="${TASK_LOCAL_PREFIX}:${CMAKE_PREFIX_PATH:-}"
export CMAKE_LIBRARY_PATH="${TASK_LOCAL_PREFIX}/lib${CMAKE_LIBRARY_PATH:+:${CMAKE_LIBRARY_PATH}}"
export CMAKE_INCLUDE_PATH="${TASK_LOCAL_PREFIX}/include${CMAKE_INCLUDE_PATH:+:${CMAKE_INCLUDE_PATH}}"
export LD_LIBRARY_PATH="${TASK_LOCAL_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_cyclonedds_cpp}"
export CMAKE_BUILD_PARALLEL_LEVEL="${TASK_BUILD_JOBS}"
export MAKEFLAGS="-j${TASK_BUILD_JOBS}"

TASK_ROS_PYTHON="${ROS_PYTHON_EXECUTABLE:-/usr/bin/python3}"
if [[ ! -x "${TASK_ROS_PYTHON}" ]]; then
  echo "[ERROR] ROS Python executable not found: ${TASK_ROS_PYTHON}" >&2
  exit 1
fi

colcon --log-base "${TASK_LOG_BASE}" build \
  --base-paths "${TASK_SOURCE_DIR}" \
  --build-base "${TASK_BUILD_BASE}" \
  --install-base "${TASK_INSTALL_BASE}" \
  --symlink-install \
  --packages-up-to fast_lio \
  --cmake-args \
    -DROS_EDITION=ROS2 \
    "-DDISTRO_ROS=${TASK_ROS_DISTRO}" \
    "-DPython3_EXECUTABLE=${TASK_ROS_PYTHON}" \
    "-DPYTHON_EXECUTABLE=${TASK_ROS_PYTHON}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_TESTING=OFF

source "${TASK_INSTALL_BASE}/setup.bash"

ros2 pkg prefix livox_ros_driver2
ros2 pkg prefix fast_lio
ros2 interface show livox_ros_driver2/msg/CustomMsg >/dev/null

TASK_FASTLIO_EXECUTABLE="${TASK_INSTALL_BASE}/fast_lio/lib/fast_lio/fastlio_mapping"
TASK_DRIVER_LIBRARY="${TASK_INSTALL_BASE}/livox_ros_driver2/lib/liblivox_ros_driver2.so"
for binary_path in "${TASK_FASTLIO_EXECUTABLE}" "${TASK_DRIVER_LIBRARY}"; do
  if [[ ! -e "${binary_path}" ]]; then
    echo "[ERROR] Expected build output is missing: ${binary_path}" >&2
    exit 1
  fi
  if ldd "${binary_path}" | grep -q 'not found'; then
    echo "[ERROR] Unresolved shared library in ${binary_path}:" >&2
    ldd "${binary_path}" | grep 'not found' >&2
    exit 1
  fi
done

TASK_OS_RELEASE="unknown"
if [[ -r /etc/os-release ]]; then
  TASK_OS_RELEASE="$(. /etc/os-release && printf '%s %s' "${ID:-unknown}" "${VERSION_ID:-unknown}")"
fi
{
  printf 'ARCHITECTURE=%s\n' "$(uname -m)"
  printf 'OS=%s\n' "${TASK_OS_RELEASE}"
  printf 'ROS_DISTRO=%s\n' "${TASK_ROS_DISTRO}"
  printf 'ROS_SETUP=%s\n' "${TASK_ROS_SETUP}"
  printf 'CMAKE=%s\n' "$("${TASK_CMAKE_BIN}" --version | head -1)"
  printf 'Livox-SDK2=%s\n' "${LIVOX_SDK2_COMMIT}"
  printf 'livox_ros_driver2=%s\n' "${LIVOX_DRIVER_COMMIT}"
  printf 'FAST_LIO_ROS2=%s\n' "${FASTLIO_COMMIT}"
  printf 'ikd-Tree=%s\n' "${IKD_TREE_COMMIT}"
  printf 'INSTALL_BASE=%s\n' "${TASK_INSTALL_BASE}"
} >"${TASK_BUILD_INFO}"

echo
echo "[OK] FAST-LIO2 workspace built and checked: ${FASTLIO_WS}"
echo "[OK] Revision record: ${TASK_BUILD_INFO}"
echo "Load it with:"
echo "  FASTLIO_WS=${FASTLIO_WS} source <t3-workspace>/scripts/enter_fastlio2_workspace.sh"
