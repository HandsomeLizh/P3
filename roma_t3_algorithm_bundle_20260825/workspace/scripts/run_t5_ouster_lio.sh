#!/usr/bin/env bash
set -e
set -o pipefail
set +u

T5_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
T5_WORKSPACE_ROOT="$(cd "${T5_SCRIPT_DIR}/.." && pwd)"
T5_CONFIG_DIR="${T5_WORKSPACE_ROOT}/src/t3_semantic_mapping/config"
T5_CONFIG_FILE="t5_ouster_fastlio.yaml"
T5_LIDAR_TOPIC="${T5_LIDAR_TOPIC:-/Car/T5/OS1/points}"
T5_IMU_TOPIC="${T5_IMU_TOPIC:-/Car/T5/OS1/imu}"
T5_OUSTER_START_SCRIPT="${T5_OUSTER_START_SCRIPT:-${HOME}/program/Lidar2/start_ouster.sh}"
T5_START_OUSTER_DRIVER="${T5_START_OUSTER_DRIVER:-auto}"
T5_OUSTER_WAIT_SEC="${T5_OUSTER_WAIT_SEC:-30}"
T5_FASTLIO_RVIZ="${T5_FASTLIO_RVIZ:-false}"

if [[ "${1:-}" == "--check" ]]; then
  T5_CHECK_ONLY=true
  shift
else
  T5_CHECK_ONLY=false
fi

if [[ ! -f "${T5_CONFIG_DIR}/${T5_CONFIG_FILE}" ]]; then
  echo "[ERROR] FAST-LIO2 Ouster config is missing: ${T5_CONFIG_DIR}/${T5_CONFIG_FILE}" >&2
  exit 1
fi
if [[ ! "${T5_OUSTER_WAIT_SEC}" =~ ^[1-9][0-9]*$ ]]; then
  echo "[ERROR] T5_OUSTER_WAIT_SEC must be a positive integer." >&2
  exit 2
fi
if [[ "${T5_START_OUSTER_DRIVER}" != "auto" && \
      "${T5_START_OUSTER_DRIVER}" != "true" && \
      "${T5_START_OUSTER_DRIVER}" != "false" ]]; then
  echo "[ERROR] T5_START_OUSTER_DRIVER must be auto, true, or false." >&2
  exit 2
fi

# Discover/start the sensor from the clean base ROS environment.  Loading the
# bundled PCL compatibility overlay first can make ros2cli use a different
# discovery environment even though the FAST-LIO2 node itself can receive the
# sensor topics.
T5_BASE_ROS_SETUP="${ROS_SETUP_FILE:-/opt/ros/humble/setup.bash}"
if [[ ! -f "${T5_BASE_ROS_SETUP}" ]]; then
  echo "[ERROR] Base ROS setup is missing: ${T5_BASE_ROS_SETUP}" >&2
  exit 1
fi
source "${T5_BASE_ROS_SETUP}"
export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_cyclonedds_cpp}"

# Match the production Ouster driver DDS configuration when it exists.
T5_CYCLONE_CONFIG="${HOME}/program/cyclonedds.xml"
if [[ -z "${CYCLONEDDS_URI:-}" && -f "${T5_CYCLONE_CONFIG}" ]]; then
  export CYCLONEDDS_URI="file://${T5_CYCLONE_CONFIG}"
fi

if [[ "${T5_CHECK_ONLY}" == true ]]; then
  source "${T5_SCRIPT_DIR}/enter_fastlio2_workspace.sh"
  ros2 pkg prefix fast_lio
  ros2 launch fast_lio mapping.launch.py --show-args >/dev/null
  echo "[OK] T5 Ouster FAST-LIO2 entry is ready."
  echo "[INFO] points=${T5_LIDAR_TOPIC}"
  echo "[INFO] imu=${T5_IMU_TOPIC}"
  echo "[INFO] config=${T5_CONFIG_DIR}/${T5_CONFIG_FILE}"
  exit 0
fi

topics_are_available() {
  local topic_list
  topic_list="$(ROS2CLI_NO_DAEMON=1 ros2 topic list 2>/dev/null || true)"
  grep -Fxq "${T5_LIDAR_TOPIC}" <<<"${topic_list}" &&
    grep -Fxq "${T5_IMU_TOPIC}" <<<"${topic_list}"
}

T5_STARTED_DRIVER_PID=""
cleanup() {
  if [[ -n "${T5_STARTED_DRIVER_PID}" ]] && \
     kill -0 "${T5_STARTED_DRIVER_PID}" 2>/dev/null; then
    kill "${T5_STARTED_DRIVER_PID}" 2>/dev/null || true
    wait "${T5_STARTED_DRIVER_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

if [[ "${T5_START_OUSTER_DRIVER}" == "true" ]] || \
   { [[ "${T5_START_OUSTER_DRIVER}" == "auto" ]] && \
     ! topics_are_available; }; then
  if [[ ! -f "${T5_OUSTER_START_SCRIPT}" ]]; then
    echo "[ERROR] Ouster driver is not publishing and its start script is missing:" >&2
    echo "        ${T5_OUSTER_START_SCRIPT}" >&2
    exit 1
  fi
  echo "[T5 LIO] Starting Ouster driver: ${T5_OUSTER_START_SCRIPT}"
  bash "${T5_OUSTER_START_SCRIPT}" &
  T5_STARTED_DRIVER_PID=$!
fi

for ((T5_WAITED=0; T5_WAITED<T5_OUSTER_WAIT_SEC; T5_WAITED++)); do
  if topics_are_available; then
    break
  fi
  sleep 1
done
if ! topics_are_available; then
  echo "[ERROR] Ouster input did not become ready within ${T5_OUSTER_WAIT_SEC}s." >&2
  echo "        points=${T5_LIDAR_TOPIC}" >&2
  echo "        imu=${T5_IMU_TOPIC}" >&2
  exit 1
fi

# Load FAST-LIO2 only after the clean ROS environment has confirmed both
# sensor publishers.  The running driver remains in its own child process.
source "${T5_SCRIPT_DIR}/enter_fastlio2_workspace.sh"

echo "[T5 LIO] Starting FAST-LIO2 for Ouster OS1-128."
echo "[T5 LIO] Output odometry: /Odometry"
echo "[T5 LIO] Registered clouds: /cloud_registered, /cloud_registered_body"
ros2 launch fast_lio mapping.launch.py \
  config_path:="${T5_CONFIG_DIR}" \
  config_file:="${T5_CONFIG_FILE}" \
  rviz:="${T5_FASTLIO_RVIZ}" \
  "$@"
