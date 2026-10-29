#!/usr/bin/env bash
set -e
set -o pipefail
set +u

T5_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
T5_WORKSPACE_ROOT="$(cd "${T5_SCRIPT_DIR}/.." && pwd)"
T5_LIDAR_TOPIC="${T5_LIDAR_TOPIC:-/Car/T5/OS1/points}"
T5_IMU_TOPIC="${T5_IMU_TOPIC:-/Car/T5/OS1/imu}"
T5_LEFT_TOPIC="${T5_LEFT_TOPIC:-/Car/T5/Cam_Left/image_raw/color}"
T5_RIGHT_TOPIC="${T5_RIGHT_TOPIC:-/Car/T5/Cam_Right/image_raw/color}"
T5_SEMANTIC_TOPIC="${T5_SEMANTIC_TOPIC:-/task2/seg_mask}"
T5_OUSTER_START_SCRIPT="${T5_OUSTER_START_SCRIPT:-${HOME}/program/Lidar2/start_ouster.sh}"
T5_START_OUSTER_DRIVER="${T5_START_OUSTER_DRIVER:-auto}"
T5_INPUT_WAIT_SEC="${T5_INPUT_WAIT_SEC:-30}"

T5_WITH_GRIDMAP=true
T5_USE_SIM_TIME=false
T5_CHECK_ONLY=false
T5_FULL_GLOBAL_MAP=""
T5_LAUNCH_ARGS=()

usage() {
  printf '%s\n' \
    'Usage: ./scripts/run_t5_fastlivo2.sh [OPTIONS] [launch_arg:=value ...]' \
    '' \
    '  --no-gridmap   Run FAST-LIVO2 odometry only.' \
    '  --sim-time     Use rosbag /clock.' \
    '  --lite-map     Keep local map, overview and SQLite; disable the complete' \
    '                 global GridMap topic and dense NPZ.' \
    '  --full-map     Enable complete global GridMap and NPZ (default).' \
    '  --check        Check the installation without starting sensors.' \
    '  -h, --help     Show this help.' \
    '' \
    'Output defaults to $HOME/P3/output/fastlivo2_YYYYMMDD_HHMMSS.' \
    'Pass output_dir:=PATH to override it.'
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-gridmap) T5_WITH_GRIDMAP=false ;;
    --sim-time) T5_USE_SIM_TIME=true ;;
    --lite-map) T5_FULL_GLOBAL_MAP=false ;;
    --full-map) T5_FULL_GLOBAL_MAP=true ;;
    --check) T5_CHECK_ONLY=true ;;
    -h|--help) usage; exit 0 ;;
    output_dir:=*) T3_OUTPUT_DIR="${1#output_dir:=}" ;;
    full_global_map:=*)
      T5_FULL_GLOBAL_MAP="${1#full_global_map:=}"
      ;;
    *) T5_LAUNCH_ARGS+=("$1") ;;
  esac
  shift
done

if [[ -n "${T5_FULL_GLOBAL_MAP}" ]]; then
  T5_LAUNCH_ARGS+=(
    "full_global_map:=${T5_FULL_GLOBAL_MAP}"
  )
fi

source "${T5_SCRIPT_DIR}/enter_fastlivo2_workspace.sh"

if [[ "${T5_CHECK_ONLY}" == true ]]; then
  ros2 pkg executables fast_livo | grep -q 'fastlivo_mapping'
  ros2 launch t3_semantic_mapping t5_fastlivo2.launch.py --show-args >/dev/null
  echo "[OK] FAST-LIVO2 and the T5 launch entry are ready."
  exit 0
fi

T3_OUTPUT_ROOT="${T3_OUTPUT_ROOT:-${HOME}/P3/output}"
if [[ -z "${T3_OUTPUT_DIR:-}" || "${T3_OUTPUT_DIR}" == "${T3_OUTPUT_ROOT}" ]]; then
  export T3_OUTPUT_DIR="${T3_OUTPUT_ROOT}/fastlivo2_$(date +%Y%m%d_%H%M%S)"
fi
export T3_OUTPUT_ROOT
mkdir -p "${T3_OUTPUT_DIR}"

topic_exists() {
  ROS2CLI_NO_DAEMON=1 ros2 topic list 2>/dev/null | grep -Fxq "$1"
}

T5_STARTED_DRIVER_PID=""
cleanup() {
  if [[ -n "${T5_STARTED_DRIVER_PID}" ]] && kill -0 "${T5_STARTED_DRIVER_PID}" 2>/dev/null; then
    kill "${T5_STARTED_DRIVER_PID}" 2>/dev/null || true
    wait "${T5_STARTED_DRIVER_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

if [[ "${T5_START_OUSTER_DRIVER}" == true ]] || \
   { [[ "${T5_START_OUSTER_DRIVER}" == auto ]] && \
     { ! topic_exists "${T5_LIDAR_TOPIC}" || ! topic_exists "${T5_IMU_TOPIC}"; }; }; then
  if [[ ! -f "${T5_OUSTER_START_SCRIPT}" ]]; then
    echo "[ERROR] Ouster topics are absent and start script is missing: ${T5_OUSTER_START_SCRIPT}" >&2
    exit 1
  fi
  bash "${T5_OUSTER_START_SCRIPT}" &
  T5_STARTED_DRIVER_PID=$!
fi

required_topics=("${T5_LIDAR_TOPIC}" "${T5_IMU_TOPIC}" "${T5_LEFT_TOPIC}")
if [[ "${T5_WITH_GRIDMAP}" == true ]]; then
  required_topics+=("${T5_RIGHT_TOPIC}" "${T5_SEMANTIC_TOPIC}")
fi

for ((waited=0; waited<T5_INPUT_WAIT_SEC; waited++)); do
  missing=false
  for topic in "${required_topics[@]}"; do
    if ! topic_exists "${topic}"; then missing=true; break; fi
  done
  [[ "${missing}" == false ]] && break
  sleep 1
done

for topic in "${required_topics[@]}"; do
  if ! topic_exists "${topic}"; then
    echo "[ERROR] Required input topic is unavailable: ${topic}" >&2
    exit 1
  fi
done

echo "[T5 FAST-LIVO2] odometry=/aft_mapped_to_init"
echo "[T5 FAST-LIVO2] propagated_odometry=/LIVO2/imu_propagate"
echo "[T5 FAST-LIVO2] semantic=${T5_SEMANTIC_TOPIC}"
echo "[T5 FAST-LIVO2] gridmap=${T5_WITH_GRIDMAP} output=${T3_OUTPUT_DIR}"

exec ros2 launch t3_semantic_mapping t5_fastlivo2.launch.py \
  use_sim_time:="${T5_USE_SIM_TIME}" \
  with_gridmap:="${T5_WITH_GRIDMAP}" \
  output_dir:="${T3_OUTPUT_DIR}" \
  left_topic:="${T5_LEFT_TOPIC}" \
  right_topic:="${T5_RIGHT_TOPIC}" \
  semantic_topic:="${T5_SEMANTIC_TOPIC}" \
  lidar_topic:="${T5_LIDAR_TOPIC}" \
  imu_topic:="${T5_IMU_TOPIC}" \
  "${T5_LAUNCH_ARGS[@]}"
