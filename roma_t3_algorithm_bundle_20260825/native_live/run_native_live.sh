#!/usr/bin/env bash
# ROS 2 setup scripts intentionally probe unset variables, so nounset cannot
# be enabled before sourcing them.
set -Eeo pipefail

ROOT="${T3_BUNDLE_ROOT:-/home/yanfa/P3/roma_t3_algorithm_bundle_20260825}"
RUN_DIR="$ROOT/run_native"
OUTPUT_ROOT="${T3_OUTPUT_ROOT:-$ROOT/output_native}"
INSTALL_SETUP="$ROOT/workspace/install_native/setup.bash"
CALIB_PATH="${T3_CALIB_PATH:-$ROOT/config/calibration_spreadsheets_parallel_24p9.yaml}"
POINTCLOUD_CONFIG="${T3_POINTCLOUD_CONFIG:-$ROOT/config/pointcloud_extrinsic_os1_mapping_only.yaml}"
LEFT_TOPIC="${T3_LEFT_TOPIC:-/Car/T5/Cam_Left/image_raw/color}"
RIGHT_TOPIC="${T3_RIGHT_TOPIC:-/Car/T5/Cam_Right/image_raw/color}"
REFERENCE_TOPIC="${T3_REFERENCE_TOPIC:-/car/pose}"
LIDAR_TOPIC="${T3_LIDAR_TOPIC:-/Car/T5/OS1/points}"
DASHBOARD_PORT="${T3_DASHBOARD_PORT:-8765}"

fail() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

pid_is_alive() {
  local pid_file="$1" pid
  [[ -f "$pid_file" ]] || return 1
  pid="$(<"$pid_file")"
  [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null
}

wait_for_log() {
  local pid_file="$1" log_file="$2" pattern="$3" timeout_sec="$4"
  local waited=0
  while (( waited < timeout_sec )); do
    grep -Fq "$pattern" "$log_file" 2>/dev/null && return 0
    pid_is_alive "$pid_file" || return 1
    sleep 1
    ((waited += 1))
  done
  return 1
}

[[ -f /opt/ros/humble/setup.bash ]] || fail "ROS 2 Humble is not installed under /opt/ros/humble"
[[ -f "$INSTALL_SETUP" ]] || fail "native workspace is not built: $INSTALL_SETUP"
[[ -f "$CALIB_PATH" ]] || fail "camera calibration is missing: $CALIB_PATH"
[[ -f "$POINTCLOUD_CONFIG" ]] || fail "point-cloud calibration is missing: $POINTCLOUD_CONFIG"
[[ -d "$ROOT/third_party/RoMa" ]] || fail "RoMa source is missing"

mkdir -p "$RUN_DIR" "$OUTPUT_ROOT"
for name in mapper dashboard reference_recorder; do
  pid_is_alive "$RUN_DIR/$name.pid" && fail "$name is already running (PID $(<"$RUN_DIR/$name.pid"))"
  rm -f "$RUN_DIR/$name.pid"
done

if ss -ltn "sport = :$DASHBOARD_PORT" | grep -q LISTEN; then
  fail "dashboard port $DASHBOARD_PORT is already in use"
fi

source /opt/ros/humble/setup.bash
source "$INSTALL_SETUP"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-10}"
export ROS_LOCALHOST_ONLY="${ROS_LOCALHOST_ONLY:-0}"
export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_cyclonedds_cpp}"
export T3_ALGORITHM_ROOT="$ROOT/third_party/RoMa"
export TORCH_HOME="$ROOT/models/torch"
export PYTHONPATH="$ROOT/third_party/RoMa${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

python3 -c 'import rclpy, torch; from roma_vo import StereoSLAMv2' >/dev/null \
  || fail "Python/ROS/RoMa native dependency check failed"

stamp="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="$OUTPUT_ROOT/roma_live_$stamp"
mkdir -p "$OUTPUT_DIR/logs"
printf '%s\n' "$OUTPUT_DIR" > "$RUN_DIR/current_output.txt"
printf '%s\n' "$OUTPUT_DIR" > "$OUTPUT_ROOT/current_output.txt"

start_service() {
  local name="$1"
  shift
  nohup setsid "$@" >"$OUTPUT_DIR/logs/$name.log" 2>&1 </dev/null &
  local pid=$!
  printf '%s\n' "$pid" > "$RUN_DIR/$name.pid"
}

start_service mapper ros2 launch t3_semantic_mapping \
  t5_roma_vo_live_parallel_24p9.launch.py \
  output_dir:="$OUTPUT_DIR" \
  calib_path:="$CALIB_PATH" \
  algorithm_root:="$ROOT/third_party/RoMa" \
  left_topic:="$LEFT_TOPIC" \
  right_topic:="$RIGHT_TOPIC" \
  use_sim_time:=false \
  image_timestamp_source:=sensor \
  image_sync_mode:=approximate \
  image_sync_slop_sec:="${T3_SYNC_SLOP_SEC:-0.07}" \
  stereo_sync_max_skew_sec:="${T3_STEREO_MAX_SKEW_SEC:-0.07}" \
  enable_semantics:=false \
  enable_imu:=false \
  enable_pointcloud_fusion:=true \
  pointcloud_fusion_config:="$POINTCLOUD_CONFIG" \
  full_global_map:=true \
  publish_realtime_metrics:=true

if ! wait_for_log "$RUN_DIR/mapper.pid" "$OUTPUT_DIR/logs/mapper.log" \
  "Mapper ready. Waiting for synchronized image input." 120; then
  tail -n 80 "$OUTPUT_DIR/logs/mapper.log" >&2 || true
  "$ROOT/native_live/stop_native_live.sh" >/dev/null 2>&1 || true
  fail "mapper did not reach the waiting-for-images state"
fi

start_service reference_recorder python3 "$ROOT/workflow/record_pose_reference.py" \
  "$OUTPUT_DIR" --topic "$REFERENCE_TOPIC"

start_service dashboard python3 "$ROOT/realtime_dashboard/t3_realtime_dashboard.py" \
  --output-dir "$OUTPUT_DIR" \
  --host 0.0.0.0 \
  --port "$DASHBOARD_PORT" \
  --estimate-topic /Car/T3/localization/odometry \
  --reference-topic "$REFERENCE_TOPIC" \
  --timing-topic /Car/T3/metrics/frame_timing \
  --global-map-topic /Car/T3/mapping/global_grid_map \
  --stability-period-sec 500

if ! wait_for_log "$RUN_DIR/dashboard.pid" "$OUTPUT_DIR/logs/dashboard.log" \
  "Dashboard telemetry ready" 30; then
  tail -n 80 "$OUTPUT_DIR/logs/dashboard.log" >&2 || true
  "$ROOT/native_live/stop_native_live.sh" >/dev/null 2>&1 || true
  fail "dashboard failed to start"
fi

if ! wait_for_log "$RUN_DIR/reference_recorder.pid" \
  "$OUTPUT_DIR/logs/reference_recorder.log" "Saving PoseStamped reference" 30; then
  tail -n 80 "$OUTPUT_DIR/logs/reference_recorder.log" >&2 || true
  "$ROOT/native_live/stop_native_live.sh" >/dev/null 2>&1 || true
  fail "ground-truth recorder failed to start"
fi

cat <<EOF
Native live pipeline is ready and waiting for data.
ROS_DOMAIN_ID: $ROS_DOMAIN_ID
Left image:    $LEFT_TOPIC
Right image:   $RIGHT_TOPIC
Ground truth:  $REFERENCE_TOPIC
Point cloud:   $LIDAR_TOPIC
Output:        $OUTPUT_DIR
Dashboard:     http://$(hostname -I | awk '{print $1}'):$DASHBOARD_PORT
Stop command:  $ROOT/native_live/stop_native_live.sh
EOF
