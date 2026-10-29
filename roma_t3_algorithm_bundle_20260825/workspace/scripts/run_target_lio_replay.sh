#!/usr/bin/env bash
set -e
set -o pipefail
set +u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
TARGET_BATCH_ROOT="${T3_BATCH_ROOT:-${HOME}/SLAM/batch_organized}"
TARGET_SEQUENCE="${1:-${TARGET_BATCH_ROOT}/2025-06-26-23-01-34}"
if [[ $# -gt 0 ]]; then
  shift
fi
TARGET_SEQUENCE_NAME="$(basename "${TARGET_SEQUENCE}")"

if [[ ! -f "${TARGET_SEQUENCE}/lidar_fastlio/trajectory.npz" ]]; then
  echo "Saved FAST-LIO trajectory is missing: ${TARGET_SEQUENCE}" >&2
  exit 2
fi

# T_lidar_body recovered from this recording's raw and registered point-cloud
# pairs. Across the checked frames, >99.9% of retained points are within 1 cm
# and the median residual is below 0.1 mm.  It is valid for deterministic
# replay only and must not be reused as a production robot calibration without
# an independent check.
T_LIDAR_BODY="[-0.005840618038, 0.999954907612, -0.007487959359, -0.056179127109, -0.999872735497, -0.005728641392, 0.014889458530, 0.002594557612, 0.014845891265, 0.007573970042, 0.999861107199, 0.177470581665, 0.0, 0.0, 0.0, 1.0]"

source "${SCRIPT_DIR}/enter_workspace.sh"
TARGET_OUTPUT_DIR="${T3_OUTPUT_ROOT}/${TARGET_SEQUENCE_NAME}_lio_replay_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${TARGET_OUTPUT_DIR}"

exec ros2 launch t3_semantic_mapping \
  practice_live_online_semantics.launch.py \
  data_dir:="${TARGET_SEQUENCE}" \
  algorithm_root:="${T3_ALGORITHM_ROOT:-${WORKSPACE_DIR}/third_party/RoMa}" \
  output_dir:="${TARGET_OUTPUT_DIR}" \
  localization_mode:=lio \
  fusion_backend:=none \
  publish_lio_odometry:=true \
  lio_child_to_lidar_confirmed:=true \
  lio_child_to_lidar_transform:="${T_LIDAR_BODY}" \
  enable_lidar_mapping:=true \
  lidar_temporal_mode:=per_point \
  lidar_point_time_field:=time \
  lidar_point_time_unit:=sec \
  lidar_publish_max_points:=5000 \
  max_stereo_skew_ms:=200.0 \
  max_frames:=20 \
  playback_rate:=0.2 \
  start_delay_sec:=25.0 \
  "$@"
