#!/usr/bin/env bash
set -e
set -o pipefail

T3_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
T3_CALIBRATION="${T3_CALIB_PATH:-}"
if [[ $# -gt 0 && "${1}" != *":="* && "${1}" != --* ]]; then
  T3_CALIBRATION="${1}"
  shift
fi
if [[ -n "${T3_CALIBRATION}" && ! -e "${T3_CALIBRATION}" ]]; then
  echo "[ERROR] Calibration path does not exist: ${T3_CALIBRATION}" >&2
  exit 2
fi

usage() {
  cat <<'EOF'
Usage: ./scripts/run_t5_roma_vo.sh [CALIB_PATH] [OPTIONS] [launch_arg:=value ...]

  --feature-matches     Enable /Car/T2/feature_matching/visualization.
  --no-feature-matches  Disable the feature-match publisher completely.
  --lite-map            Lightweight map mode:
                        no complete global GridMap topic and no dense NPZ;
                        local GridMap, global overview and SQLite stay enabled.
  --full-map            Enable complete global GridMap and NPZ
                        (the default real-vehicle visualization profile).
  --enable_pointcloud_fusion
                        Add OS1 and left/right TOF elevation evidence.
  -h, --help            Show this help message.

The ROS launch spelling publish_feature_matches:=true/false remains supported.
When enabled, match images are rendered only while a subscriber is connected.
Output defaults to $HOME/P3/output/roma_YYYYMMDD_HHMMSS.
Pass output_dir:=PATH to override it.
EOF
}

T3_FEATURE_MATCHES=""
T3_FULL_GLOBAL_MAP=""
T3_REQUESTED_OUTPUT_DIR=""
T3_POINTCLOUD_FUSION=""
T3_LAUNCH_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --feature-matches)
      T3_FEATURE_MATCHES=true
      ;;
    --no-feature-matches)
      T3_FEATURE_MATCHES=false
      ;;
    --lite-map)
      T3_FULL_GLOBAL_MAP=false
      ;;
    --full-map)
      T3_FULL_GLOBAL_MAP=true
      ;;
    --enable_pointcloud_fusion)
      T3_POINTCLOUD_FUSION=true
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    publish_feature_matches:=*)
      T3_FEATURE_MATCHES="${1#publish_feature_matches:=}"
      ;;
    full_global_map:=*)
      T3_FULL_GLOBAL_MAP="${1#full_global_map:=}"
      ;;
    output_dir:=*)
      T3_REQUESTED_OUTPUT_DIR="${1#output_dir:=}"
      ;;
    *)
      T3_LAUNCH_ARGS+=("$1")
      ;;
  esac
  shift
done

T3_OUTPUT_ROOT="${T3_OUTPUT_ROOT:-${HOME}/P3/output}"
if [[ -n "${T3_REQUESTED_OUTPUT_DIR}" ]]; then
  export T3_OUTPUT_DIR="${T3_REQUESTED_OUTPUT_DIR}"
elif [[ -z "${T3_OUTPUT_DIR:-}" || "${T3_OUTPUT_DIR}" == "${T3_OUTPUT_ROOT}" ]]; then
  export T3_OUTPUT_DIR="${T3_OUTPUT_ROOT}/roma_$(date +%Y%m%d_%H%M%S)"
fi
export T3_OUTPUT_ROOT
mkdir -p "${T3_OUTPUT_DIR}"

if [[ -n "${T3_FEATURE_MATCHES}" ]]; then
  T3_LAUNCH_ARGS+=(
    "publish_feature_matches:=${T3_FEATURE_MATCHES}"
  )
fi
if [[ -n "${T3_FULL_GLOBAL_MAP}" ]]; then
  T3_LAUNCH_ARGS+=(
    "full_global_map:=${T3_FULL_GLOBAL_MAP}"
  )
fi
if [[ -n "${T3_POINTCLOUD_FUSION}" ]]; then
  T3_LAUNCH_ARGS+=(
    "enable_pointcloud_fusion:=${T3_POINTCLOUD_FUSION}"
  )
fi

source "${T3_SCRIPT_DIR}/enter_workspace.sh"
echo "[T5 RoMa VO] output=${T3_OUTPUT_DIR}"

if [[ -n "${T3_CALIBRATION}" ]]; then
  exec ros2 launch t3_semantic_mapping t5_roma_vo.launch.py \
    calib_path:="${T3_CALIBRATION}" \
    output_dir:="${T3_OUTPUT_DIR}" \
    "${T3_LAUNCH_ARGS[@]}"
fi
exec ros2 launch t3_semantic_mapping t5_roma_vo.launch.py \
  output_dir:="${T3_OUTPUT_DIR}" \
  "${T3_LAUNCH_ARGS[@]}"
