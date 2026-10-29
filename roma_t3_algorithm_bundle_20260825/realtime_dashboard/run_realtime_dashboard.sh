#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# -gt 0 ]]; then
  OUTPUT_DIR="$1"
  shift
else
  OUTPUT_DIR="$(cat /tmp/t3_roma_current_output 2>/dev/null || true)"
fi
if [[ -z "${OUTPUT_DIR}" ]]; then
  echo "Usage: $0 /home/yanfa/P3/output/RUN_DIRECTORY" >&2
  exit 2
fi
if [[ ! -d "${OUTPUT_DIR}" ]]; then
  echo "Output directory does not exist: ${OUTPUT_DIR}" >&2
  exit 2
fi

set +u
source /opt/ros/humble/setup.bash
source /home/yanfa/P3/t3_gridmap_v2_delivery/t3_gridmap_v2_ws/install_gpu/setup.bash
set -u
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-228}"
export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_cyclonedds_cpp}"

exec python3 /home/yanfa/P3/realtime_dashboard/t3_realtime_dashboard.py \
  --output-dir "${OUTPUT_DIR}" \
  --host "${T3_DASHBOARD_HOST:-0.0.0.0}" \
  --port "${T3_DASHBOARD_PORT:-8765}" \
  --stability-period-sec "${T3_STABILITY_PERIOD_SEC:-500}" \
  "$@"
