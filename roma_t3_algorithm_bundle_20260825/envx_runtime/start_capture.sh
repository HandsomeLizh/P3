#!/usr/bin/env bash
set -eo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source /opt/ros/humble/setup.bash
source "$SCRIPT_DIR/install/setup.bash"
set -u
UE_HOST=192.168.10.22
UE_PORT=6665
CAPTURE_HZ=0.25
while [[ $# -gt 0 ]]; do
  case "$1" in
    --host) UE_HOST="$2"; shift 2;;
    --port) UE_PORT="$2"; shift 2;;
    --capture-freq) CAPTURE_HZ="$2"; shift 2;;
    --no-save) shift;;
    *) echo "Unsupported argument: $1" >&2; exit 2;;
  esac
done
# P3-local build. No sudo, setcap, global installs, or raw-image directory writes.
exec ros2 run sensor_capture_ros2 sensor_capture_node --ros-args \
  -p tcp_host:="$UE_HOST" -p tcp_port:="$UE_PORT" \
  -p capture_freq:="$CAPTURE_HZ" -p batch_count:=20000 \
  -p timestamp_mode:="${SENSOR_TIMESTAMP_MODE:-meta_relative}" \
  -p save_dir:="$SCRIPT_DIR/capture_scratch" -p save_data:=false \
  -p auto_start:=true -p big_endian:=false -p publish_thumbnail:=false \
  -p 'output_topics:=[rgb_0:=/Car/T5/Cam_Left/image_raw/color,rgb_1:=/Car/T5/Cam_Right/image_raw/color,cpulidar_0:=/Car/T5/OS1/points]'
