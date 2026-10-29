#!/usr/bin/env bash
# Replay a live session recorded by start_live_debug.sh (live_input_bag + truth bag) through the
# full P3 pipeline with the synthetic IMU on or off, then score against truth.
# Isolated ROS domain (default 91, localhost only), headless, real-time rate.
#   replay_live_session.sh <live_mapping_run_dir> --imu on|off [--noise mems|none] [--truth-bag DIR]
# Only the recorded sensor topics are played; the IMU comes from the truth bag, published
# straight to VoxelMap (/fusion/imu) so the Python sensor adapter cannot drop bursts.
set -eo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
RUN="$(cd "${1:?usage: replay_live_session.sh <run_dir> --imu on|off [--wheel on|off]}" && pwd)"; shift
IMU=''; WHEEL=on; NOISE=mems; DOMAIN=91; TRUTH=''
while [[ $# -gt 0 ]]; do
  case "$1" in
    --imu) IMU="$2"; shift 2;;
    --wheel) WHEEL="$2"; shift 2;;
    --noise) NOISE="$2"; shift 2;;
    --domain) DOMAIN="$2"; shift 2;;
    --truth-bag) TRUTH="$(realpath "$2")"; shift 2;;
    *) echo "unknown argument $1" >&2; exit 2;;
  esac
done
[[ "$IMU" == on || "$IMU" == off ]] || { echo "--imu on|off is required" >&2; exit 2; }
[[ "$WHEEL" == on || "$WHEEL" == off ]] || { echo "--wheel must be on or off" >&2; exit 2; }
BAG="$RUN/live_input_bag"
[[ -f "$BAG/metadata.yaml" ]] || { echo "missing $BAG/metadata.yaml (was the run stopped cleanly?)"; exit 2; }
if [[ -z "$TRUTH" ]]; then
  TRUTH="$(/usr/bin/python3 -c "import json;print(json.load(open('$RUN/debug_session.json'))['truth_bag'])")"
fi
source "$ROOT/scripts/env.sh" >/dev/null
export ROS_DOMAIN_ID=$DOMAIN ROS_LOCALHOST_ONLY=1
CACHE="$ROOT/results/sim_imu_bags"; mkdir -p "$CACHE"
NPZ="$CACHE/$(basename "$RUN")_${NOISE}"
[[ -f "$NPZ.npz" ]] || nice -n 10 /usr/bin/python3 "$HERE/build_sim_imu.py" --truth "$TRUTH" --noise "$NOISE" \
  --out "$NPZ" > "$NPZ.log" 2>&1 || { echo "IMU build failed, see $NPZ.log"; exit 1; }
if [[ "$IMU" == on ]]; then PROFILE="$ROOT/config/simulation_lidar_360_simimu.yaml"; WHEEL=on
elif [[ "$WHEEL" == on ]]; then PROFILE="$ROOT/config/simulation_lidar_360_wheel.yaml"
else PROFILE="$ROOT/config/simulation_lidar_360.yaml"; fi
OUT="$ROOT/results/replay_$(basename "$RUN")_imu${IMU}_wheel${WHEEL}_$(date +%Y%m%d_%H%M%S)"
/usr/bin/python3 "$ROOT/scripts/control.py" start --profile "$PROFILE" --use-sim-time --domain "$DOMAIN" --output "$OUT"
cp "$ROOT/build_manifest.json" "$OUT/build_manifest.json"
players=()
cleanup() {
  trap - EXIT INT TERM
  for pid in "${players[@]}"; do kill -INT "$pid" 2>/dev/null || true; done
  timeout 120 "$ROOT/save_map.sh" > "$OUT/replay_cleanup.log" 2>&1 || true
  "$ROOT/stop.sh" >> "$OUT/replay_cleanup.log" 2>&1 || true
}
trap cleanup EXIT
trap 'exit 130' INT TERM
if [[ "$IMU" == on ]]; then
  /usr/bin/python3 "$HERE/sim_imu_player.py" "$NPZ.npz" --topic /fusion/imu --frame-id imu \
    > "$OUT/sim_imu_player.log" 2>&1 &
  players+=($!)
fi
if [[ "$WHEEL" == on ]]; then
  # ros2 bag play keeps original stamps, so no timeline shift; telemetry lives in the truth bag.
  ( source /home/yanfa/Env_X/InterFace/install/setup.bash && exec /usr/bin/python3 \
      "$HERE/wheel_speed_player.py" "$TRUTH" ) > "$OUT/wheel_speed_player.log" 2>&1 &
  players+=($!)
fi
sleep 8   # estimator start-up
LIDAR_TOPIC="$(/usr/bin/python3 -c "import yaml;print(yaml.safe_load(open('$PROFILE'))['lidar_topic'])")"
ros2 bag play "$BAG" --clock 50 --rate 1 --read-ahead-queue-size 6 --disable-keyboard-controls \
  --topics /fusion/left /fusion/right "$LIDAR_TOPIC" > "$OUT/player.log" 2>&1
# Keep sim time advancing so the last scans are fused and mapped.
/usr/bin/python3 - "$BAG" <<'PY'
import sys,time,yaml,rclpy
from rosgraph_msgs.msg import Clock
m=yaml.safe_load(open(sys.argv[1]+'/metadata.yaml'))['rosbag2_bagfile_information']
last=m['starting_time']['nanoseconds_since_epoch']+m['duration']['nanoseconds']
rclpy.init();n=rclpy.create_node('replay_clock_tail');p=n.create_publisher(Clock,'/clock',10);t0=time.monotonic()
while time.monotonic()-t0<30:
    c=Clock();c.clock.sec,c.clock.nanosec=divmod(last+int((time.monotonic()-t0)*1e9),10**9);p.publish(c);rclpy.spin_once(n,timeout_sec=.02)
n.destroy_node();rclpy.shutdown()
PY
cleanup
nice -n 10 /usr/bin/python3 "$HERE/evaluate_against_truth.py" --run "$OUT" --truth "$NPZ.npz" \
  --out "$OUT/truth_eval.json" > /dev/null && echo "Evaluation: $OUT/truth_eval.json"
echo "Results: $OUT"
