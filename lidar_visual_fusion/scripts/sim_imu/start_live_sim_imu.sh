#!/usr/bin/env bash
# Start the passive truth recorder, the wheel-speed relay and (unless --no-imu) the synthetic
# IMU (simulation truth + noise) in the capture/control domain 10. Nothing is sent to the
# vehicle. Stop with stop_live_sim_imu.sh. Usually called by start_live_debug.sh.
#   start_live_sim_imu.sh [--no-imu] [extra sim_imu_live.py options, e.g. --noise none]
set -eo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
STATE="$ROOT/results/sim_imu_live.pids"
WITH_IMU=1
if [[ "${1:-}" == --no-imu ]]; then WITH_IMU=0; shift; fi
if [[ -f "$STATE" ]]; then
  read -r OLD_IMU OLD_REC OLD_WHEEL < "$STATE"
  for pid in $OLD_IMU $OLD_REC ${OLD_WHEEL:-none}; do
    if [[ "$pid" != none ]] && kill -0 "$pid" 2>/dev/null; then
      echo "Truth recorder / synthetic IMU / wheel relay already running: $(cat "$STATE")"; exit 1
    fi
  done
fi
OUT="$ROOT/results/sim_imu_live_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT"
source "$ROOT/scripts/env.sh" >/dev/null
source /home/yanfa/Env_X/InterFace/install/setup.bash   # /car/telemetry message type
# Same DDS settings as the live relay, which joins both domain 10 and mapping domain 57.
export ROS_DOMAIN_ID=10 ROS_LOCALHOST_ONLY=0 RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
if [[ -f /home/yanfa/program/cyclonedds.xml ]]; then
  export CYCLONEDDS_URI=file:///home/yanfa/program/cyclonedds.xml
else
  unset CYCLONEDDS_URI
fi
IMU=none
if [[ $WITH_IMU == 1 ]]; then
  # --direct-domain 57: gap-free copy straight to VoxelMap (/fusion/imu); see sim_imu_live.py.
  setsid nohup nice -n 5 /usr/bin/python3 "$HERE/sim_imu_live.py" --topic /sim/imu_from_truth \
    --direct-domain 57 --output "$OUT" "$@" > "$OUT/sim_imu_live.log" 2>&1 < /dev/null &
  IMU=$!
fi
setsid nohup nice -n 10 ros2 bag record --output "$OUT/truth_bag" --storage sqlite3 \
  /car/pose /car/odom /car/telemetry /sensor/meta /sensor/batch_status /sim/imu_from_truth \
  > "$OUT/truth_record.log" 2>&1 < /dev/null &
REC=$!
# Wheel-speed upper bound for the output gate; always on (works with and without the IMU).
setsid nohup nice -n 5 /usr/bin/python3 "$HERE/wheel_speed_relay.py" --target-domain 57 \
  > "$OUT/wheel_speed_relay.log" 2>&1 < /dev/null &
WHEEL=$!
echo "$IMU $REC $WHEEL" > "$STATE"
echo "$OUT" > "$ROOT/results/sim_imu_live.latest"
sleep 3
[[ $WITH_IMU == 1 ]] && echo "Synthetic IMU (NOT a physical sensor) -> /sim/imu_from_truth (domain 10) and /fusion/imu (domain 57)"
echo "Wheel speed bound -> /fusion/wheel_speed (domain 57)"
echo "Truth recorded to $OUT/truth_bag; do not read it until stop_live_sim_imu.sh has finished."
