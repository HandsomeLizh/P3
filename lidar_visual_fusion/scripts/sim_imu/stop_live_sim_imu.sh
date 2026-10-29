#!/usr/bin/env bash
# Stop the synthetic IMU, the wheel relay and close the truth recording cleanly
# (SIGINT writes metadata.yaml).
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
STATE="$ROOT/results/sim_imu_live.pids"
[[ -f "$STATE" ]] || { echo "Truth recorder / synthetic IMU not running"; exit 0; }
read -r IMU REC WHEEL < "$STATE"
WHEEL="${WHEEL:-none}"
for pid in $REC $IMU $WHEEL; do
  [[ "$pid" == none ]] && continue
  pkill -INT -g "$pid" 2>/dev/null || kill -INT "$pid" 2>/dev/null || true
done
alive() { [[ "$1" != none ]] && kill -0 "$1" 2>/dev/null; }
for _ in $(seq 1 30); do
  alive "$IMU" || alive "$REC" || alive "$WHEEL" || break
  sleep 1
done
rm -f "$STATE"
OUT="$(cat "$ROOT/results/sim_imu_live.latest" 2>/dev/null)"
if [[ -f "$OUT/truth_bag/metadata.yaml" ]]; then
  echo "Stopped; truth bag closed: $OUT/truth_bag"
else
  source /opt/ros/humble/setup.bash
  ros2 bag reindex "$OUT/truth_bag" -s sqlite3 >/dev/null 2>&1 && echo "Stopped; truth bag reindexed: $OUT/truth_bag" \
    || echo "Stopped; truth bag metadata missing: $OUT/truth_bag"
fi
