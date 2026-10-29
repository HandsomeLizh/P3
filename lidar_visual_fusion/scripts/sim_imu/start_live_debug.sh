#!/usr/bin/env bash
# One command for a live P3 session that also records everything needed for debugging:
#   scripts/sim_imu/start_live_debug.sh --imu on|off [--noise mems|none] [--no-eval]
# 1. truth recorder and wheel-speed relay (always), synthetic IMU only with --imu on (domain 10),
# 2. ./start_live.sh with the matching 360 deg profile (wheel gate in both; IMU optional)
#    and --record-input,
# 3. after Ctrl+C: stop the helpers, close the truth bag, link it from the run directory
#    (debug_session.json) and score the run against truth (truth_eval.json).
# Run it in place of terminal B's start_live.sh; capture and vehicle control stay as before.
# --imu on: keep the rover still for ~5 s after mapping starts (IMU gravity initialisation).
set -eo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
IMU=''; NOISE=mems; EVAL=1
while [[ $# -gt 0 ]]; do
  case "$1" in
    --imu) IMU="$2"; shift 2;;
    --noise) NOISE="$2"; shift 2;;
    --no-eval) EVAL=0; shift;;
    -h|--help) sed -n 2,10p "$0"; exit 0;;
    *) echo "unknown argument $1" >&2; exit 2;;
  esac
done
[[ "$IMU" == on || "$IMU" == off ]] || { echo "--imu on|off is required" >&2; exit 2; }
if [[ "$IMU" == on ]]; then
  PROFILE="$ROOT/config/simulation_lidar_360_simimu.yaml"   # wheel gate + synthetic IMU
  "$HERE/start_live_sim_imu.sh" --noise "$NOISE"
else
  PROFILE="$ROOT/config/simulation_lidar_360_wheel.yaml"    # wheel gate only, imu_mode off
  "$HERE/start_live_sim_imu.sh" --no-imu
fi
TRUTH_DIR="$(cat "$ROOT/results/sim_imu_live.latest")"
STARTED=$(date +%s)

finish() {
  trap - EXIT
  "$HERE/stop_live_sim_imu.sh" || true
  RUN="$(/usr/bin/python3 -c "import json;print(json.load(open('$ROOT/run_state.json'))['output'])" 2>/dev/null || true)"
  if [[ -z "$RUN" || ! -d "$RUN" || $(stat -c %Y "$RUN") -lt $STARTED ]]; then
    echo "No mapping run from this session; truth kept in $TRUTH_DIR"; return
  fi
  printf '{"imu": "%s", "noise": "%s", "profile": "%s", "truth_bag": "%s", "synthetic_imu_dir": "%s"}\n' \
    "$IMU" "$NOISE" "$PROFILE" "$TRUTH_DIR/truth_bag" "$TRUTH_DIR" > "$RUN/debug_session.json"
  echo "$RUN" > "$TRUTH_DIR/mapping_run.txt"
  if [[ $EVAL == 1 ]]; then
    echo "Scoring $RUN against truth ..."
    source "$ROOT/scripts/env.sh" >/dev/null
    nice -n 10 /usr/bin/python3 "$HERE/build_sim_imu.py" --truth "$TRUTH_DIR/truth_bag" --noise none \
      --out "$RUN/truth" > "$RUN/truth_build.log" 2>&1 \
      && nice -n 10 /usr/bin/python3 "$HERE/evaluate_against_truth.py" --run "$RUN" --truth "$RUN/truth.npz" \
           --out "$RUN/truth_eval.json" > /dev/null \
      && echo "Truth evaluation: $RUN/truth_eval.json" || echo "Scoring failed, see $RUN/truth_build.log"
  fi
  echo "Debug session: $RUN (truth: $TRUTH_DIR/truth_bag)"
}
trap finish EXIT
# A handler (not SIG_IGN, which children would inherit) lets start_live.sh own Ctrl+C;
# this script then continues with finish().
trap ':' INT TERM
[[ "$IMU" == on ]] && echo ">>> IMU on: keep the rover still for ~5 s after mapping starts."
"$ROOT/start_live.sh" --profile "$PROFILE" --record-input || true
