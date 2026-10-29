#!/usr/bin/env bash
set -Eeo pipefail

ROOT="${T3_BUNDLE_ROOT:-/home/yanfa/P3/roma_t3_algorithm_bundle_20260825}"
export T3_LEFT_TOPIC="/Car/T5/Cam_Left/image_raw/color/thumb"
export T3_RIGHT_TOPIC="/Car/T5/Cam_Right/image_raw/color/thumb"
export T3_CALIB_PATH="$ROOT/config/calibration_spreadsheets_parallel_24p9_quarter.yaml"

exec "$ROOT/native_live/run_native_live.sh"
