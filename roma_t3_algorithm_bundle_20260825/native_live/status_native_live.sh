#!/usr/bin/env bash
set -u

ROOT="${T3_BUNDLE_ROOT:-/home/yanfa/P3/roma_t3_algorithm_bundle_20260825}"
RUN_DIR="$ROOT/run_native"

for name in mapper dashboard reference_recorder; do
  pid_file="$RUN_DIR/$name.pid"
  if [[ -f "$pid_file" ]] && kill -0 "$(<"$pid_file")" 2>/dev/null; then
    printf '%-20s RUNNING  PID %s\n' "$name" "$(<"$pid_file")"
  else
    printf '%-20s STOPPED\n' "$name"
  fi
done

if [[ -f "$RUN_DIR/current_output.txt" ]]; then
  output_dir="$(<"$RUN_DIR/current_output.txt")"
  printf 'output: %s\n' "$output_dir"
  for log in mapper dashboard reference_recorder; do
    printf '\n== %s.log ==\n' "$log"
    tail -n 5 "$output_dir/logs/$log.log" 2>/dev/null || true
  done
fi
