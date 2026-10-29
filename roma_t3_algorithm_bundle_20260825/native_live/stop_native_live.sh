#!/usr/bin/env bash
set -u

ROOT="${T3_BUNDLE_ROOT:-/home/yanfa/P3/roma_t3_algorithm_bundle_20260825}"
RUN_DIR="$ROOT/run_native"

stop_group() {
  local name="$1"
  local timeout_sec="$2"
  local pid_file="$RUN_DIR/$name.pid"
  local pid
  local waited=0
  [[ -f "$pid_file" ]] || return 0
  pid="$(<"$pid_file")"
  if [[ ! "$pid" =~ ^[0-9]+$ ]] || ! kill -0 "$pid" 2>/dev/null; then
    rm -f "$pid_file"
    return 0
  fi

  printf 'Stopping %s (PID %s)...\n' "$name" "$pid"
  kill -INT -- "-$pid" 2>/dev/null || kill -INT "$pid" 2>/dev/null || true
  while kill -0 "$pid" 2>/dev/null && (( waited < timeout_sec )); do
    sleep 1
    ((waited += 1))
  done
  if kill -0 "$pid" 2>/dev/null; then
    printf '%s did not exit in %ss; sending TERM.\n' "$name" "$timeout_sec" >&2
    kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
    sleep 2
  fi
  rm -f "$pid_file"
}

# The mapper gets the longest graceful window so final maps and manifests can flush.
stop_group mapper 300
stop_group reference_recorder 30
stop_group dashboard 30

if [[ -f "$RUN_DIR/current_output.txt" ]]; then
  printf 'Results retained in: %s\n' "$(<"$RUN_DIR/current_output.txt")"
fi
