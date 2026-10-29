#!/usr/bin/env bash
set -Eeuo pipefail

# Stable entry point inside t3-x86-gpu.
exec /home/yanfa/P3/run_roma_bagtest.sh "$@"

