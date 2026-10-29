#!/usr/bin/env bash
set -e
set -o pipefail
set +u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/enter_workspace.sh"

exec ros2 launch t3_semantic_mapping \
  practice_live_precomputed.launch.py "$@"
