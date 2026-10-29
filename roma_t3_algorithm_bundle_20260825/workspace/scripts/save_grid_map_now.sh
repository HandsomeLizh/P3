#!/usr/bin/env bash
set -e
set -o pipefail
set +u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/enter_workspace.sh"

ros2 service call \
  /Car/T3/mapping/save_grid_map \
  std_srvs/srv/Trigger "{}"
