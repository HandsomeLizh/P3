#!/usr/bin/env bash
set -e
set -o pipefail
set +u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/enter_workspace.sh"

ros2 run t2_basic_semantics generate_semantic_dataset \
  --data-dir "${1:-$HOME/RoMa/data/data1223}"
