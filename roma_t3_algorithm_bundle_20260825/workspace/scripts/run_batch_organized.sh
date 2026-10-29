#!/usr/bin/env bash
set -e
set -o pipefail
set +u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 /path/to/batch_organized/SEQUENCE [launch_arg:=value ...]" >&2
  exit 2
fi

DATA_DIR="$1"
shift

if [[ ! -f "${DATA_DIR}/images/timestamps.tsv" ]]; then
  BATCH_ROOT="${DATA_DIR}"
  DATA_DIR=""
  for CANDIDATE in "${BATCH_ROOT}"/*; do
    if [[ -f "${CANDIDATE}/images/timestamps.tsv" ]]; then
      DATA_DIR="${CANDIDATE}"
      break
    fi
  done
  if [[ -z "${DATA_DIR}" ]]; then
    echo "No batch_organized sequence found under: ${BATCH_ROOT}" >&2
    exit 2
  fi
  echo "[batch_organized] Selected first sequence: ${DATA_DIR}"
fi

WORKSPACE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ALGORITHM_ROOT="${T3_ALGORITHM_ROOT:-${WORKSPACE_ROOT}/third_party/RoMa}"
source "${SCRIPT_DIR}/enter_workspace.sh"

exec ros2 launch t3_semantic_mapping \
  practice_live_online_semantics.launch.py \
  data_dir:="${DATA_DIR}" \
  algorithm_root:="${ALGORITHM_ROOT}" \
  "$@"
