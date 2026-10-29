#!/usr/bin/env bash
set -e
set -o pipefail
set +u

TASK_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TASK_RUNTIME_ROOT="$(cd "${TASK_SCRIPT_DIR}/.." && pwd)"
TASK_PYTHON="${TASK_RUNTIME_ROOT}/.venv/bin/python"

if [[ ! -x "${TASK_PYTHON}" ]]; then
  echo "[ERROR] MPRF environment is absent." >&2
  echo "        Run ${TASK_SCRIPT_DIR}/create_environment.sh first." >&2
  exit 1
fi

exec "${TASK_PYTHON}" -m mprf_runtime.service "$@"

