#!/usr/bin/env bash
set -e
set -o pipefail
set +u

TASK_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TASK_RUNTIME_ROOT="$(cd "${TASK_SCRIPT_DIR}/.." && pwd)"
TASK_PYTHON="${TASK_RUNTIME_ROOT}/.venv/bin/python"

if [[ ! -x "${TASK_PYTHON}" ]]; then
  TASK_PYTHON="$(command -v python3)"
fi

export PYTHONDONTWRITEBYTECODE=1
PYTHONPATH="${TASK_RUNTIME_ROOT}" \
  "${TASK_PYTHON}" -m unittest discover \
  -s "${TASK_RUNTIME_ROOT}/tests" -v

