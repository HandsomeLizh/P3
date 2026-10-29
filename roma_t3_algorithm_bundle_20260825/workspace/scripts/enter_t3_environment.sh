#!/usr/bin/env bash
# Compatibility alias. The project now has one system-Python environment.

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "[ERROR] Use: source $0" >&2
  exit 1
fi

if [[ $# -gt 0 ]]; then
  echo "[t3] Environment modes were removed; using system Python." >&2
fi

T3_ENTRY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${T3_ENTRY_DIR}/enter_workspace.sh"
unset T3_ENTRY_DIR
