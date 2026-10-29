#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
/usr/bin/python3 "$SCRIPT_DIR/democtl.py" start
if [[ $# -gt 0 ]]; then
  /usr/bin/python3 "$SCRIPT_DIR/democtl.py" sensors --host "$1"
fi
