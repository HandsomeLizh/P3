#!/usr/bin/env bash
set -e
set -o pipefail
set +u

TASK_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TASK_RUNTIME_ROOT="$(cd "${TASK_SCRIPT_DIR}/.." && pwd)"
TASK_VENV="${TASK_RUNTIME_ROOT}/.venv"

if ! command -v uv >/dev/null 2>&1; then
  echo "[ERROR] uv is required. Install it before creating the MPRF environment." >&2
  exit 1
fi

if [[ ! -x "${TASK_VENV}/bin/python" ]]; then
  uv python install 3.10.18
  uv venv --python 3.10.18 "${TASK_VENV}"
fi

TASK_PYTHON="${TASK_VENV}/bin/python"

# MPRF/SONATA's published environment is CUDA 12.4 + PyTorch 2.5.0. Keep it
# isolated from ROS 2 and RoMa. These x86_64 wheels are not for JetPack 5.1;
# an AGX Orin deployment uses this service on a companion GPU computer.
uv pip install --python "${TASK_PYTHON}" \
  --index-url https://download.pytorch.org/whl/cu124 \
  torch==2.5.0 torchvision==0.20.0

uv pip install --python "${TASK_PYTHON}" \
  -r "${TASK_RUNTIME_ROOT}/requirements-runtime.txt"

uv pip install --python "${TASK_PYTHON}" \
  --find-links https://data.pyg.org/whl/torch-2.5.0+cu124.html \
  'torch-scatter==2.1.2+pt25cu124'

uv pip install --python "${TASK_PYTHON}" --no-deps \
  --editable "${TASK_RUNTIME_ROOT}"

"${TASK_PYTHON}" - <<'PY'
import numpy
import scipy
import spconv.pytorch
import torch
import torch_scatter

print(f"[OK] Python/Torch: {torch.__version__}")
print(f"[OK] CUDA runtime: {torch.version.cuda}")
print(f"[OK] CUDA available: {torch.cuda.is_available()}")
print(f"[OK] NumPy/SciPy: {numpy.__version__}/{scipy.__version__}")
PY

echo "[OK] MPRF environment ready: ${TASK_VENV}"

