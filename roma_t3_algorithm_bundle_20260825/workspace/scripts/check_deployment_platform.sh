#!/usr/bin/env bash
set -e
set -o pipefail
set +u

# Read-only preflight for the two supported roles:
#   orin : project integration target from the interface coordination document
#   dev  : x86_64 Ubuntu 22.04 / ROS 2 Humble development workstation
TASK_ROLE="${1:-auto}"
if [[ "${TASK_ROLE}" != "auto" && "${TASK_ROLE}" != "orin" && \
      "${TASK_ROLE}" != "dev" ]]; then
  echo "Usage: $0 [auto|orin|dev]" >&2
  exit 2
fi

TASK_ARCH="$(uname -m)"
TASK_OS_ID="unknown"
TASK_OS_VERSION="unknown"
if [[ -r /etc/os-release ]]; then
  # shellcheck disable=SC1091
  source /etc/os-release
  TASK_OS_ID="${ID:-unknown}"
  TASK_OS_VERSION="${VERSION_ID:-unknown}"
fi

if [[ "${TASK_ROLE}" == "auto" ]]; then
  if [[ "${TASK_ARCH}" == "aarch64" ]]; then
    TASK_ROLE=orin
  else
    TASK_ROLE=dev
  fi
fi

TASK_ERRORS=0
TASK_WARNINGS=0
task_error() {
  echo "[ERROR] $*" >&2
  TASK_ERRORS=$((TASK_ERRORS + 1))
}
task_warn() {
  echo "[WARN] $*" >&2
  TASK_WARNINGS=$((TASK_WARNINGS + 1))
}
task_ok() {
  echo "[OK] $*"
}

echo "role=${TASK_ROLE}"
echo "architecture=${TASK_ARCH}"
echo "os=${TASK_OS_ID} ${TASK_OS_VERSION}"
echo "kernel=$(uname -r)"

TASK_DEVICE_MODEL="unknown"
if [[ -r /proc/device-tree/model ]]; then
  TASK_DEVICE_MODEL="$(tr -d '\000' </proc/device-tree/model)"
fi
echo "device_model=${TASK_DEVICE_MODEL}"

TASK_L4T_RELEASE=""
TASK_JETPACK_VERSION=""
if [[ -r /etc/nv_tegra_release ]]; then
  TASK_L4T_RELEASE="$(head -n 1 /etc/nv_tegra_release)"
  echo "l4t=${TASK_L4T_RELEASE}"
elif command -v dpkg-query >/dev/null 2>&1; then
  TASK_JETPACK_VERSION="$(
    dpkg-query -W -f='${Version}' nvidia-jetpack 2>/dev/null || true
  )"
  echo "jetpack_package=${TASK_JETPACK_VERSION:-not-installed}"
else
  echo "jetpack=not-detected"
fi

if [[ "${TASK_ROLE}" == "orin" ]]; then
  [[ "${TASK_ARCH}" == "aarch64" ]] || \
    task_error "The integration target must be aarch64, not ${TASK_ARCH}."
  [[ "${TASK_OS_ID}" == "ubuntu" && "${TASK_OS_VERSION}" == "20.04" ]] || \
    task_error "The current interface document locks Ubuntu 20.04 on Orin."
  if [[ "${TASK_DEVICE_MODEL,,}" != *"orin"* ]]; then
    task_warn "The device model does not identify an NVIDIA Orin."
  fi
  if [[ -n "${TASK_L4T_RELEASE}" ]]; then
    if [[ "${TASK_L4T_RELEASE}" != *"R35 (release)"* || \
          "${TASK_L4T_RELEASE}" != *"REVISION: 3.1"* ]]; then
      task_error "The interface baseline is JetPack 5.1.1 / L4T 35.3.1."
    fi
  elif [[ "${TASK_JETPACK_VERSION}" != 5.1.1* ]]; then
    task_error "JetPack 5.1.1 / L4T 35.3.1 was not detected."
  fi
else
  [[ "${TASK_ARCH}" == "x86_64" ]] || \
    task_warn "The documented development baseline is x86_64."
  [[ "${TASK_OS_ID}" == "ubuntu" && "${TASK_OS_VERSION}" == "22.04" ]] || \
    task_warn "The clean Humble development baseline is Ubuntu 22.04."
fi

TASK_ROS_SETUP="${ROS_SETUP_FILE:-}"
if [[ -z "${TASK_ROS_SETUP}" && -f /opt/ros/humble/setup.bash ]]; then
  TASK_ROS_SETUP=/opt/ros/humble/setup.bash
fi
if [[ -z "${TASK_ROS_SETUP}" || ! -f "${TASK_ROS_SETUP}" ]]; then
  task_error "A ROS 2 Humble setup.bash was not found."
else
  unset AMENT_PREFIX_PATH || true
  unset CMAKE_PREFIX_PATH || true
  unset COLCON_PREFIX_PATH || true
  # shellcheck disable=SC1090
  source "${TASK_ROS_SETUP}"
  echo "ros_setup=${TASK_ROS_SETUP}"
  echo "ros_distro=${ROS_DISTRO:-unknown}"
  [[ "${ROS_DISTRO:-}" == "humble" ]] || \
    task_error "The project interface requires ROS_DISTRO=humble."
fi

echo "rmw=${RMW_IMPLEMENTATION:-not-set}"
if [[ "${RMW_IMPLEMENTATION:-}" != "rmw_cyclonedds_cpp" ]]; then
  task_warn "Set RMW_IMPLEMENTATION=rmw_cyclonedds_cpp before integration."
fi

TASK_PYTHON="${ROS_PYTHON_EXECUTABLE:-/usr/bin/python3}"
if [[ ! -x "${TASK_PYTHON}" ]]; then
  task_error "Python executable is missing: ${TASK_PYTHON}"
else
  echo "python=${TASK_PYTHON} ($("${TASK_PYTHON}" --version 2>&1))"
  TASK_PYTHON_INSPECT_STATUS=0
  "${TASK_PYTHON}" - <<'PY' || TASK_PYTHON_INSPECT_STATUS=$?
import sys

print("python_executable=", sys.executable, sep="")
for name in ("rclpy", "cv2", "torch"):
    try:
        module = __import__(name)
        version = getattr(module, "__version__", "available")
        print(f"{name}={version}")
        if name == "torch":
            print(f"torch_cuda={module.version.cuda}")
            print(f"torch_cuda_available={module.cuda.is_available()}")
    except Exception as exception:
        print(f"{name}=MISSING ({type(exception).__name__}: {exception})")
PY
  if (( TASK_PYTHON_INSPECT_STATUS != 0 )); then
    task_error "Python environment inspection failed with status ${TASK_PYTHON_INSPECT_STATUS}."
  fi
  for TASK_REQUIRED_MODULE in rclpy cv2; do
    if ! "${TASK_PYTHON}" -c "import ${TASK_REQUIRED_MODULE}" >/dev/null 2>&1; then
      task_error "Required Python module is missing: ${TASK_REQUIRED_MODULE}."
    fi
  done
  if ! "${TASK_PYTHON}" -c 'import torch' >/dev/null 2>&1; then
    task_warn "PyTorch is missing; LIO-only can run, but the complete VO path cannot."
  elif ! "${TASK_PYTHON}" -c \
      'import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)' \
      >/dev/null 2>&1; then
    task_warn "PyTorch cannot use CUDA; validate the Jetson wheel before enabling VO."
  fi
fi

if [[ "${TASK_ROLE}" == "orin" ]]; then
  task_warn "JetPack 5.1.1, Humble, and the current RoMa package metadata have a known version conflict; use the project golden image/wheelhouse and run a native RoMa test."
fi

echo "errors=${TASK_ERRORS}"
echo "warnings=${TASK_WARNINGS}"
if (( TASK_ERRORS > 0 )); then
  exit 1
fi
task_ok "Platform preflight completed; warnings still require review."
