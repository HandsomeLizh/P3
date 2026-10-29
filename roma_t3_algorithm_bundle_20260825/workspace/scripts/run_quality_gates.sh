#!/usr/bin/env bash
set -e
set -o pipefail
set +u

TASK_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TASK_WORKSPACE="$(cd "${TASK_SCRIPT_DIR}/.." && pwd)"
TASK_ROMA_ROOT="${TASK_WORKSPACE}/third_party/RoMa"
TASK_MPRF_RUNTIME="${TASK_WORKSPACE}/mprf_runtime"
TASK_STATIC_TEST_ROOT="${TASK_WORKSPACE}/tests/static_contract"

if [[ ! -f "${TASK_ROMA_ROOT}/roma_vo.py" ]]; then
  echo "[ERROR] Frozen RoMa source not found: ${TASK_ROMA_ROOT}/roma_vo.py" >&2
  exit 1
fi

echo "[quality] Build the six self-owned ROS 2 packages."
bash "${TASK_SCRIPT_DIR}/build.sh"
source "${TASK_SCRIPT_DIR}/enter_workspace.sh"
export PYTHONPATH="${TASK_MPRF_RUNTIME}:${PYTHONPATH:-}"

TASK_TEST_PYTHON="$(command -v python3)"
export PYTHONDONTWRITEBYTECODE=1
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1

run_pytest_exact() {
  local expected="$1"
  local label="$2"
  shift 2
  local output
  local result=0
  local collected

  output="$(
    "${TASK_TEST_PYTHON}" -m pytest -q -p no:cacheprovider "$@" 2>&1
  )" || result=$?
  printf '%s\n' "${output}"
  if [[ ${result} -ne 0 ]]; then
    echo "[ERROR] ${label} failed with exit code ${result}." >&2
    exit "${result}"
  fi
  collected="$(
    sed -n 's/^\([0-9][0-9]*\) passed .*/\1/p' <<<"${output}" | tail -n 1
  )"
  if [[ "${collected}" != "${expected}" ]]; then
    echo "[ERROR] ${label} expected ${expected} passing tests," >&2
    echo "        but pytest reported ${collected:-an unrecognized count}." >&2
    exit 1
  fi
}

run_unittest_exact() {
  local expected="$1"
  local label="$2"
  shift 2
  local output
  local result=0
  local collected

  output="$("${TASK_TEST_PYTHON}" -m unittest "$@" 2>&1)" || result=$?
  printf '%s\n' "${output}"
  if [[ ${result} -ne 0 ]]; then
    echo "[ERROR] ${label} failed with exit code ${result}." >&2
    exit "${result}"
  fi
  collected="$(
    sed -n 's/^Ran \([0-9][0-9]*\) tests.*/\1/p' <<<"${output}" | tail -n 1
  )"
  if [[ "${collected}" != "${expected}" || "${output}" == *"skipped="* ]]; then
    echo "[ERROR] ${label} expected ${expected} passing, non-skipped tests." >&2
    echo "        unittest reported ${collected:-an unrecognized count}." >&2
    exit 1
  fi
}

echo "[quality] Static behavior contracts (43 tests)."
run_pytest_exact 43 "Static behavior contracts" \
  "${TASK_STATIC_TEST_ROOT}/test_ba_contract_static.py" \
  "${TASK_STATIC_TEST_ROOT}/test_gridmap_rebuild_contract_static.py" \
  "${TASK_STATIC_TEST_ROOT}/test_loop_closing_contract_static.py" \
  "${TASK_STATIC_TEST_ROOT}/test_map_transfer_contract_static.py" \
  "${TASK_STATIC_TEST_ROOT}/test_relocalization_contract_static.py" \
  "${TASK_STATIC_TEST_ROOT}/test_tracking_failure_guard_static.py"

echo "[quality] Frozen RoMa numerical regression (29 tests)."
run_pytest_exact 29 "Frozen RoMa numerical regression" \
  "${TASK_ROMA_ROOT}/tests/test_stereo_ba.py" \
  "${TASK_ROMA_ROOT}/tests/test_loop_closing.py" \
  "${TASK_ROMA_ROOT}/tests/test_relocalization.py" \
  "${TASK_ROMA_ROOT}/tests/test_tracking_status.py" \
  "${TASK_ROMA_ROOT}/tests/test_keyframe_policy.py" \
  "${TASK_ROMA_ROOT}/tests/test_mapping_contract.py"

echo "[quality] Self-owned ROS, localization, mapping and tooling tests (270 tests)."
run_unittest_exact 270 "Self-owned regression" discover \
  -s "${TASK_WORKSPACE}/scripts" -p 'test_*.py'

echo "[quality] Isolated MPRF protocol, storage and geometry tests (11 tests)."
run_unittest_exact 11 "MPRF runtime regression" discover \
  -s "${TASK_MPRF_RUNTIME}/tests" -p 'test_*.py'

echo "[quality] Dataset-format regression (7 tests)."
run_pytest_exact 7 "Dataset-format regression" \
  "${TASK_WORKSPACE}/src/t1_dataset_sim/test/test_dataset_formats.py"

echo "[quality] Task 1/Task 2 adapter field-path regression (8 tests)."
run_pytest_exact 8 "Upstream adapter regression" \
  "${TASK_WORKSPACE}/src/t3_upstream_adapters/test/test_field_path.py"

mapfile -d '' TASK_PYTHON_SOURCES < <(
  find "${TASK_WORKSPACE}/src" "${TASK_WORKSPACE}/scripts" \
    "${TASK_MPRF_RUNTIME}/mprf_runtime" \
    "${TASK_MPRF_RUNTIME}/tests" \
    "${TASK_MPRF_RUNTIME}/scripts" \
    -type f -name '*.py' -print0
)
echo "[quality] Pyflakes on ${#TASK_PYTHON_SOURCES[@]} Python files."
"${TASK_TEST_PYTHON}" -m pyflakes "${TASK_PYTHON_SOURCES[@]}"

TASK_COMPILE_CACHE="$(mktemp -d /tmp/t3_quality_compile.XXXXXX)"
cleanup_compile_cache() {
  if [[ -n "${TASK_COMPILE_CACHE:-}" && \
        "${TASK_COMPILE_CACHE}" == /tmp/t3_quality_compile.* && \
        -d "${TASK_COMPILE_CACHE}" ]]; then
    rm -rf -- "${TASK_COMPILE_CACHE}"
  fi
}
trap cleanup_compile_cache EXIT

echo "[quality] Compile all Python sources without writing into the workspace."
PYTHONPYCACHEPREFIX="${TASK_COMPILE_CACHE}" \
  "${TASK_TEST_PYTHON}" -m compileall -q \
  "${TASK_WORKSPACE}/src" "${TASK_WORKSPACE}/scripts" \
  "${TASK_MPRF_RUNTIME}/mprf_runtime" \
  "${TASK_MPRF_RUNTIME}/tests" \
  "${TASK_MPRF_RUNTIME}/scripts"

echo "[quality] Check all shell entry points."
while IFS= read -r -d '' TASK_SHELL_SOURCE; do
  bash -n "${TASK_SHELL_SOURCE}"
done < <(
  find "${TASK_WORKSPACE}/scripts" "${TASK_MPRF_RUNTIME}/scripts" \
    -maxdepth 1 -type f -name '*.sh' -print0
)

echo "[quality] Verify the MPRF source bundle and model file sizes."
"${TASK_TEST_PYTHON}" "${TASK_MPRF_RUNTIME}/scripts/verify_assets.py"

check_file_size() {
  local expected_bytes="$1"
  local target="$2"
  local actual_bytes
  actual_bytes="$(wc -c <"${target}" | tr -d '[:space:]')"
  if [[ "${actual_bytes}" != "${expected_bytes}" ]]; then
    echo "[ERROR] Frozen file size changed: ${target}" >&2
    echo "        expected=${expected_bytes} bytes" >&2
    echo "        actual=${actual_bytes} bytes" >&2
    exit 1
  fi
}

echo "[quality] Check the two formal messages and frozen RoMa boundary."
check_file_size \
  707 \
  "${TASK_WORKSPACE}/src/t3_interfaces/msg/IncrementalSemanticMap.msg"
check_file_size \
  1462 \
  "${TASK_WORKSPACE}/src/t3_interfaces/msg/SemanticMapBatch.msg"
check_file_size \
  131228 \
  "${TASK_ROMA_ROOT}/roma_vo.py"

cleanup_compile_cache
trap - EXIT

echo "[OK] Quality gates passed: core build 6/6; tests 368/368; static analysis,"
echo "     compilation, shell syntax and frozen file-size checks passed."
