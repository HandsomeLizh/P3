#!/usr/bin/env bash
set -e
set -o pipefail
set +u

TASK_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TASK_WORKSPACE_ROOT="$(cd "${TASK_SCRIPT_DIR}/.." && pwd)"
TASK_MODEL_ROOT="${1:-${TASK_WORKSPACE_ROOT}/models}"
TASK_EXPECTED_PROFILE="${T3_EXPECT_MODEL_PROFILE:-}"

check_file_size() {
  local TASK_RELATIVE_PATH="$1"
  local TASK_EXPECTED_BYTES="$2"
  local TASK_ABSOLUTE_PATH="${TASK_MODEL_ROOT}/${TASK_RELATIVE_PATH}"
  local TASK_ACTUAL_BYTES

  if [[ ! -f "${TASK_ABSOLUTE_PATH}" ]]; then
    echo "[ERROR] Model file is missing: ${TASK_RELATIVE_PATH}" >&2
    exit 1
  fi
  TASK_ACTUAL_BYTES="$(wc -c <"${TASK_ABSOLUTE_PATH}" | tr -d '[:space:]')"
  if [[ "${TASK_ACTUAL_BYTES}" != "${TASK_EXPECTED_BYTES}" ]]; then
    echo "[ERROR] Model file size mismatch: ${TASK_RELATIVE_PATH}" >&2
    echo "        expected=${TASK_EXPECTED_BYTES} bytes" >&2
    echo "        actual=${TASK_ACTUAL_BYTES} bytes" >&2
    exit 1
  fi
}

TASK_OPTIONAL_LOOP_WEIGHTS=(
  mprf/Weights/finetuned_dinov2_v3.pth
  mprf/Weights/pretrained_salad.ckpt
  mprf/Weights/retrained_salad.ckpt
  sonata/sonata.pth
)
TASK_PRESENT_LOOP_WEIGHTS=0

for TASK_RELATIVE_WEIGHT in "${TASK_OPTIONAL_LOOP_WEIGHTS[@]}"; do
  if [[ -f "${TASK_MODEL_ROOT}/${TASK_RELATIVE_WEIGHT}" ]]; then
    TASK_PRESENT_LOOP_WEIGHTS=$((TASK_PRESENT_LOOP_WEIGHTS + 1))
  fi
done

if (( TASK_PRESENT_LOOP_WEIGHTS == ${#TASK_OPTIONAL_LOOP_WEIGHTS[@]} )); then
  TASK_MODEL_PROFILE=full
elif (( TASK_PRESENT_LOOP_WEIGHTS == 0 )); then
  TASK_MODEL_PROFILE=roma_only
else
  echo "[ERROR] Loop-closure model bundle is partial:" >&2
  for TASK_RELATIVE_WEIGHT in "${TASK_OPTIONAL_LOOP_WEIGHTS[@]}"; do
    if [[ -f "${TASK_MODEL_ROOT}/${TASK_RELATIVE_WEIGHT}" ]]; then
      echo "        present: ${TASK_RELATIVE_WEIGHT}" >&2
    else
      echo "        missing: ${TASK_RELATIVE_WEIGHT}" >&2
    fi
  done
  exit 1
fi

if [[ -n "${TASK_EXPECTED_PROFILE}" && \
      "${TASK_EXPECTED_PROFILE}" != "${TASK_MODEL_PROFILE}" ]]; then
  echo "[ERROR] Expected model profile ${TASK_EXPECTED_PROFILE}," >&2
  echo "        detected ${TASK_MODEL_PROFILE}." >&2
  exit 1
fi

check_file_size torch/hub/checkpoints/roma_outdoor.pth 445647516
check_file_size torch/hub/checkpoints/dinov2_vitl14_pretrain.pth 1217586395

if [[ "${TASK_MODEL_PROFILE}" == "full" ]]; then
  check_file_size mprf/Weights/finetuned_dinov2_v3.pth 346402202
  check_file_size mprf/Weights/pretrained_salad.ckpt 352040378
  check_file_size mprf/Weights/retrained_salad.ckpt 356232514
  check_file_size sonata/sonata.pth 434008287
fi

if [[ "${TASK_MODEL_PROFILE}" == "roma_only" ]]; then
  echo "[OK] Model profile: roma_only (RoMa/DINOv2 present; MPRF/SALAD/SONATA weights intentionally absent)"
else
  echo "[OK] Model profile: full (RoMa and MPRF/SALAD/SONATA weights)"
fi
