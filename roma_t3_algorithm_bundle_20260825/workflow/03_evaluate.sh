#!/usr/bin/env bash
set -Eeuo pipefail

WS="${T3_WS:-/home/yanfa/P3/t3_gridmap_v2_delivery/t3_gridmap_v2_ws}"
EVALUATOR="${WS}/scripts/evaluate_all_metrics.py"

usage() {
  cat <<'EOF'
Usage:
  /home/yanfa/P3/bagtest_workflow/03_evaluate.sh OUTPUT_DIR [evaluator options]

Example:
  /home/yanfa/P3/bagtest_workflow/03_evaluate.sh \
    /home/yanfa/P3/output/roma_bag_full_20260824_213145

The output directory must contain pq_reference.tum or
_internal/reference_pose.tum, or an explicit --reference-tum option must be
supplied. Passive replay-time PQ samples are not accepted because
Float64MultiArray has no Header timestamp.
EOF
}

if [[ $# -lt 1 || "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  [[ $# -ge 1 ]] && exit 0
  exit 2
fi

OUTPUT_DIR="${1%/}"
shift

if [[ ! -d "${OUTPUT_DIR}" ]]; then
  echo "[ERROR] Output directory does not exist: ${OUTPUT_DIR}" >&2
  exit 2
fi
if [[ ! -f "${EVALUATOR}" ]]; then
  echo "[ERROR] Evaluator not found: ${EVALUATOR}" >&2
  exit 2
fi

explicit_reference=false
for argument in "$@"; do
  case "${argument}" in
    --reference-tum|--reference-tum=*|--bag|--bag=*)
      explicit_reference=true
      ;;
  esac
done

POSE_REFERENCE="${OUTPUT_DIR}/_internal/reference_pose.tum"
if [[ ! -f "${OUTPUT_DIR}/pq_reference.tum" && ! -f "${POSE_REFERENCE}" && "${explicit_reference}" != true ]]; then
  cat >&2 <<EOF
[ERROR] Missing authoritative PQ reference:
  ${OUTPUT_DIR}/pq_reference.tum

Do not evaluate this run from _internal/reference_pq_samples.jsonl: raw PQ has
no Header and its replay-time wall clock cannot pair with image sensor stamps.
Extract PQ from the rosbag using its SQLite messages.timestamp, or pass an
explicit --reference-tum file.
EOF
  exit 3
fi

if [[ "${explicit_reference}" != true && -f "${POSE_REFERENCE}" ]]; then
  set -- --reference-tum "${POSE_REFERENCE}" "$@"
fi

set +u
source /opt/ros/humble/setup.bash
source "${WS}/install_gpu/setup.bash"
set -u

echo "[EVAL] output=${OUTPUT_DIR}"
if [[ -f "${OUTPUT_DIR}/pq_reference.tum" ]]; then
  echo "[EVAL] reference=${OUTPUT_DIR}/pq_reference.tum"
elif [[ -f "${POSE_REFERENCE}" ]]; then
  echo "[EVAL] reference=${POSE_REFERENCE}"
fi

python3 "${EVALUATOR}" "${OUTPUT_DIR}" "$@"

echo "[EVAL] report=${OUTPUT_DIR}/all_metrics_report.md"
