#!/usr/bin/env bash
set -e
set -o pipefail
set +u

TASK_ADAPTER_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TASK_ADAPTER_WORKSPACE="$(cd "${TASK_ADAPTER_SCRIPT_DIR}/.." && pwd)"
source "${TASK_ADAPTER_SCRIPT_DIR}/enter_t1_t2_adapter_environment.sh"

TASK_INTERFACE_REPORT="${1:-${TASK_ADAPTER_WORKSPACE}/t1_t2_interface_report_$(date +%Y%m%d_%H%M%S)}"
if (( $# > 0 )); then
  shift
fi
TASK_INTERFACE_TOPICS=("$@")
if (( ${#TASK_INTERFACE_TOPICS[@]} == 0 )); then
  mapfile -t TASK_INTERFACE_TOPICS < <(
    ros2 topic list 2>/dev/null | \
      sed '/^\/parameter_events$/d; /^\/rosout$/d' | sort -u
  )
fi
if (( ${#TASK_INTERFACE_TOPICS[@]} == 0 )); then
  echo "[ERROR] No live ROS 2 topics were discovered." >&2
  echo "        Start Task 1/Task 2 and check ROS_DOMAIN_ID/network first." >&2
  exit 1
fi
mkdir -p "${TASK_INTERFACE_REPORT}/topics" \
  "${TASK_INTERFACE_REPORT}/interfaces"

{
  echo "generated_at=$(date --iso-8601=seconds)"
  echo "hostname=$(hostname)"
  echo "architecture=$(uname -m)"
  echo "ros_distro=${ROS_DISTRO:-unknown}"
  echo "rmw=${RMW_IMPLEMENTATION:-unknown}"
  printf 'topics='
  printf ' %q' "${TASK_INTERFACE_TOPICS[@]}"
  echo
} >"${TASK_INTERFACE_REPORT}/environment.txt"

ros2 node list >"${TASK_INTERFACE_REPORT}/node_list.txt" 2>&1 || true
ros2 topic list -t >"${TASK_INTERFACE_REPORT}/topic_list_with_types.txt" 2>&1 || true
timeout 20 ros2 doctor --report \
  >"${TASK_INTERFACE_REPORT}/ros2_doctor.txt" 2>&1 || true

for TASK_INTERFACE_TOPIC in "${TASK_INTERFACE_TOPICS[@]}"; do
  TASK_INTERFACE_SAFE_NAME="$(
    sed 's#^/##; s#[/:]#_#g' <<<"${TASK_INTERFACE_TOPIC}"
  )"
  if [[ -z "${TASK_INTERFACE_SAFE_NAME}" ]]; then
    TASK_INTERFACE_SAFE_NAME=root
  fi
  TASK_INTERFACE_INFO="${TASK_INTERFACE_REPORT}/topics/${TASK_INTERFACE_SAFE_NAME}.txt"
  {
    echo "topic=${TASK_INTERFACE_TOPIC}"
    echo
    echo '--- ros2 topic info --verbose ---'
    ros2 topic info --verbose "${TASK_INTERFACE_TOPIC}" || true
  } >"${TASK_INTERFACE_INFO}" 2>&1

  TASK_INTERFACE_TYPES="$(
    ros2 topic type "${TASK_INTERFACE_TOPIC}" 2>/dev/null || true
  )"
  if [[ -z "${TASK_INTERFACE_TYPES}" ]]; then
    echo "[WARN] No active type found for ${TASK_INTERFACE_TOPIC}." >&2
    continue
  fi
  while IFS= read -r TASK_INTERFACE_TYPE; do
    [[ -n "${TASK_INTERFACE_TYPE}" ]] || continue
    TASK_INTERFACE_TYPE_SAFE="$(sed 's#/#_#g' <<<"${TASK_INTERFACE_TYPE}")"
    ros2 interface show "${TASK_INTERFACE_TYPE}" \
      >"${TASK_INTERFACE_REPORT}/interfaces/${TASK_INTERFACE_TYPE_SAFE}.txt" 2>&1 || true
  done <<<"${TASK_INTERFACE_TYPES}"
done

echo "[OK] Interface report: ${TASK_INTERFACE_REPORT}"
echo "Send or inspect the whole directory; it contains no topic payload samples."
