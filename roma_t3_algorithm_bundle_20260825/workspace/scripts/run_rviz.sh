#!/usr/bin/env bash
set -e
set -o pipefail
set +u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/enter_workspace.sh"

FEATURE_MATCHES="${T3_RVIZ_FEATURE_MATCHES:-false}"
CHECK_ONLY=false
RVIZ_ARGS=()

usage() {
  cat <<'EOF'
Usage: ./scripts/run_rviz.sh [--feature-matches] [--check-config] [-- RVIZ_ARGS...]

  --feature-matches     Display /Car/T2/feature_matching/visualization.
  --no-feature-matches  Keep feature-match rendering disabled (default).
  --check-config        Prepare and validate the selected config without opening RViz.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --feature-matches)
      FEATURE_MATCHES=true
      shift
      ;;
    --no-feature-matches)
      FEATURE_MATCHES=false
      shift
      ;;
    --check-config)
      CHECK_ONLY=true
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      RVIZ_ARGS+=("$@")
      break
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

BASE_CONFIG="${PRACTICE_WS}/src/t3_semantic_mapping/rviz/t3_gridmap_v2.rviz"
SELECTED_CONFIG="${BASE_CONFIG}"
TEMP_CONFIG=""

case "${FEATURE_MATCHES,,}" in
  1|true|yes|on)
    TEMP_CONFIG="$(mktemp --suffix=.rviz t3_gridmap_feature_matches.XXXXXX)"
    python3 "${SCRIPT_DIR}/prepare_rviz_feature_config.py" \
      "${BASE_CONFIG}" "${TEMP_CONFIG}" enabled
    SELECTED_CONFIG="${TEMP_CONFIG}"
    ;;
  0|false|no|off|"")
    ;;
  *)
    echo "T3_RVIZ_FEATURE_MATCHES must be true or false" >&2
    exit 2
    ;;
esac

cleanup() {
  if [[ -n "${TEMP_CONFIG}" ]]; then
    rm -f -- "${TEMP_CONFIG}"
  fi
}
trap cleanup EXIT

if [[ "${CHECK_ONLY}" == true ]]; then
  CHECK_CONFIG="$(mktemp --suffix=.rviz t3_gridmap_check.XXXXXX)"
  python3 "${SCRIPT_DIR}/prepare_rviz_feature_config.py" \
    "${SELECTED_CONFIG}" "${CHECK_CONFIG}" \
    "$(if [[ "${FEATURE_MATCHES,,}" =~ ^(1|true|yes|on)$ ]]; then echo enabled; else echo disabled; fi)"
  rm -f -- "${CHECK_CONFIG}"
  echo "RViz configuration OK; feature_matches=${FEATURE_MATCHES,,}"
  exit 0
fi

rviz2 -d "${SELECTED_CONFIG}" "${RVIZ_ARGS[@]}"
