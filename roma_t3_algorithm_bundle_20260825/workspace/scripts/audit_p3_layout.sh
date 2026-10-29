#!/usr/bin/env bash
# Read-only inventory for the local or remote P3 directory. It never deletes.
set -e
set -o pipefail

T3_AUDIT_ROOT="${1:-${HOME}/P3}"
if [[ ! -d "${T3_AUDIT_ROOT}" ]]; then
  echo "[ERROR] Directory not found: ${T3_AUDIT_ROOT}" >&2
  exit 2
fi
T3_AUDIT_ROOT="$(realpath "${T3_AUDIT_ROOT}")"

echo "P3 audit root: ${T3_AUDIT_ROOT}"
echo
echo "== Top-level disk usage =="
du -h --max-depth=1 "${T3_AUDIT_ROOT}" 2>/dev/null | sort -h

echo
echo "== Top-level entries =="
find "${T3_AUDIT_ROOT}" -mindepth 1 -maxdepth 1 \
  -printf '%y\t%TY-%Tm-%Td %TH:%TM\t%p\n' | sort -k3

echo
echo "== Files larger than 100 MiB =="
find "${T3_AUDIT_ROOT}" -type f -size +100M \
  -printf '%s\t%TY-%Tm-%Td %TH:%TM\t%p\n' | sort -nr

echo
echo "== Generated and temporary candidates =="
find "${T3_AUDIT_ROOT}" \
  \( -type d \( \
       -name '__pycache__' -o -name '.pytest_cache' -o \
       -name build -o -name install -o -name log -o \
       -name '.codex_tmp' -o -name '.wpp_backup' -o \
       -name 'tmp*' \
     \) -o \
     -type f \( \
       -name '*.pyc' -o -name '*.inspect.ndjson' -o \
       -name '*.bak' -o -name '*~' \
     \) \) -print | sort

echo
echo "== Historical archive candidates =="
find "${T3_AUDIT_ROOT}" -type f \
  \( -name '*.before_*.tar.gz' -o -name '*candidate*.tar.gz' \) \
  -printf '%s\t%TY-%Tm-%Td\t%p\n' | sort -nr

echo
echo "[OK] Read-only audit complete. Review every target before removal."
