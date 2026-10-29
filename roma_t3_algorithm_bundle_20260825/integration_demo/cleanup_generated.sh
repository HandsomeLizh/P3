#!/usr/bin/env bash
# Only known regenerable caches and proven duplicate/broken checkpoint files.
set -euo pipefail
P3_ROOT=/home/yanfa/P3
BUNDLE_ROOT=/home/yanfa/P3/roma_t3_algorithm_bundle_20260825
OLD_CHECKPOINTS="$P3_ROOT/models/torch/hub/checkpoints"
CURRENT_CHECKPOINTS="$BUNDLE_ROOT/models/torch/hub/checkpoints"

for name in dinov2_vitl14_pretrain.pth roma_outdoor.pth; do
  old="$OLD_CHECKPOINTS/$name"
  current="$CURRENT_CHECKPOINTS/$name"
  [[ -L "$old" ]] && continue
  [[ -f "$old" && -f "$current" ]]
  [[ "$(realpath "$current")" == "$CURRENT_CHECKPOINTS/$name" ]]
  [[ "$(realpath "$old")" == "$OLD_CHECKPOINTS/$name" ]]
  if [[ "$name" == dinov2_vitl14_pretrain.pth ]]; then
    expected=d5383ea8f4877b2472eb973e0fd72d557c7da5d3611bd527ceeb1d7162cbf428
    [[ "$(sha256sum "$current" | cut -d' ' -f1)" == "$expected" ]]
    unzip -t "$current" >/dev/null
    if unzip -t "$old" >/dev/null 2>&1; then
      echo "Retaining valid nonduplicate DINO checkpoint: $old"
      continue
    fi
  else
    cmp -s "$old" "$current"
  fi
  bytes="$(stat -c %s "$old")"
  # Atomic replacement. Complete checkpoint bytes remain at the target.
  ln -s "$current" "$old.demo-link"
  mv -T "$old.demo-link" "$old"
  printf 'Replaced obsolete checkpoint with shared link: %s (%s bytes reclaimed)\n' "$old" "$bytes"
done

while IFS= read -r -d '' directory; do
  resolved="$(realpath -e "$directory")"
  case "$resolved" in
    /home/yanfa/P3/*/__pycache__|/home/yanfa/P3/*/.pytest_cache)
      [[ ! -L "$directory" ]]
      rm -rf -- "$resolved"
      printf 'Removed regenerable cache: %s\n' "$resolved"
      ;;
    *) echo "Refusing unexpected cache path: $resolved" >&2; exit 1 ;;
  esac
done < <(find "$P3_ROOT" -type d \( -name __pycache__ -o -name .pytest_cache \) -prune -print0)

for name in apt_lists apt_cache apt_sources_empty rviz_packages; do
  directory="$BUNDLE_ROOT/integration_demo/$name"
  [[ -d "$directory" ]] || continue
  [[ "$(realpath -e "$directory")" == "$directory" ]]
  rm -rf -- "$directory"
  printf 'Removed completed package-download cache: %s\n' "$directory"
done
