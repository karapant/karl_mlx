#!/usr/bin/env bash
# Clean up generated/downloaded files. Leaves source scripts untouched.
# Usage: ./cleanup.sh [--all]
#   --all  Also remove cloned karl/ repo and downloaded model weights
set -euo pipefail
cd "$(dirname "$0")"

all=false
[[ "${1:-}" == "--all" ]] && all=true

dirs=(output_karl_mlx output_karl_mlx_finetune output_karl_vanilla
      results results_network sample_images eval_images __pycache__)

for d in "${dirs[@]}"; do
  [[ -d "$d" ]] && echo "Removing $d/" && rm -rf "$d"
done

if $all; then
  [[ -d karl ]] && echo "Removing karl/" && rm -rf karl
  for f in *.ckpt *.safetensors; do
    [[ -f "$f" ]] && echo "Removing $f" && rm -f "$f"
  done
fi

echo "Done."
