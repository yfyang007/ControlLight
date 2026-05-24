#!/usr/bin/env bash
set -euo pipefail

# Create a local symlinked dataset view for ControlLight training.
#
# Optional overrides:
#   REPO_ROOT=/path/to/controllight
#   DATASET_NAME=flux2klein_alpha_interp5_20260501_unified_edgeexp
#   SRC_ROOT=/path/to/source_dataset
#   DST_ROOT=/path/to/local_dataset_view

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
DATASET_NAME="${DATASET_NAME:-flux2klein_alpha_interp5_20260501_unified_edgeexp}"
SRC_ROOT="${SRC_ROOT:-${REPO_ROOT}/dataset_sources/${DATASET_NAME}}"
DST_ROOT="${DST_ROOT:-${REPO_ROOT}/datasets/${DATASET_NAME}}"

if [[ ! -d "${SRC_ROOT}" ]]; then
  echo "Source dataset directory does not exist: ${SRC_ROOT}" >&2
  echo "Set SRC_ROOT=/path/to/source_dataset before running this script." >&2
  exit 1
fi

mkdir -p "${DST_ROOT}"

for name in \
  control \
  mask_normrgb_l01 mask_normrgb_l02 mask_normrgb_l03 mask_normrgb_l04 mask_normrgb_l05 \
  target_l01 target_l02 target_l03 target_l04 target_l05

do
  ln -sfn "${SRC_ROOT}/${name}" "${DST_ROOT}/${name}"
done

echo "Linked ControlLight dataset"
echo "  source: ${SRC_ROOT}"
echo "  target: ${DST_ROOT}"
