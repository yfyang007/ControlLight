#!/usr/bin/env bash
set -euo pipefail

# Launch distributed multi-GPU ControlLight training.
#
# Optional overrides:
#   CONFIG=config/train_flux2klein_lora.yaml
#   GPUS=0,1,2,3
#   NUM_PROCESSES=4
#   MAIN_PROCESS_PORT=29571
#   CONDA_ENV=aitoolkit

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"

# shellcheck disable=SC1091
source "${REPO_ROOT}/scripts/project_env.sh"

cd "${REPO_ROOT}"

CONFIG="${CONFIG:-${REPO_ROOT}/config/train_flux2klein_lora.yaml}"
GPUS="${GPUS:-0,1,2,3}"
NUM_PROCESSES="${NUM_PROCESSES:-4}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29571}"

export CUDA_VISIBLE_DEVICES="${GPUS}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
export NO_ALBUMENTATIONS_UPDATE="${NO_ALBUMENTATIONS_UPDATE:-1}"
export AI_TOOLKIT_FS_BARRIER_TIMEOUT_SEC="${AI_TOOLKIT_FS_BARRIER_TIMEOUT_SEC:-7200}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

echo "REPO_ROOT=${REPO_ROOT}"
echo "CONFIG=${CONFIG}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "NUM_PROCESSES=${NUM_PROCESSES}"
echo "MAIN_PROCESS_PORT=${MAIN_PROCESS_PORT}"

accelerate launch \
  --num_processes "${NUM_PROCESSES}" \
  --num_machines 1 \
  --mixed_precision bf16 \
  --main_process_port "${MAIN_PROCESS_PORT}" \
  "${REPO_ROOT}/run.py" "${CONFIG}"
