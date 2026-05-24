#!/usr/bin/env bash
set -euo pipefail

# Start the standard local multi-GPU training job.
#
# Optional overrides:
#   REPO_ROOT=/path/to/controllight
#   CONDA_ENV=aitoolkit
#   CONDA_SH=/path/to/conda.sh
#   RUN_NAME=my_experiment
#   CONFIG=config/train_flux2klein_lora.yaml
#   CUDA_VISIBLE_DEVICES=0,1,2,3
#   NUM_PROCESSES=4
#   MAIN_PROCESS_PORT=30000

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
RUN_NAME="${RUN_NAME:-controllight_lora_train}"
CONDA_ENV="${CONDA_ENV:-aitoolkit}"
CONFIG="${CONFIG:-${REPO_ROOT}/config/train_flux2klein_lora.yaml}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
NUM_PROCESSES="${NUM_PROCESSES:-4}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-30000}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/output/${RUN_NAME}}"

# shellcheck disable=SC1091
source "${REPO_ROOT}/scripts/project_env.sh"

cd "${REPO_ROOT}"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
export NO_ALBUMENTATIONS_UPDATE="${NO_ALBUMENTATIONS_UPDATE:-1}"
export AI_TOOLKIT_FS_BARRIER_TIMEOUT_SEC="${AI_TOOLKIT_FS_BARRIER_TIMEOUT_SEC:-7200}"
export AI_TOOLKIT_RUN_ID="${AI_TOOLKIT_RUN_ID:-${RUN_NAME}_formal_$(date +%s)}"

mkdir -p "${OUTPUT_DIR}"

CONFIG="${CONFIG}" \
GPUS="${CUDA_VISIBLE_DEVICES}" \
NUM_PROCESSES="${NUM_PROCESSES}" \
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT}" \
bash "${REPO_ROOT}/scripts/train/train_multigpu.sh" \
  2>&1 | tee -a "${OUTPUT_DIR}/train_4gpu.log"
