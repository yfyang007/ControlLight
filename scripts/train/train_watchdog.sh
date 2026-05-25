#!/usr/bin/env bash
set -euo pipefail

# Keep the standard ControlLight training run alive and resume from the latest checkpoint.
#
# Optional overrides:
#   REPO_ROOT=/path/to/controllight
#   CONDA_ENV=controlight
#   CONDA_SH=/path/to/conda.sh
#   CONFIG=config/train_flux2klein_lora.yaml
#   RUN_NAME=my_experiment
#   GPUS=0,1,2,3
#   NUM_PROCESSES=4
#   PORT_BASE=30000
#   STOP_STEP=6000
#   SAVE_EVERY=500

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
CONDA_ENV="${CONDA_ENV:-controlight}"
CONFIG="${CONFIG:-${REPO_ROOT}/config/train_flux2klein_lora.yaml}"
RUN_NAME="${RUN_NAME:-controllight_lora_train}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${REPO_ROOT}/output/${RUN_NAME}}"
CHECKPOINT_PREFIX="${CHECKPOINT_PREFIX:-${RUN_NAME}}"
GENERATED_CONFIG="${GENERATED_CONFIG:-/tmp/controllight_train_resume_latest.yaml}"
PROCESS_REGEX="${PROCESS_REGEX:-run\.py .*(${RUN_NAME}|controllight).*\.ya?ml}"
GPUS="${GPUS:-0,1,2,3}"
NUM_PROCESSES="${NUM_PROCESSES:-4}"
PORT_BASE="${PORT_BASE:-30000}"
STOP_STEP="${STOP_STEP:-6000}"
SAVE_EVERY="${SAVE_EVERY:-500}"
POLL_SECONDS="${POLL_SECONDS:-30}"
COOLDOWN_SECONDS="${COOLDOWN_SECONDS:-20}"
TRAIN_LOG="${TRAIN_LOG:-${CHECKPOINT_DIR}/train_4gpu.log}"
WATCHDOG_LOG="${WATCHDOG_LOG:-${CHECKPOINT_DIR}/watchdog.log}"

# shellcheck disable=SC1091
source "${REPO_ROOT}/scripts/project_env.sh"

cd "${REPO_ROOT}"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

python "${REPO_ROOT}/scripts/train/watchdog_resume.py" \
  --workdir "${REPO_ROOT}" \
  --template-config "${CONFIG}" \
  --generated-config "${GENERATED_CONFIG}" \
  --checkpoint-dir "${CHECKPOINT_DIR}" \
  --checkpoint-prefix "${CHECKPOINT_PREFIX}" \
  --process-regex "${PROCESS_REGEX}" \
  --gpus "${GPUS}" \
  --num-processes "${NUM_PROCESSES}" \
  --port-base "${PORT_BASE}" \
  --stop-step "${STOP_STEP}" \
  --save-every "${SAVE_EVERY}" \
  --poll-seconds "${POLL_SECONDS}" \
  --cooldown-seconds "${COOLDOWN_SECONDS}" \
  --train-log "${TRAIN_LOG}" \
  --watchdog-log "${WATCHDOG_LOG}"
