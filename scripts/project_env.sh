#!/usr/bin/env bash
set -euo pipefail

# Shared ControlLight environment bootstrap.
#
# Policy:
# - If the caller already has an active virtualenv, respect it. This keeps
#   standalone packaging/venv smoke tests isolated.
# - Otherwise, fall back to the project default conda environment:
#   `aitoolkit`. We intentionally do not auto-respect an arbitrary active
#   conda env, because users often invoke the wrappers from a base shell that
#   has incompatible package versions.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
DEFAULT_CONDA_ENV="${DEFAULT_CONDA_ENV:-aitoolkit}"
if [[ -z "${CONDA_ENV+x}" ]]; then
  if [[ -n "${VIRTUAL_ENV:-}" ]]; then
    CONDA_ENV=""
  else
    CONDA_ENV="${DEFAULT_CONDA_ENV}"
  fi
fi
CONDA_SH="${CONDA_SH:-}"

if [[ -n "${CONDA_ENV}" ]]; then
  if [[ -z "${CONDA_SH}" && -n "${CONDA_EXE:-}" ]]; then
    CONDA_SH="$(cd "$(dirname "${CONDA_EXE}")/../etc/profile.d" && pwd)/conda.sh"
  fi
  if [[ -z "${CONDA_SH}" ]]; then
    for candidate in \
      "${HOME}/miniconda3/etc/profile.d/conda.sh" \
      "${HOME}/anaconda3/etc/profile.d/conda.sh" \
      "/opt/conda/etc/profile.d/conda.sh"
    do
      if [[ -f "${candidate}" ]]; then
        CONDA_SH="${candidate}"
        break
      fi
    done
  fi
  if [[ -z "${CONDA_SH}" || ! -f "${CONDA_SH}" ]]; then
    echo "Failed to locate conda.sh for CONDA_ENV=${CONDA_ENV}. Set CONDA_SH explicitly." >&2
    exit 1
  fi
  # shellcheck disable=SC1090
  source "${CONDA_SH}"
  conda activate "${CONDA_ENV}"
fi

unset ALL_PROXY all_proxy HTTP_PROXY http_proxy HTTPS_PROXY https_proxy
export REPO_ROOT
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
