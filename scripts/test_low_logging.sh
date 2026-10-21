#!/usr/bin/env bash
# Fresh-VM, sequential one-epoch diagnostics for all supported models.
#
# Usage:
#   bash scripts/test_low_logging.sh
#   MODELS="3d_infomax graph trimnet" bash scripts/test_low_logging.sh
#
# Installs dependencies if .venv is absent, activates it, and tests each model
# in sequence WITHOUT --debug. Console output is minimal; each model has its
# own log. config/train.yaml and the dataset are not modified.

set -uo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAMP="$(date +%Y%m%d-%H%M%S)"
LOG_DIR="${REPO_DIR}/results/logs"
INSTALL_LOG="${LOG_DIR}/install-${STAMP}.log"
SUMMARY_FILE="${LOG_DIR}/all-models-summary-${STAMP}.txt"
MODELS="${MODELS:-3d_infomax graph trimnet}"

cd "${REPO_DIR}"
mkdir -p "${LOG_DIR}"

if [[ ! -f "installing/install.sh" ]]; then
  echo "ERROR: installing/install.sh not found. Run this script from the repository." >&2
  exit 2
fi

# Install once on a fresh VM; reuse the environment for all three tests.
if [[ ! -x ".venv/bin/python" ]]; then
  echo "Virtual environment not found; installing dependencies (output saved to ${INSTALL_LOG})..."
  if bash installing/install.sh >"${INSTALL_LOG}" 2>&1; then
    echo "Installation completed."
  else
    status=$?
    echo "ERROR: installation failed (exit ${status}). Last 100 lines:" >&2
    tail -n 100 "${INSTALL_LOG}" >&2
    exit "${status}"
  fi
else
  echo "Existing .venv found; skipping installation."
fi

if [[ ! -x ".venv/bin/python" ]]; then
  echo "ERROR: installing/install.sh completed but .venv/bin/python is missing." >&2
  echo "See install log: ${INSTALL_LOG}" >&2
  exit 2
fi

# Activate the project's virtual environment.
# shellcheck disable=SC1091
source ".venv/bin/activate"

if [[ ! -f "data/droparray_full.pt" ]]; then
  echo "ERROR: data/droparray_full.pt not found. Place the dataset there before running." >&2
  exit 2
fi

# Do not let an inherited setting alter CUDA timing for this comparison.
unset CUDA_LAUNCH_BLOCKING

overall_status=0
: > "${SUMMARY_FILE}"

for MODEL in ${MODELS}; do
  case "${MODEL}" in
    graph|3d_infomax|trimnet) ;;
    *)
      echo "Skipping unsupported model: ${MODEL}" >&2
      echo "${MODEL}: SKIPPED (unsupported model)" >> "${SUMMARY_FILE}"
      overall_status=1
      continue
      ;;
  esac

  MODEL_STAMP="$(date +%Y%m%d-%H%M%S)"
  LOG_FILE="${LOG_DIR}/${MODEL}-low-logging-${STAMP}.log"
  TEMP_CONFIG="${LOG_DIR}/train-one-epoch-${MODEL}-${STAMP}.yaml"
  CONFIG_LOG="${LOG_DIR}/config-${MODEL}-${STAMP}.log"

  echo
  echo "========== Starting model: ${MODEL} =========="
  echo "One epoch; --debug disabled."
  echo "Log: ${LOG_FILE}"

  # Create an independent temporary one-epoch config for this model.
  if ! python - "${TEMP_CONFIG}" >"${CONFIG_LOG}" 2>&1 <<'PY'
import sys
import yaml

with open("config/train.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)
config["n_epochs"] = 1
with open(sys.argv[1], "w", encoding="utf-8") as f:
    yaml.safe_dump(config, f, sort_keys=False)
PY
  then
    echo "ERROR: could not create temporary config; see ${CONFIG_LOG}" >&2
    echo "${MODEL}: FAILED (config generation)" >> "${SUMMARY_FILE}"
    overall_status=1
    rm -f "${TEMP_CONFIG}"
    continue
  fi

  {
    echo "Timestamp: $(date -Is)"
    echo "Repo: ${REPO_DIR}"
    echo "Branch: $(git branch --show-current 2>/dev/null || echo unknown)"
    echo "Commit: $(git rev-parse HEAD 2>/dev/null || echo unknown)"
    echo "Model: ${MODEL}"
    echo "Mode: one epoch; --debug disabled"
    echo "Python: $(python --version 2>&1)"
    python - <<'PY'
import torch
print("PyTorch:", torch.__version__)
print("CUDA runtime:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY
    echo
    echo "nvidia-smi:"
    nvidia-smi 2>&1 || true
    echo
    echo "Starting training at $(date -Is)"
  } >"${LOG_FILE}" 2>&1

  # Redirect directly to the per-model log; don't flood the terminal with
  # per-batch output. faulthandler can still record Python fatal-signal traces.
  PYTHONFAULTHANDLER=1 PYTHONUNBUFFERED=1 \
    python -X faulthandler -u main.py \
      --train \
      --model "${MODEL}" \
      --config "${TEMP_CONFIG}" \
      >>"${LOG_FILE}" 2>&1
  status=$?

  {
    echo
    echo "Training ended at $(date -Is)"
    echo "Python exit code: ${status}"
  } >>"${LOG_FILE}"

  rm -f "${TEMP_CONFIG}"

  if [[ "${status}" -eq 0 ]]; then
    echo "${MODEL}: PASS (exit 0)" | tee -a "${SUMMARY_FILE}"
    echo "Last 12 log lines:"
    tail -n 12 "${LOG_FILE}"
  else
    echo "${MODEL}: FAIL (exit ${status})" | tee -a "${SUMMARY_FILE}"
    overall_status=1
    echo "Last 80 log lines:"
    tail -n 80 "${LOG_FILE}"
    echo "Recent kernel messages (if permitted):"
    dmesg -T 2>/dev/null | grep -Ei 'NVRM|Xid|segfault|oom|killed process' | tail -40 || true
  fi

  echo "Full log: ${LOG_FILE}"
done

echo
echo "========== Test summary =========="
cat "${SUMMARY_FILE}"
echo "Summary file: ${SUMMARY_FILE}"
echo "Install log (if installation ran): ${INSTALL_LOG}"

exit "${overall_status}"
