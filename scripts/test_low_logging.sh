#!/usr/bin/env bash
# Fresh-VM, one-epoch diagnostic run with quiet console and full log files.
#
# Usage:
#   bash scripts/test_low_logging.sh
#   MODEL=graph bash scripts/test_low_logging.sh
#   MODEL=trimnet bash scripts/test_low_logging.sh
#
# If .venv does not exist, runs installing/install.sh and logs its output.
# Activates .venv, then runs one epoch WITHOUT --debug. config/train.yaml
# and the dataset are not modified. Console output is intentionally minimal;
# stdout/stderr from install and training are captured in results/logs/.

set -uo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL="${MODEL:-3d_infomax}"
STAMP="$(date +%Y%m%d-%H%M%S)"
LOG_DIR="${REPO_DIR}/results/logs"
LOG_FILE="${LOG_DIR}/${MODEL}-low-logging-${STAMP}.log"
INSTALL_LOG="${LOG_DIR}/install-${STAMP}.log"
TEMP_CONFIG="${LOG_DIR}/train-one-epoch-${STAMP}.yaml"

case "${MODEL}" in
  graph|3d_infomax|trimnet) ;;
  *)
    echo "ERROR: MODEL must be graph, 3d_infomax, or trimnet (got: ${MODEL})" >&2
    exit 2
    ;;
esac

cd "${REPO_DIR}"
mkdir -p "${LOG_DIR}"

if [[ ! -f "installing/install.sh" ]]; then
  echo "ERROR: installing/install.sh not found. Run this script from the repository." >&2
  exit 2
fi

# On a fresh VM, install the project if the virtual environment is absent.
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

# Activate the project's virtual environment for this shell.
# shellcheck disable=SC1091
source ".venv/bin/activate"

if [[ ! -f "data/droparray_full.pt" ]]; then
  echo "ERROR: data/droparray_full.pt not found. Place the dataset there before running." >&2
  exit 2
fi

# Create a temporary one-epoch config; leave config/train.yaml unchanged.
if ! python - "${TEMP_CONFIG}" >"${LOG_DIR}/config-${STAMP}.log" 2>&1 <<'PY'
import sys
import yaml

with open("config/train.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)
config["n_epochs"] = 1
with open(sys.argv[1], "w", encoding="utf-8") as f:
    yaml.safe_dump(config, f, sort_keys=False)
PY
then
  echo "ERROR: failed to create temporary config; see ${LOG_DIR}/config-${STAMP}.log" >&2
  exit 2
fi

cleanup() {
  rm -f "${TEMP_CONFIG}"
}
trap cleanup EXIT

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

echo "Model: ${MODEL}"
echo "One epoch, --debug disabled. Console output is kept minimal."
echo "Training log: ${LOG_FILE}"

# Do not let an inherited setting alter CUDA timing for this comparison.
unset CUDA_LAUNCH_BLOCKING

# Redirect directly to a file instead of tee: no per-batch terminal flood.
# faulthandler may write a Python traceback for fatal signals to this log.
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

echo "Exit code: ${status}"
echo "Log saved to: ${LOG_FILE}"

if [[ "${status}" -eq 0 ]]; then
  echo "Last 25 log lines:"
  tail -n 25 "${LOG_FILE}"
else
  echo "Run failed. Last 100 log lines:"
  tail -n 100 "${LOG_FILE}"
  echo
  echo "Recent kernel messages (if permitted):"
  dmesg -T 2>/dev/null | grep -Ei 'NVRM|Xid|segfault|oom|killed process' | tail -40 || true
  echo
  echo "Install log (if created): ${INSTALL_LOG}"
fi

exit "${status}"
