#!/usr/bin/env bash
# Fresh-VM, sequential one-epoch diagnostics for all supported models.
#
# Usage:
#   bash scripts/test_low_logging.sh
#   MODELS="3d_infomax graph trimnet" bash scripts/test_low_logging.sh
#
# Installs dependencies if .venv is absent, activates it, and tests each model
# in sequence WITHOUT --debug. Console output is minimal; training output and
# periodic resource snapshots (RAM, process RSS, GPU utilization/VRAM) are
# saved to separate per-model logs. config/train.yaml and dataset are untouched.

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

# shellcheck disable=SC1091
source ".venv/bin/activate"

if [[ ! -f "data/droparray_full.pt" ]]; then
  echo "ERROR: data/droparray_full.pt not found. Place the dataset there before running." >&2
  exit 2
fi

unset CUDA_LAUNCH_BLOCKING
overall_status=0
: > "${SUMMARY_FILE}"

for MODEL in ${MODELS}; do
  case "${MODEL}" in
    graph|3d_infomax|trimnet) ;;
    *)
      echo "${MODEL}: SKIPPED (unsupported model)" | tee -a "${SUMMARY_FILE}"
      overall_status=1
      continue
      ;;
  esac

  LOG_FILE="${LOG_DIR}/${MODEL}-low-logging-${STAMP}.log"
  RESOURCE_LOG="${LOG_DIR}/${MODEL}-resources-${STAMP}.log"
  TEMP_CONFIG="${LOG_DIR}/train-one-epoch-${MODEL}-${STAMP}.yaml"
  CONFIG_LOG="${LOG_DIR}/config-${MODEL}-${STAMP}.log"

  echo
  echo "========== Starting model: ${MODEL} =========="
  echo "One epoch; --debug disabled. Resource sampling every 5 seconds."
  echo "Training log: ${LOG_FILE}"
  echo "Resource log: ${RESOURCE_LOG}"

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
    echo "${MODEL}: FAIL (config generation; see ${CONFIG_LOG})" | tee -a "${SUMMARY_FILE}"
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
    nvidia-smi 2>&1 || true
    echo
    echo "Training starts: $(date -Is)"
  } >"${LOG_FILE}" 2>&1

  {
    echo "# Resource snapshots every 5 seconds"
    echo "# timestamp | host RAM | swap | training process RSS/CPU | GPU utilization and VRAM"
  } >"${RESOURCE_LOG}"

  # Background monitor: separate file, no per-batch console printing.
  monitor_resources() {
    local pid="$1"
    local logfile="$2"
    while kill -0 "$pid" 2>/dev/null; do
      {
        echo
        echo "TIME $(date -Is)"
        free -h 2>/dev/null | head -n 2 || true
        ps -p "$pid" -o pid=,stat=,%cpu=,%mem=,rss=,vsz=,etime=,comm= 2>/dev/null || true
        if command -v nvidia-smi >/dev/null 2>&1; then
          nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw --format=csv,noheader 2>/dev/null || nvidia-smi 2>/dev/null || true
        fi
        if [[ -r /proc/pressure/memory ]]; then
          echo "MEMORY_PRESSURE:"
          cat /proc/pressure/memory
        fi
        # Cgroup v2 memory limit/current can reveal container-level limits.
        for f in /sys/fs/cgroup/memory.current /sys/fs/cgroup/memory.max /sys/fs/cgroup/memory.events; do
          if [[ -r "$f" ]]; then echo "$f: $(tr '\n' ' ' < "$f")"; fi
        done
      } >>"$logfile" 2>&1
      sleep 5
    done
  }

  # Start Python as a background process so resource sampling can run alongside it.
  PYTHONFAULTHANDLER=1 PYTHONUNBUFFERED=1 \
    python -X faulthandler -u main.py \
      --train --model "${MODEL}" --config "${TEMP_CONFIG}" \
      >>"${LOG_FILE}" 2>&1 &
  TRAIN_PID=$!

  monitor_resources "$TRAIN_PID" "$RESOURCE_LOG" &
  MONITOR_PID=$!

  wait "$TRAIN_PID"
  status=$?

  # Stop sampler and reap it, regardless of training exit code.
  kill "$MONITOR_PID" 2>/dev/null || true
  wait "$MONITOR_PID" 2>/dev/null || true

  {
    echo
    echo "Training ended: $(date -Is)"
    echo "Python exit code: ${status}"
  } >>"${LOG_FILE}"

  rm -f "${TEMP_CONFIG}"

  # Report known error signatures without dumping all batch output.
  ERROR_MATCHES="$(grep -Ein 'traceback|segmentation fault|segfault|out of memory|cuda error|device-side assert|killed process|exception|runtimeerror' "${LOG_FILE}" | tail -n 30 || true)"

  if [[ "${status}" -eq 0 ]]; then
    echo "${MODEL}: PASS (exit 0)" | tee -a "${SUMMARY_FILE}"
  else
    echo "${MODEL}: FAIL (exit ${status})" | tee -a "${SUMMARY_FILE}"
    overall_status=1
  fi

  if [[ -n "${ERROR_MATCHES}" ]]; then
    {
      echo "--- Error-signature scan for ${MODEL} ---"
      echo "${ERROR_MATCHES}"
    } >>"${SUMMARY_FILE}"
  fi

  if [[ "${status}" -ne 0 ]]; then
    echo "Last 80 training-log lines:"
    tail -n 80 "${LOG_FILE}"
    echo "Recent kernel messages (if permitted):"
    dmesg -T 2>/dev/null | grep -Ei 'NVRM|Xid|segfault|oom|killed process' | tail -40 || true
  else
    echo "Last 12 training-log lines:"
    tail -n 12 "${LOG_FILE}"
  fi

  echo "Resource log: ${RESOURCE_LOG}"
  echo "Training log: ${LOG_FILE}"
done

echo
echo "========== Test summary =========="
cat "${SUMMARY_FILE}"
echo "Summary file: ${SUMMARY_FILE}"
echo "Install log (if installation ran): ${INSTALL_LOG}"

exit "${overall_status}"
