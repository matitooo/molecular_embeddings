#!/usr/bin/env bash
# One-epoch, low-logging diagnostic run for molecular_embeddings.
#
# Usage:
#   bash scripts/test_low_logging.sh
#   MODEL=graph bash scripts/test_low_logging.sh
#   MODEL=trimnet bash scripts/test_low_logging.sh
#
# This intentionally does NOT pass --debug. It writes a temporary one-epoch
# config and leaves config/train.yaml unchanged.

set -uo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL="${MODEL:-3d_infomax}"
STAMP="$(date +%Y%m%d-%H%M%S)"
LOG_DIR="${REPO_DIR}/results/logs"
LOG_FILE="${LOG_DIR}/${MODEL}-low-logging-${STAMP}.log"
TEMP_CONFIG="${LOG_DIR}/train-one-epoch-${STAMP}.yaml"

case "${MODEL}" in
  graph|3d_infomax|trimnet) ;;
  *)
    echo "ERROR: MODEL must be graph, 3d_infomax, or trimnet (got: ${MODEL})" >&2
    exit 2
    ;;
esac

cd "${REPO_DIR}"

if [[ ! -x ".venv/bin/python" ]]; then
  echo "ERROR: .venv/bin/python not found. Run installing/install.sh first." >&2
  exit 2
fi
if [[ ! -f "data/droparray_full.pt" ]]; then
  echo "ERROR: data/droparray_full.pt not found." >&2
  exit 2
fi

mkdir -p "${LOG_DIR}"
source ".venv/bin/activate"

# Create a temporary one-epoch config; leave config/train.yaml unchanged.
python - "${TEMP_CONFIG}" <<'PY'
import sys
import yaml

with open("config/train.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)
config["n_epochs"] = 1
with open(sys.argv[1], "w", encoding="utf-8") as f:
    yaml.safe_dump(config, f, sort_keys=False)
PY

cleanup() {
  rm -f "${TEMP_CONFIG}"
}
trap cleanup EXIT

echo "Repo: ${REPO_DIR}"
echo "Branch: $(git branch --show-current 2>/dev/null || echo unknown)"
echo "Model: ${MODEL}"
echo "One epoch; --debug disabled (reduced diagnostic logging)."
echo "Log: ${LOG_FILE}"
echo

# Do not let an inherited setting alter CUDA timing for this comparison.
unset CUDA_LAUNCH_BLOCKING

set +e
PYTHONFAULTHANDLER=1 PYTHONUNBUFFERED=1 \
  python -X faulthandler -u main.py \
    --train \
    --model "${MODEL}" \
    --config "${TEMP_CONFIG}" \
    2>&1 | tee "${LOG_FILE}"
status=${PIPESTATUS[0]}
set -e

echo
echo "Python exit code: ${status}"
echo "Log saved to: ${LOG_FILE}"

if [[ ${status} -ne 0 ]]; then
  echo "Run failed. Recent kernel messages (if permitted):"
  dmesg -T 2>/dev/null | grep -Ei 'NVRM|Xid|segfault|oom|killed process' | tail -40 || true
fi

exit "${status}"
