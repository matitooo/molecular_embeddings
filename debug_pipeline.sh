#!/usr/bin/env bash
set -Eeuo pipefail

# Always run from the repository root, regardless of the caller's working directory.
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

# Keep installer and activation pointed at the same virtual environment.
export VENV_DIR="${VENV_DIR:-.venv}"
export PY_VERSION="${PY_VERSION:-3.11}"

LOG_DIR="results"
LOG_FILE="$LOG_DIR/pipeline_debug.log"
mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_FILE") 2>&1
set -x

echo "Repository: $ROOT_DIR"
echo "Virtual environment: $VENV_DIR"
echo "Requested Python: $PY_VERSION"

# Creates the venv if missing, repairs a mismatched Python version, and installs dependencies.
bash installing/install.sh

if [[ ! -x "$VENV_DIR/bin/python" || ! -f "$VENV_DIR/bin/activate" ]]; then
    echo "ERROR: virtual environment was not created at $VENV_DIR" >&2
    exit 1
fi
# Use the venv's interpreter explicitly after activation to avoid picking up system Python.
source "$VENV_DIR/bin/activate"
hash -r
python --version
python -c 'import sys; print("Active interpreter:", sys.executable)'

model_types=("graph" "3d_infomax" "trimnet")
for model in "${model_types[@]}"; do
    echo "===== Starting full debug run for model: $model ====="
    python main.py \
        --train \
        --model "$model" \
        --kfold \\\n        --debug
done
