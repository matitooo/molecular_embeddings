#!/usr/bin/env bash
# One-command setup + isolated 3D-Infomax embedding diagnostic.
# Creates/repairs the project venv using the repository installer, activates it,
# then runs test_3d_infomax_only.sh. No training is started.
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

VENV_DIR="${VENV_DIR:-.venv}"
export VENV_DIR
export PY_VERSION="${PY_VERSION:-3.11}"

echo "============================================================"
echo "3D-Infomax diagnostic: environment setup + embedding test"
echo "Repository: $ROOT_DIR"
echo "Python target: $PY_VERSION"
echo "Virtual environment: $VENV_DIR"
echo "Dataset: ${DATASET_PATH:-data/droparray_full.pt}"
echo "Started: $(date -Is)"
echo "============================================================"

if [[ ! -f installing/install.sh ]]; then
  echo "ERROR: installing/install.sh not found. Run from a repository checkout."
  exit 2
fi
if [[ ! -f test_3d_infomax_only.sh ]]; then
  echo "ERROR: test_3d_infomax_only.sh not found."
  exit 2
fi

echo
echo "---- Step 1/3: create/install project environment ----"
bash installing/install.sh

echo
echo "---- Step 2/3: activate virtual environment ----"
if [[ ! -f "$VENV_DIR/bin/activate" ]]; then
  echo "ERROR: expected activation script at $VENV_DIR/bin/activate"
  exit 3
fi
# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"
echo "Active Python: $(command -v python)"
python --version

echo
echo "---- Step 3/3: run isolated 3D-Infomax embedding test ----"
# The test script uses python from PATH, so it will use the activated venv.
bash test_3d_infomax_only.sh
status=$?

echo
echo "Diagnostic finished with exit code $status at $(date -Is)"
exit "$status"
