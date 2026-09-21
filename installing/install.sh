#!/usr/bin/env bash
set -eo pipefail

# Vai alla root del progetto (install.sh sta in installing/)
cd "$(dirname "${BASH_SOURCE[0]}")/.."

VENV_DIR=".venv"
PYTHON_BIN="${PYTHON_BIN:-python3.10}"   # cambia con python3.11 ecc. se serve

command -v "$PYTHON_BIN" >/dev/null || { echo "ERRORE: $PYTHON_BIN non trovato"; exit 1; }

# Crea il venv solo se non esiste
if [ ! -d "$VENV_DIR" ]; then
    "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

echo "Python: $(which python) ($(python --version))"

python -m pip install --upgrade pip setuptools wheel

# PyTorch con CUDA 12.1 (togli questa riga se torch è già in requirements.txt con l'indice giusto)
python -m pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cu121

# Requirements del progetto
python -m pip install -r installing/requirements.txt

# DGL (verifica che questa combinazione esista, vedi nota sotto)
python -m pip install dgl==2.4.0+cu121 \
    -f https://data.dgl.ai/wheels/torch-2.2/cu121/repo.html

# Verifica
python -c "import torch, dgl; print('torch', torch.__version__, '| dgl', dgl.__version__, '| cuda:', torch.cuda.is_available())"

echo "Environment setup completed successfully."