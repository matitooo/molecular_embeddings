#!/usr/bin/env bash
set -eo pipefail

# Root del progetto (install.sh sta in installing/)
cd "$(dirname "${BASH_SOURCE[0]}")/.."

PY_VERSION="${PY_VERSION:-3.11}"
VENV_DIR="${VENV_DIR:-.venv}"
export PATH="$HOME/.local/bin:$PATH"

# 1. uv (senza root)
if ! command -v uv >/dev/null 2>&1; then
    echo "Installo uv..."
    if ! (curl -LsSf https://astral.sh/uv/install.sh | sh); then
        python3 -m pip install --user uv || python -m pip install --user uv
    fi
    hash -r
fi
command -v uv >/dev/null || { echo "ERRORE: impossibile installare uv"; exit 1; }

# 2. Ricrea il venv se ha la versione sbagliata
if [ -x "$VENV_DIR/bin/python" ]; then
    current="$("$VENV_DIR/bin/python" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
    if [ "$current" != "$PY_VERSION" ]; then
        echo "Venv con Python $current, serve $PY_VERSION: lo ricreo"
        rm -rf "$VENV_DIR"
    fi
fi

# 3. Crea il venv (uv scarica Python se manca)
if [ ! -x "$VENV_DIR/bin/python" ]; then
    uv venv --python "$PY_VERSION" --seed "$VENV_DIR"
fi
PY="$VENV_DIR/bin/python"
echo "Python: $($PY --version) in $VENV_DIR"

"$PY" -m pip install --upgrade pip setuptools wheel

# 4. Constraints: impediscono a pip di cambiare torch/numpy
CONSTRAINTS="$VENV_DIR/constraints.txt"
cat > "$CONSTRAINTS" <<EOF
torch==2.2.2
torchvision==0.17.2
torchaudio==2.2.2
numpy==1.26.4
EOF

# 5. PyTorch CUDA 12.1
"$PY" -m pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 \
    --index-url https://download.pytorch.org/whl/cu121

# 6. Requirements del progetto
"$PY" -m pip install -r installing/requirements.txt -c "$CONSTRAINTS"

# 7. DGL: prova più versioni finché una esiste per torch 2.2 + cu121
DGL_INDEX="https://data.dgl.ai/wheels/torch-2.2/cu121/repo.html"
dgl_ok=0
for v in 2.4.0 2.2.1 2.1.0; do
    echo "Provo dgl==${v}+cu121"
    if "$PY" -m pip install "dgl==${v}+cu121" -f "$DGL_INDEX" -c "$CONSTRAINTS"; then
        dgl_ok=1; break
    fi
done
[ "$dgl_ok" = 1 ] || { echo "ERRORE: nessuna wheel DGL trovata su $DGL_INDEX"; exit 1; }

# 8. Verifica
"$PY" - <<'EOF'
import torch, dgl, torch_geometric, rdkit, optuna
print("torch", torch.__version__, "| cuda:", torch.cuda.is_available())
print("dgl", dgl.__version__, "| pyg", torch_geometric.__version__)
EOF

echo "Environment setup completed successfully."