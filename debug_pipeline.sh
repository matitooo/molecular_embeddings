#!/usr/bin/env bash
set -eo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

# Installa (idempotente: riusa .venv se esiste)
bash installing/install.sh

# Attiva il venv in QUESTA shell, così main.py usa l'ambiente giusto
source .venv/bin/activate

model_types=("graph" "3d_infomax" "trimnet")

for model in "${model_types[@]}"; do
    echo "Sweeping model type: ${model}"
    python main.py --sweep --debug  --model "${model}"

    best_config="best_configurations/${model}_config.yaml"

    echo "Best parameters found, training with k-fold model: ${model}"
    python main.py \
        --train \
        --model "${model}" \
        --debug \
        --kfold \
        --config "${best_config}"
done