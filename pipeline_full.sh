#!/usr/bin/env bash

LOG_DIR="results/"
LOG_FILE="$LOG_DIR/pipeline_full.log"

mkdir -p "$LOG_DIR"

exec > >(tee -a "$LOG_FILE") 2>&1

set -x

set -eo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

bash installing/install.sh

source .venv/bin/activate

model_types=("graph" "3d_infomax" "trimnet")

for model in "${model_types[@]}"; do
    echo "Sweeping model type: ${model}"
    python main.py --sweep  --model "${model}"

    best_config="results/best_configurations/${model}_config.yaml"

    echo "Best parameters found, training with k-fold model: ${model}"
    python main.py \
        --train \
        --model "${model}" \
        --kfold \
        --config "${best_config}"
done