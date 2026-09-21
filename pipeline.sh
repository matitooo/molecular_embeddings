#!/usr/bin/env bash

set -e

# Install requirements
yes | bash installing/install.sh

# Define model types
model_types=("graph" "3d_infomax" "trimnet")

for model in "${model_types[@]}"; do

    echo "Sweeping model type: ${model}"

    # Execute sweep
    python main.py --sweep --model "${model}"

    # Locate best training parameters/config
    best_config="best_configurations/${model}_config.yaml"

    # Train using best configuration
    echo "Best parameters found, training with k-fold model: ${model}"

    python main.py \
        --train \
        --model "${model}" \
        --kfold \
        --config "${best_config}"

done