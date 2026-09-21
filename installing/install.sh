#!/bin/bash
set -e

# Initialize Conda for this shell
source /opt/miniforge3/etc/profile.d/conda.sh

# Create the environment
conda env create -f installing/environment.yml

# Activate the environment
conda activate drugpred_da

# Verify activation
echo "Conda environment: $CONDA_DEFAULT_ENV"
echo "Python: $(which python)"
echo "Pip: $(which pip)"

# Install Python requirements
pip install -r installing/requirements.txt

# Install DGL for PyTorch 2.2 + CUDA 12.1
pip install \
    dgl==2.4.0+cu121 \
    -f https://data.dgl.ai/wheels/torch-2.2/cu121/repo.html



echo "Environment setup completed successfully."