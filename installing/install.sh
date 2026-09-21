#!/bin/bash
set -e

conda env create -f installing/environment.yml
conda run -n drugpred_da pip install -r installing/requirements.txt
conda run -n drugpred_da pip install \
    dgl==2.4.0+cu121 \
    -f https://data.dgl.ai/wheels/torch-2.2/cu121/repo.html

conda init
conda activate drugpred_da
