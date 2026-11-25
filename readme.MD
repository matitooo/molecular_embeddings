# Bacteria growth inhibition prediction with molecular embeddings

This project compares graph-based molecular representations for predicting bacterial growth inhibition using the DropArray dataset. Models supported by the training code are `graph`, `3d_infomax`, and `trimnet`.

## Environment

Install the project dependencies from the repository root:

```bash
bash installing/install.sh
```

The installer creates a Python 3.11 virtual environment at `.venv` unless `VENV_DIR` is set.

## Dataset

The full training run expects `data/droparray_full.pt`. This file is tracked with Git LFS; ensure Git LFS is installed and the object is available before running.

## Full training

Run all three models with the configuration in `config/train.yaml`:

```bash
bash scripts/run_full_training.sh
```

To train one model only:

```bash
MODELS="graph" bash scripts/run_full_training.sh
```

The run script uses k-fold training, keeps the configured maximum of 50 epochs with the training loop's early stopping, and stores logs, metadata, predictions, model weights, and statistics under `results/full-training-runs/`. It also updates `results/predictions`, `results/trained_model_weights`, and `results/training_statistics` to point to the latest run.

## One-epoch diagnostics

For a short one-epoch check of all supported models, run:

```bash
bash scripts/test_low_logging.sh
```

This is a diagnostic workflow and does not replace the full training run.
