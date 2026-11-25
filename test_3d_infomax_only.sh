#!/usr/bin/env bash
# Isolated 3D-Infomax preprocessing test: loads the configured dataset and
# computes the pretrained 3D-Infomax embeddings, but does NOT start training.
set -uo pipefail

DATASET_PATH="${DATASET_PATH:-data/droparray_full.pt}"
LOG_DIR="${LOG_DIR:-results/logs}"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/infomax-only-$(date +%Y%m%d-%H%M%S).log"

# Keep the pipeline's exit code even though output is also written to a log.
exec > >(tee -a "$LOG_FILE") 2>&1

echo "============================================================"
echo "3D-Infomax isolated test"
echo "Date: $(date -Is)"
echo "PWD:  $PWD"
echo "Dataset: $DATASET_PATH"
echo "Log: $LOG_FILE"
echo "============================================================"

if [[ ! -f "$DATASET_PATH" ]]; then
  echo "ERROR: dataset not found: $DATASET_PATH"
  echo "Run this script from the repository root or set DATASET_PATH=/path/to/dataset.pt"
  exit 2
fi

echo
echo "---- Host / GPU ----"
uname -a
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
else
  echo "nvidia-smi not found"
fi

echo
echo "---- Python packages and isolated embedding run ----"
set +e
python -u - "$DATASET_PATH" <<'PY'
import os
import sys
import time
import traceback

dataset_path = sys.argv[1]
started = time.monotonic()

try:
    import torch
    print(f"Python: {sys.version.replace(chr(10), ' ')}", flush=True)
    print(f"PyTorch: {torch.__version__}", flush=True)
    print(f"PyTorch CUDA build: {torch.version.cuda}", flush=True)
    print(f"CUDA available: {torch.cuda.is_available()}", flush=True)
    if torch.cuda.is_available():
        print(f"GPU count: {torch.cuda.device_count()}", flush=True)
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}", flush=True)
    try:
        import dgl
        print(f"DGL: {dgl.__version__}", flush=True)
    except Exception as exc:
        print(f"DGL import failed: {type(exc).__name__}: {exc}", flush=True)
        raise

    from datasets import DropArray

    print(f"Starting DropArray(model='3d_infomax') at {time.strftime('%F %T')}", flush=True)
    # This performs dataset loading and 3D-Infomax embedding generation only.
    # It does not create DataLoaders for training or run an optimizer.
    data = DropArray(dataset_path, model="3d_infomax")
    embeddings = data.drug_embedding_dict
    print(f"SUCCESS: embeddings generated: {len(embeddings)}", flush=True)

    if embeddings:
        first_key = next(iter(embeddings))
        first = embeddings[first_key]
        print(f"First embedding key: {first_key!r}", flush=True)
        print(f"First embedding type: {type(first).__name__}", flush=True)
        if torch.is_tensor(first):
            print(f"First embedding shape: {tuple(first.shape)}", flush=True)
            print(f"First embedding dtype: {first.dtype}", flush=True)
            print(f"First embedding device: {first.device}", flush=True)
            print(f"First embedding finite: {bool(torch.isfinite(first).all().item())}", flush=True)
        devices = {}
        for emb in embeddings.values():
            if torch.is_tensor(emb):
                label = str(emb.device)
                devices[label] = devices.get(label, 0) + 1
        print(f"Embedding devices/counts: {devices}", flush=True)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        print(f"CUDA peak allocated: {torch.cuda.max_memory_allocated() / 1024**2:.1f} MiB", flush=True)
        print(f"CUDA peak reserved: {torch.cuda.max_memory_reserved() / 1024**2:.1f} MiB", flush=True)

    print(f"TOTAL SUCCESS in {time.monotonic() - started:.1f}s", flush=True)
except BaseException:
    traceback.print_exc()
    print(f"FAILED after {time.monotonic() - started:.1f}s", flush=True)
    sys.exit(1)
PY
status=$?
set -e

echo
echo "---- Result ----"
echo "Python exit code: $status"
echo "Finished: $(date -Is)"
if [[ "$status" -eq 0 ]]; then
  echo "3D-Infomax embedding test completed successfully; training was not run."
else
  echo "3D-Infomax isolated test failed (or the process crashed)."
  echo "Inspect the last log lines above and the log file: $LOG_FILE"
fi
exit "$status"
