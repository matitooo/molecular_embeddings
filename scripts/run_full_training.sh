#!/usr/bin/env bash
# Full k-fold training with default config, minimal terminal output, and per-run logs.
#
# Usage:
#   bash scripts/run_full_training.sh
#   MODELS="graph" bash scripts/run_full_training.sh
#
# config/train.yaml is used unchanged (currently n_epochs: 50). The existing
# train_loop early-stopping behavior remains enabled. Python source is unchanged.

set -uo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR" || exit 2

STAMP="${RUN_STAMP:-$(date +%Y%m%d-%H%M%S)-$$}"
RUN_ROOT="$REPO_DIR/results/full-training-runs"
RUN_DIR="$RUN_ROOT/$STAMP"
LOG_DIR="$RUN_DIR/logs"
OUTPUT_DIR="$RUN_DIR/outputs"
SUMMARY_FILE="$RUN_DIR/summary.txt"
MODELS="${MODELS:-3d_infomax graph trimnet}"

mkdir -p "$LOG_DIR" "$OUTPUT_DIR/predictions" "$OUTPUT_DIR/trained_model_weights" "$OUTPUT_DIR/training_statistics"
exec > >(tee -a "$RUN_DIR/orchestrator.log") 2>&1

say() { printf '[%s] %s\n' "$(date -Is)" "$*"; }

say "Full-training run: $STAMP"
say "Repository: $REPO_DIR"
say "Run directory: $RUN_DIR"
say "Models: $MODELS"

[[ -f config/train.yaml ]] || { say "ERROR: config/train.yaml not found."; exit 2; }
[[ -f data/droparray_full.pt ]] || { say "ERROR: data/droparray_full.pt not found."; exit 2; }
[[ -f installing/install.sh ]] || { say "ERROR: installing/install.sh not found."; exit 2; }

for model in $MODELS; do
  case "$model" in
    graph|3d_infomax|trimnet) ;;
    *) say "ERROR: unsupported model '$model'. Supported: graph, 3d_infomax, trimnet."; exit 2 ;;
  esac
done

# Snapshot the code/config/data identity before starting.
{
  echo "Started: $(date -Is)"
  echo "Run stamp: $STAMP"
  echo "Repository: $REPO_DIR"
  echo "Branch: $(git branch --show-current 2>/dev/null || echo unknown)"
  echo "Commit: $(git rev-parse HEAD 2>/dev/null || echo unknown)"
  echo "Git status:"
  git status --short 2>&1 || true
  echo
  echo "Training config used unchanged:"
  cat config/train.yaml
  echo
  echo "Dataset: $(readlink -f data/droparray_full.pt)"
  ls -lh data/droparray_full.pt
  sha256sum data/droparray_full.pt 2>&1 || true
  echo
  echo "Relevant environment variables:"
  env | grep -E '^(CUDA_VISIBLE_DEVICES|CUDA_LAUNCH_BLOCKING|PYTHONFAULTHANDLER|PYTHONUNBUFFERED|OMP_NUM_THREADS|MKL_NUM_THREADS|NVIDIA_VISIBLE_DEVICES)=' | sort || true
} > "$RUN_DIR/run-metadata.txt" 2>&1
cp config/train.yaml "$RUN_DIR/config.train.yaml"

# Install only if no usable project virtual environment exists.
if [[ ! -x .venv/bin/python ]]; then
  say "Virtual environment missing; installing dependencies. Full output: $LOG_DIR/install.log"
  if bash installing/install.sh > "$LOG_DIR/install.log" 2>&1; then
    say "Dependency installation completed."
  else
    status=$?
    say "ERROR: dependency installation failed (exit $status); see $LOG_DIR/install.log"
    exit "$status"
  fi
else
  say "Using existing .venv; skipping dependency installation."
fi
[[ -x .venv/bin/python ]] || { say "ERROR: .venv/bin/python is missing."; exit 2; }
# shellcheck disable=SC1091
source .venv/bin/activate

{
  echo "Captured: $(date -Is)"
  python --version 2>&1
  python -m pip freeze 2>&1
} > "$LOG_DIR/python-environment.txt"

python - > "$LOG_DIR/torch-cuda-environment.txt" 2>&1 <<'PY'
import sys
print("Python:", sys.version.replace("\n", " "))
try:
    import torch
    print("PyTorch:", torch.__version__)
    print("PyTorch CUDA runtime:", torch.version.cuda)
    print("CUDA available:", torch.cuda.is_available())
    print("GPU count:", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}:", torch.cuda.get_device_name(i))
except Exception as exc:
    print("PyTorch/GPU inspection failed:", repr(exc))
PY
nvidia-smi -q > "$LOG_DIR/nvidia-smi-before.txt" 2>&1 || nvidia-smi > "$LOG_DIR/nvidia-smi-before.txt" 2>&1 || true

# train.py and compute_base_path use fixed results/ paths. Point those paths at
# this run's dedicated outputs through symlinks, leaving Python source unchanged.
# Existing real output directories are moved into the run folder first, so model
# weights/predictions from earlier runs are not overwritten.
prepare_output_link() {
  local name="$1"
  local public_path="$REPO_DIR/results/$name"
  local link_target="full-training-runs/$STAMP/outputs/$name"
  local backup_path="$RUN_DIR/preexisting_outputs/$name"

  if [[ -L "$public_path" ]]; then
    rm "$public_path"
  elif [[ -e "$public_path" ]]; then
    mkdir -p "$(dirname "$backup_path")"
    if ! mv "$public_path" "$backup_path"; then
      say "ERROR: could not preserve existing $public_path"
      return 1
    fi
    say "Preserved previous $name at $backup_path"
  fi
  mkdir -p "$OUTPUT_DIR/$name"
  ln -s "$link_target" "$public_path"
}

mkdir -p "$REPO_DIR/results"
if ! prepare_output_link predictions || ! prepare_output_link trained_model_weights || ! prepare_output_link training_statistics; then
  say "ERROR: unable to prepare per-run output directories."
  exit 2
fi

# Sample resources every five seconds; do not stream batch output to the terminal.
monitor_resources() {
  local pid="$1"
  local logfile="$2"
  while kill -0 "$pid" 2>/dev/null; do
    {
      echo
      echo "TIME $(date -Is)"
      free -h 2>/dev/null | head -n 2 || true
      ps -p "$pid" -o pid=,ppid=,stat=,%cpu=,%mem=,rss=,vsz=,etime=,comm= 2>/dev/null || true
      if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw --format=csv,noheader 2>/dev/null || nvidia-smi 2>/dev/null || true
      fi
      if [[ -r /proc/pressure/memory ]]; then
        echo "MEMORY_PRESSURE:"
        cat /proc/pressure/memory
      fi
      for f in /sys/fs/cgroup/memory.current /sys/fs/cgroup/memory.max /sys/fs/cgroup/memory.events; do
        if [[ -r "$f" ]]; then echo "$f: $(tr '\n' ' ' < "$f")"; fi
      done
    } >> "$logfile" 2>&1
    sleep 5
  done
}

unset CUDA_LAUNCH_BLOCKING
overall_status=0
: > "$SUMMARY_FILE"

for MODEL in $MODELS; do
  TRAIN_LOG="$LOG_DIR/$MODEL-training.log"
  RESOURCE_LOG="$LOG_DIR/$MODEL-resources.log"
  MODEL_META="$LOG_DIR/$MODEL-metadata.txt"

  {
    echo "Model: $MODEL"
    echo "Started: $(date -Is)"
    echo "Branch: $(git branch --show-current 2>/dev/null || echo unknown)"
    echo "Commit: $(git rev-parse HEAD 2>/dev/null || echo unknown)"
    echo "Config: $REPO_DIR/config/train.yaml (used unchanged)"
    echo "Mode: full k-fold training; --debug disabled; smoke test disabled; --kfold enabled"
    echo "Python: $(python --version 2>&1)"
    python - <<'PY'
import torch
print("PyTorch:", torch.__version__)
print("CUDA runtime:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Current CUDA device:", torch.cuda.current_device())
    print("Current GPU:", torch.cuda.get_device_name(torch.cuda.current_device()))
PY
    echo "Command: python -X faulthandler -u main.py --train --model $MODEL --config config/train.yaml --kfold"
    echo "Training starts: $(date -Is)"
    nvidia-smi 2>&1 || true
  } > "$MODEL_META" 2>&1

  {
    echo "# Resource snapshots every 5 seconds"
    echo "# timestamp, host RAM/swap, process RSS/CPU, GPU utilization/VRAM, memory pressure/cgroup"
  } > "$RESOURCE_LOG"

  say "Starting full training for $MODEL. Training log: $TRAIN_LOG"
  PYTHONFAULTHANDLER=1 PYTHONUNBUFFERED=1 \
    python -X faulthandler -u main.py --train --model "$MODEL" --config config/train.yaml --kfold \
    > "$TRAIN_LOG" 2>&1 &
  TRAIN_PID=$!
  monitor_resources "$TRAIN_PID" "$RESOURCE_LOG" &
  MONITOR_PID=$!

  wait "$TRAIN_PID"
  status=$?
  kill "$MONITOR_PID" 2>/dev/null || true
  wait "$MONITOR_PID" 2>/dev/null || true

  {
    echo "Training ended: $(date -Is)"
    echo "Python exit code: $status"
  } >> "$MODEL_META"

  if [[ "$status" -eq 0 ]]; then
    echo "$MODEL: PASS (exit 0)" | tee -a "$SUMMARY_FILE"
  else
    echo "$MODEL: FAIL (exit $status)" | tee -a "$SUMMARY_FILE"
    overall_status=1
    {
      echo "--- Recent kernel messages after $MODEL (if permitted) ---"
      dmesg -T 2>&1 | grep -Ei 'NVRM|Xid|segfault|oom|killed process|gpu' | tail -n 100 || true
    } >> "$LOG_DIR/kernel-messages-after-$MODEL.txt" 2>&1
  fi

  grep -Ein 'traceback|segmentation fault|segfault|out of memory|cuda error|device-side assert|killed process|exception|runtimeerror' "$TRAIN_LOG" \
    | tail -n 100 > "$LOG_DIR/$MODEL-error-scan.txt" || true
  say "$MODEL finished with exit code $status. See $TRAIN_LOG."
done

{
  echo
  echo "Finished: $(date -Is)"
  echo "Overall exit code: $overall_status"
  echo "Run directory: $RUN_DIR"
  echo "Output directories: $OUTPUT_DIR/predictions, $OUTPUT_DIR/trained_model_weights, and $OUTPUT_DIR/training_statistics"
} >> "$SUMMARY_FILE"

nvidia-smi -q > "$LOG_DIR/nvidia-smi-after.txt" 2>&1 || nvidia-smi > "$LOG_DIR/nvidia-smi-after.txt" 2>&1 || true
dmesg -T > "$LOG_DIR/kernel-messages-after-all.txt" 2>&1 || true

find "$RUN_DIR" -type f ! -name manifest.txt -printf '%P\t%s bytes\n' 2>/dev/null | sort > "$RUN_DIR/manifest.txt"
say "All training runs finished. Summary: $SUMMARY_FILE"
say "Logs and model outputs are stored under: $RUN_DIR"
say "results/predictions, results/trained_model_weights, and results/training_statistics now point to this run's outputs."
exit "$overall_status"
