#!/usr/bin/env bash
set -euo pipefail
# Separate verification tool, never invoked by cairn or cairn-run.
TASK_ROOT=/scratch/hongse/mythos/cairn-rust
TASK_VISIBLE=""
for TASK_ARGUMENT in "$@"; do
  if [ "$TASK_ARGUMENT" = cuda ]; then TASK_VISIBLE=4; fi
done
exec env -i PATH=/usr/bin:/bin LANG=C.UTF-8 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
  CUDA_VISIBLE_DEVICES="$TASK_VISIBLE" CUDA_DEVICE_ORDER=PCI_BUS_ID CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  CUDA_MPS_PIPE_DIRECTORY="$TASK_ROOT/evidence/no-mps" \
  TORCHINDUCTOR_CACHE_DIR="$TASK_ROOT/evidence/oracle-cache" \
  TRITON_CACHE_DIR="$TASK_ROOT/evidence/oracle-cache/triton" \
  /scratch/hongse/dtm_energy_gpu_20261016/python/cpython-3.12-linux-x86_64-gnu/bin/python3.12 -I -S -B \
  "$TASK_ROOT/verification/python_oracle.py" \
  --reference /scratch/hongse/mythos/cairn-training/releases/20261102-v1 \
  --sites '["/scratch/hongse/mythos/cairn-training/.deps/site-packages","/scratch/hongse/mimo_ising_20261016/venv/lib/python3.12/site-packages"]' \
  --directory "$TASK_ROOT/evidence/parity" "$@"
