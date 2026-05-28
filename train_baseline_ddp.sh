#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
mkdir -p outputs logs

export OMP_NUM_THREADS=4
export PYTHONUNBUFFERED=1
export TORCH_NCCL_BLOCKING_WAIT=1
LOG=outputs/train_baseline_ddp_$(date +%Y%m%d_%H%M%S).log

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3} \
torchrun \
  --nproc_per_node=4 \
  --rdzv_endpoint=127.0.0.1:29500 \
  --rdzv_backend=c10d \
  tools/train_val.py --config configs/monodetr.yaml \
  2>&1 | tee "$LOG"
