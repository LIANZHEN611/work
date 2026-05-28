#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
mkdir -p outputs logs

export OMP_NUM_THREADS=4
export PYTHONUNBUFFERED=1
export TORCH_NCCL_BLOCKING_WAIT=1
LOG=outputs/train_optim_ddp_$(date +%Y%m%d_%H%M%S).log

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-4,5,6,7} \
torchrun \
  --nproc_per_node=4 \
  --rdzv_endpoint=127.0.0.1:29501 \
  --rdzv_backend=c10d \
  tools/train_val.py --config configs/monodetr_optim.yaml \
  2>&1 | tee "$LOG"
