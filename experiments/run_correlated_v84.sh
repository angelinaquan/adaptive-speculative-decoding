#!/usr/bin/env bash
set -euo pipefail
export SERVING_SCRATCH=/tmp/aquan-serving-v14-3043052.VFLAcm
export WANDB_BASE_URL=https://api.wandb.ai
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
"$SERVING_SCRATCH/env/bin/python" /data/scene-rep/u/aquan/speculative-tonight-20261109/run_correlated_v84.py
