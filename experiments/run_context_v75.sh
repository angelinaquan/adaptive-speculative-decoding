#!/usr/bin/env bash
set -euo pipefail
R=/data/scene-rep/u/aquan/speculative-tonight-20261109
export SERVING_SCRATCH=/tmp/aquan-serving-v14-3043052.VFLAcm WANDB_BASE_URL=https://api.wandb.ai
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
"$SERVING_SCRATCH/env/bin/python" "$R/context_probe_v75.py"
"$SERVING_SCRATCH/env/bin/python" "$R/publish_context_v75.py"
