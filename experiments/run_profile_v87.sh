#!/usr/bin/env bash
set -euo pipefail
R=/data/scene-rep/u/aquan/speculative-tonight-20261109
ROOT=/tmp/aquan-forespark-v63
export TMPDIR="$ROOT/tmp" XDG_CACHE_HOME="$ROOT/cache"
export HF_HUB_DISABLE_XET=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 NVIDIA_TF32_OVERRIDE=0
export WANDB_BASE_URL=https://api.wandb.ai SERVING_SCRATCH=/tmp/aquan-serving-v14-3043052.VFLAcm
export PYTHONPATH="$R${PYTHONPATH:+:$PYTHONPATH}"
"$SERVING_SCRATCH/env/bin/python" "$R/run_profile_v87.py"
