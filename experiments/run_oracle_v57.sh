#!/usr/bin/env bash
set -euo pipefail
R=/data/scene-rep/u/aquan/speculative-tonight-20261109
export SERVING_SCRATCH=/tmp/aquan-serving-v14-3043052.VFLAcm
export TMPDIR="$SERVING_SCRATCH/tmp"
export XDG_CACHE_HOME="$SERVING_SCRATCH/cache"
export VLLM_CACHE_ROOT="$SERVING_SCRATCH/cache/vllm"
export TORCHINDUCTOR_CACHE_DIR="$SERVING_SCRATCH/cache/inductor"
export TRITON_CACHE_DIR="$SERVING_SCRATCH/cache/triton"
export CUDA_CACHE_PATH="$SERVING_SCRATCH/cache/cuda"
export VLLM_ADAPTIVE_VERIFICATION_PROFILE_CONTEXT_LEN=512
export NVIDIA_TF32_OVERRIDE=0
export OMP_NUM_THREADS=4
export FLASHINFER_WORKSPACE_BASE="$SERVING_SCRATCH"
export FLASHINFER_AUTOTUNE_CACHE_DIR="$SERVING_SCRATCH/cache/flashinfer-autotune"
export WANDB_BASE_URL=https://api.wandb.ai
PY="$SERVING_SCRATCH/env/bin/python"
"$PY" -c "import json; assert json.load(open('$R/serving_v14_smoke.json'))['passed']"
export PYTHONPATH="$R${PYTHONPATH:+:$PYTHONPATH}"
export VLLM_BATCH_INVARIANT=1
"$PY" "$R/run_oracle_v57.py"
