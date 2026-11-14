#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT=$(cd "$(dirname "$0")/.." && pwd)
export SERVING_SCRATCH="${SERVING_SCRATCH:-$REPO_ROOT/.local}"
mkdir -p "$SERVING_SCRATCH/tmp" "$SERVING_SCRATCH/cache"
export TMPDIR="$SERVING_SCRATCH/tmp"
export XDG_CACHE_HOME="$SERVING_SCRATCH/cache"
export VLLM_CACHE_ROOT="$SERVING_SCRATCH/cache/vllm"
export TORCHINDUCTOR_CACHE_DIR="$SERVING_SCRATCH/cache/inductor"
export TRITON_CACHE_DIR="$SERVING_SCRATCH/cache/triton"
export CUDA_CACHE_PATH="$SERVING_SCRATCH/cache/cuda"
export FLASHINFER_WORKSPACE_BASE="$SERVING_SCRATCH"
export FLASHINFER_AUTOTUNE_CACHE_DIR="$SERVING_SCRATCH/cache/flashinfer-autotune"
export VLLM_ADAPTIVE_VERIFICATION_PROFILE_CONTEXT_LEN=512
export NVIDIA_TF32_OVERRIDE=0
export OMP_NUM_THREADS=4
VERSION="${1:-v14}"
if [[ "$VERSION" == v14 ]]; then
  METHODS=(ar fixed7 adaptive)
elif [[ "$VERSION" == v15 ]]; then
  METHODS=(ar fixed4 fixed1 fixed6 fixed2 fixed7 fixed3 fixed5 adaptive)
elif [[ "$VERSION" == v16 ]]; then
  exec python "$REPO_ROOT/experiments/run_serving_holdout_v16.py"
else
  echo 'Usage: bash scripts/run_probe.sh [v14|v15|v16]' >&2
  exit 2
fi
for METHOD in "${METHODS[@]}"; do
  python "$REPO_ROOT/experiments/probe_serving_${VERSION}.py" --method "$METHOD"
done
