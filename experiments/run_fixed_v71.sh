#!/usr/bin/env bash
set -euo pipefail
R=/data/scene-rep/u/aquan/speculative-tonight-20261109
ROOT=/tmp/aquan-forespark-v63
export TMPDIR="$ROOT/tmp" XDG_CACHE_HOME="$ROOT/cache"
export HF_HUB_DISABLE_XET=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 NVIDIA_TF32_OVERRIDE=0
export WANDB_BASE_URL=https://api.wandb.ai SERVING_SCRATCH=/tmp/aquan-serving-v14-3043052.VFLAcm
export PYTHONPATH="$R${PYTHONPATH:+:$PYTHONPATH}"
P="$ROOT/ForeSpark/.venv/bin/python"
trap '"$SERVING_SCRATCH/env/bin/python" "$R/publish_fixed_v71.py"' EXIT
"$P" "$R/benchmark_fixed_v71.py" --role ar --width 0 --round -1
for width in 1 2 3 4 5 6 7 8; do
 "$P" "$R/benchmark_fixed_v71.py" --role native --width "$width" --round 0
done
"$P" "$R/benchmark_fixed_v71.py" --role forced --width 3 --round 0
"$P" "$R/benchmark_fixed_v71.py" --role forced --width 3 --round 1
for width in 8 7 6 5 4 3 2 1; do
 "$P" "$R/benchmark_fixed_v71.py" --role native --width "$width" --round 1
done
"$P" "$R/analyze_fixed_v71.py"
