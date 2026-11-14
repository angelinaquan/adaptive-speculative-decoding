#!/usr/bin/env bash
set -euo pipefail
R=/data/scene-rep/u/aquan/speculative-tonight-20261109
ROOT=/tmp/aquan-forespark-v63
export TMPDIR="$ROOT/tmp" XDG_CACHE_HOME="$ROOT/cache"
export HF_HUB_DISABLE_XET=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 NVIDIA_TF32_OVERRIDE=0
export WANDB_BASE_URL=https://api.wandb.ai SERVING_SCRATCH=/tmp/aquan-serving-v14-3043052.VFLAcm
export PYTHONPATH="$R${PYTHONPATH:+:$PYTHONPATH}"
P="$ROOT/ForeSpark/.venv/bin/python"
trap '"$SERVING_SCRATCH/env/bin/python" "$R/publish_load_v76.py"' EXIT
"$P" "$R/benchmark_load_v76.py" --batch 64 --width 0 --round -1
"$P" "$R/benchmark_load_v76.py" --batch 8 --width 0 --round -1
for round in 0 1 2 3; do
 case "$round" in
 0) arms='8:3 8:8 64:3 64:8';;
 1) arms='8:8 8:3 64:8 64:3';;
 2) arms='64:3 64:8 8:3 8:8';;
 3) arms='64:8 64:3 8:8 8:3';;
 esac
 for arm in $arms; do
  "$P" "$R/benchmark_load_v76.py" --batch "${arm%:*}" --width "${arm#*:}" --round "$round"
 done
done
"$P" "$R/analyze_load_v76.py"
